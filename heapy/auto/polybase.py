"""Signal classification pipeline for Poisson source + polynomial background.

Exposes :class:`PolyBase`, which first uses :class:`~.baseline.Baseline`
to estimate a smooth baseline for Bayesian-block seeding, then refits a
BIC-selected polynomial (:class:`~.polynomial.Polynomial`) over the
non-signal bins to produce a principled background model with error
propagation. :class:`multiPolyBase` stacks independent :class:`PolyBase`
instances that share the same binning.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import bayesian_blocks

from .core_funcs import *
from .plot_funcs import *
from .baseline import Baseline
from .polynomial import Polynomial
from ..util.data import union, json_dump



class PolyBase(object):
    """Polynomial-background signal pipeline for Poisson event lists.

    A typical run is :meth:`bblock` → :meth:`basefit` → :meth:`calsnr` →
    :meth:`sorting` → :meth:`polyfit`, wrapped in :meth:`loop`. The final
    :meth:`polyfit` replaces the baseline-derived background with a
    BIC-selected polynomial fitted over non-signal bins and caches its
    coefficient covariance for error propagation.

    Attributes:
        ts: Source event time-stamps.
        bins: Bin edges (length ``N + 1``).
        cts, exp, time, rate: Histogram, exposure, bin centers, and rate.
        ignore: Optional list of ``[low, high]`` intervals to exclude from
            the polynomial fit.
        bl: :class:`~.baseline.Baseline` instance from :meth:`basefit`.
        poly: :class:`~.polynomial.Polynomial` instance from :meth:`polyfit`.
        bak, bak_err, bcts, bcts_err: Background rate and counts with errors.
        net, net_err, ncts, ncts_err: Net rate and counts with errors.
        ini_res, base_res, block_res, snr_res, sort_res, poly_res: Stage
            result dicts.

    Example:
        >>> sig = PolyBase(ts, bins)
        >>> sig.loop(p0=0.05, sigma=3)
        >>> sig.save('./out')

    Warning:
        Only apply to Poisson data with a smooth (Gaussian) background.
    """

    def __init__(self, ts, bins, exp=None, ignore=None):
        """Histogram events, derive bin geometry, and reset stage caches.

        Args:
            ts: Source event time-stamps.
            bins: Bin edges (length ``N + 1``).
            exp: Per-bin exposure times; defaults to bin widths.
            ignore: Optional ``[[low, high], ...]`` intervals excluded
                from :meth:`polyfit` weighting.

        Raises:
            TypeError: If ``exp`` and ``bins`` have mismatched sizes or any
                exposure exceeds its bin width.
        """
        
        self.ts = np.array(ts).astype(float)
        self.bins = np.array(bins).astype(float)

        cts, _ = np.histogram(self.ts, bins=self.bins)
        self.cts = np.array(cts).astype(int)

        self.lbins = self.bins[:-1]
        self.rbins = self.bins[1:]
        self.binsize = self.bins[1:] - self.bins[:-1]
        if exp is None:
            self.exp = self.binsize
        else:
            self.exp = np.array(exp).astype(float)

        if (self.exp.size + 1) != self.bins.size:
            raise TypeError("expected size(exp) + 1 = size(bins)")
        if not (self.exp <= self.binsize).all():
            raise TypeError("expected exp <= binsize")
        
        self.ignore = ignore
        
        self.time = (self.lbins + self.rbins) / 2
        self.rate = self.cts / self.exp

        self.ini_res = {'time': self.time, 'cts': self.cts, 
                        'rate': self.rate, 'exp': self.exp, 
                        'bins': self.bins, 'ignore': self.ignore}

        self.base_res = None
        self.block_res = None
        self.snr_res = None
        self.sort_res = None
        self.poly_res = None


    @classmethod
    def frombin(cls, cts, bins, exp=None, ignore=None):
        """Build a :class:`PolyBase` from a pre-binned counts histogram.

        Synthesizes uniformly-distributed event time-stamps within each
        bin to populate the histogram-based ``__init__`` path. The
        resulting timing is approximate; a ``UserWarning`` is always
        emitted.

        Args:
            cts: Source counts per bin.
            bins: Bin edges (length ``N + 1``).
            exp: Per-bin exposure times; defaults to bin widths.
            ignore: Optional ``[[low, high], ...]`` intervals excluded
                from :meth:`polyfit` weighting.

        Returns:
            A fully initialised :class:`PolyBase`.

        Raises:
            TypeError: If ``bins`` is not one element longer than ``cts``.
        """

        cts = np.array(cts)
        bins = np.array(bins)

        if bins.size != (cts.size + 1):
            raise TypeError("expected size(bins) = size(cts)+1")
        
        msg = 'rebuilt the time list, but may not accurate'
        warnings.warn(msg, UserWarning, stacklevel=2)

        chunks = []
        for t1, t2, n in zip(bins[:-1], bins[1:], cts):
            chunks.append(np.random.random(size=int(n)) * (t2 - t1) + t1)
        ts = np.concatenate(chunks) if chunks else np.array([])

        sbs_ = cls(ts, bins, exp, ignore)
        return sbs_
    
    
    def bblock(self, p0=0.05):
        """Run ``astropy.stats.bayesian_blocks`` and filter undersized gaps.

        Before :meth:`polyfit` has run, the partition is driven by source
        events alone; afterwards the net rate and its error are passed to
        the ``measures`` fitness so the polynomial background seeds the
        re-partitioning.

        Args:
            p0: False-alarm probability passed to ``bayesian_blocks``.
        """

        if self.poly_res is None:
            if len(self.ts) <= 1e4:
                edges_ = bayesian_blocks(self.ts, fitness='events', p0=p0)
            else:
                pos = np.where(self.cts > 0)[0]
                edges_ = bayesian_blocks(self.time[pos], self.cts[pos], fitness='events', p0=p0)
        else:
            edges_ = bayesian_blocks(self.time, self.net, self.net_err, fitness='measures', p0=p0)
            
        edges_[0] = max(edges_[0], self.time[0])
        edges_[-1] = min(edges_[-1], self.time[-1])

        self.edges = filter_block_edges(edges_, np.min(self.binsize))
        self.nblock = len(self.edges) - 1
        self.re_binsize = self.edges[1:] - self.edges[:-1]

        self.re_cts, _ = np.histogram(self.ts, bins=self.edges)

        self.block_res = {'edges': self.edges, 'nblock': self.nblock, 're_binsize': self.re_binsize}


    def basefit(self, weight=None):
        """Fit a smooth baseline via ``drpls`` to seed the first background.

        Runs :meth:`bblock` first when block results are missing. Zero
        weight is applied to bins inside :attr:`ignore`, and bins with
        zero counts are excluded before calling
        :meth:`~.baseline.Baseline.fit`. Populates :attr:`bl`, :attr:`bak`,
        :attr:`bcts`, and :attr:`base_res`.

        Args:
            weight: Optional per-bin weights; defaults to ones.
        """

        if self.block_res is None: self.bblock()

        if weight is None:
            weight = np.ones_like(self.time)
        else:
            weight = np.array(weight, dtype=float)

        if self.ignore is not None:
            ignore_idx = indices_in_intervals(self.lbins, self.rbins, self.ignore)
            weight[ignore_idx] = 0

        pos = np.where(self.cts > 0)[0]
        self.bl = Baseline.set_method('drpls')
        self.bl.fit(self.time[pos], self.rate[pos], w=weight[pos], lam=None, nk=None)
        self.bak = self.bl.val(self.time)
        self.bcts = self.bak * self.exp

        self.base_res = {'bcts': self.bcts + 0, 'bak': self.bak + 0}


    def calsnr(self):
        """Integrate background over each block and compute Poisson-Gauss SNR.

        Runs :meth:`basefit` first when baseline results are missing. If
        :meth:`polyfit` has already produced :attr:`poly`, the polynomial
        is also used to propagate block-level background uncertainty;
        otherwise only the baseline is integrated. Populates :attr:`snr`,
        :attr:`re_snr`, and :attr:`snr_res`.
        """

        if self.base_res is None: self.basefit()
        
        if self.poly_res is not None:
            self.re_bcts = np.zeros(self.nblock, dtype=float)
            self.re_bcts_err = np.zeros(self.nblock, dtype=float)
            for i, (l, r) in enumerate(zip(self.edges[:-1], self.edges[1:])):
                x = np.linspace(l, r, 100)
                y, y_err = self.poly.val(x)
                self.re_bcts[i] = np.trapz(y, x)
                self.re_bcts_err[i] = np.sqrt(np.trapz(y_err ** 2, x))
        else:
            self.re_bcts = np.zeros(self.nblock, dtype=float)
            for i, (l, r) in enumerate(zip(self.edges[:-1], self.edges[1:])):
                x = np.linspace(l, r, 100)
                y = self.bl.val(x)
                self.re_bcts[i] = np.trapz(y, x)

        bcts_err = getattr(self, 'bcts_err', None)
        self.snr = np.zeros_like(self.binsize)
        for i in range(len(self.binsize)):
            err_i = None if bcts_err is None else bcts_err[i]
            self.snr[i] = pg_snr(self.cts[i], self.bcts[i], err_i)

        re_bcts_err = getattr(self, 're_bcts_err', None)
        self.re_snr = np.zeros_like(self.re_binsize)
        for i in range(len(self.re_binsize)):
            err_i = None if re_bcts_err is None else re_bcts_err[i]
            self.re_snr[i] = pg_snr(self.re_cts[i], self.re_bcts[i], err_i)

        self.snr_res = {'snr': self.snr, 're_snr': self.re_snr,
                        'cts': self.cts, 'bcts': self.bcts,
                        're_cts': self.re_cts, 're_bcts': self.re_bcts}


    def sorting(self, sigma=3):
        """Classify raw bins and blocks into signal/background/bad by SNR.

        Runs :meth:`calsnr` first when SNR results are missing. Unions the
        signal and bad block intervals into :attr:`ignore_int` so
        :meth:`polyfit` can exclude them from the background fit.
        Populates ``*_idx``/``*_int`` attributes plus :attr:`sort_res`.

        Args:
            sigma: Detection threshold in units of SNR.
        """

        if self.snr_res is None: self.calsnr()

        self.sigma = sigma
        re = classify_bins(self.re_snr, self.edges[:-1], self.edges[1:], sigma)
        self.re_sig_idx, self.re_bkg_idx, self.re_bad_idx = re['sig_idx'], re['bkg_idx'], re['bad_idx']
        self.re_sig_int, self.re_bkg_int, self.re_bad_int = re['sig_int'], re['bkg_int'], re['bad_int']

        bn = classify_bins(self.snr, self.lbins, self.rbins, sigma)
        self.sig_idx, self.bkg_idx, self.bad_idx = bn['sig_idx'], bn['bkg_idx'], bn['bad_idx']
        self.sig_int, self.bkg_int, self.bad_int = bn['sig_int'], bn['bkg_int'], bn['bad_int']

        self.re_bad_index = indices_in_intervals(self.lbins, self.rbins, self.re_bad_int)
        self.re_sig_index = indices_in_intervals(self.lbins, self.rbins, self.re_sig_int)

        self.ignore_int = union(self.re_bad_int + self.re_sig_int)

        self.sort_res = {'sigma': self.sigma, 'ignore': self.ignore_int,
                         're_bkg': (self.re_bkg_int, self.re_bkg_idx),
                         're_sig': (self.re_sig_int, self.re_sig_idx, self.re_sig_index),
                         're_bad': (self.re_bad_int, self.re_bad_idx, self.re_bad_index)}


    def polyfit(self, deg=None):
        """Refit the background as a polynomial over non-signal bins.

        Drops bins inside :attr:`ignore` (derived from :meth:`sorting` if
        not set at construction) and fits the remaining rates with
        :class:`~.polynomial.Polynomial`. Updates :attr:`bak`, :attr:`bcts`,
        their errors, the net arrays, and :attr:`poly_res`.

        Args:
            deg: Polynomial degree; ``None`` lets the two-pass fit pick
                the BIC-optimal degree from ``0..4``.
        """

        if self.ignore is None:
            if self.sort_res is None: self.sorting()
            self.ignore = self.sort_res['ignore']

        ignore_idx = indices_in_intervals(self.lbins, self.rbins, self.ignore)

        notice_time = np.delete(self.time, ignore_idx)
        notice_rate = np.delete(self.rate, ignore_idx)
        notice_exp = np.delete(self.exp, ignore_idx)

        self.poly = Polynomial.set_method('2pass')
        self.poly.fit(notice_time, notice_rate, deg=deg, dx=notice_exp)
        self.bak, self.bak_err = self.poly.val(self.time)

        self.bcts = self.bak * self.exp
        self.bcts_err = self.bak_err * self.exp
        
        self.net = self.rate - self.bak
        self.net_err = np.sqrt(self.rate / self.exp + self.bak_err ** 2)
        
        self.ncts = self.net * self.exp
        self.ncts_err = self.net_err * self.exp

        self.poly_res = {'deg': self.poly.best_deg, 
                         'fit': self.poly.ls_res, 
                         'bcts': self.bcts, 
                         'bcts_err': self.bcts_err, 
                         'bak': self.bak, 
                         'bak_err': self.bak_err}
        
    
    def loop(self, p0=0.05, sigma=3, deg=None):
        """Run the full pipeline: bblock, calsnr, sorting, polyfit.

        Note that :meth:`basefit` is invoked lazily inside :meth:`calsnr`.

        Args:
            p0: False-alarm probability for Bayesian blocks.
            sigma: Detection threshold for bin classification.
            deg: Polynomial degree for the background refit (``None``
                enables BIC selection).
        """

        self.bblock(p0)
        self.calsnr()
        self.sorting(sigma)
        self.polyfit(deg)


    def save(self, savepath, suffix=''):
        """Dump all stage result dicts as JSON and write diagnostic PDFs.

        Creates ``savepath`` if missing. Writes six JSON files and three
        PDF figures (``bs``/``snr``/``sort``), each suffixed with
        ``suffix``.

        Args:
            savepath: Destination directory; created if absent.
            suffix: Optional filename suffix inserted before each extension.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.ini_res, savepath + '/ini_res%s.json'%suffix)
        json_dump(self.base_res, savepath + '/base_res%s.json'%suffix)
        json_dump(self.block_res, savepath + '/block_res%s.json'%suffix)
        json_dump(self.snr_res, savepath + '/snr_res%s.json'%suffix)
        json_dump(self.sort_res, savepath + '/sort_res%s.json'%suffix)
        json_dump(self.poly_res, savepath + '/poly_res%s.json'%suffix)

        re_rate = self.re_cts / self.re_binsize
        with rc_context_for_save():
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            plot_lightcurve_bblock(ax, self.time, self.rate, self.edges, re_rate,
                                   bak=self.bak, bak_err=self.bak_err)
            fig.savefig(savepath + '/bs%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            plot_snr(ax, self.time, self.net, self.edges, self.snr, self.re_snr, self.sigma,
                     left_label='Net light curve',
                     bblock_snr_label='Bayesian block SNR',
                     ylim_factors=(1.1, 1.2))
            fig.savefig(savepath + '/snr%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            plot_sorted(ax, self.time, self.edges, self.re_cts, self.re_binsize,
                        self.re_sig_idx, self.re_bkg_idx, self.re_bad_idx,
                        bak=self.bak, bak_err=self.bak_err)
            fig.savefig(savepath + '/sort%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)



class multiPolyBase(object):
    """Stack independent :class:`PolyBase` runs that share the same binning.

    Each input object must have already run :meth:`PolyBase.polyfit`; the
    stack then sums counts, rates, and background models (propagating
    errors in quadrature) and exposes the same Bayesian-block /
    classification pipeline as :class:`PolyBase`.

    Attributes:
        obj_list: Source :class:`PolyBase` instances.
        cts, rate, bcts, bcts_err, bak, bak_err, net, net_err: Per-bin
            sums and propagated errors (length ``N``).
        block_res, snr_res, sort_res, poly_res: Stage result dicts.

    Example:
        >>> stack = multiPolyBase([pb1, pb2, pb3])
        >>> stack.loop()
        >>> stack.save('./out')

    Warning:
        All objects in ``obj_list`` must share identical bin edges and
        must have run :meth:`PolyBase.polyfit` first.
    """

    def __init__(self, obj_list):
        """Validate, sum, and cache per-bin arrays from ``obj_list``.

        Args:
            obj_list: Non-empty list of :class:`PolyBase` instances with
                identical :attr:`~PolyBase.bins`.

        Raises:
            TypeError: If any entry is not a :class:`PolyBase`.
            RuntimeError: If any entry is missing the polyfit-derived
                attributes ``bcts``, ``bcts_err``, ``bak``, ``bak_err``,
                ``net``, or ``net_err``.
            ValueError: If bin edges differ across entries.
        """

        self.obj_list = obj_list

        required = ('bcts', 'bcts_err', 'bak', 'bak_err', 'net', 'net_err')
        for obj in self.obj_list:
            if not isinstance(obj, PolyBase):
                raise TypeError("expected obj_list contains PolyBase objects")
            missing = [a for a in required if not hasattr(obj, a)]
            if missing:
                raise RuntimeError(
                    "each PolyBase in obj_list must have run polyfit() first; "
                    f"missing attributes: {missing}")

        base_bins = obj_list[0].bins
        for obj in obj_list[1:]:
            if not np.array_equal(obj.bins, base_bins):
                raise ValueError("all PolyBase objects in obj_list must share the same bins")

        self.ts = np.concatenate([obj.ts for obj in self.obj_list])
        self.bins = self.obj_list[0].bins

        self.lbins = self.obj_list[0].lbins
        self.rbins = self.obj_list[0].rbins
        self.binsize = self.obj_list[0].binsize
        
        self.time = self.obj_list[0].time
        self.cts = np.sum([obj.cts for obj in self.obj_list], axis=0)
        self.rate = np.sum([obj.rate for obj in self.obj_list], axis=0)
        
        self.bcts = np.sum([obj.bcts for obj in self.obj_list], axis=0)
        self.bcts_err = np.sqrt(np.sum([obj.bcts_err ** 2 for obj in self.obj_list], axis=0))
        self.bak = np.sum([obj.bak for obj in self.obj_list], axis=0)
        self.bak_err = np.sqrt(np.sum([obj.bak_err ** 2 for obj in self.obj_list], axis=0))
        
        self.net = np.sum([obj.net for obj in self.obj_list], axis=0)
        self.net_err = np.sqrt(np.sum([obj.net_err ** 2 for obj in self.obj_list], axis=0))

        self.block_res = None
        self.snr_res = None
        self.sort_res = None
        
        self.poly_res = {'bcts': self.bcts, 
                         'bcts_err': self.bcts_err, 
                         'bak': self.bak, 
                         'bak_err': self.bak_err}
        
        
    def bblock(self, p0=0.05):
        """Run ``astropy.stats.bayesian_blocks`` on the net-rate stack.

        Populates :attr:`edges`, :attr:`nblock`, :attr:`re_binsize`,
        :attr:`re_cts`, and :attr:`block_res`.

        Args:
            p0: False-alarm probability passed to ``bayesian_blocks``.
        """

        edges_ = bayesian_blocks(self.time, self.net, self.net_err, fitness='measures', p0=p0)

        edges_[0] = max(edges_[0], self.time[0])
        edges_[-1] = min(edges_[-1], self.time[-1])

        self.edges = filter_block_edges(edges_, np.min(self.binsize))
        self.nblock = len(self.edges) - 1
        self.re_binsize = self.edges[1:] - self.edges[:-1]

        self.re_cts, _ = np.histogram(self.ts, bins=self.edges)

        self.block_res = {'edges': self.edges, 'nblock': self.nblock, 're_binsize': self.re_binsize}


    def calsnr(self):
        """Integrate each component polynomial over blocks and compute SNR.

        Runs :meth:`bblock` first when block results are missing. The
        block-level background uncertainty is the quadrature sum of each
        component's integrated polynomial variance. Populates :attr:`snr`,
        :attr:`re_snr`, and :attr:`snr_res`.
        """

        if self.block_res is None: self.bblock()

        self.re_bcts = np.zeros(self.nblock, dtype=float)
        self.re_bcts_err = np.zeros(self.nblock, dtype=float)
        for i, (l, r) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            x = np.linspace(l, r, 100)
            re_bcts_i, re_bcts_var_i = 0, 0
            for obj in self.obj_list:
                y, y_err = obj.poly.val(x)
                re_bcts_i += np.trapz(y, x)
                re_bcts_var_i += np.trapz(y_err ** 2, x)
            self.re_bcts[i] = re_bcts_i
            self.re_bcts_err[i] = np.sqrt(re_bcts_var_i)

        bcts_err = getattr(self, 'bcts_err', None)
        self.snr = np.zeros_like(self.binsize)
        for i in range(len(self.binsize)):
            err_i = None if bcts_err is None else bcts_err[i]
            self.snr[i] = pg_snr(self.cts[i], self.bcts[i], err_i)

        re_bcts_err = getattr(self, 're_bcts_err', None)
        self.re_snr = np.zeros_like(self.re_binsize)
        for i in range(len(self.re_binsize)):
            err_i = None if re_bcts_err is None else re_bcts_err[i]
            self.re_snr[i] = pg_snr(self.re_cts[i], self.re_bcts[i], err_i)

        self.snr_res = {'snr': self.snr, 're_snr': self.re_snr,
                        'cts': self.cts, 'bcts': self.bcts,
                        're_cts': self.re_cts, 're_bcts': self.re_bcts}


    def sorting(self, sigma=3):
        """Classify stacked bins and blocks into signal/background/bad by SNR.

        Runs :meth:`calsnr` first when SNR results are missing. Populates
        ``*_idx``/``*_int`` attributes plus :attr:`sort_res`.

        Args:
            sigma: Detection threshold in units of SNR.
        """

        if self.snr_res is None: self.calsnr()

        self.sigma = sigma
        re = classify_bins(self.re_snr, self.edges[:-1], self.edges[1:], sigma)
        self.re_sig_idx, self.re_bkg_idx, self.re_bad_idx = re['sig_idx'], re['bkg_idx'], re['bad_idx']
        self.re_sig_int, self.re_bkg_int, self.re_bad_int = re['sig_int'], re['bkg_int'], re['bad_int']

        bn = classify_bins(self.snr, self.lbins, self.rbins, sigma)
        self.sig_idx, self.bkg_idx, self.bad_idx = bn['sig_idx'], bn['bkg_idx'], bn['bad_idx']
        self.sig_int, self.bkg_int, self.bad_int = bn['sig_int'], bn['bkg_int'], bn['bad_int']

        self.re_bad_index = indices_in_intervals(self.lbins, self.rbins, self.re_bad_int)
        self.re_sig_index = indices_in_intervals(self.lbins, self.rbins, self.re_sig_int)

        self.ignore_int = union(self.re_bad_int + self.re_sig_int)

        self.sort_res = {'sigma': self.sigma, 'ignore': self.ignore_int,
                         're_bkg': (self.re_bkg_int, self.re_bkg_idx),
                         're_sig': (self.re_sig_int, self.re_sig_idx, self.re_sig_index),
                         're_bad': (self.re_bad_int, self.re_bad_idx, self.re_bad_index)}


    def loop(self, p0=0.05, sigma=3):
        """Run :meth:`bblock`, :meth:`calsnr`, and :meth:`sorting` in order.

        Args:
            p0: False-alarm probability for Bayesian blocks.
            sigma: Detection threshold for bin classification.
        """

        self.bblock(p0)
        self.calsnr()
        self.sorting(sigma)


    def save(self, savepath, suffix=''):
        """Dump block/snr/sort/poly result dicts as JSON and write PDFs.

        Creates ``savepath`` if missing. Writes four JSON files and three
        diagnostic PDFs (``bs``/``snr``/``sort``), each suffixed with
        ``suffix``.

        Args:
            savepath: Destination directory; created if absent.
            suffix: Optional filename suffix inserted before each extension.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.block_res, savepath + '/block_res%s.json'%suffix)
        json_dump(self.snr_res, savepath + '/snr_res%s.json'%suffix)
        json_dump(self.sort_res, savepath + '/sort_res%s.json'%suffix)
        json_dump(self.poly_res, savepath + '/poly_res%s.json'%suffix)

        re_rate = self.re_cts / self.re_binsize
        with rc_context_for_save():
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            plot_lightcurve_bblock(ax, self.time, self.rate, self.edges, re_rate,
                                   bak=self.bak, bak_err=self.bak_err)
            fig.savefig(savepath + '/bs%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            plot_snr(ax, self.time, self.net, self.edges, self.snr, self.re_snr, self.sigma,
                     left_label='Net light curve',
                     bblock_snr_label='Bayesian block SNR',
                     ylim_factors=(1.5, 1.1))
            fig.savefig(savepath + '/snr%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            plot_sorted(ax, self.time, self.edges, self.re_cts, self.re_binsize,
                        self.re_sig_idx, self.re_bkg_idx, self.re_bad_idx,
                        bak=self.bak, bak_err=self.bak_err)
            fig.savefig(savepath + '/sort%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)