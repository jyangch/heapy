"""Signal classification pipeline for Poisson source + Poisson background.

Provides :class:`ppSignal`, which histograms event time-stamps for both
source and background, computes SNR on raw bins and Bayesian blocks, and
sorts the bins into signal / background / bad categories.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import bayesian_blocks

from .core_funcs import *
from .plot_funcs import *
from ..util.data import json_dump



class ppSignal(object):
    """Detect and sort signal bins for Poisson data with Poisson background.

    Histograms both source and background event time-stamps onto the same
    ``bins``, derives per-bin rates/errors, and delegates classification to
    :func:`pp_snr` and :func:`classify_bins`.

    Attributes:
        ts: Source event time-stamps.
        bts: Background event time-stamps.
        bins: Bin edges (length ``N + 1``).
        backscale: Ratio scaling ``bts`` counts into the source region.
        cts, cts_err: Source histogram and Poisson error.
        bcts, bcts_err: Background histogram and Poisson error.
        time, rate, rate_err: Bin centers and source rates.
        bak, bak_err, net, net_err: Background and net rates with errors.
        ini_res, block_res, snr_res, sort_res: Stage result dicts.

    Example:
        >>> sig = ppSignal(ts, bts, bins, backscale=0.25)
        >>> sig.loop(p0=0.05, sigma=3)
        >>> sig.save('./out')

    Warning:
        Only apply to Poisson data with a Poisson background.
    """

    def __init__(self, ts, bts, bins, backscale=1, exp=None):
        """Histogram source/background event lists and derive bin geometry.

        Args:
            ts: Source event time-stamps.
            bts: Background event time-stamps.
            bins: Bin edges (length ``N + 1``).
            backscale: Ratio scaling background counts into the source
                region; defaults to ``1`` (identical extraction regions).
            exp: Per-bin exposure times; defaults to bin widths.

        Raises:
            TypeError: If ``exp`` and ``bins`` have mismatched sizes or any
                exposure exceeds its bin width.
        """
        
        self.ts = np.array(ts).astype(float)
        self.bts = np.array(bts).astype(float)
        self.bins = np.array(bins).astype(float)
        self.backscale = backscale

        cts, _ = np.histogram(self.ts, bins=self.bins)
        self.cts = np.array(cts).astype(int)
        self.cts_err = np.sqrt(self.cts)
        
        bcts, _ = np.histogram(self.bts, bins=self.bins)
        self.bcts = np.array(bcts).astype(int)
        self.bcts_err = np.sqrt(self.bcts)

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
        
        self.time = (self.lbins + self.rbins) / 2
        self.rate = self.cts / self.exp
        self.rate_err = self.cts_err / self.exp
        
        self.bak = self.bcts * self.backscale / self.exp
        self.bak_err = self.bcts_err * self.backscale / self.exp

        self.net = self.rate - self.bak
        self.net_err = np.sqrt(self.rate_err**2 + self.bak_err**2)
        
        self.ncts = self.net * self.exp
        self.ncts_err = self.net_err * self.exp

        self.ini_res = {'cts': self.cts, 'bcts': self.bcts, 
                        'time': self.time, 'rate': self.rate, 
                        'bak': self.bak, 'exp': self.exp, 
                        'backscale': self.backscale, 'bins': self.bins}

        self.block_res = None
        self.snr_res = None
        self.sort_res = None


    @classmethod
    def frombin(cls, cts, bcts, bins, backscale=1, exp=None):
        """Build a :class:`ppSignal` from pre-binned histograms.

        Synthesizes fake time-stamps uniformly within each bin to satisfy
        the histogram-based ``__init__`` path. The resulting timing is
        approximate; a ``UserWarning`` is emitted on every call.

        Args:
            cts: Source counts per bin (length ``N``).
            bcts: Background counts per bin (length ``N``).
            bins: Bin edges (length ``N + 1``).
            backscale: Ratio scaling background counts into the source
                region.
            exp: Per-bin exposure times; defaults to bin widths.

        Returns:
            A fully initialised :class:`ppSignal`.

        Raises:
            TypeError: If ``bins`` is not one element longer than ``cts``.
        """

        cts = np.array(cts)
        bcts = np.array(bcts)
        bins = np.array(bins)

        if bins.size != (cts.size + 1):
            raise TypeError("expected size(bins) = size(cts)+1")
        
        msg = 'rebuilt the time list, but may not accurate'
        warnings.warn(msg, UserWarning, stacklevel=2)

        ts_chunks = []
        for t1, t2, n in zip(bins[:-1], bins[1:], cts):
            ts_chunks.append(np.random.random(size=int(n)) * (t2 - t1) + t1)
        ts = np.concatenate(ts_chunks) if ts_chunks else np.array([])

        bts_chunks = []
        for t1, t2, n in zip(bins[:-1], bins[1:], bcts):
            bts_chunks.append(np.random.random(size=int(n)) * (t2 - t1) + t1)
        bts = np.concatenate(bts_chunks) if bts_chunks else np.array([])

        cls_ = cls(ts, bts, bins, backscale=backscale, exp=exp)
        return cls_
    
    
    def bblock(self, p0=0.05):
        """Run ``astropy.stats.bayesian_blocks`` and filter undersized gaps.

        Uses the event-time list directly when it has at most ``1e4``
        entries; otherwise falls back to the counts-per-bin array to keep
        the ``events`` fitness function tractable.

        Args:
            p0: False-alarm probability passed to ``bayesian_blocks``.
        """

        if len(self.ts) <= 1e4:
            edges_ = bayesian_blocks(self.ts, fitness='events', p0=p0)
        else:
            pos = np.where(self.cts > 0)[0]
            edges_ = bayesian_blocks(self.time[pos], self.cts[pos], fitness='events', p0=p0)
            
        edges_[0] = max(edges_[0], self.time[0])
        edges_[-1] = min(edges_[-1], self.time[-1])

        self.edges = filter_block_edges(edges_, np.min(self.binsize))
        self.nblock = len(self.edges) - 1
        self.re_binsize = self.edges[1:] - self.edges[:-1]

        self.block_res = {'edges': self.edges, 'nblock': self.nblock, 're_binsize': self.re_binsize}
    
    
    def calsnr(self):
        """Compute Poisson-Poisson SNR on raw bins and Bayesian blocks.

        Runs :meth:`bblock` first when block results are missing.
        Populates :attr:`re_cts`, :attr:`re_bcts`, :attr:`snr`,
        :attr:`re_snr`, and :attr:`snr_res`.
        """

        if self.block_res is None: self.bblock()

        self.re_cts, _ = np.histogram(self.ts, bins=self.edges)
        self.re_bcts, _ = np.histogram(self.bts, bins=self.edges)

        self.snr = np.zeros_like(self.binsize)
        for i in range(len(self.binsize)):
            self.snr[i] = pp_snr(self.cts[i], self.bcts[i], self.backscale)

        self.re_snr = np.zeros_like(self.re_binsize)
        for i in range(len(self.re_binsize)):
            self.re_snr[i] = pp_snr(self.re_cts[i], self.re_bcts[i], self.backscale)

        self.snr_res = {'snr': self.snr, 're_snr': self.re_snr, 
                        'cts': self.cts, 'bcts': self.bcts, 
                        're_cts': self.re_cts, 're_bcts': self.re_bcts}
        
        
    def sorting(self, sigma=3):
        """Classify raw bins and blocks into signal/background/bad by SNR.

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

        self.sort_res = {'sigma': self.sigma, 
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
        """Dump stage results as JSON and write three diagnostic PDFs.

        Creates ``savepath`` if missing. Writes ``ini_res``, ``block_res``,
        ``snr_res``, ``sort_res`` JSON files and ``bs``/``snr``/``sort``
        PDF figures, each suffixed with ``suffix``.

        Args:
            savepath: Destination directory; created if absent.
            suffix: Optional filename suffix inserted before each extension.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.ini_res, savepath + '/ini_res%s.json'%suffix)
        json_dump(self.block_res, savepath + '/block_res%s.json'%suffix)
        json_dump(self.snr_res, savepath + '/snr_res%s.json'%suffix)
        json_dump(self.sort_res, savepath + '/sort_res%s.json'%suffix)

        re_rate = self.re_cts / self.re_binsize
        with rc_context_for_save():
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            plot_lightcurve_bblock(ax, self.time, self.rate, self.edges, re_rate,
                                   bak=self.bak)
            fig.savefig(savepath + '/bs%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            plot_snr(ax, self.time, self.net, self.edges, self.snr, self.re_snr, self.sigma,
                     left_label='Net light curve',
                     bblock_snr_label='Baysian block SNR',
                     ylim_factors=(1.5, 1.1))
            fig.savefig(savepath + '/snr%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            plot_sorted(ax, self.time, self.edges, self.re_cts, self.re_binsize,
                        self.re_sig_idx, self.re_bkg_idx, self.re_bad_idx,
                        rate=self.rate, bak=self.bak)
            fig.savefig(savepath + '/sort%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)
