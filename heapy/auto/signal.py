"""Bayesian-block signal-classification pipelines for three data regimes.

Exposes three sibling classes that share the
``bblock`` → ``calsnr`` → ``sorting`` → ``save`` lifecycle but differ in
their statistical assumptions:

- :class:`pgSignal` -- Poisson source with a polynomial-modelled Gaussian
  background; supports a :meth:`pgSignal.from_components` factory that
  stacks multiple already-polyfit'd instances into a composite pipeline.
- :class:`ppSignal` -- Poisson source with a Poisson background.
- :class:`ggSignal` -- Gaussian-measured net light curve (background
  already subtracted upstream).

All three reuse the helpers and the :class:`~.signal_utils.SignalPlotter`
diagnostic figure from :mod:`heapy.auto.signal_utils`.

Example:
    from heapy.auto.signal import pgSignal
    sig = pgSignal(ts, bins)
    sig.loop(p0=0.05, sigma=3)
    sig.save('./out')
"""

import os

from astropy.stats import bayesian_blocks
import numpy as np

from ..util.data import union
from ..util.tools import json_dump
from .baseline import Baseline
from .polynomial import CompositePolynomial, Polynomial
from .signal_utils import (
    SignalPlotter,
    classify_bins,
    filter_block_edges,
    gauss_snr,
    indices_in_intervals,
    intervals_equal,
    pg_snr,
    plot_tau_diagnostic,
    plt_rc_context,
    pp_snr,
    time_rescaling_bblock,
)


class pgSignal:
    """Polynomial-background signal pipeline for Poisson event lists.

    Two construction paths are supported. The default :meth:`__init__`
    starts from raw event time-stamps and runs two passes of
    :meth:`bblock` → :meth:`calsnr` → :meth:`sorting` →
    :meth:`polyfit` (wrapped in :meth:`loop`); pass 1 seeds the
    background from a drpls :meth:`basefit` so the first :meth:`bblock`
    already accounts for non-stationary background via time rescaling,
    pass 2 then refines using the polynomial obtained from pass 1.
    :meth:`from_components` instead stacks pre-polyfit'd component
    instances that share the same binning, producing a composite-poly
    pipeline that runs only the second pass --
    :meth:`bblock` → :meth:`calsnr` → :meth:`sorting` -- since the
    background is already fixed.

    Attributes:
        ts: Source event time-stamps (concatenated across components in
            the composite path).
        bins: Bin edges (length ``N + 1``).
        cts, exp, time, rate: Histogram, exposure, bin centers, and rate.
            ``exp`` is only set on the raw-events construction path.
        ignore: User-supplied list of ``[low, high]`` intervals (or
            ``None``). When set, it serves as an escape hatch that
            bypasses the auto-detected ``sort_res['ignore']`` in
            :meth:`_effective_ignore`. NaN-gap intervals from
            :meth:`frombin` live on :attr:`_gap_int` and are always
            folded in regardless of this attribute.
        bl: :class:`~.baseline.Baseline` instance from :meth:`basefit`
            (raw-events path only).
        poly: :class:`~.polynomial.Polynomial` after :meth:`polyfit`, or
            :class:`~.polynomial.CompositePolynomial` after
            :meth:`from_components`.
        bak, bak_err, bcts, bcts_err: Background rate and counts with errors.
        net, net_err, ncts, ncts_err: Net rate and counts with errors.
            Populated by :meth:`polyfit` on the raw-events path and by
            :meth:`from_components` on the composite path.
        obj_list: Source components when constructed via
            :meth:`from_components`; ``None`` otherwise.
        ini_res, base_res, block_res, snr_res, sort_res, poly_res: Stage
            result dicts. ``ini_res`` and ``base_res`` are ``None`` on
            the composite path.

    Example:
        >>> sig = pgSignal(ts, bins)
        >>> sig.loop(p0=0.05, sigma=3)
        >>> sig.save('./out')

        >>> stack = pgSignal.from_components([pb1, pb2, pb3])
        >>> stack.loop()
        >>> stack.save('./out')

    Warning:
        Only apply to Poisson data with a smooth (Gaussian) background.
    """

    def __init__(self, ts, bins, exp=None, ignore=None):
        """Histogram events, derive bin geometry, and reset stage caches.

        Args:
            ts: Source event time-stamps.
            bins: Bin edges (length ``N + 1``).
            exp: Per-bin exposure times; defaults to bin widths.
            ignore: Intervals excluded from :meth:`polyfit` weighting.
                Accepts either a single pair ``[low, high]`` or a list
                of pairs ``[[low, high], ...]``; the bare pair form is
                normalized to a single-element list.

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
            raise TypeError('expected size(exp) + 1 = size(bins)')
        if not (self.exp <= self.binsize).all():
            raise TypeError('expected exp <= binsize')

        self.ignore = self._normalize_ignore(ignore)

        self.time = (self.lbins + self.rbins) / 2
        self.rate = self.cts / self.exp

        self.ini_res = {
            'time': self.time,
            'cts': self.cts,
            'rate': self.rate,
            'exp': self.exp,
            'bins': self.bins,
            'ignore': self.ignore,
        }

        self.base_res = None
        self.block_res = None
        self.snr_res = None
        self.sort_res = None
        self.poly_res = None

        self.obj_list = None
        self._gap_int = None

    @classmethod
    def frombin(cls, cts, bins, exp=None, ignore=None, random_seed=450001):
        """Build a :class:`pgSignal` from a pre-binned counts histogram.

        Synthesizes uniformly-distributed event time-stamps within each
        bin to feed :meth:`bblock`'s unbinned branch; the resulting
        timing is approximate. ``NaN`` entries in ``cts`` mark
        missing-data bins -- they are preserved verbatim on
        :attr:`cts` (so :attr:`rate`, :attr:`net`, and :attr:`ncts`
        carry ``NaN`` at those positions and propagate naturally
        through subtraction, multiplication, and plotting), no events
        are synthesized for them, and the merged gap intervals are
        recorded on :attr:`_gap_int`. :meth:`_effective_ignore` always
        folds the gap intervals into the exclusion set used by
        :meth:`basefit` and :meth:`polyfit` independently of
        user-supplied :attr:`ignore`. Gap-bin SNR comes out as ``NaN``
        and is classified as bad by :meth:`sorting`; gap boundaries
        are also fed to :meth:`bblock` so block edges pin there,
        preventing a single block from straddling a missing-data
        region.

        Args:
            cts: Source counts per bin. ``NaN`` entries mark bins with
                no valid observation; they are kept as ``NaN`` so
                downstream arithmetic propagates the missing-data
                signal without an intermediate ``0`` placeholder.
            bins: Bin edges (length ``N + 1``).
            exp: Per-bin exposure times; defaults to bin widths.
            ignore: Intervals excluded from :meth:`polyfit` weighting.
                Accepts either ``[low, high]`` or ``[[low, high], ...]``
                (see :meth:`__init__`). Stored verbatim on
                :attr:`ignore`; NaN-derived gap intervals live on
                :attr:`_gap_int` instead and never enter :attr:`ignore`.
            random_seed: Seed for the local RNG that draws synthetic
                photon times within each bin. Default ensures
                reproducibility across runs; pass ``None`` for OS
                entropy (non-reproducible).

        Returns:
            A fully initialised :class:`pgSignal`.

        Raises:
            TypeError: If ``bins`` is not one element longer than ``cts``,
                or if ``exp`` and ``bins`` have mismatched sizes, or any
                exposure exceeds its bin width.
        """

        cts = np.asarray(cts, dtype=float)
        bins = np.asarray(bins, dtype=float)

        if bins.size != (cts.size + 1):
            raise TypeError('expected size(bins) = size(cts)+1')

        nan_mask = np.isnan(cts)
        gap_int = None
        if nan_mask.any():
            raw_nan = [[float(bins[i]), float(bins[i + 1])] for i in np.where(nan_mask)[0]]
            gap_int = union(raw_nan)

        rng = np.random.default_rng(random_seed)
        chunks = []
        # No synthetic events for NaN bins; for everything else fall back to int(n).
        for t1, t2, n, is_gap in zip(bins[:-1], bins[1:], cts, nan_mask, strict=False):
            if is_gap:
                continue
            chunks.append(rng.random(size=int(n)) * (t2 - t1) + t1)
        ts = np.concatenate(chunks) if chunks else np.array([])

        # Build the instance directly so the NaN placeholders survive; the raw
        # __init__ path would re-derive cts from ts via np.histogram and lose
        # them (histogram outputs int and zeros NaN-bins by construction).
        inst = cls.__new__(cls)
        inst.ts = ts.astype(float)
        inst.bins = bins
        inst.lbins = bins[:-1]
        inst.rbins = bins[1:]
        inst.binsize = inst.rbins - inst.lbins
        inst.exp = np.array(exp).astype(float) if exp is not None else inst.binsize

        if (inst.exp.size + 1) != inst.bins.size:
            raise TypeError('expected size(exp) + 1 = size(bins)')
        if not (inst.exp <= inst.binsize).all():
            raise TypeError('expected exp <= binsize')

        inst.cts = cts if nan_mask.any() else cts.astype(int)
        inst.ignore = cls._normalize_ignore(ignore)
        inst.time = (inst.lbins + inst.rbins) / 2
        inst.rate = inst.cts / inst.exp  # NaN at gap bins propagates from here

        inst.ini_res = {
            'time': inst.time,
            'cts': inst.cts,
            'rate': inst.rate,
            'exp': inst.exp,
            'bins': inst.bins,
            'ignore': inst.ignore,
        }

        inst.base_res = None
        inst.block_res = None
        inst.snr_res = None
        inst.sort_res = None
        inst.poly_res = None

        inst.obj_list = None
        inst._gap_int = gap_int

        return inst

    @classmethod
    def from_components(cls, obj_list):
        """Stack polyfit'd :class:`pgSignal` components into a composite pipeline.

        Each component must share the same :attr:`bins` and must have
        already completed :meth:`polyfit` (i.e. expose ``bcts``,
        ``bcts_err``, ``bak``, ``bak_err``, ``net``, ``net_err``,
        ``ncts``, ``ncts_err``). The returned instance carries summed
        per-bin arrays, a :class:`~.polynomial.CompositePolynomial`
        background model, and is ready for :meth:`bblock` →
        :meth:`calsnr` → :meth:`sorting` -- :meth:`polyfit` /
        :meth:`basefit` are not applicable.

        Args:
            obj_list: Non-empty list of polyfit'd :class:`pgSignal`
                instances with identical :attr:`bins`.

        Returns:
            A composite :class:`pgSignal`. Stage caches (``block_res``,
            ``snr_res``, ``sort_res``) start at ``None``; ``ini_res`` and
            ``base_res`` stay ``None`` for the lifetime of the instance.

        Raises:
            TypeError: If any entry is not a :class:`pgSignal`.
            RuntimeError: If any entry is missing the polyfit-derived
                attributes.
            ValueError: If bin edges differ across entries.
        """

        required = ('bcts', 'bcts_err', 'bak', 'bak_err', 'net', 'net_err', 'ncts', 'ncts_err')
        for obj in obj_list:
            if not isinstance(obj, pgSignal):
                raise TypeError('expected obj_list contains pgSignal objects')
            missing = [a for a in required if not hasattr(obj, a)]
            if missing:
                raise RuntimeError(
                    'each pgSignal in obj_list must have run polyfit() first; '
                    f'missing attributes: {missing}'
                )

        base_bins = obj_list[0].bins
        for obj in obj_list[1:]:
            if not np.array_equal(obj.bins, base_bins):
                raise ValueError('all pgSignal objects in obj_list must share the same bins')

        inst = cls.__new__(cls)
        inst.obj_list = list(obj_list)

        inst.ts = np.concatenate([obj.ts for obj in obj_list])
        inst.bins = obj_list[0].bins

        inst.lbins = obj_list[0].lbins
        inst.rbins = obj_list[0].rbins
        inst.binsize = obj_list[0].binsize

        inst.time = obj_list[0].time
        inst.cts = np.sum([obj.cts for obj in obj_list], axis=0)
        inst.rate = np.sum([obj.rate for obj in obj_list], axis=0)

        inst.bcts = np.sum([obj.bcts for obj in obj_list], axis=0)
        inst.bcts_err = np.sqrt(np.sum([obj.bcts_err**2 for obj in obj_list], axis=0))
        inst.bak = np.sum([obj.bak for obj in obj_list], axis=0)
        inst.bak_err = np.sqrt(np.sum([obj.bak_err**2 for obj in obj_list], axis=0))

        inst.net = np.sum([obj.net for obj in obj_list], axis=0)
        inst.net_err = np.sqrt(np.sum([obj.net_err**2 for obj in obj_list], axis=0))

        inst.ncts = np.sum([obj.ncts for obj in obj_list], axis=0)
        inst.ncts_err = np.sqrt(np.sum([obj.ncts_err**2 for obj in obj_list], axis=0))

        inst.ignore = None

        inst.ini_res = None
        inst.base_res = None
        inst.block_res = None
        inst.snr_res = None
        inst.sort_res = None

        inst.poly = CompositePolynomial([obj.poly for obj in obj_list])
        inst.poly_res = {
            'bcts': inst.bcts,
            'bcts_err': inst.bcts_err,
            'bak': inst.bak,
            'bak_err': inst.bak_err,
        }

        gap_pool = [pair for obj in obj_list if obj._gap_int for pair in obj._gap_int]
        inst._gap_int = union(gap_pool) if gap_pool else None

        return inst

    @staticmethod
    def _normalize_ignore(ignore):
        """Coerce a user-supplied ``ignore`` argument into the nested form.

        Accepts either a bare numeric pair ``[low, high]`` (wrapped to
        ``[[low, high]]``) or an iterable of such pairs. Each pair is
        cast to a list of Python floats so downstream code can iterate
        uniformly. ``None`` passes through.

        Args:
            ignore: ``None``, ``[low, high]``, or ``[[low, high], ...]``.

        Returns:
            ``None`` or a list ``[[low, high], ...]``.
        """

        if ignore is None:
            return None
        seq = list(ignore)
        if len(seq) == 2 and np.isscalar(seq[0]) and np.isscalar(seq[1]):
            return [[float(seq[0]), float(seq[1])]]
        return [[float(low), float(upp)] for low, upp in seq]

    def _effective_ignore(self):
        """Resolve which intervals to drop from background fits at the current state.

        Resolution policy:

        - User-supplied :attr:`ignore` is an escape hatch: when set,
          returns ``union(self.ignore + _gap_int)``. The auto-detected
          ``sort_res['ignore']`` is bypassed so the user retains full
          control over which regions feed the background model.
        - Otherwise, when :attr:`sort_res` is populated, returns
          ``union(sort_res['ignore'] + _gap_int)`` -- auto-detected
          signal/bad blocks unioned with NaN gap intervals.
        - When neither :attr:`ignore` nor :attr:`sort_res` is set yet
          (e.g. inside :meth:`basefit` during pass 1), returns only the
          gap intervals (or ``[]`` if none).

        Returns:
            A list of ``[low, high]`` intervals; ``[]`` when nothing is
            excluded.
        """

        gap = list(self._gap_int) if self._gap_int else []
        if self.ignore is not None:
            return union(list(self.ignore) + gap)
        if self.sort_res is None:
            return gap
        return union(list(self.sort_res['ignore']) + gap)

    def bblock(self, p0=0.05):
        """Run time-rescaling Bayesian blocks under the current background estimate.

        The background cumulative integral is taken from :attr:`poly`
        when :meth:`polyfit` (or :meth:`from_components`) has run, and
        from the drpls :attr:`bl` set by :meth:`basefit` otherwise.
        With neither available the call degenerates to plain
        ``events``-mode Bayesian blocks. Inputs at most ``1e4`` events
        long use the unbinned branch; larger event lists fall back to
        the binned branch over :attr:`bins` (positive bins only).
        Composite instances always take the polynomial branch since
        :attr:`poly` is populated at construction.

        Args:
            p0: False-alarm probability passed to ``bayesian_blocks``.
        """

        if self.poly_res is not None:

            def bkg_integral(t):
                return self.poly.integral(t)[0]

        elif self.base_res is not None:
            bkg_integral = self.bl.integral

        else:
            bkg_integral = None

        if len(self.ts) <= 1e4:
            edges = time_rescaling_bblock(self.ts, p0=p0, bkg_integral=bkg_integral)
            mode = 'edges'
        else:
            # frombin keeps cts as float when NaN gap bins are present so the
            # missing-data semantic stays distinguishable from a measured zero.
            # astropy.bayesian_blocks(fitness='events') rejects non-integer
            # input; ``pos`` already filters NaN via ``> 0`` so the surviving
            # counts are integer-valued and safe to cast for this call.
            pos = np.where(self.cts > 0)[0]
            edges = time_rescaling_bblock(
                self.time[pos], cts=self.cts[pos].astype(int), p0=p0, bkg_integral=bkg_integral
            )
            mode = 'full'

        lowest = self.time[0]
        highest = self.time[-1]
        edges = np.clip(edges, lowest, highest)
        edges = np.unique(np.concatenate([[lowest], edges, [highest]]))

        gap_eps = np.unique([e for pair in self._gap_int for e in pair]) if self._gap_int else None
        self.edges = filter_block_edges(
            edges, np.min(self.binsize) / 1.8, protected=gap_eps, mode=mode
        )

        self.nblock = len(self.edges) - 1
        self.re_binsize = self.edges[1:] - self.edges[:-1]

        self.re_cts, _ = np.histogram(self.ts, bins=self.edges)

        self.block_res = {'edges': self.edges, 'nblock': self.nblock, 're_binsize': self.re_binsize}

    def basefit(self, weight=None):
        """Fit a smooth baseline via ``drpls`` to seed the first background.

        Runs :meth:`bblock` first when block results are missing. Zero
        weight is applied to bins inside :meth:`_effective_ignore`
        (i.e. user-supplied :attr:`ignore` unioned with
        :attr:`_gap_int`; :attr:`sort_res` is ``None`` at this stage so
        it does not contribute), and bins with zero counts are excluded
        before calling :meth:`~.baseline.Baseline.fit`. Populates
        :attr:`bl`, :attr:`bak`, :attr:`bcts`, and :attr:`base_res`. Not
        applicable on composite instances built via
        :meth:`from_components`.

        Args:
            weight: Optional per-bin weights; defaults to ones.

        Raises:
            RuntimeError: If called on a composite instance.
        """

        if isinstance(getattr(self, 'poly', None), CompositePolynomial):
            raise RuntimeError(
                'basefit is not applicable on composite instances; '
                'the background is already fixed by from_components()'
            )

        weight = np.ones_like(self.time) if weight is None else np.array(weight, dtype=float)

        exclude = self._effective_ignore()
        if exclude:
            ignore_idx = indices_in_intervals(self.lbins, self.rbins, exclude)
            weight[ignore_idx] = 0

        pos = np.where(self.cts > 0)[0]
        self.bl = Baseline.set_method('drpls')
        self.bl.fit(self.time[pos], self.rate[pos], w=weight[pos], lam=None, nk=None)
        self.bak = self.bl.val(self.time)
        self.bcts = self.bak * self.exp

        self.base_res = {'bcts': self.bcts + 0, 'bak': self.bak + 0}

    def calsnr(self):
        """Integrate the background over each block and compute Poisson-Gauss SNR.

        Runs :meth:`basefit` first when baseline results are missing
        (raw-events path only). When :attr:`poly_res` is populated --
        either by :meth:`polyfit` or by :meth:`from_components` --
        ``self.poly.val(x)`` provides the background model with error
        propagation; otherwise the baseline is integrated. Populates
        :attr:`snr`, :attr:`re_snr`, and :attr:`snr_res`.
        """

        if self.block_res is None:
            self.bblock()

        if self.base_res is None and self.poly_res is None:
            self.basefit()

        if self.poly_res is not None:
            self.re_bcts = np.zeros(self.nblock, dtype=float)
            self.re_bcts_err = np.zeros(self.nblock, dtype=float)
            for i, (left, right) in enumerate(zip(self.edges[:-1], self.edges[1:], strict=False)):
                self.re_bcts[i], self.re_bcts_err[i] = self.poly.block_integral(left, right)
        else:
            self.re_bcts = np.zeros(self.nblock, dtype=float)
            for i, (left, right) in enumerate(zip(self.edges[:-1], self.edges[1:], strict=False)):
                x = np.linspace(left, right, 100)
                y = self.bl.val(x)
                self.re_bcts[i] = np.trapezoid(y, x)

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

        self.snr_res = {
            'snr': self.snr,
            're_snr': self.re_snr,
            'cts': self.cts,
            'bcts': self.bcts,
            're_cts': self.re_cts,
            're_bcts': self.re_bcts,
        }

    def sorting(self, sigma=3):
        """Classify raw bins and blocks into signal/background/bad by SNR.

        Runs :meth:`calsnr` first when SNR results are missing. Unions the
        signal and bad block intervals into :attr:`ignore_int` so
        :meth:`polyfit` can exclude them from the background fit.
        Populates ``*_idx``/``*_int`` attributes plus :attr:`sort_res`.

        Args:
            sigma: Detection threshold in units of SNR.
        """

        if self.snr_res is None:
            self.calsnr()

        self.sigma = sigma
        re = classify_bins(self.re_snr, self.edges[:-1], self.edges[1:], sigma)
        self.re_sig_idx, self.re_bkg_idx, self.re_bad_idx = (
            re['sig_idx'],
            re['bkg_idx'],
            re['bad_idx'],
        )
        self.re_sig_int, self.re_bkg_int, self.re_bad_int = (
            re['sig_int'],
            re['bkg_int'],
            re['bad_int'],
        )

        bn = classify_bins(self.snr, self.lbins, self.rbins, sigma)
        self.sig_idx, self.bkg_idx, self.bad_idx = bn['sig_idx'], bn['bkg_idx'], bn['bad_idx']
        self.sig_int, self.bkg_int, self.bad_int = bn['sig_int'], bn['bkg_int'], bn['bad_int']

        self.re_bad_index = indices_in_intervals(self.lbins, self.rbins, self.re_bad_int)
        self.re_sig_index = indices_in_intervals(self.lbins, self.rbins, self.re_sig_int)

        self.ignore_int = union(self.re_bad_int + self.re_sig_int)

        self.sort_res = {
            'sigma': self.sigma,
            'ignore': self.ignore_int,
            're_bkg': (self.re_bkg_int, self.re_bkg_idx),
            're_sig': (self.re_sig_int, self.re_sig_idx, self.re_sig_index),
            're_bad': (self.re_bad_int, self.re_bad_idx, self.re_bad_index),
        }

    def polyfit(self, deg=None):
        """Refit the background as a polynomial over non-signal bins.

        Drops bins inside :meth:`_effective_ignore` and fits the
        remaining rates with :class:`~.polynomial.Polynomial`. The
        resolution policy (see :meth:`_effective_ignore`) is:
        user-supplied :attr:`ignore` is honoured as an escape hatch and
        bypasses the auto-derived ``sort_res['ignore']``; otherwise the
        auto-derived intervals are used. NaN-gap intervals
        (:attr:`_gap_int`) are always unioned in either branch since
        they mark "no observation", not "no signal". When neither
        :attr:`sort_res` nor :attr:`ignore` is set, :meth:`sorting` is
        triggered first to populate the auto branch; when only
        :attr:`ignore` is set (e.g. per-channel refits reusing a
        master's ignore), :meth:`sorting` is **not** triggered so the
        supplied intervals are used as-is. Updates :attr:`bak`,
        :attr:`bcts`, their errors, the net arrays, and
        :attr:`poly_res`. Not applicable on composite instances.

        Args:
            deg: Polynomial degree; ``None`` lets the two-pass fit pick
                the BIC-optimal degree from ``0..4``.

        Raises:
            RuntimeError: If called on a composite instance.
        """

        if isinstance(getattr(self, 'poly', None), CompositePolynomial):
            raise RuntimeError(
                'polyfit is not applicable on composite instances; '
                'rebuild the source components instead'
            )

        if self.sort_res is None and self.ignore is None:
            self.sorting()

        ignore = self._effective_ignore()
        ignore_idx = indices_in_intervals(self.lbins, self.rbins, ignore)

        notice_time = np.delete(self.time, ignore_idx)
        notice_rate = np.delete(self.rate, ignore_idx)
        notice_exp = np.delete(self.exp, ignore_idx)

        self.poly = Polynomial.set_method('2pass')
        self.poly.fit(notice_time, notice_rate, deg=deg, dx=notice_exp)
        self.bak, self.bak_err = self.poly.val(self.time)

        self.bcts = self.bak * self.exp
        self.bcts_err = self.bak_err * self.exp

        self.net = self.rate - self.bak
        self.net_err = np.sqrt(self.rate / self.exp + self.bak_err**2)

        self.ncts = self.net * self.exp
        self.ncts_err = self.net_err * self.exp

        self.poly_res = {
            'deg': self.poly.best_deg,
            'fit': self.poly.ls_res,
            'bcts': self.bcts,
            'bcts_err': self.bcts_err,
            'bak': self.bak,
            'bak_err': self.bak_err,
        }

    def loop(self, p0=0.05, sigma=3, deg=None, iter=False):
        """Run the full two-pass pipeline.

        Raw-events instances run two passes of ``bblock → calsnr →
        sorting → polyfit``: pass 1 seeds the background from a drpls
        baseline (so the very first :meth:`bblock` already accounts for
        non-stationary background via time rescaling); pass 2 refines
        with the polynomial obtained in pass 1. With ``iter_polyfit``
        enabled, pass 2's ``calsnr → sorting → polyfit`` triple is
        repeated (without a fresh :meth:`bblock`) until the auto-derived
        ignore intervals stabilize, capped at 10 iterations to bound
        runtime against pathological oscillation. Composite instances
        skip pass 1 entirely and never call :meth:`polyfit` since the
        background is fixed at :meth:`from_components` time.

        Args:
            p0: False-alarm probability for Bayesian blocks.
            sigma: Detection threshold for bin classification.
            deg: Polynomial degree for the background refit (``None``
                enables BIC selection); ignored on composite instances.
            iter: When ``True``, repeat the full ``bblock →
                calsnr → sorting → polyfit`` pass after pass 2 until
                the ignore intervals stabilize. Each iteration re-runs
                :meth:`bblock` with the latest polynomial so the edges
                stay consistent with the background estimate. Ignored
                on composite instances.
        """

        is_composite = isinstance(getattr(self, 'poly', None), CompositePolynomial)

        if not is_composite:
            # Pass 1: drpls baseline as background seed.
            self.basefit()
            self.bblock(p0)
            self.calsnr()
            self.sorting(sigma)
            self.polyfit(deg)

        # Pass 2: polynomial as background. Composite instances enter here directly.
        self.bblock(p0)
        self.calsnr()
        self.sorting(sigma)

        if is_composite:
            return

        self.polyfit(deg)

        if iter:
            for _ in range(10):
                prev_ignore = list(self.sort_res['ignore'])
                self.bblock(p0)
                self.calsnr()
                self.sorting(sigma)
                self.polyfit(deg)
                if intervals_equal(prev_ignore, self.sort_res['ignore']):
                    break

    def save_tau_diagnostic(self, savepath):
        """Save a 2-panel time-rescaling diagnostic PDF for debugging.

        Thin wrapper that checks the precondition (polynomial background
        must exist) and delegates rendering to
        :func:`~.signal_utils.plot_tau_diagnostic`. See that function
        for the panel layout and interpretation.

        Args:
            savepath: Destination directory; created if absent.

        Raises:
            RuntimeError: If :meth:`polyfit` (or :meth:`from_components`)
                has not run, leaving :attr:`poly` undefined.
        """

        if self.poly_res is None:
            raise RuntimeError(
                'save_tau_diagnostic requires polyfit() (or from_components()) to have run'
            )

        plot_tau_diagnostic(
            self.poly, self.ts, self.bins, self.edges, self.time, self.rate, savepath
        )

    def save(self, savepath):
        """Dump stage result dicts as JSON and write a diagnostic PDF.

        Creates ``savepath`` if missing. ``ini_res`` and ``base_res``
        are only written when populated (i.e. on the raw-events path);
        ``block_res`` / ``snr_res`` / ``sort_res`` / ``poly_res`` are
        always written. The PDF (``signal.pdf``) is a two-panel figure
        rendered through :class:`~.signal_utils.SignalPlotter`: total
        rate + Bayesian blocks on top, net rate + class-colored block
        SNR on the bottom.

        For tau-space verification of the time-rescaling step, call
        :meth:`save_tau_diagnostic` separately.

        Args:
            savepath: Destination directory; created if absent.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        if self.ini_res is not None:
            json_dump(self.ini_res, savepath + '/ini_res.json')
        if self.base_res is not None:
            json_dump(self.base_res, savepath + '/base_res.json')
        json_dump(self.block_res, savepath + '/block_res.json')
        json_dump(self.snr_res, savepath + '/snr_res.json')
        json_dump(self.sort_res, savepath + '/sort_res.json')
        json_dump(self.poly_res, savepath + '/poly_res.json')

        re_rate = self.re_cts / self.re_binsize
        re_net = (self.re_cts - self.re_bcts) / self.re_binsize

        with plt_rc_context():
            fig = SignalPlotter()
            if self._gap_int:
                fig.set_gaps(self._gap_int, self.bins)
            fig.plot_curve(self.time, self.rate, self.net, bak=self.bak, bak_err=self.bak_err)
            fig.plot_block(self.edges, re_rate, re_net)
            fig.plot_snr(self.edges, self.re_snr, self.sigma)
            fig.save(savepath + '/signal.pdf')


class ppSignal:
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
            raise TypeError('expected size(exp) + 1 = size(bins)')
        if not (self.exp <= self.binsize).all():
            raise TypeError('expected exp <= binsize')

        self.time = (self.lbins + self.rbins) / 2
        self.rate = self.cts / self.exp
        self.rate_err = self.cts_err / self.exp

        self.bak = self.bcts * self.backscale / self.exp
        self.bak_err = self.bcts_err * self.backscale / self.exp

        self.net = self.rate - self.bak
        self.net_err = np.sqrt(self.rate_err**2 + self.bak_err**2)

        self.ncts = self.net * self.exp
        self.ncts_err = self.net_err * self.exp

        self.ini_res = {
            'cts': self.cts,
            'bcts': self.bcts,
            'time': self.time,
            'rate': self.rate,
            'bak': self.bak,
            'exp': self.exp,
            'backscale': self.backscale,
            'bins': self.bins,
        }

        self.block_res = None
        self.snr_res = None
        self.sort_res = None

    @classmethod
    def frombin(cls, cts, bcts, bins, backscale=1, exp=None, random_seed=450001):
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
            random_seed: Seed for the local RNG that draws synthetic
                photon times within each bin. Default ensures
                reproducibility across runs; pass ``None`` for OS
                entropy (non-reproducible).

        Returns:
            A fully initialised :class:`ppSignal`.

        Raises:
            TypeError: If ``bins`` is not one element longer than ``cts``.
        """

        cts = np.array(cts)
        bcts = np.array(bcts)
        bins = np.array(bins)

        if bins.size != (cts.size + 1):
            raise TypeError('expected size(bins) = size(cts)+1')

        rng = np.random.default_rng(random_seed)
        ts_chunks = []
        for t1, t2, n in zip(bins[:-1], bins[1:], cts, strict=False):
            ts_chunks.append(rng.random(size=int(n)) * (t2 - t1) + t1)
        ts = np.concatenate(ts_chunks) if ts_chunks else np.array([])

        bts_chunks = []
        for t1, t2, n in zip(bins[:-1], bins[1:], bcts, strict=False):
            bts_chunks.append(rng.random(size=int(n)) * (t2 - t1) + t1)
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
            edges = bayesian_blocks(self.ts, fitness='events', p0=p0)
            mode = 'edges'
        else:
            pos = np.where(self.cts > 0)[0]
            edges = bayesian_blocks(self.time[pos], self.cts[pos], fitness='events', p0=p0)
            mode = 'full'

        lowest = self.time[0] - self.binsize[0] / 2
        highest = self.time[-1] + self.binsize[-1] / 2
        edges = np.clip(edges, lowest, highest)
        edges = np.unique(np.concatenate([[lowest], edges, [highest]]))

        self.edges = filter_block_edges(edges, np.min(self.binsize) / 1.8, mode=mode)
        self.nblock = len(self.edges) - 1
        self.re_binsize = self.edges[1:] - self.edges[:-1]

        self.block_res = {'edges': self.edges, 'nblock': self.nblock, 're_binsize': self.re_binsize}

    def calsnr(self):
        """Compute Poisson-Poisson SNR on raw bins and Bayesian blocks.

        Runs :meth:`bblock` first when block results are missing.
        Populates :attr:`re_cts`, :attr:`re_bcts`, :attr:`snr`,
        :attr:`re_snr`, and :attr:`snr_res`.
        """

        if self.block_res is None:
            self.bblock()

        self.re_cts, _ = np.histogram(self.ts, bins=self.edges)
        self.re_bcts, _ = np.histogram(self.bts, bins=self.edges)

        self.snr = np.zeros_like(self.binsize)
        for i in range(len(self.binsize)):
            self.snr[i] = pp_snr(self.cts[i], self.bcts[i], self.backscale)

        self.re_snr = np.zeros_like(self.re_binsize)
        for i in range(len(self.re_binsize)):
            self.re_snr[i] = pp_snr(self.re_cts[i], self.re_bcts[i], self.backscale)

        self.snr_res = {
            'snr': self.snr,
            're_snr': self.re_snr,
            'cts': self.cts,
            'bcts': self.bcts,
            're_cts': self.re_cts,
            're_bcts': self.re_bcts,
        }

    def sorting(self, sigma=3):
        """Classify raw bins and blocks into signal/background/bad by SNR.

        Runs :meth:`calsnr` first when SNR results are missing. Populates
        ``*_idx``/``*_int`` attributes plus :attr:`sort_res`.

        Args:
            sigma: Detection threshold in units of SNR.
        """

        if self.snr_res is None:
            self.calsnr()

        self.sigma = sigma
        re = classify_bins(self.re_snr, self.edges[:-1], self.edges[1:], sigma)
        self.re_sig_idx, self.re_bkg_idx, self.re_bad_idx = (
            re['sig_idx'],
            re['bkg_idx'],
            re['bad_idx'],
        )
        self.re_sig_int, self.re_bkg_int, self.re_bad_int = (
            re['sig_int'],
            re['bkg_int'],
            re['bad_int'],
        )

        bn = classify_bins(self.snr, self.lbins, self.rbins, sigma)
        self.sig_idx, self.bkg_idx, self.bad_idx = bn['sig_idx'], bn['bkg_idx'], bn['bad_idx']
        self.sig_int, self.bkg_int, self.bad_int = bn['sig_int'], bn['bkg_int'], bn['bad_int']

        self.re_bad_index = indices_in_intervals(self.lbins, self.rbins, self.re_bad_int)
        self.re_sig_index = indices_in_intervals(self.lbins, self.rbins, self.re_sig_int)

        self.sort_res = {
            'sigma': self.sigma,
            're_bkg': (self.re_bkg_int, self.re_bkg_idx),
            're_sig': (self.re_sig_int, self.re_sig_idx, self.re_sig_index),
            're_bad': (self.re_bad_int, self.re_bad_idx, self.re_bad_index),
        }

    def loop(self, p0=0.05, sigma=3):
        """Run :meth:`bblock`, :meth:`calsnr`, and :meth:`sorting` in order.

        Args:
            p0: False-alarm probability for Bayesian blocks.
            sigma: Detection threshold for bin classification.
        """

        self.bblock(p0)
        self.calsnr()
        self.sorting(sigma)

    def save(self, savepath):
        """Dump stage results as JSON and write a two-panel diagnostic PDF.

        Creates ``savepath`` if missing. Writes ``ini_res``,
        ``block_res``, ``snr_res``, ``sort_res`` JSON files plus
        ``signal.pdf``, a two-panel figure rendered through
        :class:`~.signal_utils.SignalPlotter` (total rate + Bayesian
        blocks on top, net rate + class-colored block SNR on the
        bottom).

        Args:
            savepath: Destination directory; created if absent.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.ini_res, savepath + '/ini_res.json')
        json_dump(self.block_res, savepath + '/block_res.json')
        json_dump(self.snr_res, savepath + '/snr_res.json')
        json_dump(self.sort_res, savepath + '/sort_res.json')

        re_rate = self.re_cts / self.re_binsize
        re_net = (self.re_cts - self.re_bcts) / self.re_binsize
        with plt_rc_context():
            fig = SignalPlotter()
            fig.plot_curve(self.time, self.rate, self.net, bak=self.bak)
            fig.plot_block(self.edges, re_rate, re_net)
            fig.plot_snr(self.edges, self.re_snr, self.sigma)
            fig.save(savepath + '/signal.pdf')


class ggSignal:
    """Detect and sort signal bins in a Gaussian net light curve.

    Designed for data that has been background-subtracted upstream and
    whose per-bin uncertainties are already Gaussian. The class never
    performs background subtraction itself; the inputs are interpreted
    as net counts and net errors throughout. For raw-counts pipelines
    that handle background internally, see :class:`ppSignal` or
    :class:`pgSignal`.

    Attributes:
        ncts: Per-bin net counts.
        ncts_err: 1-sigma uncertainty on :attr:`ncts`.
        bins: Bin edges (length ``N + 1``).
        exp: Per-bin exposure times.
        time: Bin centers.
        net: :attr:`ncts` / :attr:`exp` (net rate).
        net_err: :attr:`ncts_err` / :attr:`exp`.
        ini_res, block_res, snr_res, sort_res: Stage result dicts populated
            by :meth:`bblock`, :meth:`calsnr`, and :meth:`sorting`.

    Example:
        >>> sig = ggSignal(ncts, ncts_err, bins)
        >>> sig.loop(p0=0.05, sigma=3)
        >>> sig.save('./out')

    Warning:
        Only apply to Gaussian-distributed net data.
    """

    def __init__(self, ncts, ncts_err, bins, exp=None):
        """Store arrays and derive net rates, errors, and bin geometry.

        Args:
            ncts: Per-bin net counts matching bin count ``N``.
            ncts_err: 1-sigma uncertainty on ``ncts``; same length as ``ncts``.
            bins: Bin edges (length ``N + 1``).
            exp: Per-bin exposure times; defaults to bin widths when
                ``None``.

        Raises:
            TypeError: If ``exp`` and ``bins`` have mismatched sizes or any
                exposure exceeds its bin width.
        """

        self.ncts = np.array(ncts)
        self.ncts_err = np.array(ncts_err)
        self.bins = np.array(bins).astype(float)

        self.lbins = self.bins[:-1]
        self.rbins = self.bins[1:]
        self.binsize = self.bins[1:] - self.bins[:-1]
        if exp is None:
            self.exp = self.binsize
        else:
            self.exp = np.array(exp).astype(float)

        if (self.exp.size + 1) != self.bins.size:
            raise TypeError('expected size(exp) + 1 = size(bins)')
        if not (self.exp <= self.binsize).all():
            raise TypeError('expected exp <= binsize')

        self.time = (self.lbins + self.rbins) / 2
        self.net = self.ncts / self.exp
        self.net_err = self.ncts_err / self.exp

        self.ini_res = {
            'ncts': self.ncts,
            'ncts_err': self.ncts_err,
            'time': self.time,
            'net': self.net,
            'exp': self.exp,
            'bins': self.bins,
        }

        self.block_res = None
        self.snr_res = None
        self.sort_res = None

    def bblock(self, p0=0.05):
        """Run ``astropy.stats.bayesian_blocks`` and filter undersized gaps.

        Populates :attr:`edges`, :attr:`nblock`, :attr:`re_binsize`, and
        :attr:`block_res`.

        Args:
            p0: False-alarm probability passed to ``bayesian_blocks``.
        """

        edges = bayesian_blocks(self.time, self.net, self.net_err, fitness='measures', p0=p0)

        lowest = self.time[0] - self.binsize[0] / 2
        highest = self.time[-1] + self.binsize[-1] / 2
        edges = np.clip(edges, lowest, highest)
        edges = np.unique(np.concatenate([[lowest], edges, [highest]]))

        self.edges = filter_block_edges(edges, np.min(self.binsize) / 1.8)
        self.nblock = len(self.edges) - 1
        self.re_binsize = self.edges[1:] - self.edges[:-1]

        self.block_res = {'edges': self.edges, 'nblock': self.nblock, 're_binsize': self.re_binsize}

    def calsnr(self):
        """Compute Gaussian SNR on both raw bins and Bayesian blocks.

        Runs :meth:`bblock` first when block results are missing.
        Populates :attr:`re_ncts`, :attr:`re_ncts_err`, :attr:`snr`,
        :attr:`re_snr`, and :attr:`snr_res`.
        """

        if self.block_res is None:
            self.bblock()

        self.re_ncts, self.re_ncts_err = [], []
        for t1, t2 in zip(self.edges[:-1], self.edges[1:], strict=False):
            self.re_ncts.append(np.sum(self.ncts[(self.time >= t1) & (self.time < t2)]))
            self.re_ncts_err.append(
                np.sqrt(np.sum(self.ncts_err[(self.time >= t1) & (self.time < t2)] ** 2))
            )
        self.re_ncts = np.array(self.re_ncts)
        self.re_ncts_err = np.array(self.re_ncts_err)

        self.snr = np.zeros_like(self.binsize)
        for i in range(len(self.binsize)):
            self.snr[i] = gauss_snr(self.ncts[i], self.ncts_err[i])

        self.re_snr = np.zeros_like(self.re_binsize)
        for i in range(len(self.re_binsize)):
            self.re_snr[i] = gauss_snr(self.re_ncts[i], self.re_ncts_err[i])

        self.snr_res = {
            'snr': self.snr,
            're_snr': self.re_snr,
            're_ncts': self.re_ncts,
            're_ncts_err': self.re_ncts_err,
        }

    def sorting(self, sigma=3):
        """Classify raw bins and blocks into signal/background/bad by SNR.

        Runs :meth:`calsnr` first when SNR results are missing. Populates
        ``*_idx``/``*_int`` attributes plus :attr:`sort_res`.

        Args:
            sigma: Detection threshold in units of SNR.
        """

        if self.snr_res is None:
            self.calsnr()

        self.sigma = sigma
        re = classify_bins(self.re_snr, self.edges[:-1], self.edges[1:], sigma)
        self.re_sig_idx, self.re_bkg_idx, self.re_bad_idx = (
            re['sig_idx'],
            re['bkg_idx'],
            re['bad_idx'],
        )
        self.re_sig_int, self.re_bkg_int, self.re_bad_int = (
            re['sig_int'],
            re['bkg_int'],
            re['bad_int'],
        )

        bn = classify_bins(self.snr, self.lbins, self.rbins, sigma)
        self.sig_idx, self.bkg_idx, self.bad_idx = bn['sig_idx'], bn['bkg_idx'], bn['bad_idx']
        self.sig_int, self.bkg_int, self.bad_int = bn['sig_int'], bn['bkg_int'], bn['bad_int']

        self.re_bad_index = indices_in_intervals(self.lbins, self.rbins, self.re_bad_int)
        self.re_sig_index = indices_in_intervals(self.lbins, self.rbins, self.re_sig_int)

        self.sort_res = {
            'sigma': self.sigma,
            're_bkg': (self.re_bkg_int, self.re_bkg_idx),
            're_sig': (self.re_sig_int, self.re_sig_idx, self.re_sig_index),
            're_bad': (self.re_bad_int, self.re_bad_idx, self.re_bad_index),
        }

    def loop(self, p0=0.05, sigma=3):
        """Run :meth:`bblock`, :meth:`calsnr`, and :meth:`sorting` in order.

        Args:
            p0: False-alarm probability for Bayesian blocks.
            sigma: Detection threshold for bin classification.
        """

        self.bblock(p0)
        self.calsnr()
        self.sorting(sigma)

    def save(self, savepath):
        """Dump stage results as JSON and write a two-panel diagnostic PDF.

        Creates ``savepath`` if missing. Writes ``ini_res``,
        ``block_res``, ``snr_res``, ``sort_res`` JSON files plus
        ``signal.pdf``, a two-panel figure rendered through
        :class:`~.signal_utils.SignalPlotter` (net rate + Bayesian
        blocks on top, net rate + class-colored block SNR on the
        bottom; ggSignal passes ``net`` for both top and bottom curves
        since the input is already background-subtracted).

        Args:
            savepath: Destination directory; created if absent.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.ini_res, savepath + '/ini_res.json')
        json_dump(self.block_res, savepath + '/block_res.json')
        json_dump(self.snr_res, savepath + '/snr_res.json')
        json_dump(self.sort_res, savepath + '/sort_res.json')

        re_net = self.re_ncts / self.re_binsize
        with plt_rc_context():
            fig = SignalPlotter()
            fig.plot_curve(self.time, self.net, self.net)
            fig.plot_block(self.edges, re_net, re_net)
            fig.plot_snr(self.edges, self.re_snr, self.sigma)
            fig.save(savepath + '/signal.pdf')
