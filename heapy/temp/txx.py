"""Compute Txx duration metrics (T90, T50, etc.) for gamma-ray burst light curves.

Architecture mirrors :mod:`~heapy.temp.lag`: :class:`Txx` is a
self-contained engine that takes plain per-bin arrays (``cts``, ``bcts``,
their errors, bin edges, and an already-known ``pulse`` interval list)
plus a ``type`` tag (``'pg'``, ``'pp'``, or ``'gg'``) selecting the noise
model -- it has no dependency on ``pgSignal``/``ppSignal``/``ggSignal``.
Locating the pulse interval (Bayesian-blocks segmentation + SNR
classification) is *not* Txx's job; that lives in the ``sorting`` method
of those Signal classes, exactly as before.

Two construction paths bridge Signal-based pulse detection to this
primitive interface:

- :meth:`Txx.from_pgsignal` / :meth:`Txx.from_ppsignal` /
  :meth:`Txx.from_ggsignal` wrap an already-processed Signal instance
  (``sort_res`` populated, so ``pulse`` is known) and extract the plain
  arrays :class:`Txx` needs.
- ``pgTxx`` / ``ppTxx`` / ``ggTxx`` are lazy two-phase subclasses:
  ``__init__`` builds the raw-array Signal instance (mirroring the
  historical direct-construction API, e.g. ``pgTxx(ts, bins, ...)``) but
  defers completing :class:`Txx`'s own state until :meth:`find_pulse`
  runs the Signal's pipeline and hands off ``cts``/``bcts``/``pulse``.

Monte Carlo simulation (1000 realisations by default) propagates
uncertainties on the start/stop times.

Example:
    from heapy.temp.txx import pgTxx
    txx = pgTxx(ts, bins)
    txx.find_pulse()
    txx.calculate(xx=0.9)
    txx.save('/output/dir')

    # Or fully self-contained, given data you already have:
    from heapy.temp.txx import Txx
    txx = Txx(cts, bins, pulse=[[10.0, 25.0]], type='pg',
              bcts=bcts, bcts_err=bcts_err)
    txx.calculate(xx=0.9)
"""

import os
import warnings

import numpy as np

from ..auto.signal import ggSignal, pgSignal, ppSignal
from ..auto.signal_utils import indices_in_intervals
from ..util.data import union
from ..util.tools import format_message, json_dump, plt_rc_context
from .temp_utils import TxxPlotter, calculate_txx, generate_mc_sample, get_mc_errors, validate_input


class Txx:
    """Compute Txx duration and its Monte Carlo uncertainties.

    Self-contained: takes per-bin ``cts``/``bcts`` (or ``ncts``-only for
    ``'gg'``) and an already-known ``pulse`` interval list, with no
    dependency on ``pgSignal``/``ppSignal``/``ggSignal``. Locating the
    pulse is a separate, prior concern -- see :meth:`from_pgsignal` /
    :meth:`from_ppsignal` / :meth:`from_ggsignal`, or the ``pgTxx`` /
    ``ppTxx`` / ``ggTxx`` convenience subclasses, for bridging from a
    Signal instance's own ``sorting``-detected pulse.

    Attributes:
        pulse: List of ``[pstart, pstop]`` pairs.
        pstart, pstop: Arrays of pulse start/stop times (``pulse``
            unpacked column-wise).
        txx_res: Dictionary storing Txx results, or ``None`` before
            ``calculate`` is called.
    """

    def __init__(
        self,
        cts,
        bins,
        pulse,
        bcts=None,
        cts_err=None,
        bcts_err=None,
        backscale=1,
        exp=None,
        type='pg',
    ):
        """Initialize Txx with a fine-bin light curve and known pulse interval(s).

        Args:
            cts: 1-D array of per-bin source counts (``'pg'``/``'pp'``), or
                net (background-subtracted) counts (``'gg'``).
            bins: Bin edges (length ``len(cts) + 1``).
            pulse: List of ``[pstart, pstop]`` pairs already located by the
                caller (e.g. via a Signal class's ``sorting``); an empty
                list means no pulse.
            bcts: Background counts per bin; required for ``'pg'``/``'pp'``.
            cts_err: Count errors for ``cts``; required for ``'gg'``,
                defaults to ``sqrt(cts)`` for ``'pg'``/``'pp'``.
            bcts_err: Background count errors; required for ``'pg'``,
                defaults to ``sqrt(bcts)`` for ``'pp'``, defaults to zero
                for ``'gg'``.
            backscale: Ratio scaling ``bcts`` into the source region;
                only meaningful for ``'pp'``.
            exp: Per-bin exposure times; defaults to bin widths.
            type: Noise model: ``'pg'`` (Poisson source + Gaussian
                background), ``'pp'`` (Poisson source + Poisson
                background), or ``'gg'`` (Gaussian net counts, no separate
                background). Defaults to ``'pg'``.

        Raises:
            ValueError: If ``type`` is invalid, or a required background
                or error array is missing given ``type``.
            TypeError: If ``exp`` and ``bins`` have mismatched sizes or any
                exposure exceeds its bin width.
        """

        self.cts = np.asarray(cts, dtype=float)
        self.bins = np.asarray(bins, dtype=float)

        if len(pulse) > 0 and not isinstance(pulse[0], (list, tuple, np.ndarray)):
            pulse = [pulse]
        self.pulse = [list(p) for p in pulse]

        self.backscale = backscale

        self.lbins = self.bins[:-1]
        self.rbins = self.bins[1:]
        self.binsize = self.rbins - self.lbins
        self.exp = self.binsize if exp is None else np.asarray(exp, dtype=float)

        if (self.exp.size + 1) != self.bins.size:
            raise TypeError('expected size(exp) + 1 = size(bins)')
        if not (self.exp <= self.binsize).all():
            raise TypeError('expected exp <= binsize')

        self.type = type
        self.cts_err, self.bcts, self.bcts_err = validate_input(
            type, self.cts, cts_err, bcts, bcts_err
        )

        nan_mask = np.isnan(self.cts) | np.isnan(self.bcts)
        self.gap_int = None
        if nan_mask.any():
            raw_nan = [
                [float(self.bins[i]), float(self.bins[i + 1])] for i in np.where(nan_mask)[0]
            ]
            self.gap_int = union(raw_nan)

        self.ncts = self.cts - self.bcts * self.backscale
        self.ncts_err = np.sqrt(self.cts_err**2 + (self.bcts_err * self.backscale) ** 2)

        self.time = (self.lbins + self.rbins) / 2
        self.rate = self.cts / self.exp
        self.bak = self.bcts * self.backscale / self.exp
        self.net = self.ncts / self.exp

        self.txx_res = None

    @classmethod
    def from_pgsignal(cls, signal):
        """Wrap an already-processed :class:`~heapy.auto.signal.pgSignal` instance.

        Args:
            signal: A :class:`~heapy.auto.signal.pgSignal` instance whose
                ``sorting`` has already run (``sort_res`` populated), so
                ``signal.pulse`` is known.

        Returns:
            A new instance of ``cls`` (``Txx`` or a subclass).

        Raises:
            TypeError: If ``signal`` is not a ``pgSignal`` instance.
            RuntimeError: If ``signal.sort_res`` is ``None`` (pulse not
                yet located).
        """

        if not isinstance(signal, pgSignal):
            raise TypeError('expected signal to be a pgSignal instance')
        if signal.sort_res is None:
            raise RuntimeError('signal has no pulse yet; run loop()/sorting() first')

        inst = cls.__new__(cls)
        Txx.__init__(
            inst,
            signal.cts,
            signal.bins,
            signal.pulse,
            bcts=signal.bcts,
            bcts_err=signal.bcts_err,
            exp=signal.exp,
            type='pg',
        )

        return inst

    @classmethod
    def from_ppsignal(cls, signal):
        """Wrap an already-processed :class:`~heapy.auto.signal.ppSignal` instance.

        Args:
            signal: A :class:`~heapy.auto.signal.ppSignal` instance whose
                ``sorting`` has already run (``sort_res`` populated), so
                ``signal.pulse`` is known.

        Returns:
            A new instance of ``cls`` (``Txx`` or a subclass).

        Raises:
            TypeError: If ``signal`` is not a ``ppSignal`` instance.
            RuntimeError: If ``signal.sort_res`` is ``None`` (pulse not
                yet located).
        """

        if not isinstance(signal, ppSignal):
            raise TypeError('expected signal to be a ppSignal instance')
        if signal.sort_res is None:
            raise RuntimeError('signal has no pulse yet; run loop()/sorting() first')

        inst = cls.__new__(cls)
        Txx.__init__(
            inst,
            signal.cts,
            signal.bins,
            signal.pulse,
            bcts=signal.bcts,
            backscale=signal.backscale,
            exp=signal.exp,
            type='pp',
        )

        return inst

    @classmethod
    def from_ggsignal(cls, signal):
        """Wrap an already-processed :class:`~heapy.auto.signal.ggSignal` instance.

        Args:
            signal: A :class:`~heapy.auto.signal.ggSignal` instance whose
                ``sorting`` has already run (``sort_res`` populated), so
                ``signal.pulse`` is known.

        Returns:
            A new instance of ``cls`` (``Txx`` or a subclass).

        Raises:
            TypeError: If ``signal`` is not a ``ggSignal`` instance.
            RuntimeError: If ``signal.sort_res`` is ``None`` (pulse not
                yet located).
        """

        if not isinstance(signal, ggSignal):
            raise TypeError('expected signal to be a ggSignal instance')
        if signal.sort_res is None:
            raise RuntimeError('signal has no pulse yet; run loop()/sorting() first')

        inst = cls.__new__(cls)
        Txx.__init__(
            inst,
            signal.ncts,
            signal.bins,
            signal.pulse,
            cts_err=signal.ncts_err,
            exp=signal.exp,
            type='gg',
        )

        return inst

    def generate_mc_simulation(self, nmc, random_seed=450001):
        """Generate Monte Carlo realisations of the net count light curve.

        The sampling model is selected by ``self.type``:

        - ``'pg'``: Poisson source counts (``cts``) + Gaussian background
          (``bcts``, ``bcts_err``).
        - ``'pp'``: independent Poisson source (``cts``) and background
          (``bcts``) counts, background scaled by ``backscale``.
        - ``'gg'``: Gaussian net counts directly (``ncts``, ``ncts_err``);
          no separate background.

        Populates ``self.mc_ncts`` with the observed data in row 0. Gap
        bins (recorded on ``self.gap_int``) are marked ``NaN`` across all
        rows.

        Args:
            nmc: Number of Monte Carlo realisations to generate.
            random_seed: Seed for the local RNG used to draw samples.
                Default ensures reproducibility across runs; pass
                ``None`` for OS entropy.

        Raises:
            ValueError: If ``self.type`` is not one of ``'pg'``, ``'pp'``,
                ``'gg'``.
        """

        self.nmc = int(nmc)
        self.nsample = len(self.time)
        rng = np.random.default_rng(random_seed)

        sample = generate_mc_sample(
            self.type,
            self.cts,
            self.cts_err,
            self.bcts,
            self.bcts_err,
            self.nmc,
            rng,
            backscale=self.backscale,
        )
        self.mc_ncts = np.vstack([self.ncts, sample])

        # Mark gap bins as NaN across all MC rows so :meth:`calculate`'s
        # ``np.nancumsum`` treats them as no-contribution rather than a
        # spurious sampled value at a position with no real data.
        if self.gap_int:
            gap_idx = indices_in_intervals(self.lbins, self.rbins, self.gap_int)
            self.mc_ncts[:, gap_idx] = np.nan

    def calculate(self, xx=0.9, pulse=None, lbkg=None, rbkg=None):
        """Compute Txx duration and its uncertainties for each detected pulse.

        Optionally overrides the pulse interval(s) given at construction.
        Uncertainties are estimated via 1000 Monte Carlo realisations (see
        :meth:`generate_mc_simulation`). Results are printed to stdout and stored
        in ``self.txx_res``.

        Args:
            xx: Cumulative count fraction defining the duration, e.g. ``0.9``
                for T90 or ``0.5`` for T50.
            pulse: Override pulse interval(s). Either a single ``[pstart,
                pstop]`` pair or a list of such pairs (``[]`` clears the
                pulse to "none detected"); ``None`` keeps the existing
                ``self.pulse``.
            lbkg: Length of the background window to the left of the first
                pulse, in the same time units as the light curve. ``None``
                extends to the beginning of the data.
            rbkg: Length of the background window to the right of the last
                pulse. ``None`` extends to the end of the data.

        Returns:
            ``False`` if no pulse is detected; ``None`` on success (results
            are stored as instance attributes).
        """
        self.xx = xx

        if getattr(self, 'pulse', None) is None:
            self.find_pulse()
        detected_pulse = [list(p) for p in self.pulse]

        user_pulse = None
        if pulse is not None:
            if len(pulse) > 0 and not isinstance(pulse[0], (list, tuple, np.ndarray)):
                pulse = [pulse]
            user_pulse = [list(p) for p in pulse]
            self.pulse = user_pulse

        self.pstart = np.sort([p[0] for p in self.pulse])
        self.pstop = np.sort([p[1] for p in self.pulse])

        if len(self.pstart) == 0:
            msg = 'there is no pulse'
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False

        lbkg = np.inf if lbkg is None else lbkg
        rbkg = np.inf if rbkg is None else rbkg

        tmin = max(self.pstart[0] - lbkg, self.time[0])
        tmax = min(self.pstop[-1] + rbkg, self.time[-1])

        self.tindex = np.where((self.time >= tmin) & (self.time <= tmax))[0]

        # NaN-aware cumsum so gap bins (NaN in self.ncts) contribute zero
        # without propagating NaN through the rest of the cumulative curve.
        self.ccts = np.nancumsum(self.ncts)

        self.generate_mc_simulation(1000)

        mc_csf, mc_csf1, mc_csf2 = [], [], []
        mc_txx, mc_txx1, mc_txx2 = [], [], []

        for ncts in self.mc_ncts:
            ccts = np.nancumsum(ncts)

            txx, txx1, txx2, csf, csf1, csf2 = calculate_txx(
                self.time[self.tindex],
                ccts[self.tindex],
                self.pstart,
                self.pstop,
                self.xx,
                simple_err=False,
            )

            mc_csf.append(csf)
            mc_csf1.append(csf1)
            mc_csf2.append(csf2)
            mc_txx.append(txx)
            mc_txx1.append(txx1)
            mc_txx2.append(txx2)

        self.csf, self.csf1, self.csf2 = mc_csf[0], mc_csf1[0], mc_csf2[0]
        self.txx, self.txx1, self.txx2 = mc_txx[0], mc_txx1[0], mc_txx2[0]

        self.txx_err = get_mc_errors(np.array(mc_txx))
        self.txx1_err = get_mc_errors(np.array(mc_txx1))
        self.txx2_err = get_mc_errors(np.array(mc_txx2))

        self.txx_res = {
            'xx': self.xx,
            'txx': self.txx,
            'txx1': self.txx1,
            'txx2': self.txx2,
            'txx_err': self.txx_err,
            'txx1_err': self.txx1_err,
            'txx2_err': self.txx2_err,
            'csf': self.csf,
            'csf1': self.csf1,
            'csf2': self.csf2,
            'time': self.time,
            'ccts': self.ccts,
            'detected_pulse': detected_pulse,
            'user_pulse': user_pulse,
            'lbkg': lbkg,
            'rbkg': rbkg,
        }

        XX = int(self.xx * 100)

        msg = [
            f'{"id#":<5}{f"T{XX}":<10}{f"T{XX}-":<8}{f"T{XX}+":<8}{f"T{XX}1":<8}{f"T{XX}2":<8}'
        ] + [
            f'{i + 1:<5d}{t:<10.3f}{t_err[0]:<8.3f}{t_err[1]:<8.3f}{t1:<8.3f}{t2:<8.3f}'
            for i, (t, t_err, t1, t2) in enumerate(
                zip(
                    self.txx_res['txx'],
                    self.txx_res['txx_err'],
                    self.txx_res['txx1'],
                    self.txx_res['txx2'],
                    strict=False,
                )
            )
        ]
        print(format_message(msg))

    def save(self, savepath):
        """Save Txx results and diagnostic plots to disk.

        Serialises ``self.txx_res`` (whose ``'pulse'`` key holds the
        pulse intervals) as a JSON file and renders the two-panel
        diagnostic figure via
        :class:`~heapy.temp.temp_utils.TxxPlotter`. Gap intervals recorded
        on ``self.gap_int`` are masked in the plot. The primary curve is
        the net rate for ``type='gg'`` (already background-subtracted)
        and the total rate with background overlay otherwise.

        Args:
            savepath: Directory path where output files are written; created
                if it does not exist.

        Returns:
            ``False`` if no pulse has been detected; ``None`` on success.
        """

        if len(self.pstart) == 0:
            msg = 'there is no pulse'
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.txx_res, savepath + '/txx_res.json')

        with plt_rc_context():
            fig = TxxPlotter()
            if self.gap_int:
                fig.set_gaps(self.gap_int, self.lbins, self.rbins)
            if self.type == 'gg':
                fig.plot_curve(self.time, self.net)
            else:
                fig.plot_curve(self.time, self.rate, bak=self.bak)
            fig.plot_ccts(self.time, self.ccts, self.tindex)
            fig.plot_txx(self.txx1, self.txx2, self.csf, self.csf1, self.csf2)
            fig.save(savepath + '/txx.pdf')


class pgTxx(Txx):
    """Compute Txx durations for a Poisson-source/Gaussian-background light curve.

    Lazy two-phase convenience subclass: ``__init__`` builds a
    ``pgSignal`` from raw event times; :meth:`find_pulse` runs its
    pipeline (polynomial background fit + Bayesian-blocks pulse
    detection) and completes :class:`Txx`'s own state (``type='pg'``).
    """

    def __init__(self, ts, bins, exp=None, ignore=None):
        """Initialize pgTxx with time-tagged event data and binning.

        Args:
            ts: Array of event arrival times.
            bins: Bin edges or bin width used to build the light curve.
            exp: Exposure correction array, or ``None`` for uniform exposure.
            ignore: Time intervals to exclude from the background fit, or
                ``None`` to use all data.
        """

        self._signal = pgSignal(ts, bins, exp=exp, ignore=ignore)
        self.pulse = None
        self.txx_res = None

    @classmethod
    def frombin(cls, cts, bins, exp=None, ignore=None, random_seed=450001):
        """Build a pgTxx from a pre-binned counts histogram; see ``pgSignal.frombin``."""

        inst = cls.__new__(cls)
        inst._signal = pgSignal.frombin(cts, bins, exp=exp, ignore=ignore, random_seed=random_seed)
        inst.pulse = None
        inst.txx_res = None

        return inst

    @classmethod
    def from_components(cls, obj_list):
        """Stack polyfit'd pgSignal components; see ``pgSignal.from_components``."""

        inst = cls.__new__(cls)
        inst._signal = pgSignal.from_components(obj_list)
        inst.pulse = None
        inst.txx_res = None

        return inst

    def find_pulse(self, p0=0.05, sigma=3, deg=None, mp=True):
        """Run the wrapped pgSignal's pipeline and adopt its detected pulse.

        Runs the background polynomial fit if not already done (via
        ``loop``), or re-thresholds an existing segmentation via
        ``sorting`` without recomputing it. Then (re)initializes this
        instance's own :class:`Txx` state from the signal's ``cts``,
        ``bcts``, ``bcts_err``, and ``pulse``.

        Args:
            p0: Bayesian-blocks prior probability for a new change point.
            sigma: Minimum SNR threshold for a block to be classified as
                a pulse.
            deg: Polynomial degree for the background fit, or ``None`` to
                select automatically.
            mp: When ``True``, keep multiple separate pulse intervals.
                When ``False``, merge all intervals into one and emit a
                warning if more than one interval is found.
        """

        sig = self._signal
        if sig.sort_res is None:
            sig.loop(p0=p0, sigma=sigma, deg=deg, mp=mp)
        else:
            sig.sorting(sigma=sigma, mp=mp)

        Txx.__init__(
            self,
            sig.cts,
            sig.bins,
            sig.pulse,
            bcts=sig.bcts,
            bcts_err=sig.bcts_err,
            exp=sig.exp,
            type='pg',
        )


class ppTxx(Txx):
    """Compute Txx durations for a Poisson-source/Poisson-background light curve.

    Lazy two-phase convenience subclass: ``__init__`` builds a
    ``ppSignal`` from raw event times; :meth:`find_pulse` runs its
    pipeline (Bayesian-blocks pulse detection) and completes
    :class:`Txx`'s own state (``type='pp'``).
    """

    def __init__(self, ts, bts, bins, backscale=1, exp=None):
        """Initialize ppTxx with source and background event arrays.

        Args:
            ts: Array of source event arrival times.
            bts: Array of background event arrival times.
            bins: Bin edges or bin width used to build the light curve.
            backscale: Ratio of the source to background region size used
                for background scaling.
            exp: Exposure correction array, or ``None`` for uniform exposure.
        """

        self._signal = ppSignal(ts, bts, bins, backscale=backscale, exp=exp)
        self.pulse = None
        self.txx_res = None

    @classmethod
    def frombin(cls, cts, bcts, bins, backscale=1, exp=None, random_seed=450001):
        """Build a ppTxx from pre-binned histograms; see ``ppSignal.frombin``."""

        inst = cls.__new__(cls)
        inst._signal = ppSignal.frombin(
            cts, bcts, bins, backscale=backscale, exp=exp, random_seed=random_seed
        )
        inst.pulse = None
        inst.txx_res = None

        return inst

    def find_pulse(self, p0=0.05, sigma=3, mp=True):
        """Run the wrapped ppSignal's pipeline and adopt its detected pulse.

        Args:
            p0: Bayesian-blocks prior probability for a new change point.
            sigma: Minimum SNR threshold for a block to be classified as
                a pulse.
            mp: When ``True``, keep multiple separate pulse intervals.
                When ``False``, merge all intervals into one and emit a
                warning if more than one interval is found.
        """

        sig = self._signal
        if sig.sort_res is None:
            sig.loop(p0=p0, sigma=sigma, mp=mp)
        else:
            sig.sorting(sigma=sigma, mp=mp)

        Txx.__init__(
            self,
            sig.cts,
            sig.bins,
            sig.pulse,
            bcts=sig.bcts,
            backscale=sig.backscale,
            exp=sig.exp,
            type='pp',
        )


class ggTxx(Txx):
    """Compute Txx durations for a Gaussian-source/Gaussian-background light curve.

    Lazy two-phase convenience subclass: ``__init__`` builds a
    ``ggSignal`` from pre-background-subtracted counts; :meth:`find_pulse`
    runs its pipeline (Bayesian-blocks pulse detection) and completes
    :class:`Txx`'s own state (``type='gg'``).
    """

    def __init__(self, ncts, ncts_err, bins, exp=None):
        """Initialize ggTxx with pre-background-subtracted count data.

        Args:
            ncts: Array of net (background-subtracted) counts per bin.
            ncts_err: Array of uncertainties on ``ncts``.
            bins: Bin edges or bin width used to build the light curve.
            exp: Exposure correction array, or ``None`` for uniform exposure.
        """

        self._signal = ggSignal(ncts, ncts_err, bins, exp=exp)
        self.pulse = None
        self.txx_res = None

    def find_pulse(self, p0=0.05, sigma=3, mp=True):
        """Run the wrapped ggSignal's pipeline and adopt its detected pulse.

        Args:
            p0: Bayesian-blocks prior probability for a new change point.
            sigma: Minimum SNR threshold for a block to be classified as
                a pulse.
            mp: When ``True``, keep multiple separate pulse intervals.
                When ``False``, merge all intervals into one and emit a
                warning if more than one interval is found.
        """

        sig = self._signal
        if sig.sort_res is None:
            sig.loop(p0=p0, sigma=sigma, mp=mp)
        else:
            sig.sorting(sigma=sigma, mp=mp)

        Txx.__init__(
            self,
            sig.ncts,
            sig.bins,
            sig.pulse,
            cts_err=sig.ncts_err,
            exp=sig.exp,
            type='gg',
        )
