"""Minimum variability timescale via the nrbutler2025 Haar implementation.

This module intentionally contains one MVT algorithm. The scientific
core (``haar_denoise``, ``calculate_haar_power_spectrum``,
``calculate_haar_mvt``) lives in :mod:`~heapy.temp.temp_utils` -- it is a
faithful Python-3 port of the updated public code at
https://github.com/nrbutler/mvt, shared by Nat Butler for reproducing
Golkhou & Butler (2014) and Golkhou, Butler & Littlejohns (2015); do not
edit those three functions without re-diffing against the upstream
source.

Architecture mirrors :mod:`~heapy.temp.lag` / :mod:`~heapy.temp.txx`:
:class:`MVT` is a self-contained engine that takes a plain per-bin light
curve (``rate``, ``rate_err``, a scalar bin width ``dt``) with no
dependency on ``pgSignal``/``ppSignal``/``ggSignal``. Producing that
background-subtracted rate is not MVT's job -- that lives in the
polynomial background fit (``pgSignal.loop``) or the direct net-rate
construction (``ppSignal``/``ggSignal``).

Two construction paths bridge Signal-based background subtraction to
this primitive interface:

- :meth:`MVT.from_pgsignal` / :meth:`MVT.from_ppsignal` /
  :meth:`MVT.from_ggsignal` wrap an already-processed Signal instance and
  extract the plain ``rate``/``rate_err``/``dt`` arrays :class:`MVT`
  needs.
- ``pgMVT`` / ``ppMVT`` / ``ggMVT`` are convenience subclasses:
  ``pgMVT`` is a lazy two-phase wrapper (``__init__`` builds the
  raw-array ``pgSignal``, mirroring the historical direct-construction
  API, e.g. ``pgMVT(ts, bins)``, but defers completing :class:`MVT`'s
  own state until :meth:`pgMVT.find_background` runs the polynomial
  fit); ``ppMVT``/``ggMVT`` complete :class:`MVT`'s state eagerly since
  ``ppSignal``/``ggSignal`` need no fitting step.

Because the Haar transform is a pure cumulative-sum algorithm, it has no
notion of a data gap: :class:`MVT` requires a finite (no ``NaN``/``inf``)
``rate``/``rate_err`` and raises rather than silently propagating a gap
through every downstream scale.

Example:
    from heapy.temp.mvt import pgMVT
    mvt = pgMVT(ts, bins)
    mvt.calculate()
    mvt.save('/output/dir')

    # Or fully self-contained, given data you already have:
    from heapy.temp.mvt import MVT
    mvt = MVT(rate, rate_err, dt)
    mvt.calculate()
"""

import os

import numpy as np

from ..auto.signal import ggSignal, pgSignal, ppSignal
from ..util.tools import format_message, json_dump, plt_rc_context
from .temp_utils import MvtPlotter, calculate_haar_mvt, uniform_dt_from_bins


def _run_haar(ncts, ncts_err, bins, **kw):
    """Compatibility wrapper for older validation scripts.

    The Haar core expects rate/error/dt.  Older heapy diagnostics pass
    per-bin net counts and count errors with bin edges; for uniform bins
    this conversion is lossless up to the common scale factor.

    Args:
        ncts: Per-bin net (background-subtracted) counts.
        ncts_err: 1-sigma error on ``ncts``.
        bins: Bin edges (length ``len(ncts) + 1``); must be uniform.
        **kw: Forwarded to
            :func:`~heapy.temp.temp_utils.calculate_haar_mvt`.

    Returns:
        A result dict; see :meth:`MVT.calculate`.

    Raises:
        ValueError: If ``bins`` are not uniform, or ``ncts``/``ncts_err``
            don't match the bin count.
    """

    bins = np.asarray(bins, dtype=float)
    dt = uniform_dt_from_bins(bins)
    widths = np.diff(bins)
    ncts = np.asarray(ncts, dtype='float64')
    ncts_err = np.asarray(ncts_err, dtype='float64')
    if ncts.shape != widths.shape or ncts_err.shape != widths.shape:
        raise ValueError('ncts and ncts_err must match the bin count')
    rate = ncts / widths
    rate_err = ncts_err / widths

    mvt, mvt_err_lo, mvt_err_hi, is_upper_limit, diag = calculate_haar_mvt(rate, rate_err, dt, **kw)
    return {
        'method': 'haar',
        'mvt': mvt,
        'mvt_err_lo': mvt_err_lo,
        'mvt_err_hi': mvt_err_hi,
        'is_upper_limit': is_upper_limit,
        'diag': diag,
    }


class MVT:
    """Compute the Haar minimum variability timescale and its diagnostics.

    Self-contained: takes a plain per-bin ``rate``/``rate_err`` (already
    background-subtracted) and a scalar bin width ``dt``, with no
    dependency on ``pgSignal``/``ppSignal``/``ggSignal``. Producing that
    background-subtracted rate is a separate, prior concern -- see
    :meth:`from_pgsignal` / :meth:`from_ppsignal` / :meth:`from_ggsignal`,
    or the ``pgMVT`` / ``ppMVT`` / ``ggMVT`` convenience subclasses, for
    bridging from a Signal instance.

    Attributes:
        rate, rate_err: Background-subtracted count rate and its 1-sigma
            error, one entry per uniform time bin.
        dt: Bin width in seconds.
        time: Bin-center times used for diagnostic plotting.
        mvt_res: Dict with keys ``method``, ``mvt``, ``mvt_err_lo``,
            ``mvt_err_hi``, ``is_upper_limit``, ``diag``, or ``None``
            before :meth:`calculate` is called.
    """

    def __init__(self, rate, rate_err, dt, time=None):
        """Initialize MVT with a uniformly sampled, background-subtracted light curve.

        Args:
            rate: 1-D array of per-bin count rate, background-subtracted.
            rate_err: 1-sigma error on ``rate``; same shape as ``rate``.
            dt: Bin width in seconds; must be a positive finite scalar.
            time: Bin-center times for diagnostic plotting; defaults to
                ``dt * (arange(len(rate)) + 0.5)`` (bins starting at
                ``t=0``) when ``None``.

        Raises:
            ValueError: If ``rate``/``rate_err`` are not one-dimensional,
                differ in shape, have fewer than four bins, contain any
                non-finite value, or if ``dt`` is not a positive finite
                scalar.
        """

        self.rate = np.asarray(rate, dtype='float64')
        self.rate_err = np.asarray(rate_err, dtype='float64')

        if self.rate.ndim != 1 or self.rate_err.ndim != 1:
            raise ValueError('rate and rate_err must be one-dimensional')
        if self.rate.shape != self.rate_err.shape:
            raise ValueError('rate and rate_err must have matching shapes')
        if self.rate.size < 4:
            raise ValueError('at least four bins are required')

        n_bad = int(np.count_nonzero(~np.isfinite(self.rate) | ~np.isfinite(self.rate_err)))
        if n_bad > 0:
            raise ValueError(
                f'rate/rate_err must be finite; found {n_bad} non-finite bin(s). '
                'The Haar transform is a cumulative-sum algorithm with no gap '
                'support -- trim the light curve to a contiguous, gap-free '
                'window before constructing MVT.'
            )

        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('dt must be a positive finite scalar')

        self.time = (
            self.dt * (np.arange(self.rate.size) + 0.5)
            if time is None
            else np.asarray(time, dtype=float)
        )

        self.mvt_res = None

    @classmethod
    def from_pgsignal(cls, signal):
        """Wrap an already-processed :class:`~heapy.auto.signal.pgSignal` instance.

        Args:
            signal: A :class:`~heapy.auto.signal.pgSignal` instance whose
                polynomial background fit has already run
                (``polyfit``/``loop``/``from_components``), so
                ``signal.net``/``signal.net_err`` are populated.

        Returns:
            A new instance of ``cls`` (``MVT`` or a subclass).

        Raises:
            TypeError: If ``signal`` is not a ``pgSignal`` instance.
            RuntimeError: If ``signal`` has no background fit yet.
            ValueError: If ``signal``'s bins are not uniform in width.
        """

        if not isinstance(signal, pgSignal):
            raise TypeError('expected signal to be a pgSignal instance')
        if not hasattr(signal, 'net'):
            raise RuntimeError('signal has no background fit yet; run polyfit()/loop() first')

        return cls(signal.net, signal.net_err, uniform_dt_from_bins(signal.bins), time=signal.time)

    @classmethod
    def from_ppsignal(cls, signal):
        """Wrap an already-processed :class:`~heapy.auto.signal.ppSignal` instance.

        Args:
            signal: A :class:`~heapy.auto.signal.ppSignal` instance.

        Returns:
            A new instance of ``cls`` (``MVT`` or a subclass).

        Raises:
            TypeError: If ``signal`` is not a ``ppSignal`` instance.
            ValueError: If ``signal``'s bins are not uniform in width.
        """

        if not isinstance(signal, ppSignal):
            raise TypeError('expected signal to be a ppSignal instance')

        return cls(signal.net, signal.net_err, uniform_dt_from_bins(signal.bins), time=signal.time)

    @classmethod
    def from_ggsignal(cls, signal):
        """Wrap an already-processed :class:`~heapy.auto.signal.ggSignal` instance.

        Args:
            signal: A :class:`~heapy.auto.signal.ggSignal` instance.

        Returns:
            A new instance of ``cls`` (``MVT`` or a subclass).

        Raises:
            TypeError: If ``signal`` is not a ``ggSignal`` instance.
            ValueError: If ``signal``'s bins are not uniform in width, or
                ``signal.net`` has any gap (``NaN``) bins.
        """

        if not isinstance(signal, ggSignal):
            raise TypeError('expected signal to be a ggSignal instance')

        return cls(signal.net, signal.net_err, uniform_dt_from_bins(signal.bins), time=signal.time)

    def calculate(self, **kwargs):
        """Compute the Haar MVT for this light curve.

        Args:
            **kwargs: Forwarded to
                :func:`~heapy.temp.temp_utils.calculate_haar_mvt`
                (``tau_bg_max``, ``nrepl``, ``bin_fac``, ``afactor``,
                ``snr``, ``verbose``, ``weight``, ``drop_nonfinite``,
                ``file``). ``drop_nonfinite`` defaults to ``True`` for
                robust real-data analysis; pass ``False`` for exact
                upstream-unfiltered behavior.

        Returns:
            The result dict (also stored on ``self.mvt_res``).
        """

        mvt, mvt_err_lo, mvt_err_hi, is_upper_limit, diag = calculate_haar_mvt(
            self.rate, self.rate_err, self.dt, **kwargs
        )

        self.mvt_res = {
            'method': 'haar',
            'mvt': mvt,
            'mvt_err_lo': mvt_err_lo,
            'mvt_err_hi': mvt_err_hi,
            'is_upper_limit': is_upper_limit,
            'diag': diag,
        }

        msg = [
            f'{"mvt (s)":<15}{"mvt_le (s)":<15}{"mvt_he (s)":<15}',
            f'{mvt:<15.6g}{mvt_err_lo:<15.6g}{mvt_err_hi:<15.6g}',
            f'T_snr={diag["tsnr"]:.6g} s, T_beta={diag["tbeta"]:.6g} s, '
            f'is_upper_limit={is_upper_limit}',
        ]
        print(format_message(msg))

        return self.mvt_res

    def save(self, savepath, max_dt=100.0):
        """Save the MVT result and a diagnostic plot to disk.

        Serialises ``self.mvt_res`` as a JSON file and writes a two-panel
        PDF via :class:`~heapy.temp.temp_utils.MvtPlotter`: the light
        curve with the MVT marked, and the Haar scaleogram reproducing
        ``nrbutler/mvt``'s ``haar_power_mod`` ``doplot`` figure (see the
        upstream ``test_haar_mod.png``).

        Args:
            savepath: Directory path where output files are written;
                created if it does not exist.
            max_dt: Optional upstream ``haar_power_mod`` plotting
                parameter for the scaleogram; defaults to upstream's
                ``100.0``. Pass ``None`` to infer from the analysed time
                span.

        Raises:
            RuntimeError: If :meth:`calculate` has not been called yet.
        """

        if self.mvt_res is None:
            raise RuntimeError('call .calculate() before .save(...)')

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.mvt_res, os.path.join(savepath, 'mvt_res.json'))

        with plt_rc_context():
            fig = MvtPlotter()
            fig.plot_curve(self.time, self.rate)
            fig.plot_scaleogram(self.dt, self.time, self.mvt_res, max_dt=max_dt)
            fig.save(os.path.join(savepath, 'mvt.pdf'))


class pgMVT(MVT):
    """Compute the Haar MVT for a Poisson-source/Gaussian-background light curve.

    Lazy two-phase convenience subclass: ``__init__`` builds a
    ``pgSignal`` from raw event times; :meth:`find_background` runs its
    polynomial background fit and completes :class:`MVT`'s own state.
    """

    def __init__(self, ts, bins, exp=None, ignore=None):
        """Initialize pgMVT with time-tagged event data and binning.

        Args:
            ts: Array of event arrival times.
            bins: Bin edges or bin width used to build the light curve.
            exp: Exposure correction array, or ``None`` for uniform exposure.
            ignore: Time intervals to exclude from the background fit, or
                ``None`` to use all data.
        """

        self._signal = pgSignal(ts, bins, exp=exp, ignore=ignore)
        self.mvt_res = None

    @classmethod
    def frombin(cls, cts, bins, exp=None, ignore=None, random_seed=450001):
        """Build a pgMVT from a pre-binned counts histogram; see ``pgSignal.frombin``."""

        inst = cls.__new__(cls)
        inst._signal = pgSignal.frombin(cts, bins, exp=exp, ignore=ignore, random_seed=random_seed)
        inst.mvt_res = None

        return inst

    @classmethod
    def from_components(cls, obj_list):
        """Stack polyfit'd pgSignal components; see ``pgSignal.from_components``."""

        inst = cls.__new__(cls)
        inst._signal = pgSignal.from_components(obj_list)
        inst.mvt_res = None
        inst.find_background()

        return inst

    def find_background(self, p0=0.05, sigma=3, deg=None):
        """Run the wrapped pgSignal's background fit if not already done.

        Runs the polynomial background fit (via ``loop``) if the wrapped
        signal has no net rate yet, then (re)initializes this instance's
        own :class:`MVT` state from ``signal.net``/``signal.net_err``.

        Args:
            p0: Bayesian-blocks prior probability for a new change point.
            sigma: Minimum SNR threshold for a block to be classified as
                a pulse.
            deg: Polynomial degree for the background fit, or ``None`` to
                select automatically.
        """

        sig = self._signal
        if not hasattr(sig, 'net'):
            sig.loop(p0=p0, sigma=sigma, deg=deg)

        MVT.__init__(self, sig.net, sig.net_err, uniform_dt_from_bins(sig.bins), time=sig.time)

    def calculate(self, deg=None, **kw):
        """Run the background fit (if needed), then compute the Haar MVT; see ``MVT.calculate``."""

        if not hasattr(self, 'rate'):
            self.find_background(deg=deg)

        return super().calculate(**kw)


class ppMVT(MVT):
    """Compute the Haar MVT for a Poisson-source/Poisson-background light curve.

    Eager convenience subclass: ``ppSignal`` needs no fitting step, so
    ``__init__`` completes :class:`MVT`'s own state immediately.
    """

    def __init__(self, ts, bts, bins, backscale=1, exp=None):
        """Initialize ppMVT with source and background event arrays.

        Args:
            ts: Array of source event arrival times.
            bts: Array of background event arrival times.
            bins: Bin edges or bin width used to build the light curve.
            backscale: Ratio of the source to background region size used
                for background scaling.
            exp: Exposure correction array, or ``None`` for uniform exposure.
        """

        self._signal = ppSignal(ts, bts, bins, backscale=backscale, exp=exp)
        MVT.__init__(
            self,
            self._signal.net,
            self._signal.net_err,
            uniform_dt_from_bins(self._signal.bins),
            time=self._signal.time,
        )

    @classmethod
    def frombin(cls, cts, bcts, bins, backscale=1, exp=None, random_seed=450001):
        """Build a ppMVT from pre-binned histograms; see ``ppSignal.frombin``."""

        inst = cls.__new__(cls)
        inst._signal = ppSignal.frombin(
            cts, bcts, bins, backscale=backscale, exp=exp, random_seed=random_seed
        )
        MVT.__init__(
            inst,
            inst._signal.net,
            inst._signal.net_err,
            uniform_dt_from_bins(inst._signal.bins),
            time=inst._signal.time,
        )

        return inst


class ggMVT(MVT):
    """Compute the Haar MVT for a Gaussian net-count light curve.

    Eager convenience subclass: ``ggSignal`` needs no fitting step, so
    ``__init__`` completes :class:`MVT`'s own state immediately.

    Note:
        ``ggSignal`` may carry ``NaN`` gap bins; :class:`MVT` rejects
        non-finite input outright (the Haar transform has no gap
        support), so a gappy ``ggSignal`` raises ``ValueError`` here
        rather than silently corrupting the result.
    """

    def __init__(self, ncts, ncts_err, bins, exp=None):
        """Initialize ggMVT with pre-background-subtracted count data.

        Args:
            ncts: Array of net (background-subtracted) counts per bin.
            ncts_err: Array of uncertainties on ``ncts``.
            bins: Bin edges or bin width used to build the light curve.
            exp: Exposure correction array, or ``None`` for uniform exposure.

        Raises:
            ValueError: If the resulting net rate has any gap (``NaN``)
                bins, or non-uniform bin widths.
        """

        self._signal = ggSignal(ncts, ncts_err, bins, exp=exp)
        MVT.__init__(
            self,
            self._signal.net,
            self._signal.net_err,
            uniform_dt_from_bins(self._signal.bins),
            time=self._signal.time,
        )
