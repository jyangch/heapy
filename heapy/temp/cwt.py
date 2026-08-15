"""Continuous-wavelet spectrum diagnostics for gamma-ray burst light curves.

This module implements the CWT spectrum workflow used by
``giacomov/mvts`` / Vianello et al. (2018), kept separate from
:mod:`~heapy.temp.mvt` because the output is a wavelet spectrum plus a
null-envelope ``t_mv`` pick, not the Haar MVT scalar estimator.

Architecture mirrors :mod:`~heapy.temp.txx` and :mod:`~heapy.temp.mvt`:
:class:`CWT` is a self-contained engine that takes plain per-bin count
arrays plus a ``type`` tag (``'pg'``, ``'pp'``, or ``'gg'``) selecting the
noise model. The ``pgCWT`` / ``ppCWT`` / ``ggCWT`` convenience classes
bridge from the existing Signal data regimes.
"""

import os

import numpy as np

from ..auto.signal import ggSignal, pgSignal, ppSignal
from ..util.tools import format_message, json_dump, plt_rc_context
from .temp_utils import (
    CwtPlotter,
    calculate_cwt_spectrum,
    estimate_cwt_tmv,
    generate_mc_sample,
    resolve_analysis_window,
    resolve_time_grid,
    validate_input,
)


class CWT:
    """Compute a CWT global wavelet spectrum and ``t_mv`` estimate.

    Self-contained: takes per-bin source counts for ``'pg'``/``'pp'`` or
    net counts for ``'gg'``, plus the background/error arrays needed by
    the selected noise model. The observed spectrum is computed from the
    background-subtracted net-count series ``ncts``.

    Attributes:
        tmv_res: Dict storing the spectrum-threshold ``t_mv`` result
            after :meth:`calculate`.
    """

    def __init__(
        self,
        cts,
        bins=None,
        bcts=None,
        cts_err=None,
        bcts_err=None,
        backscale=1,
        type='pg',
        dt=None,
        time=None,
    ):
        """Initialize CWT with uniformly sampled count data.

        Args:
            cts: Source counts per bin for ``'pg'``/``'pp'``; net counts
                per bin for ``'gg'``.
            bins: Optional bin edges. A scalar is treated as legacy
                positional ``dt`` for compatibility.
            bcts: Background counts per bin; required for ``'pg'``/``'pp'``.
            cts_err: Count errors for ``cts``; required for ``'gg'`` and
                otherwise defaults to ``sqrt(cts)``.
            bcts_err: Background count errors; required for ``'pg'``.
            backscale: Ratio scaling ``bcts`` into the source region;
                meaningful for ``'pp'``.
            type: Noise model: ``'pg'``, ``'pp'``, or ``'gg'``.
            dt: Optional bin width in seconds.
            time: Optional bin-center times.
        """

        self.cts = np.asarray(cts, dtype=float)

        if self.cts.ndim != 1:
            raise ValueError('cts must be one-dimensional')

        self.type = type
        self.backscale = backscale

        self.dt, self.time, self.bins = resolve_time_grid(
            self.cts.size, bins=bins, dt=dt, time=time
        )

        self.cts_err, self.bcts, self.bcts_err = validate_input(
            type, self.cts, cts_err, bcts, bcts_err
        )

        self.ncts = self.cts - self.bcts * self.backscale
        self.ncts_err = np.sqrt(self.cts_err**2 + (self.bcts_err * self.backscale) ** 2)

        n_bad = int(np.count_nonzero(~np.isfinite(self.ncts) | ~np.isfinite(self.ncts_err)))
        if n_bad > 0:
            raise ValueError(
                f'ncts/ncts_err must be finite; found {n_bad} non-finite bin(s). '
                'The CWT spectrum has no gap support -- trim to a contiguous, '
                'gap-free window before constructing CWT.'
            )

        self.tmv_res = None

    @classmethod
    def from_pgsignal(cls, signal):
        """Wrap an already-background-fit :class:`~heapy.auto.signal.pgSignal`."""

        if not isinstance(signal, pgSignal):
            raise TypeError('expected signal to be a pgSignal instance')
        if signal.poly_res is None:
            raise RuntimeError('pgSignal has no background fit yet; run polyfit()/loop() first')

        inst = cls.__new__(cls)
        CWT.__init__(
            inst,
            signal.cts,
            signal.bins,
            bcts=signal.bcts,
            bcts_err=signal.bcts_err,
            type='pg',
        )

        return inst

    @classmethod
    def from_ppsignal(cls, signal):
        """Wrap a :class:`~heapy.auto.signal.ppSignal` instance."""

        if not isinstance(signal, ppSignal):
            raise TypeError('expected signal to be a ppSignal instance')

        inst = cls.__new__(cls)
        CWT.__init__(
            inst,
            signal.cts,
            signal.bins,
            bcts=signal.bcts,
            backscale=signal.backscale,
            type='pp',
        )

        return inst

    @classmethod
    def from_ggsignal(cls, signal):
        """Wrap a :class:`~heapy.auto.signal.ggSignal` instance."""

        if not isinstance(signal, ggSignal):
            raise TypeError('expected signal to be a ggSignal instance')

        inst = cls.__new__(cls)
        CWT.__init__(inst, signal.ncts, signal.bins, cts_err=signal.ncts_err, type='gg')

        return inst

    def generate_mc_simulation(self, nmc, random_seed=450001):
        """Generate Monte Carlo realisations of the net counts.

        Populates ``self.mc_ncts`` with the observed net-count light curve
        in row 0, followed by null-model realisations used to build the
        CWT background envelope.
        """

        self.nmc = int(nmc)
        rng = np.random.default_rng(random_seed)
        if self.type == 'pg' or self.type == 'pp':
            cts0 = np.clip(self.bcts * self.backscale, 0.0, None)
            bcts0 = self.bcts
            cts_err0 = np.sqrt(np.clip(cts0, 0.0, None))
            bcts_err0 = self.bcts_err
        elif self.type == 'gg':
            cts0 = np.zeros_like(self.cts)
            bcts0 = np.zeros_like(self.cts)
            cts_err0 = self.cts_err
            bcts_err0 = np.zeros_like(self.cts)
        else:
            raise ValueError(f'unknown type {self.type!r}')

        ncts_sample = generate_mc_sample(
            self.type,
            cts0,
            cts_err0,
            bcts0,
            bcts_err0,
            self.nmc,
            rng,
            backscale=self.backscale,
        )
        self.mc_ncts = np.vstack([self.ncts, ncts_sample])

    def calculate(
        self,
        twin=None,
        confidence=0.99,
        min_consecutive=2,
        **kwargs,
    ):
        """Compute the observed CWT spectrum and null-envelope ``t_mv``.

        Args:
            twin: Optional ``[t1, t2]`` analysis window. ``None`` uses the
                full light curve.
            confidence: Null-envelope central containment probability.
            min_consecutive: Adjacent scales above the upper envelope
                required for a detection.
            **kwargs: Forwarded to
                :func:`~heapy.temp.temp_utils.calculate_cwt_spectrum`.

        Returns:
            The result dict (also stored on ``self.tmv_res``).
        """

        analysis_index, analysis_window = resolve_analysis_window(self.time, twin)

        self.generate_mc_simulation(1000)
        mc_ncts = self.mc_ncts[:, analysis_index]

        spectrum = None
        null_spectrum_power = []
        spectrum_kwargs = dict(kwargs)
        for i, ncts_i in enumerate(mc_ncts):
            spectrum_i = calculate_cwt_spectrum(ncts_i, self.dt, **spectrum_kwargs)
            if i == 0:
                spectrum = spectrum_i
            else:
                null_spectrum_power.append(np.asarray(spectrum_i['spectrum_power'], dtype=float))

        spectrum_res = {
            key: val
            for key, val in spectrum.items()
            if key not in ('wave', 'power', 'fft', 'fftfreqs', 'freqs', 'coi')
        }

        period = np.asarray(spectrum_res['period'], dtype=float)
        spectrum_power = np.asarray(spectrum_res['spectrum_power'], dtype=float)
        msg = [
            f'{"method":<10}{"type":<8}{"nscale":<10}{"dt (s)":<12}',
            f'{"cwt":<10}{self.type:<8}{len(period):<10d}{self.dt:<12.6g}',
        ]
        print(format_message(msg))

        confidence = float(confidence)
        if not 0 < confidence < 1:
            raise ValueError('confidence must be between 0 and 1')

        null_spectrum_power = np.asarray(null_spectrum_power, dtype=float)

        alpha = 0.5 * (1.0 - confidence) * 100.0
        bg_lower, bg_median, bg_upper = np.percentile(
            null_spectrum_power, [alpha, 50.0, 100.0 - alpha], axis=0
        )

        pick = estimate_cwt_tmv(
            period,
            spectrum_power,
            bg_upper,
            min_consecutive=min_consecutive,
        )

        self.tmv_res = {
            'method': 'cwt',
            'tmv': pick['tmv'],
            'is_upper_limit': pick['is_upper_limit'],
            'quality': pick['quality'],
            'spectrum': spectrum_res,
            'confidence': confidence,
            'min_consecutive': int(min_consecutive),
            'analysis_window': analysis_window,
            'diag': {
                'period': period,
                'spectrum_power': spectrum_power,
                'bg_lower': bg_lower,
                'bg_median': bg_median,
                'bg_upper': bg_upper,
                'excess_mask': pick['excess_mask'],
                'first_index': pick['first_index'],
            },
        }

        tmv_text = 'nan' if not np.isfinite(pick['tmv']) else f'{pick["tmv"]:.6g}'
        msg = [
            f'{"tmv (s)":<15}{"quality":<15}{"upper_limit":<15}',
            f'{tmv_text:<15}{pick["quality"]:<15}{pick["is_upper_limit"]!s:<15}',
            f'null_model={self.type}, confidence={confidence:.3g}',
        ]
        print(format_message(msg))

        return self.tmv_res

    def save(self, savepath):
        """Save CWT results and diagnostic plot to disk."""

        if self.tmv_res is None:
            raise RuntimeError('call .calculate() before .save(...)')

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.tmv_res, os.path.join(savepath, 'tmv_res.json'))

        with plt_rc_context():
            fig = CwtPlotter()
            fig.plot_curve(self.time, self.ncts)
            fig.plot_analysis_window(self.tmv_res['analysis_window'])
            fig.plot_spectrum(self.tmv_res)
            fig.save(os.path.join(savepath, 'cwt.pdf'))


class pgCWT(CWT):
    """Compute CWT spectra for a Poisson-source/Gaussian-background light curve."""

    def __init__(self, ts, bins, exp=None, ignore=None):
        """Initialize pgCWT with event data and defer background fitting."""

        self._signal = pgSignal(ts, bins, exp=exp, ignore=ignore)
        self.tmv_res = None

    @classmethod
    def frombin(cls, cts, bins, exp=None, ignore=None, random_seed=450001):
        """Build a pgCWT from pre-binned counts; see ``pgSignal.frombin``."""

        inst = cls.__new__(cls)
        inst._signal = pgSignal.frombin(cts, bins, exp=exp, ignore=ignore, random_seed=random_seed)
        inst.tmv_res = None

        return inst

    @classmethod
    def from_components(cls, obj_list):
        """Stack polyfit'd pgSignal components; see ``pgSignal.from_components``."""

        inst = cls.__new__(cls)
        inst._signal = pgSignal.from_components(obj_list)
        inst.tmv_res = None
        inst.find_background()

        return inst

    def find_background(self, p0=0.05, sigma=3, deg=None):
        """Run the wrapped pgSignal background fit and complete CWT state."""

        sig = self._signal
        if sig.poly_res is None:
            sig.loop(p0=p0, sigma=sigma, deg=deg)

        CWT.__init__(
            self,
            sig.cts,
            sig.bins,
            bcts=sig.bcts,
            bcts_err=sig.bcts_err,
            type='pg',
        )

    def calculate(self, **kwargs):
        """Run background fitting if needed, then compute the CWT spectrum."""

        if not hasattr(self, 'cts'):
            self.find_background()
        return super().calculate(**kwargs)


class ppCWT(CWT):
    """Compute CWT spectra for a Poisson-source/Poisson-background light curve."""

    def __init__(self, ts, bts, bins, backscale=1, exp=None):
        """Initialize ppCWT with source/background event arrays."""

        self._signal = ppSignal(ts, bts, bins, backscale=backscale, exp=exp)
        CWT.__init__(
            self,
            self._signal.cts,
            self._signal.bins,
            bcts=self._signal.bcts,
            backscale=self._signal.backscale,
            type='pp',
        )

    @classmethod
    def frombin(cls, cts, bcts, bins, backscale=1, exp=None, random_seed=450001):
        """Build a ppCWT from pre-binned histograms; see ``ppSignal.frombin``."""

        inst = cls.__new__(cls)
        inst._signal = ppSignal.frombin(
            cts, bcts, bins, backscale=backscale, exp=exp, random_seed=random_seed
        )
        CWT.__init__(
            inst,
            inst._signal.cts,
            inst._signal.bins,
            bcts=inst._signal.bcts,
            backscale=inst._signal.backscale,
            type='pp',
        )

        return inst


class ggCWT(CWT):
    """Compute CWT spectra for a Gaussian net-count light curve."""

    def __init__(self, ncts, ncts_err, bins=None, exp=None, dt=None, time=None):
        """Initialize ggCWT with pre-background-subtracted counts."""

        bins = resolve_time_grid(len(ncts), bins=bins, dt=dt, time=time)[2]
        self._signal = ggSignal(ncts, ncts_err, bins, exp=exp)
        CWT.__init__(
            self,
            self._signal.ncts,
            bins=self._signal.bins,
            cts_err=self._signal.ncts_err,
            type='gg',
        )
