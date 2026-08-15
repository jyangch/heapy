"""Continuous-wavelet spectrum diagnostics for gamma-ray burst light curves.

This module implements the CWT spectrum workflow used by
``giacomov/mvts`` / Vianello et al. (2018), kept separate from
:mod:`~heapy.temp.mvt` because the output is a wavelet spectrum plus an
optional null-envelope ``t_mv`` pick, not the Haar MVT scalar estimator.

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
    uniform_dt_from_bins,
    validate_input,
)


class CWT:
    """Compute a CWT global wavelet spectrum and optional ``t_mv`` estimate.

    Self-contained: takes per-bin source counts for ``'pg'``/``'pp'`` or
    net counts for ``'gg'``, plus the background/error arrays needed by
    the selected noise model. The observed spectrum is computed from the
    background-subtracted net-count series ``ncts``.

    Attributes:
        cwt_res: Dict storing the observed CWT spectrum after
            :meth:`calculate`.
        tmv_res: Dict storing the spectrum-threshold ``t_mv`` result
            after :meth:`estimate_tmv`.
    """

    def __init__(
        self,
        cts,
        bins,
        bcts=None,
        cts_err=None,
        bcts_err=None,
        backscale=1,
        exp=None,
        type='pg',
    ):
        """Initialize CWT with uniformly sampled count data.

        Args:
            cts: Source counts per bin for ``'pg'``/``'pp'``; net counts
                per bin for ``'gg'``.
            bins: Bin edges (length ``len(cts) + 1``).
            bcts: Background counts per bin; required for ``'pg'``/``'pp'``.
            cts_err: Count errors for ``cts``; required for ``'gg'`` and
                otherwise defaults to ``sqrt(cts)``.
            bcts_err: Background count errors; required for ``'pg'``.
            backscale: Ratio scaling ``bcts`` into the source region;
                meaningful for ``'pp'``.
            exp: Per-bin exposure times; defaults to bin widths.
            type: Noise model: ``'pg'``, ``'pp'``, or ``'gg'``.
        """

        self.cts = np.asarray(cts, dtype=float)
        self.bins = np.asarray(bins, dtype=float)

        if self.cts.ndim != 1:
            raise ValueError('cts must be one-dimensional')
        if self.bins.ndim != 1 or self.bins.size != self.cts.size + 1:
            raise ValueError('expected size(bins) = size(cts)+1')

        self.type = type
        self.backscale = backscale

        self.lbins = self.bins[:-1]
        self.rbins = self.bins[1:]
        self.binsize = self.rbins - self.lbins
        self.exp = self.binsize if exp is None else np.asarray(exp, dtype=float)

        if (self.exp.size + 1) != self.bins.size:
            raise TypeError('expected size(exp) + 1 = size(bins)')
        if not (self.exp <= self.binsize).all():
            raise TypeError('expected exp <= binsize')

        self.cts_err, self.bcts, self.bcts_err = validate_input(
            type, self.cts, cts_err, bcts, bcts_err
        )
        self.dt = uniform_dt_from_bins(self.bins)

        self.ncts = self.cts - self.bcts * self.backscale
        self.ncts_err = np.sqrt(self.cts_err**2 + (self.bcts_err * self.backscale) ** 2)
        self.time = (self.lbins + self.rbins) / 2

        n_bad = int(np.count_nonzero(~np.isfinite(self.ncts) | ~np.isfinite(self.ncts_err)))
        if n_bad > 0:
            raise ValueError(
                f'ncts/ncts_err must be finite; found {n_bad} non-finite bin(s). '
                'The CWT spectrum has no gap support -- trim to a contiguous, '
                'gap-free window before constructing CWT.'
            )

        self.cwt_res = None
        self.tmv_res = None

    @classmethod
    def from_pgsignal(cls, signal):
        """Wrap an already-background-fit :class:`~heapy.auto.signal.pgSignal`."""

        if not isinstance(signal, pgSignal):
            raise TypeError('expected signal to be a pgSignal instance')
        if signal.poly_res is None:
            raise RuntimeError('pgSignal has no background fit yet; run polyfit()/loop() first')

        return cls(
            signal.cts,
            signal.bins,
            bcts=signal.bcts,
            bcts_err=signal.bcts_err,
            exp=signal.exp,
            type='pg',
        )

    @classmethod
    def from_ppsignal(cls, signal):
        """Wrap a :class:`~heapy.auto.signal.ppSignal` instance."""

        if not isinstance(signal, ppSignal):
            raise TypeError('expected signal to be a ppSignal instance')

        return cls(
            signal.cts,
            signal.bins,
            bcts=signal.bcts,
            backscale=signal.backscale,
            exp=signal.exp,
            type='pp',
        )

    @classmethod
    def from_ggsignal(cls, signal):
        """Wrap a :class:`~heapy.auto.signal.ggSignal` instance."""

        if not isinstance(signal, ggSignal):
            raise TypeError('expected signal to be a ggSignal instance')

        return cls(signal.ncts, signal.bins, cts_err=signal.ncts_err, exp=signal.exp, type='gg')

    def calculate(self, **kwargs):
        """Compute the observed CWT global spectrum.

        Args:
            **kwargs: Forwarded to
                :func:`~heapy.temp.temp_utils.calculate_cwt_spectrum`.

        Returns:
            The result dict (also stored on ``self.cwt_res``).
        """

        self.spectrum_kwargs = dict(kwargs)
        spectrum = calculate_cwt_spectrum(self.ncts, self.dt, **self.spectrum_kwargs)
        spectrum_public = {
            key: val
            for key, val in spectrum.items()
            if key not in ('wave', 'power', 'fft', 'fftfreqs', 'freqs', 'coi')
        }
        self.cwt_res = {
            'method': 'cwt',
            'type': self.type,
            'dt': self.dt,
            'time': self.time,
            'ncts': self.ncts,
            'ncts_err': self.ncts_err,
            'spectrum': spectrum_public,
        }

        period = np.asarray(self.cwt_res['spectrum']['period'], dtype=float)
        msg = [
            f'{"method":<10}{"type":<8}{"nscale":<10}{"dt (s)":<12}',
            f'{"cwt":<10}{self.type:<8}{len(period):<10d}{self.dt:<12.6g}',
        ]
        print(format_message(msg))

        return self.cwt_res

    def generate_null_sample(self, nmc, random_seed=450001):
        """Generate zero-signal net-count null realisations."""

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

        return generate_mc_sample(
            self.type,
            cts0,
            cts_err0,
            bcts0,
            bcts_err0,
            nmc,
            rng,
            backscale=self.backscale,
        )

    def estimate_tmv(
        self,
        nmc=1000,
        confidence=0.99,
        min_consecutive=2,
        random_seed=450001,
        **kwargs,
    ):
        """Estimate ``t_mv`` from the first CWT excess above a null envelope.

        Args:
            nmc: Number of null Monte Carlo realisations.
            confidence: Null-envelope central containment probability.
            min_consecutive: Adjacent scales above the upper envelope
                required for a detection.
            random_seed: Seed for deterministic MC sampling.
            **kwargs: Forwarded to :meth:`calculate` when the observed
                spectrum has not been computed yet, and to each null
                spectrum calculation.

        Returns:
            The result dict (also stored on ``self.tmv_res``).
        """

        if self.cwt_res is None or kwargs:
            self.calculate(**kwargs)
        else:
            kwargs = dict(getattr(self, 'spectrum_kwargs', {}))

        nmc = int(nmc)
        if nmc < 1:
            raise ValueError('nmc must be a positive integer')
        confidence = float(confidence)
        if not 0 < confidence < 1:
            raise ValueError('confidence must be between 0 and 1')

        samples = self.generate_null_sample(nmc, random_seed=random_seed)
        null_power = []
        for sample in samples:
            spectrum_i = calculate_cwt_spectrum(sample, self.dt, **kwargs)
            null_power.append(np.asarray(spectrum_i['spectrum_power'], dtype=float))
        null_power = np.asarray(null_power, dtype=float)

        alpha = 0.5 * (1.0 - confidence) * 100.0
        bg_lo, bg_median, bg_hi = np.percentile(null_power, [alpha, 50.0, 100.0 - alpha], axis=0)

        spectrum = self.cwt_res['spectrum']
        period = np.asarray(spectrum['period'], dtype=float)
        power = np.asarray(spectrum['spectrum_power'], dtype=float)
        pick = estimate_cwt_tmv(
            period,
            power,
            bg_hi,
            min_consecutive=min_consecutive,
        )

        self.tmv_res = {
            'method': 'cwt',
            'tmv': pick['tmv'],
            'tmv_err_lo': 0.0,
            'tmv_err_hi': 0.0,
            'is_upper_limit': pick['is_upper_limit'],
            'quality': pick['quality'],
            'null_model': self.type,
            'confidence': confidence,
            'nmc': nmc,
            'min_consecutive': int(min_consecutive),
            'diag': {
                'period': period,
                'spectrum_power': power,
                'bg_lo': bg_lo,
                'bg_median': bg_median,
                'bg_hi': bg_hi,
                'excess_mask': pick['excess_mask'],
                'first_index': pick['first_index'],
            },
        }

        tmv_text = 'nan' if not np.isfinite(pick['tmv']) else f'{pick["tmv"]:.6g}'
        msg = [
            f'{"tmv (s)":<15}{"quality":<15}{"upper_limit":<15}',
            f'{tmv_text:<15}{pick["quality"]:<15}{pick["is_upper_limit"]!s:<15}',
            f'null_model={self.type}, confidence={confidence:.3g}, nmc={nmc:d}',
        ]
        print(format_message(msg))

        return self.tmv_res

    def save(self, savepath):
        """Save CWT results and diagnostic plot to disk."""

        if self.cwt_res is None:
            raise RuntimeError('call .calculate() before .save(...)')

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        payload = dict(self.cwt_res)
        payload['tmv_res'] = self.tmv_res
        json_dump(payload, os.path.join(savepath, 'cwt_res.json'))

        with plt_rc_context():
            fig = CwtPlotter()
            fig.plot_curve(self.time, self.ncts)
            fig.plot_spectrum(self.cwt_res, tmv_res=self.tmv_res)
            fig.save(os.path.join(savepath, 'cwt.pdf'))


class pgCWT(CWT):
    """Compute CWT spectra for a Poisson-source/Gaussian-background light curve."""

    def __init__(self, ts, bins, exp=None, ignore=None):
        """Initialize pgCWT with event data and defer background fitting."""

        self._signal = pgSignal(ts, bins, exp=exp, ignore=ignore)
        self.cwt_res = None
        self.tmv_res = None

    @classmethod
    def frombin(cls, cts, bins, exp=None, ignore=None, random_seed=450001):
        """Build a pgCWT from pre-binned counts; see ``pgSignal.frombin``."""

        inst = cls.__new__(cls)
        inst._signal = pgSignal.frombin(cts, bins, exp=exp, ignore=ignore, random_seed=random_seed)
        inst.cwt_res = None
        inst.tmv_res = None
        return inst

    @classmethod
    def from_components(cls, obj_list):
        """Stack polyfit'd pgSignal components; see ``pgSignal.from_components``."""

        inst = cls.__new__(cls)
        inst._signal = pgSignal.from_components(obj_list)
        inst.cwt_res = None
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
            exp=sig.exp,
            type='pg',
        )

    def calculate(self, deg=None, **kwargs):
        """Run background fitting if needed, then compute the CWT spectrum."""

        if not hasattr(self, 'cts'):
            self.find_background(deg=deg)
        return super().calculate(**kwargs)

    def estimate_tmv(self, deg=None, **kwargs):
        """Run background fitting if needed, then estimate ``t_mv``."""

        if not hasattr(self, 'cts'):
            self.find_background(deg=deg)
        return super().estimate_tmv(**kwargs)


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
            exp=self._signal.exp,
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
            exp=inst._signal.exp,
            type='pp',
        )
        return inst


class ggCWT(CWT):
    """Compute CWT spectra for a Gaussian net-count light curve."""

    def __init__(self, ncts, ncts_err, bins, exp=None):
        """Initialize ggCWT with pre-background-subtracted counts."""

        self._signal = ggSignal(ncts, ncts_err, bins, exp=exp)
        CWT.__init__(
            self,
            self._signal.ncts,
            self._signal.bins,
            cts_err=self._signal.ncts_err,
            exp=self._signal.exp,
            type='gg',
        )
