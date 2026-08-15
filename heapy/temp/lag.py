"""Estimate cross-correlation time lags between two gamma-ray burst light curves.

Implements the ``Lag`` class, which computes the normalised cross-correlation
function (CCF) between a high-energy and a low-energy light curve, fits the
peak to locate the lag, and propagates uncertainties via Monte Carlo
simulation.  The Modified CCF (MCCF) variant is supported through the ``M``
box-smoothing parameter.

Example:
    from heapy.temp.lag import Lag
    lag = Lag(xcts, ycts, dt=0.064, xbcts=xbkg, ybcts=ybkg,
              xbcts_err=xbkg_err, ybcts_err=ybkg_err)
    lag.calculate(method='gp')
    lag.save('/output/dir')

    # Or built from two already-processed Signal instances (xtype/ytype,
    # backscale, and the count arrays are inferred from each signal):
    lag = Lag.from_signals(x_signal, y_signal)
    lag.calculate(method='gp')
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.linalg import cho_solve
from scipy.optimize import curve_fit, minimize_scalar
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from ..auto.signal import ggSignal, pgSignal, ppSignal
from ..util.tools import format_message, json_dump, plt_rc_context
from .temp_utils import (
    LagPlotter,
    box_smooth_batch,
    calculate_ccf_batch,
    generate_mc_sample,
    resolve_analysis_window,
    resolve_time_grid,
    validate_input,
)


class Lag:
    """Estimate the cross-correlation lag between two light curves.

    Computes the normalised CCF via FFT for the observed data and for
    ``nmc`` Monte Carlo realisations, then locates the lag by fitting or
    searching for the CCF peak.  Supports Poisson/Gaussian noise models and
    the MCCF box-smoothing variant.

    Attributes:
        lag: List ``[lag_value, lag_lower_error, lag_upper_error]`` populated
            after ``calculate`` is called.
        lag_res: Dictionary with full results, populated after ``calculate``.
        dt_analysis: Effective time resolution used for the CCF
            (:math:`M \\times dt`).
    """

    def __init__(
        self,
        xcts,
        ycts,
        bins=None,
        xcts_err=None,
        ycts_err=None,
        xbcts=None,
        ybcts=None,
        xbcts_err=None,
        ybcts_err=None,
        xbackscale=1,
        ybackscale=1,
        xtype='pg',
        ytype='pg',
        M=1,
        dt=None,
        time=None,
    ):
        """Initialize the Lag estimator with two light curves.

        Args:
            xcts: 1-D array of fine-bin source counts for the reference
                (high-energy) light curve.
            ycts: 1-D array of fine-bin source counts for the comparison
                (low-energy) light curve; must have the same length as
                ``xcts``.
            bins: Optional bin edges. A scalar is treated as legacy
                positional ``dt`` for compatibility.
            xcts_err: Count errors for ``xcts``; required when ``xtype``
                contains ``'g'``.
            ycts_err: Count errors for ``ycts``; required when ``ytype``
                contains ``'g'``.
            xbcts: Background counts for ``xcts``; required when ``xtype``
                is ``'pg'`` or ``'pp'``.
            ybcts: Background counts for ``ycts``; required when ``ytype``
                is ``'pg'`` or ``'pp'``.
            xbcts_err: Background count errors for ``xcts``; required when
                ``xtype`` is ``'pg'``.
            ybcts_err: Background count errors for ``ycts``; required when
                ``ytype`` is ``'pg'``.
            xbackscale: Ratio scaling ``xbcts`` into the source region;
                only meaningful when ``xtype`` is ``'pp'``. Defaults to 1.
            ybackscale: Ratio scaling ``ybcts`` into the source region;
                only meaningful when ``ytype`` is ``'pp'``. Defaults to 1.
            xtype: Noise model for the ``x`` channel: ``'pg'`` (Poisson
                source + Gaussian background), ``'pp'`` (Poisson source +
                Poisson background), or ``'gg'`` (Gaussian source +
                Gaussian background).
            ytype: Noise model for the ``y`` channel; same options as
                ``xtype``.
            M: Box-smoothing factor; the effective analysis bin width is
                :math:`M \\times dt`.  ``M = 1`` gives the classic CCF;
                ``M > 1`` enables MCCF.
            dt: Optional fine-bin width in seconds.
            time: Optional per-bin time grid. ``None`` uses
                bin centers derived from ``bins`` or ``dt``.

        Raises:
            ValueError: If ``xcts`` or ``ycts`` is not one-dimensional,
                they differ in shape, or either is empty.
            ValueError: If ``dt`` is not positive or ``M`` is less than 1.
            ValueError: If required background or error arrays are missing
                given the specified noise model.
        """

        self.xcts = np.asarray(xcts, dtype=float)
        self.ycts = np.asarray(ycts, dtype=float)

        if self.xcts.ndim != 1 or self.ycts.ndim != 1:
            raise ValueError('xcts and ycts must be one-dimensional arrays')
        if self.xcts.shape != self.ycts.shape:
            raise ValueError('xcts and ycts must have the same shape')
        if self.xcts.size == 0:
            raise ValueError('xcts and ycts cannot be empty')

        self.M = int(M)
        if self.M < 1:
            raise ValueError('M must be a positive integer')

        self.dt, self.time, self.bins = resolve_time_grid(
            self.xcts.size, bins=bins, dt=dt, time=time
        )

        self.xtype = xtype
        self.ytype = ytype
        self.xbackscale = xbackscale
        self.ybackscale = ybackscale

        self.xcts_err, self.xbcts, self.xbcts_err = validate_input(
            xtype, xcts, xcts_err, xbcts, xbcts_err, 'x'
        )
        self.ycts_err, self.ybcts, self.ybcts_err = validate_input(
            ytype, ycts, ycts_err, ybcts, ybcts_err, 'y'
        )

        self.xncts = self.xcts - self.xbcts * self.xbackscale
        self.yncts = self.ycts - self.ybcts * self.ybackscale
        self.nsample = len(self.xcts)

        self.model_funcs = {
            'gaussian': Lag.gaussian,
            'asymmetric_gaussian': Lag.asymmetric_gaussian,
            'double_gaussian': Lag.double_gaussian,
            'lorentzian': Lag.lorentzian,
            'pseudo_voigt': Lag.pseudo_voigt,
        }

    @staticmethod
    def _from_signal(signal):
        """Extract arrays and metadata from a Signal instance.

        Args:
            signal: A ``pgSignal``, ``ppSignal``, or ``ggSignal`` instance.
                For ``pgSignal``, its polynomial background fit must have
                already run (``bcts``/``bcts_err`` populated).

        Returns:
            An 8-tuple ``(type, cts, cts_err, bcts, bcts_err, backscale,
            dt, time)`` ready to feed into ``Lag.__init__`` (prefixed
            with ``x``/``y``).

        Raises:
            TypeError: If ``signal`` is not a recognised Signal instance.
            RuntimeError: If ``signal`` is a ``pgSignal`` whose background
                fit has not run yet.
            ValueError: If ``signal``'s bins are not uniform in width.
        """

        if isinstance(signal, pgSignal):
            if signal.poly_res is None:
                raise RuntimeError('pgSignal has no background fit yet; run polyfit()/loop() first')
            dtype, cts, cts_err, bcts, bcts_err, backscale = (
                'pg',
                signal.cts,
                None,
                signal.bcts,
                signal.bcts_err,
                1,
            )
        elif isinstance(signal, ppSignal):
            dtype, cts, cts_err, bcts, bcts_err, backscale = (
                'pp',
                signal.cts,
                None,
                signal.bcts,
                None,
                signal.backscale,
            )
        elif isinstance(signal, ggSignal):
            dtype, cts, cts_err, bcts, bcts_err, backscale = (
                'gg',
                signal.ncts,
                signal.ncts_err,
                None,
                None,
                1,
            )
        else:
            raise TypeError('expected signal to be a pgSignal, ppSignal, or ggSignal instance')

        binsize = signal.binsize
        if not np.allclose(binsize, binsize[0]):
            raise ValueError('signal bins must be uniform (constant width) for Lag')

        return dtype, cts, cts_err, bcts, bcts_err, backscale, float(binsize[0]), signal.time

    @classmethod
    def from_signals(cls, x_signal, y_signal, M=1):
        """Build a Lag from two already-built Signal instances.

        Infers ``xtype``/``ytype`` from the concrete class of ``x_signal``/
        ``y_signal`` (``pgSignal`` -> ``'pg'``, ``ppSignal`` -> ``'pp'``,
        ``ggSignal`` -> ``'gg'``) and extracts the plain arrays
        ``Lag.__init__`` needs -- including ``backscale`` for a ``ppSignal``
        channel, which a manually-constructed ``Lag`` is easy to forget.
        Avoids a per-(xtype, ytype)-combination subclass hierarchy (9
        combinations) by resolving each channel independently.

        Args:
            x_signal: A pgSignal/ppSignal/ggSignal instance for the
                reference (high-energy) channel.
            y_signal: A pgSignal/ppSignal/ggSignal instance for the
                comparison (low-energy) channel.
            M: Box-smoothing factor forwarded to ``__init__``.

        Returns:
            A new instance of ``cls``.

        Raises:
            TypeError: If either signal is not a recognised Signal instance.
            RuntimeError: If a ``pgSignal`` channel's background fit has
                not run yet.
            ValueError: If either signal's bins are non-uniform, or the two
                signals don't share the same bin width.
        """

        xtype, xcts, xcts_err, xbcts, xbcts_err, xbackscale, x_dt, x_time = cls._from_signal(
            x_signal
        )
        ytype, ycts, ycts_err, ybcts, ybcts_err, ybackscale, y_dt, y_time = cls._from_signal(
            y_signal
        )

        if not np.isclose(x_dt, y_dt):
            raise ValueError('x_signal and y_signal must share the same bin width (dt)')
        if not np.allclose(x_time, y_time):
            raise ValueError('x_signal and y_signal must share the same time grid')

        return cls(
            xcts,
            ycts,
            xcts_err=xcts_err,
            ycts_err=ycts_err,
            xbcts=xbcts,
            ybcts=ybcts,
            xbcts_err=xbcts_err,
            ybcts_err=ybcts_err,
            xbackscale=xbackscale,
            ybackscale=ybackscale,
            xtype=xtype,
            ytype=ytype,
            M=M,
            dt=x_dt,
            time=x_time,
        )

    @property
    def dt_analysis(self):
        """Effective time resolution used for the CCF in seconds.

        Returns the product :math:`M \\times dt`, which equals the width of
        each box-smoothed bin.
        """

        return self.M * self.dt

    @staticmethod
    def gaussian(x, cons, amp, mu, sigma):
        """Evaluate a Gaussian profile with a constant baseline.

        Args:
            x: Evaluation points.
            cons: Additive constant (baseline level).
            amp: Peak amplitude above the baseline.
            mu: Peak centre.
            sigma: Standard deviation.

        Returns:
            Array of profile values at each point in ``x``.
        """

        return cons + amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    @staticmethod
    def asymmetric_gaussian(x, cons, amp, mu, sigma_l, sigma_r):
        r"""Evaluate an asymmetric Gaussian profile with a constant baseline.

        Uses ``sigma_l`` for the left wing (:math:`x \\leq \\mu`) and
        ``sigma_r`` for the right wing (:math:`x > \\mu`).

        Args:
            x: Evaluation points.
            cons: Additive constant (baseline level).
            amp: Peak amplitude above the baseline.
            mu: Peak centre.
            sigma_l: Standard deviation of the left-side wing.
            sigma_r: Standard deviation of the right-side wing.

        Returns:
            Array of profile values at each point in ``x``.
        """

        gaussian_l = cons + amp * np.exp(-((x - mu) ** 2) / (2 * sigma_l**2))
        gaussian_r = cons + amp * np.exp(-((x - mu) ** 2) / (2 * sigma_r**2))

        return gaussian_l * (x <= mu) + gaussian_r * (x > mu)

    @staticmethod
    def double_gaussian(x, cons, amp1, mu1, sigma1, amp2, mu2, sigma2):
        """Evaluate a superposition of two Gaussians with a shared baseline.

        Args:
            x: Evaluation points.
            cons: Additive constant shared by both components.
            amp1: Amplitude of the first Gaussian.
            mu1: Centre of the first Gaussian.
            sigma1: Standard deviation of the first Gaussian.
            amp2: Amplitude of the second Gaussian.
            mu2: Centre of the second Gaussian.
            sigma2: Standard deviation of the second Gaussian.

        Returns:
            Array of profile values at each point in ``x``.
        """

        gaussian1 = amp1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))
        gaussian2 = amp2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2))

        return cons + gaussian1 + gaussian2

    @staticmethod
    def lorentzian(x, cons, amp, mu, gamma):
        """Evaluate a Lorentzian profile with a constant baseline.

        Args:
            x: Evaluation points.
            cons: Additive constant (baseline level).
            amp: Peak amplitude above the baseline.
            mu: Peak centre.
            gamma: Half-width at half-maximum.

        Returns:
            Array of profile values at each point in ``x``.
        """

        return cons + amp * (gamma**2 / ((x - mu) ** 2 + gamma**2))

    @staticmethod
    def pseudo_voigt(x, cons, amp, mu, sigma, eta):
        r"""Evaluate a pseudo-Voigt profile with a constant baseline.

        Mixes a Lorentzian and a Gaussian component with mixing fraction
        ``eta``.  The Lorentzian HWHM is derived from ``sigma`` via
        :math:`\\gamma = \\sigma \\sqrt{2 \\ln 2}`.

        Args:
            x: Evaluation points.
            cons: Additive constant (baseline level).
            amp: Peak amplitude above the baseline.
            mu: Peak centre.
            sigma: Gaussian standard deviation.
            eta: Lorentzian mixing fraction in ``[0, 1]``; ``eta = 1`` gives
                a pure Lorentzian, ``eta = 0`` gives a pure Gaussian.

        Returns:
            Array of profile values at each point in ``x``.
        """

        gamma = sigma * np.sqrt(2 * np.log(2))
        gaussian_part = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        lorentzian_part = (gamma**2) / ((x - mu) ** 2 + gamma**2)

        return cons + amp * (eta * lorentzian_part + (1 - eta) * gaussian_part)

    def generate_mc_simulation(self, nmc, random_seed=450001):
        """Generate Monte Carlo realisations of the net count light curve.

        The sampling model is selected by ``self.xtype`` and ``self.ytype``:

        - ``'pg'``: Poisson source counts (``xcts``) + Gaussian background
          (``xbcts``, ``xbcts_err``).
        - ``'pp'``: independent Poisson source (``xcts``) and background
          (``xbcts``) counts.
        - ``'gg'``: Gaussian source and background counts.

        Populates ``self.mc_xncts`` and ``self.mc_yncts`` with the observed
        net-count data in row 0.

        Args:
            nmc: Number of Monte Carlo realisations to generate.
            random_seed: Seed for the local RNG used to draw samples.
                Default ensures reproducibility across runs; pass
                ``None`` for OS entropy.
        """

        self.nmc = int(nmc)
        rng = np.random.default_rng(random_seed)

        xncts_sample = generate_mc_sample(
            self.xtype,
            self.xcts,
            self.xcts_err,
            self.xbcts,
            self.xbcts_err,
            self.nmc,
            rng,
            backscale=self.xbackscale,
        )
        self.mc_xncts = np.vstack([self.xncts, xncts_sample])

        yncts_sample = generate_mc_sample(
            self.ytype,
            self.ycts,
            self.ycts_err,
            self.ybcts,
            self.ybcts_err,
            self.nmc,
            rng,
            backscale=self.ybackscale,
        )
        self.mc_yncts = np.vstack([self.yncts, yncts_sample])

    def calculate(
        self,
        twin=None,
        method=None,
        width=None,
        threshold=None,
        poly_deg=None,
        spline_s=None,
        point_estimate='observed',
    ):
        """Compute the CCF lag and its Monte Carlo uncertainties.

        Performs background subtraction, applies MCCF box-smoothing when
        ``M > 1``, computes the normalised CCF for the observed data and
        all MC realisations via FFT, then fits or searches the CCF peak
        using the chosen method.  Prints a summary table and stores results
        in ``self.lag`` and ``self.lag_res``.

        Args:
            twin: Optional ``[t1, t2]`` analysis window. ``None`` uses the
                full light curve.
            method: Peak-location method.  One of ``'argmax'``,
                ``'polyfit'``, ``'spline'``, ``'gp'``, ``'gaussian'``,
                ``'asymmetric_gaussian'``, ``'double_gaussian'``,
                ``'lorentzian'``, or ``'pseudo_voigt'``.  Defaults to
                ``'argmax'`` when ``M > 1``, otherwise ``'gp'``.
            width: Restrict the fit/search to ±``width`` bins around the
                CCF peak; ``None`` uses the full CCF.
            threshold: Restrict to the region where CCF ≥ ``threshold``
                times the peak value; ignored when ``width`` is also set.
            poly_deg: Polynomial degree for the ``'polyfit'`` method;
                defaults to ``min(2, len(window) - 1)``.
            spline_s: Smoothing factor for the ``'spline'`` method;
                defaults to ``0.05``.
            point_estimate: Central value reported for the lag.  One of:

                - ``'observed'`` (default): lag from the unperturbed data.
                - ``'mean'``: mean of MC realisations.
                - ``'median'``: median of MC realisations.

                The 16th/84th-percentile interval from MC is reported
                regardless of this choice.

        Raises:
            ValueError: If ``point_estimate`` is not one of the allowed
                values, or if an unknown ``method`` is requested.
            RuntimeError: If the CCF peak fit fails for the observed data.
        """

        if method is None:
            method = 'argmax' if self.M > 1 else 'gp'

        self.analysis_index, analysis_window = resolve_analysis_window(self.time, twin)

        self.generate_mc_simulation(1000)
        mc_xncts = self.mc_xncts[:, self.analysis_index]
        mc_yncts = self.mc_yncts[:, self.analysis_index]

        if self.M > 1:
            mc_xncts = box_smooth_batch(mc_xncts, self.M)
            mc_yncts = box_smooth_batch(mc_yncts, self.M)

        self.nsample = mc_xncts.shape[1]
        self.taus = self.dt * np.arange(-self.nsample + 1, self.nsample, 1)

        self.mc_ccfs = calculate_ccf_batch(mc_xncts, mc_yncts)
        self.ccfs = self.mc_ccfs[0]

        pidx = np.argmax(self.ccfs)
        pval = self.ccfs[pidx]
        nccf = len(self.ccfs)

        if width is None and threshold is None:
            self.nidx = np.arange(nccf)
        elif width is not None:
            lo = max(0, pidx - width)
            hi = min(nccf, pidx + width + 1)
            self.nidx = np.arange(lo, hi)
        elif threshold is not None:
            lidx = pidx
            while lidx > 0 and self.ccfs[lidx - 1] >= threshold * pval:
                lidx -= 1
            ridx = pidx
            while ridx < nccf - 1 and self.ccfs[ridx + 1] >= threshold * pval:
                ridx += 1
            self.nidx = np.arange(lidx, ridx + 1)

        if method == 'polyfit' and poly_deg is None:
            poly_deg = min(2, len(self.nidx) - 1)

        if method == 'spline' and spline_s is None:
            spline_s = 0.05

        if point_estimate not in ('observed', 'mean', 'median'):
            raise ValueError(
                f"point_estimate must be 'observed'|'mean'|'median', got {point_estimate!r}"
            )

        self.nrange = (self.taus[self.nidx[0]], self.taus[self.nidx[-1]])

        if method == 'argmax':
            self.itp_taus = None
            self.itp_ccfs = None
        else:
            self.itp_taus = np.linspace(self.nrange[0], self.nrange[1], 1000)
            self.itp_ccfs = np.zeros_like(self.itp_taus, dtype=float)

        self.mc_fit_lags = []

        fit_taus = self.taus[self.nidx]
        fitted_kernel = None
        gp_L = None
        gp_K_star = None

        for i, ccfs_i in enumerate(self.mc_ccfs):
            fit_ccfs = ccfs_i[self.nidx]

            try:
                if method == 'argmax':
                    peak_pos = int(np.argmax(fit_ccfs))
                    lag_i = fit_taus[peak_pos]
                    if 0 < peak_pos < len(fit_ccfs) - 1:
                        y0 = fit_ccfs[peak_pos - 1]
                        y1 = fit_ccfs[peak_pos]
                        y2 = fit_ccfs[peak_pos + 1]
                        denom = y0 - 2 * y1 + y2
                        if denom != 0:
                            offset = 0.5 * (y0 - y2) / denom
                            if abs(offset) < 1.0:
                                lag_i = fit_taus[peak_pos] + offset * self.dt

                elif method == 'polyfit':
                    polyfit = np.polyfit(fit_taus, fit_ccfs, deg=poly_deg)
                    lag_i = minimize_scalar(
                        lambda x, p=polyfit: -np.polyval(p, x), bounds=self.nrange, method='bounded'
                    ).x
                    if i == 0:
                        self.itp_ccfs = np.polyval(polyfit, self.itp_taus)

                elif method == 'spline':
                    spline = UnivariateSpline(fit_taus, fit_ccfs, s=spline_s)
                    lag_i = minimize_scalar(
                        lambda x, s=spline: -s(x), bounds=self.nrange, method='bounded'
                    ).x
                    if i == 0:
                        self.itp_ccfs = spline(self.itp_taus)

                elif method == 'gp':
                    if fitted_kernel is None:
                        kernel = 1.0 * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.01)
                        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
                        gpr.fit(fit_taus.reshape(-1, 1), fit_ccfs)
                        fitted_kernel = gpr.kernel_
                        gp_L = gpr.L_
                        gp_K_star = fitted_kernel(
                            self.itp_taus.reshape(-1, 1), fit_taus.reshape(-1, 1)
                        )

                    alpha = cho_solve((gp_L, True), fit_ccfs)
                    mu = gp_K_star @ alpha
                    peak_pos = int(np.argmax(mu))
                    lag_i = self.itp_taus[peak_pos]
                    if 0 < peak_pos < len(mu) - 1:
                        y0, y1, y2 = mu[peak_pos - 1], mu[peak_pos], mu[peak_pos + 1]
                        denom = y0 - 2 * y1 + y2
                        if denom != 0:
                            step = self.itp_taus[1] - self.itp_taus[0]
                            offset = 0.5 * (y0 - y2) / denom
                            if abs(offset) < 1.0:
                                lag_i = self.itp_taus[peak_pos] + offset * step
                    if i == 0:
                        self.itp_ccfs = mu

                elif method in self.model_funcs:
                    func = self.model_funcs[method]
                    popt, _ = curve_fit(func, fit_taus, fit_ccfs, maxfev=5000)
                    lag_i = minimize_scalar(
                        lambda x, f=func, p=popt: -f(x, *p), bounds=self.nrange, method='bounded'
                    ).x
                    if i == 0:
                        self.itp_ccfs = func(self.itp_taus, *popt)

                else:
                    raise ValueError(f'unknown method: {method}')

                self.mc_fit_lags.append(lag_i)

            except Exception as err:
                if i == 0:
                    raise RuntimeError(
                        'failed to fit the CCF for the original light curves'
                    ) from err
                continue

        mc_fit_lags_filter = np.asarray(self.mc_fit_lags[1:])

        if point_estimate == 'observed':
            lag_bv = self.mc_fit_lags[0]
        elif point_estimate == 'mean':
            lag_bv = float(np.mean(mc_fit_lags_filter))
        else:
            lag_bv = float(np.median(mc_fit_lags_filter))

        lag_lo, lag_hi = np.percentile(mc_fit_lags_filter, [16, 84])
        lag_err = np.diff([lag_lo, lag_bv, lag_hi])
        self.lag = [lag_bv, lag_err[0], lag_err[1]]

        self.lag_res = {
            'lag': self.lag,
            'mc_fit_lags': self.mc_fit_lags,
            'width': width,
            'threshold': threshold,
            'method': method,
            'point_estimate': point_estimate,
            'M': self.M,
            'dt': self.dt,
            'analysis_window': analysis_window,
            'taus': self.taus,
            'ccfs': self.ccfs,
            'itp_taus': self.itp_taus,
            'itp_ccfs': self.itp_ccfs,
        }

        msg = [
            f'{"lag (s)":<15}{"lag_le (s)":<15}{"lag_he (s)":<15}',
            f'{self.lag[0]:<15.6g}{self.lag[1]:<15.6g}{self.lag[2]:<15.6g}',
            f'method={method}, M={self.M:d}, dt={self.dt:.3g} s, point_estimate={point_estimate}',
        ]
        print(format_message(msg))

    def save(self, savepath):
        """Save lag results and diagnostic plots to disk.

        Serialises ``lag_res`` as a JSON file and writes two PDF figures:
        a two-panel figure via
        :class:`~heapy.temp.temp_utils.LagPlotter` (the ``x``/``y`` light
        curves on top, the CCF with the fitted profile overlay on the
        bottom), and a histogram of the MC lag distribution with the
        central value and 1-sigma interval marked.

        Args:
            savepath: Directory path where output files are written; created
                if it does not exist.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.lag_res, savepath + '/lag_res.json')

        with plt_rc_context():
            fig = LagPlotter()
            fig.plot_curves(self.time, self.xncts, self.yncts)
            fig.plot_analysis_window(self.lag_res['analysis_window'])
            fig.plot_ccf(
                self.taus,
                self.mc_ccfs[0],
                self.nidx,
                itp_taus=self.itp_taus,
                itp_ccfs=self.itp_ccfs,
                lag=self.lag,
            )
            fig.save(savepath + '/lag.pdf')

            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            mc_only = self.mc_fit_lags[1:]
            lag_bins = np.linspace(min(mc_only), max(mc_only), 30)
            ax.hist(mc_only, lag_bins, density=False, histtype='step', color='b', lw=1.0)
            ax.axvline(self.lag[0], c='grey', lw=1.0)
            ax.axvline(self.lag[0] - self.lag[1], c='grey', ls='--', lw=1.0)
            ax.axvline(self.lag[0] + self.lag[2], c='grey', ls='--', lw=1.0)
            ax.set_xlabel('Lags (sec)')
            ax.set_ylabel('Counts')
            ax.set_title(
                rf'$\tau={self.lag[0]:.4g}_{{-{self.lag[1]:.4g}}}^{{+{self.lag[2]:.4g}}}~{{\rm s}}$'
            )
            ax.minorticks_on()
            ax.tick_params(axis='x', which='both', direction='in')
            ax.tick_params(axis='y', which='both', direction='in')
            ax.tick_params(which='major', width=1.0, length=5)
            ax.tick_params(which='minor', width=1.0, length=3)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            fig.savefig(savepath + '/lag_pdf.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)
