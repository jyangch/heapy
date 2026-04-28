"""Estimate cross-correlation time lags between two gamma-ray burst light curves.

Implements the ``Lag`` class, which computes the normalised cross-correlation
function (CCF) between a high-energy and a low-energy light curve, fits the
peak to locate the lag, and propagates uncertainties via Monte Carlo
simulation.  The Modified CCF (MCCF) variant is supported through the ``M``
box-smoothing parameter.

Typical usage:
    from heapy.temp.lag import Lag
    lag = Lag(xcts, ycts, dt=0.064, xbcts=xbkg, ybcts=ybkg,
              xbcts_err=xbkg_err, ybcts_err=ybkg_err)
    lag.calculate(method='gp')
    lag.save('/output/dir')
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.linalg import cho_solve
from scipy.interpolate import UnivariateSpline
from scipy.fft import rfft, irfft, next_fast_len
from scipy.optimize import curve_fit, minimize_scalar
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from ..util.data import json_dump



class Lag(object):
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
        dt,
        xcts_err=None,
        ycts_err=None,
        xbcts=None,
        ybcts=None,
        xbcts_err=None,
        ybcts_err=None,
        xtype='pg',
        ytype='pg',
        nmc=1000,
        M=1):
        """Initialize the Lag estimator with two light curves.

        Args:
            xcts: 1-D array of fine-bin source counts for the reference
                (high-energy) light curve.
            ycts: 1-D array of fine-bin source counts for the comparison
                (low-energy) light curve; must have the same length as
                ``xcts``.
            dt: Fine-bin width in seconds; must be positive.
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
            xtype: Noise model for the ``x`` channel: ``'pg'`` (Poisson
                source + Gaussian background), ``'pp'`` (Poisson source +
                Poisson background), or ``'gg'`` (Gaussian source +
                Gaussian background).
            ytype: Noise model for the ``y`` channel; same options as
                ``xtype``.
            nmc: Number of Monte Carlo realisations used for uncertainty
                estimation; must be at least 1.
            M: Box-smoothing factor; the effective analysis bin width is
                :math:`M \\times dt`.  ``M = 1`` gives the classic CCF;
                ``M > 1`` enables MCCF.

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

        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError('dt must be positive')

        self.M = int(M)
        if self.M < 1:
            raise ValueError('M must be a positive integer')

        self.nmc = int(nmc)
        if self.nmc < 1:
            raise ValueError('nmc must be at least 1')

        self.xtype = xtype
        self.ytype = ytype

        self.xcts_err, self.xbcts, self.xbcts_err = \
            self._validate(xtype, xcts, xcts_err, xbcts, xbcts_err, 'x')
        self.ycts_err, self.ybcts, self.ybcts_err = \
            self._validate(ytype, ycts, ycts_err, ybcts, ybcts_err, 'y')

        self.model_funcs = {
            'gaussian': Lag.gaussian,
            'asymmetric_gaussian': Lag.asymmetric_gaussian,
            'double_gaussian': Lag.double_gaussian,
            'lorentzian': Lag.lorentzian,
            'pseudo_voigt': Lag.pseudo_voigt}


    @property
    def dt_analysis(self):
        """Effective time resolution used for the CCF in seconds.

        Returns the product :math:`M \\times dt`, which equals the width of
        each box-smoothed bin.
        """

        return self.M * self.dt


    @staticmethod
    def _validate(dtype, cts, cts_err, bcts, bcts_err, label):

        if dtype == 'pg':
            cts_err = np.sqrt(cts) if cts_err is None else cts_err
            if bcts is None:
                raise ValueError(f'unknown {label}bcts')
            if bcts_err is None:
                raise ValueError(f'unknown {label}bcts_err')

        elif dtype == 'pp':
            cts_err = np.sqrt(cts) if cts_err is None else cts_err
            if bcts is None:
                raise ValueError(f'unknown {label}bcts')
            bcts_err = np.sqrt(bcts) if bcts_err is None else bcts_err

        elif dtype == 'gg':
            if cts_err is None:
                raise ValueError(f'unknown {label}cts_err')
            bcts = np.zeros_like(cts, dtype=float) if bcts is None else bcts
            bcts_err = np.zeros_like(cts, dtype=float) if bcts_err is None else bcts_err

        else:
            raise ValueError(f'unknown {label}type')

        return cts_err, bcts, bcts_err


    @staticmethod
    def _box_smooth(arr, M):

        if M == 1:
            return np.asarray(arr, dtype=float).copy()

        return np.convolve(arr, np.ones(M), mode='valid')


    @staticmethod
    def _box_smooth_batch(arr2d, M):

        if M == 1:
            return np.asarray(arr2d, dtype=float).copy()

        arr2d = np.asarray(arr2d, dtype=float)

        cumsum = np.concatenate([
            np.zeros((arr2d.shape[0], 1), dtype=arr2d.dtype),
            np.cumsum(arr2d, axis=1)
            ], axis=1)

        return cumsum[:, M:] - cumsum[:, :-M]


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

        return cons + amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


    @staticmethod
    def asymmetric_gaussian(x, cons, amp, mu, sigma_l, sigma_r):
        """Evaluate an asymmetric Gaussian profile with a constant baseline.

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

        gaussian_l = cons + amp * np.exp(-((x - mu) ** 2) / (2 * sigma_l ** 2))
        gaussian_r = cons + amp * np.exp(-((x - mu) ** 2) / (2 * sigma_r ** 2))

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

        gaussian1 = amp1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
        gaussian2 = amp2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))

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

        return cons + amp * (gamma ** 2 / ((x - mu) ** 2 + gamma ** 2))


    @staticmethod
    def pseudo_voigt(x, cons, amp, mu, sigma, eta):
        """Evaluate a pseudo-Voigt profile with a constant baseline.

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
        gaussian_part = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        lorentzian_part = (gamma ** 2) / ((x - mu) ** 2 + gamma ** 2)

        return cons + amp * (eta * lorentzian_part + (1 - eta) * gaussian_part)


    def _mc_sample(self, dtype, cts, cts_err, bcts, bcts_err):

        size = (self.nmc, self.nsample)

        if dtype == 'pg':
            src = np.random.poisson(lam=cts, size=size)
            bkg = np.random.normal(loc=bcts, scale=bcts_err, size=size)

        elif dtype == 'pp':
            src = np.random.poisson(lam=cts, size=size)
            bkg = np.random.poisson(lam=bcts, size=size)

        elif dtype == 'gg':
            src = np.random.normal(loc=cts, scale=cts_err, size=size)
            bkg = np.random.normal(loc=bcts, scale=bcts_err, size=size)

        else:
            raise TypeError(f"invalid dtype: {dtype}")

        return src - bkg


    def _mc_simulation(self):

        xncts_sample = self._mc_sample(self.xtype, self.xcts, self.xcts_err,
                                       self.xbcts, self.xbcts_err)
        self.mc_xncts = np.vstack([self.xncts, xncts_sample])

        yncts_sample = self._mc_sample(self.ytype, self.ycts, self.ycts_err,
                                       self.ybcts, self.ybcts_err)
        self.mc_yncts = np.vstack([self.yncts, yncts_sample])


    def _ccfs_batch(self):

        n = self.nsample
        nfft = next_fast_len(2 * n - 1)

        X_rev = rfft(self.mc_xncts[:, ::-1].copy(), n=nfft, axis=1)
        Y = rfft(self.mc_yncts, n=nfft, axis=1)
        all_ccfs = irfft(Y * X_rev, n=nfft, axis=1)[:, :2 * n - 1]

        norms = np.sqrt(np.sum(self.mc_xncts ** 2, axis=1) *
                         np.sum(self.mc_yncts ** 2, axis=1))
        zero_mask = norms == 0
        norms[zero_mask] = 1.0
        all_ccfs /= norms[:, np.newaxis]
        all_ccfs[zero_mask] = 0.0

        return all_ccfs


    def calculate(self, method=None, width=None, threshold=None,
                  poly_deg=None, spline_s=None, point_estimate='observed'):
        """Compute the CCF lag and its Monte Carlo uncertainties.

        Performs background subtraction, applies MCCF box-smoothing when
        ``M > 1``, computes the normalised CCF for the observed data and
        all MC realisations via FFT, then fits or searches the CCF peak
        using the chosen method.  Prints a summary table and stores results
        in ``self.lag`` and ``self.lag_res``.

        Args:
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

        self.xncts = self.xcts - self.xbcts
        self.yncts = self.ycts - self.ybcts
        self.nsample = len(self.xcts)
        self._mc_simulation()

        if self.M > 1:
            self.xncts = self._box_smooth(self.xncts, self.M)
            self.yncts = self._box_smooth(self.yncts, self.M)
            self.mc_xncts = self._box_smooth_batch(self.mc_xncts, self.M)
            self.mc_yncts = self._box_smooth_batch(self.mc_yncts, self.M)
            self.nsample = len(self.xncts)

        self.taus = self.dt * np.arange(-self.nsample + 1, self.nsample, 1)

        self.mc_ccfs = self._ccfs_batch()
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
                f"point_estimate must be 'observed'|'mean'|'median', "
                f"got {point_estimate!r}")

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
                    lag_i = minimize_scalar(lambda x: -np.polyval(polyfit, x),
                                            bounds=self.nrange, method='bounded').x
                    if i == 0:
                        self.itp_ccfs = np.polyval(polyfit, self.itp_taus)

                elif method == 'spline':
                    spline = UnivariateSpline(fit_taus, fit_ccfs, s=spline_s)
                    lag_i = minimize_scalar(lambda x: -spline(x),
                                            bounds=self.nrange, method='bounded').x
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
                            self.itp_taus.reshape(-1, 1),
                            fit_taus.reshape(-1, 1))

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
                    lag_i = minimize_scalar(lambda x: -func(x, *popt),
                                            bounds=self.nrange, method='bounded').x
                    if i == 0:
                        self.itp_ccfs = func(self.itp_taus, *popt)

                else:
                    raise ValueError(f'unknown method: {method}')

                self.mc_fit_lags.append(lag_i)

            except Exception:
                if i == 0:
                    raise RuntimeError('failed to fit the CCF for the original light curves')
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

        print('\n+---------------------------------------------------+')
        print(' %-15s%-15s%-15s' % ('lag (s)', 'lag_le (s)', 'lag_he (s)'))
        print(' %-15.6g%-15.6g%-15.6g' % (self.lag[0], self.lag[1], self.lag[2]))
        print(' method=%s, M=%d, dt=%.3g s, point_estimate=%s' %
              (method, self.M, self.dt, point_estimate))
        print('+---------------------------------------------------+\n')

        self.lag_res = {'lag': self.lag, 'mc_fit_lags': self.mc_fit_lags,
                        'width': width, 'threshold': threshold, 'method': method,
                        'point_estimate': point_estimate,
                        'M': self.M, 'dt': self.dt,
                        'taus': self.taus, 'ccfs': self.ccfs,
                        'itp_taus': self.itp_taus, 'itp_ccfs': self.itp_ccfs}


    def save(self, savepath):
        """Save lag results and diagnostic plots to disk.

        Serialises ``lag_res`` as a JSON file and writes two PDF figures:
        one showing the CCF with the fitted profile overlay, and one
        histogram of the MC lag distribution with the central value and
        1-sigma interval marked.

        Args:
            savepath: Directory path where output files are written; created
                if it does not exist.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.lag_res, savepath + '/lag_res.json')

        rcParams['font.family'] = 'serif'
        rcParams['font.sans-serif'] = 'STIX Two Text'
        rcParams['mathtext.fontset'] = 'stix'
        rcParams['font.size'] = 12
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.scatter(self.taus[self.nidx], self.mc_ccfs[0][self.nidx], marker='+',
                   color='r', s=10, linewidths=0.5, alpha=1.0)
        if self.itp_taus is not None:
            ax.plot(self.itp_taus, self.itp_ccfs, c='b', lw=0.5, alpha=1.0)
        ax.set_xlabel('Time delay (s)')
        ax.set_ylabel('CCF value')
        ax.minorticks_on()
        ax.tick_params(axis='x', which='both', direction='in')
        ax.tick_params(axis='y', which='both', direction='in')
        ax.tick_params(which='major', width=1.0, length=5)
        ax.tick_params(which='minor', width=1.0, length=3)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        fig.savefig(savepath + '/tau_ccf.pdf', bbox_inches='tight',
                    pad_inches=0.1, dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        mc_only = self.mc_fit_lags[1:]
        lag_bins = np.linspace(min(mc_only), max(mc_only), 30)
        ax.hist(mc_only, lag_bins, density=False, histtype='step',
                color='b', lw=1.0)
        ax.axvline(self.lag[0], c='grey', lw=1.0)
        ax.axvline(self.lag[0] - self.lag[1], c='grey', ls='--', lw=1.0)
        ax.axvline(self.lag[0] + self.lag[2], c='grey', ls='--', lw=1.0)
        ax.set_xlabel('Lags (sec)')
        ax.set_ylabel('Counts')
        ax.set_title(r'$\tau=%.4g_{-%.4g}^{+%.4g}~{\rm s}$' %
                     (self.lag[0], self.lag[1], self.lag[2]))
        ax.minorticks_on()
        ax.tick_params(axis='x', which='both', direction='in')
        ax.tick_params(axis='y', which='both', direction='in')
        ax.tick_params(which='major', width=1.0, length=5)
        ax.tick_params(which='minor', width=1.0, length=3)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        fig.savefig(savepath + '/lag_pdf.pdf', bbox_inches='tight',
                    pad_inches=0.1, dpi=300)
        plt.close(fig)
