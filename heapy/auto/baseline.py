"""Baseline estimators for 1D signals.

Exposes a unified :class:`Baseline` facade over several algorithms
(``drpls``, ``snip``, ``isnip``, ``fabc``, ``goldindec``, ``mixture``).
The fitted baseline is stored on the instance and can be evaluated at
arbitrary abscissae via cubic spline interpolation.
"""

import warnings

from astropy.stats import mad_std, sigma_clip
import numpy as np
import pybaselines
from scipy.interpolate import CubicSpline
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve


class Baseline:
    """Fit and evaluate a smooth baseline under a 1D signal.

    Wraps multiple baseline algorithms behind a single ``fit``/``val``
    interface. The SNIP family uses a Whittaker pre-smoothing step followed
    by symmetric shrinking-window sweeps.

    Attributes:
        method: Selected baseline algorithm identifier.
        x: Abscissae used for fitting (set by :meth:`fit`).
        y: Ordinates used for fitting (set by :meth:`fit`).
        mo: Baseline values sampled at :attr:`x` (set by :meth:`fit`).

    Example:
        >>> bl = Baseline.set_method('drpls')
        >>> bl.fit(x, y)
        >>> baseline = bl.val(x_new)
    """

    def __init__(self, method='drpls'):
        """Store the baseline algorithm identifier.

        Args:
            method: Algorithm name; one of ``'drpls'``, ``'snip'``,
                ``'isnip'``, ``'fabc'``, ``'goldindec'``, ``'mixture'``.
        """

        self.method = method

    @classmethod
    def set_method(cls, method='drpls'):
        """Validate ``method`` and return a configured :class:`Baseline`.

        Args:
            method: Algorithm name; currently only ``'drpls'`` passes the
                explicit check, other names from :meth:`fit` still work.

        Returns:
            A new :class:`Baseline` bound to ``method``.

        Raises:
            ValueError: If ``method`` is not ``'drpls'``.
        """

        if method == 'drpls':
            pass
        else:
            raise ValueError('invalid method')

        return cls(method=method)

    def fit(self, x, y, w=None, lam=None, nk=None):
        """Estimate the baseline of ``(x, y)`` using the configured method.

        Populates :attr:`x`, :attr:`y`, and :attr:`mo`. Defaults for
        ``lam`` and ``nk`` are derived from ``len(x)``.

        Args:
            x: Monotonic 1D abscissae.
            y: Ordinates matching ``x``.
            w: Per-point weights; defaults to ones.
            lam: Smoothing strength for Whittaker-based methods.
            nk: Number of spline knots for the mixture model.

        Raises:
            ValueError: If :attr:`method` is not a recognized algorithm.
        """

        self.x = np.array(x).astype(float)
        self.y = np.array(y).astype(float)

        if w is None:
            w = np.ones(len(y))
        if lam is None:
            lam = 10 ** (4 * (np.log10(len(x)) - 1))
        if nk is None:
            nk = max(2 * len(x) - 200, 4)

        if self.method == 'snip':
            mo = self.snip(y, x, w=w)

        elif self.method == 'isnip':
            mo = self.isnip(y, x, w=w)

        elif self.method == 'drpls':
            mo = pybaselines.whittaker.drpls(y, lam=lam, eta=0.5, weights=w)[0]

        elif self.method == 'fabc':
            mo = pybaselines.classification.fabc(y, lam=lam, weights=w)[0]

        elif self.method == 'goldindec':
            mo = pybaselines.polynomial.goldindec(y, x, poly_order=3, peak_ratio=0.1, weights=w)[0]

        elif self.method == 'mixture':
            mo = pybaselines.spline.mixture_model(
                y, p=0.001, lam=lam, diff_order=2, num_knots=nk, weights=w
            )[0]

        else:
            raise ValueError('invalid method')

        self.mo = mo

    def val(self, x):
        """Evaluate the fitted baseline at ``x`` via cubic spline interpolation.

        Args:
            x: Query abscissae; values outside the fitting range are
                extrapolated and trigger a ``UserWarning``.

        Returns:
            Interpolated baseline values at ``x``.
        """

        x = np.asarray(x)

        if x.min() < self.x.min() or x.max() > self.x.max():
            warnings.warn('Extrapolation may be imprecise', UserWarning, stacklevel=2)

        interp = CubicSpline(self.x, self.mo, extrapolate=True)

        return interp(x)

    @staticmethod
    def speyediff(N, d, format='csc'):
        """Build the sparse ``d``-th order finite-difference matrix of size ``N``.

        Used as the penalty operator in Whittaker smoothing.

        Args:
            N: Number of samples (rows of the identity before differencing).
            d: Difference order; must be non-negative.
            format: Sparse storage format passed to ``scipy.sparse.diags``.

        Returns:
            A ``(N - d) x N`` sparse difference matrix.
        """

        assert d >= 0, 'd must be non-negative'

        shape = (N - d, N)
        diagonals = np.zeros(2 * d + 1)
        diagonals[d] = 1.0
        for _i in range(d):
            diff = diagonals[:-1] - diagonals[1:]
            diagonals = diff
        offsets = np.arange(d + 1)
        spmat = sparse.diags(diagonals, offsets, shape, format=format)

        return spmat

    def whittaker_smooth(self, y, lmbd, d, w):
        """Solve the weighted Whittaker smoothing normal equations.

        Minimizes ``||W^{1/2} (y - z)||^2 + lmbd * ||D^d z||^2``.

        Args:
            y: Signal to smooth (1D).
            lmbd: Penalty strength; larger values yield smoother output.
            d: Difference order of the penalty.
            w: Per-point weights matching ``y``.

        Returns:
            Tuple ``(coefmat, z)`` -- the coefficient matrix and the
            smoothed signal.
        """

        y = np.asarray(y).ravel()
        m = len(y)
        # E = sparse.eye(m, format='csc')
        D = self.speyediff(m, d, format='csc')
        W = sparse.diags(w, 0, shape=(m, m))
        # D is real, so .conj() is a no-op; plain transpose is enough
        coefmat = W + lmbd * D.T.dot(D)
        z = spsolve(coefmat, W @ y)

        return coefmat, z

    def get_smooth(self, y, w=None, d=None, lmbd=None):
        """Iteratively sigma-clip large residuals while Whittaker-smoothing.

        Runs three passes at ``sigma = 3, 2, 1``. Points flagged as
        outliers have their weights zeroed, which effectively excludes
        peaks from the smooth estimate.

        Args:
            y: Signal to smooth (1D).
            w: Initial weights; defaults to ones.
            d: Penalty difference order; defaults to ``3``.
            lmbd: Penalty strength; defaults to a length-dependent heuristic.

        Returns:
            The smoothed signal after outlier rejection.
        """

        y = np.array(y)
        m = len(y)
        if w is None:
            w = np.ones(m)
        if d is None:
            d = 3
        if lmbd is None:
            lmbd = 10 ** (4 * (np.log10(len(y)) - 1))
        _, z = self.whittaker_smooth(y, lmbd, d, w)

        cs = y - z
        for sigma in [3, 2, 1]:
            mask = sigma_clip(cs, sigma=sigma, maxiters=5, stdfunc=mad_std).mask
            w[mask] = 0
            _, z = self.whittaker_smooth(y, lmbd, d, w)
            cs = y - z

        return z

    def _snip_core(self, y, x, w=None, d=None, lmbd=None, inte=None, pr=None, hwi=None, it=None):
        """Run the SNIP iteration core shared by :meth:`snip` and :meth:`isnip`.

        Pre-smooths ``y`` with Whittaker smoothing, downsamples into
        ``inte`` bins, runs ``it`` symmetric SNIP sweeps with shrinking
        windows, then fits a cubic polynomial through the envelope.

        Args:
            y: Signal (1D).
            x: Abscissae matching ``y``; used only for median spacing.
            w: Per-point weights; defaults to ones.
            d: Difference order for the Whittaker pre-smooth.
            lmbd: Penalty strength for the Whittaker pre-smooth.
            inte: Number of SNIP sampling intervals.
            pr: Peak ratio controlling the half window width.
            hwi: Half window width; derived from ``inte`` and ``pr`` if omitted.
            it: Number of SNIP iterations.

        Returns:
            Tuple ``(y, idx, poly_z)`` -- the input ``y``, the full integer
            index array, and the cubic-fit baseline on the SNIP envelope.
        """

        y, x = np.asarray(y), np.asarray(x)
        m = len(y)
        if w is None:
            w = np.ones(m)
        z = self.get_smooth(y, w, d, lmbd)

        # Robust to non-uniform sampling: use median spacing as representative dt
        dt = float(np.median(np.diff(x))) if x.size > 1 else 1.0
        if it is None:
            it = 5
        if inte is None:
            inte = max(round(len(z) * dt / 5), 3)
        if pr is None:
            pr = 0.4
        if hwi is None:
            hwi = max(round(inte / 2 * pr), 1)

        if it != 1:
            d1 = np.log10(hwi)
            d2 = 0
            win = np.ceil(
                np.concatenate((10 ** (d1 + np.arange(it - 1) * (d2 - d1) / (it - 1)), [d2]))
            )
            win = win.astype(int)
        else:
            win = np.array([hwi], dtype=int)

        lims = np.linspace(0, z.size - 1, inte + 1)
        lefts = np.ceil(lims[:-1]).astype(int)
        rights = np.floor(lims[1:]).astype(int)
        samp_x = np.round((lefts + rights) * 0.5)

        bsl_z = np.zeros(inte)
        for i in range(inte):
            bsl_z[i] = z[lefts[i] : rights[i] + 1].mean()

        # SNIP iteration: symmetric forward + reverse sweeps at shrinking windows
        for i in range(it):
            win0 = win[i]
            for j in range(1, inte - 1):
                v = min(j, win0, inte - j - 1)
                a = bsl_z[j - v : j + v + 1].mean()
                bsl_z[j] = min(a, bsl_z[j])
            for j in range(1, inte - 1):
                k = inte - j - 1
                v = min(j, win0, inte - j - 1)
                a = bsl_z[k - v : k + v + 1].mean()
                bsl_z[k] = min(a, bsl_z[k])

        samp_x = np.concatenate(([0], samp_x, [z.size - 1]))
        bsl_z = np.concatenate((bsl_z[:1], bsl_z, bsl_z[-1:]))
        idx = np.arange(z.size)

        poly_fit = np.polyfit(samp_x, bsl_z, deg=3)
        poly_z = np.polyval(poly_fit, idx)

        return y, idx, poly_z

    def snip(self, y, x, w=None, d=None, lmbd=None, inte=None, pr=None, hwi=None, it=None):
        """Estimate the baseline via the SNIP algorithm.

        Args:
            y: Signal (1D).
            x: Abscissae matching ``y``.
            w, d, lmbd, inte, pr, hwi, it: See :meth:`_snip_core`.

        Returns:
            Baseline sampled on ``arange(len(y))``.
        """

        _, _, poly_z = self._snip_core(y, x, w, d, lmbd, inte, pr, hwi, it)

        return poly_z

    def isnip(self, y, x, w=None, d=None, lmbd=None, inte=None, pr=None, hwi=None, it=None):
        """Estimate the baseline via iteratively refined SNIP.

        Runs :meth:`_snip_core`, then sigma-clips residuals and re-fits a
        cubic polynomial at ``sigma = 3, 2, 1`` to sharpen the envelope.

        Args:
            y: Signal (1D).
            x: Abscissae matching ``y``.
            w, d, lmbd, inte, pr, hwi, it: See :meth:`_snip_core`.

        Returns:
            Refined baseline sampled on ``arange(len(y))``.
        """

        y, idx, poly_z = self._snip_core(y, x, w, d, lmbd, inte, pr, hwi, it)

        # Iterative refinement: sigma-clip residuals, re-fit cubic polynomial
        poly_cs = y - poly_z
        for sigma in [3, 2, 1]:
            mask = sigma_clip(poly_cs, sigma=sigma, maxiters=5, stdfunc=mad_std).mask
            keep = ~np.asarray(mask)
            poly_fit = np.polyfit(idx[keep], y[keep], deg=3)
            poly_z = np.polyval(poly_fit, idx)
            poly_cs = y - poly_z

        return poly_z
