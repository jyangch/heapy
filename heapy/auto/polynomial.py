"""Weighted-least-squares polynomial fitting with BIC model selection.

Provides :class:`Polynomial`, a thin wrapper around :mod:`numpy.linalg`
that exposes a ``fit``/``val`` interface and implements the GBM/GECAM
two-pass fitting recipe used for Poisson-rate backgrounds.
"""

import warnings
import numpy as np



class Polynomial(object):
    """Fit a polynomial background via BIC-selected weighted least squares.

    The default ``'2pass'`` method iterates a uniform-weight pass followed
    by a model-variance-weighted pass across degrees ``0..4`` and keeps the
    degree with the lowest BIC. Coefficient covariance is propagated so
    :meth:`val` can return per-point 1-sigma errors.

    Attributes:
        method: Selected fitting strategy.
        x, y: Inputs to the last :meth:`fit` call.
        deg: User-supplied degree (``None`` enables BIC search).
        dx: Per-point width used to convert rate to counts for the variance.
        ls_res: Fit dict returned by the winning call to :meth:`ls_polyfit`.
        best_deg, best_bic, bic_list, deg_list: BIC selection bookkeeping.

    Example:
        >>> poly = Polynomial.set_method('2pass')
        >>> poly.fit(time, rate, dx=exp)
        >>> mo, err = poly.val(time_new)
    """

    def __init__(self, method='2pass'):
        """Store the fitting strategy identifier.

        Args:
            method: Strategy name; currently only ``'2pass'`` is supported.
        """

        self.method = method


    @classmethod
    def set_method(cls, method='2pass'):
        """Validate ``method`` and return a configured :class:`Polynomial`.

        Args:
            method: Strategy name; must be ``'2pass'``.

        Returns:
            A new :class:`Polynomial` bound to ``method``.

        Raises:
            ValueError: If ``method`` is not ``'2pass'``.
        """

        if method == '2pass':
            pass
        else:
            raise ValueError('invalid method')

        return cls(method=method)


    def fit(self, x, y, deg=None, dx=None):
        """Fit a polynomial background to ``(x, y)`` using the configured method.

        Stores inputs on the instance and delegates to :meth:`two_pass`
        (the only method currently supported).

        Args:
            x: Abscissae (1D).
            y: Rate values matching ``x``.
            deg: Polynomial degree; ``None`` searches ``0..4`` by BIC.
            dx: Per-point width converting rate variance to count variance;
                scalar or 1D matching ``x``. Defaults to ``x[1] - x[0]``.

        Raises:
            ValueError: If :attr:`method` is not ``'2pass'``.
        """
        
        self.x = np.array(x).astype(float)
        self.y = np.array(y).astype(float)
        self.deg = deg
        self.dx = dx

        self.ls_res = None

        if self.method == '2pass':
            self.two_pass()
        else:
            raise ValueError('invalid method')


    def val(self, x):
        """Evaluate the fitted polynomial and its 1-sigma uncertainty at ``x``.

        Emits a ``UserWarning`` when ``x`` falls outside the fitting
        range. Requires a prior :meth:`fit` call.

        Args:
            x: Query abscissae.

        Returns:
            Tuple ``(mo, err)`` — model values and their 1-sigma errors
            propagated from the coefficient covariance.

        Raises:
            AssertionError: If called before :meth:`fit`.
        """

        x = np.array(x)
        
        if min(x) < min(self.x):
            msg = "Extrapolation may be imprecise: %f < %f" % (min(x), min(self.x))
            warnings.warn(msg, UserWarning, stacklevel=2)

        if max(x) > max(self.x):
            msg = "Extrapolation may be imprecise: %f > %f" % (max(x), max(self.x))
            warnings.warn(msg, UserWarning, stacklevel=2)

        assert self.ls_res is not None, 'you should first perform fitting'

        mo, err = self.ls_polyval(x, coeff=self.ls_res['coeff'], covar=self.ls_res['covar'])
        return mo, err


    def two_pass(self):
        """Run the GBM/GECAM-style two-pass polynomial fit with BIC selection.

        For each candidate degree, runs a uniform-weight pass followed by
        a model-variance-weighted pass and records the BIC. The instance
        is updated with the lowest-BIC result.

        Populates :attr:`deg_list`, :attr:`bic_list`, :attr:`best_bic`,
        :attr:`best_deg`, and :attr:`ls_res`.
        
        Weighting scheme follows the GBM / GECAM background-fitting recipe:
        first pass is unweighted, second pass re-weights by model-rate
        variance so poorly-fit bins don't anchor the solution.
        Derivation: rate * livetime = counts; var(counts) = counts (Poisson);
        var(rate) = var(counts) / livetime^2 = rate / livetime.

        Raises:
            TypeError: If ``self.dx`` is a non-scalar array with a shape
                that does not match ``self.x``.
        """
        
        time = self.x
        rate = self.y

        if self.deg is None:
            self.deg_list = [0, 1, 2, 3, 4]
        else:
            self.deg_list = [self.deg]

        if self.dx is None:
            dt = time[1] - time[0]
        else:
            dt = np.array(self.dx)
            if dt.ndim != 0 and dt.shape != time.shape:
                raise TypeError("if dt is array, expected dt and time to have same length")

        # First pass: uniform weight (scale cancels out in WLS coefficients/covariance)
        weight = np.ones_like(rate)
        
        self.bic_list = []
        self.best_bic = np.inf
        self.best_deg = None
        for deg in self.deg_list:
            for ipass in range(2):
                ls_res = self.ls_polyfit(time, rate, deg, w=weight, cov=True)
                mo, _ = self.ls_polyval(time, coeff=ls_res['coeff'], covar=ls_res['covar'])
                
                if ipass == 0:
                    variance = mo / dt
                    zero = (variance <= 0)
                    weight = np.zeros_like(variance)
                    weight[~zero] = np.sqrt(1.0 / variance[~zero])
                    
            self.bic_list.append(ls_res['bic'])
                    
            if ls_res['bic'] < self.best_bic:
                self.best_bic = ls_res['bic']
                self.best_deg = deg
                self.ls_res = ls_res


    @staticmethod
    def ls_polyval(x, coeff, covar=None):
        """Evaluate a polynomial and optionally its propagated 1-sigma error.

        With ``covar`` supplied, the returned error is the diagonal of
        ``X @ covar @ X^T``, clipped at zero before the square root.

        Args:
            x: 1D query abscissae (non-empty).
            coeff: Polynomial coefficients in descending power order.
            covar: Optional coefficient covariance matrix
                (``order x order``).

        Returns:
            ``mo`` (1D model values) when ``covar`` is ``None``; otherwise
            ``(mo, err)``.

        Raises:
            TypeError: If array shapes violate the documented contracts.
        """

        if x.ndim != 1:
            raise TypeError("expected 1D vector for x")
        if x.size == 0:
            raise TypeError("expected non-empty vector for x")
        if coeff.ndim != 1:
            raise TypeError("expected 1D vector for coeff")
        if coeff.size == 0:
            raise TypeError("expected non-empty vector for coeff")

        order = coeff.shape[0]

        # set up least squares equation for powers of x
        lhs = np.vander(x, order)

        # evaluate the model values at given x
        mo = coeff @ lhs.T
        # or
        # mo = np.zeros_like(x)
        # for ci in coeff:
        #     mo = mo * x + ci

        # mo[mo < 0] = 0

        if covar is not None:
            if covar.ndim != 2:
                raise TypeError("expected 2D vector for covar")
            if covar.shape[0] != covar.shape[1]:
                raise TypeError("expected m*m matrix for covar")
            if covar.shape[0] != order:
                raise TypeError("expected each dim of covar and coeff have same length")

            # evaluate the model uncertainty at given x
            # M (i.e. var) = X M_\beta (i.e. covar) X^T
            var = lhs @ covar @ lhs.T
            err = np.sqrt(np.clip(np.diag(var), 0.0, None))

            return mo, err

        else:
            return mo


    @staticmethod
    def ls_polyfit(x, y, deg, rcond=None, w=None, cov=True):
        """Fit a weighted least-squares polynomial and return diagnostic stats.

        The Vandermonde system is column-scaled for conditioning, solved
        via :func:`numpy.linalg.lstsq`, and the resulting coefficients and
        covariance are rescaled back to the original basis. AIC and BIC
        use the weighted sum of squared residuals as the likelihood proxy.

        Args:
            x: 1D abscissae (non-empty).
            y: 1D or 2D ordinates with ``y.shape[0] == x.shape[0]``.
            deg: Non-negative polynomial degree.
            rcond: Cutoff for small singular values; defaults to
                ``len(x) * eps``.
            w: Per-point weights; defaults to ones.
            cov: ``True`` returns scaled covariance, ``"unscaled"`` skips
                the ``chi2/dof`` rescaling, ``False`` omits covariance.

        Returns:
            Dict with keys ``coeff``, ``chi2``, ``rank``, ``svs``,
            ``rcond``, ``order``, ``dof``, ``aic``, ``bic`` (plus
            ``covar`` when ``cov`` is truthy).

        Raises:
            ValueError: If ``deg < 0``.
            TypeError: If array shapes violate the documented contracts.

        Warning:
            Emits ``UserWarning`` on rank deficiency or non-positive dof.
        """

        order = int(deg) + 1
        x = np.asarray(x) + 0.0
        y = np.asarray(y) + 0.0
        
        if deg < 0:
            raise ValueError("expected deg >= 0")
        if x.ndim != 1:
            raise TypeError("expected 1D vector for x")
        if x.size == 0:
            raise TypeError("expected non-empty vector for x")
        if y.ndim < 1 or y.ndim > 2:
            raise TypeError("expected 1D or 2D array for y")
        if x.shape[0] != y.shape[0]:
            raise TypeError("expected x and y to have same length")
        
        if rcond is None:
            rcond = len(x) * np.finfo(x.dtype).eps

        if (y == 0).all():
            coeff = np.full(order, 0.0)
            res = {'coeff': coeff}
            if cov:
                covar = np.full((order, order), 0.0)
                res['covar'] = covar
            return res

        lhs = np.vander(x, order)
        rhs = y
        
        if w is not None:
            w = np.asarray(w) + 0.0
            if w.ndim != 1:
                raise TypeError("expected a 1-d array for weights")
            if w.shape[0] != y.shape[0]:
                raise TypeError("expected w and y to have the same length")
        else:
            w = np.ones_like(y) + 0.0
            
        dof = np.sum(w != 0) - order
        if dof <= 0:
            msg = 'The degree of freedom should be greater than zero'
            warnings.warn(msg, UserWarning, stacklevel=2)
            
        # X' = diag(w)X, see "Weighted least squares" in Wikipedia for details
        lhs *= w[:, np.newaxis]
        
        # y' = diag(w)y, see "Weighted least squares" in Wikipedia for details
        if rhs.ndim == 2:
            rhs *= w[:, np.newaxis]
        else:
            rhs *= w
            
        # scale lhs to improve condition number and solve
        scale = np.sqrt((lhs * lhs).sum(axis=0))
        lhs /= scale
        
        # solve WX@\beta = Wy, minimize sum((w*yi - w*f(xi, \beta))^2)
        # ==> X^TWy = (X^TWX)\beta ==> X'^Ty' = (X'^TX')\beta (uncorrelated w=sqrt(W))
        # see "Weighted least squares" in Wikipedia for details
        coeff, resids, rank, svs = np.linalg.lstsq(lhs, rhs, rcond)
        
        # broadcast scale coefficients
        coeff = (coeff.T / scale).T

        # warn on rank reduction, which indicates an ill conditioned matrix
        if rank != order:
            msg = "Polyfit may be poorly conditioned"
            warnings.warn(msg, UserWarning, stacklevel=2)

        # np.linalg.lstsq returns empty `resids` when rank < order or n <= order;
        # fall back to computing the weighted sum of squared residuals directly.
        # Note: `lhs` is column-scaled and `coeff` has already been rescaled back
        # to the original basis, so multiply coeff by `scale` to match `lhs`.
        if resids.size:
            resids = resids[0]
        else:
            fit = lhs @ (coeff.T * scale).T
            resids = float(np.sum((rhs - fit) ** 2))

        aic = resids + 2 * order
        bic = resids + order * np.log(len(x))
        
        res = {'coeff': coeff, 'chi2': resids, 'rank': rank, 'svs': svs, 'rcond': rcond, 
               'order': order,'dof': dof, 'aic': aic, 'bic': bic}
        
        if cov:
            # M^\beta (i.e. covar) = (X^TWX)^{-1} = (X'^TX)^{-1}
            # see "Weighted least squares" in Wikipedia for details
            covar = np.linalg.inv(np.dot(lhs.T, lhs))
            covar /= np.outer(scale, scale)
            if cov == "unscaled":
                fac = 1
            else:
                # Guard against dof <= 0 (already warned above): fall back to 1
                # so the returned covariance stays finite rather than inf/negative.
                fac = resids / dof if dof > 0 else 1.0
            if y.ndim == 1:
                covar *= fac
            else:
                covar = covar[:, :, np.newaxis] * fac
            res['covar'] = covar

        return res
