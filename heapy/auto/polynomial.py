import warnings
import numpy as np



class Polynomial(object):

    def __init__(self):
        pass


    @classmethod
    def set_method(cls, method='2pass'):
        if method == '2pass':
            pass
#             print(
# '''
# +-----------------+-----------+
# | polyfit method  |   2pass   |
# +-----------------+-----------+
# ''')
        else:
            raise ValueError('invalid method')
        
        cls_ = cls()
        cls_.method = method
        return cls_


    def fit(self, x, y, deg=None, dx=None):
        self.x = np.array(x).astype(float)
        self.y = np.array(y).astype(float)
        self.deg = deg
        self.dx = dx

        self.ls_res = None
        self.mo = None
        self.se = None

        if self.method == '2pass':
            self.two_pass()
        else:
            raise ValueError('invalid method')


    def val(self, x):
        x = np.array(x)
        
        if min(x) < min(self.x):
            msg = "External interpolation may be imprecise: %f < %f" % (min(x), min(self.x))
            warnings.warn(msg, InterpWarning, stacklevel=2)

        if max(x) > max(self.x):
            msg = "External interpolation may be imprecise: %f > %f" % (max(x), max(self.x))
            warnings.warn(msg, InterpWarning, stacklevel=2)

        assert self.ls_res is not None, 'you should first perform fitting'

        mo, se = self.ls_polyval(x, coeff=self.ls_res['coeff'], covar=self.ls_res['covar'])
        return mo, se


    def two_pass(self):
        # Two-pass fitting
        # First pass donot use the variances (i.e. uniform weight)
        # GBM and GECAM: First pass uses the data variances calculated from data rates
        # Second pass uses the data variances calculated from model rates
        # 1) rate * livetime = counts
        # 2) variance of counts = counts
        # 3) variance of rate = variance of counts/livetime^2 = rate/livetime
        time = self.x
        rate = self.y

        if self.deg is None:
            if len(time) <= 15:
                self.deg = 0
            elif 15 < len(time) <= 30:
                self.deg = 1
            elif 30 < len(time) <= 60:
                self.deg = 2
            else:
                self.deg = 3

        if self.dx is None:
            dt = time[1] - time[0]
        else:
            dt = np.array(self.dx)
            if dt.ndim != 0 and dt.shape != time.shape:
                raise TypeError("if dt is array, expected dt and time to have same length")

        variance = rate / dt
        zero = (variance <= 0)
        weight = np.piecewise(variance,
                              condlist=[zero, ~zero],
                              funclist=[lambda var: 1.0, lambda var: 1.0])  # uniform weight

        for ipass in range(2):
            self.ls_res = self.ls_polyfit(time, rate, self.deg, w=weight, cov=True)
            self.mo, self.se = self.ls_polyval(time, coeff=self.ls_res['coeff'], covar=self.ls_res['covar'])

            if ipass == 0:
                variance = self.mo / dt
                zero = (variance <= 0)
                weight = np.piecewise(variance,
                                      condlist=[zero, ~zero],
                                      funclist=[lambda var: 0.0, lambda var: np.sqrt(1 / var)])


    @staticmethod
    def ls_polyval(x, coeff, covar=None):
        if x.ndim != 1:
            raise TypeError("expected 1D vector for x")
        if x.size == 0:
            raise TypeError("expected non-empty vector for x")
        if coeff.ndim != 1:
            raise TypeError("expected 1D vector for coeff")
        if coeff.size < 1:
            raise TypeError("expected coeff.size >= 1")

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
            se = np.sqrt(np.diag(var))

            return mo, se

        else:
            return mo


    @staticmethod
    def ls_polyfit(x, y, deg, rcond=None, w=None, cov=True) -> dict:
        """
        Parameters
        ----------
        x: array_like, shape (M,)
        y: array_like, shape (M,)
        deg: int, degree of the fitting polynomial
        rcond: float, optional
        w: array_like, shape (M,), optional, w[i] = 1/sigma(y[i])
        cov: bool or str, optional
        #-------------------------#
        #see np.polyfit for details#
        #-------------------------#
        Returns
        -------
        coeff: polynomial coefficients
        resids: sum of squared residuals of the least squares fit
        rank: the effective rank of the scaled Vandermonde coefficient matrix
        svs: singular values of the scaled Vandermonde coefficient matrix
        covar: the covariance matrix of the polynomial coefficient estimates
        """
        if x.ndim != 1 or y.ndim != 1:
            raise TypeError("expected 1D vector for x and y")
        if x.size == 0 or y.size == 0:
            raise TypeError("expected non-empty vector for x and y")
        if x.shape != y.shape:
            raise TypeError("expected x and y to have same length")
        if deg < 0:
            raise ValueError("expected deg >= 0")

        order = int(deg) + 1

        if (y == 0).all():
            coeff = np.full(order, 0.0)
            res = {'coeff': coeff}
            if cov:
                covar = np.full((order, order), 0.0)
                res['covar'] = covar
            return res

        if rcond is None:
            rcond = len(x) * np.finfo(x.dtype).eps

        # set up least squares equation for powers of x
        lhs = np.vander(x, order)
        rhs = y + 0.0   # to avoid to change y

        # apply weighting
        if w is not None:
            if w.ndim != 1:
                raise TypeError("expected a 1-d array for weights")
            if w.shape != y.shape:
                raise TypeError("expected w and y to have the same length")
        else:
            w = np.ones_like(y)

        dof = np.sum(w != 0) - order
        if dof <= 0:
            msg = 'The degree of freedom should be great than zero'
            warnings.warn(msg, DofWarning, stacklevel=2)

        # X' = diag(w)X, see "Weighted least squares" in Wikipedia for details
        lhs *= w[:, np.newaxis]
        # y' = diag(w)y, see "Weighted least squares" in Wikipedia for details
        rhs *= w

        # scale lhs to improve condition number and solve
        scale = np.sqrt((lhs * lhs).sum(axis=0))
        lhs /= scale
        # solve WX@\beta = Wy, minimize sum((w*yi - w*f(xi, \beta))^2)
        # ==> X^TWy = (X^TWX)\beta ==> X'^Ty' = (X'^TX')\beta (uncorrelated w=sqrt(W))
        # see "Weighted least squares" in Wikipedia for details
        coeff, resids, rank, svs = np.linalg.lstsq(lhs, rhs, rcond)
        # broadcast scale coefficients
        coeff = coeff / scale

        # warn on rank reduction, which indicates an ill conditioned matrix
        if rank != order:
            msg = "Polyfit may be poorly conditioned"
            warnings.warn(msg, RankWarning, stacklevel=2)

        res = {'coeff': coeff, 'chi2': resids, 'rank': rank, 'svs': svs, 'rcond': rcond, 'dof': dof}

        if cov:
            # M^\beta (i.e. covar) = (X^TWX)^{-1} = (X'^TX)^{-1}
            # see "Weighted least squares" in Wikipedia for details
            covar = np.linalg.inv(np.dot(lhs.T, lhs))
            covar /= np.outer(scale, scale)
            if cov == "unscaled":
                fac = 1
            else:
                fac = resids / dof
            covar *= fac
            res['covar'] = covar

        return res


class RankWarning(UserWarning):
    """
    Issued by self.ls_polyfit when the Vandermonde matrix is rank deficient.
    """
    pass


class InterpWarning(UserWarning):
    """
    Issued by self.val for External interpolation.
    """
    pass


class DofWarning(UserWarning):
    """
    Issued by self.ls_polyfit for dof < 0.
    """
    pass
