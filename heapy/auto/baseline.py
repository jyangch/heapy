import warnings
import numpy as np
import pybaselines
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from astropy.stats import sigma_clip, mad_std



class Baseline(object):

    def __init__(self, method='drpls'):
        
        self.method = method


    @classmethod
    def set_method(cls, method='drpls'):
        
        if method == 'drpls':
            pass
        else:
            raise ValueError('invalid method')
        
        return cls(method=method)


    def fit(self, x, y, w=None, lam=None, nk=None):
        
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
            mo = pybaselines.spline.mixture_model(y, p=0.001, lam=lam, diff_order=2, num_knots=nk, weights=w)[0]

        else:
            raise ValueError('invalid method')

        self.mo = mo


    def val(self, x):

        x = np.asarray(x)

        if x.min() < self.x.min() or x.max() > self.x.max():
            warnings.warn("Extrapolation may be imprecise", UserWarning, stacklevel=2)

        interp = CubicSpline(self.x, self.mo, extrapolate=True)
        
        return interp(x)


    @staticmethod
    def speyediff(N, d, format='csc'):
        
        assert d >= 0, "d must be non-negative"
        
        shape = (N - d, N)
        diagonals = np.zeros(2 * d + 1)
        diagonals[d] = 1.0
        for i in range(d):
            diff = diagonals[:-1] - diagonals[1:]
            diagonals = diff
        offsets = np.arange(d + 1)
        spmat = sparse.diags(diagonals, offsets, shape, format=format)
        
        return spmat


    def whittaker_smooth(self, y, lmbd, d, w):

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
        
        y = np.array(y)
        m = len(y)
        if w is None:
            w = np.ones(m)
        if d is None:
            d = 3
        if lmbd is None:
            lmbd = 10 ** (4*(np.log10(len(y))-1))
        _, z = self.whittaker_smooth(y, lmbd, d, w)

        cs = y - z
        for sigma in [3, 2, 1]:
            mask = sigma_clip(cs, sigma=sigma, maxiters=5, stdfunc=mad_std).mask
            w[mask] = 0
            _, z = self.whittaker_smooth(y, lmbd, d, w)
            cs = y - z

        return z


    def _snip_core(self, y, x, w=None, d=None, lmbd=None, inte=None, pr=None, hwi=None, it=None):
        """
        Shared SNIP-iteration core used by both `snip` and `isnip`.

        Parameters (same meaning as snip/isnip):
          lmbd : smooth hardness
          it   : number of SNIP iterations
          inte : number of sampling intervals
          hwi  : half window width
          pr   : peak ratio

        Returns (y, idx, poly_z): input y, the full integer index array, and the
        cubic-polyfit baseline estimated from the SNIP-downsampled points.
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
            win = np.ceil(np.concatenate(
                (10 ** (d1 + np.arange(it - 1) * (d2 - d1) / (it - 1)), [d2])))
            win = win.astype(int)
        else:
            win = np.array([hwi], dtype=int)

        lims = np.linspace(0, z.size - 1, inte + 1)
        lefts = np.ceil(lims[:-1]).astype(int)
        rights = np.floor(lims[1:]).astype(int)
        samp_x = np.round((lefts + rights) * 0.5)

        bsl_z = np.zeros(inte)
        for i in range(inte):
            bsl_z[i] = z[lefts[i]:rights[i] + 1].mean()

        # SNIP iteration: symmetric forward + reverse sweeps at shrinking windows
        for i in range(it):
            win0 = win[i]
            for j in range(1, inte - 1):
                v = min(j, win0, inte - j - 1)
                a = bsl_z[j - v:j + v + 1].mean()
                bsl_z[j] = min(a, bsl_z[j])
            for j in range(1, inte - 1):
                k = inte - j - 1
                v = min(j, win0, inte - j - 1)
                a = bsl_z[k - v:k + v + 1].mean()
                bsl_z[k] = min(a, bsl_z[k])

        samp_x = np.concatenate(([0], samp_x, [z.size - 1]))
        bsl_z = np.concatenate((bsl_z[:1], bsl_z, bsl_z[-1:]))
        idx = np.arange(z.size)

        poly_fit = np.polyfit(samp_x, bsl_z, deg=3)
        poly_z = np.polyval(poly_fit, idx)

        return y, idx, poly_z


    def snip(self, y, x, w=None, d=None, lmbd=None, inte=None, pr=None, hwi=None, it=None):

        _, _, poly_z = self._snip_core(y, x, w, d, lmbd, inte, pr, hwi, it)
        
        return poly_z


    def isnip(self, y, x, w=None, d=None, lmbd=None, inte=None, pr=None, hwi=None, it=None):

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
