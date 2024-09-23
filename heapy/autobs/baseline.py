import operator
import warnings
import numpy as np
import pybaselines
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from astropy.stats import sigma_clip, mad_std
from scipy.interpolate import interp1d, UnivariateSpline



class Baseline(object):

    def __init__(self):
        pass


    @classmethod
    def set_method(cls, method='drpls'):
        if method == 'drpls':
            pass
#             print(
# '''
# +-----------------+-----------+
# | baseline method |   drpls   |
# +-----------------+-----------+
# ''')
        else:
            raise ValueError('invalid method')
        
        cls_ = cls()
        cls_.method = method
        return cls_


    def fit(self, x, y, w=None, lam=None, nk=None):
        self.x = np.array(x).astype(float)
        self.y = np.array(y).astype(float)

        if w is None:
            w = np.ones(len(y))
        if lam is None:
            lam = 10 ** (4 * (np.log10(len(x)) - 1))
        if nk is None:
            nk = 2 * len(x) - 200

        if self.method == 'snip':
            mo = self.snip(y, x, w=w)           # for same binsize

        elif self.method == 'isnip':
            mo = self.isnip(y, x, w=w)          # for same binsize

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
        x = np.array(x)

        if min(x) < min(self.x) or max(x) > max(self.x):
            msg = "External interpolation may be imprecise"
            warnings.warn(msg, InterpWarning, stacklevel=2)
            interp = UnivariateSpline(self.x, self.mo, w=None, k=3, s=0)
            mo = interp(x)
        else:
            interp = interp1d(self.x, self.mo, kind='quadratic')
            mo = interp(x)

        return mo


    @staticmethod
    def speyediff(N, d, format='csc'):
        assert not (d < 0)
        shape = (N - d, N)
        diagonals = np.zeros(2 * d + 1)
        diagonals[d] = 1.
        for i in range(d):
            diff = diagonals[:-1] - diagonals[1:]
            diagonals = diff
        offsets = np.arange(d + 1)
        spmat = sparse.diags(diagonals, offsets, shape, format=format)
        return spmat


    @staticmethod
    def whittaker_smooth(self, y, lmbd, d, w):
        y = np.array(y)
        m = len(y)
        y = np.mat(y)
        # E = sparse.eye(m, format='csc')
        D = self.speyediff(m, d, format='csc')
        W = sparse.diags(w, 0, shape=(m, m))
        coefmat = W + lmbd * D.conj().T.dot(D)
        z = spsolve(coefmat, W * y.T)
        return coefmat, z


    @staticmethod
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


    def snip(self, y, x, w=None, d=None, lmbd=None, inte=None, pr=None, hwi=None, it=None):
        #--------------------------
        #lmbd: smooth hardness
        #it: iterations
        #inte: number of samples
        #hwi: width of window
        #pr: peak ratio
        #for same binsize
        #--------------------------
        y, x = np.array(y), np.array(x)
        m = len(y)
        if w is None:
            w = np.ones(m)
        z = self.get_smooth(y, w, d, lmbd)

        dt = x[1] - x[0]
        if (it is None):
            it = 5
        if (inte is None):
            #inte = round(len(z)/(10*dt**(1/30)))
            inte = round(len(z) * dt / 5)
        if pr is None:
            pr = 0.4
        if (hwi is None):
            hwi = round(inte / 2 * pr)

        if (it != 1):
            d1 = np.log10(hwi)
            d2 = 0
            win = np.ceil(np.concatenate((10 ** (d1 + np.arange(0, it - 1, 1) * (d2 - d1) / (np.floor(it) - 1)), [d2])))
            win = np.array(win, dtype=int)
        else:
            win = np.array([hwi], dtype=int)

        lims = np.linspace(0, z.size - 1, inte + 1)
        lefts = np.array(np.ceil(lims[:-1]), dtype=int)
        rights = np.array(np.floor(lims[1:]), dtype=int)
        samp_x = np.round((lefts + rights) * 0.5)

        bsl_z = np.zeros(inte)
        for i in range(inte):
            bsl_z[i] = z[lefts[i]:rights[i] + 1].mean()

        for i in range(it):
            # Current window width
            win0 = win[i]
            # Point-wise iteration to the right
            for j in range(1, inte - 1):
                # Interval cut-off close to edges
                v = min([j, win0, inte - j - 1])
                # Baseline suppression
                a = bsl_z[j - v:j + v + 1].mean()
                bsl_z[j] = min([a, bsl_z[j]])
                # xx[j] = np.mean([a, xx[j]])
            for j in range(1, inte - 1):
                k = inte - j - 1
                # Interval cut-off close to edges
                v = min([j, win0, inte - j - 1])
                # Baseline suppression
                a = bsl_z[k - v:k + v + 1].mean()
                bsl_z[k] = min([a, bsl_z[k]])
                # xx[k] = np.mean([a,xx[k]])

        samp_x = np.concatenate(([0], samp_x, [z.size - 1]))
        bsl_z = np.concatenate((bsl_z[:1], bsl_z, bsl_z[-1:]))
        idx = np.arange(0, z.size, 1)

        poly_fit = np.polyfit(samp_x, bsl_z, deg=3)
        poly_z = np.polyval(poly_fit, idx)

        return poly_z


    def isnip(self, y, x, w=None, d=None, lmbd=None, inte=None, pr=None, hwi=None, it=None):
        #--------------------------
        #lmbd: smooth hardness
        #it: iterations
        #inte: number of samples
        #hwi: width of window
        #pr: peak ratio
        #for same binsize
        #--------------------------
        y, x = np.array(y), np.array(x)
        m = len(y)
        if w is None:
            w = np.ones(m)
        z = self.get_smooth(y, w, d, lmbd)

        dt = x[1] - x[0]
        if (it is None):
            it = 5
        if (inte is None):
            #inte = int(len(z) / 5)
            inte = round(len(z) * dt / 5)
        if pr is None:
            pr = 0.4
        if (hwi is None):
            #hwi = int(10 / dt)
            hwi = round(inte / 2 * pr)

        if (it != 1):
            d1 = np.log10(hwi)
            d2 = 0
            win = np.ceil(np.concatenate((10 ** (d1 + np.arange(0, it - 1, 1) * (d2 - d1) / (np.floor(it) - 1)), [d2])))
            win = np.array(win, dtype=int)
        else:
            win = np.array([hwi], dtype=int)

        lims = np.linspace(0, z.size - 1, inte + 1)
        lefts = np.array(np.ceil(lims[:-1]), dtype=int)
        rights = np.array(np.floor(lims[1:]), dtype=int)
        samp_x = np.round((lefts + rights) * 0.5)

        bsl_z = np.zeros(inte)
        for i in range(inte):
            bsl_z[i] = z[lefts[i]:rights[i] + 1].mean()

        for i in range(it):
            # Current window width
            win0 = win[i]
            # Point-wise iteration to the right
            for j in range(1, inte - 1):
                # Interval cut-off close to edges
                v = min([j, win0, inte - j - 1])
                # Baseline suppression
                a = bsl_z[j - v:j + v + 1].mean()
                bsl_z[j] = min([a, bsl_z[j]])
                # xx[j] = np.mean([a, xx[j]])
            for j in range(1, inte - 1):
                k = inte - j - 1
                # Interval cut-off close to edges
                v = min([j, win0, inte - j - 1])
                # Baseline suppression
                a = bsl_z[k - v:k + v + 1].mean()
                bsl_z[k] = min([a, bsl_z[k]])
                # xx[k] = np.mean([a,xx[k]])

        samp_x = np.concatenate(([0], samp_x, [z.size - 1]))
        bsl_z = np.concatenate((bsl_z[:1], bsl_z, bsl_z[-1:]))
        idx = np.arange(0, z.size, 1)

        poly_fit = np.polyfit(samp_x, bsl_z, deg=3)
        poly_z0 = np.polyval(poly_fit, idx)

        poly_cs = y - poly_z0
        for sigma in [3, 2, 1]:
            mask = sigma_clip(poly_cs, sigma=sigma, maxiters=5, stdfunc=mad_std).mask
            myfilter = list(map(operator.not_, mask))
            idx_filter = idx[myfilter]
            y_filter = y[myfilter]

            poly_fit = np.polyfit(idx_filter, y_filter, deg=3)
            poly_z = np.polyval(poly_fit, idx)
            poly_cs = y - poly_z

        return poly_z


class InterpWarning(UserWarning):
    """
    Issued by self.val for External interpolation.
    """
    pass
