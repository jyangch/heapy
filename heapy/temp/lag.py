import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.stats import sigma_clip, mad_std
from scipy.interpolate import UnivariateSpline
from scipy.fft import rfft, irfft, next_fast_len
from scipy.optimize import curve_fit, minimize_scalar
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from ..util.data import json_dump



class Lag(object):

    def __init__(self, 
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
                 mc=True, 
                 nmc=1000, 
                 ):
        """
        low energy band is taken as y
        high energy band is taken as x
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
        
        self.mc = mc
        self.nmc = int(nmc)
        
        if self.mc and self.nmc < 1:
            raise ValueError('nmc must be at least 1 when mc=True')
        self.nsample = len(self.xcts)
        
        if self.mc:

            self.xtype = xtype
            self.ytype = ytype

            self.xcts_err, self.xbcts, self.xbcts_err = \
                self._validate(xtype, xcts, xcts_err, xbcts, xbcts_err, 'x')
            self.ycts_err, self.ybcts, self.ybcts_err = \
                self._validate(ytype, ycts, ycts_err, ybcts, ybcts_err, 'y')

            self.xncts = self.xcts - self.xbcts
            self.yncts = self.ycts - self.ybcts

            self._mc_simulation()
            
        else:
            self.xncts = self.xcts
            self.yncts = self.ycts
            
            self.mc_xncts = self.xncts.reshape(1, -1)
            self.mc_yncts = self.yncts.reshape(1, -1)
            
        self.model_funcs = {
            'gaussian': Lag.gaussian, 
            'asymmetric_gaussian': Lag.asymmetric_gaussian, 
            'double_gaussian': Lag.double_gaussian, 
            'lorentzian': Lag.lorentzian, 
            'pseudo_voigt': Lag.pseudo_voigt}


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
    def ccf_band(d, x, y):

        x = np.asarray(x)
        y = np.asarray(y)
        N = len(x)
        sx = np.sum(x ** 2)
        sy = np.sum(y ** 2)
        nor = np.sqrt(sx * sy)
        low = max(0, -d)
        upp = min(N, N - d)
        sxy = np.dot(x[low:upp], y[low + d:upp + d])
        ccf = sxy / nor

        return ccf


    @staticmethod
    def ccfs_band(dt, x, y):

        x = np.asarray(x)
        y = np.asarray(y)
        N = len(x)
        tau = dt * np.arange(-N + 1, N, 1)
        ccfs = [Lag.ccf_band(d, x, y) for d in range(-N + 1, N, 1)]

        return tau, np.array(ccfs)


    @staticmethod
    def ccfs_scipy(dt, x, y):

        x = np.asarray(x)
        y = np.asarray(y)
        N = len(x)
        nor = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
        tau = dt * np.arange(-N + 1, N, 1)
        if nor == 0:
            ccfs = np.zeros(2 * N - 1, dtype=float)
        else:
            ccfs = signal.correlate(y, x, mode='full', method='auto') / nor

        return tau, ccfs


    @staticmethod
    def gaussian(x, cons, amp, mu, sigma):
        
        return cons + amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    
    
    @staticmethod
    def asymmetric_gaussian(x, cons, amp, mu, sigma_l, sigma_r):
        
        gaussian_l = cons + amp * np.exp(-((x - mu) ** 2) / (2 * sigma_l ** 2))
        gaussian_r = cons + amp * np.exp(-((x - mu) ** 2) / (2 * sigma_r ** 2))
        
        return gaussian_l * (x <= mu) + gaussian_r * (x > mu)
    
    
    @staticmethod
    def double_gaussian(x, cons, amp1, mu1, sigma1, amp2, mu2, sigma2):
        
        gaussian1 = amp1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
        gaussian2 = amp2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
        
        return cons + gaussian1 + gaussian2


    @staticmethod
    def lorentzian(x, cons, amp, mu, gamma):
        
        return cons + amp * (gamma ** 2 / ((x - mu) ** 2 + gamma ** 2))


    @staticmethod
    def pseudo_voigt(x, cons, amp, mu, sigma, eta):

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

        xnet = self._mc_sample(self.xtype, self.xcts, self.xcts_err, self.xbcts, self.xbcts_err)
        self.mc_xncts = np.vstack([self.xncts, xnet])

        ynet = self._mc_sample(self.ytype, self.ycts, self.ycts_err, self.ybcts, self.ybcts_err)
        self.mc_yncts = np.vstack([self.yncts, ynet])


    def _batch_ccfs(self):

        n = self.nsample
        nfft = next_fast_len(2 * n - 1)

        X_rev = rfft(self.mc_xncts[:, ::-1].copy(), n=nfft, axis=1)
        Y = rfft(self.mc_yncts, n=nfft, axis=1)
        all_ccfs = irfft(Y * X_rev, n=nfft, axis=1)[:, :2 * n - 1]

        norms = np.sqrt(np.sum(self.mc_xncts ** 2, axis=1) * np.sum(self.mc_yncts ** 2, axis=1))
        zero_mask = norms == 0
        norms[zero_mask] = 1.0
        all_ccfs /= norms[:, np.newaxis]
        all_ccfs[zero_mask] = 0.0

        return all_ccfs


    def calculate(self, method=None, width=None, threshold=None, poly_deg=None, spline_s=None):
        
        if method is None:
            method = 'gp'
            
        self.taus = self.dt * np.arange(- self.nsample + 1, self.nsample, 1)
        
        self.mc_ccfs = self._batch_ccfs()
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
            spline_s = 0.1
        
        self.nrange = (self.taus[self.nidx[0]], self.taus[self.nidx[-1]])
        self.itp_taus = np.linspace(self.nrange[0], self.nrange[1], 300)
        self.itp_ccfs = np.zeros_like(self.itp_taus, dtype=float)

        self.mc_fit_lags = []
        
        fit_taus = self.taus[self.nidx]
        fitted_kernel = None

        for i, ccfs_i in enumerate(self.mc_ccfs):
            fit_ccfs = ccfs_i[self.nidx]
            
            try:
                if method == 'polyfit':
                    polyfit = np.polyfit(fit_taus, fit_ccfs, deg=poly_deg)
                    lag_i = minimize_scalar(lambda x: -np.polyval(polyfit, x), bounds=self.nrange, method='bounded').x
                    if i == 0:
                        self.itp_ccfs = np.polyval(polyfit, self.itp_taus)
                
                elif method == 'spline':
                    spline = UnivariateSpline(fit_taus, fit_ccfs, s=spline_s)
                    lag_i = minimize_scalar(lambda x: -spline(x), bounds=self.nrange, method='bounded').x
                    if i == 0:
                        self.itp_ccfs = spline(self.itp_taus)
                        
                elif method == 'gp':
                    if fitted_kernel is None:
                        kernel = 1.0 * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.01)
                        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
                        gpr.fit(fit_taus.reshape(-1, 1), fit_ccfs)
                        fitted_kernel = gpr.kernel_
                        lag_i = minimize_scalar(lambda x: -gpr.predict([[x]])[0], bounds=self.nrange, method='bounded').x
                        self.itp_ccfs = gpr.predict(self.itp_taus.reshape(-1, 1))
                    else:
                        gpr = GaussianProcessRegressor(kernel=fitted_kernel, optimizer=None)
                        gpr.fit(fit_taus.reshape(-1, 1), fit_ccfs)
                        lag_i = minimize_scalar(lambda x: -gpr.predict([[x]])[0], bounds=self.nrange, method='bounded').x
                    
                elif method in self.model_funcs:
                    func = self.model_funcs[method]
                    popt, _ = curve_fit(func, fit_taus, fit_ccfs, maxfev=5000)
                    lag_i = minimize_scalar(lambda x: -func(x, *popt), bounds=self.nrange, method='bounded').x
                    if i == 0:
                        self.itp_ccfs = func(self.itp_taus, *popt)
                    
                else:
                    raise ValueError('unknown method')

                self.mc_fit_lags.append(lag_i)
                
            except Exception:
                if i == 0:
                    raise RuntimeError('failed to fit the CCF for the original light curves')
                continue

        lag_bv = self.mc_fit_lags[0]
        
        if self.mc:
            clipped = sigma_clip(self.mc_fit_lags, sigma=5, maxiters=5, stdfunc=mad_std)
            mc_fit_lags_filter = clipped.compressed()

            lag_lo, lag_hi = np.percentile(mc_fit_lags_filter, [16, 84])
            lag_err = np.diff([lag_lo, lag_bv, lag_hi])
            self.lag = [lag_bv, lag_err[0], lag_err[1]]
            
            print('\n+-----------------------------------------------+')
            print(' %-15s%-15s%-15s' % ('lag (s)', 'lag_le (s)', 'lag_he (s)'))
            print(' %-15.4f%-15.4f%-15.4f' % (self.lag[0], self.lag[1], self.lag[2]))
            print('+-----------------------------------------------+\n')
            
        else:
            self.lag = lag_bv
            
            print('\n+-----------------------------------------------+')
            print(' %-15s ' % 'lag (s)')
            print(' %-15.4f ' % self.lag)
            print('+-----------------------------------------------+\n')

        self.lag_res = {'lag': self.lag, 'mc_fit_lags': self.mc_fit_lags, 
                        'width': width, 'threshold': threshold, 'method': method, 
                        'taus': self.taus, 'ccfs': self.ccfs, 
                        'itp_taus': self.itp_taus, 'itp_ccfs': self.itp_ccfs}


    def save(self, savepath, suffix=''):
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.lag_res, savepath + '/lag_res%s.json'%suffix)

        rcParams['font.family'] = 'serif'
        rcParams['font.sans-serif'] = 'Georgia'
        rcParams['font.size'] = 12
        rcParams['pdf.fonttype'] = 42

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.scatter(self.taus[self.nidx], self.mc_ccfs[0][self.nidx], marker='+', 
                   color='r', s=20, linewidths=0.5, alpha=1.0)
        ax.plot(self.itp_taus, self.itp_ccfs, c='b', lw=0.5, alpha=1.0)
        ax.set_xlabel('Time delay (s)')
        ax.set_ylabel('CCF value')
        ax.minorticks_on()
        ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax.tick_params(which='major', width=1.0, length=5)
        ax.tick_params(which='minor', width=1.0, length=3)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        fig.savefig(savepath + '/tau_ccf%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)
        
        if self.mc:
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            lag_bins = np.linspace(min(self.mc_fit_lags), max(self.mc_fit_lags), 30)
            ax.hist(self.mc_fit_lags, lag_bins, density=False, histtype='step', color='b', lw=1.0)
            ax.axvline(self.lag[0], c='grey', lw=1.0)
            ax.axvline(self.lag[0] - self.lag[1], c='grey', ls='--', lw=1.0)
            ax.axvline(self.lag[0] + self.lag[2], c='grey', ls='--', lw=1.0)
            ax.set_xlabel('Lags (sec)')
            ax.set_ylabel('Counts')
            ax.set_title(r'$\tau=%.4f_{-%.4f}^{+%.4f}~{\rm s}$'%(self.lag[0], self.lag[1], self.lag[2]))
            ax.minorticks_on()
            ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
            ax.tick_params(which='major', width=1.0, length=5)
            ax.tick_params(which='minor', width=1.0, length=3)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            fig.savefig(savepath + '/lag_pdf%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)
