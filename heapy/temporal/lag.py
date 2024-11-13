import os
import operator
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.modeling import models
from astropy.stats import sigma_clip, mad_std
from ..util.data import json_dump



class CCF(object):

    def __init__(self):
        
        pass


    @staticmethod
    def ccf_band(d, x, y):
        
        x = np.array(x)
        y = np.array(y)
        N = len(x)
        sx = np.sum(x ** 2)
        sy = np.sum(y ** 2)
        nor = np.sqrt(sx * sy)
        low = max(1, 1 - d)
        upp = min(N, N - d)
        sxy = np.sum([x[i] * y[i+d] for i in range(low - 1, upp, 1)])
        ccf = sxy / nor
        
        return ccf


    @staticmethod
    def ccfs_band(self, dt, x, y):
        
        x = np.array(x)
        y = np.array(y)
        N = len(x)
        tau = [dt * d for d in range(-N + 1, N, 1)]
        ccfs = [self.ccf_band(d, x, y) for d in range(-N + 1, N, 1)]
        
        return np.array(tau), np.array(ccfs)


    @staticmethod
    def ccfs_scipy(dt, x, y):
        
        x = np.array(x)
        y = np.array(y)
        N = len(x)
        sx = np.sum(x ** 2)
        sy = np.sum(y ** 2)
        nor = np.sqrt(sx * sy)
        tau = [dt * d for d in range(-N + 1, N, 1)]
        # ccfs = np.correlate(y, x, mode='full') / nor
        ccfs = signal.correlate(y, x, mode='full', method='auto') / nor
        
        return np.array(tau), np.array(ccfs)



class Lag(CCF):

    def __init__(self, 
                 xcts, 
                 ycts, 
                 xbcts,
                 ybcts,
                 dt,
                 xtype='pg', 
                 ytype='pg',
                 xbcts_se=None,
                 ybcts_se=None,
                 xsigidx=None,
                 ysigidx=None):
        
        super().__init__()

        self.xcts = xcts
        self.ycts = ycts
        
        self.xcts_se = np.sqrt(xcts)
        self.ycts_se = np.sqrt(ycts)

        self.xbcts = xbcts
        self.ybcts = ybcts
        
        self.dt = dt
        
        self.xtype = xtype
        self.ytype = ytype
        
        if xbcts_se is None:
            if xtype == 'pp':
                self.xbcts_se = np.sqrt(xbcts)
            else:
                raise ValueError('unknown xbcts error')
        else:
            self.xbcts_se = xbcts_se
            
        if ybcts_se is None:
            if ytype == 'pp':
                self.ybcts_se = np.sqrt(ybcts)
            else:
                raise ValueError('unknown ybcts error')
        else:
            self.ybcts_se = ybcts_se  

        self.xncts = self.xcts - self.xbcts
        self.yncts = self.ycts - self.ybcts
        
        self.xsigidx = xsigidx
        self.ysigidx = ysigidx


    @staticmethod
    def gmo(x, amp, mean, std):
        
        g = models.Gaussian1D(amplitude=amp, mean=mean, stddev=std)
        
        return g(x)
    

    def mc_simulation(self, nmc):
        
        self.nmc = int(nmc)
        self.N = len(self.xcts)
        self.mc_xncts = [self.xncts]
        self.mc_yncts = [self.yncts]
        for _ in range(self.nmc):
            if self.xtype == 'pg':
                xmc = np.random.poisson(lam=self.xcts) \
                    - (self.xbcts_se * np.random.randn(self.N) + self.xbcts)
            elif self.xtype == 'pp':
                xmc = np.random.poisson(lam=self.xcts) \
                    - np.random.poisson(lam=self.xbcts)
            else:
                raise TypeError("invalid xtype")

            if self.ytype == 'pg':
                ymc = np.random.poisson(lam=self.ycts) \
                    - (self.ybcts_se * np.random.randn(self.N) + self.ybcts)
            elif self.ytype == 'pp':
                ymc = np.random.poisson(lam=self.ycts) \
                    - np.random.poisson(lam=self.ybcts)
            else:
                raise TypeError("invalid ytype")
            
            self.mc_xncts.append(xmc)
            self.mc_yncts.append(ymc)


    def callag(self, width=5, deg=5):
        
        self.taus, ccfs0 = self.ccfs_scipy(self.dt, self.mc_xncts[0], self.mc_yncts[0])

        pidx = np.argmax(ccfs0)
        self.nidx = np.arange(pidx - width if pidx >= width else 0, pidx + width + 1, 1)
        self.itp_taus = np.arange(self.taus[self.nidx[0]], self.taus[self.nidx[-1]], 1e-4)

        self.mc_ccfs, self.mc_itp_ccfs = [], []
        self.mc_peak_lags, self.mc_fit_lags = [], []
        for _, (x, y) in enumerate(zip(self.mc_xncts, self.mc_yncts)):
            _, ccfs_i = self.ccfs_scipy(self.dt, x, y)
            self.mc_ccfs.append(ccfs_i)

            poly_fit = np.polyfit(self.taus[self.nidx], ccfs_i[self.nidx], deg=deg)
            itp_ccfs_i = np.polyval(poly_fit, self.itp_taus)
            self.mc_itp_ccfs.append(itp_ccfs_i)

            lag_i = self.taus[np.argmax(ccfs_i)]
            self.mc_peak_lags.append(lag_i)

            lag_i = self.itp_taus[np.argmax(itp_ccfs_i)]
            self.mc_fit_lags.append(lag_i)

        lag_bv = self.mc_fit_lags[0]

        mask = sigma_clip(self.mc_fit_lags, sigma=5, maxiters=5, stdfunc=mad_std).mask
        not_mask = list(map(operator.not_, mask))
        mc_fit_lags_filter = np.array(self.mc_fit_lags)[not_mask]

        lag_lo, lag_hi = np.percentile(mc_fit_lags_filter, [16, 84])
        lag_err = np.diff([lag_lo, lag_bv, lag_hi])
        self.lag = [lag_bv, lag_err[0], lag_err[1]]

        self.lag_res = {'lag': self.lag, 'width': width, 'deg': deg, 
                        'mc_peak_lags': self.mc_peak_lags, 'mc_fit_lags': self.mc_fit_lags, 
                        'ccfs': self.mc_ccfs[0], 'itp_ccfs': self.mc_itp_ccfs[0], 
                        'taus': self.taus, 'itp_taus': self.itp_taus}

        print('\n+-----------------------------------------------+')
        print(' %-15s%-15s%-15s' % ('lag (s)', 'lag_le (s)', 'lag_he (s)'))
        print(' %-15.4f%-15.4f%-15.4f' % (self.lag[0], self.lag[1], self.lag[2]))
        print('+-----------------------------------------------+\n')


    def save(self, savepath, suffix=''):
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.lag_res, savepath + '/lag_res%s.json'%suffix)

        rcParams['font.family'] = 'serif'
        rcParams['font.sans-serif'] = 'Georgia'
        rcParams['text.usetex'] = True
        rcParams['font.size'] = 12
        rcParams['pdf.fonttype'] = 42

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.scatter(self.taus[self.nidx], self.mc_ccfs[0][self.nidx], marker='+', 
                   color='r', s=20, linewidths=0.5, alpha=1.0)
        ax.plot(self.itp_taus, self.mc_itp_ccfs[0], c='b', lw=0.5, alpha=1.0)
        for i in np.arange(1, self.nmc + 1, 100):
            ax.scatter(self.taus[self.nidx], self.mc_ccfs[i][self.nidx], marker='+', 
                       color='grey', s=20, linewidths=0.5, alpha=1.0)
            ax.plot(self.itp_taus, self.mc_itp_ccfs[i], c='grey', lw=0.5, alpha=1.0)
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
