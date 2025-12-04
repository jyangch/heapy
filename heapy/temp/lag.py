import os
import operator
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
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
    def ccfs_band(dt, x, y):
        
        x = np.array(x)
        y = np.array(y)
        N = len(x)
        tau = [dt * d for d in range(-N + 1, N, 1)]
        ccfs = [CCF.ccf_band(d, x, y) for d in range(-N + 1, N, 1)]
        
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
        ccfs = signal.correlate(y, x, mode='full', method='auto') / nor
        
        return np.array(tau), np.array(ccfs)



class Lag(CCF):

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
        
        super().__init__()

        self.xcts = xcts
        self.ycts = ycts
        
        self.dt = dt
        
        self.mc = mc
        self.nmc = nmc
        self.nsample = len(self.xcts)
        
        if self.mc:
        
            self.xtype = xtype
            self.ytype = ytype
            
            if xtype == 'pg':
                if xcts_err is None:
                    self.xcts_err = np.sqrt(xcts)
                else:
                    self.xcts_err = xcts_err
                    
                if xbcts is None:
                    raise ValueError('unknown xbcts')
                else:
                    self.xbcts = xbcts
                    
                if xbcts_err is None:
                    raise ValueError('unknown xbcts_err')
                else:
                    self.xbcts_err = xbcts_err
            
            elif xtype == 'pp':
                if xcts_err is None:
                    self.xcts_err = np.sqrt(xcts)
                else:
                    self.xcts_err = xcts_err
                    
                if xbcts is None:
                    raise ValueError('unknown xbcts')
                else:
                    self.xbcts = xbcts
                    
                if xbcts_err is None:
                    self.xbcts_err = np.sqrt(xbcts)
                else:
                    self.xbcts_err = xbcts_err
                    
            elif xtype == 'gg':
                if xcts_err is None:
                    raise ValueError('unknown xcts_err')
                else:
                    self.xcts_err = xcts_err
                    
                if xbcts is None:
                    self.xbcts = np.zeros_like(xcts)
                else:
                    self.xbcts = xbcts
                    
                if xbcts_err is None:
                    self.xbcts_err = np.zeros_like(xcts)
                else:
                    self.xbcts_err = xbcts_err
                    
            else:
                raise ValueError('unknown xtype')
                
            if ytype == 'pg':
                if ycts_err is None:
                    self.ycts_err = np.sqrt(ycts)
                else:
                    self.ycts_err = ycts_err
                    
                if ybcts is None:
                    raise ValueError('unknown ybcts')
                else:
                    self.ybcts = ybcts
                    
                if ybcts_err is None:
                    raise ValueError('unknown ybcts_err')
                else:
                    self.ybcts_err = ybcts_err
            
            elif ytype == 'pp':
                if ycts_err is None:
                    self.ycts_err = np.sqrt(ycts)
                else:
                    self.ycts_err = ycts_err
                    
                if ybcts is None:
                    raise ValueError('unknown ybcts')
                else:
                    self.ybcts = ybcts
                    
                if ybcts_err is None:
                    self.ybcts_err = np.sqrt(ybcts)
                else:
                    self.ybcts_err = ybcts_err
                    
            elif ytype == 'gg':
                if ycts_err is None:
                    raise ValueError('unknown ycts_err')
                else:
                    self.ycts_err = ycts_err
                    
                if ybcts is None:
                    self.ybcts = np.zeros_like(ycts)
                else:
                    self.ybcts = ybcts
                    
                if ybcts_err is None:
                    self.ybcts_err = np.zeros_like(ycts)
                else:
                    self.ybcts_err = ybcts_err
                    
            else:
                raise ValueError('unknown ytype')
            
            self.xncts = self.xcts - self.xbcts
            self.yncts = self.ycts - self.ybcts
            
            self._mc_simulation()
            
        else:
            self.xncts = xcts
            self.yncts = ycts
            
            self.mc_xncts = [self.xncts]
            self.mc_yncts = [self.yncts]


    @staticmethod
    def gaussian(x, cons, amp, mu, sigma):
        
        return cons + amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    
    
    @staticmethod
    def asymmetric_gaussian(x, cons, amp, mu, sigma_l, sigma_r):
        
        gaussian_l = cons + amp * np.exp(-((x - mu) ** 2) / (2 * sigma_l ** 2))
        gaussian_r = cons + amp * np.exp(-((x - mu) ** 2) / (2 * sigma_r ** 2))
        
        return gaussian_l * (x <= mu) + gaussian_r * (x > mu)


    def _mc_simulation(self):
        
        if self.xtype == 'pg':
            xsrc_sample = np.random.poisson(lam=self.xcts, size=(self.nmc, self.nsample))
            xbkg_sample = np.random.normal(loc=self.xbcts, scale=self.xbcts_err, size=(self.nmc, self.nsample))
        elif self.xtype == 'pp':
            xsrc_sample = np.random.poisson(lam=self.xcts, size=(self.nmc, self.nsample))
            xbkg_sample = np.random.poisson(lam=self.xbcts, size=(self.nmc, self.nsample))
        elif self.xtype == 'gg':
            xsrc_sample = np.random.normal(loc=self.xcts, scale=self.xcts_err, size=(self.nmc, self.nsample))
            xbkg_sample = np.random.normal(loc=self.xbcts, scale=self.xbcts_err, size=(self.nmc, self.nsample))
        else:
            raise TypeError("invalid xtype")
        
        self.mc_xncts = np.vstack([self.xncts, xsrc_sample - xbkg_sample])
        
        if self.ytype == 'pg':
            ysrc_sample = np.random.poisson(lam=self.ycts, size=(self.nmc, self.nsample))
            ybkg_sample = np.random.normal(loc=self.ybcts, scale=self.ybcts_err, size=(self.nmc, self.nsample))
        elif self.ytype == 'pp':
            ysrc_sample = np.random.poisson(lam=self.ycts, size=(self.nmc, self.nsample))
            ybkg_sample = np.random.poisson(lam=self.ybcts, size=(self.nmc, self.nsample))
        elif self.ytype == 'gg':
            ysrc_sample = np.random.normal(loc=self.ycts, scale=self.ycts_err, size=(self.nmc, self.nsample))
            ybkg_sample = np.random.normal(loc=self.ybcts, scale=self.ybcts_err, size=(self.nmc, self.nsample))
        else:
            raise TypeError("invalid ytype")

        self.mc_yncts = np.vstack([self.yncts, ysrc_sample - ybkg_sample])


    def calculate(self, width=3, method='polyfit'):
        
        self.taus, ccfs0 = self.ccfs_scipy(self.dt, self.mc_xncts[0], self.mc_yncts[0])

        pidx = np.argmax(ccfs0)
        
        if width is None:
            self.nidx = np.arange(0, len(ccfs0), 1)
        else:
            self.nidx = np.arange(pidx - width if pidx >= width else 0, pidx + width + 1, 1)
            
        self.itp_taus = np.arange(self.taus[self.nidx[0]], self.taus[self.nidx[-1]], 1e-4)
        
        mu_init = self.taus[pidx]
        sigma_init = (self.taus[self.nidx[-1]] - self.taus[self.nidx[0]]) / 6
        cons_init = (self.taus[self.nidx[-1]] + self.taus[self.nidx[0]]) / 2
        amp_init = np.max(ccfs0) - cons_init

        self.mc_ccfs, self.mc_itp_ccfs = [], []
        self.mc_peak_lags, self.mc_fit_lags = [], []
        for i, (x, y) in enumerate(zip(self.mc_xncts, self.mc_yncts)):
            _, ccfs_i = self.ccfs_scipy(self.dt, x, y)
            self.mc_ccfs.append(ccfs_i)
            
            if method == 'spline':
                cubic_spline = CubicSpline(self.taus[self.nidx], ccfs_i[self.nidx], bc_type='natural')
                itp_ccfs_i = cubic_spline(self.itp_taus)
                self.mc_itp_ccfs.append(itp_ccfs_i)
                
            elif method == 'polyfit':
                poly_fit = np.polyfit(self.taus[self.nidx], ccfs_i[self.nidx], deg=4)
                itp_ccfs_i = np.polyval(poly_fit, self.itp_taus)
                self.mc_itp_ccfs.append(itp_ccfs_i)
                
            elif method == 'gaussian':
                popt, _ = curve_fit(Lag.gaussian, self.taus[self.nidx], ccfs_i[self.nidx], 
                                    p0=[cons_init, amp_init, mu_init, sigma_init])
                itp_ccfs_i = Lag.gaussian(self.itp_taus, *popt)
                self.mc_itp_ccfs.append(itp_ccfs_i)
                
            elif method == 'asymmetric_gaussian':
                popt, _ = curve_fit(Lag.asymmetric_gaussian, self.taus[self.nidx], ccfs_i[self.nidx], 
                                    p0=[cons_init, amp_init, mu_init, sigma_init, sigma_init])
                itp_ccfs_i = Lag.asymmetric_gaussian(self.itp_taus, *popt)
                self.mc_itp_ccfs.append(itp_ccfs_i)
                
            else:
                raise ValueError('unknown method')
            
            lag_i = self.taus[np.argmax(ccfs_i)]
            self.mc_peak_lags.append(lag_i)

            lag_i = self.itp_taus[np.argmax(itp_ccfs_i)]
            self.mc_fit_lags.append(lag_i)

        lag_bv = self.mc_fit_lags[0]
        
        if self.mc:
            mask = sigma_clip(self.mc_fit_lags, sigma=5, maxiters=5, stdfunc=mad_std).mask
            not_mask = list(map(operator.not_, mask))
            mc_fit_lags_filter = np.array(self.mc_fit_lags)[not_mask]

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

        self.lag_res = {'lag': self.lag, 'width': width, 'method': method, 
                        'mc_peak_lags': self.mc_peak_lags, 'mc_fit_lags': self.mc_fit_lags, 
                        'ccfs': self.mc_ccfs[0], 'itp_ccfs': self.mc_itp_ccfs[0], 
                        'taus': self.taus, 'itp_taus': self.itp_taus}


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
        ax.plot(self.itp_taus, self.mc_itp_ccfs[0], c='b', lw=0.5, alpha=1.0)
        # if self.mc:
        #     for i in np.random.choice(np.arange(1, self.nmc + 1), 10):
        #         ax.scatter(self.taus[self.nidx], self.mc_ccfs[i][self.nidx], marker='+', 
        #             color='grey', s=20, linewidths=0.5, alpha=1.0)
        #         ax.plot(self.itp_taus, self.mc_itp_ccfs[i], c='grey', lw=0.5, alpha=1.0)
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
