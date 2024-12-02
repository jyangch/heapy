import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.stats import bayesian_blocks
from .baseline import Baseline
from .polynomial import Polynomial
from ..util.significance import pgsig, ppsig
from ..util.data import union, msg_format, json_dump



class PolyBase(object):
    
    """
    NOTE: only apply to possion data with gaussian background
    """

    def __init__(self, ts, bins, exp=None):
        self.ts = np.array(ts).astype(float)
        self.bins = np.array(bins).astype(float)

        cts, _ = np.histogram(self.ts, bins=self.bins)
        self.cts = np.array(cts).astype(int)

        self.lbins = self.bins[:-1]
        self.rbins = self.bins[1:]
        self.binsize = self.bins[1:] - self.bins[:-1]
        if exp is None:
            self.exp = self.binsize
        else:
            self.exp = np.array(exp).astype(float)

        if (self.exp.size + 1) != self.bins.size:
            raise TypeError("expected size(exp) + 1 = size(bins)")
        if not (self.exp <= self.binsize).all():
            raise TypeError("expected exp <= binsize")
        
        self.time = (self.lbins + self.rbins) / 2
        self.rate = self.cts / self.exp

        self.ini_res = {'time': self.time, 'cts': self.cts, 
                        'rate': self.rate, 'exp': self.exp, 
                        'bins': self.bins}

        # basefit method
        self.base_res = None

        # bblock method
        self.block_res = None

        # calsnr method
        self.snr_res = None

        # fetsig method
        self.sort_res = None

        # polyfit method
        self.poly_res = None


    @classmethod
    def frombin(cls, cts, bins, exp=None):
        cts = np.array(cts)
        bins = np.array(bins)

        if bins.size != (cts.size + 1):
            raise TypeError("expected size(bins) = size(cts)+1")
        
        msg = 'rebuilt the time list, but may not accurate'
        warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
        ts = np.array([])
        for t1, t2, n in zip(bins[:-1], bins[1:], cts):
            ts = np.append(ts, np.random.random(size=int(n)) * (t2 - t1) + t1)

        sbs_ = cls(ts, bins, exp)
        return sbs_
    
    
    def bblock(self, p0=0.05):
        pos = np.where(self.cts > 0)[0]
        edges_ = bayesian_blocks(self.time[pos], self.cts[pos], fitness='events', p0=p0)

        edges = [self.time[0], edges_[0], edges_[-1], self.time[-1]]
        for i in range(1, len(edges_) - 1, 1):
            flag1 = (edges_[i] - edges_[i-1]) > np.min(self.binsize) / 1.8
            flag2 = (edges_[i+1] - edges_[i]) > np.min(self.binsize) / 1.8
            if flag1 and flag2:
                edges.append(edges_[i])
        self.edges = np.unique(edges)
        self.nblock = len(self.edges) - 1
        self.re_binsize = self.edges[1:] - self.edges[:-1]
        
        self.re_cts, _ = np.histogram(self.ts, bins=self.edges)

        self.block_res = {'edges': self.edges, 'nblock': self.nblock, 're_binsize': self.re_binsize}


    def basefit(self, weight=None, ignore=None):
        if self.block_res is None: self.bblock()
        
        if weight is None:
            weight = np.ones_like(self.time)

        if ignore is not None:
            ignore_idx = []
            for i, (l, u) in enumerate(zip(self.lbins, self.rbins)):
                for low, upp in ignore:
                    if not (u <= low or l >= upp):
                        ignore_idx.append(i)
            ignore_idx = np.unique(ignore_idx).astype(int)
            weight[ignore_idx] = 0

        pos = np.where(self.cts > 0)[0]
        self.bl = Baseline.set_method('drpls')
        self.bl.fit(self.time[pos], self.rate[pos], w=weight[pos], lam=None, nk=None)
        self.bak = self.bl.val(self.time)
        self.bcts = self.bak * self.exp

        # plus zero for copy
        self.base_res = {'bcts': self.bcts + 0, 'bak': self.bak + 0}


    def calsnr(self):
        if self.base_res is None: self.basefit()
        
        if self.poly_res is not None:
            self.re_bcts = np.empty(self.nblock)
            self.re_bcts_se = np.empty(self.nblock)
            for i, (l, r) in enumerate(zip(self.edges[:-1], self.edges[1:])):
                x = np.linspace(l, r, 100)
                y, y_se = self.poly.val(x)
                self.re_bcts[i] = np.trapz(y, x)
                self.re_bcts_se[i] = np.trapz(y_se, x)
        else:
            self.re_bcts = np.empty(self.nblock)
            for i, (l, r) in enumerate(zip(self.edges[:-1], self.edges[1:])):
                x = np.linspace(l, r, 100)
                y = self.bl.val(x)
                self.re_bcts[i] = np.trapz(y, x)

        self.snr = np.zeros_like(self.binsize)
        for i in range(len(self.binsize)):
            cts_i = self.cts[i]
            bcts_i = self.bcts[i]
            try:
                bcts_se_i = self.bcts_se[i]
            except:
                if bcts_i <= 0 or cts_i <= 0:
                    snr_i = -5      # bad events
                else:
                    snr_i = ppsig(cts_i, bcts_i, 1)
            else:
                if bcts_i <= 0 or cts_i <= 0 or bcts_se_i == 0:
                    snr_i = -5      # bad events
                else:
                    snr_i = pgsig(cts_i, bcts_i, bcts_se_i)
            self.snr[i] = snr_i
            
        self.re_snr = np.zeros_like(self.re_binsize)
        for i in range(len(self.re_binsize)):
            cts_i = self.re_cts[i]
            bcts_i = self.re_bcts[i]
            size_i = self.re_binsize[i]
            alpha = size_i / (20 * self.binsize.mean())

            if alpha > 1:
                cts_i = cts_i / alpha
                bcts_i = bcts_i / alpha

            try:
                bcts_se_i = self.re_bcts_se[i]
            except:
                if bcts_i <= 0 or cts_i <= 0:
                    snr_i = -5      # bad events
                else:
                    snr_i = ppsig(cts_i, bcts_i, 1)
            else:
                if bcts_i <= 0 or cts_i <= 0 or bcts_se_i == 0:
                    snr_i = -5      # bad events
                else:
                    snr_i = pgsig(cts_i, bcts_i, bcts_se_i)
            self.re_snr[i] = snr_i

        self.snr_res = {'snr': self.snr, 're_snr': self.re_snr, 
                        'cts': self.cts, 'bcts': self.bcts, 
                        're_cts': self.re_cts, 're_bcts': self.re_bcts}


    def sorting(self, sigma=3):
        if self.snr_res is None: self.calsnr()

        self.sigma = sigma
        self.re_bkg_idx, self.re_sig_idx, self.re_bad_idx = [], [], []
        self.re_sig_int, self.re_bkg_int, self.re_bad_int = [], [], []
        self.bkg_idx, self.sig_idx, self.bad_idx = [], [], []
        self.sig_int, self.bkg_int, self.bad_int = [], [], []

        for i, snr_i in enumerate(self.re_snr):
            if snr_i > sigma:
                self.re_sig_idx.append(i)
                self.re_sig_int.append([self.edges[i], self.edges[i+1]])
            elif -5 < snr_i <= sigma:
                self.re_bkg_idx.append(i)
                self.re_bkg_int.append([self.edges[i], self.edges[i+1]])
            else:
                self.re_bad_idx.append(i)
                self.re_bad_int.append([self.edges[i], self.edges[i+1]])

        for i, snr_i in enumerate(self.snr):
            if snr_i > sigma:
                self.sig_idx.append(i)
                self.sig_int.append([self.lbins[i], self.rbins[i]])
            elif -5 < snr_i <= sigma:
                self.bkg_idx.append(i)
                self.bkg_int.append([self.lbins[i], self.rbins[i]])
            else:
                self.bad_idx.append(i)
                self.bad_int.append([self.lbins[i], self.rbins[i]])

        self.re_bkg_idx = np.array(self.re_bkg_idx, dtype=int)
        self.re_sig_idx = np.array(self.re_sig_idx, dtype=int)
        self.re_bad_idx = np.array(self.re_bad_idx, dtype=int)

        self.bkg_idx = np.array(self.bkg_idx, dtype=int)
        self.sig_idx = np.array(self.sig_idx, dtype=int)
        self.bad_idx = np.array(self.bad_idx, dtype=int)

        self.re_bad_index = []
        for i, (l, u) in enumerate(zip(self.lbins, self.rbins)):
            for low, upp in self.re_bad_int:
                if not (u <= low or l >= upp):
                    self.re_bad_index.append(i)
        self.re_bad_index = np.unique(self.re_bad_index).astype(int)

        self.re_sig_index = []
        for i, (l, u) in enumerate(zip(self.lbins, self.rbins)):
            for low, upp in self.re_sig_int:
                if not (u <= low or l >= upp):
                    self.re_sig_index.append(i)
        self.re_sig_index = np.unique(self.re_sig_index).astype(int)
        
        self.ignore = union(self.re_bad_int + self.re_sig_int)

        self.sort_res = {'sigma': self.sigma, 'ignore': self.ignore, 
                         're_bkg': (self.re_bkg_int, self.re_bkg_idx), 
                         're_sig': (self.re_sig_int, self.re_sig_idx, self.re_sig_index),
                         're_bad': (self.re_bad_int, self.re_bad_idx, self.re_bad_index)}


    def polyfit(self, deg=None, ignore=None):
        if ignore is None:
            if self.sort_res is None: self.sorting()
            ignore = self.sort_res['ignore']

        ignore_idx = []
        for i, (l, u) in enumerate(zip(self.lbins, self.rbins)):
            for low, upp in ignore:
                if not (u <= low or l >= upp):
                    ignore_idx.append(i)
        ignore_idx = np.unique(ignore_idx).astype(int)

        notice_time = np.delete(self.time, ignore_idx)
        notice_rate = np.delete(self.rate, ignore_idx)
        notice_exp = np.delete(self.exp, ignore_idx)
        # notice_lbins = np.delete(self.lbins, ignore_idx)
        # notice_rbins = np.delete(self.rbins, ignore_idx)
        # notice_binsize = notice_rbins - notice_lbins

        self.poly = Polynomial.set_method('2pass')
        self.poly.fit(notice_time, notice_rate, deg=deg, dx=notice_exp)
        self.bak, self.bak_se = self.poly.val(self.time)
        # self.bak[self.re_bad_index] = 0
        # self.bak_se[self.re_bad_index] = 0

        self.bcts = self.bak * self.exp
        self.bcts_se = self.bak_se * self.exp

        self.poly_res = {'deg': self.poly.deg,
                         'fit': self.poly.ls_res,
                         'bcts': self.bcts, 
                         'bcts_se': self.bcts_se, 
                         'bak': self.bak, 
                         'bak_se': self.bak_se}
        
    
    def loop(self, sigma=3, deg=None):
        self.calsnr()
        self.sorting(sigma)
        self.polyfit(deg)


    def save(self, savepath, suffix=''):
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.ini_res, savepath + '/ini_res%s.json'%suffix)
        json_dump(self.base_res, savepath + '/base_res%s.json'%suffix)
        json_dump(self.block_res, savepath + '/block_res%s.json'%suffix)
        json_dump(self.snr_res, savepath + '/snr_res%s.json'%suffix)
        json_dump(self.sort_res, savepath + '/sort_res%s.json'%suffix)
        json_dump(self.poly_res, savepath + '/poly_res%s.json'%suffix)

        rcParams['font.family'] = 'serif'
        rcParams['font.sans-serif'] = 'Georgia'
        rcParams['font.size'] = 12
        rcParams['pdf.fonttype'] = 42

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(self.time, self.rate, lw=1.0, c='b', label='Light curve')
        ax.plot(self.edges, np.append(self.re_cts/self.re_binsize, [(self.re_cts/self.re_binsize)[-1]]), 
                lw=1.0, c='c', drawstyle='steps-post', label='Bayesian block')
        ax.plot(self.time, self.bak, lw=1.0, c='r', label='Background')
        ax.fill_between(self.time, self.bak-self.bak_se, self.bak+self.bak_se, facecolor='red', alpha=0.5)
        ax.set_xlim([min(self.time), max(self.time)])
        ax.set_xlabel('Time')
        ax.set_ylabel('Rate')
        ax.minorticks_on()
        ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax.tick_params(which='major', width=1.0, length=5)
        ax.tick_params(which='minor', width=1.0, length=3)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.legend(frameon=False)
        fig.savefig(savepath + '/bs%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        p1, = ax.plot(self.time, self.rate - self.bak, lw=1.0, c='k', label='Net light curve')
        ax.set_xlim([min(self.time), max(self.time)])
        ax.set_xlabel('Time')
        ax.set_ylabel('Rate')
        ax.minorticks_on()
        ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax.tick_params(which='major', width=1.0, length=5)
        ax.tick_params(which='minor', width=1.0, length=3)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

        ax1 = ax.twinx()
        p2, = ax1.plot(self.time, self.snr, lw=1.0, c='b', label='SNR', drawstyle='steps-mid')
        p3, = ax1.plot(self.edges, np.append(self.re_snr, [self.re_snr[-1]]), lw=1.0, c='c', 
                       label='Bayesian block SNR', drawstyle='steps-post')
        p4 = ax1.axhline(self.sigma, lw=1.0, c='grey', ls='--', label='%.1f$\\sigma$' % self.sigma)
        ax1.set_xlim([min(self.time), max(self.time)])
        ax1.set_ylabel('SNR')

        plt.legend(handles=[p1, p2, p3, p4], frameon=False)
        fig.savefig(savepath + '/snr%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        colors = []
        for i in range(len(self.re_binsize)):
            if i in self.re_sig_idx: colors.append('m')
            if i in self.re_bkg_idx: colors.append('b')
            if i in self.re_bad_idx: colors.append('k')
        ax.bar((self.edges[:-1]+self.edges[1:])/2, self.re_cts/self.re_binsize, bottom=0, width=self.re_binsize, color=colors)
        ax.plot(self.time, self.bak, lw=1.0, c='r')
        ax.fill_between(self.time, self.bak-self.bak_se, self.bak+self.bak_se, facecolor='red', alpha=0.5)
        ax.set_xlim([min(self.time), max(self.time)])
        ax.set_xlabel('Time')
        ax.set_ylabel('Rate')
        ax.minorticks_on()
        ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax.tick_params(which='major', width=1.0, length=5)
        ax.tick_params(which='minor', width=1.0, length=3)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        fig.savefig(savepath + '/sort%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)
