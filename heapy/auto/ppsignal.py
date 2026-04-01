import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.stats import bayesian_blocks

from ..util.data import json_dump
from ..util.significance import ppsig



class ppSignal(object):
    
    """
    NOTE: only apply to possion data with possion background
    """

    def __init__(self, ts, bts, bins, backscale=1, exp=None):
        
        self.ts = np.array(ts).astype(float)
        self.bts = np.array(bts).astype(float)
        self.bins = np.array(bins).astype(float)
        self.backscale = backscale

        cts, _ = np.histogram(self.ts, bins=self.bins)
        self.cts = np.array(cts).astype(int)
        self.cts_err = np.sqrt(self.cts)
        
        bcts, _ = np.histogram(self.bts, bins=self.bins)
        self.bcts = np.array(bcts).astype(int)
        self.bcts_err = np.sqrt(self.bcts)

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
        self.rate_err = self.cts_err / self.exp
        
        self.bak = self.bcts * self.backscale / self.exp
        self.bak_err = self.bcts_err * self.backscale / self.exp

        self.net = self.rate - self.bak
        self.net_err = np.sqrt(self.rate_err**2 + self.bak_err**2)
        
        self.ncts = self.net * self.exp
        self.ncts_err = self.net_err * self.exp

        self.ini_res = {'cts': self.cts, 'bcts': self.bcts, 
                        'time': self.time, 'rate': self.rate, 
                        'bak': self.bak, 'exp': self.exp, 
                        'backscale': self.backscale, 'bins': self.bins}

        # bblock method
        self.block_res = None

        # calsnr method
        self.snr_res = None

        # fetsig method
        self.sort_res = None


    @classmethod
    def frombin(cls, cts, bcts, bins, backscale=1, exp=None):
        
        cts = np.array(cts)
        bcts = np.array(bcts)
        bins = np.array(bins)

        if bins.size != (cts.size + 1):
            raise TypeError("expected size(bins) = size(cts)+1")
        
        msg = 'rebuilt the time list, but may not accurate'
        warnings.warn(msg, UserWarning, stacklevel=2)
        
        ts = np.array([])
        bts = np.array([])
        
        for t1, t2, n in zip(bins[:-1], bins[1:], cts):
            ts = np.append(ts, np.random.random(size=int(n)) * (t2 - t1) + t1)
            
        for t1, t2, n in zip(bins[:-1], bins[1:], bcts):
            bts = np.append(bts, np.random.random(size=int(n)) * (t2 - t1) + t1)

        cls_ = cls(ts, bts, bins, backscale=backscale, exp=exp)
        return cls_
    
    
    def bblock(self, p0=0.05):
        
        if len(self.ts) <= 1e4:
            edges_ = bayesian_blocks(self.ts, fitness='events', p0=p0)
        else:
            pos = np.where(self.cts > 0)[0]
            edges_ = bayesian_blocks(self.time[pos], self.cts[pos], fitness='events', p0=p0)
            
        edges_[0] = max(edges_[0], self.time[0])
        edges_[-1] = min(edges_[-1], self.time[-1])

        edges = [edges_[0], edges_[-1]]
        for i in range(1, len(edges_) - 1, 1):
            flag1 = (edges_[i] - edges_[i-1]) > np.min(self.binsize) / 1.8
            flag2 = (edges_[i+1] - edges_[i]) > np.min(self.binsize) / 1.8
            if flag1 and flag2:
                edges.append(edges_[i])

        self.edges = np.unique(edges)
        self.nblock = len(self.edges) - 1
        self.re_binsize = self.edges[1:] - self.edges[:-1]

        self.block_res = {'edges': self.edges, 'nblock': self.nblock, 're_binsize': self.re_binsize}
    
    
    def calsnr(self):
        
        if self.block_res is None: self.bblock()
        
        self.re_cts, _ = np.histogram(self.ts, bins=self.edges)
        self.re_bcts, _ = np.histogram(self.bts, bins=self.edges)

        self.snr = np.zeros_like(self.binsize)
        for i in range(len(self.binsize)):
            cts_i = self.cts[i]
            bcts_i = self.bcts[i]
            
            if bcts_i < 0 or cts_i < 0:
                snr_i = -5      # bad events
            else:
                snr_i = ppsig(cts_i, bcts_i, self.backscale)

            self.snr[i] = snr_i
            
        self.re_snr = np.zeros_like(self.re_binsize)
        for i in range(len(self.re_binsize)):
            cts_i = self.re_cts[i]
            bcts_i = self.re_bcts[i]

            if bcts_i < 0 or cts_i < 0:
                snr_i = -5      # bad events
            else:
                snr_i = ppsig(cts_i, bcts_i, self.backscale)

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

        self.sort_res = {'sigma': self.sigma, 
                         're_bkg': (self.re_bkg_int, self.re_bkg_idx), 
                         're_sig': (self.re_sig_int, self.re_sig_idx, self.re_sig_index),
                         're_bad': (self.re_bad_int, self.re_bad_idx, self.re_bad_index)}


    def loop(self, p0=0.05, sigma=3):
        
        self.bblock(p0)
        self.calsnr()
        self.sorting(sigma)


    def save(self, savepath, suffix=''):
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.ini_res, savepath + '/ini_res%s.json'%suffix)
        json_dump(self.block_res, savepath + '/block_res%s.json'%suffix)
        json_dump(self.snr_res, savepath + '/snr_res%s.json'%suffix)
        json_dump(self.sort_res, savepath + '/sort_res%s.json'%suffix)

        rcParams['font.family'] = 'serif'
        rcParams['font.sans-serif'] = 'Georgia'
        rcParams['font.size'] = 12
        rcParams['pdf.fonttype'] = 42

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(self.time, self.rate, lw=1.0, c='b', label='Light curve')
        ax.plot(self.edges, np.append(self.re_cts/self.re_binsize, [(self.re_cts/self.re_binsize)[-1]]), 
                lw=1.0, c='c', drawstyle='steps-post', label='Bayesian block')
        ax.plot(self.time, self.bak, lw=1.0, c='r', label='Background')
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
        p1, = ax.plot(self.time, self.net, lw=1.0, c='k', label='Net light curve')
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
                       label='Baysian block SNR', drawstyle='steps-post')
        p4 = ax1.axhline(self.sigma, lw=1.0, c='grey', ls='--', label='%.1f$\\sigma$' % self.sigma)
        ax1.set_xlim([min(self.time), max(self.time)])
        ax1.set_ylabel('SNR')
        
        ratio = np.max(self.net) / np.max(self.re_snr)
        ax_ylim = [1.5 * np.min(self.net), 1.1 * np.max(self.net)]
        ax1_ylim = [lim / ratio for lim in ax_ylim]
        ax.set_ylim(ax_ylim)
        ax1.set_ylim(ax1_ylim)

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
        ax.plot(self.time, self.rate, lw=1.0, c='b', label='Light curve')
        ax.plot(self.time, self.bak, lw=1.0, c='r', label='Background')
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
