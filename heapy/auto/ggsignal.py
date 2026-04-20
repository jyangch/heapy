"""Signal classification pipeline for Gaussian-measured light curves.

Provides :class:`ggSignal`, which turns per-bin rates with Gaussian errors
into Bayesian blocks, computes SNR on raw and block scales, and sorts the
bins into signal / background / bad categories.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import bayesian_blocks

from .core_funcs import *
from .plot_funcs import *
from ..util.data import json_dump



class ggSignal(object):
    """Detect and sort signal bins in Gaussian-measured light curves.

    Designed for data whose per-bin uncertainties are already Gaussian; do
    not use this class for Poisson counts (see :class:`ppSignal` or
    :class:`PolyBase` instead).

    Attributes:
        cts: Per-bin counts (or signal).
        cts_err: 1-sigma uncertainty on :attr:`cts`.
        bins: Bin edges (length ``N + 1``).
        exp: Per-bin exposure times.
        time: Bin centers.
        rate: :attr:`cts` / :attr:`exp`.
        rate_err: :attr:`cts_err` / :attr:`exp`.
        ini_res, block_res, snr_res, sort_res: Stage result dicts populated
            by :meth:`bblock`, :meth:`calsnr`, and :meth:`sorting`.

    Example:
        >>> sig = ggSignal(cts, cts_err, bins)
        >>> sig.loop(p0=0.05, sigma=3)
        >>> sig.save('./out')

    Warning:
        Only apply to Gaussian-distributed data.
    """

    def __init__(self, cts, cts_err, bins, exp=None):
        """Store arrays and derive rates, rate errors, and bin geometry.

        Args:
            cts: Per-bin counts (or signal) matching bin count ``N``.
            cts_err: 1-sigma uncertainty on ``cts``; same length as ``cts``.
            bins: Bin edges (length ``N + 1``).
            exp: Per-bin exposure times; defaults to bin widths when
                ``None``.

        Raises:
            TypeError: If ``exp`` and ``bins`` have mismatched sizes or any
                exposure exceeds its bin width.
        """

        self.cts = np.array(cts)
        self.cts_err = np.array(cts_err)
        self.bins = np.array(bins).astype(float)

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

        self.ini_res = {'cts': self.cts, 'cts_err': self.cts_err,
                        'time': self.time, 'rate': self.rate, 
                        'exp': self.exp, 'bins': self.bins}

        self.block_res = None
        self.snr_res = None
        self.sort_res = None
    
    
    def bblock(self, p0=0.05):
        """Run ``astropy.stats.bayesian_blocks`` and filter undersized gaps.

        Populates :attr:`edges`, :attr:`nblock`, :attr:`re_binsize`, and
        :attr:`block_res`.

        Args:
            p0: False-alarm probability passed to ``bayesian_blocks``.
        """

        edges_ = bayesian_blocks(self.time, self.rate, self.rate_err, fitness='measures', p0=p0)
            
        edges_[0] = max(edges_[0], self.time[0])
        edges_[-1] = min(edges_[-1], self.time[-1])

        self.edges = filter_block_edges(edges_, np.min(self.binsize))
        self.nblock = len(self.edges) - 1
        self.re_binsize = self.edges[1:] - self.edges[:-1]

        self.block_res = {'edges': self.edges, 'nblock': self.nblock, 're_binsize': self.re_binsize}
    
    
    def calsnr(self):
        """Compute Gaussian SNR on both raw bins and Bayesian blocks.

        Runs :meth:`bblock` first when block results are missing.
        Populates :attr:`re_cts`, :attr:`re_cts_err`, :attr:`snr`,
        :attr:`re_snr`, and :attr:`snr_res`.
        """

        if self.block_res is None: self.bblock()

        self.re_cts, self.re_cts_err = [], []
        for t1, t2 in zip(self.edges[:-1], self.edges[1:]):
            self.re_cts.append(np.sum(self.cts[(self.time >= t1) & (self.time < t2)]))
            self.re_cts_err.append(np.sqrt(np.sum(self.cts_err[(self.time >= t1) & (self.time < t2)]**2)))
        self.re_cts = np.array(self.re_cts)
        self.re_cts_err = np.array(self.re_cts_err)

        self.snr = np.zeros_like(self.binsize)
        for i in range(len(self.binsize)):
            self.snr[i] = gauss_snr(self.cts[i], self.cts_err[i])

        self.re_snr = np.zeros_like(self.re_binsize)
        for i in range(len(self.re_binsize)):
            self.re_snr[i] = gauss_snr(self.re_cts[i], self.re_cts_err[i])

        self.snr_res = {'snr': self.snr, 're_snr': self.re_snr,
                        're_cts': self.re_cts, 're_cts_err': self.re_cts_err}
        
        
    def sorting(self, sigma=3):
        """Classify raw bins and blocks into signal/background/bad by SNR.

        Runs :meth:`calsnr` first when SNR results are missing. Populates
        ``*_idx``/``*_int`` attributes plus :attr:`sort_res`.

        Args:
            sigma: Detection threshold in units of SNR.
        """

        if self.snr_res is None: self.calsnr()

        self.sigma = sigma
        re = classify_bins(self.re_snr, self.edges[:-1], self.edges[1:], sigma)
        self.re_sig_idx, self.re_bkg_idx, self.re_bad_idx = re['sig_idx'], re['bkg_idx'], re['bad_idx']
        self.re_sig_int, self.re_bkg_int, self.re_bad_int = re['sig_int'], re['bkg_int'], re['bad_int']

        bn = classify_bins(self.snr, self.lbins, self.rbins, sigma)
        self.sig_idx, self.bkg_idx, self.bad_idx = bn['sig_idx'], bn['bkg_idx'], bn['bad_idx']
        self.sig_int, self.bkg_int, self.bad_int = bn['sig_int'], bn['bkg_int'], bn['bad_int']

        self.re_bad_index = indices_in_intervals(self.lbins, self.rbins, self.re_bad_int)
        self.re_sig_index = indices_in_intervals(self.lbins, self.rbins, self.re_sig_int)

        self.sort_res = {'sigma': self.sigma, 
                         're_bkg': (self.re_bkg_int, self.re_bkg_idx), 
                         're_sig': (self.re_sig_int, self.re_sig_idx, self.re_sig_index),
                         're_bad': (self.re_bad_int, self.re_bad_idx, self.re_bad_index)}


    def loop(self, p0=0.05, sigma=3):
        """Run :meth:`bblock`, :meth:`calsnr`, and :meth:`sorting` in order.

        Args:
            p0: False-alarm probability for Bayesian blocks.
            sigma: Detection threshold for bin classification.
        """

        self.bblock(p0)
        self.calsnr()
        self.sorting(sigma)


    def save(self, savepath, suffix=''):
        """Dump stage results as JSON and write three diagnostic PDFs.

        Creates ``savepath`` if missing. Writes ``ini_res``, ``block_res``,
        ``snr_res``, ``sort_res`` JSON files and ``bs``/``snr``/``sort``
        PDF figures, each suffixed with ``suffix``.

        Args:
            savepath: Destination directory; created if absent.
            suffix: Optional filename suffix inserted before each extension.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.ini_res, savepath + '/ini_res%s.json'%suffix)
        json_dump(self.block_res, savepath + '/block_res%s.json'%suffix)
        json_dump(self.snr_res, savepath + '/snr_res%s.json'%suffix)
        json_dump(self.sort_res, savepath + '/sort_res%s.json'%suffix)

        re_rate = self.re_cts / self.re_binsize
        with rc_context_for_save():
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            plot_lightcurve_bblock(ax, self.time, self.rate, self.edges, re_rate)
            fig.savefig(savepath + '/bs%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            plot_snr(ax, self.time, self.rate, self.edges, self.snr, self.re_snr, self.sigma,
                     left_label='light curve',
                     bblock_snr_label='Baysian block SNR',
                     ylim_factors=(1.5, 1.1))
            fig.savefig(savepath + '/snr%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            plot_sorted(ax, self.time, self.edges, self.re_cts, self.re_binsize,
                        self.re_sig_idx, self.re_bkg_idx, self.re_bad_idx,
                        rate=self.rate)
            fig.savefig(savepath + '/sort%s.pdf'%suffix, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)