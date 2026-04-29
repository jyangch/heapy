"""Compute Txx duration metrics (T90, T50, etc.) for gamma-ray burst light curves.

Provides three concrete classes — ``pgTxx``, ``ppTxx``, and ``ggTxx`` — that
inherit from ``pgSignal`` / ``ppSignal`` / ``ggSignal`` and add pulse detection
and cumulative-count-fraction duration analysis.  Monte Carlo simulation is
used by default to propagate uncertainties on the start/stop times.

Typical usage:
    from heapy.temp.txx import pgTxx
    txx = pgTxx(ts, bins)
    txx.calculate(xx=0.9)
    txx.save('/output/dir')
"""

import os
import warnings
import operator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import interp1d
from astropy.stats import sigma_clip, mad_std

from ..auto.signal import pgSignal, ppSignal, ggSignal
from ..util.data import generate_asymmetric_gaussian
from ..util.tools import json_dump



class pgTxx(pgSignal):
    """Compute Txx durations for a Poisson-source/Gaussian-background light curve.

    Extends ``pgSignal`` with pulse detection and cumulative-count-fraction
    duration (Txx) analysis.  Background subtraction uses the polynomial fit
    from the parent class; uncertainties are estimated via Monte Carlo
    simulation by default (1000 realisations).

    Attributes:
        pulse_res: Dictionary storing detected pulse start/stop edges, or
            ``None`` before ``find_pulse`` is called.
        txx_res: Dictionary storing Txx results, or ``None`` before
            ``calculate`` is called.
    """

    def __init__(self, ts, bins, exp=None, ignore=None):
        """Initialize pgTxx with time-tagged event data and binning.

        Args:
            ts: Array of event arrival times.
            bins: Bin edges or bin width used to build the light curve.
            exp: Exposure correction array, or ``None`` for uniform exposure.
            ignore: Time intervals to exclude from the background fit, or
                ``None`` to use all data.
        """

        super().__init__(ts, bins, exp=exp, ignore=ignore)

        self.pulse_res = None
        self.txx_res = None


    def find_pulse(self, p0=0.05, sigma=3, deg=None, mp=True):
        """Detect significant pulse intervals in the net light curve.

        Runs the background polynomial fit if not already done, computes
        net counts and rates on the rebinned grid, then identifies
        contiguous blocks whose signal-to-noise ratio exceeds ``sigma``.
        Results are stored in ``self.pstart``, ``self.pstop``, and
        ``self.pulse_res``.

        Args:
            p0: Bayesian-blocks prior probability for a new change point;
                passed to the parent ``loop`` method.
            sigma: Minimum SNR threshold for a block to be classified as
                a pulse.
            deg: Polynomial degree for the background fit, or ``None`` to
                select automatically.
            mp: When ``True``, keep multiple separate pulse intervals.
                When ``False``, merge all intervals into one and emit a
                warning if more than one interval is found.
        """

        if self.poly_res is None:
            self.loop(p0=p0, sigma=sigma, deg=deg)
            self.loop(p0=p0, sigma=sigma, deg=deg)

        self.ncts = self.cts - self.bcts
        self.net = self.rate - self.bak

        self.re_ncts = self.re_cts - self.re_bcts
        self.re_net = self.re_ncts / self.re_binsize

        start, stop = (0, 0)
        flat, up, down = (1, 2, 3)
        label = []
        nblock = len(self.re_net)
        for i, val in enumerate(self.re_net):
            if i == 0:
                label.append(start)
            elif i == (nblock - 1):
                label.append(stop)
            else:
                if val > self.re_net[i - 1]:
                    label.append(up)
                elif val < self.re_net[i - 1]:
                    label.append(down)
                else:
                    label.append(flat)

        flag = False
        pstart, pstop = [], []
        for i in range(nblock):
            if (self.re_snr[i] > sigma) and (flag is False):
                assert label[i] != down
                pstart.append(self.edges[i])
                flag = True
            elif (self.re_snr[i] <= sigma) and (flag is True):
                assert label[i] != up
                pstop.append(self.edges[i])
                flag = False

        if flag:
            if len(pstart) > 0:
                pstart.pop()
        if len(pstart) > 0:
            if pstart[0] == self.edges[0]:
                pstart = pstart[1:]
                pstop = pstop[1:]

        if (not mp) and (len(pstart) > 0):
            if len(pstart) > 1:
                msg = 'multi-pulse will be combined into one'
                warnings.warn(msg, UserWarning, stacklevel=2)
            pstart = [pstart[0]]
            pstop = [pstop[-1]]

        self.pstart = np.array(pstart)
        self.pstop = np.array(pstop)

        self.pulse_res = {'pstart': self.pstart, 'pstop': self.pstop}


    def mc_simulation(self, nmc):
        """Generate Monte Carlo realisations of the net count light curve.

        Draws Poisson samples for the source counts and Gaussian samples for
        the background counts, then stacks the net realisation matrix as
        ``self.mc_ncts`` with the observed data in row 0.

        Args:
            nmc: Number of Monte Carlo realisations to generate.
        """

        self.nmc = int(nmc)
        self.nsample = len(self.time)

        src_sample = np.random.poisson(lam=self.cts, size=(self.nmc, self.nsample))
        bkg_sample = np.random.normal(loc=self.bcts, scale=self.bcts_err, size=(self.nmc, self.nsample))

        self.mc_ncts = np.vstack([self.ncts, src_sample - bkg_sample])


    def calculate(self, xx=0.9, pstart=None, pstop=None, lbkg=None, rbkg=None, simple_err=False):
        """Compute Txx duration and its uncertainties for each detected pulse.

        Calls ``find_pulse`` if not already done.  Optionally overrides the
        detected pulse boundaries.  Uncertainties are estimated either via a
        simple propagation formula (``simple_err=True``) or via 1000 Monte
        Carlo realisations (default).  Results are printed to stdout and
        stored in ``self.txx_res``.

        Args:
            xx: Cumulative count fraction defining the duration, e.g. ``0.9``
                for T90 or ``0.5`` for T50.
            pstart: Override pulse start time(s).  A scalar is promoted to a
                length-1 array; a list is sorted in ascending order.
            pstop: Override pulse stop time(s).  Same promotion rules as
                ``pstart``.
            lbkg: Length of the background window to the left of the first
                pulse, in the same time units as the light curve.  ``None``
                extends to the beginning of the data.
            rbkg: Length of the background window to the right of the last
                pulse.  ``None`` extends to the end of the data.
            simple_err: When ``True``, use analytic error propagation instead
                of Monte Carlo simulation.

        Returns:
            ``False`` if no pulse is detected; ``None`` on success (results
            are stored as instance attributes).
        """

        if self.pulse_res is None: self.find_pulse()

        self.xx = xx

        if pstart is not None:
            if type(pstart) is not list:
                self.pstart = np.array([pstart])
            else:
                self.pstart = np.sort(pstart)

        if pstop is not None:
            if type(pstop) is not list:
                self.pstop = np.array([pstop])
            else:
                self.pstop = np.sort(pstop)

        if len(self.pstart) == 0:
            msg = 'there is no pulse'
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False

        if lbkg is None:
            lbkg = np.inf

        if rbkg is None:
            rbkg = np.inf

        tmin_ = self.pstart[0] - lbkg
        tmin = max(tmin_, self.time[0])
        tmax_ = self.pstop[-1] + rbkg
        tmax = min(tmax_, self.time[-1])
        self.tindex = np.where((self.time >= tmin) & (self.time <= tmax))[0]

        self.ccts = np.cumsum(self.ncts)

        if simple_err:
            self.txx, self.txx1, self.txx2, \
                self.txx_err, self.txx1_err, self.txx2_err, \
                    self.csf, self.csf1, self.csf2, \
                        self.csf_err, self.csf1_err, self.csf2_err \
                            = accumcts(self.time[self.tindex], self.ccts[self.tindex], self.pstart, self.pstop, self.xx, simple_err=True)

            self.txx_res = {'xx': self.xx, 'txx':self.txx, 'txx1': self.txx1, 'txx2': self.txx2,
                            'txx_err': self.txx_err, 'txx1_err': self.txx1_err, 'txx2_err': self.txx2_err,
                            'csf': self.csf, 'csf1': self.csf1, 'csf2': self.csf2,
                            'csf_err': self.csf_err, 'csf1_err': self.csf1_err, 'csf2_err': self.csf2_err}

        else:
            self.mc_simulation(1000)

            mc_csf, mc_csf1, mc_csf2 = [], [], []
            mc_txx, mc_txx1, mc_txx2 = [], [], []

            for ncts in self.mc_ncts:

                ccts = np.cumsum(ncts)

                txx, txx1, txx2, csf, csf1, csf2 \
                    = accumcts(self.time[self.tindex], ccts[self.tindex], self.pstart, self.pstop, self.xx, simple_err=False)

                mc_csf.append(csf)
                mc_csf1.append(csf1)
                mc_csf2.append(csf2)
                mc_txx.append(txx)
                mc_txx1.append(txx1)
                mc_txx2.append(txx2)

            self.csf = mc_csf[0]
            self.csf1 = mc_csf1[0]
            self.csf2 = mc_csf2[0]

            self.txx = mc_txx[0]
            self.txx1 = mc_txx1[0]
            self.txx2 = mc_txx2[0]

            self.txx_err = []
            self.txx1_err = []
            self.txx2_err = []

            mc_txx = np.array(mc_txx)
            mc_txx1 = np.array(mc_txx1)
            mc_txx2 = np.array(mc_txx2)

            for pi in range(len(self.pstart)):
                mask = sigma_clip(mc_txx[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
                not_mask = list(map(operator.not_, mask))
                mc_txx_filter = mc_txx[1:, pi][not_mask]

                txx_lo, txx_hi = np.percentile(mc_txx_filter, [16, 84])
                txx_err = np.diff([txx_lo, mc_txx[0, pi], txx_hi])
                self.txx_err.append([txx_err[0], txx_err[1]])

                mask = sigma_clip(mc_txx1[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
                not_mask = list(map(operator.not_, mask))
                mc_txx1_filter = mc_txx1[1:, pi][not_mask]

                txx1_lo, txx1_hi = np.percentile(mc_txx1_filter, [16, 84])
                txx1_err = np.diff([txx1_lo, mc_txx1[0, pi], txx1_hi])
                self.txx1_err.append([txx1_err[0], txx1_err[1]])

                mask = sigma_clip(mc_txx2[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
                not_mask = list(map(operator.not_, mask))
                mc_txx2_filter = mc_txx2[1:, pi][not_mask]

                txx2_lo, txx2_hi = np.percentile(mc_txx2_filter, [16, 84])
                txx2_err = np.diff([txx2_lo, mc_txx2[0, pi], txx2_hi])
                self.txx2_err.append([txx2_err[0], txx2_err[1]])

            self.txx_res = {'xx': self.xx, 'txx':self.txx, 'txx1': self.txx1, 'txx2': self.txx2,
                            'txx_err': self.txx_err, 'txx1_err': self.txx1_err, 'txx2_err': self.txx2_err,
                            'csf': self.csf, 'csf1': self.csf1, 'csf2': self.csf2}

        print('\n+------------------------------------------------+')
        print(' %-5s%-10s%-8s%-8s%-8s%-8s' % ('id#', 'Txx', 'Txx-', 'Txx+', 'Txx1', 'Txx2'))
        print('+-----------------------------------------------+')
        _ = list(map(print, [' %-5d%-10.3f%-8.3f%-8.3f%-8.3f%-8.3f' % (i+1, t, t_err[0], t_err[1], t1, t2)
                             for i, (t, t_err, t1, t2) in enumerate(zip(self.txx_res['txx'],
                                                                        self.txx_res['txx_err'],
                                                                        self.txx_res['txx1'],
                                                                        self.txx_res['txx2']))]))
        print('+------------------------------------------------+')


    def save(self, savepath):
        """Save Txx results and diagnostic plots to disk.

        Serialises ``pulse_res`` and ``txx_res`` as JSON files and writes a
        two-panel PDF showing the light curve with Txx boundaries (upper
        panel) and the cumulative count curve with CSF levels (lower panel).

        Args:
            savepath: Directory path where output files are written; created
                if it does not exist.

        Returns:
            ``False`` if no pulse has been detected; ``None`` on success.
        """

        if len(self.pstart) == 0:
            msg = 'there is no pulse'
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.pulse_res, savepath + '/pulse_res.json')
        json_dump(self.txx_res, savepath + '/txx_res.json')

        rcParams['font.family'] = "serif"
        rcParams['font.serif'] = ["STIX Two Text"]
        rcParams['mathtext.fontset'] = "stix"
        rcParams['font.size'] = 12
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42

        fig = plt.figure(figsize=(7, 6))
        gs = fig.add_gridspec(2, 1, wspace=0, hspace=0)
        ax1 = fig.add_subplot(gs[:1, 0])
        ax2 = fig.add_subplot(gs[1:, 0], sharex=ax1)

        ax1.plot(self.time, self.rate, color='k', lw=1.0, label='Light Curve')
        ax1.plot(self.time, self.bak, color='r', lw=1.0, label='Background')
        for i in range(len(self.txx1)):
            ax1.axvline(self.txx1[i], color='g', lw=1.0, ls='--')
            ax1.axvline(self.txx2[i], color='g', lw=1.0, ls='--')
        ax1.set_ylabel('Rate')
        ax1.set_xlim([self.time[0], self.time[-1]])
        ax1.minorticks_on()
        ax1.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax1.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax1.tick_params(which='major', width=1.0, length=5)
        ax1.tick_params(which='minor', width=1.0, length=3)
        ax1.xaxis.set_ticks_position('both')
        ax1.yaxis.set_ticks_position('both')
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.legend(frameon=False)

        ax2.plot(self.time, self.ccts, color='gray', lw=1.0)
        ax2.plot(self.time[self.tindex], self.ccts[self.tindex], color='k', lw=1.0)
        for i in range(len(self.txx1)):
            ax2.axvline(self.txx1[i], color='g', lw=1.0, ls='--')
            ax2.axvline(self.txx2[i], color='g', lw=1.0, ls='--')
        for csfi in self.csf:
            ax2.axhline(csfi, color='orange', lw=1.0)
        for i in range(len(self.csf1)):
            ax2.axhline(self.csf1[i], color='orange', lw=1.0, ls='--')
            ax2.axhline(self.csf2[i], color='orange', lw=1.0, ls='--')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Accumulated counts')
        ax2.minorticks_on()
        ax2.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax2.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax2.tick_params(which='major', width=1.0, length=5)
        ax2.tick_params(which='minor', width=1.0, length=3)
        ax2.xaxis.set_ticks_position('both')
        ax2.yaxis.set_ticks_position('both')
        fig.savefig(savepath + '/txx.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)



class ppTxx(ppSignal):
    """Compute Txx durations for a Poisson-source/Poisson-background light curve.

    Extends ``ppSignal`` with pulse detection and Txx duration analysis.
    Background subtraction uses a scaled Poisson background sample;
    uncertainties are always estimated via 1000 Monte Carlo realisations.

    Attributes:
        pulse_res: Dictionary storing detected pulse start/stop edges, or
            ``None`` before ``find_pulse`` is called.
        txx_res: Dictionary storing Txx results, or ``None`` before
            ``calculate`` is called.
    """

    def __init__(self, ts, bts, bins, backscale=1, exp=None):
        """Initialize ppTxx with source and background event arrays.

        Args:
            ts: Array of source event arrival times.
            bts: Array of background event arrival times.
            bins: Bin edges or bin width used to build the light curve.
            backscale: Ratio of the source to background region size used
                for background scaling.
            exp: Exposure correction array, or ``None`` for uniform exposure.
        """

        super().__init__(ts, bts, bins, backscale=backscale, exp=exp)

        self.pulse_res = None
        self.txx_res = None


    def find_pulse(self, p0=0.05, sigma=3, mp=True):
        """Detect significant pulse intervals in the net light curve.

        Runs the Bayesian-blocks segmentation if not already done, computes
        background-subtracted net counts on the rebinned grid, then identifies
        contiguous blocks whose SNR exceeds ``sigma``.  Results are stored in
        ``self.pstart``, ``self.pstop``, and ``self.pulse_res``.

        Args:
            p0: Bayesian-blocks prior probability for a new change point.
            sigma: Minimum SNR threshold for a block to be classified as
                a pulse.
            mp: When ``True``, keep multiple separate pulse intervals.
                When ``False``, merge all intervals into one and emit a
                warning if more than one interval is found.
        """

        if self.sort_res is None:
            self.loop(p0=p0, sigma=sigma)

        self.ncts = self.cts - self.bcts * self.backscale
        self.net = self.rate - self.bak

        self.re_ncts = self.re_cts - self.re_bcts * self.backscale
        self.re_net = self.re_ncts / self.re_binsize

        start, stop = (0, 0)
        flat, up, down = (1, 2, 3)
        label = []
        nblock = len(self.re_net)
        for i, val in enumerate(self.re_net):
            if i == 0:
                label.append(start)
            elif i == (nblock - 1):
                label.append(stop)
            else:
                if val > self.re_net[i - 1]:
                    label.append(up)
                elif val < self.re_net[i - 1]:
                    label.append(down)
                else:
                    label.append(flat)

        flag = False
        pstart, pstop = [], []
        for i in range(nblock):
            if (self.re_snr[i] > sigma) and (flag is False):
                assert label[i] != down
                pstart.append(self.edges[i])
                flag = True
            elif (self.re_snr[i] <= sigma) and (flag is True):
                assert label[i] != up
                pstop.append(self.edges[i])
                flag = False

        if flag:
            if len(pstart) > 0:
                pstart.pop()
        if len(pstart) > 0:
            if pstart[0] == self.edges[0]:
                pstart = pstart[1:]
                pstop = pstop[1:]

        if (not mp) and (len(pstart) > 0):
            if len(pstart) > 1:
                msg = 'multi-pulse will be combined into one'
                warnings.warn(msg, UserWarning, stacklevel=2)
            pstart = [pstart[0]]
            pstop = [pstop[-1]]

        self.pstart = np.array(pstart)
        self.pstop = np.array(pstop)

        self.pulse_res = {'pstart': self.pstart, 'pstop': self.pstop}


    def mc_simulation(self, nmc):
        """Generate Monte Carlo realisations of the net count light curve.

        Draws independent Poisson samples for both source and background
        counts and stacks the net realisation matrix as ``self.mc_ncts``
        with the observed data in row 0.

        Args:
            nmc: Number of Monte Carlo realisations to generate.
        """

        self.nmc = int(nmc)
        self.nsample = len(self.time)

        src_sample = np.random.poisson(lam=self.cts, size=(self.nmc, self.nsample))
        bkg_sample = np.random.poisson(lam=self.bcts, size=(self.nmc, self.nsample))

        self.mc_ncts = np.vstack([self.ncts, src_sample - bkg_sample * self.backscale])


    def calculate(self, xx=0.9, pstart=None, pstop=None, lbkg=None, rbkg=None):
        """Compute Txx duration and its uncertainties for each detected pulse.

        Calls ``find_pulse`` if not already done.  Optionally overrides the
        detected pulse boundaries.  Uncertainties are always estimated via
        1000 Monte Carlo realisations.  Results are printed to stdout and
        stored in ``self.txx_res``.

        Args:
            xx: Cumulative count fraction defining the duration, e.g. ``0.9``
                for T90 or ``0.5`` for T50.
            pstart: Override pulse start time(s).  A scalar is promoted to a
                length-1 array; a list is sorted in ascending order.
            pstop: Override pulse stop time(s).  Same promotion rules as
                ``pstart``.
            lbkg: Length of the background window to the left of the first
                pulse.  ``None`` extends to the beginning of the data.
            rbkg: Length of the background window to the right of the last
                pulse.  ``None`` extends to the end of the data.

        Returns:
            ``False`` if no pulse is detected; ``None`` on success (results
            are stored as instance attributes).
        """

        if self.pulse_res is None: self.find_pulse()

        self.xx = xx

        if pstart is not None:
            if type(pstart) is not list:
                self.pstart = np.array([pstart])
            else:
                self.pstart = np.sort(pstart)

        if pstop is not None:
            if type(pstop) is not list:
                self.pstop = np.array([pstop])
            else:
                self.pstop = np.sort(pstop)

        if len(self.pstart) == 0:
            msg = 'there is no pulse'
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False

        if lbkg is None:
            lbkg = np.inf

        if rbkg is None:
            rbkg = np.inf

        tmin_ = self.pstart[0] - lbkg
        tmin = max(tmin_, self.time[0])
        tmax_ = self.pstop[-1] + rbkg
        tmax = min(tmax_, self.time[-1])
        self.tindex = np.where((self.time >= tmin) & (self.time <= tmax))[0]

        self.ccts = np.cumsum(self.ncts)

        self.mc_simulation(1000)

        mc_csf, mc_csf1, mc_csf2 = [], [], []
        mc_txx, mc_txx1, mc_txx2 = [], [], []

        for ncts in self.mc_ncts:

            ccts = np.cumsum(ncts)

            txx, txx1, txx2, csf, csf1, csf2 \
                = accumcts(self.time[self.tindex], ccts[self.tindex], self.pstart, self.pstop, self.xx, simple_err=False)

            mc_csf.append(csf)
            mc_csf1.append(csf1)
            mc_csf2.append(csf2)
            mc_txx.append(txx)
            mc_txx1.append(txx1)
            mc_txx2.append(txx2)

        self.csf = mc_csf[0]
        self.csf1 = mc_csf1[0]
        self.csf2 = mc_csf2[0]

        self.txx = mc_txx[0]
        self.txx1 = mc_txx1[0]
        self.txx2 = mc_txx2[0]

        self.txx_err = []
        self.txx1_err = []
        self.txx2_err = []

        mc_txx = np.array(mc_txx)
        mc_txx1 = np.array(mc_txx1)
        mc_txx2 = np.array(mc_txx2)

        for pi in range(len(self.pstart)):
            mask = sigma_clip(mc_txx[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
            not_mask = list(map(operator.not_, mask))
            mc_txx_filter = mc_txx[1:, pi][not_mask]

            txx_lo, txx_hi = np.percentile(mc_txx_filter, [16, 84])
            txx_err = np.diff([txx_lo, mc_txx[0, pi], txx_hi])
            self.txx_err.append([txx_err[0], txx_err[1]])

            mask = sigma_clip(mc_txx1[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
            not_mask = list(map(operator.not_, mask))
            mc_txx1_filter = mc_txx1[1:, pi][not_mask]

            txx1_lo, txx1_hi = np.percentile(mc_txx1_filter, [16, 84])
            txx1_err = np.diff([txx1_lo, mc_txx1[0, pi], txx1_hi])
            self.txx1_err.append([txx1_err[0], txx1_err[1]])

            mask = sigma_clip(mc_txx2[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
            not_mask = list(map(operator.not_, mask))
            mc_txx2_filter = mc_txx2[1:, pi][not_mask]

            txx2_lo, txx2_hi = np.percentile(mc_txx2_filter, [16, 84])
            txx2_err = np.diff([txx2_lo, mc_txx2[0, pi], txx2_hi])
            self.txx2_err.append([txx2_err[0], txx2_err[1]])

        self.txx_res = {'xx': self.xx, 'txx':self.txx, 'txx1': self.txx1, 'txx2': self.txx2,
                        'txx_err': self.txx_err, 'txx1_err': self.txx1_err, 'txx2_err': self.txx2_err,
                        'csf': self.csf, 'csf1': self.csf1, 'csf2': self.csf2}

        print('\n+------------------------------------------------+')
        print(' %-5s%-10s%-8s%-8s%-8s%-8s' % ('id#', 'Txx', 'Txx-', 'Txx+', 'Txx1', 'Txx2'))
        print('+-----------------------------------------------+')
        _ = list(map(print, [' %-5d%-10.3f%-8.3f%-8.3f%-8.3f%-8.3f' % (i+1, t, t_err[0], t_err[1], t1, t2)
                                for i, (t, t_err, t1, t2) in enumerate(zip(self.txx_res['txx'],
                                                                        self.txx_res['txx_err'],
                                                                        self.txx_res['txx1'],
                                                                        self.txx_res['txx2']))]))
        print('+------------------------------------------------+')


    def save(self, savepath):
        """Save Txx results and diagnostic plots to disk.

        Serialises ``pulse_res`` and ``txx_res`` as JSON files and writes a
        two-panel PDF showing the light curve with Txx boundaries (upper
        panel) and the cumulative count curve with CSF levels (lower panel).

        Args:
            savepath: Directory path where output files are written; created
                if it does not exist.

        Returns:
            ``False`` if no pulse has been detected; ``None`` on success.
        """

        if len(self.pstart) == 0:
            msg = 'there is no pulse'
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.pulse_res, savepath + '/pulse_res.json')
        json_dump(self.txx_res, savepath + '/txx_res.json')

        rcParams['font.family'] = "serif"
        rcParams['font.serif'] = ["STIX Two Text"]
        rcParams['mathtext.fontset'] = "stix"
        rcParams['font.size'] = 12
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42

        fig = plt.figure(figsize=(7, 6))
        gs = fig.add_gridspec(2, 1, wspace=0, hspace=0)
        ax1 = fig.add_subplot(gs[:1, 0])
        ax2 = fig.add_subplot(gs[1:, 0], sharex=ax1)

        ax1.plot(self.time, self.rate, color='k', lw=1.0, label='Light Curve')
        ax1.plot(self.time, self.bak, color='r', lw=1.0, label='Background')
        for i in range(len(self.txx1)):
            ax1.axvline(self.txx1[i], color='g', lw=1.0, ls='--')
            ax1.axvline(self.txx2[i], color='g', lw=1.0, ls='--')
        ax1.set_ylabel('Rate')
        ax1.set_xlim([self.time[0], self.time[-1]])
        ax1.minorticks_on()
        ax1.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax1.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax1.tick_params(which='major', width=1.0, length=5)
        ax1.tick_params(which='minor', width=1.0, length=3)
        ax1.xaxis.set_ticks_position('both')
        ax1.yaxis.set_ticks_position('both')
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.legend(frameon=False)

        ax2.plot(self.time, self.ccts, color='gray', lw=1.0)
        ax2.plot(self.time[self.tindex], self.ccts[self.tindex], color='k', lw=1.0)
        for i in range(len(self.txx1)):
            ax2.axvline(self.txx1[i], color='g', lw=1.0, ls='--')
            ax2.axvline(self.txx2[i], color='g', lw=1.0, ls='--')
        for csfi in self.csf:
            ax2.axhline(csfi, color='orange', lw=1.0)
        for i in range(len(self.csf1)):
            ax2.axhline(self.csf1[i], color='orange', lw=1.0, ls='--')
            ax2.axhline(self.csf2[i], color='orange', lw=1.0, ls='--')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Accumulated counts')
        ax2.minorticks_on()
        ax2.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax2.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax2.tick_params(which='major', width=1.0, length=5)
        ax2.tick_params(which='minor', width=1.0, length=3)
        ax2.xaxis.set_ticks_position('both')
        ax2.yaxis.set_ticks_position('both')
        fig.savefig(savepath + '/txx.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)



class ggTxx(ggSignal):
    """Compute Txx durations for a Gaussian-source/Gaussian-background light curve.

    Extends ``ggSignal`` with pulse detection and Txx duration analysis.
    The input counts and their errors are treated as Gaussian variates;
    uncertainties are estimated via 1000 Monte Carlo realisations.

    Attributes:
        pulse_res: Dictionary storing detected pulse start/stop edges, or
            ``None`` before ``find_pulse`` is called.
        txx_res: Dictionary storing Txx results, or ``None`` before
            ``calculate`` is called.
    """

    def __init__(self, ncts, ncts_err, bins, exp=None):
        """Initialize ggTxx with pre-background-subtracted count data.

        Args:
            ncts: Array of net (background-subtracted) counts per bin.
            ncts_err: Array of uncertainties on ``ncts``.
            bins: Bin edges or bin width used to build the light curve.
            exp: Exposure correction array, or ``None`` for uniform exposure.
        """

        super().__init__(ncts, ncts_err, bins, exp=exp)

        self.pulse_res = None
        self.txx_res = None


    def find_pulse(self, p0=0.05, sigma=3, mp=True):
        """Detect significant pulse intervals in the net light curve.

        Runs the Bayesian-blocks segmentation if not already done, sets up
        net count arrays on the rebinned grid, then identifies contiguous
        blocks whose SNR exceeds ``sigma``.  Results are stored in
        ``self.pstart``, ``self.pstop``, and ``self.pulse_res``.

        Args:
            p0: Bayesian-blocks prior probability for a new change point.
            sigma: Minimum SNR threshold for a block to be classified as
                a pulse.
            mp: When ``True``, keep multiple separate pulse intervals.
                When ``False``, merge all intervals into one and emit a
                warning if more than one interval is found.
        """

        if self.sort_res is None:
            self.loop(p0=p0, sigma=sigma)

        self.ncts = self.cts
        self.ncts_err = self.cts_err
        self.net = self.cts / self.exp
        self.net_err = self.cts_err / self.exp

        self.re_ncts = self.re_cts
        self.re_ncts_err = self.re_cts_err
        self.re_net = self.re_ncts / self.re_binsize
        self.re_net_err = self.re_ncts_err / self.re_binsize

        start, stop = (0, 0)
        flat, up, down = (1, 2, 3)
        label = []
        nblock = len(self.re_net)
        for i, val in enumerate(self.re_net):
            if i == 0:
                label.append(start)
            elif i == (nblock - 1):
                label.append(stop)
            else:
                if val > self.re_net[i - 1]:
                    label.append(up)
                elif val < self.re_net[i - 1]:
                    label.append(down)
                else:
                    label.append(flat)

        flag = False
        pstart, pstop = [], []
        for i in range(nblock):
            if (self.re_snr[i] > sigma) and (flag is False):
                assert label[i] != down
                pstart.append(self.edges[i])
                flag = True
            elif (self.re_snr[i] <= sigma) and (flag is True):
                assert label[i] != up
                pstop.append(self.edges[i])
                flag = False

        if flag:
            if len(pstart) > 0:
                pstart.pop()
        if len(pstart) > 0:
            if pstart[0] == self.edges[0]:
                pstart = pstart[1:]
                pstop = pstop[1:]

        if (not mp) and (len(pstart) > 0):
            if len(pstart) > 1:
                msg = 'multi-pulse will be combined into one'
                warnings.warn(msg, UserWarning, stacklevel=2)
            pstart = [pstart[0]]
            pstop = [pstop[-1]]

        self.pstart = np.array(pstart)
        self.pstop = np.array(pstop)

        self.pulse_res = {'pstart': self.pstart, 'pstop': self.pstop}


    def mc_simulation(self, nmc):
        """Generate Monte Carlo realisations of the net count light curve.

        Draws Gaussian samples using ``self.ncts`` and ``self.ncts_err`` and
        stacks the realisation matrix as ``self.mc_ncts`` with the observed
        data in row 0.

        Args:
            nmc: Number of Monte Carlo realisations to generate.
        """

        self.nmc = int(nmc)
        self.nsample = len(self.time)

        sample = np.random.normal(loc=self.ncts, scale=self.ncts_err, size=(self.nmc, self.nsample))

        self.mc_ncts = np.vstack([self.ncts, sample])


    def calculate(self, xx=0.9, pstart=None, pstop=None, lbkg=None, rbkg=None):
        """Compute Txx duration and its uncertainties for each detected pulse.

        Calls ``find_pulse`` if not already done.  Optionally overrides the
        detected pulse boundaries.  Uncertainties are always estimated via
        1000 Monte Carlo realisations.  Results are printed to stdout and
        stored in ``self.txx_res``.

        Args:
            xx: Cumulative count fraction defining the duration, e.g. ``0.9``
                for T90 or ``0.5`` for T50.
            pstart: Override pulse start time(s).  A scalar is promoted to a
                length-1 array; a list is sorted in ascending order.
            pstop: Override pulse stop time(s).  Same promotion rules as
                ``pstart``.
            lbkg: Length of the background window to the left of the first
                pulse.  ``None`` extends to the beginning of the data.
            rbkg: Length of the background window to the right of the last
                pulse.  ``None`` extends to the end of the data.

        Returns:
            ``False`` if no pulse is detected; ``None`` on success (results
            are stored as instance attributes).
        """

        if self.pulse_res is None: self.find_pulse()

        self.xx = xx

        if pstart is not None:
            if type(pstart) is not list:
                self.pstart = np.array([pstart])
            else:
                self.pstart = np.sort(pstart)

        if pstop is not None:
            if type(pstop) is not list:
                self.pstop = np.array([pstop])
            else:
                self.pstop = np.sort(pstop)

        if len(self.pstart) == 0:
            msg = 'there is no pulse'
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False

        if lbkg is None:
            lbkg = np.inf

        if rbkg is None:
            rbkg = np.inf

        tmin_ = self.pstart[0] - lbkg
        tmin = max(tmin_, self.time[0])
        tmax_ = self.pstop[-1] + rbkg
        tmax = min(tmax_, self.time[-1])
        self.tindex = np.where((self.time >= tmin) & (self.time <= tmax))[0]

        self.ccts = np.cumsum(self.ncts)

        self.mc_simulation(1000)

        mc_csf, mc_csf1, mc_csf2 = [], [], []
        mc_txx, mc_txx1, mc_txx2 = [], [], []

        for ncts in self.mc_ncts:

            ccts = np.cumsum(ncts)

            txx, txx1, txx2, csf, csf1, csf2 \
                = accumcts(self.time[self.tindex], ccts[self.tindex], self.pstart, self.pstop, self.xx, simple_err=False)

            mc_csf.append(csf)
            mc_csf1.append(csf1)
            mc_csf2.append(csf2)
            mc_txx.append(txx)
            mc_txx1.append(txx1)
            mc_txx2.append(txx2)

        self.csf = mc_csf[0]
        self.csf1 = mc_csf1[0]
        self.csf2 = mc_csf2[0]

        self.txx = mc_txx[0]
        self.txx1 = mc_txx1[0]
        self.txx2 = mc_txx2[0]

        self.txx_err = []
        self.txx1_err = []
        self.txx2_err = []

        mc_txx = np.array(mc_txx)
        mc_txx1 = np.array(mc_txx1)
        mc_txx2 = np.array(mc_txx2)

        for pi in range(len(self.pstart)):
            mask = sigma_clip(mc_txx[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
            not_mask = list(map(operator.not_, mask))
            mc_txx_filter = mc_txx[1:, pi][not_mask]

            txx_lo, txx_hi = np.percentile(mc_txx_filter, [16, 84])
            txx_err = np.diff([txx_lo, mc_txx[0, pi], txx_hi])
            self.txx_err.append([txx_err[0], txx_err[1]])

            mask = sigma_clip(mc_txx1[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
            not_mask = list(map(operator.not_, mask))
            mc_txx1_filter = mc_txx1[1:, pi][not_mask]

            txx1_lo, txx1_hi = np.percentile(mc_txx1_filter, [16, 84])
            txx1_err = np.diff([txx1_lo, mc_txx1[0, pi], txx1_hi])
            self.txx1_err.append([txx1_err[0], txx1_err[1]])

            mask = sigma_clip(mc_txx2[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
            not_mask = list(map(operator.not_, mask))
            mc_txx2_filter = mc_txx2[1:, pi][not_mask]

            txx2_lo, txx2_hi = np.percentile(mc_txx2_filter, [16, 84])
            txx2_err = np.diff([txx2_lo, mc_txx2[0, pi], txx2_hi])
            self.txx2_err.append([txx2_err[0], txx2_err[1]])

        self.txx_res = {'xx': self.xx, 'txx':self.txx, 'txx1': self.txx1, 'txx2': self.txx2,
                        'txx_err': self.txx_err, 'txx1_err': self.txx1_err, 'txx2_err': self.txx2_err,
                        'csf': self.csf, 'csf1': self.csf1, 'csf2': self.csf2}

        print('\n+------------------------------------------------+')
        print(' %-5s%-10s%-8s%-8s%-8s%-8s' % ('id#', 'Txx', 'Txx-', 'Txx+', 'Txx1', 'Txx2'))
        print('+-----------------------------------------------+')
        _ = list(map(print, [' %-5d%-10.3f%-8.3f%-8.3f%-8.3f%-8.3f' % (i+1, t, t_err[0], t_err[1], t1, t2)
                                for i, (t, t_err, t1, t2) in enumerate(zip(self.txx_res['txx'],
                                                                        self.txx_res['txx_err'],
                                                                        self.txx_res['txx1'],
                                                                        self.txx_res['txx2']))]))
        print('+------------------------------------------------+')


    def save(self, savepath):
        """Save Txx results and diagnostic plots to disk.

        Serialises ``txx_res`` as a JSON file and writes a two-panel PDF
        showing the net rate light curve with Txx boundaries (upper panel)
        and the cumulative count curve with CSF levels (lower panel).

        Args:
            savepath: Directory path where output files are written; created
                if it does not exist.
        """

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.txx_res, savepath + '/txx_res.json')

        rcParams['font.family'] = "serif"
        rcParams['font.serif'] = ["STIX Two Text"]
        rcParams['mathtext.fontset'] = "stix"
        rcParams['font.size'] = 12
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42

        fig = plt.figure(figsize=(7, 6))
        gs = fig.add_gridspec(2, 1, wspace=0, hspace=0)
        ax1 = fig.add_subplot(gs[:1, 0])
        ax2 = fig.add_subplot(gs[1:, 0], sharex=ax1)

        ax1.plot(self.time, self.net, color='k', lw=1.0, label='Light Curve')
        for i in range(len(self.txx1)):
            ax1.axvline(self.txx1[i], color='g', lw=1.0, ls='--')
            ax1.axvline(self.txx2[i], color='g', lw=1.0, ls='--')
        ax1.set_ylabel('Rate')
        ax1.set_xlim([self.time[0], self.time[-1]])
        ax1.minorticks_on()
        ax1.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax1.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax1.tick_params(which='major', width=1.0, length=5)
        ax1.tick_params(which='minor', width=1.0, length=3)
        ax1.xaxis.set_ticks_position('both')
        ax1.yaxis.set_ticks_position('both')
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.legend(frameon=False)

        ax2.plot(self.time, self.ccts, color='gray', lw=1.0)
        ax2.plot(self.time[self.tindex], self.ccts[self.tindex], color='k', lw=1.0)
        for i in range(len(self.txx1)):
            ax2.axvline(self.txx1[i], color='g', lw=1.0, ls='--')
            ax2.axvline(self.txx2[i], color='g', lw=1.0, ls='--')
        for csfi in self.csf:
            ax2.axhline(csfi, color='orange', lw=1.0)
        for i in range(len(self.csf1)):
            ax2.axhline(self.csf1[i], color='orange', lw=1.0, ls='--')
            ax2.axhline(self.csf2[i], color='orange', lw=1.0, ls='--')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Accumulated counts')
        ax2.minorticks_on()
        ax2.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax2.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax2.tick_params(which='major', width=1.0, length=5)
        ax2.tick_params(which='minor', width=1.0, length=3)
        ax2.xaxis.set_ticks_position('both')
        ax2.yaxis.set_ticks_position('both')
        fig.savefig(savepath + '/txx.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)



def accumcts(time, ccts, pstart, pstop, xx, simple_err=False):
    """Compute cumulative-count-fraction levels and Txx start/stop times.

    Interpolates the cumulative count curve to 1000-point resolution when
    the native sampling is too coarse, calculates the background CSF levels
    between pulses, derives the :math:`(1-xx)/2` and :math:`1-(1-xx)/2`
    fraction levels (``csf1``, ``csf2``) for each pulse interval, and
    locates the corresponding times via ``find_txx``.

    Args:
        time: 1-D time array covering the analysis window.
        ccts: Cumulative net count array aligned with ``time``.
        pstart: Array of pulse start times; one entry per pulse.
        pstop: Array of pulse stop times; one entry per pulse.
        xx: Cumulative count fraction, e.g. ``0.9`` for T90.
        simple_err: When ``True``, also compute and return analytic
            uncertainty estimates on all CSF and Txx values.

    Returns:
        When ``simple_err`` is ``False``: a tuple
        ``(txx, txx1, txx2, csf, csf1, csf2)`` where each element is a list
        with one entry per pulse.

        When ``simple_err`` is ``True``: a tuple
        ``(txx, txx1, txx2, txx_err, txx1_err, txx2_err,
        csf, csf1, csf2, csf_err, csf1_err, csf2_err)``.
    """

    idx = np.argsort(pstop - pstart)[0]
    if len(np.where((time >= pstart[idx]) & (time <= pstop[idx]))[0]) < 1000:
        interp_dt = (pstop[idx] - pstart[idx]) / 1000
        interp_time = np.arange(time[0], time[-1] - 1e-5, interp_dt)
        interp = interp1d(time, ccts, kind='quadratic')
        interp_ccts = interp(interp_time)
    else:
        interp_time = time
        interp_ccts = ccts

    csf, csf1, csf2 = [], [], []
    txx, txx1, txx2 = [], [], []

    if simple_err:
        csf_err, csf1_err, csf2_err = [], [], []
        txx_err, txx1_err, txx2_err = [], [], []

    for l, r in zip(np.append(time[0], pstop), np.append(pstart, time[-1])):
        idx = np.where((interp_time >= l) & (interp_time <= r))[0]
        if len(idx) >= 1:
            csf_i = np.mean(interp_ccts[idx])
        else:
            csf_i = interp_ccts[np.argmin(np.abs(interp_time - l))]
        csf.append(csf_i)

        if simple_err:
            if len(idx) >= 1:
                csf_err_i = np.std(interp_ccts[idx])
            else:
                csf_err_i = 0.0
            csf_err.append(csf_err_i)

    dcsf = np.array(csf[1:]) - np.array(csf[:-1])

    for pi, (l, r) in enumerate(zip(np.append(time[0], pstop[:-1]), np.append(pstart[1:], time[-1]))):
        nn = (1 - xx) / 2
        dd = dcsf[pi] * nn

        csf1_i = csf[pi] + dd
        csf2_i = csf[pi + 1] - dd

        csf1.append(csf1_i)
        csf2.append(csf2_i)

        if simple_err:
            csf1_err_i = np.sqrt((1 - nn) ** 2 * csf_err[pi] ** 2 + nn ** 2 * csf_err[pi + 1] ** 2)
            csf2_err_i = np.sqrt(nn ** 2 * csf_err[pi] ** 2 + (1 - nn) ** 2 * csf_err[pi + 1] ** 2)

            csf1_err.append(csf1_err_i)
            csf2_err.append(csf2_err_i)

        pt = interp_time[np.where((interp_time >= l) & (interp_time <= r))]
        pcts = interp_ccts[np.where((interp_time >= l) & (interp_time <= r))]

        txx_i, txx1_i, txx2_i = find_txx(pt, pcts, csf1_i, csf2_i)

        txx.append(txx_i)
        txx1.append(txx1_i)
        txx2.append(txx2_i)

        if simple_err:
            _, txx1_lo_i, txx2_lo_i = find_txx(pt, pcts, csf1_i - csf1_err_i, csf2_i - csf2_err_i)
            _, txx1_hi_i, txx2_hi_i = find_txx(pt, pcts, csf1_i + csf1_err_i, csf2_i + csf2_err_i)
            txx1_le_i, txx1_he_i = txx1_i - txx1_lo_i, txx1_hi_i - txx1_i
            txx2_le_i, txx2_he_i = txx2_i - txx2_lo_i, txx2_hi_i - txx2_i

            txx1_i_sam = generate_asymmetric_gaussian(txx1_i, txx1_le_i, txx1_he_i, 1000)
            txx2_i_sam = generate_asymmetric_gaussian(txx2_i, txx2_le_i, txx2_he_i, 1000)
            txx_lo_i, txx_hi_i = np.percentile(txx2_i_sam - txx1_i_sam, [16, 84])
            txx_le_i, txx_he_i = txx_i - txx_lo_i, txx_hi_i - txx_i

            txx_err.append([txx_le_i, txx_he_i])
            txx1_err.append([txx1_le_i, txx1_he_i])
            txx2_err.append([txx2_le_i, txx2_he_i])

    if simple_err:
        return txx, txx1, txx2, txx_err, txx1_err, txx2_err, csf, csf1, csf2, csf_err, csf1_err, csf2_err
    else:
        return txx, txx1, txx2, csf, csf1, csf2



def find_txx(time, ccts, csf1, csf2):
    """Locate the start and stop times corresponding to CSF thresholds.

    Linearly interpolates the cumulative count curve onto a 1000-point grid,
    then scans forward to find the time ``txx1`` at which the curve crosses
    ``csf1`` from below, and ``txx2`` at which it crosses ``csf2`` from
    below.  The duration ``txx = txx2 - txx1`` is also returned.

    Args:
        time: 1-D time array for the pulse interval.
        ccts: Cumulative net count array aligned with ``time``.
        csf1: Lower cumulative-count-fraction level (start threshold).
        csf2: Upper cumulative-count-fraction level (stop threshold).

    Returns:
        A tuple ``(txx, txx1, txx2)`` where ``txx`` is the duration and
        ``txx1``, ``txx2`` are the start and stop times, respectively.
        All three values are ``0`` if no valid crossing is found.
    """

    interp_time = np.linspace(time[0], time[-1], 1000)
    interp = interp1d(time, ccts, kind='linear')
    interp_ccts = interp(interp_time)

    txx1, txx2 = 0, 0
    for i in range(1, len(interp_time)):
        if interp_ccts[i] < csf1:
            continue
        elif (interp_ccts[i-1] < csf1) and (interp_ccts[i] >= csf1):
            txx1 = interp_time[i]
            continue
        elif (csf1 <= interp_ccts[i-1] < csf2) and (csf1 < interp_ccts[i] <= csf2):
            continue
        elif (csf1 < interp_ccts[i-1] <= csf2) and (interp_ccts[i] > csf2):
            txx2 = interp_time[i-1]
            break
        else:
            continue

    txx = txx2 - txx1
    return txx, txx1, txx2
