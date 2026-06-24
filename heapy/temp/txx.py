"""Compute Txx duration metrics (T90, T50, etc.) for gamma-ray burst light curves.

Provides three concrete classes -- ``pgTxx``, ``ppTxx``, and ``ggTxx`` -- that
inherit from ``pgSignal`` / ``ppSignal`` / ``ggSignal`` and add pulse detection
and cumulative-count-fraction duration analysis.  Monte Carlo simulation is
used by default to propagate uncertainties on the start/stop times.

Example:
    from heapy.temp.txx import pgTxx
    txx = pgTxx(ts, bins)
    txx.calculate(xx=0.9)
    txx.save('/output/dir')
"""

import operator
import os
import warnings

from astropy.stats import mad_std, sigma_clip
import numpy as np

from ..auto.signal import ggSignal, pgSignal, ppSignal
from ..auto.signal_utils import detect_pulses_by_snr, indices_in_intervals
from ..util.tools import format_message, json_dump, plt_rc_context
from .temp_utils import TxxPlotter, accumcts


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

    @classmethod
    def frombin(cls, cts, bins, exp=None, ignore=None, random_seed=450001):

        inst = super().frombin(cts, bins, exp=exp, ignore=ignore, random_seed=random_seed)

        inst.pulse_res = None
        inst.txx_res = None

        return inst

    @classmethod
    def from_components(cls, obj_list):

        inst = super().from_components(obj_list)

        inst.pulse_res = None
        inst.txx_res = None

        return inst

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

        if self.sort_res is None:
            self.loop(p0=p0, sigma=sigma, deg=deg)

        self.pstart, self.pstop = detect_pulses_by_snr(self.re_snr, self.edges, sigma, mp=mp)

        self.pulse_res = {'pstart': self.pstart, 'pstop': self.pstop}

    def mc_simulation(self, nmc, random_seed=450001):
        """Generate Monte Carlo realisations of the net count light curve.

        Draws Poisson samples for the source counts and Gaussian samples for
        the background counts, then stacks the net realisation matrix as
        ``self.mc_ncts`` with the observed data in row 0.

        Args:
            nmc: Number of Monte Carlo realisations to generate.
            random_seed: Seed for the local RNG used to draw the
                Poisson/Gaussian samples. Default ensures reproducibility
                across runs; pass ``None`` for OS entropy.
        """

        self.nmc = int(nmc)
        self.nsample = len(self.time)

        rng = np.random.default_rng(random_seed)
        # NaN cts (gap bins) breaks ``rng.poisson``; substitute zero for the
        # Poisson rate at those positions and re-flag the rows below.
        src_lam = np.nan_to_num(self.cts, nan=0.0)
        src_sample = rng.poisson(lam=src_lam, size=(self.nmc, self.nsample))
        bkg_sample = rng.normal(loc=self.bcts, scale=self.bcts_err, size=(self.nmc, self.nsample))

        self.mc_ncts = np.vstack([self.ncts, src_sample - bkg_sample])

        # Mark gap bins as NaN across all MC rows so :meth:`calculate`'s
        # ``np.nancumsum`` treats them as no-contribution rather than the
        # spurious ``-bcts`` draw the Gaussian background sample would yield.
        if self.gap_int:
            gap_idx = indices_in_intervals(self.lbins, self.rbins, self.gap_int)
            self.mc_ncts[:, gap_idx] = np.nan

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

        if self.pulse_res is None:
            self.find_pulse()

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

        # NaN-aware cumsum so gap bins (NaN in self.ncts) contribute zero
        # without propagating NaN through the rest of the cumulative curve.
        self.ccts = np.nancumsum(self.ncts)

        if simple_err:
            (
                self.txx,
                self.txx1,
                self.txx2,
                self.txx_err,
                self.txx1_err,
                self.txx2_err,
                self.csf,
                self.csf1,
                self.csf2,
                self.csf_err,
                self.csf1_err,
                self.csf2_err,
            ) = accumcts(
                self.time[self.tindex],
                self.ccts[self.tindex],
                self.pstart,
                self.pstop,
                self.xx,
                simple_err=True,
            )

            self.txx_res = {
                'xx': self.xx,
                'txx': self.txx,
                'txx1': self.txx1,
                'txx2': self.txx2,
                'txx_err': self.txx_err,
                'txx1_err': self.txx1_err,
                'txx2_err': self.txx2_err,
                'csf': self.csf,
                'csf1': self.csf1,
                'csf2': self.csf2,
                'csf_err': self.csf_err,
                'csf1_err': self.csf1_err,
                'csf2_err': self.csf2_err,
            }

        else:
            self.mc_simulation(1000)

            mc_csf, mc_csf1, mc_csf2 = [], [], []
            mc_txx, mc_txx1, mc_txx2 = [], [], []

            for ncts in self.mc_ncts:
                ccts = np.nancumsum(ncts)

                txx, txx1, txx2, csf, csf1, csf2 = accumcts(
                    self.time[self.tindex],
                    ccts[self.tindex],
                    self.pstart,
                    self.pstop,
                    self.xx,
                    simple_err=False,
                )

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

            self.txx_res = {
                'xx': self.xx,
                'txx': self.txx,
                'txx1': self.txx1,
                'txx2': self.txx2,
                'txx_err': self.txx_err,
                'txx1_err': self.txx1_err,
                'txx2_err': self.txx2_err,
                'csf': self.csf,
                'csf1': self.csf1,
                'csf2': self.csf2,
            }

        XX = int(self.xx * 100)

        msg = [
            f'{"id#":<5}{f"T{XX}":<10}{f"T{XX}-":<8}{f"T{XX}+":<8}{f"T{XX}1":<8}{f"T{XX}2":<8}'
        ] + [
            f'{i + 1:<5d}{t:<10.3f}{t_err[0]:<8.3f}{t_err[1]:<8.3f}{t1:<8.3f}{t2:<8.3f}'
            for i, (t, t_err, t1, t2) in enumerate(
                zip(
                    self.txx_res['txx'],
                    self.txx_res['txx_err'],
                    self.txx_res['txx1'],
                    self.txx_res['txx2'],
                    strict=False,
                )
            )
        ]
        print(format_message(msg))

    def save(self, savepath):
        """Save Txx results and diagnostic plots to disk.

        Serialises ``pulse_res`` and ``txx_res`` as JSON files and
        delegates the two-panel diagnostic figure to
        :class:`~heapy.temp.temp_utils.TxxPlotter`. ``self.rate`` already
        carries ``NaN`` at gap bins (preserved by :meth:`frombin`), so
        only the cumulative curve needs an extra mask (registered via
        :meth:`~heapy.temp.temp_utils.TxxPlotter.set_gaps`) because
        :meth:`calculate` keeps ``ccts`` continuous through gaps via
        ``np.nancumsum``.

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

        with plt_rc_context():
            fig = TxxPlotter()
            if self.gap_int:
                fig.set_gaps(self.gap_int, self.lbins, self.rbins)
            fig.plot_curve(self.time, self.rate, bak=self.bak)
            fig.plot_ccts(self.time, self.ccts, self.tindex)
            fig.plot_txx(self.txx1, self.txx2, self.csf, self.csf1, self.csf2)
            fig.save(savepath + '/txx.pdf')


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

        self.pstart, self.pstop = detect_pulses_by_snr(self.re_snr, self.edges, sigma, mp=mp)

        self.pulse_res = {'pstart': self.pstart, 'pstop': self.pstop}

    def mc_simulation(self, nmc, random_seed=450001):
        """Generate Monte Carlo realisations of the net count light curve.

        Draws independent Poisson samples for both source and background
        counts and stacks the net realisation matrix as ``self.mc_ncts``
        with the observed data in row 0.

        Args:
            nmc: Number of Monte Carlo realisations to generate.
            random_seed: Seed for the local RNG used to draw the
                Poisson samples. Default ensures reproducibility across
                runs; pass ``None`` for OS entropy.
        """

        self.nmc = int(nmc)
        self.nsample = len(self.time)

        rng = np.random.default_rng(random_seed)
        src_sample = rng.poisson(lam=self.cts, size=(self.nmc, self.nsample))
        bkg_sample = rng.poisson(lam=self.bcts, size=(self.nmc, self.nsample))

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

        if self.pulse_res is None:
            self.find_pulse()

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

            txx, txx1, txx2, csf, csf1, csf2 = accumcts(
                self.time[self.tindex],
                ccts[self.tindex],
                self.pstart,
                self.pstop,
                self.xx,
                simple_err=False,
            )

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

        self.txx_res = {
            'xx': self.xx,
            'txx': self.txx,
            'txx1': self.txx1,
            'txx2': self.txx2,
            'txx_err': self.txx_err,
            'txx1_err': self.txx1_err,
            'txx2_err': self.txx2_err,
            'csf': self.csf,
            'csf1': self.csf1,
            'csf2': self.csf2,
        }

        XX = int(self.xx * 100)

        msg = [
            f'{"id#":<5}{f"T{XX}":<10}{f"T{XX}-":<8}{f"T{XX}+":<8}{f"T{XX}1":<8}{f"T{XX}2":<8}'
        ] + [
            f'{i + 1:<5d}{t:<10.3f}{t_err[0]:<8.3f}{t_err[1]:<8.3f}{t1:<8.3f}{t2:<8.3f}'
            for i, (t, t_err, t1, t2) in enumerate(
                zip(
                    self.txx_res['txx'],
                    self.txx_res['txx_err'],
                    self.txx_res['txx1'],
                    self.txx_res['txx2'],
                    strict=False,
                )
            )
        ]
        print(format_message(msg))

    def save(self, savepath):
        """Save Txx results and diagnostic plots to disk.

        Serialises ``pulse_res`` and ``txx_res`` as JSON files and
        delegates the two-panel diagnostic figure to
        :class:`~heapy.temp.temp_utils.TxxPlotter`.

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

        with plt_rc_context():
            fig = TxxPlotter()
            fig.plot_curve(self.time, self.rate, bak=self.bak)
            fig.plot_ccts(self.time, self.ccts, self.tindex)
            fig.plot_txx(self.txx1, self.txx2, self.csf, self.csf1, self.csf2)
            fig.save(savepath + '/txx.pdf')


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

        self.pstart, self.pstop = detect_pulses_by_snr(self.re_snr, self.edges, sigma, mp=mp)

        self.pulse_res = {'pstart': self.pstart, 'pstop': self.pstop}

    def mc_simulation(self, nmc, random_seed=450001):
        """Generate Monte Carlo realisations of the net count light curve.

        Draws Gaussian samples using ``self.ncts`` and ``self.ncts_err`` and
        stacks the realisation matrix as ``self.mc_ncts`` with the observed
        data in row 0.

        Args:
            nmc: Number of Monte Carlo realisations to generate.
            random_seed: Seed for the local RNG used to draw the
                Gaussian samples. Default ensures reproducibility across
                runs; pass ``None`` for OS entropy.
        """

        self.nmc = int(nmc)
        self.nsample = len(self.time)

        rng = np.random.default_rng(random_seed)
        sample = rng.normal(loc=self.ncts, scale=self.ncts_err, size=(self.nmc, self.nsample))

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

        if self.pulse_res is None:
            self.find_pulse()

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

            txx, txx1, txx2, csf, csf1, csf2 = accumcts(
                self.time[self.tindex],
                ccts[self.tindex],
                self.pstart,
                self.pstop,
                self.xx,
                simple_err=False,
            )

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

        self.txx_res = {
            'xx': self.xx,
            'txx': self.txx,
            'txx1': self.txx1,
            'txx2': self.txx2,
            'txx_err': self.txx_err,
            'txx1_err': self.txx1_err,
            'txx2_err': self.txx2_err,
            'csf': self.csf,
            'csf1': self.csf1,
            'csf2': self.csf2,
        }

        XX = int(self.xx * 100)

        msg = [
            f'{"id#":<5}{f"T{XX}":<10}{f"T{XX}-":<8}{f"T{XX}+":<8}{f"T{XX}1":<8}{f"T{XX}2":<8}'
        ] + [
            f'{i + 1:<5d}{t:<10.3f}{t_err[0]:<8.3f}{t_err[1]:<8.3f}{t1:<8.3f}{t2:<8.3f}'
            for i, (t, t_err, t1, t2) in enumerate(
                zip(
                    self.txx_res['txx'],
                    self.txx_res['txx_err'],
                    self.txx_res['txx1'],
                    self.txx_res['txx2'],
                    strict=False,
                )
            )
        ]
        print(format_message(msg))

    def save(self, savepath):
        """Save Txx results and diagnostic plots to disk.

        Serialises ``pulse_res`` and ``txx_res`` as JSON files and
        delegates the two-panel diagnostic figure to
        :class:`~heapy.temp.temp_utils.TxxPlotter`. ggTxx passes
        ``self.net`` as the primary curve (input is already
        background-subtracted) and omits the background overlay.

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

        with plt_rc_context():
            fig = TxxPlotter()
            fig.plot_curve(self.time, self.net)
            fig.plot_ccts(self.time, self.ccts, self.tindex)
            fig.plot_txx(self.txx1, self.txx2, self.csf, self.csf1, self.csf2)
            fig.save(savepath + '/txx.pdf')
