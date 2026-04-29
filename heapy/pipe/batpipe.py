"""Swift/BAT data reduction pipeline wrapping HEASOFT command-line tools.

Provides the ``batPipe`` class, which automates event energy conversion,
detector-plane image generation, mask weighting, light curve extraction,
and spectral product generation for Swift BAT gamma-ray burst observations.

Typical usage:
    pipe = batPipe.from_batobs('00012345678', datapath='/data/swift')
    pipe.timezero = '2024-01-01T12:00:00'
    pipe.filter_energy([15, 150])
    pipe.extract_curve(savepath='./output/curve')
    pipe.extract_spectrum(savepath='./output/spectrum')
"""

import os
import subprocess
import numpy as np
from astropy.io import fits
import plotly.graph_objs as go

from ..data.retrieve import swiftRetrieve
from ..temp.txx import ggTxx
from ..auto.signal import ggSignal
from ..util.data import rebin
from ..util.time import swift_met_to_utc, swift_utc_to_met



class batPipe(object):
    """Orchestrate the Swift/BAT event-file reduction pipeline.

    Wraps a sequence of HEASOFT tools (``bateconvert``, ``batbinevt``,
    ``bathotpix``, ``batmaskwtevt``, ``batfftimage``, ``batdrmgen``, etc.)
    to produce calibrated light curves, sky images, and spectra from raw
    unfiltered event files.  General pre-processing (energy conversion, DPI
    creation, hot-pixel masking, and mask weighting) is executed automatically
    on construction and whenever a core file path is changed.

    Attributes:
        general_savepath: Directory used for intermediate pipeline products.
        dpifile: Path to the detector-plane image FITS file.
        maskfile: Path to the hot-pixel-corrected mask FITS file.
        lcfile: Path to the most recently extracted light curve FITS file.
        specfile: Path to the most recently extracted spectrum PHA file.
        respfile: Path to the most recently generated response matrix file.
        lc_time: Time bin centres relative to ``timezero`` (seconds).
        lc_net_rate: Net count rate per detector per bin (cts/s/det).
        lc_net_rate_err: Uncertainty on ``lc_net_rate``.
        lc_net_cts: Net counts per bin.
        lc_net_cts_err: Uncertainty on ``lc_net_cts``.
        lc_net_crate: Cumulative sum of ``lc_net_rate``.
        lc_bins: Bin edges array (length = number of bins + 1).
        lc_bin_list: Array of shape ``(N, 2)`` with ``[t_left, t_right]`` per bin.
        ngoodpix: Number of good detector pixels reported by ``batbinevt``.

    Example:
        >>> pipe = batPipe.from_batobs('00012345678')
        >>> pipe.extract_curve(savepath='./lc')
    """

    os.environ["HEADASNOQUERY"] = "1"

    def __init__(self,
                 ufevtfile=None,
                 caldbfile=None,
                 detmaskfile=None,
                 attfile=None,
                 auxfile=None
                 ):
        """Initialize the pipeline and run general pre-processing.

        Triggers energy conversion, DPI generation, hot-pixel masking, and
        mask weighting unless the intermediate products already exist on disk.

        Args:
            ufevtfile: Path to the unfiltered event FITS file.
            caldbfile: Path to the CALDB gain/offset calibration file.
            detmaskfile: Path to the detector quality mask file.
            attfile: Path to the attitude (star-tracker) file.
            auxfile: Path to the auxiliary data file required by
                ``batupdatephakw``.
        """

        self._ufevtfile = ufevtfile
        self._caldbfile = caldbfile
        self._detmaskfile = detmaskfile
        self._attfile = attfile
        self._auxfile = auxfile

        self._general_processing()


    @classmethod
    def from_batobs(cls, obsid, datapath=None):
        """Construct a pipeline instance from a Swift BAT observation ID.

        Retrieves the required files for the observation via
        ``swiftRetrieve.from_batobs`` and passes them to ``__init__``.

        Args:
            obsid: Swift observation identifier string (e.g. ``'00012345678'``).
            datapath: Local root directory to search for observation data.
                Passed directly to ``swiftRetrieve.from_batobs``; uses the
                default search path when ``None``.

        Returns:
            A fully initialised ``batPipe`` instance ready for data reduction.
        """

        rtv = swiftRetrieve.from_batobs(obsid, datapath)

        ufevtfile = rtv.rtv_res['ufevt']
        caldbfile = rtv.rtv_res['caldb']
        detmaskfile = rtv.rtv_res['detmask']
        attfile = rtv.rtv_res['att']
        auxfile = rtv.rtv_res['aux']

        return cls(ufevtfile, caldbfile, detmaskfile, attfile, auxfile)


    @property
    def ufevtfile(self):
        """Return the absolute path to the unfiltered event file."""

        return os.path.abspath(self._ufevtfile)


    @ufevtfile.setter
    def ufevtfile(self, new_ufevtfile):

        self._ufevtfile = new_ufevtfile

        self._general_processing()


    @property
    def caldbfile(self):
        """Return the absolute path to the CALDB calibration file."""

        return os.path.abspath(self._caldbfile)


    @caldbfile.setter
    def caldbfile(self, new_caldbfile):

        self._caldbfile = new_caldbfile

        self._general_processing()


    @property
    def detmaskfile(self):
        """Return the absolute path to the detector quality mask file."""

        return os.path.abspath(self._detmaskfile)


    @detmaskfile.setter
    def detmaskfile(self, new_detmaskfile):

        self._detmaskfile = new_detmaskfile

        self._general_processing()


    @property
    def attfile(self):
        """Return the absolute path to the attitude file."""

        return os.path.abspath(self._attfile)


    @attfile.setter
    def attfile(self, new_attfile):

        self._attfile = new_attfile

        self._general_processing()


    @property
    def auxfile(self):
        """Return the absolute path to the auxiliary data file."""

        return os.path.abspath(self._auxfile)


    @auxfile.setter
    def auxfile(self, new_auxfile):

        self._auxfile = new_auxfile


    @property
    def ra(self):
        """Return the target right ascension in degrees from the event header.

        Reads the ``RA_OBJ`` keyword from the ``EVENTS`` extension of the
        unfiltered event file.

        Returns:
            Right ascension of the observed object in degrees (J2000).
        """

        hdu = fits.open(self.ufevtfile)
        val = hdu['EVENTS'].header['RA_OBJ']
        hdu.close()

        return val


    @property
    def dec(self):
        """Return the target declination in degrees from the event header.

        Reads the ``DEC_OBJ`` keyword from the ``EVENTS`` extension of the
        unfiltered event file.

        Returns:
            Declination of the observed object in degrees (J2000).
        """

        hdu = fits.open(self.ufevtfile)
        val = hdu['EVENTS'].header['DEC_OBJ']
        hdu.close()

        return val


    def _run_comands(self, commands):

        process = subprocess.Popen(commands,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)

        stdout, stderr = process.communicate()

        return stdout, stderr


    def _bateconvert(self, std=False):

        commands = ['bateconvert',
                    f'infile={self.ufevtfile}',
                    f'calfile={self.caldbfile}',
                    'residfile=CALDB',
                    'pulserfile=CALDB',
                    'fltpulserfile=CALDB']

        stdout, stderr = self._run_comands(commands)

        if std:
            print(stdout)
            print(stderr)


    def _batbinevt(self, std=False):

        commands = ['batbinevt',
                    'weighted=no',
                    'outunits=counts',
                    f'infile={self.ufevtfile}',
                    f'outfile={self.dpifile}',
                    'outtype=dpi',
                    'timedel=0',
                    'timebinalg=uniform',
                    'energybins=-']

        stdout, stderr = self._run_comands(commands)

        if std:
            print(stdout)
            print(stderr)


    def _bathotpix(self, std=False):

        commands = ['bathotpix',
                    f'detmask={self.detmaskfile}',
                    f'infile={self.dpifile}',
                    f'outfile={self.maskfile}']

        stdout, stderr = self._run_comands(commands)

        if std:
            print(stdout)
            print(stderr)


    def _batmaskwtevt(self, std=False):

        commands = ['batmaskwtevt',
                    f'detmask={self.maskfile}',
                    f'infile={self.ufevtfile}',
                    f'attitude={self.attfile}',
                    f'ra={self.ra}',
                    f'dec={self.dec}']

        stdout, stderr = self._run_comands(commands)

        if std:
            print(stdout)
            print(stderr)


    def _general_processing(self):

        self.general_savepath = os.path.dirname(os.path.dirname(self.ufevtfile)) + '/batpipe'

        self.dpifile = self.general_savepath + '/grb.dpi'
        self.maskfile = self.general_savepath + '/grb.mask'

        if not (os.path.exists(self.dpifile) and os.path.exists(self.maskfile)):

            if os.path.exists(self.general_savepath):
                os.rmdir(self.general_savepath)
            os.makedirs(self.general_savepath)

            self._bateconvert()
            self._batbinevt()
            self._bathotpix()
            self._batmaskwtevt()


    def _batbinevt_image(self, std=False):

        commands = ['batbinevt',
                    f'detmask={self.maskfile}',
                    'ecol=energy',
                    'weighted=no',
                    'outunits=counts',
                    f'infile={self.ufevtfile}',
                    f'outfile={self.dpi4file}',
                    'outtype=dpi',
                    'timedel=0',
                    'timebinalg=uniform',
                    'energybins=15-25, 25-50, 50-100, 100-350']

        stdout, stderr = self._run_comands(commands)

        if std:
            print(stdout)
            print(stderr)


    def _batfftimage(self, std=False):

        commands = ['batfftimage',
                    f'detmask={self.maskfile}',
                    f'infile={self.dpi4file}',
                    f'outfile={self.img4file}',
                    f'attitude={self.attfile}']

        stdout, stderr = self._run_comands(commands)

        if std:
            print(stdout)
            print(stderr)


    def extract_image(self, std=False):
        """Generate a four-band sky image via coded-aperture deconvolution.

        Runs ``batbinevt`` in DPI mode with the standard four BAT energy bands
        (15–25, 25–50, 50–100, 100–350 keV) and then deconvolves the result
        with ``batfftimage``.  Any pre-existing intermediate files are removed
        before the run.  Outputs ``grb_4.dpi`` and ``grb_4.img`` under
        ``general_savepath``.

        Args:
            std: When ``True``, print the stdout and stderr of each
                HEASOFT subprocess to the console.
        """

        self.dpi4file = self.general_savepath + '/grb_4.dpi'

        if os.path.exists(self.dpi4file):
            os.remove(self.dpi4file)

        self.img4file = self.general_savepath + '/grb_4.img'

        if os.path.exists(self.img4file):
            os.remove(self.img4file)

        self._batbinevt_image(std=std)
        self._batfftimage(std=std)


    @property
    def utcf(self):
        """Return the UTC correction factor stored in the event file header.

        Reads the ``UTCFINIT`` keyword from the ``EVENTS`` extension, which is
        the initial UTC correction (seconds) applied by the Swift clock.

        Returns:
            UTC correction factor in seconds.
        """

        hdu = fits.open(self._ufevtfile)
        val = hdu['EVENTS'].header['UTCFINIT']
        hdu.close()

        return val


    @property
    def trigtime(self):
        """Return the trigger time in Swift Mission Elapsed Time (MET).

        Reads the ``TRIGTIME`` keyword from the ``EVENTS`` extension header.

        Returns:
            Trigger time in seconds (MET).
        """

        hdu = fits.open(self._ufevtfile)
        val = hdu['EVENTS'].header['TRIGTIME']
        hdu.close()

        return val


    @property
    def tstart(self):
        """Return the observation start time in Swift MET seconds.

        Reads the ``TSTART`` keyword from the ``EVENTS`` extension header.

        Returns:
            Observation start time in seconds (MET).
        """

        hdu = fits.open(self._ufevtfile)
        val = hdu['EVENTS'].header['TSTART']
        hdu.close()

        return val


    @property
    def tstop(self):
        """Return the observation stop time in Swift MET seconds.

        Reads the ``TSTOP`` keyword from the ``EVENTS`` extension header.

        Returns:
            Observation stop time in seconds (MET).
        """

        hdu = fits.open(self._ufevtfile)
        val = hdu['EVENTS'].header['TSTOP']
        hdu.close()

        return val


    @property
    def timezero(self):
        """Return the reference time used as t=0 for all relative time axes.

        Defaults to ``trigtime`` when no explicit value has been set via the
        setter.  Can be overridden by assigning a MET float or a UTC string.

        Returns:
            Reference time in seconds (MET).
        """

        try:
            return self._timezero
        except AttributeError:
            return self.trigtime


    @timezero.setter
    def timezero(self, new_timezero):

        if isinstance(new_timezero, float):
            self._timezero = new_timezero

        elif isinstance(new_timezero, str):
            self._timezero = swift_utc_to_met(new_timezero, self.utcf)

        else:
            raise ValueError('not expected type for timezero')


    @property
    def timezero_utc(self):
        """Return ``timezero`` converted to a UTC string.

        Returns:
            ISO-format UTC string corresponding to the current ``timezero``.
        """

        return swift_met_to_utc(self.timezero, self.utcf)


    def filter_time(self, t1t2):
        """Set the time interval used for light curve and spectral extraction.

        Args:
            t1t2: Two-element list ``[t1, t2]`` giving the start and stop time
                in seconds relative to ``timezero``, or ``None`` to reset to
                the full observation span.

        Raises:
            ValueError: If ``t1t2`` is neither a list nor ``None``.
        """

        if isinstance(t1t2, (list, type(None))):
            self._time_filter = t1t2

        else:
            raise ValueError('t1t2 is extected to be list or None')


    def filter_energy(self, e1e2):
        """Set the energy band used for light curve and spectral extraction.

        Args:
            e1e2: Two-element list ``[e1, e2]`` giving the lower and upper
                energy bound in keV, or ``None`` to reset to the default
                full-band range (15–350 keV).

        Raises:
            ValueError: If ``e1e2`` is neither a list nor ``None``.
        """

        if isinstance(e1e2, (list, type(None))):
            self._energy_filter = e1e2

        else:
            raise ValueError('e1e2 is extected to be list or None')


    @property
    def time_filter(self):
        """Return the active time filter as ``[t1, t2]`` relative to ``timezero``.

        Falls back to the full observation span ``[tstart, tstop]`` (shifted by
        ``timezero``) when no filter has been set or when it was explicitly
        cleared to ``None``.

        Returns:
            Two-element list of start and stop times in seconds relative to
            ``timezero``.
        """

        try:
            self._time_filter
        except AttributeError:
            return [self.tstart - self.timezero, self.tstop - self.timezero]
        else:
            if self._time_filter is None:
                return [self.tstart - self.timezero, self.tstop - self.timezero]
            else:
                return self._time_filter


    @property
    def energy_filter(self):
        """Return the active energy filter as ``[e1, e2]`` in keV.

        Falls back to ``[15, 350]`` when no filter has been set or when it was
        explicitly cleared to ``None``.

        Returns:
            Two-element list of lower and upper energy bounds in keV.
        """

        try:
            self._energy_filter
        except AttributeError:
            return [15, 350]
        else:
            if self._energy_filter is None:
                return [15, 350]
            else:
                return self._energy_filter


    @property
    def lc_t1t2(self):
        """Return the time range used for light curve extraction.

        Falls back to ``time_filter`` when no explicit light curve interval
        has been set or when it was cleared to ``None``.

        Returns:
            Two-element list ``[t1, t2]`` in seconds relative to ``timezero``.
        """

        try:
            self._lc_t1t2
        except AttributeError:
            return self.time_filter
        else:
            if self._lc_t1t2 is not None:
                return self._lc_t1t2
            else:
                return self.time_filter


    @lc_t1t2.setter
    def lc_t1t2(self, new_lc_t1t2):

        if isinstance(new_lc_t1t2, (list, type(None))):
            self._lc_t1t2 = new_lc_t1t2
        else:
            raise ValueError('lc_t1t2 is extected to be list or None')


    @property
    def lc_interval(self):
        """Return the total duration of the light curve time range in seconds.

        Returns:
            Duration as ``lc_t1t2[1] - lc_t1t2[0]`` in seconds.
        """

        return self.lc_t1t2[1] - self.lc_t1t2[0]


    @property
    def lc_binsize(self):
        """Return the light curve bin size in seconds.

        Defaults to ``lc_interval / 300`` (yielding approximately 300 bins)
        when no explicit bin size has been set or when it was cleared to
        ``None``.

        Returns:
            Bin width in seconds.
        """

        try:
            self._lc_binsize
        except AttributeError:
            return self.lc_interval / 300
        else:
            if self._lc_binsize is not None:
                return self._lc_binsize
            else:
                return self.lc_interval / 300


    @lc_binsize.setter
    def lc_binsize(self, new_lc_binsize):

        if isinstance(new_lc_binsize, (int, float, type(None))):
            self._lc_binsize = new_lc_binsize
        else:
            raise ValueError('lc_binsize is extected to be int, float or None')


    @property
    def lc_tstart(self):
        """Return the light curve extraction start time in Swift MET seconds.

        Accounts for ``timezero`` and half a bin width so that the first bin
        centre aligns with ``lc_t1t2[0]``.

        Returns:
            Absolute start time in seconds (MET).
        """

        return self.lc_t1t2[0] + self.timezero + self.lc_binsize / 2


    @property
    def lc_tstop(self):
        """Return the light curve extraction stop time in Swift MET seconds.

        Accounts for ``timezero`` and half a bin width so that the last bin
        centre aligns with ``lc_t1t2[1]``.

        Returns:
            Absolute stop time in seconds (MET).
        """

        return self.lc_t1t2[1] + self.timezero + self.lc_binsize / 2


    @property
    def lc_ebin(self):
        """Return the energy band string passed to ``batbinevt`` for the light curve.

        Returns:
            Energy range formatted as ``'e1-e2'`` in keV (e.g. ``'15-350'``).
        """

        return f'{self.energy_filter[0]}-{self.energy_filter[1]}'


    def _batbinevt_curve(self, std=False):

        commands = ['batbinevt',
                    f'detmask={self.maskfile}',
                    f'tstart={self.lc_tstart}',
                    f'tstop={self.lc_tstop}',
                    f'infile={self.ufevtfile}',
                    f'outfile={self.lcfile}',
                    'outtype=lc',
                    f'timedel={self.lc_binsize}',
                    'timebinalg=uniform',
                    f'energybins={self.lc_ebin}']

        stdout, stderr = self._run_comands(commands)

        if std:
            print(stdout)
            print(stderr)


    def _extract_curve(self, std=False):

        self.lcfile = self.general_savepath + '/grb.lc'

        if os.path.exists(self.lcfile):
            os.remove(self.lcfile)

        self._batbinevt_curve(std=std)

        lc_hdu = fits.open(self.lcfile)
        self.lc_time = np.array(lc_hdu['RATE'].data['TIME']) - self.timezero
        self.lc_net_rate = np.array(lc_hdu['RATE'].data['RATE'])
        self.lc_net_rate_err = np.array(lc_hdu['RATE'].data['ERROR'])
        self.ngoodpix = lc_hdu['RATE'].header['NGOODPIX']
        lc_hdu.close()

        lbins, rbins = self.lc_time - self.lc_binsize / 2, self.lc_time + self.lc_binsize / 2
        self.lc_bin_list = np.vstack((lbins, rbins)).T
        self.lc_bins = np.append(self.lc_time - self.lc_binsize / 2, self.lc_time[-1] + self.lc_binsize / 2)

        self.lc_net_crate = np.cumsum(self.lc_net_rate)
        self.lc_net_cts = self.lc_net_rate * self.lc_binsize
        self.lc_net_cts_err = self.lc_net_rate_err * self.lc_binsize


    @property
    def gs_p0(self):
        """Return the Bayesian-block prior probability for signal classification.

        Controls the false-positive rate in the ``ggSignal`` Bayesian block
        analysis.  Defaults to ``0.05`` when no value has been explicitly set.

        Returns:
            Prior probability (dimensionless, in the range ``(0, 1)``).
        """

        try:
            self._gs_p0
        except AttributeError:
            return 0.05
        else:
            return self._gs_p0


    @gs_p0.setter
    def gs_p0(self, new_gs_p0):

        if isinstance(new_gs_p0, (float, int)):
            self._gs_p0 = new_gs_p0
        else:
            raise ValueError('gs_p0 is extected to be float or int')


    @property
    def gs_sigma(self):
        """Return the significance threshold for pulse detection in sigma units.

        Used by ``ggSignal`` and ``ggTxx`` to identify statistically significant
        emission intervals.  Defaults to ``3`` when no value has been set.

        Returns:
            Detection threshold in units of Gaussian standard deviations.
        """

        try:
            self._gs_sigma
        except AttributeError:
            return 3
        else:
            return self._gs_sigma


    @gs_sigma.setter
    def gs_sigma(self, new_gs_sigma):

        if isinstance(new_gs_sigma, (int, float)):
            self._gs_sigma = new_gs_sigma
        else:
            raise ValueError('gs_sigma is extected to be int or float')


    @property
    def lc_gs(self):
        """Extract the light curve and return a fitted ``ggSignal`` object.

        Runs ``_extract_curve`` on every access, then applies Bayesian block
        signal classification with ``gs_p0`` and ``gs_sigma``.

        Returns:
            A ``ggSignal`` instance with the ``loop`` analysis already executed.
        """

        self._extract_curve()

        lc_gs = ggSignal(self.lc_net_cts, self.lc_net_cts_err, self.lc_bins)
        lc_gs.loop(p0=self.gs_p0, sigma=self.gs_sigma)

        return lc_gs


    def extract_curve(self, savepath='./curve', sig=True, std=False, show=False):
        """Extract the light curve and save plots and data files to disk.

        Runs signal classification (``ggSignal``) when ``sig`` is ``True``,
        otherwise performs a plain extraction.  Produces an interactive HTML
        light curve and a cumulative count-rate plot in ``savepath``.

        Args:
            savepath: Directory where output files are written.  Created if it
                does not already exist.
            sig: When ``True``, run Bayesian block signal classification via
                ``lc_gs`` and save its results under ``savepath/ggsignal``.
            std: When ``True``, print the stdout and stderr of each HEASOFT
                subprocess to the console.
            show: When ``True``, display the Plotly figure interactively in
                the browser.
        """

        savepath = os.path.abspath(savepath)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        if sig:
            self.lc_gs.save(savepath=savepath + '/ggsignal')
        else:
            self._extract_curve(std=std)

        fig = go.Figure()
        net = go.Scatter(x=self.lc_time,
                         y=self.lc_net_rate,
                         mode='lines+markers',
                         name='net counts rate',
                         showlegend=True,
                         error_y=dict(
                             type='data',
                             array=self.lc_net_rate_err,
                             thickness=1.5,
                             width=0),
                         marker=dict(symbol='cross-thin', size=0))
        fig.add_trace(net)

        fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)')
        fig.update_yaxes(title_text=f'Counts per second per detector (binsize={self.lc_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        if show: fig.show()
        fig.write_html(savepath + '/lc.html', include_plotlyjs='cdn')
        fig.write_image(savepath + '/lc.pdf')

        fig = go.Figure()
        net = go.Scatter(x=self.lc_time,
                         y=self.lc_net_crate,
                         mode='lines',
                         name='net cumulated rate',
                         showlegend=True)
        fig.add_trace(net)

        fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)')
        fig.update_yaxes(title_text=f'Cumulated count rate (binsize={self.lc_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        fig.write_html(savepath + '/cum_lc.html', include_plotlyjs='cdn')
        fig.write_image(savepath + '/cum_lc.pdf')


    def calculate_txx(self, mp=True, xx=0.9, pstart=None, pstop=None,
                      lbkg=None, rbkg=None, savepath='./curve/duration', std=False):
        """Compute the Txx burst duration and save results.

        Uses ``ggTxx`` to locate the pulse interval via Bayesian block signal
        detection and then integrates the specified fraction ``xx`` of the
        total counts.  Results are serialised to ``savepath``.

        Args:
            mp: When ``True``, use multi-pulse mode in ``ggTxx.find_pulse``.
            xx: Fraction of total burst counts to integrate for the duration
                estimate (e.g. ``0.9`` gives T90).
            pstart: Manual override for the pulse start time relative to
                ``timezero`` in seconds.  Uses the detected start when ``None``.
            pstop: Manual override for the pulse stop time relative to
                ``timezero`` in seconds.  Uses the detected stop when ``None``.
            lbkg: Left background interval as ``[t1, t2]`` in seconds relative
                to ``timezero``.  Auto-selected when ``None``.
            rbkg: Right background interval as ``[t1, t2]`` in seconds relative
                to ``timezero``.  Auto-selected when ``None``.
            savepath: Directory where duration products are written.  Created
                if it does not already exist.
            std: When ``True``, print the stdout and stderr of each HEASOFT
                subprocess to the console.
        """

        savepath = os.path.abspath(savepath)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        self._extract_curve(std=std)

        txx = ggTxx(self.lc_net_cts, self.lc_net_cts_err, self.lc_bins)
        txx.find_pulse(p0=self.gs_p0, sigma=self.gs_sigma, mp=mp)
        txx.calculate(xx=xx, pstart=pstart, pstop=pstop, lbkg=lbkg, rbkg=rbkg)
        txx.save(savepath=savepath)


    def extract_rebin_curve(self, trange=None, min_sigma=None, min_evt=None, max_bin=None,
                            savepath='./curve', loglog=False, std=False, show=False):
        """Extract a rebinned light curve with adaptive bin sizes and save plots.

        Applies the ``rebin`` utility (``gstat`` statistic) to merge bins in
        the extracted light curve so that each merged bin meets at least one of
        the supplied significance or count thresholds.  Saves an interactive
        HTML plot and its JSON equivalent.

        Args:
            trange: Two-element list ``[t1, t2]`` in seconds relative to
                ``timezero`` restricting which bins are rebinned.  All bins are
                used when ``None``.
            min_sigma: Minimum signal-to-noise ratio per rebinned bin.  Ignored
                when ``None``.
            min_evt: Minimum number of net counts per rebinned bin.  Ignored
                when ``None``.
            max_bin: Maximum number of original bins to merge into one.  Ignored
                when ``None``.
            savepath: Directory where output files are written.  Created if it
                does not already exist.
            loglog: When ``True``, render both axes of the output plot on a
                logarithmic scale.
            std: When ``True``, print the stdout and stderr of each HEASOFT
                subprocess to the console.
            show: When ``True``, display the Plotly figure interactively in
                the browser.
        """

        savepath = os.path.abspath(savepath)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        self._extract_curve(std=std)

        if trange is not None:
            idx = (self.lc_bin_list[:, 0] >= trange[0]) * (self.lc_bin_list[:, 1] <= trange[1])
        else:
            idx = np.ones(len(self.lc_bin_list), dtype=bool)

        self.lc_rebin_list, self.lc_net_rects, self.lc_net_rects_err, _, _ = \
            rebin(
                self.lc_bin_list[idx],
                'gstat',
                self.lc_net_cts[idx],
                cts_err=self.lc_net_cts_err[idx],
                bcts=None,
                bcts_err=None,
                min_sigma=min_sigma,
                min_evt=min_evt,
                max_bin=max_bin,
                backscale=1)

        self.lc_retime = np.mean(self.lc_rebin_list, axis=1)
        self.lc_rebinsize = self.lc_rebin_list[:, 1] - self.lc_rebin_list[:, 0]
        self.lc_net_rerate = self.lc_net_rects / self.lc_rebinsize
        self.lc_net_rerate_err = self.lc_net_rects_err / self.lc_rebinsize

        fig = go.Figure()
        net = go.Scatter(x=self.lc_retime,
                         y=self.lc_net_rerate,
                         mode='lines+markers',
                         name='net lightcurve',
                         showlegend=True,
                         error_y=dict(
                             type='data',
                             array=self.lc_net_rerate_err,
                             thickness=1.5,
                             width=0),
                         marker=dict(symbol='cross-thin', size=0))
        fig.add_trace(net)

        if loglog:
            fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)', type='log')
            fig.update_yaxes(title_text='Counts per second', type='log')
        else:
            fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)')
            fig.update_yaxes(title_text='Counts per second')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        if show: fig.show()
        fig.write_html(savepath + '/rebin_lc.html', include_plotlyjs='cdn')
        fig.write_image(savepath + '/rebin_lc.pdf')


    @property
    def spec_slices(self):
        """Return the list of time intervals used for spectral extraction.

        Each element is a two-element list ``[t1, t2]`` in seconds relative to
        ``timezero``.  Defaults to ``[time_filter]`` (a single interval
        spanning the full active time range) when no slices have been set.

        Returns:
            List of ``[t1, t2]`` pairs in seconds relative to ``timezero``.
        """

        try:
            return self._spec_slices
        except AttributeError:
            return [self.time_filter]


    @spec_slices.setter
    def spec_slices(self, new_spec_slice):

        if isinstance(new_spec_slice, list):
            if isinstance(new_spec_slice[0], list):
                self._spec_slices = new_spec_slice
            else:
                raise ValueError('not expected spec_slices type')
        else:
            raise ValueError('not expected spec_slices type')


    def batbinevt_spectrum(self, std=False):
        """Extract a PHA spectrum for the current slice and apply corrections.

        Runs three HEASOFT tools in sequence:

        1. ``batbinevt`` — bins events into a 80-channel CALDB PHA file.
        2. ``batphasyserr`` — adds CALDB systematic errors to the spectrum.
        3. ``batupdatephakw`` — updates header keywords from the auxiliary file.

        The output file is written to ``self.specfile``, which must be set
        before calling this method (``extract_spectrum`` does this
        automatically).

        Args:
            std: When ``True``, print the stdout and stderr of each HEASOFT
                subprocess to the console.
        """

        commands = ['batbinevt',
                    f'detmask={self.maskfile}',
                    f'tstart={self.spec_tstart}',
                    f'tstop={self.spec_tstop}',
                    f'infile={self.ufevtfile}',
                    f'outfile={self.specfile}',
                    'outtype=pha',
                    'timedel=0',
                    'timebinalg=uniform',
                    'energybins=CALDB:80']

        stdout, stderr = self._run_comands(commands)

        if std:
            print(stdout)
            print(stderr)

        commands = ['batphasyserr',
                    f'infile={self.specfile}',
                    'syserrfile=CALDB']

        stdout, stderr = self._run_comands(commands)

        if std:
            print(stdout)
            print(stderr)

        commands = ['batupdatephakw',
                    f'infile={self.specfile}',
                    f'auxfile={self.auxfile}']

        stdout, stderr = self._run_comands(commands)

        if std:
            print(stdout)
            print(stderr)


    def batdrmgen(self, std=False):
        """Generate a detector response matrix for the current spectrum.

        Runs ``batdrmgen`` using ``self.specfile`` as input and writes the
        response to ``self.respfile``.  Both attributes must be set before
        calling this method (``extract_response`` does this automatically).

        Args:
            std: When ``True``, print the stdout and stderr of the HEASOFT
                subprocess to the console.
        """

        commands = ['batdrmgen',
                    f'infile={self.specfile}',
                    f'outfile={self.respfile}']

        stdout, stderr = self._run_comands(commands)

        if std:
            print(stdout)
            print(stderr)


    def extract_spectrum(self, savepath='./spectrum', std=False):
        """Extract PHA spectra for all time slices in ``spec_slices``.

        Iterates over every ``[t1, t2]`` pair in ``spec_slices``, sets
        ``specfile`` to a filename derived from the slice boundaries, and calls
        ``batbinevt_spectrum``.  Any pre-existing PHA file for a given slice is
        removed before extraction.

        Args:
            savepath: Directory where PHA files are written.  Created if it
                does not already exist.
            std: When ``True``, print the stdout and stderr of each HEASOFT
                subprocess to the console.
        """

        savepath = os.path.abspath(savepath)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        lslices = np.array(self.spec_slices)[:, 0]
        rslices = np.array(self.spec_slices)[:, 1]

        for l, r in zip(lslices, rslices):

            self.spec_tstart = self.timezero + l
            self.spec_tstop = self.timezero + r

            new_l = '{:+.2f}'.format(l).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_r = '{:+.2f}'.format(r).replace('-', 'm').replace('.', 'd').replace('+', 'p')

            file_name = '_'.join([new_l, new_r])

            self.specfile = savepath + f'/{file_name}.pha'

            if os.path.exists(self.specfile):
                os.remove(self.specfile)

            self.batbinevt_spectrum(std=std)


    def extract_response(self, savepath='./spectrum', std=False):
        """Generate detector response matrices for all time slices.

        Iterates over every ``[t1, t2]`` pair in ``spec_slices``.  If the
        corresponding PHA file does not already exist it is created first via
        ``batbinevt_spectrum``.  Any pre-existing response file is removed
        before ``batdrmgen`` is called.

        Args:
            savepath: Directory where RSP files are written.  Created if it
                does not already exist.
            std: When ``True``, print the stdout and stderr of each HEASOFT
                subprocess to the console.
        """

        savepath = os.path.abspath(savepath)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        lslices = np.array(self.spec_slices)[:, 0]
        rslices = np.array(self.spec_slices)[:, 1]

        for l, r in zip(lslices, rslices):

            self.spec_tstart = self.timezero + l
            self.spec_tstop = self.timezero + r

            new_l = '{:+.2f}'.format(l).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_r = '{:+.2f}'.format(r).replace('-', 'm').replace('.', 'd').replace('+', 'p')

            file_name = '_'.join([new_l, new_r])

            self.specfile = savepath + f'/{file_name}.pha'

            if not os.path.exists(self.specfile):
                self.batbinevt_spectrum(std=std)

            self.respfile = savepath + f'/{file_name}.rsp'

            if os.path.exists(self.respfile):
                os.remove(self.respfile)

            self.batdrmgen(std=std)
