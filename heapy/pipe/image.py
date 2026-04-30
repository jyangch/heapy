"""Read and process X-ray imaging event FITS files from EP and Swift/XRT.

Wraps HEASOFT tools (``xselect``, ``ximage``, ``xrtlccorr``, ``xrtmkarf``,
``xrtexpomap``, ``quzcif``) via subprocess to extract source and background
light curves, images, and spectra from detector-image event files.  The
module exposes a generic ``Image`` base class and two instrument-specific
subclasses: ``epImage`` (Einstein Probe WXT/FXT) and ``swiftImage``
(Swift/XRT).

Typical usage:
    from heapy.pipe.image import epImage
    img = epImage(file='ep_evt.fits', regfile='src.reg', bkregfile='bkg.reg')
    img.lc_t1t2 = [0, 1000]
    img.extract_curve(savepath='./curve')
"""

import os
import shutil
import subprocess

from astropy import table
from astropy.io import fits
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from ..auto.signal import ppSignal
from ..data.retrieve import epRetrieve, swiftRetrieve
from ..temp.txx import ppTxx
from ..util.data import rebin, union
from ..util.file import copy_file
from ..util.time import ep_met_to_utc, ep_utc_to_met, swift_met_to_utc, swift_utc_to_met
from ..util.tools import json_dump
from .filter import Filter

docs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/docs'


class Image:
    """Base class for processing X-ray imaging-mode event FITS files.

    Orchestrates HEASOFT command-line tool execution to separate source
    and background event lists using spatial region files, then provides
    light curve extraction, PSF pile-up diagnostics, and spectrum
    extraction.  Dead-time and vignetting corrections are delegated to
    instrument-specific subclasses.

    Attributes:
        prefix: File-prefix string prepended to output directories;
            defaults to ``''``.
        src_curvefile: Absolute path to the extracted source light curve.
        bkg_curvefile: Absolute path to the extracted background light curve.
        src_specfile: Absolute path to the extracted source spectrum.
        bkg_specfile: Absolute path to the extracted background spectrum.
        src_evtfile: Absolute path to the extracted source event file.
        bkg_evtfile: Absolute path to the extracted background event file.
        src_backscale: BACKSCAL keyword from the source spectrum header.
        bkg_backscale: BACKSCAL keyword from the background spectrum header.
        backscale: Ratio ``src_backscale / bkg_backscale`` for background
            scaling.
    """

    os.environ['HEADASNOQUERY'] = '1'

    def __init__(self, file, regfile, bkregfile):
        """Initialize Image, run xselect to separate source and background events.

        Args:
            file: Path to the input event FITS file.
            regfile: Path to the source spatial region file (DS9 format).
            bkregfile: Path to the background spatial region file (DS9 format).
        """

        self._file = file
        self._regfile = regfile
        self._bkregfile = bkregfile

        self.prefix = ''

        self._ini_xselect()

    @property
    def file(self):
        """Return the absolute path to the event FITS file."""

        return os.path.abspath(self._file)

    @file.setter
    def file(self, new_file):

        self._file = new_file

        self._ini_xselect()

    @property
    def regfile(self):
        """Return the absolute path to the source region file."""

        return os.path.abspath(self._regfile)

    @regfile.setter
    def regfile(self, new_regfile):

        self._regfile = new_regfile

        self._ini_xselect()

    @property
    def bkregfile(self):
        """Return the absolute path to the background region file."""

        return os.path.abspath(self._bkregfile)

    @bkregfile.setter
    def bkregfile(self, new_bkregfile):

        self._bkregfile = new_bkregfile

        self._ini_xselect()

    @staticmethod
    def _run_xselect(commands):

        process = subprocess.Popen(
            'xselect',
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = process.communicate(input='\n'.join(commands) + '\nexit\nno\n')

        return stdout, stderr

    @staticmethod
    def _run_ximage(commands):

        process = subprocess.Popen(
            'ximage',
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = process.communicate(input='\n'.join(commands))

        return stdout, stderr

    def _run_commands(self, commands):

        process = subprocess.Popen(
            commands,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = process.communicate()

        return stdout, stderr

    def _ini_xselect(self):

        curve_savepath = os.path.dirname(self.file) + f'/{self.prefix}curve'

        self.src_curvefile = curve_savepath + '/src.lc'
        self.bkg_curvefile = curve_savepath + '/bkg.lc'

        spectra_savepath = os.path.dirname(self.file) + f'/{self.prefix}spectra'

        self.src_specfile = spectra_savepath + '/src.pi'
        self.bkg_specfile = spectra_savepath + '/bkg.pi'

        events_savepath = os.path.dirname(self.file) + f'/{self.prefix}events'

        self.src_evtfile = events_savepath + '/src.evt'
        self.bkg_evtfile = events_savepath + '/bkg.evt'

        if os.path.exists(curve_savepath):
            shutil.rmtree(curve_savepath)
        os.makedirs(curve_savepath)

        if os.path.exists(spectra_savepath):
            shutil.rmtree(spectra_savepath)
        os.makedirs(spectra_savepath)

        if os.path.exists(events_savepath):
            shutil.rmtree(events_savepath)
        os.makedirs(events_savepath)

        commands = [
            'xsel',
            'read events',
            os.path.dirname(self.file),
            self.file.split('/')[-1],
            'yes',
            f'filter region {self.regfile}',
            'extract curve',
            f'save curve {self.src_curvefile}',
            'extract spectrum',
            f'save spectrum {self.src_specfile}',
            'extract events',
            f'save events {self.src_evtfile}',
            'no',
            'clear events',
            'clear region',
            f'filter region {self.bkregfile}',
            'extract curve',
            f'save curve {self.bkg_curvefile}',
            'extract spectrum',
            f'save spectrum {self.bkg_specfile}',
            'extract events',
            f'save events {self.bkg_evtfile}',
            'no',
        ]

        _, _ = self._run_xselect(commands)

        src_hdu = fits.open(self.src_specfile)
        self.src_backscale = src_hdu['SPECTRUM'].header['BACKSCAL']
        src_hdu.close()

        bkg_hdu = fits.open(self.bkg_specfile)
        self.bkg_backscale = bkg_hdu['SPECTRUM'].header['BACKSCAL']
        bkg_hdu.close()

        self.backscale = self.src_backscale / self.bkg_backscale

        hdu = fits.open(self.file)
        self._event = table.Table.read(hdu['EVENTS'])
        self._gti = table.Table.read(hdu['GTI'])
        self._timezero = hdu['EVENTS'].header['TSTART']
        self._filter = Filter(self._event)
        hdu.close()

        src_hdu = fits.open(self.src_evtfile)
        self._src_event = table.Table.read(src_hdu['EVENTS'])
        self._src_filter = Filter(self._src_event)
        src_hdu.close()

        bkg_hdu = fits.open(self.bkg_evtfile)
        self._bkg_event = table.Table.read(bkg_hdu['EVENTS'])
        self._bkg_filter = Filter(self._bkg_event)
        bkg_hdu.close()

        self._filter_info = {'time': None, 'pi': None, 'tag': None}

    @property
    def event(self):
        """Return the filtered full-field event table."""

        return self._filter.evt

    @property
    def src_event(self):
        """Return the filtered source-region event table."""

        return self._src_filter.evt

    @property
    def bkg_event(self):
        """Return the filtered background-region event table."""

        return self._bkg_filter.evt

    @property
    def gti(self):
        """Return merged good-time intervals as an (N, 2) array in seconds.

        Values are expressed relative to ``timezero``.
        """

        tstart = self._gti['START'] - self.timezero
        tstop = self._gti['STOP'] - self.timezero

        return np.array(union(np.vstack((tstart, tstop)).T))

    @property
    def timezero(self):
        """Return the reference time (TSTART from the EVENTS header)."""

        return self._timezero

    @timezero.setter
    def timezero(self, new_timezero):

        self._timezero = new_timezero

    @property
    def timezero_utc(self):
        """Return the reference time as a UTC string, or ``None`` if unavailable."""

        return None

    @property
    def psf_modelfile(self):
        """Return the ximage PSF model filename, or ``None`` if not defined."""

        return None

    @property
    def psf_fitradius(self):
        """Return the PSF fit radius in arcseconds, or ``None`` if not defined."""

        return None

    def slice_time(self, t1t2):
        """Trim all event tables in-place to a relative time window.

        Permanently discards events outside ``[t1, t2]`` from the full,
        source, and background event tables and resets all active filters.

        Args:
            t1t2: Two-element sequence ``[t1, t2]`` in seconds relative to
                ``timezero``.
        """

        t1, t2 = t1t2

        met_t1 = self.timezero + t1
        met_t2 = self.timezero + t2

        met_ts = self._event['TIME']
        flt = (met_ts >= met_t1) & (met_ts <= met_t2)
        self._event = self._event[flt]

        src_met_ts = self._src_event['TIME']
        flt = (src_met_ts >= met_t1) & (src_met_ts <= met_t2)
        self._src_event = self._src_event[flt]

        bkg_met_ts = self._bkg_event['TIME']
        flt = (bkg_met_ts >= met_t1) & (bkg_met_ts <= met_t2)
        self._bkg_event = self._bkg_event[flt]

        self._clear_filter()

    def filter_time(self, t1t2):
        """Apply a reversible time filter to all event tables.

        Passing ``None`` removes the current time filter.

        Args:
            t1t2: Two-element list ``[t1, t2]`` in seconds relative to
                ``timezero``, or ``None`` to clear the filter.

        Raises:
            ValueError: If ``t1t2`` is neither a list nor ``None``.
        """

        if t1t2 is None:
            expr = None

        elif isinstance(t1t2, list):
            t1, t2 = t1t2

            met_t1 = self.timezero + t1
            met_t2 = self.timezero + t2

            expr = f'(TIME >= {met_t1}) * (TIME <= {met_t2})'

        else:
            raise ValueError('t1t2 is extected to be list or None')

        self._time_filter = t1t2

        self._filter_info['time'] = expr

        self._filter_update()

    def filter_pi(self, p1p2):
        """Apply a reversible PI channel range filter to all event tables.

        Args:
            p1p2: Two-element list ``[p1, p2]`` of integer PI channel
                bounds (inclusive), or ``None`` to clear the filter.

        Raises:
            ValueError: If ``p1p2`` is neither a list nor ``None``.
        """

        if p1p2 is None:
            expr = None

        elif isinstance(p1p2, list):
            p1, p2 = p1p2
            expr = f'(PI >= {p1}) * (PI <= {p2})'

        else:
            raise ValueError('p1p2 is extected to be list or None')

        self._pi_filter = p1p2

        self._filter_info['pi'] = expr

        self._filter_update()

    def filter(self, expr):
        """Apply an arbitrary boolean expression filter to all event tables.

        Args:
            expr: Boolean expression string compatible with
                ``astropy.table.Table`` evaluation, or ``None`` to clear.
        """

        self._filter_info['tag'] = expr

        self._filter_update()

    def _filter_update(self):

        self._clear_filter()

        self._filter.eval(self._filter_info['time'])
        self._filter.eval(self._filter_info['pi'])
        self._filter.eval(self._filter_info['tag'])

        self._src_filter.eval(self._filter_info['time'])
        self._src_filter.eval(self._filter_info['pi'])
        self._src_filter.eval(self._filter_info['tag'])

        self._bkg_filter.eval(self._filter_info['time'])
        self._bkg_filter.eval(self._filter_info['pi'])
        self._bkg_filter.eval(self._filter_info['tag'])

    def _clear_filter(self):

        self._filter.clear()
        self._src_filter.clear()
        self._bkg_filter.clear()

    @property
    def filter_info(self):
        """Return the current filter expression dictionary."""

        return self._filter_info

    @property
    def time_filter(self):
        """Return the active time-filter window ``[t1, t2]`` in seconds.

        When no time filter is set, returns the floored/ceiled span of the
        full event table relative to ``timezero``.
        """

        if self._filter_info['time'] is None:
            return [
                np.floor(np.min(self._event['TIME'])) - self.timezero,
                np.ceil(np.max(self._event['TIME'])) - self.timezero,
            ]
        else:
            return self._time_filter

    @property
    def pi_filter(self):
        """Return the active PI filter window ``[p1, p2]``.

        When no PI filter is set, returns the full PI range of the event table.
        """

        if self._filter_info['pi'] is None:
            return [np.min(self._event['PI']), np.max(self._event['PI'])]
        else:
            return self._pi_filter

    def extract_image(self, savepath='./image', show=False, std=False):
        """Extract a detector image and write it as an HTML heatmap and FITS file.

        Calls xselect to produce the image FITS file within the active
        time and PI filters, then generates a log-scaled 2-D histogram
        from the filtered event table and saves an interactive HTML plot.

        Args:
            savepath: Directory where image files are written.
            show: If ``True``, display the plot interactively in the browser.
            std: If ``True``, print xselect stdout and stderr for debugging.
        """

        savepath = os.path.abspath(savepath)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        self.imagefile = savepath + '/image.img'

        if os.path.exists(self.imagefile):
            os.remove(self.imagefile)

        scc_start = self.timezero + self.time_filter[0]
        scc_stop = self.timezero + self.time_filter[1]

        pi_start = self.pi_filter[0]
        pi_stop = self.pi_filter[1]

        commands = [
            'xsel',
            'read events',
            os.path.dirname(self.file),
            self.file.split('/')[-1],
            'yes',
            'filter time scc',
            f'{scc_start}, {scc_stop}',
            'x',
            f'filter pha_cutoff {pi_start} {pi_stop}',
            'extract image',
            f'save image {self.imagefile}',
        ]

        stdout, stderr = self._run_xselect(commands)

        if std:
            print(stdout)
            print(stderr)

        H, xedges, yedges = np.histogram2d(self.event['X'], self.event['Y'], bins=128)
        H[H == 0] = 1

        fig = go.Figure()
        image = go.Heatmap(x=xedges, y=yedges, z=np.log10(H.T), colorscale='Jet')
        fig.add_trace(image)

        fig.update_layout(template='plotly_white', height=700, width=700)

        if show:
            fig.show()
        fig.write_html(savepath + '/image.html', include_plotlyjs='cdn')
        fig.write_image(savepath + '/image.pdf')

    @property
    def src_ts(self):
        """Return filtered source-region event times in seconds relative to ``timezero``."""

        return np.array(self.src_event['TIME']) - self.timezero

    @property
    def bkg_ts(self):
        """Return filtered background-region event times in seconds relative to ``timezero``."""

        return np.array(self.bkg_event['TIME']) - self.timezero

    @property
    def lc_t1t2(self):
        """Return the light curve time window ``[t1, t2]`` in seconds.

        Falls back to ``time_filter`` when ``_lc_t1t2`` has not been set
        or is ``None``.
        """

        try:
            _ = self._lc_t1t2
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
        """Return the total duration of the light curve window in seconds."""

        return self.lc_t1t2[1] - self.lc_t1t2[0]

    @property
    def lc_binsize(self):
        """Return the light curve bin width in seconds.

        Defaults to ``lc_interval / 300`` when not explicitly set.
        """

        try:
            _ = self._lc_binsize
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
    def lc_src_ts(self):
        """Return source-region event times within the light curve window in seconds."""

        idx = (self.src_ts >= self.lc_t1t2[0]) & (self.src_ts <= self.lc_t1t2[1])

        return self.src_ts[idx]

    @property
    def lc_bkg_ts(self):
        """Return background-region event times within the light curve window in seconds."""

        idx = (self.bkg_ts >= self.lc_t1t2[0]) & (self.bkg_ts <= self.lc_t1t2[1])

        return self.bkg_ts[idx]

    @property
    def lc_bins(self):
        """Return the light curve bin edges array in seconds."""

        return np.arange(self.lc_t1t2[0], self.lc_t1t2[1] + 1e-5, self.lc_binsize)

    @property
    def lc_bin_list(self):
        """Return light curve bin boundaries as an (N, 2) array in seconds."""

        lbins, rbins = self.lc_bins[:-1], self.lc_bins[1:]

        return np.vstack((lbins, rbins)).T

    @property
    def lc_mask(self):
        """Return a boolean mask selecting bins that fall entirely within a GTI."""

        return np.any(
            (self.lc_bin_list[:, None, 0] >= self.gti[:, 0])
            & (self.lc_bin_list[:, None, 1] <= self.gti[:, 1]),
            axis=1,
        )

    @property
    def lc_mask_bin_list(self):
        """Return light curve bins restricted to GTI-covered intervals."""

        return self.lc_bin_list[self.lc_mask]

    @property
    def lc_exps(self):
        """Return the exposure per light curve bin in seconds.

        Subclasses may override this to apply vignetting or dead-time
        corrections.  The base implementation returns ``lc_binsize`` for
        all bins (uniform exposure, no correction).
        """

        return self.lc_binsize

    @property
    def lc_mask_exps(self):
        """Return GTI-masked light curve exposures in seconds."""

        return self.lc_exps[self.lc_mask]

    @property
    def lc_time(self):
        """Return light curve bin center times in seconds."""

        return np.mean(self.lc_bin_list, axis=1)

    @property
    def lc_mask_time(self):
        """Return GTI-masked light curve bin center times in seconds."""

        return self.lc_time[self.lc_mask]

    @property
    def lc_time_err(self):
        """Return half-widths of light curve bins in seconds."""

        lbins, rbins = self.lc_bins[:-1], self.lc_bins[1:]

        return (rbins - lbins) / 2

    @property
    def lc_mask_time_err(self):
        """Return GTI-masked light curve bin half-widths in seconds."""

        return self.lc_time_err[self.lc_mask]

    @property
    def lc_src_cts(self):
        """Return source counts per light curve bin."""

        return np.histogram(self.lc_src_ts, bins=self.lc_bins)[0]

    @property
    def lc_mask_src_cts(self):
        """Return GTI-masked source counts per light curve bin."""

        return self.lc_src_cts[self.lc_mask]

    @property
    def lc_src_cts_err(self):
        """Return Poisson 1-sigma uncertainties on source counts per bin."""

        return np.sqrt(self.lc_src_cts)

    @property
    def lc_mask_src_cts_err(self):
        """Return GTI-masked Poisson 1-sigma uncertainties on source counts."""

        return self.lc_src_cts_err[self.lc_mask]

    @property
    def lc_src_rate(self):
        """Return source count rate per bin in counts per second."""

        return self.lc_src_cts / self.lc_exps

    @property
    def lc_mask_src_rate(self):
        """Return GTI-masked source count rate in counts per second."""

        return self.lc_src_rate[self.lc_mask]

    @property
    def lc_src_rate_err(self):
        """Return 1-sigma uncertainty on source count rate in counts per second."""

        return self.lc_src_cts_err / self.lc_exps

    @property
    def lc_mask_src_rate_err(self):
        """Return GTI-masked 1-sigma uncertainty on source count rate."""

        return self.lc_src_rate_err[self.lc_mask]

    @property
    def lc_bkg_cts(self):
        """Return background counts per light curve bin."""

        return np.histogram(self.lc_bkg_ts, bins=self.lc_bins)[0]

    @property
    def lc_mask_bkg_cts(self):
        """Return GTI-masked background counts per light curve bin."""

        return self.lc_bkg_cts[self.lc_mask]

    @property
    def lc_bkg_cts_err(self):
        """Return Poisson 1-sigma uncertainties on background counts per bin."""

        return np.sqrt(self.lc_bkg_cts)

    @property
    def lc_mask_bkg_cts_err(self):
        """Return GTI-masked Poisson 1-sigma uncertainties on background counts."""

        return self.lc_bkg_cts_err[self.lc_mask]

    @property
    def lc_bkg_rate(self):
        """Return background count rate scaled to the source region in counts per second."""

        return self.lc_bkg_cts / self.lc_exps * self.backscale

    @property
    def lc_mask_bkg_rate(self):
        """Return GTI-masked scaled background rate in counts per second."""

        return self.lc_bkg_rate[self.lc_mask]

    @property
    def lc_bkg_rate_err(self):
        """Return 1-sigma uncertainty on the scaled background rate in counts per second."""

        return self.lc_bkg_cts_err / self.lc_exps * self.backscale

    @property
    def lc_mask_bkg_rate_err(self):
        """Return GTI-masked 1-sigma uncertainty on scaled background rate."""

        return self.lc_bkg_rate_err[self.lc_mask]

    @property
    def lc_net_rate(self):
        """Return background-subtracted net count rate in counts per second."""

        return self.lc_src_rate - self.lc_bkg_rate

    @property
    def lc_mask_net_rate(self):
        """Return GTI-masked net count rate in counts per second."""

        return self.lc_net_rate[self.lc_mask]

    @property
    def lc_net_rate_err(self):
        """Return 1-sigma uncertainty on net count rate in counts per second."""

        return np.sqrt(self.lc_src_rate_err**2 + self.lc_bkg_rate_err**2)

    @property
    def lc_mask_net_rate_err(self):
        """Return GTI-masked 1-sigma uncertainty on net count rate."""

        return self.lc_net_rate_err[self.lc_mask]

    @property
    def lc_net_cts(self):
        """Return net (background-subtracted) counts per light curve bin."""

        return self.lc_net_rate * self.lc_exps

    @property
    def lc_mask_net_cts(self):
        """Return GTI-masked net counts per light curve bin."""

        return self.lc_net_cts[self.lc_mask]

    @property
    def lc_net_cts_err(self):
        """Return 1-sigma uncertainty on net counts per light curve bin."""

        return self.lc_net_rate_err * self.lc_exps

    @property
    def lc_mask_net_cts_err(self):
        """Return GTI-masked 1-sigma uncertainty on net counts per bin."""

        return self.lc_net_cts_err[self.lc_mask]

    @property
    def lc_net_ccts(self):
        """Return cumulative net counts over all light curve bins."""

        return np.cumsum(self.lc_net_cts)

    @property
    def ps_p0(self):
        """Return the signal-detection significance threshold p0 (default 0.05)."""

        try:
            _ = self._ps_p0
        except AttributeError:
            return 0.05
        else:
            return self._ps_p0

    @ps_p0.setter
    def ps_p0(self, new_ps_p0):

        if isinstance(new_ps_p0, (float, int)):
            self._ps_p0 = new_ps_p0
        else:
            raise ValueError('ps_p0 is extected to be float or int')

    @property
    def ps_sigma(self):
        """Return the signal-detection outlier threshold in sigma (default 3)."""

        try:
            _ = self._ps_sigma
        except AttributeError:
            return 3
        else:
            return self._ps_sigma

    @ps_sigma.setter
    def ps_sigma(self, new_ps_sigma):

        if isinstance(new_ps_sigma, (int, float)):
            self._ps_sigma = new_ps_sigma
        else:
            raise ValueError('ps_sigma is extected to be int or float')

    @property
    def lc_ps(self):
        """Compute and return a ``ppSignal`` object for the light curve.

        Fits a source+background model to the binned counts using
        ``ppSignal.loop`` and returns the result each time this property
        is accessed.
        """

        lc_ps = ppSignal(self.src_ts, self.bkg_ts, self.lc_bins, backscale=self.backscale)
        lc_ps.loop(p0=self.ps_p0, sigma=self.ps_sigma)

        return lc_ps

    def extract_curve(self, savepath='./curve', sig=True, show=False):
        """Extract source, background, and net light curves and save as HTML/JSON.

        When ``sig`` is enabled, runs ``ppSignal`` signal detection and
        saves results under ``savepath/ppsignal``.  Produces a three-panel
        rate light curve and a cumulative net-counts plot.

        Args:
            savepath: Directory where output files are written.
            sig: If ``True``, run signal detection and save the
                ``ppSignal`` output.
            show: If ``True``, display plots interactively in the browser.
        """

        savepath = os.path.abspath(savepath)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        if sig:
            self.lc_ps.save(savepath=savepath + '/ppsignal')

        fig = go.Figure()
        src = go.Scatter(
            x=self.lc_mask_time,
            y=self.lc_mask_src_rate,
            mode='markers',
            name='src counts rate',
            showlegend=True,
            error_x=dict(type='data', array=self.lc_mask_time_err, thickness=1.5, width=0),
            error_y=dict(type='data', array=self.lc_mask_src_rate_err, thickness=1.5, width=0),
            marker=dict(symbol='circle', size=3),
        )
        bkg = go.Scatter(
            x=self.lc_mask_time,
            y=self.lc_mask_bkg_rate,
            mode='markers',
            name='bkg counts rate',
            showlegend=True,
            error_x=dict(type='data', array=self.lc_mask_time_err, thickness=1.5, width=0),
            error_y=dict(type='data', array=self.lc_mask_bkg_rate_err, thickness=1.5, width=0),
            marker=dict(symbol='circle', size=3),
        )
        net = go.Scatter(
            x=self.lc_mask_time,
            y=self.lc_mask_net_rate,
            mode='markers',
            name='net counts rate',
            showlegend=True,
            error_x=dict(type='data', array=self.lc_mask_time_err, thickness=1.5, width=0),
            error_y=dict(type='data', array=self.lc_mask_net_rate_err, thickness=1.5, width=0),
            marker=dict(symbol='circle', size=3),
        )

        fig.add_trace(src)
        fig.add_trace(bkg)
        fig.add_trace(net)

        fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)', range=self.lc_t1t2)
        fig.update_yaxes(title_text=f'Counts per second (binsize={self.lc_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        if show:
            fig.show()
        fig.write_html(savepath + '/lc.html', include_plotlyjs='cdn')
        fig.write_image(savepath + '/lc.pdf')

        fig = go.Figure()
        net = go.Scatter(
            x=self.lc_time,
            y=self.lc_net_ccts,
            mode='lines',
            name='net cumulated counts',
            showlegend=True,
        )

        fig.add_trace(net)

        fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)', range=self.lc_t1t2)
        fig.update_yaxes(title_text=f'Cumulated counts (binsize={self.lc_binsize} s)')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        fig.write_html(savepath + '/cum_lc.html', include_plotlyjs='cdn')
        fig.write_image(savepath + '/cum_lc.pdf')

    def calculate_txx(
        self,
        mp=True,
        xx=0.9,
        pstart=None,
        pstop=None,
        lbkg=None,
        rbkg=None,
        savepath='./curve/duration',
    ):
        """Compute the Txx duration of a transient from the light curve.

        Uses ``ppTxx`` to detect the pulse via ``ppSignal`` and then
        calculates the interval enclosing fraction ``xx`` of the total
        net counts.

        Args:
            mp: If ``True``, use multi-peak pulse-finding mode.
            xx: Fraction of total counts to enclose; e.g. 0.9 for T90.
            pstart: Forced pulse start in seconds relative to ``timezero``,
                or ``None`` for automatic detection.
            pstop: Forced pulse stop in seconds relative to ``timezero``,
                or ``None`` for automatic detection.
            lbkg: Left background window ``[t1, t2]``, or ``None``.
            rbkg: Right background window ``[t1, t2]``, or ``None``.
            savepath: Directory where duration results are written.
        """

        savepath = os.path.abspath(savepath)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        txx = ppTxx(self.src_ts, self.bkg_ts, self.lc_bins, self.backscale)
        txx.find_pulse(p0=self.ps_p0, sigma=self.ps_sigma, mp=mp)
        txx.calculate(xx=xx, pstart=pstart, pstop=pstop, lbkg=lbkg, rbkg=rbkg)
        txx.save(savepath=savepath)

    def extract_rebin_curve(
        self,
        tranges=None,
        min_sigma=None,
        min_evt=None,
        max_bin=None,
        savepath='./curve',
        loglog=False,
        step=False,
        show=False,
    ):
        """Rebin the net light curve using cstat and save as HTML and JSON.

        Merges bins within each range in ``tranges`` until the significance
        criterion is met.  Background scaling is applied via ``backscale``.

        Args:
            tranges: List of ``[t1, t2]`` windows in seconds; a single
                ``[t1, t2]`` list is also accepted.  Defaults to the full
                light curve range.
            min_sigma: Minimum signal-to-noise ratio per merged bin, or
                ``None`` to skip.
            min_evt: Minimum source counts per merged bin, or ``None``.
            max_bin: Maximum number of raw bins to merge, or ``None``.
            savepath: Directory where output files are written.
            loglog: If ``True``, use logarithmic axes on both axes.
            step: If ``True``, overlay a step-function representation.
            show: If ``True``, display the plot interactively in the browser.

        Raises:
            ValueError: If ``tranges`` is not a list.
        """

        savepath = os.path.abspath(savepath)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        if tranges is not None:
            if isinstance(tranges, list) and not isinstance(tranges[0], list):
                tranges = [tranges]
            elif isinstance(tranges, list) and isinstance(tranges[0], list):
                tranges = tranges
            else:
                raise ValueError('trange is expected to be a list')
        else:
            tranges = [[np.min(self.lc_bin_list[:, 0]), np.max(self.lc_bin_list[:, 1])]]

        lc_rebin_list, lc_src_rects, lc_src_rects_err, lc_bkg_rebcts, lc_bkg_rebcts_err = (
            [],
            [],
            [],
            [],
            [],
        )

        for trange in tranges:
            idx = (self.lc_bin_list[:, 0] >= trange[0]) * (self.lc_bin_list[:, 1] <= trange[1])

            new_bins, new_cts, new_cts_err, new_bcts, new_bcts_err = rebin(
                self.lc_bin_list[idx],
                'cstat',
                self.lc_src_cts[idx],
                cts_err=self.lc_src_cts_err[idx],
                bcts=self.lc_bkg_cts[idx],
                bcts_err=self.lc_bkg_cts_err[idx],
                min_sigma=min_sigma,
                min_evt=min_evt,
                max_bin=max_bin,
                backscale=self.backscale,
            )

            lc_rebin_list.append(new_bins)
            lc_src_rects.append(new_cts)
            lc_src_rects_err.append(new_cts_err)
            lc_bkg_rebcts.append(new_bcts)
            lc_bkg_rebcts_err.append(new_bcts_err)

        self.lc_rebin_list = np.vstack(lc_rebin_list)
        self.lc_src_rects = np.hstack(lc_src_rects)
        self.lc_src_rects_err = np.hstack(lc_src_rects_err)
        self.lc_bkg_rebcts = np.hstack(lc_bkg_rebcts)
        self.lc_bkg_rebcts_err = np.hstack(lc_bkg_rebcts_err)

        self.lc_retime = np.mean(self.lc_rebin_list, axis=1)
        self.lc_rebinsize = self.lc_rebin_list[:, 1] - self.lc_rebin_list[:, 0]
        self.lc_retime_err = self.lc_rebinsize / 2
        self.lc_net_rects = self.lc_src_rects - self.lc_bkg_rebcts * self.backscale
        self.lc_net_rects_err = np.sqrt(
            self.lc_src_rects_err**2 + (self.lc_bkg_rebcts_err * self.backscale) ** 2
        )
        self.lc_net_rerate = self.lc_net_rects / self.lc_rebinsize
        self.lc_net_rerate_err = self.lc_net_rects_err / self.lc_rebinsize

        self.lc_retime_step = self.lc_rebin_list.flatten()
        self.lc_net_rerate_step = np.repeat(self.lc_net_rerate, 2)

        fig = go.Figure()
        net = go.Scatter(
            x=self.lc_retime,
            y=self.lc_net_rerate,
            mode='markers',
            name='net lightcurve',
            showlegend=True,
            error_x=dict(type='data', array=self.lc_retime_err, thickness=1.5, width=0),
            error_y=dict(type='data', array=self.lc_net_rerate_err, thickness=1.5, width=0),
            marker=dict(color='#636EFA', symbol='circle', size=3),
        )
        fig.add_trace(net)

        if step:
            net_step = go.Scatter(
                x=self.lc_retime_step,
                y=self.lc_net_rerate_step,
                mode='lines',
                showlegend=False,
                line=dict(width=1.5, color='#636EFA'),
            )
            fig.add_trace(net_step)

        if loglog:
            fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)', type='log')
            fig.update_yaxes(title_text='Counts per second', type='log')
        else:
            fig.update_xaxes(title_text=f'Time since {self.timezero_utc} (s)')
            fig.update_yaxes(title_text='Counts per second')
        fig.update_layout(template='plotly_white', height=600, width=800)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        if show:
            fig.show()
        fig.write_html(savepath + '/rebin_lc.html', include_plotlyjs='cdn')
        fig.write_image(savepath + '/rebin_lc.pdf')

    def check_pileup(self, std=False, show=False):
        """Diagnose pile-up by fitting a PSF model to the radial profile.

        Runs ximage to extract the encircled energy fraction (EEF) and PSF
        radial profile from the image, fits the instrument PSF model, and
        saves an interactive two-panel plot.  The ratio of the fitted
        normalisation to the nominal model indicates the degree of pile-up.

        Args:
            std: If ``True``, print ximage stdout and stderr for debugging.
            show: If ``True``, display the plot interactively in the browser.
        """

        psf_savepath = os.path.dirname(self.file) + '/psf'

        psf_savepath = os.path.abspath(psf_savepath)

        if not os.path.exists(psf_savepath):
            os.makedirs(psf_savepath)

        try:
            imagefile = self.imagefile
        except AttributeError:
            imagefile = self.file

        qdpfile = psf_savepath + '/psf.qdp'

        if os.path.exists(qdpfile):
            os.remove(qdpfile)

        modfile = psf_savepath + '/psf.mod'

        if os.path.exists(modfile):
            os.remove(modfile)

        commands = [
            f'read {imagefile}',
            'cpd /xtk',
            'disp',
            'back',
            'psf/cur',
            'col off 1 2 3 4 6',
            f'rescale x {self.psf_fitradius}',
            f'model {self.psf_modelfile}',
            '\n',
            'fit',
            'rescale',
            'plot',
            f'wdata {qdpfile[:-4]}',
            f'wmodel {modfile[:-4]}',
            'exit',
            'exit',
        ]

        work_path = os.getcwd()
        os.chdir(docs_path + '/psf_model')
        stdout, stderr = self._run_ximage(commands)
        if os.path.exists(docs_path + '/psf_model/psf.qdp'):
            os.remove(docs_path + '/psf_model/psf.qdp')
        os.chdir(work_path)

        if std:
            print(stdout)
            print(stderr)

        with open(qdpfile) as f:
            qdp_lines = f.readlines()

        qdp_data_lines = [
            line
            for line in qdp_lines
            if line.strip()
            and not line.strip().startswith('!')
            and not line.strip().startswith('@')
            and not line.strip().upper().startswith('READ')
        ]
        qdp_data = np.loadtxt(qdp_data_lines)

        with open(modfile) as f:
            mod_lines = f.readlines()

        mod_param = float(mod_lines[1].split()[0])

        radius_arr = np.logspace(np.log10(qdp_data[0, 0]), np.log10(qdp_data[-1, 0]), 100)
        psf_arr = self._psf_model(radius_arr, mod_param)

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.5, 0.5],
            shared_xaxes=True,
            horizontal_spacing=0,
            vertical_spacing=0.05,
        )

        eef = go.Scatter(
            x=qdp_data[:, 0],
            y=qdp_data[:, 4],
            mode='markers',
            name='Encircled energy fraction',
            showlegend=True,
            error_x=dict(type='data', array=qdp_data[:, 1], thickness=1.5, width=0),
            error_y=dict(type='data', array=qdp_data[:, 5], thickness=1.5, width=0),
            marker=dict(symbol='circle', size=3),
        )
        eef_nominal_model = go.Scatter(
            x=np.append(qdp_data[:, 0] - qdp_data[:, 1], qdp_data[-1, 0] + qdp_data[-1, 1]),
            y=np.append(qdp_data[:, 6], qdp_data[-1, 6]),
            mode='lines',
            name='EEF nominal model',
            line_shape='hv',
            showlegend=True,
        )
        psf = go.Scatter(
            x=qdp_data[:, 0],
            y=qdp_data[:, 7],
            mode='markers',
            name='Point spread function',
            showlegend=True,
            error_x=dict(type='data', array=qdp_data[:, 1], thickness=1.5, width=0),
            error_y=dict(type='data', array=qdp_data[:, 8], thickness=1.5, width=0),
            marker=dict(symbol='circle', size=3),
        )
        psf_nominal_model = go.Scatter(
            x=np.append(qdp_data[:, 0] - qdp_data[:, 1], qdp_data[-1, 0] + qdp_data[-1, 1]),
            y=np.append(qdp_data[:, 9], qdp_data[-1, 9]),
            mode='lines',
            name='PSF nominal model',
            line_shape='hv',
            showlegend=True,
        )
        psf_best_model = go.Scatter(
            x=radius_arr,
            y=psf_arr,
            name='PSF best model',
            showlegend=True,
            mode='lines',
            line=dict(width=2),
        )

        fig.add_trace(eef, row=1, col=1)
        fig.add_trace(eef_nominal_model, row=1, col=1)
        fig.add_trace(psf, row=2, col=1)
        fig.add_trace(psf_nominal_model, row=2, col=1)
        fig.add_trace(psf_best_model, row=2, col=1)

        fig.update_xaxes(title_text='', row=1, col=1, type='log')
        fig.update_xaxes(title_text='Radius (arcsec)', row=2, col=1, type='log')
        fig.update_yaxes(title_text='Encircled energy fraction', row=1, col=1, type='log')
        fig.update_yaxes(title_text='PSF (ct/sq.arcsec/s)', row=2, col=1, type='log')
        fig.update_layout(template='plotly_white', height=700, width=700)
        fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='bottom'))

        if show:
            fig.show()
        fig.write_html(psf_savepath + '/psf.html', include_plotlyjs='cdn')
        fig.write_image(psf_savepath + '/psf.pdf')

    @property
    def spec_slices(self):
        """Return the list of ``[t1, t2]`` spectral time slices in seconds.

        Defaults to ``[time_filter]`` when ``spec_slices`` has not been set.
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

    def extract_spectrum(self, savepath='./spectrum', std=False):
        """Extract source and background spectra for each spectral slice.

        Calls xselect for each slice in ``spec_slices`` to produce
        time-filtered ``.src`` and ``.bkg`` spectrum files.  Also writes
        ``timezero.json`` and ``spec_slices.json`` for downstream
        reference.

        Args:
            savepath: Directory where spectrum files are written.
            std: If ``True``, print xselect stdout and stderr for debugging.
        """

        savepath = os.path.abspath(savepath)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        json_dump(self.timezero, savepath + '/timezero.json')

        json_dump(self.spec_slices, savepath + '/spec_slices.json')

        lslices = np.array(self.spec_slices)[:, 0]
        rslices = np.array(self.spec_slices)[:, 1]

        for left, right in zip(lslices, rslices, strict=False):
            scc_start = self.timezero + left
            scc_stop = self.timezero + right

            new_left = f'{left:+.2f}'.replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_right = f'{right:+.2f}'.replace('-', 'm').replace('.', 'd').replace('+', 'p')

            file_name = '_'.join([new_left, new_right])

            src_specfile = savepath + f'/{file_name}.src'
            bkg_specfile = savepath + f'/{file_name}.bkg'

            if os.path.exists(src_specfile):
                os.remove(src_specfile)

            if os.path.exists(bkg_specfile):
                os.remove(bkg_specfile)

            commands = [
                'xsel',
                'read events',
                os.path.dirname(self.file),
                self.file.split('/')[-1],
                'yes',
                'filter time scc',
                f'{scc_start}, {scc_stop}',
                'x',
                f'filter region {self.regfile}',
                'extract spectrum',
                f'save spectrum {src_specfile}',
                'clear region',
                f'filter region {self.bkregfile}',
                'extract spectrum',
                f'save spectrum {bkg_specfile}',
            ]

            stdout, stderr = self._run_xselect(commands)

            if std:
                print(stdout)
                print(stderr)


class epImage(Image):
    """Read and process Einstein Probe (EP) WXT or FXT event FITS files.

    Extends ``Image`` with EP-specific time-system conversions, an ARM
    region for background exclusion in WXT data, and PSF pile-up
    diagnostics using the FXT King+Gaussian PSF model.  RMF/ARF response
    files are copied directly from the calibration database.
    """

    def __init__(
        self, file, regfile, bkregfile, armregfile=None, arm=False, rmffile=None, arffile=None
    ):
        """Initialize epImage and run xselect to separate source and background events.

        Args:
            file: Path to the EP event FITS file.
            regfile: Path to the source region file (DS9 format).
            bkregfile: Path to the background region file (DS9 format).
            armregfile: Path to an optional ARM exclusion region file, or
                ``None``.
            arm: If ``True``, combine ``bkregfile`` and ``armregfile``
                into a composite background region.
            rmffile: Path to the redistribution matrix file (RMF), or
                ``None``.
            arffile: Path to the ancillary response file (ARF), or
                ``None``.
        """

        self._file = file
        self._regfile = regfile
        self._bkregfile = bkregfile
        self._armregfile = armregfile
        self._arm = arm
        self._rmffile = rmffile
        self._arffile = arffile

        self.prefix = ''

        self._ini_xselect()

    @classmethod
    def from_wxtobs(cls, obsid, srcid, datapath=None):
        """Retrieve EP WXT observation data and construct an epImage instance.

        Fetches event, region, ARM region, RMF, and ARF files for the
        specified WXT observation and source.

        Args:
            obsid: EP observation ID string.
            srcid: Source ID within the observation.
            datapath: Local directory containing the observation data, or
                ``None`` to use the default retrieval path.
        """

        rtv = epRetrieve.from_wxtobs(obsid, srcid, datapath)

        file = rtv.rtv_res['evt']
        regfile = rtv.rtv_res['reg']
        bkregfile = rtv.rtv_res['bkreg']
        armregfile = rtv.rtv_res['armreg']
        rmffile = rtv.rtv_res['rmf']
        arffile = rtv.rtv_res['arf']

        return cls(file, regfile, bkregfile, armregfile, False, rmffile, arffile)

    @classmethod
    def from_fxtobs(cls, obsid, module, datapath=None):
        """Retrieve EP FXT observation data and construct an epImage instance.

        Fetches event, region, RMF, and ARF files for the specified FXT
        observation and detector module.

        Args:
            obsid: EP observation ID string.
            module: FXT detector module identifier (e.g. ``'FXT-A'``).
            datapath: Local directory containing the observation data, or
                ``None`` to use the default retrieval path.
        """

        rtv = epRetrieve.from_fxtobs(obsid, module, datapath)

        file = rtv.rtv_res['evt']
        regfile = rtv.rtv_res['reg']
        bkregfile = rtv.rtv_res['bkreg']
        rmffile = rtv.rtv_res['rmf']
        arffile = rtv.rtv_res['arf']

        return cls(file, regfile, bkregfile, rmffile=rmffile, arffile=arffile)

    @property
    def armregfile(self):
        """Return the absolute path to the ARM exclusion region file, or ``None``."""

        if self._armregfile is None:
            return None
        else:
            return os.path.abspath(self._armregfile)

    @armregfile.setter
    def armregfile(self, new_armregfile):

        self._armregfile = new_armregfile

        self._ini_xselect()

    @property
    def arm(self):
        """Return ``True`` if the ARM region is included in the background."""

        return self._arm

    @arm.setter
    def arm(self, new_arm):

        self._arm = new_arm

        self._ini_xselect()

    @property
    def bkregfile(self):
        """Return the background region file path, including the ARM region if active."""

        if self.arm and self.armregfile:
            return f'"{os.path.abspath(self._bkregfile)} {self.armregfile}"'
        else:
            return f'{os.path.abspath(self._bkregfile)}'

    @property
    def rmffile(self):
        """Return the absolute path to the RMF file, or ``None`` if not set."""

        if self._rmffile is None:
            return None
        else:
            return os.path.abspath(self._rmffile)

    @rmffile.setter
    def rmffile(self, new_rmffile):

        self._rmffile = new_rmffile

    @property
    def arffile(self):
        """Return the absolute path to the ARF file, or ``None`` if not set."""

        if self._arffile is None:
            return None
        else:
            return os.path.abspath(self._arffile)

    @arffile.setter
    def arffile(self, new_arffile):

        self._arffile = new_arffile

    @property
    def timezero(self):
        """Return the reference time in EP MET seconds."""

        return self._timezero

    @timezero.setter
    def timezero(self, new_timezero):

        if isinstance(new_timezero, float):
            self._timezero = new_timezero

        elif isinstance(new_timezero, str):
            self._timezero = ep_utc_to_met(new_timezero)

        else:
            raise ValueError('not expected type for timezero')

    @property
    def timezero_utc(self):
        """Return the reference time as an EP UTC string."""

        return ep_met_to_utc(self.timezero)

    @staticmethod
    def _psf_model(r, p):

        rc = 5.1964
        beta = 1.5792
        sigma = 8.7997
        w = 0.0844

        king_part = (1 + (r / rc) ** 2) ** (-beta)
        gauss_part = np.exp(-(r**2) / (2 * sigma**2))

        psf = p * (king_part + w * gauss_part)
        return psf

    @property
    def psf_modelfile(self):
        """Return the ximage PSF model filename for EP FXT (``'fxt_psf.cod'``)."""

        return 'fxt_psf.cod'

    @property
    def psf_fitradius(self):
        """Return the PSF fit outer radius in arcseconds (default 30)."""

        try:
            _ = self._psf_fitradius
        except AttributeError:
            return 30
        else:
            if self._psf_fitradius is not None:
                return self._psf_fitradius
            else:
                return 30

    @psf_fitradius.setter
    def psf_fitradius(self, new_psf_fitradius):

        if isinstance(new_psf_fitradius, (int, type(None))):
            self._psf_fitradius = new_psf_fitradius
        else:
            raise ValueError('psf_fitradius is extected to be int or None')

    def extract_response(self, savepath='./spectrum'):
        """Copy the RMF and ARF files to the spectrum output directory.

        Args:
            savepath: Directory where ``ep.rmf`` and ``ep.arf`` are written.

        Raises:
            AssertionError: If ``rmffile`` has not been set.
            AssertionError: If ``arffile`` has not been set.
        """

        assert self.rmffile is not None, 'rmffile is not set, cannot extract response'
        assert self.arffile is not None, 'arffile is not set, cannot extract response'

        savepath = os.path.abspath(savepath)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        rsp_rmffile = savepath + '/ep.rmf'
        rsp_arffile = savepath + '/ep.arf'

        copy_file(self.rmffile, rsp_rmffile)
        copy_file(self.arffile, rsp_arffile)


class swiftImage(Image):
    """Read and process Swift/XRT event FITS files in PC or WT mode.

    Extends ``Image`` with Swift-specific time-system conversions (using
    the UTCFINIT keyword), vignetting-corrected exposure computation via
    ``xrtlccorr``, per-slice ARF generation via ``xrtmkarf``, and RMF
    look-up via ``quzcif``.
    """

    def __init__(self, file, regfile, bkregfile, attfile, xhdfile, mode):
        """Initialize swiftImage and run xselect to separate source and background events.

        Args:
            file: Path to the Swift/XRT event FITS file.
            regfile: Path to the source region file (DS9 format).
            bkregfile: Path to the background region file (DS9 format).
            attfile: Path to the Swift attitude file.
            xhdfile: Path to the Swift XRT housekeeping header file.
            mode: Observation mode string (e.g. ``'pc'`` or ``'wt'``);
                used as a file-name prefix.
        """

        self._file = file
        self._regfile = regfile
        self._bkregfile = bkregfile
        self._attfile = attfile
        self._xhdfile = xhdfile
        self._mode = mode

        self.prefix = f'{mode}_'

        self._ini_xselect()

    @classmethod
    def from_xrtobs(cls, obsid, mode, datapath=None):
        """Retrieve Swift/XRT observation data and construct a swiftImage instance.

        Fetches event, region, attitude, housekeeping header, and mode
        files for the specified XRT observation.

        Args:
            obsid: Swift observation ID string.
            mode: XRT observation mode (``'pc'`` or ``'wt'``).
            datapath: Local directory containing the observation data, or
                ``None`` to use the default retrieval path.
        """

        rtv = swiftRetrieve.from_xrtobs(obsid, mode, datapath)

        file = rtv.rtv_res['evt']
        regfile = rtv.rtv_res['reg']
        bkregfile = rtv.rtv_res['bkreg']
        attfile = rtv.rtv_res['att']
        xhdfile = rtv.rtv_res['xhd']
        mode = rtv.rtv_res['mode']

        return cls(file, regfile, bkregfile, attfile, xhdfile, mode)

    @property
    def attfile(self):
        """Return the absolute path to the Swift attitude file."""

        return os.path.abspath(self._attfile)

    @attfile.setter
    def attfile(self, new_attfile):

        self._attfile = new_attfile

    @property
    def xhdfile(self):
        """Return the absolute path to the XRT housekeeping header file."""

        return os.path.abspath(self._xhdfile)

    @xhdfile.setter
    def xhdfile(self, new_xhdfile):

        self._xhdfile = new_xhdfile

    @property
    def mode(self):
        """Return the XRT observation mode string (e.g. ``'pc'`` or ``'wt'``)."""

        return self._mode

    @property
    def utcf(self):
        """Return the UTCFINIT correction factor from the event file header."""

        hdu = fits.open(self._file)
        return hdu['EVENTS'].header['UTCFINIT']

    @property
    def timezero(self):
        """Return the reference time in Swift MET seconds."""

        return self._timezero

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
        """Return the reference time as a Swift UTC string."""

        return swift_met_to_utc(self.timezero, self.utcf)

    @staticmethod
    def _psf_model(r, p):

        rc = 3.726
        beta = 1.305
        sigma = 7.422
        w = 0.0807

        king_part = (1 + (r / rc) ** 2) ** (-beta)
        gauss_part = np.exp(-(r**2) / (2 * sigma**2))

        psf = p * (king_part + w * gauss_part)
        return psf

    @property
    def psf_modelfile(self):
        """Return the ximage PSF model filename for Swift/XRT (``'xrt_psf.cod'``)."""

        return 'xrt_psf.cod'

    @property
    def psf_fitradius(self):
        """Return the PSF fit outer radius in arcseconds (default 15)."""

        try:
            _ = self._psf_fitradius
        except AttributeError:
            return 15
        else:
            if self._psf_fitradius is not None:
                return self._psf_fitradius
            else:
                return 15

    @psf_fitradius.setter
    def psf_fitradius(self, new_psf_fitradius):

        if isinstance(new_psf_fitradius, (int, type(None))):
            self._psf_fitradius = new_psf_fitradius
        else:
            raise ValueError('psf_fitradius is extected to be int or None')

    @property
    def lc_exps(self):
        """Return vignetting-corrected exposure per light curve bin in seconds.

        Runs ``xrtlccorr`` on first access to produce a correction factor
        file and interpolates the 10-second cadence correction factors onto
        the light curve bin centres.
        """

        curve_savepath = os.path.dirname(self.file) + f'/{self.prefix}curve'

        src_corrfile = curve_savepath + '/src.corr'

        if not os.path.exists(src_corrfile):
            src_corr_curvefile = curve_savepath + '/src_corr.lc'
            src_instrfile = curve_savepath + '/src_srawinstr.img'

            commands = [
                'xrtlccorr',
                'clobber=yes',
                'regionfile=None',
                f'lcfile={self.src_curvefile}',
                f'outfile={src_corr_curvefile}',
                f'corrfile={src_corrfile}',
                f'attfile={self.attfile}',
                f'outinstrfile={src_instrfile}',
                f'infile={self.file}',
                f'hdfile={self.xhdfile}',
            ]

            stdout, stderr = self._run_commands(commands)

            if not os.path.exists(src_corrfile):
                print(stdout)
                print(stderr)

        hdu = fits.open(src_corrfile)
        time_10s = np.array(hdu['LCCORRFACT'].data['TIME'])
        factor_10s = np.array(hdu['LCCORRFACT'].data['CORRFACT'])

        diff = np.abs(self.lc_time[:, None] - time_10s[None, :])
        self.lc_factor = factor_10s[np.argmin(diff, axis=1)]

        return self.lc_binsize / self.lc_factor

    def extract_response(self, savepath='./spectrum', std=False):
        """Generate per-slice ARF and RMF response files for Swift/XRT.

        For each spectral slice, extracts a time-filtered event file,
        computes an exposure map, generates an ARF with ``xrtmkarf``, and
        queries the calibration database (``quzcif``) for the appropriate
        RMF.

        Args:
            savepath: Directory where response files are written.
            std: If ``True``, print command-line stdout and stderr for
                debugging.
        """

        savepath = os.path.abspath(savepath)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        lslices = np.array(self.spec_slices)[:, 0]
        rslices = np.array(self.spec_slices)[:, 1]

        for left, right in zip(lslices, rslices, strict=False):
            scc_start = self.timezero + left
            scc_stop = self.timezero + right

            new_left = f'{left:+.2f}'.replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_right = f'{right:+.2f}'.replace('-', 'm').replace('.', 'd').replace('+', 'p')

            file_name = '_'.join([new_left, new_right])

            exposure_savepath = savepath + '/exposure'

            if not os.path.exists(exposure_savepath):
                os.makedirs(exposure_savepath)

            evtfile = exposure_savepath + f'/{file_name}.evt'

            if os.path.exists(evtfile):
                os.remove(evtfile)

            commands = [
                'xsel',
                'read event',
                os.path.dirname(self.file),
                self.file.split('/')[-1],
                'yes',
                'filter time scc',
                f'{scc_start}, {scc_stop}',
                'x',
                'extract event copyall=yes',
                f'save event {evtfile}',
                'no',
            ]

            stdout, stderr = self._run_xselect(commands)

            if std:
                print(stdout)
                print(stderr)

            expfile = exposure_savepath + f'/{file_name}_ex.img'

            commands = [
                'xrtexpomap',
                'clobber=yes',
                f'infile={evtfile}',
                f'attfile={self.attfile}',
                f'hdfile={self.xhdfile}',
                f'outdir={exposure_savepath}/',
                f'stemout={file_name}',
            ]

            stdout, stderr = self._run_commands(commands)

            if std:
                print(stdout)
                print(stderr)

            src_specfile = savepath + f'/{file_name}.src'

            if not os.path.exists(src_specfile):
                self.extract_spectrum(savepath=savepath, std=std)

            rsp_arffile = savepath + f'/{file_name}.arf'

            commands = [
                'xrtmkarf',
                'clobber=yes',
                f'expofile={expfile}',
                f'phafile={src_specfile}',
                'psfflag=yes',
                f'outfile={rsp_arffile}',
                'srcx=-1',
                'srcy=-1',
            ]

            stdout, stderr = self._run_commands(commands)

            if std:
                print(stdout)
                print(stderr)

            rsp_rmffile = savepath + f'/{file_name}.rmf'
            date_time = swift_met_to_utc((scc_start + scc_stop) / 2, self.utcf)
            date = date_time.split('T')[0]
            time = date_time.split('T')[1]

            commands = [
                'quzcif',
                'mission=SWIFT',
                'instrument=XRT',
                'detector=-',
                'filter=-',
                'codename=matrix',
                f'date={date}',
                f'time={time}',
                'expr=datamode.eq.windowed.and.grade.eq.G0:2.and.XRTVSUB.eq.6',
            ]

            stdout, stderr = self._run_commands(commands)

            copy_file(stdout.split()[0], rsp_rmffile)

            if std:
                print(stdout)
                print(stderr)
