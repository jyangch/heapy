"""Retrieve data files from high-energy astrophysics mission archives.

Provides mission-specific retrieval classes — ``gbmRetrieve``,
``gecamRetrieve``, ``gridRetrieve``, ``epRetrieve``, and
``swiftRetrieve`` — all inheriting from the base ``Retrieve`` class.
Each class exposes one or more ``from_*`` classmethods that locate or
download the required data files and return a populated instance.

Typical usage:
    from heapy.data.retrieve import gbmRetrieve
    rtv = gbmRetrieve.from_burst('bn180703949')
    tte_file = rtv.rtv_res['tte']['n0']
"""

import os
import gzip
import shutil
import warnings
import pandas as pd
from astropy.time import Time, TimeDelta

from .filefinder import FileFinder



class Retrieve(object):
    """Base container for mission data retrieval results.

    Stores the dictionary returned by a ``from_*`` factory method and
    provides a common superclass for all mission-specific retrieval classes.

    Attributes:
        rtv_res: Dictionary containing paths to retrieved data files and
            the query parameters used to find them.
    """

    def __init__(self, rtv_res):
        """Initialize Retrieve with a retrieval result dictionary.

        Args:
            rtv_res: Dictionary of retrieved file paths and metadata,
                as produced by the ``from_*`` classmethods of subclasses.
        """

        self.rtv_res = rtv_res



class gbmRetrieve(Retrieve):
    """Retrieve Fermi GBM data files for a burst or a UTC time window.

    Wraps ``FileFinder`` to locate TTE, HEALPix, and position-history
    files from a local cache or the Fermi FTP server at
    ``ftp://129.164.179.23``.
    """

    def __init__(self, rtv_res):
        """Initialize gbmRetrieve with a retrieval result dictionary.

        Args:
            rtv_res: Dictionary of retrieved file paths and metadata.
        """

        super().__init__(rtv_res)


    @classmethod
    def from_burst(cls, burstid, datapath=None, skip_tte=False, skip_healpix=False):
        """Retrieve GBM burst data files for all 14 detectors.

        Locates TTE event files and the all-sky HEALPix localisation file
        for the given burst ID.  Files are downloaded from the Fermi FTP
        server into the local cache when not already present.

        Args:
            burstid: GBM burst identifier string (e.g. ``'bn180703949'``).
            datapath: Root directory of the local GBM burst data cache.
                Defaults to ``'/Users/junyang/Data/fermi/data/gbm/bursts'``.
            skip_tte: When ``True``, skip retrieval of TTE files.
            skip_healpix: When ``True``, skip retrieval of the HEALPix
                localisation file.

        Returns:
            A ``gbmRetrieve`` instance whose ``rtv_res`` dictionary contains
            ``'burstid'``, ``'datapath'``, ``'tte'`` (dict keyed by detector
            name), and ``'healpix'`` (path or ``None``).
        """

        if datapath is None:
            datapath = '/Users/junyang/Data/fermi/data/gbm/bursts'

        dataurl = 'ftp://129.164.179.23/fermi/data/gbm/bursts'

        year = '20' + burstid[2:4]

        local_dir = datapath + '/' + year + '/' + burstid + '/current'
        if not os.path.isdir(local_dir): os.makedirs(local_dir)

        ftp_url = dataurl + '/' + year + '/' + burstid + '/current'

        ff = FileFinder(local_dir=local_dir, ftp_url=ftp_url)

        dets = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        tte_dict = {}

        if not skip_tte:
            for det in dets:
                tte_feature = 'glg_tte_' + det + '_' + burstid + '_v*fit'
                tte_file = ff.find(tte_feature)
                tte_dict[det] = tte_file[-1] if tte_file else None

        if not skip_healpix:
            healpix_feature = 'glg_healpix_all_' + burstid + '_v*fit'
            healpix_file = ff.find(healpix_feature)
            healpix = healpix_file[-1] if healpix_file else None
        else:
            healpix = None

        rtv_res = {'burstid': burstid, 'datapath': datapath,
                   'tte': tte_dict, 'healpix': healpix}

        rtv = cls(rtv_res)

        return rtv


    @classmethod
    def from_utc(cls, utc, t1, t2, datapath=None, skip_tte=False, skip_poshist=False):
        """Retrieve GBM daily data files covering a UTC time window.

        Determines the hourly and daily date ranges spanned by
        ``[utc + t1, utc + t2]``, then locates TTE and position-history
        files for each relevant hour and day.

        Args:
            utc: Reference UTC time as an ISO-T string or an
                ``astropy.time.Time`` object.
            t1: Start offset relative to ``utc`` in seconds.
            t2: Stop offset relative to ``utc`` in seconds.
            datapath: Root directory of the local GBM daily data cache.
                Defaults to ``'/Users/junyang/Data/fermi/data/gbm/daily'``.
            skip_tte: When ``True``, skip retrieval of TTE files.
            skip_poshist: When ``True``, skip retrieval of position-history
                files.

        Returns:
            A ``gbmRetrieve`` instance whose ``rtv_res`` dictionary contains
            ``'utc'``, ``'t1'``, ``'t2'``, ``'datapath'``, ``'tte'`` (dict
            keyed by detector name, each value a list of hourly paths), and
            ``'poshist'`` (list of daily paths).

        Note:
            A ``UserWarning`` is emitted when the requested time range spans
            more than two hours.
        """

        if datapath is None:
            datapath = '/Users/junyang/Data/fermi/data/gbm/daily'

        dataurl = 'ftp://129.164.179.23/fermi/data/gbm/daily'

        ff = FileFinder(local_dir=datapath, ftp_url=dataurl)

        if isinstance(utc, Time) == False:
            utc = Time(utc, format='isot', scale='utc')

        t1 = TimeDelta(t1, format='sec')
        t2 = TimeDelta(t2, format='sec')

        tstart = utc + t1
        tstop = utc + t2

        year_start, month_start, day_start, hour_start = list(tstart.ymdhms)[:4]
        year_stop, month_stop, day_stop, hour_stop = list(tstop.ymdhms)[:4]

        start = '%d-%02d-%02d %02d' % (year_start, month_start, day_start, hour_start)+':00:00'
        stop = '%d-%02d-%02d %02d' % (year_stop, month_stop, day_stop, hour_stop) + ':00:00'
        dates_perH = pd.date_range(start, stop, freq='h')
        dates_perD = pd.date_range(start[:10], stop[:10], freq='D')

        if len(dates_perH) > 2:
            msg = 'the time range is too long'
            warnings.warn(msg, UserWarning, stacklevel=2)

        poshist_list = []
        dets = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        tte_dict = {det:[] for det in dets}

        if not skip_tte:
            for date in dates_perH:
                year = '%d' % date.year
                month = '%.2d' % date.month
                day = '%.2d' % date.day
                hour = '%.2d' % date.hour

                local_dir = datapath + '/' + year + '/' + month + '/' + day + '/current'
                if not os.path.isdir(local_dir): os.makedirs(local_dir)

                ftp_url = dataurl + '/' + year + '/' + month + '/' + day + '/current'

                ff.local_dir = local_dir
                ff.ftp_url = ftp_url

                for det in dets:
                    tte_feature = 'glg_tte_' + det + '_' + year[-2:] + month + day + '_' + hour + 'z_v*fit.gz'
                    tte_file = ff.find(tte_feature)
                    tte_dict[det].append(tte_file[-1] if tte_file else None)

        if not skip_poshist:
            for date in dates_perD:
                year = '%d' % date.year
                month = '%.2d' % date.month
                day = '%.2d' % date.day

                local_dir = datapath + '/' + year + '/' + month + '/' + day + '/current'
                if not os.path.isdir(local_dir): os.makedirs(local_dir)

                ftp_url = dataurl + '/' + year + '/' + month + '/' + day + '/current'

                ff.local_dir = local_dir
                ff.ftp_url = ftp_url

                poshist_feature = 'glg_poshist_all_' + year[-2:] + month + day + '_v*fit'
                poshist_file = ff.find(poshist_feature)
                poshist_list.append(poshist_file[-1] if poshist_file else None)

        rtv_res = {'utc': utc.value, 't1': t1.value, 't2': t2.value, 'datapath': datapath,
                   'tte': tte_dict, 'poshist': poshist_list}

        rtv = cls(rtv_res)

        return rtv



class gecamRetrieve(Retrieve):
    """Retrieve GECAM data files for a burst or a UTC time window.

    Wraps ``FileFinder`` to locate GECAM GRD event, background spectrum,
    background time, and position/attitude files from the local data cache.
    """

    def __init__(self, rtv_res):
        """Initialize gecamRetrieve with a retrieval result dictionary.

        Args:
            rtv_res: Dictionary of retrieved file paths and metadata.
        """

        super().__init__(rtv_res)


    @classmethod
    def from_burst(cls, burstid, datapath=None):
        """Retrieve GECAM burst data files from the local cache.

        Searches for GRD event, background spectrum, background time, and
        position/attitude files associated with the given burst ID.

        Args:
            burstid: GECAM burst identifier string
                (e.g. ``'GRB240101A'``).
            datapath: Root directory of the local GECAM burst data cache.
                Defaults to ``'/Users/junyang/Data/gecam/bursts'``.

        Returns:
            A ``gecamRetrieve`` instance whose ``rtv_res`` dictionary
            contains ``'burstid'``, ``'datapath'``, ``'grd_evt'``,
            ``'grd_bspec'``, ``'grd_btime'``, and ``'posatt'``
            (each a file path or ``None``).
        """

        if datapath is None:
            datapath = '/Users/junyang/Data/gecam/bursts'

        year = '20' + burstid[2:4]
        month = burstid[4:6]

        local_dir = datapath + '/' + year + '/' + month + '/' + burstid

        ff = FileFinder(local_dir=local_dir)

        grd_evt_feature = '*g_evt_' + burstid + '*'
        grd_evt_file = ff.find(grd_evt_feature)
        grd_evt = grd_evt_file[-1] if grd_evt_file else None

        grd_bspec_feature = '*g_bspec_' + burstid + '*'
        grd_bspec_file = ff.find(grd_bspec_feature)
        grd_bspec = grd_bspec_file[-1] if grd_bspec_file else None

        grd_btime_feature = '*g_btime_' + burstid + '*'
        grd_btime_file = ff.find(grd_btime_feature)
        grd_btime = grd_btime_file[-1] if grd_btime_file else None

        posatt_feature = '*_posatt_' + burstid + '*'
        posatt_file = ff.find(posatt_feature)
        posatt = posatt_file[-1] if posatt_file else None

        rtv_res = {'burstid': burstid, 'datapath': datapath,
                   'grd_evt': grd_evt, 'grd_bspec': grd_bspec,
                   'grd_btime': grd_btime, 'posatt': posatt}

        rtv = cls(rtv_res)

        return rtv


    @classmethod
    def from_utc(cls, utc, t1, t2, datapath=None):
        """Retrieve GECAM daily data files covering a UTC time window.

        Determines the hourly date range spanned by ``[utc + t1, utc + t2]``
        and locates GRD event, background spectrum, background time, and
        position/attitude files for each relevant hour.

        Args:
            utc: Reference UTC time as an ISO-T string or an
                ``astropy.time.Time`` object.
            t1: Start offset relative to ``utc`` in seconds.
            t2: Stop offset relative to ``utc`` in seconds.
            datapath: Root directory of the local GECAM daily data cache.
                Defaults to ``'/Users/junyang/Data/gecam/daily'``.

        Returns:
            A ``gecamRetrieve`` instance whose ``rtv_res`` dictionary
            contains ``'utc'``, ``'t1'``, ``'t2'``, ``'datapath'``,
            ``'grd_evt'``, ``'grd_bspec'``, ``'grd_btime'``, and
            ``'posatt'`` (each a list of hourly paths).

        Note:
            A ``UserWarning`` is emitted when the requested time range spans
            more than two hours.
        """

        if datapath is None:
            datapath = '/Users/junyang/Data/gecam/daily'

        ff = FileFinder(local_dir=datapath)

        if isinstance(utc, Time) == False:
            utc = Time(utc, format='isot', scale='utc')

        t1 = TimeDelta(t1, format='sec')
        t2 = TimeDelta(t2, format='sec')

        tstart = utc + t1
        tstop = utc + t2

        year_start, month_start, day_start, hour_start = list(tstart.ymdhms)[:4]
        year_stop, month_stop, day_stop, hour_stop = list(tstop.ymdhms)[:4]

        start = '%d-%02d-%02d %02d' % (year_start, month_start, day_start, hour_start)+':00:00'
        stop = '%d-%02d-%02d %02d' % (year_stop, month_stop, day_stop, hour_stop) + ':00:00'
        dates_perH = pd.date_range(start, stop, freq='h')

        if len(dates_perH) > 2:
            msg = 'the time range is too long'
            warnings.warn(msg, UserWarning, stacklevel=2)

        grd_evt_list = []
        grd_bspec_list = []
        grd_btime_list = []
        posatt_list = []

        for date in dates_perH:
            year = '%d' % date.year
            month = '%.2d' % date.month
            day = '%.2d' % date.day
            hour = '%.2d' % date.hour

            grd_evt_local_dir = datapath + '/' + year + '/' + month + '/' + day + '/GECAM_B/GRD_evt'
            ff.local_dir = grd_evt_local_dir
            grd_evt_feature = '*g_evt_' + year[-2:] + month + day + '_' + hour + '*'
            grd_evt_file = ff.find(grd_evt_feature)
            grd_evt_list.append(grd_evt_file[-1] if grd_evt_file else None)

            grd_bspec_local_dir = datapath + '/' + year + '/' + month + '/' + day + '/GECAM_B/GRD_bspec'
            ff.local_dir = grd_bspec_local_dir
            grd_bspec_feature = '*g_bspec_' + year[-2:] + month + day + '_' + hour + '*'
            grd_bspec_file = ff.find(grd_bspec_feature)
            grd_bspec_list.append(grd_bspec_file[-1] if grd_bspec_file else None)

            grd_btime_local_dir = datapath + '/' + year + '/' + month + '/' + day + '/GECAM_B/GRD_btime'
            ff.local_dir = grd_btime_local_dir
            grd_btime_feature = '*g_btime_' + year[-2:] + month + day + '_' + hour + '*'
            grd_btime_file = ff.find(grd_btime_feature)
            grd_btime_list.append(grd_btime_file[-1] if grd_btime_file else None)

            posatt_local_dir = datapath+ '/' + year + '/' + month + '/' + day + '/GECAM_B/posatt'
            ff.local_dir = posatt_local_dir
            posatt_feature = '*_posatt_' + year[-2:] + month + day + '_' + hour + '*'
            posatt_file = ff.find(posatt_feature)
            posatt_list.append(posatt_file[-1] if posatt_file else None)

        rtv_res = {'utc': utc.value, 't1': t1.value, 't2': t2.value, 'datapath': datapath,
                   'grd_evt': grd_evt_list, 'grd_bspec': grd_bspec_list,
                   'grd_btime': grd_btime_list, 'posatt': posatt_list}

        rtv = cls(rtv_res)

        return rtv



class gridRetrieve(Retrieve):
    """Retrieve GRID data files for a UTC time window.

    Wraps ``FileFinder`` to locate GRID TTE event and detector response
    files from the local data cache.
    """

    def __init__(self, rtv_res):
        """Initialize gridRetrieve with a retrieval result dictionary.

        Args:
            rtv_res: Dictionary of retrieved file paths and metadata.
        """

        super().__init__(rtv_res)


    @classmethod
    def from_utc(cls, utc, t1, t2, det, datapath=None):
        """Retrieve GRID daily data files for one detector over a UTC window.

        Determines the daily date range spanned by ``[utc + t1, utc + t2]``
        and locates TTE event and response files for the specified detector.

        Args:
            utc: Reference UTC time as an ISO-T string or an
                ``astropy.time.Time`` object.
            t1: Start offset relative to ``utc`` in seconds.
            t2: Stop offset relative to ``utc`` in seconds.
            det: Detector index as a string; must be one of
                ``'0'``, ``'1'``, ``'2'``, or ``'3'``.
            datapath: Root directory of the local GRID data cache.
                Defaults to ``'/Users/junyang/Data/grid/G05'``.

        Returns:
            A ``gridRetrieve`` instance whose ``rtv_res`` dictionary contains
            ``'utc'``, ``'t1'``, ``'t2'``, ``'det'``, ``'datapath'``,
            ``'tte'``, and ``'rsp'`` (each a list of daily paths).

        Raises:
            AssertionError: If ``det`` is not in ``['0', '1', '2', '3']``.

        Note:
            A ``UserWarning`` is emitted when the requested time range spans
            more than two days.
        """

        if datapath is None:
            datapath = '/Users/junyang/Data/grid/G05'

        ff = FileFinder(local_dir=datapath)

        if isinstance(utc, Time) == False:
            utc = Time(utc, format='isot', scale='utc')

        t1 = TimeDelta(t1, format='sec')
        t2 = TimeDelta(t2, format='sec')

        tstart = utc + t1
        tstop = utc + t2

        year_start, month_start, day_start, hour_start = list(tstart.ymdhms)[:4]
        year_stop, month_stop, day_stop, hour_stop = list(tstop.ymdhms)[:4]

        start = '%d-%02d-%02d %02d' % (year_start, month_start, day_start, hour_start)+':00:00'
        stop = '%d-%02d-%02d %02d' % (year_stop, month_stop, day_stop, hour_stop) + ':00:00'
        dates_perD = pd.date_range(start, stop, freq='D')

        if len(dates_perD) > 2:
            msg = 'the time range is too long'
            warnings.warn(msg, UserWarning, stacklevel=2)

        dets = ['%d' % i for i in range(0, 4)]
        assert det in dets, 'invalid detector: %s' % det
        tte_list = []
        rsp_list = []

        for date in dates_perD:
            year = '%d' % date.year
            month = '%.2d' % date.month
            day = '%.2d' % date.day

            tte_local_dir = datapath + '/' + year + '/' + month + '/' + day
            ff.local_dir = tte_local_dir
            tte_feature = '*_tte_' + year[-2:] + month + day + '*'
            tte_file = ff.find(tte_feature)
            tte_list.append(tte_file[-1] if tte_file else None)

            rsp_local_dir = datapath + '/' + year + '/' + month + '/' + day
            ff.local_dir = rsp_local_dir
            rsp_feature = '*_det' + det + '_*.rsp'
            rsp_file = ff.find(rsp_feature)
            rsp_list.append(rsp_file[-1] if rsp_file else None)

        rtv_res = {'utc': utc.value, 't1': t1.value, 't2': t2.value, 'det': det,
                   'datapath': datapath, 'tte': tte_list, 'rsp': rsp_list}

        rtv = cls(rtv_res)

        return rtv



class epRetrieve(Retrieve):
    """Retrieve Einstein Probe WXT and FXT data files for an observation.

    Wraps ``FileFinder`` to locate calibrated event files, spectral response
    matrices, auxiliary response files, and region files from the local data
    cache for both WXT and FXT instruments.
    """

    def __init__(self, rtv_res):
        """Initialize epRetrieve with a retrieval result dictionary.

        Args:
            rtv_res: Dictionary of retrieved file paths and metadata.
        """

        super().__init__(rtv_res)


    @classmethod
    def from_wxtobs(cls, obsid, srcid, datapath=None):
        """Retrieve EP/WXT observation data files for a given source.

        Locates the calibrated event file, RMF, ARF, source region, and
        background region files for the specified observation and source.
        If the source region file is absent it is synthesised from the
        background region file.

        Args:
            obsid: WXT observation ID string.
            srcid: Source identifier string used in file name patterns.
            datapath: Root directory of the local EP WXT data cache.
                Defaults to ``'/Users/junyang/Data/ep/WXT'``.

        Returns:
            An ``epRetrieve`` instance whose ``rtv_res`` dictionary contains
            ``'satelite'`` (``'WXT'``), ``'obsid'``, ``'srcid'``, ``'evt'``,
            ``'rmf'``, ``'arf'``, ``'armreg'``, ``'reg'``, and ``'bkreg'``
            (each a file path or ``None``).
        """

        if datapath is None:
            datapath = '/Users/junyang/Data/ep/WXT'

        local_dir = datapath + '/' + obsid

        ff = FileFinder(local_dir=local_dir)

        evt_feature = 'ep*_cl.evt'
        evt_file = ff.find(evt_feature)
        evt = evt_file[-1] if evt_file else None

        rmf_feature = 'ep*.rmf'
        rmf_file = ff.find(rmf_feature)
        rmf = rmf_file[-1] if rmf_file else None

        arf_feature = 'ep*' + srcid + '.arf'
        arf_file = ff.find(arf_feature)
        arf = arf_file[-1] if arf_file else None

        armreg_feature = 'ep*arm.reg'
        armreg_file = ff.find(armreg_feature)
        armreg = armreg_file[-1] if armreg_file else None

        bkreg_feature = 'ep*' + srcid + 'bk.reg'
        bkreg_file = ff.find(bkreg_feature)
        bkreg = bkreg_file[-1] if bkreg_file else None

        reg_feature = 'ep*' + srcid + '.reg'
        reg_file = ff.find(reg_feature)
        reg = reg_file[-1] if reg_file else None

        if reg is None:

            reg = bkreg[:-6] + '.reg'

            with open(bkreg, 'r') as f_obj:
                items = f_obj.readlines()[0].split()

            with open(reg, 'w') as f_obj:
                f_obj.write('circle ' + items[1] + ' 67)')

        rtv_res = {'satelite': 'WXT', 'obsid': obsid, 'srcid': srcid,
                   'evt': evt, 'rmf': rmf, 'arf': arf,
                   'armreg': armreg, 'reg': reg, 'bkreg': bkreg}

        rtv = cls(rtv_res)

        return rtv


    @classmethod
    def from_fxtobs(cls, obsid, module, datapath=None):
        """Retrieve EP/FXT observation data files for a given module.

        Locates the calibrated event file, source-specific (or fallback
        generic) RMF and ARF, and source and background region files.

        Args:
            obsid: FXT observation ID string.
            module: FXT module identifier (e.g. ``'A'`` or ``'B'``).
            datapath: Root directory of the local EP FXT data cache.
                Defaults to ``'/Users/junyang/Data/ep/FXT'``.

        Returns:
            An ``epRetrieve`` instance whose ``rtv_res`` dictionary contains
            ``'satelite'`` (``'FXT'``), ``'obsid'``, ``'module'``, ``'evt'``,
            ``'rmf'``, ``'arf'``, ``'reg'``, and ``'bkreg'`` (each a file
            path or ``None``).
        """

        if datapath is None:
            datapath = '/Users/junyang/Data/ep/FXT'

        local_dir = datapath + '/' + obsid

        ff = FileFinder(local_dir=local_dir)

        evt_feature = f'fxt_{module}_*_cl*.fits'
        evt_file = ff.find(evt_feature)
        evt = evt_file[-1] if evt_file else None

        src_rmf_feature = f'fxt_{module}_*_src*.rmf'
        src_rmf_file = ff.find(src_rmf_feature)
        src_rmf = src_rmf_file[-1] if src_rmf_file else None

        if src_rmf is not None:
            rmf = src_rmf
        else:
            rmf_feature = f'fxt_{module}_*.rmf'
            rmf_file = ff.find(rmf_feature)
            rmf = rmf_file[-1] if rmf_file else None

        src_arf_feature = f'fxt_{module}_*_src*.arf'
        src_arf_file = ff.find(src_arf_feature)
        src_arf = src_arf_file[-1] if src_arf_file else None

        if src_arf is not None:
            arf = src_arf
        else:
            arf_feature = f'fxt_{module}_*.arf'
            arf_file = ff.find(arf_feature)
            arf = arf_file[-1] if arf_file else None

        reg_feature = f'fxt_{module}_*_src*.reg'
        reg_file = ff.find(reg_feature)
        reg = reg_file[-1] if reg_file else None

        bkreg_feature = f'fxt_{module}_*_bkg*.reg'
        bkreg_file = ff.find(bkreg_feature)
        bkreg = bkreg_file[-1] if bkreg_file else None

        rtv_res = {'satelite': 'FXT', 'obsid': obsid, 'module': module,
                   'evt': evt, 'rmf': rmf, 'arf': arf, 'reg': reg, 'bkreg': bkreg}

        rtv = cls(rtv_res)

        return rtv


    @classmethod
    def from_fxtsrc(cls, obsid, module, datapath=None):
        """Retrieve EP/FXT pre-extracted source and background products.

        Locates pre-extracted source and background event files, PHA spectra,
        RMF, and ARF for the specified observation and module.

        Args:
            obsid: FXT observation ID string.
            module: FXT module identifier (e.g. ``'A'`` or ``'B'``).
            datapath: Root directory of the local EP FXT data cache.
                Defaults to ``'/Users/junyang/Data/ep/FXT'``.

        Returns:
            An ``epRetrieve`` instance whose ``rtv_res`` dictionary contains
            ``'satelite'`` (``'FXT'``), ``'obsid'``, ``'module'``,
            ``'src_evt'``, ``'bkg_evt'``, ``'src_spec'``, ``'bkg_spec'``,
            ``'rmf'``, and ``'arf'`` (each a file path or ``None``).
        """

        if datapath is None:
            datapath = '/Users/junyang/Data/ep/FXT'

        local_dir = datapath + '/' + obsid

        ff = FileFinder(local_dir=local_dir)

        src_evt_feature = f'fxt_{module}_*_cl_src_*.fits'
        src_evt_file = ff.find(src_evt_feature)
        src_evt = src_evt_file[-1] if src_evt_file else None

        bkg_evt_feature = f'fxt_{module}_*_cl_bkg_*.fits'
        bkg_evt_file = ff.find(bkg_evt_feature)
        bkg_evt = bkg_evt_file[-1] if bkg_evt_file else None

        src_spec_feature = f'fxt_{module}_*_src_*.pha'
        src_spec_file = ff.find(src_spec_feature)
        src_spec = src_spec_file[-1] if src_spec_file else None

        bkg_spec_feature = f'fxt_{module}_*_bkg_*.pha'
        bkg_spec_file = ff.find(bkg_spec_feature)
        bkg_spec = bkg_spec_file[-1] if bkg_spec_file else None

        rmf_feature = f'fxt_{module}_*_src_*.rmf'
        rmf_file = ff.find(rmf_feature)
        rmf = rmf_file[-1] if rmf_file else None

        arf_feature = f'fxt_{module}_*_src_*.arf'
        arf_file = ff.find(arf_feature)
        arf = arf_file[-1] if arf_file else None

        rtv_res = {'satelite': 'FXT', 'obsid': obsid, 'module': module,
                   'src_evt': src_evt, 'bkg_evt': bkg_evt,
                   'src_spec': src_spec, 'bkg_spec': bkg_spec,
                   'rmf': rmf, 'arf': arf}

        rtv = cls(rtv_res)

        return rtv



class swiftRetrieve(Retrieve):
    """Retrieve Swift XRT and BAT data files for an observation.

    Wraps ``FileFinder`` to locate calibrated event files, response files,
    and auxiliary files for Swift XRT (WT or PC mode) and BAT observations
    from the local data cache.
    """

    def __init__(self, rtv_res):
        """Initialize swiftRetrieve with a retrieval result dictionary.

        Args:
            rtv_res: Dictionary of retrieved file paths and metadata.
        """

        super().__init__(rtv_res)


    @classmethod
    def from_xrtobs(cls, obsid, mode, datapath=None):
        """Retrieve Swift XRT observation data files.

        Locates the calibrated event file (decompressing from ``.gz`` when
        necessary), source and background region files, the XRT housekeeping
        file, and the attitude file.

        Args:
            obsid: Swift observation ID string.
            mode: XRT readout mode; must be ``'wt'`` (Windowed Timing) or
                ``'pc'`` (Photon Counting).
            datapath: Root directory of the local Swift data cache.
                Defaults to ``'/Users/junyang/Data/swift'``.

        Returns:
            A ``swiftRetrieve`` instance whose ``rtv_res`` dictionary
            contains ``'satelite'`` (``'XRT'``), ``'obsid'``, ``'mode'``,
            ``'evt'``, ``'reg'``, ``'bkreg'``, ``'xhd'``, and ``'att'``
            (each a file path or ``None``).

        Raises:
            AssertionError: If ``mode`` is not ``'wt'`` or ``'pc'``.
        """

        assert mode in ['wt', 'pc'], 'xrt mode only allowed to be wt or pc'

        if datapath is None:
            datapath = '/Users/junyang/Data/swift'

        local_dir = datapath + '/' + obsid

        ff = FileFinder(local_dir=local_dir + '/xrt/event')

        evtgz_feature = f'sw{obsid}x{mode}*po_cl.evt.gz'
        evtgz_file = ff.find(evtgz_feature)
        evtgz = evtgz_file[-1] if evtgz_file else None
        evt = evtgz[:-3]

        if not os.path.exists(evt):
            with gzip.open(evtgz, 'rb') as f_in:
                with open(evt, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        bkreg_feature = f'sw{obsid}x{mode}*pobk.reg'
        bkreg_file = ff.find(bkreg_feature)
        bkreg = bkreg_file[-1] if bkreg_file else None

        reg_feature = f'sw{obsid}x{mode}*po.reg'
        reg_file = ff.find(reg_feature)
        reg = reg_file[-1] if reg_file else None

        ff = FileFinder(local_dir=local_dir + '/xrt/hk')
        xhd_feature = 'sw*xhd.hk.gz'
        xhd_file = ff.find(xhd_feature)
        xhd = xhd_file[-1] if xhd_file else None

        ff = FileFinder(local_dir=local_dir + '/auxil')
        att_feature = 'sw*pat.fits.gz'
        att_file = ff.find(att_feature)
        att = att_file[-1] if att_file else None

        rtv_res = {'satelite': 'XRT', 'obsid': obsid, 'mode': mode,
                   'evt': evt, 'reg': reg, 'bkreg': bkreg, 'xhd': xhd, 'att': att}

        rtv = cls(rtv_res)

        return rtv


    @classmethod
    def from_batobs(cls, obsid, datapath=None):
        """Retrieve Swift BAT observation data files.

        Locates the unfiltered BAT event file (decompressing from ``.gz``
        when necessary), calibration database file, detector mask, satellite
        attitude file, and auxiliary event file.

        Args:
            obsid: Swift observation ID string.
            datapath: Root directory of the local Swift data cache.
                Defaults to ``'/Users/junyang/Data/swift'``.

        Returns:
            A ``swiftRetrieve`` instance whose ``rtv_res`` dictionary
            contains ``'satelite'`` (``'BAT'``), ``'obsid'``, ``'ufevt'``,
            ``'caldb'``, ``'detmask'``, ``'att'``, and ``'aux'`` (each a
            file path or ``None``).
        """

        if datapath is None:
            datapath = '/Users/junyang/Data/swift'

        local_dir = datapath + '/' + obsid

        ff = FileFinder(local_dir=local_dir + '/bat/event')
        ufevtgz_feature = 'sw*bevshsp_uf.evt.gz'
        ufevtgz_file = ff.find(ufevtgz_feature)
        ufevtgz = ufevtgz_file[-1] if ufevtgz_file else None
        ufevt = ufevtgz[:-3]

        if not os.path.exists(ufevt):
            with gzip.open(ufevtgz, 'rb') as f_in:
                with open(ufevt, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        ff = FileFinder(local_dir=local_dir + '/bat/hk')
        caldb_feature = 'sw*bgocb.hk.gz'
        caldb_file = ff.find(caldb_feature)
        caldb = caldb_file[-1] if caldb_file else None

        ff = FileFinder(local_dir=local_dir + '/bat/hk')
        detmask_feature = 'sw*bdecb.hk.gz'
        detmask_file = ff.find(detmask_feature)
        detmask = detmask_file[-1] if detmask_file else None

        ff = FileFinder(local_dir=local_dir + '/auxil')
        att_feature = 'sw*sat.fits.gz'
        att_file = ff.find(att_feature)
        att = att_file[-1] if att_file else None

        ff = FileFinder(local_dir=local_dir + '/bat/event')
        aux_feature = 'sw*bevtr.fits.gz'
        aux_file = ff.find(aux_feature)
        aux = aux_file[-1] if aux_file else None

        rtv_res = {'satelite': 'BAT', 'obsid': obsid,
                   'ufevt': ufevt, 'caldb': caldb,
                   'detmask': detmask, 'att': att, 'aux': aux}

        rtv = cls(rtv_res)

        return rtv
