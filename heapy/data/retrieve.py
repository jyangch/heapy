import os
import ftplib
import warnings
import pandas as pd
from tqdm import tqdm
from urllib.parse import urlparse
from astropy.time import Time, TimeDelta
from ..util.data import msg_format



class Retrieve(object):
    
    def __init__(self, rtv_res):
        
        self.rtv_res = rtv_res



class gbmRetrieve(Retrieve):

    def __init__(self, rtv_res):
        
        super().__init__(rtv_res)


    @classmethod
    def from_burst(cls, burstid, datapath=None):
        
        if datapath is None:
            datapath = '/Users/jyang/Documents/fermi/data/gbm/bursts'
            
        dataurl = 'ftp://129.164.179.23/fermi/data/gbm/bursts'

        year = '20' + burstid[2:4]
        
        local_dir = datapath + '/' + year + '/' + burstid + '/current'
        if not os.path.isdir(local_dir): os.makedirs(local_dir)
        
        ftp_url = dataurl + '/' + year + '/' + burstid + '/current'
        
        ff = FileFinder(local_dir=local_dir, ftp_url=ftp_url)

        dets = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        tte_dict = {}
        cspec_pha_dict = {}
        cspec_rsp_dict = {}
        ctime_pha_dict = {}
        ctime_rsp_dict = {}

        for det in dets:
            tte_feature = 'glg_tte_' + det + '_' + burstid + '_v*fit'
            tte_file = ff.find(tte_feature)
            tte_dict[det] = tte_file[-1] if tte_file else None

            cspec_feature = 'glg_cspec_' + det + '_' + burstid + '_v*pha'
            cspec_file = ff.find(cspec_feature)
            cspec_pha_dict[det] = cspec_file[-1] if cspec_file else None

            cspec_feature = 'glg_cspec_' + det + '_' + burstid + '_v*rsp2'
            cspec_file = ff.find(cspec_feature)
            cspec_rsp_dict[det] = cspec_file[-1] if cspec_file else None

            ctime_feature = 'glg_ctime_' + det + '_' + burstid + '_v*pha'
            ctime_file = ff.find(ctime_feature)
            ctime_pha_dict[det] = ctime_file[-1] if ctime_file else None

            ctime_feature = 'glg_ctime_' + det + '_' + burstid + '_v*rsp2'
            ctime_file = ff.find(ctime_feature)
            ctime_rsp_dict[det] = ctime_file[-1] if ctime_file else None
            
        healpix_feature = 'glg_healpix_all_' + burstid + '_v*fit'
        healpix_file = ff.find(healpix_feature)
        healpix = healpix_file[-1] if healpix_file else None

        rtv_res = {'burstid': burstid, 'datapath': datapath, 'tte': tte_dict, 
                   'cspec_pha': cspec_pha_dict, 'cspec_rsp': cspec_rsp_dict, 
                   'ctime_pha': ctime_pha_dict, 'ctime_rsp': ctime_rsp_dict, 
                   'healpix': healpix}
        
        rtv = cls(rtv_res)
        
        return rtv


    @classmethod
    def from_utc(cls, utc, t1, t2, datapath=None):
        
        if datapath is None:
            datapath = '/Users/jyang/Documents/fermi/data/gbm/daily'
            
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
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)

        poshist_list = []
        dets = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        tte_dict = {det:[] for det in dets}
        cspec_pha_dict = {det:[] for det in dets}
        ctime_pha_dict = {det:[] for det in dets}

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

        for date in dates_perD:
            year = '%d' % date.year
            month = '%.2d' % date.month
            day = '%.2d' % date.day

            local_dir = datapath + '/' + year + '/' + month + '/' + day + '/current'
            if not os.path.isdir(local_dir): os.makedirs(local_dir)
            
            ftp_url = dataurl + '/' + year + '/' + month + '/' + day + '/current'
            
            ff.local_dir = local_dir
            ff.ftp_url = ftp_url
            
            for det in dets:
                cspec_feature = 'glg_cspec_' + det + '_' + year[-2:] + month + day + '_v*pha'
                cspec_file = ff.find(cspec_feature)
                cspec_pha_dict[det].append(cspec_file[-1] if cspec_file else None)

                ctime_feature = 'glg_ctime_' + det + '_' + year[-2:] + month + day + '_v*pha'
                ctime_file = ff.find(ctime_feature)
                ctime_pha_dict[det].append(ctime_file[-1] if ctime_file else None)

            poshist_feature = 'glg_poshist_all_' + year[-2:] + month + day + '_v*fit'
            poshist_file = ff.find(poshist_feature)
            poshist_list.append(poshist_file[-1] if poshist_file else None)

        rtv_res = {'utc': utc.value, 't1': t1.value, 't2': t2.value, 'datapath': datapath, 
                   'tte': tte_dict, 'cspec_pha': cspec_pha_dict, 
                   'ctime_pha': ctime_pha_dict, 'poshist': poshist_list}
        
        rtv = cls(rtv_res)
        
        return rtv



class gecamRetrieve(Retrieve):

    def __init__(self, rtv_res):
        
        super().__init__(rtv_res)


    @classmethod
    def from_burst(cls, burstid, datapath=None):
        
        if datapath is None:
            datapath = '/Users/jyang/Documents/gecam/bursts'

        year = '20' + burstid[2:4]
        month = burstid[4:6]

        local_dir = datapath + '/' + year + '/' + month + '/' + burstid
        
        ff = FileFinder(local_dir=local_dir)

        grd_evt_feature = 'g_evt_' + burstid + '*'
        grd_evt_file = ff.find(grd_evt_feature)
        grd_evt = grd_evt_file[-1] if grd_evt_file else None

        grd_bspec_feature = 'g_bspec_' + burstid + '*'
        grd_bspec_file = ff.find(grd_bspec_feature)
        grd_bspec = grd_bspec_file[-1] if grd_bspec_file else None

        grd_btime_feature = 'g_btime_' + burstid + '*'
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
        
        if datapath is None:
            datapath = '/Users/jyang/Documents/gecam/daily'
            
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
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)

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
            grd_evt_feature = 'g_evt_' + year[-2:] + month + day + '_' + hour + '*'
            grd_evt_file = ff.find(grd_evt_feature)
            grd_evt_list.append(grd_evt_file[-1] if grd_evt_file else None)

            grd_bspec_local_dir = datapath + '/' + year + '/' + month + '/' + day + '/GECAM_B/GRD_bspec'
            ff.local_dir = grd_bspec_local_dir
            grd_bspec_feature = 'g_bspec_' + year[-2:] + month + day + '_' + hour + '*'
            grd_bspec_file = ff.find(grd_bspec_feature)
            grd_bspec_list.append(grd_bspec_file[-1] if grd_bspec_file else None)

            grd_btime_local_dir = datapath + '/' + year + '/' + month + '/' + day + '/GECAM_B/GRD_btime'
            ff.local_dir = grd_btime_local_dir
            grd_btime_feature = 'g_btime_' + year[-2:] + month + day + '_' + hour + '*'
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



class epRetrieve(Retrieve):

    def __init__(self, rtv_res):
        
        super().__init__(rtv_res)
        
        
    @classmethod
    def from_wxtobs(cls, obsname, srcid, datapath=None): 
        
        if datapath is None:
            datapath = '/Users/jyang/Documents/EinsteinProbe/WXT'
            
        local_dir = datapath + '/' + obsname
        
        ff = FileFinder(local_dir=local_dir)
        
        evt_feature = 'ep*_cl.evt'
        evt_file = ff.find(evt_feature)
        evt = evt_file[-1] if evt_file else None

        rmf_feature = 'ep*.rmf'
        rmf_file = ff.find(rmf_feature)
        rmf = rmf_file[-1] if rmf_file else None
        
        armreg_feature = 'ep*arm.reg'
        armreg_file = ff.find(armreg_feature)
        armreg = armreg_file[-1] if armreg_file else None
        
        arf_feature = 'ep*' + srcid + '.arf'
        arf_file = ff.find(arf_feature)
        arf = arf_file[-1] if arf_file else None
        
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

        rtv_res = {'satelite': 'WXT', 'obsname': obsname, 'srcid': srcid, 
                   'evt': evt, 'rmf': rmf, 'arf': arf, 
                   'armreg': armreg, 'reg': reg, 'bkreg': bkreg}
        
        rtv = cls(rtv_res)
        
        return rtv


    @classmethod
    def from_fxtobs(cls, obsname, srcid, datapath=None): 
        
        if datapath is None:
            datapath = '/Users/jyang/Documents/EinsteinProbe/FXT'
            
        local_dir = datapath + '/' + obsname
        
        ff = FileFinder(local_dir=local_dir)
        
        evt_feature = 'fxt_*_cl_*.fits'
        evt_file = ff.find(evt_feature)
        evt = evt_file[-1] if evt_file else None

        rmf_feature = 'fxt_*.rmf'
        rmf_file = ff.find(rmf_feature)
        rmf = rmf_file[-1] if rmf_file else None
        
        arf_feature = 'fxt_*.arf'
        arf_file = ff.find(arf_feature)
        arf = arf_file[-1] if arf_file else None
        
        reg_feature = 'fxt_*' + srcid + '.reg'
        reg_file = ff.find(reg_feature)
        reg = reg_file[-1] if reg_file else None
        
        bkreg_feature = 'fxt_*' + srcid + 'bk.reg'
        bkreg_file = ff.find(bkreg_feature)
        bkreg = bkreg_file[-1] if bkreg_file else None
        
        rtv_res = {'satelite': 'FXT', 'obsname': obsname, 'srcid': srcid, 
                   'evt': evt, 'rmf': rmf, 'arf': arf, 
                   'reg': reg, 'bkreg': bkreg}
        
        rtv = cls(rtv_res)
        
        return rtv


    @classmethod
    def from_fxtsrc(cls, obsname, datapath=None):
        
        if datapath is None:
            datapath = '/Users/jyang/Documents/EinsteinProbe/FXT'
            
        local_dir = datapath + '/' + obsname
        
        ff = FileFinder(local_dir=local_dir)
        
        src_evt_feature = 'fxt_*_src*cl_*.fits'
        src_evt_file = ff.find(src_evt_feature)
        src_evt = src_evt_file[-1] if src_evt_file else None
        
        bkg_evt_feature = 'fxt_*_bkg*cl_*.fits'
        bkg_evt_file = ff.find(bkg_evt_feature)
        bkg_evt = bkg_evt_file[-1] if bkg_evt_file else None
        
        src_spec_feature = 'fxt_*_src_*.pha'
        src_spec_file = ff.find(src_spec_feature)
        src_spec = src_spec_file[-1] if src_spec_file else None
        
        bkg_spec_feature = 'fxt_*_bkg_*.pha'
        bkg_spec_file = ff.find(bkg_spec_feature)
        bkg_spec = bkg_spec_file[-1] if bkg_spec_file else None

        rmf_feature = 'fxt_*.rmf'
        rmf_file = ff.find(rmf_feature)
        rmf = rmf_file[-1] if rmf_file else None
        
        arf_feature = 'fxt_*.arf'
        arf_file = ff.find(arf_feature)
        arf = arf_file[-1] if arf_file else None
        
        rtv_res = {'satelite': 'FXT', 'obsname': obsname, 
                   'src_evt': src_evt, 'bkg_evt': bkg_evt, 
                   'src_spec': src_spec, 'bkg_spec': bkg_spec, 
                   'rmf': rmf, 'arf': arf}
        
        rtv = cls(rtv_res)
        
        return rtv



class FileFinder:
    
    def __init__(self, local_dir, ftp_url=None):

        self._local_dir = os.path.abspath(local_dir)
        self._ftp_url = urlparse(ftp_url) if ftp_url else None
        
        self.local_files = None
        self.ftp_files = None
        self.ftp_connection = None
        
        
    @property
    def local_dir(self):
        
        return self._local_dir
    
    
    @local_dir.setter
    def local_dir(self, new_local_dir):
        
        self._local_dir = os.path.abspath(new_local_dir)
        
        
    @property
    def ftp_url(self):
        
        return self._ftp_url
    
    
    @ftp_url.setter
    def ftp_url(self, new_ftp_url):
        
        old_ftp_hostname = self._ftp_url.hostname if self._ftp_url else None
        
        self._ftp_url = urlparse(new_ftp_url) if new_ftp_url else None
        
        new_ftp_hostname = self._ftp_url.hostname if self._ftp_url else None
        
        if old_ftp_hostname and old_ftp_hostname != new_ftp_hostname:
            
            if self.ftp_connection:
                
                self.ftp_connection.quit()
                self.ftp_connection = None
    

    def __del__(self):

        if self.ftp_connection:
            self.ftp_connection.quit()


    def find(self, feature):

        self.local_files = self._get_files_from_local()
        matching_local_files = self._match_files(self.local_files, feature)

        if matching_local_files:
            
            return matching_local_files

        if self.ftp_url:
            self.ftp_files = self._get_files_from_ftp()
            matching_ftp_files = self._match_files(self.ftp_files, feature)

            if matching_ftp_files:
                
                downloaded_files_in_local = []
                
                pbar = tqdm(matching_ftp_files)
                
                for ftp_file_to_download in pbar:
                    
                    pbar.set_description(f'downloading {os.path.basename(ftp_file_to_download)}')
                    
                    local_file_to_write = os.path.join(self.local_dir, os.path.basename(ftp_file_to_download))
                    
                    self._download_file_from_ftp(ftp_file_to_download, local_file_to_write)
                    
                    downloaded_files_in_local.append(local_file_to_write)

                return downloaded_files_in_local

        warnings.warn(f"No files found matching the feature: {feature}", UserWarning)
        return None


    def _get_files_from_local(self):

        if not os.path.exists(self.local_dir):
            warnings.warn(f"Directory '{self.local_dir}' does not exist.", UserWarning)
            return []

        return [os.path.join(self.local_dir, f) for f in os.listdir(self.local_dir) 
                if os.path.isfile(os.path.join(self.local_dir, f))]


    def _get_files_from_ftp(self):
        
        self._ensure_ftp_connection()

        ftp_path = self.ftp_url.path

        try:
            return self.ftp_connection.nlst(ftp_path)
        except ftplib.all_errors as e:
            warnings.warn(f"FTP error: {str(e)}", UserWarning)
            return []


    def _download_file_from_ftp(self, ftp_file_path, local_file_path):

        self._ensure_ftp_connection()

        try:
            with open(local_file_path, 'wb') as local_file:
                self.ftp_connection.retrbinary(f"RETR {ftp_file_path}", local_file.write)
        except ftplib.all_errors as e:
            warnings.warn(f"FTP download error: {str(e)}", UserWarning)


    def _ensure_ftp_connection(self):

        if self.ftp_connection is None:
            
            ftp_host = self.ftp_url.hostname
            ftp_user = self.ftp_url.username or 'anonymous'
            ftp_pass = self.ftp_url.password or ''

            try:
                self.ftp_connection = ftplib.FTP_TLS(ftp_host)
                self.ftp_connection.login(user=ftp_user, passwd=ftp_pass)
                self.ftp_connection.prot_p()
                print(f"Connected to FTP: {ftp_host}")
            except ftplib.all_errors as e:
                warnings.warn(f"FTP connection error: {str(e)}", UserWarning)
                self.ftp_connection = None
                
        else:
            
            if not self._is_ftp_connection_alive():
                print("FTP connection lost, reconnecting...")
                self.ftp_connection = None
                self._ensure_ftp_connection()


    def _is_ftp_connection_alive(self):

        try:
            self.ftp_connection.voidcmd("NOOP")
            return True
        except (ftplib.error_temp, ftplib.error_perm, ftplib.error_proto, OSError):
            return False


    def _match_files(self, files_in_dir, feature):

        if not files_in_dir:
            return []

        feature_list = [f for f in feature.split('*') if f]

        starts_with = feature.startswith('*')
        ends_with = feature.endswith('*')
        
        matching_files = []

        for file in files_in_dir:
            file_name = os.path.basename(file)

            if not starts_with and not file_name.startswith(feature_list[0]):
                continue

            if not ends_with and not file_name.endswith(feature_list[-1]):
                continue

            match = True
            pos = 0
            for feat in feature_list:
                pos = file_name.find(feat, pos)
                if pos == -1:
                    match = False
                    break
                pos += len(feat)

            if match:
                matching_files.append(file)

        return matching_files
