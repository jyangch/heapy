import os
import re
import warnings
import pandas as pd
from astropy.time import Time, TimeDelta
from ..util.data import msg_format



class Retrieve(object):
    
    def __init__(self, rtv_res):
        
        self.rtv_res = rtv_res
        
        
    @staticmethod
    def findfile(dir, feature):
        
        if (os.path.exists(dir)):
            dirnames = os.listdir(dir)
            filelist = []
            fil_number = 0
            fil_result_number = 0
            featurelist = [i for i in re.split('[*]', feature) if i != '']
            for_number = len(featurelist)
            fileresult = [[] for i in range(for_number)]
            
            for eve in range(for_number):
                
                if (eve == 0):
                    
                    fe_number = len(featurelist[eve])
                    
                    for sample in dirnames:
                        if (os.path.isfile(dir + sample)):
                            filelist.append(sample)
                            fil_number = fil_number + 1
                            
                    if (fil_number != 0):
                        for i in filelist:
                            i_number = len(i)
                            n = i_number - fe_number + 1
                            for j in range(n):
                                if (i[j:j + fe_number] == featurelist[eve]):
                                    fileresult[eve].append(i)
                                    fil_result_number = fil_result_number + 1
                                    break
                                
                        if (fil_result_number == 0):
                            msg = 'can not find any file with the feature:' + '\n' + feature
                            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
                            return None
                        
                        else:
                            fil_result_number = 0
                            
                    else:
                        msg = 'there is no file in this dir'
                        warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
                        return None
                    
                else:
                    
                    fe_number = len(featurelist[eve])
                    
                    for i in fileresult[eve - 1]:
                        i_number = len(i)
                        n = i_number - fe_number + 1
                        for j in range(n):
                            if (i[j:j + fe_number] == featurelist[eve]):
                                fileresult[eve].append(i)
                                fil_result_number = fil_result_number + 1
                                break
                            
                    if (fil_result_number == 0):
                        msg = 'can not find any file with the feature:' + '\n' + feature
                        warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
                        return None
                    
                    else:
                        fil_result_number = 0
                        
            file = fileresult[for_number - 1]
            
            if len(file) > 1:
                msg = 'may find needless file:' + '\n' \
                    + '%s\n' % '\n'.join(file) \
                        + 'will only keep the last one'
                warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
                file = [name for name in file if name[0] != '.']
            filepath = dir + file[-1]
            
            return filepath
        
        else:
            msg = 'can not find the dir named [' + dir + ']'
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
            return None



class gbmRetrieve(Retrieve):

    def __init__(self, rtv_res):
        
        super().__init__(rtv_res)


    @classmethod
    def from_burst(cls, burstid, datapath=None):
        
        if datapath is None:
            datapath = '/fermi/data/gbm/bursts'

        year = '20' + burstid[2:4]
        link = datapath + '/' + year + '/' + burstid + '/current/'

        healpix_list = []
        dets = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        tte_dict = {det:[] for det in dets}
        cspec_pha_dict = {det:[] for det in dets}
        cspec_rsp_dict = {det:[] for det in dets}
        ctime_pha_dict = {det:[] for det in dets}
        ctime_rsp_dict = {det:[] for det in dets}

        for det in dets:
            tte_fm = 'glg_tte_' + det + '_' + burstid + '_v*'
            tte_file = cls.findfile(link, tte_fm)
            tte_dict[det].append(tte_file)

            cspec_pha_fm = 'glg_cspec_' + det + '_' + burstid + '_v*pha'
            cspec_pha_file = cls.findfile(link, cspec_pha_fm)
            cspec_pha_dict[det].append(cspec_pha_file)

            cspec_rsp_fm = 'glg_cspec_' + det + '_' + burstid + '_v*rsp2'
            cspec_rsp_file = cls.findfile(link, cspec_rsp_fm)
            cspec_rsp_dict[det].append(cspec_rsp_file)

            ctime_pha_fm = 'glg_ctime_' + det + '_' + burstid + '_v*pha'
            ctime_pha_file = cls.findfile(link, ctime_pha_fm)
            ctime_pha_dict[det].append(ctime_pha_file)

            ctime_rsp_fm = 'glg_ctime_' + det + '_' + burstid + '_v*rsp2'
            ctime_rsp_file = cls.findfile(link, ctime_rsp_fm)
            ctime_rsp_dict[det].append(ctime_rsp_file)
            
        healpix_fm = 'glg_healpix_all_' + burstid + '_v*'
        healpix_file = cls.findfile(link, healpix_fm)
        healpix_list.append(healpix_file)

        rtv_res = {'burstid': burstid, 'datapath': datapath, 'tte': tte_dict, 
                   'cspec_pha': cspec_pha_dict, 'cspec_rsp': cspec_rsp_dict, 
                   'ctime_pha': ctime_pha_dict, 'ctime_rsp': ctime_rsp_dict, 
                   'healpix': healpix_list}
        rtv = cls(rtv_res)
        return rtv


    @classmethod
    def from_utc(cls, utc, t1, t2, datapath=None):
        
        if datapath is None:
            datapath = '/data-share/SSS_SHARE/DATA/SHAO_DOWN/data'

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
        dates_perH = pd.date_range(start, stop, freq='H')
        dates_perD = pd.date_range(start[:10], stop[:10], freq='D')

        if len(dates_perH) > 2:
            msg = 'the time range is too long'
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)

        poshist_list = []
        dets = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        tte_dict = {det:[] for det in dets}
        cspec_pha_dict = {det:[] for det in dets}
        ctime_pha_dict = {det:[] for det in dets}

        for date in enumerate(dates_perH):
            year = '%d' % date.year
            month = '%.2d' % date.month
            day = '%.2d' % date.day
            hour = '%.2d' % date.hour

            link = datapath + '/' + year + '/' + month + '/' + day + '/'

            for det in dets:
                tte_fm = 'glg_tte_' + det + '_' + year[-2:] + month + day + '_' + hour + 'z_v*'
                tte_file = cls.findfile(link, tte_fm)
                tte_dict[det].append(tte_file)

        for date in enumerate(dates_perD):
            year = '%d' % date.year
            month = '%.2d' % date.month
            day = '%.2d' % date.day

            link = datapath + '/' + year + '/' + month + '/' + day + '/'

            for det in dets:
                cspec_pha_fm = 'glg_cspec_' + det + '_' + year[-2:] + month + day + '_v*pha'
                cspec_pha_file = cls.findfile(link, cspec_pha_fm)
                cspec_pha_dict[det].append(cspec_pha_file)

                ctime_pha_fm = 'glg_ctime_' + det + '_' + year[-2:] + month + day + '_v*pha'
                ctime_pha_file = cls.findfile(link, ctime_pha_fm)
                ctime_pha_dict[det].append(ctime_pha_file)

            poshist_fm = 'glg_poshist_all_' + year[-2:] + month + day + '_v*'
            poshist_file = cls.findfile(link, poshist_fm)
            poshist_list.append(poshist_file)

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
            datapath = '/fermi/data/gbm/bursts'

        year = '20' + burstid[2:4]
        month = burstid[4:6]

        link = datapath + '/' + year + '/' + month + '/' + burstid + '/'

        grd_evt_fm = 'g_evt_' + burstid
        grd_evt_file = cls.findfile(link, grd_evt_fm)

        grd_bspec_fm = 'g_bspec_' + burstid
        grd_bspec_file = cls.findfile(link, grd_bspec_fm)

        grd_btime_fm = 'g_btime_' + burstid
        grd_btime_file = cls.findfile(link, grd_btime_fm)

        posatt_fm = '_posatt_' + burstid
        posatt_file = cls.findfile(link, posatt_fm)

        rtv_res = {'burstid': burstid, 'datapath': datapath, 
                   'grd_evt': [grd_evt_file], 'grd_bspec': [grd_bspec_file], 
                   'grd_btime': [grd_btime_file], 'posatt': [posatt_file]}
        rtv = cls(rtv_res)
        return rtv


    @classmethod
    def from_utc(cls, utc, t1, t2, datapath=None):
        
        if datapath is None:
            datapath = '/data-share/SSS_SHARE/DATA/GECAM/daily'

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
        dates_perH = pd.date_range(start, stop, freq='H')

        if len(dates_perH) > 2:
            msg = 'the time range is too long'
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)

        grd_evt_list = []
        grd_bspec_list = []
        grd_btime_list = []
        posatt_list = []

        for date in enumerate(dates_perH):
            year = '%d' % date.year
            month = '%.2d' % date.month
            day = '%.2d' % date.day
            hour = '%.2d' % date.hour

            grd_evt_link = datapath + '/' + year + '/' + month + '/' + day + '/GECAM_B/GRD_evt/'
            grd_evt_fm = 'g_evt_' + year[-2:] + month + day + '_' + hour
            grd_evt_file = cls.findfile(grd_evt_link, grd_evt_fm)
            grd_evt_list.append(grd_evt_file)

            grd_bspec_link = datapath + '/' + year + '/' + month + '/' + day + '/GECAM_B/GRD_bspec/'
            grd_bspec_fm = 'g_bspec_' + year[-2:] + month + day + '_' + hour
            grd_bspec_file = cls.findfile(grd_bspec_link, grd_bspec_fm)
            grd_bspec_list.append(grd_bspec_file)

            grd_btime_link = datapath + '/' + year + '/' + month + '/' + day + '/GECAM_B/GRD_btime/'
            grd_btime_fm = 'g_btime_' + year[-2:] + month + day + '_' + hour
            grd_btime_file = cls.findfile(grd_btime_link, grd_btime_fm)
            grd_btime_list.append(grd_btime_file)

            posatt_link = datapath+ '/' + year + '/' + month + '/' + day + '/GECAM_B/posatt/'
            posatt_fm = '_posatt_' + year[-2:] + month + day + '_' + hour
            posatt_file = cls.findfile(posatt_link, posatt_fm)
            posatt_list.append(posatt_file)

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
            datapath = '/Users/jyang/Documents/research_works/EinsteinProbe/WXT'
            
        link = datapath + '/' + obsname + '/'
        
        evt_fm = 'ep*_cl.evt'
        evt_file = cls.findfile(link, evt_fm)

        rmf_fm = 'ep*.rmf'
        rmf_file = cls.findfile(link, rmf_fm)
        
        armreg_fm = 'ep*arm.reg'
        armreg_file = cls.findfile(link, armreg_fm)
        
        arf_fm = 'ep*' + srcid + '.arf'
        arf_file = cls.findfile(link, arf_fm)
        
        bkreg_fm = 'ep*' + srcid + 'bk.reg'
        bkreg_file = cls.findfile(link, bkreg_fm)
        
        reg_fm = 'ep*' + srcid + '.reg'
        reg_file = cls.findfile(link, reg_fm)
        
        if reg_file is None:
            
            reg_file = bkreg_file[:-6] + '.reg'
            
            with open(bkreg_file, 'r') as f_obj:
                items = f_obj.readlines()[0].split()
                
            with open(reg_file, 'w') as f_obj:
                f_obj.write('circle ' + items[1] + ' 67)')

        rtv_res = {'satelite': 'WXT', 'obsname': obsname, 'srcid': srcid, 
                   'evt': evt_file, 'rmf': rmf_file, 'arf': arf_file, 
                   'armreg': armreg_file, 'reg': reg_file, 'bkreg': bkreg_file}
        
        rtv = cls(rtv_res)
        
        return rtv


    @classmethod
    def from_fxtobs(cls, obsname, srcid, datapath=None): 
        
        if datapath is None:
            datapath = '/Users/jyang/Documents/research_works/EinsteinProbe/FXT'
            
        link = datapath + '/' + obsname + '/'
        
        evt_fm = 'fxt_*_cl_*.fits'
        evt_file = cls.findfile(link, evt_fm)

        rmf_fm = 'fxt_*.rmf'
        rmf_file = cls.findfile(link, rmf_fm)
        
        arf_fm = 'fxt_*.arf'
        arf_file = cls.findfile(link, arf_fm)
        
        reg_fm = 'fxt_*' + srcid + '.reg'
        reg_file = cls.findfile(link, reg_fm)
        
        bkreg_fm = 'fxt_*' + srcid + 'bk.reg'
        bkreg_file = cls.findfile(link, bkreg_fm)
        
        rtv_res = {'satelite': 'FXT', 'obsname': obsname, 'srcid': srcid, 
                   'evt': evt_file, 'rmf': rmf_file, 'arf': arf_file, 
                   'reg': reg_file, 'bkreg': bkreg_file}
        
        rtv = cls(rtv_res)
        
        return rtv


    @classmethod
    def from_fxtsrc(cls, obsname, datapath=None):
        
        if datapath is None:
            datapath = '/Users/jyang/Documents/research_works/EinsteinProbe/FXT'
            
        link = datapath + '/' + obsname + '/'
        
        src_evt_fm = 'fxt_*_src*cl_*.fits'
        src_evt_file = cls.findfile(link, src_evt_fm)
        
        bkg_evt_fm = 'fxt_*_bkg*cl_*.fits'
        bkg_evt_file = cls.findfile(link, bkg_evt_fm)
        
        src_spec_fm = 'fxt_*_src_*.pha'
        src_spec_file = cls.findfile(link, src_spec_fm)
        
        bkg_spec_fm = 'fxt_*_bkg_*.pha'
        bkg_spec_file = cls.findfile(link, bkg_spec_fm)

        rmf_fm = 'fxt_*.rmf'
        rmf_file = cls.findfile(link, rmf_fm)
        
        arf_fm = 'fxt_*.arf'
        arf_file = cls.findfile(link, arf_fm)
        
        rtv_res = {'satelite': 'FXT', 'obsname': obsname, 
                   'src_evt': src_evt_file, 'bkg_evt': bkg_evt_file, 
                   'src_spec': src_spec_file, 'bkg_spec': bkg_spec_file, 
                   'rmf': rmf_file, 'arf': arf_file}
        
        rtv = cls(rtv_res)
        
        return rtv
