import os
import gzip
import shutil
import warnings
import pandas as pd
from astropy.time import Time, TimeDelta
from .filefinder import FileFinder
from ..util.data import msg_format



class Retrieve(object):
    
    def __init__(self, rtv_res):
        
        self.rtv_res = rtv_res



class gbmRetrieve(Retrieve):

    def __init__(self, rtv_res):
        
        super().__init__(rtv_res)


    @classmethod
    def from_burst(cls, burstid, datapath=None, skip_tte=False, skip_healpix=False):
        
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
        # cspec_pha_dict = {}
        # cspec_rsp_dict = {}
        # ctime_pha_dict = {}
        # ctime_rsp_dict = {}

        if not skip_tte:
            for det in dets:
                tte_feature = 'glg_tte_' + det + '_' + burstid + '_v*fit'
                tte_file = ff.find(tte_feature)
                tte_dict[det].append(tte_file[-1] if tte_file else None)

                # cspec_feature = 'glg_cspec_' + det + '_' + burstid + '_v*pha'
                # cspec_file = ff.find(cspec_feature)
                # cspec_pha_dict[det] = cspec_file[-1] if cspec_file else None

                # cspec_feature = 'glg_cspec_' + det + '_' + burstid + '_v*rsp2'
                # cspec_file = ff.find(cspec_feature)
                # cspec_rsp_dict[det] = cspec_file[-1] if cspec_file else None

                # ctime_feature = 'glg_ctime_' + det + '_' + burstid + '_v*pha'
                # ctime_file = ff.find(ctime_feature)
                # ctime_pha_dict[det] = ctime_file[-1] if ctime_file else None

                # ctime_feature = 'glg_ctime_' + det + '_' + burstid + '_v*rsp2'
                # ctime_file = ff.find(ctime_feature)
                # ctime_rsp_dict[det] = ctime_file[-1] if ctime_file else None
            
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
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)

        poshist_list = []
        dets = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        tte_dict = {det:[] for det in dets}
        # cspec_pha_dict = {det:[] for det in dets}
        # ctime_pha_dict = {det:[] for det in dets}

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
                
                # for det in dets:
                #     cspec_feature = 'glg_cspec_' + det + '_' + year[-2:] + month + day + '_v*pha'
                #     cspec_file = ff.find(cspec_feature)
                #     cspec_pha_dict[det].append(cspec_file[-1] if cspec_file else None)

                #     ctime_feature = 'glg_ctime_' + det + '_' + year[-2:] + month + day + '_v*pha'
                #     ctime_file = ff.find(ctime_feature)
                #     ctime_pha_dict[det].append(ctime_file[-1] if ctime_file else None)

                poshist_feature = 'glg_poshist_all_' + year[-2:] + month + day + '_v*fit'
                poshist_file = ff.find(poshist_feature)
                poshist_list.append(poshist_file[-1] if poshist_file else None)

        rtv_res = {'utc': utc.value, 't1': t1.value, 't2': t2.value, 'datapath': datapath, 
                   'tte': tte_dict, 'poshist': poshist_list}
        
        rtv = cls(rtv_res)
        
        return rtv



class gecamRetrieve(Retrieve):

    def __init__(self, rtv_res):
        
        super().__init__(rtv_res)


    @classmethod
    def from_burst(cls, burstid, datapath=None):
        
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

    def __init__(self, rtv_res):
        
        super().__init__(rtv_res)
        
        
    @classmethod
    def from_utc(cls, utc, t1, t2, det, datapath=None):
        
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
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
            
        dets = ['%d' % i for i in range(0, 4)]
        msg = 'invalid detector: %s' % det
        assert det in dets, msg_format(msg)

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

    def __init__(self, rtv_res):
        
        super().__init__(rtv_res)
        
        
    @classmethod
    def from_wxtobs(cls, obsid, srcid, datapath=None): 
        
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

        rtv_res = {'satelite': 'WXT', 'obsid': obsid, 'srcid': srcid, 
                   'evt': evt, 'rmf': rmf, 'arf': arf, 
                   'armreg': armreg, 'reg': reg, 'bkreg': bkreg}
        
        rtv = cls(rtv_res)
        
        return rtv


    @classmethod
    def from_fxtobs(cls, obsid, module, datapath=None): 
        
        if datapath is None:
            datapath = '/Users/junyang/Data/ep/FXT'
            
        local_dir = datapath + '/' + obsid
        
        ff = FileFinder(local_dir=local_dir)
        
        evt_feature = f'fxt_{module}_*_cl_*.fits'
        evt_file = ff.find(evt_feature)
        evt = evt_file[-1] if evt_file else None
        
        reg_feature = f'fxt_{module}_*.reg'
        reg_file = ff.find(reg_feature)
        reg = reg_file[-1] if reg_file else None
        
        bkreg_feature = f'fxt_{module}_*bk.reg'
        bkreg_file = ff.find(bkreg_feature)
        bkreg = bkreg_file[-1] if bkreg_file else None
        
        rtv_res = {'satelite': 'FXT', 'obsid': obsid, 'module': module, 
                   'evt': evt, 'reg': reg, 'bkreg': bkreg}
        
        rtv = cls(rtv_res)
        
        return rtv


    @classmethod
    def from_fxtsrc(cls, obsid, module, datapath=None):
        
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

    def __init__(self, rtv_res):
        
        super().__init__(rtv_res)
        
        
    @classmethod
    def from_xrtobs(cls, obsid, mode, datapath=None): 
        
        assert mode in ['wt', 'pc'], 'xrt mode only allowed to be wt or pc'
        
        if datapath is None:
            datapath = '/Users/junyang/Data/swift'
            
        local_dir = datapath + '/' + obsid + '/xrt/event'
        
        ff = FileFinder(local_dir=local_dir)
        
        try:
            evt_feature = f'sw{obsid}x{mode}*po_cl.evt'
            evt_file = ff.find(evt_feature)
            evt = evt_file[-1] if evt_file else None
        except:
            evtgz_feature = f'sw{obsid}x{mode}*po_cl.evt.gz'
            evtgz_file = ff.find(evtgz_feature)
            evtgz = evtgz_file[-1] if evtgz_file else None
            evt = evtgz[:-3]
            with gzip.open(evtgz, 'rb') as f_in:
                with open(evt, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        bkreg_feature = f'sw{obsid}x{mode}*pobk.reg'
        bkreg_file = ff.find(bkreg_feature)
        bkreg = bkreg_file[-1] if bkreg_file else None
        
        reg_feature = f'sw{obsid}x{mode}*po.reg'
        reg_file = ff.find(reg_feature)
        reg = reg_file[-1] if reg_file else None

        rtv_res = {'satelite': 'XRT', 'obsid': obsid, 'mode': mode,
                   'evt': evt, 'reg': reg, 'bkreg': bkreg}
        
        rtv = cls(rtv_res)
        
        return rtv
    
    
    @classmethod
    def from_batobs(cls, obsid, datapath=None):
        
        if datapath is None:
            datapath = '/Users/junyang/Data/swift'
            
        local_dir = datapath + '/' + obsid
        
        ff = FileFinder(local_dir=local_dir + '/bat/event')
        try:
            ufevt_feature = 'sw*bevshsp_uf.evt'
            ufevt_file = ff.find(ufevt_feature)
            ufevt = ufevt_file[-1] if ufevt_file else None
        except:
            ufevtgz_feature = 'sw*bevshsp_uf.evt.gz'
            ufevtgz_file = ff.find(ufevtgz_feature)
            ufevtgz = ufevtgz_file[-1] if ufevtgz_file else None
            ufevt = ufevtgz[:-3]
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
