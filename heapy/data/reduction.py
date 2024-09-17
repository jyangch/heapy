import numpy as np
from astropy import table
from astropy.io import fits
from astropy import units as u
from .retrieve import gbmRetrieve, gecamRetrieve
from ..util.data import msg_format, ch_to_energy
from ..util.time import fermi_utc_to_met, gecam_utc_to_met



class gbmTTE(object):

    def __init__(self, ttefile):
        
        self._ttefile = ttefile
        
        self._read()


    @classmethod
    def from_rtv(cls, rtv, det):
        
        dets = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        msg = 'invalid detector: %s' % det
        assert det in dets, msg_format(msg)

        if isinstance(rtv, gbmRetrieve):
            ttefile = rtv.rtv_res['tte'][det]
        elif isinstance(rtv, dict):
            if 'tte' in rtv:
                ttefile = rtv['tte'][det]
            else:
                msg = 'tte is not a key of rvt'
                raise ValueError(msg_format(msg))
        else:
            msg = 'rvt is not the expected format'
            raise ValueError(msg_format(msg))
            
        msg = 'no retrieved tte file'
        assert ttefile != [], msg_format(msg)
        
        return cls(ttefile)
    
    
    @property
    def ttefile(self):
        
        return self._ttefile
    
    
    @ttefile.setter
    def ttefile(self, new_ttefile):
        
        self._ttefile = new_ttefile
        
        self._read()
        
        
    def _read(self):
        
        event_list = list()
        gti_list = list()
        
        for i in range(len(self.ttefile)):
            hdu = fits.open(self.ttefile[i])
            event = table.Table.read(hdu['EVENTS'])
            ebound = table.Table.read(hdu['EBOUNDS'])
            gti = table.Table.read(hdu['GTI'])
            hdu.close()
            
            pha = np.array(event['PHA']).astype(int)
            ch = np.array(ebound['CHANNEL']).astype(int)
            emin = np.array(ebound['E_MIN'])
            emax = np.array(ebound['E_MAX'])
            energy = ch_to_energy(pha, ch, emin, emax)
            event['ENERGY'] = energy * u.keV
            
            event_list.append(event)
            gti_list.append(gti)
            
        self._event = table.vstack(event_list)
        self._gti = table.vstack(gti_list)

        self._event = table.unique(self._event, keys=['TIME', 'PHA'])
        self._gti = table.unique(self._gti, keys=['START', 'STOP'])

        self._event.sort('TIME')
        self._gti.sort('START')
        
        self._filter = Filter(self._event)
        
    
    @property
    def event(self):
        
        return self._filter.evt
    
    
    @property
    def filter_info(self):
        
        self._filter.info()
        
        return None
    
    
    def filter(self, expr):
        
        self._filter.eval(expr)
    
    
    def time_filter(self, t1, t2, utc=None):
        
        if utc is None:
            met_t1 = t1
            met_t2 = t2
        else:
            met = fermi_utc_to_met(utc)
            met_t1 = round(met + t1, 4)
            met_t2 = round(met + t2, 4)
            
        expr = f'(TIME >= {met_t1}) * (TIME <= {met_t2})'
        self._filter.eval(expr)


    def energy_filter(self, e1, e2):

        expr = f'(ENERGY >= {e1}) * (ENERGY <= {e2})'
        self._filter.eval(expr)
        
        
    def pha_filter(self, p1, p2):
        
        expr = f'(PHA >= {p1}) * (PHA <= {p2})'
        self._filter.eval(expr)


    def clear_filter(self):
        
        self._filter.clear()
        
        
    def to_lc(self):
        
        pass
    
    
    def to_pha(self):
        
        pass
        
        
        
class gecamEVT(object):

    def __init__(self, evtfile, det):
        
        self._evtfile = evtfile
        self._det = det
        
        self._read()


    @classmethod
    def from_rtv(cls, rtv, det):
        
        dets = ['%02d' % i for i in range(1, 26)]
        msg = 'invalid detector: %s' % det
        assert det in dets, msg_format(msg)

        if isinstance(rtv, gecamRetrieve):
            evtfile = rtv.rtv_res['grd_evt']
        elif isinstance(rtv, dict):
            if 'tte' in rtv:
                evtfile = rtv['grd_evt']
            else:
                msg = 'grd_evt is not a key of rvt'
                raise ValueError(msg_format(msg))
        else:
            msg = 'rvt is not the expected format'
            raise ValueError(msg_format(msg))
            
        msg = 'no retrieved evt file'
        assert evtfile != [], msg_format(msg)
        
        return cls(evtfile, det)
    
    
    @property
    def evtfile(self):
        
        return self._evtfile
    
    
    @evtfile.setter
    def evtfile(self, new_evtfile):
        
        self._evtfile = new_evtfile
        
        self._read()
        
        
    @property
    def det(self):
        
        return self._det
        
        
    def _read(self):
        
        event_list = list()
        gti_list = list()
        
        for i in range(len(self.evtfile)):
            hdu = fits.open(self.evtfile[i])
            event = table.Table.read(hdu[f'EVENTS{self.det}'])
            ebound = table.Table.read(hdu['EBOUNDS'])
            gti = table.Table.read(hdu['GTI'])
            hdu.close()
            
            pi = np.array(event['PI']).astype(int)
            ch = np.array(ebound['CHANNEL']).astype(int)
            emin = np.array(ebound['E_MIN'])
            emax = np.array(ebound['E_MAX'])
            energy = ch_to_energy(pi, ch, emin, emax)
            event['ENERGY'] = energy * u.keV
            
            event_list.append(event)
            gti_list.append(gti)
            
        self._event = table.vstack(event_list)
        self._gti = table.vstack(gti_list)

        self._event = table.unique(self._event, keys=['TIME', 'PI', 'GAIN_TYPE'])
        self._gti = table.unique(self._gti, keys=['START', 'STOP'])

        self._event.sort('TIME')
        self._gti.sort('START')
        
        self._filter = Filter(self._event)
        
        
    @property
    def event(self):
        
        return self._filter.evt
    
    
    @property
    def filter_info(self):
        
        self._filter.info()
        
        return None
        
    
    def filter(self, expr):
        
        self._filter.eval(expr)
        
        
    def time_filter(self, t1, t2, utc=None):
        
        if utc is None:
            met_t1 = t1
            met_t2 = t2
        else:
            met = gecam_utc_to_met(utc)
            met_t1 = round(met + t1, 4)
            met_t2 = round(met + t2, 4)
            
        expr = f'(TIME >= {met_t1}) * (TIME <= {met_t2})'
        self._filter.eval(expr)


    def energy_filter(self, e1, e2):

        expr = f'(ENERGY >= {e1}) * (ENERGY <= {e2})'
        self._filter.eval(expr)
        
        
    def pha_filter(self, p1, p2):
        
        expr = f'(PI >= {p1}) * (PI <= {p2})'
        self._filter.eval(expr)


    def gain_filter(self, gt):
        
        expr = f'GAIN_TYPE == {gt}'
        self._filter.eval(expr)
        
        
    def lc_filter(self):
        
        expr = '((GAIN_TYPE == 0) & (PI > 44) & (PI < 205)) | ((GAIN_TYPE == 1) & (PI >= 205) & (PI < 448))'
        self._filter.eval(expr)


    def clear_filter(self):
        
        self._filter.clear()



class Filter(object):
    
    def __init__(self, evt):
        
        msg = 'evt is not the type of astropy.tabel.Table'
        assert isinstance(evt, table.Table), msg_format(msg)
        
        self._evt = evt
        self.evt = self._evt.copy()
        
        self.exprs = []
        
    
    @classmethod
    def from_fits(cls, file, idx=None):
        
        hdu = fits.open(file)
        
        if idx is not None:
            evt = hdu[idx]
        else:
            try:
                evt = hdu['EVENTS']
            except KeyError:
                raise KeyError('EVENTS extension not found!')
            
        evt = table.Table.read(evt)
        hdu.close()

        return cls(evt)

    
    @property
    def tags(self):
        
        return self._evt.colnames
    
    
    @property
    def base(self):
        
        return {tag: self.evt[tag] for tag in self.tags}
    
    
    def info(self, tag=None):
        
        if tag is None:
            print(self.evt.info)
        else:
            msg = '%s is not one of tags' % tag
            assert tag in self.tags, msg_format(msg)
            
            print(self.evt[tag].info)
            
        print('\n'.join(self.exprs))


    def eval(self, expr):
        
        flt = eval(expr, {}, self.base)
        self.evt = self.evt[flt]
        
        self.exprs.append(expr)
    
    
    def clear(self):
        
        self.evt = self._evt.copy()
        
        self.exprs = []
