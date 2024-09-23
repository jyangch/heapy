import numpy as np
from astropy import table
from astropy.io import fits
from astropy import units as u
from .event import Event
from .retrieve import gbmRetrieve, gecamRetrieve
from ..util.data import msg_format
from ..util.time import fermi_met_to_utc, gecam_met_to_utc



class Reduction(object):
    
    def __init__(self, file):
        
        self._file = file
        
        self._read()
        
        
    def _read(self):
        
        if self.file is not None:
            hdu = fits.open(self.file)
            self._event = table.Table.read(hdu['EVENTS'])
            self._ebound = table.Table.read(hdu['EBOUNDS'])
            self._gti = table.Table.read(hdu['GTI'])
            
            self._timezero = np.min(self._event['TIME'])
            self._filter = Filter(self._event)
            
    
    @staticmethod
    def _ch_to_energy(pi, ch, e1, e2):
        
        energy = np.zeros_like(pi)
        
        for i, chi in enumerate(ch):
            chi_idx = np.where(pi == chi)[0]
            chi_energy = Reduction._energy_of_ch(len(chi_idx), e1[i], e2[i])
            energy[chi_idx] = chi_energy
            
        return energy


    @staticmethod
    def _energy_of_ch(n, e1, e2):
        
        energy_arr = np.random.random_sample(n)
        energy = e1 + (e2 - e1) * energy_arr
        
        return energy


    @property
    def file(self):
        
        return self._file
    
    
    @file.setter
    def file(self, new_file):
        
        self._file = new_file
        
        self._read()
    
    
    @property
    def event(self):
        
        return self._filter.evt
    
    
    @property
    def gti(self):
        
        return self._gti
    
    
    @property
    def ebound(self):
        
        return self._ebound
    
    
    @property
    def timezero(self):
        
        return self._timezero
    
    
    @timezero.setter
    def timezero(self, new_timezero):
        
        if isinstance(new_timezero, float):
            self._timezero = new_timezero
            
        else:
            msg = 'not expected type for timezero'
            raise ValueError(msg_format(msg))
    
    
    @property
    def filter_info(self):
        
        self._filter.info()
        
        return None
    
    
    def filter(self, expr):
        
        self._filter.eval(expr)
        
        
    def time_filter(self, t1, t2):

        met_t1 = self.timezero + t1
        met_t2 = self.timezero + t2
            
        expr = f'(TIME >= {met_t1}) * (TIME <= {met_t2})'
        self._filter.eval(expr)


    def energy_filter(self, e1, e2):

        expr = f'(ENERGY >= {e1}) * (ENERGY <= {e2})'
        self._filter.eval(expr)
        
        
    def clear_filter(self):
        
        self._filter.clear()


    @property
    def to_event(self):
        
        return Event(self)


    def __add__(self, other):
        
        if not isinstance(other, Reduction):
            raise ValueError('"other" argument should be Reduction type')
        
        merge_event = table.vstack([self.event, other.event])
        merge_event.sort('TIME')
        
        cls = Reduction(file=None)
        cls._event = merge_event
        cls._timezero = self.timezero
        cls._filter = Filter(cls._event)
    
        return cls


    def __radd__(self, other):
        
        return self.__add__(other)



class gbmTTE(Reduction):

    def __init__(self, file):
        
        super().__init__(file)


    @classmethod
    def from_utc(cls, utc, det):
        
        dets = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        msg = 'invalid detector: %s' % det
        assert det in dets, msg_format(msg)
        
        rtv = gbmRetrieve.from_utc(utc=utc, t1=-200, t2=500)

        ttefile = rtv.rtv_res['tte'][det]
            
        msg = 'no retrieved tte file'
        assert ttefile != [], msg_format(msg)
        
        return cls(ttefile)
    

    def _read(self):
        
        event_list = list()
        gti_list = list()
        
        for i in range(len(self.file)):
            hdu = fits.open(self.file[i])
            event = table.Table.read(hdu['EVENTS'])
            ebound = table.Table.read(hdu['EBOUNDS'])
            gti = table.Table.read(hdu['GTI'])
            hdu.close()
            
            pha = np.array(event['PHA']).astype(int)
            ch = np.array(ebound['CHANNEL']).astype(int)
            emin = np.array(ebound['E_MIN'])
            emax = np.array(ebound['E_MAX'])
            energy = Reduction._ch_to_energy(pha, ch, emin, emax)
            event['ENERGY'] = energy * u.keV
            event['DEAD_TIME'] = (pha == 127) * 10.0 + (pha < 127) * 2.6
            
            event_list.append(event)
            gti_list.append(gti)
            
        self._event = table.vstack(event_list)
        self._gti = table.vstack(gti_list)
        self._ebound = ebound

        self._event = table.unique(self._event, keys=['TIME', 'PHA'])
        self._gti = table.unique(self._gti, keys=['START', 'STOP'])

        self._event.sort('TIME')
        self._gti.sort('START')
        
        self._timezero = np.min(self._event['TIME'])
        
        self._filter = Filter(self._event)
        
        
    @property
    def timezero_utc(self):
        
        return fermi_met_to_utc(self.timezero)
        
        
    def pha_filter(self, p1, p2):
        
        expr = f'(PHA >= {p1}) * (PHA <= {p2})'
        self._filter.eval(expr)


    def lc_filter(self, type='NaI'):
        
        if type == 'NaI':
            self.energy_filter(10, 1000)
        elif type == 'BGO':
            self.energy_filter(300, 38000)
        else:
            raise ValueError(f'not expected type: {type}')



class gecamEVT(Reduction):

    def __init__(self, file, det, gain_type):
        
        self._file = file
        self.det = det
        self.gain_type = gain_type
        
        self._read()


    @classmethod
    def from_utc(cls, utc, det, gain_type):
        
        rtv = gecamRetrieve.from_utc(utc=utc, t1=-200, t2=500)

        evtfile = rtv.rtv_res['grd_evt']
            
        msg = 'no retrieved evt file'
        assert evtfile != [], msg_format(msg)
        
        return cls(evtfile, det, gain_type)
        
        
    @property
    def det(self):
        
        return self._det
    
    
    @det.setter
    def det(self, new_det):
        
        dets = ['%02d' % i for i in range(1, 26)]
        msg = 'invalid detector: %s' % new_det
        assert new_det in dets, msg_format(msg)
        
        self._det = new_det


    @property
    def gain_type(self):
        
        return self._gain_type
    
    
    @gain_type.setter
    def gain_type(self, new_gain_type):
        
        gain_types = ['HG', 'LG']
        msg = 'invalid gain type: %s' % new_gain_type
        assert new_gain_type in gain_types, msg_format(msg)
        
        self._gain_type = new_gain_type


    def _read(self):
        
        event_list = list()
        gti_list = list()
        
        for i in range(len(self.file)):
            hdu = fits.open(self.file[i])
            event = table.Table.read(hdu[f'EVENTS{self.det}'])
            ebound = table.Table.read(hdu['EBOUNDS'])
            gti = table.Table.read(hdu['GTI'])
            hdu.close()
            
            gt = event['GAIN_TYPE']
            if self.gain_type == 'HG':
                event = event[gt == 0]
            else:
                event = event[gt == 1]
            
            pi = np.array(event['PI']).astype(int)
            ch = np.array(ebound['CHANNEL']).astype(int)
            emin = np.array(ebound['E_MIN'])
            emax = np.array(ebound['E_MAX'])
            energy = Reduction._ch_to_energy(pi, ch, emin, emax)
            event['ENERGY'] = energy * u.keV
            
            event_list.append(event)
            gti_list.append(gti)
            
        self._event = table.vstack(event_list)
        self._gti = table.vstack(gti_list)

        self._event = table.unique(self._event, keys=['TIME', 'PI', 'GAIN_TYPE'])
        self._gti = table.unique(self._gti, keys=['START', 'STOP'])

        self._event.sort('TIME')
        self._gti.sort('START')
        
        self._timezero = np.min(self._event['TIME'])
        
        self._filter = Filter(self._event)
        
        
    @property
    def timezero_utc(self):
        
        return gecam_met_to_utc(self.timezero)
        
        
    def pha_filter(self, p1, p2):
        
        expr = f'(PI >= {p1}) * (PI <= {p2})'
        self._filter.eval(expr)


    def gain_filter(self, gt):
        
        expr = f'GAIN_TYPE == {gt}'
        self._filter.eval(expr)
        
        
    def lc_filter(self):
        
        if self.gain_type == 'HG':
            self.pha_filter(45, 204)
        else:
            self.pha_filter(205, 447)



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
