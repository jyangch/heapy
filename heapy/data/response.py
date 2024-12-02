import os
import numpy as np
from astropy.io import fits
from gbm_drm_gen.drmgen_tte import DRMGenTTE
from .retrieve import gbmRetrieve
from ..util.file import copy, remove
from ..util.time import fermi_utc_to_met, fermi_met_to_utc



class gbmResponse(object):
    
    det_name_lookup = {
        "NAI_00": "n0",
        "NAI_01": "n1",
        "NAI_02": "n2",
        "NAI_03": "n3",
        "NAI_04": "n4",
        "NAI_05": "n5",
        "NAI_06": "n6",
        "NAI_07": "n7",
        "NAI_08": "n8",
        "NAI_09": "n9",
        "NAI_10": "na",
        "NAI_11": "nb",
        "BGO_00": "b0",
        "BGO_01": "b1"}
    
    def __init__(self, 
                 tte_file, 
                 poshist_file):

        self.tte_file = tte_file
        self.poshist_file = poshist_file
        
        self._read()
    
    
    @classmethod
    def from_utc(cls, utc, det):
        
        dets = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        msg = 'invalid detector: %s' % det
        assert det in dets, msg
        
        rtv = gbmRetrieve.from_utc(utc=utc, t1=0, t2=0)
        
        tte_file = rtv.rtv_res['tte'][det]
        
        msg = 'no retrieved tte file'
        assert tte_file != [], msg
        
        poshist_file = rtv.rtv_res['poshist']
        
        msg = 'no retrieved poshist file'
        assert poshist_file != [], msg
        
        return cls(utc, det, tte_file[0], poshist_file[0])
    
    
    def _read(self):
        
        with fits.open(self.tte_file) as f:
            
            self._det = f['PRIMARY'].header['DETNAM']
            self._timezero = np.min(f['EVENTS'].data['TIME'])
    
    
    @property
    def det(self):
            
        return gbmResponse.det_name_lookup[self._det]


    @property
    def timezero(self):
        
        return self._timezero
    
    
    @timezero.setter
    def timezero(self, new_timezero):
        
        if isinstance(new_timezero, float):
            self._timezero = new_timezero
        else:
            raise ValueError('not expected type for timezero')
        
        
    @property
    def timezero_utc(self):
        
        return fermi_met_to_utc(self.timezero)
    
    
    @property
    def spec_slices(self):
        
        try:
            return self._spec_slices
        except AttributeError:
            return [[-1, 1]]
        
        
    @spec_slices.setter
    def spec_slices(self, new_spec_slice):
        
        if isinstance(new_spec_slice, list):
            if isinstance(new_spec_slice[0], list):
                self._spec_slices = new_spec_slice
            else:
                raise ValueError('not expected spec_slices type')
        else:
            raise ValueError('not expected spec_slices type')
    
    
    def extract_response(self, ra, dec, savepath='./spectrum'):
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
        cwd = os.getcwd()
        os.chdir(savepath)
        
        lslices = np.array(self.spec_slices)[:, 0]
        rslices = np.array(self.spec_slices)[:, 1]
    
        for _, (l, r) in enumerate(zip(lslices, rslices)):
            new_l = '{:+.2f}'.format(l).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_r = '{:+.2f}'.format(r).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            
            file_name = '_'.join([new_l, new_r])
        
            drm = DRMGenTTE(
                tte_file=self.tte_file, 
                time=(l + r) / 2, 
                poshist=self.poshist_file, 
                T0=self.timezero, 
                mat_type=2, 
                occult=False)
        
            drm.to_fits(
                ra=ra, 
                dec=dec, 
                filename=file_name, 
                overwrite=True)
            
            copy(f'{file_name}_{self.det}.rsp', f'{file_name}.rsp')
            remove(f'{file_name}_{self.det}.rsp')
            
        os.chdir(cwd)
