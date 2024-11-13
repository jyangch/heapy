import os
import numpy as np
from gbm_drm_gen.drmgen_tte import DRMGenTTE
from .retrieve import gbmRetrieve
from ..util.file import copy, remove
from ..util.time import fermi_utc_to_met



class gbmResponse(object):
    
    def __init__(self, 
                 utc, 
                 det, 
                 cspec_file, 
                 poshist_file):
        
        self.utc = utc
        self.det = det
        self.cspec_file = cspec_file
        self.poshist_file = poshist_file
    
    
    @classmethod
    def from_utc(cls, utc, det):
        
        dets = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
        msg = 'invalid detector: %s' % det
        assert det in dets, msg
        
        rtv = gbmRetrieve.from_utc(utc=utc, t1=0, t2=0)
        
        cspec_file = rtv.rtv_res['cspec_pha'][det]
        
        msg = 'no retrieved cspec file'
        assert cspec_file != [], msg
        
        poshist_file = rtv.rtv_res['poshist']
        
        msg = 'no retrieved poshist file'
        assert poshist_file != [], msg
        
        return cls(utc, det, cspec_file[0], poshist_file[0])
    
    
    def extract_response(self, ra, dec, spec_slices, savepath='./spectrum'):
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
        cwd = os.getcwd()
        os.chdir(savepath)
        
        met = fermi_utc_to_met(self.utc)
        
        lslices = np.array(spec_slices)[:, 0]
        rslices = np.array(spec_slices)[:, 1]
    
        for _, (l, r) in enumerate(zip(lslices, rslices)):
            new_l = '{:+.2f}'.format(l).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            new_r = '{:+.2f}'.format(r).replace('-', 'm').replace('.', 'd').replace('+', 'p')
            
            file_name = '-'.join([new_l, new_r])
        
            drm = DRMGenTTE(
                det_name=self.det, 
                time=(l + r) / 2, 
                cspecfile=self.cspec_file, 
                poshist=self.poshist_file, 
                T0=met,
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
