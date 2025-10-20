import numpy as np
import pandas as pd
from astropy import table
from astropy.io import fits
from daily_search.zjh_location import zjh_loc
from ..pipe.event import gbmTTE
from ..data.retrieve import gbmRetrieve



class gbmLocation(object):
    
    def __init__(self, tte_file, poshist_file):
        
        self.tte_file = tte_file
        self.poshist_file = poshist_file
        
        self._read()


    @classmethod
    def from_utc(cls, utc):
        
        rtv = gbmRetrieve.from_utc(utc=utc, t1=-500, t2=500)
        
        tte_file = rtv.rtv_res['tte']
        
        poshist_file = rtv.rtv_res['poshist']
        
        return cls(tte_file, poshist_file)
    
    
    def _read(self):
        
        self.tte_data = {}
        
        for det, file in self.tte_file.items():
            self.tte_data[det] = {}
            
            tte = gbmTTE(file)
            
            self.tte_data[det]['ch_E'] = pd.DataFrame({'CHANNEL': tte.channel, 
                                                       'E_MIN': tte.channel_emin,
                                                       'E_MAX': tte.channel_emax})
            
            self.tte_data[det]['events'] = pd.DataFrame({'TIME': np.array(tte.event['TIME']).astype(float),
                                                         'PHA': np.array(tte.event['PHA']).astype(int)})
        
        poshist_list = []
        for file in self.poshist_file:
            hdu = fits.open(file)
            pos = table.Table.read(hdu[1])
            poshist_list.append(pos)
            
        poshist = table.vstack(poshist_list)
        poshist = table.unique(poshist, keys=['SCLK_UTC'])
        poshist.sort('SCLK_UTC')
        
        col_names = ['SCLK_UTC','QSJ_1','QSJ_2','QSJ_3','QSJ_4','POS_X','POS_Y','POS_Z','SC_LAT','SC_LON']
        self.poshist_data = poshist[col_names].to_pandas()
    
    
    def extract_location(self, utc, t1, t2, binsize, snr=3, savepath='./location'):
        
        zjh_loc(utc, t1, t2, self.tte_data, self.poshist_data, savepath, binsize, snr)