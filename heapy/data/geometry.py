import os
from gbm.data import PosHist
import matplotlib.pyplot as plt
from gbm.plot.lib import sky_point
from gbm.plot import SkyPlot, EarthPlot
from .retrieve import gbmRetrieve
from ..util.time import fermi_utc_to_met


class gbmGeometry(object):
    
    def __init__(self, 
                 utc, 
                 poshist_file):
        
        self.utc = utc
        self.met = fermi_utc_to_met(utc)
        self.poshist_file = poshist_file
        
        self.poshist = PosHist.open(poshist_file)
    
    
    @classmethod
    def from_utc(cls, utc):
        
        rtv = gbmRetrieve.from_utc(utc=utc, t1=0, t2=0)
        
        poshist_file = rtv.rtv_res['poshist']
        
        msg = 'no retrieved poshist file'
        assert poshist_file != [], msg
        
        return cls(utc, poshist_file[0])
    
    
    def saa_passage(self):
        
        return self.poshist.get_saa_passage(self.met)
    
    
    def location_visible(self, ra, dec):
        
        return self.poshist.location_visible(ra, dec, self.met)
    
    
    def detector_angle(self, ra, dec, det):
        
        return self.poshist.detector_angle(ra, dec, det, self.met)
    
    
    def extract_skymap(self, ra, dec, savepath='./geometry'):
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        skyplot = SkyPlot()
        skyplot.add_poshist(self.poshist, trigtime=self.met)
    
        sky_point(ra, dec, skyplot.ax, marker='*', s=75, facecolor='r', edgecolor='r', flipped=True, fermi=False)
        plt.savefig(savepath + '/sky_map.pdf')
        
        
    def extract_earthmap(self, dt=1000, savepath='./geometry'):

        earthplot = EarthPlot()
        earthplot.add_poshist(self.poshist, trigtime=self.met, time_range=(self.met-dt, self.met+dt))
        
        plt.savefig(savepath + '/earth_map.pdf')
