import os
import numpy as np
from gbm.data import PosHist
import matplotlib.pyplot as plt
from gbm.plot.lib import sky_point
from gbm.plot import SkyPlot, EarthPlot
from .retrieve import gbmRetrieve



class gbmGeometry(object):
    
    def __init__(self, file):
        
        self._file = file
        
        self._read()
    
    
    @classmethod
    def from_utc(cls, utc):
        
        rtv = gbmRetrieve.from_utc(utc=utc, t1=0, t2=0)
        
        poshist_file = rtv.rtv_res['poshist']
        
        msg = 'no retrieved poshist file'
        assert poshist_file != [], msg
        
        return cls(utc, poshist_file)
    
    
    def _read(self):
        
        self._poshist = [PosHist.open(file) for file in self._file]
        
        self._poshist_time_range = np.array([poshist.time_range for poshist in self._poshist])
    
    
    @property
    def file(self):
        
        return self._file
    
    
    @file.setter
    def file(self, new_file):
        
        self._file = new_file
        
        self._read()
        
        
    def poshist(self, met):

        fi = -1
        for i, (start, end) in enumerate(self._poshist_time_range):
            if (met >= start) & (met <= end):
                fi = i
                break
        
        if fi == -1: raise ValueError(f'uncovered time: {met}')
                
        return self._poshist[fi]

    
    def saa_passage(self, met):
        
        met = np.array(met)
        
        results = np.full(met.shape, None)
        
        for i, (start, end) in enumerate(self._poshist_time_range):
            mask = (met >= start) & (met <= end)
            results[mask] = self._poshist[i].get_saa_passage(met[mask])
        
        return results
    
    
    def location_visible(self, ra, dec, met):
        
        met = np.array(met)
        
        results = np.full(met.shape, None)
        
        for i, (start, end) in enumerate(self._poshist_time_range):
            mask = (met >= start) & (met <= end)
            results[mask] = self._poshist[i].location_visible(ra, dec, met[mask])
        
        return results
    
    
    def sun_visible(self, met):
        
        met = np.array(met)
        
        results = np.full(met.shape, None)
        
        for i, (start, end) in enumerate(self._poshist_time_range):
            mask = (met >= start) & (met <= end)
            results[mask] = self._poshist[i].get_sun_visibility(met[mask])
        
        return results


    def detector_angle(self, ra, dec, det, met):
        
        met = np.array(met)
        
        results = np.full(met.shape, None)
        
        for i, (start, end) in enumerate(self._poshist_time_range):
            mask = np.where((met >= start) & (met <= end))[0]
            for j in mask:
                results[j] = self._poshist[i].detector_angle(ra, dec, det, met[j])
        
        return results
    
    
    def extract_skymap(self, ra, dec, met, savepath='./geometry'):
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        skyplot = SkyPlot()
        skyplot.add_poshist(self.poshist(met), trigtime=met)
    
        sky_point(ra, dec, skyplot.ax, marker='*', s=75, facecolor='r', edgecolor='r', flipped=True, fermi=False)
        plt.savefig(savepath + '/sky_map.pdf')
        
        
    def extract_earthmap(self, met, dt=1000, savepath='./geometry'):
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        earthplot = EarthPlot()
        earthplot.add_poshist(self.poshist(met), trigtime=met, time_range=(met-dt, met+dt))
        
        plt.savefig(savepath + '/earth_map.pdf')
