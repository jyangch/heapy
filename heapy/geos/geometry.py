import os
import numpy as np
from astropy import table
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from scipy.interpolate import make_interp_spline
from astropy.coordinates import angular_separation, get_sun

from .skymap import gbmSkyMap
from .detector import gbmDetector
from ..data.retrieve import gbmRetrieve
from ..util.data import split_bool_mask
from ..util.time import fermi_met_to_utc
from ..util.coord import (
    gcrs_to_itrs, 
    calc_earth_angular_radius, 
    get_geocenter_radec, 
    calc_mcilwain_l, 
    spacecraft_to_radec, 
    radec_to_spacecraft)



class gbmGeometry(object):
    
    def __init__(self, file):
        
        self._file = file
        
        self._read()
    
    
    @classmethod
    def from_utc(cls, utc, t1=-200, t2=500):
        
        rtv = gbmRetrieve.from_utc(utc=utc, t1=t1, t2=t2)
        
        poshist_file = rtv.rtv_res['poshist']
        
        msg = 'no retrieved poshist file'
        assert poshist_file != [], msg
        
        return cls(poshist_file)
        
        
    def _read(self):
        
        if isinstance(self._file, str):
            self._file = [self._file]
        
        poshist_list = list()
        
        for i in range(len(self.file)):
            hdu = fits.open(self.file[i])
            poshist = table.Table.read(hdu['GLAST POS HIST'])
            hdu.close()
            
            poshist_list.append(poshist)
        
        self._poshist = table.vstack(poshist_list)
        self._poshist = table.unique(self._poshist, keys=['SCLK_UTC'])
        self._poshist.sort('SCLK_UTC')
        
        self._met = np.array(self._poshist['SCLK_UTC'])
        self._utc = fermi_met_to_utc(self._met)
        
        self._quaternion = np.column_stack((
            self._poshist['QSJ_1'], 
            self._poshist['QSJ_2'], 
            self._poshist['QSJ_3'], 
            self._poshist['QSJ_4']))
        self._quaternion_interp = make_interp_spline(self._met, self._quaternion, k=3, axis=0)
        
        self._coords_gcrs = np.column_stack((
            self._poshist['POS_X'], 
            self._poshist['POS_Y'], 
            self._poshist['POS_Z']))
        self._coords_gcrs_interp = make_interp_spline(self._met, self._coords_gcrs, k=3, axis=0)
        
        self._coords_itrs = gcrs_to_itrs(self._coords_gcrs, self._utc)
        self._coords_itrs_interp = make_interp_spline(self._met, self._coords_itrs, k=3, axis=0)
        
        self._earth_angular_radius = calc_earth_angular_radius(self._coords_itrs[:, 2])
        self._earth_angular_radius_interp = make_interp_spline(self._met, self._earth_angular_radius, k=3)
        
        self._geocenter_radec = get_geocenter_radec(self._coords_gcrs)
        self._geocenter_radec_interp = make_interp_spline(self._met, self._geocenter_radec, k=3, axis=0)
        
        self._vel = np.column_stack((
            self._poshist['VEL_X'], 
            self._poshist['VEL_Y'], 
            self._poshist['VEL_Z']))
        self._vel_interp = make_interp_spline(self._met, self._vel, k=3, axis=0)
        
        self._angvel = np.column_stack((
            self._poshist['WSJ_1'], 
            self._poshist['WSJ_2'], 
            self._poshist['WSJ_3']))
        self._angvel_interp = make_interp_spline(self._met, self._angvel, k=3, axis=0)
        
        self._sun_visible = ((self._poshist['FLAGS'] == 1) | (self._poshist['FLAGS'] == 3))
        self._sun_visible_interp = make_interp_spline(self._met, self._sun_visible.astype(float), k=3)
        
        self._saa_passage = ((self._poshist['FLAGS'] == 2) | (self._poshist['FLAGS'] == 3))
        self._saa_passage_interp = make_interp_spline(self._met, self._saa_passage.astype(float), k=3)
        
        self._gti = split_bool_mask(self._saa_passage, self._met, selection_value=False)
        self._sun_occulted = split_bool_mask(self._sun_visible, self._met, selection_value=False)


    @property
    def file(self):
        
        return self._file
    
    
    @file.setter
    def file(self, new_file):
        
        self._file = new_file
        
        self._read()
        
        
    @property
    def poshist(self):
        
        return self._poshist
    
    
    @property
    def telescope(self):
        
        return 'GLAST'
    
    
    @property
    def met(self):
        
        return self._met
    
    
    @property
    def utc(self):
        
        return self._utc
    
    
    @property
    def quaternion(self):
        
        return self._quaternion
    
    
    @property
    def coords_gcrs(self):
        
        return self._coords_gcrs
    
    
    @property
    def coords_itrs(self):
        
        return self._coords_itrs
    
    
    @property
    def earth_angular_radius(self):
        
        return self._earth_angular_radius
    
    
    @property
    def geocenter_radec(self):
        
        return self._geocenter_radec
    
    
    @property
    def vel(self):
        
        return self._vel
    
    
    @property
    def angvel(self):
        
        return self._angvel
    
    
    @property
    def sun_visible(self):
        
        return self._sun_visible
    
    
    @property
    def saa_passage(self):
        
        return self._saa_passage
    
    
    @property
    def gti(self):
        
        return self._gti
    
    
    @property
    def sun_occulted(self):
        
        return self._sun_occulted
    
    
    def get_quaternion(self, met):
        
        return self._quaternion_interp(met)

    
    def get_coords_gcrs(self, met):
        
        return self._coords_gcrs_interp(met)
    
    
    def get_coords_itrs(self, met):
        
        return self._coords_itrs_interp(met)
    
    
    def get_earth_angular_radius(self, met):
        
        return self._earth_angular_radius_interp(met)
    
    
    def get_geocenter_radec(self, met):
        
        return self._geocenter_radec_interp(met)
    
    
    def get_vel(self, met):
        
        return self._vel_interp(met)
    
    
    def get_angvel(self, met):
        
        return self._angvel_interp(met)
    
    
    def get_sun_location(self, met):
        
        utc = fermi_met_to_utc(met)
        utc = Time(utc, scale='utc', format='isot')
        
        sun = get_sun(utc)
        
        return sun.ra.degree, sun.dec.degree
    
    
    def get_sun_visible(self, met):
        
        return self._sun_visible_interp(met) >= 0.5
    
    
    def get_saa_passage(self, met):
        
        return self._saa_passage_interp(met) >= 0.5
    
    
    def get_mcilwaine_l(self, met):
        
        coords_itrs = self.get_coords_itrs(met)
        
        if coords_itrs.ndim == 1:
            return calc_mcilwain_l(coords_itrs[0], coords_itrs[1])
        else:
            return calc_mcilwain_l(coords_itrs[:, 0], coords_itrs[:, 1])


    def to_fermi_frame(self, ra, dec, met):
        
        quat = self.get_quaternion(met)
        az, zen = radec_to_spacecraft(ra, dec, quat, deg=True)
        
        return az, zen
    
    
    def to_sky_frame(self, az, zen, met):
        
        quat = self.get_quaternion(met)
        ra, dec = spacecraft_to_radec(az, zen, quat, deg=True)
        
        return ra, dec
    
    
    def get_location_visible(self, ra, dec, met):
        
        geo = self.get_geocenter_radec(met)
        earth_radius = self.get_earth_angular_radius(met)
        
        geo_arr = np.atleast_2d(geo)
        
        separation = angular_separation(
            ra * u.deg, dec * u.deg,
            geo_arr[:, 0] * u.deg, geo_arr[:, 1] * u.deg).to_value(u.deg)
    
        visible = separation > earth_radius
    
        return visible[0] if geo.ndim == 1 else visible


    def get_detector_pointing(self, det, met):

        det = gbmDetector.from_name(det)
        det_az, det_zen = det.azimuth, det.zenith
        det_ra, det_dec = self.to_sky_frame(det_az, det_zen, met)
        
        return det_ra, det_dec


    def get_detector_angle(self, ra, dec, det, met):

        det_ra, det_dec = self.get_detector_pointing(det, met)
        
        angle = angular_separation(
            ra*u.deg, dec*u.deg, 
            det_ra*u.deg, det_dec*u.deg).to_value(u.deg)

        return angle


    def extract_skymap(self, met, ra=None, dec=None, savepath='./geometry'):
        
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        skymap = gbmSkyMap()
        skymap.plot_earth(self, met)
        skymap.plot_sun(self, met)
        skymap.plot_galactic()
        skymap.plot_all_detectors(self, met)
        
        if ra is not None and dec is not None:
            skymap.add_source(ra, dec)

        skymap.save(savepath + '/sky_map_poshist.pdf')
