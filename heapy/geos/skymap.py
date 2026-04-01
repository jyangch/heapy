import warnings
import numpy as np
import astropy.units as u
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.coordinates import SkyCoord
warnings.filterwarnings("ignore", message="Approximating coordinate system")



class gbmSkyMap(object):
    
    def __init__(self, figsize=(10, 5), central_longitude=180, flipped=True):
        
        self.fig = plt.figure(figsize=figsize, dpi=300)
        
        self.ax = self.fig.add_subplot(1, 1, 1, \
            projection=ccrs.Mollweide(central_longitude=central_longitude))
        self.ax.set_facecolor("#f5eddf")
        
        self.central_longitude = central_longitude
        self.flipped = flipped
        
        self._setup_gridlines()


    def _setup_gridlines(self):

        self.ax.gridlines(xlocs=range(-180, 181, 30), ylocs=range(-90, 91, 15), 
                          color='gray', alpha=0.5, linestyle='--')
        self.ax.set_global()

        if self.flipped:
            self.ax.invert_xaxis()

        lons = np.arange(0, 360, 30)
        for lon in lons:
            if lon == self.central_longitude - 180 or lon == self.central_longitude + 180:
                continue
            self.ax.text(lon, 0, f"{lon}$^\\circ$", transform=ccrs.Geodetic(), 
                         ha='center', va='center', alpha=0.5)
            
        lats = np.arange(-75, 76, 15)
        for lat in lats:
            if lat == 0: continue
            
            self.ax.text(self.central_longitude, lat, f"{lat}$^\\circ$", transform=ccrs.Geodetic(), 
                         ha='center', va='center', alpha=0.5)


    def _generate_circle(self, ra, dec, radius, steps=100):

        center = SkyCoord(ra * u.deg, dec * u.deg, frame='gcrs')
        angles = np.linspace(0, 360, steps) * u.deg
        circle = center.directional_offset_by(angles, radius * u.deg)
        
        return circle.ra.degree, circle.dec.degree
    
    
    def plot_earth(self, geom, met):

        ra, dec = geom.get_geocenter_radec(met)
        radius = geom.get_earth_angular_radius(met)

        self.ax.tissot(rad_km=radius * 111.3195, lons=[ra], lats=[dec], 
                       facecolor='#90d7ec', alpha=0.5, zorder=1, edgecolor='#90d7ec')


    def plot_detector(self, geom, met, det, color='#969696'):

        ra, dec = geom.get_detector_pointing(det, met)

        self.ax.tissot(rad_km=10 * 111.3195, lons=[ra], lats=[dec], 
                       facecolor=color, alpha=0.5, zorder=2, edgecolor=color)

        self.ax.text(ra, dec, str(det).lower(), transform=ccrs.Geodetic(), 
                     ha='center', va='center', color=color, fontweight='bold', zorder=3)


    def plot_all_detectors(self, geom, met, highlight_dets=None):

        all_dets = [f'n{i}' for i in range(10)] + ['na', 'nb', 'b0', 'b1']
        highlight_dets = highlight_dets or []
        
        for det in all_dets:
            color = '#f26522' if det in highlight_dets else '#969696'
            self.plot_detector(geom, met, det, color=color)


    def plot_sun(self, geom, met):

        ra, dec = geom.get_sun_location(met)

        self.ax.scatter(ra, dec, transform=ccrs.Geodetic(), 
                        s=70, c='gold', marker='o', zorder=4)


    def plot_galactic(self):
        
        l = np.linspace(0, 360, 360)
        b = np.zeros_like(l)
        
        plane = SkyCoord(l=l*u.deg, b=b*u.deg, frame='galactic').transform_to('gcrs')
        
        self.ax.scatter(plane.ra.degree, plane.dec.degree, transform=ccrs.Geodetic(), 
                        s=5, color='k', alpha=0.3, zorder=0)
        
        center = SkyCoord(l=0*u.deg, b=0*u.deg, frame='galactic').transform_to('gcrs')
        
        self.ax.scatter(center.ra.degree, center.dec.degree, transform=ccrs.Geodetic(),
                        s=30, color='k', alpha=0.5, zorder=1)


    def add_source(self, ra, dec, name=None, marker='*', color='red', size=100):

        self.ax.scatter(ra, dec, transform=ccrs.Geodetic(), 
                        s=size, c=color, marker=marker, zorder=5, linewidths=0)
        if name:
            self.ax.text(ra, dec + 3, name, transform=ccrs.Geodetic(), 
                         ha='center', va='bottom', color=color, zorder=5)


    def add_healpix(self, hpx, gradient=True, clevels=None, plot_meta=True):
        
        if clevels is None:
            clevels = [0.997, 0.955, 0.687]
        
        approx_res = np.sqrt(hpx.pixel_area)
        num_ra = int(np.clip(np.floor(360.0 / approx_res * 1.0), 720, 2880))
        num_dec = int(np.clip(np.floor(180.0 / approx_res * 1.0), 360, 1440))

        if gradient:
            prob_grid, ra_axis, dec_axis = hpx.get_probability_map(
                numpts_ra=num_ra, numpts_dec=num_dec, density=True)
            
            base_cmap = plt.colormaps.get_cmap('RdPu')
            color_array = base_cmap(np.linspace(0, 1, 256))
            color_array[:, -1] = np.linspace(0.0, 1.0, 256)
            
            cmap = mcolors.ListedColormap(color_array)
            cmap.set_bad(alpha=0.0)
            
            RA, DEC = np.meshgrid(ra_axis, dec_axis)
            
            norm = mcolors.PowerNorm(gamma=0.3, vmin=np.nanmin(prob_grid), vmax=np.nanmax(prob_grid))
            
            self.ax.pcolormesh(RA, DEC, prob_grid, transform=ccrs.PlateCarree(), 
                               cmap=cmap, norm=norm, shading='auto', zorder=2)

        for level in clevels:
            vertices_list = hpx.get_confidence_contours(
                level, numpts_ra=num_ra, numpts_dec=num_dec)

            for vertices in vertices_list:
                self.ax.plot(vertices[:, 0], vertices[:, 1], transform=ccrs.Geodetic(), 
                                color='k', linewidth=2.0, alpha=0.9, zorder=4)

        if plot_meta:
            if hpx.sun_location:
                sun_ra, sun_dec = hpx.sun_location
                self.ax.scatter(sun_ra, sun_dec, transform=ccrs.Geodetic(), 
                                s=70, c='gold', marker='o', zorder=4)

            if hpx.geo_location:
                geo_ra, geo_dec = hpx.geo_location
                self.ax.tissot(rad_km=hpx.geo_radius * 111.3195, lons=[geo_ra], lats=[geo_dec], 
                               facecolor='#90d7ec', alpha=0.5, zorder=1, edgecolor='#90d7ec')

            all_dets = [f'n{i}' for i in range(10)] + ['na', 'nb', 'b0', 'b1']
            for det in all_dets:
                ptr_name = f"{det}_pointing"
                if hasattr(hpx, ptr_name):
                    det_ra, det_dec = getattr(hpx, ptr_name)
                    self.ax.tissot(rad_km=10 * 111.3195, lons=[det_ra], lats=[det_dec], 
                                   facecolor='#969696', alpha=0.5, zorder=2, edgecolor='#969696')
                    self.ax.text(det_ra, det_dec, det.lower(), transform=ccrs.PlateCarree(), 
                                 ha='center', va='center', color='#969696', fontweight='bold', zorder=3)


    def show(self):
        
        plt.tight_layout()
        plt.show()
        
        
    def save(self, filename, dpi=300):
        
        plt.tight_layout()
        self.fig.savefig(filename, dpi=dpi)
