"""Fermi/GBM spacecraft geometry backed by position-history FITS files.

Provides ``gbmGeometry``, which reads one or more GBM position-history
(poshist) FITS files, stacks them into a single timeline, and exposes
spline-interpolated spacecraft state — quaternion, GCRS/ITRS position,
velocity, angular velocity, Earth angular radius, geocenter direction,
and Sun/SAA flag — evaluated at arbitrary Mission Elapsed Time (MET)
values.
"""

import os

from astropy import table, units as u
from astropy.coordinates import angular_separation, get_sun
from astropy.io import fits
from astropy.time import Time
import numpy as np
from scipy.interpolate import make_interp_spline

from ..data.retrieve import gbmRetrieve
from ..util.coord import (
    calc_earth_angular_radius,
    calc_mcilwain_l,
    gcrs_to_itrs,
    get_geocenter_radec,
    radec_to_spacecraft,
    spacecraft_to_radec,
)
from ..util.data import split_bool_mask
from ..util.time import fermi_met_to_utc
from .detector import gbmDetector
from .skymap import gbmSkyMap


class gbmGeometry:
    """Read GBM position-history FITS files and provide interpolated spacecraft state.

    Parses the ``GLAST POS HIST`` extension from one or more poshist files,
    merges and deduplicates the timeline, then constructs cubic B-spline
    interpolators for quaternion, GCRS/ITRS coordinates, velocity, angular
    velocity, Earth angular radius, geocenter direction, and Sun/SAA flag.
    All ``get_*`` methods evaluate these interpolators at user-supplied MET
    values and return the corresponding physical quantities.
    """

    def __init__(self, file):
        """Initialize by reading one or more poshist FITS files.

        Args:
            file: Path string or list of path strings pointing to GBM
                position-history FITS files.  When a list is given, the
                files are stacked and deduplicated before interpolation.
        """

        self._file = file

        self._read()

    @classmethod
    def from_utc(cls, utc, t1=-200, t2=500):
        """Construct a ``gbmGeometry`` by downloading the poshist for a UTC time.

        Retrieves the appropriate GBM position-history file via
        ``gbmRetrieve``, covering a time window ``[utc + t1, utc + t2]``
        seconds around the trigger time.

        Args:
            utc: UTC trigger time string in ISO 8601 format
                (e.g. ``'2017-08-17T12:41:04'``).
            t1: Start offset in seconds relative to ``utc``.  Defaults to
                ``-200``.
            t2: End offset in seconds relative to ``utc``.  Defaults to
                ``500``.

        Returns:
            A ``gbmGeometry`` instance initialised from the downloaded
            poshist file.

        Raises:
            AssertionError: If the retrieval returns no poshist file.
        """

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

        self._quaternion = np.column_stack(
            (
                self._poshist['QSJ_1'],
                self._poshist['QSJ_2'],
                self._poshist['QSJ_3'],
                self._poshist['QSJ_4'],
            )
        )
        self._quaternion_interp = make_interp_spline(self._met, self._quaternion, k=3, axis=0)

        self._coords_gcrs = np.column_stack(
            (self._poshist['POS_X'], self._poshist['POS_Y'], self._poshist['POS_Z'])
        )
        self._coords_gcrs_interp = make_interp_spline(self._met, self._coords_gcrs, k=3, axis=0)

        self._coords_itrs = gcrs_to_itrs(self._coords_gcrs, self._utc)
        self._coords_itrs_interp = make_interp_spline(self._met, self._coords_itrs, k=3, axis=0)

        self._earth_angular_radius = calc_earth_angular_radius(self._coords_itrs[:, 2])
        self._earth_angular_radius_interp = make_interp_spline(
            self._met, self._earth_angular_radius, k=3
        )

        self._geocenter_radec = get_geocenter_radec(self._coords_gcrs)
        self._geocenter_radec_interp = make_interp_spline(
            self._met, self._geocenter_radec, k=3, axis=0
        )

        self._vel = np.column_stack(
            (self._poshist['VEL_X'], self._poshist['VEL_Y'], self._poshist['VEL_Z'])
        )
        self._vel_interp = make_interp_spline(self._met, self._vel, k=3, axis=0)

        self._angvel = np.column_stack(
            (self._poshist['WSJ_1'], self._poshist['WSJ_2'], self._poshist['WSJ_3'])
        )
        self._angvel_interp = make_interp_spline(self._met, self._angvel, k=3, axis=0)

        self._sun_visible = (self._poshist['FLAGS'] == 1) | (self._poshist['FLAGS'] == 3)
        self._sun_visible_interp = make_interp_spline(
            self._met, self._sun_visible.astype(float), k=3
        )

        self._saa_passage = (self._poshist['FLAGS'] == 2) | (self._poshist['FLAGS'] == 3)
        self._saa_passage_interp = make_interp_spline(
            self._met, self._saa_passage.astype(float), k=3
        )

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
        """Return the interpolated spacecraft attitude quaternion at ``met``.

        Args:
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            Quaternion array of shape ``(4,)`` for scalar input or
            ``(N, 4)`` for array input, with components
            ``[q1, q2, q3, q4]``.
        """

        return self._quaternion_interp(met)

    def get_coords_gcrs(self, met):
        """Return the interpolated spacecraft position in the GCRS frame at ``met``.

        Args:
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            Position array of shape ``(3,)`` for scalar input or ``(N, 3)``
            for array input, with Cartesian components ``[X, Y, Z]`` in
            metres.
        """

        return self._coords_gcrs_interp(met)

    def get_coords_itrs(self, met):
        """Return the interpolated spacecraft position in the ITRS frame at ``met``.

        Args:
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            Position array of shape ``(3,)`` for scalar input or ``(N, 3)``
            for array input, with Cartesian components ``[X, Y, Z]`` in
            metres.
        """

        return self._coords_itrs_interp(met)

    def get_earth_angular_radius(self, met):
        """Return the interpolated angular radius of Earth as seen from the spacecraft.

        Args:
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            Angular radius in degrees, scalar for scalar input or array of
            shape ``(N,)`` for array input.
        """

        return self._earth_angular_radius_interp(met)

    def get_geocenter_radec(self, met):
        """Return the interpolated geocenter direction in sky coordinates at ``met``.

        Args:
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            Array of shape ``(2,)`` for scalar input or ``(N, 2)`` for array
            input, containing ``[RA, Dec]`` of the geocenter in degrees.
        """

        return self._geocenter_radec_interp(met)

    def get_vel(self, met):
        """Return the interpolated spacecraft velocity vector at ``met``.

        Args:
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            Velocity array of shape ``(3,)`` for scalar input or ``(N, 3)``
            for array input, with Cartesian components ``[Vx, Vy, Vz]`` in
            metres per second.
        """

        return self._vel_interp(met)

    def get_angvel(self, met):
        """Return the interpolated spacecraft angular velocity vector at ``met``.

        Args:
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            Angular velocity array of shape ``(3,)`` for scalar input or
            ``(N, 3)`` for array input, with components
            ``[wx, wy, wz]`` in radians per second.
        """

        return self._angvel_interp(met)

    def get_sun_location(self, met):
        """Return the Sun's sky coordinates (RA, Dec) at ``met``.

        Computes the Sun position via ``astropy.coordinates.get_sun`` after
        converting ``met`` to UTC.

        Args:
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            A tuple ``(ra, dec)`` of the Sun's right ascension and
            declination in degrees.
        """

        utc = fermi_met_to_utc(met)
        utc = Time(utc, scale='utc', format='isot')

        sun = get_sun(utc)

        return sun.ra.degree, sun.dec.degree

    def get_sun_visible(self, met):
        """Return whether the Sun is visible (not occulted) at ``met``.

        Evaluates the interpolated Sun-visibility flag derived from the
        poshist ``FLAGS`` column.

        Args:
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            Boolean scalar or array; ``True`` when the Sun is not occulted.
        """

        return self._sun_visible_interp(met) >= 0.5

    def get_saa_passage(self, met):
        """Return whether the spacecraft is in the South Atlantic Anomaly at ``met``.

        Evaluates the interpolated SAA flag derived from the poshist
        ``FLAGS`` column.

        Args:
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            Boolean scalar or array; ``True`` when the spacecraft is inside
            the SAA.
        """

        return self._saa_passage_interp(met) >= 0.5

    def get_mcilwaine_l(self, met):
        """Return the McIlwain L-shell parameter at the spacecraft position at ``met``.

        Args:
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            McIlwain L value as a scalar or array, dimensionless.
        """

        coords_itrs = self.get_coords_itrs(met)

        if coords_itrs.ndim == 1:
            return calc_mcilwain_l(coords_itrs[0], coords_itrs[1])
        else:
            return calc_mcilwain_l(coords_itrs[:, 0], coords_itrs[:, 1])

    def to_fermi_frame(self, ra, dec, met):
        """Convert sky coordinates to spacecraft azimuth and zenith at ``met``.

        Args:
            ra: Right ascension of the sky position in degrees.
            dec: Declination of the sky position in degrees.
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            A tuple ``(az, zen)`` of azimuth and zenith angles in the
            spacecraft body frame, in degrees.
        """

        quat = self.get_quaternion(met)
        az, zen = radec_to_spacecraft(ra, dec, quat, deg=True)

        return az, zen

    def to_sky_frame(self, az, zen, met):
        """Convert spacecraft body-frame coordinates to sky (RA, Dec) at ``met``.

        Args:
            az: Azimuth angle in the spacecraft body frame, in degrees.
            zen: Zenith angle in the spacecraft body frame, in degrees.
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            A tuple ``(ra, dec)`` of right ascension and declination in
            degrees.
        """

        quat = self.get_quaternion(met)
        ra, dec = spacecraft_to_radec(az, zen, quat, deg=True)

        return ra, dec

    def get_location_visible(self, ra, dec, met):
        """Check whether a sky position is above the Earth limb at ``met``.

        Computes the angular separation between ``(ra, dec)`` and the
        geocenter, then compares it to the Earth angular radius.

        Args:
            ra: Right ascension of the target position in degrees.
            dec: Declination of the target position in degrees.
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            Boolean scalar when ``met`` is scalar, or boolean array of
            shape ``(N,)`` when ``met`` is array-like; ``True`` when the
            position is not occulted by Earth.
        """

        geo = self.get_geocenter_radec(met)
        earth_radius = self.get_earth_angular_radius(met)

        geo_arr = np.atleast_2d(geo)

        separation = angular_separation(
            ra * u.deg, dec * u.deg, geo_arr[:, 0] * u.deg, geo_arr[:, 1] * u.deg
        ).to_value(u.deg)

        visible = separation > earth_radius

        return visible[0] if geo.ndim == 1 else visible

    def get_detector_pointing(self, det, met):
        """Return the sky coordinates of a detector boresight at ``met``.

        Looks up the spacecraft body-frame azimuth and zenith of ``det``
        and converts them to equatorial coordinates using the interpolated
        quaternion.

        Args:
            det: Detector short name string (e.g. ``'n0'``, ``'b1'``).
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            A tuple ``(ra, dec)`` of the detector boresight right ascension
            and declination in degrees.
        """

        det = gbmDetector.from_name(det)
        det_az, det_zen = det.azimuth, det.zenith
        det_ra, det_dec = self.to_sky_frame(det_az, det_zen, met)

        return det_ra, det_dec

    def get_detector_angle(self, ra, dec, det, met):
        """Return the angular separation between a sky position and a detector boresight.

        Args:
            ra: Right ascension of the target position in degrees.
            dec: Declination of the target position in degrees.
            det: Detector short name string (e.g. ``'n0'``, ``'b1'``).
            met: Mission Elapsed Time in seconds, scalar or array-like.

        Returns:
            Angular separation in degrees, scalar or array matching the
            shape of ``met``.
        """

        det_ra, det_dec = self.get_detector_pointing(det, met)

        angle = angular_separation(
            ra * u.deg, dec * u.deg, det_ra * u.deg, det_dec * u.deg
        ).to_value(u.deg)

        return angle

    def extract_skymap(self, met, ra=None, dec=None, savepath='./geometry'):
        """Generate and save a full-sky map PDF for the spacecraft state at ``met``.

        Renders Earth occultation, Sun position, Galactic plane, and all
        detector pointings onto a sky map.  An optional source marker is
        added when both ``ra`` and ``dec`` are provided.  The output PDF is
        written to ``savepath/sky_map_poshist.pdf``.

        Args:
            met: Mission Elapsed Time in seconds at which the map is
                evaluated.
            ra: Right ascension of an optional source marker, in degrees.
                Ignored when ``dec`` is ``None``.
            dec: Declination of an optional source marker, in degrees.
                Ignored when ``ra`` is ``None``.
            savepath: Directory path where the PDF is saved.  Created if it
                does not already exist.  Defaults to ``'./geometry'``.
        """

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
