"""Coordinate transformation utilities for satellite and spacecraft data analysis.

This module provides functions to convert between celestial and terrestrial
reference frames (GCRS/ITRS), compute Earth geometry parameters visible from
orbit, estimate the McIlwain L-shell, and transform directions between the
spacecraft body frame and the GCRS sky frame using quaternion attitude data.
"""

import os

from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation, SkyCoord
from astropy.time import Time
import numpy as np
from scipy.spatial.transform import Rotation


def gcrs_to_itrs(scpos, utc, utc_format='isot'):
    """Convert GCRS Cartesian position to ITRS geodetic coordinates.

    Transforms one or more spacecraft positions expressed in the Geocentric
    Celestial Reference System (GCRS) into geodetic latitude, longitude, and
    altitude in the International Terrestrial Reference System (ITRS).

    Args:
        scpos: Spacecraft position(s) in metres. Single position as a
            shape-(3,) array ``[x, y, z]``; batch as a shape-(N, 3) array
            ``[[x1, y1, z1], ...]``.
        utc: UTC timestamp(s) matching the format given by ``utc_format``.
            Single value as a scalar string; batch as a 1-D array of strings
            with length N matching ``scpos``.
        utc_format: Astropy time format string for parsing ``utc``.
            Defaults to ``'isot'``.

    Returns:
        Geodetic coordinates in units of (degrees, degrees, km). Single
        position returned as a shape-(3,) array ``[lat, lon, alt]``; batch
        returned as a shape-(N, 3) array ``[[lat1, lon1, alt1], ...]``.

    Raises:
        ValueError: If ``scpos`` is not shape (3,) or (N, 3), if ``utc`` is
            not a scalar or shape-(N,) array, or if the number of positions
            does not match the number of timestamps.
    """

    scpos_arr = np.asarray(scpos)
    utc_arr = np.asarray(utc)

    if scpos_arr.ndim == 1 and utc_arr.ndim == 0 and scpos_arr.shape[0] == 3:
        batch = False
    elif scpos_arr.ndim == 2 and utc_arr.ndim == 1 and scpos_arr.shape[1] == 3:
        batch = True
    else:
        raise ValueError(
            'Invalid input shapes: scpos should be (3,) or (N, 3), \
            utc should be scalar or (N,)'
        )

    scpos_arr = np.atleast_2d(scpos_arr)
    utc_arr = np.atleast_1d(utc_arr)

    if scpos_arr.shape[0] != utc_arr.shape[0]:
        raise ValueError('The number of coordinate positions must match the number of times')

    t = Time(utc_arr, format=utc_format, scale='utc')
    cart_gcrs = CartesianRepresentation(
        x=scpos_arr[:, 0], y=scpos_arr[:, 1], z=scpos_arr[:, 2], unit=u.m
    )

    coords_gcrs = GCRS(cart_gcrs, obstime=t)
    coords_itrs = coords_gcrs.transform_to(ITRS(obstime=t))

    location = coords_itrs.earth_location

    lla = np.column_stack((location.lat.deg, location.lon.deg, location.height.to(u.km).value))

    return lla[0] if not batch else lla


def itrs_to_gcrs(lla, utc, utc_format='isot'):
    """Convert ITRS geodetic coordinates to GCRS Cartesian position.

    Transforms one or more geodetic positions expressed in the International
    Terrestrial Reference System (ITRS) into Cartesian coordinates in the
    Geocentric Celestial Reference System (GCRS).

    Args:
        lla: Geodetic coordinate(s) in units of (degrees, degrees, km).
            Single position as a shape-(3,) array ``[lat, lon, alt]``; batch
            as a shape-(N, 3) array ``[[lat1, lon1, alt1], ...]``.
        utc: UTC timestamp(s) matching the format given by ``utc_format``.
            Single value as a scalar string; batch as a 1-D array of strings
            with length N matching ``lla``.
        utc_format: Astropy time format string for parsing ``utc``.
            Defaults to ``'isot'``.

    Returns:
        Spacecraft position(s) in metres. Single position returned as a
        shape-(3,) array ``[x, y, z]``; batch returned as a shape-(N, 3)
        array ``[[x1, y1, z1], ...]``.

    Raises:
        ValueError: If ``lla`` is not shape (3,) or (N, 3), if ``utc`` is
            not a scalar or shape-(N,) array, or if the number of positions
            does not match the number of timestamps.
    """

    lla_arr = np.asarray(lla)
    utc_arr = np.asarray(utc)

    if lla_arr.ndim == 1 and utc_arr.ndim == 0 and lla_arr.shape[0] == 3:
        batch = False
    elif lla_arr.ndim == 2 and utc_arr.ndim == 1 and lla_arr.shape[1] == 3:
        batch = True
    else:
        raise ValueError(
            'Invalid input shapes: lla should be (3,) or (N, 3), \
            utc should be scalar or (N,)'
        )

    lla_arr = np.atleast_2d(lla_arr)
    utc_arr = np.atleast_1d(utc_arr)

    if lla_arr.shape[0] != utc_arr.shape[0]:
        raise ValueError('The number of coordinate positions must match the number of times')

    t = Time(utc_arr, format=utc_format, scale='utc')

    loc = EarthLocation(
        lat=lla_arr[:, 0] * u.deg, lon=lla_arr[:, 1] * u.deg, height=lla_arr[:, 2] * u.km
    )

    itrs_coords = loc.get_itrs(obstime=t)
    gcrs_coords = itrs_coords.transform_to(GCRS(obstime=t))

    xyz = np.column_stack(
        (
            gcrs_coords.cartesian.x.value,
            gcrs_coords.cartesian.y.value,
            gcrs_coords.cartesian.z.value,
        )
    )

    return xyz[0] if not batch else xyz


def calc_earth_angular_radius(alt, alt_unit=u.km):
    """Calculate the angular radius of the Earth seen from a given altitude.

    Computes the half-angle of the Earth's disk as viewed by an observer at
    the specified altitude above the surface, using the WGS-84 equatorial
    radius as the Earth radius.

    Args:
        alt: Altitude above the Earth's surface. Scalar or array, expressed
            in the units given by ``alt_unit``.
        alt_unit: Astropy unit for ``alt``. Defaults to ``u.km``.

    Returns:
        Angular radius of the Earth in degrees. Scalar or array matching the
        shape of ``alt``.
    """

    R_EARTH = 6378.137

    h = (alt * alt_unit).to(u.km).value
    h = np.maximum(h, 0.0)

    sin_theta = R_EARTH / (R_EARTH + h)
    half_angle_rad = np.arcsin(np.clip(sin_theta, -1.0, 1.0))

    return np.rad2deg(half_angle_rad)


def get_geocenter_radec(scpos):
    """Return the RA/Dec of the Earth center as seen from the spacecraft.

    Computes the right ascension and declination in the GCRS frame of the
    direction pointing from the spacecraft toward the geocenter.

    Args:
        scpos: Spacecraft position(s) in metres in GCRS Cartesian coordinates.
            Single position as a shape-(3,) array ``[x, y, z]``; batch as a
            shape-(N, 3) array.

    Returns:
        Geocenter direction(s) as (RA, Dec) pairs in degrees. Single position
        returned as a shape-(2,) array ``[ra, dec]``; batch returned as a
        shape-(N, 2) array ``[[ra1, dec1], ...]``.

    Raises:
        ValueError: If ``scpos`` is not shape (3,) or (N, 3).
    """

    scpos_arr = np.asarray(scpos)

    if scpos_arr.ndim == 1 and scpos_arr.shape[0] == 3:
        batch = False
    elif scpos_arr.ndim == 2 and scpos_arr.shape[1] == 3:
        batch = True
    else:
        raise ValueError('Invalid input shape: scpos should be (3,) or (N, 3)')

    scpos_arr = np.atleast_2d(scpos_arr)

    vec_to_earth = CartesianRepresentation(
        x=-scpos_arr[:, 0] * u.m,
        y=-scpos_arr[:, 1] * u.m,
        z=-scpos_arr[:, 2] * u.m,
    )

    coord = SkyCoord(vec_to_earth, frame='gcrs')

    geo = np.column_stack((coord.ra.deg, coord.dec.deg))

    return geo[0] if not batch else geo


def calc_mcilwain_l(latitude, longitude):
    """Estimate the McIlwain L-shell parameter from geodetic coordinates.

    Computes an empirical polynomial estimate of the McIlwain L-shell value
    using pre-computed coefficient tables stored in the package data directory.
    The estimate is valid for geodetic latitudes in the range [-30, 30] degrees
    and for longitudes in [0, 360) degrees.

    Args:
        latitude: Geodetic latitude in degrees. Scalar or 1-D array; must
            be in the range [-30, 30].
        longitude: Geodetic longitude in degrees. Scalar or 1-D array of the
            same length as ``latitude``; values are wrapped to [0, 360).

    Returns:
        Estimated McIlwain L-shell value(s). Scalar when inputs are scalars;
        array matching the shape of ``latitude`` when inputs are arrays.

    Raises:
        ValueError: If ``latitude`` or ``longitude`` have incompatible shapes,
            or if any latitude value is outside [-30, 30].
    """

    lat = np.asarray(latitude)
    lon = np.asarray(longitude)

    if lat.ndim == 0 and lon.ndim == 0:
        batch = False
    elif lat.ndim == 1 and lon.ndim == 1 and lat.shape[0] == lon.shape[0]:
        batch = True
    else:
        raise ValueError(
            'Invalid input shapes: \
            latitude and longitude should be scalars or 1D arrays of the same length'
        )

    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon) % 360.0

    if np.any((lat < -30) | (lat > 30)):
        raise ValueError('Latitude out of range [-30, 30]')

    idx1 = (lon / 10.0).astype(int)
    idx2 = (idx1 + 1) % 36
    f = (lon / 10.0) - idx1

    coeffs_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'docs',
        'McIlwainL_coeffs',
        'McIlwainL_Coeffs.npy',
    )
    poly_coeffs = np.load(coeffs_file)

    c1 = poly_coeffs[idx1]
    c2 = poly_coeffs[idx2]

    lat_powers = np.column_stack([np.ones_like(lat), lat, lat**2, lat**3])

    l_left = np.sum(c1 * lat_powers, axis=1)
    l_right = np.sum(c2 * lat_powers, axis=1)

    mc_l = (1.0 - f) * l_left + f * l_right

    return mc_l.reshape(np.shape(latitude)) if batch else mc_l[0]


def spacecraft_to_radec(az, zen, quat, deg=True):
    """Convert spacecraft body-frame Az/Zen angles to GCRS RA/Dec.

    Rotates a direction given in the spacecraft body frame (azimuth and
    zenith angle) into the GCRS celestial frame using the spacecraft attitude
    quaternion. Supports three broadcast modes: one direction with N attitudes,
    N directions with one attitude, or N directions with N attitudes.

    Args:
        az: Azimuth angle(s) in the spacecraft body frame. Scalar or array;
            units controlled by ``deg``.
        zen: Zenith angle(s) in the spacecraft body frame. Scalar or array
            with the same shape as ``az``; units controlled by ``deg``.
        quat: Spacecraft attitude quaternion(s) in Fermi standard order
            ``[q1, q2, q3, q4]`` mapped to ``[x, y, z, w]``. Shape must be
            (4,) for a single quaternion or (N, 4) for a batch.
        deg: If ``True``, interpret ``az`` and ``zen`` as degrees and return
            ``ra`` and ``dec`` in degrees. If ``False``, use radians throughout.
            Defaults to ``True``.

    Returns:
        Tuple ``(ra, dec)`` giving right ascension and declination in the GCRS
        frame. Scalars when all inputs are scalar; arrays otherwise. Units are
        degrees when ``deg=True``, radians when ``deg=False``.

    Raises:
        ValueError: If ``az`` and ``zen`` do not have the same shape, if
            ``quat`` does not have shape (4,) or (N, 4), or if the number of
            positions and quaternions cannot be broadcast (must be 1:N, N:1,
            or N:N).
    """

    az_arr = np.asarray(az)
    zen_arr = np.asarray(zen)
    quat_arr = np.asarray(quat)

    if az_arr.shape != zen_arr.shape:
        raise ValueError('Azimuth and zenith arrays must have the same shape')

    if (quat_arr.ndim == 1 and quat_arr.shape[0] != 4) or (
        quat_arr.ndim == 2 and quat_arr.shape[1] != 4
    ):
        raise ValueError('Quaternion array must have shape (4,) or (N, 4)')

    batch = not (az_arr.ndim == 0 and zen_arr.ndim == 0 and quat_arr.ndim == 1)

    az_arr = np.atleast_1d(az_arr)
    zen_arr = np.atleast_1d(zen_arr)
    quat_arr = np.atleast_2d(quat_arr)

    if deg:
        az_rad = np.deg2rad(az_arr)
        zen_rad = np.deg2rad(zen_arr)
    else:
        az_rad, zen_rad = az_arr, zen_arr

    sin_zen = np.sin(zen_rad)
    pos_body = np.column_stack(
        [sin_zen * np.cos(az_rad), sin_zen * np.sin(az_rad), np.cos(zen_rad)]
    )

    rot = Rotation.from_quat(quat_arr)
    dcm = rot.as_matrix()

    num_pos = pos_body.shape[0]
    num_quat = dcm.shape[0]

    if num_pos == 1 and num_quat > 1:
        # modele 1: 1 position + N quaternions (track a single source over time)
        # Matrix (N, 3, 3) multiplied by vector (1, 3) -> (N, 3)
        cart_pos = np.einsum('nij,j->ni', dcm, pos_body[0])
    elif num_pos > 1 and num_quat == 1:
        # modele 2: N positions + 1 quaternion (same time multiple sources)
        # Matrix (1, 3, 3) multiplied by vector (N, 3) -> (N, 3)
        cart_pos = np.einsum('ij,nj->ni', dcm[0], pos_body)
    elif num_pos == num_quat:
        # modele 3: N positions + N quaternions (standard pipeline)
        # Matrix (N, 3, 3) multiplied by vector (N, 3) -> (N, 3)
        cart_pos = np.einsum('nij,nj->ni', dcm, pos_body)
    else:
        raise ValueError(
            f'Dimension mismatch: Number of \
            positions ({num_pos}) and quaternions ({num_quat}) must be 1:N, N:1, or N:N'
        )

    x, y, z = cart_pos[:, 0], cart_pos[:, 1], cart_pos[:, 2]
    z = np.clip(z, -1.0, 1.0)

    dec = np.arcsin(z)
    ra = np.arctan2(y, x) % (2 * np.pi)

    if deg:
        ra, dec = np.rad2deg(ra), np.rad2deg(dec)

    if not batch:
        return ra[0], dec[0]

    return ra, dec


def radec_to_spacecraft(ra, dec, quat, deg=True):
    """Convert GCRS RA/Dec to spacecraft body-frame Az/Zen angles.

    Rotates a celestial direction given in the GCRS frame (right ascension and
    declination) into the spacecraft body frame (azimuth and zenith angle)
    using the inverse of the spacecraft attitude quaternion. Supports three
    broadcast modes: one direction with N attitudes, N directions with one
    attitude, or N directions with N attitudes.

    Args:
        ra: Right ascension in the J2000/GCRS frame. Scalar or array; units
            controlled by ``deg``.
        dec: Declination in the J2000/GCRS frame. Scalar or array with the
            same shape as ``ra``; units controlled by ``deg``.
        quat: Spacecraft attitude quaternion(s) in Fermi standard order
            ``[q1, q2, q3, q4]`` mapped to ``[x, y, z, w]``. Shape must be
            (4,) for a single quaternion or (N, 4) for a batch.
        deg: If ``True``, interpret ``ra`` and ``dec`` as degrees and return
            ``az`` and ``zen`` in degrees. If ``False``, use radians
            throughout. Defaults to ``True``.

    Returns:
        Tuple ``(az, zen)`` giving the azimuth and zenith angle in the
        spacecraft body frame. Scalars when all inputs are scalar; arrays
        otherwise. Units are degrees when ``deg=True``, radians when
        ``deg=False``.

    Raises:
        ValueError: If ``ra`` and ``dec`` do not have the same shape, if
            ``quat`` does not have shape (4,) or (N, 4), or if the number of
            positions and quaternions cannot be broadcast (must be 1:N, N:1,
            or N:N).
    """

    ra_arr = np.asarray(ra)
    dec_arr = np.asarray(dec)
    quat_arr = np.asarray(quat)

    if ra_arr.shape != dec_arr.shape:
        raise ValueError('RA and Dec arrays must have the same shape')

    if (quat_arr.ndim == 1 and quat_arr.shape[0] != 4) or (
        quat_arr.ndim == 2 and quat_arr.shape[1] != 4
    ):
        raise ValueError('Quaternion array must have shape (4,) or (N, 4)')

    batch = not (ra_arr.ndim == 0 and dec_arr.ndim == 0 and quat_arr.ndim == 1)

    ra_arr = np.atleast_1d(ra_arr)
    dec_arr = np.atleast_1d(dec_arr)
    quat_arr = np.atleast_2d(quat_arr)

    if deg:
        ra_rad = np.deg2rad(ra_arr)
        dec_rad = np.deg2rad(dec_arr)
    else:
        ra_rad, dec_rad = ra_arr, dec_arr

    cos_dec = np.cos(dec_rad)
    pos_sky = np.column_stack([cos_dec * np.cos(ra_rad), cos_dec * np.sin(ra_rad), np.sin(dec_rad)])

    rot = Rotation.from_quat(quat_arr)
    dcm_inv = rot.inv().as_matrix()

    num_pos = pos_sky.shape[0]
    num_quat = dcm_inv.shape[0]

    if num_pos == 1 and num_quat > 1:
        # modele 1: 1 position + N quaternions (track a single source over time)
        # Matrix (N, 3, 3) multiplied by vector (1, 3) -> (N, 3)
        cart_pos_body = np.einsum('nij,j->ni', dcm_inv, pos_sky[0])
    elif num_pos > 1 and num_quat == 1:
        # modele 2: N positions + 1 quaternion (same time multiple sources)
        # Matrix (1, 3, 3) multiplied by vector (N, 3) -> (N, 3)
        cart_pos_body = np.einsum('ij,nj->ni', dcm_inv[0], pos_sky)
    elif num_pos == num_quat:
        # modele 3: N positions + N quaternions (standard pipeline)
        # Matrix (N, 3, 3) multiplied by vector (N, 3) -> (N, 3)
        cart_pos_body = np.einsum('nij,nj->ni', dcm_inv, pos_sky)
    else:
        raise ValueError(
            f'Dimension mismatch: Number of \
            positions ({num_pos}) and quaternions ({num_quat}) must be 1:N, N:1, or N:N'
        )

    x, y, z = cart_pos_body[:, 0], cart_pos_body[:, 1], cart_pos_body[:, 2]

    zen = np.arccos(np.clip(z, -1.0, 1.0))

    az = np.arctan2(y, x) % (2 * np.pi)

    if deg:
        az, zen = np.rad2deg(az), np.rad2deg(zen)

    if not batch:
        return az[0], zen[0]

    return az, zen
