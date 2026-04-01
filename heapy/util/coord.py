import os
import numpy as np
from astropy.time import Time
from astropy import units as u
from scipy.spatial.transform import Rotation
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation, SkyCoord



def gcrs_to_itrs(scpos, utc, utc_format='isot'):
    """
    Convert coordinates 
    from GCRS (Geocentric Celestial Reference System) 
    to ITRS (International Terrestrial Reference System).
    
    Parameters:
    ----------
    scpos : array_like, in units of (m, m, m)
        single: [x, y, z]
        batch: [[x1, y1, z1], ...]
    utc : str or array_like
        single: 'utc'
        batch: ['utc1', ...]
        
    Returns:
    -------
    lla : array_like, in units of (deg, deg, km)
        single: [lat, lon, alt]
        batch: [[lat1, lon1, alt1], ...]
    """
    
    scpos_arr = np.asarray(scpos)
    utc_arr = np.asarray(utc)
    
    if scpos_arr.ndim == 1 \
        and utc_arr.ndim == 0 \
            and scpos_arr.shape[0] == 3:
        batch = False
    elif scpos_arr.ndim == 2 \
        and utc_arr.ndim == 1 \
            and scpos_arr.shape[1] == 3:
        batch = True
    else:
        raise ValueError('Invalid input shapes: scpos should be (3,) or (N, 3), \
            utc should be scalar or (N,)')
        
    scpos_arr = np.atleast_2d(scpos_arr)
    utc_arr = np.atleast_1d(utc_arr)
    
    if scpos_arr.shape[0] != utc_arr.shape[0]:
        raise ValueError('The number of coordinate positions must match the number of times')
    
    t = Time(utc_arr, format=utc_format, scale='utc')
    cart_gcrs = CartesianRepresentation(
        x=scpos_arr[:, 0], 
        y=scpos_arr[:, 1], 
        z=scpos_arr[:, 2], 
        unit=u.m)
    
    coords_gcrs = GCRS(cart_gcrs, obstime=t)
    coords_itrs = coords_gcrs.transform_to(ITRS(obstime=t))
    
    location = coords_itrs.earth_location
    
    lla = np.column_stack((
        location.lat.deg, 
        location.lon.deg, 
        location.height.to(u.km).value))
    
    return lla[0] if not batch else lla


def itrs_to_gcrs(lla, utc, utc_format='isot'):
    """
    Convert coordinates
    from ITRS (International Terrestrial Reference System)
    to GCRS (Geocentric Celestial Reference System).
    
    Parameters:
    ----------
    lla : array_like, in units of (deg, deg, km)
        single: [lat, lon, alt]
        batch: [[lat1, lon1, alt1], ...]
    utc : str or array_like
        single: 'utc'
        batch: ['utc1', ...]
        
    Returns:
    -------
    scpos : array_like, in units of (m, m, m)
        single: [x, y, z]
        batch: [[x1, y1, z1], ...]
    """
    
    lla_arr = np.asarray(lla)
    utc_arr = np.asarray(utc)
    
    if lla_arr.ndim == 1 \
        and utc_arr.ndim == 0 \
            and lla_arr.shape[0] == 3:
        batch = False
    elif lla_arr.ndim == 2 \
        and utc_arr.ndim == 1 \
            and lla_arr.shape[1] == 3:
        batch = True
    else:
        raise ValueError('Invalid input shapes: lla should be (3,) or (N, 3), \
            utc should be scalar or (N,)')

    lla_arr = np.atleast_2d(lla_arr)
    utc_arr = np.atleast_1d(utc_arr)
    
    if lla_arr.shape[0] != utc_arr.shape[0]:
        raise ValueError('The number of coordinate positions must match the number of times')

    t = Time(utc_arr, format=utc_format, scale='utc')

    loc = EarthLocation(
        lat=lla_arr[:, 0] * u.deg, 
        lon=lla_arr[:, 1] * u.deg, 
        height=lla_arr[:, 2] * u.km)
    
    itrs_coords = loc.get_itrs(obstime=t)
    gcrs_coords = itrs_coords.transform_to(GCRS(obstime=t))

    xyz = np.column_stack((
        gcrs_coords.cartesian.x.value,
        gcrs_coords.cartesian.y.value,
        gcrs_coords.cartesian.z.value))

    return xyz[0] if not batch else xyz


def calc_earth_angular_radius(alt, alt_unit=u.km):
    """
    Calculate the angular radius of the Earth at a given altitude.
    
    Parameters:
    ----------
    alt : float or np.array
        altitude above the Earth's surface
    alt_unit : astropy.unit
        unit of the input altitude, default is km
        
    Returns:
    -------
    half_angle : np.array
        angular radius of the Earth (in degrees)
    """

    R_EARTH = 6378.137 
    
    h = (alt * alt_unit).to(u.km).value
    h = np.maximum(h, 0.0) 
    
    sin_theta = R_EARTH / (R_EARTH + h)
    half_angle_rad = np.arcsin(np.clip(sin_theta, -1.0, 1.0))
    
    return np.rad2deg(half_angle_rad)


def get_geocenter_radec(scpos):
    """
    Calculate the RA/Dec of the Earth center as seen from the satellite.
    
    Parameters:
    ----------
    scpos : array_like, in units of (m, m, m)
        
    Returns:
    -------
    ra : array_like, in units of degrees
    dec : array_like, in units of degrees
    """
    
    scpos_arr = np.asarray(scpos)
    
    if scpos_arr.ndim == 1 \
        and scpos_arr.shape[0] == 3:
        batch = False
    elif scpos_arr.ndim == 2 \
        and scpos_arr.shape[1] == 3:
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
    
    geo = np.column_stack((
        coord.ra.deg, 
        coord.dec.deg))
    
    return geo[0] if not batch else geo


def calc_mcilwain_l(latitude, longitude):
    """
    Estimate the McIlwain L-shell parameter based on the satellite's latitude and longitude.
    
    Parameters:
    ----------
    latitude : float or array_like
        Geodetic latitude in degrees (range: -30 to 30)
    longitude : float or array_like
        Geodetic longitude in degrees (range: 0 to 360)
        
    Returns:
    -------
    mc_l : float or array_like
        Estimated McIlwain L-shell parameter
    """
    
    lat = np.asarray(latitude)
    lon = np.asarray(longitude)
    
    if lat.ndim == 0 \
        and lon.ndim == 0:
        batch = False
    elif lat.ndim == 1 \
        and lon.ndim == 1 \
            and lat.shape[0] == lon.shape[0]:
        batch = True
    else:
        raise ValueError('Invalid input shapes: \
            latitude and longitude should be scalars or 1D arrays of the same length')
    
    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon) % 360.0
    
    if np.any((lat < -30) | (lat > 30)):
        raise ValueError("Latitude out of range [-30, 30]")

    idx1 = (lon / 10.0).astype(int)
    idx2 = (idx1 + 1) % 36
    f = (lon / 10.0) - idx1
    
    coeffs_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'docs', 'McIlwainL_coeffs', 'McIlwainL_Coeffs.npy')
    poly_coeffs = np.load(coeffs_file)
    
    c1 = poly_coeffs[idx1]
    c2 = poly_coeffs[idx2]
    
    lat_powers = np.column_stack([np.ones_like(lat), lat, lat**2, lat**3])
    
    l_left = np.sum(c1 * lat_powers, axis=1)
    l_right = np.sum(c2 * lat_powers, axis=1)
    
    mc_l = (1.0 - f) * l_left + f * l_right
    
    return mc_l.reshape(np.shape(latitude)) if batch else mc_l[0]


def spacecraft_to_radec(az, zen, quat, deg=True):
    """
    Convert spacecraft body coordinates (Az/Zen) to celestial coordinates (RA/Dec).

    Parameters:
    ----------
    az, zen : float or np.array
        Satellite azimuth and zenith angles in the spacecraft body frame.
    quat : np.array
        Quaternion array. Shape can be (4,) or (N, 4).
        Note: Assumes Fermi standard order [q1, q2, q3, q4] -> [x, y, z, w]
    deg : bool
        Whether the input and output are in degrees. Default is True.
        
    Returns:
    -------
    ra, dec : float or np.array
        Right Ascension and Declination in the GCRS frame.
    """
    
    az_arr = np.asarray(az)
    zen_arr = np.asarray(zen)
    quat_arr = np.asarray(quat)
    
    if az_arr.shape != zen_arr.shape:
        raise ValueError('Azimuth and zenith arrays must have the same shape')
    
    if quat_arr.ndim == 1 and quat_arr.shape[0] != 4:
        raise ValueError('Quaternion array must have shape (4,) or (N, 4)')
    elif quat_arr.ndim == 2 and quat_arr.shape[1] != 4:
        raise ValueError('Quaternion array must have shape (4,) or (N, 4)')
    
    if az_arr.ndim == 0 \
        and zen_arr.ndim == 0 \
            and quat_arr.ndim == 1:
        batch = False
    else:
        batch = True

    az_arr = np.atleast_1d(az_arr)
    zen_arr = np.atleast_1d(zen_arr)
    quat_arr = np.atleast_2d(quat_arr)

    if deg:
        az_rad = np.deg2rad(az_arr)
        zen_rad = np.deg2rad(zen_arr)
    else:
        az_rad, zen_rad = az_arr, zen_arr

    sin_zen = np.sin(zen_rad)
    pos_body = np.column_stack([
        sin_zen * np.cos(az_rad),
        sin_zen * np.sin(az_rad),
        np.cos(zen_rad)])

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
        raise ValueError(f"Dimension mismatch: Number of \
            positions ({num_pos}) and quaternions ({num_quat}) must be 1:N, N:1, or N:N")

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
    """
    Convert celestial coordinates (RA/Dec) to spacecraft body coordinates (Az/Zen).

    Parameters:
    ----------
    ra, dec : float or np.array
        Right Ascension and Declination in the J2000/GCRS frame.
    quat : np.array
        Quaternion array. Shape can be (4,) or (N, 4).
    deg : bool
        Whether the input and output are in degrees. Default is True.
        
    Returns:
    -------
    az, zen : float or np.array
        Satellite azimuth and zenith angles in the spacecraft body frame.
    """
    
    ra_arr = np.asarray(ra)
    dec_arr = np.asarray(dec)
    quat_arr = np.asarray(quat)
    
    if ra_arr.shape != dec_arr.shape:
        raise ValueError('RA and Dec arrays must have the same shape')
    
    if quat_arr.ndim == 1 and quat_arr.shape[0] != 4:
        raise ValueError('Quaternion array must have shape (4,) or (N, 4)')
    elif quat_arr.ndim == 2 and quat_arr.shape[1] != 4:
        raise ValueError('Quaternion array must have shape (4,) or (N, 4)')
    
    if ra_arr.ndim == 0 \
        and dec_arr.ndim == 0 \
            and quat_arr.ndim == 1:
        batch = False
    else:
        batch = True

    ra_arr = np.atleast_1d(ra_arr)
    dec_arr = np.atleast_1d(dec_arr)
    quat_arr = np.atleast_2d(quat_arr)

    if deg:
        ra_rad = np.deg2rad(ra_arr)
        dec_rad = np.deg2rad(dec_arr)
    else:
        ra_rad, dec_rad = ra_arr, dec_arr

    cos_dec = np.cos(dec_rad)
    pos_sky = np.column_stack([
        cos_dec * np.cos(ra_rad),
        cos_dec * np.sin(ra_rad),
        np.sin(dec_rad)
        ])

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
        raise ValueError(f"Dimension mismatch: Number of \
            positions ({num_pos}) and quaternions ({num_quat}) must be 1:N, N:1, or N:N")

    x, y, z = cart_pos_body[:, 0], cart_pos_body[:, 1], cart_pos_body[:, 2]
    
    zen = np.arccos(np.clip(z, -1.0, 1.0))
    
    az = np.arctan2(y, x) % (2 * np.pi)

    if deg:
        az, zen = np.rad2deg(az), np.rad2deg(zen)

    if not batch:
        return az[0], zen[0]
    
    return az, zen
