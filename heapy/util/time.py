import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta


def hxmt_met_to_utc(met):
    """
    Convert HXMT Mission Elapsed Time (MET) to UTC.

    Parameters:
    ----------
    met : float
        The Mission Elapsed Time (MET) in seconds since the reference epoch.
        
    Returns:
    -------
    str
        The corresponding UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).
    """

    dt = TimeDelta(met + 441763197.0, format='sec')
    ref_tt = Time('1998-01-01T00:00:00', format='isot', scale='tt')
    now_utc = (ref_tt + dt).value

    return now_utc


def hxmt_utc_to_met(utc, format='isot'):
    """
    Convert UTC time to HXMT Mission Elapsed Time (MET).

    Parameters:
    ----------
    utc : str
        The UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).
    format : str, optional
        The format of the input UTC time. Default is 'isot'.

    Returns:
    -------
    float
        The corresponding HXMT MET in seconds since the reference epoch.
    """
    
    now_tt = Time(utc, scale='tt', format=format)
    met = now_tt.cxcsec - 441763197.0

    return met


def fermi_met_to_utc(met):
    """
    Convert Fermi Mission Elapsed Time (MET) to UTC.

    Parameters:
    ----------
    met : float
        The Fermi MET in seconds since the reference epoch.

    Returns:
    -------
    str
        The corresponding UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).
    """

    dt = TimeDelta(met, format='sec')
    ref_utc = Time('2001-01-01T00:00:00.00', scale='utc', format='isot')
    now_utc = (ref_utc + dt).value

    return now_utc


def fermi_utc_to_met(utc, format='isot'):
    """
    Convert UTC time to Fermi Mission Elapsed Time (MET).

    Parameters:
    ----------
    utc : str
        The UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).
    format : str, optional
        The format of the input UTC time. Default is 'isot'.

    Returns:
    -------
    float
        The corresponding Fermi MET in seconds since the reference epoch.
    """
    
    ref_utc = Time('2001-01-01T00:00:00.00', scale='utc', format='isot')
    now_utc = Time(utc, scale='utc', format=format)
    met = (now_utc - ref_utc).sec

    return met


def fermi_utc_goback(utc, poshist_file):
    """
    Calculate a UTC time that is a certain period before the given UTC time,
    
    Parameters:
    ----------
    utc : str
        The UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).
    poshist_file : str
        The path to the position history FITS file.

    Returns:
    -------
    str
        The calculated UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).
    """
    
    poshist = fits.open(poshist_file)[1].data
    nt = np.size(poshist)

    pos = np.zeros((nt, 3), float)
    pos[:, 0] = poshist['POS_X']
    pos[:, 1] = poshist['POS_Y']
    pos[:, 2] = poshist['POS_Z']
    
    G = 6.67428e-11
    M = 5.9722e24
    r = (np.sum(pos ** 2.0, 1)) ** (1 / 2.0)
    r_avg = np.average(r)
    r_cubed = (r_avg) ** 3.0
    factor = r_cubed / (G * M)
    period = 2.0 * np.pi * np.sqrt(factor)

    utc = Time(utc, scale='utc', format='isot')
    dt = TimeDelta(period * 30, format='sec')
    goback_utc = (utc - dt).value
    
    return goback_utc


def gecam_met_to_utc(met):
    """
    Convert GECAM Mission Elapsed Time (MET) to UTC.

    Parameters:
    ----------
    met : float
        The GECAM MET in seconds since the reference epoch.

    Returns:
    -------
    str
        The corresponding UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).
    """

    dt = TimeDelta(met, format='sec')
    ref_utc = Time('2019-01-01T00:00:00.00', format='isot', scale='tt')
    now_utc = (ref_utc + dt).value

    return now_utc


def gecam_utc_to_met(utc, format='isot'):
    """
    Convert UTC time to GECAM Mission Elapsed Time (MET).

    Parameters:
    ----------
    utc : str
        The UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).
    format : str, optional
        The format of the input UTC time. Default is 'isot'.

    Returns:
    -------
    float
        The corresponding GECAM MET in seconds since the reference epoch.
    """

    now_utc = Time(utc, scale='tt', format=format)
    ref_utc = Time('2019-01-01T00:00:00.00', format='isot', scale='tt')
    met = (now_utc - ref_utc).sec
    
    return met


def grid_met_to_utc(met):
    """
    Convert GRID Mission Elapsed Time (MET) to UTC.

    Parameters:
    ----------
    met : float
        The GRID MET in seconds since the reference epoch.

    Returns:
    -------
    str
        The corresponding UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).

    """
    
    now_utc = Time(met, scale='utc', format='unix').to_value('isot')
    
    return now_utc


def grid_utc_to_met(isot, format='isot'):
    """
    Convert UTC time to GRID Mission Elapsed Time (MET).

    Parameters:
    ----------
    isot : str
        The UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).
    format : str, optional
        The format of the input UTC time. Default is 'isot'.

    Returns:
    -------
    float
        The corresponding GRID MET in seconds since the reference epoch.
    """
    
    now_utc = Time(isot, scale='utc', format=format)
    met = now_utc.to_value('unix')
    
    return met


def ep_utc_to_met(utc, format='isot'):
    """
    Convert UTC time to EP Mission Elapsed Time (MET).

    Parameters:
    ----------
    utc : str
        The UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).
    format : str, optional
        The format of the input UTC time. Default is 'isot'.

    Returns:
    -------
    float
        The corresponding EP MET in seconds since the reference epoch.
    """

    ref_utc = Time('2020-01-01T00:00:00.000', format='isot', scale='utc')
    now_utc = Time(utc, format=format, scale='utc')
    met = (now_utc - ref_utc).sec

    return met


def ep_met_to_utc(met):
    """
    Convert EP Mission Elapsed Time (MET) to UTC.

    Parameters:
    ----------
    met : float
        The EP MET in seconds since the reference epoch.

    Returns:
    -------
    str
        The corresponding UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).
    """
    
    ref_utc = Time('2020-01-01T00:00:00.000', format='isot', scale='utc')
    dt = TimeDelta(met, format='sec')
    now_utc = (ref_utc + dt).value

    return now_utc


def leia_utc_to_met(utc, format='isot'):
    """
    Convert UTC time to LEIA Mission Elapsed Time (MET).

    Parameters:
    ----------
    utc : str
        The UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).
    format : str, optional
        The format of the input UTC time. Default is 'isot'.

    Returns:
    -------
    float
        The corresponding LEIA MET in seconds since the reference epoch.
    """

    ref_utc = Time('2021-01-01T00:00:00.000', format='isot', scale='utc')
    now_utc = Time(utc, format=format, scale='utc')
    met = (now_utc - ref_utc).sec

    return met


def leia_met_to_utc(met):
    """
    Convert LEIA Mission Elapsed Time (MET) to UTC.

    Parameters:
    ----------
    met : float
        The LEIA MET in seconds since the reference epoch.

    Returns:
    -------
    str
        The corresponding UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).
    """

    ref_utc = Time('2021-01-01T00:00:00.000', format='isot', scale='utc')
    dt = TimeDelta(met, format='sec')
    now_utc = (ref_utc + dt).value

    return now_utc


def swift_met_to_utc(met, utcf):
    """
    Convert Swift Mission Elapsed Time (MET) to UTC.

    Parameters:
    ----------
    met : float
        The Swift MET in seconds since the reference epoch.
    utcf : float
        The UTC correction factor in seconds.

    Returns:
    -------
    str
        The corresponding UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).

    """

    dt = TimeDelta(met + utcf, format='sec')
    ref_tt = Time('2001-01-01T00:00:00.00', scale='tt', format='isot')
    now_utc = (ref_tt + dt).value

    return now_utc


def swift_utc_to_met(utc, utcf, format='isot'):
    """
    Convert UTC time to Swift Mission Elapsed Time (MET).

    Parameters:
    ----------
    utc : str
        The UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).
    utcf : float
        The UTC correction factor in seconds.
    format : str, optional
        The format of the input UTC time. Default is 'isot'.

    Returns:
    -------
    float
        The corresponding Swift MET in seconds since the reference epoch.

    """
    
    ref_tt = Time('2001-01-01T00:00:00.00', scale='tt', format='isot')
    now_utc = Time(utc, scale='tt', format=format)
    met = (now_utc - ref_tt).sec - utcf

    return met