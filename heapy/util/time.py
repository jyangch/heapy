"""MET/UTC conversion utilities for X-ray and gamma-ray astronomy missions.

Provides bidirectional conversion between Mission Elapsed Time (MET) and UTC
for HXMT, Fermi, GECAM, GRID, EP, LEIA, and Swift. Each mission defines its
own reference epoch and timescale; refer to the individual function pairs for
details.
"""

from astropy.io import fits
from astropy.time import Time, TimeDelta
import numpy as np


def hxmt_met_to_utc(met):
    """Convert HXMT Mission Elapsed Time (MET) to UTC.

    The HXMT MET is defined relative to 1998-01-01T00:00:00 TT, with an
    offset of 441763197.0 seconds applied before the conversion.  The
    ``hxmt_met_to_utc`` / ``hxmt_utc_to_met`` pair handles all HXMT
    time-system bookkeeping.

    Args:
        met: float, MET in seconds since the HXMT reference epoch.

    Returns:
        UTC time string in ISO format (YYYY-MM-DDTHH:MM:SS).
    """

    dt = TimeDelta(met + 441763197.0, format='sec')
    ref_tt = Time('1998-01-01T00:00:00', format='isot', scale='tt')
    now_utc = (ref_tt + dt).value

    return now_utc


def hxmt_utc_to_met(utc, format='isot'):
    """Convert UTC time to HXMT Mission Elapsed Time (MET).

    Args:
        utc: str, UTC time string (YYYY-MM-DDTHH:MM:SS).
        format: str, ``astropy.time`` format of ``utc``. Default is ``'isot'``.

    Returns:
        HXMT MET in seconds since the reference epoch.
    """

    now_tt = Time(utc, scale='tt', format=format)
    met = now_tt.cxcsec - 441763197.0

    return met


def fermi_met_to_utc(met):
    """Convert Fermi Mission Elapsed Time (MET) to UTC.

    The Fermi MET epoch is 2001-01-01T00:00:00 UTC.  The
    ``fermi_met_to_utc`` / ``fermi_utc_to_met`` pair handles all Fermi
    time-system bookkeeping.

    Args:
        met: float, Fermi MET in seconds since 2001-01-01T00:00:00 UTC.

    Returns:
        UTC time string in ISO format (YYYY-MM-DDTHH:MM:SS).
    """

    dt = TimeDelta(met, format='sec')
    ref_utc = Time('2001-01-01T00:00:00.00', scale='utc', format='isot')
    now_utc = (ref_utc + dt).value

    return now_utc


def fermi_utc_to_met(utc, format='isot'):
    """Convert UTC time to Fermi Mission Elapsed Time (MET).

    Args:
        utc: str, UTC time string (YYYY-MM-DDTHH:MM:SS).
        format: str, ``astropy.time`` format of ``utc``. Default is ``'isot'``.

    Returns:
        Fermi MET in seconds since 2001-01-01T00:00:00 UTC.
    """

    ref_utc = Time('2001-01-01T00:00:00.00', scale='utc', format='isot')
    now_utc = Time(utc, scale='utc', format=format)
    met = (now_utc - ref_utc).sec

    return met


def fermi_utc_goback(utc, poshist_file):
    r"""Return the UTC time approximately 30 orbital periods before ``utc``.

    Computes the Fermi orbital period from spacecraft position vectors stored
    in ``poshist_file`` using Kepler's third law
    (:math:`T = 2\\pi\\sqrt{r^3 / (GM)}`), then subtracts 30 periods from the
    input time.

    Args:
        utc: str, UTC time in ISO format (YYYY-MM-DDTHH:MM:SS).
        poshist_file: str, path to the Fermi position history FITS file
            containing ``POS_X``, ``POS_Y``, and ``POS_Z`` columns in metres.

    Returns:
        UTC time string in ISO format (YYYY-MM-DDTHH:MM:SS) that is ~30
        orbital periods before ``utc``.
    """

    poshist = fits.open(poshist_file)[1].data
    nt = np.size(poshist)

    pos = np.zeros((nt, 3), float)
    pos[:, 0] = poshist['POS_X']
    pos[:, 1] = poshist['POS_Y']
    pos[:, 2] = poshist['POS_Z']

    G = 6.67428e-11
    M = 5.9722e24
    r = (np.sum(pos**2.0, 1)) ** (1 / 2.0)
    r_avg = np.average(r)
    r_cubed = (r_avg) ** 3.0
    factor = r_cubed / (G * M)
    period = 2.0 * np.pi * np.sqrt(factor)

    utc = Time(utc, scale='utc', format='isot')
    dt = TimeDelta(period * 30, format='sec')
    goback_utc = (utc - dt).value

    return goback_utc


def gecam_met_to_utc(met):
    """Convert GECAM Mission Elapsed Time (MET) to UTC.

    The GECAM MET epoch is 2019-01-01T00:00:00 TT.  The
    ``gecam_met_to_utc`` / ``gecam_utc_to_met`` pair handles all GECAM
    time-system bookkeeping.

    Args:
        met: float, GECAM MET in seconds since 2019-01-01T00:00:00 TT.

    Returns:
        UTC time string in ISO format (YYYY-MM-DDTHH:MM:SS).
    """

    dt = TimeDelta(met, format='sec')
    ref_utc = Time('2019-01-01T00:00:00.00', format='isot', scale='tt')
    now_utc = (ref_utc + dt).value

    return now_utc


def gecam_utc_to_met(utc, format='isot'):
    """Convert UTC time to GECAM Mission Elapsed Time (MET).

    Args:
        utc: str, UTC time string (YYYY-MM-DDTHH:MM:SS).
        format: str, ``astropy.time`` format of ``utc``. Default is ``'isot'``.

    Returns:
        GECAM MET in seconds since 2019-01-01T00:00:00 TT.
    """

    now_utc = Time(utc, scale='tt', format=format)
    ref_utc = Time('2019-01-01T00:00:00.00', format='isot', scale='tt')
    met = (now_utc - ref_utc).sec

    return met


def grid_met_to_utc(met):
    """Convert GRID Mission Elapsed Time (MET) to UTC.

    The GRID MET is Unix time (seconds since 1970-01-01T00:00:00 UTC).  The
    ``grid_met_to_utc`` / ``grid_utc_to_met`` pair handles all GRID
    time-system bookkeeping.

    Args:
        met: float, GRID MET as a Unix timestamp in seconds.

    Returns:
        UTC time string in ISO format (YYYY-MM-DDTHH:MM:SS).
    """

    now_utc = Time(met, scale='utc', format='unix').to_value('isot')

    return now_utc


def grid_utc_to_met(isot, format='isot'):
    """Convert UTC time to GRID Mission Elapsed Time (MET).

    Args:
        isot: str, UTC time string (YYYY-MM-DDTHH:MM:SS).
        format: str, ``astropy.time`` format of ``isot``. Default is ``'isot'``.

    Returns:
        GRID MET as a Unix timestamp in seconds.
    """

    now_utc = Time(isot, scale='utc', format=format)
    met = now_utc.to_value('unix')

    return met


def ep_utc_to_met(utc, format='isot'):
    """Convert UTC time to EP Mission Elapsed Time (MET).

    The EP MET epoch is 2020-01-01T00:00:00 UTC.  The
    ``ep_utc_to_met`` / ``ep_met_to_utc`` pair handles all EP
    time-system bookkeeping.

    Args:
        utc: str, UTC time string (YYYY-MM-DDTHH:MM:SS).
        format: str, ``astropy.time`` format of ``utc``. Default is ``'isot'``.

    Returns:
        EP MET in seconds since 2020-01-01T00:00:00 UTC.
    """

    ref_utc = Time('2020-01-01T00:00:00.000', format='isot', scale='utc')
    now_utc = Time(utc, format=format, scale='utc')
    met = (now_utc - ref_utc).sec

    return met


def ep_met_to_utc(met):
    """Convert EP Mission Elapsed Time (MET) to UTC.

    Args:
        met: float, EP MET in seconds since 2020-01-01T00:00:00 UTC.

    Returns:
        UTC time string in ISO format (YYYY-MM-DDTHH:MM:SS).
    """

    ref_utc = Time('2020-01-01T00:00:00.000', format='isot', scale='utc')
    dt = TimeDelta(met, format='sec')
    now_utc = (ref_utc + dt).value

    return now_utc


def leia_utc_to_met(utc, format='isot'):
    """Convert UTC time to LEIA Mission Elapsed Time (MET).

    The LEIA MET epoch is 2021-01-01T00:00:00 UTC.  The
    ``leia_utc_to_met`` / ``leia_met_to_utc`` pair handles all LEIA
    time-system bookkeeping.

    Args:
        utc: str, UTC time string (YYYY-MM-DDTHH:MM:SS).
        format: str, ``astropy.time`` format of ``utc``. Default is ``'isot'``.

    Returns:
        LEIA MET in seconds since 2021-01-01T00:00:00 UTC.
    """

    ref_utc = Time('2021-01-01T00:00:00.000', format='isot', scale='utc')
    now_utc = Time(utc, format=format, scale='utc')
    met = (now_utc - ref_utc).sec

    return met


def leia_met_to_utc(met):
    """Convert LEIA Mission Elapsed Time (MET) to UTC.

    Args:
        met: float, LEIA MET in seconds since 2021-01-01T00:00:00 UTC.

    Returns:
        UTC time string in ISO format (YYYY-MM-DDTHH:MM:SS).
    """

    ref_utc = Time('2021-01-01T00:00:00.000', format='isot', scale='utc')
    dt = TimeDelta(met, format='sec')
    now_utc = (ref_utc + dt).value

    return now_utc


def swift_met_to_utc(met, utcf):
    """Convert Swift Mission Elapsed Time (MET) to UTC.

    The Swift MET epoch is 2001-01-01T00:00:00 TT.  A UTC correction factor
    ``utcf`` (provided in the event FITS header) is added to ``met`` before
    conversion.  The ``swift_met_to_utc`` / ``swift_utc_to_met`` pair handles
    all Swift time-system bookkeeping.

    Args:
        met: float, Swift MET in seconds since 2001-01-01T00:00:00 TT.
        utcf: float, UTC correction factor in seconds (from the FITS header
            keyword ``UTCFINIT`` or equivalent).

    Returns:
        UTC time string in ISO format (YYYY-MM-DDTHH:MM:SS).
    """

    dt = TimeDelta(met + utcf, format='sec')
    ref_tt = Time('2001-01-01T00:00:00.00', scale='tt', format='isot')
    now_utc = (ref_tt + dt).value

    return now_utc


def swift_utc_to_met(utc, utcf, format='isot'):
    """Convert UTC time to Swift Mission Elapsed Time (MET).

    Args:
        utc: str, UTC time string (YYYY-MM-DDTHH:MM:SS).
        utcf: float, UTC correction factor in seconds.
        format: str, ``astropy.time`` format of ``utc``. Default is ``'isot'``.

    Returns:
        Swift MET in seconds since 2001-01-01T00:00:00 TT.
    """

    ref_tt = Time('2001-01-01T00:00:00.00', scale='tt', format='isot')
    now_utc = Time(utc, scale='tt', format=format)
    met = (now_utc - ref_tt).sec - utcf

    return met
