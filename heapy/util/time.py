import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta


def hxmt_met_to_utc(met):

    dt = TimeDelta(met + 441763197.0, format='sec')
    ref_tt = Time('1998-01-01T00:00:00', format='isot', scale='tt')
    now_utc = (ref_tt + dt).value

    # or below: 
    # mjdshort = met - 3
    # mjd = float(mjdshort) / 86400. + 55927
    # mjdtime = Time(mjd, format='mjd')
    # isot = mjdtime.isot

    return now_utc


def hxmt_utc_to_met(utc, format='isot'):

    now_tt = Time(utc, scale='tt', format=format)
    met = now_tt.cxcsec - 441763197.0

    return met


def fermi_met_to_utc(met):

    dt = TimeDelta(met, format='sec')
    ref_utc = Time('2001-01-01T00:00:00.00', scale='utc', format='isot')
    now_utc = (ref_utc + dt).value

    return now_utc


def fermi_utc_to_met(utc, format='isot'):

    ref_utc = Time('2001-01-01T00:00:00.00', scale='utc', format='isot')
    now_utc = Time(utc, scale='utc', format=format)
    met = (now_utc - ref_utc).sec

    return met


def fermi_utc_goback(utc, poshist_file):
    
    poshist = fits.open(poshist_file)[1].data
    nt = np.size(poshist)
    sc_time = poshist['SCLK_UTC']
    sc_quat = np.zeros((nt,4),float)
    sc_pos = np.zeros((nt,3),float)
    sc_coords = np.zeros((nt,2),float)
    try:
        sc_coords[:,0] = poshist['SC_LON']
        sc_coords[:,1] = poshist['SC_LAT']
    except:
        msg = ''
        msg += '*** No geographical coordinates available '
        msg += 'for this file: %s' % poshist_file
        print(msg)

    sc_quat[:,0] = poshist['QSJ_1']
    sc_quat[:,1] = poshist['QSJ_2']
    sc_quat[:,2] = poshist['QSJ_3']
    sc_quat[:,3] = poshist['QSJ_4']
    sc_pos[:,0] = poshist['POS_X']
    sc_pos[:,1] = poshist['POS_Y']
    sc_pos[:,2] = poshist['POS_Z']
    
    G = 6.67428e-11
    M = 5.9722e24
    r = (np.sum(sc_pos ** 2.0, 1)) ** (1 / 2.0)
    r_avg = np.average(r)
    r_cubed = (r_avg) ** 3.0
    factor = r_cubed / (G * M)
    period = 2.0 * np.pi * np.sqrt(factor)

    utc = Time(utc, scale='utc', format='isot')
    dt = TimeDelta(period * 30, format='sec')
    goback_utc = (utc - dt).value
    
    return goback_utc


def gecam_met_to_utc(met):

    dt = TimeDelta(met, format='sec')
    ref_utc = Time('2019-01-01T00:00:00.00', format='isot', scale='tt')
    now_utc = (ref_utc + dt).value

    return now_utc


def gecam_utc_to_met(utc, format='isot'):

    now_utc = Time(utc, scale='tt', format=format)
    ref_utc = Time('2019-01-01T00:00:00.00', format='isot', scale='tt')
    met = (now_utc - ref_utc).sec
    
    return met


def grid_met_to_utc(met):
    
    now_utc = Time(met, scale='utc', format='unix').to_value('isot')
    
    return now_utc


def grid_utc_to_met(isot, format='isot'):
    
    now_utc = Time(isot, scale='utc', format=format)
    met = now_utc.to_value('unix')
    
    return met


def ep_utc_to_met(utc, format='isot'):

    ref_utc = Time('2020-01-01T00:00:00.000', format='isot', scale='utc')
    now_utc = Time(utc, format=format, scale='utc')
    met = (now_utc - ref_utc).sec

    return met


def ep_met_to_utc(met):

    ref_utc = Time('2020-01-01T00:00:00.000', format='isot', scale='utc')
    dt = TimeDelta(met, format='sec')
    now_utc = (ref_utc + dt).value

    return now_utc


def leia_utc_to_met(utc, format='isot'):

    ref_utc = Time('2021-01-01T00:00:00.000', format='isot', scale='utc')
    now_utc = Time(utc, format=format, scale='utc')
    met = (now_utc - ref_utc).sec

    return met


def leia_met_to_utc(met):

    ref_utc = Time('2021-01-01T00:00:00.000', format='isot', scale='utc')
    dt = TimeDelta(met, format='sec')
    now_utc = (ref_utc + dt).value

    return now_utc


def swift_met_to_utc(met, utcf):

    dt = TimeDelta(met + utcf, format='sec')
    ref_tt = Time('2001-01-01T00:00:00.00', scale='tt', format='isot')
    now_utc = (ref_tt + dt).value

    return now_utc


def swift_utc_to_met(utc, utcf, format='isot'):

    ref_tt = Time('2001-01-01T00:00:00.00', scale='tt', format='isot')
    now_utc = Time(utc, scale='tt', format=format)
    met = (now_utc - ref_tt).sec - utcf

    return met