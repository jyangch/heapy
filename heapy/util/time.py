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
