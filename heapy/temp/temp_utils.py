"""Shared helpers and plotter for the temporal analysis pipeline classes.

Bundles shared utilities consumed by :mod:`heapy.temp.txx`,
:mod:`heapy.temp.lag`, and :mod:`heapy.temp.mvt`:

- Cumulative-count-fraction math (:func:`calculate_txx`, :func:`find_txx`)
  that derives Txx start/stop times from a pulse's cumulative net
  count curve. Both functions are stateless and reusable outside the
  Txx classes.
- The Haar minimum-variability-timescale core (:func:`haar_denoise`,
  :func:`calculate_haar_power_spectrum`, :func:`calculate_haar_mvt`),
  consumed by :mod:`heapy.temp.mvt`. This is a faithful Python-3 port of
  the public code at https://github.com/nrbutler/mvt, shared by Nat
  Butler for reproducing Golkhou & Butler (2014) and Golkhou, Butler &
  Littlejohns (2015); do not edit these three functions' numerics
  without re-diffing against the upstream source.
- :func:`uniform_dt_from_bins`, a bin-edge validation helper shared by
  every Signal-to-MVT bridging path.
- Monte Carlo sampling, box smoothing, and CCF batch calculation helpers
  shared by the temporal analysis classes.
- :class:`TxxPlotter`, a composable two-panel diagnostic figure that
  every ``save()`` method in :mod:`heapy.temp.txx` shares (mirrors the
  :class:`~heapy.auto.signal_utils.SignalPlotter` design).
"""

import operator

from astropy.stats import mad_std, sigma_clip
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

from ..auto.signal_utils import indices_in_intervals
from ..util.data import generate_asymmetric_gaussian


def validate_input(dtype, cts, cts_err, bcts, bcts_err, label=''):
    """Validate input arrays for temporal analysis.

    Args:
        dtype: Noise model: ``'pg'`` (Poisson source + Gaussian
            background), ``'pp'`` (Poisson source + Poisson
            background), or ``'gg'`` (Gaussian net counts, no separate
            background).
        cts: Primary count array.
        cts_err: Primary count error array.
        bcts: Background count array.
        bcts_err: Background count error array.
        label: Label for error messages. Defaults to empty string.
    Returns:
        Tuple of (cts_err, bcts, bcts_err) after validation.
    """

    cts = np.asarray(cts, dtype=float)

    if dtype == 'pg':
        cts_err = np.sqrt(cts) if cts_err is None else cts_err
        if bcts is None:
            raise ValueError(f'unknown {label}bcts')
        if bcts_err is None:
            raise ValueError(f'unknown {label}bcts_err')

    elif dtype == 'pp':
        cts_err = np.sqrt(cts) if cts_err is None else cts_err
        if bcts is None:
            raise ValueError(f'unknown {label}bcts')
        bcts_err = np.sqrt(bcts) if bcts_err is None else bcts_err

    elif dtype == 'gg':
        if cts_err is None:
            raise ValueError(f'unknown {label}cts_err')
        bcts = np.zeros_like(cts, dtype=float) if bcts is None else bcts
        bcts_err = np.zeros_like(cts, dtype=float) if bcts_err is None else bcts_err

    else:
        raise ValueError(f'unknown {label}type')

    return (
        np.asarray(cts_err, dtype=float),
        np.asarray(bcts, dtype=float),
        np.asarray(bcts_err, dtype=float),
    )


def generate_mc_sample(dtype, cts, cts_err, bcts, bcts_err, nmc, rng, backscale=1):
    """Generate Monte Carlo realisations of background-subtracted counts.

    Args:
        dtype: Noise model: ``'pg'`` (Poisson source + Gaussian
            background), ``'pp'`` (Poisson source + Poisson background),
            or ``'gg'`` (Gaussian source + Gaussian background).
        cts: Source or net-count expectation per bin.
        cts_err: Source/net-count errors for Gaussian sampling.
        bcts: Background expectation per bin.
        bcts_err: Background errors for Gaussian sampling.
        nmc: Number of Monte Carlo realisations to generate.
        rng: ``numpy.random.Generator`` used for all random draws.
        backscale: Multiplicative scale applied to sampled background.
            Defaults to 1.

    Returns:
        A ``(nmc, nsample)`` array of net-count realisations.

    Raises:
        ValueError: If ``dtype`` is not supported.
    """

    cts = np.asarray(cts, dtype=float)
    cts_err = np.asarray(cts_err, dtype=float)
    bcts = np.asarray(bcts, dtype=float)
    bcts_err = np.asarray(bcts_err, dtype=float)
    size = (int(nmc), cts.size)

    if dtype == 'pg':
        src = rng.poisson(lam=np.nan_to_num(cts, nan=0.0), size=size)
        bkg = rng.normal(loc=bcts, scale=bcts_err, size=size)

    elif dtype == 'pp':
        src = rng.poisson(lam=np.nan_to_num(cts, nan=0.0), size=size)
        bkg = rng.poisson(lam=np.nan_to_num(bcts, nan=0.0), size=size)

    elif dtype == 'gg':
        src = rng.normal(loc=cts, scale=cts_err, size=size)
        bkg = rng.normal(loc=bcts, scale=bcts_err, size=size)

    else:
        raise ValueError(f'unknown dtype {dtype!r}, expected pg, pp, or gg')

    return src - bkg * backscale


def calculate_txx(time, ccts, pstart, pstop, xx, simple_err=False, random_seed=450001):
    """Calculate Txx and its uncertainties from a cumulative count curve.

    Interpolates the cumulative count curve to 1000-point resolution when
    the native sampling is too coarse, calculates the background CSF levels
    between pulses, derives the :math:`(1-xx)/2` and :math:`1-(1-xx)/2`
    fraction levels (``csf1``, ``csf2``) for each pulse interval, and
    locates the corresponding times via :func:`find_txx`.

    Args:
        time: 1-D time array covering the analysis window.
        ccts: Cumulative net count array aligned with ``time``.
        pstart: Array of pulse start times; one entry per pulse.
        pstop: Array of pulse stop times; one entry per pulse.
        xx: Cumulative count fraction, e.g. ``0.9`` for T90.
        simple_err: When ``True``, also compute and return analytic
            uncertainty estimates on all CSF and Txx values.
        random_seed: Seed for the local RNG used by the asymmetric
            Gaussian sampler in the ``simple_err`` branch. Default
            ensures reproducibility; ignored when ``simple_err=False``.

    Returns:
        When ``simple_err`` is ``False``: a tuple
        ``(txx, txx1, txx2, csf, csf1, csf2)`` where each element is a list
        with one entry per pulse.

        When ``simple_err`` is ``True``: a tuple
        ``(txx, txx1, txx2, txx_err, txx1_err, txx2_err,
        csf, csf1, csf2, csf_err, csf1_err, csf2_err)``.
    """

    rng = np.random.default_rng(random_seed)

    idx = np.argsort(pstop - pstart)[0]
    if len(np.where((time >= pstart[idx]) & (time <= pstop[idx]))[0]) < 1000:
        interp_dt = (pstop[idx] - pstart[idx]) / 1000
        interp_time = np.arange(time[0], time[-1] - 1e-5, interp_dt)
        # Quadratic splines need >= 3 nodes; coarse light curves (e.g. the
        # ~3 s Konus-Wind binning) can leave only 2 samples in the analysis
        # window, so fall back to linear interpolation there.
        interp_kind = 'quadratic' if len(time) >= 3 else 'linear'
        interp = interp1d(time, ccts, kind=interp_kind)
        interp_ccts = interp(interp_time)
    else:
        interp_time = time
        interp_ccts = ccts

    csf, csf1, csf2 = [], [], []
    txx, txx1, txx2 = [], [], []

    if simple_err:
        csf_err, csf1_err, csf2_err = [], [], []
        txx_err, txx1_err, txx2_err = [], [], []

    for left, right in zip(np.append(time[0], pstop), np.append(pstart, time[-1]), strict=False):
        idx = np.where((interp_time >= left) & (interp_time <= right))[0]
        if len(idx) >= 1:
            csf_i = np.mean(interp_ccts[idx])
        else:
            csf_i = interp_ccts[np.argmin(np.abs(interp_time - left))]
        csf.append(csf_i)

        if simple_err:
            csf_err_i = np.std(interp_ccts[idx]) if len(idx) >= 1 else 0.0
            csf_err.append(csf_err_i)

    dcsf = np.array(csf[1:]) - np.array(csf[:-1])

    for pi, (left, right) in enumerate(
        zip(np.append(time[0], pstop[:-1]), np.append(pstart[1:], time[-1]), strict=False)
    ):
        nn = (1 - xx) / 2
        dd = dcsf[pi] * nn

        csf1_i = csf[pi] + dd
        csf2_i = csf[pi + 1] - dd

        csf1.append(csf1_i)
        csf2.append(csf2_i)

        if simple_err:
            csf1_err_i = np.sqrt((1 - nn) ** 2 * csf_err[pi] ** 2 + nn**2 * csf_err[pi + 1] ** 2)
            csf2_err_i = np.sqrt(nn**2 * csf_err[pi] ** 2 + (1 - nn) ** 2 * csf_err[pi + 1] ** 2)

            csf1_err.append(csf1_err_i)
            csf2_err.append(csf2_err_i)

        pt = interp_time[np.where((interp_time >= left) & (interp_time <= right))]
        pcts = interp_ccts[np.where((interp_time >= left) & (interp_time <= right))]

        txx_i, txx1_i, txx2_i = find_txx(pt, pcts, csf1_i, csf2_i)

        txx.append(txx_i)
        txx1.append(txx1_i)
        txx2.append(txx2_i)

        if simple_err:
            _, txx1_lo_i, txx2_lo_i = find_txx(pt, pcts, csf1_i - csf1_err_i, csf2_i - csf2_err_i)
            _, txx1_hi_i, txx2_hi_i = find_txx(pt, pcts, csf1_i + csf1_err_i, csf2_i + csf2_err_i)
            txx1_le_i, txx1_he_i = txx1_i - txx1_lo_i, txx1_hi_i - txx1_i
            txx2_le_i, txx2_he_i = txx2_i - txx2_lo_i, txx2_hi_i - txx2_i

            txx1_i_sam = generate_asymmetric_gaussian(txx1_i, txx1_le_i, txx1_he_i, 1000, rng=rng)
            txx2_i_sam = generate_asymmetric_gaussian(txx2_i, txx2_le_i, txx2_he_i, 1000, rng=rng)
            txx_lo_i, txx_hi_i = np.percentile(txx2_i_sam - txx1_i_sam, [16, 84])
            txx_le_i, txx_he_i = txx_i - txx_lo_i, txx_hi_i - txx_i

            txx_err.append([txx_le_i, txx_he_i])
            txx1_err.append([txx1_le_i, txx1_he_i])
            txx2_err.append([txx2_le_i, txx2_he_i])

    if simple_err:
        return (
            txx,
            txx1,
            txx2,
            txx_err,
            txx1_err,
            txx2_err,
            csf,
            csf1,
            csf2,
            csf_err,
            csf1_err,
            csf2_err,
        )
    else:
        return txx, txx1, txx2, csf, csf1, csf2


def find_txx(time, ccts, csf1, csf2):
    """Locate the start and stop times corresponding to CSF thresholds.

    Linearly interpolates the cumulative count curve onto a 1000-point grid,
    then scans forward to find the time ``txx1`` at which the curve crosses
    ``csf1`` from below, and ``txx2`` at which it crosses ``csf2`` from
    below.  The duration ``txx = txx2 - txx1`` is also returned.

    Args:
        time: 1-D time array for the pulse interval.
        ccts: Cumulative net count array aligned with ``time``.
        csf1: Lower cumulative-count-fraction level (start threshold).
        csf2: Upper cumulative-count-fraction level (stop threshold).

    Returns:
        A tuple ``(txx, txx1, txx2)`` where ``txx`` is the duration and
        ``txx1``, ``txx2`` are the start and stop times, respectively.
        All three values are ``0`` if no valid crossing is found.
    """

    interp_time = np.linspace(time[0], time[-1], 1000)
    interp = interp1d(time, ccts, kind='linear')
    interp_ccts = interp(interp_time)

    txx1, txx2 = 0, 0
    for i in range(1, len(interp_time)):
        if interp_ccts[i] < csf1:
            continue
        elif (interp_ccts[i - 1] < csf1) and (interp_ccts[i] >= csf1):
            txx1 = interp_time[i]
            continue
        elif (csf1 <= interp_ccts[i - 1] < csf2) and (csf1 < interp_ccts[i] <= csf2):
            continue
        elif (csf1 < interp_ccts[i - 1] <= csf2) and (interp_ccts[i] > csf2):
            txx2 = interp_time[i - 1]
            break
        else:
            continue

    txx = txx2 - txx1
    return txx, txx1, txx2


def get_mc_errors(mc_values):
    """Get per-pulse 1-sigma error bars from Monte Carlo realisations.

    Args:
        mc_values: Monte Carlo realisations.
    Returns:
        List of [lower error, upper error] for each pulse.
    """

    errors = []
    for pi in range(mc_values.shape[1]):
        mask = sigma_clip(mc_values[1:, pi], sigma=5, maxiters=5, stdfunc=mad_std).mask
        not_mask = list(map(operator.not_, mask))
        filtered = mc_values[1:, pi][not_mask]

        lo, hi = np.percentile(filtered, [16, 84])
        err = np.diff([lo, mc_values[0, pi], hi])
        errors.append([err[0], err[1]])

    return errors


def box_smooth(arr, M):
    """Box-smooth an array.

    Args:
        arr: Array to smooth.
        M: Box width.
    Returns:
        Smoothed array.
    """

    if M == 1:
        return np.asarray(arr, dtype=float).copy()

    return np.convolve(arr, np.ones(M), mode='valid')


def box_smooth_batch(arr2d, M):
    """Box-smooth a 2D array.

    Args:
        arr2d: 2D array to smooth.
        M: Box width.
    Returns:
        Smoothed 2D array.
    """

    if M == 1:
        return np.asarray(arr2d, dtype=float).copy()

    arr2d = np.asarray(arr2d, dtype=float)

    cumsum = np.concatenate(
        [np.zeros((arr2d.shape[0], 1), dtype=arr2d.dtype), np.cumsum(arr2d, axis=1)], axis=1
    )

    return cumsum[:, M:] - cumsum[:, :-M]


def calculate_ccf_batch(mc_xncts, mc_yncts):
    """Calculate the cross-correlation function for a batch of Monte Carlo realisations.

    Args:
        mc_xncts: Monte Carlo realisations of the x-axis data.
        mc_yncts: Monte Carlo realisations of the y-axis data.
    Returns:
        Cross-correlation function for the batch.
    """

    n = mc_xncts.shape[1]
    nfft = next_fast_len(2 * n - 1)

    X_rev = rfft(mc_xncts[:, ::-1].copy(), n=nfft, axis=1)
    Y = rfft(mc_yncts, n=nfft, axis=1)
    all_ccfs = irfft(Y * X_rev, n=nfft, axis=1)[:, : 2 * n - 1]

    norms = np.sqrt(np.sum(mc_xncts**2, axis=1) * np.sum(mc_yncts**2, axis=1))
    zero_mask = norms == 0
    norms[zero_mask] = 1.0
    all_ccfs /= norms[:, np.newaxis]
    all_ccfs[zero_mask] = 0.0

    return all_ccfs


def uniform_dt_from_bins(bins):
    """Derive a scalar bin width from bin edges, requiring uniform spacing.

    Args:
        bins: 1-D array of bin edges (length >= 2).

    Returns:
        The median bin width as a float.

    Raises:
        ValueError: If ``bins`` is not a one-dimensional array of at
            least two edges, or if the bin widths are not uniform to
            within a relative tolerance of ``1e-7``.
    """

    bins = np.asarray(bins, dtype=float)
    if bins.ndim != 1 or bins.size < 2:
        raise ValueError('bins must be a one-dimensional edge array')

    widths = np.diff(bins)
    dt = float(np.median(widths))
    if not np.allclose(widths, dt, rtol=1e-7, atol=max(1e-12, abs(dt) * 1e-9)):
        raise ValueError('the Haar MVT core requires uniform bins')

    return dt


def haar_denoise(data, err=None, thresh_fac=1.0, estimate_noise=False, soft=False):
    """Denoise a 1-D series via non-decimated Haar wavelet hard-thresholding.

    Port of ``nrbutler/mvt`` ``haar_denoise.py``: extends the series by
    mirror reflection to the next power-of-two length, computes the
    non-decimated (a-trous style) Haar wavelet coefficients at every
    dyadic scale via cumulative sums, hard-thresholds each coefficient
    against a noise level ``thresh_fac * noise`` (scaled per-coefficient
    by the propagated variance when ``err`` is given), and reconstructs
    by inverting the transform. :func:`calculate_haar_mvt` calls this
    twice to build a smoothed, non-negative weight curve that downweights
    low-signal bins in :func:`calculate_haar_power_spectrum`.

    Args:
        data: 1-D series to denoise.
        err: Optional 1-sigma error per point. When given, thresholding
            uses the propagated coefficient variance instead of a flat
            noise level, and ``estimate_noise`` normalizes by it too.
        thresh_fac: Threshold multiplier; larger values denoise more
            aggressively.
        estimate_noise: When ``True``, estimate the noise level from the
            median absolute first difference of ``data`` (Donoho-Johnstone
            style) instead of treating ``thresh_fac`` as an absolute
            level.
        soft: When ``True``, soft-threshold (shrink toward zero) surviving
            coefficients instead of leaving them untouched.

    Returns:
        Denoised series, same length as ``data``.

    Raises:
        ValueError: If ``data`` is not one-dimensional, or ``err`` does
            not match ``data``'s shape.
    """

    data = np.asarray(data, dtype='float64')
    if data.ndim != 1:
        raise ValueError('data must be one-dimensional')

    cx = 1.0 * data
    ln0 = len(data)
    if ln0 == 0:
        return cx

    use_err = err is not None and len(err) != 0
    if use_err:
        err = np.asarray(err, dtype='float64')
        if err.shape != data.shape:
            raise ValueError('err must match data shape')
        vx = err**2

    n = int(np.ceil(np.log2(ln0)))
    ln = 2**n

    l1 = 0
    if ln > ln0:
        l1 = int(0.5 * (ln - ln0))
        l2 = ln - ln0 - l1
        cx = np.hstack((cx[:l1][::-1], cx, cx[-l2:][::-1]))
        if use_err:
            vx = np.hstack((vx[:l1][::-1], vx, vx[-l2:][::-1]))

    noise = 1.0
    if estimate_noise:
        if use_err:
            err2 = (1.0 / np.sqrt(2.0)) * np.sqrt(err[1:] ** 2 + err[:-1] ** 2)
            noise = 1.05 * np.median(np.abs(data[1:] - data[:-1]) / err2)
        else:
            noise = 1.05 * np.median(np.abs(data[1:] - data[:-1]))

    x0 = cx.mean()
    cx -= x0
    xm = np.empty(ln, dtype='float64')
    x_recon = np.zeros(ln, dtype='float64') + x0
    cx[:] = cx.cumsum()
    if use_err:
        vx[:] = vx.cumsum()
        vxm = np.empty(ln, dtype='float64')

    tlt = 1.386 * (thresh_fac * noise) ** 2
    for m in range(n):
        scl = 2 ** (n - m - 1)

        xm[: -2 * scl] = 2 * cx[scl:-scl] - cx[: -2 * scl] - cx[2 * scl :]
        xm[-2 * scl : -scl] = 2 * cx[-scl:] - cx[-2 * scl : -scl] - cx[:scl] - cx[-1]
        xm[-scl:] = 2 * cx[:scl] - cx[-scl:] - cx[scl : 2 * scl] + cx[-1]

        if use_err:
            vxm[: -2 * scl] = vx[2 * scl :] - vx[: -2 * scl]
            vxm[-2 * scl :] = vx[: 2 * scl] + vx[-1] - vx[-2 * scl :]
        else:
            vxm = 2 * scl

        h = xm * xm <= tlt * m * vxm
        xm[h] = 0
        if soft:
            mh = ~h
            if use_err:
                xm[mh] *= np.sqrt(1.0 - tlt * m * vxm[mh] / xm[mh] ** 2)
            else:
                xm[mh] *= np.sqrt(1.0 - tlt * m * vxm / xm[mh] ** 2)

        xm[:] = xm[::-1].cumsum()
        x_recon[2 * scl :] += (2 * xm[scl:-scl] - xm[: -2 * scl] - xm[2 * scl :])[::-1] / (
            2 * scl
        ) ** 2
        x_recon[scl : 2 * scl] += (2 * xm[-scl:] - xm[-2 * scl : -scl] - xm[:scl] - xm[-1])[
            ::-1
        ] / (2 * scl) ** 2
        x_recon[:scl] += (2 * xm[:scl] - xm[-scl:] - xm[scl : 2 * scl] + xm[-1])[::-1] / (
            2 * scl
        ) ** 2

    return x_recon[l1 : l1 + ln0]


def calculate_haar_power_spectrum(data, error, weight, dt=1.0, osamp=32.0, nrepl=1, bfac=4.0):
    """Compute the weighted, noise-corrected Haar wavelet power spectrum.

    Port of ``nrbutler/mvt`` ``haar_nondec_regular_err_wt.py``. For a
    regularly-sampled series, evaluates the non-decimated (sliding)
    Haar wavelet coefficient at each of a set of dyadic-plus-oversampled
    timescales via cumulative sums, weights each coefficient by
    ``weight``, and separately propagates the coefficient's noise
    variance from ``error``. Adjacent scales are then averaged down onto
    a coarser, ``bfac``-controlled output grid. This per-scale structure
    function is what :func:`calculate_haar_mvt` searches for the
    signal-to-noise-power crossing that defines the MVT.

    Args:
        data: 1-D regularly-sampled series (e.g. background-subtracted
            count rate).
        error: 1-sigma error per point; same shape as ``data``.
        weight: Non-negative per-point weight (typically a denoised copy
            of ``data`` from :func:`haar_denoise`) used to downweight
            low-signal bins in the per-scale average.
        dt: Bin width in seconds; scales the returned timescale bounds.
        osamp: Oversampling factor for the internal (pre-averaging) scale
            grid; must be ``>= bfac``.
        nrepl: Number of times to replicate the series end-to-end before
            transforming (variance-reduction trick for short series).
        bfac: Bin factor controlling the density of the output timescale
            grid relative to the dyadic scales.

    Returns:
        A 5-tuple ``(dt_lo, dt_hi, power, noise_power, power_err)``, each
        a 1-D array over output timescale bins: the bin's lower and
        upper edge in seconds, the weighted wavelet power, its
        noise-only counterpart (the zero-signal expectation from
        ``error``), and the propagated 1-sigma error on ``power``.

    Raises:
        ValueError: If ``data``, ``error``, and ``weight`` do not share
            a matching one-dimensional shape.
    """

    data = np.asarray(data, dtype='float64')
    error = np.asarray(error, dtype='float64')
    weight = np.asarray(weight, dtype='float64')
    if data.shape != error.shape or data.shape != weight.shape:
        raise ValueError('data, error, and weight must have matching shapes')
    if data.ndim != 1:
        raise ValueError('data must be one-dimensional')

    cx = np.hstack((0, np.cumsum(data)))
    wt = np.hstack((0, np.cumsum(weight)))
    vx = np.hstack((0, np.cumsum(error**2)))

    for _ in range(nrepl - 1):
        cx = np.hstack((cx, cx[1:-1] + cx[-1]))
        wt = np.hstack((wt, wt[1:-1] + wt[-1]))
        vx = np.hstack((vx, vx[1:-1] + vx[-1]))

    nmax = len(cx) - 1
    lscl_max = int(np.ceil(np.log2(nmax)))

    if bfac <= 0:
        bfac = 1.0
    if osamp < bfac:
        osamp = bfac

    if bfac < osamp:
        scl_out = 2 ** (np.arange(lscl_max, dtype='int32'))
        scl2 = 2.0 * np.round(2 ** (np.arange(lscl_max * bfac, dtype='float64') / bfac) / 2.0)
        scl_out = np.hstack((scl_out, scl2)).astype('int32')
        scl_out.sort()
        scl_out = np.unique(scl_out)
        scl_out = scl_out[(scl_out > 0) * (2 * scl_out <= nmax)]

    scales = 2 ** (np.arange(lscl_max, dtype='int32'))
    scl2 = 2.0 * np.round(2 ** (np.arange(lscl_max * osamp, dtype='float64') / osamp) / 2.0)
    scales = np.hstack((scales, scl2)).astype('int32')
    scales.sort()
    scales = np.unique(scales)
    scales = scales[(scales > 0) * (2 * scales <= nmax)]
    if bfac >= osamp:
        scl_out = 1 * scales

    nscales = len(scales)
    pspec = np.zeros(nscales, dtype='float64')
    pspec0 = np.zeros(nscales, dtype='float64')
    vpspec = np.zeros(nscales, dtype='float64')

    with np.errstate(divide='ignore', invalid='ignore'):
        for k in range(nscales):
            scl = scales[k]
            scl2 = scl**2
            cfac = nmax / (nmax - 2.0 * scl + 1)

            wav2 = (
                cx[2 * scl : nmax + 1] - 2 * cx[scl : nmax - scl + 1] + cx[: nmax - 2 * scl + 1]
            ) ** 2
            vwav = vx[2 * scl : nmax + 1] - vx[: nmax - 2 * scl + 1]

            wt0 = (wt[2 * scl : nmax + 1] - wt[: nmax - 2 * scl + 1]) / (2.0 * scl)
            wts = wt0.mean()

            pspec[k] = (wav2 * wt0).sum() * cfac / scl2 / wts
            pspec0[k] = (vwav * wt0).sum() * cfac / scl2 / wts
            vpspec[k] = (
                (((vwav * wt0) ** 2).sum() / scl2 / wts**2 + 0.5 * pspec0[k]) * cfac**2 / scl
            )

    scl1, scl2 = scl_out[:-1], scl_out[1:]
    nscales1 = len(scl1)
    psp = np.zeros(nscales1, dtype='float64')
    psp0 = np.zeros(nscales1, dtype='float64')
    dpsp = np.zeros(nscales1, dtype='float64')

    with np.errstate(invalid='ignore'):
        for i in range(nscales1):
            h = (scales >= scl1[i]) * (scales < scl2[i])
            nh = h.sum()
            if nh > 0:
                scl1[i], scl2[i] = scales[h].min(), scales[h].max()
                psp[i] = pspec[h].sum() / nh
                psp0[i] = pspec0[h].sum() / nh
                dpsp[i] = np.sqrt(vpspec[h].sum() * (nrepl + 1.0) / nh)

    return dt * scl1, dt * scl2, psp, psp0, dpsp


def drop_local_noise_floor_spikes(tau, noise_power, factor=100.0, half_window=2):
    """Mask finite scaleogram bins whose noise floor is a sharp local spike."""

    tau = np.asarray(tau, dtype=float)
    noise_power = np.asarray(noise_power, dtype=float)
    keep = np.isfinite(tau) & np.isfinite(noise_power) & (noise_power > 0)
    idx = np.where(keep)[0]
    if idx.size < 2 * half_window + 1:
        return keep

    log_noise = np.log(noise_power[idx])
    for pos, src_idx in enumerate(idx):
        lo = max(0, pos - half_window)
        hi = min(idx.size, pos + half_window + 1)
        neigh = np.r_[log_noise[lo:pos], log_noise[pos + 1 : hi]]
        if neigh.size >= half_window and log_noise[pos] - np.median(neigh) > np.log(factor):
            keep[src_idx] = False

    return keep


def calculate_haar_mvt(
    rate,
    rate_err,
    dt,
    *,
    tau_bg_max=0.01,
    nrepl=2,
    bin_fac=4,
    afactor=1.0,
    snr=3.0,
    verbose=False,
    weight=True,
    drop_nonfinite=True,
    file='mvt',
):
    """Compute the Haar minimum variability timescale (MVT) of a light curve.

    Port of ``nrbutler/mvt`` ``haar_power_mod.py``: builds the
    signal-weighted Haar power spectrum via
    :func:`calculate_haar_power_spectrum`, estimates and subtracts the
    Poisson/Gaussian noise floor (``pspec0``, rescaled by ``afactor`` or,
    when ``afactor < 0``, by a data-driven factor measured below
    ``tau_bg_max``), locates the shortest significant timescale (``snr``
    sigma above the noise floor), then fits a broken power law (flat
    noise branch + rising signal branch, slope free) to the log-log
    structure function to locate the break timescale ``tmin`` -- the
    MVT. When too few scales are significant, returns a ``snr``-sigma
    upper limit instead of a measurement.

    Args:
        rate, rate_err: Uniformly sampled, background-subtracted light
            curve and its 1-sigma errors.
        dt: Bin width in seconds.
        tau_bg_max: Largest timescale used to estimate the zero level
            when ``afactor < 0``.
        nrepl, bin_fac, afactor, snr, weight: Parameters preserved from
            ``nrbutler/mvt``; see the upstream ``haar_power_mod``
            docstring for their exact roles.
        verbose: Print a one-line summary (``a factor``, or
            ``T_snr``/``T_beta``/``T_min``) as the original code does.
        drop_nonfinite: When ``True``, drop scales where
            ``power``/``noise_power``/``power_err`` are non-finite before
            searching for the break, and also drop finite bins whose
            noise floor is a sharp local numerical spike (on by default
            for robust real-data analysis). Pass ``False`` to reproduce
            upstream's unfiltered default behavior exactly.
        file: Label used in the ``verbose`` print statements.

    Returns:
        A 5-tuple ``(mvt, mvt_err_lo, mvt_err_hi, is_upper_limit, diag)``:
        the MVT in seconds, its lower/upper 1-sigma errors (both ``0``
        when ``is_upper_limit`` is ``True``), whether the result is a
        ``snr``-sigma upper limit rather than a measurement, and a dict
        of diagnostic arrays/scalars (the full scaleogram, the fitted
        break parameters, and the input settings) suitable for plotting
        or JSON serialisation.

    Raises:
        ValueError: If ``rate``/``rate_err`` are not one-dimensional,
            differ in shape, have fewer than four bins, or if ``dt`` is
            not a positive finite scalar.
    """

    rate = np.asarray(rate, dtype='float64')
    rate_err = np.asarray(rate_err, dtype='float64')
    if rate.ndim != 1 or rate_err.ndim != 1:
        raise ValueError('rate and rate_err must be one-dimensional')
    if rate.shape != rate_err.shape:
        raise ValueError('rate and rate_err must have matching shapes')
    if rate.size < 4:
        raise ValueError('at least four bins are required')
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be a positive finite scalar')

    if weight:
        wt = haar_denoise(rate, rate_err)
        wt = haar_denoise(wt, rate_err).clip(0.0)
    else:
        wt = np.ones(len(rate), dtype='float64')

    dta, dta1, pspec, pspec0, dpspec = calculate_haar_power_spectrum(
        rate,
        rate_err,
        wt,
        dt=dt,
        nrepl=nrepl,
        bfac=bin_fac,
        osamp=bin_fac * 8,
    )

    tau_all = 0.5 * (dta + dta1)
    g = pspec0 > 0
    first_dropped_tau = np.nan
    first_noise_spike_tau = np.nan
    if drop_nonfinite:
        finite_mask = np.isfinite(pspec) & np.isfinite(pspec0) & np.isfinite(dpspec)
        spike_keep = drop_local_noise_floor_spikes(tau_all, pspec0)
        dropped = g & ~(finite_mask & spike_keep)
        noise_spikes = g & finite_mask & ~spike_keep
        if dropped.any():
            first_dropped_tau = float(tau_all[np.flatnonzero(dropped)[0]])
        if noise_spikes.any():
            first_noise_spike_tau = float(tau_all[np.flatnonzero(noise_spikes)[0]])
        g &= finite_mask
        g &= spike_keep
    dta = dta[g]
    dta1 = dta1[g]
    pspec = pspec[g]
    pspec0 = pspec0[g]
    dpspec = dpspec[g]
    tau = 0.5 * (dta + dta1)
    if tau.size == 0:
        return (
            dt,
            0.0,
            0.0,
            True,
            {'mode': 'nrbutler2025', 'reason': 'no-positive-noise-power'},
        )

    with np.errstate(divide='ignore', invalid='ignore'):
        tmax = tau[(pspec / pspec0).argmax()]

    tsnr, tbeta, tmin, dtmin, slope, sigma_tsnr, sigma_tmin = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    tmin_fit, mu0, ib1 = 0.0, 0.0, 0
    otype = 'limit'

    g = tau < tau_bg_max
    if g.sum() < 2 or afactor > 0:
        afactor = abs(afactor)
        pspec0 *= afactor
        dpspec *= afactor
    else:
        a = np.median(pspec[g] / pspec0[g])
        pspec0 *= a
        dpspec *= a
        if verbose:
            print(f' {file} a factor: {a:f}')

    pspec_raw = pspec.copy()
    pspec0_raw = pspec0.copy()
    dpspec_raw = dpspec.copy()
    with np.errstate(invalid='ignore'):
        pspec = pspec - pspec0

    g = pspec < snr * dpspec
    g2 = ~g
    g *= tau < tmax
    wi1 = np.where(g)[0]
    i1 = wi1[-1] if len(wi1) > 0 else 0
    g[:i1] = True
    g2[:i1] = False
    k = 0
    while i1 > 0 and pspec[i1] > dpspec[i1] and k < bin_fac:
        g[i1] = False
        g2[i1] = True
        i1 -= 1
        k += 1

    reason = None
    if g2.sum() < 2:
        reason = 'not-enough-significant-data'
        if verbose:
            print(f'{file} Not enough significant data!')
    else:
        pspm = pspec[g2].max()
        pspec /= pspm
        pspec0 /= pspm
        dpspec /= pspm

        tsnr = tau[i1 + 1].max()

        y = pspec[g2] - pspec0[g2]
        h = np.where(y > 0)[0]
        fix_beta = False
        if len(h) > 2:
            ib1 = h[0]
            ib0 = ib1 - 1
            if ib0 >= 0:
                tbeta = tau[g2][ib0]
                if y[ib1] != y[ib0]:
                    tbeta -= y[ib0] * (tau[g2][ib1] - tau[g2][ib0]) / (y[ib1] - y[ib0])
            else:
                fix_beta = True
        else:
            tbeta = tau.max()

        xx = np.log(tau[g2])
        yy = np.log(pspec[g2]) - 2.0 * xx
        dyy = dpspec[g2] / pspec[g2]
        vmu = 1.0 / (1.0 / dyy**2).cumsum()
        mu = (yy / dyy**2).cumsum() * vmu

        thresh = (0.5 * np.sqrt(dyy[1:] ** 2 + bin_fac * vmu[:-1])).clip(0.1 * np.log(2.0))
        wib1 = np.where(mu[:-1] - yy[1:] > thresh)[0]
        ib1 = len(mu) - 2
        if len(wib1) > 0:
            ib1 = wib1[0]
        ib0 = ib1 - 1
        if ib0 < 0:
            ib0 = 0
        mu0 = (mu[ib0] + mu[ib1]) / 2.0

        x = xx[: ib1 + bin_fac + 1]
        y = yy[: ib1 + bin_fac + 1]
        dy = dyy[: ib1 + bin_fac + 1]
        dy2 = dy * dy

        b1 = (y / dy2).sum()
        m11 = (1.0 / dy2).sum()
        par = np.array([mu0, 0.5, 0.0])

        def break_fun(xb):
            if np.isnan(xb):
                xb = x.min()
            xl = x < xb
            xu = ~xl
            m12 = ((x[xu] - xb) / dy2[xu]).sum()
            m22 = (((x[xu] - xb) ** 2) / dy2[xu]).sum()
            det2 = m11 * m22 - m12**2
            b2 = (y[xu] * (x[xu] - xb) / dy2[xu]).sum()

            if det2 > 0:
                par[0] = (m22 * b1 - m12 * b2) / det2
                par[1] = (m11 * b2 - m12 * b1) / det2

            if xl.sum() > 0:
                chi0 = ((y[xl] - par[0]) ** 2 / dy2[xl]).sum()
                m01 = -par[1] * (1.0 / dy2[xu]).sum()
                m00 = -par[1] * m01
                m02 = ((y[xu] - par[0]) / dy2[xu]).sum() - 2.0 * par[1] * m12
                par[2] = 1.0 / np.sqrt(
                    m00 + (m01 * (m02 * m12 - m01 * m22) + m02 * (m01 * m12 - m02 * m11)) / det2
                )
            else:
                chi0 = 0.0

            return chi0 + ((y[xu] - par[1] * (x[xu] - xb) - par[0]) ** 2 / dy2[xu]).sum()

        res = minimize_scalar(break_fun, bounds=(x.min(), x.max() - 1.0e-5), method='bounded')
        c0, xmin = res['fun'], res['x']
        dxmin = 1.0 * par[2]
        if np.isnan(dxmin):
            dxmin = xmin

        mu0, slope = par[0], 1.0 + 0.5 * par[1]
        sigma_tsnr, sigma_tmin = np.exp(0.5 * mu0 + x[0]), np.exp(0.5 * mu0 + xmin)

        if xmin >= x[1] and xmin - dxmin > x[0]:
            otype = 'measurement'
        else:
            for _ in range(3):
                c1 = break_fun(xmin + np.log(1.0 + dxmin))
                dxmin = max(0.5 * dxmin, min(dxmin / (1.0e-5 + abs(c1 - c0)), 1.5 * dxmin))
            sigma_tmin *= np.exp(slope * np.log(1.0 + snr * dxmin * np.sqrt(bin_fac)))

        tmin = np.exp(xmin)
        dtmin = tmin * dxmin * np.sqrt(bin_fac)
        tmin_fit = float(tmin)

        if fix_beta:
            y = 2 * np.log(tau) + mu0 - np.log(pspec0)
            h = np.where(y > 0)[0]
            if len(h) > 2:
                ib1a = h[0]
                ib0a = ib1a - 1
                tbeta = tau[ib0a]
                if y[ib1a] != y[ib0a]:
                    tbeta -= y[ib0a] * (tau[ib1a] - tau[ib0a]) / (y[ib1a] - y[ib0a])

        if verbose:
            print(f' {file} T_snr={tsnr:f} T_beta={tbeta:f} T_min={tmin:f} +/- {dtmin:f}')

        if otype == 'limit':
            tmin += snr * dtmin
            dtmin = 0.0

    diag = {
        'mode': 'nrbutler2025',
        'tau': tau.tolist(),
        'dta': dta.tolist(),
        'dta1': dta1.tolist(),
        'pspec_raw': pspec_raw.tolist(),
        'pspec0_raw': pspec0_raw.tolist(),
        'dpspec_raw': dpspec_raw.tolist(),
        'pspec': pspec.tolist(),
        'pspec0': pspec0.tolist(),
        'dpspec': dpspec.tolist(),
        'noise_mask': g.tolist(),
        'signal_mask': g2.tolist(),
        'tsnr': float(tsnr),
        'tbeta': float(tbeta),
        'tmin': float(tmin),
        'dtmin': float(dtmin),
        'tmin_fit': tmin_fit,
        'mu0': float(mu0),
        'ib1': int(ib1),
        'slope': float(slope),
        'sigma_tsnr': float(sigma_tsnr),
        'sigma_tmin': float(sigma_tmin),
        'otype': otype,
        'tau_bg_max': float(tau_bg_max),
        'nrepl': int(nrepl),
        'bin_fac': int(bin_fac),
        'afactor': float(afactor),
        'snr': float(snr),
        'weight': bool(weight),
        'drop_nonfinite': bool(drop_nonfinite),
        'first_dropped_tau': float(first_dropped_tau),
        'first_noise_spike_tau': float(first_noise_spike_tau),
    }
    if reason is not None:
        diag['reason'] = reason

    return float(tmin), float(dtmin), float(dtmin), (otype == 'limit'), diag


def set_diagnostic_axis(ax):
    """Apply the shared tick/spine styling used by the temp-module plotter classes.

    Args:
        ax: Matplotlib Axes to style in place.
    """

    ax.minorticks_on()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
    ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
    ax.tick_params(which='major', width=1.0, length=5)
    ax.tick_params(which='minor', width=1.0, length=3)


def plot_haar_scaleogram(ax, dt, time, mvt_res, max_dt='auto'):
    """Reproduce ``nrbutler/mvt``'s ``haar_power_mod`` scaleogram plot on ``ax``.

    Ports the ``doplot`` block of the upstream ``haar_power_mod`` function
    (see the reference ``test_haar_mod.png``) onto a caller-supplied Axes:
    the sqrt-transformed flux-variation points with error bars, the fitted
    noise-floor branch (flat) and signal branch (broken power law), the
    break-point marker, a family of dotted power-law guide lines through
    the peak point, and the below-threshold points drawn as their
    ``snr``-sigma upper limits. Operates on copies of the diagnostic
    arrays, so ``mvt_res`` is left untouched (safe to call more than
    once, e.g. before JSON serialisation).

    Args:
        ax: Matplotlib Axes to draw on.
        dt: Bin width in seconds (upstream's ``min_dt``).
        time: Bin-center times of the analysed light curve; used when
            ``max_dt`` is ``'auto'`` and no dropped scale is available,
            or when ``max_dt`` is explicitly ``None``.
        mvt_res: A result dict as returned by
            :meth:`~heapy.temp.mvt.MVT.calculate`.
        max_dt: Optional scaleogram plotting limit. ``'auto'`` uses the
            first scale dropped by the robust non-finite/noise-spike
            filter when available, otherwise the analysed time span.
            Pass a number (for example upstream's ``100.0``) for an
            explicit fixed plotting range, or ``None`` to infer from the
            analysed time span.
    """

    diag = mvt_res['diag']
    is_upper_limit = mvt_res['is_upper_limit']
    mvt_val = mvt_res['mvt']
    dtmin_val = mvt_res['mvt_err_lo']

    tau = np.asarray(diag['tau'], dtype=float)
    dta = np.asarray(diag['dta'], dtype=float)
    dta1 = np.asarray(diag['dta1'], dtype=float)
    pspec = np.asarray(diag['pspec'], dtype=float).copy()
    pspec0 = np.asarray(diag['pspec0'], dtype=float).copy()
    dpspec = np.asarray(diag['dpspec'], dtype=float).copy()
    g = np.asarray(diag['noise_mask'], dtype=bool)
    g2 = np.asarray(diag['signal_mask'], dtype=bool)
    snr = diag['snr']

    ax.set_xlabel(r'$\Delta t$ [s]')
    ax.set_ylabel(r'Flux Variation $\sigma_{X,\Delta t}$')

    min_dt = dt
    if max_dt == 'auto':
        first_dropped_tau = float(diag.get('first_dropped_tau', np.nan))
        if np.isfinite(first_dropped_tau) and first_dropped_tau > min_dt:
            max_dt = first_dropped_tau
        else:
            max_dt = float(time[-1] - time[0]) if len(time) > 1 else dt
    elif max_dt is None:
        max_dt = float(time[-1] - time[0]) if len(time) > 1 else dt
    else:
        max_dt = float(max_dt)

    in_window = tau <= max_dt
    g_plot = g & in_window
    g2_plot = g2 & in_window

    if 'ib1' in diag and g2_plot.sum() >= 2:
        mu0 = diag['mu0']
        slope = diag['slope']
        tmin_fit = diag['tmin_fit']

        pspec[g2_plot] = np.sqrt(np.clip(pspec[g2_plot], 0, None))
        dpspec[g2_plot] /= 2.0 * np.where(pspec[g2_plot] > 0, pspec[g2_plot], np.nan)
        pspec0 = np.sqrt(np.clip(pspec0, 0, None))

        xx1 = np.array([min_dt / 2, tmin_fit])
        xx2 = np.array([tmin_fit, max_dt * 2])
        ax.plot(xx1, xx1 * np.exp(mu0 / 2.0), 'r-', alpha=0.5)
        ax.plot(
            xx2,
            np.exp(0.5 * mu0 - (slope - 1) * np.log(tmin_fit) + slope * np.log(xx2)),
            'r-',
            alpha=0.5,
        )
        ax.errorbar(
            tau[g2_plot],
            pspec[g2_plot],
            yerr=dpspec[g2_plot],
            xerr=0.5 * (dta1 - dta)[g2_plot],
            fmt='bo',
            capsize=0,
            linestyle='None',
            markersize=3,
        )
        if not is_upper_limit and tmin_fit <= max_dt:
            ax.plot(
                tmin_fit,
                tmin_fit * np.exp(mu0 / 2.0),
                marker='o',
                ms=7,
                mfc='none',
                mec='m',
                mew=2.5,
                linestyle='None',
            )

        pspec_g2 = pspec[g2_plot]
        finite_g2 = np.isfinite(pspec_g2) & (pspec_g2 > 0)
        if finite_g2.any():
            i0 = np.flatnonzero(finite_g2)[np.argmax(pspec_g2[finite_g2])]
            x1, y1 = tau[g2_plot][i0], pspec_g2[i0]
            xx = np.array([min_dt / 2, max_dt * 2])
            for i in range(-12, 13, 2):
                ax.plot(xx, y1 * xx / x1 * 2.0**i, 'k--', alpha=0.4, lw=0.8)

            ax.set_xlim(tau[g2_plot][finite_g2].min() / 4.0, tau[g2_plot][finite_g2].max() * 1.5)
            ax.set_ylim(pspec_g2[finite_g2].min() / 2.0, pspec_g2[finite_g2].max() * 1.5)

        if g_plot.sum() > 0:
            ax.plot(
                tau[g_plot], np.sqrt(np.clip(pspec[g_plot], 0, None) + snr * dpspec[g_plot]), 'bv'
            )
    else:
        ok = (pspec > 0) & in_window
        if ok.any():
            ax.errorbar(
                tau[ok],
                np.sqrt(pspec[ok]),
                yerr=0.5 * dpspec[ok] / np.sqrt(pspec[ok]),
                fmt='bo',
                capsize=0,
                linestyle='None',
                markersize=3,
            )

    ax.set_xscale('log')
    ax.set_yscale('log')
    legend_label = (
        r'$\Delta t_{\rm min}<$' + rf'{mvt_val:.4f} s'
        if is_upper_limit
        else r'$\Delta t_{\rm min}=$' + rf'{mvt_val:.4f} $\pm$ {dtmin_val:.4f} s'
    )
    ax.legend(
        [Line2D([], [], linestyle='None')],
        [legend_label],
        loc='upper left',
        frameon=True,
        handlelength=0,
        handletextpad=0,
    )


class TxxPlotter:
    """Composable two-panel diagnostic figure for Txx duration analysis.

    Top panel (:attr:`ax_top`): primary light-curve trace with optional
    background overlay and vertical Txx start/stop markers. Bottom
    panel (:attr:`ax_bot`): cumulative net counts -- the full series in
    gray and the analysis-window slice in black -- with horizontal CSF
    level lines and the same Txx vertical markers overlaid. Compose by
    calling :meth:`plot_curve`, :meth:`plot_ccts`, and :meth:`plot_txx`
    in any order, then :meth:`save` or :meth:`show`. Mirrors the
    :class:`~heapy.auto.signal_utils.SignalPlotter` API style.

    Attributes:
        fig: Underlying matplotlib Figure.
        ax_top: Top-panel Axes (light curve + background).
        ax_bot: Bottom-panel Axes (cumulative counts).
    """

    def __init__(self, figsize=(7, 6)):
        """Create an empty two-panel figure with shared x-axis.

        Args:
            figsize: Width and height of the figure in inches.
        """

        self.fig = plt.figure(figsize=figsize)
        gs = self.fig.add_gridspec(2, 1, wspace=0, hspace=0)
        self.ax_top = self.fig.add_subplot(gs[:1, 0])
        self.ax_bot = self.fig.add_subplot(gs[1:, 0], sharex=self.ax_top)
        set_diagnostic_axis(self.ax_top)
        set_diagnostic_axis(self.ax_bot)
        plt.setp(self.ax_top.get_xticklabels(), visible=False)
        self.ax_top.set_ylabel('Rate (cts/s)')
        self.ax_bot.set_xlabel('Time (s)')
        self.ax_bot.set_ylabel('Accumulated counts')

        self._gaps = None
        self._bin_lbins = None
        self._bin_rbins = None

    def set_gaps(self, gap_int, lbins, rbins):
        """Register missing-data intervals to mask in :meth:`plot_ccts`.

        Cumulative counts at gap-bin positions are replaced with
        ``NaN`` so the curve renders as a break rather than a spurious
        plateau (gap-bin ``ncts`` is treated as zero contribution by
        ``np.nancumsum`` in the caller, leaving the running total
        finite at those positions until this mask is applied).

        Args:
            gap_int: List of ``[low, high]`` intervals; empty or
                falsy disables masking.
            lbins: Per-bin left edges aligned with the ``ccts`` array
                later passed to :meth:`plot_ccts`.
            rbins: Per-bin right edges (same length as ``lbins``).
        """

        self._gaps = gap_int
        self._bin_lbins = lbins
        self._bin_rbins = rbins

    def plot_curve(self, time, primary, bak=None):
        """Draw the primary light curve and optional background on the top panel.

        Args:
            time: Per-bin time grid.
            primary: Primary rate (or net rate) array plotted in black
                as the light curve.
            bak: Optional background rate plotted in red. ``None``
                skips the background line (used by gg-flavoured Txx
                where the input is already net).
        """

        self.ax_top.plot(time, primary, color='k', lw=1.0, label='Light Curve')
        if bak is not None:
            self.ax_top.plot(time, bak, color='r', lw=1.0, label='Background')
        self.ax_top.set_xlim([time[0], time[-1]])
        self.ax_top.legend(frameon=False)

    def plot_ccts(self, time, ccts, tindex):
        """Draw cumulative counts on the bottom panel.

        Full ``ccts`` is rendered in gray; the analysis-window slice
        ``ccts[tindex]`` is overlaid in black. When :meth:`set_gaps`
        has been called, gap-bin positions are masked to ``NaN`` so
        matplotlib breaks the line there.

        Args:
            time: Per-bin time grid aligned with ``ccts``.
            ccts: Cumulative net count array.
            tindex: Integer index array selecting the analysis window.
        """

        ccts = np.asarray(ccts, dtype=float).copy()
        if self._gaps:
            idx = indices_in_intervals(self._bin_lbins, self._bin_rbins, self._gaps)
            ccts[idx] = np.nan
        self.ax_bot.plot(time, ccts, color='gray', lw=1.0)
        self.ax_bot.plot(time[tindex], ccts[tindex], color='k', lw=1.0)

    def plot_txx(self, txx1, txx2, csf, csf1, csf2):
        """Overlay Txx vertical markers and CSF horizontal lines.

        Args:
            txx1: Pulse start times (one per pulse); drawn as dashed
                green vertical lines on both panels.
            txx2: Pulse stop times (one per pulse); drawn the same way.
            csf: Background CSF levels (one per quiescent segment);
                drawn as solid orange horizontal lines on the bottom
                panel.
            csf1: Lower CSF thresholds (one per pulse); drawn as dashed
                orange horizontal lines on the bottom panel.
            csf2: Upper CSF thresholds; drawn the same way as ``csf1``.
        """

        for t1, t2 in zip(txx1, txx2, strict=False):
            self.ax_top.axvline(t1, color='g', lw=1.0, ls='--')
            self.ax_top.axvline(t2, color='g', lw=1.0, ls='--')
            self.ax_bot.axvline(t1, color='g', lw=1.0, ls='--')
            self.ax_bot.axvline(t2, color='g', lw=1.0, ls='--')
        for c in csf:
            self.ax_bot.axhline(c, color='orange', lw=1.0)
        for c1, c2 in zip(csf1, csf2, strict=False):
            self.ax_bot.axhline(c1, color='orange', lw=1.0, ls='--')
            self.ax_bot.axhline(c2, color='orange', lw=1.0, ls='--')

    def show(self):
        """Display the figure interactively."""

        plt.tight_layout()
        plt.show()

    def save(self, filename, dpi=300):
        """Save the figure to ``filename`` and close it.

        Args:
            filename: Output file path; format inferred from extension.
            dpi: Resolution in dots per inch.
        """

        self.fig.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=dpi)
        plt.close(self.fig)


class LagPlotter:
    """Composable two-panel diagnostic figure for Lag cross-correlation analysis.

    Top panel (:attr:`ax_top`): the ``x`` and ``y`` net-count light
    curves overlaid. Bottom panel (:attr:`ax_bot`): the CCF as a
    function of time delay, restricted to the fit/search window, with
    the fitted (interpolated) peak profile overlaid when available.
    Compose by calling :meth:`plot_curves` and :meth:`plot_ccf` in any
    order, then :meth:`save` or :meth:`show`. Mirrors the
    :class:`TxxPlotter` API style.

    Attributes:
        fig: Underlying matplotlib Figure.
        ax_top: Top-panel Axes (x/y light curves).
        ax_bot: Bottom-panel Axes (CCF vs. time delay).
    """

    def __init__(self, figsize=(6, 8)):
        """Create an empty two-panel figure.

        Args:
            figsize: Width and height of the figure in inches.
        """

        self.fig = plt.figure(figsize=figsize)
        gs = self.fig.add_gridspec(5, 1, hspace=0.5)
        self.ax_top = self.fig.add_subplot(gs[0:2, 0])
        self.ax_bot = self.fig.add_subplot(gs[2:5, 0])
        set_diagnostic_axis(self.ax_top)
        set_diagnostic_axis(self.ax_bot)
        self.ax_top.set_xlabel('Time (s)')
        self.ax_top.set_ylabel('Counts')
        self.ax_bot.set_xlabel('Time delay (s)')
        self.ax_bot.set_ylabel('CCF value')

    def plot_curves(self, time, xncts, yncts):
        """Draw the ``x`` and ``y`` light curves on the top panel.

        Args:
            time: Per-bin time grid, shared by ``xncts`` and ``yncts``.
            xncts: Reference (high-energy) channel net counts.
            yncts: Comparison (low-energy) channel net counts.
        """

        self.ax_top.plot(time, xncts, color='k', lw=1.0, label='x')
        self.ax_top.plot(time, yncts, color='r', lw=1.0, label='y')
        self.ax_top.set_xlim([time[0], time[-1]])
        self.ax_top.legend(frameon=False)

    def plot_ccf(self, taus, ccf, nidx, itp_taus=None, itp_ccfs=None):
        """Draw the CCF and its fitted peak profile on the bottom panel.

        Args:
            taus: Full time-delay grid.
            ccf: CCF values aligned with ``taus`` (the observed, i.e.
                unperturbed, realisation).
            nidx: Integer index array selecting the fit/search window
                plotted as ``+`` markers.
            itp_taus: Interpolated time-delay grid for the fitted peak
                profile overlay, or ``None`` to skip it (e.g. the
                ``'argmax'`` method has no continuous fit).
            itp_ccfs: Fitted CCF values aligned with ``itp_taus``.
        """

        self.ax_bot.scatter(
            taus[nidx], ccf[nidx], marker='+', color='k', s=10, linewidths=0.5, alpha=1.0
        )
        if itp_taus is not None:
            self.ax_bot.plot(itp_taus, itp_ccfs, c='r', lw=0.5, alpha=1.0)

    def show(self):
        """Display the figure interactively."""

        plt.tight_layout()
        plt.show()

    def save(self, filename, dpi=300):
        """Save the figure to ``filename`` and close it.

        Args:
            filename: Output file path; format inferred from extension.
            dpi: Resolution in dots per inch.
        """

        self.fig.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=dpi)
        plt.close(self.fig)


class MvtPlotter:
    """Composable two-panel diagnostic figure for Haar MVT analysis.

    Top panel (:attr:`ax_top`): the light curve with the fitted MVT
    marked. Bottom panel (:attr:`ax_bot`): the Haar scaleogram,
    reproducing ``nrbutler/mvt``'s ``haar_power_mod`` ``doplot`` figure
    (see :func:`_plot_haar_scaleogram`). Compose by calling
    :meth:`plot_curve` and :meth:`plot_scaleogram` in any order, then
    :meth:`save` or :meth:`show`. Mirrors the :class:`TxxPlotter` API
    style.

    Attributes:
        fig: Underlying matplotlib Figure.
        ax_top: Top-panel Axes (light curve).
        ax_bot: Bottom-panel Axes (Haar scaleogram).
    """

    def __init__(self, figsize=(6, 8)):
        """Create an empty two-panel figure.

        Args:
            figsize: Width and height of the figure in inches.
        """

        self.fig = plt.figure(figsize=figsize)
        gs = self.fig.add_gridspec(5, 1, hspace=0.5)
        self.ax_top = self.fig.add_subplot(gs[0:2, 0])
        self.ax_bot = self.fig.add_subplot(gs[2:5, 0])
        set_diagnostic_axis(self.ax_top)
        set_diagnostic_axis(self.ax_bot)
        self.ax_top.set_xlabel('Time (s)')
        self.ax_top.set_ylabel('Rate (cts/s)')

    def plot_curve(self, time, rate):
        """Draw the light curve on the top panel.

        Args:
            time: Per-bin time grid.
            rate: Background-subtracted count rate aligned with ``time``.
        """

        self.ax_top.plot(time, rate, color='k', lw=1.0)
        self.ax_top.set_xlim([time[0], time[-1]])

    def plot_scaleogram(self, dt, time, mvt_res, max_dt='auto'):
        """Draw the Haar scaleogram on the bottom panel; see :func:`plot_haar_scaleogram`.

        Args:
            dt: Bin width in seconds.
            time: Bin-center times of the analysed light curve.
            mvt_res: A result dict as returned by
                :meth:`~heapy.temp.mvt.MVT.calculate`.
            max_dt: Optional scaleogram plotting limit; defaults to
                ``'auto'``. Pass ``100.0`` for upstream's fixed
                demonstration extent.
        """

        plot_haar_scaleogram(self.ax_bot, dt, time, mvt_res, max_dt=max_dt)

    def show(self):
        """Display the figure interactively."""

        plt.tight_layout()
        plt.show()

    def save(self, filename, dpi=300):
        """Save the figure to ``filename`` and close it.

        Args:
            filename: Output file path; format inferred from extension.
            dpi: Resolution in dots per inch.
        """

        self.fig.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=dpi)
        plt.close(self.fig)
