"""Minimum variability timescale (MVT) for GRB light curves.

Implements three independent algorithms accessible from three data-type
subclasses that mirror :mod:`heapy.temp.txx`:

* :class:`pgMVT(pgSignal) <heapy.auto.signal.pgSignal>` -- Poisson source plus
  Gaussian-fit background (Swift/BAT style).
* :class:`ppMVT(ppSignal) <heapy.auto.signal.ppSignal>` -- Poisson source plus
  Poisson background sample (Fermi/GBM style).
* :class:`ggMVT(ggSignal) <heapy.auto.signal.ggSignal>` -- pre-subtracted
  Gaussian net rates.

Each subclass exposes ``.calculate(method=...)`` selecting one of:

* ``method='cwt'``  -- Vianello et al. 2018 (Continuous Wavelet Transform with
  Mexican Hat + Liu 2007 rectification + 99 % Monte Carlo confidence band).
  ``ggMVT`` uses a Gaussian background MC; ``pgMVT``/``ppMVT`` use a Poisson
  MC at the measured off-pulse rate.
* ``method='haar'`` -- Golkhou & Butler 2014 (non-decimated Haar wavelet
  structure function on log-rate; t_min from departure of sigma_X,Dt from the
  sigma_X,Dt proportional to Dt smooth-signal expectation at the 2 sigma
  Dchi^2 level).  Distribution-agnostic: relies only on ``ncts_err``.
* ``method='mepsa'`` -- Camisasca et al. 2023 / Maccary et al. 2025 (multi-bin
  MEPSA peak detection; FWHM_min from direct half-max measurement on the
  rebinned light curve at the optimal detection scale).

The three outputs are **not** mutually interchangeable: ``cwt`` and ``haar``
return values close to the rise time of the narrowest pulse, while ``mepsa``
returns its FWHM (~2x larger).

Example::

    mvt = pgMVT(ts_evt, bins, ignore=[-2, 20])
    mvt.calculate(method='cwt')
    mvt.save('/output/mvt_cwt')
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..auto.signal import ggSignal, pgSignal, ppSignal
from ..util.tools import json_dump, plt_rc_context

VALID_METHODS = ("cwt", "haar", "mepsa")


@dataclass
class _MVTResult:
    method: str
    mvt: float
    mvt_err_lo: float
    mvt_err_hi: float
    is_upper_limit: bool
    diag: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "method": self.method,
            "mvt": self.mvt,
            "mvt_err_lo": self.mvt_err_lo,
            "mvt_err_hi": self.mvt_err_hi,
            "is_upper_limit": self.is_upper_limit,
            "diag": self.diag,
        }


# ---------- MEPSA peak detector helpers (Task 2) -----------------------------

def _rebin_mean(x, factor):
    """Mean-rebin a 1-D array by ``factor`` (drops the tail that doesn't fit)."""
    n = (x.shape[0] // factor) * factor
    if n == 0:
        return np.empty(0, dtype=x.dtype)
    return x[:n].reshape(-1, factor).mean(axis=1)


def _moving_mean(x, window):
    """Centre-excluded moving mean (same length as ``x``).

    Edges fall back to the global mean until the window is fully populated.
    The centre bin is excluded so a true spike does not bias its own local
    baseline.
    """
    n = x.shape[0]
    half = window // 2
    out = np.empty(n)
    g_mean = float(x.mean())
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo <= 1:
            out[i] = g_mean
            continue
        block = np.concatenate([x[lo:i], x[i + 1:hi]])
        out[i] = block.mean()
    return out


def _snr_threshold(dt_ms, snr_thresholds):
    """Log-linear interpolation of the per-Δt SNR threshold (Camisasca 2023 Table 3)."""
    keys = np.array(sorted(snr_thresholds.keys()), dtype=float)
    vals = np.array([snr_thresholds[int(k)] for k in keys])
    if dt_ms <= keys[0]:
        return float(vals[0])
    if dt_ms >= keys[-1]:
        return float(vals[-1])
    return float(np.interp(np.log10(dt_ms), np.log10(keys), vals))


_DEFAULT_SNR_THRESHOLDS = {1: 7.0, 4: 6.8, 64: 6.4, 1000: 6.0}


def _mepsa_scan(net, dt, *, bkg_rate=None,
                rebin_factors=(1, 4, 16, 64, 256),
                smoothing_windows=(5, 11, 21),
                snr_thresholds=None, max_rebin=256):
    """Simplified MEPSA-like multi-scale peak detector.

    For every ``(R, M)`` combination with ``R <= max_rebin``:

    1. Rebin ``net`` by ``R`` (mean over ``R`` adjacent bins).
    2. Compute the centre-excluded moving mean over a window of ``M`` bins.
       Per Camisasca 2023 condition (i), the local mean is clamped at a
       robust baseline level (25th percentile of the rebinned LC) so a
       broad pulse cannot mask itself.
    3. Compute a per-bin Poisson noise estimate
       ``sqrt((rb + bkg) / R)``.  ``bkg`` is inferred from the most
       negative values of ``net`` -- baseline-only bins typically sit at
       ``-bkg``.  This Gehrels 1986 regime estimator is robust where the
       literal centre-excluded ``movstd`` is degenerate (low-count flat
       blocks) or pulse-contaminated.
    4. Form ``SNR = (rb - mean_eff) / noise`` per bin.
    5. Accept bins whose SNR exceeds the per-``dt`` threshold (Camisasca
       2023 Table 3) AND which are local maxima of the SNR series.

    Group accepted detections across ``(R, M)`` by proximity in time
    (within one ``R · dt``).  For each group, report the detection at the
    best (highest SNR) ``(R, M)``.

    Returns a list of dicts with keys:
        ``idx``      -- bin index on the *original* grid
        ``dt_det``   -- best ``R · dt`` (seconds)
        ``snr``      -- peak SNR at the best ``(R, M)``
        ``rebin``    -- best R
        ``window``   -- best M
        ``n_adiac``  -- number of ``(R, M)`` combinations detecting the peak
    """
    if snr_thresholds is None:
        snr_thresholds = _DEFAULT_SNR_THRESHOLDS
    net = np.asarray(net, dtype=float)
    # The implied background count level subtracted to make ``net``.  When
    # supplied explicitly (preferred), use it directly.  Otherwise fall back
    # to inferring from the lower-percentile values of ``net`` -- baseline-
    # only bins of a Poisson-subtracted net typically sit at ``-bkg*dt``.
    # Used as a Poisson-noise floor below.
    if bkg_rate is not None:
        bkg_offset = float(bkg_rate) * dt
    else:
        bkg_offset = max(0.0, -float(np.percentile(net, 5)))
    candidates = []
    for R in rebin_factors:
        if R > max_rebin:
            continue
        rb = _rebin_mean(net, R)
        if rb.shape[0] < 5:
            continue
        rb_dt = dt * R
        # Camisasca 2023 condition (ii): dt_det must be at least 2x the original bin.
        if R > 1 and rb_dt < 2 * dt:
            continue
        thr = _snr_threshold(rb_dt * 1e3, snr_thresholds)
        # Robust baseline level from the off-pulse portion of ``rb``
        # (lower-quartile values).  Used as a floor for the centre-excluded
        # local mean so a broad pulse cannot mask itself (Camisasca 2023
        # condition i).
        baseline = float(np.percentile(rb, 25))
        # Per-bin Poisson noise estimate on ``rb``: rb is a mean of ``R``
        # Poisson counts per bin with mean ~= ``rb + bkg_offset``; hence
        # var(rb) = (rb + bkg_offset) / R.  This gives a noise estimate
        # that grows with the local count level (Gehrels 1986 regime),
        # making single-count statistical outliers in a near-empty
        # baseline have the *correct* low Poisson SNR rather than the
        # inflated Gaussian SNR from a degenerate centre-excluded std.
        poisson_noise = np.sqrt(
            np.maximum(rb + bkg_offset, bkg_offset) / R
        )
        for M in smoothing_windows:
            if M < 3 or M >= rb.shape[0]:
                continue
            mean = _moving_mean(rb, M)
            mean_eff = np.minimum(mean, baseline)
            # Use the per-bin Poisson noise as the noise estimate.  The
            # centre-excluded local std (the spec's literal prescription)
            # is fragile both at low counts (collapses to ~0 in flat blocks)
            # and on broad pulses (inflates by tracking the pulse shape);
            # ``poisson_noise`` is robust to both.
            std_eff = poisson_noise
            snr = (rb - mean_eff) / std_eff
            for i in range(1, snr.shape[0] - 1):
                if snr[i] < thr:
                    continue
                if snr[i] <= snr[i - 1] or snr[i] <= snr[i + 1]:
                    continue
                orig_idx = i * R + R // 2
                candidates.append({
                    "idx": int(orig_idx),
                    "dt_det": float(rb_dt),
                    "snr": float(snr[i]),
                    "rebin": int(R),
                    "window": int(M),
                })

    # Group by proximity in original-grid index space.
    candidates.sort(key=lambda c: c["idx"])
    groups = []
    for c in candidates:
        for g in groups:
            ref = g[0]
            tol_idx = max(int(round(ref["dt_det"] / dt)),
                          int(round(c["dt_det"] / dt)))
            if abs(c["idx"] - ref["idx"]) <= tol_idx:
                g.append(c)
                break
        else:
            groups.append([c])

    peaks = []
    for g in groups:
        best = max(g, key=lambda c: c["snr"])
        peaks.append({
            "idx": best["idx"],
            "dt_det": best["dt_det"],
            "snr": best["snr"],
            "rebin": best["rebin"],
            "window": best["window"],
            "n_adiac": len(g),
        })
    return peaks


def _fwhm_around(y, idx):
    """Linear-interpolated FWHM around a local maximum at ``idx``.

    Returns ``(fwhm_bins, lx, rx)`` where ``fwhm_bins`` is in bin units and
    ``lx``/``rx`` are the float-valued half-max crossing indices.  Falls back
    to ``(NaN, NaN, NaN)`` when no crossing can be found.
    """
    n = y.shape[0]
    if idx <= 0 or idx >= n - 1:
        return float("nan"), float("nan"), float("nan")
    ymax = float(y[idx])
    if ymax <= 0:
        return float("nan"), float("nan"), float("nan")
    half = ymax / 2.0
    lx = float("nan")
    for j in range(idx, 0, -1):
        if (y[j - 1] - half) * (y[j] - half) <= 0:
            denom = y[j] - y[j - 1]
            lx = (j - 1) if denom == 0 else (j - 1) + (half - y[j - 1]) / denom
            break
    rx = float("nan")
    for j in range(idx, n - 1):
        if (y[j] - half) * (y[j + 1] - half) <= 0:
            denom = y[j + 1] - y[j]
            rx = float(j) if denom == 0 else j + (half - y[j]) / denom
            break
    if np.isnan(lx) or np.isnan(rx):
        return float("nan"), lx, rx
    return rx - lx, lx, rx


# ---------- CWT helpers (Task 4) ---------------------------------------------

def _next_pow2(n):
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


def _cwt_global_spectrum(counts, dt, dj=0.125, s0=None):
    """Mexican-Hat CWT global spectrum with Liu 2007 rectification.

    Returns ``(periods_sec, rectified_global_ws)``.  Pads to the next
    power-of-two length with the input mean, then standardises the
    signal to zero mean / unit variance so the data and Monte Carlo
    spectra are directly comparable independent of input amplitude.
    Per Vianello 2018 eq. ``W(dt) = var * sum_t |W(t,dt)|^2 / N`` --
    the ``var`` prefactor is absorbed by standardising the input.
    """
    import pycwt
    n = counts.shape[0]
    n2 = _next_pow2(n)
    pad = np.full(n2 - n, counts.mean())
    x = np.concatenate([counts, pad]).astype(float)
    std = x.std()
    if std == 0:
        # Degenerate input (all-zero realisation): treat as no excess power.
        # Fall through with std=1 so we still produce a full-length spectrum
        # of zeros — needed by the Monte Carlo band stacker.
        std = 1.0
    x = (x - x.mean()) / std
    if s0 is None:
        s0 = 2 * dt
    J = max(int(np.floor(np.log2(n2 * dt / s0) / dj)), 1)
    mother = pycwt.DOG(2)  # Mexican Hat
    wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(x, dt, dj, s0, J, mother)
    power = np.abs(wave) ** 2
    global_ws = power.sum(axis=1) / n2
    rectified = global_ws / scales
    periods = scales * mother.flambda()
    return periods, rectified


def _cwt_background_band(noise_model, dt, n_bins, n_sim, sig_level, rng,
                         *, bkg_rate=None, gauss_sigma=None):
    """Monte Carlo upper-envelope of the rectified CWT spectrum.

    ``noise_model='poisson'`` requires ``bkg_rate``; ``noise_model='gaussian'``
    requires ``gauss_sigma`` (per-bin standard deviation).
    """
    spectra = []
    periods = None
    for _ in range(n_sim):
        if noise_model == "poisson":
            sim = rng.poisson(bkg_rate * dt, size=n_bins).astype(float)
        elif noise_model == "gaussian":
            sim = rng.normal(0.0, gauss_sigma, size=n_bins)
        else:
            raise ValueError(f"unknown noise_model {noise_model!r}")
        p, ws = _cwt_global_spectrum(sim, dt)
        spectra.append(ws)
        if periods is None:
            periods = p
    arr = np.array(spectra)
    upper = np.percentile(arr, sig_level, axis=0)
    median = np.percentile(arr, 50.0, axis=0)
    return periods, median, upper


# ---------- algorithm free-function stubs (filled in by later tasks) ----------

def _run_cwt(ncts, ncts_err, bins, *, bkg_rate, noise_model,
             n_sim=1000, sig_level=99.0, dj=0.125, random_seed=450001):
    """Vianello 2018 CWT minimum variability timescale.

    Args:
        ncts: net counts per bin.
        ncts_err: 1-sigma errors per bin (used for Gaussian MC std).
        bins: bin edges, length ``N+1``.
        bkg_rate: off-pulse background rate in cts/s (required for Poisson MC).
        noise_model: ``'poisson'`` for pg/pp; ``'gaussian'`` for gg.
        n_sim: Monte Carlo realisations (paper uses 10 000; default 1 000).
        sig_level: upper percentile for the noise envelope (paper uses 99.0).
    """
    ncts = np.asarray(ncts, dtype=float)
    ncts_err = np.asarray(ncts_err, dtype=float)
    bins = np.asarray(bins, dtype=float)
    dt = float(bins[1] - bins[0])
    n = ncts.shape[0]
    rng = np.random.default_rng(random_seed)

    if noise_model == "poisson":
        if bkg_rate is None:
            raise ValueError("Poisson CWT MC requires bkg_rate (cts/s)")
        # Reconstruct observed counts (= ncts + bkg) so the CWT sees the same
        # noise statistics as the simulated backgrounds.
        obs = ncts + bkg_rate * dt
        periods, ws = _cwt_global_spectrum(obs, dt, dj=dj)
        _, median, upper = _cwt_background_band(
            "poisson", dt, n, n_sim, sig_level, rng, bkg_rate=bkg_rate,
        )
    elif noise_model == "gaussian":
        # ncts are already net; the Gaussian MC matches their noise std.
        gauss_sigma = float(np.mean(ncts_err))
        periods, ws = _cwt_global_spectrum(ncts, dt, dj=dj)
        _, median, upper = _cwt_background_band(
            "gaussian", dt, n, n_sim, sig_level, rng, gauss_sigma=gauss_sigma,
        )
    else:
        raise ValueError(f"unknown noise_model {noise_model!r}")

    crossing = np.where(ws > upper)[0]
    diag = {
        "noise_model": noise_model,
        "periods": periods.tolist(),
        "ws": ws.tolist(),
        "bkg_median": median.tolist(),
        "bkg_upper": upper.tolist(),
    }
    if crossing.size == 0:
        return _MVTResult(method="cwt", mvt=float(periods[0]),
                          mvt_err_lo=0.0, mvt_err_hi=0.0,
                          is_upper_limit=True, diag=diag)
    i0 = int(crossing[0])
    err_lo = float(periods[i0] - periods[i0 - 1]) if i0 > 0 else 0.0
    err_hi = float(periods[i0 + 1] - periods[i0]) if i0 + 1 < periods.size else 0.0
    return _MVTResult(method="cwt", mvt=float(periods[i0]),
                      mvt_err_lo=err_lo, mvt_err_hi=err_hi,
                      is_upper_limit=False, diag=diag)


def _rebin_to_snr(net, err, lbins, rbins, target_snr=5.0):
    """Greedy adjacent-bin grouping until every group has SNR >= ``target_snr``.

    Returns ``(rate, drate, out_lb, out_rb)``.  ``rate`` is in cts/sec.
    """
    net = np.asarray(net, float)
    err = np.asarray(err, float)
    lb = np.asarray(lbins, float)
    rb = np.asarray(rbins, float)
    n = net.shape[0]
    rate, drate, out_lb, out_rb = [], [], [], []
    i = 0
    while i < n:
        c = net[i]
        v = err[i] ** 2
        lo = lb[i]
        hi = rb[i]
        j = i + 1
        while True:
            if v > 0 and c > 0 and (c / np.sqrt(v)) >= target_snr:
                break
            if j >= n:
                break
            c += net[j]
            v += err[j] ** 2
            hi = rb[j]
            j += 1
        exposure = hi - lo
        if exposure > 0:
            rate.append(c / exposure)
            drate.append(np.sqrt(v) / exposure)
            out_lb.append(lo)
            out_rb.append(hi)
        i = j
    return (np.array(rate), np.array(drate),
            np.array(out_lb), np.array(out_rb))


def _haar_structure_function(rate, drate, lbins, rbins,
                              n_scales=40, scale_min=None, scale_max=None):
    """Non-decimated Haar SF on log-rate (Golkhou 2014 eq 6-8).

    Implements Golkhou & Butler 2014's per-position binning by local Δt:
    every (position i, scale s) tuple contributes its own wavelet value and
    its own local Δt = (ct[i+2s] - 2·ct[i+s] + ct[i]) / s, where ct is the
    cumsum of bin-centre times.  Contributions are accumulated into
    ``n_scales`` log-spaced Δt bins so that positions ON a narrow pulse
    (fine local Δt) and positions OFF (wide local Δt) populate separate
    output scales, recovering narrow-pulse structure that the previous
    median-collapse implementation lost on adaptive-rebinned light curves.

    Returns ``(delta_t, sigma2, sigma2_noise, sigma2_err)`` or ``None`` if
    the light curve has too few positive-rate bins, or no valid (i, s)
    contributions land in any Δt bin.  Output arrays only include Δt bins
    with at least 4 contributions (sparser bins are dropped).
    """
    valid = (rate > 0) & (drate > 0)
    if valid.sum() < 16:
        return None
    rate = rate[valid]
    drate = drate[valid]
    lb = lbins[valid]
    rb = rbins[valid]
    t_centre = 0.5 * (lb + rb)
    x = np.log(rate)
    dx = drate / rate
    cx = np.concatenate([[0.0], np.cumsum(x)])
    vx = np.concatenate([[0.0], np.cumsum(dx ** 2)])
    ct = np.concatenate([[0.0], np.cumsum(t_centre)])
    n = x.shape[0]
    if scale_min is None:
        scale_min = 1
    if scale_max is None:
        scale_max = max(2, n // 4)
    scales = np.unique(np.round(np.geomspace(scale_min, scale_max, n_scales)).astype(int))
    scales = scales[(scales > 0) & (2 * scales < n)]
    if scales.size == 0:
        return None

    # Step 1: accumulate (local_dt, w², dw0²) over all (i, s).  Per
    # Golkhou & Butler 2014, every position contributes to whichever
    # local-Δt bin its (i, s) window falls into.
    local_dt_list = []
    w2_list = []
    dw02_list = []
    for s in scales:
        ii = np.arange(0, n - 2 * s + 1)
        w = (cx[ii + 2 * s] - 2 * cx[ii + s] + cx[ii]) / s
        dw0_sq = (vx[ii + 2 * s] - vx[ii]) / (s * s)
        loc = (ct[ii + 2 * s] - 2 * ct[ii + s] + ct[ii]) / s
        local_dt_list.append(loc)
        w2_list.append(w * w)
        dw02_list.append(dw0_sq)
    local_dt = np.concatenate(local_dt_list)
    w2_all = np.concatenate(w2_list)
    dw02_all = np.concatenate(dw02_list)

    # Step 2: keep only finite, positive local_dt.
    keep = np.isfinite(local_dt) & (local_dt > 0) & np.isfinite(w2_all) & np.isfinite(dw02_all)
    local_dt = local_dt[keep]
    w2_all = w2_all[keep]
    dw02_all = dw02_all[keep]
    if local_dt.size == 0:
        return None

    # Step 3: define output Δt bin edges, log-spaced.
    dt_lo = float(local_dt.min())
    dt_hi = float(local_dt.max())
    if dt_lo == dt_hi:
        # Degenerate single-Δt case: fall back to one bin.
        edges = np.array([dt_lo * (1 - 1e-9), dt_hi * (1 + 1e-9)])
    else:
        edges = np.geomspace(dt_lo, dt_hi, n_scales + 1)

    # Step 4: per-bin reduction (uniform mean of w², dw0² across contributions).
    # Tried noise-inverse weighting `1/dw0²` (spec recommendation) but on real
    # GRB data it down-weighted the burst-edge wavelet contributions and lost
    # the on-pulse signal; uniform mean reproduces the OLD code's signal
    # recovery while the per-position Δt binning still gives finer resolution.
    delta_t = []
    sigma2 = []
    sigma2_noise = []
    sigma2_err = []
    # np.digitize: indices 1..n_bins are inside [edges[i-1], edges[i]); we
    # also want the right-most point at edges[-1] to be included.
    idx = np.digitize(local_dt, edges, right=False)
    idx = np.clip(idx, 1, edges.size - 1)
    for b in range(1, edges.size):
        sel = idx == b
        nh = int(sel.sum())
        if nh < 4:
            continue
        mean_w2 = float(np.mean(w2_all[sel]))
        mean_dw02 = float(np.mean(dw02_all[sel]))
        delta_t.append(float(np.sqrt(edges[b - 1] * edges[b])))
        sigma2.append(mean_w2 - mean_dw02)
        sigma2_noise.append(mean_dw02)
        sigma2_err.append(float(np.sqrt(2.0 / nh) * mean_dw02))

    if not delta_t:
        return None
    return (np.asarray(delta_t), np.asarray(sigma2),
            np.asarray(sigma2_noise), np.asarray(sigma2_err))


def _run_haar(ncts, ncts_err, bins, *, target_snr=5.0,
              n_scales=40, dchi2_threshold=4.0):
    """Golkhou & Butler 2014 Haar structure-function MVT."""
    ncts = np.asarray(ncts, dtype=float)
    ncts_err = np.asarray(ncts_err, dtype=float)
    bins = np.asarray(bins, dtype=float)
    lbins = bins[:-1]
    rbins = bins[1:]
    dt = float(bins[1] - bins[0])

    rate, drate, lb, rb = _rebin_to_snr(ncts, ncts_err, lbins, rbins,
                                         target_snr=target_snr)
    if rate.size < 16:
        return _MVTResult(method="haar", mvt=dt,
                          mvt_err_lo=0.0, mvt_err_hi=0.0,
                          is_upper_limit=True,
                          diag={"reason": "too-few-rebinned-bins"})
    sf = _haar_structure_function(rate, drate, lb, rb, n_scales=n_scales)
    if sf is None:
        return _MVTResult(method="haar", mvt=dt,
                          mvt_err_lo=0.0, mvt_err_hi=0.0,
                          is_upper_limit=True,
                          diag={"reason": "log-rate-undefined-on-too-many-bins"})
    delta_t, sigma2, sigma2_noise, sigma2_err = sf
    pos = (sigma2 > 0) & (sigma2 > 3 * sigma2_err)
    if pos.sum() < 4:
        return _MVTResult(method="haar", mvt=float(delta_t[0]),
                          mvt_err_lo=0.0, mvt_err_hi=0.0,
                          is_upper_limit=True,
                          diag={"reason": "no-significant-scaleogram-points",
                                "delta_t": delta_t.tolist(),
                                "sigma2": sigma2.tolist(),
                                "sigma2_err": sigma2_err.tolist()})
    sig = np.sqrt(np.maximum(sigma2, 0.0))
    sig_err = 0.5 * sigma2_err / np.maximum(sig, 1e-30)
    pos_idx = np.where(pos)[0]
    fit_idx = pos_idx[:max(3, pos_idx.size // 3)]
    slope, *_ = np.linalg.lstsq(delta_t[fit_idx].reshape(-1, 1),
                                sig[fit_idx], rcond=None)
    slope = float(slope[0])
    model = slope * delta_t
    dchi2 = np.cumsum(((sig - model) / np.maximum(sig_err, 1e-30)) ** 2)
    base = dchi2[fit_idx[-1]]
    excess = dchi2 - base
    cross = np.where(excess[fit_idx[-1] + 1:] >= dchi2_threshold)[0]
    if cross.size == 0:
        return _MVTResult(method="haar", mvt=float(delta_t[-1]),
                          mvt_err_lo=0.0, mvt_err_hi=0.0,
                          is_upper_limit=True,
                          diag={"reason": "no-departure-from-smooth-model",
                                "delta_t": delta_t.tolist(),
                                "sigma2": sigma2.tolist(),
                                "model": model.tolist()})
    cross_idx = int(fit_idx[-1] + 1 + cross[0])
    mvt_val = float(delta_t[cross_idx])
    err_lo = float(delta_t[cross_idx] - delta_t[max(cross_idx - 1, 0)])
    err_hi = float(delta_t[min(cross_idx + 1, delta_t.size - 1)] - delta_t[cross_idx])
    return _MVTResult(method="haar", mvt=mvt_val,
                      mvt_err_lo=err_lo, mvt_err_hi=err_hi,
                      is_upper_limit=False,
                      diag={"delta_t": delta_t.tolist(),
                            "sigma2": sigma2.tolist(),
                            "sigma2_noise": sigma2_noise.tolist(),
                            "sigma2_err": sigma2_err.tolist(),
                            "smooth_slope": slope,
                            "fit_idx": fit_idx.tolist()})


def _run_mepsa(ncts, ncts_err, bins, *, bkg_rate=None,
               rebin_factors=(1, 4, 16, 64, 256),
               smoothing_windows=(5, 11, 21), snr_thresholds=None,
               use_camisasca_calibration=False):
    """MEPSA-based FWHM_min following Camisasca 2023 / Maccary 2025."""
    ncts = np.asarray(ncts, dtype=float)
    bins = np.asarray(bins, dtype=float)
    dt = float(bins[1] - bins[0])
    peaks = _mepsa_scan(ncts, dt, bkg_rate=bkg_rate,
                        rebin_factors=rebin_factors,
                        smoothing_windows=smoothing_windows,
                        snr_thresholds=snr_thresholds)
    if not peaks:
        smallest_dt = dt * min(rebin_factors)
        return _MVTResult(method="mepsa", mvt=smallest_dt,
                          mvt_err_lo=0.0, mvt_err_hi=0.0,
                          is_upper_limit=True, diag={"peaks": []})

    fwhms = []
    for p in peaks:
        if use_camisasca_calibration:
            snr_ratio = p["snr"] / 4.7 - 1.0
            if snr_ratio <= 0:
                continue
            fwhm = (10 ** -0.31) * p["dt_det"] * snr_ratio ** 0.60 * p["n_adiac"] ** 1.06
            sig = fwhm * (10 ** 0.13 - 1)
        else:
            rb = _rebin_mean(ncts, p["rebin"])
            rb_idx = p["idx"] // p["rebin"]
            fwhm_bins, _, _ = _fwhm_around(rb, rb_idx)
            if np.isnan(fwhm_bins):
                continue
            fwhm = fwhm_bins * p["dt_det"]
            sig = 0.35 * fwhm   # Maccary 2025 quotes ~35% systematic for this estimator
        fwhms.append((fwhm, sig, p))

    if not fwhms:
        smallest_dt = dt * min(rebin_factors)
        return _MVTResult(method="mepsa", mvt=smallest_dt,
                          mvt_err_lo=0.0, mvt_err_hi=0.0,
                          is_upper_limit=True,
                          diag={"peaks": peaks})

    fwhm, sig, narrowest = min(fwhms, key=lambda t: t[0])
    return _MVTResult(method="mepsa", mvt=float(fwhm),
                      mvt_err_lo=float(sig), mvt_err_hi=float(sig),
                      is_upper_limit=False,
                      diag={"peaks": peaks,
                            "narrowest_peak": narrowest,
                            "calibration": "camisasca_A3" if use_camisasca_calibration else "direct_half_max"})


# ---------- helper: dispatch table per data-type subclass --------------------

def _dispatch(method, ncts, ncts_err, bins, *, bkg_rate, noise_model, **kw):
    if method not in VALID_METHODS:
        raise ValueError(
            f"method must be one of {VALID_METHODS!r}, got {method!r}"
        )
    if method == "cwt":
        return _run_cwt(ncts, ncts_err, bins,
                        bkg_rate=bkg_rate, noise_model=noise_model, **kw)
    if method == "haar":
        return _run_haar(ncts, ncts_err, bins, **kw)
    if method == "mepsa":
        return _run_mepsa(ncts, ncts_err, bins, bkg_rate=bkg_rate, **kw)
    raise AssertionError(method)  # pragma: no cover


# ---------- save helper, shared by all subclasses ----------------------------

def _save(savepath, mvt_res, time, net_rate, method):
    from .temp_utils import MVTPlotter  # local import to avoid cycle
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    json_dump(mvt_res.to_dict(), os.path.join(savepath, "mvt_res.json"))
    with plt_rc_context():
        plotter = MVTPlotter()
        plotter.plot_lc(time, net_rate, mvt_res.mvt, mvt_res.is_upper_limit)
        plotter.plot_diag(method, mvt_res.diag)
        plotter.save(os.path.join(savepath, "mvt.pdf"))


# ---------- subclasses -------------------------------------------------------

class pgMVT(pgSignal):
    """MVT for Poisson source + Gaussian-fit background light curves.

    Construct exactly like :class:`heapy.temp.txx.pgTxx`: ``pgMVT(ts, bins,
    exp=None, ignore=None)``.  ``calculate()`` ensures :meth:`polyfit` has run
    (and runs the full :meth:`loop` pipeline with ``deg`` if not), then
    dispatches to the requested algorithm using ``noise_model='poisson'`` and
    ``bkg_rate = mean(self.bak)``.
    """

    def __init__(self, ts, bins, exp=None, ignore=None):
        super().__init__(ts, bins, exp=exp, ignore=ignore)
        self.mvt_res: Optional[_MVTResult] = None

    @classmethod
    def frombin(cls, cts, bins, exp=None, ignore=None, random_seed=450001):
        inst = super().frombin(cts, bins, exp=exp, ignore=ignore,
                               random_seed=random_seed)
        inst.mvt_res = None
        return inst

    @classmethod
    def from_components(cls, obj_list):
        inst = super().from_components(obj_list)
        inst.mvt_res = None
        return inst

    def _ensure_background(self, p0=0.05, sigma=3, deg=None):
        """Run the full pgSignal background pipeline if it has not been run.

        Mirrors :meth:`heapy.temp.txx.pgTxx.find_pulse`'s gating: when
        ``sort_res`` is ``None``, delegate to :meth:`loop` which executes
        ``basefit -> bblock -> calsnr -> sorting -> polyfit`` (twice on the
        raw-events path).  For composite instances :meth:`loop` skips the
        polynomial steps entirely; in either case ``self.bak`` is populated
        afterwards.
        """
        if self.sort_res is None:
            self.loop(p0=p0, sigma=sigma, deg=deg)

    def calculate(self, method="cwt", deg=None, **kw):
        """Run an MVT algorithm; ensures background polynomial is fitted first."""
        if method not in VALID_METHODS:
            raise ValueError(
                f"method must be one of {VALID_METHODS!r}, got {method!r}"
            )
        self._ensure_background(deg=deg)
        bkg_rate = float(np.mean(self.bak))
        self.mvt_res = _dispatch(method, self.ncts, self.ncts_err, self.bins,
                                 bkg_rate=bkg_rate, noise_model="poisson", **kw)
        return self.mvt_res

    def save(self, savepath):
        if self.mvt_res is None:
            raise RuntimeError("call .calculate() before .save(...)")
        _save(savepath, self.mvt_res, self.time, self.net, self.mvt_res.method)


class ppMVT(ppSignal):
    """MVT for Poisson source + Poisson background light curves.

    Construct exactly like :class:`heapy.temp.txx.ppTxx`: ``ppMVT(ts, bts,
    bins, backscale=1, exp=None)``.  ``self.ncts / self.ncts_err / self.bak``
    are populated in ``__init__``; ``calculate()`` only chooses the algorithm.
    """

    def __init__(self, ts, bts, bins, backscale=1, exp=None):
        super().__init__(ts, bts, bins, backscale=backscale, exp=exp)
        self.mvt_res: Optional[_MVTResult] = None

    @classmethod
    def frombin(cls, cts, bcts, bins, backscale=1, exp=None,
                random_seed=450001):
        inst = super().frombin(cts, bcts, bins, backscale=backscale, exp=exp,
                               random_seed=random_seed)
        inst.mvt_res = None
        return inst

    def calculate(self, method="cwt", **kw):
        if method not in VALID_METHODS:
            raise ValueError(
                f"method must be one of {VALID_METHODS!r}, got {method!r}"
            )
        bkg_rate = float(np.mean(self.bak))
        self.mvt_res = _dispatch(method, self.ncts, self.ncts_err, self.bins,
                                 bkg_rate=bkg_rate, noise_model="poisson", **kw)
        return self.mvt_res

    def save(self, savepath):
        if self.mvt_res is None:
            raise RuntimeError("call .calculate() before .save(...)")
        _save(savepath, self.mvt_res, self.time, self.net, self.mvt_res.method)


class ggMVT(ggSignal):
    """MVT for Gaussian-net-rate light curves (no background model needed).

    Construct exactly like :class:`heapy.temp.txx.ggTxx`: ``ggMVT(ncts,
    ncts_err, bins, exp=None)``.  The CWT method uses a **Gaussian** Monte
    Carlo background drawn at ``std = mean(ncts_err)``; Haar and MEPSA are
    distribution-agnostic.
    """

    def __init__(self, ncts, ncts_err, bins, exp=None):
        super().__init__(ncts, ncts_err, bins, exp=exp)
        self.mvt_res: Optional[_MVTResult] = None

    def calculate(self, method="cwt", **kw):
        if method not in VALID_METHODS:
            raise ValueError(
                f"method must be one of {VALID_METHODS!r}, got {method!r}"
            )
        bkg_rate = None  # gg has no Poisson background concept
        self.mvt_res = _dispatch(method, self.ncts, self.ncts_err, self.bins,
                                 bkg_rate=bkg_rate, noise_model="gaussian", **kw)
        return self.mvt_res

    def save(self, savepath):
        if self.mvt_res is None:
            raise RuntimeError("call .calculate() before .save(...)")
        _save(savepath, self.mvt_res, self.time, self.net, self.mvt_res.method)
