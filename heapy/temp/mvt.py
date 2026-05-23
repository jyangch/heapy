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


def _moving_stats(x, window):
    """Centre-excluded moving mean and std (same length as ``x``).

    Edges fall back to the global mean/std until the window is fully populated.
    The centre bin is excluded from each local statistic so that a true spike
    does not inflate its own noise estimate.  When the local std collapses to
    zero (or below numerical precision -- e.g. a flat block of identical
    floats), it is replaced by the global std so a numerical artefact does
    not produce a spuriously divergent SNR.
    """
    n = x.shape[0]
    half = window // 2
    out_mean = np.empty(n)
    out_std = np.empty(n)
    g_mean = float(x.mean())
    g_std = float(x.std(ddof=1)) if n > 1 else 1.0
    std_floor = max(g_std, 1.0) * 1e-9
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo <= 1:
            out_mean[i] = g_mean
            out_std[i] = g_std
            continue
        block = np.concatenate([x[lo:i], x[i + 1:hi]])
        out_mean[i] = block.mean()
        out_std[i] = block.std(ddof=1) if block.size > 1 else g_std
    fallback = g_std if g_std > std_floor else 1.0
    out_std = np.where(out_std > std_floor, out_std, fallback)
    return out_mean, out_std


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


def _mepsa_scan(net, dt, rebin_factors=(1, 4, 16, 64, 256),
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
    # Estimate the implied background count level subtracted to make ``net``.
    # ``net = obs - bkg``: baseline-only bins typically sit at ``-bkg``, so
    # the magnitude of the lower-percentile values bounds the bkg level.
    # Used as a Poisson-noise floor below.
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
        if rb_dt < 2 * dt and R != 1:
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
            mean, std = _moving_stats(rb, M)
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


# ---------- algorithm free-function stubs (filled in by later tasks) ----------

def _run_cwt(ncts, ncts_err, bins, *, bkg_rate, noise_model,
             n_sim=1000, sig_level=99.0, dj=0.125, random_seed=450001):
    """Vianello 2018 CWT MVT.  Filled in by Task 4."""
    raise NotImplementedError("Task 4 fills this in")


def _run_haar(ncts, ncts_err, bins, *, target_snr=5.0,
              n_scales=40, dchi2_threshold=4.0):
    """Golkhou 2014 Haar structure-function MVT.  Filled in by Task 5."""
    raise NotImplementedError("Task 5 fills this in")


def _run_mepsa(ncts, ncts_err, bins, *, rebin_factors=(1, 4, 16, 64, 256),
               smoothing_windows=(5, 11, 21), snr_thresholds=None,
               use_camisasca_calibration=False):
    """Camisasca 2023 / Maccary 2025 MEPSA-FWHM MVT.  Filled in by Task 3."""
    raise NotImplementedError("Task 3 fills this in")


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
        return _run_mepsa(ncts, ncts_err, bins, **kw)
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
