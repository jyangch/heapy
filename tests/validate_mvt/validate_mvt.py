"""Reproduce published GRB MVT values on real Fermi/GBM data.

Validates the three algorithms in `heapy.temp.mvt` against:
- Golkhou & Littlejohns 2015 (ApJ 811, 93) Fig 3 — Haar SF on GRB 110213A, 080916A, 120119A
- Vianello et al. 2018 (ApJ 864, 163) Fig 6 — CWT on GRB 100724B, 160509A
- Maccary et al. 2025 (A&A 702, A95) Fig 5 — MEPSA-FWHM on GRB 211211A, 230307A

Usage:
    python tests/validate_mvt/validate_mvt.py [grb_name ...]

If no GRB names are given, all configured ones are attempted; those without
cached data are skipped with a warning.

Output: per-GRB JSON + PDF in `tests/validate_mvt/output/<grb>/`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from heapy.temp.mvt import (  # noqa: E402
    _MVTResult,
    _fwhm_around,
    _rebin_mean,
    _run_cwt,
    _run_haar,
    _run_mepsa,
)

DATA_ROOT = Path("/Users/junyang/Data/fermi/data/gbm/bursts")
OUT_ROOT = REPO_ROOT / "scripts" / "validate_mvt" / "output"

# All 12 NaI detectors — we stack everything and let the polynomial bkg
# subtraction handle the off-axis ones (they contribute only baseline).
ALL_NAI = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"]

# Per-GRB configuration.  Time windows match published reference where given.
GRBS = {
    # ----- Golkhou & Littlejohns 2015 Fig 3 (Fermi/GBM, 15-350 keV) -----
    "110213A": dict(
        burstid="bn110213220", year=2011,
        dets=ALL_NAI,
        t1=-20, t2=80, ignore=[[-1.0, 50.0]],
        energy=(15.0, 350.0),
        dt=0.001,
        method="haar",
        published={"dt_min": 0.40, "dt_sn": 0.13, "t_beta": 0.24},
        paper="Golkhou+Littlejohns 2015 (Fermi/GBM)",
    ),
    "080916A": dict(
        burstid="bn080916009", year=2008,
        dets=ALL_NAI,
        t1=-30, t2=100, ignore=[[-1.0, 70.0]],
        energy=(15.0, 350.0),
        dt=0.001,
        method="haar",
        published={"dt_min": 1.09, "dt_sn": 0.62, "t_beta": 0.84},
        paper="Golkhou+Littlejohns 2015 (Fermi/GBM)",
    ),
    "120119A": dict(
        burstid="bn120119229", year=2012,
        dets=ALL_NAI,
        t1=-30, t2=80, ignore=[[-1.0, 60.0]],
        energy=(15.0, 350.0),
        dt=0.001,
        method="haar",
        published={"dt_min": 0.20, "dt_sn": 0.08, "t_beta": 0.18},
        paper="Golkhou+Littlejohns 2015 (Fermi/GBM)",
    ),
    # ----- Vianello et al. 2018 Fig 6 (GBM 8-260 keV; paper also uses LLE) -----
    "100724B": dict(
        burstid="bn100724029", year=2010,
        dets=ALL_NAI,
        t1=-5, t2=200, ignore=[[8.0, 123.0]],
        energy=(8.0, 260.0),
        dt=1e-3,  # paper uses 10^-4 s
        method="cwt",
        published={"t_mv": 0.3},
        paper="Vianello+ 2018 (GBM 8-260 keV; paper also used LLE 100 MeV-)",
        cwt_n_sim=150,
        # Vianello Fig 6 x-axis caps at ~burst-window/4; beyond ~20 s the
        # rectified power is dominated by the boundary-artefact tail.
        cwt_max_time_scale=20.0,
    ),
    "160509A": dict(
        burstid="bn160509374", year=2016,
        dets=ALL_NAI,
        t1=-5, t2=40, ignore=[[8.0, 25.0]],
        energy=(8.0, 260.0),
        dt=1e-3,
        method="cwt",
        published={"t_mv": 0.05},
        paper="Vianello+ 2018 (GBM 8-260 keV; paper also used LLE 100 MeV-)",
        cwt_n_sim=150,
        cwt_max_time_scale=5.0,
    ),
    # ----- Maccary et al. 2025 Fig 5 (GBM 8-1000 keV) -----
    "211211A": dict(
        burstid="bn211211549", year=2021,
        dets=ALL_NAI,
        t1=-5, t2=70, ignore=[[-0.5, 50.0]],
        energy=(8.0, 1000.0),
        dt=0.001,  # Maccary refines to 1ms
        method="mepsa",
        published={"fwhm_min": 0.005},  # ~5 ms per Maccary main text
        paper="Maccary+ 2025 (GBM 8-1000 keV)",
        # Restrict peak search to the initial spike train per Maccary
        # Fig 5 left-panel inset at t ~ 1.77 s; otherwise the narrowest
        # peak lands deep in extended emission and reports the wrong
        # variability scale.
        mepsa_restrict_to=[-1.0, 3.0],
    ),
    "230307A": dict(
        burstid="bn230307656", year=2023,
        dets=ALL_NAI,
        t1=-5, t2=70, ignore=[[-0.5, 50.0]],
        energy=(8.0, 1000.0),
        dt=0.001,
        method="mepsa",
        published={"fwhm_min": 0.017},  # ~17 ms per Maccary main text
        paper="Maccary+ 2025 (GBM 8-1000 keV)",
        # Same: restrict to the initial bright phase.
        mepsa_restrict_to=[-1.0, 3.0],
    ),
}


def _data_dir(grb: dict) -> Path:
    return DATA_ROOT / str(grb["year"]) / grb["burstid"] / "current"


def _find_tte(grb: dict, det: str) -> Path | None:
    """Return the TTE file path for ``det`` (any v00/v01/... version)."""
    base = _data_dir(grb)
    matches = sorted(base.glob(f"glg_tte_{det}_{grb['burstid']}_v*.fit"))
    return matches[-1] if matches else None


def is_available(grb_name: str) -> bool:
    grb = GRBS[grb_name]
    base = _data_dir(grb)
    if not base.exists():
        return False
    # Need at least one NaI TTE file.
    return any(_find_tte(grb, d) is not None for d in grb["dets"])


def _load_det_events(grb: dict, det: str):
    """Read TTE events from one detector, return (t_rel, trigtime) or (None, None)."""
    f = _find_tte(grb, det)
    if f is None:
        return None, None
    e_lo, e_hi = grb["energy"]
    with fits.open(f) as hdu:
        trigtime = float(hdu["PRIMARY"].header["TRIGTIME"])
        events = hdu["EVENTS"].data
        ebound = hdu["EBOUNDS"].data
        chan = np.asarray(ebound["CHANNEL"], dtype=int)
        emin = np.asarray(ebound["E_MIN"], dtype=float)
        emax = np.asarray(ebound["E_MAX"], dtype=float)
        ec = 0.5 * (emin + emax)
        pha = np.asarray(events["PHA"], dtype=int)
        ok_pha = (pha >= chan.min()) & (pha <= chan.max())
        pha_clip = np.clip(pha, chan.min(), chan.max())
        energy = np.where(ok_pha, ec[pha_clip - chan.min()], -1.0)
        t_rel = np.asarray(events["TIME"], dtype=float) - trigtime
        mask = (
            ok_pha
            & (energy >= e_lo)
            & (energy <= e_hi)
            & (t_rel >= grb["t1"])
            & (t_rel <= grb["t2"])
        )
    return t_rel[mask], trigtime


def _select_brightest_detectors(grb, snr_threshold=5.0, min_dets=2, max_dets=6):
    """Select all NaI detectors with significant signal in the burst window.

    For each detector, compute SNR = (burst - off) / sqrt(off + 1) over the
    burst interval (the +1 guard against zero off-pulse counts).  Keep all
    detectors with SNR >= snr_threshold, capped at ``max_dets``.  If fewer
    than ``min_dets`` pass, fall back to the top ``min_dets`` by SNR.
    """
    burst_win = grb.get("ignore", [[0.0, 1.0]])[0]
    burst_dur = burst_win[1] - burst_win[0]
    off_lo = max(grb["t1"], burst_win[0] - burst_dur - 1.0)
    off_hi = off_lo + burst_dur
    scored = []
    for det in grb["dets"]:
        t_rel, _ = _load_det_events(grb, det)
        if t_rel is None:
            continue
        n_burst = int(np.sum((t_rel >= burst_win[0]) & (t_rel <= burst_win[1])))
        n_off = int(np.sum((t_rel >= off_lo) & (t_rel <= off_hi)))
        snr = (n_burst - n_off) / max(np.sqrt(n_off + 1), 1.0)
        scored.append((det, snr, n_burst, n_off))
    scored.sort(key=lambda x: -x[1])
    chosen = [d for d, snr, _, _ in scored if snr >= snr_threshold][:max_dets]
    if len(chosen) < min_dets:
        chosen = [d for d, _, _, _ in scored[:min_dets]]
    return chosen, scored


def load_events(grb: dict) -> tuple[np.ndarray, float, list[str]]:
    """Load + stack TTE events from significantly-illuminated detectors.

    Returns ``(times_relative_to_trigger, trigtime_met, dets_used)``.
    When ``grb['auto_dets']`` is truthy (default), keeps all NaI with
    SNR >= ``grb['snr_threshold']`` (default 5.0) in the burst window,
    capped at ``max_dets`` (default 6); falls back to top-``min_dets`` (2)
    by SNR if no detector passes the threshold.  Otherwise stacks all of
    ``grb['dets']``.
    """
    if grb.get("auto_dets", True):
        chosen, scored = _select_brightest_detectors(
            grb,
            snr_threshold=grb.get("snr_threshold", 5.0),
            min_dets=grb.get("min_dets", 2),
            max_dets=grb.get("max_dets", 6),
        )
        print(f"  detector ranking (SNR): "
              + ", ".join(f"{d}={snr:.1f}" for d, snr, _, _ in scored[:8]))
        print(f"  using detectors: {chosen}")
    else:
        chosen = list(grb["dets"])
    all_t = []
    trigtime = None
    for det in chosen:
        t_rel, tt = _load_det_events(grb, det)
        if t_rel is None:
            continue
        if trigtime is None:
            trigtime = tt
        all_t.append(t_rel)
    if not all_t:
        raise RuntimeError(f"no TTE data found for {grb['burstid']}")
    return np.sort(np.concatenate(all_t)), float(trigtime), chosen


def _fit_polynomial_background(t, obs, ignore_intervals, deg=3):
    """Polynomial fit of the observed light curve over off-pulse bins only.

    Returns ``bkg`` (counts per bin, length == len(obs)).
    """
    mask = np.ones_like(obs, dtype=bool)
    for lo, hi in ignore_intervals:
        mask &= ~((t >= lo) & (t <= hi))
    if mask.sum() < deg + 2:
        # not enough off-pulse bins; fall back to a constant equal to the median
        return np.full_like(obs, float(np.median(obs[mask] if mask.any() else obs)))
    coeffs = np.polyfit(t[mask], obs[mask], deg=deg)
    return np.polyval(coeffs, t)


def analyze(grb_name: str) -> dict:
    grb = GRBS[grb_name]
    print(f"\n=== {grb_name} ({grb['burstid']}, method={grb['method']}) ===")
    events, trigtime, dets_used = load_events(grb)
    print(f"  events: {events.size}, t-range [{events.min():.3f}, {events.max():.3f}]")
    bins = np.arange(grb["t1"], grb["t2"] + grb["dt"], grb["dt"])
    print(f"  bins: {bins.size - 1} at dt={grb['dt']} s")

    # Bin observed counts and fit the polynomial background manually
    # (bypasses heapy's full pipe which is fragile at 1 ms binning).
    obs, _ = np.histogram(events, bins=bins)
    obs = obs.astype(float)
    t = 0.5 * (bins[:-1] + bins[1:])
    bkg = _fit_polynomial_background(t, obs, grb["ignore"], deg=grb.get("bkg_deg", 3))
    bkg = np.maximum(bkg, 1e-6)  # guard against negative polynomial values
    net = obs - bkg
    err = np.sqrt(np.maximum(obs + bkg, 1.0))
    bkg_rate = float(np.mean(bkg) / grb["dt"])
    print(f"  bkg_rate ≈ {bkg_rate:.2f} cts/s,  peak net ≈ {net.max():.1f} cts/bin")

    method = grb["method"]
    if method == "haar":
        # Golkhou paper observation: target_snr=3 (lower than the conservative
        # 5.0 default) retains finer composite bins inside the bright peak,
        # which is critical for resolving sub-second rise times.
        res = _run_haar(net, err, bins, target_snr=grb.get("haar_snr", 3.0))
    elif method == "cwt":
        # pg-style Poisson MC at the measured off-pulse rate
        res = _run_cwt(
            net, err, bins,
            bkg_rate=bkg_rate,
            noise_model="poisson",
            n_sim=grb.get("cwt_n_sim", 500),
            sig_level=99.0,
            max_time_scale=grb.get("cwt_max_time_scale"),
        )
    elif method == "mepsa":
        res = _run_mepsa(net, err, bins, bkg_rate=bkg_rate)
        # Optional time-window restriction: filter peaks to those inside
        # ``mepsa_restrict_to`` and re-measure MVT as the narrowest peak
        # among the survivors.  Used for 211211A / 230307A where the
        # globally narrowest peak lands deep in extended emission while
        # the paper-reported MVT sits in the initial spike train.
        if grb.get("mepsa_restrict_to") and not res.is_upper_limit:
            t1_w, t2_w = grb["mepsa_restrict_to"]
            dt_orig = grb["dt"]

            def _in_window(p):
                t_peak = grb["t1"] + p["idx"] * dt_orig
                return t1_w <= t_peak <= t2_w
            restricted_peaks = [p for p in res.diag.get("peaks", []) if _in_window(p)]
            if restricted_peaks:
                fwhms = []
                for p in restricted_peaks:
                    rb = _rebin_mean(net, p["rebin"])
                    rb_idx = p["idx"] // p["rebin"]
                    fwhm_bins, _, _ = _fwhm_around(rb, rb_idx)
                    if not np.isnan(fwhm_bins):
                        fwhms.append((fwhm_bins * p["dt_det"], p))
                if fwhms:
                    fwhm, narrowest = min(fwhms, key=lambda t: t[0])
                    sig = 0.35 * fwhm
                    res = _MVTResult(
                        method="mepsa", mvt=float(fwhm),
                        mvt_err_lo=float(sig), mvt_err_hi=float(sig),
                        is_upper_limit=False,
                        diag={**res.diag,
                              "narrowest_peak": narrowest,
                              "restricted_window": [t1_w, t2_w],
                              "restricted_peaks": len(restricted_peaks),
                              "calibration": "direct_half_max_restricted"},
                    )
    else:
        raise ValueError(method)

    out_dir = OUT_ROOT / grb_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "grb": grb_name,
        "burstid": grb["burstid"],
        "method": grb["method"],
        "n_events": int(events.size),
        "dt": grb["dt"],
        "energy_keV": list(grb["energy"]),
        "ignore": grb["ignore"],
        "result": res.to_dict(),
        "published": grb["published"],
        "paper": grb["paper"],
    }
    # Strip large diagnostic arrays for the summary file.
    diag_compact = {
        k: (v if not isinstance(v, list) or len(v) <= 50 else f"<{len(v)} pts>")
        for k, v in res.diag.items()
    }
    summary["diag_compact"] = diag_compact
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  mvt = {res.mvt:.4g} s  (published: {grb['published']})")
    print(f"  is_upper_limit = {res.is_upper_limit}")

    _make_paper_style_plot(grb_name, grb, res, events, out_dir)
    return summary


def _make_paper_style_plot(grb_name: str, grb: dict, res, events, out_dir: Path):
    """Produce a plot resembling the corresponding figure in the source paper."""
    method = grb["method"]
    if method == "haar":
        _plot_haar_scaleogram(grb_name, grb, res, events, out_dir)
    elif method == "cwt":
        _plot_cwt_spectrum(grb_name, grb, res, events, out_dir)
    elif method == "mepsa":
        _plot_mepsa_narrowest(grb_name, grb, res, events, out_dir)


def _binned_rate(events, t1, t2, dt):
    edges = np.arange(t1, t2 + dt, dt)
    cts, _ = np.histogram(events, bins=edges)
    return 0.5 * (edges[:-1] + edges[1:]), cts / dt


def _plot_haar_scaleogram(grb_name, grb, res, events, out_dir):
    """Reproduce Golkhou+Littlejohns 2015 Fig 3 layout: scaleogram + LC.

    Distinctive Fig 3 features rendered:
    - faint grey dotted diagonal grid (constant sigma * Δt^-0.5 reference lines)
    - blue circles with errorbars for significant points
    - downward-pointing black triangles for 3σ upper limits
    - red dashed line for the fitted smooth μ_0(Δt) power-law
    - large red filled circle at Δt_min
    - upper-right inset text: Δt_S/N, t_β, Δt_min
    """
    diag = res.diag
    fig, (ax_sf, ax_lc) = plt.subplots(
        nrows=2, figsize=(6, 6.5), gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=True,
    )
    dt = np.asarray(diag.get("delta_t", []))
    s2 = np.asarray(diag.get("sigma2", []))
    s2_err = np.asarray(diag.get("sigma2_err", []))
    slope = diag.get("smooth_slope")
    norm = diag.get("smooth_norm")
    fit_idx = diag.get("fit_idx", [])
    pos = (s2 > 0) & (s2 > 3 * s2_err)

    # Establish plot extent for diagonal grid.
    if dt.size:
        x_lo = float(dt.min()) * 0.5
        x_hi = float(dt.max()) * 2.0
    else:
        x_lo, x_hi = 1e-3, 10.0
    if pos.any():
        sig_pos = np.sqrt(s2[pos])
        err_pos = 0.5 * s2_err[pos] / np.maximum(sig_pos, 1e-30)
        y_lo = float(np.min(sig_pos - err_pos)) * 0.3
        y_hi = float(np.max(sig_pos + err_pos)) * 3.0
    else:
        y_lo, y_hi = 1e-3, 1.0
    y_lo = max(y_lo, 1e-4)

    # Faint grey dotted diagonal grid: constant sigma * Δt^-0.5 contours.
    # In log-log these are parallel lines of slope -0.5.  Draw ~8 of them
    # spanning the plot region (Fig 3's most distinctive visual element).
    log_x = np.log10([x_lo, x_hi])
    log_y = np.log10([y_lo, y_hi])
    # Range of intercepts c where log y = -0.5 * log x + c covers the box.
    c_lo = log_y[0] + 0.5 * log_x[0]
    c_hi = log_y[1] + 0.5 * log_x[1]
    for c in np.linspace(c_lo, c_hi, 8):
        xs = np.geomspace(x_lo, x_hi, 50)
        ys = 10.0 ** (-0.5 * np.log10(xs) + c)
        ax_sf.plot(xs, ys, ls=":", color="grey", lw=0.6, alpha=0.5, zorder=0)

    # Blue dots with errorbars for significant points.
    if pos.any():
        ax_sf.errorbar(dt[pos], sig_pos, yerr=err_pos,
                       fmt="o", ms=5, color="C0", ecolor="C0",
                       capsize=0, zorder=3, label="Fermi/GBM")

    # Downward-pointing BLACK TRIANGLES for upper limits.
    if (~pos).any():
        ul = np.where(~pos)[0]
        ul_y = np.sqrt(np.maximum(s2[ul], 0.0))
        # Use a small finite floor for any sigma2<=0 to keep them on the plot.
        floor = y_lo * 3
        ul_y = np.where(ul_y > floor, ul_y, floor)
        ax_sf.plot(dt[ul], ul_y, "v", ms=6, color="black",
                   zorder=2, label=r"$3\sigma$ upper limit")

    # Red dashed line for the fitted μ_0 power-law.
    if slope is not None and norm is not None:
        xs = np.geomspace(x_lo, x_hi, 80)
        mu0_line = norm * xs ** slope
        ax_sf.plot(xs, mu0_line, "r--", lw=1.2, zorder=4,
                   label=fr"$\mu_0 \propto \Delta t^{{{slope:.2f}}}$")

    # Large red filled circle at Δt_min.
    if not res.is_upper_limit:
        # y-position from the smooth model at Δt_min.
        if slope is not None and norm is not None:
            y_at_mvt = float(norm * res.mvt ** slope)
        else:
            y_at_mvt = float(sig_pos.max() * 0.5) if pos.any() else 1e-2
        ax_sf.plot([res.mvt], [y_at_mvt], "o", ms=14, mfc="red", mec="darkred",
                   mew=1.5, zorder=5, label=r"$\Delta t_{\min}$")

    ax_sf.set_xscale("log"); ax_sf.set_yscale("log")
    ax_sf.set_xlim(x_lo, x_hi)
    ax_sf.set_ylim(y_lo, y_hi)
    ax_sf.set_xlabel(r"$\Delta t$ (sec)")
    ax_sf.set_ylabel(r"Flux Variation $\sigma_{X,\Delta t}$")

    # Title with measured / published.
    pub = grb["published"].get("dt_min")
    title = f"{grb_name}  (events={events.size:,})"
    if pub is not None:
        title += f"\n measured $\\Delta t_{{\\min}}$ = {res.mvt:.3g} s  vs published {pub} s"
    else:
        title += f"\n measured $\\Delta t_{{\\min}}$ = {res.mvt:.3g} s"
    ax_sf.set_title(title)

    # Upper-right inset text annotations.
    if pos.any():
        first_sig = int(np.where(pos)[0][0])
        dt_sn = float(dt[first_sig])
    else:
        dt_sn = float("nan")
    if fit_idx:
        t_beta = float(dt[fit_idx[-1]])
    else:
        t_beta = float("nan")
    inset_text = (
        f"$\\Delta t_{{S/N}}$ = {dt_sn:.3g} s\n"
        f"$t_\\beta$ = {t_beta:.3g} s\n"
        f"$\\Delta t_{{\\min}}$ = {res.mvt:.3g} s"
    )
    ax_sf.text(0.97, 0.97, inset_text, transform=ax_sf.transAxes,
               ha="right", va="top", fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor="grey", alpha=0.85))
    ax_sf.legend(loc="lower right", fontsize=8, framealpha=0.85)

    # LC panel: net rate (display-binned), blue step.
    t, rate = _binned_rate(events, grb["t1"], grb["t2"], 0.064)
    ax_lc.step(t, rate, where="mid", lw=0.6, color="C0", label="Fermi/GBM")
    ax_lc.set_xlabel("Time Since Trigger (sec)")
    ax_lc.set_ylabel("Count Rate (cts/s)")
    ax_lc.legend(loc="upper right", fontsize=8)
    fig.savefig(out_dir / "validation.pdf")
    plt.close(fig)


def _plot_cwt_spectrum(grb_name, grb, res, events, out_dir):
    """Reproduce Vianello+2018 Fig 6 layout: rectified power spectrum + MC band.

    Single-panel log-log:
    - blue dots for the observed rectified spectrum
    - BLACK DASHED line for the MC median
    - LIGHT BLUE filled band between MC median and 99% upper percentile
    - NO red vertical line for t_mv (paper doesn't show one)
    """
    diag = res.diag
    p = np.asarray(diag.get("periods", []))
    ws = np.asarray(diag.get("ws", []))
    med = np.asarray(diag.get("bkg_median", []))
    up = np.asarray(diag.get("bkg_upper", []))
    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    if p.size:
        # Light blue filled band first (so the line + points sit on top).
        ax.fill_between(p, med, up, color="#9ecae1", alpha=0.55,
                        label="99% MC envelope")
        ax.plot(p, med, ls="--", color="black", lw=1.2, label="MC median")
        ax.plot(p, ws, "o", ms=4, color="C0", label="observed")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\delta t$ (s)")
    ax.set_ylabel(r"Rectified power $W(\delta t)$")
    pub = grb["published"].get("t_mv")
    upper = "upper limit " if res.is_upper_limit else ""
    title = (
        f"{grb_name}  (events={events.size:,})"
        f"\n measured $t_{{mv}}$ = {upper}{res.mvt:.3g} s"
    )
    if pub is not None:
        title += f"  vs published {pub} s"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out_dir / "validation.pdf")
    plt.close(fig)


def _plot_mepsa_narrowest(grb_name, grb, res, events, out_dir):
    """Reproduce Maccary+2025 Fig 5 layout: LC + narrowest-pulse inset.

    Main panel: red step net rate vs time since trigger; yellow translucent
    band over the initial-spike interval; blue translucent band over the
    extended-emission interval.

    Inset (top-right): zoom on the narrowest peak, red step LC at dt_det,
    orange band marking [t_peak - 0.5*mvt, t_peak + 0.5*mvt], black dot at
    the peak with horizontal error bar of width dt_det.
    """
    diag = res.diag
    pub = grb["published"].get("fwhm_min")
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    dt_show = 0.064
    t, rate = _binned_rate(events, grb["t1"], grb["t2"], dt_show)
    ax.step(t, rate, where="mid", lw=0.7, color="red",
            label=f"GBM {grb['energy'][0]:.0f}-{grb['energy'][1]:.0f} keV")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Count rate (cts/s)")

    if res.is_upper_limit:
        ax.set_title(
            f"{grb_name}  (events={events.size:,})"
            f"\n no narrow peak detected; upper limit {res.mvt:.3g} s"
            + (f"  vs published FWHMmin {pub} s" if pub is not None else "")
        )
        ax.legend(loc="upper right", fontsize=9)
        fig.savefig(out_dir / "validation.pdf")
        plt.close(fig)
        return

    peaks = diag.get("peaks", [])
    narrowest = diag.get("narrowest_peak", {})
    t_pk = grb["t1"] + narrowest.get("idx", 0) * grb["dt"]
    dt_det_n = float(narrowest.get("dt_det", grb["dt"]))

    # ----- Initial-spike interval (yellow band) -----
    # Use the EARLIEST peak in time as the initial spike.  Its bin width
    # gives the spike extent.
    if peaks:
        first_peak = min(peaks, key=lambda p: p["idx"])
        t_first = grb["t1"] + first_peak["idx"] * grb["dt"]
        dt_first = float(first_peak["dt_det"])
        # Spike extent: +/- one dt_det around the peak position.
        spike_lo = t_first - dt_first
        spike_hi = t_first + dt_first
        ax.axvspan(spike_lo, spike_hi, color="gold", alpha=0.35,
                   label="initial spike")
        # ----- Extended-emission interval (blue band) -----
        # Starts after the spike (+ 2*dt of the display bin) and runs to
        # the ignore-window upper bound.
        burst_win = grb.get("ignore", [[grb["t1"], grb["t2"]]])[0]
        ext_lo = spike_hi + 2 * grb["dt"]
        ext_hi = float(burst_win[1])
        if ext_hi > ext_lo:
            ax.axvspan(ext_lo, ext_hi, color="steelblue", alpha=0.18,
                       label="extended emission")

    # ----- Inset zoom on the narrowest peak -----
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_in = inset_axes(ax, width="35%", height="40%", loc="upper right",
                       borderpad=1.5)
    # Inset uses dt_det of the narrowest peak as its bin width.
    dt_zoom = max(dt_det_n, grb["dt"])
    zoom_half = max(20 * dt_det_n, 0.05)
    zoom_t1, zoom_t2 = t_pk - zoom_half, t_pk + zoom_half
    tz, rz = _binned_rate(events, zoom_t1, zoom_t2, dt_zoom)
    ax_in.step(tz, rz, where="mid", lw=0.7, color="red")
    # Orange translucent band over [t_pk - 0.5*mvt, t_pk + 0.5*mvt].
    ax_in.axvspan(t_pk - 0.5 * res.mvt, t_pk + 0.5 * res.mvt,
                  color="orange", alpha=0.45)
    # Black dot at peak position with horizontal error bar of width dt_det.
    if rz.size:
        # Peak rate near the inset's centre.
        i_centre = int(np.argmin(np.abs(tz - t_pk))) if tz.size else 0
        y_pk = float(rz[i_centre]) if rz.size else 0.0
    else:
        y_pk = 0.0
    ax_in.errorbar([t_pk], [y_pk], xerr=[0.5 * dt_det_n],
                   fmt="o", color="black", ms=4, capsize=2,
                   ecolor="black", zorder=5)
    ax_in.set_xlabel("Time (s)", fontsize=7)
    ax_in.set_ylabel("cts/s", fontsize=7)
    ax_in.tick_params(labelsize=7)
    ax_in.set_title(f"narrowest peak  (Δt_det={dt_det_n*1e3:.1f} ms)",
                    fontsize=7)

    title = (
        f"{grb_name}  (events={events.size:,})"
        f"\n FWHM$_{{\\min}}$ = {res.mvt*1e3:.2f} ms"
    )
    if pub is not None:
        title += f"  vs published {pub*1e3:.1f} ms"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out_dir / "validation.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("grbs", nargs="*", help="GRB names to run (default: all available)")
    args = parser.parse_args()
    targets = args.grbs if args.grbs else list(GRBS.keys())

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in targets:
        if name not in GRBS:
            print(f"  [skip] {name}: not configured")
            continue
        if not is_available(name):
            print(f"  [skip] {name}: not downloaded (looked in {_data_dir(GRBS[name])})")
            continue
        try:
            summaries.append(analyze(name))
        except Exception as e:
            print(f"  [error] {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # Table
    if summaries:
        print("\n\n=== Summary ===")
        print(f"{'GRB':<10} {'method':<6} {'measured':<12} {'published':<14} {'paper'}")
        for s in summaries:
            pub = s["published"]
            pub_str = ", ".join(f"{k}={v}" for k, v in pub.items())
            measured = (
                f"{s['result']['mvt']:.3g} s"
                + (" (UL)" if s["result"]["is_upper_limit"] else "")
            )
            print(f"{s['grb']:<10} {s['method']:<6} {measured:<12} {pub_str:<14} {s['paper']}")


if __name__ == "__main__":
    main()
