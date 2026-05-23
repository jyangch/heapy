"""Reproduce published GRB MVT values on real Fermi/GBM data.

Validates the three algorithms in `heapy.temp.mvt` against:
- Golkhou & Littlejohns 2015 (ApJ 811, 93) Fig 3 — Haar SF on GRB 110213A, 080916A, 120119A
- Vianello et al. 2018 (ApJ 864, 163) Fig 6 — CWT on GRB 100724B, 160509A
- Maccary et al. 2025 (A&A 702, A95) Fig 5 — MEPSA-FWHM on GRB 211211A, 230307A

Usage:
    python scripts/validate_mvt/validate_mvt.py [grb_name ...]

If no GRB names are given, all configured ones are attempted; those without
cached data are skipped with a warning.

Output: per-GRB JSON + PDF in `scripts/validate_mvt/output/<grb>/`.
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


def load_events(grb: dict) -> tuple[np.ndarray, float]:
    """Load + stack TTE events from all configured detectors.

    Returns ``(times_relative_to_trigger, trigtime_met)``.  Applies the
    energy filter using each TTE file's own EBOUNDS table.
    """
    all_t = []
    trigtime = None
    e_lo, e_hi = grb["energy"]
    for det in grb["dets"]:
        f = _find_tte(grb, det)
        if f is None:
            continue
        with fits.open(f) as hdu:
            tt = float(hdu["PRIMARY"].header["TRIGTIME"])
            if trigtime is None:
                trigtime = tt
            events = hdu["EVENTS"].data
            ebound = hdu["EBOUNDS"].data
            chan = np.asarray(ebound["CHANNEL"], dtype=int)
            emin = np.asarray(ebound["E_MIN"], dtype=float)
            emax = np.asarray(ebound["E_MAX"], dtype=float)
            ec = 0.5 * (emin + emax)
            # PHA -> channel index (CHANNEL is 0..N-1)
            pha = np.asarray(events["PHA"], dtype=int)
            ok_pha = (pha >= chan.min()) & (pha <= chan.max())
            pha_clip = np.clip(pha, chan.min(), chan.max())
            energy = np.where(ok_pha, ec[pha_clip - chan.min()], -1.0)
            t_rel = np.asarray(events["TIME"], dtype=float) - tt
            mask = (
                ok_pha
                & (energy >= e_lo)
                & (energy <= e_hi)
                & (t_rel >= grb["t1"])
                & (t_rel <= grb["t2"])
            )
            all_t.append(t_rel[mask])
    if not all_t:
        raise RuntimeError(f"no TTE data found for {grb['burstid']}")
    return np.sort(np.concatenate(all_t)), float(trigtime)


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
    events, trigtime = load_events(grb)
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
        res = _run_haar(net, err, bins)
    elif method == "cwt":
        # pg-style Poisson MC at the measured off-pulse rate
        res = _run_cwt(
            net, err, bins,
            bkg_rate=bkg_rate,
            noise_model="poisson",
            n_sim=grb.get("cwt_n_sim", 500),
            sig_level=99.0,
        )
    elif method == "mepsa":
        res = _run_mepsa(net, err, bins, bkg_rate=bkg_rate)
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
    """Reproduce Golkhou+Littlejohns 2015 Fig 3 layout: scaleogram + LC."""
    diag = res.diag
    fig, (ax_sf, ax_lc) = plt.subplots(
        nrows=2, figsize=(6, 6.5), gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=True,
    )
    dt = np.asarray(diag.get("delta_t", []))
    s2 = np.asarray(diag.get("sigma2", []))
    s2_err = np.asarray(diag.get("sigma2_err", []))
    slope = diag.get("smooth_slope")
    # Significance gate, mimicking the paper's "3sigma excess" markers.
    pos = (s2 > 0) & (s2 > 3 * s2_err)
    if pos.sum():
        sig = np.sqrt(s2[pos])
        ax_sf.errorbar(dt[pos], sig,
                       yerr=0.5 * s2_err[pos] / np.maximum(sig, 1e-30),
                       fmt="o", ms=4, color="C0", label="Fermi/GBM")
    # Upper-limit triangles for non-significant scales.
    if (~pos).sum():
        ax_sf.plot(dt[~pos], np.maximum(np.sqrt(np.maximum(s2[~pos], 0)), 1e-3),
                   "v", ms=5, color="black", label="$3\\sigma$ upper")
    if slope is not None:
        x_line = np.geomspace(dt.min(), dt.max(), 60)
        ax_sf.plot(x_line, slope * x_line, "r--", lw=1, label=r"$\sigma \propto \Delta t$")
    ax_sf.set_xscale("log"); ax_sf.set_yscale("log")
    ax_sf.set_xlabel(r"$\Delta t$ (sec)")
    ax_sf.set_ylabel(r"Flux Variation $\sigma_{X,\Delta t}$")
    if not res.is_upper_limit:
        ax_sf.axvline(res.mvt, color="red", ls=":", alpha=0.6)
        ax_sf.plot([res.mvt], [np.sqrt(max(s2[pos].max() if pos.any() else 1e-3, 1e-3)) * 0.7],
                   "o", ms=10, mfc="red", mec="darkred", label=r"$\Delta t_{\min}$")
    title = f"{grb_name}  (events={events.size:,})"
    pub = grb["published"].get("dt_min")
    ax_sf.set_title(
        title
        + f"\n  measured $\\Delta t_{{\\min}}$ = {res.mvt:.3g} s"
        + (f"  vs published {pub} s" if pub is not None else "")
    )
    ax_sf.legend(loc="lower right", fontsize=8)
    # LC panel
    t, rate = _binned_rate(events, grb["t1"], grb["t2"], 0.064)  # 64 ms bins for display
    ax_lc.step(t, rate, where="mid", lw=0.6, color="C0")
    ax_lc.set_xlabel("Time Since Trigger (sec)")
    ax_lc.set_ylabel("Count Rate (cts/s)")
    fig.savefig(out_dir / "validation.pdf")
    plt.close(fig)


def _plot_cwt_spectrum(grb_name, grb, res, events, out_dir):
    """Reproduce Vianello+2018 Fig 6 layout: rectified power spectrum + MC band."""
    diag = res.diag
    p = np.asarray(diag.get("periods", []))
    ws = np.asarray(diag.get("ws", []))
    med = np.asarray(diag.get("bkg_median", []))
    up = np.asarray(diag.get("bkg_upper", []))
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    if p.size:
        ax.plot(p, ws, "o", ms=4, color="C0", label="observed")
        ax.plot(p, med, "k--", lw=1.0, label="bkg median")
        ax.fill_between(p, med, up, alpha=0.25, color="C0", label="99% upper")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\delta t$ (s)")
    ax.set_ylabel("Power")
    pub = grb["published"].get("t_mv")
    upper = "upper limit " if res.is_upper_limit else ""
    ax.set_title(
        f"{grb_name}  (events={events.size:,})"
        f"\n  measured $t_{{mv}}$ = {upper}{res.mvt:.3g} s"
        + (f"  vs published {pub} s" if pub is not None else "")
    )
    if not res.is_upper_limit:
        ax.axvline(res.mvt, color="red", ls=":", alpha=0.7)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out_dir / "validation.pdf")
    plt.close(fig)


def _plot_mepsa_narrowest(grb_name, grb, res, events, out_dir):
    """Reproduce Maccary+2025 Fig 5 layout: LC + narrowest-pulse inset."""
    diag = res.diag
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    # Display LC at a sensible bin width.
    dt_show = 0.064
    t, rate = _binned_rate(events, grb["t1"], grb["t2"], dt_show)
    ax.step(t, rate, where="mid", lw=0.6, color="red", label=f"GBM {grb['energy'][0]:.0f}-{grb['energy'][1]:.0f} keV")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Count rate (cts/s)")
    pub = grb["published"].get("fwhm_min")
    if res.is_upper_limit:
        ax.set_title(
            f"{grb_name}  (events={events.size:,})"
            f"\n  no narrow peak detected; upper limit {res.mvt:.3g} s"
            + (f"  vs published FWHMmin {pub} s" if pub is not None else "")
        )
    else:
        narrowest = diag.get("narrowest_peak", {})
        t_pk = grb["t1"] + narrowest.get("idx", 0) * grb["dt"]
        ax.axvline(t_pk, color="orange", ls="-", alpha=0.7)
        # Inset zoom around the narrowest peak.
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_in = inset_axes(ax, width="35%", height="35%", loc="upper right")
        dt_zoom = grb["dt"]
        zoom_half = max(20 * narrowest.get("dt_det", 0.01), 0.05)
        zoom_t1, zoom_t2 = t_pk - zoom_half, t_pk + zoom_half
        tz, rz = _binned_rate(events, zoom_t1, zoom_t2, dt_zoom)
        ax_in.step(tz, rz, where="mid", lw=0.6, color="red")
        ax_in.axvspan(t_pk - 0.5 * res.mvt, t_pk + 0.5 * res.mvt,
                      color="orange", alpha=0.3)
        ax_in.set_xlabel("Time [s]", fontsize=7)
        ax_in.set_ylabel("cts/s", fontsize=7)
        ax_in.tick_params(labelsize=7)
        title = (
            f"{grb_name}  (events={events.size:,})"
            f"\n  FWHM_min = {res.mvt*1e3:.2f} ms"
            + (f"  vs published {pub*1e3:.1f} ms" if pub is not None else "")
        )
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
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
