"""Point-by-point comparison: our MVT algorithms vs the published reference
implementations from the algorithm authors themselves.

The goal of this script is to settle once and for all whether the residual
2-4x discrepancies on the real-GRB validation are due to algorithm bugs in
our code, or to methodological differences (detector mask, energy band,
background subtraction).

Method:
  1. Generate ONE synthetic Poisson-realised FRED light curve with known
     rise/decay parameters.
  2. Feed the SAME LC into:
     - Our CWT implementation                       (heapy.temp.mvt._cwt_global_spectrum)
     - Vianello's mvts.wavelet_spectrum             (giacomov/mvts upstream)
  3. Feed the SAME (log-rate, log-rate-err) into:
     - Our Haar SF implementation                   (heapy.temp.mvt._haar_structure_function)
     - Golkhou's inlined non-decimated Haar formula (zgolkhou/GRB_Lightcurve_MinimumVariability-tmin- haar_nondec.py)
  4. Print point-by-point ratios.

Reference repos must be cloned to:
    /tmp/mvt-research/vianello                                  (Vianello's mvts)
    /tmp/mvt-research/GRB_Lightcurve_MinimumVariability-tmin--master  (Golkhou's repo)

Reproduce:
    python tests/validate_mvt/reference_comparison.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VIANELLO_PATH = Path("/tmp/mvt-research/vianello")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VIANELLO_PATH))
warnings.filterwarnings("ignore")

from heapy.temp.mvt import _cwt_global_spectrum, _haar_structure_function  # noqa: E402


def synth_fred(seed=42, dt=0.01, N=4096, rise=0.5, decay=1.5, p=1.5,
               amp=200.0, bkg_rate=10.0):
    """Single FRED pulse with Poisson noise, padded to a power-of-2 length
    so Vianello's CWT acceptance test passes.
    """
    rng = np.random.default_rng(seed)
    bins = np.arange(N + 1) * dt - N * dt / 2
    t = 0.5 * (bins[:-1] + bins[1:])
    profile = np.where(
        t < 0,
        amp * np.exp(-(np.abs(t) / rise) ** p),
        amp * np.exp(-(np.abs(t) / decay) ** p),
    )
    bkg_per_bin = bkg_rate * dt
    obs = rng.poisson(profile + bkg_per_bin).astype(float)
    return t, bins, obs, bkg_rate, bkg_per_bin


def compare_cwt(t, obs, dt):
    """Run our CWT and Vianello's mvts on the same LC; print ratios."""
    from mvts import mvts as vmvts

    print("\n=== CWT (Vianello 2018) — reference vs ours ===\n")
    result, _ = vmvts.wavelet_spectrum(
        t, obs, dt, t[0], t[-1], plot=False, quiet=True,
    )
    periods_ref = np.array(result["period"])
    scales_ref = np.array(result["scale"])
    ws_rect_ref = np.array(result["global_ws"]) / scales_ref

    # Use Vianello's actual default dj=0.25 (mvts.py:41 uses 0.125*2) for an
    # element-by-element comparison on identical scale grids.  Our default
    # is dj=0.125 (finer), which produces 2x as many scales — the algorithms
    # still agree, but a like-for-like comparison needs matched dj.
    periods_ours, ws_rect_ours = _cwt_global_spectrum(obs, dt, dj=0.25)

    print(f"  reference: {len(periods_ref)} scales, "
          f"period ∈ [{periods_ref.min():.4g}, {periods_ref.max():.4g}]")
    print(f"  ours:      {len(periods_ours)} scales, "
          f"period ∈ [{periods_ours.min():.4g}, {periods_ours.max():.4g}]")
    print()
    n = min(len(periods_ref), len(periods_ours))
    ratios = ws_rect_ours[:n] / np.where(
        np.abs(ws_rect_ref[:n]) > 1e-30, ws_rect_ref[:n], np.inf,
    )
    print(f"  Max |ratio − 1| across all {n} scales: {np.max(np.abs(ratios - 1)):.2e}")
    print(f"  Mean ratio:                {np.mean(ratios):.8f}")
    print()
    print(f"  {'period (s)':>12} {'Vianello W/scale':>20} "
          f"{'ours W/scale':>18} {'ratio':>10}")
    for i in [0, 6, 12, 18, 24, 30, 36, 42]:
        if i >= n:
            continue
        print(f"  {periods_ref[i]:>12.4g} {ws_rect_ref[i]:>20.4g} "
              f"{ws_rect_ours[i]:>18.4g} {ratios[i]:>10.5f}")


def compare_haar(rate, drate, lbins, rbins):
    """Run our Haar SF and Golkhou's exact inlined wavelet on the same data."""
    print("\n=== Haar SF (Golkhou 2014) — reference vs ours ===\n")

    # Inline Golkhou's exact non-decimated Haar (from his haar_nondec.py
    # lines 65-74): for every (position i, scale s) compute the wavelet
    # coefficient and the local Δt, then bin (w², dw0²) by local Δt.
    x = np.log(rate)
    dx = drate / rate
    cx = np.concatenate([[0.0], np.cumsum(x)])
    vx = np.concatenate([[0.0], np.cumsum(dx ** 2)])
    t_centre = 0.5 * (lbins + rbins)
    ct = np.concatenate([[0.0], np.cumsum(t_centre)])
    n = x.shape[0]

    scales = np.unique(np.round(np.geomspace(1, max(2, n // 4), 40)).astype(int))
    scales = scales[(scales > 0) & (2 * scales < n)]

    all_dt, all_w2, all_dw02 = [], [], []
    for s in scales:
        ii = np.arange(0, n - 2 * s + 1)
        w = (cx[ii + 2 * s] - 2 * cx[ii + s] + cx[ii]) / s
        dw0 = np.sqrt(vx[ii + 2 * s] - vx[ii]) / s
        local_dt = (ct[ii + 2 * s] - 2 * ct[ii + s] + ct[ii]) / s
        all_dt.extend(local_dt.tolist())
        all_w2.extend((w ** 2).tolist())
        all_dw02.extend((dw0 ** 2).tolist())
    all_dt = np.array(all_dt)
    all_w2 = np.array(all_w2)
    all_dw02 = np.array(all_dw02)
    valid = (all_dt > 0) & np.isfinite(all_dt) & np.isfinite(all_w2)
    adt, aw2, adw0 = all_dt[valid], all_w2[valid], all_dw02[valid]

    dt_bins = np.geomspace(adt.min(), adt.max(), 41)
    golkhou_dt = np.sqrt(dt_bins[:-1] * dt_bins[1:])
    golkhou_s2 = np.full_like(golkhou_dt, np.nan)
    for i in range(len(dt_bins) - 1):
        sel = (adt >= dt_bins[i]) & (adt < dt_bins[i + 1])
        if sel.sum() < 4:
            continue
        golkhou_s2[i] = np.mean(aw2[sel]) - np.mean(adw0[sel])

    delta_t_ours, sigma2_ours, _, _ = _haar_structure_function(
        rate, drate, lbins, rbins, n_scales=40,
    )

    print(f"  reference: {(~np.isnan(golkhou_s2)).sum()} non-NaN bins")
    print(f"  ours:      {len(delta_t_ours)} bins")
    print()
    print(f"  {'Δt (s)':>12} {'Golkhou σ²':>18} {'ours σ²':>18} "
          f"{'ratio':>10}")
    for dt_target in [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]:
        if dt_target < golkhou_dt.min() or dt_target > golkhou_dt.max():
            continue
        i_g = int(np.argmin(np.abs(golkhou_dt - dt_target)))
        i_o = int(np.argmin(np.abs(delta_t_ours - dt_target)))
        if np.isnan(golkhou_s2[i_g]) or golkhou_s2[i_g] == 0:
            continue
        ratio = float(sigma2_ours[i_o] / golkhou_s2[i_g])
        print(f"  {dt_target:>12.4g} {golkhou_s2[i_g]:>18.4g} "
              f"{sigma2_ours[i_o]:>18.4g} {ratio:>10.4f}")


def main():
    print("=" * 60)
    print("Reference comparison: our algorithms vs upstream implementations")
    print("=" * 60)
    print("\nSynthetic FRED: rise=0.5 s, decay=1.5 s, FWHM≈1.56 s, "
          "amp=200, bkg=10 cts/s, dt=10 ms, N=4096")

    t, bins, obs, bkg_rate, bkg_per_bin = synth_fred()
    compare_cwt(t, obs, dt=float(bins[1] - bins[0]))

    # For Haar: drop zero-count bins so log(rate) is finite
    mask = obs > 0
    rate = obs[mask] / (bins[1] - bins[0])
    drate = np.sqrt(obs[mask]) / (bins[1] - bins[0])
    lb = bins[:-1][mask]
    rb = bins[1:][mask]
    compare_haar(rate, drate, lb, rb)

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print("If all ratios are within ±5% of 1.0, the two implementations")
    print("are numerically identical on this LC.  Any residual real-GRB")
    print("discrepancy must come from data preprocessing (detector mask,")
    print("energy band, background subtraction), not from the algorithm.")


if __name__ == "__main__":
    main()
