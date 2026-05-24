"""Systematic synthetic-recovery study of the three MVT algorithms.

Generates FRED pulses with KNOWN rise time / FWHM across 0.005-10 s and runs
each algorithm to measure systematic bias.  If an algorithm is correct, the
recovered MVT should match the ground truth within ~30% (the published
between-algorithm scatter on the same GRB).

Outputs:
    scripts/validate_mvt/output/synthetic/recovery_<algo>.csv
    scripts/validate_mvt/output/synthetic/recovery_<algo>.pdf
    scripts/validate_mvt/output/synthetic/RECOVERY_REPORT.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from heapy.temp.mvt import _run_cwt, _run_haar, _run_mepsa  # noqa: E402

OUT_DIR = REPO_ROOT / "scripts" / "validate_mvt" / "output" / "synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fred(t, t0, rise, decay, amp, p=1.5):
    """FRED profile identical to Camisasca 2023 eq A.1."""
    return np.where(
        t < t0,
        amp * np.exp(-(np.abs(t - t0) / rise) ** p),
        amp * np.exp(-(np.abs(t - t0) / decay) ** p),
    )


def fred_fwhm(rise, decay, p=1.5):
    """Analytic FWHM of a FRED pulse: |t - t0| where profile = amp/2.

    Solve exp(-(|dt|/scale)^p) = 1/2  ⇒  |dt| = scale * (ln 2)^(1/p).
    """
    half_rise = rise * (np.log(2)) ** (1 / p)
    half_decay = decay * (np.log(2)) ** (1 / p)
    return half_rise + half_decay


def make_fred_lc(rise, decay=None, peak_snr=30.0, dt=None, t_span=None,
                 bkg_rate=10.0, seed=42):
    """Synthesize a Poisson-realised FRED light curve with known parameters.

    Args:
        rise: FRED rise time (s).
        decay: FRED decay time (s).  Default 3*rise (Camisasca convention).
        peak_snr: peak SNR per sample at the chosen ``dt``.  Used to back-solve
            the amplitude so the peak Poisson SNR ≈ this value.
        dt: bin width.  Default rise/20 (resolves the rise with 20 samples).
        t_span: total LC span (s).  Default 20 * (rise + decay).
        bkg_rate: flat background in cts/s.
        seed: RNG seed.

    Returns:
        (net, err, bins, bkg_rate, truth) where ``truth`` is a dict with
        keys ``rise, decay, fwhm, p``.
    """
    rng = np.random.default_rng(seed)
    if decay is None:
        decay = 3 * rise
    if dt is None:
        dt = rise / 20.0
    if t_span is None:
        t_span = max(20.0 * (rise + decay), 4.0)
    bins = np.arange(-t_span / 2, t_span / 2 + dt, dt)
    t = 0.5 * (bins[:-1] + bins[1:])
    # Back-solve amplitude so peak SNR ≈ peak_snr at dt
    # SNR_peak = amp / sqrt(amp + bkg_rate*dt)
    # ⇒ amp = (snr² + snr*sqrt(snr² + 4*bkg)) / 2
    bkg_per_bin = bkg_rate * dt
    snr = peak_snr
    amp = 0.5 * (snr * snr + snr * np.sqrt(snr * snr + 4 * bkg_per_bin))
    profile = fred(t, t0=0.0, rise=rise, decay=decay, amp=amp)
    obs = rng.poisson(profile + bkg_per_bin).astype(float)
    bkg = np.full_like(obs, bkg_per_bin)
    net = obs - bkg
    err = np.sqrt(obs + bkg)
    truth = {
        "rise": rise, "decay": decay, "p": 1.5,
        "fwhm": fred_fwhm(rise, decay, p=1.5),
        "amp": amp, "dt": dt, "t_span": t_span,
        "peak_snr": peak_snr, "bkg_rate": bkg_rate,
    }
    return net, err, bins, bkg_rate, truth


def study(algo, rise_values, peak_snr=30.0):
    """Run ``algo`` on FRED pulses with each rise time; return list of dicts."""
    results = []
    for r in rise_values:
        net, err, bins, bkg_rate, truth = make_fred_lc(rise=r, peak_snr=peak_snr)
        if algo == "haar":
            res = _run_haar(net, err, bins, target_snr=3.0)
            measured = res.mvt
            expected = truth["rise"]
            label = "rise"
        elif algo == "cwt":
            res = _run_cwt(net, err, bins, bkg_rate=bkg_rate,
                           noise_model="poisson", n_sim=100, sig_level=99.0)
            measured = res.mvt
            expected = truth["rise"]
            label = "rise"
        elif algo == "mepsa":
            res = _run_mepsa(net, err, bins, bkg_rate=bkg_rate)
            measured = res.mvt
            expected = truth["fwhm"]
            label = "fwhm"
        else:
            raise ValueError(algo)
        results.append({
            "rise": r,
            "expected": expected,
            "expected_label": label,
            "measured": measured,
            "is_upper_limit": res.is_upper_limit,
            "ratio": measured / expected if expected > 0 else float("nan"),
            "fwhm": truth["fwhm"],
            "peak_snr": peak_snr,
        })
        print(f"  rise={r:.4g}s, expected_{label}={expected:.4g}s, "
              f"measured={measured:.4g}s, ratio={measured / expected:.2f}, "
              f"upper_limit={res.is_upper_limit}")
    return results


def write_csv(rows, path):
    with open(path, "w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")


def plot_recovery(algo, rows, path):
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    expected = np.array([r["expected"] for r in rows])
    measured = np.array([r["measured"] for r in rows])
    ul = np.array([r["is_upper_limit"] for r in rows], dtype=bool)
    ax.loglog(expected[~ul], measured[~ul], "o", ms=7, label="measured")
    if ul.any():
        ax.loglog(expected[ul], measured[ul], "v", ms=7, color="gray",
                  alpha=0.5, label="upper limit")
    diag = np.geomspace(expected.min() * 0.5, expected.max() * 2, 50)
    ax.loglog(diag, diag, "k--", label="ideal y=x", alpha=0.6)
    ax.loglog(diag, 2 * diag, ":", color="red", alpha=0.4)
    ax.loglog(diag, 0.5 * diag, ":", color="red", alpha=0.4, label="±factor 2")
    ax.set_xlabel(f"Truth ({rows[0]['expected_label']}, s)")
    ax.set_ylabel("Recovered MVT (s)")
    ax.set_title(f"{algo.upper()}: synthetic recovery (peak_snr={rows[0]['peak_snr']:.0f})")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.2)
    fig.savefig(path)
    plt.close(fig)


def summarise(rows):
    """Compute geometric-mean ratio and scatter, excluding upper limits."""
    log_ratios = []
    for r in rows:
        if r["is_upper_limit"] or r["ratio"] <= 0:
            continue
        log_ratios.append(np.log10(r["ratio"]))
    if not log_ratios:
        return None
    log_ratios = np.array(log_ratios)
    return {
        "geo_mean_ratio": 10 ** float(np.mean(log_ratios)),
        "log_scatter_dex": float(np.std(log_ratios)),
        "n_used": len(log_ratios),
        "n_upper_limits": sum(1 for r in rows if r["is_upper_limit"]),
    }


def main():
    rise_values = np.geomspace(0.005, 10.0, 14)
    print("=== Synthetic recovery study ===")
    print(f"  rise grid: {rise_values}")
    print(f"  peak_snr=30 (bright)")
    print()
    summaries = {}
    for algo in ("haar", "cwt", "mepsa"):
        print(f"--- {algo} ---")
        rows = study(algo, rise_values)
        write_csv(rows, OUT_DIR / f"recovery_{algo}.csv")
        plot_recovery(algo, rows, OUT_DIR / f"recovery_{algo}.pdf")
        summaries[algo] = summarise(rows)
        print(f"  -> {summaries[algo]}\n")

    md = OUT_DIR / "RECOVERY_REPORT.md"
    with open(md, "w") as f:
        f.write("# Synthetic Recovery Study\n\n")
        f.write("FRED pulses with known rise / decay / FWHM swept across "
                "rise ∈ [5 ms, 10 s].\n\n")
        f.write("Peak SNR per bin = 30 (bright, well-detectable).\n\n")
        f.write("| Algorithm | Expected metric | Geo-mean ratio | log-scatter (dex) | upper-limits | N used |\n")
        f.write("|---|---|---|---|---|---|\n")
        for algo, s in summaries.items():
            if s is None:
                f.write(f"| {algo} | — | — | — | all | 0 |\n")
                continue
            expected_label = "rise" if algo != "mepsa" else "fwhm"
            f.write(f"| {algo} | {expected_label} | {s['geo_mean_ratio']:.2f} | "
                    f"{s['log_scatter_dex']:.2f} | {s['n_upper_limits']} | {s['n_used']} |\n")
        f.write("\n**Interpretation:** an unbiased algorithm has "
                "geometric-mean ratio ≈ 1.0 and log-scatter ≲ 0.15 dex (35%).  "
                "Persistent ratio ≠ 1 indicates a systematic bias in the "
                "algorithm implementation.\n")
    print(f"\nWrote summary to {md}")


if __name__ == "__main__":
    main()
