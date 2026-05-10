"""Simulation tests for the pgSignal time-rescaling pipeline.

User-requested ad-hoc tests written to verify the BB refactor (drpls
seed -> time-rescaling BB -> polyfit, two-pass with optional iteration).
Each TEST block prints ground truth alongside recovered values so the
user can visually inspect accuracy and the failure modes flagged in the
methodology discussion. Diagnostic PDFs from every pgSignal instance
are written under ``test/figures/<label>/``. Safe to delete when no
longer needed.

Run:
    python test/test_pgsignal_simulation.py
"""

import os

import numpy as np
from astropy.stats import bayesian_blocks

from heapy.auto.signal import pgSignal

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


def make_grb(bins, bkg_func, sig_func, rng):
    """Synthesize photon arrival times from given bkg(t) + sig(t) rate funcs."""

    times = (bins[:-1] + bins[1:]) / 2
    binsize = bins[1:] - bins[:-1]
    rate = bkg_func(times) + sig_func(times)
    counts = rng.poisson(rate * binsize)
    chunks = [rng.uniform(bins[i], bins[i + 1], n) for i, n in enumerate(counts) if n > 0]
    return np.concatenate(chunks) if chunks else np.array([])


def signal_window(intervals):
    """Return (min_low, max_high) over a list of [low, high] intervals."""

    if not intervals:
        return None, None
    return min(p[0] for p in intervals), max(p[1] for p in intervals)


def dump_plots(sig, label):
    """Save signal.pdf and signal_tau.pdf for ``sig`` under FIGURES_DIR/label/."""

    out = os.path.join(FIGURES_DIR, label)
    sig.save(out)
    if sig.poly_res is not None:
        sig.save_tau_diagnostic(out)


def main():
    rng = np.random.default_rng(2026)
    bins = np.arange(-50.0, 100.0 + 0.064, 0.064)

    # ------------------------------------------------------------------
    print('=' * 70)
    print('TEST A: Top-hat pulse on quadratic background -- T_start/T_end recovery')
    print('=' * 70)

    true_t_start, true_t_end = 10.0, 30.0
    bkg = lambda t: 1500.0 + 1.5 * t - 0.005 * t ** 2
    sig = lambda t: 500.0 * ((t >= true_t_start) & (t < true_t_end))

    ts_a = make_grb(bins, bkg, sig, rng)
    print(f'  N photons       : {len(ts_a):,}')
    print(f'  true pulse      : [{true_t_start}, {true_t_end}]  width={true_t_end - true_t_start}s')
    print(f'  bkg drift over window: {bkg(-50):.0f} -> {bkg(100):.0f} cts/s '
          f'({(bkg(100) / bkg(-50) - 1) * 100:+.1f}%)')

    s_a = pgSignal(ts_a, bins)
    s_a.loop(p0=0.05, sigma=3)
    dump_plots(s_a, 'A_default')
    new_lo, new_hi = signal_window(s_a.sort_res['re_sig'][0])
    print(f'\n  new pipeline    : T_start={new_lo:6.3f}, T_end={new_hi:6.3f}')
    print(f'    error         : dT_start={new_lo - true_t_start:+.3f}, '
          f'dT_end={new_hi - true_t_end:+.3f}')
    print(f'    poly deg picked: {s_a.poly.best_deg}  (truth=2)')
    print(f'    blocks total   : {len(s_a.edges) - 1}, '
          f'signal blocks: {len(s_a.sort_res["re_sig"][0])}')

    pos = np.where(s_a.cts > 0)[0]
    old_edges = bayesian_blocks(s_a.time[pos], s_a.cts[pos], fitness='events', p0=0.05)
    print(f'\n  old (events,no bkg): {len(old_edges) - 1} blocks total')
    in_pulse = sum(1 for e in old_edges[1:-1] if true_t_start - 1 < e < true_t_end + 1)
    out_pulse = (len(old_edges) - 2) - in_pulse
    print(f'    edges near true pulse [{true_t_start - 1},{true_t_end + 1}]: {in_pulse}')
    print(f'    edges elsewhere (likely spurious from bkg drift): {out_pulse}')

    # ------------------------------------------------------------------
    print()
    print('=' * 70)
    print('TEST B: Two well-separated pulses')
    print('=' * 70)

    bkg2 = lambda t: 1200.0 + 0.8 * t
    sig2 = lambda t: 600.0 * (((t >= -15) & (t < -5)) | ((t >= 40) & (t < 55)))

    ts_b = make_grb(bins, bkg2, sig2, rng)
    s_b = pgSignal(ts_b, bins)
    s_b.loop(p0=0.05, sigma=3)
    dump_plots(s_b, 'B_multipulse')
    print(f'  N photons       : {len(ts_b):,}')
    print(f'  true pulses     : [-15, -5] and [40, 55]')
    print(f'  recovered signal blocks:')
    for low, high in s_b.sort_res['re_sig'][0]:
        print(f'    [{low:6.3f}, {high:6.3f}]')

    # ------------------------------------------------------------------
    print()
    print('=' * 70)
    print('TEST C: Multi-detector via from_components')
    print('=' * 70)

    weights = [0.9, 0.6, 0.3]
    bkg_amps = [1300, 1700, 2100]

    components = []
    for i, (w, b) in enumerate(zip(weights, bkg_amps, strict=False)):
        bkg_i = lambda t, b=b: b + 1.0 * t - 0.003 * t ** 2
        sig_i = lambda t, w=w: w * 800.0 * np.exp(-0.5 * ((t - 20.0) / 5.0) ** 2)
        ts_i = make_grb(bins, bkg_i, sig_i, rng)
        p = pgSignal(ts_i, bins)
        p.loop(p0=0.05, sigma=3)
        dump_plots(p, f'C_det{i}')
        components.append(p)
        cl, ch = signal_window(p.sort_res['re_sig'][0])
        cl_str = f'{cl:.3f}' if cl is not None else '--'
        ch_str = f'{ch:.3f}' if ch is not None else '--'
        print(f'  det {i} (w={w}, bkg~{b}): N={len(ts_i):,}, sig=[{cl_str}, {ch_str}]')

    combo = pgSignal.from_components(components)
    combo.loop(p0=0.05, sigma=3)
    dump_plots(combo, 'C_composite')
    cl, ch = signal_window(combo.sort_res['re_sig'][0])
    print(f'\n  composite        : N={len(combo.ts):,}, sig=[{cl:.3f}, {ch:.3f}]')
    print(f'  composite blocks : {len(combo.edges) - 1} total, '
          f'{len(combo.sort_res["re_sig"][0])} signal')

    # ------------------------------------------------------------------
    print()
    print('=' * 70)
    print('TEST D: iter_polyfit convergence and consistency on TEST A data')
    print('=' * 70)

    s_default = pgSignal(ts_a, bins)
    s_default.loop(p0=0.05, sigma=3)
    dump_plots(s_default, 'D_default')

    s_iter = pgSignal(ts_a, bins)
    s_iter.loop(p0=0.05, sigma=3, iter=True)
    dump_plots(s_iter, 'D_iter')

    print(f'  default 2-pass : ignore = {s_default.sort_res["ignore"]}')
    print(f'  iter=True      : ignore = {s_iter.sort_res["ignore"]}')
    print(f'  identical?     : {s_default.sort_res["ignore"] == s_iter.sort_res["ignore"]}')
    print(f'  poly deg same? : default={s_default.poly.best_deg}, iter={s_iter.poly.best_deg}')

    # ------------------------------------------------------------------
    print()
    print('=' * 70)
    print('TEST E: tau-space dN/dtau in known bkg vs signal windows (TEST A data)')
    print('=' * 70)

    tau_at_bins = s_a.poly.integral(s_a.bins)[0]
    tau_at_ts = np.interp(np.sort(s_a.ts), s_a.bins, tau_at_bins)

    def tau_rate(t_lo, t_hi):
        tau_lo = np.interp(t_lo, s_a.bins, tau_at_bins)
        tau_hi = np.interp(t_hi, s_a.bins, tau_at_bins)
        n = ((tau_at_ts > tau_lo) & (tau_at_ts < tau_hi)).sum()
        return n / (tau_hi - tau_lo)

    bkg_lo = tau_rate(-40, -20)
    bkg_hi = tau_rate(50, 80)
    pulse_rate = tau_rate(10, 30)
    print(f'  dN/dtau in bkg-only [-40,-20]   : {bkg_lo:.4f}  (expect ~1.000)')
    print(f'  dN/dtau in bkg-only [50, 80]    : {bkg_hi:.4f}  (expect ~1.000)')
    print(f'  dN/dtau inside pulse [10, 30]   : {pulse_rate:.4f}  (expect > 1)')
    print(f'  ratio (pulse / bkg)             : {pulse_rate / bkg_lo:.3f}  '
          f'(truth = 1 + sig_amp/bkg ~ {1 + 500 / bkg(20):.3f})')

    # ------------------------------------------------------------------
    print()
    print('=' * 70)
    print('TEST F: extreme cubic background slope (stress test) -- default')
    print('=' * 70)

    bkg3 = lambda t: 1000.0 + 5.0 * t + 0.05 * t ** 2 - 0.0005 * t ** 3
    sig3 = lambda t: 400.0 * ((t >= 10) & (t < 25))
    ts_f = make_grb(bins, bkg3, sig3, rng)
    print(f'  bkg(-50)={bkg3(-50):.0f}, bkg(0)={bkg3(0):.0f}, bkg(100)={bkg3(100):.0f} cts/s')
    print(f'  drift: {(bkg3(100) / bkg3(-50) - 1) * 100:+.1f}%, true pulse [10, 25]')

    s_f = pgSignal(ts_f, bins)
    s_f.loop(p0=0.05, sigma=3)
    dump_plots(s_f, 'F_default')
    cl, ch = signal_window(s_f.sort_res['re_sig'][0])
    print(f'\n  default 2-pass:')
    print(f'    T_start={cl:.3f}, T_end={ch:.3f} (truth: [10, 25])')
    print(f'    poly deg picked: {s_f.poly.best_deg}  (truth=3)')
    print(f'    signal blocks  : {len(s_f.sort_res["re_sig"][0])}')
    for low, high in s_f.sort_res['re_sig'][0]:
        print(f'      [{low:7.3f}, {high:7.3f}]')

    pos3 = np.where(s_f.cts > 0)[0]
    old_edges3 = bayesian_blocks(s_f.time[pos3], s_f.cts[pos3], fitness='events', p0=0.05)
    in_pulse3 = sum(1 for e in old_edges3[1:-1] if 9 < e < 26)
    out_pulse3 = (len(old_edges3) - 2) - in_pulse3
    print(f'\n  old events-BB : {len(old_edges3) - 1} blocks, '
          f'{in_pulse3} edges in pulse, {out_pulse3} elsewhere')

    # ------------------------------------------------------------------
    print()
    print('=' * 70)
    print('TEST G: extreme cubic bkg -- can iter_polyfit / manual ignore recover?')
    print('=' * 70)

    # G.1 iter on the same data as TEST F
    s_g_iter = pgSignal(ts_f, bins)
    s_g_iter.loop(p0=0.05, sigma=3, iter=True)
    dump_plots(s_g_iter, 'G1_iter')
    cl, ch = signal_window(s_g_iter.sort_res['re_sig'][0])
    print(f'  G.1 iter=True:')
    print(f'    T_start={cl:.3f}, T_end={ch:.3f} (truth: [10, 25])')
    print(f'    poly deg picked: {s_g_iter.poly.best_deg}')
    print(f'    signal blocks  : {len(s_g_iter.sort_res["re_sig"][0])}')
    for low, high in s_g_iter.sort_res['re_sig'][0]:
        print(f'      [{low:7.3f}, {high:7.3f}]')
    print(f'    ignore final   : {s_g_iter.sort_res["ignore"]}')

    # G.2 Manual ignore as escape hatch (truth-aware, simulating user knowledge)
    s_g_manual = pgSignal(ts_f, bins, ignore=[[10.0, 25.0]])
    s_g_manual.loop(p0=0.05, sigma=3)
    dump_plots(s_g_manual, 'G2_manual_tight')
    cl, ch = signal_window(s_g_manual.sort_res['re_sig'][0])
    print(f'\n  G.2 manual ignore=[[10, 25]] (escape hatch):')
    cl_str = f'{cl:.3f}' if cl is not None else '--'
    ch_str = f'{ch:.3f}' if ch is not None else '--'
    print(f'    T_start={cl_str}, T_end={ch_str} (truth: [10, 25])')
    print(f'    poly deg picked: {s_g_manual.poly.best_deg}')
    print(f'    signal blocks  : {len(s_g_manual.sort_res["re_sig"][0])}')
    for low, high in s_g_manual.sort_res['re_sig'][0]:
        print(f'      [{low:7.3f}, {high:7.3f}]')

    # G.3 Slightly looser ignore (over-bracketing the pulse)
    s_g_loose = pgSignal(ts_f, bins, ignore=[[5.0, 30.0]])
    s_g_loose.loop(p0=0.05, sigma=3)
    dump_plots(s_g_loose, 'G3_manual_loose')
    cl, ch = signal_window(s_g_loose.sort_res['re_sig'][0])
    cl_str = f'{cl:.3f}' if cl is not None else '--'
    ch_str = f'{ch:.3f}' if ch is not None else '--'
    print(f'\n  G.3 manual ignore=[[5, 30]] (over-bracketed):')
    print(f'    T_start={cl_str}, T_end={ch_str} (truth: [10, 25])')
    print(f'    signal blocks  : {len(s_g_loose.sort_res["re_sig"][0])}')


if __name__ == '__main__':
    main()
