"""Tests for heapy.temp.mvt."""
import numpy as np
import pytest

from heapy.temp.mvt import ggMVT, pgMVT, ppMVT, _MVTResult
from heapy.temp.mvt import _mepsa_scan  # noqa: E402


def test_ggMVT_constructs_from_ncts(fred_gg):
    net, err, bins, bkg_rate, _ = fred_gg
    mvt = ggMVT(net, err, bins)
    assert mvt.mvt_res is None
    assert mvt.ncts is net or np.array_equal(mvt.ncts, net)


def test_pgMVT_constructs_from_events(fred_events):
    src, bkg, t1, t2, dt, _ = fred_events
    all_t = np.concatenate([src, bkg])
    bins = np.arange(t1, t2 + dt, dt)
    mvt = pgMVT(all_t, bins, ignore=[-1.0, 5.0])
    assert mvt.mvt_res is None
    assert mvt.poly_res is None  # polyfit not yet run


def test_ppMVT_constructs_from_two_event_lists(fred_events):
    src, bkg, t1, t2, dt, _ = fred_events
    bins = np.arange(t1, t2 + dt, dt)
    mvt = ppMVT(src, bkg, bins, backscale=1.0)
    assert mvt.mvt_res is None
    assert mvt.ncts is not None  # ppSignal computes ncts in __init__


def test_calculate_rejects_unknown_method(fred_gg):
    net, err, bins, bkg_rate, _ = fred_gg
    mvt = ggMVT(net, err, bins)
    with pytest.raises(ValueError, match="method"):
        mvt.calculate(method="bogus")


def test_mepsa_scan_finds_fred_pulse(fred_gg):
    net, err, bins, bkg_rate, t0 = fred_gg
    dt = bins[1] - bins[0]
    peaks = _mepsa_scan(net, dt)
    assert len(peaks) >= 1
    strongest = max(peaks, key=lambda p: p["snr"])
    t = 0.5 * (bins[:-1] + bins[1:])
    assert abs(t[strongest["idx"]] - t0) < strongest["dt_det"]


def test_mepsa_scan_rejects_pure_noise(noise_gg):
    net, err, bins, bkg_rate = noise_gg
    dt = bins[1] - bins[0]
    peaks = _mepsa_scan(net, dt)
    assert len(peaks) == 0


def test_ggMVT_mepsa_recovers_fred_fwhm(fred_gg):
    net, err, bins, bkg_rate, _ = fred_gg
    mvt = ggMVT(net, err, bins)
    mvt.calculate(method="mepsa")
    res = mvt.mvt_res
    assert res.is_upper_limit is False
    # FRED (rise=0.5, decay=1.5, p=1.5) ⇒ FWHM ~ 1.4 s; allow generous tol.
    assert 0.7 < res.mvt < 2.5


def test_ggMVT_mepsa_upper_limit_on_noise(noise_gg):
    net, err, bins, bkg_rate = noise_gg
    mvt = ggMVT(net, err, bins)
    mvt.calculate(method="mepsa")
    assert mvt.mvt_res.is_upper_limit is True


def test_ggMVT_cwt_recovers_fred_rise_time(fred_gg):
    pytest.importorskip("pycwt")
    net, err, bins, bkg_rate, _ = fred_gg
    mvt = ggMVT(net, err, bins)
    mvt.calculate(method="cwt", n_sim=200)
    res = mvt.mvt_res
    assert res.is_upper_limit is False
    # FRED rise = 0.5 s; CWT should hit something between ~0.05 s and ~1.5 s.
    assert 0.05 < res.mvt < 1.5


def test_ggMVT_cwt_upper_limit_on_noise(noise_gg):
    pytest.importorskip("pycwt")
    net, err, bins, bkg_rate = noise_gg
    mvt = ggMVT(net, err, bins)
    mvt.calculate(method="cwt", n_sim=200)
    assert mvt.mvt_res.is_upper_limit is True


def test_ppMVT_cwt_uses_poisson_mc(fred_events):
    pytest.importorskip("pycwt")
    src, bkg, t1, t2, dt, _ = fred_events
    bins = np.arange(t1, t2 + dt, dt)
    mvt = ppMVT(src, bkg, bins, backscale=1.0)
    mvt.calculate(method="cwt", n_sim=200)
    assert mvt.mvt_res.is_upper_limit is False
    # Confirms _run_cwt accepted noise_model='poisson' path without crashing.


def test_ggMVT_haar_recovers_fred_rise_time(fred_gg):
    net, err, bins, bkg_rate, _ = fred_gg
    mvt = ggMVT(net, err, bins)
    mvt.calculate(method="haar")
    res = mvt.mvt_res
    assert res.is_upper_limit is False
    # FRED rise=0.5s, Haar's intrinsic scale ~0.13×FWHM (synthetic-recovery).
    # With the smallest-3-significant-points smooth-norm fit, departure
    # detection is at smaller Δt than with the wider fit windows.
    assert 0.01 < res.mvt < 3.0
    # delta_t should differ across scales now; mvt_err_hi must be > 0.
    assert res.mvt_err_hi > 0.0


def test_ggMVT_haar_upper_limit_on_noise(noise_gg):
    """Haar declares an upper limit on pure noise -- the SNR-rebin step yields
    too few composite bins to compute a structure function."""
    net, err, bins, bkg_rate = noise_gg
    mvt = ggMVT(net, err, bins)
    mvt.calculate(method="haar")
    assert mvt.mvt_res.is_upper_limit is True


def test_ggMVT_save_writes_json_and_pdf(tmp_path, fred_gg):
    net, err, bins, bkg_rate, _ = fred_gg
    mvt = ggMVT(net, err, bins)
    mvt.calculate(method="haar")
    mvt.save(str(tmp_path))
    assert (tmp_path / "mvt_res.json").exists()
    assert (tmp_path / "mvt.pdf").exists()


def test_ggMVT_save_before_calculate_raises(tmp_path, fred_gg):
    net, err, bins, bkg_rate, _ = fred_gg
    mvt = ggMVT(net, err, bins)
    with pytest.raises(RuntimeError, match="calculate"):
        mvt.save(str(tmp_path))


@pytest.mark.parametrize("method", ["cwt", "haar", "mepsa"])
def test_ggMVT_save_works_for_all_methods(tmp_path, fred_gg, method):
    pytest.importorskip("pycwt")  # gates the cwt path; haar/mepsa pass through
    net, err, bins, bkg_rate, _ = fred_gg
    mvt = ggMVT(net, err, bins)
    kw = {"n_sim": 100} if method == "cwt" else {}
    mvt.calculate(method=method, **kw)
    mvt.save(str(tmp_path / method))
    assert (tmp_path / method / "mvt_res.json").exists()
    assert (tmp_path / method / "mvt.pdf").exists()


def test_pgMVT_recovers_fred_via_haar(fred_events):
    src, bkg, t1, t2, dt, _ = fred_events
    all_t = np.concatenate([src, bkg])
    bins = np.arange(t1, t2 + dt, dt)
    mvt = pgMVT(all_t, bins, ignore=[-2.0, 5.0])
    mvt.calculate(method="haar")
    assert mvt.poly_res is not None
    assert mvt.mvt_res.is_upper_limit is False
    assert 0.05 < mvt.mvt_res.mvt < 2.0


def test_ppMVT_recovers_fred_via_mepsa(fred_events):
    src, bkg, t1, t2, dt, _ = fred_events
    bins = np.arange(t1, t2 + dt, dt)
    mvt = ppMVT(src, bkg, bins, backscale=1.0)
    mvt.calculate(method="mepsa")
    assert mvt.mvt_res.is_upper_limit is False
    assert mvt.mvt_res.mvt > 0


def test_ggMVT_cwt_uses_gaussian_mc(fred_gg):
    pytest.importorskip("pycwt")
    net, err, bins, bkg_rate, _ = fred_gg
    mvt = ggMVT(net, err, bins)
    mvt.calculate(method="cwt", n_sim=100)
    assert mvt.mvt_res.diag["noise_model"] == "gaussian"


def test_pgMVT_cwt_uses_poisson_mc(fred_events):
    pytest.importorskip("pycwt")
    src, bkg, t1, t2, dt, _ = fred_events
    all_t = np.concatenate([src, bkg])
    bins = np.arange(t1, t2 + dt, dt)
    mvt = pgMVT(all_t, bins, ignore=[-2.0, 5.0])
    mvt.calculate(method="cwt", n_sim=100)
    assert mvt.mvt_res.diag["noise_model"] == "poisson"


def test_haar_recovers_narrow_pulse_on_sparse_baseline(rng):
    """Haar SF should resolve a narrow bright pulse buried in a broad faint envelope.

    Construct a light curve mimicking the 120119A failure mode: a broad,
    low-SNR envelope (FWHM ~10 s, ~0.2 cts/bin) plus a narrow bright pulse
    (FWHM ~50 ms, peak 50 cts/bin) at the same time.  After SNR-rebin the
    envelope produces many *medium*-width composite bins (~100 ms) while
    the narrow peak keeps fine 1 ms bins — the OLD median-Δt collapse picks
    the medium width and reports MVT >= 1 s.  With per-position binning,
    the on-pulse positions contribute to fine-Δt bins and recover the pulse.
    """
    dt = 0.001
    bins = np.arange(-50.0, 50.0 + dt, dt)
    t = 0.5 * (bins[:-1] + bins[1:])
    bkg_rate = 10.0  # flat background, 0.01 cts/bin
    # Broad envelope: FWHM ~12 s, peak 200 cts/s -> ~0.2 cts/bin at 1 ms
    broad = 200.0 * dt * np.exp(-(t ** 2) / (2 * 5.0 ** 2))
    # Narrow Gaussian pulse, FWHM ~50 ms, peak amp 50 cts/bin
    narrow = 50.0 * np.exp(-(t ** 2) / (2 * (0.050 / 2.3548) ** 2))
    profile = narrow + broad
    obs = rng.poisson(profile + bkg_rate * dt).astype(float)
    bkg = bkg_rate * dt * np.ones_like(obs)
    net = obs - bkg
    err = np.sqrt(obs + bkg)
    mvt = ggMVT(net, err, bins)
    mvt.calculate(method="haar")
    res = mvt.mvt_res
    assert res.is_upper_limit is False
    # Should now resolve sub-second variability (old median-Δt gave >1 s)
    assert res.mvt < 1.0, f"Haar MVT {res.mvt:.3g} s should be < 1.0 s for a 50ms pulse"


def test_haar_returns_dense_scaleogram_on_uniform_bins(fred_gg):
    """The refactored Haar should produce >=20 Δt bins even on uniform-bin input."""
    from heapy.temp.mvt import _haar_structure_function, _rebin_to_snr
    net, err, bins, _, _ = fred_gg
    lbins = bins[:-1]; rbins = bins[1:]
    rate, drate, lb, rb = _rebin_to_snr(net, err, lbins, rbins, target_snr=5.0)
    out = _haar_structure_function(rate, drate, lb, rb)
    assert out is not None
    delta_t, sigma2, sigma2_noise, sigma2_err = out
    assert len(delta_t) >= 20, f"got {len(delta_t)} bins, expected >=20"
    # delta_t should be monotonically increasing
    assert (np.diff(delta_t) > 0).all()
    # at least some bins should show significant signal
    assert (sigma2 > 3 * sigma2_err).sum() >= 4


def test_three_methods_ordered_on_fred(fred_gg):
    pytest.importorskip("pycwt")
    net, err, bins, bkg_rate, _ = fred_gg
    results = {}
    for method in ("cwt", "haar", "mepsa"):
        mvt = ggMVT(net, err, bins)
        kw = {"n_sim": 200} if method == "cwt" else {}
        mvt.calculate(method=method, **kw)
        if not mvt.mvt_res.is_upper_limit:
            results[method] = mvt.mvt_res.mvt
    assert set(results) == {"cwt", "haar", "mepsa"}
    assert results["mepsa"] > min(results["cwt"], results["haar"])
