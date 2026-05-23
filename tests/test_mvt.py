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


@pytest.mark.parametrize("method", ["cwt", "haar", "mepsa"])
def test_ggMVT_dispatch_raises_until_impls_added(fred_gg, method):
    net, err, bins, bkg_rate, _ = fred_gg
    mvt = ggMVT(net, err, bins)
    with pytest.raises(NotImplementedError):
        mvt.calculate(method=method)


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
