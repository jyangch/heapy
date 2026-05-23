"""Tests for heapy.temp.mvt."""
import numpy as np
import pytest

from heapy.temp.mvt import ggMVT, pgMVT, ppMVT, _MVTResult


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
