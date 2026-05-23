"""Shared pytest fixtures for heapy.temp.mvt tests."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(450001)


def _fred_profile(t, t0=0.0, rise=0.5, decay=1.5, p=1.5, amp=200.0):
    """FRED (Fast Rise Exponential Decay) shape, identical to Camisasca 2023 eq A.1."""
    left = amp * np.exp(-(np.abs(t - t0) / rise) ** p)
    right = amp * np.exp(-(np.abs(t - t0) / decay) ** p)
    return np.where(t < t0, left, right)


@pytest.fixture
def fred_gg(rng):
    """Bright FRED light curve already reduced to (net cts, err, bins) — feeds ggMVT directly.

    rise=0.5 s, decay=1.5 s, FWHM ~ 1.4 s, peak amp 200 cts/bin at 4 ms binning,
    flat 10 cts/s background subtracted.

    Returns (ncts, ncts_err, bins, bkg_rate, t_peak).
    """
    dt = 0.004
    bins = np.arange(-10.0, 20.0 + dt, dt)
    t = 0.5 * (bins[:-1] + bins[1:])
    bkg_rate = 10.0
    profile = _fred_profile(t)
    obs = rng.poisson(profile + bkg_rate * dt).astype(float)
    bkg = bkg_rate * dt * np.ones_like(obs)
    net = obs - bkg
    err = np.sqrt(obs + bkg)
    return net, err, bins, bkg_rate, 0.0


@pytest.fixture
def noise_gg(rng):
    """Flat-Poisson noise-only light curve as (net, err, bins, bkg_rate)."""
    dt = 0.004
    bins = np.arange(0.0, 60.0 + dt, dt)
    t = 0.5 * (bins[:-1] + bins[1:])
    bkg_rate = 10.0
    expected = bkg_rate * dt * np.ones_like(t)
    obs = rng.poisson(expected).astype(float)
    bkg = expected
    net = obs - bkg
    err = np.sqrt(obs + bkg)
    return net, err, bins, bkg_rate


@pytest.fixture
def fred_events(rng):
    """Raw event arrival-time arrays (source + background) for pp/pgMVT.from_events tests.

    Returns (src_times, bkg_times, t1, t2, dt, t_peak).
    Same FRED+10 cts/s bkg as `fred_gg`, but expressed as event lists.
    """
    t1, t2, dt = -10.0, 20.0, 0.004
    bkg_rate = 10.0
    n_bkg = rng.poisson(bkg_rate * (t2 - t1))
    bkg_times = np.sort(rng.uniform(t1, t2, size=n_bkg))
    # Sample from the FRED profile by inverse-CDF on a fine grid.
    fine = np.linspace(t1, t2, 200_000)
    p = _fred_profile(fine)
    p = np.clip(p, 0, None)
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    n_src = int(p.sum() * (fine[1] - fine[0]) / 1.0)  # integral ~ #counts at amp=200/bin*0.004s
    # Use larger sample for stable recovery.
    n_src = max(n_src, 3000)
    u = rng.uniform(size=n_src)
    src_times = np.interp(u, cdf, fine)
    return src_times, bkg_times, t1, t2, dt, 0.0
