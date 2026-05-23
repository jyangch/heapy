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
