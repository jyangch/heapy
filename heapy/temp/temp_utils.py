"""Shared helpers and plotter for the ``txx`` module's pipeline classes.

Bundles two groups of utilities consumed by :mod:`heapy.temp.txx`:

- Cumulative-count-fraction math (:func:`accumcts`, :func:`find_txx`)
  that derives Txx start/stop times from a pulse's cumulative net
  count curve. Both functions are stateless and reusable outside the
  Txx classes.
- :class:`TxxPlotter`, a composable two-panel diagnostic figure that
  every ``save()`` method in :mod:`heapy.temp.txx` shares (mirrors the
  :class:`~heapy.auto.signal_utils.SignalPlotter` design).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from ..auto.signal_utils import indices_in_intervals
from ..util.data import generate_asymmetric_gaussian


def accumcts(time, ccts, pstart, pstop, xx, simple_err=False, random_seed=450001):
    """Compute cumulative-count-fraction levels and Txx start/stop times.

    Interpolates the cumulative count curve to 1000-point resolution when
    the native sampling is too coarse, calculates the background CSF levels
    between pulses, derives the :math:`(1-xx)/2` and :math:`1-(1-xx)/2`
    fraction levels (``csf1``, ``csf2``) for each pulse interval, and
    locates the corresponding times via :func:`find_txx`.

    Args:
        time: 1-D time array covering the analysis window.
        ccts: Cumulative net count array aligned with ``time``.
        pstart: Array of pulse start times; one entry per pulse.
        pstop: Array of pulse stop times; one entry per pulse.
        xx: Cumulative count fraction, e.g. ``0.9`` for T90.
        simple_err: When ``True``, also compute and return analytic
            uncertainty estimates on all CSF and Txx values.
        random_seed: Seed for the local RNG used by the asymmetric
            Gaussian sampler in the ``simple_err`` branch. Default
            ensures reproducibility; ignored when ``simple_err=False``.

    Returns:
        When ``simple_err`` is ``False``: a tuple
        ``(txx, txx1, txx2, csf, csf1, csf2)`` where each element is a list
        with one entry per pulse.

        When ``simple_err`` is ``True``: a tuple
        ``(txx, txx1, txx2, txx_err, txx1_err, txx2_err,
        csf, csf1, csf2, csf_err, csf1_err, csf2_err)``.
    """

    rng = np.random.default_rng(random_seed)

    idx = np.argsort(pstop - pstart)[0]
    if len(np.where((time >= pstart[idx]) & (time <= pstop[idx]))[0]) < 1000:
        interp_dt = (pstop[idx] - pstart[idx]) / 1000
        interp_time = np.arange(time[0], time[-1] - 1e-5, interp_dt)
        interp = interp1d(time, ccts, kind='quadratic')
        interp_ccts = interp(interp_time)
    else:
        interp_time = time
        interp_ccts = ccts

    csf, csf1, csf2 = [], [], []
    txx, txx1, txx2 = [], [], []

    if simple_err:
        csf_err, csf1_err, csf2_err = [], [], []
        txx_err, txx1_err, txx2_err = [], [], []

    for left, right in zip(np.append(time[0], pstop), np.append(pstart, time[-1]), strict=False):
        idx = np.where((interp_time >= left) & (interp_time <= right))[0]
        if len(idx) >= 1:
            csf_i = np.mean(interp_ccts[idx])
        else:
            csf_i = interp_ccts[np.argmin(np.abs(interp_time - left))]
        csf.append(csf_i)

        if simple_err:
            csf_err_i = np.std(interp_ccts[idx]) if len(idx) >= 1 else 0.0
            csf_err.append(csf_err_i)

    dcsf = np.array(csf[1:]) - np.array(csf[:-1])

    for pi, (left, right) in enumerate(
        zip(np.append(time[0], pstop[:-1]), np.append(pstart[1:], time[-1]), strict=False)
    ):
        nn = (1 - xx) / 2
        dd = dcsf[pi] * nn

        csf1_i = csf[pi] + dd
        csf2_i = csf[pi + 1] - dd

        csf1.append(csf1_i)
        csf2.append(csf2_i)

        if simple_err:
            csf1_err_i = np.sqrt((1 - nn) ** 2 * csf_err[pi] ** 2 + nn**2 * csf_err[pi + 1] ** 2)
            csf2_err_i = np.sqrt(nn**2 * csf_err[pi] ** 2 + (1 - nn) ** 2 * csf_err[pi + 1] ** 2)

            csf1_err.append(csf1_err_i)
            csf2_err.append(csf2_err_i)

        pt = interp_time[np.where((interp_time >= left) & (interp_time <= right))]
        pcts = interp_ccts[np.where((interp_time >= left) & (interp_time <= right))]

        txx_i, txx1_i, txx2_i = find_txx(pt, pcts, csf1_i, csf2_i)

        txx.append(txx_i)
        txx1.append(txx1_i)
        txx2.append(txx2_i)

        if simple_err:
            _, txx1_lo_i, txx2_lo_i = find_txx(pt, pcts, csf1_i - csf1_err_i, csf2_i - csf2_err_i)
            _, txx1_hi_i, txx2_hi_i = find_txx(pt, pcts, csf1_i + csf1_err_i, csf2_i + csf2_err_i)
            txx1_le_i, txx1_he_i = txx1_i - txx1_lo_i, txx1_hi_i - txx1_i
            txx2_le_i, txx2_he_i = txx2_i - txx2_lo_i, txx2_hi_i - txx2_i

            txx1_i_sam = generate_asymmetric_gaussian(txx1_i, txx1_le_i, txx1_he_i, 1000, rng=rng)
            txx2_i_sam = generate_asymmetric_gaussian(txx2_i, txx2_le_i, txx2_he_i, 1000, rng=rng)
            txx_lo_i, txx_hi_i = np.percentile(txx2_i_sam - txx1_i_sam, [16, 84])
            txx_le_i, txx_he_i = txx_i - txx_lo_i, txx_hi_i - txx_i

            txx_err.append([txx_le_i, txx_he_i])
            txx1_err.append([txx1_le_i, txx1_he_i])
            txx2_err.append([txx2_le_i, txx2_he_i])

    if simple_err:
        return (
            txx,
            txx1,
            txx2,
            txx_err,
            txx1_err,
            txx2_err,
            csf,
            csf1,
            csf2,
            csf_err,
            csf1_err,
            csf2_err,
        )
    else:
        return txx, txx1, txx2, csf, csf1, csf2


def find_txx(time, ccts, csf1, csf2):
    """Locate the start and stop times corresponding to CSF thresholds.

    Linearly interpolates the cumulative count curve onto a 1000-point grid,
    then scans forward to find the time ``txx1`` at which the curve crosses
    ``csf1`` from below, and ``txx2`` at which it crosses ``csf2`` from
    below.  The duration ``txx = txx2 - txx1`` is also returned.

    Args:
        time: 1-D time array for the pulse interval.
        ccts: Cumulative net count array aligned with ``time``.
        csf1: Lower cumulative-count-fraction level (start threshold).
        csf2: Upper cumulative-count-fraction level (stop threshold).

    Returns:
        A tuple ``(txx, txx1, txx2)`` where ``txx`` is the duration and
        ``txx1``, ``txx2`` are the start and stop times, respectively.
        All three values are ``0`` if no valid crossing is found.
    """

    interp_time = np.linspace(time[0], time[-1], 1000)
    interp = interp1d(time, ccts, kind='linear')
    interp_ccts = interp(interp_time)

    txx1, txx2 = 0, 0
    for i in range(1, len(interp_time)):
        if interp_ccts[i] < csf1:
            continue
        elif (interp_ccts[i - 1] < csf1) and (interp_ccts[i] >= csf1):
            txx1 = interp_time[i]
            continue
        elif (csf1 <= interp_ccts[i - 1] < csf2) and (csf1 < interp_ccts[i] <= csf2):
            continue
        elif (csf1 < interp_ccts[i - 1] <= csf2) and (interp_ccts[i] > csf2):
            txx2 = interp_time[i - 1]
            break
        else:
            continue

    txx = txx2 - txx1
    return txx, txx1, txx2


class TxxPlotter:
    """Composable two-panel diagnostic figure for Txx duration analysis.

    Top panel (:attr:`ax_top`): primary light-curve trace with optional
    background overlay and vertical Txx start/stop markers. Bottom
    panel (:attr:`ax_bot`): cumulative net counts -- the full series in
    gray and the analysis-window slice in black -- with horizontal CSF
    level lines and the same Txx vertical markers overlaid. Compose by
    calling :meth:`plot_curve`, :meth:`plot_ccts`, and :meth:`plot_txx`
    in any order, then :meth:`save` or :meth:`show`. Mirrors the
    :class:`~heapy.auto.signal_utils.SignalPlotter` API style.

    Attributes:
        fig: Underlying matplotlib Figure.
        ax_top: Top-panel Axes (light curve + background).
        ax_bot: Bottom-panel Axes (cumulative counts).
    """

    def __init__(self, figsize=(7, 6)):
        """Create an empty two-panel figure with shared x-axis.

        Args:
            figsize: Width and height of the figure in inches.
        """

        self.fig = plt.figure(figsize=figsize)
        gs = self.fig.add_gridspec(2, 1, wspace=0, hspace=0)
        self.ax_top = self.fig.add_subplot(gs[:1, 0])
        self.ax_bot = self.fig.add_subplot(gs[1:, 0], sharex=self.ax_top)
        self._style_axis(self.ax_top)
        self._style_axis(self.ax_bot)
        plt.setp(self.ax_top.get_xticklabels(), visible=False)
        self.ax_top.set_ylabel('Rate')
        self.ax_bot.set_xlabel('Time')
        self.ax_bot.set_ylabel('Accumulated counts')

        self._gaps = None
        self._bin_lbins = None
        self._bin_rbins = None

    @staticmethod
    def _style_axis(ax):

        ax.minorticks_on()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
        ax.tick_params(which='major', width=1.0, length=5)
        ax.tick_params(which='minor', width=1.0, length=3)

    def set_gaps(self, gap_int, lbins, rbins):
        """Register missing-data intervals to mask in :meth:`plot_ccts`.

        Cumulative counts at gap-bin positions are replaced with
        ``NaN`` so the curve renders as a break rather than a spurious
        plateau (gap-bin ``ncts`` is treated as zero contribution by
        ``np.nancumsum`` in the caller, leaving the running total
        finite at those positions until this mask is applied).

        Args:
            gap_int: List of ``[low, high]`` intervals; empty or
                falsy disables masking.
            lbins: Per-bin left edges aligned with the ``ccts`` array
                later passed to :meth:`plot_ccts`.
            rbins: Per-bin right edges (same length as ``lbins``).
        """

        self._gaps = gap_int
        self._bin_lbins = lbins
        self._bin_rbins = rbins

    def plot_curve(self, time, primary, bak=None):
        """Draw the primary light curve and optional background on the top panel.

        Args:
            time: Per-bin time grid.
            primary: Primary rate (or net rate) array plotted in black
                as the light curve.
            bak: Optional background rate plotted in red. ``None``
                skips the background line (used by gg-flavoured Txx
                where the input is already net).
        """

        self.ax_top.plot(time, primary, color='k', lw=1.0, label='Light Curve')
        if bak is not None:
            self.ax_top.plot(time, bak, color='r', lw=1.0, label='Background')
        self.ax_top.set_xlim([time[0], time[-1]])
        self.ax_top.legend(frameon=False)

    def plot_ccts(self, time, ccts, tindex):
        """Draw cumulative counts on the bottom panel.

        Full ``ccts`` is rendered in gray; the analysis-window slice
        ``ccts[tindex]`` is overlaid in black. When :meth:`set_gaps`
        has been called, gap-bin positions are masked to ``NaN`` so
        matplotlib breaks the line there.

        Args:
            time: Per-bin time grid aligned with ``ccts``.
            ccts: Cumulative net count array.
            tindex: Integer index array selecting the analysis window.
        """

        ccts = np.asarray(ccts, dtype=float).copy()
        if self._gaps:
            idx = indices_in_intervals(self._bin_lbins, self._bin_rbins, self._gaps)
            ccts[idx] = np.nan
        self.ax_bot.plot(time, ccts, color='gray', lw=1.0)
        self.ax_bot.plot(time[tindex], ccts[tindex], color='k', lw=1.0)

    def plot_txx(self, txx1, txx2, csf, csf1, csf2):
        """Overlay Txx vertical markers and CSF horizontal lines.

        Args:
            txx1: Pulse start times (one per pulse); drawn as dashed
                green vertical lines on both panels.
            txx2: Pulse stop times (one per pulse); drawn the same way.
            csf: Background CSF levels (one per quiescent segment);
                drawn as solid orange horizontal lines on the bottom
                panel.
            csf1: Lower CSF thresholds (one per pulse); drawn as dashed
                orange horizontal lines on the bottom panel.
            csf2: Upper CSF thresholds; drawn the same way as ``csf1``.
        """

        for t1, t2 in zip(txx1, txx2, strict=False):
            self.ax_top.axvline(t1, color='g', lw=1.0, ls='--')
            self.ax_top.axvline(t2, color='g', lw=1.0, ls='--')
            self.ax_bot.axvline(t1, color='g', lw=1.0, ls='--')
            self.ax_bot.axvline(t2, color='g', lw=1.0, ls='--')
        for c in csf:
            self.ax_bot.axhline(c, color='orange', lw=1.0)
        for c1, c2 in zip(csf1, csf2, strict=False):
            self.ax_bot.axhline(c1, color='orange', lw=1.0, ls='--')
            self.ax_bot.axhline(c2, color='orange', lw=1.0, ls='--')

    def show(self):
        """Display the figure interactively."""

        plt.tight_layout()
        plt.show()

    def save(self, filename, dpi=300):
        """Save the figure to ``filename`` and close it.

        Args:
            filename: Output file path; format inferred from extension.
            dpi: Resolution in dots per inch.
        """

        self.fig.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=dpi)
        plt.close(self.fig)


class MVTPlotter:
    """Two-panel diagnostic for :mod:`heapy.temp.mvt`.

    Top panel: net light curve with the MVT value annotated.
    Bottom panel: method-specific scaleogram or wavelet power spectrum.
    """

    def __init__(self):
        self.fig, (self.ax_lc, self.ax_spec) = plt.subplots(
            nrows=2, figsize=(7, 7), constrained_layout=True,
        )

    def plot_lc(self, time, net_rate, mvt, is_upper_limit):
        self.ax_lc.step(time, net_rate, where="mid", lw=0.8)
        self.ax_lc.set_xlabel("Time (s)")
        self.ax_lc.set_ylabel("Net rate")
        label = (r"MVT $\leq$ " if is_upper_limit else "MVT = ") + f"{mvt:.3g} s"
        self.ax_lc.set_title(label)

    def plot_diag(self, method, diag):
        if method == "cwt":
            p = np.array(diag.get("periods", []))
            ws = np.array(diag.get("ws", []))
            up = np.array(diag.get("bkg_upper", []))
            med = np.array(diag.get("bkg_median", []))
            if p.size:
                ws_plot = np.where(ws > 0, ws, np.nan)
                up_plot = np.where(up > 0, up, np.nan)
                med_plot = np.where(med > 0, med, np.nan)
                self.ax_spec.plot(p, ws_plot, "o", ms=4, label="observed")
                self.ax_spec.plot(p, med_plot, "--k", label="bkg median")
                self.ax_spec.plot(p, up_plot, ":", label="upper")
                self.ax_spec.set_xscale("log")
                self.ax_spec.set_yscale("log")
            self.ax_spec.set_xlabel(r"$\delta t$ (s)")
            self.ax_spec.set_ylabel("Rectified W")
        elif method == "haar":
            dt = np.array(diag.get("delta_t", []))
            s2 = np.array(diag.get("sigma2", []))
            err = np.array(diag.get("sigma2_err", []))
            slope = diag.get("smooth_slope")
            if dt.size:
                mask = s2 > 0
                self.ax_spec.errorbar(
                    dt[mask], np.sqrt(s2[mask]),
                    yerr=0.5 * err[mask] / np.sqrt(np.maximum(s2[mask], 1e-30)),
                    fmt="o", ms=4,
                )
                if slope:
                    self.ax_spec.plot(dt, slope * dt, "r--",
                                       label=r"$\sigma \propto \Delta t$")
            self.ax_spec.set_xscale("log")
            self.ax_spec.set_yscale("log")
            self.ax_spec.set_xlabel(r"$\Delta t$ (s)")
            self.ax_spec.set_ylabel(r"$\sigma_{X,\Delta t}$")
        elif method == "mepsa":
            peaks = diag.get("peaks", [])
            if peaks:
                dt = np.array([p["dt_det"] for p in peaks])
                snr = np.array([p["snr"] for p in peaks])
                self.ax_spec.semilogx(dt, snr, "o")
                self.ax_spec.set_xlabel(r"$\Delta t_{\rm det}$ (s)")
                self.ax_spec.set_ylabel("Peak SNR")
        handles, labels = self.ax_spec.get_legend_handles_labels()
        if handles:
            self.ax_spec.legend(loc="best", fontsize=8)

    def save(self, filename):
        self.fig.savefig(filename)
        plt.close(self.fig)
