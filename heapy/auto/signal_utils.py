"""Shared helpers and plotter for the ``signal`` module's pipeline classes.

Bundles four groups of utilities consumed by :mod:`heapy.auto.signal`:

- Bin / interval helpers (:func:`indices_in_intervals`,
  :func:`filter_block_edges`, :func:`classify_bins`,
  :func:`intervals_equal`).
- Bayesian-blocks driver :func:`time_rescaling_bblock` that wraps
  ``astropy.stats.bayesian_blocks`` with optional non-stationary
  background correction (Scargle 2013, Appendix B).
- One-bin SNR estimators (:func:`pg_snr`, :func:`pp_snr`,
  :func:`gauss_snr`) wrapping :mod:`heapy.util.significance`.
- :class:`SignalPlotter`, the composable two-panel diagnostic figure
  used by every ``save()`` method in :mod:`heapy.auto.signal`, plus
  :func:`plot_tau_diagnostic`, an opt-in tau-space verification PDF.
"""

import os
import warnings

from astropy.stats import bayesian_blocks
import matplotlib.pyplot as plt
import numpy as np

from ..util.significance import pgsig, ppsig
from ..util.tools import plt_rc_context


def indices_in_intervals(lbins, rbins, intervals):
    """Return sorted unique bin indices that overlap any given interval.

    A bin ``[lbins[i], rbins[i])`` overlaps interval ``[low, upp)`` when
    neither ``rbins[i] <= low`` nor ``lbins[i] >= upp`` holds.

    Args:
        lbins: Left edges of each bin.
        rbins: Right edges of each bin (same length as ``lbins``).
        intervals: Iterable of ``(low, upp)`` pairs.

    Returns:
        Sorted unique integer indices; empty result is
        ``np.array([], dtype=int)`` with shape ``(0,)``.
    """

    idx = []
    for i, (left, right) in enumerate(zip(lbins, rbins, strict=False)):
        for low, upp in intervals:
            if not (right <= low or left >= upp):
                idx.append(i)
                break

    return np.unique(idx).astype(int) if idx else np.array([], dtype=int)


def time_rescaling_bblock(t, cts=None, bkg_integral=None, p0=0.05):
    """Run Bayesian blocks with optional non-stationary background correction.

    When ``bkg_integral`` is supplied, transforms the time axis via
    ``tau(t) = integral_0^t B(s) ds`` so the inhomogeneous Poisson
    process becomes homogeneous (Scargle 2013, Appendix B), runs
    ``events`` mode Bayesian blocks in ``tau``-space, then maps the
    resulting edges back to the original time axis through cubic
    interpolation on the ``(tau, t)`` pairs. Without ``bkg_integral``,
    delegates to ``astropy.stats.bayesian_blocks`` unchanged.

    Args:
        t: Photon arrival times (``cts=None``, unbinned branch) or
            sorted bin centers (``cts`` supplied, binned branch).
        cts: Per-bin counts; ``None`` selects the unbinned branch where
            each ``t`` entry counts as one event.
        bkg_integral: Callable ``t -> tau`` returning the cumulative
            background integral; must be strictly increasing on the
            input grid. ``None`` skips the time-rescaling transformation.
        p0: False-alarm probability passed to ``bayesian_blocks``.

    Returns:
        Block edges in the original time axis (sorted, length >= 2).

    Raises:
        ValueError: If ``bkg_integral`` is not strictly increasing on
            the input grid.
    """

    t = np.asarray(t, dtype=float)

    if cts is None:
        # Match astropy events-mode internals: dedupe and sort photon times
        t = np.sort(np.unique(t))
    else:
        cts = np.asarray(cts)

    if bkg_integral is None:
        if cts is None:
            return bayesian_blocks(t, fitness='events', p0=p0)
        return bayesian_blocks(t, cts, fitness='events', p0=p0)

    tau = np.asarray(bkg_integral(t), dtype=float)
    if len(tau) > 1 and not np.all(np.diff(tau) > 0):
        raise ValueError('bkg_integral must be strictly increasing on the input grid')

    if cts is None:
        edges_tau = bayesian_blocks(tau, fitness='events', p0=p0)
    else:
        edges_tau = bayesian_blocks(tau, cts, fitness='events', p0=p0)

    # Map tau-space block edges back to t-space via direct Voronoi-midpoint
    # lookup (matching threeML/bbbd). Avoids the float roundoff that
    # np.interp accumulates when tau cancels near the bkg_integral's
    # zero anchor (subnormal-scale noise at edges adjacent to that point).
    tau_v = np.concatenate([tau[:1], 0.5 * (tau[1:] + tau[:-1]), tau[-1:]])
    t_v = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
    idx = np.clip(np.searchsorted(tau_v, edges_tau), 0, len(t_v) - 1)
    return np.asarray(t_v[idx], dtype=float)


def intervals_equal(a, b):
    """Test whether two interval lists describe the same set.

    Compares ``[low, high]`` pairs as sorted floats; differences in
    ordering between the two inputs are tolerated.

    Args:
        a: First list of ``[low, high]`` pairs, or ``None``.
        b: Second list of ``[low, high]`` pairs, or ``None``.

    Returns:
        ``True`` when both inputs are ``None`` or when the two lists
        contain the same intervals up to ordering.
    """

    if a is None or b is None:
        return a is None and b is None
    if len(a) != len(b):
        return False
    a_sorted = sorted((float(p[0]), float(p[1])) for p in a)
    b_sorted = sorted((float(p[0]), float(p[1])) for p in b)
    return a_sorted == b_sorted


def filter_block_edges(edges_raw, merge_threshold, protected=None, mode='full'):
    """Drop Bayesian-block edges that produce undersized blocks.

    Both modes share one mechanism: flag interior edges for removal, then
    return the surviving edges with the endpoints kept and ``protected``
    points force-injected. Only the flagging rule differs.

    - ``mode='full'`` (binned grid): a block narrower than ``merge_threshold``
      is below the sampling resolution, so an interior edge is flagged when
      either adjacent block is undersized, or when it crowds a protected
      point. This checks every block.
    - ``mode='edges'`` (unbinned event-time blocks): every block is already
      significant at the ``p0`` level -- narrow spikes included -- so interior
      edges are kept verbatim and only an undersized *first* or *last* block
      (the inserted observation-window boundary, which may be spuriously thin)
      is merged into its neighbor.

    Args:
        edges_raw: Candidate block edges (1D, sorted, endpoints clamped).
        merge_threshold: Minimum block width; a block narrower than this is
            merged into a neighbor. Callers typically pass the smallest raw
            bin size relaxed by a safety factor (e.g. ``min(binsize) / 1.8``).
        protected: Optional 1D array-like of edges that must appear in
            the output. Points outside ``(edges_raw[0], edges_raw[-1])``
            are silently ignored.
        mode: ``'full'`` to filter every block (default), or ``'edges'`` to
            filter only the first and last block.

    Returns:
        Sorted unique edges with undersized neighbors removed.

    Raises:
        ValueError: If ``mode`` is neither ``'full'`` nor ``'edges'``.
    """

    if mode not in ('full', 'edges'):
        raise ValueError(f"mode must be 'full' or 'edges', got {mode!r}")

    if protected is None or len(protected) == 0:
        protected = np.empty(0, dtype=float)
    else:
        protected = np.asarray(protected, dtype=float)
        protected = protected[(protected > edges_raw[0]) & (protected < edges_raw[-1])]

    edges = np.asarray(edges_raw, dtype=float)
    drop = np.zeros(len(edges), dtype=bool)

    if mode == 'full':
        for i in range(1, len(edges) - 1):
            drop[i] = (edges[i] - edges[i - 1]) < merge_threshold or (
                edges[i + 1] - edges[i]
            ) < merge_threshold
    elif len(edges) >= 3:
        drop[1] |= (edges[1] - edges[0]) < merge_threshold
        drop[-2] |= (edges[-1] - edges[-2]) < merge_threshold

    return np.unique(np.concatenate([edges[~drop], protected]))


def classify_bins(snr, left, right, sigma, bad_sentinel=-5):
    """Split bins by SNR into signal, background, and bad categories.

    Classification:

    - ``snr > sigma`` -- signal.
    - ``bad_sentinel < snr <= sigma`` -- background.
    - otherwise (typically ``snr <= bad_sentinel``) -- bad.

    Args:
        snr: Per-bin signal-to-noise values.
        left: Left edges of each bin.
        right: Right edges of each bin (same length as ``left``).
        sigma: Detection threshold above which a bin becomes signal.
        bad_sentinel: Threshold below which a bin is flagged bad.

    Returns:
        Dict with keys ``sig_idx``/``bkg_idx``/``bad_idx`` (``np.ndarray``
        of ``int``) and ``sig_int``/``bkg_int``/``bad_int`` (lists of
        ``[left[i], right[i]]`` pairs).
    """

    sig_idx, bkg_idx, bad_idx = [], [], []
    sig_int, bkg_int, bad_int = [], [], []
    for i, s in enumerate(snr):
        pair = [float(left[i]), float(right[i])]
        if s > sigma:
            sig_idx.append(i)
            sig_int.append(pair)
        elif bad_sentinel < s <= sigma:
            bkg_idx.append(i)
            bkg_int.append(pair)
        else:
            bad_idx.append(i)
            bad_int.append(pair)

    return {
        'sig_idx': np.array(sig_idx, dtype=int),
        'bkg_idx': np.array(bkg_idx, dtype=int),
        'bad_idx': np.array(bad_idx, dtype=int),
        'sig_int': sig_int,
        'bkg_int': bkg_int,
        'bad_int': bad_int,
    }


def detect_pulses_by_snr(re_snr, edges, sigma, mp=True):
    """Identify pulse intervals from per-block SNR by left-to-right scan.

    Walks the Bayesian blocks once: opens an interval at the first
    block whose SNR exceeds ``sigma`` and closes it at the next block
    whose SNR falls back to or below ``sigma``. The threshold is
    asymmetric -- ``snr == sigma`` closes an open interval but does
    not open a new one.

    The signed-SNR convention of :func:`pg_snr` / :func:`pp_snr` /
    :func:`gauss_snr` makes the ``> sigma`` test self-sufficient: a
    statistically significant downward fluctuation has negative SNR and
    cannot open a pulse, so no separate ``re_net > 0`` check is needed.

    Edge handling:

    - A leading pulse that opens at ``edges[0]`` is dropped, since its
      true onset lies before the observation window.
    - A trailing pulse that never closes (``re_snr`` stays above
      ``sigma`` through the final block) is dropped.
    - A trailing pulse that does close inside the last block is kept.

    Args:
        re_snr: Per-block signal-to-noise values.
        edges: Block edges (length ``len(re_snr) + 1``).
        sigma: SNR threshold above which a block is considered part of
            a pulse.
        mp: When ``True``, keep every detected sub-pulse. When ``False``,
            collapse them into a single span from the first ``pstart``
            to the last ``pstop``; any quiescent gap between sub-pulses
            is absorbed into that span (which inflates the resulting
            duration). A warning is emitted when more than one sub-pulse
            was detected.

    Returns:
        ``(pstart, pstop)`` -- two ``np.ndarray`` of equal length giving
        the start and stop times of each detected pulse.
    """

    pstart, pstop = [], []
    flag = False
    for i in range(len(re_snr)):
        if re_snr[i] > sigma and not flag:
            pstart.append(edges[i])
            flag = True
        elif re_snr[i] <= sigma and flag:
            pstop.append(edges[i])
            flag = False

    if flag and len(pstart) > 0:
        pstart.pop()
    if len(pstart) > 0 and pstart[0] == edges[0]:
        pstart = pstart[1:]
        pstop = pstop[1:]

    if not mp and len(pstart) > 0:
        if len(pstart) > 1:
            msg = 'multi-pulse will be combined into one'
            warnings.warn(msg, UserWarning, stacklevel=2)
        pstart = [pstart[0]]
        pstop = [pstop[-1]]

    return np.array(pstart), np.array(pstop)


def pg_snr(cts_i, bcts_i, bcts_err_i=None):
    """Compute Poisson-with-Gaussian-background SNR for one bin.

    Falls back to the pure-Poisson estimator :func:`ppsig` when no
    background uncertainty is supplied. Returns ``-5`` (a bad-bin
    sentinel) for non-positive inputs or zero background error, and
    ``NaN`` when ``cts_i`` itself is non-finite (signals a missing-data
    bin so the caller and downstream :func:`classify_bins` can keep
    treating it as bad without invoking the underlying SNR estimator,
    which is not NaN-safe).

    Args:
        cts_i: Observed source counts in the bin.
        bcts_i: Expected background counts in the bin.
        bcts_err_i: Background-count uncertainty; ``None`` selects the
            Poisson-only branch.

    Returns:
        SNR value, ``-5`` when the bin is unusable, or ``NaN`` when
        ``cts_i`` is not finite.
    """

    if not np.isfinite(cts_i):
        return np.nan

    if bcts_err_i is None:
        if bcts_i <= 0 or cts_i <= 0:
            return -5
        return ppsig(cts_i, bcts_i, 1)

    if bcts_i <= 0 or cts_i <= 0 or bcts_err_i == 0:
        return -5

    return pgsig(cts_i, bcts_i, bcts_err_i)


def pp_snr(cts_i, bcts_i, backscale):
    """Compute Poisson-with-Poisson-background SNR for one bin.

    Args:
        cts_i: Observed source counts in the bin.
        bcts_i: Observed background counts in the bin.
        backscale: Ratio scaling background counts into the source region.

    Returns:
        SNR value, or ``-5`` when either count is negative.
    """

    if bcts_i < 0 or cts_i < 0:
        return -5

    return ppsig(cts_i, bcts_i, backscale)


def gauss_snr(cts_i, cts_err_i):
    """Compute Gaussian SNR ``cts_i / cts_err_i`` for one bin.

    Args:
        cts_i: Observed counts (or signal) in the bin.
        cts_err_i: 1-sigma uncertainty on ``cts_i``.

    Returns:
        SNR value, or ``-5`` when ``cts_err_i <= 0``.
    """

    if cts_err_i <= 0:
        return -5

    return cts_i / cts_err_i


class SignalPlotter:
    """Composable two-panel diagnostic figure for signal classification.

    Top panel (``ax_top``) hosts total quantities (raw rate, total
    Bayesian-block step trace, optional background); bottom panel
    (``ax_bot``) hosts net quantities, with a twin right axis lazily
    created by :meth:`plot_snr` for the per-block SNR overlay.
    :meth:`plot_block` tracks the bottom net-block step trace so that
    :meth:`plot_snr` can remove it before drawing the SNR layer.
    Compose by calling :meth:`plot_curve`, :meth:`plot_block`, and
    :meth:`plot_snr` in any order, then :meth:`save` or :meth:`show`.

    Attributes:
        fig: The underlying ``matplotlib`` Figure.
        ax_top: Top-panel axes (total light-curve quantities).
        ax_bot: Bottom-panel left axes (net light-curve quantities).
        ax_bot_twinx: Twin right axes on the bottom panel, created
            lazily by :meth:`plot_snr`; ``None`` until then.
        ax_bot_block_line: Handle to the net Bayesian-block line drawn
            on the bottom panel by :meth:`plot_block`, retained so
            :meth:`plot_snr` can remove it; ``None`` otherwise.
    """

    def __init__(self, figsize=(8, 8)):
        """Create an empty two-panel figure with shared x-axis.

        Args:
            figsize: Width and height of the figure in inches.
        """

        self.fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        plt.subplots_adjust(hspace=0, wspace=0)
        self.ax_top, self.ax_bot = axes
        self.ax_bot_twinx = None
        self.ax_bot_block_line = None
        self._style_axis(self.ax_top)
        self._style_axis(self.ax_bot)

        self._gaps = None
        self._bin_lbins = None
        self._bin_rbins = None

    def set_gaps(self, gaps, bins):
        """Register missing-data intervals to mask in subsequent plot_* calls.

        Bins / blocks that overlap any of ``gaps`` are replaced with
        ``NaN`` in the plot arrays so the missing region renders as a
        break rather than a spurious zero-rate excursion. Only affects
        rendering; the caller's input arrays are not mutated and the
        JSON dumps in :class:`~heapy.auto.signal.pgSignal.save` still
        carry the raw values.

        Args:
            gaps: List of ``[low, high]`` intervals; empty / falsy
                disables masking.
            bins: Per-bin edges (length ``N + 1``) used by
                :meth:`plot_curve` to derive bin lefts / rights for the
                overlap test. :meth:`plot_block` and :meth:`plot_snr`
                use their own ``edges`` argument and do not consult this.
        """

        self._gaps = gaps
        self._bin_lbins = bins[:-1]
        self._bin_rbins = bins[1:]

    @staticmethod
    def _style_axis(ax):

        ax.minorticks_on()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
        ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')

    @staticmethod
    def _align_zero(ax_left, ax_right):
        """Align ``y = 0`` on a pair of twinned axes.

        Extends each axis only downward (never crops live data) so the
        zero line lands at the same vertical position on both. Bails
        out without changes when either axis's data is fully
        non-positive (zero alignment is ill-defined in that case).

        Args:
            ax_left: Primary axes (e.g. the bottom-panel left axis).
            ax_right: Twin axes (e.g. the bottom-panel ``twinx`` axis).
        """

        lo_l, hi_l = ax_left.get_ylim()
        lo_r, hi_r = ax_right.get_ylim()

        lo_l, hi_l = min(lo_l, 0.0), max(hi_l, 0.0)
        lo_r, hi_r = min(lo_r, 0.0), max(hi_r, 0.0)

        if hi_l <= 0 or hi_r <= 0:
            return

        f = max(-lo_l / (hi_l - lo_l), -lo_r / (hi_r - lo_r))
        if f >= 1:
            return

        scale = -f / (1 - f) if f > 0 else 0.0
        ax_left.set_ylim(scale * hi_l, hi_l)
        ax_right.set_ylim(scale * hi_r, hi_r)

    def plot_curve(self, time, rate, net, bak=None, bak_err=None):
        """Draw raw light curves on both panels.

        Top panel: ``rate`` (blue) plus optional background (red, with a
        1-sigma shaded band when ``bak_err`` is given). Bottom panel:
        ``net`` (black). Caches ``net`` for the y-limit synchronization
        in :meth:`plot_snr`.

        Args:
            time: Abscissae of the per-bin series.
            rate: Total per-bin rate plotted on the top panel.
            net: Net per-bin rate plotted on the bottom panel; pass
                ``rate`` when no background subtraction applies.
            bak: Optional background rate plotted on the top panel.
            bak_err: Optional 1-sigma uncertainty on ``bak``.
        """

        rate = np.asarray(rate, dtype=float).copy()
        net = np.asarray(net, dtype=float).copy()
        if self._gaps:
            idx = indices_in_intervals(self._bin_lbins, self._bin_rbins, self._gaps)
            rate[idx] = np.nan
            net[idx] = np.nan

        self.ax_top.plot(time, rate, lw=1.0, c='b', label='Light curve')
        if bak is not None:
            self.ax_top.plot(time, bak, lw=1.0, c='r', label='Background')
            if bak_err is not None:
                self.ax_top.fill_between(
                    time, bak - bak_err, bak + bak_err, facecolor='red', alpha=0.5
                )
        self.ax_top.set_xlim([min(time), max(time)])
        self.ax_top.set_ylabel('Rate (cts/s)')
        self.ax_top.legend(frameon=False)

        self.ax_bot.plot(time, net, lw=1.0, c='b', label='Net light curve')
        self.ax_bot.set_xlim([min(time), max(time)])
        self.ax_bot.set_xlabel('Time (s)')
        self.ax_bot.set_ylabel('Rate (cts/s)')
        self.ax_bot.legend(frameon=False)

        self._net = net

    def plot_block(self, edges, re_rate, re_net):
        """Overlay Bayesian-block step traces on both panels.

        Top panel: total block rate. Bottom panel: net block rate; the
        bottom line is tracked so that :meth:`plot_snr` can remove it
        when replacing it with the SNR overlay.

        Args:
            edges: Bayesian-block edges (length ``N + 1``).
            re_rate: Total block rate per block (top panel).
            re_net: Net block rate per block (bottom panel).
        """

        re_rate = np.asarray(re_rate, dtype=float).copy()
        re_net = np.asarray(re_net, dtype=float).copy()
        if self._gaps:
            idx = indices_in_intervals(edges[:-1], edges[1:], self._gaps)
            re_rate[idx] = np.nan
            re_net[idx] = np.nan

        self.ax_top.plot(
            edges,
            np.append(re_rate, [re_rate[-1]]),
            lw=1.0,
            c='c',
            drawstyle='steps-post',
            label='Bayesian block',
        )
        self.ax_top.legend(frameon=False)

        (line,) = self.ax_bot.plot(
            edges,
            np.append(re_net, [re_net[-1]]),
            lw=1.0,
            c='c',
            drawstyle='steps-post',
            label='Bayesian block',
        )
        self.ax_bot_block_line = line
        self.ax_bot.legend(frameon=False)

    def plot_snr(self, edges, re_snr, sigma):
        """Replace the bottom net-block trace with a twin-axis SNR overlay.

        Removes any net Bayesian-block line previously drawn by
        :meth:`plot_block` on the bottom panel, then creates a twin
        right axis carrying the per-block SNR as a cyan ``steps-post``
        trace plus a dashed detection threshold at ``sigma``. The
        bottom-panel legend is rebuilt to merge handles from both axes,
        and ``y = 0`` on the left and twin-x axes is aligned via
        :meth:`_align_zero` (each axis is only extended downward,
        never cropped).

        Args:
            edges: Bayesian-block edges (length ``N + 1``).
            re_snr: Per-block SNR (length ``N``).
            sigma: Detection threshold drawn as a dashed horizontal line.
        """

        re_snr = np.asarray(re_snr, dtype=float).copy()
        if self._gaps:
            idx = indices_in_intervals(edges[:-1], edges[1:], self._gaps)
            re_snr[idx] = np.nan

        if self.ax_bot_block_line is not None:
            self.ax_bot_block_line.remove()
            self.ax_bot_block_line = None

        if self.ax_bot_twinx is None:
            self.ax_bot_twinx = self.ax_bot.twinx()

        ax = self.ax_bot_twinx
        ax.plot(
            edges,
            np.append(re_snr, [re_snr[-1]]),
            lw=1.0,
            c='c',
            drawstyle='steps-post',
            label='Bayesian block SNR',
        )
        ax.axhline(0.0, lw=1.0, c='red', ls='--')
        ax.axhline(sigma, lw=1.0, c='grey', ls='--', label=f'{sigma:.1f}$\\sigma$')
        ax.set_ylabel('Significance (sigma)')

        bot_legend = self.ax_bot.get_legend()
        if bot_legend is not None:
            bot_legend.remove()

        h_l, l_l = self.ax_bot.get_legend_handles_labels()
        h_r, l_r = ax.get_legend_handles_labels()
        ax.legend(h_l + h_r, l_l + l_r, frameon=False)

        self._align_zero(self.ax_bot, self.ax_bot_twinx)

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

        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close(self.fig)


def plot_tau_diagnostic(poly, ts, bins, edges, time, rate, savepath):
    """Save a 2-panel time-rescaling diagnostic PDF for debugging.

    Top panel: time-axis raw rate ``rate(t)``, polynomial background
    overlay (with a 1-sigma band), and Bayesian-block edges drawn as
    vertical dashed lines. Bottom panel: tau-axis photon density
    ``dN/dtau`` from a uniform-tau histogram of ``ts``, with the BB
    edges mapped through ``poly.integral`` and a horizontal line at
    ``dN/dtau = 1`` marking the level a correctly-modelled background
    would sit at after time rescaling. Signal segments rise above the
    unity line.

    Useful for verifying that the polynomial background is adequate --
    when ``dN/dtau`` deviates strongly from 1 over intervals known to
    be background-only, the polynomial is biased and the BB edges may
    be unreliable.

    The integral on the (small) ``bins`` grid is interpolated to photon
    times so the call stays cheap regardless of how many photons live
    in ``ts``.

    Args:
        poly: Background model exposing ``val(x) -> (mo, err)`` and
            ``integral(x) -> (mo, err)``; typically a fitted
            :class:`~.polynomial.Polynomial` or
            :class:`~.polynomial.CompositePolynomial`.
        ts: Source photon arrival times.
        bins: Histogram bin edges (length ``N + 1``).
        edges: Bayesian-block edges (length ``n + 1``).
        time: Per-bin centers used for the time-axis curve.
        rate: Per-bin source rate used for the time-axis curve.
        savepath: Destination directory; created if absent. The PDF is
            written to ``savepath/signal_tau.pdf``.
    """

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    ts_sorted = np.sort(ts)
    # Evaluate the integral on the (small) bins grid, then interpolate to
    # photon times. Calling poly.integral directly on ts hits the
    # err-computation path; the err itself is unused for the diagnostic
    # and the interp residual is far below Poisson noise.
    tau_grid_t = poly.integral(bins)[0]
    tau_at_ts = np.interp(ts_sorted, bins, tau_grid_t)
    tau_edges = np.interp(edges, bins, tau_grid_t)

    n_tau_bins = len(bins) - 1
    tau_lo = float(np.min(tau_at_ts))
    tau_hi = float(np.max(tau_at_ts))
    tau_grid = np.linspace(tau_lo, tau_hi, n_tau_bins + 1)
    cts_tau, _ = np.histogram(tau_at_ts, bins=tau_grid)
    rate_tau = cts_tau / np.diff(tau_grid)
    tau_center = 0.5 * (tau_grid[:-1] + tau_grid[1:])

    bak, bak_err = poly.val(time)

    with plt_rc_context():
        fig, (ax_t, ax_tau) = plt.subplots(2, 1, figsize=(8, 8))
        plt.subplots_adjust(hspace=0.25)

        ax_t.plot(time, rate, lw=1.0, c='b', label='Light curve')
        ax_t.plot(time, bak, lw=1.0, c='r', label='Background')
        ax_t.fill_between(time, bak - bak_err, bak + bak_err, color='r', alpha=0.3)
        for e in edges[1:-1]:
            ax_t.axvline(e, ls='--', c='k', lw=0.6, alpha=0.6)
        ax_t.set_xlim([bins[0], bins[-1]])
        ax_t.set_xlabel('Time (s)')
        ax_t.set_ylabel('Rate (cts/s)')
        ax_t.set_title('time axis')
        ax_t.legend(frameon=False)
        ax_t.minorticks_on()

        ax_tau.plot(tau_center, rate_tau, lw=1.0, c='b', label='dN / dtau')
        ax_tau.axhline(1.0, ls='--', c='r', lw=1.0, label='Expected bkg = 1')
        for e in tau_edges[1:-1]:
            ax_tau.axvline(e, ls='--', c='k', lw=0.6, alpha=0.6)
        ax_tau.set_xlim([tau_lo, tau_hi])
        ax_tau.set_xlabel('tau (cumulative background counts)')
        ax_tau.set_ylabel('dN / dtau')
        ax_tau.set_title('tau axis (time-rescaled)')
        ax_tau.legend(frameon=False)
        ax_tau.minorticks_on()

        fig.savefig(savepath + '/signal_tau.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
