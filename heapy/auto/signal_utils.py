"""Shared helpers and plotter for the ``signal`` module's pipeline classes.

Bundles three groups of utilities consumed by :mod:`heapy.auto.signal`:

- Bin / interval helpers (:func:`indices_in_intervals`,
  :func:`filter_block_edges`, :func:`classify_bins`).
- One-bin SNR estimators (:func:`pg_snr`, :func:`pp_snr`,
  :func:`gauss_snr`) wrapping :mod:`heapy.util.significance`.
- :class:`SignalPlotter`, the composable two-panel diagnostic figure
  used by every ``save()`` method in :mod:`heapy.auto.signal`.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..util.significance import pgsig, ppsig


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


def filter_block_edges(edges_raw, min_binsize, margin=1.8):
    """Drop interior Bayesian-block edges that produce undersized blocks.

    An interior edge is kept only when both neighboring gaps exceed
    ``min_binsize / margin``. The two endpoint edges are always preserved.
    ``edges_raw`` must already be clamped to the data range by the caller.

    Args:
        edges_raw: Candidate block edges (1D, sorted, endpoints clamped).
        min_binsize: Minimum physical bin size from the raw histogram.
        margin: Safety factor relaxing ``min_binsize``; larger keeps more edges.

    Returns:
        Sorted unique edges with undersized neighbors removed.
    """

    edges = [edges_raw[0], edges_raw[-1]]
    for i in range(1, len(edges_raw) - 1):
        flag1 = (edges_raw[i] - edges_raw[i - 1]) > min_binsize / margin
        flag2 = (edges_raw[i + 1] - edges_raw[i]) > min_binsize / margin
        if flag1 and flag2:
            edges.append(edges_raw[i])

    return np.unique(edges)


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
        pair = [left[i], right[i]]
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


def pg_snr(cts_i, bcts_i, bcts_err_i=None):
    """Compute Poisson-with-Gaussian-background SNR for one bin.

    Falls back to the pure-Poisson estimator :func:`ppsig` when no
    background uncertainty is supplied. Returns ``-5`` (a bad-bin
    sentinel) for non-positive inputs or zero background error.

    Args:
        cts_i: Observed source counts in the bin.
        bcts_i: Expected background counts in the bin.
        bcts_err_i: Background-count uncertainty; ``None`` selects the
            Poisson-only branch.

    Returns:
        SNR value, or ``-5`` when the bin is unusable.
    """

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


def plt_rc_context():
    """Return the rc-context preset wrapped around every ``SignalPlotter`` save.

    Bundles the project's PDF-friendly overrides -- Inter sans-serif
    body text, ``stixsans`` math, and TrueType (type-42) embedding for
    the ``pdf`` and ``ps`` backends -- so saved figures stay
    typographically consistent regardless of the global rcParams at
    call time.

    Returns:
        A :class:`matplotlib.RcParams` context manager applying the
        preset overrides; use in a ``with`` block around figure
        rendering.
    """

    plt_rc = {
        'font.family': 'serif',
        'font.serif': ['STIX Two Text'],
        'mathtext.fontset': 'stix',
        'font.size': 12,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }

    return plt.rc_context(plt_rc)


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
        self.ax_bot.set_xlabel('Time')
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
