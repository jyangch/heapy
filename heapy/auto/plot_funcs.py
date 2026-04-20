"""Matplotlib helpers shared by the ``save`` methods of the signal classes.

Provides a common serif/PDF-friendly rc context and three diagnostic
plotting helpers (light curve + Bayesian blocks, SNR twin-axis, sorted
block bar chart).

Attributes:
    SAVE_RC (dict): Matplotlib rc overrides applied via
        :func:`rc_context_for_save`.
"""

import numpy as np
import matplotlib.pyplot as plt


SAVE_RC = {
    'font.family': 'serif',
    'font.sans-serif': ['Georgia'],
    'font.size': 12,
    'pdf.fonttype': 42,
}


def rc_context_for_save():
    """Return an ``rc_context`` preset tuned for saved PDF figures.

    Returns:
        A :class:`matplotlib.RcParams` context manager applying
        :data:`SAVE_RC`.
    """

    return plt.rc_context(SAVE_RC)


def _style_axis(ax):
    """Apply the shared minor-tick and inward-tick styling to ``ax``.

    Args:
        ax: Target matplotlib ``Axes``.
    """

    ax.minorticks_on()
    ax.tick_params(axis='x', which='both', direction='in', labelcolor='k', colors='k')
    ax.tick_params(axis='y', which='both', direction='in', labelcolor='k', colors='k')
    ax.tick_params(which='major', width=1.0, length=5)
    ax.tick_params(which='minor', width=1.0, length=3)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')


def plot_lightcurve_bblock(ax, time, rate, edges, re_rate, bak=None, bak_err=None):
    """Plot a light curve overlaid with its Bayesian-block step function.

    Args:
        ax: Target matplotlib ``Axes``.
        time: Abscissae of the raw light curve.
        rate: Raw rate values matching ``time``.
        edges: Bayesian-block edges (length ``N + 1``).
        re_rate: Block-averaged rate per block (length ``N``).
        bak: Optional background rate matching ``time``.
        bak_err: Optional 1-sigma uncertainty on ``bak``.
    """

    ax.plot(time, rate, lw=1.0, c='b', label='Light curve')
    ax.plot(edges, np.append(re_rate, [re_rate[-1]]),
            lw=1.0, c='c', drawstyle='steps-post', label='Bayesian block')
    if bak is not None:
        ax.plot(time, bak, lw=1.0, c='r', label='Background')
        if bak_err is not None:
            ax.fill_between(time, bak - bak_err, bak + bak_err,
                            facecolor='red', alpha=0.5)
    ax.set_xlim([min(time), max(time)])
    ax.set_xlabel('Time')
    ax.set_ylabel('Rate')
    _style_axis(ax)
    ax.legend(frameon=False)


def plot_snr(ax, time, left_curve, edges, snr, re_snr, sigma,
             left_label='Net light curve',
             bblock_snr_label='Bayesian block SNR',
             ylim_factors=(1.5, 1.1)):
    """Plot a light curve on ``ax`` with a twin-axis SNR overlay.

    The left axis shows ``left_curve``; the right axis shows per-bin and
    per-block SNR together with a horizontal detection threshold at
    ``sigma``. Y-limits on both axes are synchronized via ``ylim_factors``.

    Args:
        ax: Target matplotlib ``Axes``.
        time: Abscissae of ``left_curve`` and ``snr``.
        left_curve: Series plotted on the left axis (e.g. net rate).
        edges: Bayesian-block edges (length ``N + 1``).
        snr: Per-bin SNR matching ``time``.
        re_snr: Per-block SNR (length ``N``).
        sigma: Detection threshold drawn as a dashed horizontal line.
        left_label: Legend label for ``left_curve``.
        bblock_snr_label: Legend label for the block SNR step trace.
        ylim_factors: ``(low_factor, high_factor)`` multipliers applied to
            ``min``/``max`` of ``left_curve`` for the left-axis limits.
    """

    p1, = ax.plot(time, left_curve, lw=1.0, c='k', label=left_label)
    ax.set_xlim([min(time), max(time)])
    ax.set_xlabel('Time')
    ax.set_ylabel('Rate')
    _style_axis(ax)

    ax1 = ax.twinx()
    p2, = ax1.plot(time, snr, lw=1.0, c='b', label='SNR', drawstyle='steps-mid')
    p3, = ax1.plot(edges, np.append(re_snr, [re_snr[-1]]),
                   lw=1.0, c='c', label=bblock_snr_label, drawstyle='steps-post')
    p4 = ax1.axhline(sigma, lw=1.0, c='grey', ls='--', label='%.1f$\\sigma$' % sigma)
    ax1.set_xlim([min(time), max(time)])
    ax1.set_ylabel('SNR')

    ratio = np.max(left_curve) / np.max(re_snr)
    fa, fb = ylim_factors
    ax_ylim = [fa * np.min(left_curve), fb * np.max(left_curve)]
    ax1_ylim = [lim / ratio for lim in ax_ylim]
    ax.set_ylim(ax_ylim)
    ax1.set_ylim(ax1_ylim)

    plt.legend(handles=[p1, p2, p3, p4], frameon=False)


def plot_sorted(ax, time, edges, re_cts, re_binsize,
                sig_idx, bkg_idx, bad_idx,
                rate=None, bak=None, bak_err=None):
    """Bar-chart the block rates colored by signal/background/bad class.

    Signal blocks are magenta, background blocks are blue, bad blocks are
    black. An optional raw rate trace and a background curve can be
    overlaid for context.

    Args:
        ax: Target matplotlib ``Axes``.
        time: Abscissae of the optional raw rate trace.
        edges: Bayesian-block edges (length ``N + 1``).
        re_cts: Counts per block.
        re_binsize: Block widths (length ``N``).
        sig_idx: Indices of signal blocks.
        bkg_idx: Indices of background blocks.
        bad_idx: Indices of bad blocks.
        rate: Optional raw rate trace matching ``time``.
        bak: Optional background rate matching ``time``.
        bak_err: Optional 1-sigma uncertainty on ``bak``.
    """

    colors = []
    for i in range(len(re_binsize)):
        if i in sig_idx: colors.append('m')
        if i in bkg_idx: colors.append('b')
        if i in bad_idx: colors.append('k')
    ax.bar((edges[:-1] + edges[1:]) / 2, re_cts / re_binsize,
           bottom=0, width=re_binsize, color=colors)
    if rate is not None:
        ax.plot(time, rate, lw=1.0, c='b', label='Light curve')
    if bak is not None:
        ax.plot(time, bak, lw=1.0, c='r', label='Background')
        if bak_err is not None:
            ax.fill_between(time, bak - bak_err, bak + bak_err,
                            facecolor='red', alpha=0.5)
    ax.set_xlim([min(time), max(time)])
    ax.set_xlabel('Time')
    ax.set_ylabel('Rate')
    _style_axis(ax)
