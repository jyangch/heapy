"""Pure helpers shared by the signal classes in this package.

Bundles interval/bin utilities and small SNR wrappers so that
:mod:`ggsignal`, :mod:`ppsignal`, and :mod:`polybase` can import them via a
single ``from .core_funcs import *``.
"""

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
    for i, (l, u) in enumerate(zip(lbins, rbins)):
        for low, upp in intervals:
            if not (u <= low or l >= upp):
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

    - ``snr > sigma`` — signal.
    - ``bad_sentinel < snr <= sigma`` — background.
    - otherwise (typically ``snr <= bad_sentinel``) — bad.

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