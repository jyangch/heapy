"""General-purpose data manipulation utilities for heapy.

Provides helpers for 2D matrix operations (transpose, pad, pop),
interval arithmetic (intersection, union), histogram rebinning with
statistical significance criteria (gstat / cstat / pgstat), array
utilities (scale, asymmetric Gaussian sampling), and LaTeX formatting
helpers.
"""

import warnings
import numpy as np
from itertools import zip_longest

from .significance import pgsig, ppsig


def transpose_2d_matrix(data):
    """Transpose a 2D list (matrix).

    Args:
        data: A 2D list to be transposed.

    Returns:
        The transposed 2D list, or the original ``data`` unchanged if it
        is empty or its first row is empty.
    """

    if not data or not data[0]:
        return data

    return list(map(list, zip(*data)))


def pad_2d_matrix(data, fillvalue='-'):
    """Pad a 2D list so that all rows share the same length.

    Shorter rows are extended on the right with ``fillvalue``.

    Args:
        data: A 2D list to be padded.
        fillvalue: The value used to extend shorter rows. Default is ``'-'``.

    Returns:
        The padded 2D list, or ``False`` if ``data`` is not a 2D list or
        padding fails.
    """

    if not data or not isinstance(data[0], (list, np.ndarray, tuple)):
        msg = 'Input "data" must be a 2-D list or array.'
        warnings.warn(msg, UserWarning, stacklevel=2)
        return False

    try:
        transposed = zip_longest(*data, fillvalue=fillvalue)
        padded_data = list(map(list, zip(*transposed)))
        return padded_data
    except Exception as e:
        warnings.warn(f"Padding failed: {e}", UserWarning, stacklevel=2)
        return False


def pop_2d_matrix(data, popvalue='-'):
    """Remove all occurrences of a value from every row of a 2D list.

    Args:
        data: A 2D list from which to remove the specified value.
        popvalue: The value to be removed from each row. Default is ``'-'``.

    Returns:
        The 2D list with every occurrence of ``popvalue`` removed, or
        ``False`` if ``data`` is not a 2D list or the operation fails.
    """

    if not data or not isinstance(data[0], (list, np.ndarray, tuple)):
        msg = 'Input "data" must be a 2-D list or array-like structure.'
        warnings.warn(msg, UserWarning, stacklevel=2)
        return False

    try:
        return [[item for item in row if item != popvalue] for row in data]
    except Exception as e:
        warnings.warn(f"Popping failed: {e}", UserWarning, stacklevel=2)
        return False


def intersection(A, B):
    """Compute the intersection of two lists of intervals.

    Args:
        A: List of intervals in the form
            ``[[a1_start, a1_end], [a2_start, a2_end], ...]``.
        B: List of intervals in the form
            ``[[b1_start, b1_end], [b2_start, b2_end], ...]``.

    Returns:
        A list of intervals representing the intersection of ``A`` and
        ``B``.  Returns an empty list if either input is empty.
    """

    if not A or not B:
        return []

    arr_a = np.array(A)
    arr_b = np.array(B)

    arr_a = arr_a[np.argsort(arr_a[:, -1])]
    arr_b = arr_b[np.argsort(arr_b[:, -1])]

    i, j = 0, 0
    res = []
    while i < len(arr_a) and j < len(arr_b):
        a_start, a_end = arr_a[i, 0], arr_a[i, 1]
        b_start, b_end = arr_b[j, 0], arr_b[j, 1]

        low = max(a_start, b_start)
        high = min(a_end, b_end)

        if low < high:
            res.append([low, high])

        if a_end < b_end:
            i += 1
        else:
            j += 1

    return res


def union(bins):
    """Compute the union of a list of intervals.

    Overlapping or adjacent intervals are merged into a single interval.

    Args:
        bins: List of intervals in the form
            ``[[start1, end1], [start2, end2], ...]``.

    Returns:
        A list of non-overlapping intervals representing the union of all
        input intervals, sorted by start value.  Returns an empty list if
        ``bins`` is ``None`` or empty.
    """

    if bins is None or len(bins) == 0:
        return []

    sorted_bins = sorted(bins, key=lambda x: x[0])

    res = []
    for current in sorted_bins:
        if not res or current[0] > res[-1][1]:
            res.append(list(current[:2]))
        else:
            res[-1][1] = max(res[-1][1], current[1])

    return res


def rebin(bins,
          stat,
          cts,
          cts_err=None,
          bcts=None,
          bcts_err=None,
          min_sigma=None,
          min_evt=None,
          max_bin=None,
          backscale=None):
    """Greedily merge adjacent bins until SNR and event thresholds are met.

    Adjacent bins are accumulated until the combined signal-to-noise ratio
    reaches ``min_sigma`` **and** the net counts reach ``min_evt``, or until
    ``max_bin`` bins have been consumed.  Any remainder at the end of the
    array is folded into the last accepted bin.

    Args:
        bins: Input bin edges as a list of ``[start, end]`` intervals.
        stat: Statistical method used to compute significance.  Must be one
            of ``'gstat'``, ``'cstat'``, ``'pgstat'``, or ``None``.
        cts: Counts array, one entry per bin.
        cts_err: Count errors, required when ``stat='gstat'``.  Defaults to
            an all-zero array.
        bcts: Background counts, required when ``stat`` is ``'cstat'`` or
            ``'pgstat'``.  Defaults to an all-zero array.
        bcts_err: Background count errors, required when
            ``stat='pgstat'``.  Defaults to an all-zero array.
        min_sigma: Minimum SNR threshold.  Default is ``-inf`` (no lower
            bound).
        min_evt: Minimum net-count threshold.  Default is ``0``.
        max_bin: Maximum number of original bins to combine.  Default is
            ``inf`` (unlimited).
        backscale: Scale factor applied to background counts when computing
            net counts and significance.  Default is ``1``.

    Returns:
        A tuple ``(new_bins, new_cts, new_cts_err, new_bcts, new_bcts_err)``
        where each element is a ``np.ndarray``.  ``new_bins`` contains the
        merged ``[start, end]`` intervals; the remaining arrays hold the
        accumulated counts, count errors, background counts, and background
        count errors for each merged bin.
    """

    _allowed_stat = ['gstat', 'cstat', 'pgstat', None]

    assert stat in _allowed_stat, f'unsupported stat: {stat}'

    if stat == 'gstat':
        assert cts_err is not None

    if stat == 'cstat':
        assert bcts is not None

    if stat == 'pgstat':
        assert bcts is not None
        assert bcts_err is not None

    if cts_err is None:
        cts_err = np.zeros_like(cts, dtype=float)

    if bcts is None:
        bcts = np.zeros_like(cts, dtype=float)

    if bcts_err is None:
        bcts_err = np.zeros_like(cts, dtype=float)

    if min_sigma is None:
        min_sigma = -np.inf

    if min_evt is None:
        min_evt = 0

    if max_bin is None:
        max_bin = np.inf

    if backscale is None:
        backscale = 1

    new_bins, new_cts, new_cts_err, new_bcts, new_bcts_err = [], [], [], [], []
    cc, cb, cc_err, cb_err, j, k = 0, 0, 0, 0, 0, 0

    for i in range(len(bins)):
        ci = cts[i]
        bi = bcts[i]
        ci_err = cts_err[i]
        bi_err = bcts_err[i]

        cc += ci
        cb += bi
        cc_err = np.sqrt(cc_err ** 2 + ci_err ** 2)
        cb_err = np.sqrt(cb_err ** 2 + bi_err ** 2)

        if stat == 'gstat':
            sigma = cc / cc_err
        elif stat == 'pgstat':
            sigma = pgsig(cc, cb * backscale, cb_err * backscale)
        elif stat == 'cstat':
            sigma = ppsig(cc, cb * backscale, 1)
        elif stat is None:
            sigma = 0
        else:
            raise AttributeError(f'unsupported stat: {stat}')

        evt = cc - cb * backscale

        if ((sigma >= min_sigma) and (evt >= min_evt)) or ((j-i+1) == max_bin):
            new_bins.append([bins[j][0], bins[i][1]])
            new_cts.append(cc)
            new_bcts.append(cb)
            new_cts_err.append(cc_err)
            new_bcts_err.append(cb_err)
            cc, cb, cc_err, cb_err = 0, 0, 0, 0
            j = i + 1
            k += 1

        if (i == len(bins) - 1) and ((sigma < min_sigma) or (evt < min_evt)) and (j-i+1) < max_bin:
            if k >= 1:
                new_bins[k-1][1] = bins[i][1]
                new_cts[k-1] = new_cts[k-1] + cc
                new_bcts[k-1] = new_bcts[k-1] + cb
                new_cts_err[k-1] = np.sqrt(new_cts_err[k-1] ** 2 + cc_err ** 2)
                new_bcts_err[k-1] = np.sqrt(new_bcts_err[k-1] ** 2 + cb_err ** 2)
            else:
                new_bins.append([bins[j][0], bins[i][1]])
                new_cts.append(cc)
                new_bcts.append(cb)
                new_cts_err.append(cc_err)
                new_bcts_err.append(cb_err)

    new_bins = np.array(new_bins)
    new_cts = np.array(new_cts)
    new_cts_err = np.array(new_cts_err)
    new_bcts = np.array(new_bcts)
    new_bcts_err = np.array(new_bcts_err)

    return new_bins, new_cts, new_cts_err, new_bcts, new_bcts_err


def multi_rebin(bins,
                stat_list,
                cts_list,
                cts_err_list=None,
                bcts_list=None,
                bcts_err_list=None,
                min_sigma_list=None,
                min_evt_list=None,
                backscale_list=None):
    """Rebin bins by simultaneously satisfying criteria across multiple datasets.

    Operates like :func:`rebin` but requires every detector dataset to meet
    its individual SNR and event thresholds before a merged bin is accepted.
    A bin boundary is only emitted when **all** datasets pass their
    respective criteria.

    Args:
        bins: Input bin edges as a ``np.ndarray`` of ``[start, end]``
            intervals.
        stat_list: List of statistical methods, one per dataset.  Each entry
            must be one of ``'gstat'``, ``'cstat'``, ``'pgstat'``, or
            ``None``.
        cts_list: List of counts arrays, one array per dataset.
        cts_err_list: List of count-error arrays, one per dataset.  Required
            for datasets using ``'gstat'``.  Defaults to all-zero arrays.
        bcts_list: List of background-count arrays, one per dataset.
            Required for datasets using ``'cstat'`` or ``'pgstat'``.
            Defaults to all-zero arrays.
        bcts_err_list: List of background-count-error arrays, one per
            dataset.  Required for datasets using ``'pgstat'``.  Defaults to
            all-zero arrays.
        min_sigma_list: List of minimum SNR thresholds, one per dataset.
            Each ``None`` entry defaults to ``-inf``.
        min_evt_list: List of minimum net-count thresholds, one per dataset.
            Each ``None`` entry defaults to ``0``.
        backscale_list: List of background scale factors, one per dataset.
            Each ``None`` entry defaults to ``1``.

    Returns:
        A tuple
        ``(new_bins, new_cts_list, new_cts_err_list, new_bcts_list, new_bcts_err_list)``
        where ``new_bins`` is a ``np.ndarray`` of merged ``[start, end]``
        intervals and each remaining element is a list of ``np.ndarray``
        objects holding the accumulated counts, count errors, background
        counts, and background count errors for every dataset.
    """

    _allowed_stat = ['gstat', 'cstat', 'pgstat', None]

    multi = len(cts_list)

    if stat_list is None:
        stat_list = [None] * multi

    if cts_err_list is None:
        cts_err_list = [None] * multi

    if bcts_list is None:
        bcts_list = [None] * multi

    if bcts_err_list is None:
        bcts_err_list = [None] * multi

    if min_sigma_list is None:
        min_sigma_list = [None] * multi

    if min_evt_list is None:
        min_evt_list = [None] * multi

    if backscale_list is None:
        backscale_list = [None] * multi

    for n in range(multi):

        assert stat_list[n] in _allowed_stat, f'unsupported stat: {stat_list[n]}'

        if stat_list[n] == 'gstat':
            assert cts_err_list[n] is not None

        if stat_list[n] == 'cstat':
            assert bcts_list[n] is not None

        if stat_list[n] == 'pgstat':
            assert bcts_list[n] is not None
            assert bcts_err_list[n] is not None

        if cts_err_list[n] is None:
            cts_err_list[n] = np.zeros_like(cts_list[n], dtype=float)

        if bcts_list[n] is None:
            bcts_list[n] = np.zeros_like(cts_list[n], dtype=float)

        if bcts_err_list[n] is None:
            bcts_err_list[n] = np.zeros_like(cts_list[n], dtype=float)

        if min_sigma_list[n] is None:
            min_sigma_list[n] = -np.inf

        if min_evt_list[n] is None:
            min_evt_list[n] = 0

        if backscale_list[n] is None:
            backscale_list[n] = 1

    new_bins, new_cts_list, new_cts_err_list = [], [[] for _ in range(multi)], [[] for _ in range(multi)]
    new_bcts_list, new_bcts_err_list = [[] for _ in range(multi)], [[] for _ in range(multi)]

    j, k = 0, 0
    cc_list, cb_list = [0] * multi, [0] * multi
    cc_err_list, cb_err_list = [0] * multi, [0] * multi
    rebin_flag = [False] * multi
    last_flag = [False] * multi

    for i in range(len(bins)):
        for n in range(multi):
            ci = cts_list[n][i]
            bi = bcts_list[n][i]
            ci_err = cts_err_list[n][i]
            bi_err = bcts_err_list[n][i]

            cc_list[n] = cc_list[n] + ci
            cb_list[n] = cb_list[n] + bi
            cc_err_list[n] = np.sqrt(cc_err_list[n] ** 2 + ci_err ** 2)
            cb_err_list[n] = np.sqrt(cb_err_list[n] ** 2 + bi_err ** 2)

            if stat_list[n] == 'gstat':
                sigma = cc_list[n] / cc_err_list[n]
            elif stat_list[n] == 'pgstat':
                sigma = pgsig(cc_list[n], cb_list[n] * backscale_list[n], cb_err_list[n] * backscale_list[n])
            elif stat_list[n] == 'cstat':
                sigma = ppsig(cc_list[n], cb_list[n] * backscale_list[n], 1)
            elif stat_list[n] is None:
                sigma = 0
            else:
                raise AttributeError(f'unsupported stat: {stat_list[n]}')

            evt = cc_list[n] - cb_list[n] * backscale_list[n]

            if ((sigma >= min_sigma_list[n]) and (evt >= min_evt_list[n])):
                rebin_flag[n] = True

            if (i == len(bins) - 1) and ((sigma < min_sigma_list[n]) or (evt < min_evt_list[n])):
                last_flag[n] = True

        if False not in rebin_flag:
            new_bins.append([bins[j][0], bins[i][1]])
            for n in range(multi):
                new_cts_list[n].append(cc_list[n])
                new_bcts_list[n].append(cb_list[n])
                new_cts_err_list[n].append(cc_err_list[n])
                new_bcts_err_list[n].append(cb_err_list[n])

            rebin_flag = [False] * multi
            cc_list, cb_list = [0] * multi, [0] * multi
            cc_err_list, cb_err_list = [0] * multi, [0] * multi
            j = i + 1
            k += 1

        if True in last_flag:
            if k >= 1:
                for n in range(multi):
                    new_bins[k-1][1] = bins[i][1]
                    new_cts_list[n][k-1] = new_cts_list[n][k-1] + cc_list[n]
                    new_bcts_list[n][k-1] = new_bcts_list[n][k-1] + cb_list[n]
                    new_cts_err_list[n][k-1] = np.sqrt(new_cts_err_list[n][k-1] ** 2 + cc_err_list[n] ** 2)
                    new_bcts_err_list[n][k-1] = np.sqrt(new_bcts_err_list[n][k-1] ** 2 + cb_err_list[n] ** 2)
            else:
                for n in range(multi):
                    new_cts_list[n].append(cc_list[n])
                    new_bcts_list[n].append(cb_list[n])
                    new_cts_err_list[n].append(cc_err_list[n])
                    new_bcts_err_list[n].append(cb_err_list[n])

    new_bins = np.array(new_bins)
    new_cts_list = [np.array(cts) for cts in new_cts_list]
    new_cts_err_list = [np.array(cts_err) for cts_err in new_cts_err_list]
    new_bcts_list = [np.array(bcts) for bcts in new_bcts_list]
    new_bcts_err_list = [np.array(bcts_err) for bcts_err in new_bcts_err_list]

    return new_bins, new_cts_list, new_cts_err_list, new_bcts_list, new_bcts_err_list


def split_bool_mask(mask, times, selection_value=False):
    """Convert a boolean mask into time intervals where the mask equals a value.

    Args:
        mask: Boolean array (``dtype=bool``) aligned with ``times``.
        times: Array of time values corresponding to each mask element.
        selection_value: The boolean value that defines the selected
            segments.  Default is ``False``.

    Returns:
        A list of ``(t_start, t_stop)`` tuples covering every contiguous
        run of elements in ``mask`` that equal ``selection_value``.
        Returns an empty list when no such elements exist or when ``mask``
        is empty.
    """

    if len(mask) == 0:
        return []

    if np.all(mask == selection_value):
        return [(times[0], times[-1])]
    if np.all(mask != selection_value):
        return []

    changes = np.diff(mask.astype(int))
    idx = np.where(changes != 0)[0] + 1

    boundaries = np.concatenate(([0], idx, [len(mask)]))

    segs = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        if mask[start_idx] == selection_value:
            t_start = times[start_idx]
            t_stop = times[boundaries[i+1] - 1]
            segs.append((t_start, t_stop))

    return segs


def scale_of_one(seq, seq_min=None):
    """Min-max normalise a sequence to the range [0, 1].

    Args:
        seq: Array-like sequence to normalise.
        seq_min: Minimum value used as the lower bound of the
            normalisation range.  When ``None``, the minimum of ``seq``
            is used.  Default is ``None``.

    Returns:
        A ``np.ndarray`` with values scaled to ``[0, 1]``.  Returns an
        all-zero array of the same shape when the effective range is zero.
    """

    seq = np.asanyarray(seq)

    s_max = seq.max()
    s_min = seq.min() if seq_min is None else seq_min

    s_range = s_max - s_min

    if s_range == 0:
        return np.zeros_like(seq, dtype=float)

    return (seq - s_min) / s_range


def format_err_latex(value, low, upp, precision=2):
    r"""Format a central value with asymmetric errors as a LaTeX string.

    Expresses the result in scientific notation as
    :math:`v \times 10^{n}` with lower and upper error bars.  When
    ``value`` is zero the exponent is omitted.

    Args:
        value: The central value.
        low: The lower error magnitude (i.e. ``value - lower_bound``).
        upp: The upper error magnitude (i.e. ``upper_bound - value``).
        precision: Number of decimal places in the formatted output.
            Default is ``2``.

    Returns:
        A LaTeX string of the form
        ``$v_s^{+u_s}_{-l_s}\times 10^{n}$`` suitable for use in
        matplotlib labels or LaTeX documents.
    """

    v, l, u = map(np.longdouble, [value, low, upp])

    if v == 0:
        return f"${0:.{precision}f}_{{-{l:.{precision}f}}}^{{+{u:.{precision}f}}}$"

    exponent = int(np.floor(np.log10(np.abs(v))))
    factor = 10.0 ** exponent

    v_s, l_s, u_s = v / factor, l / factor, u / factor

    fmt = f".{precision}f"
    return (f"${v_s:{fmt}}_{{-{l_s:{fmt}}}}^{{+{u_s:{fmt}}}}"
            f"\\times 10^{{{exponent}}}$")


def escape_latex_chars(text):
    """Escape special LaTeX characters in a string by prepending backslashes.

    Currently escapes ``_`` and ``^``.

    Args:
        text: The input string to escape.  Non-string values are returned
            unchanged.

    Returns:
        The escaped string, or ``text`` unmodified if it is not a
        ``str``.
    """

    if not isinstance(text, str):
        return text

    return text.replace('_', r'\_').replace('^', r'\^')


def get_items_by_idx(data, indices):
    """Retrieve items from a sequence by a list of indices.

    Args:
        data: The input sequence (list, tuple, or ``np.ndarray``) from
            which to retrieve items.
        indices: The indices of the items to retrieve (list, tuple, or
            ``np.ndarray``).

    Returns:
        A list of items from ``data`` at the positions given by
        ``indices``.

    Raises:
        TypeError: If ``data`` or ``indices`` is not a list, tuple, or
            ``np.ndarray``.
        IndexError: If any index in ``indices`` is out of range and the
            NumPy fancy-indexing path fails.
    """

    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError(f"Expected data to be a sequence, got {type(data)}")
    if not isinstance(indices, (list, tuple, np.ndarray)):
        raise TypeError(f"Expected indices to be a sequence, got {type(indices)}")

    try:
        return np.asanyarray(data)[np.asanyarray(indices)].tolist()
    except IndexError:
        raise [data[i] for i in indices]


def generate_asymmetric_gaussian(mean, std_left, std_right, size):
    """Generate samples from an asymmetric Gaussian distribution.

    Draws standard normal samples and rescales negative deviations by
    ``std_left`` and positive deviations by ``std_right``, then shifts
    the result to ``mean``.

    Args:
        mean: The mean (centre) of the distribution.
        std_left: Standard deviation applied to samples below the mean.
        std_right: Standard deviation applied to samples above the mean.
        size: Number of samples to generate.

    Returns:
        A ``np.ndarray`` of length ``size`` drawn from the asymmetric
        Gaussian distribution.
    """

    samples = np.random.randn(size)

    mask_left = samples <= 0
    mask_right = samples > 0

    samples[mask_left] *= std_left
    samples[mask_right] *= std_right

    return samples + mean


def format_message(msg, min_width=30):
    """Format a message as a bordered box string.

    Args:
        msg: The message to format.  A ``str`` is split on newlines; a
            list or tuple has each element converted to a string.  Any
            other type is converted with ``str()``.
        min_width: Minimum total width of the box interior.  Default is
            ``30``.

    Returns:
        A multi-line string with ``+---+`` top and bottom borders and
        ``| ... |`` side borders around the message content.  Returns an
        empty string if no non-empty lines are found.
    """

    if isinstance(msg, str):
        lines = [line.strip() for line in msg.split('\n') if line.strip()]
    elif isinstance(msg, (list, tuple)):
        lines = [str(line).strip() for line in msg if str(line).strip()]
    else:
        lines = [str(msg)]

    if not lines:
        return ""

    content_width = max(max(len(line) for line in lines) + 2, min_width)

    horizontal_border = f"+{'-' * content_width}+"

    formatted_lines = [f"| {line.ljust(content_width - 2)} |" for line in lines]

    formatted_msg = "\n".join([
        "",
        horizontal_border,
        *formatted_lines,
        horizontal_border])

    return formatted_msg
