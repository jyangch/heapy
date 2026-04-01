import json
import warnings
import numpy as np
from pathlib import Path
from itertools import zip_longest
from datetime import datetime, date

from .significance import pgsig, ppsig


def transpose_2d_matrix(data):
    """
    Transpose a 2D list (matrix).
    
    Parameters:
    data (list of list): A 2D list to be transposed.
    
    Returns:
    list of list: The transposed 2D list.
    """

    if not data or not data[0]:
        return data

    return list(map(list, zip(*data)))


def pad_2d_matrix(data, fillvalue='-'):
    """
    Pad a 2D list (matrix) with a specified fill value 
    to ensure all rows have the same length.

    Parameters:
    data (list of list): A 2D list to be padded.
    fillvalue: The value to use for padding shorter rows. Default is '-'.
    
    Returns:
    list of list: The padded 2D list, or False if the input is not a 2D list.
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
    """
    Remove all occurrences of a specified value from a 2D list (matrix).
    
    Parameters:
    data (list of list): A 2D list from which to remove the specified value.
    popvalue: The value to be removed from the 2D list. Default is '-'.
    
    Returns:
    list of list: The 2D list with the specified value removed, 
    or False if the input is not a 2D list.
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
    """
    Compute the intersection of two lists of intervals A and B.
    
    Parameters:
    ----------
    A : list of intervals [[a1_start, a1_end], [a2_start, a2_end], ...]
    B : list of intervals [[b1_start, b1_end], [b2_start, b2_end], ...]
    
    Returns:
    -------
    list of intervals representing the intersection of A and B
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
    """
    Compute the union of a list of intervals.
    
    Parameters:
    ----------
    bins : list of intervals [[start1, end1], [start2, end2], ...]
    
    Returns:
    -------
    list of intervals representing the union of the input intervals
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
    """
    Rebin the input data based on the specified statistical criteria.

    Parameters:
    ----------
    bins : list of intervals [[start1, end1], [start2, end2], ...]
        The input bins to be rebinned.
    stat : str
        The statistical method to use ('gstat', 'cstat', 'pgstat', or None).
    cts : array-like
        The counts for each bin.
    cts_err : array-like, optional
        The errors for the counts. Required if stat is 'gstat'.
    bcts : array-like, optional
        The background counts. Required if stat is 'cstat' or 'pgstat'.
    bcts_err : array-like, optional
        The errors for the background counts. Required if stat is 'pgstat'.
    min_sigma : float, optional
        The minimum sigma threshold for rebinning. Default is -inf.
    min_evt : int, optional
        The minimum event threshold for rebinning. Default is 0.
    max_bin : int, optional
        The maximum number of bins to combine. Default is inf.
    backscale : float, optional
        The backscale factor for background counts. Default is 1.

    Returns:
    -------
    new_bins : np.ndarray
        The rebinned intervals.
    new_cts : np.ndarray
        The rebinned counts.
    new_cts_err : np.ndarray
        The errors for the rebinned counts.
    new_bcts : np.ndarray
        The rebinned background counts.
    new_bcts_err : np.ndarray
        The errors for the rebinned background counts.
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
    """
    Rebin the input data for multiple sets of counts and background counts 
    based on the specified statistical criteria.
    
    Parameters
    ----------
    bins : np.ndarray
        The input bins.
    stat_list : list
        The list of statistical methods for each set of counts.
    cts_list : list
        The list of counts arrays.
    cts_err_list : list, optional
        The list of errors for the counts arrays.
    bcts_list : list, optional
        The list of background counts arrays.
    bcts_err_list : list, optional
        The list of errors for the background counts arrays.
    min_sigma_list : list, optional
        The list of minimum sigma values for each set of counts.
    min_evt_list : list, optional
        The list of minimum event values for each set of counts.
    backscale_list : list, optional
        The list of backscale values for each set of counts.

    Returns
    -------
    new_bins : np.ndarray
        The rebinned intervals.
    new_cts_list : list of np.ndarray
        The list of rebinned counts arrays.
    new_cts_err_list : list of np.ndarray
        The list of errors for the rebinned counts arrays.
    new_bcts_list : list of np.ndarray
        The list of rebinned background counts arrays.
    new_bcts_err_list : list of np.ndarray
        The list of errors for the rebinned background counts arrays.
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
    """
    Convert a boolean mask into a list of time intervals 
    where the mask is selection_value.
    
    Parameters:
    ----------
    mask : np.array(dtype=bool)
    times : np.array
    selection_value : bool
    
    Returns:
    -------
    list of tuples (t_start, t_stop) where mask is selection_value
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
    """
    Scale a sequence to the range [0, 1].
    
    Parameters:
    ----------
    seq : array-like
    seq_min : float, optional
        Minimum value to use for scaling. 
        If None, the minimum of seq will be used.
        
    Returns:
    -------
    np.ndarray
        Scaled sequence with values in the range [0, 1]
    """

    seq = np.asanyarray(seq)

    s_max = seq.max()
    s_min = seq.min() if seq_min is None else seq_min
    
    s_range = s_max - s_min

    if s_range == 0:
        return np.zeros_like(seq, dtype=float)
    
    return (seq - s_min) / s_range


def format_err_latex(value, low, upp, precision=2):
    """
    Format a value with asymmetric errors into a LaTeX string.
    
    Parameters:
    ----------
    value : float
        The central value.
    low : float
        The lower error (value - low).
    upp : float
        The upper error (upp - value).
    precision : int, optional
        Number of decimal places to include in the output. Default is 2.
        
    Returns:
    -------
    str
        A LaTeX-formatted string representing the value with asymmetric errors
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
    """
    Add backslashes before special characters in a string for LaTeX formatting.
    
    Parameters:
    ----------
    text : str
        The input string to be formatted.
        
    Returns:
    -------
    str
        The input string with backslashes added before 
        special characters for LaTeX formatting
    """
    
    if not isinstance(text, str):
        return text
    
    return text.replace('_', r'\_').replace('^', r'\^')


def get_items_by_idx(data, indices):
    """
    Retrieve items from a sequence (list, tuple, or numpy array) based on a list of indices.
    
    Parameters:
    ----------
    data : list, tuple, or np.ndarray
        The input sequence from which to retrieve items.
    indices : list, tuple, or np.ndarray
        The indices of the items to retrieve from the data sequence.
        
    Returns:
    -------
    list
        A list of items from the data sequence corresponding to the provided indices.
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
    """
    Generate samples from an asymmetric Gaussian distribution.
    
    Parameters:
    ----------
    mean : float
        The mean of the distribution.
    std_left : float
        The standard deviation for the left side of the distribution (values less than the mean).
    std_right : float
        The standard deviation for the right side of the distribution (values greater than the mean).
    size : int
        The number of samples to generate.
        
    Returns:
    -------
    np.ndarray
        An array of samples drawn from the specified asymmetric Gaussian distribution.
    """

    samples = np.random.randn(size)

    mask_left = samples <= 0
    mask_right = samples > 0
    
    samples[mask_left] *= std_left
    samples[mask_right] *= std_right
    
    return samples + mean


def format_boxed_message(msg, min_width=30):
    """
    Format a message as a boxed string with borders.

    Parameters:
    ----------
    msg : str, list, or tuple
        The message to be formatted. Can be a single string or a list/tuple of strings.
    min_width : int, optional
        The minimum width of the boxed message. Default is 30.

    Returns:
    -------
    str
        The formatted boxed message as a string
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


class JsonEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that can handle numpy data types and datetime objects.
    """
    
    def default(self, obj):
        if isinstance(obj, (np.integer, np.bool_)):
            return int(obj) if isinstance(obj, np.integer) else bool(obj)
            
        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, (np.ndarray, set)):
            return obj.tolist() if isinstance(obj, np.ndarray) else list(obj)

        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        return super().default(obj)


def json_dump(data, filepath, indent=4, ensure_ascii=False):
    """
    Dump data to a JSON file with support for numpy data types and datetime objects.
    
    Parameters:
    ----------
    data : any
        The data to be dumped to JSON. Can include numpy data types and datetime objects.
    filepath : str or Path
        The path to the JSON file where the data will be saved.
    indent : int, optional
        The number of spaces to use for indentation in the JSON file. Default is 4.
    ensure_ascii : bool, optional
        Whether to escape non-ASCII characters in the JSON output. Default is False. 
    """
    
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, cls=JsonEncoder)
