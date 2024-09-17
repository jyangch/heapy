import json
import numpy as np
from .significance import pgsig, ppsig


def poisson_binsize(points, time_edge, p):
    
    mean_value = float(points) / (time_edge[1] - time_edge[0])

    if mean_value == 0:
        binsize_p = time_edge[1] - time_edge[0]
        return binsize_p
    else:
        binsize_p = np.log(1 - p) / (-mean_value)
        if binsize_p > (time_edge[1] - time_edge[0]):
            binsize_p = time_edge[1] - time_edge[0]
            return binsize_p
        elif binsize_p > 1:
            return np.ceil(binsize_p)
        elif binsize_p < 1:
            return binsize_p


def ch_to_energy(pi, ch, e1, e2):
    
    energy = np.zeros_like(pi)
    for i, chi in enumerate(ch):
        chi_idx = np.where(pi == chi)[0]
        chi_energy = energy_of_ch(len(chi_idx), e1[i], e2[i])
        energy[chi_idx] = chi_energy
        
    return energy


def energy_of_ch(n, e1, e2):
    
    energy_arr = np.random.random_sample(n)
    energy = e1 + (e2 - e1) * energy_arr
    
    return energy


def intersection(A, B):
    
    #A = [[0,2], [5,10], [13,23], [24,25]]
    #B = [[1,5], [8,12], [15,24], [25,26]]
    #--------------
    #sort A and B
    #--------------
    
    A1 = np.array([i[-1] for i in A])
    B1 = np.array([i[-1] for i in B])
    A = np.array(A)[np.argsort(A1)]
    B = np.array(B)[np.argsort(B1)]

    i, j = 0, 0
    res = []
    while i < len(A) and j < len(B):
        a1, a2 = A[i][0], A[i][1]
        b1, b2 = B[j][0], B[j][1]
        if b2 > a1 and a2 > b1:
            res.append([max(a1, b1), min(a2, b2)])
        if b2 < a2: j += 1
        else: i += 1
        
    return res


def union(bins):
    
    if len(bins) == 0:
        return []
    
    #--------------
    #sort bins
    #--------------
    
    bins1 = np.array([bin[0] for bin in bins])
    bins = np.array(bins)[np.argsort(bins1)]
    bins = bins.tolist()

    res = [bins[0]]
    for i in range(1, len(bins)):
        a1, a2 = res[-1][0], res[-1][1]
        b1, b2 = bins[i][0], bins[i][1]
        if b2 >= a1 and a2 >= b1:
            res[-1] = [min(a1, b1), max(a2, b2)]
        else: res.append(bins[i])

    return res


def rebin(bins, 
          cts, 
          bcts, 
          bcts_err, 
          min_sigma=None, 
          min_evt=None, 
          max_bin=None,
          backscale=1, 
          stat=None):
    
    bins, cts, bcts = list(bins), list(cts), list(bcts)
    new_bins, new_cts, new_bcts, new_bcts_err = [], [], [], []
    cc, cb, cb_err, j, k = 0, 0, 0, 0, 0
    
    if min_sigma is None:
        min_sigma = -np.inf
    
    if min_evt is None:
        min_evt = 0
        
    if max_bin is None:
        max_bin = np.inf
    
    for i in range(len(cts)):
        ci = cts[i]
        bi = bcts[i]
        bi_err = bcts_err[i]
        cc += ci
        cb += bi
        cb_err = np.sqrt(cb_err ** 2 + bi_err ** 2)
        
        if stat is None: stat = 'pgstat'
        if stat == 'pgstat':
            sigma = pgsig(cc, cb * backscale, cb_err * backscale)
        elif stat == 'cstat':
            sigma = ppsig(cc, cb * backscale, 1)
        else:
            raise AttributeError(f'unsupported stat: {stat}')
        
        evt = cc - cb * backscale
        
        if ((sigma >= min_sigma) and (evt >= min_evt)) or ((j-i+1) == max_bin):
            new_bins.append([bins[j][0], bins[i][1]])
            new_cts.append(cc)
            new_bcts.append(cb)
            new_bcts_err.append(cb_err)
            cc, cb, cb_err = 0, 0, 0
            j = i + 1
            k += 1
            
        if (i == len(cts) - 1) and ((sigma < min_sigma) or (evt < min_evt)) and (j-i+1) < max_bin:
            if k >= 1:
                new_bins[k-1][1] = bins[i][1]
                new_cts[k-1] = new_cts[k-1] + cc
                new_bcts[k-1] = new_bcts[k-1] + cb
                new_bcts_err[k-1] = np.sqrt(new_bcts_err[k-1] ** 2 + cb_err ** 2)
            else:
                new_bins.append([bins[j][0], bins[i][1]])
                new_cts.append(cc)
                new_bcts.append(cb)
                new_bcts_err.append(cb_err)
                    
    new_bins = np.array(new_bins)
    new_cts = np.array(new_cts)
    new_bcts = np.array(new_bcts)
    new_bcts_err = np.array(new_bcts_err)
            
    return new_bins, new_cts, new_bcts, new_bcts_err


def scale_of_one(seq, seq_min=None):
    
    seq_max = max(seq)
    if seq_min is None:
        seq_min = min(seq)
    if seq_max == seq_min:
        scale_seq = seq - seq_min
    else:
        scale_seq = (seq - seq_min) / (seq_max - seq_min)
        
    return scale_seq


def err_latex(value, low, upp):
    value = np.float128(value)
    low = np.float128(low)
    upp = np.float128(upp)

    value_str = '%.2e' % value
    e_index = value_str.find('e')
    if e_index == -1:
        print('value is not infinite: %s'%value)
        expr = '$%s_{-%s}^{+%s}$'%(value_str, value_str, value_str)
    else:
        v = value_str[:e_index]
        times = int(value_str[(e_index + 1):])
        l = '%.2f'%(low/(10**times))
        u = '%.2f'%(upp/(10**times))
        expr = '$'+v+'_{-'+l+'}^{+'+u+r'}\times10^{%d}$'%times
    return expr


def add_slash(string_):
    list_ = list(string_)
    strint__ = ''
    for i in list_:
        if i == '_':
            strint__ = strint__ + r'\_'
        elif i == '^':
            strint__ = strint__ + r'\^'
        else:
            strint__ = strint__ + i
    return strint__


def listidx(list, idx):
    assert type(list) is list, '"list" should be a list!'
    assert type(idx) is list, '"idx" should be a list!'
    list_ = [list[i] for i in idx]
    return list_


def asym_gaus_gen(mean, std1, std2, size):
    
    sam = np.random.randn(size)
    sam[sam <= 0] = sam[sam <= 0] * std1
    sam[sam > 0] = sam[sam > 0] * std2
    sam = sam + mean
    
    return sam


def msg_format(msg):
    
    msg_ = ''
    uppline = '\n+' + '-' * 48 + '+\n'
    lowline = '+' + '-' * 48 + '+'
    if type(msg) is list:
        msg_ += ('\n'.join(msg) + '\n')
    elif type(msg) is str:
        for mi in msg.split('\n'):
            mi = mi.strip()
            if mi != '':
                msg_ += (' ' + mi + '\n')
    msg_format = uppline + msg_ + lowline
    
    return msg_format
        
        
class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NpEncoder, self).default(obj)
