import json
import numpy as np
from .significance import pgsig, ppsig


def intersection(A, B):
    
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
          stat, 
          cts, 
          cts_err=None,
          bcts=None, 
          bcts_err=None, 
          min_sigma=None, 
          min_evt=None, 
          max_bin=None,
          backscale=None):
    
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
        cts_err = np.zeros_like(cts).astype(float)
        
    if bcts is None:
        bcts = np.zeros_like(cts).astype(float)
        
    if bcts_err is None:
        bcts_err = np.zeros_like(cts).astype(float)
        
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
            cts_err_list[n] = np.zeros_like(cts_list[n]).astype(float)
            
        if bcts_list[n] is None:
            bcts_list[n] = np.zeros_like(cts_list[n]).astype(float)
            
        if bcts_err_list[n] is None:
            bcts_err_list[n] = np.zeros_like(cts_list[n]).astype(float)
            
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


class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, set):
            return list(obj)
        else:
            return super(JsonEncoder, self).default(obj)
        
        
def json_dump(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, cls=JsonEncoder)
