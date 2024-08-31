import os
import re
import warnings
import numpy as np
import subprocess as sp
from .data import msg_format


def transpose(data):
    if len(data[0]) == 0:
        return data
    trans_data = [[row[col] for row in data] for col in range(len(data[0]))]
    return trans_data


def fill2D(data, fill_value='--'):
    if type(data[0]) is not list and (type(data[0]) is not np.ndarray):
        msg = 'data should be 2-D list or array'
        warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
        return False
    nrow = len(data)
    ncol = max([len(data[i]) for i in range(nrow)])
    data = [[row[i] if i < len(row) else fill_value for i in range(ncol)] for row in data]
    return data


def pop2D(data, pop_value='--'):
    if type(data[0]) is not list and (type(data[0]) is not np.ndarray):
        msg = 'data should be 2-D list or array'
        warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
        return False
    nrow = len(data)
    ncol = len(data[0])
    data = [[data[r][c] for c in range(ncol) if data[r][c] != pop_value] for r in range(nrow)]
    return data


def savetxt(file, data, fmt=None, trans=False, header=None):
    if data is None:
        data = [[]]
    elif len(data) == 0:
        data = [[]]
    elif (type(data[0]) is not list) and (type(data[0]) is not np.ndarray):
        msg = 'data is 1-D array or list'
        warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
        data = [data]

    data = fill2D(data)
    if trans: data = transpose(data)
    row = len(data)
    col = len(data[0])

    if col == 0:
        f = open(file, 'w+')
        _ = [f.write(''.join(data[r]) + '\n') for r in range(row)]
        f.close()
        return True

    if fmt is None:
        fmt = [['s'] * col] * row
    elif type(fmt) is str:
        fmt = [[fmt] * col] * row
    elif (type(fmt) is list) and (type(fmt[0]) is not list):
        if len(fmt) != col:
            msg = 'the fmt lenth should equal to col(after trans)'
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
            return False
        else:
            fmt = [fmt] * row
    elif (type(fmt) is list) and (type(fmt[0]) is list):
        if trans: fmt = transpose(fmt)
        if len(fmt) != row or len(fmt[0]) != col:
            msg = 'the fmt shape should be same with data'
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
            return False
    else:
        msg = 'wrong fmt'
        warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
        return False

    data = [['--' if data[r][c] == '--' else ('%'+fmt[r][c])%data[r][c] for c in  range(col)] for r in range(row)]

    if header is None:
        header = []
        length = [max([len(data[r][c]) for r in range(row)]) for c in range(col)]
    else:
        if len(header) != col:
            msg = 'the header lenth should equal to col(after trans)'
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
            return False
        else:
            header = ['%s' % h for h in header]
            length = [max([len(data[r][c]) for r in range(row)] + [len(header[c])]) for c in range(col)]

    header = [h + ' ' * (length[c] + 4 - len(h)) for c, h in enumerate(header)]
    data = [[data[r][c] + ' ' * (length[c] + 4 - len(data[r][c])) for c in range(col)] for r in range(row)]

    f = open(file, 'w+')
    f.write('' if len(header) == 0 else ''.join(header) + '\n')
    _ = [f.write(''.join(data[r]) + '\n') for r in range(row)]
    f.close()
    return True


def loadtxt(file, fmt=None, trans=False):
    data = []
    f = open(file, 'r')
    for line in f:
        data.append([i for i in re.split(r'[\t\n\s]', line) if i != ''])
    f.close()

    row = len(data)
    col = len(data[0])

    if col == 0:
        return data

    for r in range(row):
        for c in range(col):
            try:
                data[r][c] = int(data[r][c])
            except ValueError:
                try:
                    data[r][c] = float(data[r][c])
                except ValueError:
                    data[r][c] = data[r][c]

    if fmt is not None:
        if type(fmt) is str:
            fmt = [[fmt] * col] * row
        elif (type(fmt) is list) and (type(fmt[0]) is not list):
            if len(fmt) != col:
                msg = 'the fmt lenth should equal to col(after trans)'
                warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
                return False
            else:
                fmt = [fmt] * row
        elif (type(fmt) is list) and (type(fmt[0]) is list):
            if len(fmt) != row or len(fmt[0]) != col:
                msg = 'the fmt shape should be same with data'
                warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
                return False
        else:
            msg = 'wrong fmt'
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)

        for r in range(row):
            for c in range(col):
                try:
                    data[r][c] = ('%' + fmt[r][c]) % data[r][c]
                except TypeError:
                    msg = 'format %s is not suitable for data %s'%(fmt[r][c], data[r][c])
                    warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
    if trans: data = transpose(data)
    return data


def cat(file):
    with open(file) as f_obj:
        for line in f_obj:
            print(line.rstrip())


def copy(f1, f2):
    if os.path.isfile(f1):
        sp.call('cp -rf ' + f1 + ' ' + f2, shell=True)
    else:
        msg = 'FILE NOT FOUND: ' + f1
        warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
        f2_name = f2.split('/')[-1]
        sp.call('touch ' + os.path.dirname(f2) + '/%s_Not_Found.txt'%f2_name, shell=True)


def searchfile(keywords, root):
    filelist = []
    for root, dirs, files in os.walk(root):
        for name in files:
            filelist.append(os.path.join(root, name))
    try:
        filesname = []
        fileslocation = []
        for i in filelist:
            if os.path.isfile(i):
                flag = True
                filename = os.path.split(i)[1]
                for keyword in keywords:
                    if keyword not in filename:
                        flag = False
                if flag and filename[0] != '.':
                    filesname.append(filename)
                    fileslocation.append(i)
        return sorted(filesname), sorted(fileslocation)
    except:
        msg = 'Something wrong'
        warnings.warn(msg_format(msg), UserWarning, stacklevel=2)


def findfile(dir1, feature):
    if (os.path.exists(dir1)):
        dirnames = os.listdir(dir1)
        filelist = []
        fil_number = 0
        fil_result_number = 0
        featurelist = [i for i in re.split('[*]', feature) if i != '']
        for_number = len(featurelist)
        fileresult = [[] for i in range(for_number)]
        for eve in range(for_number):
            if (eve == 0):
                fe_number = len(featurelist[eve])
                for sample in dirnames:
                    if (os.path.isfile(dir1 + sample)):
                        filelist.append(sample)
                        fil_number = fil_number + 1
                if (fil_number != 0):
                    for i in filelist:
                        i_number = len(i)
                        n = i_number - fe_number + 1
                        for j in range(n):
                            if (i[j:j + fe_number] == featurelist[eve]):
                                fileresult[eve].append(i)
                                fil_result_number = fil_result_number + 1
                                break
                    if (fil_result_number == 0):
                        msg = 'do not find any file that has the feature with [' + feature + ']'
                        warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
                        return []
                    else:
                        fil_result_number = 0
                else:
                    msg = 'there is no file in this dir'
                    warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
                    return []
            else:
                fe_number = len(featurelist[eve])
                for i in fileresult[eve - 1]:
                    i_number = len(i)
                    n = i_number - fe_number + 1
                    for j in range(n):
                        if (i[j:j + fe_number] == featurelist[eve]):
                            fileresult[eve].append(i)
                            fil_result_number = fil_result_number + 1
                            break
                if (fil_result_number == 0):
                    msg = 'do not find any file that has the feature with [' + feature + ']'
                    warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
                    return []
                else:
                    fil_result_number = 0
        return fileresult[for_number - 1]
    else:
        msg = 'do not find the dir named [' + dir1 + ']'
        warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
        return False


def findcheck(file, path):
    if file is not False:
        if len(file) > 0:
            if len(file) > 1:
                msg = 'may find needless file:\n%s\nwill only keep last one' % '\n'.join(file)
                warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
                file = [name for name in file if name[0] != '.']
            filepath = path + file[-1]
        else:
            msg = 'can not find this file'
            warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
            filepath = None
    else:
        filepath = None
    return filepath


def getfilelist(dir1):
    if (os.path.exists(dir1)):
        dirnames = os.listdir(dir1)
        dirlist = []
        dir_number = 0
        filelist = []
        file_number = 0
        for sample in dirnames:
            if (os.path.isfile(dir1 + sample)):
                filelist.append(sample)
                file_number = file_number + 1
            elif(os.path.isdir(dir1 + sample)):
                dirlist.append(sample)
                dir_number = dir_number + 1
        c = {dir1: filelist}
        if (dir_number != 0):
            for sname in dirlist:
                dir2 = dir1+sname+'/'
                k = getfilelist(dir2)
                c.update(k)
        return c
    else:
        msg = 'do not find the dir named [' + dir1 + ']'
        warnings.warn(msg_format(msg), UserWarning, stacklevel=2)
        return False