import os
import re
import shutil
import warnings
import numpy as np
import subprocess as sp
from pathlib import Path

from .data import transpose_2d_matrix, pad_2d_matrix


def cat_file(file_path, encoding='utf-8'):
    """
    Display the contents of a text file to the console.
    
    Parameters:
    ----------
    file_path : str
        The path to the text file to be displayed.
    encoding : str, optional
        The encoding of the text file. Default is 'utf-8'.
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    try:
        with open(file_path, mode='r', encoding=encoding) as f:
            for line in f:
                print(line.rstrip())
    except UnicodeDecodeError:
        print(f"Error: Could not decode '{file_path}' using {encoding}. Try a different encoding.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
        
def copy_file(src, dst):
    """
    Copy a file or directory from src to dst. If src does not exist, 
    a warning is issued and a marker file is created at dst.
    
    Parameters:
    ----------
    src : str
        The source file or directory to be copied.
    dst : str
        The destination path where the file or directory should be copied to.
    """
    
    src_path = Path(src)
    dst_path = Path(dst)

    if src_path.exists():
        try:
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
        except Exception as e:
            warnings.warn(f"Copy failed from {src} to {dst}: {e}", UserWarning)

    else:
        msg = f"FILE NOT FOUND: {src}"
        warnings.warn(msg, UserWarning, stacklevel=2)


def remove_file(file_path):
    """
    Remove a file at the specified path. If the file does not exist, a warning is issued. 
    If the file cannot be removed due to permissions or other issues, a warning is also issued.
    
    Parameters:
    ----------
    file_path : str
        The path to the file to be removed.
    """
    
    path = Path(file_path)
    
    if path.is_file():
        try:
            path.unlink()
        except PermissionError:
            warnings.warn(f"PERMISSION DENIED: Could not remove {file_path}", RuntimeWarning)
        except Exception as e:
            warnings.warn(f"FAILED TO REMOVE {file_path}: {e}", RuntimeWarning)

    else:
        msg = f"FILE NOT FOUND: {file_path}"
        warnings.warn(msg, UserWarning, stacklevel=2)


def find_file(root_dir, pattern, recursive=True):
    """
    Search for files in a directory that match a given pattern. 
    The pattern can be a string with wildcards or a list of keywords.
    
    Parameters:
    ----------
    root_dir : str
        The directory to search within.
    pattern : str or list of str
        The pattern to match file names against.
        If a string is provided, it can include '*' as a wildcard.
        If a list of strings is provided, all keywords must be present in the file name.
    recursive : bool, optional
        Whether to search subdirectories recursively. Default is True.
        
    Returns:
    -------
    tuple or None
        A tuple containing a list of matching file names and their corresponding paths,
        or None if no matches are found or if an error occurs.
    """
    
    root = Path(root_dir)
    if not root.exists():
        warnings.warn(f"Path not found: {root_dir}", UserWarning, stacklevel=2)
        return None

    if isinstance(pattern, str):
        keywords = [k for k in re.split(r'[*]', pattern) if k]
    else:
        keywords = list(pattern)

    search_func = root.rglob if recursive else root.glob
    matches = []

    try:
        for path in search_func('*'):
            if path.is_file():
                filename = path.name
                if not filename.startswith('.') and all(kw in filename for kw in keywords):
                    matches.append(str(path.absolute()))

        if not matches:
            msg = f"No files found matching: {pattern} in {root_dir}"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return None

        matches.sort()

        return matches

    except Exception as e:
        warnings.warn(f"Error during search: {e}", UserWarning, stacklevel=2)
        return None


def savetxt(file, data, fmt=None, trans=False, header=None):
    """
    Save a 2D array or list to a text file with specified formatting and optional header.
    
    Parameters:
    ----------
    file : str
        The path to the file where the data will be saved.
    data : array-like
        The 2D array or list to be saved.
    fmt : str or list of str, optional
        The format for each element. Default is None.
    trans : bool, optional
        Whether to transpose the data before saving. Default is False.
    header : list of str, optional
        The header for the columns. Default is None.
        
    Returns:
    -------
    bool
        True if the file was saved successfully, False otherwise.
    """
    
    if data is None:
        data = [[]]
    elif len(data) == 0:
        data = [[]]
    elif (type(data[0]) is not list) and (type(data[0]) is not np.ndarray):
        msg = 'data is 1-D array or list'
        warnings.warn(msg, UserWarning, stacklevel=2)
        data = [data]

    data = pad_2d_matrix(data)
    if trans: data = transpose_2d_matrix(data)
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
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False
        else:
            fmt = [fmt] * row
    elif (type(fmt) is list) and (type(fmt[0]) is list):
        if trans: fmt = transpose_2d_matrix(fmt)
        if len(fmt) != row or len(fmt[0]) != col:
            msg = 'the fmt shape should be same with data'
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False
    else:
        msg = 'wrong fmt'
        warnings.warn(msg, UserWarning, stacklevel=2)
        return False

    data = [['--' if data[r][c] == '--' else ('%'+fmt[r][c])%data[r][c] for c in range(col)] for r in range(row)]

    if header is None:
        header = []
        length = [max([len(data[r][c]) for r in range(row)]) for c in range(col)]
    else:
        if len(header) != col:
            msg = 'the header length should equal to col(after trans)'
            warnings.warn(msg, UserWarning, stacklevel=2)
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
    """
    Load data from a text file and return it as a 2D list, with optional formatting and transposition.
    
    Parameters:
    ----------
    file : str
        The path to the file to be loaded.
    fmt : str or list of str, optional
        The format for each element. Default is None.
    trans : bool, optional
        Whether to transpose the data after loading. Default is False.
        
    Returns:
    -------
    list
        The loaded 2D list.
    """
    
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
                warnings.warn(msg, UserWarning, stacklevel=2)
                return False
            else:
                fmt = [fmt] * row
        elif (type(fmt) is list) and (type(fmt[0]) is list):
            if len(fmt) != row or len(fmt[0]) != col:
                msg = 'the fmt shape should be same with data'
                warnings.warn(msg, UserWarning, stacklevel=2)
                return False
        else:
            msg = 'wrong fmt'
            warnings.warn(msg, UserWarning, stacklevel=2)

        for r in range(row):
            for c in range(col):
                try:
                    data[r][c] = ('%' + fmt[r][c]) % data[r][c]
                except TypeError:
                    msg = 'format %s is not suitable for data %s'%(fmt[r][c], data[r][c])
                    warnings.warn(msg, UserWarning, stacklevel=2)
    if trans: data = transpose_2d_matrix(data)
    return data
