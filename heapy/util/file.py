"""File-system utilities for reading, writing, copying, and searching files.

This module provides convenience wrappers around standard library I/O
operations, including displaying text files, copying or removing files and
directories with warning-based error handling, searching for files by glob
pattern or keyword list, and saving/loading 2-D data tables to and from
formatted plain-text files.
"""

import os
from pathlib import Path
import re
import shutil
import warnings

import numpy as np

from .data import pad_2d_matrix, transpose_2d_matrix


def cat_file(file_path, encoding='utf-8'):
    """Print the contents of a text file to the console.

    Args:
        file_path: Path to the text file to be displayed.
        encoding: Character encoding used to open the file.
            Defaults to ``'utf-8'``.

    Raises:
        FileNotFoundError: If ``file_path`` does not refer to an existing file.
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    try:
        with open(file_path, encoding=encoding) as f:
            for line in f:
                print(line.rstrip())
    except UnicodeDecodeError:
        print(f"Error: Could not decode '{file_path}' using {encoding}. Try a different encoding.")
    except Exception as e:
        print(f'An unexpected error occurred: {e}')


def copy_file(src, dst):
    """Copy a file or directory from src to dst.

    Copies the source path to the destination. Directories are copied
    recursively. If ``src`` does not exist, a ``UserWarning`` is issued and
    no destination is created.

    Args:
        src: Source file or directory path to copy.
        dst: Destination path to copy the file or directory to.
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
            warnings.warn(f'Copy failed from {src} to {dst}: {e}', UserWarning, stacklevel=2)

    else:
        msg = f'FILE NOT FOUND: {src}'
        warnings.warn(msg, UserWarning, stacklevel=2)


def remove_file(file_path):
    """Remove a file at the specified path.

    Issues a ``UserWarning`` if the file does not exist, a ``RuntimeWarning``
    if the file cannot be removed due to a permission error, and a
    ``RuntimeWarning`` for any other removal failure.

    Args:
        file_path: Path to the file to be removed.
    """

    path = Path(file_path)

    if path.is_file():
        try:
            path.unlink()
        except PermissionError:
            warnings.warn(
                f'PERMISSION DENIED: Could not remove {file_path}', RuntimeWarning, stacklevel=2
            )
        except Exception as e:
            warnings.warn(f'FAILED TO REMOVE {file_path}: {e}', RuntimeWarning, stacklevel=2)

    else:
        msg = f'FILE NOT FOUND: {file_path}'
        warnings.warn(msg, UserWarning, stacklevel=2)


def find_file(root_dir, pattern, recursive=True):
    """Search for files matching a pattern within a directory tree.

    Accepts either a glob-style string (with ``*`` as wildcard) or a list of
    keyword strings. All keywords must appear in the file name for a file to
    be considered a match. Hidden files (names starting with ``.``) are
    excluded.

    Args:
        root_dir: Root directory to search within.
        pattern: Pattern to match file names against. A string may contain
            ``'*'`` as a wildcard; all non-wildcard substrings must be present
            in the file name. A list of strings requires every element to be
            present in the file name.
        recursive: If ``True``, search subdirectories recursively.
            Defaults to ``True``.

    Returns:
        Sorted list of absolute path strings for all matching files, or
        ``None`` if no matches are found, the directory does not exist, or
        an error occurs during the search.
    """

    root = Path(root_dir)
    if not root.exists():
        warnings.warn(f'Path not found: {root_dir}', UserWarning, stacklevel=2)
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
            msg = f'No files found matching: {pattern} in {root_dir}'
            warnings.warn(msg, UserWarning, stacklevel=2)
            return None

        matches.sort()

        return matches

    except Exception as e:
        warnings.warn(f'Error during search: {e}', UserWarning, stacklevel=2)
        return None


def savetxt(file, data, fmt=None, trans=False, header=None):
    """Save a 2-D array or list to a formatted plain-text file.

    Writes ``data`` to ``file`` with column-aligned, fixed-width formatting.
    An optional header row and per-element format strings are supported. The
    data can be transposed before writing.

    Args:
        file: Path to the output file.
        data: 2-D array or list of lists to save. A 1-D input is promoted to
            a single-row 2-D structure with a warning.
        fmt: Format specifier(s) for the data elements. ``None`` formats
            everything as strings. A single string applies the same format to
            all elements. A 1-D list of strings of length equal to the number
            of columns (after optional transposition) applies one format per
            column. A 2-D list must match the shape of the (possibly
            transposed) data. Defaults to ``None``.
        trans: If ``True``, transpose ``data`` (and ``fmt`` if 2-D) before
            writing. Defaults to ``False``.
        header: Column header strings. Must have the same length as the number
            of columns (after optional transposition). Defaults to ``None``.

    Returns:
        ``True`` if the file was saved successfully, ``False`` if a format or
        header validation error occurred.
    """

    if data is None or len(data) == 0:
        data = [[]]
    elif (type(data[0]) is not list) and (type(data[0]) is not np.ndarray):
        msg = 'data is 1-D array or list'
        warnings.warn(msg, UserWarning, stacklevel=2)
        data = [data]

    data = pad_2d_matrix(data)
    if trans:
        data = transpose_2d_matrix(data)
    row = len(data)
    col = len(data[0])

    if col == 0:
        with open(file, 'w+') as f:
            _ = [f.write(''.join(data[r]) + '\n') for r in range(row)]
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
        if trans:
            fmt = transpose_2d_matrix(fmt)
        if len(fmt) != row or len(fmt[0]) != col:
            msg = 'the fmt shape should be same with data'
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False
    else:
        msg = 'wrong fmt'
        warnings.warn(msg, UserWarning, stacklevel=2)
        return False

    data = [
        ['--' if data[r][c] == '--' else ('%' + fmt[r][c]) % data[r][c] for c in range(col)]
        for r in range(row)
    ]

    if header is None:
        header = []
        length = [max([len(data[r][c]) for r in range(row)]) for c in range(col)]
    else:
        if len(header) != col:
            msg = 'the header length should equal to col(after trans)'
            warnings.warn(msg, UserWarning, stacklevel=2)
            return False
        else:
            header = [f'{h}' for h in header]
            length = [
                max([len(data[r][c]) for r in range(row)] + [len(header[c])]) for c in range(col)
            ]

    header = [h + ' ' * (length[c] + 4 - len(h)) for c, h in enumerate(header)]
    data = [
        [data[r][c] + ' ' * (length[c] + 4 - len(data[r][c])) for c in range(col)]
        for r in range(row)
    ]

    with open(file, 'w+') as f:
        f.write('' if len(header) == 0 else ''.join(header) + '\n')
        _ = [f.write(''.join(data[r]) + '\n') for r in range(row)]
    return True


def loadtxt(file, fmt=None, trans=False):
    """Load a plain-text file and return its contents as a 2-D list.

    Reads whitespace-delimited rows from ``file``. Each cell is automatically
    cast to ``int`` or ``float`` where possible; otherwise it remains a string.
    An optional format specifier can re-format the values after loading, and
    the result can be transposed before returning.

    Args:
        file: Path to the text file to load.
        fmt: Format specifier(s) applied to each element after type conversion.
            ``None`` skips re-formatting. A single string applies the same
            format to all elements. A 1-D list of strings of length equal to
            the number of columns applies one format per column. A 2-D list
            must match the shape of the loaded data. Defaults to ``None``.
        trans: If ``True``, transpose the loaded data before returning.
            Defaults to ``False``.

    Returns:
        Loaded data as a 2-D list of rows. Returns ``False`` if a format
        validation error occurs.
    """

    data = []
    with open(file) as f:
        for line in f:
            data.append([i for i in re.split(r'[\t\n\s]', line) if i != ''])

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
                    msg = f'format {fmt[r][c]} is not suitable for data {data[r][c]}'
                    warnings.warn(msg, UserWarning, stacklevel=2)
    if trans:
        data = transpose_2d_matrix(data)
    return data
