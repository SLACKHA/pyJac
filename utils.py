"""Module containing utility functions.
"""

# Standard libraries
import os
import errno

# list of supported languages
#langs = ['c', 'cuda', 'fortran', 'matlab']
langs = ['c', 'cuda']

# source code file extensions based on language
file_ext = dict(c = '.c', cuda = '.cu',
                fortran = '.f90', matlab = '.m'
                )

# line endings dependent on language
line_end = dict(c = ';\n',
                cuda = ';\n',
                fortran = '\n',
                matlab = ';\n'
                )

#exp10 functions for various languages
exp_10_fun = dict(c = "pow(10.0, ",
                  cuda = 'exp10(',
                  fortran = 'exp(log(10.0) * ',
                  matlab = 'exp(log(10.0) * '
                  )

def read_str_num(string, sep = None):
    """Returns a list of floats pulled from a string.

    Delimiter is optional; if not specified, uses whitespace.

    Parameters
    ----------
    string : str
        String to be parsed.
    sep : str, optional
        Delimiter (default is None, which means consecutive whitespace).

    Returns
    -------
    list of float
        Floats separated by `sep` in `string`.

    """

    # separate string into space-delimited strings of numbers
    num_str = string.split(sep)
    return [float(n) for n in num_str]


def split_str(seq, length):
    """Separate a string seq into length-sized pieces.

    Parameters
    ----------
    seq : str
        String containing sequence of smaller strings of constant length.
    length : int
        Length of individual sequences.

    Returns
    -------
    list of str
        List of strings of length `length` from `seq`.

    """
    return [seq[i : i + length] for i in range(0, len(seq), length)]


def create_dir(path):
    """Creates a new directory based on input path.

    No error if path already exists, but other error is reported.

    Parameters
    ----------
    path : str
        Path of directory to be created

    Returns
    -------
    None

    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
