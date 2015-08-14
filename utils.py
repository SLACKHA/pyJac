"""Module containing utility functions.
"""

# Standard libraries
import os
import errno
from math import log10, floor

# local includes
import CUDAParams

line_start = '  '
comment = dict(c='//', cuda='//',
                fortran='!', matlab='%')

# list of supported languages
langs = ['c', 'cuda', 'fortran', 'matlab']

# source code file extensions based on language
file_ext = dict(c='.c', cuda='.cu',
                fortran='.f90', matlab='.m'
                )

# header extensions based on language
header_ext = dict(c='.h', cuda='.cuh')

# line endings dependent on language
line_end = dict(c=';\n', cuda=';\n',
                fortran='\n', matlab=';\n'
                )

# exp10 functions for various languages
exp_10_fun = dict(c="pow(10.0, ", cuda='exp10(',
                  fortran='exp(log(10) * ', matlab='exp(log(10.0) * ')

# the characters to format an index into an array per language
array_chars = dict(c="[{}]", cuda="[{}]",
                   fortran="({})", matlab="({})")


def round_sig(x, sig=8):
    if x == 0:
        return 0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def read_str_num(string, sep=None):
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
    return [seq[i: i + length] for i in range(0, len(seq), length)]


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


def get_array(lang, name, index, twod=None):
    """
    Given a language and an index, returns the proper string index formatted into the appropriate array characters (
    e.g. [] or ())

    Parameters
    ----------
    lang : str
        One of the accepted languages
    name : str
        The name of the array
    index : int
        The index to format
    twod : int, optional
        If not None and the lang is fortan or matlab this will be formatted as a second index in the array
    """

    if lang in ['fortran', 'matlab']:
        if twod is not None:
            return name +'({}, {})'.format(index + 1, twod + 1)
        return name + array_chars[lang].format(index + 1)
    return name + array_chars[lang].format(index)


def get_index(lang, index):
    """
    Given an integer index this function will return the proper string version of the index
    based on the language and other considerations

    Parameters
    ----------
    lang : str
        One of the supported languages
    index : int

    Returns
    -------
    str
        The string corresponding to the correct index to be formatted into the code
    """

    retval = None
    if lang in ['fortran', 'matlab']:
        return str(index + 1)
    if lang in ['c', 'cuda']:
        return str(index)

def reassign_species_lists(reacs, specs):
    """
    Given a list of ReacInfo, and SpecInfo's, this method will update the ReacInfo's
    reactants / products / third body list to integers representing the species' index in the list
    """

    species_map = {sp.name: i for i, sp in enumerate(specs)}
    for rxn in reacs:
        rxn.reac = [species_map[sp] for sp in rxn.reac]
        rxn.prod = [species_map[sp] for sp in rxn.prod]
        rxn.thd_body_eff = [(species_map[thd[0]], thd[1]) for thd in rxn.thd_body_eff]
        if rxn.pdep_sp:
            rxn.pdep_sp = species_map[rxn.pdep_sp]

def is_integer(val):
    """Returns whether a value is an integer regardless of whether it's a float or it's an int"""
    if isinstance(val, int):
        return True
    elif isinstance(val, float) and val.is_integer():
        return True
    else:
        #last ditch effort
        try:
            return int(val) == float(val)
        except:
            return False
