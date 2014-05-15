"""Module containing utility functions.
"""

# list of supported languages
langs = ['c', 'cuda', 'fortran', 'matlab']

# source code file extensions based on language
file_ext = dict(c = '.c', cuda = '.cu',
                fortran = '.f90', matlab = '.m'
                )

# line endings dependent on language
line_end = dict(c = ';\n', cuda = ';\n',
                fortran = '\n', matlab = ';\n'
                )

def read_str_num(string, sep = None):
    """Returns a list of floats pulled from a string.
    
    Delimiter is optional; if not specified, uses whitespace.
    
    Keyword arguments:
    string -- string to be parsed
    sep    -- (optional) delimiter
    """
        
    # separate string into space-delimited strings of numbers
    num_str = string.split(sep)
    return [float(n) for n in num_str]


def split_str(seq, length):
    """Separate a string seq into length-sized pieces.
    
    Keyword arguments:
    seq    -- string containing sequence of smaller strings of constant length
    length -- length of individual sequences
    """
    return [seq[i : i + length] for i in range(0, len(seq), length)]


def create_dir(path):
    """Creates a new directory based on input path.
    
    No error if path already exists, but other error is reported.
    
    Keyword arguments:
    path -- path of directory to be created
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise