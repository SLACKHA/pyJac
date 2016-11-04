# -*- coding: utf-8 -*-
"""
Module that maintains abstract file classes that ease file I/O
"""

#system imports
import os

#local imports
from .. import utils

def get_standard_headers(lang):
    """
    Returns a list of standard headers to include for a given target language

    Parameters
    ----------
    lang : str
        The target language
    """

    utils.check_lang(lang)
    if lang == 'opencl':
        return ['cl.h']
    elif lang == 'cuda':
        return ['cuda.h']
    return []


def get_header_file(name, lang, mode='w'):
    """
    Returns the appropriate FileWriter class for a header file for the given language

    Parameters
    ----------
    name : str
        The full path and name of the output file
    lang : str
        The target language
    mode : str
        The file mode, 'w' by default
    """

    return FileWriter(name, lang, mode=mode, is_header=True)

def get_file(name, lang, mode='w'):
    """
    Returns the appropriate FileWriter class for a regular file for the given language

    Parameters
    ----------
    name : str
        The full path and name of the output file
    lang : str
        The target language
    mode : str
        The file mode, 'w' by default
    """

    return FileWriter(name, lang, mode=mode, is_header=False)

class FileWriter(object):
    """
    The base FileWriter class.
    Defines various functions to be reimplmented
    and provides some base definitions

    Attributes
    ----------
    name : str
        The full path and name of the file
    mode : str
        The file i/o mode
    lang : str
        The target language
    headers : list of str
        The headers to include
    std_headers : list of str
        The system headers to include
    lines : list of str
        The lines to write
    is_header : bool
        If true, this is a header file
    """

    def __init__(self, name, lang, mode='w', is_header=False):
        self.name = name
        self.mode = mode
        self.lang = lang
        utils.check_lang(lang)
        self.headers = []
        self.std_headers = []
        self.is_header = is_header
        if self.is_header:
            self.headers = ['header']
            self.std_headers = get_standard_headers(lang)
        self.lines = []

    def __enter__(self):
        self.file = open(self.name, self.mode)
        return self

    def __exit__(self, type, value, traceback):
        self.write()
        self.file.close()

    def write(self):
        lines = []
        filename = os.path.basename(self.name)
        filename, ext = filename.split('.')
        if self.is_header:
            filename, ext = filename.upper(), ext.upper()
            lines.append('#ifndef {}_{}'.format(filename, ext))
            lines.append('#define {}_{}'.format(filename, ext))
        else:
            self.headers.append(filename)

        ext = utils.header_ext[self.lang]
        for header in self.std_headers:
            lines.append('#include <{}>'.format(header))
        for header in self.headers:
            if not any(header.endswith(x) for x in utils.header_ext.values()):
                header = header + utils.header_ext[self.lang]
            if not (header.endswith('>') or header.endswith('"')): 
                lines.append('#include "{}"'.format(header,
                    ext))
            else:
                lines.append(header)

        lines.extend(self.lines)
        lines.append('#endif')
        self.file.write('\n'.join(lines))


    def add_headers(self, headers):
        if isinstance(headers, list):
            self.headers.extend(headers)
        else:
            self.headers.append(headers)

    def add_lines(self, lines):
        if isinstance(lines, str):
            lines = lines.split('\n')

        self.lines.extend(lines)