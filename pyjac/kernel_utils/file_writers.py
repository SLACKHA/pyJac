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
        return []
    elif lang == 'cuda':
        return []
    return []

def get_preamble(lang):
    utils.check_lang(lang)
    return []


def get_header_file(name, lang, mode='w', **kwargs):
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

    return FileWriter(name, lang, mode=mode, is_header=True, **kwargs)

def get_file(name, lang, mode='w', **kwargs):
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

    return FileWriter(name, lang, mode=mode, is_header=False, **kwargs)

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
    include_own_header : bool
        If true, include a header of the same name.  Cannot be header file
    use_filter : bool
        If true, use the default filter for this file type:
            None for header files
            Preamble filters for source files
    """

    def __init__(self, name, lang, mode='w', is_header=False,
                    include_own_header=False, use_filter=True):
        self.name = name
        self.mode = mode
        self.lang = lang
        utils.check_lang(lang)
        self.headers = []
        self.std_headers = []
        self.is_header = is_header
        self.include_own_header = include_own_header
        if self.is_header:
            self.headers = ['mechanism'] if self.name != 'mechanism' + utils.header_ext[lang] else []
            self.std_headers = get_standard_headers(lang)
            self.filter = lambda x: x
            assert not self.include_own_header, 'Cannot include this file in itself'
        else:
            self.filter = self.preamble_filter
            self.preamble = get_preamble(lang)
        if not use_filter:
            self.filter = lambda x: x
        self.lines = []
        self.defines = []

    def __enter__(self):
        self.file = open(self.name, self.mode)
        return self

    def __exit__(self, type, value, traceback):
        self.write()
        self.file.close()

    def preamble_filter(self, lines):
        #check things outside of function definitions for duplicated lines
        #first, find the kernel text

        seen = set()
        out_lines = []
        in_preamble = True
        brace_counter = 0
        for line in lines:
            if 'void' in line:
                in_preamble = False
                assert brace_counter == 0

            if in_preamble:
                #check for dupes
                if line not in seen or not line.strip():
                    seen.add(line)
                    out_lines.append(line)
            else:
                out_lines.append(line)

            #update braces
            if not in_preamble and '{' in line:
                brace_counter += 1
            if not in_preamble and '}' in line:
                brace_counter -= 1
                if brace_counter == 0:
                    in_preamble = True

        return out_lines


    def write(self):
        lines = []
        filename = os.path.basename(self.name)
        filename, ext = filename.split('.')
        if self.is_header:
            filename, ext = filename.upper(), ext.upper()
            lines.append('#ifndef {}_{}'.format(filename, ext))
            lines.append('#define {}_{}'.format(filename, ext))
        else:
            if self.include_own_header:
                self.headers.append(filename)
            self.lines.extend(self.preamble)

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

        if self.is_header and self.defines:
            lines.extend(['#define {name} ({value})'.format(
                name=x[0], value=x[1]) for x in self.defines])

        lines.extend(self.lines)
        if self.is_header:
            lines.append('#endif')
        self.file.write('\n'.join(lines))


    def add_headers(self, headers):
        if isinstance(headers, list):
            self.headers.extend(headers)
        else:
            self.headers.append(headers)

    def add_define(self, name, value):
        self.defines.append((name, value))

    def add_lines(self, lines):
        if isinstance(lines, str):
            lines = lines.split('\n')

        lines = self.filter(lines)

        self.lines.extend(lines)