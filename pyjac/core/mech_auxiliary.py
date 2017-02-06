"""Writes mechanism header and output testing files
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import sys
import itertools

# Local imports
from .. import utils
from ..kernel_utils import file_writers as filew

def write_mechanism_header(path, lang, specs, reacs):
    with filew.get_header_file(os.path.join(path,
                'mechanism' + utils.header_ext[lang]), lang) as file:
        #define NR, NS, NN, etc.

        file.add_define('NS', len(specs) - 1)
        file.add_define('NR', len(reacs))
        file.add_define('NN', len(specs))