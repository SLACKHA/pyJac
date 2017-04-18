# Python 2 compatibility
from __future__ import print_function
from __future__ import division

import sys

from ..pywrap import pywrap_gen
from ..pywrap import parallel_compiler

class TestPywrap_gen(object):
    """
    """
    def test_imported(self):
        """Ensure pywrap_gen module imported.
        """
        assert 'pyjac.pywrap.pywrap_gen' in sys.modules

class TestParallelCompiler(object):
    """
    """
    def test_imported(self):
        """Ensure parallel_compiler module imported.
        """
        assert 'pyjac.pywrap.parallel_compiler' in sys.modules
