# Python 2 compatibility
from __future__ import print_function
from __future__ import division

import sys

from .. import utils

class TestUtils(object):
    """
    """
    def test_imported(self):
        """Ensure utils module imported.
        """
        assert 'pyjac.utils' in sys.modules
