# Python 2 compatibility
from __future__ import print_function
from __future__ import division

import sys

from ..functional_tester import partially_stirred_reactor
from ..functional_tester import test

class TestPartiallyStirredReactor(object):
    """
    """
    def test_imported(self):
        """Ensure partially_stirred_reactor module imported.
        """
        assert 'pyjac.functional_tester.partially_stirred_reactor' in sys.modules

class TestTest(object):
    """
    """
    def test_imported(self):
        """Ensure test module imported.
        """
        assert 'pyjac.functional_tester.test' in sys.modules
