# Python 2 compatibility
from __future__ import print_function
from __future__ import division

import sys

from ..core import cache_optimizer
from ..core import chem_utilities
from ..core import create_jacobian
from ..core import mech_auxiliary
from ..core import rate_subs
from ..core import shared_memory

class TestCacheOptimizer(object):
    """
    """
    def test_imported(self):
        """Ensure cache_optimizer module imported.
        """
        assert 'pyjac.core.cache_optimizer' in sys.modules

class TestChemUtilities(object):
    """
    """
    def test_imported(self):
        """Ensure chem_utilities module imported.
        """
        assert 'pyjac.core.chem_utilities' in sys.modules

class TestCreateJacobian(object):
    """
    """
    def test_imported(self):
        """Ensure create_jacobian module imported.
        """
        assert 'pyjac.core.create_jacobian' in sys.modules

class TestMechAuxiliary(object):
    """
    """
    def test_imported(self):
        """Ensure mech_auxiliary module imported.
        """
        assert 'pyjac.core.mech_auxiliary' in sys.modules

class TestRateSubs(object):
    """
    """
    def test_imported(self):
        """Ensure rate_subs module imported.
        """
        assert 'pyjac.core.rate_subs' in sys.modules

class TestSharedMemory(object):
    """
    """
    def test_imported(self):
        """Ensure shared_memory module imported.
        """
        assert 'pyjac.core.shared_memory' in sys.modules
