import sys

from pyjac.core import cache_optimizer
from pyjac.core import chem_utilities
from pyjac.core import create_jacobian
from pyjac.core import mech_auxiliary
from pyjac.core import rate_subs
from pyjac.core import shared_memory

class TestCacheOptimizer:
    """
    """
    def test_imported(self):
        """Ensure cache_optimizer module imported.
        """
        assert 'pyjac.core.cache_optimizer' in sys.modules

class TestChemUtilities:
    """
    """
    def test_imported(self):
        """Ensure chem_utilities module imported.
        """
        assert 'pyjac.core.chem_utilities' in sys.modules

class TestCreateJacobian:
    """
    """
    def test_imported(self):
        """Ensure create_jacobian module imported.
        """
        assert 'pyjac.core.create_jacobian' in sys.modules

class TestMechAuxiliary:
    """
    """
    def test_imported(self):
        """Ensure mech_auxiliary module imported.
        """
        assert 'pyjac.core.mech_auxiliary' in sys.modules

class TestRateSubs:
    """
    """
    def test_imported(self):
        """Ensure rate_subs module imported.
        """
        assert 'pyjac.core.rate_subs' in sys.modules

class TestSharedMemory:
    """
    """
    def test_imported(self):
        """Ensure shared_memory module imported.
        """
        assert 'pyjac.core.shared_memory' in sys.modules
