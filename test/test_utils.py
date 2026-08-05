import sys

from pyjac import utils

class TestUtils(object):
    """
    """
    def test_imported(self):
        """Ensure utils module imported.
        """
        assert 'pyjac.utils' in sys.modules
