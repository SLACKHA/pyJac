import sys

from pyjac.libgen import libgen

class TestLibgen(object):
    """
    """
    def test_imported(self):
        """Ensure libgen module imported.
        """
        assert 'pyjac.libgen.libgen' in sys.modules
