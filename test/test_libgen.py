import sys


class TestLibgen:
    """ """

    def test_imported(self):
        """Ensure libgen module imported."""
        assert 'pyjac.libgen.libgen' in sys.modules
