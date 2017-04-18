"""
Test for _version.py
"""
# Standard libraries
import pkg_resources

# Local imports
from .._version import __version__


def test_semantic_version():
    pkg_resources.parse_version(__version__)
