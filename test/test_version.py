"""Tests for pyjac._version."""

from packaging.version import Version

from pyjac._version import __version__, __version_info__


def test_version_is_pep440():
    """__version__ parses as a valid PEP 440 version."""
    Version(__version__)


def test_version_info_matches_version():
    """__version_info__ is the parsed form of __version__."""
    assert __version_info__ == (1, 0, 6)
    assert '.'.join(str(part) for part in __version_info__) == __version__
