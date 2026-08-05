"""Import smoke tests for pyjac.pywrap.

``parallel_compiler`` still imports ``distutils``, removed from the standard
library in Python 3.12. The strict xfail below is the record of that known
breakage: once the setuptools port lands it will XPASS and fail the suite,
which is the signal to delete the marker.
"""

import importlib
import sys

import pytest

from pyjac.pywrap import pywrap_gen  # noqa: F401


def test_pywrap_gen_imported():
    assert 'pyjac.pywrap.pywrap_gen' in sys.modules


@pytest.mark.xfail(
    raises=ImportError,
    strict=True,
    reason='imports distutils, removed from the stdlib in Python 3.12 (port pending)',
)
def test_parallel_compiler_imported():
    importlib.import_module('pyjac.pywrap.parallel_compiler')
