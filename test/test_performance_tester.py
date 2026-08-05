"""Import smoke test for pyjac.performance_tester.

The module imports ``cantera.ck2cti``, removed in Cantera 3.0 in favour of
``cantera.ck2yaml``. The strict xfail below is the record of that known
breakage: once the Cantera 3.x port lands it will XPASS and fail the suite,
which is the signal to delete the marker.
"""

import importlib

import pytest


@pytest.mark.xfail(
    raises=ImportError,
    strict=True,
    reason='imports cantera.ck2cti, removed in Cantera 3.0 (port pending)',
)
def test_performance_tester_imported():
    importlib.import_module('pyjac.performance_tester.performance_tester')
