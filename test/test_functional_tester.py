"""Import smoke tests for pyjac.functional_tester.

Both modules import ``cantera.ck2cti``, removed in Cantera 3.0 in favour of
``cantera.ck2yaml``. The strict xfail markers below are the record of that
known breakage: once the Cantera 3.x port lands these will XPASS and fail the
suite, which is the signal to delete the markers.
"""

import importlib

import pytest

CK2CTI = pytest.mark.xfail(
    raises=ImportError,
    strict=True,
    reason='imports cantera.ck2cti, removed in Cantera 3.0 (port pending)',
)


@CK2CTI
def test_partially_stirred_reactor_imported():
    importlib.import_module('pyjac.functional_tester.partially_stirred_reactor')


@CK2CTI
def test_test_module_imported():
    importlib.import_module('pyjac.functional_tester.test')
