"""Tests for pyjac.functional_tester.

Both modules import again now that ``cantera.ck2cti`` -- removed in Cantera 3.0
along with the CTI format itself -- has been replaced by ``ck2yaml``.

The modules are not yet fully functional: ``is_pdep`` and the reaction-type
checks in ``test`` still dispatch on Cantera reaction classes removed in 3.0,
which is part of the Cantera 3.x port rather than this work.
"""

import shutil
import sys

import cantera as ct

from pyjac.functional_tester import partially_stirred_reactor, test as ftest

from conftest import GOLDEN_MECHS


def test_partially_stirred_reactor_imported():
    assert 'pyjac.functional_tester.partially_stirred_reactor' in sys.modules
    assert hasattr(partially_stirred_reactor, 'run_simulation')


def test_test_module_imported():
    assert 'pyjac.functional_tester.test' in sys.modules


def test_convert_mech_writes_loadable_yaml(tmp_path):
    """Chemkin input converts to YAML that Cantera can read.

    The old implementation called ``ck2cti`` and produced a ``.cti`` file;
    neither the converter nor the format survives in Cantera 3.x.
    """
    mech = tmp_path / 'h2o2.inp'
    shutil.copy(GOLDEN_MECHS['h2o2'], mech)

    converted = ftest.convert_mech(str(mech))

    assert converted.endswith('.yaml')
    gas = ct.Solution(converted)
    assert gas.n_species == 9
    assert gas.n_reactions == 28
