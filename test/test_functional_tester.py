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
import pytest

from conftest import GOLDEN_MECHS
from pyjac.functional_tester import partially_stirred_reactor
from pyjac.functional_tester import test as ftest


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


def test_legacy_cantera_formats_are_refused(tmp_path):
    """.cti and .xml were removed in Cantera 3.0.

    The tester used to treat anything that was not .cti/.xml as Chemkin, so a
    YAML mechanism was fed to ck2yaml as though it were Chemkin input.
    """
    for suffix in ('.cti', '.xml'):
        legacy = tmp_path / f'mech{suffix}'
        legacy.write_text('')
        with pytest.raises(NotImplementedError) as excinfo:
            ftest.test('c', str(tmp_path), str(tmp_path), str(legacy))
        message = str(excinfo.value)
        assert 'cti2yaml' in message or 'ctml2yaml' in message


def test_yaml_mechanisms_are_not_converted(tmp_path, monkeypatch):
    """A Cantera YAML mechanism must reach Cantera unconverted.

    The format check used to treat anything that was not .cti/.xml as Chemkin,
    so a YAML mechanism was handed to ck2yaml as though it were Chemkin input.
    """
    converted = []
    monkeypatch.setattr(
        ftest, 'convert_mech', lambda *args, **kwargs: converted.append(args)
    )

    class ReachedCantera(Exception):
        """Raised in place of loading the mechanism, to stop the run early."""

    def stop(*args, **kwargs):
        raise ReachedCantera

    monkeypatch.setattr(ftest.ct, 'Solution', stop)

    mech = tmp_path / 'mech.yaml'
    mech.write_text('')

    with pytest.raises(ReachedCantera):
        ftest.test('c', str(tmp_path), str(tmp_path), str(mech))

    assert not converted, 'a .yaml mechanism was sent through ck2yaml'


def test_chemkin_mechanisms_are_converted(tmp_path, monkeypatch):
    """A Chemkin mechanism is still converted before Cantera sees it."""
    converted = []

    def fake_convert(mech_filename, therm_filename=None):
        converted.append(mech_filename)
        return str(tmp_path / 'converted.yaml')

    monkeypatch.setattr(ftest, 'convert_mech', fake_convert)

    class ReachedCantera(Exception):
        """Raised in place of loading the mechanism, to stop the run early."""

    def stop(*args, **kwargs):
        raise ReachedCantera

    monkeypatch.setattr(ftest.ct, 'Solution', stop)

    mech = tmp_path / 'mech.inp'
    mech.write_text('')

    with pytest.raises(ReachedCantera):
        ftest.test('c', str(tmp_path), str(tmp_path), str(mech))

    assert converted == [str(mech)]
