"""The incomplete Fortran and Matlab backends must fail fast and clearly.

Neither was ever finished -- ``utils.header_ext`` has no entry for them, so
generation used to die with a bare ``KeyError`` on the first file written,
after having already created the output directory. They now raise
``NotImplementedError`` before doing any work.
"""

import argparse

import pytest

from pyjac import utils
from pyjac.__main__ import main
from pyjac.core.create_jacobian import create_jacobian

from conftest import GOLDEN_MECHS

UNSUPPORTED = sorted(set(utils.langs) - set(utils.supported_langs))


def test_supported_langs_is_a_subset_of_langs():
    assert set(utils.supported_langs) <= set(utils.langs)
    assert utils.supported_langs == ['c', 'cuda']


@pytest.mark.parametrize('lang', UNSUPPORTED)
def test_create_jacobian_rejects_unsupported_lang(lang, tmp_path):
    """The error names the language and the supported alternatives."""
    with pytest.raises(NotImplementedError) as excinfo:
        create_jacobian(lang, mech_name=str(GOLDEN_MECHS['h2o2']),
                        build_path=str(tmp_path))
    message = str(excinfo.value)
    assert lang in message
    assert 'c, cuda' in message


@pytest.mark.parametrize('lang', UNSUPPORTED)
def test_rejected_before_writing_anything(lang, tmp_path):
    """Generation bails out before creating output, not part-way through."""
    build_path = tmp_path / 'out'
    with pytest.raises(NotImplementedError):
        create_jacobian(lang, mech_name=str(GOLDEN_MECHS['h2o2']),
                        build_path=str(build_path))
    assert not build_path.exists(), 'output directory created despite failure'


@pytest.mark.parametrize('lang', UNSUPPORTED)
def test_cli_reports_error_and_exits_nonzero(lang, tmp_path, capsys):
    """The CLI turns the exception into a message and a non-zero status."""
    args = argparse.Namespace(
        lang=lang, input=str(GOLDEN_MECHS['h2o2']), thermo=None,
        cache_optimizer=False, initial_conditions='', num_blocks=8,
        num_threads=64, no_shared=False, L1_preferred=True, multi_thread=1,
        force_optimize=False, build_path=str(tmp_path), skip_jac=False,
        last_species=None, auto_diff=False,
    )
    assert main(args) == 2
    assert 'not implemented' in capsys.readouterr().err.lower()


def test_cli_main_accepts_supplied_args(tmp_path):
    """main() honours an args namespace instead of ignoring it.

    Previously the body sat inside ``if args is None``, so passing arguments
    made main() a silent no-op.
    """
    args = argparse.Namespace(
        lang='c', input=str(GOLDEN_MECHS['h2o2']), thermo=None,
        cache_optimizer=False, initial_conditions='', num_blocks=8,
        num_threads=64, no_shared=False, L1_preferred=True, multi_thread=1,
        force_optimize=False, build_path=str(tmp_path), skip_jac=False,
        last_species=None, auto_diff=False,
    )
    assert main(args) == 0
    assert (tmp_path / 'jacob.c').is_file()
