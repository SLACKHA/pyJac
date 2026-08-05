"""Tests for the pyJac command-line interface.

The parser lives in ``pyjac.__main__``.
"""

import pytest

from pyjac import utils
from pyjac._version import __version__
from pyjac.__main__ import build_parser, get_parser

from conftest import GOLDEN_MECHS

UNSUPPORTED = sorted(set(utils.langs) - set(utils.supported_langs))


def test_parser_is_not_in_utils():
    """utils no longer exposes the CLI parser."""
    assert not hasattr(utils, 'get_parser')
    assert 'get_parser' not in utils.__all__


def test_version_flag_prints_version(capsys, monkeypatch):
    """--version prints the version and exits 0."""
    monkeypatch.setattr('sys.argv', ['pyjac', '--version'])
    with pytest.raises(SystemExit) as excinfo:
        get_parser()
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f'pyJac {__version__}'


def test_short_version_flag(capsys, monkeypatch):
    """-v is accepted as well."""
    monkeypatch.setattr('sys.argv', ['pyjac', '-v'])
    with pytest.raises(SystemExit):
        get_parser()
    assert __version__ in capsys.readouterr().out


@pytest.mark.parametrize('lang', UNSUPPORTED)
def test_unsupported_lang_exits_via_parser(lang, capsys, monkeypatch):
    """An incomplete backend is rejected at parse time, before any work."""
    monkeypatch.setattr(
        'sys.argv',
        ['pyjac', '--lang', lang, '--input', str(GOLDEN_MECHS['h2o2'])],
    )
    with pytest.raises(SystemExit) as excinfo:
        get_parser()
    assert excinfo.value.code == 2
    assert 'not implemented' in capsys.readouterr().err


def test_missing_mechanism_file_exits(capsys, monkeypatch):
    """A missing mechanism is reported by the parser, not as an IOError."""
    monkeypatch.setattr(
        'sys.argv', ['pyjac', '--lang', 'c', '--input', 'does_not_exist.inp']
    )
    with pytest.raises(SystemExit) as excinfo:
        get_parser()
    assert excinfo.value.code == 2
    assert 'mechanism file not found' in capsys.readouterr().err


def test_missing_thermo_file_exits(capsys, monkeypatch):
    """A supplied but missing thermo database is likewise caught early."""
    monkeypatch.setattr('sys.argv', [
        'pyjac', '--lang', 'c', '--input', str(GOLDEN_MECHS['h2o2']),
        '--thermo', 'does_not_exist.dat',
    ])
    with pytest.raises(SystemExit) as excinfo:
        get_parser()
    assert excinfo.value.code == 2
    assert 'thermodynamic database not found' in capsys.readouterr().err


def test_valid_arguments_parse(monkeypatch, tmp_path):
    """A well-formed command line produces the expected namespace."""
    monkeypatch.setattr('sys.argv', [
        'pyjac', '--lang', 'c', '--input', str(GOLDEN_MECHS['h2o2']),
        '-b', str(tmp_path),
    ])
    args = get_parser()
    assert args.lang == 'c'
    assert args.input == str(GOLDEN_MECHS['h2o2'])
    assert args.build_path == str(tmp_path)
    assert args.auto_diff is False
    assert args.skip_jac is False


def test_lang_choices_still_list_every_backend():
    """All four languages remain selectable, so the error can explain itself.

    Dropping them from ``choices`` would make argparse emit a bare "invalid
    choice", which says nothing about why fortran and matlab are unavailable.
    """
    parser = build_parser()
    lang_action = next(a for a in parser._actions if a.dest == 'lang')
    assert sorted(lang_action.choices) == sorted(utils.langs)
