"""Tests for pyjac.libgen."""

import pytest

from pyjac.libgen import file_struct, generate_library, libgen


def test_public_api():
    assert callable(generate_library)
    assert callable(libgen.compiler)
    assert 'cuda' in libgen.cmd_compile


def test_missing_compiler_fails_instead_of_hanging(tmp_path, monkeypatch):
    """A missing compiler must return an error, not deadlock the process pool.

    ``compiler`` runs in a ``multiprocessing.Pool`` worker. Calling
    ``sys.exit`` there leaves the parent's ``pool.map`` waiting for a result
    that never arrives, so the build hung rather than reporting.
    """
    monkeypatch.setitem(libgen.cmd_compile, 'c', str(tmp_path / 'no-such-compiler'))
    struct = file_struct(
        'c',
        'c',
        'jacob',
        [str(tmp_path)],
        [],
        str(tmp_path),
        str(tmp_path),
        shared=False,
    )

    assert compiler_returns_error(struct)


def compiler_returns_error(struct):
    return libgen.compiler(struct) == -1


@pytest.mark.parametrize('lang', ['c', 'cuda', 'icc'])
def test_every_language_has_a_compiler_and_flags(lang):
    assert lang in libgen.cmd_compile
    assert lang in libgen.flags
