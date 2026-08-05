"""Characterization tests: generated source code must not change.

These lock in the exact output of the code generator as it stood before the
modernization work began (pyJac 1.0.6, via the Chemkin input path). A diff here
means a refactor altered generated code -- either a bug, or an intentional
change that needs the fixtures re-recorded with ``test/regenerate_golden.py``
and called out in CHANGELOG.md.
"""

import subprocess

import pytest

from pyjac.core.create_jacobian import create_jacobian

from conftest import GOLDEN_LANGS, GOLDEN_MECHS


def _generate(mech, lang, dest):
    """Generate source for ``mech`` in ``lang`` under ``dest``."""
    create_jacobian(lang, mech_name=str(mech), build_path=str(dest))


def _relative_files(root):
    return sorted(p.relative_to(root) for p in root.rglob('*') if p.is_file())


@pytest.mark.parametrize('lang', GOLDEN_LANGS)
@pytest.mark.parametrize('mech', sorted(GOLDEN_MECHS))
def test_generated_source_matches_golden(mech, lang, tmp_path, golden_dir):
    """Regenerated source is byte-identical to the recorded golden output."""
    expected_dir = golden_dir / mech / lang
    assert expected_dir.is_dir(), f'no golden output recorded for {mech}/{lang}'

    _generate(GOLDEN_MECHS[mech], lang, tmp_path)

    expected_files = _relative_files(expected_dir)
    actual_files = _relative_files(tmp_path)
    assert actual_files == expected_files, (
        f'{mech}/{lang}: generated file list changed'
    )

    mismatched = []
    for rel in expected_files:
        expected = (expected_dir / rel).read_bytes()
        actual = (tmp_path / rel).read_bytes()
        if expected != actual:
            mismatched.append(str(rel))

    assert not mismatched, (
        f'{mech}/{lang}: generated source differs from golden output for '
        f'{mismatched}. If this change is intentional, re-record with '
        f'`python test/regenerate_golden.py` and document it in CHANGELOG.md.'
    )


@pytest.mark.parametrize('mech', sorted(GOLDEN_MECHS))
def test_generation_is_deterministic(mech, tmp_path):
    """Two runs of the generator produce identical output."""
    first, second = tmp_path / 'first', tmp_path / 'second'
    _generate(GOLDEN_MECHS[mech], 'c', first)
    _generate(GOLDEN_MECHS[mech], 'c', second)

    assert _relative_files(first) == _relative_files(second)
    for rel in _relative_files(first):
        assert (first / rel).read_bytes() == (second / rel).read_bytes(), (
            f'{mech}: {rel} differs between two generation runs'
        )


@pytest.mark.compiler
@pytest.mark.parametrize('mech', sorted(GOLDEN_MECHS))
def test_generated_c_compiles(mech, tmp_path, c_compiler):
    """Generated C compiles cleanly, with no warnings from -Wall -Wextra.

    ``-Wno-unused-parameter`` is expected: the generated ODE right-hand sides
    take a time argument ``t`` that an autonomous chemical system never uses.
    """
    _generate(GOLDEN_MECHS[mech], 'c', tmp_path)

    sources = sorted(tmp_path.glob('*.c'))
    assert sources, 'generator produced no C sources'

    failures = []
    for source in sources:
        result = subprocess.run(
            [
                c_compiler, '-std=c99', '-O2', '-fPIC',
                '-Wall', '-Wextra', '-Wno-unused-parameter',
                '-I', str(tmp_path), '-c', str(source), '-o', '/dev/null',
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or result.stderr.strip():
            failures.append(f'{source.name}:\n{result.stderr}')

    assert not failures, 'generated C did not compile cleanly:\n' + '\n'.join(failures)
