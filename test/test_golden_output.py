"""Characterization tests: generated source code must not change.

These lock in the exact output of the code generator as it stood before the
modernization work began (pyJac 1.0.6, via the Chemkin input path). A diff here
means a refactor altered generated code -- either a bug, or an intentional
change that needs the fixtures re-recorded with ``test/regenerate_golden.py``
and called out in CHANGELOG.md.

Coverage spans the C and CUDA backends, the Adept autodifferentiation variant,
and CUDA without the shared-memory manager, so that mechanical refactors of the
string-assembly code in ``rate_subs`` and ``create_jacobian`` are checked on
every branch they touch.

The cache optimizer is deliberately excluded from byte comparison: it is a
randomized greedy search using unseeded ``np.random``, so its output differs
between runs. It gets a smoke-and-compile test instead.
"""

import subprocess

import pytest

from pyjac.core.create_jacobian import create_jacobian

from conftest import GOLDEN_MECHS, GOLDEN_VARIANTS


def _generate(mech, lang, dest, **kwargs):
    """Generate source for ``mech`` in ``lang`` under ``dest``."""
    create_jacobian(lang, mech_name=str(mech), build_path=str(dest), **kwargs)


def _relative_files(root):
    return sorted(p.relative_to(root) for p in root.rglob('*') if p.is_file())


def _compile_sources(sources, include_dir, compiler):
    """Compile each source, returning a list of failure descriptions."""
    failures = []
    for source in sources:
        result = subprocess.run(
            [
                compiler, '-std=c99', '-O2', '-fPIC',
                '-Wall', '-Wextra', '-Wno-unused-parameter',
                '-I', str(include_dir), '-c', str(source), '-o', '/dev/null',
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or result.stderr.strip():
            failures.append(f'{source.name}:\n{result.stderr}')
    return failures


@pytest.mark.parametrize('variant', sorted(GOLDEN_VARIANTS))
@pytest.mark.parametrize('mech', sorted(GOLDEN_MECHS))
def test_generated_source_matches_golden(mech, variant, tmp_path, golden_dir):
    """Regenerated source is byte-identical to the recorded golden output."""
    expected_dir = golden_dir / mech / variant
    assert expected_dir.is_dir(), f'no golden output recorded for {mech}/{variant}'

    lang, kwargs = GOLDEN_VARIANTS[variant]
    _generate(GOLDEN_MECHS[mech], lang, tmp_path, **kwargs)

    expected_files = _relative_files(expected_dir)
    actual_files = _relative_files(tmp_path)
    assert actual_files == expected_files, (
        f'{mech}/{variant}: generated file list changed'
    )

    mismatched = [
        str(rel) for rel in expected_files
        if (expected_dir / rel).read_bytes() != (tmp_path / rel).read_bytes()
    ]
    assert not mismatched, (
        f'{mech}/{variant}: generated source differs from golden output for '
        f'{mismatched}. If this change is intentional, re-record with '
        f'`python test/regenerate_golden.py` and document it in CHANGELOG.md.'
    )


@pytest.mark.parametrize('variant', sorted(GOLDEN_VARIANTS))
@pytest.mark.parametrize('mech', sorted(GOLDEN_MECHS))
def test_generation_is_deterministic(mech, variant, tmp_path):
    """Two runs of the generator produce identical output."""
    lang, kwargs = GOLDEN_VARIANTS[variant]
    first, second = tmp_path / 'first', tmp_path / 'second'
    _generate(GOLDEN_MECHS[mech], lang, first, **kwargs)
    _generate(GOLDEN_MECHS[mech], lang, second, **kwargs)

    assert _relative_files(first) == _relative_files(second)
    for rel in _relative_files(first):
        assert (first / rel).read_bytes() == (second / rel).read_bytes(), (
            f'{mech}/{variant}: {rel} differs between two generation runs'
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

    failures = _compile_sources(sources, tmp_path, c_compiler)
    assert not failures, 'generated C did not compile cleanly:\n' + '\n'.join(failures)


@pytest.mark.compiler
@pytest.mark.slow
@pytest.mark.parametrize('mech', sorted(GOLDEN_MECHS))
def test_cache_optimized_generation(mech, tmp_path, c_compiler):
    """The cache-optimizer path generates a complete, compilable source set.

    Output is not byte-compared: ``cache_optimizer`` runs a randomized greedy
    search on unseeded ``np.random``, so successive runs legitimately differ.
    This asserts the path runs to completion, emits the same set of files as an
    unoptimized build, and produces C that still compiles.
    """
    pytest.importorskip('bitarray', reason='cache optimization requires bitarray')

    plain, optimized = tmp_path / 'plain', tmp_path / 'optimized'
    _generate(GOLDEN_MECHS[mech], 'c', plain)
    _generate(GOLDEN_MECHS[mech], 'c', optimized,
              optimize_cache=True, force_optimize=True)

    # The optimizer additionally writes its memoized ordering next to the source.
    produced = {rel.name for rel in _relative_files(optimized)} - {'optimized.pickle'}
    assert produced == {rel.name for rel in _relative_files(plain)}, (
        'cache-optimized build produced a different set of source files'
    )

    sources = sorted(optimized.glob('*.c'))
    failures = _compile_sources(sources, optimized, c_compiler)
    assert not failures, (
        'cache-optimized C did not compile cleanly:\n' + '\n'.join(failures)
    )
