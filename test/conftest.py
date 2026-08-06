"""Shared pytest fixtures and path helpers for the pyJac test suite."""

import pathlib
import shutil

import pytest

TEST_DIR = pathlib.Path(__file__).parent.resolve()
REPO_ROOT = TEST_DIR.parent
FIXTURE_DIR = TEST_DIR / 'fixtures'
GOLDEN_DIR = FIXTURE_DIR / 'golden'
MECH_DIR = FIXTURE_DIR / 'mechanisms'

#: Mechanisms with golden output, mapped to their Chemkin source file.
GOLDEN_MECHS = {
    'h2o2': REPO_ROOT / 'data' / 'h2o2.inp',
    'rxn_types': MECH_DIR / 'rxn_types.inp',
}

#: Generation variants with recorded golden output, as
#: ``name -> (lang, create_jacobian kwargs)``. Each exercises a distinct branch
#: of the generators, so that a refactor touching any of them is caught.
#:
#: ``cuda`` deliberately keeps ``no_shared=False`` (the default), so the plain
#: variant covers the shared-memory manager and ``cuda-noshared`` covers its
#: absence.
GOLDEN_VARIANTS = {
    'c': ('c', {}),
    'cuda': ('cuda', {}),
    'c-autodiff': ('c', {'auto_diff': True}),
    'cuda-noshared': ('cuda', {'no_shared': True}),
}

#: Kept for readability where only the plain per-language variants are meant.
GOLDEN_LANGS = ('c', 'cuda')


@pytest.fixture(scope='session')
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope='session')
def golden_dir():
    return GOLDEN_DIR


@pytest.fixture
def to_cantera_yaml(tmp_path):
    """Convert a Chemkin mechanism to Cantera YAML, returning the new path.

    Converting on demand rather than committing a second copy keeps the two
    readers provably fed from the same source mechanism.
    """
    ck2yaml = pytest.importorskip('cantera.ck2yaml')

    def convert(chemkin_path, thermo_path=None):
        chemkin_path = pathlib.Path(chemkin_path)
        out_name = tmp_path / (chemkin_path.stem + '.yaml')
        ck2yaml.convert(
            str(chemkin_path),
            thermo_file=str(thermo_path) if thermo_path else None,
            out_name=str(out_name),
            permissive=True,
            quiet=True,
        )
        return out_name

    return convert


@pytest.fixture
def c_compiler():
    """Path to a C compiler, skipping the test if none is available."""
    for candidate in ('cc', 'gcc', 'clang'):
        found = shutil.which(candidate)
        if found:
            return found
    pytest.skip('no C compiler found on PATH')
