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

#: Languages for which golden output is recorded.
GOLDEN_LANGS = ('c', 'cuda')


@pytest.fixture(scope='session')
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope='session')
def golden_dir():
    return GOLDEN_DIR


@pytest.fixture
def c_compiler():
    """Path to a C compiler, skipping the test if none is available."""
    for candidate in ('cc', 'gcc', 'clang'):
        found = shutil.which(candidate)
        if found:
            return found
    pytest.skip('no C compiler found on PATH')
