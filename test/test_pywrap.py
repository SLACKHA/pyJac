"""Tests for pyjac.pywrap.

``parallel_compiler`` and the four ``*_setup.py.in`` templates now use
``setuptools._distutils`` instead of ``distutils``, which was removed from the
standard library in Python 3.12.
"""

import ast
import sys
from pathlib import Path
from string import Template

import pytest

from pyjac.pywrap import parallel_compiler, pywrap_gen

TEMPLATE_DIR = Path(parallel_compiler.__file__).parent
TEMPLATES = sorted(TEMPLATE_DIR.glob('*_setup.py.in'))


def test_pywrap_gen_imported():
    assert 'pyjac.pywrap.pywrap_gen' in sys.modules
    assert hasattr(pywrap_gen, 'generate_wrapper')


def test_parallel_compiler_imported():
    assert 'pyjac.pywrap.parallel_compiler' in sys.modules
    assert callable(parallel_compiler.parallel_compile)


def test_no_stdlib_distutils_imports():
    """Nothing may import the removed stdlib distutils."""
    offenders = []
    for path in [Path(parallel_compiler.__file__), *TEMPLATES]:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(('import distutils', 'from distutils')):
                offenders.append(f'{path.name}: {stripped}')
    assert not offenders, 'stdlib distutils still imported:\n' + '\n'.join(offenders)


def test_templates_found():
    """Guard against the glob silently matching nothing."""
    assert len(TEMPLATES) == 4


@pytest.mark.parametrize('template', TEMPLATES, ids=lambda p: p.name)
def test_template_is_valid_python_once_filled_in(template):
    """Each setup template must parse after placeholder substitution."""
    filled = Template(template.read_text()).safe_substitute(
        homepath='/tmp/home', buildpath='/tmp/build',
        outpath='/tmp/out', libname='libc_pyjac.a',
    )
    ast.parse(filled, filename=template.name)
