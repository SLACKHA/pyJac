"""Tests for pyjac.pywrap.

``parallel_compiler`` and the four ``*_setup.py.in`` templates use
``setuptools._distutils`` rather than ``distutils``, which was removed from the
standard library in Python 3.12.
"""

import ast
import importlib.machinery
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from string import Template

import pytest

from conftest import GOLDEN_MECHS
from pyjac.core import CUDAParams
from pyjac.core.create_jacobian import create_jacobian
from pyjac.pywrap import generate_wrapper, parallel_compiler, pywrap_gen

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
        homepath='/tmp/home',
        buildpath='/tmp/build',
        outpath='/tmp/out',
        libname='libc_pyjac.a',
    )
    ast.parse(filled, filename=template.name)


def test_build_invokes_the_running_interpreter():
    """The wrapper build must use sys.executable.

    Reconstructing a ``pythonX.Y`` name resolves it against PATH, which escapes
    the active virtual environment and lands on an interpreter without Cython,
    NumPy or setuptools installed.
    """
    source = Path(pywrap_gen.__file__).read_text()
    assert 'sys.executable' in source
    assert 'python{sys.version_info' not in source
    assert "f'python{" not in source


@pytest.mark.parametrize('template', TEMPLATES, ids=lambda p: p.name)
def test_templates_import_parallel_compiler_absolutely(template):
    """Templates must not rely on living beside parallel_compiler.

    The filled-in setup script is written to the build directory, so a bare
    ``import parallel_compiler`` would not resolve.
    """
    text = template.read_text()
    assert 'from pyjac.pywrap import parallel_compiler' in text
    assert '\nimport parallel_compiler' not in text


@pytest.mark.compiler
@pytest.mark.slow
def test_generate_wrapper_end_to_end(tmp_path, monkeypatch, c_compiler):
    """Generate, compile, wrap, import, and call the built module.

    Exercises the whole pipeline: Chemkin input to C, C to a static library,
    Cython wrapper, and finally the extension module's own entry points.
    """
    pytest.importorskip('Cython', reason='building the wrapper requires Cython')
    pytest.importorskip('setuptools', reason='building the wrapper requires setuptools')

    monkeypatch.chdir(tmp_path)
    create_jacobian('c', mech_name=str(GOLDEN_MECHS['h2o2']), build_path='out')
    generate_wrapper('c', 'out', out_dir=str(tmp_path))

    built = [
        path
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        for path in tmp_path.glob(f'pyjacob*{suffix}')
    ]
    assert built, f'no extension module produced; got {list(tmp_path.iterdir())}'

    # the package directory must stay clean -- it is read-only once installed.
    # Cython emits its .c beside the .pyx it compiles, so the wrapper sources
    # are staged into the build directory first.
    stray = list(TEMPLATE_DIR.glob('*_setup.py'))
    stray += list(TEMPLATE_DIR.glob('*_wrapper.c'))
    assert not stray, f'wrapper build wrote into the package directory: {stray}'

    script = textwrap.dedent(
        """
        import numpy as np, pyjacob
        y = np.zeros(9); y[0] = 1000.0; y[1] = 0.05; y[2] = 0.2
        dy = np.zeros(9)
        pyjacob.py_dydt(0.0, 101325.0, y, dy)
        jac = np.zeros(81)
        pyjacob.py_eval_jacobian(0.0, 101325.0, y, jac)
        assert np.all(np.isfinite(dy)), 'dydt produced non-finite values'
        assert np.all(np.isfinite(jac)), 'jacobian produced non-finite values'
        assert np.count_nonzero(jac) > 0, 'jacobian is entirely zero'
        print('OK')
        """
    )
    result = subprocess.run(
        [sys.executable, '-c', script], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout


@pytest.mark.cuda
@pytest.mark.compiler
@pytest.mark.slow
def test_generate_cuda_wrapper_builds(tmp_path, monkeypatch):
    """The CUDA wrapper builds against a current toolkit.

    Only nvcc is needed, not a GPU, so this runs anywhere the toolkit is
    installed. Nothing covered this before: the end-to-end test above is C
    only, and the CUDA job in CI goes through ``pyjac.libgen``, which compiles
    generated sources directly and never touches the wrapper templates. That
    left the templates free to keep requiring things the toolkit had dropped.
    """
    if shutil.which('nvcc') is None:
        pytest.skip('nvcc not on PATH')
    pytest.importorskip('Cython', reason='building the wrapper requires Cython')
    pytest.importorskip('setuptools', reason='building the wrapper requires setuptools')

    monkeypatch.chdir(tmp_path)
    create_jacobian('cuda', mech_name=str(GOLDEN_MECHS['h2o2']), build_path='out')
    generate_wrapper(
        'cuda', 'out', out_dir=str(tmp_path), cuda_arch=CUDAParams.DEFAULT_ARCH
    )

    built = [
        path
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        for path in tmp_path.glob(f'cu_pyjacob*{suffix}')
    ]
    assert built, f'no CUDA extension module produced; got {list(tmp_path.iterdir())}'

    stray = list(TEMPLATE_DIR.glob('*_setup.py'))
    stray += list(TEMPLATE_DIR.glob('*_wrapper.c'))
    assert not stray, f'wrapper build wrote into the package directory: {stray}'


def load_flag_forwarder():
    """Extract forward_host_flags from the CUDA template without running it.

    The template calls locate_cuda() at import, which needs a toolkit, so the
    function is pulled out and exec'd on its own.
    """
    text = (TEMPLATE_DIR / 'pyjacob_cuda_setup.py.in').read_text()
    tree = ast.parse(text)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == 'forward_host_flags'
    )
    namespace = {}
    exec(ast.unparse(node), namespace)  # noqa: S102 - this repository's own source
    return namespace['forward_host_flags']


def test_host_only_link_flags_are_forwarded_to_the_host_compiler():
    """nvcc rejects host compiler flags rather than ignoring them.

    The link command is built from Python's own build configuration, so it
    carries whatever the interpreter was compiled with. Recent CPython
    contributes -fno-strict-overflow and -Wsign-compare, neither of which nvcc
    accepts; they have to arrive via -Xcompiler.
    """
    forward_host_flags = load_flag_forwarder()

    result = forward_host_flags(
        [
            '/usr/local/cuda/bin/nvcc',
            '-fno-strict-overflow',
            '-Wsign-compare',
            '-DNDEBUG',
            '-O3',
            '-Wall',
            '-fPIC',
            '-shared',
            '-L/somewhere/lib',
            '-pthread',
            '-Wl,--rpath=/somewhere/lib',
        ]
    )

    # the compiler stays first, and what nvcc understands is passed straight on
    assert result[0] == '/usr/local/cuda/bin/nvcc'
    for kept in ('-DNDEBUG', '-O3', '-shared', '-L/somewhere/lib'):
        assert kept in result
        assert result[result.index(kept) - 1] != '-Xcompiler'

    # everything host-specific is handed over instead of passed directly
    for handed_over in (
        '-fno-strict-overflow',
        '-Wsign-compare',
        '-Wall',
        '-fPIC',
        '-pthread',
        '-Wl,--rpath=/somewhere/lib',
    ):
        assert handed_over in result, f'{handed_over} was dropped entirely'
        assert result[result.index(handed_over) - 1] == '-Xcompiler', (
            f'{handed_over} would be passed straight to nvcc'
        )


def test_every_linker_list_is_rewritten_for_nvcc():
    """All of distutils' linker lists are rewritten, not just linker_so.

    Which list distutils builds the link command from depends on the
    extension's language. This one declares C++, so the command comes from
    compiler_cxx plus the linker portion of linker_so_cxx. Rewriting only
    linker_so left that portion handing raw host flags to nvcc, which rejects
    them.
    """
    text = (TEMPLATE_DIR / 'pyjacob_cuda_setup.py.in').read_text()
    names = None
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Assign) and any(
            getattr(target, 'id', None) == 'LINKER_LISTS' for target in node.targets
        ):
            names = set(ast.literal_eval(node.value))
    assert names is not None, 'LINKER_LISTS is gone; the rewrite may be incomplete'

    required = {
        'linker_so',
        'linker_exe',
        'linker_so_cxx',
        'linker_exe_cxx',
        'compiler_cxx',
    }
    assert required <= names, f'not rewritten for nvcc: {sorted(required - names)}'
