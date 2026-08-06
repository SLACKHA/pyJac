"""The CUDA target architecture must be configurable, not hardcoded.

pyJac emitted ``-arch=sm_20`` at every compile site. Fermi support was removed
in CUDA 9 (2017), so the CUDA backend could not build on any current toolkit.

These tests exercise the Python side only. Compiling the generated CUDA needs
nvcc and is covered by the CUDA job in CI.
"""

import re
from pathlib import Path
from string import Template

import pytest

from pyjac.core import CUDAParams
from pyjac.libgen import file_struct, flags, libgen
from pyjac.pywrap import pywrap_gen

TEMPLATE_DIR = Path(pywrap_gen.__file__).parent


def test_no_hardcoded_fermi_architecture():
    """sm_20 must not appear anywhere; it cannot be compiled today."""
    offenders = []
    for path in [*Path('src/pyjac').rglob('*.py'), *TEMPLATE_DIR.glob('*.in')]:
        if 'sm_20' in path.read_text():
            offenders.append(str(path))
    assert not offenders, f'sm_20 still hardcoded in: {offenders}'


def test_default_arch_is_a_valid_compute_capability():
    assert re.fullmatch(r'sm_\d{2,3}', CUDAParams.DEFAULT_ARCH)


def test_cuda_flags_carry_no_architecture():
    """The arch is a per-build choice, so it must not sit in the static flags."""
    assert not any('-arch' in flag for flag in flags['cuda'])


def test_compiler_emits_the_requested_arch():
    """file_struct carries the arch through to the nvcc command line."""
    struct = file_struct('cuda', 'cuda', 'jacob', ['.'], [], '.', '.', shared=False)
    assert struct.cuda_arch == CUDAParams.DEFAULT_ARCH

    struct.cuda_arch = 'sm_90'
    # compiler() builds its argument list before invoking nvcc; rebuild the
    # same prefix here rather than requiring a CUDA toolkit to be installed
    args = [libgen.cmd_compile['cuda'], *flags['cuda'], f'-arch={struct.cuda_arch}']
    assert '-arch=sm_90' in args
    assert not any(a == '-arch=sm_20' for a in args)


def test_c_builds_are_unaffected():
    """Only the CUDA path gains an arch flag."""
    struct = file_struct('c', 'c', 'jacob', ['.'], [], '.', '.', shared=False)
    assert struct.build_lang == 'c'
    assert not any('-arch' in flag for flag in flags['c'])


def test_cuda_setup_template_takes_the_arch():
    """The wrapper's setup template substitutes the arch like other options."""
    template = TEMPLATE_DIR / 'pyjacob_cuda_setup.py.in'
    text = template.read_text()
    assert '$cudaarch' in text
    assert 'sm_20' not in text

    filled = Template(text).safe_substitute(
        homepath='/tmp/home',
        buildpath='/tmp/build',
        outpath='/tmp/out',
        libname='libcu_pyjac.a',
        cudaarch='sm_86',
    )
    assert '-arch=sm_86' in filled
    assert '$cudaarch' not in filled


# --------------------------------------------------------------------------
# Resource limits: Fermi's numbers understate every architecture since.
# --------------------------------------------------------------------------


def test_register_count_is_an_integer():
    """regcount feeds nvcc's -maxrregcount, which rejects a float."""
    for blocks, threads in [(8, 64), (8, 128), (16, 256), (1, 32), (64, 1024)]:
        count = CUDAParams.get_register_count(blocks, threads)
        assert isinstance(count, int), f'{blocks}x{threads} gave {count!r}'


def test_register_count_respects_the_hardware_cap():
    """A tiny launch cannot ask for more registers than a thread may hold."""
    assert CUDAParams.get_register_count(1, 1) == CUDAParams.MAX_REGISTERS_PER_THREAD
    assert CUDAParams.MAX_REGISTERS_PER_THREAD == 255


def test_register_count_stays_positive():
    """A very large launch still needs at least one register per thread."""
    assert CUDAParams.get_register_count(1024, 1024) == 1


def test_register_budget_is_post_fermi():
    assert CUDAParams.REGISTERS_PER_SM == 65536


def test_shared_memory_is_the_portable_static_limit():
    """48 KB is the most a kernel gets without opting in via cudaFuncSetAttribute.

    Volta and later expose more, but only as dynamic shared memory that the
    kernel must request. pyJac declares its shared memory statically.
    """
    assert CUDAParams.STATIC_SHARED_BYTES == 49152
    assert CUDAParams.get_shared_size(L1_Preferred=False) == 49152 / 8
    assert CUDAParams.get_L1_size(L1_Preferred=True) == 49152 / 8


@pytest.mark.parametrize('no_shared', [False, True])
def test_launch_bounds_written(tmp_path, no_shared):
    """launch_bounds.cuh and regcount are emitted with usable values."""
    CUDAParams.write_launch_bounds(
        str(tmp_path), blocks_per_sm=8, num_threads=64, no_shared=no_shared
    )

    bounds = (tmp_path / 'launch_bounds.cuh').read_text()
    assert '#define TARGET_BLOCK_SIZE (64)' in bounds
    assert '#define TARGET_BLOCKS (8)' in bounds

    regcount = (tmp_path / 'regcount').read_text()
    assert regcount == '128'
    assert int(regcount) <= CUDAParams.MAX_REGISTERS_PER_THREAD
