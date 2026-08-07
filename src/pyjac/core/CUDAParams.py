"""
Module containing parameters that control CUDA code generation


Parameters
----------

Jacob_Unroll : int
  The number of reactions to attempt to place in each Jacobian reaction update subfile
Jacob_Spec_Unroll : int
  The number of species to attempt to place in each Jacobian species update subfile
Rates_Unroll : int
  The number of reactions to limit each reaction rate subfile to
Max_Lines : int
  The number of lines to attempt to limit each Jacobian reaction update subfile to
Max_Spec_Lines : int
  The number of lines to attempt to limit each Jacobian species update subfile to

"""

# Standard libraries
import os
from math import floor

Jacob_Unroll = 40
Jacob_Spec_Unroll = 40
Rates_Unroll = 250
Max_Lines = 10000
Max_Spec_Lines = 5000

DEFAULT_ARCH = 'sm_75'
"""str: default CUDA compute capability to generate code for

Turing, the oldest architecture that compiles offline on every CUDA toolkit
pyJac is tested against. CUDA 13 dropped offline compilation for Volta, so
``sm_70`` builds on 12.x but fails on 13.x; ``sm_75`` is the floor common to
both.

Override with ``--cuda-arch`` to match the target hardware; ``native`` compiles
for the GPU present on the build machine (CUDA 11.5 and later).
"""

REGISTERS_PER_SM = 65536
"""int: 32-bit registers per multiprocessor, for compute capability 5.0+

Fermi and Kepler had 32768. Every architecture from Maxwell onward has 65536.
"""

MAX_REGISTERS_PER_THREAD = 255
"""int: hardware cap on registers a single thread may use, compute 5.0+

Fermi capped this at 63.
"""

STATIC_SHARED_BYTES = 49152
"""int: shared memory per block usable without an explicit opt-in

Compute capability 7.0 and later expose more shared memory than this -- 96 KB
on Volta, up to 228 KB on Hopper -- but only to kernels that request it through
``cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)``.
pyJac declares its shared memory statically, so 48 KB is the portable limit.
"""


def get_L1_size(L1_Preferred):
    """
    Returns the size (in number of doubles) of the L1 cache

    Parameters
    ----------
    L1_Preferred : bool
        If true, prefer a larger L1 cache over more shared memory (recommended)
    """
    if L1_Preferred:
        return STATIC_SHARED_BYTES / 8  # doubles
    else:
        return 16384 / 8  # doubles


def get_shared_size(L1_Preferred):
    """
    Returns the size (in number of doubles) of shared memory

    Parameters
    ----------
    L1_Preferred : bool
        If true, prefer a larger L1 cache over more shared memory (recommended)
    """
    if not L1_Preferred:
        return STATIC_SHARED_BYTES / 8  # doubles
    else:
        return 16384 / 8  # doubles


def get_register_count(num_blocks, num_threads):
    """
    Returns the number of registers available per thread

    The result is written to the ``regcount`` file for use as nvcc's
    ``-maxrregcount``, which requires an integer.

    Parameters
    ----------
    num_blocks : int
        The number of blocks to target per kernel launch
    num_threads : int
        The number of threads to target per kernel launch
    """
    per_thread = REGISTERS_PER_SM // (num_blocks * num_threads)
    return max(min(per_thread, MAX_REGISTERS_PER_THREAD), 1)


def write_launch_bounds(
    builddir, blocks_per_sm=8, num_threads=64, L1_PREFERRED=True, no_shared=False
):
    """Creates the launch_bounds.cuh file that may be included by CUDA solvers

    Parameters
    ----------

    builddir : str
        The directory to place the source file in
    blocks_per_sm : int, optional
        The number of blocks to target per kernel launch
    num_threads : int, optional
        The number of threads per block in the per kernel launch
    L1_PREFERRED : bool, optional
        If true, prefer a larger L1 cache over more shared memory (recommended)
    no_shared : bool, optional
        If false, turn off shared memory

    Returns
    -------
    None

    """
    shared_per_block = (
        int(floor(get_shared_size(L1_PREFERRED) / blocks_per_sm))
        if not no_shared
        else 0
    )
    with open(os.path.join(builddir, 'launch_bounds.cuh'), 'w') as file:
        file.write(
            '#ifndef LAUNCH_BOUNDS_CUH\n'
            '#define LAUNCH_BOUNDS_CUH\n'
            f'#define TARGET_BLOCK_SIZE ({num_threads})\n'
            + f'#define TARGET_BLOCKS ({blocks_per_sm})\n'
            + ('' if no_shared else '//shared memory active\n')
            + f'#define SHARED_SIZE ({shared_per_block}'
            + ' * sizeof(double))\n'
            + (
                '//Large L1 cache active\n#define PREFERL1\n'
                if L1_PREFERRED
                else '//Large shared memory active\n'
            )
            + '#endif\n'
        )
    with open(os.path.join(builddir, 'regcount'), 'w') as file:
        file.write(f'{get_register_count(blocks_per_sm, num_threads)}')
