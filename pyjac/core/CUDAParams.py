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


def get_L1_size(L1_Preferred):
    """
    Returns the size (in number of doubles) of the L1 cache for sm_20

    Parameters
    ----------
    L1_Preferred : bool
        If true, prefer a larger L1 cache over more shared memory (recommended)
    """
    if L1_Preferred:
        return 49152 / 8  # doubles
    else:
        return 16384 / 8  # doubles


def get_shared_size(L1_Preferred):
    """
    Returns the size (in number of doubles) of shared memory for sm_20

    Parameters
    ----------
    L1_Preferred : bool
        If true, prefer a larger L1 cache over more shared memory (recommended)
    """
    if not L1_Preferred:
        return 49152 / 8  # doubles
    else:
        return 16384 / 8  # doubles


def get_register_count(num_blocks, num_threads):
    """
    Returns the number of registers available per block for sm_20

    Parameters
    ----------
    num_blocks : int
        The number of blocks to target per kernel launch
    num_threads : int
        The number of threads to target per kernel launch
    """
    return max(min((32768 / num_blocks) / num_threads, 63), 1)


def write_launch_bounds(builddir, blocks_per_sm=8, num_threads=64,
                        L1_PREFERRED=True, no_shared=False
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
    shared_per_block = (int(floor(get_shared_size(L1_PREFERRED) / blocks_per_sm))
                        if not no_shared
                        else 0
                        )
    with open(os.path.join(builddir, 'launch_bounds.cuh'), "w") as file:
        file.write('#ifndef LAUNCH_BOUNDS_CUH\n'
                   '#define LAUNCH_BOUNDS_CUH\n'
                   '#define TARGET_BLOCK_SIZE ({})\n'.format(num_threads) +
                   '#define TARGET_BLOCKS ({})\n'.format(blocks_per_sm) +
                   ('' if no_shared else '//shared memory active\n') +
                   '#define SHARED_SIZE ({}'.format(shared_per_block) +
                   ' * sizeof(double))\n' +
                   ('//Large L1 cache active\n#define PREFERL1\n'
                    if L1_PREFERRED else '//Large shared memory active\n'
                    ) + '#endif\n'
                   )
    with open(os.path.join(builddir, 'regcount'), 'w') as file:
        file.write('{}'.format(get_register_count(blocks_per_sm, num_threads)))
