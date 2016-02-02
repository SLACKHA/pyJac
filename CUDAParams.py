"""Module containing parameters that control CUDA code generation
"""

# Standard libraries
import os
from math import floor


class JacRatesCacheStrat:
    Include, Exclude = range(2)

JacRateStrat = JacRatesCacheStrat.Include
ResetOnJacUnroll = True

Jacob_Unroll = 40
Jacob_Spec_Unroll = 40
Rates_Unroll = 250
Max_Lines = 10000
Max_Spec_Lines = 5000


def get_L1_size(L1_Preferred):
    if L1_Preferred:
        return 49152 / 8  # doubles
    else:
        return 16384 / 8  # doubles


def get_shared_size(L1_Preferred):
    if not L1_Preferred:
        return 49152 / 8  # doubles
    else:
        return 16384 / 8  # doubles


def get_register_count(num_blocks, num_threads):
    return max(min((32768 / num_blocks) / num_threads, 63), 1)


def write_launch_bounds(builddir, blocks_per_sm=8, num_threads=64, L1_PREFERRED=True, no_shared=False):
    shared_per_block = int(floor(get_shared_size(L1_PREFERRED) / blocks_per_sm)) if not no_shared else 0
    with open(os.path.join(builddir, 'launch_bounds.cuh'), "w") as file:
        file.write('#ifndef LAUNCH_BOUNDS_CUH\n'
                   '#define LAUNCH_BOUNDS_CUH\n'
                   '#define TARGET_BLOCK_SIZE ({})\n'.format(num_threads) +
                   '#define TARGET_BLOCKS ({})\n'.format(blocks_per_sm) +
                   ('' if no_shared else '//shared memory active\n') +
                   '#define SHARED_SIZE ({} * sizeof(double))\n'.format(shared_per_block) +
                   ('//Large L1 cache active\n#define PREFERL1\n' if L1_PREFERRED else '//Large shared memory active\n') +
                   '#endif\n')
    with open(os.path.join(builddir, 'regcount'), 'w') as file:
        file.write('{}'.format(get_register_count(blocks_per_sm, num_threads)))
