"""Module containing parameters that control CUDA code generation
"""
import os
from math import floor

class CudaMemStrats:
    Global, Local = range(2)

class JacRatesCacheStrat:
    Include, Exclude = range(2)

MemoryStrategy = CudaMemStrats.Local
JacRateStrat = JacRatesCacheStrat.Include

Jacob_Unroll = 20

def is_global():
    return MemoryStrategy == CudaMemStrats.Global

def get_L1_size(L1_Preferred):
    if L1_Preferred:
        return 49152 / 8 #doubles
    else:
        return 16384 / 8 #doubles

def get_shared_size(L1_Preferred):
    if not L1_Preferred:
        return 49152 / 8 #doubles
    else:
        return 16384 / 8 #doubles

def get_register_count(num_blocks, num_threads):
    return max(min((32768 / num_blocks) / num_threads, 63), 1)

def write_launch_bounds(builddir, blocks_per_sm = 8, num_threads = 64, L1_PREFERRED=True):
    shared_per_block = int(floor(get_shared_size(L1_PREFERRED) / blocks_per_sm))
    with open(os.path.join(builddir, 'launch_bounds.cuh'), "w") as file:
            file.write('#ifndef LAUNCH_BOUNDS_CUH\n'
                       '#define LAUNCH_BOUNDS_CUH\n'
                       '#define TARGET_BLOCK_SIZE ({})\n'.format(num_threads) + 
                       '#define TARGET_BLOCKS ({})\n'.format(blocks_per_sm) +
                       '#define SHARED_SIZE ({} * sizeof(double))\n'.format(shared_per_block) + 
                       ('#define PREFERL1\n' if L1_PREFERRED else '') +
                       '#endif\n')
    with open(os.path.join(builddir, 'regcount'), 'w') as file:
        file.write('{}'.format(get_register_count(blocks_per_sm, num_threads)))