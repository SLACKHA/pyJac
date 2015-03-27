"""Handles shared memory usage to accelerate memory accesses for CUDA"""

from math import floor
import CUDAParams
import utils
import os
def write_blank(builddir, blocks_per_sm = 8, num_threads = 64, L1_PREFERRED = True):
    with open(os.path.join(builddir, "launch_bounds.cuh"), "w") as file:
        file.write('#ifndef LAUNCH_BOUNDS_CUH\n'
                   '#define LAUNCH_BOUNDS_CUH\n'
                   '#define TARGET_BLOCK_SIZE ({})\n'.format(num_threads) + 
                   '#define TARGET_BLOCKS ({})\n'.format(blocks_per_sm) +
                   '#define SHARED_SIZE (0)\n' +
                   ('#define PREFERL1\n' if L1_PREFERRED else '') +
                   '#endif\n')
    with open(os.path.join(builddir, 'regcount'), 'w') as file:
        file.write(CUDAParams.get_register_count(blocks_per_sm, num_threads))
class shared_memory_manager(object):
    def __init__(self, builddir, blocks_per_sm = 8, num_threads = 64, L1_PREFERRED=True):
        SHARED_MEMORY_SIZE = CUDAParams.get_shared_size(L1_PREFERRED)
        self.blocks_per_sm = blocks_per_sm
        self.num_threads = num_threads
        self.skeleton = 'shared_temp[{}]'
        self.shared_dict = []
        self.shared_per_block = int(floor(SHARED_MEMORY_SIZE / self.blocks_per_sm))
        self.shared_per_thread = int(floor(self.shared_per_block / self.num_threads))
        self.shared_indexes = range(self.shared_per_thread)
        self.eviction_marking = []
        with open(os.path.join(builddir, 'launch_bounds.cuh'), "w") as file:
            file.write('#ifndef LAUNCH_BOUNDS_CUH\n'
                       '#define LAUNCH_BOUNDS_CUH\n'
                       '#define TARGET_BLOCK_SIZE ({})\n'.format(self.num_threads) + 
                       '#define TARGET_BLOCKS ({})\n'.format(self.blocks_per_sm) +
                       '#define SHARED_SIZE ({} * sizeof(double))\n'.format(self.shared_per_block) + 
                       ('#define PREFERL1\n' if L1_PREFERRED else '') +
                       '#endif\n')
        with open(os.path.join(builddir, 'regcount'), 'w') as file:
            file.write(CUDAParams.get_register_count(blocks_per_sm, num_threads))

    def reset(self):
        self.shared_dict = []
        self.shared_indexes = range(self.shared_per_thread)
        self.eviction_marking = []

    def write_init(self, file, indent = 4):
        file.write(''.join([' ' for i in range(indent)]) +   'extern __shared__ double ' + self.skeleton.format('') + utils.line_end['cuda'])

    def load_into_shared(self, file, variables, estimated_usage = None, indent=2):
        old_variables = [x[0] for x in self.shared_dict]
        old_index = [x[1] for x in self.shared_dict]
        #see if anything has been marked for eviction
        if len(self.eviction_marking):
            for i, var in enumerate(old_variables):
                if var not in variables and self.eviction_marking[i]:
                    #remove it
                    index = next(s for s in self.shared_dict if s[0] == var)
                    self.shared_dict.remove(index)
                    self.shared_indexes.append(index[1])
        if estimated_usage is not None:
            variables = [(x[1], estimated_usage[x[0]]) for x in sorted(enumerate(variables), key = lambda x: estimated_usage[x[0]], reverse=True)]
        for variable in variables:
            if estimated_usage is not None:
                var, usage = variable
            else:
                var = variable
                usage = None
            if not any(var == val[0] for val in self.shared_dict):
                if usage == 0 or usage == 1:
                    continue
                if len(self.shared_dict) < self.shared_per_thread:
                    self.shared_dict.append((var, self.shared_indexes.pop(0)))
        if estimated_usage:
            #add any usage = 1 ones if space
            for var, usage in variables:
                if not any(var == val[0] for val in self.shared_dict):
                    if len(self.shared_dict) < self.shared_per_thread:
                        self.shared_dict.append((var, self.shared_indexes.pop(0)))
        #need to write loads for any new vars
        for var in self.shared_dict:
            if not var[0] in old_variables:
                file.write(''.join([' ' for i in range(indent)]) + self.__get_string(var[1]) + ' = ' + var[0] + utils.line_end['cuda'])

    def evict(self, variables):
        for variable in variables:
            found = next((x for x in self.shared_dict if x[0] == variable), None)
            if found is not None:
                self.shared_dict.remove(found)
                self.shared_indexes.append(found[1])

    def mark_for_eviction(self, variables):
        """Like the eviction method, but only evicts if not used in the next load"""
        self.eviction_marking = [var[0] in variables for var in self.shared_dict]
    def __get_string(self, index):
        if index == 0:
            return self.skeleton.format('threadIdx.x')
        else:
            return self.skeleton.format('threadIdx.x + {} * blockDim.x'.format(index))          

    def get_index(self, variable):
        return next((x for x in self.shared_dict if variable == x[0]), None)

    def get_array(self, lang, variable, index, twod=None):
        if index is None:
            stringified = variable
        else:
            stringified = utils.get_array(lang, variable, index)
        var = self.get_index(stringified)
        if var is None:
            return stringified
        else:
            return self.__get_string(var[1])

