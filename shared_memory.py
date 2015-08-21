"""Handles shared memory usage to accelerate memory accesses for CUDA"""

from math import floor
import CUDAParams
import utils
import os


class shared_memory_manager(object):
    def __init__(self, blocks_per_sm=8, num_threads=64, L1_PREFERRED=True):
        SHARED_MEMORY_SIZE = CUDAParams.get_shared_size(L1_PREFERRED)
        self.blocks_per_sm = blocks_per_sm
        self.num_threads = num_threads
        self.skeleton = 'shared_temp[{}]'
        self.shared_dict = []
        self.shared_per_block = int(floor(SHARED_MEMORY_SIZE / self.blocks_per_sm))
        self.shared_per_thread = int(floor(self.shared_per_block / self.num_threads))
        self.shared_indexes = range(self.shared_per_thread)
        self.eviction_marking = [False for i in range(self.shared_per_thread)]
        self.on_eviction = None

    def force_eviction(self):
        for index in self.shared_dict[:]:
            # remove it
            self.shared_dict.remove(index)
            self.shared_indexes.append(index[1])
            if self.on_eviction is not None:
                self.on_eviction(index[0], self.__get_string(index[1]))

    def set_on_eviction(self, func):
        self.on_eviction = func

    def reset(self):
        self.shared_dict = []
        self.shared_indexes = range(self.shared_per_thread)
        self.eviction_marking = []

    def write_init(self, file, indent=4):
        file.write(''.join([' ' for i in range(indent)]) + 'extern __shared__ double ' + self.skeleton.format('') +
                   utils.line_end['cuda'])

    def load_into_shared(self, file, variables, estimated_usage=None, indent=2, load=None):
        old_variables = [x[0] for x in self.shared_dict]
        old_index = [x[1] for x in self.shared_dict]
        # simply clear old
        # self.shared_dict = []
        # self.shared_indexes = range(self.shared_per_thread)
        # see if anything has been marked for eviction
        if len(self.eviction_marking):
            for i, var in enumerate(old_variables):
                if var not in variables and self.eviction_marking[i]:
                    # remove it
                    index = next(s for s in self.shared_dict if s[0] == var)
                    self.shared_dict.remove(index)
                    self.shared_indexes.append(index[1])
                    if self.on_eviction is not None:
                        self.on_eviction(var, self.__get_string(index[1]))
        if estimated_usage is not None:
            variables = [(x[1], estimated_usage[x[0]]) for x in
                         sorted(enumerate(variables), key=lambda x: estimated_usage[x[0]], reverse=True)]
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
                    if var in old_variables:
                        index = old_index[old_variables.index(var)]
                        if index not in self.shared_indexes:
                            # someone checked it out, swap
                            stored = next((x for x in self.shared_dict if index == x[1]), None)
                            self.shared_dict.remove(stored)
                            self.shared_dict.append((stored[0], self.shared_indexes.pop(0)))
                            if self.on_eviction is not None:
                                print(stored)
                                self.on_eviction(stored[0], self.__get_string(stored[1]))
                    else:
                        index = self.shared_indexes.pop(0)
                    self.shared_dict.append((var, index))
        if estimated_usage:
            # add any usage = 1 ones if space
            for var, usage in variables:
                if not any(var == val[0] for val in self.shared_dict):
                    if len(self.shared_dict) < self.shared_per_thread:
                        self.shared_dict.append((var, self.shared_indexes.pop(0)))
        # need to write loads for any new vars
        for var in self.shared_dict:
            if not var[0] in old_variables:
                if load is not None:
                    index = next(i for i, v in enumerate(variables) if v[0] == var[0])
                    if not load[index]:
                        continue
                file.write(' ' * indent + self.__get_string(var[1]) + ' = ' + var[0] + utils.line_end[
                        'cuda'])

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
