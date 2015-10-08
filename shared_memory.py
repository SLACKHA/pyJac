"""Handles shared memory usage to accelerate memory accesses for CUDA"""

from math import floor
import CUDAParams
import utils
import os

class variable(object):
    def __init__(self, base, index, lang='cuda'):
        self.base = base
        self.index = index
        self.last_use_count = 0
        self.lang = lang
    def __eq__(self, other):
        return self.base == other.base and self.index == other.index
    def reset(self):
        self.last_use_count = 0
    def update(self):
        self.last_use_count += 1
    def to_string(self):
        return utils.to_string(self.lang, self.base, self.index)

class shared_memory_manager(object):
    def __init__(self, blocks_per_sm=8, num_threads=64, L1_PREFERRED=True):
        SHARED_MEMORY_SIZE = CUDAParams.get_shared_size(L1_PREFERRED)
        self.blocks_per_sm = blocks_per_sm
        self.num_threads = num_threads
        self.skeleton = 'shared_temp[{}]'
        self.shared_dict = {}
        self.shared_per_block = int(floor(SHARED_MEMORY_SIZE / self.blocks_per_sm))
        self.shared_per_thread = int(floor(self.shared_per_block / self.num_threads))
        self.shared_indexes = [True for i in range(self.shared_per_thread)]
        self.eviction_marking = [False for i in range(self.shared_per_thread)]
        self.on_eviction = None
        self.self_eviction_strategy = lambda x: x.last_use_count > 2

    def force_eviction(self):
        for shared_index in self.shared_dict.iterkeys(): 
            self.evict(shared_index)

    def evict_longest_gap(self):
        """evicts the entry that has gone the longest without use"""
        if len(self.shared_dict):
            ind = max(x for x in self.shared_dict if self.mark_for_eviction[x],
                      key=lambda k: self.shared_dict[k].last_use_count)
            self.evict(ind)

    def evict(self, shared_index):
        del self.shared_dict[shared_index]
        self.shared_indexes.append(shared_index)
        self.mark_for_eviction[shared_index] = False
        if self.on_eviction is not None:
            self.on_eviction(var, self.__get_string(shared_index))
    def add_to_dictionary(self, val):
        assert len(self.shared_indexes)
        self.shared_dict[self.shared_indexes.pop()] = val

    def set_on_eviction(self, func):
        self.on_eviction = func

    def reset(self):
        self.shared_dict = {}
        self.shared_indexes = range(self.shared_per_thread)
        self.eviction_marking = []
        self.on_eviction = None

    def write_init(self, file, indent=4):
        file.write(''.join([' ' for i in range(indent)]) + 'extern __shared__ double ' + 
                   self.skeleton.format('') + utils.line_end['cuda'])

    def load_into_shared(self, file, variables, estimated_usage=None, indent=2, load=True):
        #save old variables
        old_index, old_variables = zip(*self.shared_dict.iteritems())

        #update all the old variables usage counts
        for x in old_variables:
            x.update()

        #check for self_eviction
        if self.self_eviction_strategy is not None:
            for ind, val in self.shared_dict.iteritems():
                #if qualifies for self eviction and not in current set
                if self.self_eviction_strategy(val) and not val in variables:
                    self.mark_for_eviction[ind] = True

        #sort by usage if available
        if estimated_usage is not None:
            variables = [(x[1], estimated_usage[x[0]]) for x in
                         sorted(enumerate(variables), key=lambda x: estimated_usage[x[0]], reverse=True)]

        #now update for new variables
        for thevar in variables:
            if estimated_usage is not None:
                var, usage = thevar
            else:
                var = thevar
                usage = None
            #don't re-add if it's already in
            if not var in self.shared_dict.itervalues():
                #skip barely used ones
                if usage <= 1:
                    continue
                #if we have something marked for eviction, now's the time
                if len(self.shared_dict) >= self.shared_per_thread and \
                        self.mark_for_eviction.count(True):
                        self.evict_longest_gap()
                #add it if possible
                if len(self.shared_dict) < self.shared_per_thread:
                    self.add_to_dictionary(var)

        if estimated_usage:
            # add any usage = 1 ones if space
            for var, usage in variables:
                if not var in self.shared_dict.itervalues()
                    if len(self.shared_dict) < self.shared_per_thread:
                        self.add_to_dictionary(var)
        if load is True:
            # need to write loads for any new vars
            for ind, val in self.shared_dict.itervalues():
                if not var in old_variables:
                    file.write(' ' * indent + self.__get_string(ind) + ' = ' + var.to_string() +
                                utils.line_end['cuda'])

        return [x for x in self.shared_dict.itervalues() if not x in old_variables]

    def mark_for_eviction(self, variables):
        """Like the eviction method, but only evicts if not used in the next load"""
        self.eviction_marking = [var in variables for var in self.shared_dict.itervalues()]

    def __get_string(self, index):
        if index == 0:
            return self.skeleton.format('threadIdx.x')
        else:
            return self.skeleton.format('threadIdx.x + {} * blockDim.x'.format(index))

    def get_index(self, thevar):
        return next((x for x in self.shared_dict if thevar == self.shared_dict[x]), None)

    def get_array(self, lang, thevar, index, twod=None):
        var = variable(thevar, index)
        ind = self.get_index(var)
        #mark as used
        self.shared_dict[ind].reset()
        return self.__get_string(var[1])
