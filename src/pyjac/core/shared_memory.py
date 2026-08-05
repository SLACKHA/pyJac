"""Handles shared memory usage to accelerate memory accesses for CUDA
"""

# Standard libraries
import os
from math import floor

# Local imports
from .. import utils
from . import CUDAParams

class variable(object):
    """
    Class that represents an array/index pair.  Used for in the internal
    dicitonary of the `shared_memory_manager` for identification and
    tracking of `variable` usage for eviction.
    """
    def __init__(self, base, index, lang='cuda'):
        """
        Creates a `variable` with given base and index

        Parameters
        ----------

        base : str
            The name of the array
        index : int
            The index in the array

        """
        self.base = base
        self.index = index
        self.last_use_count = 0
        self.lang = lang

    def __eq__(self, other):
        """Tests `variable` equality"""
        if self.index is None:
            return self.base == other.base
        return self.base == other.base and self.index == other.index

    def reset(self):
        """Reset the usage count of this `variable`
        """
        self.last_use_count = 0

    def update(self):
        """Increment the usage count of this `variable`
        """
        self.last_use_count += 1

    def to_string(self):
        """Converts this `variable` to a string representation
        """
        if self.index is None:
            return self.base
        return utils.get_array(self.lang, self.base, self.index)


class shared_memory_manager(object):
    """Manager for GPU shared memory.
    """
    def __init__(self, blocks_per_sm=8, num_threads=64, L1_PREFERRED=True):
        """Creates a shared memory manager

        Parameters
        ----------
        blocks_per_sm : int, optional
            The number of blocks / streaming multiprocessor to target
        num_threads : int, optional
            The number of threads / block expected in kernel launches
        L1_PREFERRED : bool, optional
            Whether or not to prefer a larger L1 cache over more shared memory
            (recommended).

        Notes
        -----
        For ease, a single SMM is used in the entire program.
        Thus this class has methods for setting the state/behaviour
        (e.g., `reset`, and `set_on_eviction`).

        """
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
        self.self_eviction_strategy = lambda x: x.last_use_count >= 2

    def force_eviction(self):
        """Forces eviction of the manager's internal dictionary.

        Notes
        -----
        The internal dictionary will be reset, and (if supplied) the
        `on_eviction` function will be called on each evicted entry.

        """
        key_copy = [x for x in self.shared_dict.keys()]
        for shared_index in key_copy:
            self.evict(shared_index)

    def evict_longest_gap(self):
        """Evicts entry in the internal dictionary the longest without use.
        """
        if len(self.shared_dict):
            ind = max((x for x in self.shared_dict if self.eviction_marking[x]),
                      key=lambda k: self.shared_dict[k].last_use_count
                      )
            self.evict(ind)

    def evict(self, shared_index):
        """Removes the entry at shared_index from the internal dictionary

        Parameters
        ----------
        shared_index : int
            The key to remove from the internal dictionary

        Notes
        -----
        If set, `on_eviction` will be called.

        """
        var = self.shared_dict[shared_index]
        del self.shared_dict[shared_index]
        self.shared_indexes.append(shared_index)
        self.eviction_marking[shared_index] = False
        if self.on_eviction is not None:
            self.on_eviction(var, self.__get_string(shared_index), shared_index)

    def add_to_dictionary(self, val):
        """Adds the value to the next available dictionary location

        Parameters
        ----------
        val : `variable`
            The value to add to the dictionary

        """
        assert len(self.shared_indexes)
        self.shared_dict[self.shared_indexes.pop()] = val

    def set_on_eviction(self, func):
        """Sets a callback function that is called upon eviction of a variable from the
           internal dictionary

        Parameters
        ----------
        func : `function`
            Function that takes one arguement (the evicted variable)

        Returns
        -------
        None

        """
        self.on_eviction = func

    def reset(self):
        """Resets the SMM for use by other methods/callers

        Returns
        -------
        None

        """
        self.shared_dict = {}
        self.shared_indexes = list(range(self.shared_per_thread))
        self.eviction_marking = [False for x in range(self.shared_per_thread)]
        self.on_eviction = None

    def write_init(self, file, indent=4):
        """Convenience method to define shared memory for CUDA

        Parameters
        ----------
        file : `File`
            Open `File` object to write to
        indent : int, optional
            The number of spaces to use in the indent

        Returns
        -------
        None

        """
        file.write(''.join([' ' for i in range(indent)]) +
                   'extern volatile __shared__ double ' +
                   self.skeleton.format('') + utils.line_end['cuda']
                   )

    def load_into_shared(self, file, variables, estimated_usage=None,
                         indent=2, load=True
                         ):
        """The main SMM method, loads/evicts variables based upon estimated
        usage and stagnancy.

        Parameters
        ----------
        file : `File`
            Open `File` object to write to
        variables : list of `variable`
            List of variables to consider loading
        estimated_usage : list of float, optional
            If specified, these will be used to prioritize variable additon
        indent : int, optional
            The number of spaces to use in the indentation
        load : bool, optional
            If ``True`` (default), a load into the internal dictionary will be
            written to the file. If ``False``, this will must be handled by
            the calling routine.

        Returns
        -------
        List of `bool` to indicate if variables are loaded in shared memory.

        """
        #save old variables
        old_index = []
        old_variables = []
        if len(self.shared_dict):
            old_index, old_variables = zip(*self.shared_dict.items())

        #update all the old variables usage counts
        for x in old_variables:
            x.update()

        #check for self_eviction
        if self.self_eviction_strategy is not None:
            for ind, val in self.shared_dict.items():
                #if qualifies for self eviction and not in current set
                if self.self_eviction_strategy(val) and not val in variables:
                    self.eviction_marking[ind] = True
                elif val in variables:
                    self.eviction_marking[ind] = False

        #sort by usage if available
        if estimated_usage is not None:
            variables = [(x[1], estimated_usage[x[0]]) for x in
                         sorted(enumerate(variables),
                         key=lambda x: estimated_usage[x[0]], reverse=True)
                         ]

        #now update for new variables
        for thevar in variables:
            if estimated_usage is not None:
                var, usage = thevar
            else:
                var = thevar
                usage = None
            #don't re-add if it's already in
            if not var in self.shared_dict.values():
                #skip barely used ones
                if usage <= 1:
                    continue
                #if we have something marked for eviction, now's the time
                if (len(self.shared_dict) >= self.shared_per_thread and
                    self.eviction_marking.count(True)
                    ):
                    self.evict_longest_gap()
                #add it if possible
                if len(self.shared_dict) < self.shared_per_thread:
                    self.add_to_dictionary(var)

        if estimated_usage:
            # add any usage = 1 ones if space
            for var, usage in variables:
                if not var in self.shared_dict.values():
                    if len(self.shared_dict) < self.shared_per_thread:
                        self.add_to_dictionary(var)
        if load is True:
            # need to write loads for any new vars
            for ind, val in self.shared_dict.items():
                if not val in old_variables:
                    file.write(' ' * indent + self.__get_string(ind) +
                               ' = ' + val.to_string() +
                               utils.line_end['cuda']
                               )

        return {k:(v not in old_variables)
                for k, v in self.shared_dict.items()
                }

    def mark_for_eviction(self, variables):
        """Marks variables for possible eviction upon next load_into_shared call

        Parameters
        ----------
        variables : list of `variable`
            List of variables to consider for eviction

        """
        self.eviction_marking = [var in variables for var in
                                 self.shared_dict.values()
                                 ]

    def __get_string(self, index):
        """Convenience method to get correct GPU shared memory addressing

        Parameters
        ----------
        index : int
            Index of GPU block.

        Returns
        -------
        str
            String with shared memory addressing.

        """
        if index == 0:
            return self.skeleton.format('threadIdx.x')
        else:
            return self.skeleton.format('threadIdx.x + '
                                        '{} * blockDim.x'.format(index)
                                        )

    def get_index(self, var):
        """Checks to see if a variable is in the internal dictionary.
        If so returns internal index and variable

        Parameters
        ----------
        var : `variable`
            The variable to check

        Returns
        -------
        our_ind : int
            Index of variable in internal dictionary
        our_var : `variable`
            Variable found in internal dictionary

        """
        our_ind, our_var = next((val for val in self.shared_dict.items()
                                if val[1] == var), (None, None)
                                )
        return our_ind, our_var

    def get_array(self, lang, thevar, index, twod=None):
        """A substitute for `utils.get_array`.

        If the variable is in our internal dictionary returns shared memory
        address, otherwise calls `utils.get_array`.

        Parameters
        ----------
        lang : {'c', 'cuda'}
            Programming language.
        thevar : `variable`
            Variable of interest.
        index : int
            Index in the array.
        twod : int
            Not used in this function.

        Returns
        -------
        name : str
            String with indexed array.

        """
        var = variable(thevar, index, lang)
        our_ind, our_var = self.get_index(var)
        if our_var is not None:
            #mark as used
            our_var.reset()
            #and return the shared string
            name = self.__get_string(our_ind)
        else:
            name = var.to_string()
        return name
