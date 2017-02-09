"""
memory_manager.py - generators for defining / allocating / transfering memory
for kernel creation
"""

from .. import utils

from string import Template
import numpy as np
import loopy as lp

class memory_manager(object):
    """
    Aids in defining & allocating arrays for various languages
    """

    def __init__(self, lang):
        """
        Parameters
        ----------
        lang : str
            The language used in this memory initializer
        """
        self.arrays = []
        self.in_arrays = []
        self.out_arrays = []
        self.lang = lang
        self.has_init = {}
        self.memory_types = {'c' : 'double*',
                             'opencl' : 'cl_mem'}
        self.host_langs = {'opencl' : 'c',
                           'c' : 'c'}
        self.alloc_templates = {'opencl' :
                    Template('${name} = clCreateBuffer(context, ${memflag},'
                    ' ${buff_size} * sizeof(double),'
                    ' NULL, &return_code)'),
                'c' : Template('${name} = (double*)malloc(${buff_size} * sizeof(double))')}
        self.copy_in_templates = {'opencl' :
                    Template('clEnqueueWriteBuffer(queue, ${name}, CL_TRUE,'
                        ' 0, ${buff_size} * sizeof(double),'
                        ' ${host_buff}, 0, NULL, NULL)'),
                    'c' :
                    Template('memcpy(${name}, ${host_buff},'
                             ' ${buff_size} * sizeof(double))')}
        self.copy_out_templates = {'opencl' :
                    Template('clEnqueueReadBuffer(queue, ${name}, CL_TRUE,'
                        ' 0, ${buff_size} * sizeof(double),'
                        ' ${host_buff}, 0, NULL, NULL)'),
                    'c' :
                    Template('memcpy(${host_buff}, ${name},'
                             ' ${buff_size} * sizeof(double))')}
        self.free_template = {'opencl' :
                    Template('clReleaseMemObject(${name})'),
                    'c' : Template('free(${name})')}

        self.host_inits = []

    def get_check_err_call(self, call):
        if self.lang == 'opencl':
            return Template('check_err(${call})').safe_substitute(call=call)
        else:
            return call

    def add_arrays(self, arrays=[], has_init={}, in_arrays=[],
        out_arrays=[]):
        """
        Adds arrays to the manager

        Parameters
        ----------
        arrays : list of :class:`lp.GlobalArg`
            The arrays to declare
        has_init : dict
            A mapping of array name -> constant initializer value
            Indiciating that these arrays must be set before execution
        in_arrays : list of str
            The array names that form the input to this kernel
        out_arrays : list of str
            The array names that form the output of this kernel

        Returns
        -------
        None
        """
        self.arrays.extend([x for x in arrays if not isinstance(x, lp.ValueArg)])
        self.in_arrays.extend(in_arrays)
        self.out_arrays.extend(out_arrays)
        self.has_init.update(has_init)

    @property
    def host_arrays(self):
        return self.in_arrays + self.out_arrays + self.host_inits

    @property
    def host_lang(self):
        return self.host_langs[self.lang]

    def get_defns(self):
        """
        Returns the definition strings for this memory manager's arrays

        Parameters
        ----------
        None

        Returns
        -------
        defn_str : str
            A string of global memory definitions
        """

        def __add(arraylist, lang, prefix, defn_list):
            for arr in arraylist:
                defn_list.append(self.memory_types[lang] + ' ' + prefix +
                    arr + utils.line_end[lang])

        defns = []
        #get all 'device' defns
        __add([x.name for x in self.arrays], self.lang, 'd_', defns)

        #process any host array that needs init but isn't an input array
        for arr in sorted(self.has_init):
            if not arr in self.host_arrays:
                self.host_inits.append(arr)

        #get host defns
        __add(self.host_inits, self.host_lang, 'h_', defns)

        #return defn string
        return '\n'.join(sorted(set(defns)))

    def get_mem_allocs(self, alloc_inputs=False):
        """
        Returns the allocation strings for this memory manager's arrays

        Parameters
        ----------
        alloc_inputs : bool
            If true, only define host arrays in self.in_arrays

        Returns
        alloc_str : str
            The string of memory allocations
        """

        def __get_alloc(name, lang, prefix):
            #if we're allocating inputs, call them something different
            #than the input args from python
            post_fix = '_local' if alloc_inputs else ''

            #if it's opencl, we need to declare the buffer type
            memflag = None
            if lang == 'opencl':
                memflag = 'CL_MEM_READ_WRITE'

            #find the corresponding loopy array, to get sizes
            dev_arr = next(x for x in self.arrays if x.name == name)

            #generate allocs
            alloc = self.alloc_templates[lang].safe_substitute(
                name=prefix + name + post_fix,
                memflag=memflag,
                buff_size=self._get_size(dev_arr))

            if alloc_inputs:
                #add a type
                alloc = self.memory_types[self.host_lang] + ' ' + alloc

            #generate allocs
            return_list = [alloc]

            #add error checking
            if lang == 'opencl':
                return_list.append(self.get_check_err_call('return_code'))

            #return
            return '\n'.join([r + utils.line_end[lang] for r in return_list])

        to_alloc = [next(x for x in self.arrays if x.name == y) for y in self.in_arrays] \
                    if alloc_inputs else self.arrays
        prefix = 'h_' if alloc_inputs else 'd_'
        lang = self.lang if not alloc_inputs else self.host_lang
        alloc_list = [__get_alloc(arr.name, lang, prefix) for arr in to_alloc]

        #do memsets where applicable
        if not alloc_inputs:
            for arr in sorted(self.has_init):
                assert arr in self.host_arrays, 'Cannot initialize device memory to a constant'
                prefix = 'h_'
                if arr not in self.in_arrays:
                    #needs to be alloced, since it isn't passed in
                    alloc_list.append(__get_alloc(arr, self.host_lang, prefix))
                if not alloc_inputs:
                    #find initial value
                    init_v = self.has_init[arr]
                    #find corresponding device array
                    dev_arr = next(x for x in self.arrays if x.name == arr)
                    if init_v == 0:
                        alloc_list.append(Template('memset(${prefix}${name}, 0, '
                                            '${buff_size} * sizeof(double))${end}').safe_substitute(
                                            prefix=prefix,
                                            name=arr,
                                            buff_size=self._get_size(dev_arr),
                                            end=utils.line_end[self.lang] + '\n'
                                            ))
                    else:
                        alloc_list.append(Template('for(int i_setter = 0; i_setter < ${buff_size}; ++i_setter)\n'
                                                   '    ${prefix}${name}[i_setter] = ${value}${end}').safe_substitute(
                                                    prefix=prefix,
                                                    name=arr,
                                                    buff_size=self._get_size(dev_arr),
                                                    value=init_v,
                                                    end=utils.line_end[self.lang] + '\n'))

        return '\n'.join(alloc_list)

    def _get_size(self, arr, subs_n='problem_size'):
        size = arr.shape
        str_size = []
        #remove 'problem_size' from shape if present, as it's baked into the various defns
        for x in size:
            #but allow for substitutions
            if str(x) == 'problem_size' and subs_n:
                str_size.append(subs_n)
        size = [x for x in size if str(x) != 'problem_size']

        if size:
            size = str(np.cumprod(size, dtype=np.int32)[-1])
            str_size.append(size)
        return ' * '.join(str_size)

    def _mem_transfers(self, to_device=True):
        arr_list = self.in_arrays if to_device else self.out_arrays
        arr_maps = {x : next(y for y in self.arrays if x == y.name) for x in arr_list}
        templates = self.copy_in_templates if to_device else self.copy_out_templates

        return '\n'.join([templates[self.lang].safe_substitute(
            name='d_' + arr, host_buff='h_' + arr,
            buff_size=self._get_size(arr_maps[arr])) +
                utils.line_end[self.lang] for arr in arr_list])

    def get_mem_transfers_in(self):
        """
        Generates the memory transfers into the device before kernel execution

        Parameters
        ----------
        None

        Returns
        -------
        mem_transfer_in : str
            The string to perform the memory transfers before execution
        """

        return self._mem_transfers(to_device=True)

    def get_mem_transfers_out(self):
        """
        Generates the memory transfers into the back to the host after kernel execution

        Parameters
        ----------
        None

        Returns
        -------
        mem_transfer_out : str
            The string to perform the memory transfers back to the host after execution
        """

        return self._mem_transfers(to_device=False)

    def get_mem_frees(self, free_inputs=False):
        """
        Returns code to free the allocated buffers

        Parameters
        ----------
        free_inputs : bool
            If true, we're freeing the local version of the input arrays

        Returns
        -------
        mem_free_str : str
            The generated code
        """

        if not free_inputs:
            #device memory
            frees = [self.get_check_err_call(self.free_template[self.lang].safe_substitute(
                name='d_' + arr.name)) for arr in self.arrays]

            #host memory
            frees.extend(
                [self.free_template[self.host_lang].safe_substitute(
                    name='h_' + arr) for arr in self.host_inits])
        else:
            frees = [self.free_template[self.host_lang].safe_substitute(
                    name='h_' + arr) for arr in self.in_arrays]

        return '\n'.join([x + utils.line_end[self.lang] for x in sorted(set(frees))])