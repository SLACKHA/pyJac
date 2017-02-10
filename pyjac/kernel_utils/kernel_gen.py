"""
kernel_gen.py - generators used for kernel creation
"""

import shutil
import textwrap
import os
import re
from string import Template

import loopy as lp
import pyopencl as cl
import numpy as np
from loopy.kernel.data import temp_var_scope as scopes

from . import file_writers as filew
from .memory_manager import memory_manager
from .. import site_conf as site
from .. import utils
from ..loopy import loopy_utils as lp_utils

script_dir = os.path.abspath(os.path.dirname(__file__))
TINV_PREINST_KEY = 'Tinv'
TLOG_PREINST_KEY = 'logT'
PLOG_PREINST_KEY = 'logP'

class wrapping_kernel_generator(object):
    def __init__(self, loopy_opts, name, kernels,
        external_kernels=[],
        input_arrays=[],
        output_arrays=[],
        init_arrays={},
        test_size=None,
        auto_diff=False,
        depends_on=[],
        array_props={}):
        """
        Parameters
        ----------
        loopy_opts : :class:`LoopyOptions`
            The specified user options
        name : str
            The kernel name to use
        kernels : list of :class:`loopy.LoopKernel`
            The kernels / calls to wrap
        external_kernels : list of :class:`loopy.LoopKernel`
            External kernels that must be called, but not implemented in this file
        input_arrays : list of str
            The names of the input arrays of this kernel
        output_arrays : list of str
            The names of the output arrays of this kernel
        init_arrays : dict
            A mapping of name -> initializer value for arrays in
            this kernel that require constant value initalization
        test_size : int
            If specified, the # of conditions to test
        auto_diff : bool
            If true, this will be used for automatic differentiation
        depends_on : list of :class:`wrapping_kernel_generator`
            If supplied, this kernel depends on the supplied depencies
        array_props : dict
            Mapping of various switches to array names:
                doesnt_need_init
                    * Arrays in this list do not need initialization [defined for host arrays only]
        """

        self.loopy_opts = loopy_opts
        self.lang = loopy_opts.lang
        self.mem = memory_manager(self.lang)
        self.name = name
        self.kernels = kernels
        self.external_kernels = external_kernels
        self.test_size = test_size
        self.auto_diff = auto_diff

        #update the memory manager
        self.mem.add_arrays(in_arrays=input_arrays,
            out_arrays=output_arrays, has_init=init_arrays)

        self.set_knl_arg_array_template = Template(self.mem.get_check_err_call('clSetKernelArg(kernel,'
                                        '${arg_index}, ${arg_size}, ${arg_value})'))
        self.set_knl_arg_value_template = Template(self.mem.get_check_err_call('clSetKernelArg(kernel,'
                                        '${arg_index}, ${arg_size}, ${arg_value})'))

        self.filename = ''
        self.bin_name = ''
        self.header_name = ''
        self.file_prefix = ''

        self.depends_on = depends_on[:]
        self.array_props = array_props.copy()
        self.all_arrays = []

    def add_depencencies(self, k_gens):
        """
        Adds the supplied :class:`wrapping_kernel_generator`s to this
        one's dependency list.  Functionally this means that this kernel generator
        will know how to compile and execute functions from the dependencies

        Parameters
        ----------
        k_gens : list of :class:`wrapping_kernel_generator`
            The dependencies to add to this kernel
        """

        self.depends_on.extend(k_gens)

    def _make_kernels(self):
        """
        Turns the supplied kernel infos into loopy kernels,
        and vectorizes them!

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        #TODO: need to update loopy to allow pointer args
        #to functions, in the meantime use a OpenCL Template

        #now create the kernels!
        target = lp_utils.get_target(self.lang, self.loopy_opts.device)
        for i, info in enumerate(self.kernels):
            if info in self.external_kernels:
                continue
            #create kernel from k_gen.knl_info
            self.kernels[i] = self.make_kernel(info, target, self.test_size)
            #apply vectorization if possible
            if info.can_vectorize:
                self.kernels[i] = apply_vectorization(self.loopy_opts,
                    info.var_name, self.kernels[i])
            #apply any specializations if supplied
            if info.vectorization_specializer:
                self.kernels[i] = info.vectorization_specializer(self.kernels[i])
            #and add a mangler
            #func_manglers.append(create_function_mangler(kernels[i]))
            #set the editor
            self.kernels[i] = lp_utils.set_editor(self.kernels[i])

        #and finally register functions
        #for func in func_manglers:
        #    knl = lp.register_function_manglers(knl, [func])

        #need to call make_kernels on dependencies
        for x in self.depends_on:
            x._make_kernels()

    def __copy_deps(self, scan_path, out_path, change_extension=True):
        deps = [x for x in os.listdir(scan_path) if os.path.isfile(
            os.path.join(scan_path, x)) and not x.endswith('.in')]
        for dep in deps:
            dep_dest = dep
            if change_extension and not dep.endswith(utils.file_ext[self.lang]):
                dep_dest = dep[:dep.rfind('.')] + utils.file_ext[self.lang]
            shutil.copyfile(os.path.join(scan_path, dep),
                os.path.join(out_path, dep_dest))

    def generate(self, path, data_order=None, data_filename='data.bin'):
        """
        Generates wrapping kernel, compiling program (if necessary) and
        calling / executing program for this kernel

        Parameters
        ----------
        path : str
            The output path
        data_order : {'C', 'F'}
            If specified, the ordering of the binary input data
            which may differ from the loopy order
        data_filename : Optional[str]
            If specified, the path to the data file for reading / execution
            via the command line

        Returns
        -------
        None
        """
        utils.create_dir(path)
        self._make_kernels()
        self._generate_wrapping_kernel(path)
        self._generate_compiling_program(path)
        self._generate_calling_program(path, data_filename)
        self._generate_calling_header(path)
        self._generate_common(path)

        #finally, copy any dependencies to the path
        lang_dir = os.path.join(script_dir, self.lang)
        self.__copy_deps(lang_dir, path)

    def _generate_common(self, path, data_order=None):
        if data_order is None:
            data_order = self.loopy_opts.order

        common_dir = os.path.join(script_dir, 'common')
        #get the initial condition reader
        with open(os.path.join(common_dir,
                    'read_initial_conditions.c.in'), 'r') as file:
            file_src = Template(file.read())

        with filew.get_file(os.path.join(path, 'read_initial_conditions'
                + utils.file_ext[self.lang]), self.lang,
                    use_filter=False) as file:
            file.add_lines(file_src.safe_substitute(
                mechanism='mechanism' + utils.header_ext[self.lang],
                data_order=data_order))

        #and any other deps
        self.__copy_deps(common_dir, path)

    def _get_pass(self, argv, include_type=True, postfix=''):
            return '{type}h_{name}'.format(
                type=utils.type_map[argv.dtype] + '* ' if include_type else '',
                name=argv.name  + postfix)

    def _generate_calling_header(self, path):
        assert self.filename or self.bin_name, 'Cannot generate calling header before wrapping kernel is generated...'
        with open(os.path.join(script_dir, self.lang,
                    'kernel.h.in'), 'r') as file:
            file_src = Template(file.read())

        self.header_name = os.path.join(path,
                            self.file_prefix + self.name + utils.header_ext[self.mem.host_lang])
        with filew.get_file(os.path.join(path, self.header_name), self.lang,
            use_filter=False) as file:
            file.add_lines(file_src.safe_substitute(
                input_args=', '.join([self._get_pass(next(x for x in self.mem.arrays if x.name == a))
            for a in self.mem.in_arrays]),
                knl_name=self.name))

    def _generate_calling_program(self, path, data_filename):
        """
        Needed for all languages, this generates a simple C file that
        reads in data, sets up the kernel call, executes, etc.

        Parameters
        ----------
        path : str
            The output path to write files to
        data_filename : str
            The path to the data file for command line input

        Returns
        -------
        None
        """

        assert self.filename or self.bin_name, 'Cannot generate calling program before wrapping kernel is generated...'

        #find definitions
        mem_declares = self.mem.get_defns()

        #and input args

        #these are the args in the kernel defn
        knl_args = ', '.join([self._get_pass(next(x for x in self.mem.arrays if x.name == a))
            for a in self.mem.in_arrays])
        #these are the args passed to the kernel (exclude type)
        input_args = ', '.join([self._get_pass(next(x for x in self.mem.arrays if x.name == a), False)
            for a in self.mem.in_arrays])
        #these are passed from the main method (exclude type, add _local postfix)
        local_input_args = ', '.join([self._get_pass(next(x for x in self.mem.arrays if x.name == a), False,
            '_local') for a in self.mem.in_arrays])
        #create doc strings
        knl_args_doc = []
        knl_args_doc_template = Template(
"""
${name} : ${type}
    ${desc}
""")
        for x in self.mem.in_arrays:
            if x == 'T_arr':
                knl_args_doc.append(knl_args_doc_template.safe_substitute(
                    name=x, type='double*', desc='The array of temperatures'))
            elif x == 'P_arr':
                knl_args_doc.append(knl_args_doc_template.safe_substitute(
                    name=x, type='double*', desc='The array of pressures'))
            elif x == 'conc':
                knl_args_doc.append(knl_args_doc_template.safe_substitute(
                    name=x, type='double*', desc='The array of concentrations in {}-order').format(
                    self.loopy_opts.order))
            elif x == 'wdot':
                knl_args_doc.append(knl_args_doc_template.safe_substitute(
                    name=x, type='double*', desc='The array of species rates, in {}-order').format(
                    self.loopy_opts.order))
            else:
                raise Exception('Argument documentation not found for arg {}'.format(x))

        knl_args_doc = '\n'.join(knl_args_doc)
        #these are args passed in (from main, or python)
        #that require initialization, and hence must be passed to mem_init
        input_initialized_args = ', ' + ', '.join([
            self._get_pass(next(x for x in self.mem.arrays if x.name == a), False)
            for a in self.mem.in_arrays if a in self.mem.has_init])
        #and the type included form thereof (for defn's)
        input_initialized_args_defn = ', ' + ', '.join([
            self._get_pass(next(x for x in self.mem.arrays if x.name == a))
            for a in self.mem.in_arrays if a in self.mem.has_init])
        #and finally the local versions
        input_initialized_args_local = ', ' + ', '.join([
            self._get_pass(next(x for x in self.mem.arrays if x.name == a), False,
                '_local')
            for a in self.mem.in_arrays if a in self.mem.has_init])
        #memory transfers in
        mem_in = self.mem.get_mem_transfers_in()
        #memory transfers out
        mem_out = self.mem.get_mem_transfers_out()
        #vec width
        vec_width = self.vec_width
        if not vec_width:
            #set to default
            vec_width = 1
        #platform
        platform_str = self.loopy_opts.platform.get_info(cl.platform_info.VENDOR)
        #build options
        build_options = self.build_options
        #memory allocations
        mem_allocs = self.mem.get_mem_allocs()
        #input allocs
        input_allocs = self.mem.get_mem_allocs(True)
        #read args are those that aren't initalized elsewhere
        read_args = ', '.join(['h_' + x + '_local' for x in self.mem.in_arrays
            if x not in self.mem.has_init])
        #kernel arg setting
        kernel_arg_sets = self.get_kernel_arg_setting()
        #memory frees
        mem_frees = self.mem.get_mem_frees()
        #input frees
        input_frees = self.mem.get_mem_frees(True)
        #kernel list
        kernel_paths = [self.bin_name] + [x.bin_name for x in self.depends_on]
        kernel_paths = ', '.join('"{}"'.format(x) for x in kernel_paths if x.strip())

        #get template
        with open(os.path.join(script_dir, self.lang,
                    'kernel.c.in'), 'r') as file:
            file_src = Template(file.read())

        with filew.get_file(os.path.join(path, self.name + '_main' + utils.file_ext[self.lang]),
                            self.lang, use_filter=False) as file:
                file.add_lines(file_src.safe_substitute(
                    mem_declares=mem_declares,
                    platform=platform_str,
                    build_options=self.build_options,
                    knl_args=knl_args,
                    knl_args_doc=knl_args_doc,
                    knl_name=self.name,
                    input_args=input_args,
                    local_input_args=local_input_args,
                    input_initialized_args=input_initialized_args,
                    input_initialized_args_defn=input_initialized_args_defn,
                    input_initialized_args_local=input_initialized_args_local,
                    mem_transfers_in=mem_in,
                    mem_transfers_out=mem_out,
                    vec_width=vec_width,
                    kernel_paths=kernel_paths,
                    mem_allocs=mem_allocs,
                    kernel_arg_set=kernel_arg_sets,
                    mem_frees=mem_frees,
                    input_frees=input_frees,
                    read_args=read_args,
                    order=self.loopy_opts.order,
                    device_type=str(self.loopy_opts.device_type),
                    num_source=1, #only 1 program / binary is built
                    data_filename=data_filename,
                    input_allocs=input_allocs
                    ))


    def get_kernel_arg_setting(self):
        """
        Needed for OpenCL, this generates the code that sets the kernel args

        Parameters
        ----------
        None

        Returns
        -------
        knl_arg_set_str : str
            The code that sets opencl kernel args
        """

        kernel_arg_sets = []
        for i, arg in enumerate(self.all_arrays):
            if not isinstance(arg, lp.ValueArg):
                kernel_arg_sets.append(
                    self.set_knl_arg_array_template.safe_substitute(
                        arg_index=i,
                        arg_size='sizeof({})'.format('d_' + arg.name),
                        arg_value='&d_' + arg.name)
                        )
            else:
                kernel_arg_sets.append(
                    self.set_knl_arg_value_template.safe_substitute(
                        arg_index=i,
                        arg_size='sizeof({})'.format(arg.name),
                        arg_value='&' + arg.name))

        return '\n'.join([x + utils.line_end[self.lang] for x in kernel_arg_sets])

    def _generate_compiling_program(self, path):
        """
        Needed for OpenCL, this generates a simple C file that
        compiles and stores the binary OpenCL kernel generated w/ the wrapper

        Parameters
        ----------
        path : str
            The output path to write files to

        Returns
        -------
        None
        """

        assert self.filename, 'Cannot generate compiler before wrapping kernel is generated...'
        if self.depends_on:
            assert [x.filename for x in self.depends_on], ('Cannot generate compiler before wrapping kernel '
                'for dependencies are generated...')

        self.build_options = ''
        if self.lang == 'opencl':
            with open(os.path.join(script_dir, self.lang,
                    'opencl_kernel_compiler.c.in'),
                 'r') as file:
                file_str = file.read()
                file_src = Template(file_str)

            #get the platform from the options
            platform_str = self.loopy_opts.platform.get_info(cl.platform_info.VENDOR)

            #for the build options, we turn to the siteconf
            self.build_options = ['-I' + x for x in site.CL_INC_DIR + [path]]
            self.build_options.extend(site.CL_FLAGS)
            self.build_options.append('-cl-std=CL{}'.format(site.CL_VERSION))
            self.build_options = ' '.join(self.build_options)

            file_list = [self.filename] + [x.filename for x in self.depends_on]
            file_list = ', '.join('"{}"'.format(x) for x in file_list)

            self.bin_name = self.filename[:self.filename.index(
                                utils.file_ext[self.lang])] + '.bin'

            with filew.get_file(os.path.join(path, self.name + '_compiler'
                                 + utils.file_ext[self.lang]),
                            self.lang, use_filter=False) as file:
                file.add_lines(file_src.safe_substitute(
                    filenames=file_list,
                    outname=self.bin_name,
                    platform=platform_str,
                    build_options=self.build_options,
                    num_source=1+len(self.depends_on) #compiler expects all source strings
                    ))


    def _generate_wrapping_kernel(self, path):
        """
        Generates the wrapping kernel

        Parameters
        ----------
        path : str
            The output path to write files to

        Returns
        -------
        None
        """

        assert all(isinstance(x, lp.LoopKernel) or next((y for y in self.external_kernels if x.name == y.name), None)
            for x in self.kernels), 'Cannot generate wrapper before calling _make_kernels'

        if self.depends_on:
            #generate wrappers for dependencies
            for x in self.depends_on:
                x._generate_wrapping_kernel(path)

        self.file_prefix = ''
        if self.auto_diff:
            self.file_prefix = 'ad_'

        #first, load the wrapper as a template
        with open(os.path.join(script_dir, self.lang,
                    'wrapping_kernel{}.in'.format(utils.file_ext[self.lang])),
                 'r') as file:
            file_str = file.read()
            file_src = Template(file_str)

        #Find the list of all arguements needed for this kernel
        #this may change in the future

        kernel_data = []
        #need to find mapping of externel kernels to depends
        for x in self.external_kernels:
            knl = next((y for dep in self.depends_on for y in dep.kernels if y.name == x.name), None)
            assert knl, 'Cannot find external kernel {} in any dependencies'.format(x.name)
            my_knl_ind = next((i for i, k in enumerate(self.kernels) if x.name == k.name), None)
            #now replace
            self.kernels[my_knl_ind] = knl

        #now scan through all our (and externel) kernels
        #and compile the args
        defines = [arg for knl in self.kernels for arg in knl.args if
                        not isinstance(arg, lp.TemporaryVariable)]
        nameset = sorted(set(d.name for d in defines))
        args = []
        for name in nameset:
            #check for dupes
            same_name = [x for x in defines if x.name == name]
            assert all(same_name[0] == y for y in same_name[1:])
            same_name = same_name[0]
            same_name.read_only = False
            kernel_data.append(same_name)

        self.all_arrays = kernel_data[:]
        self.mem.add_arrays(kernel_data)

        #generate the kernel definition
        self.vec_width = self.loopy_opts.depth
        if self.vec_width is None:
            self.vec_width = self.loopy_opts.width
        if self.vec_width is None:
            self.vec_width = 0
        #create a dummy kernel to get the defn
        knl = lp.make_kernel('{{[i, j]: 0 <= i,j < {}}}'.format(self.vec_width),
            '<>temp = i',
            kernel_data,
            name=self.name,
            target=lp_utils.get_target(self.lang, self.loopy_opts.device)
            )
        if self.vec_width:
            knl = lp.tag_inames(knl, [('i', 'l.0')])
        defn_str = lp_utils.get_header(knl)

        #next create the call instructions
        def __gen_call(knl, idx, condition=None):
            call = Template('${name}(${args})${end}').safe_substitute(
                    name=knl.name,
                    args=','.join([arg.name for arg in knl.args
                            if not isinstance(arg, lp.TemporaryVariable)]),
                    end=utils.line_end[self.lang]
                    #dep='id=call_{}{}'.format(idx, ', dep=call_{}'.format(idx - 1) if idx > 0 else '')
                )
            if condition:
                call = Template(
    """
    #ifdef ${cond}
        ${call}
    #endif
    """            ).safe_substitute(cond=condition, call=call)
            return call

        instructions = '\n'.join(__gen_call(knl, i)
            for i, knl in enumerate(self.kernels))

        #and finally, generate the additional kernels [excluding additional knls]
        additional_kernels = '\n'.join([lp_utils.get_code(k) for k in self.kernels
            if not any(y.name == k.name for y in self.external_kernels)])

        self.filename = os.path.join(path,
                            self.file_prefix + self.name + utils.file_ext[self.lang])
        #create the file
        with filew.get_file(self.filename, self.lang, include_own_header=True) as file:
            instructions = _find_indent(file_str, 'body', instructions)
            lines = file_src.safe_substitute(
                        defines='',
                        func_define=defn_str,
                        body=instructions,
                        additional_kernels=additional_kernels).split('\n')

            if self.auto_diff:
                lines = [x.replace('double', 'adouble') for x in lines]
            file.add_lines(lines)
            if self.depends_on:
                file.add_headers([x.name for x in self.depends_on])

        #and the header file
        headers = [lp_utils.get_header(knl) + utils.line_end[self.lang]
                        for knl in self.kernels] + [defn_str + utils.line_end[self.lang]]
        with filew.get_header_file(os.path.join(path, self.file_prefix + self.name
                                 + utils.header_ext[self.lang]), self.lang) as file:

            lines = '\n'.join(headers).split('\n')
            if self.auto_diff:
                file.add_headers('adept.h')
                file.add_lines('using adept::adouble;\n')
                lines = [x.replace('double', 'adouble') for x in lines]
            file.add_lines(lines)

    def make_kernel(self, info, target, test_size):
        """
        Convience method to create loopy kernels from kernel_info

        Parameters
        ----------
        info : :class:`knl_info`
            The rate contstant info to generate the kernel from
        target : :class:`loopy.TargetBase`
            The target to generate code for
        test_size : int/str
            The integer (or symbolic) problem size

        Returns
        -------
        knl : :class:`loopy.LoopKernel`
            The generated loopy kernel
        """

        #various precomputes
        pre_inst = {TINV_PREINST_KEY : '<> T_inv = 1 / T_arr[j]',
                    TLOG_PREINST_KEY : '<> logT = log(T_arr[j])',
                    PLOG_PREINST_KEY : '<> logP = log(P_arr[j])'}

        #and the skeleton kernel
        skeleton = """
        for j
            ${pre}
            for ${var_name}
                ${main}
            end
            ${post}
        end
        """

        #convert instructions into a list for convienence
        instructions = info.instructions
        if isinstance(instructions, str):
            instructions = textwrap.dedent(info.instructions)
            instructions = [x for x in instructions.split('\n') if x.strip()]

        #load inames
        inames = [info.var_name, 'j']

        #add map instructions
        instructions = info.maps + instructions

        #look for extra inames, ranges
        iname_range = []

        assumptions = info.assumptions[:]

        #find the start index for 'i'
        if isinstance(info.indicies, tuple):
            i_start = info.indicies[0]
            i_end = info.indicies[1]
        else:
            i_start = 0
            i_end = info.indicies.size

        #add to ranges
        iname_range.append('{}<={}<{}'.format(i_start, info.var_name, i_end))
        iname_range.append('{}<=j<{}'.format(0, test_size))

        if isinstance(test_size, str):
            assumptions.append('{0} > 0'.format(test_size))
            #get vector width
            vec_width = None
            if self.loopy_opts.depth or self.loopy_opts.width:
                vec_width = self.loopy_opts.depth if self.loopy_opts.depth else self.loopy_opts.width
            if vec_width:
                assumptions.append('{0} mod {1} = 0'.format(
                    test_size, vec_width))

        for iname, irange in info.extra_inames:
            inames.append(iname)
            iname_range.append(irange)

        #construct the kernel args
        pre_instructions = [pre_inst[k] if k in pre_inst else k
                                for k in info.pre_instructions]

        post_instructions = info.post_instructions[:]

        def subs_preprocess(key, value):
            #find the instance of ${key} in kernel_str
            result = _find_indent(skeleton, key, value)
            return Template(result).safe_substitute(var_name=info.var_name)

        kernel_str = Template(skeleton).safe_substitute(
            var_name=info.var_name,
            pre=subs_preprocess('${pre}', '\n'.join(pre_instructions)),
            post=subs_preprocess('${post}', '\n'.join(post_instructions)),
            main=subs_preprocess('${main}', '\n'.join(instructions)))

        #finally do extra subs
        if info.extra_subs:
            kernel_str = Template(kernel_str).safe_substitute(
                **info.extra_subs)

        iname_arr = []
        #generate iname strings
        for iname, irange in zip(*(inames,iname_range)):
            iname_arr.append(Template(
                '{[${iname}]:${irange}}').safe_substitute(
                iname=iname,
                irange=irange
                ))

        #make the kernel
        knl = lp.make_kernel(iname_arr,
            kernel_str,
            kernel_data=info.kernel_data,
            name=info.name,
            target=target,
            assumptions=' and '.join(assumptions)
        )
        #fix parameters
        if info.parameters:
            knl = lp.fix_parameters(knl, **info.parameters)
        #prioritize and return
        knl = lp.prioritize_loops(knl, inames)
        return knl

def handle_indicies(indicies, reac_ind, out_map, kernel_data,
                        outmap_name='out_map', alternate_indicies=None,
                        force_zero=False, force_map=False, scope=scopes.PRIVATE):
    """Consolidates the commonly used indicies mapping steps

    Parameters
    ----------
    indicies: :class:`numpy.ndarray`
        The list of indicies
    reac_ind : str
        The reaction index variable (used in mapping)
    out_map : dict
        The dictionary to store the mapping result in (if any)
    kernel_data : list of :class:`loopy.KernelArgument`
        The data to pass to the kernel (may be added to)
    outmap_name : str, optional
        The name to use in mapping
    alternate_indicies : :class:`numpy.ndarray`
        An alternate list of indicies that can be substituted in to the mapping
    force_zero : bool
        If true, any indicies that don't start with zero require a map (e.g. for
            smaller arrays)
    force_map : bool
        If true, forces use of a map
    scope : :class:`loopy.temp_var_scope`
        The scope of the temporary variable definition, if necessary
    Returns
    -------
    indicies : :class:`numpy.ndarray` OR tuple of int
        The transformed indicies
    """

    check = indicies if alternate_indicies is None else alternate_indicies
    if check[0] + check.size - 1 == check[-1] and \
            (not force_zero or check[0] == 0) and \
            not force_map:
        #if the indicies are contiguous, we can get away with an
        check = (check[0], check[0] + check.size)
    else:
        #need an output map
        out_map[reac_ind] = outmap_name
        #add to kernel data
        outmap_lp = lp.TemporaryVariable(outmap_name,
            shape=lp.auto,
            initializer=check.astype(dtype=np.int32),
            read_only=True, scope=scope)
        kernel_data.append(outmap_lp)

    return check

def apply_vectorization(loopy_opts, inner_ind, knl):
    """
    Applies wide / deep vectorization to a generic rateconst kernel

    Parameters
    ----------
    loopy_opts : :class:`loopy_options` object
        A object containing all the loopy options to execute
    inner_ind : str
        The inner loop index variable
    knl : :class:`loopy.LoopKernel`
        The kernel to transform

    Returns
    -------
    knl : :class:`loopy.LoopKernel`
        The transformed kernel
    """


    #before doing anything, find vec width
    #and split variable
    vec_width = None
    to_split = None
    i_tag = inner_ind
    j_tag = 'j'
    if loopy_opts.depth:
        to_split = inner_ind
        vec_width = loopy_opts.depth
        i_tag += '_outer'
    elif loopy_opts.width:
        to_split = 'j'
        vec_width = loopy_opts.width
        j_tag += '_outer'

    #fix for variable too small for vectorization
    clean = knl.copy()
    def __ggs(insn_ids):
        import pdb;pdb.set_trace()
        grid_size, lsize = clean.get_grid_sizes_for_insn_ids(
            insn_ids)
        lsize = local_size if vec_width is None else \
                    vec_width
        return grid_size, vec_width

    #if we're splitting
    #apply specified optimizations
    if to_split:
        #and assign the l0 axis to the correct variable
        knl = lp.split_iname(knl, to_split, vec_width, inner_tag='l.0')
        #and tag the 'j' variable as global
        knl = lp.tag_inames(knl, [(j_tag, 'g.0')])
        #finally apply the fix above
        knl = knl.copy(get_grid_sizes_for_insn_ids=__ggs)

    #now do unr / ilp
    if loopy_opts.unr is not None:
        knl = lp.split_iname(knl, i_tag, loopy_opts.unr, inner_tag='unr')
    elif loopy_opts.ilp:
        knl = lp.tag_inames(knl, [(i_tag, 'ilp')])

    return knl

class knl_info(object):
    """
    A composite class that contains the various parameters, etc.
    needed to create a simple kernel

    name : str
        The kernel name
    instructions : str or list of str
        The kernel instructions
    pre_instructions : list of str
        The instructions to execute before the inner loop
    post_instructions : list of str
        The instructions to execute after end of inner loop but before end
        of outer loop
    var_name : str
        The inner loop variable
    kernel_data : list of :class:`loopy.ArrayBase`
        The arguements / temporary variables for this kernel
    maps : list of str
        A list of variable mapping instructions
        see :method:`loopy_utils.generate_mapping_instruction`
    extra_inames : list of tuple
        A list of (iname, domain) tuples the form the extra loops in this kernel
    indicies : :class:`numpy.ndarray` or tuple
        The list of indicies to run this kernel on,
        see :method:`handle_indicies`
    assumptions : list of str
        Assumptions to pass to the loopy kernel
    parameters : dict
        Dictionary of parameter values to fix in the loopy kernel
    extra subs : dict
        Dictionary of extra string substitutions to make in kernel generation
    can_vectorize : bool
        If true, can vectorize this kernel
    vectorization_specializer : function
        If specified, use this specialization function to fix problems that would arise
        in vectorization
    """
    def __init__(self, name, instructions, pre_instructions=[],
            post_instructions=[],
            var_name='i', kernel_data=None,
            maps=[], extra_inames=[], indicies=[],
            assumptions=[], parameters={},
            extra_subs={},
            can_vectorize=True,
            vectorization_specializer=None):
        self.name = name
        self.instructions = instructions
        self.pre_instructions = pre_instructions[:]
        self.post_instructions = post_instructions[:]
        self.var_name = var_name
        self.kernel_data = kernel_data[:]
        self.maps = maps[:]
        self.extra_inames = extra_inames[:]
        self.indicies = indicies[:]
        self.assumptions = assumptions[:]
        self.parameters = parameters.copy()
        self.extra_subs = extra_subs
        self.can_vectorize = can_vectorize
        self.vectorization_specializer = vectorization_specializer

class MangleGen(object):
    def __init__(self, name, arg_dtypes, result_dtypes):
        self.name = name
        self.arg_dtypes = arg_dtypes
        self.result_dtypes = result_dtypes

    def __call__(self, kernel, name, arg_dtypes):
        if name != self.name:
            return None
        assert arg_dtypes == self.arg_dtypes
        from loopy.kernel.data import CallMangleInfo
        return CallMangleInfo(
            target_name=self.name,
            result_dtypes=self.result_dtypes,
            arg_dtypes=self.arg_dtypes)


def create_function_mangler(kernel, return_dtypes=()):
    """
    Returns a function mangler to interface loopy kernels with function calls
    to other kernels (e.g. falloff rates from the rate kernel, etc.)

    Parameters
    ----------
    kernel : :class:`loopy.LoopKernel`
        The kernel to create an interface for
    return_dtypes : list :class:`numpy.dtype` returned from the kernel, optional
        Most likely an empty list
    Returns
    -------
    func : :method:`MangleGen`.__call__
        A function that will return a :class:`loopy.kernel.data.CallMangleInfo` to
        interface with the calling :class:`loopy.LoopKernel`
    """

    dtypes = []
    for arg in kernel.args:
        if not isinstance(arg, lp.TemporaryVariable):
            dtypes.append(arg.dtype)
    mg = MangleGen(kernel.name, tuple(dtypes), return_dtypes)
    return mg.__call__

def _find_indent(template_str, key, value):
    """
    Finds and returns a formatted value containing the appropriate
    whitespace to put 'value' in place of 'key' for template_str

    Parameters
    ----------
    template_str : str
        The string to sub into
    key : str
        The key in the template string
    value : str
        The string to format

    Returns
    -------
    formatted_value : str
        The formatted string
    """

    #find the instance of ${key} in kernel_str
    whitespace = None
    for i, line in enumerate(template_str.split('\n')):
        if key in line:
            #get whitespace
            whitespace = re.match(r'\s*', line).group()
            break
    result = [line if i == 0 else whitespace + line for i, line in
                enumerate(textwrap.dedent(value).splitlines())]
    return '\n'.join(result)