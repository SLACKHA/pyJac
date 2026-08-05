"""Writes mechanism header and output testing files
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import sys
import itertools

# Local imports
from .. import utils
from . import chem_utilities as chem

def write_mechanism_initializers(path, lang, specs, reacs,
                                 fwd_spec_mapping, back_spec_mapping,
                                 initial_conditions='', cache_optimized=False,
                                 last_spec=None, auto_diff=False):
    """Writes mechanism-specific files.

    Parameters
    ----------
    path : str
        Path where files are being written.
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Language type.
    specs : list of `SpecInfo`
        List of species in the mechanism.
    reacs : list of `ReacInfo`
        List of reactions in the mechanism.
    fwd_spec_mapping : list of int
        A mapping of the original mechanism to the new species order
    back_spec_mapping : list of int
        A mapping of the new species order to the original mechanism
    initial_conditions : Optional[str]
        A comma separated list of the initial conditions to use in form
        T,P,X (e.g. '800,1,H2=1.0,O2=0.5'). Temperature in K, P in atm.
    cache_optimized : Optional[bool]
        If ``True``, use the greedy optimizer to attempt to improve cache hit rates
    last_spec : Optional[str]
        If specified, the species to assign to the last index.
        Typically should be N2, Ar, He or another inert bath gas
    auto_diff : Optional[bool]
        If ``True``, generate files for Adept auto_differention library.

    Returns
    -------
    None

    """
    if lang in ['matlab', 'fortran']:
        raise NotImplementedError

    if auto_diff:
        with open(os.path.join(path, 'ad_jac.c'), 'w') as file:
            #need to write the auto_diff jacobian
            file.write("""
#include <vector>
#include "adept.h"
#include "header.h"
#include "ad_dydt.h"
void eval_jacob(const double t, const double p, const double* y,
                         double* jac) {
    using adept::adouble; // Import Stack and adouble from adept
    adept::Stack stack; // Where the derivative information is stored
    std::vector<adouble> in(NSP); // Vector of active input variables
    adept::set_values(&in[0], NSP, y); // Initialize adouble inputs
    adouble pres = p;
    stack.new_recording(); // Start recording
    std::vector<adouble> out(NSP); // Create vector of active output variables
    dydt(t, pres, &in[0], &out[0]); // Run algorithm
    stack.independent(&in[0], NSP); // Identify independent variables
    stack.dependent(&out[0], NSP); // Identify dependent variables
    stack.jacobian(jac); // Compute & store Jacobian in jac
}
            """
            )

    # some information variables
    have_rev_rxns = any(reac.rev for reac in reacs)
    have_pdep_rxns = any(reac.thd_body or reac.pdep for reac in reacs)

    gpu_memory = {'y' : 'NSP',
                  'dy' : 'NSP',
                  'conc' : 'NSP',
                  'fwd_rates' : 'FWD_RATES',
                  'rev_rates' : 'REV_RATES',
                  'spec_rates' : 'NSP',
                  'cp' : 'NSP',
                  'h' : 'NSP',
                  'dBdT' : 'NSP',
                  'jac' : 'NSP * NSP',
                  'var' : '1'
                  }
    if any(len(specs) - 1 in reac.reac + reac.prod and
           utils.get_nu(len(specs) - 1, reac) for reac in reacs
           ):
        gpu_memory['J_nplusjplus'] = 'NSP'
    if any(r.pdep or r.thd_body for r in reacs):
        gpu_memory['pres_mod'] = 'PRES_MOD_RATES'
    if any(r.cheb for r in reacs):
        dim = max(rxn.cheb_n_temp for rxn in reacs if rxn.cheb)
        gpu_memory['dot_prod'] = str(dim)

    # the mechanism header defines a number of useful preprocessor defines, as
    # well as defining method stubs for setting initial conditions
    with open(os.path.join(path, 'mechanism{}'.format(utils.header_ext[lang])),
              'w'
              ) as file:

        file.write('#ifndef MECHANISM_{}\n'.format(utils.header_ext[lang][1:]) +
                   '#define MECHANISM_{}\n\n'.format(utils.header_ext[lang][1:])
                   )

        if lang == 'cuda':
            file.write('#ifdef __GNUG__\n'
                       '#include <cuda.h>\n'
                       '#include <cuda_runtime.h>\n'
                       '#include <helper_cuda.h>\n'
                       '#include "launch_bounds.cuh"\n'
                       '#include "gpu_macros.cuh"\n'
                       '#endif\n'
                       )
            file.write('\nstruct mechanism_memory {\n')
            for array in gpu_memory:
                file.write('  double * {};\n'.format(array))
            file.write('};\n\n')
        if lang == 'c':
            file.write('#include <string.h>\n') # for memset

        # make cache optimized easy to recognize
        if cache_optimized:
            file.write('//Cache Optimized\n')
        file.write('//last_spec {}\n'.format(last_spec))

        # convience: write species indexes
        file.write('/* Species Indexes\n')
        file.write('\n'.join('{}  {}'.format(i, spec.name)
                             for i, spec in enumerate(specs)
                             )
                   )
        file.write('\n*/\n\n')

        file.write('//Number of species\n'
                   '#define NSP {}\n'.format(len(specs)) +
                   '//Number of variables. NN = NSP + 1 (temperature)\n' +
                   '#define NN {}\n'.format(len(specs) + 1)
                   )
        file.write('//Number of forward reactions\n' +
                   '#define FWD_RATES {}\n'.format(len(reacs)) +
                   '//Number of reversible reactions\n'+
                   '#define REV_RATES {}\n'.format(
                   len([reac for reac in reacs if reac.rev]))
                   )
        file.write('//Number of reactions with pressure modified rates\n')
        file.write(
            '#define PRES_MOD_RATES {}\n\n'.format(
            len([reac for reac in reacs if reac.pdep or reac.thd_body]))
            )

        file.write(
            '//Must be implemented by user on a per '
            'mechanism basis in mechanism{}\n'.format(utils.file_ext[lang]) +
            'void set_same_initial_conditions(int, double**, double**);\n\n'
            )
        file.write('#if defined (RATES_TEST) || defined (PROFILER)\n'
                   '    void write_jacobian_and_rates_output(int NUM);\n'
                   '#endif\n'
                   '//apply masking of ICs for cache optimized mechanisms\n'
                   'void apply_mask(double*);\n'
                   'void apply_reverse_mask(double*);\n'
                   )

        file.write('#endif\n\n')

    # now the mechanism file
    with open(os.path.join(path, 'mechanism' + utils.file_ext[lang]), 'w') as file:
        file.write(
            '#include "mass_mole{}"\n'.format(utils.header_ext[lang]) +
            '#include <stdio.h>\n'
            '#include "mechanism{}"\n'.format(utils.header_ext[lang])
            )
        if lang == 'cuda':
            file.write('#include "gpu_memory.cuh"\n')

        file.write('    //apply masking of ICs for cache optimized mechanisms\n')
        file.write('    void apply_mask(double* y_specs) {\n')
        if cache_optimized or last_spec != len(specs) - 1:
            file.write('        double temp [NSP];\n'
                       '        memcpy(temp, y_specs, NSP * sizeof(double));\n'
                       )
            for i, spec in enumerate(fwd_spec_mapping):
                file.write('        y_specs[{0}] = temp[{1}];\n'.format(i, spec))
        file.write('    }\n')

        file.write('    //reverse masking of ICs for cache optimized mechanisms\n')
        file.write('    void apply_reverse_mask(double* y_specs) {\n')
        if cache_optimized or last_spec != len(specs) - 1:
            file.write('        double temp [NSP];\n'
                       '        memcpy(temp, y_specs, NSP * sizeof(double));\n'
                       )
            for i, spec in enumerate(back_spec_mapping):
                file.write('        y_specs[{0}] = temp[{1}];\n'.format(i, spec))
        file.write('    }\n')

        needed_arr = ['y', 'var']
        needed_arr = ['double** ' + a + '_host' for a in needed_arr]
        file.write('void set_same_initial_conditions(int NUM, '
                   '{}) \n'.format(', '.join(needed_arr)) +
                   '{\n'
                   '    double Xi [NSP] = {0.0};\n'
                   '    //set initial mole fractions here\n\n'
                   '    //Normalize mole fractions to sum to one\n'
                   '    double Xsum = 0.0;\n'
                   )

        mole_list = []
        T0 = 1600
        P = 1
        if initial_conditions != "":
            try:
                conditions = [x.strip() for x in initial_conditions.split(",")]
                if len(conditions) < 3:
                    print('Initial conditions improperly specified, '
                          'expecting form T,P,Species1=...,Species2=...'
                          )
                    sys.exit(1)
            except:
                print('Error in initial conditions list, not comma separated')
                sys.exit(1)
            try:
                T0 = float(conditions[0])
                P = float(conditions[1])
                mole_list = conditions[2:]
            except:
                print('Could not parse initial T or P as floats...')
                sys.exit(1)
            try:
                mole_list = [x.split('=') for x in mole_list]
            except:
                print('Error in initial mole list, initial moles do not '
                      'follow SPECIES_NAME=VALUE format'
                      )
                sys.exit(1)
            try:
                mole_list = [(split[0], float(split[1])) for split in mole_list]
            except:
                print('Unknown (non-float) value found '
                      'as initial mole number in list'
                      )
                sys.exit(1)
            try:
                mole_list = [(next(i for i, spec in enumerate(specs)
                              if spec.name == split[0]), split[1]
                              ) for split in mole_list
                             ]
            except:
                print('Unknown species in initial mole list')
                sys.exit(1)
        for x in mole_list:
            file.write('    Xi[{}] = {}'.format(x[0], x[1]) +
                       utils.line_end[lang]
                       )
        file.write(
            '    for (int j = 0; j < NSP; ++ j) {\n'
            '        Xsum += Xi[j];\n'
            '    }\n'
            '    if (Xsum == 0.0) {\n'
            '        printf("Use of the set initial conditions function '
            'requires user implementation!\\n");\n'
            '        exit(-1);\n'
            '    }\n'
            '    for (int j = 0; j < NSP; ++ j) {\n'
            '        Xi[j] /= Xsum;\n'
            '    }\n\n'
            '    //convert to mass fractions\n'
            '    double Yi[NSP - 1] = {0.0};\n'
            '    mole2mass(Xi, Yi);\n\n'
            '    //set initial pressure, units [PA]\n' +
            '    double P = {};\n'.format(chem.PA * P) +
            '    // set intial temperature, units [K]\n' +
            '    double T0 = {};\n\n'.format(T0)
            )
        file.write(
            '    (*y_host) = (double*)malloc(NUM * NSP * sizeof(double));\n'
            '    (*var_host) = (double*)malloc(NUM * sizeof(double));\n')
        file.write(
            '    //load temperature and mass fractions for all threads (cells)\n'
            '    for (int i = 0; i < NUM; ++i) {\n'
            '        (*y_host)[i] = T0;\n'
            '        //loop through species\n'
            '        for (int j = 1; j < NSP; ++j) {\n'
            '            (*y_host)[i + NUM * j] = Yi[j - 1];\n'
            '        }\n'
            '    }\n\n'
            '#ifdef CONV\n'
            '    //calculate density\n'
            '    double rho = getDensity(T0, P, Xi);\n'
            '#endif\n\n'
            '    for (int i = 0; i < NUM; ++i) {\n'
            '#ifdef CONV\n'
            '        (*var_host)[i] = rho;\n'
            '#elif defined(CONP)\n'
            '        (*var_host)[i] = P;\n'
            '#endif\n'
            '    }\n'
        )

        file.write('}\n\n')

    if lang == 'cuda':
        mem_template = 'double* {};'
        with open(os.path.join(path, 'gpu_memory.cuh'), 'w') as file:
            file.write('#ifndef GPU_MEMORY_CUH\n'
                       '#define GPU_MEMORY_CUH\n'
                       '\n'
                       '#include "header{}"\n'.format(utils.header_ext[lang]) +
                       '#include "gpu_macros.cuh"\n'
                       '\n'
                       )
            file.write('void initialize_gpu_memory(int, mechanism_memory**,'
                       ' mechanism_memory**);\n'
                       'size_t required_mechanism_size();\n'
                       'void free_gpu_memory(mechanism_memory**, '
                       'mechanism_memory**);\n'
                       '\n'
                       '#endif\n'
                       )

        with open(os.path.join(path, 'gpu_memory.cu'), 'w') as file:
            init_template = 'initialize_pointer(&((*d_mem)->{}), {} * padded)'
            free_template = 'cudaErrorCheck(cudaFree({}))'
            err_check = '  cudaErrorCheck( {} );\n'
            file.write('#include "gpu_memory.cuh"\n'
                       '\n'
                       )

            file.write('size_t required_mechanism_size() {\n'
                       '  //returns the total required size for the mechanism per thread\n'
                       '  size_t mech_size = 0;\n'
                       )
            for array, size in gpu_memory.items():
                file.write('  //{}\n'.format(array) +
                           '  mech_size += {};\n'.format(size)
                           )
            file.write('  //y_device\n'
                       '  mech_size += NSP;\n'
                       '  //pres_device\n'
                       '  mech_size += 1;\n'
                       '  return mech_size * sizeof(double);\n'
                       '}\n'
                       )

            file.write(
                'void initialize_gpu_memory(int padded, mechanism_memory** h_mem,'
                ' mechanism_memory** d_mem)\n'
                '{\n'
                '  //init vectors\n'
                '  // Allocate storage for the device struct\n' +
                err_check.format('cudaMalloc(d_mem, sizeof(mechanism_memory))') +
                '  //allocate the device arrays on the host pointer\n'
                )

            for array, size in gpu_memory.items():
                file.write(
                    err_check.format(
                    'cudaMalloc(&((*h_mem)->{}), {}'.format(array, size) +
                    ' * padded * sizeof(double))')
                    )

            zero_vals = ['spec_rates', 'dy', 'jac']
            for x in zero_vals:
                file.write(
                    utils.line_start + 'cudaErrorCheck( '
                    'cudaMemset((*h_mem)->{}, 0, {}'.format(x, gpu_memory[x]) +
                    ' * padded * sizeof(double)) )' +
                    utils.line_end[lang]
                    )

            file.write(
                utils.line_start + 'cudaErrorCheck( '
                'cudaMemcpy(*d_mem, *h_mem, sizeof(mechanism_memory), '
                'cudaMemcpyHostToDevice) )' +
                utils.line_end[lang]
                )
            file.write(utils.line_start + utils.comment[lang] +
                       'zero out required values\n'
                       )

            file.write('}\n')
            file.write('void free_gpu_memory(mechanism_memory** h_mem, '
                       'mechanism_memory** d_mem)\n'
                       '{\n'
                       )
            for array in gpu_memory:
                file.write(utils.line_start +
                           free_template.format('(*h_mem)->{}'.format(array)) +
                           utils.line_end[lang]
                           )
            file.write(utils.line_start +
                       free_template.format('*d_mem') +
                       utils.line_end[lang]
                       )
            file.write('}\n')

    if lang == 'cuda':
        with open(os.path.join(path, 'gpu_macros.cuh'), 'w') as file:
            file.write(
    '#ifndef GPU_MACROS_CUH\n'
    '#define GPU_MACROS_CUH\n'
    '#include <stdio.h>\n'
    '#include <cuda.h>\n'
    '#include <cuda_runtime.h>\n'
    '#include <helper_cuda.h>\n'
    '\n'
    '#define GRID_DIM (blockDim.x * gridDim.x)\n'
    '#define T_ID (threadIdx.x + blockIdx.x * blockDim.x)\n'
    '#define INDEX(i) (T_ID + (i) * GRID_DIM)\n'
    '\n'
    )

            file.write(
    '#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }\n'
    'inline void gpuAssert(cudaError_t code, const char *file, '
    'int line, bool abort=true)\n'
    '{\n'
    '    if (code != cudaSuccess)\n'
    '    {\n'
    '        fprintf(stderr,"GPUassert: %s %s %d\\n", '
    'cudaGetErrorString(code), file, line);\n'
    '        if (abort) exit(code);\n'
    '    }\n'
    '}\n'
    )
            file.write('#endif\n')


def write_header(path, lang):
    """Writes minimal header file used by all other source files.

    Parameters
    ----------
    path : str
        Path where files are being written.
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Language type.

    Returns
    -------
    None

    """
    if lang in ['matlab', 'fortran']:
        raise NotImplementedError

    with open(os.path.join(path, 'header' + utils.header_ext[lang]), 'w') as file:
        file.write('#ifndef HEAD\n'
                   '#define HEAD\n'
                   '#include <stdlib.h>\n'
                   '#include <math.h>'
                   '\n'
                   '/** Constant pressure or volume. */\n'
                   '#define CONP\n'
                   '//#define CONV\n'
                   '\n'
                   '/** Include mechanism header to get NSP and NN **/\n'
                   '#include "mechanism{}"\n'.format(utils.header_ext[lang]) +
                   '// OpenMP\n'
                   '#ifdef _OPENMP\n'
                   ' #include <omp.h>\n'
                   '#else\n'
                   ' #define omp_get_max_threads() 1\n'
                   ' #define omp_get_num_threads() 1\n'
                   '#endif\n'
                   '#endif\n'
                  )
