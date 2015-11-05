"""Writes mechanism header and output testing files"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import sys
import itertools

# Local imports
import chem_utilities as chem
import utils

def write_mechanism_initializers(path, lang, specs, reacs, initial_conditions='',
                                 old_spec_order=None, old_rxn_order=None,
                                 cache_optimized=False, last_spec=None):
    if lang in ['matlab', 'fortran']:
        raise NotImplementedError

    # some information variables
    have_rev_rxns = any(reac.rev for reac in reacs)
    have_pdep_rxns = any(reac.thd_body or reac.pdep for reac in reacs)

    # the mechanism header defines a number of useful preprocessor defines, as
    # well as defining method stubs for setting initial conditions
    with open(os.path.join(path, 'mechanism{}'.format(utils.header_ext[lang])),
              'w') as file:

        file.write('#ifndef MECHANISM_{}\n'.format(utils.header_ext[lang][1:]) +
                   '#define MECHANISM_{}\n\n'.format(utils.header_ext[lang][1:])
                   )

        # make cache optimized easy to recognize
        if cache_optimized:
            file.write('//Cache Optimized\n')
        file.write('//last_spec {}\n'.format(last_spec))

        # convience: write species indexes
        file.write('/* Species Indexes\n')
        file.write('\n'.join('{}  {}'.format(i, spec.name)
                             for i, spec in enumerate(specs)))
        file.write('*/\n\n')

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
        file.write('#define PRES_MOD_RATES {}\n\n'.format(
            len([reac for reac in reacs if reac.pdep or reac.thd_body]))
            )

        file.write(
            '//Must be implemented by user on a per mechanism basis in mechanism{}\n'.format(utils.file_ext[lang]) +
            '{} set_same_initial_conditions(int NUM,{} double**, double**);\n\n'.format(
                'int' if lang == 'cuda' else 'void', ' double**, double**, ' if lang == 'cuda' else '')
            )
        file.write('#if defined (RATES_TEST) || defined (PROFILER)\n'
                   '    void write_jacobian_and_rates_output(int NUM);\n'
                   '#endif\n'
                   '//apply masking of ICs for cache optimized mechanisms\n'
                   'void apply_mask(double*);\n'
                   'void apply_reverse_mask(double*);\n'
                   )

        file.write('#endif\n\n')

    reversed_specs = []
    for i in range(len(specs)):
        reversed_specs.append(old_spec_order.index(i))

    # now the mechanism file
    with open(os.path.join(path, 'mechanism' + utils.file_ext[lang]), 'w') as file:
        file.write('#include "mass_mole{}"\n'.format(
          utils.header_ext[lang]) +
        '#include <stdio.h>\n')

        if lang == 'cuda':
            file.write('#include <cuda.h>\n'
                       '#include <cuda_runtime.h>\n'
                       '#include <helper_cuda.h>\n'
                       '#include "launch_bounds.cuh"\n'
                       '#include "gpu_macros.cuh"\n'
                       '#include "gpu_memory.cuh"\n')
        if lang == 'c':
            file.write('#include <string.h>\n') # for memset

        file.write('    //apply masking of ICs for cache optimized mechanisms\n')
        file.write('    void apply_mask(double* y_specs) {\n')
        if cache_optimized or last_spec != len(specs) - 1:
            file.write('        double temp [NSP];\n'
                       '        memcpy(temp, y_specs, NSP * sizeof(double));\n')
            for i, spec in enumerate(old_spec_order):
                file.write('        y_specs[{0}] = temp[{1}];\n'.format(spec, i))
        file.write('    }\n')

        file.write('    //reverse masking of ICs for cache optimized mechanisms\n')
        file.write('    void apply_reverse_mask(double* y_specs) {\n')
        if cache_optimized or last_spec != len(specs) - 1:
            file.write('        double temp [NSP];\n'
                       '        memcpy(temp, y_specs, NSP * sizeof(double));\n')
            for i, spec in enumerate(reversed_specs):
                file.write('        y_specs[{0}] = temp[{1}];\n'.format(spec, i))
        file.write('    }\n')

        needed_arr = ['y', 'pres']
        needed_arr_conv = ['y', 'rho']
        if lang == 'cuda':
            needed_arr = [['double** ' + a + '_host', 'double** d_' + a]
                          for a in needed_arr]
            needed_arr = [a for a in itertools.chain(*needed_arr)]
            needed_arr_conv = [
                ['double** ' + a + '_host', 'double** d_' + a] for a in needed_arr_conv]
            needed_arr_conv = [a for a in itertools.chain(*needed_arr_conv)]
        else:
            needed_arr = ['double** ' + a + '_host' for a in needed_arr]
            needed_arr_conv = ['double** ' + a + '_host' for a in needed_arr_conv]
        file.write('#ifdef CONP\n'
                   '{} set_same_initial_conditions(int NUM, {}) \n'.format('int' if lang == 'cuda' else 'void',
                                                                           ', '.join(needed_arr)) +
                   '#elif CONV\n'
                   '{} set_same_initial_conditions(int NUM, {}) \n'.format('int' if lang == 'cuda' else 'void',
                                                                           ', '.join(needed_arr_conv)) +
                   '#endif\n'
                   )
        file.write('{\n')
        if lang == 'cuda':
            file.write('    int grid_size = round(((double)NUM) / ((double)TARGET_BLOCK_SIZE));\n')
            file.write('    if (grid_size == 0)\n'
                       '        grid_size = 1;\n')
            # do cuda mem init and copying
            file.write(
                '#ifdef CONP\n'
                '    int padded = initialize_gpu_memory(NUM, TARGET_BLOCK_SIZE, grid_size, d_y, d_pres);\n'
                '#elif CONV\n'
                '    int padded = initialize_gpu_memory(NUM, TARGET_BLOCK_SIZE, grid_size, d_y, d_rho);\n'
                '#endif\n'
            )
        else:
            file.write('    int padded = NUM;\n')
        file.write('    double Xi [NSP] = {0.0};\n'
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
                    print("Initial conditions improperly specified, expecting form T,P,Species1=...,Species2=...")
                    sys.exit(1)
            except:
                print("Error in initial conditions list, not comma separated")
                sys.exit(1)
            try:
                T0 = float(conditions[0])
                P = float(conditions[1])
                mole_list = conditions[2:]
            except:
                print("Could not parse initial T or P as floats...")
                sys.exit(1)
            try:
                mole_list = [x.split("=") for x in mole_list]
            except:
                print("Error in initial mole list, initial moles do not follow SPECIES_NAME=VALUE format")
                sys.exit(1)
            try:
                mole_list = [(split[0], float(split[1])) for split in mole_list]
            except:
                print("Unknown (non-float) value found as initial mole number in list")
                sys.exit(1)
            try:
                mole_list = [(next(i for i, spec in enumerate(specs) if spec.name == split[0]), split[1]) for split in
                             mole_list]
            except:
                print("Unknown species in initial mole list")
                sys.exit(1)
        for x in mole_list:
            file.write('    Xi[{}] = {}'.format(x[0], x[1]) + utils.line_end[lang])
        file.write(
            '    for (int j = 0; j < NSP; ++ j) {\n'
            '        Xsum += Xi[j];\n'
            '    }\n'
            '    if (Xsum == 0.0) {\n'
            '        printf("Use of the set initial conditions function requires user implementation!");\n'
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
        if lang == 'cuda':
            file.write(
                '    cudaMallocHost((void**)y_host, padded * NSP * sizeof(double));\n'
                '#ifdef CONP\n'
                '    cudaMallocHost((void**)pres_host, padded * sizeof(double));\n'
                '#elif defined(CONV)\n'
                '    cudaMallocHost((void**)rho_host, padded * sizeof(double));\n'
                '#endif\n'
            )
        else:
            file.write(
                '    (*y_host) = malloc(padded * NSP * sizeof(double));\n'
                '#ifdef CONP\n'
                '    (*pres_host) = malloc(padded * sizeof(double));\n'
                '#elif defined(CONV)\n'
                '    (*rho_host) = malloc(padded * sizeof(double));\n'
                '#endif\n'
            )
        file.write(
            '    //load temperature and mass fractions for all threads (cells)\n'
            '    for (int i = 0; i < padded; ++i) {\n'
            '        (*y_host)[i] = T0;\n'
            '        //loop through species\n'
            '        for (int j = 1; j < NSP; ++j) {\n'
            '            (*y_host)[i + padded * j] = Yi[j - 1];\n'
            '        }\n'
            '    }\n\n'
            '#ifdef CONV\n'
            '    //calculate density\n'
            '    double rho = getDensity(T0, P, Xi);\n'
            '#endif\n\n'
            '    for (int i = 0; i < padded; ++i) {\n'
            '#ifdef CONV\n'
            '        (*rho_host)[i] = rho;\n'
            '#elif defined(CONP)\n'
            '        (*pres_host)[i] = P;\n'
            '#endif\n'
            '    }\n'
        )
        if lang == 'cuda':  # copy memory over
            file.write('    return padded;\n')

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
            file.write('#ifdef CONP\n'
                       'int initialize_gpu_memory(int NUM, int block_size, int grid_size, double** y_device, '
                       'double** pres_device);\n'
                       'void free_gpu_memory(double* y_device, double* pres_device);\n'
                       '#else\n'
                       'int initialize_gpu_memory(int NUM, int block_size, int grid_size, double** y_device, '
                       'double** rho_device);\n'
                       'void free_gpu_memory(double* y_device, double* rho_device);\n'
                       '#endif\n'
                       '\n'
                       '#endif\n')

        with open(os.path.join(path, 'gpu_memory.cu'), 'w') as file:
            init_template = 'initialize_pointer(&{}, {});'
            free_template = 'cudaErrorCheck(cudaFree({}));'
            file.write('#include "gpu_memory.cuh"\n'
                       '\n')

            file.write('void initialize_pointer(double** ptr, int size) {\n'
                       '    cudaErrorCheck(cudaMalloc((void**)ptr, size * sizeof(double)));\n'
                       '    cudaErrorCheck(cudaMemset(*ptr, 0, size * sizeof(double)));\n'
                       '}\n')
            file.write('#ifdef CONP\n'
                       'int initialize_gpu_memory(int NUM, int block_size, int grid_size, double** y_device, '
                       'double** pres_device)\n'
                       '#else\n'
                       'int initialize_gpu_memory(int NUM, int block_size, int grid_size, double** y_device, '
                       'double** rho_device)\n'
                       '#endif\n'
                       '{\n'
                       '    int padded = grid_size * block_size > NUM ? grid_size * block_size : NUM;\n'
                       '    cudaErrorCheck(cudaMalloc((void**)y_device, padded * NSP * sizeof(double)));\n'
                       '#ifdef CONP\n'
                       '    cudaErrorCheck(cudaMalloc((void**)pres_device, padded * sizeof(double)));\n'
                       '#else\n'
                       '    cudaErrorCheck(cudaMalloc((void**)rho_device, padded * sizeof(double)));\n'
                       '#endif\n'
                       '    return padded;\n'
                       '}\n'
                       )
            file.write('#ifdef CONP\n'
                       'void free_gpu_memory(double* y_device, double* pres_device)\n'
                       '#else\n'
                       'void free_gpu_memory(double* y_device, double* rho_device)\n'
                       '#endif\n'
                       '{\n'
                       '    cudaErrorCheck(cudaFree(y_device));\n'
                       '#ifdef CONP\n'
                       '    cudaErrorCheck(cudaFree(pres_device));\n'
                       '#else\n'
                       '    cudaErrorCheck(cudaFree(rho_device));\n'
                       '#endif\n'
                       '}\n'
                       )

    if lang == 'cuda':
        with open(os.path.join(path, 'gpu_macros.cuh'), 'w') as file:
            file.write('#ifndef GPU_MACROS_CUH\n'
                       '#define GPU_MACROS_CUH\n'
                       '#include <stdio.h>\n'
                       '#include <cuda.h>\n'
                       '#include <cuda_runtime.h>\n'
                       '#include <helper_cuda.h>\n'
                       '\n'
                       )

            file.write('#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }\n'
                       'inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)\n'
                       '{\n'
                       '    if (code != cudaSuccess)\n'
                       '    {\n'
                       '        fprintf(stderr,"GPUassert: %s %s %d\\n", cudaGetErrorString(code), file, line);\n'
                       '        if (abort) exit(code);\n'
                       '    }\n'
                       '}\n'
                       )
            file.write('#endif\n')


def write_header(path, lang):
    """Writes minimal header file used by all other source files.

    :param path:
        Path where files are being written.
    :param lang: {'c', 'cuda', 'fortran', 'matlab'}
        Language type.
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
