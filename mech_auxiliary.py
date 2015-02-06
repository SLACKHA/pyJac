"""Writes mechanism header and output testing files"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import sys
import math

# Local imports
import chem_utilities as chem
import utils
import CUDAParams

def __write_rate_evaluator(file, have_rev_rxns, have_pdep_rxns, T, P, Pretty_P):
    file.write('        fprintf(fp, "{}, {} atm\\n");\n'.format(T, Pretty_P) +
               '        Yi[0] = {};\n'.format(T) +
               '        get_concentrations({}, Yi, conc);\n'.format(P) + 
               '        eval_rxn_rates({}, conc, fwd_rates{}'.format(T, 'rev_rates);\n' if have_rev_rxns else ');\n')
               )
    if have_pdep_rxns:
        file.write('        get_rxn_pres_mod ({}, {}, conc, pres_mod);\n'.format(T, P))
    file.write('        eval_spec_rates (fwd_rates,{}{} dy);\n'.format(' rev_rates,' if have_rev_rxns else '', ' pres_mod,' if have_pdep_rxns else '') +
               '        write_rates(fp,{}{} dy);\n'.format(' rev_rates,' if have_rev_rxns else '', ' pres_mod,' if have_pdep_rxns else '') +
               '        eval_jacob(0, {}, Yi, jacob);\n'.format(P) + 
               '        write_jacob(fp, jacob);\n'
               )

def write_mechanism_initializers(path, lang, specs, reacs):
    if lang in ['matlab', 'fortran']:
        raise NotImplementedError

    #some information variables
    have_rev_rxns = any(reac.rev for reac in reacs)
    have_pdep_rxns = any(reac.thd or reac.pdep for reac in reacs)


    #the mechanism header defines a number of useful preprocessor defines, as well as defining method stubs for setting initial conditions
    with open(path + 'mechanism.{}h'.format('cu' if lang == 'cuda' else ''), 'w') as file:

        file.write('#ifndef MECHANISM_{}H\n'.format('cu' if lang == 'cuda' else '') + 
                   '#define MECHANISM_{}H\n\n'.format('cu' if lang == 'cuda' else ''))

        #convience: write species indexes
        file.write('/* Species Indexes\n')
        file.write('\n'.join('{}  {}'.format(i, spec.name) for i, spec in enumerate(specs)))
        file.write('*/\n\n')

        #defines
        file.write("//Number of species\n")
        file.write('#define NSP {}\n'.format(len(specs)))
        file.write("//Number of variables. NN = NSP + 1 (temperature)\n")
        file.write('#define NN {}\n'.format(len(specs) + 1))
        file.write('//Number of forward reactions\n')
        file.write('#define FWD_RATES {}\n'.format(len(reacs)))
        file.write('//Number of reversible reactions\n')
        file.write('#define REV_RATES {}\n'.format(len([reac for reac in reacs if reac.rev])))
        file.write('//Number of reactions with pressure modified rates\n')
        file.write('#define PRES_MOD_RATES {}\n\n'.format(len([reac for reac in reacs if reac.pdep or reac.thd])))

        if lang == 'cuda':
          file.write('#ifdef __cplusplus\n'
                     'extern "C" {\n'
                     '#endif\n')

        file.write('    //Must be implemented by user on a per mechanism basis in mechanism.c\n'
                   '    #ifdef CONP\n'
                   '    {} set_same_initial_conditions(int NUM,{} double* y_host, double* pres_host);\n'\
                        .format(' int block_size, int grid_size, ' if lang == 'cuda' else '', 'int' if lang == 'cuda' else 'void') + 
                   '    #elif CONV\n'
                   '    {} set_same_initial_conditions(int NUM,{} double* y_host, double* pres_host, double* rho_host);\n'\
                        .format(' int block_size, int grid_size, ' if lang == 'cuda' else '', 'int' if lang == 'cuda' else 'void') + 
                   '    #endif\n'
                   '    #ifdef RATES_TEST\n'
                   '    void write_jacobian_and_rates_output();\n'
                   '    #endif\n'
                   )

        if lang == 'cuda':
          #close previous extern
          file.write('#ifdef __cplusplus\n'
                     '}\n'
                     '#endif\n\n')

        file.write('#endif\n\n')

    #now the mechanism file
    with open(path + 'mechanism.c{}'.format('u' if lang == 'cuda' else ''), 'w') as file:
        file.write('#include <stdio.h>\n'
                   '#include "mass_mole.h"\n'
                   '#include "mechanism.{}h"\n'.format('cu' if lang == 'cuda' else '') + 
                   '#ifdef RATES_TEST\n'
                   '    #include "rates.h"\n'
                   '    #include "derivs.h"\n'
                   '#endif\n')
        if lang == 'cuda':
            file.write('#include "gpu_memory.cuh"\n'
                       '#include "gpu_macros.cuh"\n'
                       'extern __device__ double* y;\n'
                       'extern __device__ double* pres;\n'
                       '#ifdef CONV\n'
                       '    extern __device__ double* rho;\n'
                       '#endif\n'
                       )
        file.write('#ifdef CONP\n'
                   '    {} set_same_initial_conditions(int NUM, double* y_host, double* pres_host) {{\n'.format('int' if lang == 'cuda' else 'void') + 
                   '#elif CONV\n'
                   '    {} set_same_initial_conditions(int NUM, double* y_host, double* pres_host, double* rho_host) {{\n'.format('int' if lang == 'cuda' else 'void') +
                   '#endif\n'
                   )
        if lang == 'cuda':
            #do cuda mem init and copying
            file.write('        int size = initialize_gpu_memory(NUM, block_size, grid_size);\n')
        else:
            file.write('        int size = NUM;\n')
        file.write('        y_host = (double*)malloc(NN * size * sizeof(double));\n'
                   '        pres_host = (double*)malloc(size * sizeof(double));\n'
                   '#ifdef CONV\n'
                   '        rho_host = (double*)malloc(size * sizeof(double));\n'
                   '#endif'
                   '        double Xi [NSP] = {0.0};\n'
                   '        //set initial mole fractions here\n\n'
                   '        //Normalize mole fractions to sum to one\n'
                   '        double Xsum = 0.0;\n'
                   '        for (int j = 0; j < NSP; ++ j) {\n'
                   '            Xsum += Xi[j];\n'
                   '        }\n'
                   '        if (Xsum == 0.0) {\n'
                   '            printf("Use of the set initial conditions function requires user implementation!");\n'
                   '            exit(-1);\n'
                   '        }\n'
                   '        for (int j = 0; j < NSP; ++ j) {\n'
                   '            Xi[j] /= Xsum;\n'
                   '        }\n\n'
                   '        //convert to mass fractions\n'
                   '        double Yi[NSP];\n'
                   '        mole2mass(Xi, Yi);\n\n'
                   '        //set initial pressure, units [dyn/cm^2]\n'
                   '        double pres = 1.01325e6;\n'
                   '        // set intial temperature, units [K]\n'
                   '        double T0 = 1600;\n\n'
                   '        //load temperature and mass fractions for all threads (cells)\n'
                   '        for (int i = 0; i < size; ++i) {\n'
                   '            y_host[i] = T0;\n'
                   '            //loop through species\n'
                   '            for (int j = 1; j < NN; ++j) {\n'
                   '                y_host[i + NUM * j] = Yi[j - 1];\n'
                   '            }\n'
                   '        }\n\n'
                   '        #ifdef CONV\n'
                   '        //calculate density\n'
                   '        double rho = getDensity(T0, pres, Xi);\n'
                   '        #endif\n\n'
                   '        for (int i = 0; i < size; ++i) {\n'
                   '            #ifdef CONV\n'
                   '            rho_host[i] = rho;\n'
                   '            #endif\n'
                   '            pres_host[i] = pres;\n'
                   '        }\n'
                   )
        if lang == 'cuda': #copy memory over
            file.write('        cudaMemcpy(y, y_host, size * NN * sizeof(double), cudaMemcpyHostToDevice);\n'
                       '        cudaMemcpy(pres, pres_host, size * sizeof(double), cudaMemcpyHostToDevice);\n'
                       '#ifdef CONV\n'
                       '        cudaMemcpy(rho, rho_host, size * sizeof(double), cudaMemcpyHostToDevice);\n'
                       '#endif\n')

        file.write('    }\n\n')
        file.write('#ifdef RATES_TEST\n')
        #write utility function that finds concentrations at a given state
        file.write('    void get_concentrations(double pres, const double* y, double* conc) {\n'
                   '        double rho;\n'
                   )

        line = '        rho = '
        isfirst = True
        for sp in specs:
            if len(line) > 70:
                line += '\n'
                file.write(line)
                line = '         '
            
            if not isfirst: line += ' + '
            line += '(y[{}] / {})'.format(specs.index(sp) + 1, sp.mw)
            isfirst = False

        line += ';\n'
        file.write(line)
        line = '        rho = pres / ({:.8e} * y[0] * rho);\n\n'.format(chem.RU)
        file.write(line)


        # calculation of species molar concentrations
        file.write('        // species molar concentrations\n')
        # loop through species
        for sp in specs:
            isp = specs.index(sp)
            line = '        conc[{}] = rho * y[{}] / '.format(isp, isp)
            line += '{}'.format(sp.mw) + utils.line_end[lang]
            file.write(line)
        
        file.write('    }\n\n')


        line = 'void write_rates(FILE* fp, const double* fwd_rates'
        if have_rev_rxns:
            line += ', const double* rev_rates'
        if have_pdep_rxns:
            line += ', const double* pres_mod'
        line += 'const double dy*) {\n'
        file.write(line)
        #convience method to write rates to file
        file.write('        for(int i = 0; i < FWD_RATES; ++i) {\n'
                   '            fprintf(fp, "%.15le\\n", fwd_rates[i]);\n'
                   '        }\n'
                   )
        if have_rev_rxns:
            file.write('        for(int i = 0; i < REV_RATES; ++i) {\n'
                       '            fprintf(fp, "%.15le\\n", rev_rates[i]);\n'
                       '        }\n'
                       )
        if have_pdep_rxns:
            file.write('        for(int i = 0; i < PRES_MOD_RATES; i++) {\n'
                       '            fprintf(fp, "%.15le\\n", pres_mod[i]);\n'
                       '        }\n'
                       )
        file.write('        for(int i = 0; i < NSP; i++) {\n'
                   '            fprintf(fp, "%.15le\\n", dy[i]);\n'
                   '        }\n'
                   '    }\n\n'
                  )

        file.write('    void write_jacob(FILE* fp, const double* jacob) {\n'
                   '        for (int i = 0; i < NN * NN; ++i) {\n'
                   '            fprintf(fp, "%.15le\\n", jacob[i]);\n'
                   '        }\n'
                   '    }\n\n'
                   )


        file.write('    void write_jacobian_and_rates_output() {\n'
                   '        //set mass fractions to unity to turn on all reactions\n'
                   '        double Yi[NN];\n'
                   '        for (int i = 1; i < NN; ++i) {\n'
                   '            Yi[i] = 1.0 / ((double)NSP);\n'
                   '        }\n'
                   '        double conc[NSP];\n'
                   '        double fwd_rates[FWD_RATES];\n'
                   )
        if have_rev_rxns:
            file.write('        double rev_rates[REV_RATES];\n')
        if have_pdep_rxns:
            file.write('        double pres_mod[PRES_MOD_RATES];\n')

        file.write('        double dy[NSP];\n'
                   '        double jacob[NN * NN];\n'
                   '        //evaluate and write rates for various conditions\n'
                   '        FILE* fp = fopen ("rates_data.txt", "w");\n'
                   )
        __write_rate_evaluator(file, have_rev_rxns, have_pdep_rxns, '800', '1.01325e6', '1')
        __write_rate_evaluator(file, have_rev_rxns, have_pdep_rxns, '1600', '1.01325e6', '1')
        __write_rate_evaluator(file, have_rev_rxns, have_pdep_rxns, '800', '1.01325e7', '10')
        file.write('        fclose(fp);\n'
                   '    }\n'
                   )
        file.write('#endif')

    if lang == 'cuda' and CUDAParams.is_global():
        with open(path + 'gpu_memory.cuh', 'w') as file:
            file.write('#ifndef GPU_MEMORY_CUH\n'
                       '#define GPU_MEMORY_CUH\n'
                       '\n'
                       '#include "mechanism.cuh"\n'
                       '\n'
                       'int initialize_gpu_memory(int NUM, int block_size, int grid_size);\n'
                       'void free_gpu_memory();\n'
                       '\n'
                       '#endif\n')

        with open(path + 'gpu_memory.cu', 'w') as file:
            file.write('//define global memory pointers')
            template = '__device__ double* {};'
            host_template = 'double* d_{};'
            conp = ['h', 'cp']
            conv = ['u', 'cv']
            arrays = ['y', 'dy', 'conc', 'fwd_rates' if have_rev_rxns else '', 'rev_rates' if have_rev_rxns else '', 'rates' if have_rev_rxns else '',\
             'pres_mod' if have_pdep_rxns else '', 'sp_rates']
            arrays = [arr for arr in arrays if arr.strip()]

            all_arrays_conv = arrays + conv
            all_arrays_conp = arrays + conp
            method_template = 'double* {}_in'
            setter_template = '{} = {}_in;'

            file.write('\n#ifdef CONP\n' + 
                        '\n'.join([template.format(arr) for arr in conp]) +
                        '\n#else\n' +
                        '\n'.join([template.format(arr) for arr in conv]) +
                        '\n#endif\n'
                        )
            file.write('\n'.join([template.format(arr) for arr in arrays]))
            file.write('\n\n'
                       '#ifdef CONP\n'
                       '__global__ pointer_set_kernel(' +
                       ', '.join([method_template.format(arr) for arr in all_arrays_conp]) +
                       ') {\n  ' +
                       '\n  '.join([setter_template.format(arr, arr) for arr in all_arrays_conp]) + 
                       '\n}\n'
                       '#else\n'
                       '__global__ pointer_set_kernel(' +
                       ', '.join([method_template.format(arr) for arr in all_arrays_conv]) +
                       ') {\n  ' +
                       '\n  '.join([setter_template.format(arr, arr) for arr in all_arrays_conv]) + 
                       '\n}\n'
                       '#endif\n'
                       )

            file.write('\n#ifdef CONP\n' + 
            '\n'.join([host_template.format(arr) for arr in conp]) +
            '\n#else\n' +
            '\n'.join([host_template.format(arr) for arr in conv]) +
            '\n#endif\n' +
            '\n'.join([host_template.format(arr) for arr in arrays])
            )
            file.write('\n')
            malloc_template = 'cudaMalloc((void**)&{}, {});'
            free_template = 'cudaFree({});'
            file.write('\nvoid initialize_gpu_memory(int NUM, int block_size, int grid_size) {\n'
                       '  int padded = grid_size * block_size;\n'
                       '  padded = padded > NUM ? padded : NUM;\n'
                       '#ifdef CONP\n  ' +
                       '\n  '.join([malloc_template.format('d_' + arr, 'NN * padded * sizeof(double)') for arr in all_arrays_conp[:2]]) +
                       '\n  ' +
                       '\n  '.join([malloc_template.format('d_' + arr, 'NSP * padded * sizeof(double)') for arr in all_arrays_conp[2:]]) +
                       '\n  pointer_set_kernel<<<1, 1>>>({});'.format(', '.join(all_arrays_conp)) +
                       '\n#else\n  ' +
                       '\n  '.join([malloc_template.format('d_' + arr, 'NN * padded * sizeof(double)') for arr in all_arrays_conv[:2]]) +
                       '\n  ' +
                       '\n  '.join([malloc_template.format('d_' + arr, 'NSP * padded * sizeof(double)') for arr in all_arrays_conv[2:]]) +
                       '\n  pointer_set_kernel<<<1, 1>>>({});'.format(', '.join(all_arrays_conv)) +
                       '\n#endif\n'
                       '  return padded;\n'
                       '}'
                       )
            file.write('\n')
            file.write('\n\nvoid free_gpu_memory() {\n'
                       '#ifdef CONP\n  ' +
                       '\n  '.join([free_template.format('d_' + arr) for arr in all_arrays_conp]) +
                       '\n#else\n  ' +
                       '\n  '.join([free_template.format('d_' + arr) for arr in all_arrays_conv]) +
                       '\n#endif\n'
                       '}\n'
                       )