"""Writes mechanism header and output testing files"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import sys
import math
import itertools

# Local imports
import chem_utilities as chem
import utils
import CUDAParams


def __write_kernels(file, have_rev_rxns, have_pdep_rxns):
    """
    Writes kernels that simply act as shells to call the various reaction/species/jacobian
    """
    if have_rev_rxns:
        file.write('__global__ void k_eval_rxn_rates(const double T, const double* conc, double* fwd_rates, double* rev_rates) {\n'
                   '    eval_rxn_rates(T, conc, fwd_rates, rev_rates);\n'
                   '}\n'
                   )
    else:
        file.write('__global__ void k_eval_rxn_rates(const double T, const double* conc, double* rates) {\n'
                   '    eval_rxn_rates(T, conc, rates);\n'
                   '}\n'
                   )
    if have_pdep_rxns:
        file.write('__global__ void k_get_rxn_pres_mod(const double T, const double P, const double* conc, double* pres_mod) {\n'
                   '    get_rxn_pres_mod(T, P, conc, pres_mod);\n'
                   '}\n')
    file.write('__global__ void k_eval_spec_rates({}{} double* dy) {{\n'.format('const double* fwd_rates, const double* rev_rates,' if have_rev_rxns else 'const double* rates,',
                                                                                ' const double* pres_mod,' if have_pdep_rxns else '') +
               '    eval_spec_rates({}{} dy);\n'.format('fwd_rates, rev_rates,' if have_rev_rxns else 'rates,',
                                                        ' pres_mod,' if have_pdep_rxns else '') +
               '}\n'
               )
    file.write('__global__ void k_eval_jacob(const double t, const double P, double* y, double* jac) {\n'
               '    eval_jacob(t, P, y, jac);\n'
               '}\n')
    file.write('\n')


def __write_c_rate_evaluator(file, have_rev_rxns, have_pdep_rxns, T, P, Pretty_P):
    file.write('        fprintf(fp, "{}, {} atm\\n");\n'.format(T, Pretty_P) +
               '        y_host[0] = {};\n'.format(T) +
               '        get_concentrations({}, y_host, conc_host);\n'.format(P)
               )
    if have_rev_rxns:
        file.write(
            '        eval_rxn_rates({}, conc_host, fwd_rates_host, rev_rates_host);\n'.format(T))
    else:
        file.write(
            '        eval_rxn_rates({}, conc_host, rates_host);\n'.format(T))
    if have_pdep_rxns:
        file.write(
            '        get_rxn_pres_mod ({}, {}, conc_host, pres_mod_host);\n'.format(T, P))
    file.write('        eval_spec_rates ({}{} dy_host);\n'.format('fwd_rates_host, rev_rates_host,' if have_rev_rxns else 'rates_host, ', ' pres_mod_host,' if have_pdep_rxns else '') +
               '        write_rates(fp,{}{} dy_host);\n'.format(' fwd_rates_host, rev_rates_host,' if have_rev_rxns else ' rates_host, ', ' pres_mod_host,' if have_pdep_rxns else '') +
               '        eval_jacob(0, {}, y_host, jacob_host);\n'.format(P) +
               '        write_jacob(fp, jacob_host);\n'
               )


def __write_cuda_rate_evaluator(file, have_rev_rxns, have_pdep_rxns, T, P, Pretty_P):
    file.write('    fprintf(fp, "{}, {} atm\\n");\n'.format(T, Pretty_P) +
               '    y_host[0] = {};\n'.format(T) +
               '    get_concentrations({}, y_host, conc_host);\n'.format(P))
    file.write('    for (int i = 0; i < NUM; ++i) {\n'
               '        for (int j = 0; j < NSP; ++j) {\n'
               '            conc_host_full[i + j * NUM] = conc_host[j];\n'
               '        }\n'
               '    }\n'
               )
    file.write(
        '    cudaErrorCheck(cudaMemcpy(d_conc, conc_host_full, NUM * NSP * sizeof(double), cudaMemcpyHostToDevice));\n')
    if have_rev_rxns:
        file.write('    k_eval_rxn_rates<<<grid_size, block_size>>>({}, d_conc, d_fwd_rates, d_rev_rates);\n'.format(T) +
                   '    cudaErrorCheck(cudaMemcpy(fwd_rates_host_full, d_fwd_rates, NUM * FWD_RATES * sizeof(double), cudaMemcpyDeviceToHost));\n'
                   '    cudaErrorCheck(cudaMemcpy(rev_rates_host_full, d_rev_rates, NUM * REV_RATES * sizeof(double), cudaMemcpyDeviceToHost));\n')
        file.write('    for (int j = 0; j < FWD_RATES; ++j) {\n'
                   '            fwd_rates_host[j] = fwd_rates_host_full[j * NUM];\n'
                   '    }\n'
                   )
        file.write('    for (int j = 0; j < REV_RATES; ++j) {\n'
                   '        rev_rates_host[j] = rev_rates_host_full[j * NUM];\n'
                   '    }\n'
                   )
    else:
        file.write(
            '    k_eval_rxn_rates<<<grid_size, block_size>>>({}, d_conc, d_rates);\n'.format(T) +
            '    cudaErrorCheck(cudaMemcpy(rates_host_full, d_rates, NUM * RATES * sizeof(double), cudaMemcpyDeviceToHost));\n'
        )
        file.write('    for (int j = 0; j < RATES; ++j) {\n'
                   '        rates_host[j] = rates_host_full[j * NUM];\n'
                   '    }\n'
                   )
    if have_pdep_rxns:
        file.write(
            '    k_get_rxn_pres_mod<<<grid_size, block_size>>>({}, {}, d_conc, d_pres_mod);\n'.format(T, P) +
            '    cudaErrorCheck(cudaMemcpy(pres_mod_host_full, d_pres_mod, NUM * PRES_MOD_RATES * sizeof(double), cudaMemcpyDeviceToHost));\n'
        )
        file.write('    for (int j = 0; j < PRES_MOD_RATES; ++j) {\n'
                   '        pres_mod_host[j] = pres_mod_host_full[j * NUM];\n'
                   '    }\n'
                   )

    if have_rev_rxns:
        file.write('    k_eval_spec_rates<<<grid_size, block_size>>>(d_fwd_rates, d_rev_rates,{} d_dy);\n'.format(' d_pres_mod,' if have_pdep_rxns else '') +
                   '    cudaErrorCheck(cudaMemcpy(dy_host_full, d_dy, NUM * NN * sizeof(double), cudaMemcpyDeviceToHost));\n'
                   '    for (int j = 0; j < NN; ++j) {\n'
                   '        dy_host[j] = dy_host_full[j * NUM];\n'
                   '    }\n' +
                   '    write_rates(fp,{}{} dy_host);\n'.format(' fwd_rates_host, rev_rates_host,' if have_rev_rxns else ' rates,', ' pres_mod_host,' if have_pdep_rxns else '') +
                   '    k_eval_jacob<<<grid_size, block_size>>>(0, {}, d_y, d_jac);\n'.format(P) +
                   '    cudaErrorCheck(cudaMemcpy(jacob_host_full, d_jac, NUM * NN * NN * sizeof(double), cudaMemcpyDeviceToHost));\n'
                   '    for (int j = 0; j < NN * NN; ++j) {\n'
                   '        jacob_host[j] = jacob_host_full[j * NUM];\n'
                   '    }\n'
                   '    write_jacob(fp, jacob_host);\n'
                   )


def write_mechanism_initializers(path, lang, specs, reacs):
    if lang in ['matlab', 'fortran']:
        raise NotImplementedError

    # some information variables
    have_rev_rxns = any(reac.rev for reac in reacs)
    have_pdep_rxns = any(reac.thd or reac.pdep for reac in reacs)

    # the mechanism header defines a number of useful preprocessor defines, as
    # well as defining method stubs for setting initial conditions
    with open(path + 'mechanism.{}h'.format('cu' if lang == 'cuda' else ''), 'w') as file:

        file.write('#ifndef MECHANISM_{}H\n'.format('cu' if lang == 'cuda' else '') +
                   '#define MECHANISM_{}H\n\n'.format('cu' if lang == 'cuda' else ''))

        # convience: write species indexes
        file.write('/* Species Indexes\n')
        file.write('\n'.join('{}  {}'.format(i, spec.name)
                             for i, spec in enumerate(specs)))
        file.write('*/\n\n')

        # defines
        if lang == 'cuda' and CUDAParams.is_global():
            file.write('//Global Memory\n'
                       '#define GLOBAL_MEM\n')
        file.write("//Number of species\n")
        file.write('#define NSP {}\n'.format(len(specs)))
        file.write("//Number of variables. NN = NSP + 1 (temperature)\n")
        file.write('#define NN {}\n'.format(len(specs) + 1))
        if have_rev_rxns:
            file.write('//Number of forward reactions\n')
            file.write('#define FWD_RATES {}\n'.format(len(reacs)))
            file.write('//Number of reversible reactions\n')
            file.write('#define REV_RATES {}\n'.format(
                len([reac for reac in reacs if reac.rev])))
        else:
            file.write('//Number of reactions\n'
                       '#define RATES {}\n'.format(len(reac)))
        if have_pdep_rxns:
            file.write('//Number of reactions with pressure modified rates\n')
            file.write('#define PRES_MOD_RATES {}\n\n'.format(
                len([reac for reac in reacs if reac.pdep or reac.thd])))

        if lang == 'cuda':
            file.write('#ifdef __cplusplus\n'
                       'extern "C" {\n'
                       '#endif\n')

        file.write('    //Must be implemented by user on a per mechanism basis in mechanism.c\n'
                   '    #ifdef CONP\n'
                   '    {} set_same_initial_conditions(int NUM,{} double* y_host, double* pres_host);\n'
                   .format('int' if lang == 'cuda' else 'void', ' int block_size, int grid_size, ' if lang == 'cuda' else '') +
                   '    #elif CONV\n'
                   '    {} set_same_initial_conditions(int NUM,{} double* y_host, double* pres_host, double* rho_host);\n'
                   .format('int' if lang == 'cuda' else 'void', ' int block_size, int grid_size, ' if lang == 'cuda' else '') +
                   '    #endif\n'
                   '    #ifdef RATES_TEST\n'
                   )
        if lang == 'c':
            file.write('    void write_jacobian_and_rates_output();\n'
                       '    #endif\n'
                       '\n'
                       )
        else:
            file.write('    void write_jacobian_and_rates_output(int NUM, int block_size, int grid_size);\n'
                       '    #endif\n'
                       '\n'
                       )

        if lang == 'cuda':
            # close previous extern
            file.write('#ifdef __cplusplus\n'
                       '}\n'
                       '#endif\n\n')

        file.write('#endif\n\n')

    # now the mechanism file
    with open(path + 'mechanism.c{}'.format('u' if lang == 'cuda' else ''), 'w') as file:
        file.write('#include <stdio.h>\n'
                   '#include "mass_mole.h"\n'
                   '#include "mechanism.{}h"\n'.format('cu' if lang == 'cuda' else '') +
                   '#ifdef RATES_TEST\n'
                   '    #include "rates.{}h"\n'.format('cu' if lang == 'cuda' else '') +
                   '    #include "jacob.{}h"\n'.format('cu' if lang == 'cuda' else '') +
                   '#endif\n')
        if lang == 'cuda':
            file.write('#include <cuda.h>\n'
                       '#include <cuda_runtime.h>\n'
                       '#include <helper_cuda.h>\n'
                       )

            file.write('#include "gpu_macros.cuh"\n')
            if CUDAParams.is_global():
                file.write('#include "gpu_memory.cuh"\n\n'
                           'extern double* d_y;\n'
                           'extern double* d_pres;\n'
                           'extern double* d_dy;\n'
                           '#ifdef CONV\n'
                           '    extern double* d_rho;\n'
                           '#endif\n'
                           '#ifdef RATES_TEST\n'
                           '    extern double* d_conc;\n'
                           )
                if have_rev_rxns:
                    file.write('    extern double* d_fwd_rates;\n'
                               '    extern double* d_rev_rates;\n')
                else:
                    file.write('    extern double* d_rates;\n')
                if have_pdep_rxns:
                    file.write('    extern double* d_pres_mod;\n')
                file.write('    extern double* d_jac;\n')
                file.write('#endif\n\n')
            __write_kernels(file, have_rev_rxns, have_pdep_rxns)

        needed_arr = ['y', 'pres']
        needed_arr_conv = ['y', 'pres', 'rho']
        if lang == 'cuda':
            if CUDAParams.is_global():
                needed_arr = ['double* ' + a + '_host' for a in needed_arr]
                needed_arr_conv = [
                    'double* ' + a + '_host' for a in needed_arr_conv]
            else:
                needed_arr = [['double* ' + a + '_host', 'double* ' + a + '_d']
                              for a in needed_arr]
                needed_arr = [a for a in itertools.chain(*needed_arr)]
                needed_arr_conv = [
                    ['double* ' + a + '_host', 'double* ' + a + '_d'] for a in needed_arr_conv]
                needed_arr_conv = [
                    a + '_d' for a in itertools.chain(*needed_arr_conv)]
        file.write('#ifdef CONP\n'
                   '{} set_same_initial_conditions(int NUM{}, {}) \n'.format('int' if lang == 'cuda' else 'void',
                                                                             ', int block_size, int grid_size' if lang == 'cuda' else '', ', '.join(needed_arr)) +
                   '#elif CONV\n'
                   '{} set_same_initial_conditions(int NUM{}, {}) \n'.format('int' if lang == 'cuda' else 'void',
                                                                             ', int block_size, int grid_size' if lang == 'cuda' else '', ', '.join(needed_arr_conv)) +
                   '#endif\n'
                   )
        file.write('{\n')
        if lang == 'cuda':
                # do cuda mem init and copying
            file.write(
                '    int size = initialize_gpu_memory(NUM, block_size, grid_size);\n')
        else:
            file.write('    int size = NUM;\n')
        file.write('    y_host = (double*)malloc(NN * size * sizeof(double));\n'
                   '    pres_host = (double*)malloc(size * sizeof(double));\n'
                   '#ifdef CONV\n'
                   '    rho_host = (double*)malloc(size * sizeof(double));\n'
                   '#endif\n'
                   '    double Xi [NSP] = {0.0};\n'
                   '    //set initial mole fractions here\n\n'
                   '    //Normalize mole fractions to sum to one\n'
                   '    double Xsum = 0.0;\n'
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
                   '    double Yi[NSP];\n'
                   '    mole2mass(Xi, Yi);\n\n'
                   '    //set initial pressure, units [dyn/cm^2]\n'
                   '    double P = 1.01325e6;\n'
                   '    // set intial temperature, units [K]\n'
                   '    double T0 = 1600;\n\n'
                   '    //load temperature and mass fractions for all threads (cells)\n'
                   '    for (int i = 0; i < size; ++i) {\n'
                   '        y_host[i] = T0;\n'
                   '        //loop through species\n'
                   '        for (int j = 1; j < NN; ++j) {\n'
                   '            y_host[i + NUM * j] = Yi[j - 1];\n'
                   '        }\n'
                   '    }\n\n'
                   '#ifdef CONV\n'
                   '    //calculate density\n'
                   '    double rho = getDensity(T0, pres, Xi);\n'
                   '#endif\n\n'
                   '    for (int i = 0; i < size; ++i) {\n'
                   '#ifdef CONV\n'
                   '        rho_host[i] = rho;\n'
                   '#endif\n'
                   '        pres_host[i] = P;\n'
                   '    }\n'
                   )
        if lang == 'cuda':  # copy memory over
            file.write('    cudaMemcpyToSymbol(d_y, y_host, size * NN * sizeof(double));\n'
                       '    cudaMemcpyToSymbol(d_pres, pres_host, size * sizeof(double));\n'
                       '#ifdef CONV\n'
                       '    cudaMemcpyToSymbol(d_rho, rho_host, size * sizeof(double));\n'
                       '#endif\n')
            file.write('    return size;\n')

        file.write('}\n\n')
        # write utility function that finds concentrations at a given state
        file.write('void get_concentrations(double P, const double* y_host, double* conc_host) {\n'
                   '    double rho;\n'
                   )

        line = '    rho = '
        isfirst = True
        for sp in specs:
            if len(line) > 70:
                line += '\n'
                file.write(line)
                line = '         '

            if not isfirst:
                line += ' + '
            line += '(y_host[{}] / {})'.format(specs.index(sp) + 1, sp.mw)
            isfirst = False

        line += ';\n'
        file.write(line)
        line = '    rho = P / ({:.8e} * y_host[0] * rho);\n\n'.format(chem.RU)
        file.write(line)

        # calculation of species molar concentrations
        file.write('    // species molar concentrations\n')
        # loop through species
        for sp in specs:
            isp = specs.index(sp)
            line = '    conc_host[{}] = rho * y_host[{}] / '.format(isp, isp)
            line += '{}'.format(sp.mw) + utils.line_end[lang]
            file.write(line)

        file.write('}\n\n')

        line = 'void write_rates(FILE* fp, const double* fwd_rates_host'
        if have_rev_rxns:
            line += ', const double* rev_rates_host'
        if have_pdep_rxns:
            line += ', const double* pres_mod_host'
        line += ', const double* dy_host) {\n'
        file.write(line)
        # convience method to write rates to file
        file.write('    for(int i = 0; i < FWD_RATES; ++i) {\n'
                   '        fprintf(fp, "%.15le\\n", fwd_rates_host[i]);\n'
                   '    }\n'
                   )
        if have_rev_rxns:
            file.write('    for(int i = 0; i < REV_RATES; ++i) {\n'
                       '        fprintf(fp, "%.15le\\n", rev_rates_host[i]);\n'
                       '    }\n'
                       )
        if have_pdep_rxns:
            file.write('    for(int i = 0; i < PRES_MOD_RATES; i++) {\n'
                       '        fprintf(fp, "%.15le\\n", pres_mod_host[i]);\n'
                       '    }\n'
                       )
        file.write('    for(int i = 0; i < NN; i++) {\n'
                   '        fprintf(fp, "%.15le\\n", dy_host[i]);\n'
                   '    }\n'
                   '}\n\n'
                   )

        file.write('void write_jacob(FILE* fp, const double* jacob_host) {\n'
                   '    for (int i = 0; i < NN * NN; ++i) {\n'
                   '        fprintf(fp, "%.15le\\n", jacob_host[i]);\n'
                   '    }\n'
                   '}\n\n'
                   )

        if lang != 'cuda':
            file.write('    void write_jacobian_and_rates_output() {\n'
                       '        //set mass fractions to unity to turn on all reactions\n'
                       '        double y_host[NN];\n'
                       '        for (int i = 1; i < NN; ++i) {\n'
                       '            y_host[i] = 1.0 / ((double)NSP);\n'
                       '        }\n'
                       '        double conc_host[NSP];\n'
                       )
            if have_rev_rxns:
                file.write('        double fwd_rates_host[FWD_RATES];\n'
                           '        double rev_rates_host[REV_RATES];\n')
            else:
                file.write('        double rates_host[RATES];\n')
            if have_pdep_rxns:
                file.write('        double pres_mod_host[PRES_MOD_RATES];\n')

            file.write('        double dy_host[NN];\n'
                       '        double jacob_host[NN * NN];\n'
                       '        //evaluate and write rates for various conditions\n'
                       '        FILE* fp = fopen ("rates_data.txt", "w");\n'
                       )
            __write_c_rate_evaluator(
                file, have_rev_rxns, have_pdep_rxns, '800', '1.01325e6', '1')
            __write_c_rate_evaluator(
                file, have_rev_rxns, have_pdep_rxns, '1600', '1.01325e6', '1')
            __write_c_rate_evaluator(
                file, have_rev_rxns, have_pdep_rxns, '800', '1.01325e7', '10')
            file.write('        fclose(fp);\n'
                       '    }\n'
                       )
        else:
            file.write('void write_jacobian_and_rates_output(int NUM, int block_size, int grid_size) {\n'
                       '    //set mass fractions to unity to turn on all reactions\n'
                       '    double y_host[NN];\n'
                       '    for (int i = 1; i < NN; ++i) {\n'
                       '        y_host[i] = 1.0 / ((double)NSP);\n'
                       '    }\n'
                       '    double conc_host[NSP];\n'
                       '    int padded = initialize_gpu_memory(NUM, block_size, grid_size);\n'
                       '    NUM = padded > NUM ? padded : NUM;\n'
                       '    double* conc_host_full = (double*)malloc(NUM * NSP * sizeof(double));\n'
                       )
            if have_rev_rxns:
                file.write('    double* fwd_rates_host_full = (double*)malloc(NUM * FWD_RATES * sizeof(double));\n'
                           '    double* rev_rates_host_full = (double*)malloc(NUM * REV_RATES * sizeof(double));\n'
                           '    double fwd_rates_host[FWD_RATES];\n'
                           '    double rev_rates_host[REV_RATES];\n'
                           )
            else:
                file.write('    double* rates_host_full = (double*)malloc(NUM * RATES * sizeof(double));\n'
                           '    double rates_host[RATES];\n')
            if have_pdep_rxns:
                file.write('    double* pres_mod_host_full = (double*)malloc(NUM * PRES_MOD_RATES * sizeof(double));\n'
                           '    double pres_mod_host[PRES_MOD_RATES];\n')

            file.write('    double dy_host[NN];\n'
                       '    double jacob_host[NN * NN];\n'
                       '    double* dy_host_full = (double*)malloc(NUM * NN * sizeof(double));\n'
                       '    double* jacob_host_full = (double*)malloc(NUM * NN * NN * sizeof(double));\n'
                       '    //evaluate and write rates for various conditions\n'
                       '    FILE* fp = fopen ("rates_data.txt", "w");\n'
                       )

        if not CUDAParams.is_global():
            # need to define arrays
            file.write('    double* y = cudaMalloc(NUM * NN * sizeof(double));\n'
                       '    double* pres = cudaMalloc(NUM * sizeof(double));\n'
                       '#ifdef CONV\n'
                       '    double* rho = cudaMalloc(NUM * sizeof(double));\n'
                       '#endif\n'
                       '    double* conc = cudaMalloc(NUM * NSP * sizeof(double));\n'
                       )
            if have_rev_rxns:
                file.write('    double* fwd_rates = cudaMalloc(NUM * FWD_RATES * sizeof(double));\n'
                           '    double* rev_rates = cudaMalloc(NUM * REV_RATES * sizeof(double));\n')
            else:
                file.write(
                    '    double* rates = cudaMalloc(NUM * RATES * sizeof(double));\n')
            if have_pdep_rxns:
                file.write(
                    '    double* pres_mod = cudaMalloc(NUM * PRES_MOD_RATES * sizeof(double));\n')
            file.write(
                '    double* jac = cudaMalloc(NUM * NN * NN * sizeof(double));\n')

        __write_cuda_rate_evaluator(
            file, have_rev_rxns, have_pdep_rxns, '800', '1.01325e6', '1')
        __write_cuda_rate_evaluator(
            file, have_rev_rxns, have_pdep_rxns, '1600', '1.01325e6', '1')
        __write_cuda_rate_evaluator(
            file, have_rev_rxns, have_pdep_rxns, '800', '1.01325e7', '10')
        file.write('    fclose(fp);\n'
                   '    free(conc_host_full);\n'
                   '    free(dy_host_full);\n'
                   '    free(jacob_host_full);\n'
                   )
        if have_rev_rxns:
            file.write('    free(fwd_rates_host_full);\n'
                       '    free(rev_rates_host_full);\n'
                       )
        else:
            file.write('    free(rates_host_full);\n')
        if have_pdep_rxns:
            file.write('    free(pres_mod_host_full);\n')

        if not CUDAParams.is_global():
            file.write('    cudaErrorCheck(cudaFree(y));\n'
                       '    cudaErrorCheck(cudaFree(pres));\n'
                       '#ifdef CONV\n'
                       '    cudaErrorCheck(cudaFree(rho));\n'
                       '#endif'
                       '    cudaErrorCheck(cudaFree(conc));\n'
                       )
            if have_rev_rxns:
                file.write('    cudaErrorCheck(cudaFree(fwd_rates));\n')
                file.write('    cudaErrorCheck(cudaFree(rev_rates));\n')
            else:
                file.write('    cudaErrorCheck(cudaFree(rates));\n')

            if have_pdep_rxns:
                file.write('    cudaErrorCheck(cudaFree(pres_mod));\n')
                file.write('    cudaErrorCheck(cudaFree(jac));\n')
        else:
            file.write('    free_gpu_memory();\n')

        file.write('}\n')

    if lang == 'cuda':
        with open(path + 'gpu_memory.cuh', 'w') as file:
            file.write('#ifndef GPU_MEMORY_CUH\n'
                       '#define GPU_MEMORY_CUH\n'
                       '\n'
                       '#include "header.h"\n'
                       '#include "mechanism.cuh"\n'
                       '#include "gpu_macros.cuh"\n'
                       '\n'
                       )
            if CUDAParams.is_global():
                file.write('int initialize_gpu_memory(int NUM, int block_size, int grid_size);\n'
                           'void free_gpu_memory();\n'
                           '\n'
                           '#endif\n')
            else:
                file.write('#ifdef CONP\n'
                           'int initialize_gpu_memory(int NUM, int block_size, int grid_size, double* y_device, double* pres_device);\n'
                           'void free_gpu_memory(double* y_device, double* pres_device);\n'
                           '#else\n'
                           'int initialize_gpu_memory(int NUM, int block_size, int grid_size, double* y_device, double* rho_device);\n'
                           'void free_gpu_memory(double* y_device, double* rho_device);\n'
                           '#endif\n'
                           '\n'
                           '#endif\n')

        with open(path + 'gpu_memory.cu', 'w') as file:
            malloc_template = 'cudaErrorCheck(cudaMalloc((void**)&{}, {} * sizeof(double)));'
            free_template = 'cudaErrorCheck(cudaFree({}));'
            file.write('#include "gpu_memory.cuh"\n'
                       '\n')
            if CUDAParams.is_global():
                file.write('//define global memory pointers')
                template = '__device__ double* {};'
                host_template = 'double* d_{};'
                conp = [(['h', 'cp'], 'NSP'), (['pres'], '1')]
                conv = [(['u', 'cv'], 'NSP'), (['rho'], '1')]
                arrays = [(['jac'], 'NN * NN')]
                arrays.append(
                    (['y', 'dy', 'f_temp', 'error'], 'NN'))
                arrays.append((['conc'], 'NSP'))
                if have_rev_rxns:
                    arrays.append((['fwd_rates'], 'FWD_RATES'))
                    arrays.append((['rev_rates'], 'REV_RATES'))
                else:
                    arrays.append((['rates'], 'RATES'))
                if have_pdep_rxns:
                    arrays.append((['pres_mod'], 'PRES_MOD_RATES'))

                all_arrays_conv = arrays + conv
                all_arrays_conp = arrays + conp
                method_template = 'double* {}_in'
                setter_template = '{} = {}_in;'

                flat_arrays = [
                    x for x in itertools.chain(*[arr[0] for arr in arrays])]
                flat_conp = [
                    x for x in itertools.chain(*[arr[0] for arr in conp])]
                flat_conv = [
                    x for x in itertools.chain(*[arr[0] for arr in conv])]
                file.write('\n#ifdef CONP\n' +
                           '\n'.join([template.format(arr) for arr in flat_conp]) +
                           '\n#else\n' +
                           '\n'.join([template.format(arr) for arr in flat_conv]) +
                           '\n#endif\n'
                           )
                file.write(
                    '\n'.join([template.format(a) for a in flat_arrays]))
                file.write('\n\n'
                           '#ifdef CONP\n'
                           '__global__ void pointer_set_kernel(' +
                           ', '.join([method_template.format(a) for a in flat_conp + flat_arrays]) +
                           ') {\n  ' +
                           '\n  '.join([setter_template.format(a, a) for a in flat_conp + flat_arrays]) +
                           '\n}\n'
                           '#else\n'
                           '__global__ void pointer_set_kernel(' +
                           ', '.join([method_template.format(a) for a in flat_conv + flat_arrays]) +
                           ') {\n  ' +
                           '\n  '.join([setter_template.format(a, a) for a in flat_conv + flat_arrays]) +
                           '\n}\n'
                           '#endif\n'
                           )

                file.write('\n#ifdef CONP\n' +
                           '\n'.join([host_template.format(a) for a in flat_conp]) +
                           '\n#else\n' +
                           '\n'.join([host_template.format(a) for a in flat_conv]) +
                           '\n#endif\n' +
                           '\n'.join([host_template.format(a)
                                      for a in flat_arrays])
                           )
                file.write('\n')
                file.write('\nint initialize_gpu_memory(int NUM, int block_size, int grid_size) {\n'
                           '  NUM = grid_size * block_size > NUM ? grid_size * block_size : NUM;\n'
                           '#ifdef CONP\n')
                for arr in all_arrays_conp:
                    for a in arr[0]:
                        file.write(
                            '  ' + malloc_template.format('d_' + a, 'NUM * ' + arr[1]) + '\n')

                file.write('\n  pointer_set_kernel<<<1, 1>>>({});'.format(', '.join(['d_' + a for a in flat_conp + flat_arrays])) +
                           '\n#else\n'
                           )

                for arr in all_arrays_conv:
                    for a in arr[0]:
                        file.write(
                            '  ' + malloc_template.format('d_' + a, 'NUM * ' + arr[1]) + '\n')

                file.write(
                    '\n  pointer_set_kernel<<<1, 1>>>({});'.format(', '.join(['d_' + a for a in flat_conv + flat_arrays])) +
                    '\n#endif\n'
                    '  return NUM;\n'
                    '}'
                )
                file.write('\n')
                file.write('\n\nvoid free_gpu_memory() {\n'
                           '#ifdef CONP\n  ' +
                           '\n  '.join([free_template.format('d_' + a) for a in flat_conp + flat_arrays]) +
                           '\n#else\n  ' +
                           '\n  '.join([free_template.format('d_' + a) for a in flat_conv + flat_arrays]) +
                           '\n#endif\n'
                           '}\n'
                           )
            else:
                file.write('#ifdef CONP\n'
                           'int initialize_gpu_memory(int NUM, int block_size, int grid_size, double* y_device, double* pres_device)\n'
                           '#else\n'
                           'int initialize_gpu_memory(int NUM, int block_size, int grid_size, double* y_device, double* rho_device)\n'
                           '#endif\n'
                           '{\n'
                           '    NUM = grid_size * block_size; > NUM ? grid_size * block_size; : NUM;\n'
                           '    cudaErrorCheck(&y_device, NUM * NN * sizeof(double));\n'
                           '#ifdef CONP\n'
                           '    cudaErrorCheck(&pres_device, NN * sizeof(double));\n'
                           '#else\n'
                           '    cudaErrorCheck(&rho_device, NUM * NN * sizeof(double));\n'
                           '#endif\n'
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
        with open(path + 'gpu_macros.cuh', 'w') as file:
            file.write('#ifndef GPU_MACROS_CUH\n'
                       '#define GPU_MACROS_CUH\n'
                       '#include <stdio.h>\n'
                       '#include <cuda.h>\n'
                       '#include <cuda_runtime.h>\n'
                       '#include <helper_cuda.h>\n'
                       '\n'
                       )
            if CUDAParams.is_global():
                file.write('#define GLOBAL_MEM\n')

            file.write('#define CU_LINEAR_OFFSET(I) (threadIdx.x + blockIdx.x * blockDim.x + (I) * blockDim.x * gridDim.x)\n'
                       '\n'
                       '#ifdef GLOBAL_MEM\n'
                       '    #define INDEX(I) (CU_LINEAR_OFFSET(I))\n'
                       '#else\n'
                       '    #define INDEX(I) ((I))\n'
                       '#endif\n'
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
