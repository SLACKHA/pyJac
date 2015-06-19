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

def __write_kernels(file, have_rev_rxns, have_pdep_rxns):
    """ Writes kernels that simply act as shells to call the various
    reaction/species/jacobian.
    """
    if not CUDAParams.is_global():
        if have_rev_rxns:
            file.write(
                '#ifdef PROFILER\n'
                '__global__ void k_eval_rxn_rates(const double T) {\n'
                '    double conc_local[NSP] = {[0 ... NSP - 1] = 1.0};\n'
                '    double fwd_rates_local[FWD_RATES];\n'
                '    double rev_rates_local[REV_RATES];\n'
                '    eval_rxn_rates(T, conc_local, fwd_rates_local,'
                ' rev_rates_local);\n'
                '}\n'
                '#elif RATES_TEST\n'
                '__global__ void k_eval_rxn_rates(const int NUM, const '
                'double T, const double* conc, double* fwd_rates, double* '
                'rev_rates) {\n'
                '    double conc_local[NSP];\n'
                '    double fwd_rates_local[FWD_RATES];\n'
                '    double rev_rates_local[REV_RATES];\n'
                '    //copy in\n'
                '    for (int i = 0; i < NSP; i++) {\n'
                '        conc_local[i] = conc[i * NUM + threadIdx.x + '
                'blockIdx.x * blockDim.x];\n'
                '    }\n'
                '    eval_rxn_rates(T, conc_local, fwd_rates_local, '
                'rev_rates_local);\n'
                '    //copy back\n'
                '    for (int i = 0; i < FWD_RATES; i++) {\n'
                '        fwd_rates[i * NUM + threadIdx.x + blockIdx.x * '
                'blockDim.x] = fwd_rates_local[i];\n'
                '    }\n'
                '    for (int i = 0; i < REV_RATES; i++) {\n'
                '        rev_rates[i * NUM + threadIdx.x + blockIdx.x * '
                'blockDim.x] = rev_rates_local[i];\n'
                '    }\n'
                '}\n'
                '#endif\n'
                )
        else:
            file.write(
                '#ifdef PROFILER\n'
                '__global__ void k_eval_rxn_rates(const double T) {\n'
                '    double conc_local[NSP] = {[0 ... NSP - 1] = 1.0};\n'
                '    double rates_local[RATES];\n'
                '    eval_rxn_rates(T, conc_local, rates_local);\n'
                '}\n'
                '#elif RATES_TEST\n'
                '__global__ void k_eval_rxn_rates(const int NUM, '
                'const double T, const double* conc, double* rates) {\n'
                '    double conc_local[NSP];\n'
                '    double rates_local[RATES];\n'
                '    //copy in\n'
                '    for (int i = 0; i < NSP; i++) {\n'
                '        conc_local[i] = conc[i * NUM + threadIdx.x + '
                'blockIdx.x * blockDim.x];\n'
                '    }\n'
                '    eval_rxn_rates(T, conc_local, rates_local);\n'
                '    //copy back\n'
                '    for (int i = 0; i < RATES; i++) {\n'
                '        rates[i * NUM + threadIdx.x + blockIdx.x * '
                'blockDim.x] = rates_local[i];\n'
                '    }\n'
                '}\n'
                '#endif\n'
                )
        if have_pdep_rxns:
            file.write(
                '#ifdef PROFILER\n'
                '__global__ void k_get_rxn_pres_mod(const double T, '
                'const double P) {\n'
                '    double conc_local[NSP] = {[0 ... NSP - 1] = 1.0};\n'
                '    double pres_mod_local[PRES_MOD_RATES];\n'
                '    get_rxn_pres_mod(T, P, conc_local, pres_mod_local);\n'
                '}\n'
                '#elif RATES_TEST\n'
                '__global__ void k_get_rxn_pres_mod(const int NUM, '
                'const double T, const double P, double* conc, '
                'double* pres_mod) {'
                '    double conc_local[NSP];\n'
                '    double pres_mod_local[PRES_MOD_RATES];\n'
                '    //copy in\n'
                '    for (int i = 0; i < NSP; i++) {\n'
                '        conc_local[i] = conc[i * NUM + threadIdx.x + '
                'blockIdx.x * blockDim.x];\n'
                '    }\n'
                '    get_rxn_pres_mod(T, P, conc_local, pres_mod_local);\n'
                '    //copy back\n'
                '    for (int i = 0; i < PRES_MOD_RATES; i++) {\n'
                '        pres_mod[i * NUM + threadIdx.x + blockIdx.x * '
                'blockDim.x] = pres_mod_local[i];\n'
                '    }\n'
                '}\n'
                '#endif\n'
                )
        file.write('#ifdef PROFILER\n'
                   '__global__ void k_eval_spec_rates() {\n'
                   )
        if have_rev_rxns:
            file.write('    double fwd_rates_local[FWD_RATES] = '
                       '{[0 ... FWD_RATES - 1] = 1.0};\n'
                       '    double rev_rates_local[REV_RATES] = '
                       '{[0 ... REV_RATES - 1] = 1.0};\n'
                       )
        else:
            file.write('    double rates_local[RATES] = '
                       '{[0 ... RATES - 1] = 1.0};\n'
                       )
        if have_pdep_rxns:
            file.write('    double pres_mod_local[PRES_MOD_RATES] = '
                       '{[0 ... PRES_MOD_RATES - 1] = 1.0};\n'
                       if have_pdep_rxns else ''
                       )
        file.write(
                   '    double dy_local[NN];\n'
                   '    eval_spec_rates('
                   )
        if have_rev_rxns:
            file.write('fwd_rates_local, rev_rates_local, ')
        else:
            file.write('rates_local, ')
        if have_pdep_rxns:
            file.write('pres_mod_local, ')
        file.write('dy_local);\n')
        file.write('}\n'
                   '#elif RATES_TEST\n'
                   '__global__ void k_eval_spec_rates(const int NUM, '
                  )
        if have_rev_rxns:
            file.write('const double* fwd_rates, const double* rev_rates, ')
        else:
            file.write('const double* rates, ')
        if have_pdep_rxns:
            file.write('const double* pres_mod, ')
        file.write('double* dy) {\n')
        file.write('    //copy in\n')
        if have_rev_rxns:
            file.write('    double fwd_rates_local[FWD_RATES];\n'
                       '    double rev_rates_local[REV_RATES];\n'
                       )
        else:
            file.write('    double rates_local[RATES];\n')
        if have_pdep_rxns:
            file.write('    double pres_mod_local[PRES_MOD_RATES];\n'
                       if have_pdep_rxns else ''
                       )
        file.write('    double dy_local[NN];\n')
        if have_rev_rxns:
            file.write('    for (int i = 0; i < FWD_RATES; i++) {\n'
                       '        fwd_rates_local[i] = fwd_rates[i * NUM + '
                       'threadIdx.x + blockIdx.x * blockDim.x];\n'
                       '    }\n'
                       '    for (int i = 0; i < REV_RATES; i++) {\n'
                       '        rev_rates_local[i] = rev_rates[i * NUM + '
                       'threadIdx.x + blockIdx.x * blockDim.x];\n'
                       '    }\n'
                       )
        else:
            file.write('    for (int i = 0; i < RATES; i++) {\n'
                       '        rates_local[i] = rates[i * NUM + threadIdx.x '
                       '+ blockIdx.x * blockDim.x];\n'
                       '    }\n'
                       )
        if have_pdep_rxns:
            file.write('    for (int i = 0; i < PRES_MOD_RATES; i++) {\n'
                       '        pres_mod_local[i] = pres_mod[i * NUM + '
                       'threadIdx.x + blockIdx.x * blockDim.x];\n'
                       '    }\n'
                       )
        file.write('    eval_spec_rates(')
        if have_rev_rxns:
            file.write('fwd_rates_local, rev_rates_local, ')
        else:
            file.write('rates_local, ')
        if have_pdep_rxns:
            file.write('pres_mod_local, ')
        file.write('dy_local);\n')
        file.write(
                   '   //copy back\n'
                   '   for (int i = 0; i < NN; i++) {\n'
                   '        dy[i * NUM + threadIdx.x + blockIdx.x * '
                   'blockDim.x] = dy_local[i];\n'
                   '    }\n'
                   '}\n'
                   )
        file.write('#endif\n')
        file.write(
            '#ifdef PROFILER\n'
            '__global__ void k_eval_dy(const double T, const double P) {\n'
            '    double y_local[NN] = {T, [1 ... NN - 1] = 1.0 / NSP};\n'
            '    double dy_local[NN];\n'
            '    dydt(T, P, y_local, dy_local);\n'
            '}\n'
            '#elif RATES_TEST\n'
            '__global__ void k_eval_dy(const int NUM, const double T, '
            'const double P, const double* y, double* dy) {\n'
            '    double y_local[NN];\n'
            '    double dy_local[NN];\n'
            '    //copy in\n'
            '    for (int i = 0; i < NN; i++) {\n'
            '        y_local[i] = y[i * NUM + threadIdx.x + '
            'blockIdx.x * blockDim.x];\n'
            '    }\n'
            '    dydt(T, P, y_local, dy_local);\n'
            '    //copy back\n'
            '    for (int i = 0; i < NN; i++) {\n'
            '        dy[i * NUM + threadIdx.x + blockIdx.x * blockDim.x] = '
            'dy_local[i];\n'
            '    }\n'
            '}\n'
            '#endif\n'
            )
        file.write(
            '#ifdef PROFILER\n'
            '__global__ void k_eval_jacob(const double T, const double P) {\n'
            '    double y_local[NN] = {T, [1 ... NN - 1] = 1.0 / NSP};\n'
            '    double jac_local[NN * NN];\n'
            '    eval_jacob(0, P, y_local, jac_local);\n'
            '}\n'
            '#elif RATES_TEST\n'
            '__global__ void k_eval_jacob(const int NUM, const double t, '
            'const double P, double* y, double* jac) {\n'
            '    double y_local[NN];\n'
            '    double jac_local[NN * NN] = {0.0};\n'
            '    //copy in\n'
            '    for (int i = 0; i < NN; i++) {\n'
            '        y_local[i] = y[i * NUM + threadIdx.x + '
            'blockIdx.x * blockDim.x];\n'
            '    }\n'
            '    eval_jacob(0, P, y_local, jac_local);\n'
            '    for (int i = 0; i < NN * NN; i++) {\n'
            '        jac[i * NUM + threadIdx.x + blockIdx.x * blockDim.x] = '
            'jac_local[i];\n'
            '    }\n'
            '}\n'
            '#endif\n'
            )
        file.write('\n')
    else:
        if have_rev_rxns:
            file.write(
                '__global__ void k_eval_rxn_rates(const double T) {\n'
                '    eval_rxn_rates(T, memory_pointers.conc, '
                'memory_pointers.fwd_rates, memory_pointers.rev_rates);\n'
                '}\n'
                )
        else:
            file.write(
                '__global__ void k_eval_rxn_rates(const double T) {\n'
                '    eval_rxn_rates(T, memory_pointers.conc, '
                'memory_pointers.rates);\n'
                '}\n'
                )
        if have_pdep_rxns:
            file.write(
                '__global__ void k_get_rxn_pres_mod(const double T, '
                'const double P) {\n'
                '    get_rxn_pres_mod(T, P, memory_pointers.conc, '
                'memory_pointers.pres_mod);\n'
                '}\n'
                )
        file.write(
            '__global__ void k_eval_spec_rates() {\n    eval_spec_rates('
            '{}'.format('memory_pointers.fwd_rates, '
                        'memory_pointers.rev_rates,' if have_rev_rxns else
                        'memory_pointers.rates,'
                        )
            '{} memory_pointers.dy);\n'.format(' memory_pointers.pres_mod,'
                                               if have_pdep_rxns else ''
                                               ) +
            '}\n'
            )
        file.write(
            '__global__ void k_eval_dy(const double T, const double P) {\n'
            '    dydt(T, P, memory_pointers.y, memory_pointers.dy);\n'
            '}\n'
            )
        file.write(
            '__global__ void k_eval_jacob(const double t, const double P) {\n'
            '    eval_jacob(t, P, memory_pointers.y, memory_pointers.jac);\n'
            '}\n'
            )
        file.write('\n')


def __write_c_rate_evaluator(file, have_rev_rxns, have_pdep_rxns,
                             T, P, Pretty_P
                             ):
    file.write('    fprintf(fp, "{}K, {} atm\\n");\n'.format(T, Pretty_P) +
               '    y_host[0] = {};\n'.format(T) +
               '    get_concentrations({}, y_host, conc_host);\n'.format(P)
               )
    if have_rev_rxns:
        file.write(
            '    eval_rxn_rates({}, '.format(T) +
            'conc_host, fwd_rates_host, rev_rates_host);\n'
            )
    else:
        file.write(
            '    eval_rxn_rates({}, conc_host, rates_host);\n'.format(T))
    if have_pdep_rxns:
        file.write('    get_rxn_pres_mod ({}, {}, '.format(T, P) +
                   'conc_host, pres_mod_host);\n'
                   )
    file.write(
        '    eval_spec_rates ({}{}'.format('fwd_rates_host, rev_rates_host,'
                                           if have_rev_rxns else
                                           'rates_host,', ' pres_mod_host,'
                                           if have_pdep_rxns else ''
                                           ) +
        ' spec_rates_host);\n'
        '    dydt({}, {}, y_host, dy_host);\n'.format(T, P) +
        '    write_rates(fp,{}{}'.format(' fwd_rates_host, rev_rates_host,'
                                         if have_rev_rxns else
                                         ' rates_host,', ' pres_mod_host,'
                                         if have_pdep_rxns else ''
                                         ) +
        ' spec_rates_host, dy_host);\n'
        '    eval_jacob(0, {}, y_host, jacob_host);\n'.format(P) +
        '    write_jacob(fp, jacob_host);\n'
        )


def __write_cuda_rate_evaluator(file, have_rev_rxns, have_pdep_rxns,
                                T, P, Pretty_P
                                ):
    descriptor = 'd_' if not CUDAParams.is_global() else 'host_memory->'
    file.write('    fprintf(fp, "{}K, {} atm\\n");\n'.format(T, Pretty_P) +
               '    y_host[0] = {};\n'.format(T) +
               '    get_concentrations({}, y_host, conc_host);\n'.format(P))
    file.write('#ifdef CONV\n' +
               '    double rho = getDensity({}, {}, Xi);\n'.format(T, P) +
               '#endif\n'
               '    for (int i = 0; i < padded; ++i) {\n'
               '        for (int j = 0; j < NSP; ++j) {\n'
               '            conc_host_full[i + j * padded] = conc_host[j];\n'
               '        }\n' +
               '        y_host_full[i] = {};\n'.format(T) +
               '#ifdef CONP\n' +
               '        pres_host_full[i] = {};\n'.format(P) +
               '#elif CONV\n'
               '        rho_host_full[i] = rho;\n'
               '#endif\n'
               '    }\n'
               )
    file.write(
        '    cudaErrorCheck(cudaMemcpy({}conc, '.format(descriptor) +
        'conc_host_full, padded * NSP * sizeof(double), '
        'cudaMemcpyHostToDevice));\n'
        )
    if have_rev_rxns:
        file.write(
            '#ifdef PROFILER\n'
            '    cuProfilerStart();\n'
            '    k_eval_rxn_rates<<<grid_size, block_size>>>'
            '({});\n'.format(T) +
            '    cuProfilerStop();\n'
            '#elif RATES_TEST\n'
            '    k_eval_rxn_rates<<<grid_size, block_size>>>(padded, '
            '{}, '.format(T) +
            '{});\n'.format(
                '{0}conc, {0}fwd_rates, {0}rev_rates'.format(descriptor)
                if not CUDAParams.is_global() else ''
                )
            )
        file.write(
            '    cudaErrorCheck(cudaMemcpy(fwd_rates_host_full, '
            '{}fwd_rates, padded * FWD_RATES '.format(descriptor) +
            '* sizeof(double), cudaMemcpyDeviceToHost));\n'
            '    cudaErrorCheck(cudaMemcpy(rev_rates_host_full, '
            '{}rev_rates, padded * REV_RATES '.format(descriptor) +
            '* sizeof(double), cudaMemcpyDeviceToHost));\n'
            )
        file.write(
            '    for (int j = 0; j < FWD_RATES; ++j) {\n'
            '            fwd_rates_host[j] = '
            'fwd_rates_host_full[j * padded];\n'
            '    }\n'
            )
        file.write(
            '    for (int j = 0; j < REV_RATES; ++j) {\n'
            '        rev_rates_host[j] = rev_rates_host_full[j * padded];\n'
            '    }\n'
            '#endif\n'
            )
    else:
        file.write(
            '#ifdef PROFILER\n'
            '    cuProfilerStart();\n'
            '    k_eval_rxn_rates<<<grid_size, block_size>>>'
            '({});\n'.format(T) +
            '    cuProfilerStop();\n'
            '#elif RATES_TEST\n'
            '    k_eval_rxn_rates<<<grid_size, block_size>>>(padded, '
            '{}{});\n'.format(T, ', {0}conc, {0}rates'.format(descriptor)
                              if not CUDAParams.is_global() else ''
                              )
            )
        file.write(
            '    cudaErrorCheck(cudaMemcpy(rates_host_full, '
            '{}rates, padded * RATES '.format(descriptor) +
            '* sizeof(double), cudaMemcpyDeviceToHost));\n'
            )

        file.write(
            '    for (int j = 0; j < RATES; ++j) {\n'
            '        rates_host[j] = rates_host_full[j * padded];\n'
            '    }\n'
            '#endif\n'
            )
    if have_pdep_rxns:
        file.write(
            '#ifdef PROFILER\n'
            '    cuProfilerStart();\n'
            '    k_get_rxn_pres_mod<<<grid_size, block_size>>>('
            '{}, {});\n'.format(T, P) +
            '    cuProfilerStop();\n'
            '#elif RATES_TEST\n'
            '    k_get_rxn_pres_mod<<<grid_size, block_size>>>(padded, '
            '{}, {}{});\n'.format(T, P,
                                  ', {0}conc, {0}pres_mod'.format(descriptor)
                                  if not CUDAParams.is_global() else ''
                                  )
            )
        file.write(
            '    cudaErrorCheck(cudaMemcpy(pres_mod_host_full, '
            '{}pres_mod, padded * PRES_MOD_RATES '.format(descriptor) +
            '* sizeof(double), cudaMemcpyDeviceToHost));\n'
            )
        file.write(
            '    for (int j = 0; j < PRES_MOD_RATES; ++j) {\n'
            '        pres_mod_host[j] = pres_mod_host_full[j * padded];\n'
            '    }\n'
            '#endif\n'
            )

    if have_rev_rxns:
        file.write(
            '#ifdef PROFILER\n'
            '    cuProfilerStart();\n'
            '    k_eval_spec_rates<<<grid_size, block_size>>>();\n'
            '    cuProfilerStop();\n'
            '#elif RATES_TEST\n'
            '    k_eval_spec_rates<<<grid_size, block_size>>>(padded, '
            '{});\n'.format(('{0}fwd_rates, {0}rev_rates' + (', {0}pres_mod'
                            if have_pdep_rxns else '') +
                            ', {0}spec_rates').format(descriptor)
                            if not CUDAParams.is_global() else ''
                            )
            )
    else:
        file.write(
            '#ifdef PROFILER\n'
            '    cuProfilerStart();\n'
            '    k_eval_spec_rates<<<grid_size, block_size>>>({});\n'
            '    cuProfilerStop();\n'
            '#elif RATES_TEST\n'
            '    k_eval_spec_rates<<<grid_size, block_size>>>(padded, '
            '{});\n'.format(('{0}rates' + (', {0}pres_mod' if have_pdep_rxns
                            else '') + ', {0}spec_rates').format(descriptor)
                            )
            )
    file.write(
        '    cudaErrorCheck(cudaMemcpy(spec_rates_host_full, '
        '{}spec_rates, padded * NSP * sizeof(double), '.format(descriptor) +
        'cudaMemcpyDeviceToHost));\n'
        )
    file.write(
        '    for (int j = 0; j < NSP; ++j) {\n'
        '        spec_rates_host[j] = spec_rates_host_full[j * padded];\n'
        '    }\n'
        '#endif\n'
        )
    file.write(
        '#ifdef PROFILER\n'
        '    cuProfilerStart();\n'
        '    k_eval_dy<<<grid_size, block_size>>>({}, {});\n'.format(T, P) +
        '    cuProfilerStop();\n'
        '#elif RATES_TEST\n'
        '    cudaErrorCheck(cudaMemcpy({}y, y_host_full'.format(descriptor) +
        ', padded * NN * sizeof(double), cudaMemcpyHostToDevice));\n'
        '    k_eval_dy<<<grid_size, block_size>>>(padded, '
        '{}, {}, {});\n'.format(T, P,
                                ('{0}y' + (', {0}dy' if have_pdep_rxns
                                 else '')
                                 ).format(descriptor)
                                 if not CUDAParams.is_global() else ''
                                )
        )
    file.write('    cudaErrorCheck(cudaMemcpy(dy_host_full, '
               '{}dy, '.format(descriptor) +
               'padded * NN * sizeof(double), cudaMemcpyDeviceToHost));\n'
               )
    file.write(
               '    for (int j = 0; j < NN; ++j) {\n'
               '        dy_host[j] = dy_host_full[j * padded];\n'
               '    }\n'
               '#endif\n'
               )
    file.write(
        '#ifdef RATES_TEST\n'
        '    write_rates(fp,{}{}'.format(' fwd_rates_host, rev_rates_host,'
                                         if have_rev_rxns else ' rates_host,',
                                         ' pres_mod_host,' if have_pdep_rxns
                                         else ''
                                         ) +
        ' spec_rates_host, dy_host);\n'
        '#endif\n'
        )
    file.write(
        '#ifdef CONP\n'
        '    cudaErrorCheck(cudaMemcpy({}pres, '.format(descriptor) +
        'pres_host_full, padded * sizeof(double), cudaMemcpyHostToDevice));\n'
        '#elif CONV\n'
        '    cudaErrorCheck(cudaMemcpy({}rho, '.format(descriptor) +
        'rho_host_full, padded * sizeof(double), cudaMemcpyHostToDevice));\n'
        '#endif\n'
        )
    file.write(
        '#ifdef PROFILER\n'
        '    cuProfilerStart();\n'
        '    k_eval_jacob<<<grid_size, block_size>>>({}, {});\n'.format(T,P) +
        '    cuProfilerStop();\n'
        '#elif RATES_TEST\n'
        '    k_eval_jacob<<<grid_size, block_size>>>(padded, 0, '
        '{}{});\n'.format(P, ', {0}y, {0}jac'.format(descriptor)
                          if not CUDAParams.is_global() else ''
                          ) +
        '    cudaErrorCheck(cudaMemcpy(jacob_host_full, '
        '{}jac, padded * NN * NN * sizeof(double), '.format(descriptor) +
        'cudaMemcpyDeviceToHost));\n'
        '    for (int j = 0; j < NN * NN; ++j) {\n'
        '        jacob_host[j] = jacob_host_full[j * padded];\n'
        '    }\n'
        '    write_jacob(fp, jacob_host);\n'
        '#endif\n'
        )


def write_mechanism_initializers(path, lang, specs, reacs):
    """
    """
    if lang in ['matlab', 'fortran']:
        raise NotImplementedError

    # The mechanism header defines a number of useful preprocessor defines,
    # as well as defining method stubs for setting initial conditions.
    with open(path + 'mechanism.h', 'w') as file:

        file.write('#ifndef MECHANISM_H\n'
                   '#define MECHANISM_H\n\n')

        #convience: write species indexes
        file.write('/* Species Indexes\n')
        file.write('\n'.join('{}  {}'.format(i, spec.name)
                   for i, spec in enumerate(specs))
                   )
        file.write('*/\n\n')

        #defines
        file.write("//Number of species\n")
        file.write('#define NSP {}\n'.format(len(specs)))
        file.write("//Number of variables. NN = NSP + 1 (temperature)\n")
        file.write('#define NN {}\n'.format(len(specs) + 1))
        file.write('//Number of forward reactions\n')
        file.write('#define FWD_RATES {}\n'.format(len(reacs)))
        file.write('//Number of reversible reactions\n')
        file.write('#define REV_RATES {}\n'.format(len([reac for reac in
                                                   reacs if reac.rev])
                                                   )
                   )
        file.write('//Number of reactions with pressure modified rates\n')
        file.write('#define PRES_MOD_RATES '
                   '{}\n\n'.format(len([reac for reac in reacs
                                   if reac.pdep or reac.thd])
                                   )
                   )

        #to work with both c and cuda
        file.write('#ifdef __cplusplus\n'
                   'extern "C" {\n'
                   '#endif\n')

        file.write('    //Must be implemented by user on a per mechanism '
                   'basis in mechanism.c\n'
                   '    #ifdef CONP\n'
                   '    void set_same_initial_conditions(int NUM, double* '
                   'y_host, double* pres_host);\n'
                   '    #elif CONV\n'
                   '    void set_same_initial_conditions(int NUM, double* '
                   'y_host, double* pres_host, double* rho_host);\n'
                   '    #endif\n'
                   '    #ifdef RATES_TEST\n'
                   '    void write_jacobian_and_rates_output();\n'
                   '    #endif\n'
                   )

        #close previous extern
        file.write('#ifdef __cplusplus\n'
                   '}\n'
                   '#endif\n\n')

        file.write('#endif\n\n')

    #now the mechanism file
    with open(path + 'mechanism.c', 'w') as file:
        file.write('#include <stdio.h>\n'
                   '#include "mass_mole.h"\n'
                   '#include "mechanism.h"\n'
                   '#ifdef RATES_TEST\n'
                   '    #include "rates.h"\n'
                   '    #include "derivs.h"\n'
                   '#endif\n')
        file.write(
            '#ifdef CONP\n'
            '    void set_same_initial_conditions(int NUM, double* y_host, '
            'double* pres_host) {\n'
            '#elif CONV\n'
            '    void set_same_initial_conditions(int NUM, double* y_host, '
            'double* pres_host, double* rho_host) {\n'
            '#endif\n'
            '        double Xi [NSP] = {0.0};\n'
            '        //set initial mole fractions here\n\n'
            '        //Normalize mole fractions to sum to one\n'
            '        double Xsum = 0.0;\n'
            '        for (int j = 0; j < NSP; ++ j) {\n'
            '            Xsum += Xi[j];\n'
            '        }\n'
            '        if (Xsum == 0.0) {\n'
            '            printf("Use of the set initial conditions function '
            'requires user implementation!");\n'
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
            '        //load temperature and mass fractions for all '
            'threads (cells)\n'
            '        for (int i = 0; i < NUM; ++i) {\n'
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
            '        for (int i = 0; i < NUM; ++i) {\n'
            '            #ifdef CONV\n'
            '            rho_host[i] = rho;\n'
            '            #endif\n'
            '            pres_host[i] = pres;\n'
            '        }\n'
            '    }\n\n'
            )
        file.write('#ifdef RATES_TEST\n')
        #write utility function that finds concentrations at a given state
        file.write('    void get_concentrations(double pres, const double* y,'
                   ' double* conc) {\n'
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
        line = ('        rho = pres / '
                '({:.8e} * y[0] * rho);\n\n'.format(chem.RU)
                )
        file.write(line)


        # calculation of species molar concentrations
        file.write('        // species molar concentrations\n')
        # loop through species
        for sp in specs:
            isp = specs.index(sp)
            line = '    conc[{}] = rho * y[{}] / '.format(isp, isp + 1)
            line += '{}'.format(sp.mw) + utils.line_end[lang]
            file.write(line)

        file.write('    }\n\n')


        #convience method to write rates to file
        file.write('    void write_rates(FILE* fp, const double* fwd_rates, '
                   'const double* rev_rates, const double* pres_mod, '
                   'const double* sp_rates, const double* dy) {\n'
                   '        fprintf(fp, "Forward Rates\\n");\n'
                   '        for(int i = 0; i < FWD_RATES; ++i) {\n'
                   '            fprintf(fp, "%.15le\\n", fwd_rates[i]);\n'
                   '        }\n'
                   '        fprintf(fp, "Rev Rates\\n");\n'
                   '        for(int i = 0; i < REV_RATES; ++i) {\n'
                   '            fprintf(fp, "%.15le\\n", rev_rates[i]);\n'
                   '        }\n'
                   '        fprintf(fp, "Pres Mod Rates\\n");\n'
                   '        for(int i = 0; i < PRES_MOD_RATES; i++) {\n'
                   '            fprintf(fp, "%.15le\\n", pres_mod[i]);\n'
                   '        }\n'
                   '        fprintf(fp, "Spec Rates\\n");\n'
                   '        for(int i = 0; i < NSP; i++) {\n'
                   '             fprintf(fp, "%.15le\\n", sp_rates[i]);\n'
                   '        }\n'
                   '        fprintf(fp, "dy\\n");\n'
                   '        for(int i = 0; i < NN; i++) {\n'
                   '            fprintf(fp, "%.15le\\n", dy[i]);\n'
                   '        }\n'
                   '    }\n\n'
                  )

        file.write('    void write_jacob(FILE* fp, const double* jacob) {\n'
                   '        fprintf(fp, "Jacob\\n");\n'
                   '        for (int i = 0; i < NN * NN; ++i) {\n'
                   '            fprintf(fp, "%.15le\\n", jacob[i]);\n'
                   '        }\n'
                   '    }\n\n'
                   )


        file.write(
            '    void write_jacobian_and_rates_output() {\n'
            '        //set mass fractions to unity to turn on all reactions\n'
            '        double Yi[NN];\n'
            '        double sum = 0;'
            '        for (int i = 1; i < NN; ++i) {\n'
            '            Yi[i] = (1.0 + i - 1);\n'
            '            sum += Yi[i];\n'
            '        }\n'
            '        //Normalize\n'
            '        for (int i = 1; i < NN; ++i) {\n'
            '            Yi[i] /= sum;\n'
            '        }\n'
            '        double conc[NSP];\n'
            '        double fwd_rates[FWD_RATES];\n'
            '        double rev_rates[REV_RATES];\n'
            '        double pres_mod[PRES_MOD_RATES];\n'
            '        double sp_rates[NSP];\n'
            '        double dy[NN];\n'
            '        double jacob[NN * NN];\n'
            '        //evaluate and write rates for various conditions\n'
            '        FILE* fp = fopen ("rates_data.txt", "w");\n'
            '        fprintf(fp, "800K, 1 atm\\n");\n'
            '        Yi[0] = 800;\n'
            '        get_concentrations(1.01325e6, Yi, conc);\n'
            '        eval_rxn_rates(800, conc, fwd_rates, rev_rates);\n'
            '        get_rxn_pres_mod (800, 1.01325e6, conc, pres_mod);\n'
            '        eval_spec_rates (fwd_rates, rev_rates, pres_mod, '
            'sp_rates);\n'
            '        dydt(800, 1.01325e6, Yi, dy);\n'
            '        write_rates(fp, fwd_rates, rev_rates, pres_mod, '
            'sp_rates, dy);\n'
            '        eval_jacob(0, 1.01325e6, Yi, jacob);\n'
            '        write_jacob(fp, jacob);\n'
            '        fprintf(fp, "1600K, 1 atm\\n");\n'
            '        Yi[0] = 1600;\n'
            '        get_concentrations(1.01325e6, Yi, conc);\n'
            '        eval_rxn_rates(1600, conc, fwd_rates, rev_rates);\n'
            '        get_rxn_pres_mod (1600, 1.01325e6, conc, pres_mod);\n'
            '        eval_spec_rates (fwd_rates, rev_rates, pres_mod, '
            'sp_rates);\n'
            '        dydt(1600, 1.01325e6, Yi, dy);\n'
            '        write_rates(fp, fwd_rates, rev_rates, pres_mod, '
            'sp_rates, dy);\n'
            '        eval_jacob(0, 1.01325e6, Yi, jacob);\n'
            '        write_jacob(fp, jacob);\n'
            '        fprintf(fp, "800K, 10 atm\\n");\n'
            '        Yi[0] = 800;\n'
            '        get_concentrations(1.01325e7, Yi, conc);\n'
            '        eval_rxn_rates(800, conc, fwd_rates, rev_rates);\n'
            '        get_rxn_pres_mod (800, 1.01325e7, conc, pres_mod);\n'
            '        eval_spec_rates (fwd_rates, rev_rates, pres_mod, '
            'sp_rates);\n'
            '        dydt(800, 1.01325e7, Yi, dy);\n'
            '        write_rates(fp, fwd_rates, rev_rates, pres_mod, '
            'sp_rates, dy);\n'
            '        eval_jacob(0, 1.01325e7, Yi, jacob);\n'
            '        write_jacob(fp, jacob);\n'
            '        fclose(fp);\n'
            '    }\n'
            )
        file.write('#endif')


def write_header(path, lang):
    """Writes minimal header file used by all other source files.

    :param path:
        Path where files are being written.
    :param lang: {'c', 'cuda', 'fortran', 'matlab'}
        Language type.
    """
    if lang in ['matlab', 'fortran']:
        raise NotImplementedError

    with open(path + 'header.h', 'w') as file:
        file.write('#include <stdlib.h>\n'
                   '#include <math.h>\n'
                   '#include <float.h>\n'
                   '\n'
                   '/** Constant pressure or volume. */\n'
                   '#define CONP\n'
                   '//#define CONV\n'
                   '\n'
                   '/** Include mechanism header to get NSP and NN **/\n'
                   '#ifdef __cplusplus\n'
                   ' #include "mechanism.cuh"\n'
                   '#else\n'
                   ' #include "mechanism.h"\n'
                   '#endif\n'
                  )
