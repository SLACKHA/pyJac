# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import sys
import subprocess
from argparse import ArgumentParser

# More Python 2 compatibility
if sys.version_info.major == 3:
    from itertools import zip
elif sys.version_info.major == 2:
    from itertools import izip as zip

# Related modules
import numpy as np

try:
    import cantera as ct
    from cantera import ck2cti
except ImportError:
    print('Error: Cantera must be installed.')
    raise

# Local imports
import utils
import create_jacobian

# Compiler based on language
cmd_compile = dict(c='gcc',
                   cuda='nvcc',
                   fortran='gfortran'
                   )

# Flags based on language
flags = dict(c='-std=c99',
             cuda=['-arch=sm_20', '-g', '-G', '-O0',
                   '-I/usr/local/cuda/include/',
                   '-I/usr/local/cuda/samples/common/inc/',
                   '-dc'],
             fortran='')

libs = dict(c='-lm',
            cuda='-arch=sm_20',
            fortran='')


class ReactorConstPres(object):

    def __init__(self, gas):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorConstPres object.
        self.gas = gas
        self.P = gas.P

    def __call__(self, t=None, y=None):
        """the ODE function, y' = f(t,y) """

        # State vector is [T, Y_1, Y_2, ... Y_K]
        # Only set state if something is input
        if y is not None:
            self.gas.set_unnormalized_mass_fractions(y[1:])
            self.gas.TP = y[0], self.P
        rho = self.gas.density

        wdot = self.gas.net_production_rates
        dTdt = - (np.dot(self.gas.partial_molar_enthalpies, wdot) /
                  (rho * self.gas.cp)
                  )
        dYdt = wdot * self.gas.molecular_weights / rho

        return np.hstack((dTdt, dYdt))


class ReactorConstVol(object):

    def __init__(self, gas):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorConstVol object.
        self.gas = gas
        self.density = gas.density

    def __call__(self, t=None, y=None):
        """the ODE function, y' = f(t,y) """

        # State vector is [T, Y_1, Y_2, ... Y_K]
        # Only set state if something is input
        if y is not None:
            self.gas.set_unnormalized_mass_fractions(y[1:])
            self.gas.TD = y[0], self.density

        wdot = self.gas.net_production_rates
        dTdt = - (np.dot(self.gas.partial_molar_int_energies, wdot) /
                  (self.density * self.gas.cv)
                  )
        dYdt = wdot * self.gas.molecular_weights / self.density

        return np.hstack((dTdt, dYdt))


def convert_mech(mech_filename, therm_filename=None):
    """Convert a mechanism and return a string with the filename.

    Convert a CHEMKIN format mechanism to the Cantera CTI format using
    the Cantera built-in script `ck2cti`.

    :param mech_filename:
        Filename of the input CHEMKIN format mechanism. The converted
        CTI file will have the same name, but with `.cti` extension.
    :param thermo_filename:
        Filename of the thermodynamic database. Optional if the
        thermodynamic database is present in the mechanism input.
    """

    arg = ['--input=' + mech_filename]
    if therm_filename is not None:
        arg.append('--thermo=' + therm_filename)

    # Convert the mechanism
    ck2cti.main(arg)
    mech_filename = mech_filename[:-4] + '.cti'
    print('Mechanism conversion successful, written to '
          '{}'.format(mech_filename)
          )
    return mech_filename


def __write_header_include(file, lang):
    file.write(
        '#include <stdlib.h>\n'
        '#include <stdio.h>\n'
        '\n'
    )

    file_list = ['mechanism', 'chem_utils', 'rates', 'dydt', 'jacob']
    file.write('\n'.join(['#include "header.h"'] +
                         ['#include "{}.{}"'.format(
                             the_file, 'h' if lang == 'c' else 'cuh')
                          for the_file in file_list]) + '\n')

    if lang == 'cuda':
        file.write('#include <cuda.h>\n'
                   '#include <cuda_runtime.h>\n'
                   '#include <helper_cuda.h>\n'
                   '#include "launch_bounds.cuh"\n'
                   '#include "gpu_macros.cuh"\n'
                   '#include "gpu_memory.cuh"\n\n')


def __write_output_methods(file):
    file.write(
        'void write_conc (FILE* fp, const double* conc) {\n'
        '  fprintf(fp, "%d\\n", NSP);\n'
        '  for(int i = 0; i < NSP; ++i) {\n'
        '    fprintf(fp, "%.15e\\n", conc[i]);\n'
        '  }\n'
        '}\n'
        '\n'
        'void write_rates (FILE* fp, const double* fwd_rates, '
        'const double* rev_rates,\n'
        '                  const double* pres_mod, '
        'const double* sp_rates, const double* dy) {\n'
        '\n'
        '  fprintf(fp, "%d\\n", FWD_RATES);\n'
        '  for(int i = 0; i < FWD_RATES; ++i) {\n'
        '      fprintf(fp, "%.15e\\n", fwd_rates[i]);\n'
        '  }\n'
        '\n'
        '  fprintf(fp, "%d\\n", REV_RATES);\n'
        '  for(int i = 0; i < REV_RATES; ++i) {\n'
        '      fprintf(fp, "%.15e\\n", rev_rates[i]);\n'
        '  }\n'
        '\n'
        '  fprintf(fp, "%d\\n", PRES_MOD_RATES);\n'
        '  for(int i = 0; i < PRES_MOD_RATES; i++) {\n'
        '      fprintf(fp, "%.15e\\n", pres_mod[i]);\n'
        '  }\n'
        '\n'
        '  fprintf(fp, "%d\\n", NSP);\n'
        '  for(int i = 0; i < NSP; i++) {\n'
        '       fprintf(fp, "%.15e\\n", sp_rates[i]);\n'
        '  }\n'
        '\n'
        '  fprintf(fp, "%d\\n", NN);\n'
        '  for(int i = 0; i < NN; i++) {\n'
        '      fprintf(fp, "%.15e\\n", dy[i]);\n'
        '  }\n'
        '\n'
        '  return;\n'
        '}\n'
        '\n'
        '\n'
        'void write_jacob (FILE* fp, const double* jacob) {\n'
        '\n'
        '  fprintf(fp, "%d\\n", NN * NN);\n'
        '  for (int i = 0; i < NN * NN; ++i) {\n'
        '      fprintf(fp, "%.15e\\n", jacob[i]);\n'
        '  }\n'
        '\n'
        '  return;\n'
        '}\n'
        '\n'
        '\n'
    )


def __write_condition_reader(file):
    file.write(
        '  int buff_size = 1024;\n'
        '  char buffer [buff_size];\n'
        '  double data[NN + 1];\n'
        '  for (int i = 0; i < NN + 1; ++i) {\n'
        '    if (fgets (buffer, buff_size, fp) != NULL) {\n'
        '      data[i] = strtod(buffer, NULL);\n'
        '    }\n'
        '  }\n'
        '  fclose (fp);\n'
        '\n'
        '  y[0] = data[0];\n'
        '  pres = data[1];\n'
        '  for (int i = 0; i < NSP; ++i) {\n'
        '    y[i + 1] = data[i + 2];\n'
        '  }\n')


def __write_kernels(file):
    file.write(
        '  __global__\n'
        'void k_eval_conc(const double* y, double pres, double* conc) {\n'
        '   double rho, mw_avg;\n'
        '   eval_conc(y[0], pres, &y[1], &mw_avg, &rho, conc);\n'
        '}\n\n'
    )
    file.write(
        '  __global__\n'
        'void k_eval_rxn_rates(const double* y, double pres, '
        'const double* conc, double* fwd_rates,'
        ' double* rev_rates) {\n'
        '   eval_rxn_rates(y[0], pres, conc, fwd_rates, rev_rates);\n'
        '}\n\n'
    )
    file.write(
        '  __global__\n'
        'void k_get_rxn_pres_mod(const double* y, double pres,'
        ' const double* conc, double* pres_mod) {\n'
        '   get_rxn_pres_mod(y[0], pres, conc, pres_mod);\n'
        '}\n\n'
    )
    file.write(
        '  __global__\n'
        'void k_eval_spec_rates(const double* fwd_rates,'
        ' const double* rev_rates, const double* pres_mod'
        ', double* sp_rates) {\n'
        '   eval_spec_rates(fwd_rates, rev_rates, pres_mod, sp_rates);\n'
        '}\n\n'
    )
    file.write(
        '  __global__\n'
        'void k_eval_dy_dt(double pres, const double* y, double* dy) {\n'
        '   dydt(0, pres, y, dy);\n'
        '}\n\n'
    )
    file.write(
        '  __global__\n'
        'void k_eval_jacob(double pres, const double* y, double* jacob) {\n'
        '   eval_jacob(0, pres, y, jacob);\n'
        '}\n\n'
    )


def write_cuda_test(build_dir):
    with open(build_dir + os.path.sep + 'test.cu', 'w') as f:
        __write_header_include(f, 'cuda')
        f.write(
            '#ifndef SHARED_SIZE\n'
            '  #define SHARED_SIZE (0)\n'
            '#endif\n\n'
        )
        __write_output_methods(f)
        __write_kernels(f)
        f.write('int main (void) {\n'
                '\n'
                '  FILE* fp = fopen ("test/input.txt", "r");\n'
                '  double y[NN], pres;\n'
                '\n'
                )

        __write_condition_reader(f)

        f.write(
            '  double conc[NSP] = {0};\n'
            '  double fwd_rates[FWD_RATES] = {0};\n'
            '  double rev_rates[REV_RATES]= {0};\n'
            '  double pres_mod[PRES_MOD_RATES] = {0};\n'
            '  double spec_rates[NSP] = {0};\n'
            '  double dy[NN] = {0};\n'
            '  double jacob[NN * NN] = {0};\n'
            '\n'
            '  fp = fopen ("test/output.txt", "w");\n'
            '\n'
            '  cudaErrorCheck(cudaSetDevice(0));\n'
            '  cudaErrorCheck(cudaDeviceSetCacheConfig('
            'cudaFuncCachePreferL1));\n' +
            '\n'.join(['  double *{} = 0;'.format(x) for x in
                       ['d_y', 'd_conc', 'd_fwd_rates', 'd_rev_rates',
                        'd_pres_mod_rates', 'd_spec_rates', 'd_jacob',
                        'd_dy']]) + '\n'
            '  //initialize cuda variables\n'
            '  cudaErrorCheck(cudaMalloc((void**)&d_y, '
            'NN * sizeof(double)));\n'
            '  cudaErrorCheck(cudaMalloc((void**)&d_conc,'
            ' NSP * sizeof(double)));\n'
            '  cudaErrorCheck(cudaMalloc((void**)&d_fwd_rates,'
            ' FWD_RATES * sizeof(double)));\n'
            '  cudaErrorCheck(cudaMalloc((void**)&d_rev_rates,'
            ' REV_RATES * sizeof(double)));\n'
            '  cudaErrorCheck(cudaMalloc((void**)&d_pres_mod_rates,'
            ' PRES_MOD_RATES * sizeof(double)));\n'
            '  cudaErrorCheck(cudaMalloc((void**)&d_dy,'
            ' NN * sizeof(double)));\n'
            '  cudaErrorCheck(cudaMalloc((void**)&d_spec_rates,'
            ' NSP * sizeof(double)));\n'
            '  cudaErrorCheck(cudaMalloc((void**)&d_jacob,'
            ' NN * NN * sizeof(double)));\n'
            '  //copy mass fractions\n'
            '  cudaErrorCheck(cudaMemcpy(d_y, y, NN * sizeof(double),'
            ' cudaMemcpyHostToDevice));\n'
            '  k_eval_conc<<<1, 1, SHARED_SIZE>>>(d_y, pres, d_conc);\n'
            '  k_eval_rxn_rates<<<1, 1, SHARED_SIZE>>>(d_y, pres, d_conc,'
            ' d_fwd_rates, d_rev_rates);\n'
            '  k_get_rxn_pres_mod<<<1, 1, SHARED_SIZE>>>(d_y, pres, d_conc,'
            ' d_pres_mod_rates);\n'
            '  k_eval_spec_rates<<<1, 1, SHARED_SIZE>>>'
            '(d_fwd_rates, d_rev_rates, d_pres_mod_rates, d_spec_rates);\n'
            '  k_eval_dy_dt<<<1, 1, SHARED_SIZE>>>(pres, d_y, d_dy);\n'
            '  k_eval_jacob<<<1, 1, SHARED_SIZE>>>(pres, d_y, d_jacob);\n'
            '  //copy back\n'
            '  cudaErrorCheck(cudaMemcpy(conc, d_conc,'
            ' NSP * sizeof(double), cudaMemcpyDeviceToHost));\n'
            '  cudaErrorCheck(cudaMemcpy(fwd_rates, d_fwd_rates,'
            ' FWD_RATES * sizeof(double), cudaMemcpyDeviceToHost));\n'
            '  cudaErrorCheck(cudaMemcpy(rev_rates, d_rev_rates,'
            ' REV_RATES * sizeof(double), cudaMemcpyDeviceToHost));\n'
            '  cudaErrorCheck(cudaMemcpy(pres_mod, d_pres_mod_rates,'
            ' PRES_MOD_RATES * sizeof(double), cudaMemcpyDeviceToHost));\n'
            '  cudaErrorCheck(cudaMemcpy(dy, d_dy,'
            ' NN * sizeof(double), cudaMemcpyDeviceToHost));\n'
            '  cudaErrorCheck(cudaMemcpy(spec_rates, d_spec_rates,'
            ' NSP * sizeof(double), cudaMemcpyDeviceToHost));\n'
            '  cudaErrorCheck(cudaMemcpy(jacob, d_jacob,'
            ' NN * NN * sizeof(double), cudaMemcpyDeviceToHost));\n'
            '  //write\n'
            '  write_conc (fp, conc);\n'
            '  write_rates (fp, fwd_rates, rev_rates, '
            'pres_mod, spec_rates, dy);\n'
            '  write_jacob (fp, jacob);\n'
            '\n'
            '  fclose (fp);\n'
            '\n'
            '  //finally free cuda variables\n'
            '  cudaErrorCheck(cudaFree(d_y));\n'
            '  cudaErrorCheck(cudaFree(d_conc));\n'
            '  cudaErrorCheck(cudaFree(d_fwd_rates));\n'
            '  cudaErrorCheck(cudaFree(d_rev_rates));\n'
            '  cudaErrorCheck(cudaFree(d_pres_mod_rates));\n'
            '  cudaErrorCheck(cudaFree(d_dy));\n'
            '  cudaErrorCheck(cudaFree(d_spec_rates));\n'
            '  cudaErrorCheck(cudaFree(d_jacob));\n'
            '  return 0;\n'
            '}\n'

        )


def write_c_test(build_dir):
    with open(build_dir + os.path.sep + 'test.c', 'w') as f:
        __write_header_include(f, 'c')
        __write_output_methods(f)
        f.write('int main (void) {\n'
                '\n'
                '  FILE* fp = fopen ("test/input.txt", "r");\n'
                '  double y[NN], pres;\n'
                '\n'
                )

        __write_condition_reader(f)

        f.write(
            '  double mw_avg;\n'
            '  double rho;\n'
            '  double conc[NSP] = {0};\n'
            '  double fwd_rates[FWD_RATES] = {0};\n'
            '  double rev_rates[REV_RATES] = {0};\n'
            '  double pres_mod[PRES_MOD_RATES] = {0};\n'
            '  double sp_rates[NSP] = {0};\n'
            '  double dy[NN] = {0};\n'
            '  double jacob[NN * NN] = {0};\n'
            '\n'
            '  fp = fopen ("test/output.txt", "w");\n'
            '\n'
            '  eval_conc (y[0], pres, &y[1], &mw_avg, &rho, conc);\n'
            '  write_conc (fp, conc);\n'
            '\n'
            '  eval_rxn_rates (y[0], pres, conc, fwd_rates, rev_rates);\n'
            '  get_rxn_pres_mod (y[0], pres, conc, pres_mod);\n'
            '  eval_spec_rates (fwd_rates, rev_rates, pres_mod, sp_rates);\n'
            '  dydt (0.0, pres, y, dy);\n'
            '  write_rates (fp, fwd_rates, rev_rates, '
            'pres_mod, sp_rates, dy);\n'
            '\n'
            '  eval_jacob (0.0, pres, y, jacob);\n'
            '  write_jacob (fp, jacob);\n'
            '\n'
            '  fclose (fp);\n'
            '\n'
            '  return 0;\n'
            '}\n'
            '\n'
        )

    return None


def eval_jacobian(dydt, order):
    """
    """
    abs_tol = 1.e-16
    rel_tol = 1.e-8

    y = np.hstack((dydt.gas.T, dydt.gas.Y))

    if order == 2:
        x_coeffs = np.array([-1., 1.])
        y_coeffs = np.array([-0.5, 0.5])
    elif order == 4:
        x_coeffs = np.array([-2., -1., 1., 2.])
        y_coeffs = np.array([1. / 12., -2. / 3., 2. / 3., -1. / 12.])
    elif order == 6:
        x_coeffs = np.array([-3., -2., -1., 1., 2., 3.])
        y_coeffs = np.array([-1. / 60., 3. / 20., -3. / 4.,
                             3. / 4., -3. / 20., 1. / 60.
                             ])

    sqrt_rnd = np.sqrt(np.finfo(float).eps)
    err_wt = abs(y) * rel_tol + abs_tol

    r0 = (1000. * rel_tol * np.finfo(float).eps * len(y) *
          np.sqrt(np.sum(np.power(err_wt * dydt(), 2)) / len(y))
          )

    jacob = np.zeros(len(y) ** 2)
    for j, y_j in enumerate(y):
        y_temp = np.copy(y)
        r = max(sqrt_rnd * abs(y_j), r0 / err_wt[j])

        for x_c, y_c in zip(x_coeffs, y_coeffs):
            y_temp[j] = y_j + x_c * r
            jacob[np.arange(len(y)) + len(y) * j] += y_c * dydt(y=y_temp)

        jacob[np.arange(len(y)) + len(y) * j] /= r

    return jacob


def test(lang, build_dir, mech_filename, therm_filename=None):
    """
    """

    test_dir = '.' + os.path.sep + 'test'
    utils.create_dir(test_dir)

    # First check for appropriate Compiler
    try:
        subprocess.check_call(['which', cmd_compile[lang]])
    except subprocess.CalledProcessError:
        print('Error: appropriate compiler for language not found.')
        sys.exit(1)

    # generate jacobian
    create_jacobian.create_jacobian(
        lang, mech_filename, therm_filename,
        optimize_cache=False, build_path=build_dir)

    # Interpret reaction mechanism file, depending on Cantera or
    # Chemkin format.
    if not mech_filename.endswith(tuple(['.cti', '.xml'])):
        # Chemkin format; need to convert first.
        mech_filename = convert_mech(mech_filename, therm_filename)

    # Write test driver
    if lang == 'c':
        write_c_test(build_dir)
    elif lang == 'cuda':
        write_cuda_test(build_dir)

    # Compile generated source code
    files = ['chem_utils', 'dydt', 'jacob', 'spec_rates',
             'rxn_rates', 'rxn_rates_pres_mod', 'test'
             ]

    for f in files:
        args = [cmd_compile[lang]]
        args.extend(flags[lang])
        args.extend([
            '-I.' + os.path.sep + build_dir,
            '-c', os.path.join(build_dir, f + utils.file_ext[lang]),
            '-o', os.path.join(test_dir, f + '.o')
        ])
        args = [val for val in args if val.strip()]
        try:
            subprocess.check_call(args)
        except subprocess.CalledProcessError:
            print('Error: compilation failed for ' + f + utils.file_ext[lang])
            sys.exit(1)

    # Link into executable
    args = ([cmd_compile[lang]] +
            [os.path.join(test_dir, f + '.o') for f in files] +
            [libs[lang]] +
            ['-o', os.path.join(test_dir, 'test')]
            )
    try:
        subprocess.check_call(args)
    except subprocess.CalledProcessError:
        print('Error: linking of test program failed.')
        sys.exit(1)

    # Now generate data and check results
    gas = ct.Solution(mech_filename)

    # Need to get reversible reactions and those for which
    # pressure modification applies.
    idx_rev = [i for i, rxn in enumerate(gas.reactions()) if rxn.reversible]
    idx_pmod = [i for i, rxn in enumerate(gas.reactions()) if
                isinstance(rxn, ct.ThreeBodyReaction) or
                isinstance(rxn, ct.FalloffReaction) or
                isinstance(rxn, ct.ChemicallyActivatedReaction)
                ]
    # Index of element in idx_pmod that corresponds to reversible reaction
    idx_pmod_rev = [
        i for i, idx in enumerate(idx_pmod) if gas.reaction(idx).reversible]
    # Index of reversible reaction that also has pressure dependent
    # modification
    idx_rev_pmod = [i for i, idx in enumerate(idx_rev) if
                    isinstance(gas.reaction(idx), ct.ThreeBodyReaction) or
                    isinstance(gas.reaction(idx), ct.FalloffReaction) or
                    isinstance(
        gas.reaction(idx), ct.ChemicallyActivatedReaction)
    ]

    num_trials = 10
    rand_temp = np.random.uniform(800., 2000., num_trials)
    rand_pres = np.random.uniform(1., 40., num_trials) * ct.one_atm
    rand_mass = np.random.uniform(0., 1., (num_trials, gas.n_species))
    for (temp, pres, mass_frac) in zip(rand_temp, rand_pres, rand_mass):
        # Normalize mass fractions
        mass_frac /= sum(mass_frac)

        print()
        print('Testing condition {} / {}'.format(np.where(rand_temp == temp)
                                                 [0][0] + 1, num_trials))

        with open(os.path.join(test_dir, 'input.txt'), 'w') as f:
            f.write('{:.15e}\n'.format(temp))
            f.write('{:.15e}\n'.format(pres))
            for val in mass_frac:
                f.write('{:.15e}\n'.format(val))

        # Run testing executable to get output printed to file
        subprocess.check_call(os.path.join(test_dir, 'test'))

        gas.TPY = temp, pres, mass_frac

        # Derivative source term
        ode = ReactorConstPres(gas)

        # Now read output from test program
        data = np.genfromtxt(os.path.join(test_dir, 'output.txt'))

        num = int(data[0])
        test_conc = data[1: num + 1]
        data = data[num + 1:]
        err = np.linalg.norm((test_conc - gas.concentrations) /
                             gas.concentrations, 2
                             ) * 100.
        print('L2 norm error in concentration: {:.2e} %'.format(err))

        num = int(data[0])
        test_fwd_rates = data[1: num + 1]
        data = data[num + 1:]

        num = int(data[0])
        test_rev_rates = data[1: num + 1]
        data = data[num + 1:]

        num = int(data[0])
        test_pres_mod = data[1: num + 1]
        data = data[num + 1:]

        # Modify forward and reverse rates with pressure modification
        test_fwd_rates[idx_pmod] *= test_pres_mod
        test_rev_rates[idx_rev_pmod] *= test_pres_mod[idx_pmod_rev]

        err = np.linalg.norm((test_fwd_rates - gas.forward_rates_of_progress)
                             / gas.forward_rates_of_progress, 2
                             ) * 100.
        print('L2 norm error in forward reaction rates: {:.2e}'.format(err))

        err = np.linalg.norm((test_rev_rates -
                              gas.reverse_rates_of_progress[idx_rev]
                              ) / gas.reverse_rates_of_progress[idx_rev], 2
                             ) * 100.
        print('L2 norm error in reverse reaction rates: {:.2e}'.format(err))

        # Species production rates
        num = int(data[0])
        test_spec_rates = data[1: num + 1]
        data = data[num + 1:]
        err = np.linalg.norm(test_spec_rates - gas.net_production_rates, 2)
        err *= 100. / max(gas.net_production_rates)
        print('Percentage of maximum: {:.2e} %'.format(err))

        num = int(data[0])
        test_dydt = data[1: num + 1]
        data = data[num + 1:]
        err = np.linalg.norm(test_dydt - ode(), 2)
        print('L2 norm relative error in dydt: {:.2e}'.format(err))
        err *= 100. / max(ode())
        print('Percentage of maximum: {:.2e} %'.format(err))

        num = int(data[0])
        test_jacob = data[1: num + 1]

        # Calculate "true" Jacobian numerically
        jacob = eval_jacobian(ode, 6)
        err = np.linalg.norm(test_jacob - jacob, 2)
        print('L2 norm relative error in Jacobian: {:.2e}'.format(err))
        err *= 100. / max(jacob)
        print('Percentage of maximum: {:.2e} %'.format(err))
        print()

    # Cleanup all files in test directory.
    #for f in os.listdir(test_dir):
    #    os.remove(os.path.join(test_dir, f))
    #os.rmdir(test_dir)

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Tests create_jacobian versus a finite difference'
        ' Cantera jacobian\n')
    parser.add_argument('-l', '--lang',
                        type=str,
                        choices=utils.langs,
                        required=True,
                        help='Programming language for output '
                             'source files.')
    parser.add_argument('-b', '--build_dir',
                        type=str,
                        default='out' + os.path.sep,
                        help='The directory the jacob/rates/tester'
                        ' files will be generated and built in.')
    parser.add_argument('-i', '--input',
                        type=str,
                        required=True,
                        help='Input mechanism filename (e.g., mech.dat).')
    parser.add_argument('-t', '--thermo',
                        type=str,
                        default=None,
                        help='Thermodynamic database filename (e.g., '
                             'therm.dat), or nothing if in mechanism.')
    args = parser.parse_args()
    test(args.lang, args.build_dir, args.input, args.thermo)
