# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import sys
import subprocess
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

try:
    import numdifftools
except ImportError:
    print('Error: numdifftools must be installed.')
    raise

# Local imports
import utils
from pyJac import create_jacobian
import partially_stirred_reactor as pasr

# Compiler based on language
cmd_compile = dict(c='gcc',
                   cuda='nvcc',
                   fortran='gfortran'
                   )

# Flags based on language
flags = dict(c=['-std=c99'],
             cuda=['-arch=sm_20',
                   '-I/usr/local/cuda/include/',
                   '-I/usr/local/cuda/samples/common/inc/',
                   '-dc'],
             fortran='')

libs = dict(c=['-lm', '-std=c99'],
            cuda='-arch=sm_20',
            fortran='')


class ReactorConstPres(object):

    def __init__(self, gas):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorConstPres object.
        self.gas = gas
        self.P = gas.P

    def __call__(self, y=None):
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

    def __call__(self, y=None):
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
    arg.append('--permissive')

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
                          for the_file in file_list]) + '\n'
               )

    if lang == 'cuda':
        file.write('#include <cuda.h>\n'
                   '#include <cuda_runtime.h>\n'
                   '#include <helper_cuda.h>\n'
                   '#include "launch_bounds.cuh"\n'
                   '#include "gpu_macros.cuh"\n'
                   '#include "gpu_memory.cuh"\n\n'
                   )


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
        '  }\n'
        )


def __write_kernels(file, pmod):
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
    if pmod:
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


def write_cuda_test(build_dir, rev, pmod):
    with open(build_dir + os.path.sep + 'test.cu', 'w') as f:
        __write_header_include(f, 'cuda')
        f.write(
            '#ifndef SHARED_SIZE\n'
            '  #define SHARED_SIZE (0)\n'
            '#endif\n\n'
        )
        __write_output_methods(f)
        __write_kernels(f, pmod)
        f.write('int main (void) {\n'
                '\n'
                '  FILE* fp = fopen ("test/input.txt", "r");\n'
                '  double y[NN], pres;\n'
                '\n'
                )

        __write_condition_reader(f)

        the_arrays = {}
        the_arrays['NN'] = ['d_y', 'd_dy']
        the_arrays['NSP'] = ['d_conc', 'd_spec_rates']
        the_arrays['FWD_RATES'] = ['d_fwd_rates']
        the_arrays['REV_RATES'] = ['d_rev_rates']
        the_arrays['PRES_MOD_RATES'] = ['d_pres_mod_rates']
        the_arrays['NN * NN'] = ['d_jacob']
        f.write(
            '  double conc[NSP] = {0};\n'
            '  double fwd_rates[FWD_RATES] = {0};\n'
            )
        f.write(
            '  double rev_rates[REV_RATES]= {0};\n' if rev
            else '  double* rev_rates = 0;\n'
        )
        f.write(
            '  double pres_mod[PRES_MOD_RATES] = {0};\n' if pmod
            else '  double* pres_mod = 0;\n'
        )
        f.write(
            '  double spec_rates[NSP] = {0};\n'
            '  double dy[NN] = {0};\n'
            '  double jacob[NN * NN] = {0};\n'
            '\n'
            '  fp = fopen ("test/output.txt", "w");\n'
            '\n'
            '  cudaErrorCheck(cudaSetDevice(0));\n'
            '  cudaErrorCheck(cudaDeviceSetCacheConfig('
            'cudaFuncCachePreferL1));\n'
            '  //initialize cuda variables\n'
        )
        for size in the_arrays:
            for array in the_arrays[size]:
                f.write('  double* {} = 0;\n'.format(array))
                f.write('  cudaErrorCheck(cudaMalloc((void**)&'
                        '{}, {} * sizeof(double)));\n'.format(array, size))
                f.write('  cudaErrorCheck(cudaMemset({}, 0, '.format(array) +
                        '{} * sizeof(double)));\n'.format(size))
        f.write(
            '  //copy mass fractions\n'
            '  cudaErrorCheck(cudaMemcpy(d_y, y, NN * sizeof(double),'
            ' cudaMemcpyHostToDevice));\n'
            '  k_eval_conc<<<1, 1, SHARED_SIZE>>>(d_y, pres, d_conc);\n'
            '  k_eval_rxn_rates<<<1, 1, SHARED_SIZE>>>(d_y, pres, d_conc,'
            ' d_fwd_rates, d_rev_rates);\n'
            )
        if pmod:
            f.write(
            '  k_get_rxn_pres_mod<<<1, 1, SHARED_SIZE>>>(d_y, pres, d_conc,'
            ' d_pres_mod_rates);\n'
            )
        f.write(
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


def write_c_test(build_dir, pmod):
    with open(build_dir + os.path.sep + 'test.c', 'w') as f:
        __write_header_include(f, 'c')
        __write_output_methods(f)
        #__write_fd_jacob(f, 'c')
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
            )
        if pmod:
            f.write(
            '  double pres_mod[PRES_MOD_RATES] = {0};\n'
            '  get_rxn_pres_mod (y[0], pres, conc, pres_mod);\n'
            )
        else:
            f.write('double* pres_mod = 0;\n')
        f.write(
            '  eval_spec_rates (fwd_rates, rev_rates, pres_mod, sp_rates);\n'
            '  dydt (0.0, pres, y, dy);\n'
            '  write_rates (fp, fwd_rates, rev_rates, '
            'pres_mod, sp_rates, dy);\n'
            '\n'
            '  eval_jacob (0.0, pres, y, jacob);\n'
            '  write_jacob (fp, jacob);\n'
            #'  double fd_jacob[NN * NN] = {0};\n'
            #'  eval_fd_jacob(0.0, pres, y, fd_jacob);\n'
            #'  write_jacob(fp, fd_jacob);\n'
            '\n'
            '  fclose (fp);\n'
            '\n'
            '  return 0;\n'
            '}\n'
            '\n'
        )

    return None


class analytic_eval_jacob:
    def __init__(self, pressure):
        self.pres = pressure
        import pyjacob
        self.jac = pyjacob

    def dydt(self, y):
        dy = np.zeros_like(y)
        self.jac.py_dydt(0, self.pres, y, dy)
        return dy

    def eval_jacobian(self, gas, order):
        """
        """
        abs_tol = 1.e-20
        rel_tol = 1.e-8

        y = np.hstack((gas.T, gas.Y))
        step = np.array([1e-15 for x in range(len(y))])
        step[0] = abs_tol

        jacob = numdifftools.Jacobian(self.dydt, order=order, method='central', full_output=True)
        arr = jacob(y)
        fd = np.array(arr[0])
        index = np.array(arr[-1].index)
        return fd.T.flatten()

def is_pdep(rxn):
    return (isinstance(rxn, ct.ThreeBodyReaction) or
    isinstance(rxn, ct.FalloffReaction) or
    isinstance(rxn, ct.ChemicallyActivatedReaction))

def run_pasr(pasr_input_file, mech_filename, pasr_output_file):
    # Run PaSR to get data
    pasr_input = pasr.parse_input_file(pasr_input_file)
    state_data = pasr.run_simulation(
                    mech_filename,
                    pasr_input['case'],
                    pasr_input['temperature'],
                    pasr_input['pressure'],
                    pasr_input['equivalence ratio'],
                    pasr_input['fuel'],
                    pasr_input['oxidizer'],
                    pasr_input['complete products'],
                    pasr_input['number of particles'],
                    pasr_input['residence time'],
                    pasr_input['mixing time'],
                    pasr_input['pairing time'],
                    pasr_input['number of residence times']
                    )
    if pasr_output_file:
        np.save(pasr_output_file, state_data)
    return state_data

class cpyjac_evaluator(object):
    def __init__(self):
        import pyjacob
        pass
    def eval_conc(self, temp, pres, mass_frac, mw_avg, rho, conc):
        pyjacob.eval_conc(self, temp, pres, mass_frac, mw_avg, rho, conc)
    def eval_rxn_rates(self, temp, pres, conc, fwd_rates, rev_rates):
        pyjacob.eval_rxn_rates(temp, pres, conc, fwd_rates, rev_rates)
    def get_rxn_pres_mod(self, temp, pres, conc, pres_mod):
        pyjacob.get_rxn_pres_mod(temp, pres, conc, pres_mod)
    def eval_spec_rates(self, fwd_rates, rev_rates, pres_mod, spec_rates):
        pyjacob.eval_spec_rates(fwd_rates, rev_rates, pres_mod, spec_rates)
    def dydt(self, t, pres, dummy, dydt):
        pyjacob.eval_dydt(t, pres, dummy, dydt)
    def eval_jacobian(self, pres, dummy, jacob):
        pyjacob.eval_jacobian(pres, dummy, jacob)
    def update(self, index):
        pass

class cupyjac_evaluator(cpyjac_evaluator):
    def __init__(self, gas, state_data):
        import cu_pyjacob as pyjacob

        def czeros(shape):
            arr = np.zeros(shape)
            return arr.flatten(order='c')
        def reshaper(arr, shape):
            return arr.reshape(shape, order='f').astype(np.dtype('d'), order='c')

        idx_rev = [i for i, rxn in enumerate(gas.reactions()) if rxn.reversible]
        idx_pmod = [i for i, rxn in enumerate(gas.reactions()) if
                is_pdep(rxn)
                ]

        cuda_state = state_data[:, 1:]
        num_cond = cuda_state.shape[0]
        #init vectors
        test_conc = czeros((num_cond, gas.n_species))
        test_fwd_rates = czeros((num_cond,gas.n_reactions))
        test_rev_rates = czeros((num_cond,len(idx_rev)))
        test_pres_mod = czeros((num_cond,len(idx_pmod)))
        test_spec_rates = czeros((num_cond,gas.n_species))
        test_dydt = czeros((num_cond,gas.n_species + 1))
        test_jacob = czeros((num_cond,(gas.n_species + 1) * (gas.n_species + 1)))

        mw_avg = czeros(num_cond)
        rho = czeros(num_cond)
        temp = cuda_state[:, 0].flatten(order='c')
        pres = cuda_state[:, 1].flatten(order='c')
        mass_frac = cuda_state[:, 2:].flatten(order='f').astype(np.dtype('d'), order='c')
        y_dummy = cuda_state[:, [0] + range(2, cuda_state.shape[1])].flatten(order='f').astype(np.dtype('d'), order='c')

        pyjacob.py_eval_conc(num_cond, temp, pres, mass_frac, mw_avg, rho, test_conc)
        pyjacob.py_eval_rxn_rates(num_cond, temp, pres, test_conc, test_fwd_rates, test_rev_rates)
        pyjacob.py_get_rxn_pres_mod(num_cond, temp, pres, test_conc, test_pres_mod)
        pyjacob.py_eval_spec_rates(num_cond, test_fwd_rates, test_rev_rates, test_pres_mod, test_spec_rates)
        pyjacob.py_dydt(num_cond, pres, y_dummy, test_dydt)
        pyjacob.py_eval_jacobian(num_cond, pres, y_dummy, test_jacob)

        #reshape for comparison
        self.test_conc = reshaper(test_conc, (num_cond, gas.n_species))
        self.test_fwd_rates = reshaper(test_fwd_rates, (num_cond, gas.n_reactions))
        self.test_rev_rates = reshaper(test_rev_rates, (num_cond, len(idx_rev)))
        self.test_pres_mod = reshaper(test_pres_mod, (num_cond, len(idx_pmod)))
        self.test_spec_rates = reshaper(test_spec_rates, (num_cond,gas.n_species))
        self.test_dydt = reshaper(test_dydt, (num_cond,gas.n_species + 1))
        self.test_jacob = reshaper(test_jacob, (num_cond, (gas.n_species + 1) * (gas.n_species + 1)))
        self.index = 0

    def update(self, index):
        self.index = index
    def eval_conc(self, temp, pres, mass_frac, mw_avg, rho, conc):
        conc[:] = self.test_conc[self.index, :]
    def eval_rxn_rates(self, temp, pres, conc, fwd_rates, rev_rates):
        fwd_rates[:] = self.test_fwd_rates[self.index, :]
        rev_rates[:] = self.test_rev_rates[self.index, :]
    def get_rxn_pres_mod(self, temp, pres, conc, pres_mod):
        pres_mod[:] = self.test_pres_mod[self.index, :]
    def eval_spec_rates(self, fwd_rates, rev_rates, pres_mod, spec_rates):
        spec_rates[:] = self.test_spec_rates[self.index, :]
    def dydt(self, t, pres, y, dydt):
        dydt[:] = self.test_dydt[self.index, :]
    def eval_jacobian(self, t, pres, y, jacob):
        jacob[:] = self.test_jacob[self.index, :]




def test(lang, build_dir, mech_filename, therm_filename=None,
         pasr_input_file='pasr_input.yaml', generate_jacob=True,
         seed=None, pasr_output_file=None):
    """
    """

    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    # First check for appropriate Compiler
    try:
        subprocess.check_call(['which', cmd_compile[lang]])
    except subprocess.CalledProcessError:
        print('Error: appropriate compiler for language not found.')
        sys.exit(1)

    #remove the old jaclist
    try:
        os.remove('out/jacobs/jac_list_c')
    except:
        pass
    try:
        os.remove('out/jacobs/jac_list_cuda')
    except:
        pass


    if generate_jacob:
        # Create Jacobian and supporting source code files
        create_jacobian(lang, mech_filename, therm_name=therm_filename,
                        optimize_cache=False, build_path=build_dir,
                        no_shared=True
                        )

    # Interpret reaction mechanism file, depending on Cantera or
    # Chemkin format.
    if not mech_filename.endswith(tuple(['.cti', '.xml'])):
        # Chemkin format; need to convert first.
        mech_filename = convert_mech(mech_filename, therm_filename)

    #write and compile the dydt python wrapper
    try:
        os.remove('pyjacob.so')
    except:
        pass
    try:
        os.remove('cu_pyjacob.so')
    except:
        pass

    #get the cantera object
    gas = ct.Solution(mech_filename)
    pmod = any([is_pdep(rxn) for rxn in gas.reactions()])
    rev = any(rxn.reversible for rxn in gas.reactions())

    # Now generate data and check results

    # Need to get reversible reactions and those for which
    # pressure modification applies.
    idx_rev = [i for i, rxn in enumerate(gas.reactions()) if rxn.reversible]
    idx_pmod = [i for i, rxn in enumerate(gas.reactions()) if
                is_pdep(rxn)
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

    if pasr_output_file:
        #load old test data
        try:
            state_data = np.load(pasr_output_file)
        except Exception, e:
            # Run PaSR to get data
            print('Could not load saved pasr data... re-running')
            state_data = run_pasr(pasr_input_file, mech_filename, pasr_output_file)
    else:
        # Run PaSR to get data
        state_data = run_pasr(pasr_input_file, mech_filename, pasr_output_file)
    # Reshape array to treat time steps and particles the same
    state_data = state_data.reshape(state_data.shape[0] * state_data.shape[1],
                                    state_data.shape[2]
                                    )

    #need to compile this anyways, it's way easier to get the analytical
    #jacobian evaulator to use the c interface
    subprocess.check_call(['python2.7', os.getcwd() + os.path.sep + 'pyjacob_setup.py', 'build_ext', '--inplace'])
    if lang == 'cuda':
        subprocess.check_call(['python2.7', os.getcwd() + os.path.sep + 'pyjacob_cuda_setup.py', 'build_ext', '--inplace'])
        pyjacob = cupyjac_evaluator(gas, state_data)
    else:
        pyjacob = cpyjac_evaluator()

    num_trials = len(state_data)

    err_dydt = np.zeros(num_trials)
    err_dydt_zero = np.zeros(num_trials)
    err_jac_norm = np.zeros(num_trials)
    err_jac = np.zeros(num_trials)
    err_jac_thr = np.zeros(num_trials)
    err_jac_max = np.zeros(num_trials)
    err_jac_zero = np.zeros(num_trials)

    for i, state in enumerate(state_data):
        #update index in case we're using cuda
        pyjacob.update(i)

        temp = state[1]
        pres = state[2]
        mass_frac = state[3:]

        ajac = analytic_eval_jacob(pres)

        #init vectors
        test_conc = np.zeros(gas.n_species)
        test_fwd_rates = np.zeros(gas.n_reactions)
        test_rev_rates = np.zeros(len(idx_rev))
        test_pres_mod = np.zeros(len(idx_pmod))
        test_spec_rates = np.zeros(gas.n_species)
        test_dydt = np.zeros(gas.n_species + 1)
        test_jacob = np.zeros((gas.n_species + 1) * (gas.n_species + 1))

        print()
        print('Testing condition {} / {}'.format(i + 1, num_trials))

        # Run testing executable to get output printed to file
        #subprocess.check_call(os.path.join(test_dir, 'test'))

        gas.TPY = temp, pres, mass_frac

        # Derivative source term
        ode = ReactorConstPres(gas)

        mw_avg = 0
        rho = 0
        #get conc
        pyjacob.eval_conc(temp, pres, mass_frac, mw_avg, rho, test_conc)

        non_zero = np.where(test_conc > 0.)[0]
        err = abs((test_conc[non_zero] - gas.concentrations[non_zero]) /
                  gas.concentrations[non_zero]
                  )
        max_err = np.max(err)
        loc = non_zero[np.argmax(err)]
        err = np.linalg.norm(err) * 100.
        print('L2 norm error in non-zero concentration: {:.2e} %'.format(err))
        print('Max error in non-zero concentration: {:.2e} % @ species {}'
            .format(max_err * 100., loc))

        #get rates
        pyjacob.eval_rxn_rates(temp, pres, test_conc, test_fwd_rates, test_rev_rates)
        pyjacob.get_rxn_pres_mod(temp, pres, test_conc, test_pres_mod)

        # Modify forward and reverse rates with pressure modification
        test_fwd_rates[idx_pmod] *= test_pres_mod
        test_rev_rates[idx_rev_pmod] *= test_pres_mod[idx_pmod_rev]

        non_zero = np.where(test_fwd_rates > 0.)[0]
        err = abs((test_fwd_rates[non_zero] -
                  gas.forward_rates_of_progress[non_zero]) /
                  gas.forward_rates_of_progress[non_zero]
                  )
        max_err = np.max(err)
        loc = non_zero[np.argmax(err)]
        err = np.linalg.norm(err) * 100.
        print('L2 norm error in non-zero forward reaction rates: {:.2e}%'.format(err))
        print('Max error in non-zero forward reaction rates: {:.2e}% @ reaction {}'.
              format(max_err * 100., loc))

        if idx_rev:
            non_zero = np.where(test_rev_rates > 0.)[0]
            err = abs((test_rev_rates[non_zero] -
                      (gas.reverse_rates_of_progress[idx_rev])[non_zero]) /
                      (gas.reverse_rates_of_progress[idx_rev])[non_zero]
                      )
            max_err = np.max(err)
            loc = non_zero[np.argmax(err)]
            err = np.linalg.norm(err) * 100.
            print('L2 norm error in non-zero reverse reaction rates: {:.2e}%'.format(err))
            print('Max error in non-zero reverse reaction rates: {:.2e}% @ reaction {}'.
                  format(max_err * 100., loc))

        # Species production rates
        pyjacob.eval_spec_rates(test_fwd_rates, test_rev_rates, test_pres_mod, test_spec_rates)
        non_zero = np.where(test_spec_rates != 0.)[0]
        zero = np.where(test_spec_rates == 0.)[0]
        err = abs((test_spec_rates[non_zero] -
                  gas.net_production_rates[non_zero]) /
                  gas.net_production_rates[non_zero]
                  )
        max_err = np.max(err)
        loc = non_zero[np.argmax(err)]
        err = np.linalg.norm(err) * 100.
        print('L2 norm relative error of non-zero net production rates: '
              '{:.2e} %'.format(err)
              )
        print('Max error in non-zero net production rates: {:.2e}% '
              '@ species {}'.format(max_err * 100., loc)
              )
        err = np.linalg.norm(
            test_spec_rates[zero] - gas.net_production_rates[zero])
        print(
            'L2 norm difference of "zero" net production rates: {:.2e}'
            .format(err))

        y_dummy = np.hstack((temp, mass_frac))
        pyjacob.dydt(0, pres, y_dummy, test_dydt)
        non_zero = np.where(test_dydt != 0.)[0]
        zero = np.where(test_dydt == 0.)[0]
        err = abs((test_dydt[non_zero] - ode()[non_zero]) /
                  ode()[non_zero]
                  )
        max_err = np.max(err)
        loc = non_zero[np.argmax(err)]
        err = np.linalg.norm(err) * 100.
        err_dydt[i] = err
        print('L2 norm relative error of non-zero dydt: {:.2e} %'.format(err))
        print('Max error in non-zero dydt: {:.2e}% '
              '@ index {}'.format(max_err * 100., loc)
              )
        err = np.linalg.norm(test_dydt[zero] - ode()[zero])
        err_dydt_zero[i] = err
        print('L2 norm difference of "zero" dydt: {:.2e}'.format(err))

        pyjacob.eval_jacobian(0, pres, y_dummy, test_jacob)
        non_zero = np.where(abs(test_jacob) > np.linalg.norm(test_jacob) / 1.e20)[0]
        zero = np.where(test_jacob == 0.)[0]

        jacob = ajac.eval_jacobian(gas, 6)
        err = abs((test_jacob[non_zero] - jacob[non_zero]) /
                  jacob[non_zero]
                  )
        max_err = np.max(err)
        loc = non_zero[np.argmax(err)]
        err = np.linalg.norm(err) * 100.
        print('Max error in non-zero Jacobian: {:.2e}% '
              '@ index {} %'.format(max_err * 100., loc))
        print('L2 norm of relative error of Jacobian: '
              '{:.2e} %'.format(err))
        err_jac_max[i] = max_err
        err_jac[i] = err

        # Thresholded error
        non_zero = np.where(abs(test_jacob) > np.linalg.norm(test_jacob) / 1.e8)[0]
        err = abs((test_jacob[non_zero] - jacob[non_zero]) /
                  jacob[non_zero]
                  )
        err = np.linalg.norm(err) * 100.
        err_jac_thr[i] = err
        print('L2 norm of thresholded relative error of Jacobian: '
              '{:.2e} %'.format(err))

        err = np.linalg.norm(test_jacob - jacob) / np.linalg.norm(jacob)
        err_jac_norm[i] = err
        print('L2 norm error of Jacobian: {:.2e}'.format(err))

        err = np.linalg.norm(test_jacob[zero] - jacob[zero])
        err_jac_zero[i] = err
        print('L2 norm difference of "zero" Jacobian: '
              '{:.2e}'.format(err))

    plt.semilogy(state_data[:,1], err_jac_norm, 'o')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Jacobian matrix norm error')
    pp = PdfPages('Jacobian_error_norm.pdf')
    pp.savefig()
    pp.close()

    plt.figure()
    plt.semilogy(state_data[:,1], err_jac, 'o')
    plt.xlabel('Temperature [K]')
    plt.ylabel('Jacobian matrix relative error norm [%]')
    pp = PdfPages('Jacobian_relative_error.pdf')
    pp.savefig()
    pp.close()

    # Save all error arrays
    np.savez('error_arrays.npz', err_dydt=err_dydt, err_jac_norm=err_jac_norm,
              err_jac=err_jac, err_jac_thr=err_jac_thr)

    # Cleanup all files in test directory.
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
    for f in test_files + ['pyjacob.so', 'pyjacob_wrapper.c']:
        os.remove(f)
    os.rmdir(test_dir)

    # Now clean build directory
    for root, dirs, files in os.walk('./build', topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir('./build')

    return 0

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Tests pyJac versus a finite difference'
                    ' Cantera jacobian\n'
        )
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
    parser.add_argument('-m', '--mech',
                        type=str,
                        required=True,
                        help='Mechanism filename (e.g., mech.dat).')
    parser.add_argument('-t', '--thermo',
                        type=str,
                        default=None,
                        help='Thermodynamic database filename (e.g., '
                             'therm.dat), or nothing if in mechanism.')
    parser.add_argument('-i', '--input',
                        type=str,
                        default='pasr_input.yaml',
                        help='Perfectly stirred reactor input file for '
                             'generating test data (e.g, pasr_input.yaml)')
    parser.add_argument('-dng', '--do_not_generate',
                        action='store_false',
                        dest='generate_jacob',
                        default=True,
                        help='Use this option to have the tester utilize'
                        ' the already existant jacobian files')
    parser.add_argument('-s', '--seed',
                        type=int,
                        default=None,
                        help='The seed to be used for random numbers')
    parser.add_argument('-p', '--pasr_output',
                        type=str,
                        default=None,
                        help='An optional saved .npy file that has the '
                             'resulting pasr data (to speed testing)')
    args = parser.parse_args()
    test(args.lang, args.build_dir, args.mech, args.thermo, args.input,
         args.generate_jacob, args.seed, args.pasr_output
         )
