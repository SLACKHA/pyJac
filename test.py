# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import sys
import subprocess

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

# Compiler based on language
cmd_compile = dict(c = 'gcc',
                   cuda = 'nvcc',
                   fortran = 'gfortran'
                   )

# Source code extension based on language
src_ext = dict(c = '.c', cuda = '.cu', fortran = '.f90', matlab = '.m')


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


def write_c_test(build_dir):
    with open(build_dir + os.path.sep + 'test.c', 'w') as f:
        f.write(
            '#include <stdlib.h>\n'
            '#include <stdio.h>\n'
            '\n'
            '#include "header.h"\n'
            '#include "mechanism.h"\n'
            '#include "chem_utils.h"\n'
            '#include "rates.h"\n'
            '#include "dydt.h"\n'
            '#include "jacob.h"\n'
            '\n'
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
            'int main (void) {\n'
            '\n'
            '  FILE* fp = fopen ("test/input.txt", "r");\n'
            '  double y[NN], pres;\n'
            '\n'
            '  int buff_size = 1024;\n'
            '  char buffer [buff_size];\n'
            '  char* ptr, *eptr;\n'
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
            '\n'
            '  double conc[NSP];\n'
            '  double fwd_rates[FWD_RATES];\n'
            '  double rev_rates[REV_RATES];\n'
            '  double pres_mod[PRES_MOD_RATES];\n'
            '  double sp_rates[NSP];\n'
            '  double dy[NN];\n'
            '  double jacob[NN * NN];\n'
            '\n'
            '  fp = fopen ("test/output.txt", "w");\n'
            '\n'
            '  eval_conc (y[0], pres, &y[1], conc);\n'
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

    sqrt_rnd = sqrt(np.finfo(float).eps)
    err_wt = abs(y) * rel_tol + abs_tol

    r0 = (1000. * rel_tol * np.finfo(float).eps * len(y) *
          sqrt(np.sum(np.power(err_wt * dydt(), 2)) / len(y))
          )

    jacob = np.zeros(len(y) ** 2)
    for j, y_j  in enumerate(y):
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

    # Interpret reaction mechanism file, depending on Cantera or
    # Chemkin format.
    if not mech_filename.endswith(tuple(['.cti','.xml'])):
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
        args = [cmd_compile[lang], '-I.' + os.path.sep + build_dir,
                '-c', os.path.join(build_dir, f + src_ext[lang]),
                '-o', os.path.join(test_dir, f + '.o')
                ]
        try:
            subprocess.check_call(args)
        except subprocess.CalledProcessError:
            print('Error: compilation failed for ' + f + src_ext[lang])
            sys.exit(1)

    # Link into executable
    args = ([cmd_compile[lang]] +
            [os.path.join(test_dir, f + '.o') for f in files] +
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
    idx_pmod_rev = [i for i, idx in enumerate(idx_pmod) if gas.reaction(idx).reversible]
    # Index of reversible reaction that also has pressure dependent modification
    idx_rev_pmod = [i for i, idx in enumerate(idx_rev) if
                    isinstance(gas.reaction(idx), ct.ThreeBodyReaction) or
                    isinstance(gas.reaction(idx), ct.FalloffReaction) or
                    isinstance(gas.reaction(idx), ct.ChemicallyActivatedReaction)
                    ]

    num_trials = 10
    rand_temp = np.random.uniform(800., 2000., num_trials)
    rand_pres = np.random.uniform(1., 40., num_trials) * ct.one_atm
    rand_mass = np.random.uniform(0., 1., (num_trials, gas.n_species))
    for (temp, pres, mass_frac) in zip(rand_temp, rand_pres, rand_mass):
        # Normalize mass fractions
        mass_frac /= sum(mass_frac)

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
        test_conc = data[1 : num + 1]
        data = data[num + 1 :]
        err = np.linalg.norm((test_conc - gas.concentrations) /
                             gas.concentrations, 2
                             ) * 100.
        print('L2 norm error in concentration: {:.2e} %'.format(err))

        num = int(data[0])
        test_fwd_rates = data[1 : num + 1]
        data = data[num + 1 :]

        num = int(data[0])
        test_rev_rates = data[1 : num + 1]
        data = data[num + 1 :]

        num = int(data[0])
        test_pres_mod = data[1 : num + 1]
        data = data[num + 1 :]

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
        test_spec_rates = data[1 : num + 1]
        data = data[num + 1 :]
        err = np.linalg.norm(test_spec_rates - gas.net_production_rates, 2)
        print('L2 norm relative error in species rates: {:.2e}'.format(err))
        err *= 100. / max(gas.net_production_rates)
        print('Percentage of maximum: {:.2e} %'.format(err))

        num = int(data[0])
        test_dydt = data[1 : num + 1]
        data = data[num + 1 :]
        err = np.linalg.norm(test_dydt - ode(), 2)
        print('L2 norm relative error in dydt: {:.2e}'.format(err))
        err *= 100. / max(ode())
        print('Percentage of maximum: {:.2e} %'.format(err))

        num = int(data[0])
        test_jacob = data[1 : num + 1]

        # Calculate "true" Jacobian numerically
        jacob = eval_jacobian(ode, 6)
        err = np.linalg.norm(test_jacob - jacob, 2)
        print('L2 norm relative error in Jacobian: {:.2e}'.format(err))
        err *= 100. / max(jacob)
        print('Percentage of maximum: {:.2e} %'.format(err))

    # Cleanup all files in test directory.
    for f in os.listdir(test_dir):
        os.remove(os.path.join(test_dir, f))
    os.rmdir(test_dir)
