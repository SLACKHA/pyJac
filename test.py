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
        """Evaluate finite difference Jacobian.

        Uses Richardson extrapolation applied to central finite difference to
        to achieve much higher accuracy.
        """
        #abs_tol = 1.e-20
        #rel_tol = 1.e-8

        y = np.hstack((gas.T, gas.Y))
        #step = np.array([1e-15 for x in range(len(y))])
        #step[0] = abs_tol

        jacob = numdifftools.Jacobian(self.dydt, order=order,
                                      method='central', full_output=True
                                      )
        arr = jacob(y)
        fd = np.array(arr[0])
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
    def check_optimized(self, build_dir, filename='mechanism.h'):
        with open(os.path.join(build_dir, filename), 'r') as file:
            opt = False
            for line in file.readlines():
                if 'Cache Optimized' in line:
                    opt = True
                    break
        self.cache_opt = opt
        if self.cache_opt:
            with open(os.path.join(build_dir, 'optimized.pickle'), 'rb') as file:
                specs = pickle.load(file)
                reacs = pickle.load(file)
                self.fwd_rxn_map = np.array(pickle.load(file))
                self.fwd_rev_rxn_map = np.array([i for i in self.fwd_rxn_map if reacs[i].rev])
                self.fwd_spec_map = np.array(pickle.load(file))
                self.fwd_pdep_map = np.array(pickle.load(file))
                self.fwd_dydt_map = np.array([0] + self.fwd_spec_map)
                spec_ordering = pickle.load(file)
                rxn_ordering = pickle.load(file)

            self.sp_map = np.array(spec_ordering)
            self.rxn_map = np.array(rxn_ordering)

            rev_reacs = [i for i, rxn in enumerate(reacs) if rxn.rev]
            if rev_reacs:
                old_rev_order = [i for i in old_rxn_order if reacs[i].rev]
                self.rev_rxn_map = np.array([rev_reacs.index(rxn) for rxn in old_rev_order])
            else:
                self.rev_rxn_map = np.array([])

            pdep_reacs = [i for i, rxn in enumerate(reacs) if rxn.pdep or rxn.thd_body]
            if pdep_reacs:
                old_pdep_order = [rxn for rxn in old_rxn_order if reacs[rxn].pdep or reacs[rxn].thd_body]
                self.pdep_rxn_map = np.array([pdep_reacs.index(rxn) for rxn in old_pdep_order])
            else:
                self.pdep_rxn_map = np.array([])

            self.dydt_map = np.array([0] + spec_ordering)
            #handle jacobian map separately
            #dT/dT doesn't change
            self.jac_map = [0]
            #updated species map (+ 1 to account for dT entry)
            self.jac_map.extend([x + 1 for x in self.sp_map])
            for i in range(len(self.sp_map)):
                #dT / dY entry
                self.jac_map.append((self.sp_map[i] + 1) * (len(specs) + 1))
                for j in range(len(self.sp_map)):
                    self.jac_map.append((self.sp_map[i] + 1) * (len(specs) + 1) + self.sp_map[j] + 1)
            self.jac_map = np.array(self.jac_map)
        else:
            self.sp_map = None
            self.rxn_map = None
            self.rev_rxn_map = None
            self.pdep_rxn_map = None
            self.dydt_map = None
            self.jac_map = None

        
    def __init__(self, build_dir, gas, module_name='pyjacob'):
        self.idx_rev = [i for i, rxn in enumerate(gas.reactions()) if rxn.reversible]
        self.idx_pmod = [i for i, rxn in enumerate(gas.reactions()) if
                is_pdep(rxn)
                ]
        self.check_optimized(build_dir)
        self.pyjac = __import__(module_name)


    def eval_conc(self, temp, pres, mass_frac, conc):
        mw_avg = 0
        rho = 0
        if self.cache_opt:
            test_mf[:] = mass_frac[self.sp_map]
            self.pyjac.py_eval_conc(temp, pres, mass_frac, mw_avg, rho, conc)
            conc[:] = conc[self.sp_map]
        else:
            self.pyjac.py_eval_conc(temp, pres, mass_frac, mw_avg, rho, conc)
    def eval_rxn_rates(self, temp, pres, conc, fwd_rates, rev_rates):
        if self.cache_opt:
            test_conc[:] = conc[self.fwd_spec_map]
            self.pyjac.py_eval_rxn_rates(temp, pres, test_conc,
             fwd_rates, rev_rates)
            fwd_rates[:] = fwd_rates[self.rxn_map]
            rev_rates[:] = rev_rates[self.rev_rxn_map]
        else:
            self.pyjac.py_eval_rxn_rates(temp, pres, conc, fwd_rates, rev_rates)
    def get_rxn_pres_mod(self, temp, pres, conc, pres_mod):
        if self.cache_opt:
            test_conc[:] = conc[self.fwd_spec_map]
            pyjacob.py_get_rxn_pres_mod(temp, pres, test_conc, pres_mod)
            pres_mod[:] = pres_mod[self.pdep_rxn_map]
        else:
            self.pyjac.py_get_rxn_pres_mod(temp, pres, conc, pres_mod)
    def eval_spec_rates(self, fwd_rates, rev_rates, pres_mod, spec_rates):
        if self.cache_opt:
            test_fwd[:] = fwd_rates[self.fwd_rxn_map]
            test_rev[:] = rev_rates[self.fwd_rev_rxn_map]
            test_pdep[:] = pres_mod[self.fwd_pdep_map]
            self.pyjac.py_eval_spec_rates(test_fwd, test_rev, test_pdep, spec_rates)
            spec_rates[:] = spec_rates[self.sp_map]
        else:
            self.pyjac.py_eval_spec_rates(fwd_rates, rev_rates, pres_mod, spec_rates)
    def dydt(self, t, pres, y, dydt):
        if self.cache_opt:
            test_y[:] = y[self.fwd_dydt_map]
            pyjacob.py_dydt(t, pres, test_y, dydt)
            dydt[:] = dydt[self.dydt_map]
        else:   
            self.pyjac.py_dydt(t, pres, y, dydt)
    def eval_jacobian(self, t, pres, y, jacob):
        if self.cache_opt:
            test_y[:] = y[self.fwd_dydt_map]
            self.pyjac.py_eval_jacobian(pres, test_y, jacob)
            jacob[:] = jacob[self.jac_map]
        else:
            self.pyjac.py_eval_jacobian(t, pres, y, jacob)
    def update(self, index):
        pass

class cupyjac_evaluator(cpyjac_evaluator):
    def __init__(self, build_dir, gas, state_data):
        super(cupyjac_evaluator, self).__init__(build_dir, gas, 'cu_pyjacob')

        def czeros(shape):
            arr = np.zeros(shape)
            return arr.flatten(order='c')
        def reshaper(arr, shape, reorder=None):
            arr = arr.reshape(shape, order='f').astype(np.dtype('d'), order='c')
            if reorder is not None:
                arr = arr[:, reorder]
            return arr

        if not self.cache_opt:
            self.fwd_spec_map = np.arange(gas.n_species)

        cuda_state = state_data[:, 1:]
        num_cond = cuda_state.shape[0]
        #init vectors
        test_conc = czeros((num_cond, gas.n_species))
        test_fwd_rates = czeros((num_cond,gas.n_reactions))
        test_rev_rates = czeros((num_cond,len(self.idx_rev)))
        test_pres_mod = czeros((num_cond,len(self.idx_pmod)))
        test_spec_rates = czeros((num_cond,gas.n_species))
        test_dydt = czeros((num_cond,gas.n_species + 1))
        test_jacob = czeros((num_cond,(gas.n_species + 1) * (gas.n_species + 1)))

        mw_avg = czeros(num_cond)
        rho = czeros(num_cond)
        temp = cuda_state[:, 0].flatten(order='c')
        pres = cuda_state[:, 1].flatten(order='c')
        mass_frac = cuda_state[:, [2 + x for x in self.fwd_spec_map]].flatten(order='f')\
                                .astype(np.dtype('d'), order='c')
        y_dummy = cuda_state[:, [0] + [2 + x for x in self.fwd_spec_map]].flatten(order='f')\
                                .astype(np.dtype('d'), order='c')

        self.pyjac.py_eval_conc(num_cond, temp, pres, mass_frac, mw_avg, rho, test_conc)
        self.pyjac.py_eval_rxn_rates(num_cond, temp, pres, test_conc, test_fwd_rates, test_rev_rates)
        self.pyjac.py_get_rxn_pres_mod(num_cond, temp, pres, test_conc, test_pres_mod)
        self.pyjac.py_eval_spec_rates(num_cond, test_fwd_rates, test_rev_rates, test_pres_mod, test_spec_rates)
        self.pyjac.py_dydt(num_cond, 0, pres, y_dummy, test_dydt)
        self.pyjac.py_eval_jacobian(num_cond, 0, pres, y_dummy, test_jacob)

        #reshape for comparison
        self.test_conc = reshaper(test_conc, (num_cond, gas.n_species), self.sp_map)
        self.test_fwd_rates = reshaper(test_fwd_rates, (num_cond, gas.n_reactions), self.rxn_map)
        self.test_rev_rates = reshaper(test_rev_rates, (num_cond, len(self.idx_rev)), self.rev_rxn_map)
        self.test_pres_mod = reshaper(test_pres_mod, (num_cond, len(self.idx_pmod)), self.pdep_rxn_map)
        self.test_spec_rates = reshaper(test_spec_rates, (num_cond,gas.n_species), self.sp_map)
        self.test_dydt = reshaper(test_dydt, (num_cond,gas.n_species + 1), self.dydt_map)
        self.test_jacob = reshaper(test_jacob, (num_cond, (gas.n_species + 1) * (gas.n_species + 1)),
                            self.jac_map)
        self.index = 0

    def update(self, index):
        self.index = index
    def eval_conc(self, temp, pres, mass_frac, conc):
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
    #need to compile this anyways, it's way easier to get the analytical
    #jacobian evaulator to use the c interface
    subprocess.check_call(['python2.7', os.getcwd() + os.path.sep +
                           'pyjacob_setup.py', 'build_ext', '--inplace'
                           ])

    try:
        os.remove('cu_pyjacob.so')
    except:
        pass
    if lang == 'cuda':
        subprocess.check_call(['python2.7', os.getcwd() + os.path.sep + 
                               'pyjacob_cuda_setup.py', 'build_ext', '--inplace'
                               ])

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
            state_data = run_pasr(pasr_input_file, mech_filename,
                                  pasr_output_file
                                  )
    else:
        # Run PaSR to get data
        state_data = run_pasr(pasr_input_file, mech_filename,
                              pasr_output_file
                              )
    # Reshape array to treat time steps and particles the same
    if len(state_data.shape) == 3:
        state_data = state_data.reshape(state_data.shape[0] *
                                        state_data.shape[1],
                                        state_data.shape[2]
                                        )
    
    if lang == 'cuda':
        pyjacob = cupyjac_evaluator(build_dir, gas, state_data)
    else:
        pyjacob = cpyjac_evaluator(build_dir, gas)

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

        gas.TPY = temp, pres, mass_frac

        # Derivative source term
        ode = ReactorConstPres(gas)

        #get conc
        pyjacob.eval_conc(temp, pres, mass_frac, test_conc)

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
        pyjacob.eval_rxn_rates(temp, pres, test_conc,
                                  test_fwd_rates, test_rev_rates
                                  )
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
        print('L2 norm error in non-zero forward reaction rates: '
              '{:.2e}%'.format(err)
              )
        print('Max error in non-zero forward reaction rates: '
              '{:.2e}% @ reaction {}'.format(max_err * 100., loc)
              )

        if idx_rev:
            non_zero = np.where(test_rev_rates > 0.)[0]
            err = abs((test_rev_rates[non_zero] -
                      (gas.reverse_rates_of_progress[idx_rev])[non_zero]) /
                      (gas.reverse_rates_of_progress[idx_rev])[non_zero]
                      )
            max_err = np.max(err)
            loc = non_zero[np.argmax(err)]
            err = np.linalg.norm(err) * 100.
            print('L2 norm error in non-zero reverse reaction rates: '
                  '{:.2e}%'.format(err)
                  )
            print('Max error in non-zero reverse reaction rates: '
                  '{:.2e}% @ reaction {}'.format(max_err * 100., loc)
                  )

        # Species production rates
        pyjacob.eval_spec_rates(test_fwd_rates, test_rev_rates,
                                test_pres_mod, test_spec_rates
                                )

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
        non_zero = np.where(abs(test_jacob) > 1.e-30)[0]
        zero = np.where(test_jacob == 0.)[0]

        jacob = ajac.eval_jacobian(gas, 6)
        err = abs((test_jacob[non_zero] - jacob[non_zero]) /
                  jacob[non_zero]
                  )
        max_err = np.max(err)
        loc = non_zero[np.argmax(err)]
        err = np.linalg.norm(err) * 100.
        print('Max error in non-zero Jacobian: {:.2e}% '
              '@ index {}'.format(max_err * 100., loc))
        print('L2 norm of relative error of Jacobian: '
              '{:.2e} %'.format(err))
        err_jac_max[i] = max_err
        err_jac[i] = err

        # Thresholded error
        non_zero = np.where(abs(test_jacob) >
                            np.linalg.norm(test_jacob) / 1.e20
                            )[0]
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

    #plt.semilogy(state_data[:,1], err_jac_norm, 'o')
    #plt.xlabel('Temperature [K]')
    #plt.ylabel('Jacobian matrix norm error')
    #pp = PdfPages('Jacobian_error_norm.pdf')
    #pp.savefig()
    #pp.close()

    #plt.figure()
    #plt.semilogy(state_data[:,1], err_jac, 'o')
    #plt.xlabel('Temperature [K]')
    #plt.ylabel('Jacobian matrix relative error norm [%]')
    #pp = PdfPages('Jacobian_relative_error.pdf')
    #pp.savefig()
    #pp.close()

    # Save all error arrays
    np.savez('error_arrays.npz', err_dydt=err_dydt, err_jac_norm=err_jac_norm,
              err_jac=err_jac, err_jac_thr=err_jac_thr)

    # Cleanup all compiled files.
    for f in ['pyjacob.so', 'pyjacob_wrapper.c']:
        os.remove(f)
    if lang == 'cuda':
        for f in ['cu_pyjacob.so', 'pyjacob_cuda_wrapper.cpp']:
            os.remove(f)

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
                        help='Partially stirred reactor input file for '
                             'generating test data (e.g, pasr_input.yaml)')
    parser.add_argument('-dng', '--do_not_generate',
                        action='store_false',
                        dest='generate_jacob',
                        default=True,
                        help='Use this option to have the tester utilize '
                             'existing Jacobian files')
    parser.add_argument('-s', '--seed',
                        type=int,
                        default=None,
                        help='The seed to be used for random numbers')
    parser.add_argument('-p', '--pasr_output',
                        type=str,
                        default=None,
                        help='An optional saved .npy file that has the '
                             'resulting PaSR data (to speed testing)')
    args = parser.parse_args()
    test(args.lang, args.build_dir, args.mech, args.thermo, args.input,
         args.generate_jacob, args.seed, args.pasr_output
         )
