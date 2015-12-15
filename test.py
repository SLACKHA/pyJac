# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import re
import sys
import subprocess
import pickle
from argparse import ArgumentParser

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

    def __call__(self):
        """the ODE function, y' = f(t,y) """

        # State vector is [T, Y_1, Y_2, ... Y_K]
        # Only set state if something is input
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

    def __call__(self):
        """the ODE function, y' = f(t,y) """

        # State vector is [T, Y_1, Y_2, ... Y_K]
        # Only set state if something is input

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
    def __init__(self, pressure, fwd_spec_map):
        self.pres = pressure
        self.fwd_spec_map = fwd_spec_map
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

        y = np.hstack((gas.T, gas.Y[self.fwd_spec_map][:-1]))
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
    def __copy(self, B):
        A = np.empty_like(B)
        A[:] = B
        return A
    
    def check_numbers(self, build_dir, gas, filename='mechanism.h'):
        """Ensures numbers of species and forward reaction match.
        """
        with open(os.path.join(build_dir, filename), 'r') as file:
            n_spec = None
            n_reac = None
            for line in file.readlines():
                if n_spec is None:
                    match = re.search(r'^#define NSP (\d+)$', line)
                    if match:
                        n_spec = int(match.group(1))
                
                if n_reac is None:
                    match = re.search(r'^#define FWD_RATES (\d+)$', line)
                    if match:
                        n_reac = int(match.group(1))
                
                if n_spec is not None and n_reac is not None:
                    break
        
        if n_spec != gas.n_species:
            print('Error: species counts do not match between '
                  'mechanism.h and Cantera.')
            sys.exit(1)
        if n_reac != gas.n_reactions:
            print('Error: forward reaction counts do not match between '
                  'mechanism.h and Cantera.')
            sys.exit(1)
    
    def check_optimized(self, build_dir, gas, filename='mechanism.h'):
        self.fwd_spec_map = np.array(range(gas.n_species))
        with open(os.path.join(build_dir, filename), 'r') as file:
            opt = False
            last_spec = None
            for line in file.readlines():
                if 'Cache Optimized' in line:
                    opt = True
                match = re.search(r'^//last_spec (\d+)$', line)
                if match:
                    last_spec = int(match.group(1))
                if opt and last_spec is not None:
                    break

        self.last_spec = last_spec
        self.cache_opt = opt
        self.dydt_mask = np.array([0] + [x + 1 for x in range(gas.n_species) if x != last_spec])
        if self.cache_opt:
            with open(os.path.join(build_dir, 'optimized.pickle'), 'rb') as file:
                dummy = pickle.load(file)
                dummy = pickle.load(file)
                self.fwd_spec_map = np.array(pickle.load(file))
                self.fwd_rxn_map = np.array(pickle.load(file))
                self.back_spec_map = np.array(pickle.load(file))
                self.back_rxn_map = np.array(pickle.load(file))
        elif last_spec != gas.n_species - 1:
            #still need to treat it as a cache optimized
            self.cache_opt = True

            self.fwd_spec_map, self.back_spec_map = utils.get_species_mappings(gas.n_species, last_spec)
            self.fwd_spec_map = np.array(self.fwd_spec_map)
            self.back_spec_map = np.array(self.back_spec_map)

            self.fwd_rxn_map = np.array(range(gas.n_reactions))
            self.back_rxn_map = np.array(range(gas.n_reactions))
        else:
            self.fwd_spec_map = range(gas.n_species)
            self.back_spec_map = range(gas.n_species)
            self.fwd_rxn_map = np.array(range(gas.n_reactions))
            self.back_rxn_map = np.array(range(gas.n_reactions))

        #assign the rest
        n_spec = gas.n_species
        n_reac = gas.n_reactions

        self.fwd_dydt_map = np.array([0] + [x + 1 for x in self.fwd_spec_map])

        self.fwd_rev_rxn_map = np.array([i for i in self.fwd_rxn_map if
                gas.reaction(i).reversible])
        rev_reacs = self.fwd_rev_rxn_map.shape[0]
        self.back_rev_rxn_map = np.sort(self.fwd_rev_rxn_map)
        self.back_rev_rxn_map = np.array([np.where(self.fwd_rev_rxn_map == x)[0][0]
                                    for x in self.back_rev_rxn_map])
        self.fwd_rev_rxn_map = np.array([np.where(self.back_rev_rxn_map == x)[0][0]
                                    for x in range(rev_reacs)])

        self.fwd_pdep_map = [self.fwd_rxn_map[i] for i in range(n_reac)
                                if is_pdep(gas.reaction(self.fwd_rxn_map[i]))]
        pdep_reacs = len(self.fwd_pdep_map)
        self.back_pdep_map = sorted(self.fwd_pdep_map)
        self.back_pdep_map = np.array([self.fwd_pdep_map.index(x)
                                for x in self.back_pdep_map])
        self.fwd_pdep_map = np.array([np.where(self.back_pdep_map == x)[0][0]
                                for x in range(pdep_reacs)])

        self.back_dydt_map = np.array([0] + [x + 1 for x in self.back_spec_map])

    def __init__(self, build_dir, gas, module_name='pyjacob', filename='mechanism.h'):
        self.check_numbers(build_dir, gas, filename)
        self.check_optimized(build_dir, gas, filename)
        self.pyjac = __import__(module_name)

    def eval_conc(self, temp, pres, mass_frac, conc):
        mw_avg = 0
        rho = 0
        if self.cache_opt:
            test_mass_frac = self.__copy(mass_frac[self.fwd_spec_map])
            self.pyjac.py_eval_conc(temp, pres, test_mass_frac, mw_avg, rho, conc)
            conc[:] = conc[self.back_spec_map]
        else:
            self.pyjac.py_eval_conc(temp, pres, mass_frac, mw_avg, rho, conc)
    def eval_rxn_rates(self, temp, pres, conc, fwd_rates, rev_rates):
        if self.cache_opt:
            test_conc = self.__copy(conc[self.fwd_spec_map])
            self.pyjac.py_eval_rxn_rates(temp, pres, test_conc,
             fwd_rates, rev_rates)
            fwd_rates[:] = fwd_rates[self.back_rxn_map]
            rev_rates[:] = rev_rates[self.back_rev_rxn_map]
        else:
            self.pyjac.py_eval_rxn_rates(temp, pres, conc, fwd_rates, rev_rates)
    def get_rxn_pres_mod(self, temp, pres, conc, pres_mod):
        if self.cache_opt:
            test_conc = self.__copy(conc[self.fwd_spec_map])
            self.pyjac.py_get_rxn_pres_mod(temp, pres, test_conc, pres_mod)
            pres_mod[:] = pres_mod[self.back_pdep_map]
        else:
            self.pyjac.py_get_rxn_pres_mod(temp, pres, conc, pres_mod)
    def eval_spec_rates(self, fwd_rates, rev_rates, pres_mod, spec_rates):
        if self.cache_opt:
            test_fwd = self.__copy(fwd_rates[self.fwd_rxn_map])
            test_rev = self.__copy(rev_rates[self.fwd_rev_rxn_map])
            test_pdep = self.__copy(pres_mod[self.fwd_pdep_map])
            self.pyjac.py_eval_spec_rates(test_fwd, test_rev, test_pdep, spec_rates)
            spec_rates[:] = spec_rates[self.back_spec_map]
        else:
            self.pyjac.py_eval_spec_rates(fwd_rates, rev_rates, pres_mod, spec_rates)
    def dydt(self, t, pres, y, dydt):
        if self.cache_opt:
            test_y = self.__copy(y[self.fwd_dydt_map])
            test_dydt = np.zeros_like(test_y)
            self.pyjac.py_dydt(t, pres, test_y, test_dydt)
            dydt[self.dydt_mask] = test_dydt[self.back_dydt_map[self.dydt_mask]]
        else:
            self.pyjac.py_dydt(t, pres, y, dydt)
    def eval_jacobian(self, t, pres, y, jacob):
        if self.cache_opt:
            test_y = self.__copy(y[self.fwd_dydt_map][:])
            self.pyjac.py_eval_jacobian(t, pres, test_y, jacob)
        else:
            self.pyjac.py_eval_jacobian(t, pres, y, jacob)
    def update(self, index):
        self.index = index


class cupyjac_evaluator(cpyjac_evaluator):
    def __init__(self, build_dir, gas, state_data):
        super(cupyjac_evaluator, self).__init__(build_dir, gas, 'cu_pyjacob', 'mechanism.cuh')

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

        num_rev = np.array([rxn.reversible for rxn in gas.reactions()]).sum()
        num_pdep = np.array([is_pdep(rxn) for rxn in gas.reactions()]).sum()
        cuda_state = state_data[:, 1:]
        num_cond = cuda_state.shape[0]
        #init vectors
        test_conc = czeros((num_cond, gas.n_species))
        test_fwd_rates = czeros((num_cond,gas.n_reactions))
        test_rev_rates = czeros((num_cond,num_rev))
        test_pres_mod = czeros((num_cond,num_pdep))
        test_spec_rates = czeros((num_cond,gas.n_species))
        test_dydt = czeros((num_cond, gas.n_species + 1))
        test_jacob = czeros((num_cond,(gas.n_species) * (gas.n_species)))

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
        self.test_conc = reshaper(test_conc, (num_cond, gas.n_species), self.back_spec_map)
        self.test_fwd_rates = reshaper(test_fwd_rates, (num_cond, gas.n_reactions), self.back_rxn_map)
        self.test_rev_rates = reshaper(test_rev_rates, (num_cond, num_rev), self.back_rev_rxn_map)
        self.test_pres_mod = reshaper(test_pres_mod, (num_cond, num_pdep), self.back_pdep_map)
        self.test_spec_rates = reshaper(test_spec_rates, (num_cond,gas.n_species), self.back_spec_map)
        self.test_dydt = reshaper(test_dydt, (num_cond, gas.n_species + 1), self.back_dydt_map)
        self.test_jacob = reshaper(test_jacob, (num_cond, (gas.n_species) * (gas.n_species)))
        self.index = 0

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


class tchem_evaluator(cpyjac_evaluator):
    def __init__(self, build_dir, gas, state_data, mechfile, thermofile,
                 module_name='py_tchem', filename='mechanism.h'
                 ):
        self.tchem = __import__(module_name)

        if thermofile == None:
            thermofile = mechfile

        # TChem needs array of full species mass fractions
        self.y_mask = np.array([0] + [x + 2 for x in range(gas.n_species)])
        def czeros(shape):
            arr = np.zeros(shape)
            return arr.flatten(order='c')
        def reshaper(arr, shape, reorder=None):
            arr = arr.reshape(shape, order='c').astype(np.dtype('d'), order='c')
            if reorder is not None:
                arr = arr[:, reorder]
            return arr

        states = state_data[:, 1:]
        num_cond = states.shape[0]
        #init vectors
        test_conc = czeros((num_cond, gas.n_species))
        test_fwd_rates = czeros((num_cond,gas.n_reactions))
        test_rev_rates = czeros((num_cond,gas.n_reactions))
        test_spec_rates = czeros((num_cond,gas.n_species))
        test_dydt = czeros((num_cond, gas.n_species + 1))
        test_jacob = czeros((num_cond, (gas.n_species) * (gas.n_species)))

        pres = states[:, 1].flatten(order='c')
        y_dummy = states[:, [x for x in self.y_mask]
                         ].flatten(order='c').astype(np.dtype('d'), order='c')

        self.tchem.py_eval_jacobian(mechfile, thermofile, num_cond,
                                    pres, y_dummy, test_conc, test_fwd_rates,
                                    test_rev_rates, test_spec_rates,
                                    test_dydt, test_jacob
                                    )

        #reshape for comparison
        self.test_conc = reshaper(test_conc, (num_cond, gas.n_species))
        self.test_fwd_rates = reshaper(test_fwd_rates,
                                       (num_cond, gas.n_reactions)
                                       )
        self.test_rev_rates = reshaper(test_rev_rates,
                                       (num_cond, gas.n_reactions)
                                       )
        self.test_spec_rates = reshaper(test_spec_rates,
                                        (num_cond, gas.n_species)
                                        )
        self.test_dydt = reshaper(test_dydt, (num_cond, gas.n_species + 1))
        self.test_jacob = reshaper(test_jacob, (num_cond,
                                   (gas.n_species) * (gas.n_species))
                                   )
        self.index = 0

    def get_conc(self, conc):
        conc[:] = self.test_conc[self.index, :]
    def get_rxn_rates(self, fwd_rates, rev_rates):
        fwd_rates[:] = self.test_fwd_rates[self.index, :]
        rev_rates[:] = self.test_rev_rates[self.index, :]
    def get_spec_rates(self, spec_rates):
        spec_rates[:] = self.test_spec_rates[self.index, :]
    def get_dydt(self, dydt):
        dydt[:] = self.test_dydt[self.index, :-1]
    def get_jacobian(self, jacob):
        jacob[:] = self.test_jacob[self.index, :]


def test(lang, build_dir, mech_filename, therm_filename=None,
         pasr_input_file='pasr_input.yaml', generate_jacob=True,
         compile_jacob=True, seed=None, pasr_output_file=None, last_spec=None,
         cache_optimization=False, no_shared=False, tchem_flag=False):
    """Compares pyJac results against Cantera (and optionally TChem) using
    state data from PaSR simulations.

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

    if generate_jacob and not compile_jacob:
        print('Warning: Jacobian files being generated but not compiled.')

    if generate_jacob:
        #remove the old jaclist
        try:
            os.remove(os.path.join(build_dir, 'jacobs/jac_list_c'))
        except:
            pass
        try:
            os.remove(os.path.join(build_dir, 'jacobs/jac_list_cuda'))
        except:
            pass

        # Create Jacobian and supporting source code files
        create_jacobian(lang, mech_filename, therm_name=therm_filename,
                        optimize_cache=cache_optimization, build_path=build_dir,
                        no_shared=no_shared, last_spec=last_spec
                        )
        if lang == 'cuda':
            #if it's cuda, we need to make sure the c files
            #are generated using the same manner
            #as we're going to use the c interface for testing the jacobian
            create_jacobian('c', mech_filename, therm_name=therm_filename,
                        optimize_cache=cache_optimization, build_path=build_dir,
                        no_shared=no_shared, last_spec=last_spec
                        )

    # Interpret reaction mechanism file, depending on Cantera or
    # Chemkin format.
    if not mech_filename.endswith(tuple(['.cti', '.xml'])):
        # Chemkin format; need to convert first.
        ck_mech_filename = mech_filename
        mech_filename = convert_mech(mech_filename, therm_filename)
    elif tchem_flag:
        tchem_flag = False
        print('TChem validation disabled; '
              'not compatible with Cantera mechanism.'
              )

    if compile_jacob:
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
                                   'pyjacob_cuda_setup.py',
                                   'build_ext', '--inplace'
                                   ])

        try:
            os.remove('py_tchem.so')
        except:
            pass
        if tchem_flag:
            subprocess.check_call(['python2.7', os.getcwd() + os.path.sep +
                                   'pytchem_setup.py',
                                   'build_ext', '--inplace'
                                   ])

    # get the cantera object
    gas = ct.Solution(mech_filename)
    pmod = any([is_pdep(rxn) for rxn in gas.reactions()])
    rev = any(rxn.reversible for rxn in gas.reactions())

    # Now generate data and check results

    # Need to get reversible reactions and those for which
    # pressure modification applies.
    idx_rev = [i for i, rxn in enumerate(gas.reactions()) if rxn.reversible]
    idx_pmod = [i for i, rxn in enumerate(gas.reactions()) if is_pdep(rxn)]
    # Index of element in idx_pmod that corresponds to reversible reaction
    idx_pmod_rev = [
        i for i, idx in enumerate(idx_pmod) if gas.reaction(idx).reversible
        ]
    # Index of reversible reaction that also has pressure dependent
    # modification
    idx_rev_pmod = [
                i for i, idx in enumerate(idx_rev) if
                isinstance(gas.reaction(idx), ct.ThreeBodyReaction) or
                isinstance(gas.reaction(idx), ct.FalloffReaction) or
                isinstance(gas.reaction(idx), ct.ChemicallyActivatedReaction)
                ]

    # Check mechanism for Plog or Chebyshev reactions... if any, can't
    # compare Jacobian to TChem
    if any([isinstance(rxn, ct.PlogReaction) or
            isinstance(rxn, ct.ChebyshevReaction) for rxn in gas.reactions()
            ]):
        print('TChem comparison disabled; '
              'not compatible with Plog or Chebyshev reactions.'
              )
        tchem_flag = False

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

    # The test data from Cantera does not use the 1 - Yn approach
    # and thus mass is not explicitly, strictly conserved
    # Although this effect is typically very small e.g. O(1e-16)
    # it can have a large effect on the error in certain cases
    # e.g. if you designate a radical like OH as the last species
    # and that radical also has a mass fraction on that order of
    # magnitude
    # --------------------------------------------------------------
    # Therefore to maintain 100% consistancy and to ensure the mass
    # fractions we put into our code (which does not normalize)
    # are strictly equal to unity, we renormalize them now by running
    # them through Cantera again

    for i in range(len(state_data)):
        gas.TPY = state_data[i, 1], state_data[i, 2], state_data[i, 3:]
        state_data[i, 3:] = gas.Y

    if lang == 'cuda':
        pyjacob = cupyjac_evaluator(build_dir, gas, state_data)
    else:
        pyjacob = cpyjac_evaluator(build_dir, gas)

    tchem_jac = None
    if tchem_flag:
        tchem_jac = tchem_evaluator(build_dir, gas, state_data,
                                    ck_mech_filename, therm_filename
                                    )

    num_trials = len(state_data)

    err_dydt = np.zeros(num_trials)
    err_dydt_zero = np.zeros(num_trials)
    err_jac_norm = np.zeros(num_trials)
    err_jac = np.zeros(num_trials)
    err_jac_thr = np.zeros(num_trials)
    err_jac_max = np.zeros(num_trials)
    err_jac_zero = np.zeros(num_trials)
    err_jac_tchem = np.zeros(num_trials)

    for i, state in enumerate(state_data):
        #update index in case we're using cuda
        pyjacob.update(i)

        if tchem_flag:
            tchem_jac.update(i)

        temp = state[1]
        pres = state[2]
        mass_frac = state[3:]

        ajac = analytic_eval_jacob(pres, pyjacob.fwd_spec_map)

        gas.TPY = temp, pres, mass_frac

        #get conc
        test_conc = np.zeros(gas.n_species)
        pyjacob.eval_conc(temp, pres, mass_frac, test_conc)

        #get reaction rates of production
        test_fwd_rates = np.zeros(gas.n_reactions)
        test_rev_rates = np.zeros(len(idx_rev))
        test_pres_mod = np.zeros(len(idx_pmod))
        pyjacob.eval_rxn_rates(temp, pres, test_conc,
                                  test_fwd_rates, test_rev_rates
                                  )
        pyjacob.get_rxn_pres_mod(temp, pres, test_conc, test_pres_mod)

        # Species production rates
        test_spec_rates = np.zeros(gas.n_species)
        pyjacob.eval_spec_rates(test_fwd_rates, test_rev_rates,
                                test_pres_mod, test_spec_rates
                                )

        # Derivative source terms terms
        test_dydt = np.zeros(gas.n_species + 1)
        y_dummy = np.hstack((temp, mass_frac))
        pyjacob.dydt(0, pres, y_dummy, test_dydt)
        ode = ReactorConstPres(gas)

        # Jacobian matrix
        test_jacob = np.zeros((gas.n_species) * (gas.n_species))
        pyjacob.eval_jacobian(0, pres, y_dummy, test_jacob)
        jacob = ajac.eval_jacobian(gas, 6)


        print()
        print('Testing condition {} / {}'.format(i + 1, num_trials))

        # Calculate error in concentrations
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

        # Calculate error in species net production rates
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

        # Calculate error in derivative source terms

        #need to mask the resulting dydt vectors to avoid comparison
        #of the new last species
        t_dydt = test_dydt[pyjacob.dydt_mask]
        ode_dydt = ode()[pyjacob.dydt_mask]

        non_zero = np.where(t_dydt != 0.)[0]
        zero = np.where(t_dydt == 0.)[0]
        err = abs((t_dydt[non_zero] - ode_dydt[non_zero]) /
                  ode_dydt[non_zero]
                  )
        max_err = np.max(err)
        loc = pyjacob.dydt_mask[non_zero[np.argmax(err)]]
        err = np.linalg.norm(err) * 100.
        err_dydt[i] = err
        print('L2 norm relative error of non-zero dydt: {:.2e} %'.format(err))
        print('Max error in non-zero dydt: {:.2e}% '
              '@ index {}'.format(max_err * 100., loc)
              )
        err = np.linalg.norm(t_dydt[zero] - ode_dydt[zero])
        err_dydt_zero[i] = err
        print('L2 norm difference of "zero" dydt: {:.2e}'.format(err))

        # Calculate error in Jacobian matrix
        non_zero = np.where(abs(test_jacob) > 1.e-30)[0]
        zero = np.where(test_jacob == 0.)[0]
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

        # Compare against TChem, if enabled
        if tchem_flag:
            tchem_conc = np.zeros(gas.n_species)
            tchem_jac.get_conc(tchem_conc)
            non_zero = np.where(tchem_conc > 0.)[0]
            err = abs((test_conc[non_zero] - tchem_conc[non_zero]) /
                      tchem_conc[non_zero]
                      )
            max_err = np.max(err)
            loc = non_zero[np.argmax(err)]
            err = np.linalg.norm(err) * 100.
            print('L2 norm difference with TChem concentration: {:.2e} %'.format(err))
            print('Max difference with TChem concentration: {:.2e} % @ species {}'
                .format(max_err * 100., loc))

            tchem_fwd_rates = np.zeros(gas.n_reactions)
            tchem_rev_rates = np.zeros(gas.n_reactions)
            tchem_jac.get_rxn_rates(tchem_fwd_rates, tchem_rev_rates)
            non_zero = np.where(tchem_fwd_rates > 0.)[0]
            err = abs((test_fwd_rates[non_zero] - tchem_fwd_rates[non_zero]) /
                      tchem_fwd_rates[non_zero]
                      )
            max_err = np.max(err)
            loc = non_zero[np.argmax(err)]
            err = np.linalg.norm(err) * 100.
            print('L2 norm difference with TChem forward reaction rates: '
                  '{:.2e}%'.format(err)
                  )
            print('Max difference with TChem forward reaction rates: '
                  '{:.2e}% @ reaction {}'.format(max_err * 100., loc)
                  )

            if idx_rev:
                non_zero = np.where(tchem_rev_rates > 0.)[0]
                err = abs((test_rev_rates[non_zero] -
                          (tchem_rev_rates[idx_rev])[non_zero]) /
                          (tchem_rev_rates[idx_rev])[non_zero]
                          )
                max_err = np.max(err)
                loc = non_zero[np.argmax(err)]
                err = np.linalg.norm(err) * 100.
                print('L2 norm difference with TChem reverse reaction rates: '
                      '{:.2e}%'.format(err)
                      )
                print('Max difference with TChem reverse reaction rates: '
                      '{:.2e}% @ reaction {}'.format(max_err * 100., loc)
                      )

            tchem_spec_rates = np.zeros(gas.n_species)
            tchem_jac.get_spec_rates(tchem_spec_rates)
            non_zero = np.where(tchem_spec_rates != 0.)[0]
            err = abs((test_spec_rates[non_zero] - tchem_spec_rates[non_zero])
                       / tchem_spec_rates[non_zero]
                      )
            max_err = np.max(err)
            loc = non_zero[np.argmax(err)]
            err = np.linalg.norm(err) * 100.
            print('L2 norm relative difference with TChem net production rates: '
                  '{:.2e} %'.format(err)
                  )
            print('Max difference with TChem net production rates: {:.2e}% '
                  '@ species {}'.format(max_err * 100., loc)
                  )

            tchem_dydt = np.zeros(gas.n_species)
            tchem_jac.get_dydt(tchem_dydt)
            non_zero = np.where(tchem_dydt != 0.)[0]
            err = abs((test_dydt[non_zero] - tchem_dydt[non_zero]) /
                       tchem_dydt[non_zero]
                      )
            max_err = np.max(err)
            loc = non_zero[np.argmax(err)]
            err = np.linalg.norm(err) * 100.
            print('L2 norm relative difference with TChem dydt: '
                  '{:.2e} %'.format(err)
                  )
            print('Max difference with TChem dydt: {:.2e}% '
                  '@ species {}'.format(max_err * 100., loc)
                  )

            tchem_jacob = np.zeros(gas.n_species * gas.n_species)
            tchem_jac.get_jacobian(tchem_jacob)
            non_zero = np.where(abs(tchem_jacob) > 1.e-30)[0]

            err = abs((test_jacob[non_zero] - tchem_jacob[non_zero]) /
                      tchem_jacob[non_zero]
                      )
            loc = non_zero[np.argmax(err)]
            print('Max difference with non-zero TChem Jacobian: {:.2e}% '
                  '@ index {}'.format(np.max(err) * 100., loc))
            err = np.linalg.norm(err) * 100.
            print('L2 norm of relative difference with TChem Jacobian: '
                  '{:.2e} %'.format(err))
            err_jac_tchem[i] = err


    # Save all error arrays
    np.savez('error_arrays.npz',
             err_dydt=err_dydt, err_jac_norm=err_jac_norm,
             err_jac=err_jac, err_jac_thr=err_jac_thr
             )

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

    # Cleanup TChem crud
    if tchem_flag:
        for f in ['periodictable.dat', 'kmod.echo', 'kmod.err', 'kmod.list',
                  'kmod.out', 'math_elem.dat', 'math_falloff.dat',
                  'math_nasapol7.dat', 'math_reac.dat', 'math_spec.dat',
                  'math_trdbody.dat']:
            os.remove(f)

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
                             'source files'
                        )
    parser.add_argument('-b', '--build_dir',
                        type=str,
                        default='out' + os.path.sep,
                        help='The directory the jacob/rates/tester'
                             ' files will be generated and built in'
                        )
    parser.add_argument('-m', '--mech',
                        type=str,
                        required=True,
                        help='Mechanism filename (e.g., mech.dat)'
                        )
    parser.add_argument('-t', '--thermo',
                        type=str,
                        default=None,
                        help='Thermodynamic database filename (e.g., '
                             'therm.dat), or nothing if in mechanism'
                        )
    parser.add_argument('-i', '--input',
                        type=str,
                        default='pasr_input.yaml',
                        help='Partially stirred reactor input file for '
                             'generating test data (e.g, pasr_input.yaml)'
                        )
    parser.add_argument('-dng', '--do_not_generate',
                        action='store_false',
                        dest='generate_jacob',
                        default=True,
                        help='Use this option to have the tester use '
                             'existing Jacobian files'
                        )
    parser.add_argument('-s', '--seed',
                        type=int,
                        default=None,
                        help='The seed to be used for random numbers'
                        )
    parser.add_argument('-p', '--pasr_output',
                        type=str,
                        default=None,
                        help='An optional saved .npy file that has the '
                             'resulting PaSR data (to speed testing)'
                             )
    parser.add_argument('-ls', '--last_spec',
                        type=str,
                        default=None,
                        help='The last species, to pass to pyJac'
                        )
    parser.add_argument('-co', '--cache_optimization',
                        default=False,
                        action='store_true',
                        help='Use to enable cache optimization in pyJac'
                        )
    parser.add_argument('-nosmem', '--no_shared',
                        default=False,
                        action='store_true',
                        help='Use to disable shared memory usage in pyJac '
                             '(CUDA only)'
                        )
    parser.add_argument('-tc', '--tchem',
                        default=False,
                        action='store_true',
                        help='Activate TChem comparison (false by default)'
                        )
    parser.add_argument('-dnc', '--do_not_compile',
                        action='store_false',
                        dest='compile_jacob',
                        default=True,
                        help='Use this option to force the tester to use '
                             'previously compiled files'
                        )
    args = parser.parse_args()
    test(args.lang, args.build_dir, args.mech, args.thermo, args.input,
         args.generate_jacob, args.compile_jacob, args.seed, args.pasr_output,
         args.last_spec, args.cache_optimization, args.no_shared, args.tchem
         )
