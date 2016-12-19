#modules
import cantera as ct
import numpy as np
import unittest
import loopy as lp

#system
import os

#local imports
from ..sympy.sympy_interpreter import load_equations
from ..core.mech_interpret import read_mech_ct

test_size = 10000

class storage(object):
    def __init__(self, conp_vars, conp_eqs, conv_vars,
        conv_eqs, gas, specs, reacs):
        self.conp_vars = conp_vars
        self.conp_eqs = conp_eqs
        self.conv_vars = conv_vars
        self.conv_eqs = conv_eqs
        self.gas = gas
        self.specs = specs
        self.reacs = reacs
        self.test_size = test_size

        #create states
        self.T = np.random.uniform(600, 2200, size=test_size)
        self.P = np.random.uniform(0.5, 50, size=test_size) * ct.one_atm
        self.Y = np.random.uniform(0, 1, size=(self.gas.n_species, test_size))
        self.concs = np.empty_like(self.Y)
        self.fwd_rate_constants = np.zeros((self.gas.n_reactions, test_size))

        #third body indicies
        thd_inds = np.array([i for i, x in enumerate(gas.reactions())
            if isinstance(x, ct.FalloffReaction) or
                isinstance(x, ct.ChemicallyActivatedReaction) or
                isinstance(x, ct.ThreeBodyReaction)])
        self.ref_thd = np.zeros((thd_inds.size, test_size))
        thd_eff_maps = []
        for i in thd_inds:
            default = gas.reaction(i).default_efficiency
            #fill all missing with default
            thd_eff_map = [default if j not in gas.reaction(i).efficiencies
                              else gas.reaction(i).efficiencies[j] for j in gas.species_names]
            thd_eff_maps.append(np.array(thd_eff_map))
        thd_eff_maps = np.array(thd_eff_maps)

        for i in range(test_size):
            self.gas.TPY = self.T[i], self.P[i], self.Y[:, i]
            self.concs[:, i] = self.gas.concentrations[:]

            #store various information
            self.fwd_rate_constants[:, i] = gas.forward_rate_constants[:]
            self.ref_thd[:, i] = np.dot(thd_eff_maps, self.concs[:, i])


class TestClass(unittest.TestCase):
    #global setup var
    _is_setup = False
    _store = None
    @property
    def store(self):
        return self._store
    @store.setter
    def store(self, val):
        self._store = val
    @property
    def is_setup(self):
        return self._is_setup
    @is_setup.setter
    def is_setup(self, val):
        self._is_setup = val

    def setUp(self):
        lp.set_caching_enabled(False)
        if not self.is_setup:
            #load equations
            conp_vars, conp_eqs = load_equations(True)
            conv_vars, conv_eqs = load_equations(False)
            self.dirpath = os.path.dirname(os.path.realpath(__file__))
            #load the gas
            gas = ct.Solution(os.path.join(self.dirpath, 'test.cti'))
            #the mechanism
            elems, specs, reacs = read_mech_ct(os.path.join(self.dirpath, 'test.cti'))
            self.store = storage(conp_vars, conp_eqs, conv_vars,
                conv_eqs, gas, specs, reacs)
            self.is_setup = True
