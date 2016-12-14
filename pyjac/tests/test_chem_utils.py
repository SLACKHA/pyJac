#compatibility
from builtins import range

#system
import os
import filecmp
from collections import OrderedDict

#local imports
from ..core.rate_subs import polyfit_kernel_gen, write_chem_utils
from ..loopy.loopy_utils import auto_run, loopy_options
from ..utils import create_dir
from . import TestClass

#modules
from optionloop import OptionLoop
import cantera as ct
import numpy as np
from nose.plugins.attrib import attr

class SubTest(TestClass):
    def __subtest(self, T, ref_ans, ref_ans_T,
        varname, nicename, eqs):
        oploop = OptionLoop(OrderedDict([('lang', ['opencl']),
            ('width', [4, None]),
            ('depth', [4, None]),
            ('ilp', [True, False]),
            ('unr', [None, 4]),
            ('device', ['0:0', '1'])]))

        specs = self.store.specs
        test_size = self.store.test_size
        for i, state in enumerate(oploop):
            try:
                opt = loopy_options(**{x : state[x] for x in state if x != 'device'})
                knl = polyfit_kernel_gen(varname, nicename, eqs, specs,
                                            opt, test_size=test_size)
                ref = ref_ans_T if state['depth'] else ref_ans
                assert auto_run(knl, ref, device=state['device'],
                    T_arr=T)
            except Exception as e:
                if not(state['width'] and state['depth']):
                    raise e

    def __populate(self, func):
        T = self.store.T
        specs = self.store.specs
        test_size = self.store.test_size
        ref_ans = np.zeros((len(specs), test_size))
        for i in range(test_size):
            for j in range(len(specs)):
                ref_ans[j, i] = func(j, i, T)

        ref_ans_T = ref_ans.T.copy()
        return T, ref_ans, ref_ans_T

    @attr('long')
    def test_cp(self):
        T, ref_ans, ref_ans_T = self.__populate(lambda j, i, T: self.store.gas.species(
            j).thermo.cp(T[i]))
        self.__subtest(T, ref_ans, ref_ans_T, '{C_p}[k]',
            'cp', self.store.conp_eqs)

    @attr('long')
    def test_cv(self):
        T, ref_ans, ref_ans_T = self.__populate(lambda j, i, T: self.store.gas.species(
            j).thermo.cp(T[i]) - ct.gas_constant)
        ref_ans_T = ref_ans.T.copy()

        self.__subtest(T, ref_ans, ref_ans_T, '{C_v}[k]',
            'cv', self.store.conp_eqs)

    @attr('long')
    def test_h(self):
        T, ref_ans, ref_ans_T = self.__populate(lambda j, i, T: self.store.gas.species(
            j).thermo.h(T[i]))
        self.__subtest(T, ref_ans, ref_ans_T, 'H[k]',
            'h', self.store.conp_eqs)

    @attr('long')
    def test_u(self):
        T, ref_ans, ref_ans_T = self.__populate(lambda j, i, T: self.store.gas.species(
            j).thermo.h(T[i]) - T[i] * ct.gas_constant)
        self.__subtest(T, ref_ans, ref_ans_T, 'U[k]',
            'u', self.store.conv_eqs)

    def test_write_chem_utils(self):
        script_dir = os.path.abspath(os.path.dirname(__file__))
        build_dir = os.path.join(script_dir, 'out')
        create_dir(build_dir)
        write_chem_utils(build_dir, self.store.specs,
            {'conp' : self.store.conp_eqs, 'conv' : self.store.conv_eqs},
                loopy_options(lang='opencl',
                    width=None, depth=None, ilp=False,
                    unr=None))

        assert filecmp.cmp(os.path.join(build_dir, 'chem_utils.oh'),
                        os.path.join(script_dir, 'blessed', 'chem_utils.oh'))
        assert filecmp.cmp(os.path.join(build_dir, 'chem_utils.co'),
                        os.path.join(script_dir, 'blessed', 'chem_utils.co'))
