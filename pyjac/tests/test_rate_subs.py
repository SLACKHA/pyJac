#compatibility
from builtins import range

#system
import os
import filecmp

#local imports
from ..core.rate_subs import rate_const_kernel_gen, get_rate_eqn
from ..core.loopy_utils import auto_run, loopy_options, RateSpecialization
from ..utils import create_dir
from . import TestClass

#modules
from optionloop import OptionLoop
import cantera as ct
import numpy as np
from nose.plugins.attrib import attr

class SubTest(TestClass):
    def __populate(self):
        T = self.store.T
        ref_const = self.store.fwd_rate_constants
        return T, ref_const, ref_const.T.copy()

    def test_get_rate_eqs(self):
        eqs = {'conp' : self.store.conp_eqs, 
                'conv' : self.store.conv_eqs}
        pre, eq = get_rate_eqn(eqs)

        #first check they are equal
        assert 'exp(' + str(pre) + ')' == eq

        #second check the form
        assert eq == 'exp(logA + beta[i] * T_inv - Ta[i] * Ea[i])'

    @attr('long')
    def test_rate_constants(self):
        T = self.store.T
        ref_const = self.store.fwd_rate_constants
        ref_const_T = ref_const.T.copy()

        oploop = OptionLoop({'lang': ['opencl'],
            'width' : [4, None],
            'depth' : [4, None],
            'ilp' : [True, False],
            'unr' : [None, 4],
            'order' : ['cpu', 'gpu'],
            'device' : ['0:0', '1'],
            'ratespec' : [x for x in RateSpecialization],
            'ratespec_kernels' : [True, False]
            })

        for state in oploop:
            try:
                opt = loopy_options(**{x : state[x] for x in state if x != 'device'})
                knl = rate_const_kernel_gen(rate_eqn, reacs,
                                loopy_opt, test_size=None)

                ref = ref_ans if state['order'] == 'gpu' else ref_ans_T
                assert auto_run(knl, ref, device=state['device'],
                    T_arr=T)
            except Exception as e:
                if not(state['width'] and state['depth']):
                    raise e
