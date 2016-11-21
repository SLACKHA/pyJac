#compatibility
from builtins import range

#system
import os
import filecmp
from collections import OrderedDict

#local imports
from ..core.rate_subs import rate_const_kernel_gen, get_rate_eqn, assign_rates
from ..core.loopy_utils import auto_run, loopy_options, RateSpecialization
from ..utils import create_dir
from . import TestClass
from ..core.reaction_types import reaction_type

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
        assert eq == 'exp(A[i] - T_inv*Ta[i] + beta[i]*logT)'

    def test_assign_rates(self):
        reacs = self.store.reacs
        result = assign_rates(reacs, RateSpecialization.fixed)

        #test rate type
        assert np.all(result['simple']['type'] == 0)

        #import gas in cantera for testing
        gas = ct.Solution('test.cti')

        def __tester(result):
            #test return value
            assert 'simple' in result and 'cheb' in result and 'plog' in result

            #test num, mask, and offset
            plog_inds, plog_reacs = zip(*[(i, x) for i, x in enumerate(gas.reactions())
                    if isinstance(x, ct.PlogReaction)])
            assert result['plog']['num'] == len(plog_inds)
            assert np.array_equal(result['plog']['mask'], np.array(plog_inds))
            if np.all(np.diff(result['plog']['mask']) == 1):
                assert result['plog']['offset'] == result['plog']['mask'][0]
            else:
                assert result['plog']['offset'] is None

            cheb_inds, cheb_reacs = zip(*[(i, x) for i, x in enumerate(gas.reactions())
                    if isinstance(x, ct.ChebyshevReaction)])
            assert result['cheb']['num'] == len(cheb_inds)
            assert np.array_equal(result['cheb']['mask'], np.array(cheb_inds))
            if np.all(np.diff(result['cheb']['mask']) == 1):
                assert result['cheb']['offset'] == result['cheb']['mask'][0]
            else:
                assert result['cheb']['offset'] is None

            simple_inds = sorted(list(set(range(gas.n_reactions)).difference(
                set(plog_inds).union(set(cheb_inds)))))
            assert result['simple']['num'] == len(simple_inds)
            assert np.array_equal(result['simple']['mask'], np.array(simple_inds))
            if np.all(np.diff(result['simple']['mask']) == 1):
                assert result['simple']['offset'] == result['simple']['mask'][0]
            else:
                assert result['simple']['offset'] is None

        __tester(result)

        result = assign_rates(reacs, RateSpecialization.hybrid)

        def __get_vals(reac):
            try:
                Ea = reac.rate.activation_energy
                b = reac.rate.temperature_exponent
            except:
                if isinstance(reac, ct.FalloffReaction) and not isinstance(reac, ct.ChemicallyActivatedReaction):
                    Ea = reac.high_rate.activation_energy
                    b = reac.high_rate.temperature_exponent
                else:
                    Ea = reac.low_rate.activation_energy
                    b = reac.low_rate.temperature_exponent
            return Ea, b

        #test rate type
        rtypes = []
        for reac in gas.reactions():
            if not (isinstance(reac, ct.PlogReaction) or isinstance(reac, ct.ChebyshevReaction)):
                Ea, b = __get_vals(reac)
                if Ea == 0 and b == 0:
                    rtypes.append(0)
                elif Ea == 0 and int(b) == b:
                    rtypes.append(1)
                else:
                    rtypes.append(2)
        assert np.array_equal(result['simple']['type'], np.array(rtypes))
        __tester(result)

        result = assign_rates(reacs, RateSpecialization.full)

        #test rate type
        rtypes = []
        for reac in gas.reactions():
            if not (isinstance(reac, ct.PlogReaction) or isinstance(reac, ct.ChebyshevReaction)):
                Ea, b = __get_vals(reac)
                if Ea == 0 and b == 0:
                    rtypes.append(0)
                elif Ea == 0 and int(b) == b:
                    rtypes.append(1)
                elif Ea == 0:
                    rtypes.append(2)
                elif b == 0:
                    rtypes.append(3)
                else:
                    rtypes.append(4)
        assert np.array_equal(result['simple']['type'], np.array(rtypes))
        __tester(result)


    @attr('long')
    def test_rate_constants(self):
        T = self.store.T
        ref_const = self.store.fwd_rate_constants
        ref_const_T = ref_const.T.copy()

        eqs = {'conp' : self.store.conp_eqs,
                'conv' : self.store.conv_eqs}
        pre, eq = get_rate_eqn(eqs)

        oploop = OptionLoop(OrderedDict([('lang', ['opencl']),
            ('width', [4, None]),
            ('depth', [4, None]),
            ('ilp', [True, False]),
            ('unr', [None, 4]),
            ('order', ['C', 'F']),
            ('device', ['0:0', '1']),
            ('rate_spec', [x for x in RateSpecialization]),
            ('rate_spec_kernels', [True, False])
            ]))

        reacs = self.store.reacs
        compare_mask = [i for i, x in enumerate(reacs) if x.match((reaction_type.elementary,))]

        for i, state in enumerate(oploop):
            if state['width'] is not None and state['depth'] is not None:
                continue
            opt = loopy_options(**{x : state[x] for x in
                state if x != 'device'})
            knl = rate_const_kernel_gen(eq, pre, reacs, opt,
                test_size=self.store.test_size)

            ref = ref_const if state['order'] == 'F' else ref_const_T
            assert auto_run(knl, ref, device=state['device'],
                T_arr=T, compare_mask=compare_mask,
                compare_axis=1 if state['order'] == 'C' else 0)
