#compatibility
from builtins import range

#system
import os
import filecmp
from collections import OrderedDict

#local imports
from ..core.rate_subs import (rate_const_kernel_gen, get_rate_eqn, assign_rates,
    get_simple_arrhenius_rates, get_plog_arrhenius_rates, get_cheb_arrhenius_rates,
    make_rateconst_kernel, apply_rateconst_vectorization)
from ..loopy.loopy_utils import (auto_run, loopy_options, RateSpecialization, get_code,
    get_target)
from ..utils import create_dir
from . import TestClass
from ..core.reaction_types import reaction_type, falloff_form, thd_body_type

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
        pre = get_rate_eqn(eqs)

        #check the form
        assert 'exp(' + str(pre) + ')' == 'exp(A[i] - T_inv*Ta[i] + beta[i]*logT)'

        pre = get_rate_eqn(eqs, index='j')

        #check the form
        assert 'exp(' + str(pre) + ')' == 'exp(A[j] - T_inv*Ta[j] + beta[j]*logT)'

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

            #test num, map
            plog_inds, plog_reacs = zip(*[(i, x) for i, x in enumerate(gas.reactions())
                    if isinstance(x, ct.PlogReaction)])
            assert result['plog']['num'] == len(plog_inds)
            assert np.allclose(result['plog']['map'], np.array(plog_inds))
            #check values
            assert np.array_equal(result['plog']['num_P'], [len(p.rates) for p in plog_reacs])
            for i, reac_params in enumerate(result['plog']['params']):
                act_energy_ratios = []
                for j, rates in enumerate(plog_reacs[i].rates):
                    assert np.isclose(reac_params[j][0], rates[0])
                    assert np.isclose(reac_params[j][1], rates[1].pre_exponential_factor)
                    assert np.isclose(reac_params[j][2], rates[1].temperature_exponent)
                    act_energy_ratios.append(reac_params[j][3] / rates[1].activation_energy)
                #for the activation energies, we simply check that the ratios are the same
                assert np.all(np.isclose(act_energy_ratios, act_energy_ratios[0]))

            cheb_inds, cheb_reacs = zip(*[(i, x) for i, x in enumerate(gas.reactions())
                    if isinstance(x, ct.ChebyshevReaction)])
            assert result['cheb']['num'] == len(cheb_inds)
            assert np.allclose(result['cheb']['map'], np.array(cheb_inds))

            simple_inds = sorted(list(set(range(gas.n_reactions)).difference(
                set(plog_inds).union(set(cheb_inds)))))
            assert result['simple']['num'] == len(simple_inds)
            assert np.allclose(result['simple']['map'], np.array(simple_inds))

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
        assert np.allclose(result['simple']['type'], np.array(rtypes))
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
        assert np.allclose(result['simple']['type'], np.array(rtypes))
        __tester(result)

        #test the thd / falloff / chem assignments
        assert np.allclose(result['fall']['map'],
            [i for i, x in enumerate(gas.reactions()) if (isinstance(x,
                ct.FalloffReaction) or isinstance(x, ct.ChemicallyActivatedReaction))])
        fall_reacs = [gas.reaction(y) for y in result['fall']['map']]
        #test fall vs chemically activated
        assert np.allclose(result['fall']['type'],
            np.array([reaction_type.fall if (isinstance(x, ct.FalloffReaction) and not
                isinstance(x, ct.ChemicallyActivatedReaction)) else reaction_type.chem for x in
                fall_reacs], dtype=np.int32))
        #test blending func
        blend_types = []
        for x in fall_reacs:
            if isinstance(x.falloff, ct.TroeFalloff):
                blend_types.append(falloff_form.troe)
            elif isinstance(x.falloff, ct.SriFalloff):
                blend_types.append(falloff_form.sri)
            else:
                blend_types.append(falloff_form.lind)
        assert np.allclose(result['fall']['blend'], np.array(blend_types, dtype=np.int32))
        #and finally test the third body stuff
        #test map
        third_reac_inds = [i for i, x in enumerate(gas.reactions()) if (isinstance(x,
                ct.FalloffReaction) or isinstance(x, ct.ChemicallyActivatedReaction)
                or isinstance(x, ct.ThreeBodyReaction))]
        assert np.allclose(result['thd']['map'], third_reac_inds)
        #construct types, efficiencies, species, and species numbers
        thd_type = []
        thd_eff = []
        thd_sp = []
        thd_sp_num = []
        for ind in third_reac_inds:
            eff_dict = gas.reaction(ind).efficiencies
            eff = sorted(eff_dict, key=lambda x:gas.species_index(x))
            if not len(eff):
                thd_type.append(thd_body_type.unity)
            elif (len(eff) == 1 and eff_dict[eff[0]] == 1 and
                    gas.reaction(ind).default_efficiency == 0):
                thd_type.append(thd_body_type.species)
            else:
                thd_type.append(thd_body_type.mix)
            thd_sp_num.append(len(eff))
            for spec in eff:
                thd_sp.append(gas.species_index(spec))
                thd_eff.append(eff_dict[spec])
        #and test
        assert np.allclose(result['thd']['type'], np.array(thd_type, dtype=np.int32))
        assert np.allclose(result['thd']['eff'], thd_eff)
        assert np.allclose(result['thd']['spec_num'], thd_sp_num)
        assert np.allclose(result['thd']['spec'], thd_sp)

    def __test_rateconst_type(self, rtype):
        """
        Performs tests for a single reaction rate type

        Parameters
        ----------
        rtype : {'simple', 'plog', 'cheb'}
            The reaction type to test
        """

        T = self.store.T
        P = self.store.P
        ref_const = self.store.fwd_rate_constants
        ref_const_T = ref_const.T.copy()

        eqs = {'conp' : self.store.conp_eqs,
                'conv' : self.store.conv_eqs}

        target = get_target('opencl')

        oploop = OptionLoop(OrderedDict([('lang', ['opencl']),
            ('width', [4, None]),
            ('depth', [4, None]),
            ('ilp', [True, False]),
            ('unr', [None, 4]),
            ('device', ['0:0', '1']),
            ('rate_spec', [x for x in RateSpecialization]),
            ('rate_spec_kernels', [True, False])
            ]))

        reacs = self.store.reacs
        masks = {
            'simple' : (
                np.array([i for i, x in enumerate(reacs) if x.match((reaction_type.elementary,))]),
                get_simple_arrhenius_rates),
            'plog' : (
                np.array([i for i, x in enumerate(reacs) if x.match((reaction_type.plog,))]),
                get_plog_arrhenius_rates),
            'cheb' : (
                np.array([i for i, x in enumerate(reacs) if x.match((reaction_type.cheb,))]),
                get_cheb_arrhenius_rates)}

        compare_mask, rate_func = masks[rtype]

        for i, state in enumerate(oploop):
            if state['width'] is not None and state['depth'] is not None:
                continue
            opt = loopy_options(**{x : state[x] for x in
                state if x != 'device'})
            #find rate info
            rate_info = assign_rates(reacs, opt.rate_spec)
            #create the kernel info
            infos = rate_func(eqs, opt, rate_info,
                        test_size=self.store.test_size)

            if not isinstance(infos, list):
                try:
                    infos = list(infos)
                except:
                    infos = [infos]

            kernels = []
            for info in infos:
                #create kernel
                knl = make_rateconst_kernel(info, target, self.store.test_size)

                #apply vectorization
                knl = apply_rateconst_vectorization(opt, info.reac_ind, knl)

                kernels.append(knl)

            ref = ref_const if state['width'] else ref_const_T
            args = {'T_arr' : T}
            if rtype != 'simple':
                args['P_arr'] =  P

            assert auto_run(kernels, ref, device=state['device'],
                **args,
                compare_mask=compare_mask,
                compare_axis=1 if state['width'] is None else 0), \
                'Evaluate {} rates failed'.format(name)


    @attr('long')
    def test_simple_rate_constants(self):
        self.__test_rateconst_type('simple')

    @attr('long')
    def test_plog_rate_constants(self):
        self.__test_rateconst_type('plog')

    @attr('long')
    def test_cheb_rate_constants(self):
        self.__test_rateconst_type('cheb')
