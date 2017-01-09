#system
import os
import filecmp
from collections import OrderedDict

#local imports
from ..core.rate_subs import (rate_const_kernel_gen, get_rate_eqn, assign_rates,
    get_simple_arrhenius_rates, get_plog_arrhenius_rates, get_cheb_arrhenius_rates,
    make_rateconst_kernel, apply_rateconst_vectorization, get_thd_body_concs)
from ..loopy.loopy_utils import (auto_run, loopy_options, RateSpecialization, get_code,
    get_target, get_device_list)
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
        specs = self.store.specs
        result = assign_rates(reacs, specs, RateSpecialization.fixed)

        #test rate type
        assert np.all(result['simple']['type'] == 0)

        #import gas in cantera for testing
        gas = self.store.gas

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

        result = assign_rates(reacs, specs, RateSpecialization.hybrid)

        def __get_vals(reac, fall=False):
            try:
                Ea = reac.rate.activation_energy
                b = reac.rate.temperature_exponent
                if fall:
                    return None, None #don't care
            except:
                if not fall:
                    #want the normal rates
                    if isinstance(reac, ct.FalloffReaction) and not isinstance(reac, ct.ChemicallyActivatedReaction):
                        rate = reac.high_rate
                    else:
                        rate = reac.low_rate
                else:
                    #want the other rates
                    if isinstance(reac, ct.FalloffReaction) and not isinstance(reac, ct.ChemicallyActivatedReaction):
                        rate = reac.low_rate
                    else:
                        rate = reac.high_rate
                Ea = rate.activation_energy
                b = rate.temperature_exponent
            return Ea, b

        def test_assign(type_max, fall):
            #test rate type
            rtypes = []
            for reac in gas.reactions():
                if not (isinstance(reac, ct.PlogReaction) or isinstance(reac, ct.ChebyshevReaction)):
                    Ea, b = __get_vals(reac, fall)
                    if Ea is None:
                        continue
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
                    rtypes[-1] = min(rtypes[-1], type_max)
            return rtypes

        #test rate type
        assert np.allclose(result['simple']['type'],
            test_assign(2, False))
        assert np.allclose(result['fall']['type'],
            test_assign(2, True))
        __tester(result)

        result = assign_rates(reacs, specs, RateSpecialization.full)

        #test rate type
        assert np.allclose(result['simple']['type'],
            test_assign(5, False))
        assert np.allclose(result['fall']['type'],
            test_assign(5, True))
        __tester(result)

        #test the thd / falloff / chem assignments
        assert np.allclose(result['fall']['map'],
            [i for i, x in enumerate(gas.reactions()) if (isinstance(x,
                ct.FalloffReaction) or isinstance(x, ct.ChemicallyActivatedReaction))])
        fall_reacs = [gas.reaction(y) for y in result['fall']['map']]
        #test fall vs chemically activated
        assert np.allclose(result['fall']['ftype'],
            np.array([reaction_type.fall if (isinstance(x, ct.FalloffReaction) and not
                isinstance(x, ct.ChemicallyActivatedReaction)) else reaction_type.chem for x in
                fall_reacs], dtype=np.int32) - int(reaction_type.fall))
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
        #test parameters
        #troe
        troe_reacs = [x for x in fall_reacs if isinstance(x.falloff, ct.TroeFalloff)]
        troe_par = [x.falloff.parameters for x in troe_reacs]
        troe_a, troe_T3, troe_T1, troe_T2 = [np.array(x) for x in zip(*troe_par)]
        assert np.allclose(result['fall']['troe']['a'], troe_a)
        assert np.allclose(result['fall']['troe']['T3'], troe_T3)
        assert np.allclose(result['fall']['troe']['T1'], troe_T1)
        assert np.allclose(result['fall']['troe']['T2'], troe_T2)
        #and map
        assert np.allclose([fall_reacs.index(x) for x in troe_reacs],
            result['fall']['troe']['map'])
        #sri
        sri_reacs = [x for x in fall_reacs if isinstance(x.falloff, ct.SriFalloff)]
        sri_par = [x.falloff.parameters for x in sri_reacs]
        sri_a, sri_b, sri_c, sri_d, sri_e = [np.array(x) for x in zip(*sri_par)]
        assert np.allclose(result['fall']['sri']['a'], sri_a)
        assert np.allclose(result['fall']['sri']['b'], sri_b)
        assert np.allclose(result['fall']['sri']['c'], sri_c)
        assert np.allclose(result['fall']['sri']['d'], sri_d)
        assert np.allclose(result['fall']['sri']['e'], sri_e)
        #and map
        assert np.allclose([fall_reacs.index(x) for x in sri_reacs],
            result['fall']['sri']['map'])

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

    def __generic_rate_tester(self, func, ref_ans, ref_ans_T, args, mask=None,
        variable_loop_length=False):
        """
        A generic testing method that can be used for rate constants, third bodies, ...

        Parameters
        ----------
        func : function
            The function to test
        ref_ans : :class:`numpy.ndarray`
            The answer in 'wide' form
        ref_ans_T : :class:`numpy.ndarray`
            The answer in 'deep' form
        args : list of `loopy.KernelArguement`
            The args to pass to the kernel
        mask : :class:`numpy.ndarray`
            If not none, the compare mask to use in testing
        variable_loop_length : bool
            ILP interferes with this type of kernel, turn it off
        """

        eqs = {'conp' : self.store.conp_eqs,
                'conv' : self.store.conv_eqs}
        oploop = OptionLoop(OrderedDict([('lang', ['opencl']),
            ('width', [4, None]),
            ('depth', [4, None]),
            ('order', ['C', 'F']),
            ('ilp', [False]),
            ('unr', [None, 4]),
            ('device', get_device_list()),
            ('rate_spec', [x for x in RateSpecialization]),
            ('rate_spec_kernels', [True, False])
            ]))

        reacs = self.store.reacs
        specs = self.store.specs

        target = get_target('opencl')

        for i, state in enumerate(oploop):
            if state['width'] is not None and state['depth'] is not None:
                continue
            opt = loopy_options(**{x : state[x] for x in
                state if x != 'device'})
            #find rate info
            rate_info = assign_rates(reacs, specs, opt.rate_spec)
            #create the kernel info
            infos = func(eqs, opt, rate_info,
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

            ref = ref_ans if state['order'] == 'F' else ref_ans_T

            args_copy = args.copy()
            for key in args_copy:
                if hasattr(args_copy[key], '__call__'):
                    #it's a function
                    args_copy[key] = args_copy[key](state['order'])

            assert auto_run(kernels, ref, device=state['device'],
                compare_mask=mask,
                compare_axis=1 if state['order'] == 'C' else 0,
                **args_copy), \
                'Evaluate {} rates failed'.format(func.__name__)

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

        args = {'T_arr' : T}
        if rtype != 'simple':
            args['P_arr'] =  P

        compare_mask, rate_func = masks[rtype]

        self.__generic_rate_tester(rate_func, ref_const, ref_const_T, args,
            mask=compare_mask)


    @attr('long')
    def test_simple_rate_constants(self):
        self.__test_rateconst_type('simple')

    @attr('long')
    def test_plog_rate_constants(self):
        self.__test_rateconst_type('plog')

    @attr('long')
    def test_cheb_rate_constants(self):
        self.__test_rateconst_type('cheb')

    @attr('long')
    def test_thd_body_concs(self):
        T = self.store.T
        P = self.store.P
        concs = self.store.concs
        ref_ans = self.store.ref_thd.copy()
        ref_ans_T = self.store.ref_thd.T.copy()
        args = { 'T_arr' : T,
                 'P_arr' : P,
                 'conc' : lambda x: concs[:-1, :].copy() if x == 'F'
                            else concs[:-1, :].T.copy()}
        self.__generic_rate_tester(get_thd_body_concs, ref_ans, ref_ans_T, args,
            variable_loop_length=True)
