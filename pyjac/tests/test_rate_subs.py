#system
import os
import filecmp
from collections import OrderedDict, defaultdict
import logging
logging.getLogger('root').setLevel(logging.WARNING)

#local imports
from ..core.rate_subs import (rate_const_kernel_gen, get_rate_eqn, assign_rates,
    get_simple_arrhenius_rates, get_plog_arrhenius_rates, get_cheb_arrhenius_rates,
    make_rateconst_kernel, apply_rateconst_vectorization, get_thd_body_concs,
    get_reduced_pressure_kernel, get_sri_kernel, get_troe_kernel, get_rev_rates,
    get_rxn_pres_mod, get_rop, get_rop_net)
from ..loopy.loopy_utils import (auto_run, loopy_options, RateSpecialization, get_code,
    get_target, get_device_list, populate, kernel_call)
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

        #test fwd / rev maps, nu, species etc.
        assert result['fwd']['num'] == gas.n_reactions
        assert np.array_equal(result['fwd']['map'], np.arange(gas.n_reactions))
        rev_inds = np.array([i for i in range(gas.n_reactions) if gas.is_reversible(i)])
        assert np.array_equal(result['rev']['map'], rev_inds)
        assert result['rev']['num'] == rev_inds.size
        fwd_specs = []
        fwd_nu = []
        rev_specs = []
        rev_nu = []
        nu_sum = []
        net_nu = []
        net_specs = []
        net_num_specs = []
        for i, reac in enumerate(gas.reactions()):
            temp_nu_sum = defaultdict(lambda: 0)
            for spec, nu in sorted(reac.reactants.items(), key=lambda x: gas.species_index(x[0])):
                fwd_specs.append(gas.species_index(spec))
                fwd_nu.append(nu)
                temp_nu_sum[spec] -= nu
            assert result['fwd']['num_spec'][i] == len(reac.reactants)
            for spec, nu in sorted(reac.products.items(), key=lambda x: gas.species_index(x[0])):
                if i in rev_inds:
                    rev_specs.append(gas.species_index(spec))
                    rev_nu.append(nu)
                temp_nu_sum[spec] += nu
            if i in rev_inds:
                rev_ind = np.where(rev_inds == i)[0]
                assert result['rev']['num_spec'][rev_ind] == len(reac.products)
            temp_specs, temp_nu_sum = zip(*[(gas.species_index(x[0]), x[1]) for x in
                sorted(temp_nu_sum.items(), key=lambda x: gas.species_index(x[0]))])
            net_specs.extend(temp_specs)
            net_num_specs.append(len(temp_specs))
            net_nu.extend(temp_nu_sum)
            nu_sum.append(sum(temp_nu_sum))
        assert np.allclose(fwd_specs, result['fwd']['specs'])
        assert np.allclose(fwd_nu, result['fwd']['nu'])
        assert np.allclose(rev_specs, result['rev']['specs'])
        assert np.allclose(rev_nu, result['rev']['nu'])
        assert np.allclose(nu_sum, result['net']['nu_sum'])
        assert np.allclose(net_nu, result['net']['nu'])
        assert np.allclose(net_num_specs, result['net']['num_spec'])
        assert np.allclose(net_specs, result['net']['specs'])

        def __get_rate(reac, fall=False):
            try:
                Ea = reac.rate.activation_energy
                b = reac.rate.temperature_exponent
                if fall:
                    return None
                return reac.rate
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
                return rate
            return Ea, b

        def __tester(result, spec_type):
            #test return value
            assert 'simple' in result and 'cheb' in result and 'plog' in result

            #test num, map
            plog_inds, plog_reacs = zip(*[(i, x) for i, x in enumerate(gas.reactions())
                    if isinstance(x, ct.PlogReaction)])
            cheb_inds, cheb_reacs = zip(*[(i, x) for i, x in enumerate(gas.reactions())
            if isinstance(x, ct.ChebyshevReaction)])

            def rate_checker(our_params, ct_params, rate_forms, force_act_nonlog=False):
                act_energy_ratios = []
                for ourvals, ctvals, form in zip(*(our_params, ct_params, rate_forms)):
                    #activation energy, check rate form
                    #if it's fixed specialization, or the form >= 2
                    if (spec_type == RateSpecialization.fixed or form >= 2) and not force_act_nonlog:
                        #it's in log form
                        assert np.isclose(ourvals[0], np.log(ctvals.pre_exponential_factor))
                    else:
                        assert np.isclose(ourvals[0], ctvals.pre_exponential_factor)
                    #temperature exponent doesn't change w/ form
                    assert np.isclose(ourvals[1], ctvals.temperature_exponent)
                    #activation energy, either the ratios should be constant or
                    #it should be zero
                    if ourvals[2] == 0 or ctvals.activation_energy == 0:
                        assert ourvals[2] == ctvals.activation_energy
                    else:
                        act_energy_ratios.append(ourvals[2] / ctvals.activation_energy)
                #check that all activation energy ratios are the same
                assert np.all(np.isclose(act_energy_ratios, act_energy_ratios[0]))

            #check rate values
            assert np.array_equal(result['plog']['num_P'], [len(p.rates) for p in plog_reacs])
            for i, reac_params in enumerate(result['plog']['params']):
                for j, rates in enumerate(plog_reacs[i].rates):
                    assert np.isclose(reac_params[j][0], rates[0])
                #plog uses a weird form, so use force_act_nonlog
                rate_checker([rp[1:] for rp in reac_params], [rate[1] for rate in plog_reacs[i].rates],
                    [2 for rate in plog_reacs[i].rates], force_act_nonlog=True)

            simple_inds = sorted(list(set(range(gas.n_reactions)).difference(
                set(plog_inds).union(set(cheb_inds)))))
            assert result['simple']['num'] == len(simple_inds)
            assert np.allclose(result['simple']['map'], np.array(simple_inds))
            #test the simple reaction rates
            simple_reacs = [gas.reaction(i) for i in simple_inds]
            rate_checker([(result['simple']['A'][i], result['simple']['b'][i],
                result['simple']['Ta'][i]) for i in range(result['simple']['num'])],
                [__get_rate(reac, False) for reac in simple_reacs],
                result['simple']['type'])

            #test the falloff (alternate) rates
            fall_reacs = [gas.reaction(i) for i in result['fall']['map']]
            rate_checker([(result['fall']['A'][i], result['fall']['b'][i],
                result['fall']['Ta'][i]) for i in range(result['fall']['num'])],
                [__get_rate(reac, True) for reac in fall_reacs],
                result['fall']['type'])

        __tester(result, RateSpecialization.fixed)


        result = assign_rates(reacs, specs, RateSpecialization.hybrid)

        def test_assign(type_max, fall):
            #test rate type
            rtypes = []
            for reac in gas.reactions():
                if not (isinstance(reac, ct.PlogReaction) or isinstance(reac, ct.ChebyshevReaction)):
                    rate = __get_rate(reac, fall)
                    if rate is None:
                        continue
                    Ea = rate.activation_energy
                    b = rate.temperature_exponent
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
        __tester(result, RateSpecialization.hybrid)

        result = assign_rates(reacs, specs, RateSpecialization.full)

        #test rate type
        assert np.allclose(result['simple']['type'],
            test_assign(5, False))
        assert np.allclose(result['fall']['type'],
            test_assign(5, True))
        __tester(result, RateSpecialization.full)

        #ALL BELOW HERE ARE INDEPENDENT OF SPECIALIZATIONS
        cheb_inds, cheb_reacs = zip(*[(i, x) for i, x in enumerate(gas.reactions())
            if isinstance(x, ct.ChebyshevReaction)])
        assert result['cheb']['num'] == len(cheb_inds)
        assert np.allclose(result['cheb']['map'], np.array(cheb_inds))

        plog_inds, plog_reacs = zip(*[(i, x) for i, x in enumerate(gas.reactions())
            if isinstance(x, ct.PlogReaction)])
        assert result['plog']['num'] == len(plog_inds)
        assert np.allclose(result['plog']['map'], np.array(plog_inds))

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

    def __generic_rate_tester(self, func, kernel_calls, do_ratespec=False, do_ropsplit=None):
        """
        A generic testing method that can be used for rate constants, third bodies, ...

        Parameters
        ----------
        func : function
            The function to test
        kernel_calls : :class:`kernel_call` or list thereof
            Contains the masks and reference answers for kernel testing
        do_ratespec : bool
            If true, test rate specializations and kernel splitting for simple rates
        do_ropsplit : bool
            If true, test kernel splitting for rop_net
        """

        eqs = {'conp' : self.store.conp_eqs,
                'conv' : self.store.conv_eqs}
        oploop = [('lang', ['opencl']),
            ('width', [4, None]),
            ('depth', [4, None]),
            ('order', ['C', 'F']),
            ('ilp', [False]),
            ('unr', [None, 4]),
            ('device', get_device_list()),
            ]
        ratespec_loop = [
            ('rate_spec', [x for x in RateSpecialization]),
            ('rate_spec_kernels', [True, False])]
        ropsplit_loop = [
            ('rop_net_kernels', [True])]
        if do_ratespec:
            oploop = oploop + ratespec_loop
        if do_ropsplit:
            oploop = oploop + ropsplit_loop
        oploop = OptionLoop(OrderedDict(oploop))

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
                knl = apply_rateconst_vectorization(opt, info.var_name, knl)

                kernels.append(knl)

            #create a list of answers to check
            try:
                for kc in kernel_calls:
                    kc.set_state(state['order'])
            except:
                kernel_calls.set_state(state['order'])

            assert auto_run(kernels, kernel_calls, device=state['device']), \
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

        #create the kernel call
        kc = kernel_call('rateconst_' + rtype,
                            ref_const, compare_mask=compare_mask, **args)

        self.__generic_rate_tester(rate_func, kc, rtype == 'simple')


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
        args = { 'T_arr' : T,
                 'P_arr' : P,
                 'conc' : lambda x: concs.copy() if x == 'F'
                            else concs.T.copy()}

        #create the kernel call
        kc = kernel_call('rateconst_thd', ref_ans, **args)
        self.__generic_rate_tester(get_thd_body_concs, kc)

    @attr('long')
    def test_reduced_pressure(self):
        T = self.store.T
        ref_thd = self.store.ref_thd.copy()
        ref_ans = self.store.ref_Pr.copy()
        kf_vals = {}
        kf_fall_vals = {}
        args = { 'T_arr' : T,
                 'kf' : lambda x: kf_vals[x],
                 'kf_fall' : lambda x: kf_fall_vals[x],
                 'thd_conc' : lambda x: ref_thd.copy() if x == 'F'
                            else ref_thd.T.copy()
                 }

        def __tester(eqs, loopy_opts, rate_info, test_size):
            #check if we've found the kf / kf_fall values yet
            if loopy_opts.order not in kf_vals:
                #first we have to get the simple arrhenius rates
                #in order to evaluate the reduced pressure
                target = get_target(loopy_opts.lang)

                device = get_device_list()[0]

                #first with falloff parameters
                infos = get_simple_arrhenius_rates(eqs, loopy_opts, rate_info, test_size,
                        falloff=True)
                knl_list = []
                for info in infos:
                    #make kernel
                    knl = make_rateconst_kernel(info, target, test_size)
                    #apply vectorization
                    knl = apply_rateconst_vectorization(loopy_opts, info.var_name, knl)
                    #and add to list
                    knl_list.append(knl)
                kc = kernel_call('rateconst_kf_fall', [], **{'T_arr' : T})
                kc.set_state(loopy_opts.order)
                kf_fall_vals[loopy_opts.order] = populate(knl_list, kc, device=device)[0][0]

                #next with regular parameters
                infos = get_simple_arrhenius_rates(eqs, loopy_opts, rate_info, test_size)
                knl_list = []
                for info in infos:
                    #make kernel
                    knl = make_rateconst_kernel(info, target, test_size)
                    #apply vectorization
                    knl = apply_rateconst_vectorization(loopy_opts, info.var_name, knl)
                    #and add to list
                    knl_list.append(knl)
                kc = kernel_call('rateconst_kf', [], **{'T_arr' : T})
                kc.set_state(loopy_opts.order)
                kf_vals[loopy_opts.order] = populate(knl_list, kc, device=device)[0][0]

            #finally we can call the reduced pressure evaluator
            return get_reduced_pressure_kernel(eqs, loopy_opts, rate_info, test_size)

        #create the kernel call
        kc = kernel_call('rateconst_pred', ref_ans, **args)
        self.__generic_rate_tester(__tester, kc)

    @attr('long')
    def test_sri_falloff(self):
        T = self.store.T
        ref_Pr = self.store.ref_Pr.copy()
        ref_ans = self.store.ref_Sri.copy().squeeze()
        args = { 'Pr' :  lambda x: ref_Pr.copy() if x == 'F'
                         else ref_Pr.T.copy(),
                 'T_arr' : T
               }

        #get SRI reaction mask
        sri_mask = np.where(np.in1d(self.store.fall_inds, self.store.sri_inds))[0]
        #create the kernel call
        kc = kernel_call('rateconst_fall_sri', ref_ans, out_mask=[0],
                                    compare_mask=sri_mask, **args)
        self.__generic_rate_tester(get_sri_kernel, kc)

    @attr('long')
    def test_troe_falloff(self):
        T = self.store.T
        ref_Pr = self.store.ref_Pr.copy()
        ref_ans = self.store.ref_Troe.copy()
        args = { 'Pr' :  lambda x: ref_Pr.copy() if x == 'F'
                         else ref_Pr.T.copy(),
                 'T_arr' : T
               }

        #get Troe reaction mask
        troe_mask = np.where(np.in1d(self.store.fall_inds, self.store.troe_inds))[0]
        #create the kernel call
        kc = kernel_call('rateconst_fall_troe', ref_ans, out_mask=[0],
                                    compare_mask=troe_mask, **args)
        self.__generic_rate_tester(get_troe_kernel, kc)

    @attr('long')
    def test_rev_rates(self):
        ref_fwd_rates = self.store.fwd_rate_constants.copy()
        ref_kc = self.store.equilibrium_constants.copy()
        ref_B = self.store.ref_B_rev.copy()
        ref_rev = self.store.rev_rate_constants.copy()
        args={'B' : lambda x: ref_B.copy() if x == 'F' else ref_B.T.copy(),
                'kf' : lambda x: ref_fwd_rates.copy() if x == 'F' else ref_fwd_rates.T.copy()}

        #create the kernel call
        kc = kernel_call('rateconst_Kc', [ref_kc, ref_rev],
                                    out_mask=[0, 1], **args)

        self.__generic_rate_tester(get_rev_rates, kc)

    @attr('long')
    def test_pressure_mod(self):
        ref_fwd_rates = self.store.fwd_rate_constants.copy()
        ref_pres_mod = self.store.ref_pres_mod.copy()
        ref_Pr = self.store.ref_Pr.copy()
        ref_Fi = self.store.ref_Fall.copy()
        ref_thd = self.store.ref_thd.copy()

        args = {'Fi' : lambda x: ref_Fi.copy() if x == 'F' else ref_Fi.T.copy(),
                'thd_conc' : lambda x: ref_thd.copy() if x == 'F' else ref_thd.T.copy(),
                'Pr' : lambda x: ref_Pr.copy() if x == 'F' else ref_Pr.T.copy()}

        thd_only_inds = np.where(np.logical_not(np.in1d(self.store.thd_inds,
                                    self.store.fall_inds)))[0]
        thd_rxn_inds = self.store.thd_inds[thd_only_inds]
        fall_only_inds = np.where(np.in1d(self.store.thd_inds,
                                    self.store.fall_inds))[0]
        fall_rxn_inds = self.store.fall_inds[:]

        #create the kernel call
        kc = [kernel_call('rateconst_ci_thd', [ref_pres_mod],
                        out_mask=[0],
                        compare_mask=[thd_only_inds],
                        input_mask=['Fi', 'Pr'],
                        strict_name_match=True, **args),
              kernel_call('rateconst_ci_fall', [ref_pres_mod],
                        out_mask=[0],
                        compare_mask=[fall_only_inds],
                        input_mask=['thd_conc'],
                        strict_name_match=True, **args)]
        self.__generic_rate_tester(get_rxn_pres_mod, kc)

    @attr('long')
    def test_rop(self):
        fwd_rate_constants = self.store.fwd_rate_constants.copy()
        rev_rate_constants = self.store.rev_rate_constants.copy()
        fwd_rxn_rate = self.store.fwd_rxn_rate.copy()
        rev_rxn_rate = self.store.rev_rxn_rate.copy()
        concs = self.store.concs.copy()

        args={'kf' : lambda x: fwd_rate_constants.copy() if x == 'F' else fwd_rate_constants.T.copy(),
                'kr' : lambda x: rev_rate_constants.copy() if x == 'F' else rev_rate_constants.T.copy(),
                'conc' : lambda x: concs.copy() if x == 'F' else concs.T.copy()}

        kc = [kernel_call('rateconst_rop_fwd', [fwd_rxn_rate],
                        input_mask=['kr'],
                        strict_name_match=True, **args),
              kernel_call('rateconst_rop_rev', [rev_rxn_rate],
                        input_mask=['kf'],
                        strict_name_match=True, **args)]
        self.__generic_rate_tester(get_rop, kc)

    @attr('long')
    def test_rop_net(self):
        fwd_removed = self.store.fwd_rxn_rate.copy()
        fwd_removed[self.store.thd_inds, :] = fwd_removed[self.store.thd_inds, :] / self.store.ref_pres_mod
        thd_in_rev = np.where(np.in1d(self.store.thd_inds, self.store.rev_inds))[0]
        rev_update_map = np.where(np.in1d(self.store.rev_inds, self.store.thd_inds[thd_in_rev]))[0]
        rev_removed = self.store.rev_rxn_rate.copy()
        rev_removed[rev_update_map, :] = rev_removed[rev_update_map, :] / self.store.ref_pres_mod[thd_in_rev, :]
        args={'rop_fwd' : lambda x: fwd_removed.copy() if x == 'F' else fwd_removed.T.copy(),
                'rop_rev' : lambda x: rev_removed.copy() if x == 'F' else rev_removed.T.copy(),
                'pres_mod' : lambda x: self.store.ref_pres_mod.copy() if x == 'F' else
                                        self.store.ref_pres_mod.T.copy()}

        #first test w/o the splitting
        kc = kernel_call('rateconst_rop_net', [self.store.rxn_rates], **args)
        #self.__generic_rate_tester(get_rop_net, kc)

        def __input_mask(self, arg_name):
            names = ['fwd', 'rev', 'pres_mod']
            return next(x for x in names if x in self.name) in arg_name

        def __chainer(self, out_vals):
            self.kernel_args['rop_net'] = out_vals[-1][0]

        #next test with splitting
        kc = [kernel_call('rateconst_rop_net_fwd', [self.store.rxn_rates],
            input_mask=__input_mask, strict_name_match=True,
            check=False, **args),
              kernel_call('rateconst_rop_net_rev', [self.store.rxn_rates],
            input_mask=__input_mask, strict_name_match=True,
            check=False, chain=__chainer, **args),
              kernel_call('rateconst_rop_net_pres_mod', [self.store.rxn_rates],
            input_mask=__input_mask, strict_name_match=True,
            chain=__chainer, **args)]
        self.__generic_rate_tester(get_rop_net, kc, do_ropsplit=True)