# -*- coding: utf-8 -*-
"""Module for writing species/reaction rate subroutines.

This is kept separate from Jacobian creation module in order
to create only the rate subroutines if desired.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import sys
import math
import os
import logging
from string import Template
from collections import OrderedDict

# Non-standard librarys
import sympy as sp
import loopy as lp
import numpy as np
from loopy.kernel.data import temp_var_scope as scopes
from ..loopy import loopy_utils as lp_utils

# Local imports
from .. import utils
from . import chem_model as chem
from . import mech_interpret as mech
from ..kernel_utils import kernel_gen as k_gen
from ..sympy import sympy_utils as sp_utils
from . reaction_types import reaction_type, falloff_form, thd_body_type, reversible_type

def assign_rates(reacs, specs, rate_spec):
    """
    From a given set of reactions, determine the rate types for evaluation

    Parameters
    ----------
    reacs : list of `ReacInfo`
        The reactions in the mechanism
    specs : list of `SpecInfo`
        The species in the mechanism
    rate_spec : `RateSpecialization` enum
        The specialization option specified

    Notes
    -----

    For simple Arrhenius evaluations, the rate type keys are:

    if rate_spec == RateSpecialization.full
        0 -> kf = A
        1 -> kf = A * T * T * T ...
        2 -> kf = exp(logA + b * logT)
        3 -> kf = exp(logA - Ta / T)
        4 -> kf = exp(logA + b * logT - Ta / T)

    if rate_spec = RateSpecialization.hybrid
        0 -> kf = A
        1 -> kf = A * T * T * T ...
        2 -> kf = exp(logA + b * logT - Ta / T)

    if rate_spec == lp_utils.rate_specialization.fixed
        0 -> kf = exp(logA + b * logT - Ta / T)

    Returns
    -------
    rate_info : dict of parameters
        Keys are 'simple', 'plog', 'cheb', 'fall', 'chem', 'thd'
        Values are further dictionaries including addtional rate info, number,
        offset, maps, etc.

    Notes
    -----
        Note that the reactions in 'fall', 'chem' and 'thd' are also in 'simple'
        Further, there are duplicates between 'thd' and 'fall' / 'chem'
    """

    #determine specialization
    full = rate_spec == lp_utils.RateSpecialization.full
    hybrid = rate_spec == lp_utils.RateSpecialization.hybrid
    fixed = rate_spec == lp_utils.RateSpecialization.fixed

    #find fwd / reverse rate parameters
    #first, the number of each
    rev_map = np.array([i for i, x in enumerate(reacs) if x.rev], dtype=np.int32)
    num_rev = len(rev_map)
    #next, find the species / nu values
    fwd_spec = []
    fwd_num_spec = []
    fwd_nu = []
    rev_spec = []
    rev_num_spec = []
    rev_nu = []
    nu_sum = []
    net_num_spec = []
    net_nu = []
    net_spec = []
    fwd_allnu_integer = True
    rev_allnu_integer = True
    for rxn in reacs:
        #fwd
        fwd_spec.extend(rxn.reac[:])
        fwd_num_spec.append(len(rxn.reac))
        fwd_nu.extend(rxn.reac_nu[:])
        if rxn.rev:
            #and rev
            rev_spec.extend(rxn.prod[:])
            rev_num_spec.append(len(rxn.prod))
            rev_nu.extend(rxn.prod_nu[:])
        #finally, net values
        spec = list(sorted(set(rxn.reac + rxn.prod)))
        net_spec.extend(spec)
        net_num_spec.append(len(spec))
        nu = [utils.get_nu(isp, rxn) for isp in spec]
        net_nu.extend(nu)
        #and nu sum for equilibrium constants
        nu_sum.append(sum(nu))

    #create numpy versions
    fwd_spec = np.array(fwd_spec, dtype=np.int32)
    fwd_num_spec = np.array(fwd_num_spec, dtype=np.int32)
    if any(not utils.is_integer(nu) for nu in fwd_nu):
        fwd_nu = np.array(fwd_nu)
        fwd_allnu_integer = False
    else:
        fwd_nu = np.array(fwd_nu, dtype=np.int32)
    rev_spec = np.array(rev_spec, dtype=np.int32)
    rev_num_spec = np.array(rev_num_spec, dtype=np.int32)
    if any(not utils.is_integer(nu) for nu in rev_nu):
        rev_nu = np.array(rev_nu)
        fwd_allnu_integer = False
    else:
        rev_nu = np.array(rev_nu, dtype=np.int32)

    net_nu_integer = all(utils.is_integer(nu) for nu in net_nu)
    if net_nu_integer:
        nu_sum = np.array(nu_sum, dtype=np.int32)
        net_nu = np.array(net_nu, dtype=np.int32)
    else:
        nu_sum = np.array(nu_sum)
        net_nu = np.array(net_nu)
    net_num_spec = np.array(net_num_spec, dtype=np.int32)
    net_spec = np.array(net_spec, dtype=np.int32)

    #sometimes we need the net properties forumlated per species rather than per reaction
    #as above
    spec_to_reac = []
    spec_nu = []
    spec_reac_count = []
    spec_list = []
    for ispec, spec in enumerate(specs):
        #first, find all non-zero nu reactions
        reac_list = [x for x in [(irxn, utils.get_nu(ispec, rxn)) for irxn, rxn in
                        enumerate(reacs)] if x[1]]
        if reac_list:
            reac_list, nu_list = zip(*reac_list)
            spec_to_reac.extend(reac_list)
            spec_nu.extend(nu_list)
            spec_reac_count.append(len(reac_list))
            spec_list.append(ispec)

    spec_to_reac = np.array(spec_to_reac, dtype=np.int32)
    if net_nu_integer:
        spec_nu = np.array(spec_nu, dtype=np.int32)
    else:
        spec_nu = np.array(spec_nu)
    spec_reac_count = np.array(spec_reac_count, dtype=np.int32)
    spec_list = np.array(spec_list, dtype=np.int32)


    def __seperate(reacs, matchers):
        #find all reactions / indicies that match this offset
        rate = [(i, x) for i, x in enumerate(reacs) if any(x.match(y) for y in matchers)]
        mapping = []
        num = 0
        if rate:
            mapping, rate = zip(*rate)
            mapping = np.array(mapping, dtype=np.int32)
            rate = list(rate)
            num = len(rate)

        return rate, mapping, num

    #count / seperate reactions with simple arrhenius rates
    simple_rate, simple_map, num_simple = __seperate(
        reacs, [reaction_type.elementary, reaction_type.thd,
                    reaction_type.fall, reaction_type.chem])

    def __specialize(rates, fall=False):
        fall_types = None
        num = len(rates)
        rate_type = np.zeros((num,), dtype=np.int32)
        if fall:
            fall_types = np.zeros((num,), dtype=np.int32)
        #reaction parameters
        A = np.zeros((num,), dtype=np.float64)
        b = np.zeros((num,), dtype=np.float64)
        Ta = np.zeros((num,), dtype=np.float64)

        for i, reac in enumerate(rates):
            if (reac.high or reac.low) and fall:
                if reac.high:
                    Ai, bi, Tai = reac.high
                    fall_types[i] = 1 #mark as chemically activated
                else:
                    #we want k0, hence default factor is fine
                    Ai, bi, Tai = reac.low
                    fall_types[i] = 0 #mark as falloff
            else:
                #assign rate params
                Ai, bi, Tai = reac.A, reac.b, reac.E
            #generic assign
            A[i] = np.log(Ai)
            b[i] = bi
            Ta[i] = Tai

            if fixed:
                rate_type[i] = 0
                continue
            #assign rate types
            if bi == 0 and Tai == 0:
                A[i] = Ai
                rate_type[i] = 0
            elif bi == int(bi) and bi and Tai == 0:
                A[i] = Ai
                rate_type[i] = 1
            elif Tai == 0 and bi != 0:
                rate_type[i] = 2
            elif bi == 0 and Tai != 0:
                rate_type[i] = 3
            else:
                rate_type[i] = 4
            if not full:
                rate_type[i] = rate_type[i] if rate_type[i] <= 1 else 2
        return rate_type, A, b, Ta, fall_types

    simple_rate_type, A, b, Ta, _ = __specialize(simple_rate)

    #finally determine the advanced rate evaulation types
    plog_reacs, plog_map, num_plog = __seperate(
        reacs, [reaction_type.plog])

    #create the plog arrays
    num_pressures = []
    plog_params = []
    for p in plog_reacs:
        num_pressures.append(len(p.plog_par))
        plog_params.append(p.plog_par)
    num_pressures = np.array(num_pressures, dtype=np.int32)

    cheb_reacs, cheb_map, num_cheb = __seperate(
        reacs, [reaction_type.cheb])

    #create the chebyshev arrays
    cheb_n_pres = []
    cheb_n_temp = []
    cheb_plim = []
    cheb_tlim = []
    cheb_coeff = []
    for cheb in cheb_reacs:
        cheb_n_pres.append(cheb.cheb_n_pres)
        cheb_n_temp.append(cheb.cheb_n_temp)
        cheb_coeff.append(cheb.cheb_par)
        cheb_plim.append(cheb.cheb_plim)
        cheb_tlim.append(cheb.cheb_tlim)

    #find falloff types
    fall_reacs, fall_map, num_fall = __seperate(
        reacs, [reaction_type.fall, reaction_type.chem])
    fall_rate_type, fall_A, fall_b, fall_Ta, fall_types = __specialize(fall_reacs, True)
    #find blending type
    blend_type = np.array([next(int(y) for y in x.type if isinstance(
        y, falloff_form)) for x in fall_reacs], dtype=np.int32)
    #seperate parameters based on blending type
    #sri
    sri_map = np.where(blend_type == int(falloff_form.sri))[0].astype(dtype=np.int32)
    sri_reacs = [reacs[fall_map[i]] for i in sri_map]
    sri_par = [reac.sri_par for reac in sri_reacs]
    #now fill in defaults as needed
    for par_set in sri_par:
        if len(par_set) != 5:
            par_set.extend([1, 0])
    if len(sri_par):
        sri_a, sri_b, sri_c, sri_d, sri_e = [np.array(x, dtype=np.float64) for x in zip(*sri_par)]
    else:
        sri_a, sri_b, sri_c, sri_d, sri_e = [np.empty(shape=(0,)) for i in range(5)]
    #and troe
    troe_map = np.where(blend_type == int(falloff_form.troe))[0].astype(dtype=np.int32)
    troe_reacs = [reacs[fall_map[i]] for i in troe_map]
    troe_par = [reac.troe_par for reac in troe_reacs]
    #now fill in defaults as needed
    for par_set in troe_par:
        if len(par_set) != 4:
            par_set.append(0)
    troe_a, troe_T3, troe_T1, troe_T2 = [np.array(x, dtype=np.float64) for x in zip(*troe_par)]

    #find third-body types
    thd_reacs, thd_map, num_thd = __seperate(
        reacs, [reaction_type.fall, reaction_type.chem, reaction_type.thd])
    #find third body type
    thd_type = np.array([next(int(y) for y in x.type if isinstance(
        y, thd_body_type)) for x in thd_reacs], dtype=np.int32)
    #find the species indicies
    thd_spec_num = []
    thd_spec = []
    thd_eff = []
    for x in thd_reacs:
        if x.match(thd_body_type.species):
            thd_spec_num.append(1)
            thd_spec.append(x.pdep_sp)
            thd_eff.append(1)
        elif x.match(thd_body_type.unity):
            thd_spec_num.append(0)
        else:
            thd_spec_num.append(len(x.thd_body_eff))
            spec, eff = zip(*x.thd_body_eff)
            thd_spec.extend(spec)
            thd_eff.extend(eff)
    thd_spec_num = np.array(thd_spec_num, dtype=np.int32)
    thd_spec = np.array(thd_spec, dtype=np.int32)
    thd_eff = np.array(thd_eff, dtype=np.float64)

    return {'simple' : {'A' : A, 'b' : b, 'Ta' : Ta, 'type' : simple_rate_type,
                'num' : num_simple, 'map' : simple_map},
            'plog' : {'map' : plog_map, 'num' : num_plog,
            'num_P' : num_pressures, 'params' : plog_params},
            'cheb' : {'map' : cheb_map, 'num' : num_cheb,
                'num_P' : cheb_n_pres, 'num_T' : cheb_n_temp,
                'params' : cheb_coeff, 'Tlim' : cheb_tlim,
                'Plim' : cheb_plim},
            'fall' : {'map' : fall_map, 'num' : num_fall,
                'ftype' : fall_types, 'blend' : blend_type,
                'A' : fall_A, 'b' : fall_b, 'Ta' : fall_Ta,
                'type' : fall_rate_type,
                'sri' :
                    {'map' : sri_map,
                     'num' : sri_map.size,
                     'a' : sri_a,
                     'b' : sri_b,
                     'c' : sri_c,
                     'd' : sri_d,
                     'e' : sri_e
                    },
                 'troe' :
                    {'map' : troe_map,
                     'num' : troe_map.size,
                     'a' : troe_a,
                     'T3' : troe_T3,
                     'T1' : troe_T1,
                     'T2' : troe_T2
                    }
                },
            'thd' : {'map' : thd_map, 'num' : num_thd,
                'type' : thd_type, 'spec_num' : thd_spec_num,
                'spec' : thd_spec, 'eff' : thd_eff},
            'fwd' : {'map' : np.arange(len(reacs)), 'num' : len(reacs),
                'num_spec' :  fwd_num_spec, 'specs' : fwd_spec,
                'nu' : fwd_nu, 'allint' : fwd_allnu_integer},
            'rev' : {'map' : rev_map, 'num' : num_rev,
                'num_spec' :  rev_num_spec, 'specs' : rev_spec,
                'nu' : rev_nu, 'allint' : rev_allnu_integer},
            'net' : {'num_spec' : net_num_spec, 'nu_sum' : nu_sum,
                'nu' : net_nu, 'specs' : net_spec,
                'allint' : net_nu_integer},
            'net_per_spec' : {'reac_count' : spec_reac_count, 'nu' : spec_nu,
                'reacs' : spec_to_reac, 'map' : spec_list,
                'allint' : net_nu_integer},
            'Nr' : len(reacs),
            'Ns' : len(specs)}


def __1Dcreator(name, numpy_arg, index='${reac_ind}', scope=scopes.PRIVATE):
    """
    Simple convenience method for creating 1D loopy arrays from
    Numpy args

    Parameters
    ----------
    name : str
        The loopy arg name
    numpy_arg : :class:`numpy.ndarray`
        The numpy array to use as initializer
    index : str, optional
        The string form of the index used in loopy code. Defaults to ${reac_ind}
    scope : :class:`loopy.temp_var_scope`
        The scope to use for the temporary variable. Defaults to PRIVATE
    """
    arg_lp = lp.TemporaryVariable(name, shape=numpy_arg.shape, initializer=numpy_arg,
                                  scope=scope, dtype=numpy_arg.dtype, read_only=True)
    arg_str = name + '[{}]'.format(index)
    return arg_lp, arg_str

def get_temperature_rate(eqs, loopy_opts, rate_info, conp=True, test_size=None):
    """Generates instructions, kernel arguements, and data for the temperature derivative
    kernel

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    conp : bool
        If true, generate equations using constant pressure assumption
        If false, use constant volume equations
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator for both equation types
    """

    #here, the equation form _does_ matter
    if conp:
        term = next(x for x in eqs['conp'] if str(x) == 'frac{text{d} T }{text{d} t }')
        term = eqs['conp'][term]
    else:
        term = next(x for x in eqs['conv'] if str(x) == 'frac{text{d} T }{text{d} t }')
        term = eqs['conv'][term]

    #first, create all arrays
    kernel_data = []

    if test_size == 'problem_size':
        kernel_data.append(lp.ValueArg(test_size, dtype=np.int32))

    #figure out if we need to do all species rates
    indicies = k_gen.handle_indicies(np.arange(rate_info['Ns']), '${var_name}', None, kernel_data)

    if conp:
        h_lp, h_str, map_result = lp_utils.get_loopy_arg('h',
                            indicies=['${var_name}', 'j'],
                            dimensions=(rate_info['Ns'], test_size),
                            order=loopy_opts.order)
        cp_lp, cp_str, map_result = lp_utils.get_loopy_arg('cp',
                        indicies=['${var_name}', 'j'],
                        dimensions=(rate_info['Ns'], test_size),
                        order=loopy_opts.order)
        kernel_data.extend([h_lp, cp_lp])
    else:
        u_lp, u_str, map_result = lp_utils.get_loopy_arg('u',
                        indicies=['${var_name}', 'j'],
                        dimensions=(rate_info['Ns'], test_size),
                        order=loopy_opts.order)
        cv_lp, cv_str, map_result = lp_utils.get_loopy_arg('cv',
                        indicies=['${var_name}', 'j'],
                        dimensions=(rate_info['Ns'], test_size),
                        order=loopy_opts.order)
        kernel_data.extend([u_lp, cv_lp])

    concs_lp, concs_str, map_result = lp_utils.get_loopy_arg('conc',
                        indicies=['${var_name}', 'j'],
                        dimensions=(rate_info['Ns'], test_size),
                        order=loopy_opts.order)

    omega_dot_lp, omega_dot_str, _ = lp_utils.get_loopy_arg('wdot',
                        indicies=['${omega_ind}', 'j'],
                        dimensions=(rate_info['Ns'] + 1, test_size),
                        order=loopy_opts.order)

    omega_ind = '${var_name} + 1'

    kernel_data.extend([concs_lp, omega_dot_lp])

    #put together conv/conp terms
    if conp:
        term = sp_utils.sanitize(term, subs ={
              'H[k]' : h_str,
              'dot{omega}[k]' : omega_dot_str,
              '[C][k]' : concs_str,
              '{C_p}[k]' : cp_str
            })
    else:
        term = sp_utils.sanitize(term, subs ={
              'U[k]' : u_str,
              'dot{omega}[k]' : omega_dot_str,
              '[C][k]' : concs_str,
              '{C_v}[k]' : cv_str
            })
    #now split into upper / lower halves
    factor = -1
    def separate(term):
        upper = sp.Mul(*[x for x in sp.Mul.make_args(term) if not x.has(sp.Pow) and x.has(sp.Sum)])
        lower = factor / (term / upper) #take inverse
        upper = sp.Mul(*[x if not x.has(sp.Sum) else x.function for x in sp.Mul.make_args(upper)])
        lower = lower.function
        return upper, lower

    upper, lower = separate(term)

    pre_instructions = Template("""
    ${omega_dot_str} = ${factor} * simul_reduce(sum, ${var_name}, ${upper_term}) / simul_reduce(sum, ${var_name}, ${lower_term}) {id=sum, dep=*}
    """).safe_substitute(
        omega_dot_str=omega_dot_str,
        factor=factor)
    pre_instructions = Template(pre_instructions).safe_substitute(omega_ind=0)

    instructions = ''

    instructions = Template(pre_instructions).safe_substitute(
        upper_term=upper,
        lower_term=lower)
    instructions = Template(instructions).safe_substitute(
        omega_ind=omega_ind)

    #finally do vectorization ability and specializer
    can_vectorize = loopy_opts.depth is None
    vec_spec = None
    if loopy_opts.width is not None:
        def __vec_spec(knl):
            name = 'sum_i_update' if not loopy_opts.unr else 'sum_i_outer_i_inner_update'
            #split the reduction
            knl = lp.split_reduction_outward(knl, 'j_outer')
            #and aremove the sum_0 barrier
            knl = lp.preprocess_kernel(knl)
            instruction_list = [insn if insn.id != 'sum_0'
                else insn.copy(no_sync_with=
                    insn.no_sync_with | frozenset([(name, 'any')]))
                for insn in knl.instructions]
            return knl.copy(instructions=instruction_list)
        vec_spec = __vec_spec

    return k_gen.knl_info(name='temperature_rate',
                           pre_instructions=[instructions],
                           instructions='',
                           var_name='i',
                           kernel_data=kernel_data,
                           indicies=indicies,
                           extra_subs = {'spec_ind' : 'ispec'},
                           can_vectorize=can_vectorize,
                           vectorization_specializer=vec_spec)

def get_spec_rates(eqs, loopy_opts, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for the net species rate
    kernel

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    #find summation direction and consistency check
    over_reac = loopy_opts.spec_rates_sum_over_reac
    deep = loopy_opts.depth is not None
    if deep and over_reac:
        logging.warn('Cannot use summation over reaction with a deep vectorization [not currently supported].'
                     '  Disabling...')
        over_reac = False

    var_name = 'i'
    kernel_data = []
    if test_size == 'problem_size':
        kernel_data.append(lp.ValueArg(test_size, dtype=np.int32))
    extra_inames =[]
    maps = {}
    maplist = []
    if over_reac:
        #various indicies
        reac_ind = var_name
        omega_ind = 'spec_ind'

        indicies = k_gen.handle_indicies(np.arange(rate_info['Nr']), '${var_name}', maps, kernel_data)

        #create species rates kernel
        #all species in reaction
        spec_lp, spec_str = __1Dcreator('reac_to_spec', rate_info['net']['specs'],
                                                '${spec_map}', scope=scopes.GLOBAL)

        #total # species in reaction
        num_spec_lp, num_spec_str, _ = lp_utils.get_loopy_arg('total_spec_per_reac',
                                                   ['${var_name}'],
                                                   dimensions=rate_info['net']['num_spec'].shape,
                                                   order=loopy_opts.order,
                                                   initializer=rate_info['net']['num_spec'],
                                                   dtype=rate_info['net']['num_spec'].dtype)

        #species offsets
        net_spec_offsets = np.array(
            np.cumsum(rate_info['net']['num_spec']) - rate_info['net']['num_spec'], dtype=np.int32)
        num_spec_offsets_lp, num_spec_offsets_str, _ = lp_utils.get_loopy_arg('total_spec_per_reac_offset',
                                                   ['${var_name}'],
                                                   dimensions=net_spec_offsets.shape,
                                                   order=loopy_opts.order,
                                                   initializer=net_spec_offsets,
                                                   dtype=net_spec_offsets.dtype)

        #nu array
        net_nu_lp, net_nu_str, _ = lp_utils.get_loopy_arg('net_nu',
                                            ['${spec_map}'],
                                            dimensions=rate_info['net']['nu'].shape,
                                            initializer=rate_info['net']['nu'],
                                            dtype=rate_info['net']['nu'].dtype,
                                            order=loopy_opts.order)

        #update kernel args
        kernel_data.extend([spec_lp, num_spec_lp, num_spec_offsets_lp, net_nu_lp])

    else:
        #various indicies
        reac_ind = 'reac_ind'
        omega_ind = var_name

        indicies = k_gen.handle_indicies(rate_info['net_per_spec']['map'], '${var_name}', maps, kernel_data,
                                    outmap_name='nonzero_specs', scope=scopes.GLOBAL)
        #reaction per species
        spec_to_reac = rate_info['net_per_spec']['reacs']
        spec_to_reac_lp, spec_to_reac_str, _ = lp_utils.get_loopy_arg('spec_to_reac',
                                                       ['${rxn_map}'],
                                                       dimensions=spec_to_reac.shape,
                                                       order=loopy_opts.order,
                                                       initializer=spec_to_reac,
                                                       dtype=spec_to_reac.dtype)
        #total # reactions per species
        spec_reac_count = rate_info['net_per_spec']['reac_count']
        spec_reac_count_lp, spec_rxn_counts_str, map_instructs = lp_utils.get_loopy_arg('spec_reac_count',
                                                       ['${var_name}'],
                                                       dimensions=spec_reac_count.shape,
                                                       order=loopy_opts.order,
                                                       initializer=spec_reac_count,
                                                       dtype=spec_reac_count.dtype)
        #species offsets
        spec_offsets = spec_offsets = np.array(
            np.cumsum(spec_reac_count) - spec_reac_count, dtype=np.int32)
        spec_offsets_lp, spec_offsets_str, map_instructs = lp_utils.get_loopy_arg('spec_offsets',
                                                       ['${var_name}'],
                                                       dimensions=spec_offsets.shape,
                                                       order=loopy_opts.order,
                                                       initializer=spec_offsets,
                                                       dtype=spec_offsets.dtype)

        #nu array
        spec_net_nu = rate_info['net_per_spec']['nu']
        spec_net_nu_lp, spec_net_nu_str, _ = lp_utils.get_loopy_arg('spec_net_nu',
                                                        ['${rxn_map}'],
                                                        dimensions=spec_net_nu.shape,
                                                        initializer=spec_net_nu,
                                                        dtype=spec_net_nu.dtype,
                                                        order=loopy_opts.order)
        kernel_data.extend([spec_to_reac_lp, spec_reac_count_lp, spec_offsets_lp, spec_net_nu_lp])

    #net ROP
    rop_net_lp, rop_net_str, _ = lp_utils.get_loopy_arg('rop_net',
                    indicies=['${reac_ind}', 'j'],
                    dimensions=[rate_info['Nr'], test_size],
                    order=loopy_opts.order)

    #add output map if needed
    if '${var_name}' in maps:
        omega_ind = 'omega_ind'
        maplist.append(lp_utils.generate_map_instruction(var_name,
                                                         omega_ind,
                                                         maps['${var_name}'],
                                                         affine=' + 1'))

    #and finally the wdot
    omega_dot_lp, omega_dot_str, _ = lp_utils.get_loopy_arg('wdot',
                    indicies=['${omega_ind}', 'j'],
                    dimensions=[rate_info['Ns'] + 1, test_size],
                    order=loopy_opts.order)

    #update args
    kernel_data.extend([rop_net_lp, omega_dot_lp])

    #and finally instructions
    if over_reac:
        #now the instructions
        instructions = Template(
        """
        <>net_rate = ${rop_net_str} {id=rate_init}
        <>offset = ${num_spec_offsets_str}
        <>num_spec = ${num_spec_str}
        for ispec
            <> spec_map = offset + ispec
            <> spec_ind = ${spec_str} + 1 # (offset by one for Tdot)
            <> nu = ${nu_str}
            ${omega_dot_str} = ${omega_dot_str} + nu * net_rate {dep=rate_update*}
        end
        """).safe_substitute(rop_net_str=rop_net_str,
                             spec_str=spec_str,
                             nu_str=net_nu_str,
                             omega_dot_str=omega_dot_str,
                             num_spec_offsets_str=num_spec_offsets_str,
                             num_spec_str=num_spec_str)
        instructions = Template(instructions).safe_substitute(
            reac_ind=var_name,
            spec_map='spec_map',
            spec_ind='spec_ind')

        instructions = '\n'.join([x for x in instructions.split('\n') if x.strip()])
        extra_inames = [('ispec', '0 <= ispec < num_spec')]
    else:
        #now the instructions
        instructions = Template(
        """
        <>num_rxn = ${spec_rxn_counts_str}
        <>rxn_offset = ${spec_offsets_str}
        for irxn
            <>rxn_map = rxn_offset + irxn
            <>${reac_ind} = ${spec_to_reac_str}
            ${omega_dot_str} = ${omega_dot_str} + ${spec_net_nu_str} * ${rop_net_str}
        end
        """).safe_substitute(spec_rxn_counts_str=spec_rxn_counts_str,
                             spec_offsets_str=spec_offsets_str,
                             spec_to_reac_str=spec_to_reac_str,
                             rop_net_str=rop_net_str,
                             spec_net_nu_str=spec_net_nu_str,
                             omega_dot_str=omega_dot_str,
                             omega_ind=omega_ind)

        instructions = Template(instructions).safe_substitute(
            rxn_map='rxn_map',
            omega_ind=omega_ind,
            reac_ind=reac_ind)
        instructions = '\n'.join([x for x in instructions.split('\n') if x.strip()])
        extra_inames = [('irxn', '0 <= irxn < num_rxn')]

    return k_gen.knl_info(name='spec_rates',
                           instructions=instructions,
                           var_name=var_name,
                           kernel_data=kernel_data,
                           maps=maplist,
                           extra_inames=extra_inames,
                           indicies=indicies,
                           extra_subs = {'reac_ind' : reac_ind,
                                         'omega_ind' : omega_ind})


def get_rop_net(eqs, loopy_opts, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for the net Rate of Progress
    kernels

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    #create net rop kernel
    reac_ind = '${reac_ind}'
    rev_ind = '${reac_ind}'
    pres_ind = '${reac_ind}'

    kernel_data = {}
    kernel_data['fwd'] = []
    kernel_list = ['fwd']
    maplist = {}
    maps = {}
    indicies = {}

    separated_kernels = loopy_opts.rop_net_kernels
    if separated_kernels:
        kernel_data['rev'] = []
        kernel_data['pres_mod'] = []
        kernel_list = ['fwd', 'rev', 'pres_mod'] #ordering matters

    for x in kernel_list:
        maps[x] = {}
        maplist[x] = []

    def __add_data(knl, data):
        if separated_kernels:
            kernel_data[knl].append(data)
        else:
            kernel_data['fwd'].append(data)

    def __add_to_all(data):
        for kernel in kernel_list:
            __add_data(kernel, data)

    if test_size == 'problem_size':
        __add_to_all(lp.ValueArg('problem_size', dtype=np.int32))


    indicies['fwd'] = k_gen.handle_indicies(np.arange(rate_info['Nr']), '', None, kernel_data)
    if separated_kernels:
        if rate_info['rev']['num']:
            indicies['rev'] = k_gen.handle_indicies(np.arange(rate_info['rev']['num'], dtype=np.int32),
                '', None, kernel_data)
        if rate_info['thd']['num']:
            indicies['pres_mod'] = k_gen.handle_indicies(np.arange(rate_info['thd']['num'], dtype=np.int32),
                '', None, kernel_data)

    #create the fwd rop array / str
    rop_fwd_lp, rop_fwd_str, _ = lp_utils.get_loopy_arg('rop_fwd',
                    indicies=['${reac_ind}', 'j'],
                    dimensions=[rate_info['Nr'], test_size],
                    order=loopy_opts.order)

    __add_data('fwd', rop_fwd_lp)

    if rate_info['Nr'] != rate_info['rev']['num'] and rate_info['rev']['num']:
        if not separated_kernels:
            #only have one kernel, so all maps / data go here
            kernel = 'fwd'
            rev_ind = 'rev_ind'
            #if not all reversible
            #create the reverse map
            rev_map = np.arange(rate_info['Nr'], dtype=np.int32)
            rev_map[np.where(np.logical_not(np.in1d(rev_map, rate_info['rev']['map'])))[0]] = -1
            rev_map[rate_info['rev']['map']] = np.arange(rate_info['rev']['num'], dtype=np.int32)
            rev_map_lp, rev_map_str = __1Dcreator('rev_map', rev_map, '${reac_ind}', scope=scopes.PRIVATE)
            #and add map instruction
            maplist[kernel].append(lp_utils.generate_map_instruction('${reac_ind}', rev_ind, rev_map_lp.name))
            __add_data(kernel, rev_map_lp)
        else:
            kernel = 'rev'
            #need an output map
            maps[kernel]['${reac_ind}'] = 'rev_ind'
            rev_map_lp, rev_map_str = __1Dcreator('rev_map', rate_info['rev']['map'],
                '${reac_ind}', scope=scopes.PRIVATE)
            maplist[kernel].append(lp_utils.generate_map_instruction('${reac_ind}', maps[kernel]['${reac_ind}'],
                rev_map_lp.name))
            __add_data(kernel, rev_map_lp)



    if rate_info['rev']['num']:
        #have reversible reaction
        #create the rev rop array / str
        rop_rev_lp, rop_rev_str, _ = lp_utils.get_loopy_arg('rop_rev',
                    indicies=[rev_ind, 'j'],
                    dimensions=[rate_info['rev']['num'], test_size],
                    order=loopy_opts.order)
        __add_data('rev', rop_rev_lp)

    #create the pressure modification array
    if rate_info['Nr'] != rate_info['thd']['num'] and rate_info['thd']['num']:
        if not separated_kernels:
            #only have one kernel, so all maps / data go here
            kernel = 'fwd'
            pres_ind = 'pres_ind'
            #if have pdep reactions and not all reacs are pdep, create map
            pres_map = np.arange(rate_info['Nr'], dtype=np.int32)
            pres_map[np.where(np.logical_not(np.in1d(pres_map, rate_info['thd']['map'])))[0]] = -1
            pres_map[rate_info['thd']['map']] = np.arange(rate_info['thd']['num'], dtype=np.int32)
            pres_map_lp, pres_map_str = __1Dcreator('pres_map', pres_map, '${reac_ind}', scope=scopes.PRIVATE)
            #and add map instruction
            maplist[kernel].append(lp_utils.generate_map_instruction('${reac_ind}', pres_ind, pres_map_lp.name))
            __add_data(kernel, pres_map_lp)
        else:
            kernel = 'pres_mod'
            #need an output map
            maps[kernel]['${reac_ind}'] = 'pres_ind'
            pres_map_lp, pres_map_str = __1Dcreator('thd_map', rate_info['thd']['map'], #named thd_map as used elsewhere
                '${reac_ind}', scope=scopes.PRIVATE)
            maplist[kernel].append(lp_utils.generate_map_instruction('${reac_ind}', maps[kernel]['${reac_ind}'],
                pres_map_lp.name))
            __add_data(kernel, pres_map_lp)

    if rate_info['thd']['num']:
        pres_mod_lp, pres_mod_str, _ = lp_utils.get_loopy_arg('pres_mod',
                                              [pres_ind, 'j'],
                                              dimensions=(rate_info['thd']['num'], test_size),
                                              order=loopy_opts.order)
        __add_data('pres_mod', pres_mod_lp)

    #add rop net to all kernels:
    rop_strs = {}
    for name in kernel_list:
        var_name = '${reac_ind}'
        if var_name in maps[name]:
            var_name = maps[name][var_name]
        rop_net_lp, rop_net_str, _ = lp_utils.get_loopy_arg('rop_net',
                        indicies=[var_name, 'j'],
                        dimensions=[rate_info['Nr'], test_size],
                        order=loopy_opts.order)
        __add_data(name, rop_net_lp)
        rop_strs[name] = rop_net_str

    if not separated_kernels:
        #now the instructions
        instructions = Template(
        """
        <>net_rate = ${rop_fwd_str} {id=rate_update}
        ${rev_update}
        ${pmod_update}
        ${rop_net_str} = net_rate {dep=rate_update*}
        """).safe_substitute(rop_fwd_str=rop_fwd_str,
                             rop_net_str=rop_strs['fwd'])

        #reverse update
        if rate_info['rev']['num']:
            rev_update_instructions = Template(
        """
        if rev_ind >= 0
            net_rate = net_rate - ${rop_rev_str} {id=rate_update_rev, dep=rate_update}
        end
        """).safe_substitute(
                rev_map_str=rev_map_str,
                rop_rev_str=rop_rev_str)
        else:
            rev_update_instructions = ''

        #pmod update
        if rate_info['thd']['num']:
            pmod_update_instructions = Template(
        """
        if pres_ind >= 0
            net_rate = net_rate * ${pres_mod_str} {id=rate_update_pmod, dep=rate_update${rev_dep}}
        end
        """).safe_substitute(
                rev_dep=':rate_update_rev' if rate_info['rev']['num'] else '',
                pres_map_str=pres_map_str,
                pres_mod_str=pres_mod_str)
        else:
            pmod_update_instructions = ''

        instructions = Template(instructions).safe_substitute(
            rev_update=rev_update_instructions,
            pmod_update=pmod_update_instructions)

        instructions = Template(instructions).safe_substitute(
            reac_ind='i',
            rev_ind=rev_ind,
            pres_ind=pres_ind)

        instructions = '\n'.join([x for x in instructions.split('\n') if x.strip()])

        return k_gen.knl_info(name='rop_net',
                           instructions=instructions,
                           var_name='i',
                           kernel_data=kernel_data['fwd'],
                           maps=maplist['fwd'],
                           indicies=indicies['fwd'],
                           extra_subs = {'reac_ind' : 'i'})

    else:
        infos = []
        for kernel in kernel_list:
            if kernel == 'fwd':
                instructions = Template(
            """
            ${rop_net_str} = ${rop_fwd_str}
                    """).safe_substitute(rop_fwd_str=rop_fwd_str,
                                         rop_net_str=rop_strs['fwd'])
            elif kernel == 'rev':
                instructions = Template(
            """
            ${rop_net_str} = ${rop_net_str} - ${rop_rev_str}
                    """).safe_substitute(rop_rev_str=rop_rev_str,
                                         rop_net_str=rop_strs['rev'])
            else:
                instructions = Template(
            """
            ${rop_net_str} = ${rop_net_str} * ${pres_mod_str}
                    """).safe_substitute(pres_mod_str=pres_mod_str,
                                         rop_net_str=rop_strs['pres_mod'])

            instructions = '\n'.join([x for x in instructions.split('\n') if x.strip()])
            infos.append(k_gen.knl_info(name='rop_net_{}'.format(kernel),
                           instructions=instructions,
                           var_name='i',
                           kernel_data=kernel_data[kernel],
                           maps=maplist[kernel],
                           indicies=indicies[kernel],
                           extra_subs = {'reac_ind' : 'i'}))
        return infos


def get_rop(eqs, loopy_opts, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for the Rate of Progress
    kernels

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    #create ROP kernels

    def __rop_create(direction):
        mapname = {}
        maplist = []
        reac_ind = 'i'

        #indicies
        kernel_data = []
        if test_size == 'problem_size':
            kernel_data.append(lp.ValueArg(test_size, dtype=np.int32))
        indicies = k_gen.handle_indicies(np.arange(rate_info[direction]['num']), '${reac_ind}', mapname, kernel_data)

        #we need species lists, nu lists, etc.
        #first create offsets for map
        net_spec_offsets = np.array(
            np.cumsum(rate_info[direction]['num_spec']) - rate_info[direction]['num_spec'], dtype=np.int32)
        num_spec_offsets_lp, num_spec_offsets_str, map_result = lp_utils.get_loopy_arg('spec_offset_{}'.format(direction),
                                                       ['${reac_ind}'],
                                                       dimensions=net_spec_offsets.shape,
                                                       order=loopy_opts.order,
                                                       initializer=net_spec_offsets,
                                                       dtype=net_spec_offsets.dtype,
                                                       map_name=mapname,
                                                       map_result='${spec_offset}')
        if '${reac_ind}' in maplist:
            maplist.append(map_instructs['${reac_ind}'])

        #nu lists
        nu_lp, nu_str, _ = lp_utils.get_loopy_arg('nu_{}'.format(direction),
                                                ['${spec_map}'],
                                                dimensions=rate_info[direction]['nu'].shape,
                                                initializer=rate_info[direction]['nu'],
                                                dtype=rate_info[direction]['nu'].dtype,
                                                order=loopy_opts.order)

        #species lists
        spec_lp, spec_str, _ = lp_utils.get_loopy_arg('spec_{}'.format(direction),
                                                ['${spec_map}'],
                                                dimensions=rate_info[direction]['specs'].shape,
                                                initializer=rate_info[direction]['specs'],
                                                dtype=rate_info[direction]['specs'].dtype,
                                                order=loopy_opts.order)

        #species counts
        num_spec_lp, num_spec_str, map_result = lp_utils.get_loopy_arg('num_spec_{}'.format(direction),
                                                       ['${reac_ind}'],
                                                       dimensions=rate_info[direction]['num_spec'].shape,
                                                       order=loopy_opts.order,
                                                       initializer=rate_info[direction]['num_spec'],
                                                       dtype=rate_info[direction]['num_spec'].dtype,
                                                       map_result='${spec_offset}')

        #rate constants
        rateconst_arr, rateconst_str, _ = lp_utils.get_loopy_arg('kf' if direction == 'fwd' else 'kr',
                        indicies=['${reac_ind}', 'j'],
                        dimensions=[rate_info[direction]['num'], test_size],
                        order=loopy_opts.order)

        #concentrations
        concs_lp, concs_str, _ = lp_utils.get_loopy_arg('conc',
                                                        indicies=['${spec_ind}', 'j'],
                                                        dimensions=(rate_info['Ns'], test_size),
                                                        order=loopy_opts.order)

        #and finally the ROP values
        rop_arr, rop_str, _ = lp_utils.get_loopy_arg('ropeval_{}'.format(direction),
                        indicies=['${reac_ind}', 'j'],
                        dimensions=[rate_info[direction]['num'], test_size],
                        order=loopy_opts.order)

        #update kernel data
        kernel_data.extend([num_spec_offsets_lp, concs_lp, nu_lp, spec_lp, num_spec_lp, rateconst_arr, rop_arr])

        #instructions
        rop_instructions = Template(Template(
    """
    <>rop_temp = ${rateconst_str} {id=rop_init}
    <>spec_offset = ${num_spec_offsets_str}
    <>num_spec = ${num_spec_str}
    for ispec
        <>spec_map = spec_offset + ispec
        <>spec_ind = ${spec_str} {id=spec_ind}
    ${rop_temp_eval}
    end
    ${rop_str} = rop_temp {dep=rop_fin*}
    """).safe_substitute(rateconst_str=rateconst_str,
                         rop_str=rop_str,
                         num_spec_offsets_str=num_spec_offsets_str,
                         spec_str=spec_str,
                         num_spec_str=num_spec_str))

        #if all integers, it's much faster to use multiplication
        allint_eval = Template(
    """
    <>conc_temp = 1.0d {id=conc_init}
    <>nu = ${nu_str}
    for inu
        conc_temp = conc_temp * ${concs_str} {id=conc_update, dep=spec_ind:conc_init}
    end
    rop_temp = rop_temp * conc_temp {id=rop_fin, dep=conc_update}""").safe_substitute(
                         nu_str=nu_str,
                         concs_str=concs_str)
        #cleanup
        allint_eval = '\n'.join('    ' + line for line in allint_eval.split('\n') if line)

        #if we need to use powers, do so
        fractional_eval = Template(
    """
    if int(${nu_sum}) == ${nu_sum}
    ${allint}
    else
        rop_temp = rop_temp * (${concs_str})**(${nu_str}) {id=rop_fin2}
    end
    """).safe_substitute(allint=allint_eval,
                         nu_str=nu_str,
                         concs_str=concs_str)
        #cleanup
        fractional_eval = '\n'.join('    ' + line for line in fractional_eval.split('\n') if line)


        if not rate_info[direction]['allint']:
            rop_instructions = rop_instructions.safe_substitute(
                rop_temp_eval=fractional_eval)
        else:
            rop_instructions = rop_instructions.safe_substitute(
                rop_temp_eval=allint_eval)
        rop_instructions = Template(rop_instructions).safe_substitute(
                                 reac_ind=reac_ind,
                                 spec_map='spec_map',
                                 spec_ind='spec_ind')

        #and finally extra inames
        extra_inames = [('ispec', '0 <= ispec < num_spec'),
                        ('inu', '0 <= inu < nu')]

        #and return the rateconst
        return k_gen.knl_info(name='rop_eval_{}'.format(direction),
                       instructions=rop_instructions,
                       var_name=reac_ind,
                       kernel_data=kernel_data,
                       maps=maplist,
                       extra_inames=extra_inames,
                       indicies=indicies,
                       extra_subs={'reac_ind' : reac_ind})

    infos = [__rop_create('fwd')]
    if rate_info['rev']['num']:
        infos.append(__rop_create('rev'))
    return infos



def get_rxn_pres_mod(eqs, loopy_opts, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for pressure modification
    term of the forward reaction rates.

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    #start developing the ci kernel
    #rate info and reac ind
    reac_ind = 'i'
    kernel_data = []
    if test_size == 'problem_size':
        kernel_data.append(lp.ValueArg('problem_size', dtype=np.int32))

    #set of eqn's doesn't matter
    conp_eqs = eqs['conp']

    #create the third body conc pres-mod kernel

    #first, we need all third body reactions that are not falloff
    num_thd = rate_info['thd']['num']
    non_fall_thd = np.where(np.logical_not(np.in1d(rate_info['thd']['map'],
                        rate_info['fall']['map'])))[0]

    thd_map = {}
    thd_maplist = []
    indicies = k_gen.handle_indicies(non_fall_thd, '${reac_ind}', thd_map, kernel_data,
                        outmap_name='thd_only_ind')

    #next the thd_body_conc's
    num_thd = rate_info['thd']['num']
    reac_mapname = '${reac_ind}_map'
    thd_lp, thd_str, map_instructs = lp_utils.get_loopy_arg('thd_conc',
                                                indicies=['${reac_ind}', 'j'],
                                                dimensions=(num_thd, test_size),
                                                order=loopy_opts.order,
                                                map_name=thd_map,
                                                map_result=reac_mapname)
    #add the map
    if '${reac_ind}' in map_instructs:
        thd_maplist.append(map_instructs['${reac_ind}'])

    #and the pressure mod term
    pres_mod_lp, pres_mod_str, map_instructs = lp_utils.get_loopy_arg('pres_mod',
                                      ['${reac_ind}', 'j'],
                                      dimensions=(num_thd, test_size),
                                      order=loopy_opts.order,
                                      map_name=thd_map)

    thd_instructions = Template(
"""
${pres_mod} = ${thd_conc} {dep=decl}

""").safe_substitute(pres_mod=pres_mod_str,
                     thd_conc=thd_str)
    thd_instructions = Template(thd_instructions).safe_substitute(reac_ind=reac_ind)
    #and the args
    kernel_data.extend([thd_lp, pres_mod_lp])

    #add to the info list
    info_list = [
        k_gen.knl_info(name='ci_thd',
                   instructions=thd_instructions,
                   var_name=reac_ind,
                   kernel_data=kernel_data,
                   maps=thd_maplist,
                   indicies=indicies,
                   extra_subs={'reac_ind' : reac_ind})]

    #and now the falloff kernel
    kernel_data = []
    if test_size == 'problem_size':
        kernel_data.append(lp.ValueArg('problem_size', dtype=np.int32))
    fall_maplist = []
    fall_map = {}
    indicies = k_gen.handle_indicies(np.arange(rate_info['fall']['num'], dtype=np.int32),
                        '${reac_ind}', fall_map, kernel_data,
                        outmap_name='fall_inds', scope=scopes.GLOBAL)

    #the falloff vs chemically activated indicator
    fall_type = rate_info['fall']['ftype']
    fall_type_lp, fall_type_str, map_instructs = lp_utils.get_loopy_arg('fall_type',
                                      ['${reac_ind}'],
                                      dimensions=(fall_type.shape),
                                      order=loopy_opts.order,
                                      initializer=fall_type,
                                      dtype=fall_type.dtype)

    #the blending term
    Fi_lp, Fi_str, _ = lp_utils.get_loopy_arg('Fi',
                            indicies=['${reac_ind}', 'j'],
                            dimensions=[rate_info['fall']['num'], test_size],
                            order=loopy_opts.order)
    #the Pr array
    Pr_lp, Pr_str, _ = lp_utils.get_loopy_arg('Pr',
                            indicies=['${reac_ind}', 'j'],
                            dimensions=[rate_info['fall']['num'], test_size],
                            order=loopy_opts.order)

    #find the falloff -> thd map
    map_name = 'pres_mod_map'
    pres_mod_ind = 'pres_mod_ind'
    pres_mod_map = np.where(np.in1d(rate_info['thd']['map'], rate_info['fall']['map']))[0]
    pres_mod_map, pres_mod_str = __1Dcreator(map_name, pres_mod_map, '${pres_mod_ind}')
    fall_maplist.append(lp_utils.generate_map_instruction(
                                        newname=pres_mod_ind,
                                        map_arr=map_name,
                                        oldname='${reac_ind}'))


    #and the pressure mod term
    pres_mod_lp, pres_mod_str, _ = lp_utils.get_loopy_arg('pres_mod',
                                      ['${pres_mod_ind}', 'j'],
                                      dimensions=(num_thd, test_size),
                                      order=loopy_opts.order)

    #update the args
    kernel_data.extend([Fi_lp, Pr_lp, fall_type_lp, pres_mod_lp, pres_mod_map])

    fall_instructions = Template(
"""
<>ci_temp = ${Fi_str} / (1 + ${Pr_str}) {id=ci_decl}
if ${fall_type} == 0
    ci_temp = ci_temp * ${Pr_str} {id=ci_update, dep=ci_decl}
end
${pres_mod} = ci_temp {dep=ci_update}
""").safe_substitute(Fi_str=Fi_str,
                     Pr_str=Pr_str,
                     pres_mod=pres_mod_str,
                     fall_type=fall_type_str)
    fall_instructions = Template(fall_instructions).safe_substitute(reac_ind=reac_ind,
        pres_mod_ind=pres_mod_ind)

    #add to the info list
    info_list.append(
        k_gen.knl_info(name='ci_fall',
                   instructions=fall_instructions,
                   var_name=reac_ind,
                   kernel_data=kernel_data,
                   maps=fall_maplist,
                   indicies=indicies,
                   extra_subs={'reac_ind' : reac_ind}))

    return info_list



def get_rev_rates(eqs, loopy_opts, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for reverse reaction rates

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """
    #start developing the Kc kernel
    #rate info and reac ind
    reac_ind = 'i'
    kernel_data = []

    if test_size == 'problem_size':
        kernel_data.append(lp.ValueArg(test_size, dtype=np.int32))

    #set of eqn's doesn't matter
    conp_eqs = eqs['conp']

    #add the reverse map
    maps = []
    rev_map = {}
    indicies = k_gen.handle_indicies(rate_info['rev']['map'], '${reac_ind}', rev_map, kernel_data)

    #find Kc equation
    Kc_sym = next(x for x in conp_eqs if str(x) == '{K_c}[i]')
    Kc_eqn = conp_eqs[Kc_sym]
    nu_sym = next(x for x in Kc_eqn.free_symbols if str(x) == 'nu[k, i]')
    B_sym  = next(x for x in Kc_eqn.free_symbols if str(x) == 'B[k]')
    kr_sym = next(x for x in conp_eqs if str(x) == '{k_r}[i]')
    kf_sym = next(x for x in conp_eqs if str(x) == '{k_f}[i]')

    #create nu_sum
    nu_sum_lp, nu_sum_str, map_result = lp_utils.get_loopy_arg('nu_sum',
                                                   ['${reac_ind}'],
                                                   dimensions=rate_info['net']['nu_sum'].shape,
                                                   order=loopy_opts.order,
                                                   initializer=rate_info['net']['nu_sum'],
                                                   dtype=rate_info['net']['nu_sum'].dtype,
                                                   map_name=rev_map)
    if '${reac_ind}' in map_result:
        maps.append(map_result['${reac_ind}'])
    #all species in reaction
    spec_lp, spec_str = __1Dcreator('allspec_in_reac', rate_info['net']['specs'],
                                            '${spec_map}', scope=scopes.GLOBAL)
    #total # species in reaction
    num_spec_lp, num_spec_str, map_result = lp_utils.get_loopy_arg('total_spec_per_reac',
                                                   ['${reac_ind}'],
                                                   dimensions=rate_info['net']['num_spec'].shape,
                                                   order=loopy_opts.order,
                                                   initializer=rate_info['net']['num_spec'],
                                                   dtype=rate_info['net']['num_spec'].dtype,
                                                   map_name=rev_map)
    #species offsets
    net_spec_offsets = np.array(
        np.cumsum(rate_info['net']['num_spec']) - rate_info['net']['num_spec'], dtype=np.int32)
    num_spec_offsets_lp, num_spec_offsets_str, map_result = lp_utils.get_loopy_arg('total_spec_per_reac_offset',
                                                   ['${reac_ind}'],
                                                   dimensions=net_spec_offsets.shape,
                                                   order=loopy_opts.order,
                                                   initializer=net_spec_offsets,
                                                   dtype=net_spec_offsets.dtype,
                                                   map_name=rev_map)
    #B array
    B_lp, B_str, _ = lp_utils.get_loopy_arg('B',
                                            ['${spec_ind}', 'j'],
                                            dimensions=(rate_info['Ns'], test_size),
                                            order=loopy_opts.order)
    net_nu_lp, net_nu_str, _ = lp_utils.get_loopy_arg('net_nu',
                                            ['${spec_map}'],
                                            dimensions=rate_info['net']['nu'].shape,
                                            initializer=rate_info['net']['nu'],
                                            dtype=rate_info['net']['nu'].dtype,
                                            order=loopy_opts.order)

    #and the Kc array
    Kc_lp, Kc_str, _ = lp_utils.get_loopy_arg('Kc',
                                              ['${reac_ind}', 'j'],
                                              dimensions=(rate_info['rev']['num'], test_size),
                                              order=loopy_opts.order)

    #modify Kc equation
    Kc_eqn = sp_utils.sanitize(conp_eqs[Kc_sym],
                              symlist={'nu[k, i]' : nu_sym,
                                       'B[k]' : B_sym},
                              subs={sp.Sum(nu_sym, (sp.Idx('k'), 1, sp.Symbol('N_s'))) : nu_sum_str})
    Kc_eqn_Pres = next(x for x in sp.Mul.make_args(Kc_eqn) if x.has(sp.Symbol('R_u')))
    Kc_eqn_exp = Kc_eqn / Kc_eqn_Pres
    Kc_eqn_exp = sp_utils.sanitize(Kc_eqn_exp,
                                   symlist={'nu[k, i]' : nu_sym,
                                            'B[k]' : B_sym},
                                   subs={sp.Sum(B_sym * nu_sym, (sp.Idx('k'), 1, sp.Symbol('N_s'))) : 'B_sum'})

    #create the kf array / str
    kf_arr, kf_str, map_result = lp_utils.get_loopy_arg('kf',
                    indicies=['${reac_ind}', 'j'],
                    dimensions=[rate_info['Nr'], test_size],
                    order=loopy_opts.order,
                    map_name=rev_map)

    #create the kr array / str
    kr_arr, kr_str, _ = lp_utils.get_loopy_arg('kr',
                    indicies=['${reac_ind}', 'j'],
                    dimensions=[rate_info['rev']['num'], test_size],
                    order=loopy_opts.order)

    #get the kr eqn
    Kc_temp_str = 'Kc_temp'
    #for some reason this substitution is poorly behaved
    #hence we just do this rather than deriving from sympy for the moment
    kr_eqn = sp.Symbol(kf_str) / sp.Symbol(Kc_temp_str)
    #kr_eqn = sp_utils.sanitize(conp_eqs[kr_sym][(reversible_type.non_explicit,)],
    #                           symlist={'{k_f}[i]' : sp.Symbol('kf[i]'),
    #                                    '{K_c}[i]' : sp.Symbol('Kc[i]')},
    #                           subs={'kf[i]' : kf_str,
    #                                'Kc[i]' : Kc_temp_str})

    #update kernel data
    kernel_data.extend([nu_sum_lp, spec_lp, num_spec_lp, num_spec_offsets_lp,
        B_lp, Kc_lp, net_nu_lp, kf_arr, kr_arr])

    #create the pressure product loop
    pressure_prod = Template(
    """<> P_sum_end = abs(${nu_sum}) {id=P_bound}
    <> P_sum = 1.0d {id=P_init}
    if ${nu_sum} > 0
        <> P_val = P_a / R_u {id=P_val_decl}
    else
        P_val = R_u / P_a {id=P_val_decl1}
    end
    for P_sum_ind
        P_sum = P_sum * P_val {id=P_accum, dep=P_val_decl:P_val_decl1:P_bound:P_init}
    end
    #P_sum = pown(P_val, P_sum_end)
    """).safe_substitute(nu_sum=nu_sum_str)

    if not rate_info['net']['allint']:
        #if not all integers, need to add outer if statment to check integer status
        pressure_prod = Template("""
    if int(${nu_sum}) == ${nu_sum}
    ${pprod}
    else
        P_sum = (P_a / R_u)**(${nu_sum}) {id=P_accum}
    end""").safe_substitute(nu_sum=nu_sum_str,
                            pprod='\n'.join('    ' + line for line in
                                           pressure_prod.split('\n') if line))

    #and the b sum loop
    Bsum_inst = Template(
    """<>num_spec = ${num_spec} {id=B_bound}
    <>offset = ${spec_offset} {id=offset}
    <>B_sum = 0 {id=B_init}
    for spec_count
        <>spec = ${spec_mapper} {dep=offset:B_bound}
        if ${net_nu} != 0
            B_sum = B_sum + ${net_nu} * ${B_val} {id=B_accum, dep=B_init}
        end
    end
    B_sum = exp(B_sum) {id=B_final, dep=B_accum}""").safe_substitute(num_spec=num_spec_str,
                        spec_offset=num_spec_offsets_str,
                        spec_mapper=spec_str,
                        nu_val=nu_sum_str,
                        net_nu=net_nu_str,
                        B_val=B_str
                        )
    Bsum_inst = Template(Bsum_inst).safe_substitute(
        spec_map='offset + spec_count',
        spec_ind='spec')

    Rate_assign = Template(
"""<>${Kc_temp_str} = P_sum * B_sum {dep=P_accum:B_final}
${Kc_val} = ${Kc_temp_str}
${kr_val} = ${rev_eqn}
""").safe_substitute(Kc_val=Kc_str,
                     Kc_temp_str=Kc_temp_str,
                     kr_val=kr_str,
                     rev_eqn=kr_eqn)

    instructions = '\n'.join([Bsum_inst, pressure_prod, Rate_assign])
    instructions = Template(instructions).safe_substitute(reac_ind=reac_ind)

    #create the extra inames
    extra_inames = [('P_sum_ind', '0 <= P_sum_ind < {}'.format('P_sum_end')),
                    ('spec_count', '0 <= spec_count < {}'.format('num_spec'))]


    #and return the rateinfo
    return k_gen.knl_info(name='rateconst_Kc',
                   instructions=instructions,
                   var_name=reac_ind,
                   kernel_data=kernel_data,
                   maps=maps,
                   extra_inames=extra_inames,
                   indicies=indicies,
                   parameters={'P_a' : np.float64(chem.PA), 'R_u' : np.float64(chem.RU)},
                   extra_subs={'reac_ind' : reac_ind})

def get_thd_body_concs(eqs, loopy_opts, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for third body concentrations

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    Ns = rate_info['Ns']
    num_thd = rate_info['thd']['num']
    reac_ind = 'i'

    #create args
    indicies = ['${species_ind}', 'j']
    concs_lp, concs_str, _ = lp_utils.get_loopy_arg('conc',
                                                    indicies=indicies,
                                                    dimensions=(Ns, test_size),
                                                    order=loopy_opts.order)
    concs_str = Template(concs_str)

    indicies = ['${reac_ind}', 'j']
    thd_lp, thd_str, _ = lp_utils.get_loopy_arg('thd_conc',
                                                indicies=indicies,
                                                dimensions=(num_thd, test_size),
                                                order=loopy_opts.order)

    T_arr = lp.GlobalArg('T_arr', shape=(test_size,), dtype=np.float64)
    P_arr = lp.GlobalArg('P_arr', shape=(test_size,), dtype=np.float64)

    thd_eff_ns = np.ones(num_thd)
    num_specs = rate_info['thd']['spec_num'].copy()
    spec_list = rate_info['thd']['spec'].copy()
    thd_effs = rate_info['thd']['eff'].copy()

    last_spec = Ns - 1
    #first, we must do some surgery to get _our_ form of the thd-body efficiencies
    for i in range(num_specs.size):
        num = num_specs[i]
        offset = np.sum(num_specs[:i])
        #check if Ns has a non-default efficiency
        if last_spec in spec_list[offset:offset+num]:
            ind = np.where(spec_list[offset:offset+num] == last_spec)[0][0]
            #set the efficiency
            thd_eff_ns[i] = thd_effs[offset + ind]
            #delete from the species list
            spec_list = np.delete(spec_list, offset + ind)
            #delete from effiency list
            thd_effs = np.delete(thd_effs, offset + ind)
            #subtract from species num
            num_specs[i] -= 1
        #and subtract from efficiencies
        thd_effs[offset:offset+num_specs[i]] -= thd_eff_ns[i]
        if thd_eff_ns[i] != 1:
            #we need to add all the other species :(
            #get updated species list
            to_add = np.array(range(Ns - 1), dtype=np.int32)
            #and efficiencies
            eff = np.array([1 - thd_eff_ns[i] for spec in range(Ns - 1)])
            eff[spec_list[offset:offset+num_specs[i]]] = thd_effs[offset:offset+num_specs[i]]
            #delete from the species list / efficiencies
            spec_list = np.delete(spec_list, range(offset, offset+num_specs[i]))
            thd_effs = np.delete(thd_effs,  range(offset, offset+num_specs[i]))
            #insert
            spec_list = np.insert(spec_list, offset, to_add)
            thd_effs = np.insert(thd_effs, offset, eff)
            #and update number
            num_specs[i] = to_add.size


    #and temporary variables:
    thd_type_lp, thd_type_str = __1Dcreator('thd_type', rate_info['thd']['type'], scope=scopes.GLOBAL)
    thd_eff_lp, thd_eff_str = __1Dcreator('thd_eff', thd_effs)
    thd_spec_lp, thd_spec_str = __1Dcreator('thd_spec', spec_list)
    thd_num_spec_lp, thd_num_spec_str = __1Dcreator('thd_spec_num', num_specs)
    thd_eff_ns_lp, thd_eff_ns_str = __1Dcreator('thd_eff_ns', thd_eff_ns)

    #calculate offsets
    offsets = np.cumsum(num_specs, dtype=np.int32) - num_specs
    thd_offset_lp, thd_offset_str = __1Dcreator('thd_offset', offsets)

    #kernel data
    kernel_data = [T_arr, P_arr, concs_lp, thd_lp, thd_type_lp,
                   thd_eff_lp, thd_spec_lp, thd_num_spec_lp, thd_offset_lp,
                   thd_eff_ns_lp]

    #maps
    out_map = {}
    outmap_name = 'out_map'
    indicies = rate_info['thd']['map'].astype(dtype=np.int32)
    indicies = k_gen.handle_indicies(indicies, reac_ind, out_map, kernel_data)
    #extra loops
    extra_inames = [('k', '0 <= k < num')]

    #generate instructions
    instructions = Template("""
<> offset = ${offset}
<> num_temp = ${num_spec_str} {id=num0}
if ${type_str} == 1 # single species
    <> thd_temp = ${conc_spec} {id=thd0, dep=num0}
    num_temp = 0 {id=num1, dep=num0}
else
    thd_temp = P_arr[j] * ${thd_eff_ns_str} / (R * T_arr[j]) {id=thd1, dep=num0}
end
<> num = num_temp {dep=num*}
for k
    thd_temp = thd_temp + ${thd_eff} * ${conc_thd_spec} {id=thdcalc}
end
${thd_str} = thd_temp {dep=thd*}
""")

    #sub in instructions
    instructions = Template(
        instructions.safe_substitute(
            offset=thd_offset_str,
            type_str=thd_type_str,
            conc_spec=concs_str.safe_substitute(
                    species_ind=thd_spec_str),
            num_spec_str=thd_num_spec_str,
            thd_eff=Template(thd_eff_str).safe_substitute(reac_ind='offset + k'),
            conc_thd_spec=concs_str.safe_substitute(
                species_ind=Template(thd_spec_str).safe_substitute(
                    reac_ind='offset + k')),
            thd_str=thd_str,
            thd_eff_ns_str=thd_eff_ns_str
        )
    ).safe_substitute(reac_ind=reac_ind)


    #create info
    info = k_gen.knl_info('eval_thd_body_concs',
                         instructions=instructions,
                         var_name=reac_ind,
                         kernel_data=kernel_data,
                         extra_inames=extra_inames,
                         indicies=indicies,
                         parameters={'R' : chem.RU},
                         extra_subs={'reac_ind' : reac_ind})
    return info

def get_cheb_arrhenius_rates(eqs, loopy_opts, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for cheb rate constants

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator

    """

    #the equation set doesn't matter for this application
    #just use conp
    conp_eqs = eqs['conp']

    rate_eqn_pre = get_rate_eqn(eqs)
    #find the cheb equation
    cheb_eqn = next(x for x in conp_eqs if str(x) == 'log({k_f}[i])/log(10)')
    cheb_form, cheb_eqn = cheb_eqn, conp_eqs[cheb_eqn][(reaction_type.cheb,)]
    cheb_form = sp.Pow(10, sp.Symbol('kf_temp'))

    #make nice symbols
    Tinv = sp.Symbol('T_inv')
    logP = sp.Symbol('logP')
    Pmax, Pmin, Tmax, Tmin = sp.symbols('Pmax Pmin Tmax Tmin')
    Pred, Tred = sp.symbols('Pred Tred')

    #get tilde{T}, tilde{P}
    T_red = next(x for x in conp_eqs if str(x) == 'tilde{T}')
    P_red = next(x for x in conp_eqs if str(x) == 'tilde{P}')

    Pred_eqn = sp_utils.sanitize(conp_eqs[P_red], subs={sp.log(sp.Symbol('P_{min}')) : Pmin,
                                               sp.log(sp.Symbol('P_{max}')) : Pmax,
                                               sp.log(sp.Symbol('P')) : logP})

    Tred_eqn = sp_utils.sanitize(conp_eqs[T_red], subs={sp.S.One / sp.Symbol('T_{min}') : Tmin,
                                               sp.S.One / sp.Symbol('T_{max}') : Tmax,
                                               sp.S.One / sp.Symbol('T') : Tinv})

    #number of cheb reactions
    num_cheb = rate_info['cheb']['num']
    # degree of pressure polynomial per reaction
    num_P = np.array(rate_info['cheb']['num_P'], dtype=np.int32)
    # degree of temperature polynomial per reaction
    num_T = np.array(rate_info['cheb']['num_T'], dtype=np.int32)

    #max degrees in mechanism
    maxP = int(np.max(num_P))
    maxT = int(np.max(num_T))
    minP = int(np.min(num_P))
    minT = int(np.min(num_T))
    poly_max = int(np.maximum(maxP, maxT))

    #now we start defining parameters / temporary variable

    #workspace vars
    pres_poly_lp = lp.TemporaryVariable('pres_poly', shape=(poly_max,),
        dtype=np.float64, scope=scopes.PRIVATE, read_only=False)
    temp_poly_lp = lp.TemporaryVariable('temp_poly', shape=(poly_max,),
        dtype=np.float64, scope=scopes.PRIVATE, read_only=False)
    pres_poly_str = Template('pres_poly[${pres_poly_ind}]')
    temp_poly_str = Template('temp_poly[${temp_poly_ind}]')

    #chebyshev parameters
    params = np.zeros((num_cheb, maxT, maxP))
    for i, p in enumerate(rate_info['cheb']['params']):
        params[i, :num_T[i], :num_P[i]] = p[:, :]

    indicies=['${reac_ind}', '${temp_poly_ind}', '${pres_poly_ind}']
    params_lp = lp.TemporaryVariable('cheb_params', shape=params.shape,
        initializer=params, scope=scopes.GLOBAL, read_only=True)
    params_str = Template('cheb_params[' + ','.join(indicies) +  ']')

    #finally the min/maxs & param #'s
    numP_lp = lp.TemporaryVariable('cheb_numP', shape=num_P.shape,
        initializer=num_P, dtype=np.int32, read_only=True, scope=scopes.GLOBAL)
    numP_str = 'cheb_numP[${reac_ind}]'
    numT_lp = lp.TemporaryVariable('cheb_numT', shape=num_T.shape,
        initializer=num_T, dtype=np.int32, read_only=True, scope=scopes.GLOBAL)
    numT_str = 'cheb_numT[${reac_ind}]'

    # limits for cheby polys
    Plim = np.log(np.array(rate_info['cheb']['Plim'], dtype=np.float64))
    Tlim = 1. / np.array(rate_info['cheb']['Tlim'], dtype=np.float64)

    indicies = ['${reac_ind}', '${lim_ind}']
    plim_lp = lp.TemporaryVariable('cheb_plim', shape=Plim.shape,
        initializer=Plim, scope=scopes.GLOBAL, read_only=True)
    tlim_lp = lp.TemporaryVariable('cheb_tlim', shape=Tlim.shape,
        initializer=Tlim, scope=scopes.GLOBAL, read_only=True)
    plim_str = Template('cheb_plim[' + ','.join(indicies) + ']')
    tlim_str = Template('cheb_tlim[' + ','.join(indicies) + ']')

    T_arr, T_arr_str, _ = lp_utils.get_loopy_arg('T_arr',
                          indicies=['j'],
                          dimensions=[test_size],
                          order=loopy_opts.order,
                          dtype=np.float64)
    P_arr, P_arr_str, _= lp_utils.get_loopy_arg('P_arr',
                          indicies=['j'],
                          dimensions=[test_size],
                          order=loopy_opts.order,
                          dtype=np.float64)
    kernel_data = [params_lp, numP_lp, numT_lp, plim_lp, tlim_lp,
                    pres_poly_lp, temp_poly_lp, T_arr, P_arr]

    if test_size == 'problem_size':
        kernel_data.append(lp.ValueArg(test_size, dtype=np.int32))

    reac_ind = 'i'
    out_map = {}
    outmap_name = 'out_map'
    indicies = rate_info['cheb']['map'].astype(dtype=np.int32)
    Nr = rate_info['Nr']

    indicies = k_gen.handle_indicies(indicies, reac_ind,
                      out_map, kernel_data, outmap_name=outmap_name)

    #get the proper kf indexing / array
    kf_arr, kf_str, map_result = lp_utils.get_loopy_arg('kf',
                    indicies=[reac_ind, 'j'],
                    dimensions=[Nr, test_size],
                    order=loopy_opts.order,
                    map_name=out_map)

    maps = [map_result[reac_ind]]

    #add to kernel data
    kernel_data.append(kf_arr)

    #preinstructions
    preinstructs = [k_gen.PLOG_PREINST_KEY, k_gen.TINV_PREINST_KEY]

    #extra loops
    pres_poly_ind = 'k'
    temp_poly_ind = 'm'
    extra_inames = [(pres_poly_ind, '0 <= {} < {}'.format(pres_poly_ind, maxP)),
                    (temp_poly_ind, '0 <= {} < {}'.format(temp_poly_ind, maxT)),
                    ('p', '2 <= p < {}'.format(poly_max))]

    instructions = Template("""
<>Pmin = ${Pmin_str}
<>Tmin = ${Tmin_str}
<>Pmax = ${Pmax_str}
<>Tmax = ${Tmax_str}
<>Tred = ${Tred_str}
<>Pred = ${Pred_str}
<>numP = ${numP_str} {id=plim}
<>numT = ${numT_str} {id=tlim}
${ppoly_0} = 1
${ppoly_1} = Pred
${tpoly_0} = 1
${tpoly_1} = Tred
#<> poly_end = max(numP, numT)
#compute polynomial terms
for p
    if p < numP
        ${ppoly_p} = 2 * Pred * ${ppoly_pm1} - ${ppoly_pm2} {id=ppoly, dep=plim}
    end
    if p < numT
        ${tpoly_p} = 2 * Tred * ${tpoly_pm1} - ${tpoly_pm2} {id=tpoly, dep=tlim}
    end
end
<> kf_temp = 0
for m
    <>temp = 0
    for k
        temp = temp + ${ppoly_k} * ${chebpar_km} {id=temp, dep=ppoly:tpoly}
    end
    kf_temp = kf_temp + ${tpoly_m} * temp {id=kf, dep=temp}
end

${kf_str} = exp10(kf_temp) {dep=kf}
""")

    instructions = Template(instructions.safe_substitute(
                    kf_str=kf_str,
                    Tred_str=str(Tred_eqn),
                    Pred_str=str(Pred_eqn),
                    Pmin_str=plim_str.safe_substitute(lim_ind=0),
                    Pmax_str=plim_str.safe_substitute(lim_ind=1),
                    Tmin_str=tlim_str.safe_substitute(lim_ind=0),
                    Tmax_str=tlim_str.safe_substitute(lim_ind=1),
                    ppoly_0=pres_poly_str.safe_substitute(pres_poly_ind=0),
                    ppoly_1=pres_poly_str.safe_substitute(pres_poly_ind=1),
                    ppoly_k=pres_poly_str.safe_substitute(pres_poly_ind=pres_poly_ind),
                    ppoly_p=pres_poly_str.safe_substitute(pres_poly_ind='p'),
                    ppoly_pm1=pres_poly_str.safe_substitute(pres_poly_ind='p - 1'),
                    ppoly_pm2=pres_poly_str.safe_substitute(pres_poly_ind='p - 2'),
                    tpoly_0=temp_poly_str.safe_substitute(temp_poly_ind=0),
                    tpoly_1=temp_poly_str.safe_substitute(temp_poly_ind=1),
                    tpoly_m=temp_poly_str.safe_substitute(temp_poly_ind=temp_poly_ind),
                    tpoly_p=temp_poly_str.safe_substitute(temp_poly_ind='p'),
                    tpoly_pm1=temp_poly_str.safe_substitute(temp_poly_ind='p - 1'),
                    tpoly_pm2=temp_poly_str.safe_substitute(temp_poly_ind='p - 2'),
                    chebpar_km=params_str.safe_substitute(temp_poly_ind=temp_poly_ind,
                        pres_poly_ind=pres_poly_ind),
                    numP_str=numP_str,
                    numT_str=numT_str,
                    kf_eval=str(cheb_form),
                    num_cheb=num_cheb)).safe_substitute(reac_ind=reac_ind)

    return k_gen.knl_info('rateconst_cheb', instructions=instructions, pre_instructions=preinstructs,
                     var_name=reac_ind, kernel_data=kernel_data, maps=maps,
                     extra_inames=extra_inames, indicies=indicies,
                     extra_subs={'reac_ind' : reac_ind})


def get_plog_arrhenius_rates(eqs, loopy_opts, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for p-log rate constants

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator

    """

    rate_eqn = get_rate_eqn(eqs)

    #find the plog equation
    plog_eqn = next(x for x in eqs['conp'] if str(x) == 'log({k_f}[i])')
    plog_form, plog_eqn = plog_eqn, eqs['conp'][plog_eqn][(reaction_type.plog,)]

    #now we do some surgery to obtain a form w/o 'logs' as we'll take them
    #explicitly in python
    logP = sp.Symbol('logP')
    logP1 = sp.Symbol('low[0]')
    logP2 = sp.Symbol('hi[0]')
    logk1 = sp.Symbol('logk1')
    logk2 = sp.Symbol('logk2')
    plog_eqn = sp_utils.sanitize(plog_eqn, subs={sp.log(sp.Symbol('k_1')) : logk1,
                                        sp.log(sp.Symbol('k_2')) : logk2,
                                        sp.log(sp.Symbol('P')) : logP,
                                        sp.log(sp.Symbol('P_1')) : logP1,
                                        sp.log(sp.Symbol('P_2')) : logP2})

    #and specialize the k1 / k2 equations
    A1 = sp.Symbol('low[1]')
    b1 = sp.Symbol('low[2]')
    Ta1 = sp.Symbol('low[3]')
    k1_eq = sp_utils.sanitize(rate_eqn, subs={sp.Symbol('A[i]') : A1,
                                         sp.Symbol('beta[i]') : b1,
                                         sp.Symbol('Ta[i]') : Ta1})
    A2 = sp.Symbol('hi[1]')
    b2 = sp.Symbol('hi[2]')
    Ta2 = sp.Symbol('hi[3]')
    k2_eq = sp_utils.sanitize(rate_eqn, subs={sp.Symbol('A[i]') : A2,
                                         sp.Symbol('beta[i]') : b2,
                                         sp.Symbol('Ta[i]') : Ta2})

    # total # of plog reactions
    num_plog = rate_info['plog']['num']
    # number of parameter sets per reaction
    num_params = rate_info['plog']['num_P']

    #create the loopy equivalents
    params = rate_info['plog']['params']
    #create a copy
    params_temp = [p[:] for p in params]
    #max # of parameters for sizing
    maxP = np.max(num_params)

    #for simplicity, we're going to use a padded form
    params = np.zeros((4, num_plog, maxP))
    for m in range(4):
        for i, numP in enumerate(num_params):
            for j in range(numP):
                params[m, i, j] = params_temp[i][j][m]

    #take the log of P and A
    hold = np.seterr(divide='ignore')
    params[0, :, :] = np.log(params[0, :, :])
    params[1, :, :] = np.log(params[1, :, :])
    params[np.where(np.isinf(params))] = 0
    np.seterr(**hold)

    #default params indexing order
    inds = ['${m}', '${reac_ind}', '${param_ind}']
    #make loopy version
    plog_params_lp = lp.TemporaryVariable('plog_params', shape=params.shape,
        initializer=params, scope=scopes.GLOBAL, read_only=True)
    param_str = Template('plog_params[' + ','.join(inds) + ']')

    #and finally the loopy version of num_params
    num_params_lp = lp.TemporaryVariable('plog_num_params', shape=lp.auto,
        initializer=num_params, read_only=True, scope=scopes.GLOBAL)

    #create temporary variables
    low_lp = lp.TemporaryVariable('low', shape=(4,), scope=scopes.PRIVATE, dtype=np.float64)
    hi_lp = lp.TemporaryVariable('hi', shape=(4,), scope=scopes.PRIVATE, dtype=np.float64)
    T_arr = lp.GlobalArg('T_arr', shape=(test_size,), dtype=np.float64)
    P_arr = lp.GlobalArg('P_arr', shape=(test_size,), dtype=np.float64)

    #start creating the k_gen.knl_info's
    #data
    kernel_data = [plog_params_lp, num_params_lp, T_arr,
                        P_arr, low_lp, hi_lp]

    if test_size == 'problem_size':
        kernel_data.append(lp.ValueArg(test_size, dtype=np.int32))

    #reac ind
    reac_ind = 'i'

    #extra loops
    extra_inames = [('k', '0 <= k < {}'.format(maxP - 1)), ('m', '0 <= m < 4')]

    #see if we need an output mask
    out_map = {}
    indicies = rate_info['plog']['map'].astype(dtype=np.int32)
    Nr = rate_info['Nr']

    indicies = k_gen.handle_indicies(indicies, reac_ind, out_map,
                    kernel_data, outmap_name='plog_inds')

    #get the proper kf indexing / array
    kf_arr, kf_str, map_result = lp_utils.get_loopy_arg('kf',
                    indicies=[reac_ind, 'j'],
                    dimensions=[Nr, test_size],
                    map_name=out_map,
                    order=loopy_opts.order)
    kernel_data.append(kf_arr)

    #handle map info
    maps = []
    if reac_ind in map_result:
        maps.append(map_result[reac_ind])

    #instructions
    instructions = Template(Template(
"""
        <>lower = logP <= ${pressure_lo} #check below range
        if lower
            <>lo_ind = 0 {id=ind00}
            <>hi_ind = 0 {id=ind01}
        end
        <>numP = plog_num_params[${reac_ind}] - 1
        <>upper = logP > ${pressure_hi} #check above range
        if upper
            lo_ind = numP {id=ind10}
            hi_ind = numP {id=ind11}
        end
        <>oor = lower or upper
        for k
            #check that
            #1. inside this reactions parameter's still
            #2. inside pressure range
            <> midcheck = (k <= numP) and (logP > ${pressure_mid_lo}) and (logP <= ${pressure_mid_hi})
            if midcheck
                lo_ind = k {id=ind20}
                hi_ind = k + 1 {id=ind21}
            end
        end
        for m
            low[m] = ${pressure_general_lo} {id=lo, dep=ind*}
            hi[m] = ${pressure_general_hi} {id=hi, dep=ind*}
        end
        <>logk1 = ${loweq} {id=a1, dep=lo}
        <>logk2 = ${hieq} {id=a2, dep=hi}
        <>kf_temp = logk1 {id=a_oor}
        if not oor
            kf_temp = ${plog_eqn} {id=a_found, dep=a1:a2}
        end
        ${kf_str} = exp(kf_temp) {id=kf, dep=aoor:a_found}
""").safe_substitute(loweq=k1_eq, hieq=k2_eq, plog_eqn=plog_eqn,
                    kf_str=kf_str,
                    pressure_lo=param_str.safe_substitute(m=0, param_ind=0),
                    pressure_hi=param_str.safe_substitute(m=0, param_ind='numP'),
                    pressure_mid_lo=param_str.safe_substitute(m=0, param_ind='k'),
                    pressure_mid_hi=param_str.safe_substitute(m=0, param_ind='k + 1'),
                    pressure_general_lo=param_str.safe_substitute(m='m', param_ind='lo_ind'),
                    pressure_general_hi=param_str.safe_substitute(m='m', param_ind='hi_ind')
                    )
    ).safe_substitute(reac_ind=reac_ind)

    #and return
    return [k_gen.knl_info(name='rateconst_plog', instructions=instructions,
        pre_instructions=[k_gen.TINV_PREINST_KEY, k_gen.TLOG_PREINST_KEY, k_gen.PLOG_PREINST_KEY],
        var_name=reac_ind, kernel_data=kernel_data,
        maps=maps, extra_inames=extra_inames, indicies=indicies,
        extra_subs={'reac_ind' : reac_ind})]


def get_reduced_pressure_kernel(eqs, loopy_opts, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for the reduced
    pressure evaluation kernel

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator

    """

    reac_ind = 'i'
    conp_eqs = eqs['conp'] #conp / conv irrelevant for rates

    #create the various necessary arrays

    kernel_data = []

    map_instructs = []
    inmaps = {}

    #falloff index mapping
    indicies = ['${reac_ind}', 'j']
    k_gen.handle_indicies(rate_info['fall']['map'], '${reac_ind}', inmaps, kernel_data,
                            outmap_name='fall_map',
                            force_zero=True)

    #simple arrhenius rates
    kf_arr, kf_str, map_inst = lp_utils.get_loopy_arg('kf',
                    indicies=indicies,
                    dimensions=[rate_info['Nr'], test_size],
                    order=loopy_opts.order,
                    map_name=inmaps)
    if '${reac_ind}' in map_inst:
        map_instructs.append(map_inst['${reac_ind}'])

    #simple arrhenius rates using falloff (alternate) parameters
    kf_fall_arr, kf_fall_str, _ = lp_utils.get_loopy_arg('kf_fall',
                        indicies=indicies,
                        dimensions=[rate_info['fall']['num'], test_size],
                        order=loopy_opts.order)

    #temperatures
    T_arr = lp.GlobalArg('T_arr', shape=(test_size,), dtype=np.float64)

    #create a Pr array
    Pr_arr, Pr_str, _ = lp_utils.get_loopy_arg('Pr',
                        indicies=indicies,
                        dimensions=[rate_info['fall']['num'], test_size],
                        order=loopy_opts.order)

    #third-body concentrations
    thd_indexing_str = '${reac_ind}'
    thd_index_map = {}

    #check if a mapping is needed
    thd_inds = rate_info['thd']['map']
    fall_inds = rate_info['fall']['map']
    if not np.array_equal(thd_inds, fall_inds):
        #add a mapping for falloff index -> third body conc index
        thd_map_name = 'thd_map'
        thd_map = np.where(np.in1d(thd_inds, fall_inds))[0].astype(np.int32)

        k_gen.handle_indicies(thd_map, thd_indexing_str, thd_index_map,
            kernel_data, outmap_name=thd_map_name, force_map=True)

    #create the array
    indicies = ['${reac_ind}', 'j']
    thd_conc_lp, thd_conc_str, map_inst = lp_utils.get_loopy_arg('thd_conc',
                                                 indicies=indicies,
                                                 dimensions=(rate_info['thd']['num'], test_size),
                                                 order=loopy_opts.order,
                                                 map_name=thd_index_map,
                                                 map_result='thd_i')
    if '${reac_ind}' in map_inst:
        map_instructs.append(map_inst['${reac_ind}'])

    #and finally the falloff types
    fall_type_lp, fall_type_str = __1Dcreator('fall_type', rate_info['fall']['ftype'],
        scope=scopes.GLOBAL)

    #append all arrays to the kernel data
    kernel_data.extend([T_arr, thd_conc_lp, kf_arr, kf_fall_arr, Pr_arr,
        fall_type_lp])

    #create Pri eqn
    Pri_sym = next(x for x in conp_eqs if str(x) == 'P_{r, i}')
    #make substituions to get a usable form
    pres_mod_sym = sp.Symbol(thd_conc_str)
    Pri_eqn = sp_utils.sanitize(conp_eqs[Pri_sym][(thd_body_type.mix,)],
                       subs={'[X]_i' : pres_mod_sym,
                            'k_{0, i}' : 'k0',
                            'k_{infty, i}' : 'kinf'}
                      )

    #create instruction set
    pr_instructions = Template("""
if fall_type[${reac_ind}]
    #chemically activated
    <>k0 = ${kf_str} {id=k0_c}
    <>kinf = ${kf_fall_str} {id=kinf_c}
else
    #fall-off
    kinf = ${kf_str} {id=kinf_f}
    k0 = ${kf_fall_str} {id=k0_f}
end
${Pr_str} = ${Pr_eq} {dep=k*}
""")

    #sub in strings
    pr_instructions = Template(pr_instructions.safe_substitute(
    kf_str=kf_str,
    kf_fall_str=kf_fall_str,
    Pr_str=Pr_str,
    Pr_eq=Pri_eqn)).safe_substitute(
        reac_ind=reac_ind)

    #and finally return the resulting info
    return [k_gen.knl_info('red_pres',
                     instructions=pr_instructions,
                     var_name=reac_ind,
                     kernel_data=kernel_data,
                     indicies=np.array(range(rate_info['fall']['num']), dtype=np.int32),
                     maps=map_instructs,
                     extra_subs={'reac_ind' : reac_ind})]

def get_troe_kernel(eqs, loopy_opts, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for the Troe
    falloff evaluation kernel

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator

    """

    #set of equations is irrelevant for non-derivatives
    conp_eqs = eqs['conp']

    #rate info and reac ind
    reac_ind = 'i'
    kernel_data = []

    if test_size == 'problem_size':
        kernel_data.append(lp.ValueArg(test_size, dtype=np.int32))

    #add the troe map
    #add sri map
    out_map = {}
    outmap_name = 'out_map'
    indicies = np.array(rate_info['fall']['troe']['map'], dtype=np.int32)
    indicies = k_gen.handle_indicies(indicies, '${reac_ind}', out_map, kernel_data,
                                force_zero=True)

    #create the Pr loopy array / string
    Pr_lp, Pr_str, map_result = lp_utils.get_loopy_arg('Pr',
                            indicies=['${reac_ind}', 'j'],
                            dimensions=[rate_info['fall']['num'], test_size],
                            order=loopy_opts.order,
                            map_name=out_map)

    #create Fi loopy array / string
    Fi_lp, Fi_str, map_result = lp_utils.get_loopy_arg('Fi',
                        indicies=['${reac_ind}', 'j'],
                        dimensions=[rate_info['fall']['num'], test_size],
                        order=loopy_opts.order,
                        map_name=out_map)

    #add to the kernel maps
    maps = []
    if '${reac_ind}' in map_result:
        maps.append(map_result['${reac_ind}'])

    #create the Fcent loopy array / str
    Fcent_lp, Fcent_str, _ = lp_utils.get_loopy_arg('Fcent',
                        indicies=['${reac_ind}', 'j'],
                        dimensions=[rate_info['fall']['troe']['num'], test_size],
                        order=loopy_opts.order)

    #create the Atroe loopy array / str
    Atroe_lp, Atroe_str, _ = lp_utils.get_loopy_arg('Atroe',
                        indicies=['${reac_ind}', 'j'],
                        dimensions=[rate_info['fall']['troe']['num'], test_size],
                        order=loopy_opts.order)

    #create the Fcent loopy array / str
    Btroe_lp, Btroe_str, _ = lp_utils.get_loopy_arg('Btroe',
                        indicies=['${reac_ind}', 'j'],
                        dimensions=[rate_info['fall']['troe']['num'], test_size],
                        order=loopy_opts.order)

    #create the temperature array
    T_lp = lp.GlobalArg('T_arr', shape=(test_size,), dtype=np.float64)

    #update the kernel_data
    kernel_data.extend([Pr_lp, T_lp, Fi_lp, Fcent_lp, Atroe_lp, Btroe_lp])

    #find the falloff form equations
    Fi_sym = next(x for x in conp_eqs if str(x) == 'F_{i}')
    keys = conp_eqs[Fi_sym]
    Fi = {}
    for key in keys:
        fall_form = next(x for x in key if isinstance(x, falloff_form))
        Fi[fall_form] = conp_eqs[Fi_sym][key]

    #get troe syms / eqs
    Fcent = next(x for x in conp_eqs if str(x) == 'F_{cent}')
    Atroe = next(x for x in conp_eqs if str(x) == 'A_{Troe}')
    Btroe = next(x for x in conp_eqs if str(x) == 'B_{Troe}')
    Fcent_eq, Atroe_eq, Btroe_eq = conp_eqs[Fcent], conp_eqs[Atroe], conp_eqs[Btroe]

    #get troe params and create arrays
    troe_a, troe_T3, troe_T1, troe_T2 = rate_info['fall']['troe']['a'], \
         rate_info['fall']['troe']['T3'], \
         rate_info['fall']['troe']['T1'], \
         rate_info['fall']['troe']['T2']
    troe_a_lp, troe_a_str = __1Dcreator('troe_a', troe_a, scope=scopes.GLOBAL)
    troe_T3_lp, troe_T3_str = __1Dcreator('troe_T3', troe_T3, scope=scopes.GLOBAL)
    troe_T1_lp, troe_T1_str = __1Dcreator('troe_T1', troe_T1, scope=scopes.GLOBAL)
    troe_T2_lp, troe_T2_str = __1Dcreator('troe_T2', troe_T2, scope=scopes.GLOBAL)
    #update the kernel_data
    kernel_data.extend([troe_a_lp, troe_T3_lp, troe_T1_lp, troe_T2_lp])
    #sub into eqs
    Fcent_eq = sp_utils.sanitize(Fcent_eq, subs={
            'a' : troe_a_str,
            'T^{*}' : troe_T1_str,
            'T^{***}' : troe_T3_str,
            'T^{**}' : troe_T2_str,
        })

    #now separate into optional / base parts
    Fcent_base_eq = sp.Add(*[x for x in sp.Add.make_args(Fcent_eq) if not sp.Symbol(troe_T2_str) in x.free_symbols])
    Fcent_opt_eq = Fcent_eq - Fcent_base_eq

    #develop the Atroe / Btroe eqs
    Atroe_eq = sp_utils.sanitize(Atroe_eq, subs=OrderedDict([
            ('F_{cent}', Fcent_str),
            ('P_{r, i}', Pr_str),
            (sp.log(sp.Symbol(Pr_str), 10), 'logPr'),
            (sp.log(sp.Symbol(Fcent_str), 10), sp.Symbol('logFcent'))
        ]))

    Btroe_eq = sp_utils.sanitize(Btroe_eq, subs=OrderedDict([
            ('F_{cent}', Fcent_str),
            ('P_{r, i}', Pr_str),
            (sp.log(sp.Symbol(Pr_str), 10), 'logPr'),
            (sp.log(sp.Symbol(Fcent_str), 10), sp.Symbol('logFcent'))
        ]))

    Fcent_temp_str = 'Fcent_temp'
    #finally, work on the Fi form
    Fi_eq = sp_utils.sanitize(Fi[falloff_form.troe], subs=OrderedDict([
            ('F_{cent}', Fcent_temp_str),
            ('A_{Troe}', Atroe_str),
            ('B_{Troe}', Btroe_str)
        ]))

    #separate into Fcent and power
    Fi_base_eq = next(x for x in Fi_eq.args if str(x) == Fcent_temp_str)
    Fi_pow_eq = next(x for x in Fi_eq.args if str(x) != Fcent_temp_str)
    Fi_pow_eq = sp_utils.sanitize(Fi_pow_eq, subs=OrderedDict([
            (sp.Pow(sp.Symbol(Atroe_str), 2), sp.Symbol('Atroe_squared')),
            (sp.Pow(sp.Symbol(Btroe_str), 2), sp.Symbol('Btroe_squared'))
        ]))

    #make the instructions
    troe_instructions = Template(
    """
    <>T = T_arr[j]
    <>${Fcent_temp} = ${Fcent_base_eq} {id=Fcent_decl} #this must be a temporary to avoid a race on future assignments
    if ${troe_T2_str} != 0
        ${Fcent_temp} = ${Fcent_temp} + ${Fcent_opt_eq} {id=Fcent_decl2, dep=Fcent_decl}
    end
    ${Fcent_str} = ${Fcent_temp} {dep=Fcent_decl*}
    <>logFcent = log10(${Fcent_temp}) {dep=Fcent_decl*}
    <>logPr = log10(${Pr_str}) {id=Pr_decl}
    <>Atroe_temp = ${Atroe_eq} {dep=Fcent_decl*:Pr_decl}
    <>Btroe_temp = ${Btroe_eq} {dep=Fcent_decl*:Pr_decl}
    ${Atroe_str} = Atroe_temp #this must be a temporary to avoid a race on future assignments
    ${Btroe_str} = Btroe_temp #this must be a temporary to avoid a race on future assignments
    <>Atroe_squared = Atroe_temp * Atroe_temp
    <>Btroe_squared = Btroe_temp * Btroe_temp
    ${Fi_str} = ${Fi_base_eq}**(${Fi_pow_eq}) {dep=Fcent_decl*:Pr_decl}
    """
    ).safe_substitute(Fcent_temp=Fcent_temp_str,
                     Fcent_str=Fcent_str,
                     Fcent_base_eq=Fcent_base_eq,
                     Fcent_opt_eq=Fcent_opt_eq,
                     troe_T2_str=troe_T2_str,
                     Pr_str=Pr_str,
                     Atroe_eq=Atroe_eq,
                     Btroe_eq=Btroe_eq,
                     Atroe_str=Atroe_str,
                     Btroe_str=Btroe_str,
                     Fi_str=Fi_str,
                     Fi_base_eq=Fi_base_eq,
                     Fi_pow_eq=Fi_pow_eq)
    troe_instructions = Template(troe_instructions).safe_substitute(reac_ind=reac_ind)

    return [k_gen.knl_info('fall_troe',
                     instructions=troe_instructions,
                     var_name=reac_ind,
                     kernel_data=kernel_data,
                     indicies=indicies,
                     maps=maps,
                     extra_subs={'reac_ind' : reac_ind})]


def get_sri_kernel(eqs, loopy_opts, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for the SRI
    falloff evaluation kernel

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator

    """

    #set of equations is irrelevant for non-derivatives
    conp_eqs = eqs['conp']
    reac_ind = 'i'

    #Create the temperature array
    T_arr = lp.GlobalArg('T_arr', shape=(test_size,), dtype=np.float64)
    kernel_data = [T_arr]

    if test_size == 'problem_size':
        kernel_data.append(lp.ValueArg(test_size, dtype=np.int32))

    #figure out if we need to do any mapping of the input variable
    out_map = {}
    outmap_name = 'out_map'
    indicies = np.array(rate_info['fall']['sri']['map'], dtype=np.int32)
    indicies = k_gen.handle_indicies(indicies, '${reac_ind}', out_map, kernel_data,
                                force_zero=True)

    #start creating SRI kernel
    Fi_sym = next(x for x in conp_eqs if str(x) == 'F_{i}')
    keys = conp_eqs[Fi_sym]
    Fi = {}
    for key in keys:
        fall_form = next(x for x in key if isinstance(x, falloff_form))
        Fi[fall_form] = conp_eqs[Fi_sym][key]

    #find Pr symbol
    Pri_sym = next(x for x in conp_eqs if str(x) == 'P_{r, i}')

    #create Fi array / mapping
    Fi_lp, Fi_str, map_result = lp_utils.get_loopy_arg('Fi',
                        indicies=['${reac_ind}', 'j'],
                        dimensions=[rate_info['fall']['num'], test_size],
                        order=loopy_opts.order,
                        map_name=out_map)
    #and Pri array / mapping
    Pr_lp, Pr_str, map_result = lp_utils.get_loopy_arg('Pr',
                        indicies=['${reac_ind}', 'j'],
                        dimensions=[rate_info['fall']['num'], test_size],
                        order=loopy_opts.order,
                        map_name=out_map)
    maps = []
    if '${reac_ind}' in map_result:
        maps.append(map_result['${reac_ind}'])
    kernel_data.extend([Fi_lp, Pr_lp])

    #get SRI symbols
    X_sri_sym = next(x for x in Fi[falloff_form.sri].free_symbols if str(x) == 'X')
    a_sri_sym = next(x for x in Fi[falloff_form.sri].free_symbols if str(x) == 'a')
    b_sri_sym = next(x for x in Fi[falloff_form.sri].free_symbols if str(x) == 'b')
    c_sri_sym = next(x for x in Fi[falloff_form.sri].free_symbols if str(x) == 'c')
    d_sri_sym = next(x for x in Fi[falloff_form.sri].free_symbols if str(x) == 'd')
    e_sri_sym = next(x for x in Fi[falloff_form.sri].free_symbols if str(x) == 'e')
    #get SRI params and create arrays
    sri_a, sri_b, sri_c, sri_d, sri_e = rate_info['fall']['sri']['a'], \
        rate_info['fall']['sri']['b'], \
        rate_info['fall']['sri']['c'], \
        rate_info['fall']['sri']['d'], \
        rate_info['fall']['sri']['e']
    X_sri_lp, X_sri_str, _ = lp_utils.get_loopy_arg('X',
                        indicies=['${reac_ind}', 'j'],
                        dimensions=[rate_info['fall']['sri']['num'], test_size],
                        order=loopy_opts.order)
    sri_a_lp, sri_a_str = __1Dcreator('sri_a', sri_a, scope=scopes.GLOBAL)
    sri_b_lp, sri_b_str = __1Dcreator('sri_b', sri_b, scope=scopes.GLOBAL)
    sri_c_lp, sri_c_str = __1Dcreator('sri_c', sri_c, scope=scopes.GLOBAL)
    sri_d_lp, sri_d_str = __1Dcreator('sri_d', sri_d, scope=scopes.GLOBAL)
    sri_e_lp, sri_e_str = __1Dcreator('sri_e', sri_e, scope=scopes.GLOBAL)
    kernel_data.extend([X_sri_lp, sri_a_lp, sri_b_lp, sri_c_lp, sri_d_lp, sri_e_lp])

    #create SRI eqs
    X_sri_eq = conp_eqs[X_sri_sym].subs(sp.Pow(sp.log(Pri_sym, 10), 2), 'logPr * logPr')
    Fi_sri_eq = Fi[falloff_form.sri]
    Fi_sri_eq = sp_utils.sanitize(Fi_sri_eq,
        subs={
            'a' : sri_a_str,
            'b' : sri_b_str,
            'c' : sri_c_str,
            'd' : sri_d_str,
            'e' : sri_e_str,
            'X' : 'X_temp'
        })
    #do some surgery on the Fi_sri_eq to get the optional parts
    Fi_sri_base = next(x for x in sp.Mul.make_args(Fi_sri_eq)
                       if any(str(y) == sri_a_str for y in x.free_symbols))
    Fi_sri_opt = Fi_sri_eq / Fi_sri_base
    Fi_sri_d_opt = next(x for x in sp.Mul.make_args(Fi_sri_opt)
                       if any(str(y) == sri_d_str for y in x.free_symbols))
    Fi_sri_e_opt = next(x for x in sp.Mul.make_args(Fi_sri_opt)
                       if any(str(y) == sri_e_str for y in x.free_symbols))

    #create instruction set
    sri_instructions = Template(Template("""
<>T = T_arr[j]
<>logPr = log10(${pr_str})
<>X_temp = ${Xeq} {id=X_decl} #this must be a temporary to avoid a race on Fi_temp assignment
<>Fi_temp = ${Fi_sri} {id=Fi_decl, dep=X_decl}
if ${d_str} != 1.0
    Fi_temp = Fi_temp * ${d_eval} {id=Fi_decl1, dep=Fi_decl}
end
if ${e_str} != 0.0
    Fi_temp = Fi_temp * ${e_eval} {id=Fi_decl2, dep=Fi_decl}
end
${Fi_str} = Fi_temp {dep=Fi_decl*}
${X_str} = X_temp
""").safe_substitute(logPr_eval='logPr',
                     pr_str=Pr_str,
                     X_str=X_sri_str,
                     Xeq=X_sri_eq,
                     Fi_sri=Fi_sri_base,
                     d_str=sri_d_str,
                     d_eval=Fi_sri_d_opt,
                     e_str=sri_e_str,
                     e_eval=Fi_sri_e_opt,
                     Fi_str=Fi_str)).safe_substitute(
                            reac_ind=reac_ind)

    return [k_gen.knl_info('fall_sri',
                     instructions=sri_instructions,
                     var_name=reac_ind,
                     kernel_data=kernel_data,
                     indicies=indicies,
                     maps=maps,
                     extra_subs={'reac_ind' : reac_ind})]



def get_simple_arrhenius_rates(eqs, loopy_opts, rate_info, test_size=None,
        falloff=False):
    """Generates instructions, kernel arguements, and data for specialized forms
    of simple (non-pressure dependent) rate constants

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly
    falloff : bool
        If true, generate rate kernel for the falloff rates, i.e. either
        k0 or kinf depending on whether the reaction is falloff or chemically activated

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator

    """

    #find options, sizes, etc.
    if falloff:
        tag = 'fall'
        name_mod = '_fall'
        Nr = rate_info['fall']['num']
    else:
        tag = 'simple'
        name_mod = ''
        Nr = rate_info['Nr']

    #first assign the reac types, parameters
    full = loopy_opts.rate_spec == lp_utils.RateSpecialization.full
    hybrid = loopy_opts.rate_spec == lp_utils.RateSpecialization.hybrid
    fixed = loopy_opts.rate_spec == lp_utils.RateSpecialization.fixed
    separated_kernels = loopy_opts.rate_spec_kernels
    if fixed and separated_kernels:
        separated_kernels = False
        logging.warn('Cannot use separated kernels with a fixed RateSpecialization, '
            'disabling...')

    #define loopy arrays
    A_name = 'A{}'.format(name_mod)
    A_lp = lp.TemporaryVariable(A_name, shape=lp.auto,
        initializer=rate_info[tag]['A'],
        read_only=True, scope=scopes.GLOBAL)
    b_name = 'beta{}'.format(name_mod)
    b_lp = lp.TemporaryVariable(b_name, shape=lp.auto,
        initializer=rate_info[tag]['b'],
        read_only=True, scope=scopes.GLOBAL)
    Ta_name = 'Ta{}'.format(name_mod)
    Ta_lp = lp.TemporaryVariable(Ta_name, shape=lp.auto,
        initializer=rate_info[tag]['Ta'],
        read_only=True, scope=scopes.GLOBAL)
    T_arr = lp.GlobalArg('T_arr', shape=(test_size,), dtype=np.float64)
    simple_arrhenius_data = [A_lp, b_lp, Ta_lp, T_arr]

    if test_size == 'problem_size':
        simple_arrhenius_data += [lp.ValueArg(test_size, dtype=np.int32)]

    #if we need the rtype array, add it
    if not separated_kernels and not fixed:
        rtype_lp = lp.TemporaryVariable('rtype{}'.format(name_mod), shape=lp.auto,
            initializer=rate_info[tag]['type'],
            read_only=True, scope=scopes.PRIVATE)
        simple_arrhenius_data.append(rtype_lp)

    #the reaction index variable
    reac_ind = 'i'

    maps = []
    #figure out if we need to do any mapping of the input variable
    inmap_name = 'in_map'
    if separated_kernels:
        reac_ind = 'i_dummy'
        maps.append(lp_utils.generate_map_instruction(
                                            newname='i',
                                            map_arr=inmap_name,
                                            oldname=reac_ind))
    #get rate equations
    rate_eqn_pre = get_rate_eqn(eqs)
    rate_eqn_pre = sp_utils.sanitize(rate_eqn_pre,
                        symlist= {
                            'A[i]' : A_name + '[i]',
                            'Ta[i]' : Ta_name + '[i]',
                            'beta[i]' : b_name + '[i]',
                            })

    #put rateconst info args in dict for unpacking convenience
    extra_args = {'kernel_data' : simple_arrhenius_data,
                  'var_name' : reac_ind,
                  'maps' : maps,
                  'extra_subs' : {'reac_ind' : reac_ind}}

    default_preinstructs = [k_gen.TINV_PREINST_KEY, k_gen.TLOG_PREINST_KEY]

    #generic kf assigment str
    kf_assign = Template("${kf_str} = ${rate}")
    expkf_assign = Template("${kf_str} = exp(${rate})")

    #various specializations of the rate form
    specializations = {}
    i_a_only = k_gen.knl_info(name='a_only{}'.format(name_mod),
        instructions=kf_assign.safe_substitute(rate='${A_name}[i]'),
        **extra_args)
    i_beta_int = k_gen.knl_info(name='beta_int{}'.format(name_mod),
        pre_instructions=[k_gen.TINV_PREINST_KEY],
        instructions="""
        <> T_val = T_arr[j] {id=a1}
        <> negval = ${b_name}[i] < 0
        if negval
            T_val = T_inv {id=a2, dep=a1}
        end
        ${kf_str} = ${A_name}[i] * T_val {id=a3, dep=a2}
        ${beta_iter}
        """,
        **extra_args)
    i_beta_exp = k_gen.knl_info('rateconst_beta_exp{}'.format(name_mod),
        instructions=expkf_assign.safe_substitute(rate=str(rate_eqn_pre.subs(Ta_name, 0))),
        pre_instructions=default_preinstructs,
        **extra_args)
    i_ta_exp = k_gen.knl_info('rateconst_ta_exp{}'.format(name_mod),
        instructions=expkf_assign.safe_substitute(rate=str(rate_eqn_pre.subs(b_name, 0))),
        pre_instructions=default_preinstructs,
        **extra_args)
    i_full = k_gen.knl_info('rateconst_full{}'.format(name_mod),
        instructions=expkf_assign.safe_substitute(rate=str(rate_eqn_pre)),
        pre_instructions=default_preinstructs,
        **extra_args)

    #set up the simple arrhenius rate specializations
    if fixed:
        specializations[0] = i_full
    else:
        specializations[0] = i_a_only
        specializations[1] = i_beta_int
        if full:
            specializations[2] = i_beta_exp
            specializations[3] = i_ta_exp
            specializations[4] = i_full
        else:
            specializations[2] = i_full

    if not separated_kernels and not fixed:
        #need to enclose each branch in it's own if statement
        if len(specializations) > 1:
            instruction_list = []
            inds = set()
            for i in specializations:
                instruction_list.append('<>test_{0} = rtype[i] == {0} {{id=d{0}}}'.format(i))
                instruction_list.append('if test_{0}'.format(i))
                instruction_list.extend(['\t' + x for x in specializations[i].instructions.split('\n')])
                instruction_list.append('end')
        #and combine them
        specializations = {-1 : k_gen.knl_info('singlekernel',
            instructions='\n'.join(instruction_list),
            pre_instructions=[k_gen.TINV_PREINST_KEY, k_gen.TLOG_PREINST_KEY],
            **extra_args)}

    spec_copy = specializations.copy()
    #and do some finalizations for the specializations
    for rtype, info in spec_copy.items():
        #first, get indicies
        if rtype < 0:
            #select all for joined kernel
            info.indicies = np.arange(0, rate_info[tag]['type'].size, dtype=np.int32)
        else:
            #otherwise choose just our rtype
            info.indicies = np.where(rate_info[tag]['type'] == rtype)[0].astype(dtype=np.int32)

        if not info.indicies.size:
            #kernel doesn't act on anything, remove it
            del specializations[rtype]
            continue

        #check maxb / iteration
        beta_iter = ''
        if (separated_kernels and (info.name == i_beta_int.name)) or \
            (not separated_kernels and not fixed):
            #find max b exponent
            maxb_test = rate_info[tag]['b'][
                    np.where(rate_info[tag]['type'] == rtype)]
            if maxb_test.size:
                maxb = int(np.max(np.abs(maxb_test)))
                #if we need to iterate
                if maxb > 1:
                    #add an extra iname, and the resulting iteraton loop
                    info.extra_inames.append(('k', '1 <= maxb < {}'.format(maxb)))
                    beta_iter = """
                <> btest = abs(${b_name}[i])
                for k
                    <>inbounds = k < btest
                    if inbounds
                        ${kf_str} = ${kf_str} * T_val {dep=a2}
                    end
                end"""

        #check if we have an input map
        if info.var_name != 'i':
            if info.indicies[0] + info.indicies.size - 1 == info.indicies[-1]:
                #it'll end up in offset form, hence we don't need a map
                info.var_name = 'i'
                info.maps = []
            else:
                #need to add the input map to kernel data
                inmap_lp = lp.TemporaryVariable(inmap_name,
                    shape=lp.auto,
                    initializer=info.indicies,
                    read_only=True, scope=scopes.PRIVATE)
                info.kernel_data.append(inmap_lp)

        #check if we need an output map
        out_map = {}
        outmap_name = 'out_map'
        alt_inds = None
        if not falloff:
            alt_inds = rate_info[tag]['map'][info.indicies]
        info.indicies = k_gen.handle_indicies(info.indicies, info.var_name,
                      out_map, info.kernel_data, outmap_name=outmap_name,
                      alternate_indicies=alt_inds)

        #get the proper kf indexing / array
        kf_arr, kf_str, map_result = lp_utils.get_loopy_arg('kf' + name_mod,
                        indicies=[info.var_name, 'j'],
                        dimensions=[Nr, test_size],
                        map_name=out_map,
                        order=loopy_opts.order)
        info.kernel_data.append(kf_arr)

        #handle map info
        if info.var_name in map_result:
            info.maps.append(map_result[info.var_name])

        #substitute in whatever beta_iter / kf_str we found
        info.instructions = Template(
                        Template(info.instructions).safe_substitute(
                            beta_iter=beta_iter)
                    ).safe_substitute(kf_str=kf_str,
                                      A_name=A_name,
                                      b_name=b_name,
                                      Ta_name=Ta_name)

    return list(specializations.values())


def write_specrates_kernel(eqs, reacs, specs,
                            loopy_opts, conp=True,
                            test_size=None, auto_diff=False):
    """Helper function that generates kernels for
       evaluation of reaction rates / rate constants / and species rates

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    reacs : list of :class:`ReacInfo`
        List of species in the mechanism.
    specs : list of :class:`SpecInfo`
        List of species in the mechanism.
    loopy_opts : :class:`loopy_options` object
        A object containing all the loopy options to execute
    conp : bool
        If true, generate equations using constant pressure assumption
        If false, use constant volume equations
    test_size : int
        If not None, this kernel is being used for testing.
    auto_diff : bool
        If ``True``, generate files for Adept autodifferention library.

    Returns
    -------
    kernel_gen : :class:`wrapping_kernel_generator`
        The generator responsible for creating the resulting code

    """

    rate_info = assign_rates(reacs, specs, loopy_opts.rate_spec)

    if test_size is None:
        test_size = 'problem_size'

    kernels = []

    func_manglers = []

    def __add_knl(knls, klist=None):
        if klist is None:
            klist = kernels
        try:
            klist.extend(knls)
        except:
            klist.append(knls)

    #get the simple arrhenius k_gen.knl_info's
    __add_knl(get_simple_arrhenius_rates(eqs, loopy_opts,
        rate_info, test_size=test_size))

    #check for plog
    if rate_info['plog']['num']:
        #generate the plog kernel
        __add_knl(get_plog_arrhenius_rates(eqs, loopy_opts,
            rate_info, test_size=test_size))

    #check for chebyshev
    if rate_info['cheb']['num']:
        __add_knl(get_cheb_arrhenius_rates(eqs, loopy_opts,
            rate_info, test_size=test_size))


    #check for falloff
    if rate_info['fall']['num']:
        __add_knl(get_simple_arrhenius_rates(eqs, loopy_opts,
            rate_info, test_size=test_size, falloff=True))
        if rate_info['fall']['troe']['num']:
            __add_knl(get_troe_kernel(eqs, loopy_opts,
                rate_info, test_size=test_size))
        if rate_info['fall']['sri']['num']:
            __add_knl(get_sri_kernel(eqs, loopy_opts,
                rate_info, test_size=test_size))

    #check for reverse rates
    if rate_info['rev']['num']:
        __add_knl(get_rev_rates(eqs, loopy_opts,
                rate_info, test_size=test_size))

    #check for pressure modification terms
    if rate_info['thd']['num']:
        #add the initial third body conc eval kernel
        __add_knl(get_thd_body_concs(eqs, loopy_opts,
            rate_info, test_size))
        #and the Pr evals
        __add_knl(get_rxn_pres_mod(eqs, loopy_opts,
            rate_info, test_size))

    #add ROP
    __add_knl(get_rop(eqs, loopy_opts,
        rate_info, test_size))
    #add ROP net
    __add_knl(get_rop_net(eqs, loopy_opts,
        rate_info, test_size))
    #add spec rates
    __add_knl(get_spec_rates(eqs, loopy_opts,
        rate_info, test_size))

    external_kernels = []
    if conp:
        #get h / cp evals
        __add_knl(polyfit_kernel_gen('h', eqs['conp'], specs, loopy_opts,
            test_size))
        __add_knl(polyfit_kernel_gen('cp', eqs['conp'], specs, loopy_opts,
            test_size))
        external_kernels.extend(kernels[-2:])
    else:
        #and u / cv
        __add_knl(polyfit_kernel_gen('u', eqs['conv'], specs, loopy_opts,
            test_size))
        __add_knl(polyfit_kernel_gen('cv', eqs['conv'], specs, loopy_opts,
            test_size))
        external_kernels.extend(kernels[-2:])
    #and temperature rates
    __add_knl(get_temperature_rate(eqs, loopy_opts,
        rate_info, test_size=test_size, conp=conp))

    return k_gen.wrapping_kernel_generator(
            loopy_opts=loopy_opts,
            name='species_rates_kernel',
            kernels=kernels,
            external_kernels=external_kernels,
            input_arrays=['T_arr', 'P_arr', 'conc', 'wdot'],
            output_arrays=['wdot'],
            init_arrays={'wdot' : 0,
                         'Fi' : 1},
            auto_diff=auto_diff,
            test_size=test_size)


def get_rate_eqn(eqs, index='i'):
    """Helper routine that returns the Arrenhius rate constant in exponential
    form.

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    index : str
        The index to generate the equations for, 'i' by default
    Returns
    -------
    rate_eqn_pre : `sympy.Expr`
        The rate constant before taking the exponential (sympy does odd things upon doing so)
        This is used for various simplifications

    """

    conp_eqs = eqs['conp']

    #define some dummy symbols for loopy writing
    E_sym = sp.Symbol('Ta[{ind}]'.format(ind=index))
    A_sym = sp.Symbol('A[{ind}]'.format(ind=index))
    T_sym = sp.Symbol('T')
    b_sym = sp.Symbol('beta[{ind}]'.format(ind=index))
    symlist = {'Ta[i]' : E_sym,
               'A[i]' : A_sym,
               'T' : T_sym,
               'beta[i]' : b_sym}
    Tinv_sym = sp.Symbol('T_inv')
    logA_sym = sp.Symbol('A[{ind}]'.format(ind=index))
    logT_sym = sp.Symbol('logT')

    #the rate constant is indep. of conp/conv, so just use conp for simplicity
    kf_eqs = [x for x in conp_eqs if str(x) == '{k_f}[i]']

    #do some surgery on the equations
    kf_eqs = {key: (x, conp_eqs[x][key]) for x in kf_eqs for key in conp_eqs[x]}

    #first load the arrenhius rate equation
    rate_eqn = next(kf_eqs[x] for x in kf_eqs if reaction_type.elementary in x)[1]
    rate_eqn = sp_utils.sanitize(rate_eqn,
                        symlist=symlist,
                        subs={sp.Symbol('{E_{a}}[i]') / (sp.Symbol('R_u') * T_sym) :
                            E_sym * Tinv_sym})

    #finally, alter to exponential form:
    rate_eqn_pre = sp.log(A_sym) + sp.log(T_sym) * b_sym - E_sym * Tinv_sym
    rate_eqn_pre = rate_eqn_pre.subs([(sp.log(A_sym), logA_sym),
                                      (sp.log(T_sym), logT_sym)])

    return rate_eqn_pre


def polyfit_kernel_gen(nicename, eqs, specs,
                            loopy_opts, test_size=None):
    """Helper function that generates kernels for
       evaluation of various thermodynamic species properties

    Parameters
    ----------
    nicename : str
        The variable name to use in generated code
    eqs : dict of `sympy.Symbol`
        Dictionary defining conditional equations for the variables (keys)
    specs : list of `SpecInfo`
        List of species in the mechanism.
    lang : {'c', 'cuda', 'opencl'}
        Programming language.
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    test_size : int
        If not None, this kernel is being used for testing.

    Returns
    -------
    knl : :class:`loopy.LoopKernel`
        The generated loopy kernel for code generation / testing

    """

    if test_size is None:
        test_size = 'problem_size'

    if loopy_opts.width is not None and loopy_opts.depth is not None:
        raise Exception('Cannot specify both SIMD/SIMT width and depth')

    #mapping of nicename -> varname
    var_maps = {'cp' : '{C_p}[k]',
        'h' : 'H[k]', 'cv' : '{C_v}[k]',
        'u' : 'U[k]', 'b' : 'B[k]'}
    varname = var_maps[nicename]

    var = next(v for v in eqs.keys() if str(v) == varname)
    eq = eqs[var]
    poly_dim = specs[0].hi.shape[0]
    Ns = len(specs)

    #pick out a values and T_mid
    a_lo = np.zeros((Ns, poly_dim), dtype=np.float64)
    a_hi = np.zeros((Ns, poly_dim), dtype=np.float64)
    T_mid = np.zeros((Ns,), dtype=np.float64)
    for ind, spec in enumerate(specs):
        a_lo[ind, :] = spec.lo[:]
        a_hi[ind, :] = spec.hi[:]
        T_mid[ind] = spec.Trange[1]

    #get correctly ordered arrays / strings
    indicies = ['k', '${param_val}']
    a_lo_lp = lp.TemporaryVariable('a_lo', shape=a_lo.shape, initializer=a_lo,
        scope=scopes.GLOBAL, read_only=True)
    a_hi_lp = lp.TemporaryVariable('a_hi', shape=a_hi.shape, initializer=a_hi,
        scope=scopes.GLOBAL, read_only=True)
    a_lo_str = Template('a_lo[' + ','.join(indicies) + ']')
    a_hi_str = Template('a_hi[' + ','.join(indicies) + ']')
    T_mid_lp = lp.TemporaryVariable('T_mid', shape=T_mid.shape, initializer=T_mid, read_only=True,
                                scope=scopes.GLOBAL)

    k = sp.Idx('k')
    lo_eq_str = str(eq.subs([(sp.IndexedBase('a')[k, i],
        a_lo_str.safe_substitute(param_val=i)) for i in range(poly_dim)]))
    hi_eq_str = str(eq.subs([(sp.IndexedBase('a')[k, i],
        a_hi_str.safe_substitute(param_val=i)) for i in range(poly_dim)]))

    target = lp_utils.get_target(loopy_opts.lang)

    #create the input arrays arrays
    T_lp = lp.GlobalArg('T_arr', shape=(test_size,), dtype=np.float64)
    out_lp, out_str, _ = lp_utils.get_loopy_arg(nicename,
                    indicies=['k', 'j'],
                    dimensions=(Ns, test_size),
                    order=loopy_opts.order)

    knl_data = [a_lo_lp, a_hi_lp, T_mid_lp, T_lp, out_lp]

    if test_size == 'problem_size':
        knl_data = [lp.ValueArg('problem_size', dtype=np.int32)] + knl_data

    return k_gen.knl_info(instructions=Template("""
        for j
            <> T = T_arr[j]
            for k
                if T < T_mid[k]
                    ${out_str} = ${lo_eq}
                else
                    ${out_str} = ${hi_eq}
                end
            end
        end
        """).safe_substitute(out_str=out_str, lo_eq=lo_eq_str, hi_eq=hi_eq_str),
        kernel_data=knl_data,
        name='eval_{}'.format(nicename),
        parameters={'R_u' : chem.RU},
        var_name='k',
        indicies=k_gen.handle_indicies(np.arange(Ns, dtype=np.int32), 'k', None, []))


def write_chem_utils(specs, eqs, loopy_opts,
                        test_size=None, auto_diff=False):
    """Write subroutine to evaluate species thermodynamic properties.

    Notes
    -----
    Thermodynamic properties include:  enthalpy, energy, specific heat
    (constant pressure and volume).

    Parameters
    ----------
    specs : list of `SpecInfo`
        List of species in the mechanism.
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    test_size : int
        If not None, this kernel is being used for testing.
    auto_diff : bool
        If ``True``, generate files for Adept autodifferention library.

    Returns
    -------
    global_defines : list of :class:`loopy.TemporaryVariable`
        The global variables for this kernel that need definition in the memory manager

    """

    if test_size is None:
        test_size = 'problem_size'

    file_prefix = ''
    if auto_diff:
        file_prefix = 'ad_'

    target = lp_utils.get_target(loopy_opts.lang)

    #generate the kernels
    conp_eqs = eqs['conp']
    conv_eqs = eqs['conv']

    nicenames = ['cp', 'h', 'cv', 'u', 'b']
    kernels = []
    headers = []
    code = []
    for nicename in nicenames:
        eq = conp_eqs if nicename in ['h', 'cp'] else conv_eqs
        kernels.append(polyfit_kernel_gen(nicename,
            eq, specs, loopy_opts, test_size))

    return k_gen.wrapping_kernel_generator(
        loopy_opts=loopy_opts,
        name='chem_utils',
        kernels=kernels,
        input_arrays=['T_arr'],
        output_arrays=nicenames,
        auto_diff=auto_diff,
        test_size=test_size
        )


if __name__ == "__main__":
    from . import create_jacobian
    args = utils.get_parser()
    create_jacobian(lang=args.lang,
                mech_name=args.input,
                therm_name=args.thermo,
                optimize_cache=args.cache_optimizer,
                initial_state=args.initial_conditions,
                num_blocks=args.num_blocks,
                num_threads=args.num_threads,
                no_shared=args.no_shared,
                L1_preferred=args.L1_preferred,
                multi_thread=args.multi_thread,
                force_optimize=args.force_optimize,
                build_path=args.build_path,
                skip_jac=True,
                last_spec=args.last_species,
                auto_diff=args.auto_diff
                )
