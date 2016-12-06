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
import re
import logging
import itertools
from string import Template
import textwrap

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
from . import CUDAParams
from . import cache_optimizer as cache
from . import mech_auxiliary as aux
from . import shared_memory as shared
from . import file_writers as filew
from ..sympy import sympy_utils as sp_utils
from . reaction_types import reaction_type


def rxn_rate_const(A, b, E):
    r"""Returns line with reaction rate calculation (after = sign).

    Parameters
    ----------
    A : float
        Arrhenius pre-exponential coefficient
    b : float
        Arrhenius temperature exponent
    E : float
        Arrhenius activation energy

    Returns
    -------
    line : str
        String with expression for reaction rate.

    Notes
    -----
    Form of the reaction rate constant (from, e.g., Lu and Law [1]_):

    .. math::
        :nowrap:

        k_f = \begin{cases}
        A & \text{if } \beta = 0 \text{ and } T_a = 0 \\
        \exp \left( \log A + \beta \log T \right) &
        \text{if } \beta \neq 0 \text{ and } \text{if } T_a = 0 \\
        \exp \left( \log A + \beta \log T - T_a / T \right)	&
        \text{if } \beta \neq 0 \text{ and } T_a \neq 0 \\
        \exp \left( \log A - T_a / T \right)
        & \text{if } \beta = 0 \text{ and } T_a \neq 0 \\
        A \prod^b T	& \text{if } T_a = 0 \text{ and }
        b \in \mathbb{Z} \text{ (integers) }
        \end{cases}

    References
    ----------

    .. [1] TF Lu and CK Law, "Toward accommodating realistic fuel chemistry
       in large-scale computations," Progress in Energy and Combustion
       Science, vol. 35, pp. 192-215, 2009. doi:10.1016/j.pecs.2008.10.002

    """

    line = ''

    if A > 0:
        logA = math.log(A)

        if not E:
            # E = 0
            if not b:
                # b = 0
                line += str(A)
            else:
                # b != 0
                if isinstance(b, int):
                    line += str(A)
                    for i in range(b):
                        line += ' * T'
                else:
                    line += 'exp({:.16e}'.format(logA)
                    if b > 0:
                        line += ' + ' + str(b)
                    else:
                        line += ' - ' + str(abs(b))
                    line += ' * logT)'
        else:
            # E != 0
            if not b:
                # b = 0
                line += 'exp({:.16e}'.format(logA) + ' - ({:.16e} / T))'.format(E)
            else:
                # b!= 0
                line += 'exp({:.16e}'.format(logA)
                if b > 0:
                    line += ' + ' + str(b)
                else:
                    line += ' - ' + str(abs(b))
                line += ' * logT - ({:.16e} / T))'.format(E)
    elif A < 0:
        #a < 0, can't take the log of it
        #the reaction, should also be a duplicate to make any sort of sense
        if not E:
            #E = 0
            if not b:
                #b = 0
                line += str(A)
            else:
                #b != 0
                if utils.is_integer(b):
                    line += str(A)
                    for i in range(int(b)):
                        line += ' * T'
                else:
                    line += '{:.16e} * exp('.format(A)
                    if b > 0:
                        line += str(b)
                    else:
                        line += '-' + str(abs(b))
                    line += ' * logT)'
        else:
            #E != 0
            if not b:
                # b = 0
                line += '{:.16e} * exp(-({:.16e} / T))'.format(A, E)
            else:
                # b!= 0
                line += '{:.16e} * exp('.format(A)
                if b > 0:
                    line += str(b)
                else:
                    line += '-' + str(abs(b))
                line += ' * logT - ({:.16e} / T))'.format(E)

    else:
      raise NotImplementedError

    return line


def get_cheb_rate(lang, rxn, write_defns=True):
    """
    Given a reaction, and a temperature and pressure, this routine
    will generate code to evaluate the Chebyshev rate efficiently.

    Note
    ----
    Assumes existence of variables dot_prod* of sized at least rxn.cheb_n_temp
    Pred and Tred, T and pres, and kf, cheb_temp_0 and cheb_temp_1.

    Parameters
    ----------
    lang : str
        Programming language
    rxn : `ReacInfo`
        Reaction with Chebyshev pressure dependence.
    write_defns : bool, optional
        If ``True`` (default), write calculation of ``Tred`` and ``Pred``.

    Returns
    -------
    line : list of `str`
        Line with evaluation of Chebyshev reaction rate.

    """

    line_list = []
    tlim_inv_sum = 1.0 / rxn.cheb_tlim[0] + 1.0 / rxn.cheb_tlim[1]
    tlim_inv_sub = 1.0 / rxn.cheb_tlim[1] - 1.0 / rxn.cheb_tlim[0]
    if write_defns:
        line_list.append(
                'Tred = ((2.0 / T) - ' +
                '{:.8e}) / {:.8e}'.format(tlim_inv_sum, tlim_inv_sub)
                )

    plim_log_sum = (math.log10(rxn.cheb_plim[0]) +
                    math.log10(rxn.cheb_plim[1])
                    )
    plim_log_sub = (math.log10(rxn.cheb_plim[1]) -
                    math.log10(rxn.cheb_plim[0])
                    )
    if write_defns:
        line_list.append(
                'Pred = (2.0 * log10(pres) - ' +
                '{:.8e}) / {:.8e}'.format(plim_log_sum, plim_log_sub)
                )

    line_list.append('cheb_temp_0 = 1')
    line_list.append('cheb_temp_1 = Pred')
    #start pressure dot product
    for i in range(rxn.cheb_n_temp):
        line_list.append(utils.get_array(lang, 'dot_prod', i) +
          '= {:.8e} + Pred * {:.8e}'.format(rxn.cheb_par[i, 0],
            rxn.cheb_par[i, 1]))

    #finish pressure dot product
    update_one = True
    for j in range(2, rxn.cheb_n_pres):
        if update_one:
            new = 1
            old = 0
        else:
            new = 0
            old = 1
        line = 'cheb_temp_{}'.format(old)
        line += ' = 2 * Pred * cheb_temp_{}'.format(new)
        line += ' - cheb_temp_{}'.format(old)
        line_list.append(line)
        for i in range(rxn.cheb_n_temp):
            line_list.append(utils.get_array(lang, 'dot_prod', i)  +
              ' += {:.8e} * cheb_temp_{}'.format(
                rxn.cheb_par[i, j], old))

        update_one = not update_one

    line_list.append('cheb_temp_0 = 1')
    line_list.append('cheb_temp_1 = Tred')
    #finally, do the temperature portion
    line_list.append('kf = ' + utils.get_array(lang, 'dot_prod', 0) +
                     ' + Tred * ' + utils.get_array(lang, 'dot_prod', 1))

    update_one = True
    for i in range(2, rxn.cheb_n_temp):
        if update_one:
            new = 1
            old = 0
        else:
            new = 0
            old = 1
        line = 'cheb_temp_{}'.format(old)
        line += ' = 2 * Tred * cheb_temp_{}'.format(new)
        line += ' - cheb_temp_{}'.format(old)
        line_list.append(line)
        line_list.append('kf += ' + utils.get_array(lang, 'dot_prod', i) +
                         ' * ' + 'cheb_temp_{}'.format(old))

        update_one = not update_one

    line_list.append('kf = ' + utils.exp_10_fun[lang] + 'kf)')
    line_list = [utils.line_start + line + utils.line_end[lang] for
                  line in line_list]

    return ''.join(line_list)

def assign_rates(reacs, rate_spec):
    """
    From a given set of reactions, determine the rate types for evaluation

    Parameters
    ----------
    reacs : list of `ReacInfo`
        The reactions in the mechanism
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
        Keys are 'simple', 'plog', 'cheb'
        Values are further dictionaries including addtional rate info, including
            number, offset, maps, and (in the case of simple) additional rate
            info (i.e. A, Ta, b, rate type and maximum b-value)

    """

    #determine specialization
    full = rate_spec == lp_utils.RateSpecialization.full
    hybrid = rate_spec == lp_utils.RateSpecialization.hybrid
    fixed = rate_spec == lp_utils.RateSpecialization.fixed

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

    simple_rate_type = np.zeros((num_simple,), dtype=np.int32)
    #reaction parameters
    A = np.zeros((num_simple,), dtype=np.float64)
    b = np.zeros((num_simple,), dtype=np.float64)
    Ta = np.zeros((num_simple,), dtype=np.float64)

    i = 0
    for i, reac in enumerate(simple_rate):
        #assign rate params
        A[i] = np.log(reac.A)
        b[i] = reac.b
        Ta[i] = reac.E
        if fixed:
            simple_rate_type[i] = 0
            continue
        #assign rate types
        if reac.b == 0 and reac.E == 0:
            A[i] = reac.A
            simple_rate_type[i] = 0
        elif reac.b == int(reac.b) and reac.b and reac.E == 0:
            A[i] = reac.A
            simple_rate_type[i] = 1
        elif reac.E == 0 and reac.b != 0:
            simple_rate_type[i] = 2
        elif reac.b == 0 and reac.E != 0:
            simple_rate_type[i] = 3
        else:
            simple_rate_type[i] = 4
        if not full:
            simple_rate_type[i] = simple_rate_type[i] if simple_rate_type[i] <= 1 else 2

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
    num_cheb_p = []
    num_cheb_t = []
    cheb_coeff = []
    for cheb in cheb_reacs:
        cheb_coeff.append(cheb.cheb_par)

    return {'simple' : {'A' : A, 'b' : b, 'Ta' : Ta, 'type' : simple_rate_type,
                'num' : num_simple, 'map' : simple_map},
            'plog' : {'map' : plog_map, 'num' : num_plog,
            'num_P' : num_pressures, 'params' : plog_params},
            'cheb' : {'map' : cheb_map, 'num' : num_cheb},
            'Nr' : len(reacs)}

class rateconst_info(object):
    def __init__(self, name, instructions, pre_instructions=[],
        reac_ind='i', kernel_data=None,
        maps={}, extra_inames=[], indicies=[]):
        self.name = name
        self.instructions = instructions
        self.pre_instructions = pre_instructions[:]
        self.reac_ind = reac_ind
        self.kernel_data = kernel_data[:]
        self.maps = maps.copy()
        self.extra_inames = extra_inames[:]
        self.indicies = indicies[:]

__TINV_PREINST_KEY = 'Tinv'
__TLOG_PREINST_KEY = 'logT'
__PLOG_PREINST_KEY = 'logP'

def get_plog_arrhenius_rates(eqs, loopy_opt, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for p-log rate constants

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opt : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`rateconst_info`
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
    # incidies of plog reactions
    plog_inds = np.array(rate_info['plog']['map'])

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
    params[0, :, :] = np.log(params[0, :, :])
    params[1, :, :] = np.log(params[1, :, :])
    params[np.where(np.isinf(params))] = 0

    #default indexing order
    inds = ['${m}', '${reac_ind}', '${param_ind}']

    if loopy_opt.order == 'C':
        #in c-continguous mode, we want parameters from
        #subsequent reactions to be adjacent
        params = params.swapaxes(1, 2).copy()

        #swap indicies
        temp = inds[2]
        inds[2] = inds[1]
        inds[1] = temp

    #make loopy version
    plog_params_lp = lp_utils.get_loopy_arg('plog_params', indicies=inds,
                                                dimensions=params.shape,
                                                order=loopy_opt.order, #order taken care of already
                                                initializer=params,
                                                scope=scopes.GLOBAL)
    #get the param indexing
    param_str = Template(plog_params_lp['arg_str'])

    #and the actual parameter array
    plog_params_lp = plog_params_lp['arg']

    #and finally the loopy version of num_params
    num_params_lp = lp.TemporaryVariable('plog_num_params', shape=lp.auto,
        initializer=num_params, read_only=True, scope=scopes.GLOBAL)

    #create temporary variables
    low_lp = lp.TemporaryVariable('low', shape=(4,), scope=scopes.PRIVATE, dtype=np.float64)
    hi_lp = lp.TemporaryVariable('hi', shape=(4,), scope=scopes.PRIVATE, dtype=np.float64)
    T_arr = lp.GlobalArg('T_arr', shape=(test_size,), dtype=np.float64)
    P_arr = lp.GlobalArg('P_arr', shape=(test_size,), dtype=np.float64)
    plog_inds_lp = lp.TemporaryVariable('plog_inds', scope=scopes.GLOBAL, shape=lp.auto,
                                       initializer=plog_inds, read_only=True)

    #start creating the rateconst_info's
    #data
    kernel_data = [plog_params_lp, num_params_lp, T_arr,
                        P_arr, low_lp, hi_lp, plog_inds_lp]

    #reac ind
    reac_ind = 'i'

    #extra loops
    extra_inames = [('k', '0 <= k < {}'.format(maxP - 1)), ('m', '0 <= m < 4')]

    out_map = {}
    outmap_name = 'out_map'
    #see if we need an output mask
    indicies = rate_info['plog']['map'].astype(dtype=np.int32)
    Nr = rate_info['Nr']

    if indicies[0] + indicies.size == indicies[-1]:
        #if the indicies are contiguous, we can get away with an
        #offset
        indicies = (indicies[0], indicies[-1])
    else:
        #need an output map
        out_map[reac_ind] = outmap_name
        #add to kernel data
        outmap_lp = lp.TemporaryVariable(outmap_name,
            shape=lp.auto,
            initializer=rate_info['plog']['map'],
            read_only=True, scope=scopes.PRIVATE)
        kernel_data.append(outmap_lp)

    #get the proper kf indexing / array
    result = lp_utils.get_loopy_arg('kf',
                    [reac_ind, 'j'],
                    [Nr, test_size],
                    order=loopy_opt.order,
                    map_name=out_map)

    #add to kernel data
    kf_arr = result['arg']
    kernel_data.append(kf_arr)

    #get correct str for instructions
    kf_str = result['arg_str']

    #handle map info
    maps = []
    if reac_ind in out_map:
        maps.append(result['map_instructs'][reac_ind])

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
    return [rateconst_info(name='plog', instructions=instructions,
        pre_instructions=[__TINV_PREINST_KEY, __TLOG_PREINST_KEY, __PLOG_PREINST_KEY],
        reac_ind=reac_ind, kernel_data=kernel_data,
        maps=maps, extra_inames=extra_inames, indicies=indicies)]


def get_simple_arrhenius_rates(eqs, loopy_opt, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for specialized forms
    of simple (non-pressure dependent) rate constants

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opt : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`rateconst_info`
        The generated infos for feeding into the kernel generator

    """

    #find options, sizes, etc.
    Nr = rate_info['Nr']

    #first assign the reac types, parameters
    full = loopy_opt.rate_spec == lp_utils.RateSpecialization.full
    hybrid = loopy_opt.rate_spec == lp_utils.RateSpecialization.hybrid
    fixed = loopy_opt.rate_spec == lp_utils.RateSpecialization.fixed
    separated_kernels = loopy_opt.rate_spec_kernels
    if fixed and separated_kernels:
        separated_kernels = False
        logging.info('Cannot use separated kernels with a fixed RateSpecialization, '
            'disabling...')

    #define loopy arrays
    A_lp = lp.TemporaryVariable('A', shape=lp.auto,
        initializer=rate_info['simple']['A'],
        read_only=True, scope=scopes.GLOBAL)
    b_lp = lp.TemporaryVariable('beta', shape=lp.auto,
        initializer=rate_info['simple']['b'],
        read_only=True, scope=scopes.GLOBAL)
    Ta_lp = lp.TemporaryVariable('Ta', shape=lp.auto,
        initializer=rate_info['simple']['Ta'],
        read_only=True, scope=scopes.GLOBAL)
    T_arr = lp.GlobalArg('T_arr', shape=(test_size,), dtype=np.float64)
    simple_arrhenius_data = [A_lp, b_lp, Ta_lp, T_arr]

    #if we need the rtype array, add it
    if not separated_kernels and not fixed:
        rtype_lp = lp.TemporaryVariable('rtype', shape=lp.auto,
            initializer=rate_info['simple']['type'],
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

    #put rateconst info args in dict for unpacking convenience
    extra_args = {'kernel_data' : simple_arrhenius_data,
                  'reac_ind' : reac_ind,
                  'maps' : maps}

    default_preinstructs = [__TINV_PREINST_KEY, __TLOG_PREINST_KEY]

    #generic kf assigment str
    kf_assign = Template("${kf_str} = ${rate}")
    expkf_assign = Template("${kf_str} = exp(${rate})")

    #various specializations of the rate form
    specializations = {}
    i_a_only = rateconst_info(name='a_only',
        instructions=kf_assign.safe_substitute(rate='A[i]'),
        **extra_args)
    i_beta_int = rateconst_info(name='beta_int',
        pre_instructions=[__TINV_PREINST_KEY],
        instructions="""
        <> T_val = T_arr[j] {id=a1}
        <> negval = beta[i] < 0
        if negval
            T_val = T_inv {id=a2, dep=a1}
        end
        ${kf_str} = A[i] * T_val {id=a3, dep=a2}
        ${beta_iter}
        """,
        **extra_args)
    i_beta_exp = rateconst_info('beta_exp',
        instructions=expkf_assign.safe_substitute(rate=str(rate_eqn_pre.subs('Ta[i]', 0))),
        pre_instructions=default_preinstructs,
        **extra_args)
    i_ta_exp = rateconst_info('ta_exp',
        instructions=expkf_assign.safe_substitute(rate=str(rate_eqn_pre.subs('beta[i]', 0))),
        pre_instructions=default_preinstructs,
        **extra_args)
    i_full = rateconst_info('full',
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
        specializations = {-1 : rateconst_info('singlekernel',
            instructions='\n'.join(instruction_list),
            pre_instructions=[__TINV_PREINST_KEY, __TLOG_PREINST_KEY],
            **extra_args)}

    spec_copy = specializations.copy()
    #and do some finalizations for the specializations
    for rtype, info in spec_copy.items():
        #first, get indicies
        if rtype < 0:
            #select all for joined kernel
            info.indicies = np.arange(0, rate_info['simple']['type'].size, dtype=np.int32)
        else:
            #otherwise choose just our rtype
            info.indicies = np.where(rate_info['simple']['type'] == rtype)[0].astype(dtype=np.int32)

        if not info.indicies.size:
            #kernel doesn't act on anything, remove it
            del specializations[rtype]
            continue

        #check maxb / iteration
        beta_iter = ''
        if (separated_kernels and (info.name == i_beta_int.name)) or \
            (not separated_kernels and not fixed):
            #find max b exponent
            maxb_test = rate_info['simple']['b'][
                    np.where(rate_info['simple']['type'] == rtype)]
            if maxb_test.size:
                maxb = int(np.max(np.abs(maxb_test)))
                #if we need to iterate
                if maxb > 1:
                    #add an extra iname, and the resulting iteraton loop
                    info.extra_inames.append(('k', '1 <= maxb < {}'.format(maxb)))
                    beta_iter = """
                <> btest = abs(beta[i])
                for k
                    <>inbounds = k < btest
                    if inbounds
                        ${kf_str} = ${kf_str} * T_val {dep=a2}
                    end
                end"""

        #check if we have an input map
        if info.reac_ind != 'i':
            #need to add the input map to kernel data
            inmap_lp = lp.TemporaryVariable(inmap_name,
                shape=lp.auto,
                initializer=info.indicies,
                read_only=True, scope=scopes.PRIVATE)
            info.kernel_data.append(inmap_lp)

        #check if we need an output map
        out_map = {}
        outmap_name = 'out_map'
        if info.indicies[0] + info.indicies.size == info.indicies[-1]:
            #if the indicies are contiguous, we can get away with an
            #offset
            info.indicies = (info.indicies[0], info.indicies[-1])
        else:
            #need an output map
            out_map[reac_ind] = outmap_name
            #add to kernel data
            outmap_lp = lp.TemporaryVariable(outmap_name,
                shape=lp.auto,
                initializer=rate_info['simple']['map'][
                                info.indicies],
                read_only=True, scope=scopes.PRIVATE)
            info.kernel_data.append(outmap_lp)

        #get the proper kf indexing / array
        result = lp_utils.get_loopy_arg('kf',
                        [reac_ind, 'j'],
                        [Nr, test_size],
                        order=loopy_opt.order,
                        map_name=out_map)
        #add to kernel data
        kf_arr = result['arg']
        info.kernel_data.append(kf_arr)

        #get correct str for instructions
        kf_str = result['arg_str']

        #handle map info
        if reac_ind in out_map:
            info.maps.append(result['map_instructs'][reac_ind])

        #substitute in whatever beta_iter / kf_str we found
        info.instructions = Template(
                        Template(info.instructions).safe_substitute(
                            beta_iter=beta_iter)
                    ).safe_substitute(kf_str=kf_str)

    return specializations.values()

def rate_const_simple_kernel_gen(eqs, reacs,
                            loopy_opt, test_size=None):
    """Helper function that generates kernels for
       evaluation of simple arrhenius rate forms

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    reacs : list of `ReacInfo`
        List of species in the mechanism.
    loopy_opt : `loopy_options` object
        A object containing all the loopy options to execute
    test_size : int
        If not none, this kernel is being used for testing. Hence we need to size the arrays accordingly

    Returns
    -------
    knl_list : list of `loopy.kernel`
        The generated loopy kernel(s) for code generation / testing

    """

    #find language / target
    lang = loopy_opt.lang
    target = lp_utils.get_target(lang)

    if test_size is None:
        test_size = 'n'

    #determine rate evaluation types, indicies etc.
    rate_info = assign_rates(reacs, loopy_opt.rate_spec)

    #various precomputes
    pre_inst = {__TINV_PREINST_KEY : '<> T_inv = 1 / T_arr[j]',
                __TLOG_PREINST_KEY : '<> logT = log(T_arr[j])',
                __PLOG_PREINST_KEY : '<> logP = log(P_arr[j])'}

    #and the skeleton kernel
    skeleton = """
    for j
        ${pre}
        for ${reac_ind}
            ${main}
        end
    end
    """

    kernels = {}

    #get the simple arrhenius rateconst_info's
    kernels['simple'] = get_simple_arrhenius_rates(eqs, loopy_opt, rate_info, test_size=test_size)

    #check for plog
    if any(r.plog for r in reacs):
        #generate the plog kernel
        kernels['plog'] = get_plog_arrhenius_rates(eqs, loopy_opt, rate_info, test_size=test_size)

    #convience method to create kernel
    def make_kernel(info):
        #convert instructions into a list for convienence
        if isinstance(info.instructions, str):
            instructions = textwrap.dedent(info.instructions)
            instructions = [x for x in instructions.split('\n') if x.strip()]

        #load inames
        inames = [info.reac_ind, 'j']

        #add map instructions
        instructions = info.maps + instructions

        #look for extra inames, ranges
        iname_range = []

        #find the start index for 'i'
        if isinstance(info.indicies, tuple):
            i_start = info.indicies[0]
            i_end = info.indicies[1]
        else:
            i_start = 0
            i_end = info.indicies.size

        #add to ranges
        iname_range.append('{}<={}<{}'.format(i_start, info.reac_ind, i_end))
        iname_range.append('{}<=j<{}'.format(0, test_size))

        for iname, irange in info.extra_inames:
            inames.append(iname)
            iname_range.append(irange)

        #construct the kernel args
        pre_instructions = [pre_inst[k] for k in info.pre_instructions]
        def subs_preprocess(key, value):
            #find the instance of ${key} in kernel_str
            whitespace = None
            for i, line in enumerate(skeleton.split('\n')):
                if key in line:
                    #get whitespace
                    whitespace = re.match(r'\s*', line).group()
                    break
            result = [line if i == 0 else whitespace + line for i, line in
                        enumerate(textwrap.dedent(value).splitlines())]
            return '\n'.join(result)


        kernel_str = Template(skeleton).safe_substitute(
            reac_ind=info.reac_ind,
            pre=subs_preprocess('${pre}', '\n'.join(pre_instructions)),
            main=subs_preprocess('${main}', '\n'.join(instructions)))

        #make the kernel
        knl = lp.make_kernel('{[' + ','.join(inames) + ']:' +
            ' and '.join(iname_range) + '}',
            kernel_str,
            kernel_data=info.kernel_data,
            name='rateconst_' + info.name,
            target=target
        )
        #prioritize and return
        knl = lp.prioritize_loops(knl, inames)
        return knl

    knl_list = {}
    #now create the kernels!

    for eval_type in kernels:
        knl_list[eval_type] = []
        for info in kernels[eval_type]:
            knl_list[eval_type].append((info.reac_ind, make_kernel(info)))

        #stub for special handling of various evaluation types here

        #apply other optimizations
        for i, (reac_ind, knl) in enumerate(knl_list[eval_type]):
            #now apply specified optimizations
            if loopy_opt.depth is not None:
                #and assign the l0 axis to 'i'
                knl = lp.split_iname(knl, reac_ind, loopy_opt.depth, inner_tag='l.0')
                #assign g0 to 'j'
                knl = lp.tag_inames(knl, [('j', 'g.0')])
            elif loopy_opt.width is not None:
                #make the kernel a block of specifed width
                knl = lp.split_iname(knl, 'j', loopy_opt.width, inner_tag='l.0')
                #assign g0 to 'i'
                knl = lp.tag_inames(knl, [('j_outer', 'g.0')])

            #now do unr / ilp
            i_tag = reac_ind + '_outer' if loopy_opt.depth is not None else reac_ind
            if loopy_opt.unr is not None:
                knl = lp.split_iname(knl, i_tag, loopy_opt.unr, inner_tag='unr')
            elif loopy_opt.ilp:
                knl = lp.tag_inames(knl, [(i_tag, 'ilp')])

            knl_list[eval_type][i] = knl

    return knl_list

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


def write_rate_constants(path, specs, reacs, eqs, opts, auto_diff=False):
    """Write subroutine(s) to evaluate reaction rates constants.

    Parameters
    ----------
    path : str
        Path to build directory for file.
    specs : list of `SpecInfo`
        List of species in the mechanism.
    reacs : list of `ReacInfo`
        List of reactions in the mechanism.
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    opts : `loopy_options` object
        A object containing all the loopy options to execute
    auto_diff : bool
        If ``True``, generate files for Adept autodifferention library.
    Returns
    -------
    None

    """

    file_prefix = ''
    double_type = 'double'
    pres_ref = ''
    if auto_diff:
        file_prefix = 'ad_'
        double_type = 'adouble'
        pres_ref = '&'

    target = utils.get_target(opts.lang)

    #generate the rate kernels
    knl_list = rate_const_simple_kernel_gen(eqs, reacs, opts)



def write_rxn_pressure_mod(path, lang, specs, reacs,
                           fwd_rxn_mapping, smm=None, auto_diff=False
                           ):
    """Write subroutine to for reaction pressure dependence modifications.

    Parameters
    ----------
    path : str
        Path to build directory for file.
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Language type.
    specs : list of `SpecInfo`
        List of species in mechanism.
    reacs : list of `ReacInfo`
        List of reactions in mechanism.
    fwd_rxn_mapping : List of int
        The order of the reaction in the original mechanism
    smm : `shared_memory_manager`, optional
        If not ```None```, `shared_memory_manager` to use for CUDA optimizations
    auto_diff : bool, optional
        If ```True```, generate files for Adept autodifferention library.

    Returns
    -------
    None

    """
    double_type = 'double'
    file_prefix = ''
    pres_ref = ''
    if auto_diff:
        double_type = 'adouble'
        file_prefix = 'ad_'
        pres_ref = '&'
    filename = file_prefix + 'rxn_rates_pres_mod' + utils.file_ext[lang]
    file = open(os.path.join(path, filename), 'w')

    # headers
    if lang in ['c', 'cuda']:
        file.write('#include <math.h>\n'
                   '#include "header{1}"\n'
                   '#include "{0}rates{1}"\n'.format(file_prefix, utils.header_ext[lang])
                   )

        if auto_diff:
            file.write('#include "adept.h"\n'
                       'using adept::adouble;\n'
                       '#define fmax(a, b) (a.value() > b ? a : adouble(b))\n')
        file.write('\n')

    # list of reactions with third-body or pressure-dependence
    pdep_reacs = []
    thd_flag = False
    pdep_flag = False
    troe_flag = False
    sri_flag = False
    for i_rxn, reac in enumerate(reacs):
        if reac.thd_body:
            # add reaction index to list
            thd_flag = True
            pdep_reacs.append(i_rxn)
        elif reac.pdep:
            # add reaction index to list
            pdep_flag = True
            pdep_reacs.append(i_rxn)

            if reac.troe and not troe_flag: troe_flag = True
            if reac.sri and not sri_flag: sri_flag = True

    line = ''
    if lang == 'cuda': line = '__device__ '

    if lang in ['c', 'cuda']:
        line += ('void get_rxn_pres_mod (const {0} T, const {0}{1} pres, '
                 'const {0} * {2} C, {0} * {2} pres_mod) {{\n'.format(
                double_type, pres_ref, utils.restrict[lang])
                 )
    elif lang == 'fortran':
        line += 'subroutine get_rxn_pres_mod ( T, pres, C, pres_mod )\n\n'

        # fortran needs type declarations
        line += ('  implicit none\n'
                 '  double precision, intent(in) :: T, pres, '
                 'C({})\n'.format(len(specs)) +
                 '  double precision, intent(out) :: '
                 'pres_mod({})\n'.format(len(pdep_reacs)) +
                 '  \n'
                 '  double precision :: logT, m\n')
    elif lang == 'matlab':
        line += ('function pres_mod = get_rxn_pres_mod (T, pres, C)\n\n'
                 '  pres_mod = zeros({},1);\n'.format(len(pdep_reacs))
                 )
    file.write(line)

    get_array = utils.get_array
    if lang == 'cuda' and smm is not None:
        smm.reset()
        get_array = smm.get_array
        smm.write_init(file, indent=2)

    # declarations for third-body variables
    if thd_flag or pdep_flag:
        if lang == 'c':
            file.write('  // third body variable declaration\n'
                       '  {0} thd;\n\n'.format(double_type)
                       )
        elif lang == 'cuda':
            file.write('  // third body variable declaration\n'
                       '  register double thd;\n'
                       '\n'
                       )
        elif lang == 'fortran':
            file.write('  ! third body variable declaration\n'
                       '  double precision :: thd\n'
                       )

    # declarations for pressure-dependence variables
    if pdep_flag:
        if lang == 'c':
            file.write('  // pressure dependence variable declarations\n'
                       '  {0} k0;\n'
                       '  {0} kinf;\n'
                       '  {0} Pr;\n'
                       '\n'.format(double_type)
                       )
            if troe_flag:
                # troe variables
                file.write('  // troe variable declarations\n'
                           '  {0} logFcent;\n'
                           '  {0} A;\n'
                           '  {0} B;\n'
                           '\n'.format(double_type)
                           )
            if sri_flag:
                # sri variables
                file.write('  // sri variable declarations\n')
                file.write('  {0} X;\n'
                           '\n'.format(double_type)
                           )
        elif lang == 'cuda':
            file.write('  // pressure dependence variable declarations\n')
            file.write('  register double k0;\n'
                       '  register double kinf;\n'
                       '  register double Pr;\n'
                       '\n'
                       )
            if troe_flag:
                # troe variables
                file.write('  // troe variable declarations\n'
                           '  register double logFcent;\n'
                           '  register double A;\n'
                           '  register double B;\n'
                           '\n'
                           )
            if sri_flag:
                # sri variables
                file.write('  // sri variable declarations\n')
                file.write('  register double X;\n'
                           '\n')
        elif lang == 'fortran':
            file.write('  ! pressure dependence variable declarations\n'
                       '  double precision :: k0, kinf, Pr\n'
                       '\n'
                       )
            if troe_flag:
                # troe variables
                file.write('  ! troe variable declarations\n'
                           '  double precision :: logFcent, A, B\n'
                           '\n'
                           )
            if sri_flag:
                # sri variables
                file.write('  ! sri variable declarations\n')
                file.write('  double precision :: X\n'
                           '\n')

    if lang == 'c':
        file.write('  {} logT = log(T);\n'
                   '  {} m = pres / ({:.8e} * T);\n'.format(double_type,
                                                    double_type, chem.RU)
                   )
    elif lang == 'cuda':
        file.write('  register double logT = log(T);\n'
                   '  register double m = pres / ('
                   '{:.8e} * T);\n'.format(chem.RU)
                   )
    elif lang == 'fortran':
        file.write('  logT = log(T)\n'
                   '  m = pres / ({:.8e} * T)\n'.format(chem.RU)
                   )
    elif lang == 'matlab':
        file.write('  logT = log(T);\n'
                   '  m = pres / ({:.8e} * T);\n'.format(chem.RU)
                   )

    file.write('\n')

    pind = 0
    # loop through third-body and pressure-dependent reactions
    for rind in range(len(reacs)):
        reac = reacs[rind]  # index in reaction list

        if not (reac.pdep or reac.thd_body):
            continue

        printind = fwd_rxn_mapping[rind]
        # print reaction index
        if lang in ['c', 'cuda']:
            line = '  // reaction ' + str(printind)
        elif lang == 'fortran':
            line = '  ! reaction ' + str(printind + 1)
        elif lang == 'matlab':
            line = '  % reaction ' + str(printind + 1)
        line += utils.line_end[lang]
        file.write(line)

        if reac.thd_body_eff:
            if lang == 'cuda' and smm is not None:
                the_vars = []
                indexes = sorted([sp[0] for sp in reac.thd_body_eff
                                 if sp[1] != 1.0]
                                 )
                the_vars = [shared.variable('C', index) for index in indexes]
                # estimate usages as the number of consecutive reactions
                usages = []
                for sp_i in indexes:
                    temp = i_rxn + 1
                    count = 0
                    while temp < len(reacs):
                        rxn = reacs[temp]
                        if sp_i in set([x[0] for x in rxn.thd_body_eff
                                       if sp[1] != 1.0]
                                       ):
                            count += 1
                        else:
                            break
                        temp += 1
                    usages.append(count)
                smm.load_into_shared(file, the_vars, usages)

        # third-body reaction
        if reac.thd_body:

            line = '  ' + get_array(lang, 'pres_mod', pind) + ' = m'

            for sp in reac.thd_body_eff:
                if sp[1] == 1.0:
                    continue
                elif sp[1] > 1.0:
                    line += ' + {}'.format(sp[1] - 1.0)
                elif sp[1] < 1.0:
                    line += ' - {}'.format(1.0 - sp[1])
                line += ' * ' + get_array(lang, 'C', sp[0])

            line += utils.line_end[lang]
            file.write(line)

        # pressure dependence
        if reac.pdep:
            if reac.pdep_sp is None:
                line = '  thd = m'
                for sp in reac.thd_body_eff:
                    if sp[1] == 1.0:
                        continue
                    elif sp[1] > 1.0:
                        line += ' + {}'.format(sp[1] - 1.0)
                    elif sp[1] < 1.0:
                        line += ' - {}'.format(1.0 - sp[1])
                    line += ' * ' + get_array(lang, 'C', sp[0])
                file.write(line + utils.line_end[lang])

            # low-pressure limit rate
            line = '  k0 = '
            if reac.low:
                line += rxn_rate_const(reac.low[0],
                                       reac.low[1],
                                       reac.low[2]
                                       )
            else:
                line += rxn_rate_const(reac.A, reac.b, reac.E)

            line += utils.line_end[lang]
            file.write(line)

            # high-pressure limit rate
            line = '  kinf = '
            if reac.high:
                line += rxn_rate_const(reac.high[0],
                                       reac.high[1],
                                       reac.high[2]
                                       )
            else:
                line += rxn_rate_const(reac.A, reac.b, reac.E)

            line += utils.line_end[lang]
            file.write(line)

            # reduced pressure
            if reac.pdep_sp is not None:
                line = ('  Pr = k0 * ' +
                        get_array(lang, 'C', reac.pdep_sp) + ' / kinf'
                        )
            else:
                line = '  Pr = k0 * thd / kinf'
            line += utils.line_end[lang]
            file.write(line)

            simple = False
            if reac.troe:
                # Troe form
                line = ('  logFcent = log10( fmax('
                        '{:.8e} * '.format(1.0 - reac.troe_par[0])
                        )
                if reac.troe_par[1] > 0.0:
                    line += 'exp(-T / {:.8e})'.format(reac.troe_par[1])
                else:
                    line += 'exp(T / {:.8e})'.format(abs(reac.troe_par[1]))

                line += ' + {:.8e} * '.format(reac.troe_par[0])
                if reac.troe_par[2] > 0.0:
                    line += 'exp(-T / {:.8e})'.format(reac.troe_par[2])
                else:
                    line += 'exp(T / {:.8e})'.format(abs(reac.troe_par[2]))

                if len(reac.troe_par) == 4 and reac.troe_par[3] != 0.0:
                    line += ' + '
                    if reac.troe_par[3] > 0.0:
                        line += 'exp(-{:.8e} / T)'.format(reac.troe_par[3])
                    else:
                        line += 'exp({:.8e} / T)'.format(abs(reac.troe_par[3]))
                line += ', 1.0e-300))' + utils.line_end[lang]
                file.write(line)

                line = ('  A = log10(fmax(Pr, 1.0e-300)) - '
                        '0.67 * logFcent - 0.4' +
                        utils.line_end[lang]
                        )
                file.write(line)

                line = ('  B = 0.806 - 1.1762 * logFcent - '
                        '0.14 * log10(fmax(Pr, 1.0e-300))' +
                        utils.line_end[lang]
                        )
                file.write(line)

                line = ('  ' + get_array(lang, 'pres_mod', pind) +
                        ' = ' + utils.exp_10_fun[lang]
                        )
                line += 'logFcent / (1.0 + A * A / (B * B))) '

            elif reac.sri:
                # SRI form

                line = ('  X = 1.0 / (1.0 + log10(fmax(Pr, 1.0e-300)) * '
                        'log10(fmax(Pr, 1.0e-300)))' +
                        utils.line_end[lang]
                        )
                file.write(line)

                line = '  ' + get_array(lang, 'pres_mod', pind)
                line += ' = pow({:.6} * '.format(reac.sri_par[0])
                # Need to check for negative parameters, and
                # skip "-" sign if so.
                if reac.sri_par[1] > 0.0:
                    line += 'exp(-{:.6} / T)'.format(reac.sri_par[1])
                else:
                    line += 'exp({:.6} / T)'.format(abs(reac.sri_par[1]))

                if reac.sri_par[2] > 0.0:
                    line += ' + exp(-T / {:.6}), X) '.format(reac.sri_par[2])
                else:
                    line += ' + exp(T / {:.6}), X) '.format(abs(reac.sri_par[2]))

                if (len(reac.sri_par) == 5 and
                        reac.sri_par[3] != 1.0 and reac.sri_par[4] != 0.0):
                    line += ('* {:.8e} * '.format(reac.sri_par[3]) +
                             'pow(T, {:.6}) '.format(reac.sri_par[4])
                             )
            else:
                # simple falloff fn (i.e. F = 1)
                simple = True
                line = '  ' + get_array(lang, 'pres_mod', pind) + ' = '
                # regardless of F formulation
                if reac.low:
                    # unimolecular/recombination fall-off reaction
                    line += ' Pr / (1.0 + Pr)'
                elif reac.high:
                    # chemically-activated bimolecular reaction
                    line += '1.0 / (1.0 + Pr)'

            if not simple:
                # regardless of F formulation
                if reac.low:
                    # unimolecular/recombination fall-off reaction
                    line += '* Pr / (1.0 + Pr)'
                elif reac.high:
                    # chemically-activated bimolecular reaction
                    line += '/ (1.0 + Pr)'

            line += utils.line_end[lang]
            file.write(line)

        # space in between each reaction
        file.write('\n')
        pind += 1

    if lang in ['c', 'cuda']:
        file.write('} // end get_rxn_pres_mod\n\n')
    elif lang == 'fortran':
        file.write('end subroutine get_rxn_pres_mod\n\n')
    elif lang == 'matlab':
        file.write('end\n\n')

    file.close()

    return


def write_spec_rates(path, lang, specs, reacs, fwd_spec_mapping,
                    fwd_rxn_mapping, smm=None, auto_diff=False):
    """Write subroutine to evaluate species rates of production.

    Parameters
    ----------
    path : str
        Path to build directory for file.
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Programming language.
    specs : list of `SpecInfo`
        List of species in mechanism.
    reacs : list of `ReacInfo`
        List of reactions in mechanism.
    fwd_spec_mapping : list of int
        the index of the species in the original mechanism
    fwd_rxn_mapping : list of int
        the index of the reactions in the original mechanism
    smm : shared_memory_manager, optional
        If not ```None```, `shared_memory_manager` to use for CUDA optimizations
    auto_diff : bool, optional
        If ```True```, generate files for Adept autodifferention library.

    Returns
    -------
    seen : list of `bool`, ``True`` if species rate i is not identically zero

    """

    double_type = 'double'
    file_prefix =''
    if auto_diff:
        double_type = 'adouble'
        file_prefix = 'ad_'

    filename = file_prefix + 'spec_rates' + utils.file_ext[lang]
    file = open(os.path.join(path, filename), 'w')

    if lang in ['c', 'cuda']:
        file.write('#include "header{}"\n'.format(utils.header_ext[lang])
                   )
        #if lang == 'cuda' and smm is not None:
        #    file.write('#include <assert.h>\n')
        if auto_diff:
            file.write('#include "adept.h"\n'
                       'using adept::adouble;\n')
        file.write('#include "{}rates{}"\n'.format(file_prefix, utils.header_ext[lang]))
        file.write('\n')

    num_s = len(specs)
    num_r = len(reacs)
    rev_reacs = [i for i, rxn in enumerate(reacs) if rxn.rev]
    num_rev = len(rev_reacs)

    # pressure dependent reactions
    pdep_reacs = []
    for i_rxn, reac in enumerate(reacs):
        if reac.thd_body or reac.pdep:
            # add reaction index to list
            pdep_reacs.append(i_rxn)

    line = ''
    if lang == 'cuda': line = '__device__ '

    if lang in ['c', 'cuda']:
        line += ('void eval_spec_rates (const {0} * {1} fwd_rates,'
                 ' const {0} * {1} rev_rates, const {0} * {1} pres_mod,'
                 ' {0} * {1} sp_rates, {0} * {1} dy_N) {{\n'.format(double_type,
                 utils.restrict[lang])
                 )
    elif lang == 'fortran':
        line += ('subroutine eval_spec_rates (fwd_rates, rev_rates,'
                 ' pres_mod, sp_rates, dy_N)\n\n'
                 )

        # fortran needs type declarations
        line += '  implicit none\n'
        line += ('  double precision, intent(in) :: '
                 'fwd_rates({0}), rev_rates({0}), '.format(num_r) +
                 'pres_mod({})\n'.format(len(pdep_reacs))
                 )
        line += ('  double precision, intent(out) :: '
                 'sp_rates({}), dy_N\n'.format(num_s) +
                 '\n'
                 )
    elif lang == 'matlab':
        line += ('function sp_rates = eval_spec_rates (fwd_rates,'
                 ' rev_rates, pres_mod, dy_N)\n\n'
                 )
        line += '  sp_rates = zeros({},1);\n'.format(len(specs))
    file.write(line)


    def __get_var(spind):
        if spind + 1 == len(specs):
            line = '  ' + get_array(lang, '(*dy_N)', None)
        else:
            line = '  ' + get_array(lang, 'sp_rates', spind)
        return line

    def __get_smm_var(sp):
        return shared.variable('sp_rates', sp) if sp + 1 != len(specs) \
            else shared.variable('(*dy_N)', None)

    #if lang == 'cuda' and smm is not None:
    #    file.write('  assert(threadIdx.x + {} * blockDim.x < {});\n'.format(
    #        smm.shared_per_thread - 1, smm.shared_per_block))
    first_use = [True for spec in specs]
    first_smem_use = {}
    seen = [False for spec in specs]
    new_loads = []
    def __on_eviction(sp, shared, shared_ind):
        index = len(specs) - 1 if sp.index is None else sp.index
        file.write('  {} {}= {}'.format(sp.to_string(),
                    # is only a += if the species in question has been updated
                    # previously
                   '+' if not first_use[index] else '',
                   shared) + utils.line_end[lang])
        first_use[index] = False

    get_array = utils.get_array
    if lang == 'cuda' and smm is not None:
        smm.reset()
        get_array = smm.get_array
        smm.write_init(file, indent=2)
        smm.set_on_eviction(__on_eviction)

    #loop through reaction
    for rind in range(len(reacs)):
        print_ind = fwd_rxn_mapping[rind]
        file.write(utils.line_start + utils.comment[lang] +
                    'rxn {}'.format(print_ind) + '\n')
        rxn = reacs[rind]
        #get allowed species
        my_specs = [x for x in set(rxn.reac + rxn.prod)
                        if utils.get_nu(x, rxn) != 0.]
        if lang == 'cuda' and smm is not None:
            the_vars = [__get_smm_var(sp) for sp in my_specs]
            # estimate usages
            usages = []
            for sp in set(rxn.reac + rxn.prod):
                temp = rind + 1
                while (temp < len(reacs) and
                      utils.get_nu(sp, reacs[temp])) != 0.:
                    temp += 1
                usages.append(temp - rind - 1)
            first_smem_use = smm.load_into_shared(file, the_vars,
                                                  usages, load=False
                                                  )

        # loop through species
        for spind in my_specs:
            sp = specs[spind]

            #find nu
            nu = utils.get_nu(spind, rxn)
            if nu == 0.0:
                continue

            file.write(utils.line_start + utils.comment[lang] +
                       'sp {}'.format(fwd_spec_mapping[spind]) + '\n'
                       )

            sign = '-' if nu < 0 else '+'
            line = __get_var(spind)
            if lang == 'cuda' and smm is not None:
                tempvar = __get_smm_var(spind)
                smem_ind, smem_var = smm.get_index(tempvar)
                if smem_ind is not None:
                    #this is loaded into shared memory
                    if first_smem_use[smem_ind]:
                        #if it's the first time the value has been used
                        #use an ='s
                        line += ' = {}'.format(sign if sign == '-' else '')
                    else:
                        #otherwise +/- =
                        line += ' {}= '.format(sign)
                    first_smem_use[smem_ind] = False
                else:
                    #this is not loaded into shared memory
                    if first_use[spind]:
                        #if it's the first time the value has been used
                        #use an ='s
                        line += ' = {}'.format(sign if sign == '-' else '')
                    else:
                        #otherwise +/- =
                        line += ' {}= '.format(sign)
                    first_use[spind] = False
            else:
                line += ' {}= '.format(sign if not first_use[spind] else '')
                if not seen[spind] and nu < 0:
                    line += sign
                first_use[spind] = False
            nu = abs(nu)
            if nu != 1.0:
                if utils.is_integer(nu):
                    line += '{} * '.format(float(nu))
                else:
                    line += '{:3} * '.format(nu)
            if rxn.rev:
                rxn_out = (
                    '(' + get_array(lang, 'fwd_rates', rind) +
                    ' - ' + get_array(lang, 'rev_rates',
                                rev_reacs.index(rind)) + ')'
                    )
            else:
                rxn_out = get_array(lang, 'fwd_rates', rind)

            seen[spind] = True

            # pressure dependence modification
            if rxn.thd_body or rxn.pdep:
                pind = pdep_reacs.index(rind)
                rxn_out += ' * ' + get_array(lang, 'pres_mod', pind)

            # if lang == 'cuda':
            #    rxn_out = get_array(lang, rxn_out, None, preformed=True)
            line += rxn_out

            # done with this species
            line += utils.line_end[lang]
            file.write(line)
        file.write('\n')

    for i, seen_sp in enumerate(seen):
        if not seen_sp:
            file.write(utils.line_start + utils.comment[lang] +
                'sp {}'.format(fwd_spec_mapping[i]) + '\n')
            file.write(__get_var(i) +
                       ' = 0.0' + utils.line_end[lang]
                       )

    if lang == 'cuda' and smm is not None:
        smm.force_eviction()
        smm.set_on_eviction(None)

    if lang in ['c', 'cuda']:
        file.write('} // end eval_spec_rates\n\n')
    elif lang == 'fortran':
        file.write('end subroutine eval_spec_rates\n\n')
    elif lang == 'matlab':
        file.write('end\n\n')

    file.close()

    return seen

def polyfit_kernel_gen(varname, nicename, eqs, specs,
                            loopy_opt, test_size=None):
    """Helper function that generates kernels for
       evaluation of various thermodynamic species properties

    Parameters
    ----------
    varname : str
        The variable to generate the kernel for
    nicename : str
        The variable name to use in generated code
    eqs : dict of `sympy.Symbol`
        Dictionary defining conditional equations for the variables (keys)
    specs : list of `SpecInfo`
        List of species in the mechanism.
    lang : {'c', 'cuda', 'opencl'}
        Programming language.
    loopy_opt : `loopy_options` object
        A object containing all the loopy options to execute
    test_size : int
        If not none, this kernel is being used for testing. Hence we need to size the arrays accordingly

    Returns
    -------
    knl : `loopy.kernel`
        The generated loopy kernel for code generation / testing

    """

    if loopy_opt.width is not None and loopy_opt.depth is not None:
        raise Exception('Cannot specify both SIMD/SIMT width and depth')

    var = next(v for v in eqs.keys() if str(v) == varname)
    eq = eqs[var]
    poly_dim = specs[0].hi.shape[0]
    Ns = len(specs)

    k = sp.Idx('k')
    if loopy_opt.order == 'gpu':
        eq_lo = str(eq.subs([(sp.IndexedBase('a')[k, i],
                    sp.IndexedBase('a_lo')[k, i]) for i in range(poly_dim)]))
        eq_hi = str(eq.subs([(sp.IndexedBase('a')[k, i],
                    sp.IndexedBase('a_hi')[k, i]) for i in range(poly_dim)]))
    else:
        eq_lo = eq.subs(sp.IndexedBase('a')[k, 0], sp.IndexedBase('a_lo')[k, 0])
        eq_lo = str(eq.subs([(sp.IndexedBase('a')[k, i],
                    sp.IndexedBase('a_lo')[i, k]) for i in range(poly_dim)]))
        eq_hi = str(eq.subs([(sp.IndexedBase('a')[k, i],
                    sp.IndexedBase('a_hi')[i, k]) for i in range(poly_dim)]))

    #pick out a values and T_mid
    a_lo = np.zeros((Ns, poly_dim), dtype=np.float64)
    a_hi = np.zeros((Ns, poly_dim), dtype=np.float64)
    T_mid = np.zeros((Ns,), dtype=np.float64)
    for ind, spec in enumerate(specs):
        a_lo[ind, :] = spec.lo[:]
        a_hi[ind, :] = spec.hi[:]
        T_mid[ind] = spec.Trange[1]

    #need to transpose poly arrays
    if loopy_opt.order == 'cpu':
        a_lo = a_lo.T.copy()
        a_hi = a_hi.T.copy()

    #loopy variables
    a_lo_lp = lp.TemporaryVariable('a_lo', shape=a_lo.shape, initializer=a_lo, read_only=True,
                                scope=scopes.GLOBAL)
    a_hi_lp = lp.TemporaryVariable('a_hi', shape=a_hi.shape, initializer=a_hi, read_only=True,
                                scope=scopes.GLOBAL)
    T_mid_lp = lp.TemporaryVariable('T_mid', shape=T_mid.shape, initializer=T_mid, read_only=True,
                                scope=scopes.GLOBAL)

    target = utils.get_target(loopy_opt.lang)

    #create the loopy arrays
    test_size = 1 if test_size is None else test_size
    T_lp = lp.GlobalArg('T_arr', shape=(test_size,), dtype=np.float64)
    out_lp = lp.GlobalArg(nicename, shape=lp.auto, dtype=np.float64)
    #correct indexing
    indexed = nicename + ('[k, i]' if loopy_opt.order == 'gpu' else '[i, k]')

    #first we generate the kernel
    knl = lp.make_kernel('{{[k, i]: 0<=k<{} and 0<=i<{}}}'.format(Ns,
        test_size),
        """
        for i
            <> T = T_arr[i]
            for k
                <> Tcond = T < T_mid[k]
                if Tcond
                    {0} = {1}
                end
                if not Tcond
                    {0} = {2}
                end
            end
        end
        """.format(indexed, eq_lo, eq_hi),
        target=target,
        kernel_data=[a_lo_lp, a_hi_lp, T_mid_lp, T_lp, out_lp],
        name='eval_{}'.format(nicename)
        )

    knl = lp.prioritize_loops(knl, ['i', 'k'])

    if loopy_opt.depth is not None:
        #and assign the l0 axis to 'k'
        knl = lp.split_iname(knl, 'k', loopy_opt.depth, inner_tag='l.0')
        #assign g0 to 'i'
        knl = lp.tag_inames(knl, [('i', 'g.0')])
    elif loopy_opt.width is not None:
        #make the kernel a block of specifed width
        knl = lp.split_iname(knl, 'i', loopy_opt.width, inner_tag='l.0')
        #assign g0 to 'i'
        knl = lp.tag_inames(knl, [('i_outer', 'g.0')])

    #specify R
    knl = lp.fix_parameters(knl, R_u=chem.RU)

    #now do unr / ilp
    k_tag = 'k_outer' if loopy_opt.depth is not None else 'k'
    if loopy_opt.unr is not None:
        knl = lp.split_iname(knl, k_tag, loopy_opt.unr, inner_tag='unr')
    elif loopy_opt.ilp:
        knl = lp.tag_inames(knl, [(k_tag, 'ilp')])

    return knl


def write_chem_utils(path, specs, eqs, opts, auto_diff=False):
    """Write subroutine to evaluate species thermodynamic properties.

    Notes
    -----
    Thermodynamic properties include:  enthalpy, energy, specific heat
    (constant pressure and volume).

    Parameters
    ----------
    path : str
        Path to build directory for file.
    specs : list of `SpecInfo`
        List of species in the mechanism.
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    opts : `loopy_options` object
        A object containing all the loopy options to execute
    auto_diff : bool
        If ``True``, generate files for Adept autodifferention library.
    Returns
    -------
    None

    """

    file_prefix = ''
    if auto_diff:
        file_prefix = 'ad_'

    target = utils.get_target(opts.lang)

    #generate the kernels
    conp_eqs = eqs['conp']
    conv_eqs = eqs['conv']


    namelist = ['cp', 'cv', 'h', 'u']
    kernels = []
    headers = []
    code = []
    for varname, nicename in [('{C_p}[k]', 'cp'),
        ('H[k]', 'h'), ('{C_v}[k]', 'cv'),
        ('U[k]', 'u')]:
        eq = conp_eqs if nicename in ['h', 'cp'] else conv_eqs
        kernels.append(polyfit_kernel_gen(varname, nicename,
            eq, specs, opts))

    #get headers
    for i in range(len(namelist)):
        headers.append(lp_utils.get_header(kernels[i]) + utils.line_end[opts.lang])

    #and code
    for i in range(len(namelist)):
        code.append(lp_utils.get_code(kernels[i]))

    #need to filter out double definitions of constants in code
    preamble = []
    inkernel = False
    for line in code[0].split('\n'):
        if not re.search(r'eval_\w+', line):
            preamble.append(line)
        else:
            break
    code = [line for line in '\n'.join(code).split('\n') if line not in preamble]


    #now write
    with filew.get_header_file(os.path.join(path, file_prefix + 'chem_utils'
                             + utils.header_ext[opts.lang]), opts.lang) as file:

        if auto_diff:
            file.add_headers('adept.h')
            file.add_lines('using adept::adouble;\n')
            lines = [x.replace('double', 'adouble') for x in lines]

        lines = '\n'.join(headers).split('\n')
        file.add_lines(lines)


    with filew.get_file(os.path.join(path, file_prefix + 'chem_utils'
                             + utils.file_ext[opts.lang]), opts.lang) as file:
        file.add_lines(preamble + code)


def write_derivs(path, lang, specs, reacs, specs_nonzero, auto_diff=False):
    """Writes derivative function file and header.

    Parameters
    ----------
    path : str
        Path to build directory for file.
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Programming language.
    specs : list of `SpecInfo`
        List of species in the mechanism.
    reacs : list of `ReacInfo`
        List of reactions in the mechanism.
    specs_nonzero : list of bool
        List of `bool` indicating species with zero net production
    auto_diff : bool, optional
        If ``True``, generate files for Adept autodifferention library.

    Returns
    -------
    None

    """


    file_prefix = ''
    double_type = 'double'
    pres_ref = ''
    if auto_diff:
        file_prefix = 'ad_'
        double_type = 'adouble'
        pres_ref = '&'

    pre = ''
    if lang == 'cuda': pre = '__device__ '

    # first write header file
    file = open(os.path.join(path, file_prefix + 'dydt' +
                            utils.header_ext[lang]), 'w')
    file.write('#ifndef DYDT_HEAD\n'
               '#define DYDT_HEAD\n'
               '\n'
               '#include "header{}"\n'.format(utils.header_ext[lang]) +
               '\n'
               )
    if auto_diff:
        file.write('#include "adept.h"\n'
                   'using adept::adouble;\n')
    file.write('{0}void dydt (const double, const {1}{2}, '
               'const {1} * {3}, {1} * {3}'.format(pre, double_type, pres_ref,
                                utils.restrict[lang]) +
               ('' if lang == 'c' else
                    ', const mechanism_memory * {}'.format(utils.restrict[lang])) +
               ');\n'
               '\n'
               '#endif\n'
               )
    file.close()

    filename = file_prefix + 'dydt' + utils.file_ext[lang]
    file = open(os.path.join(path, filename), 'w')

    file.write('#include "header{}"\n'.format(utils.header_ext[lang]))

    file.write('#include "{0}chem_utils{1}"\n'
               '#include "{0}rates{1}"\n'.format(file_prefix,
                                            utils.header_ext[lang]))
    if lang == 'cuda':
        file.write('#include "gpu_memory.cuh"\n'
                   )
    file.write('\n')
    if auto_diff:
        file.write('#include "adept.h"\n'
                   'using adept::adouble;\n')

    ##################################################################
    # constant pressure
    ##################################################################
    file.write('#if defined(CONP)\n\n')

    line = (pre + 'void dydt (const double t, const {0}{1} pres, '
                  'const {0} * {2} y, {0} * {2} dy{3}) {{\n\n'.format(
                  double_type, pres_ref, utils.restrict[lang],
                  ', const mechanism_memory * {} d_mem'.format(utils.restrict[lang])
                  if lang == 'cuda' else '')
            )
    file.write(line)

    # calculation of species molar concentrations
    file.write('  // species molar concentrations\n')
    file.write(('  {0} conc[{1}]'.format(double_type, len(specs)) if lang != 'cuda'
               else '  double * {} conc = d_mem->conc'.format(utils.restrict[lang]))
               + utils.line_end[lang]
               )

    file.write('  {0} y_N;\n'.format(double_type))
    file.write('  {0} mw_avg;\n'.format(double_type))
    file.write('  {0} rho;\n'.format(double_type))

    # Simply call subroutine
    file.write('  eval_conc (' + utils.get_array(lang, 'y', 0) +
               ', pres, &' + (utils.get_array(lang, 'y', 1) if lang != 'cuda'
                                else 'y[GRID_DIM]') + ', '
               '&y_N, &mw_avg, &rho, conc);\n\n'
               )

    # evaluate reaction rates
    rev_reacs = [i for i, rxn in enumerate(reacs) if rxn.rev]
    if lang == 'cuda':
        file.write('  double * {} fwd_rates = d_mem->fwd_rates'.format(utils.restrict[lang])
                   + utils.line_end[lang])
    else:
        file.write('  // local arrays holding reaction rates\n'
                   '  {} fwd_rates[{}];\n'.format(double_type, len(reacs))
                   )
    if rev_reacs and lang == 'cuda':
        file.write('  double * {} rev_rates = d_mem->rev_rates'.format(utils.restrict[lang])
                   + utils.line_end[lang])
    elif rev_reacs:
        file.write('  {} rev_rates[{}];\n'.format(double_type, len(rev_reacs)))
    else:
        file.write('  {}* rev_rates = 0;\n'.format(double_type))
    cheb = False
    if any(rxn.cheb for rxn in reacs) and lang == 'cuda':
        cheb = True
        file.write('  double * {} dot_prod = d_mem->dot_prod'.format(utils.restrict[lang])
                   + utils.line_end[lang])
    file.write('  eval_rxn_rates (' + utils.get_array(lang, 'y', 0) +
               ', pres, conc, fwd_rates, rev_rates{});\n\n'.format(', dot_prod'
                    if cheb else '')
               )

    # reaction pressure dependence
    num_dep_reacs = sum([rxn.thd_body or rxn.pdep for rxn in reacs])
    if num_dep_reacs > 0:
        file.write('  // get pressure modifications to reaction rates\n')
        if lang == 'cuda':
            file.write('  double * {} pres_mod = d_mem->pres_mod'.format(utils.restrict[lang]) +
                   utils.line_end[lang])
        else:
            file.write('  {} pres_mod[{}];\n'.format(double_type, num_dep_reacs))
        file.write('  get_rxn_pres_mod (' + utils.get_array(lang, 'y', 0) +
                   ', pres, conc, pres_mod);\n'
                   )
    else:
        file.write('  {}* pres_mod = 0;\n'.format(double_type))
    file.write('\n')

    if lang == 'cuda':
        file.write('  double * {} spec_rates = d_mem->spec_rates'.format(utils.restrict[lang]) +
                   utils.line_end[lang])

    # species rate of change of molar concentration
    file.write('  // evaluate species molar net production rates\n')
    if lang != 'cuda':
        file.write('  {} dy_N;\n'.format(double_type))
    file.write('  eval_spec_rates (fwd_rates, rev_rates, pres_mod, ')
    if lang == 'c':
        file.write('&' + utils.get_array(lang, 'dy', 1) + ', &dy_N)')
    elif lang == 'cuda':
        file.write('spec_rates, &' + utils.get_array(lang, 'spec_rates', len(specs) - 1) +
                    ')')
    file.write(utils.line_end[lang])

    # evaluate specific heat
    file.write('  // local array holding constant pressure specific heat\n')
    file.write(('  {} cp[{}]'.format(double_type, len(specs)) if lang != 'cuda'
               else '  double * {} cp = d_mem->cp'.format(utils.restrict[lang]))
               + utils.line_end[lang])
    file.write('  eval_cp (' + utils.get_array(lang, 'y', 0) + ', cp)'
               + utils.line_end[lang] + '\n')

    file.write('  // constant pressure mass-average specific heat\n')
    line = '  {} cp_avg = '.format(double_type)
    isfirst = True
    for isp, sp in enumerate(specs[:-1]):
        if len(line) > 70:
            line += '\n'
            file.write(line)
            line = '             '

        if not isfirst: line += ' + '

        line += '(' + utils.get_array(lang, 'cp', isp) + \
                ' * ' + utils.get_array(lang, 'y', isp + 1) + ')'

        isfirst = False

    if not isfirst: line += ' + '
    line += '(' + utils.get_array(lang, 'cp', len(specs) - 1) + ' * y_N)'
    file.write(line + utils.line_end[lang] + '\n')

    file.write('  // local array for species enthalpies\n' +
              ('  {} h[{}]'.format(double_type, len(specs)) if lang != 'cuda'
               else '  double * {} h = d_mem->h'.format(utils.restrict[lang]))
               + utils.line_end[lang])
    file.write('  eval_h(' + utils.get_array(lang, 'y', 0) + ', h);\n')

    # energy equation
    file.write('  // rate of change of temperature\n')
    line = ('  ' + utils.get_array(lang, 'dy', 0) +
            ' = (-1.0 / (rho * cp_avg)) * ('
            )
    isfirst = True
    for isp, sp in enumerate(specs):
        if not specs_nonzero[isp]:
            continue
        if len(line) > 70:
            line += '\n'
            file.write(line)
            line = '       '

        if not isfirst: line += ' + '

        if lang == 'c':
            arr = utils.get_array(lang, 'dy', isp + 1) if isp < len(specs) - 1 else 'dy_N'
        elif lang == 'cuda':
            arr = utils.get_array(lang, 'spec_rates', isp)
        line += ('(' + arr + ' * ' +
                 utils.get_array(lang, 'h', isp) + ' * {:.16e})'.format(sp.mw)
                 )

        isfirst = False
    line += ')' + utils.line_end[lang] + '\n'
    file.write(line)


    line = ''
    # rate of change of species mass fractions
    file.write('  // calculate rate of change of species mass fractions\n')
    for isp, sp in enumerate(specs[:-1]):
        if lang == 'c':
            file.write('  ' + utils.get_array(lang, 'dy', isp + 1) +
                   ' *= ({:.16e} / rho);\n'.format(sp.mw)
                   )
        elif lang == 'cuda':
            file.write('  ' + utils.get_array(lang, 'dy', isp + 1) +
                       ' = ' + utils.get_array(lang, 'spec_rates', isp) +
                       ' * ({:.16e} / rho);\n'.format(sp.mw)
                       )
    file.write('\n')

    file.write('} // end dydt\n\n')

    ##################################################################
    # constant volume
    ##################################################################
    file.write('#elif defined(CONV)\n\n')

    file.write(pre + 'void dydt (const double t, const {0} rho, '
                     'const {0} * {1} y, {0} * {1} dy{2}) {{\n\n'.format(double_type,
                     utils.restrict[lang],
                     '' if lang != 'cuda' else
                     ', mechanism_memory * {} d_mem'.format(utils.restrict[lang]))
               )

    # calculation of species molar concentrations
    file.write('  // species molar concentrations\n')
    file.write(('  {0} conc[{1}]'.format(double_type, len(specs)) if lang != 'cuda'
               else '  double * {} conc = d_mem->conc'.format(utils.restrict[lang]))
               + utils.line_end[lang]
               )

    file.write('  {} y_N;\n'.format(double_type))
    file.write('  {} mw_avg;\n'.format(double_type))
    file.write('  {} pres;\n'.format(double_type))

    # Simply call subroutine
    file.write('  eval_conc_rho (' + utils.get_array(lang, 'y', 0) +
               'rho, &' + (utils.get_array(lang, 'y', 1) if lang != 'cuda'
                                else 'y[GRID_DIM]') + ', ' +
               '&y_N, &mw_avg, &pres, conc);\n\n'
               )

    # evaluate reaction rates
    rev_reacs = [i for i, rxn in enumerate(reacs) if rxn.rev]
    if lang == 'cuda':
        file.write('  double * {} fwd_rates = d_mem->fwd_rates'.format(utils.restrict[lang])
                   + utils.line_end[lang])
    else:
        file.write('  // local arrays holding reaction rates\n'
                   '  {} fwd_rates[{}];\n'.format(double_type, len(reacs))
                   )
    if rev_reacs and lang == 'cuda':
        file.write('  double * {} rev_rates = d_mem->rev_rates'.format(utils.restrict[lang])
                   + utils.line_end[lang])
    elif rev_reacs:
        file.write('  {} rev_rates[{}];\n'.format(double_type, len(rev_reacs)))
    else:
        file.write('  {}* rev_rates = 0;\n'.format(double_type))
    file.write('  eval_rxn_rates (' + utils.get_array(lang, 'y', 0) + ', '
               'pres, conc, fwd_rates, rev_rates);\n\n'
               )

    # reaction pressure dependence
    num_dep_reacs = sum([rxn.thd_body or rxn.pdep for rxn in reacs])
    if num_dep_reacs > 0:
        file.write('  // get pressure modifications to reaction rates\n')
        if lang == 'cuda':
            file.write('  double * {} pres_mod = d_mem->pres_mod'.format(utils.restrict[lang]) +
                   utils.line_end[lang])
        else:
            file.write('  {} pres_mod[{}];\n'.format(double_type, num_dep_reacs))
        file.write('  get_rxn_pres_mod (' + utils.get_array(lang, 'y', 0) +
                   ', pres, conc, pres_mod);\n'
                   )
    else:
        file.write('  {}* pres_mod = 0;\n'.format(double_type))
    file.write('\n')

    # species rate of change of molar concentration
    file.write('  // evaluate species molar net production rates\n'
               '  {} dy_N;'.format(double_type) +
               '  eval_spec_rates (fwd_rates, rev_rates, pres_mod, ')
    file.write('&' + utils.get_array(lang, 'dy', 1) if lang != 'cuda'
           else '&dy[GRID_DIM]')
    file.write(', &dy_N)' + utils.line_end[lang] + '\n')

    # evaluate specific heat
    file.write(('  {} cv[{}]'.format(double_type, len(specs)) if lang != 'cuda'
               else '  double * {} cv = d_mem->cp'.format(utils.restrict[lang]))
               + utils.line_end[lang])
    file.write('  eval_cv(' + utils.get_array(lang, 'y', 0) + ', cv);\n\n')

    file.write('  // constant volume mass-average specific heat\n')
    line = '  {} cv_avg = '.format(double_type)
    isfirst = True
    for idx, sp in enumerate(specs[:-1]):
        if len(line) > 70:
            line += '\n'
            file.write(line)
            line = '             '
        line += ' + ' if not isfirst else ''
        line += ('(' + utils.get_array(lang, 'cv', idx) + ' * ' +
                 utils.get_array(lang, 'y', idx + 1) + ')'
                 )

        isfirst = False
    line += '(' + utils.get_array(lang, 'cv', len(specs) - 1) + ' * y_N)'
    file.write(line + utils.line_end[lang] + '\n')

    # evaluate internal energy
    file.write('  // local array for species internal energies\n' +
              ('  {} u[{}]'.format(double_type, len(specs)) if lang != 'cuda'
               else '  double * {} u = d_mem->h'.format(utils.restrict[lang]))
               + utils.line_end[lang])
    file.write('  eval_u (' + utils.get_array(lang, 'y', 0) + ', u);\n\n')

    # energy equation
    file.write('  // rate of change of temperature\n')
    line = ('  ' + utils.get_array(lang, 'dy', 0) +
            ' = (-1.0 / (rho * cv_avg)) * ('
            )
    isfirst = True
    for isp, sp in enumerate(specs):
        if not specs_nonzero[isp]:
            continue
        if len(line) > 70:
            line += '\n'
            file.write(line)
            line = '       '

        if not isfirst: line += ' + '

        if lang == 'c':
            arr = utils.get_array(lang, 'dy', isp + 1) if isp < len(specs) - 1 else 'dy_N'
        elif lang == 'cuda':
            arr = utils.get_array(lang, 'spec_rates', isp)
        line += ('(' + arr + ' * ' +
                 utils.get_array(lang, 'u', isp) + ' * {:.16e})'.format(sp.mw)
                 )

        isfirst = False
    line += ')' + utils.line_end[lang] + '\n'
    file.write(line)


    # rate of change of species mass fractions
    file.write('  // calculate rate of change of species mass fractions\n')
    for isp, sp in enumerate(specs[:-1]):
        if lang == 'c':
            file.write('  ' + utils.get_array(lang, 'dy', isp + 1) +
                   ' *= ({:.16e} / rho);\n'.format(sp.mw)
                   )
        elif lang == 'cuda':
            file.write('  ' + utils.get_array(lang, 'dy', isp + 1) +
                       ' = ' + utils.get_array(lang, 'spec_rates', isp) +
                       ' * ({:.16e} / rho);\n'.format(sp.mw)
                       )

    file.write('\n')

    file.write('} // end dydt\n\n')

    file.write('#endif\n')

    file.close()
    return


def write_mass_mole(path, lang, specs):
    """Write files for mass/molar concentration and density conversion utility.

    Parameters
    ----------
    path : str
        Path to build directory for file.
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Programming language.
    specs : list of `SpecInfo`
        List of species in mechanism.

    Returns
    -------
    None

    """

    # Create header file
    if lang in ['c', 'cuda']:
        arr_lang = 'c'
        file = open(os.path.join(path, 'mass_mole{}'.format(
            utils.header_ext[lang])), 'w')

        file.write(
            '#ifndef MASS_MOLE_HEAD\n'
            '#define MASS_MOLE_HEAD\n'
            '\n'
            '#include "header{0}"\n'
            '\n'
            'void mole2mass (const double*, double*);\n'
            'void mass2mole (const double*, double*);\n'
            'double getDensity (const double, const double, const double*);\n'
            '\n'
            '#endif\n'.format(utils.header_ext[lang])
            )
        file.close()

    # Open file; both C and CUDA programs use C file (only used on host)
    filename = 'mass_mole' + utils.file_ext[lang]
    file = open(os.path.join(path, filename), 'w')

    if lang in ['c', 'cuda']:
        file.write('#include "mass_mole{}"\n\n'.format(
            utils.header_ext[lang]))

    ###################################################
    # Documentation and function/subroutine initializaton for mole2mass
    if lang in ['c', 'cuda']:
        file.write('/** Function converting species mole fractions to '
                   'mass fractions.\n'
                   ' *\n'
                   ' * \param[in]  X  array of species mole fractions\n'
                   ' * \param[out] Y  array of species mass fractions\n'
                   ' */\n'
                   'void mole2mass (const double * X, double * Y) {\n'
                   '\n'
                   )
    elif lang == 'fortran':
        file.write(
        '!-----------------------------------------------------------------\n'
        '!> Subroutine converting species mole fractions to mass fractions.\n'
        '!! @param[in]  X  array of species mole fractions\n'
        '!! @param[out] Y  array of species mass fractions\n'
        '!-----------------------------------------------------------------\n'
        'subroutine mole2mass (X, Y)\n'
        '  implicit none\n'
        '  double, dimension(:), intent(in) :: X\n'
        '  double, dimension(:), intent(out) :: X\n'
        '  double :: mw_avg\n'
        '\n'
        )

    file.write('  // mole fraction of final species\n')
    file.write(utils.line_start + 'double X_N' + utils.line_end[lang])
    line = '  X_N = 1.0 - ('
    isfirst = True
    for isp in range(len(specs) - 1):
        if len(line) > 70:
            line += '\n'
            file.write(line)
            line = '               '

        if not isfirst: line += ' + '

        line += utils.get_array(arr_lang, 'X', isp)

        isfirst = False
    line += ')'
    file.write(line + utils.line_end[lang])

    # calculate molecular weight
    if lang in ['c', 'cuda']:
        file.write('  // average molecular weight\n'
                   '  double mw_avg = 0.0;\n'
                   )
        for isp in range(len(specs) - 1):
            sp = specs[isp]
            file.write('  mw_avg += ' + utils.get_array(arr_lang, 'X', isp) +
                       ' * {:.16e};\n'.format(sp.mw)
                       )
        file.write(utils.line_start + 'mw_avg += X_N * ' +
                       '{:.16e};\n'.format(specs[-1].mw)
                       )
    elif lang == 'fortran':
        file.write('  ! average molecular weight\n'
                   '  mw_avg = 0.0\n'
                   )
        for isp, sp in enumerate(specs):
            file.write('  mw_avg = mw_avg + '
                       'X({}) * '.format(isp + 1) +
                       '{:.16e}\n'.format(sp.mw)
                       )
    file.write('\n')

    # calculate mass fractions
    if lang in ['c', 'cuda']:
        file.write('  // calculate mass fractions\n')
        for isp in range(len(specs) - 1):
            sp = specs[isp]
            file.write('  ' + utils.get_array(arr_lang, 'Y', isp) +
                        ' = ' +
                        utils.get_array(arr_lang, 'X', isp) +
                       ' * {:.16e} / mw_avg;\n'.format(sp.mw)
                       )
        file.write('\n'
                   '} // end mole2mass\n'
                   '\n'
                   )
    elif lang == 'fortran':
        file.write('  ! calculate mass fractions\n')
        for isp, sp in enumerate(specs):
            file.write('  Y({0}) = X({0}) * '.format(isp + 1) +
                       '{:.16e} / mw_avg\n'.format(sp.mw)
                       )
        file.write('\n'
                   'end subroutine mole2mass\n'
                   '\n'
                   )

    ################################
    # Documentation and function/subroutine initialization for mass2mole

    if lang in ['c', 'cuda']:
        file.write('/** Function converting species mass fractions to mole '
                   'fractions.\n'
                   ' *\n'
                   ' * \param[in]  Y  array of species mass fractions\n'
                   ' * \param[out] X  array of species mole fractions\n'
                   ' */\n'
                   'void mass2mole (const double * Y, double * X) {\n'
                   '\n'
                   )
    elif lang == 'fortran':
        file.write('!-------------------------------------------------------'
                   '----------\n'
                   '!> Subroutine converting species mass fractions to mole '
                   'fractions.\n'
                   '!! @param[in]  Y  array of species mass fractions\n'
                   '!! @param[out] X  array of species mole fractions\n'
                   '!-------------------------------------------------------'
                   '----------\n'
                   'subroutine mass2mole (Y, X)\n'
                   '  implicit none\n'
                   '  double, dimension(:), intent(in) :: Y\n'
                   '  double, dimension(:), intent(out) :: X\n'
                   '  double :: mw_avg\n'
                   '\n'
                   )

    # calculate Y_N
    file.write('  // mass fraction of final species\n')
    file.write(utils.line_start + 'double Y_N' + utils.line_end[lang])
    line = '  Y_N = 1.0 - ('
    isfirst = True
    for isp in range(len(specs) - 1):
        if len(line) > 70:
            line += '\n'
            file.write(line)
            line = '               '

        if not isfirst: line += ' + '

        line += utils.get_array(arr_lang, 'Y', isp)

        isfirst = False
    line += ')'
    file.write(line + utils.line_end[lang])

    # calculate average molecular weight
    if lang in ['c', 'cuda']:
        file.write('  // average molecular weight\n')
        file.write('  double mw_avg = 0.0;\n')
        for isp in range(len(specs) - 1):
            file.write('  mw_avg += ' + utils.get_array(arr_lang, 'Y', isp) +
                       ' / {:.16e};\n'.format(specs[isp].mw)
                       )
        file.write('  mw_avg += Y_N / ' +
                       '{:.16e};\n'.format(specs[-1].mw)
                       )
        file.write('  mw_avg = 1.0 / mw_avg;\n')
    elif lang == 'fortran':
        file.write('  ! average molecular weight\n')
        file.write('  mw_avg = 0.0\n')
        for isp, sp in enumerate(specs):
            file.write('  mw_avg = mw_avg + '
                       'Y({}) / '.format(isp + 1) +
                       '{:.16e}\n'.format(sp.mw)
                       )
    file.write('\n')

    # calculate mole fractions
    if lang in ['c', 'cuda']:
        file.write('  // calculate mole fractions\n')
        for isp in range(len(specs) - 1):
            file.write('  ' + utils.get_array(arr_lang, 'X', isp)
                      + ' = ' +
                      utils.get_array(arr_lang, 'Y', isp) +
                       ' * mw_avg / {:.16e};\n'.format(specs[isp].mw)
                       )
        file.write('\n'
                   '} // end mass2mole\n'
                   '\n'
                   )
    elif lang == 'fortran':
        file.write('  ! calculate mass fractions\n')
        for isp, sp in enumerate(specs):
            file.write('  X({0}) = Y({0}) * '.format(isp + 1) +
                       'mw_avg / {:.16e}\n'.format(sp.mw)
                       )
        file.write('\n'
                   'end subroutine mass2mole\n'
                   '\n'
                   )

    ###############################
    # Documentation and subroutine/function initialization for getDensity

    if lang in ['c', 'cuda']:
        file.write('/** Function calculating density from mole fractions.\n'
                   ' *\n'
                   ' * \param[in]  temp  temperature\n'
                   ' * \param[in]  pres  pressure\n'
                   ' * \param[in]  X     array of species mole fractions\n'
                   r' * \return     rho  mixture mass density' + '\n'
                   ' */\n'
                   'double getDensity (const double temp, const double '
                   'pres, '
                   'const double * X) {\n'
                   '\n'
                   )
    elif lang == 'fortran':
        file.write('!-------------------------------------------------------'
                   '----------\n'
                   '!> Function calculating density from mole fractions.\n'
                   '!! @param[in]  temp  temperature\n'
                   '!! @param[in]  pres  pressure\n'
                   '!! @param[in]  X     array of species mole fractions\n'
                   '!! @return     rho   mixture mass density' + '\n'
                   '!-------------------------------------------------------'
                   '----------\n'
                   'function mass2mole (temp, pres, X) result(rho)\n'
                   '  implicit none\n'
                   '  double, intent(in) :: temp, pres\n'
                   '  double, dimension(:), intent(in) :: X\n'
                   '  double :: mw_avg, rho\n'
                   '\n'
                   )

    file.write('  // mole fraction of final species\n')
    file.write(utils.line_start + 'double X_N' + utils.line_end[lang])
    line = '  X_N = 1.0 - ('
    isfirst = True
    for isp in range(len(specs) - 1):
        if len(line) > 70:
            line += '\n'
            file.write(line)
            line = '               '

        if not isfirst: line += ' + '

        line += utils.get_array(arr_lang, 'X', isp)

        isfirst = False
    line += ')'
    file.write(line + utils.line_end[lang])

    # get molecular weight
    if lang in ['c', 'cuda']:
        file.write('  // average molecular weight\n'
                   '  double mw_avg = 0.0;\n'
                   )
        for isp in range(len(specs) - 1):
            file.write('  mw_avg += ' + utils.get_array(arr_lang, 'X', isp) +
                       ' * {:.16e};\n'.format(specs[isp].mw)
                       )
        file.write(utils.line_start + 'mw_avg += X_N * ' +
               '{:.16e};\n'.format(specs[-1].mw))
        file.write('\n')
    elif lang == 'fortran':
        file.write('  ! average molecular weight\n'
                   '  mw_avg = 0.0\n'
                   )
        for isp, sp in enumerate(specs):
            file.write('  mw_avg = mw_avg + '
                       'X({}) * '.format(isp + 1) +
                       '{:.16e}\n'.format(sp.mw)
                       )
        file.write('\n')

    # calculate density
    if lang in ['c', 'cuda']:
        file.write('  return pres * mw_avg / ({:.8e} * temp);'.format(chem.RU))
        file.write('\n')
    else:
        line = '  rho = pres * mw_avg / ({:.8e} * temp)'.format(chem.RU)
        line += utils.line_end[lang]
        file.write(line)

    if lang in ['c', 'cuda']:
        file.write('} // end getDensity\n\n')
    elif lang == 'fortran':
        file.write('end function getDensity\n\n')

    file.close()
    return

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
