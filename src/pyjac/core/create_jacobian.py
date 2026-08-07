#! /usr/bin/env python
"""Creates source code for calculating analytical Jacobian matrix."""

# Standard libraries
import math
import os
import sys

# Local imports
from .. import utils
from . import CParams, CUDAParams
from . import cache_optimizer as cache
from . import chem_utilities as chem
from . import mech_auxiliary as aux
from . import mech_interpret as mech
from . import rate_subs as rate
from . import shared_memory as shared


def calculate_shared_memory(rxn_ind, rxn, specs, reacs, rev_reacs, pdep_reacs):
    """Estimates usage of the various variables for a given reaction

    Parameters
    ----------
    rxn_ind : into
        Index of reaction of interest.
    rxn : `ReacInfo`
        Reaction of interest.
    specs : list of `SpecInfo`
        List of species.
    reacs : list of `ReacInfo`
        Full list of reactions.
    rev_reacs : list of `ReacInfo`
        List of reversible reactions.
    pdep_reacs : list of `ReacInfo`
        List of pressure-dependent reactions.

    Returns
    -------
    variable_list :

    usages :


    """
    # need to figure out shared memory stuff
    variable_list = []
    usages = []
    fwd_usage = 3
    rev_usage = 3
    pres_mod_usage = 0 if not (rxn.pdep or rxn.thd_body) else (3 if rxn.thd_body else 2)
    reac_usages = [0 for i in range(len(rxn.reac))]
    prod_usages = [0 for i in range(len(rxn.prod))]
    # add variables
    variable_list.append(shared.variable('fwd_rates', rxn_ind))
    if rxn.rev:
        variable_list.append(shared.variable('rev_rates', rev_reacs.index(rxn_ind)))
    if rxn.pdep or rxn.thd_body:
        variable_list.append(shared.variable('pres_mod', pdep_reacs.index(rxn_ind)))
    for sp in set(rxn.reac + rxn.prod + [x[0] for x in rxn.thd_body_eff]):
        variable_list.append(shared.variable('conc', sp))

    for i, sp in enumerate(rxn.reac):
        nu = rxn.reac_nu[i]
        if nu - 1 > 0:
            reac_usages[i] += 1
        for sp2 in range(len(specs)):
            if nu - 1 > 0:
                reac_usages[i] += nu - 1
            if rxn.pdep or rxn.thd_body:
                pres_mod_usage += 1
            if sp == sp2:
                continue
            ind = next((ind for ind, spec in enumerate(rxn.reac) if spec == sp2), None)
            if ind is not None:
                reac_usages[ind] += 1

    if rxn.rev:
        for i, sp in enumerate(rxn.prod):
            nu = rxn.prod_nu[i]
            for sp2 in range(len(specs)):
                if nu - 1 > 0:
                    prod_usages[i] += nu - 1
                # already counted in reac
                #                if rxn.pdep or rxn.thd_body:
                #                    pres_mod_usage += 1
                if sp == sp2:
                    continue
                ind = next(
                    (ind for ind, spec in enumerate(rxn.prod) if spec == sp2), None
                )
                if ind is not None:
                    prod_usages[ind] += 1

    usages.append(fwd_usage)
    if rxn.rev:
        usages.append(rev_usage)
    if rxn.pdep or rxn.thd_body:
        usages.append(pres_mod_usage)
    for sp in set(rxn.reac + rxn.prod + [x[0] for x in rxn.thd_body_eff]):
        u = 0
        if sp in rxn.reac:
            u += reac_usages[rxn.reac.index(sp)]
        if sp in rxn.prod:
            u += prod_usages[rxn.prod.index(sp)]
        if sp in rxn.thd_body_eff:
            u += 1
        usages.append(u)

    return variable_list, usages


def write_dr_dy(file, lang, rev_reacs, rxn, rxn_ind, pres_rxn_ind, get_array):
    """Writes evaluation of the (non-pressure dependent part) of the
    reaction rate R that is independent of species

    Parameters
    ----------
    file : ``file`` object
        The open file object to write to
    lang : str
        The Programming language
    rev_reacs : list of `ReacInfo`
        The list of reverisble reactions
    rxn : `ReacInfo`
        The reaction to consider
    rxn_ind : int
        The index of the reaction in the mechanism
    pres_rxn_ind : int
        The index of the reaction in the pressure dependent reactions
    get_array : function
        The SMM binded get_array function (or utils.get_array) as required

    Returns
    -------
    None

    """
    # write the T_Pr and T_Fi terms if needed
    if (rxn.pdep or rxn.thd_body) and (rxn.thd_body_eff or rxn.pdep_sp):
        jline = utils.line_start + 'pres_mod_temp = '
        if rxn.pdep:
            jline += '('
            # dPr/dYj contribution
            if rxn.low:
                # unimolecular/recombination
                jline += '(1.0 / (1.0 + Pr))'
            elif rxn.high:
                # chem-activated bimolecular
                jline += '(-Pr / (1.0 + Pr))'
            if rxn.troe:
                jline += (
                    ' - log(fmax(Fcent, 1.0e-300)) * 2.0 * A * (B * '
                    f'{1.0 / math.log(10.0):.16}' + ' + A * '
                    f'{0.14 / math.log(10.0):.16}) / '
                    + '(B * B * B * (1.0 + A * A / (B * B)) '
                    '* (1.0 + A * A / (B * B)))'
                )
            elif rxn.sri:
                jline += (
                    '- X * X * '
                    f'{2.0 / math.log(10.0):.16} * ' + 'log10(fmax(Pr, 1.0e-300)) * '
                    f'log({rxn.sri_par[0]:.4} * '
                    + f'exp({-rxn.sri_par[1]:.4} / T) + '
                    + f'exp(T / {-rxn.sri_par[2]:.4}))'
                )

            jline += ') * '
        if rxn.rev:
            jline += '(' + get_array(lang, 'fwd_rates', rxn_ind)
            jline += ' - ' + get_array(lang, 'rev_rates', rev_reacs.index(rxn_ind))
            jline += ')'
        else:
            jline += get_array(lang, 'fwd_rates', rxn_ind)
        file.write(jline + utils.line_end[lang])

    jline = '  j_temp = -mw_avg * rho_inv * '
    # next, contribution from dR/dYj
    # namely the T_dy independent term
    if rxn.pdep or rxn.thd_body:
        jline += get_array(lang, 'pres_mod', pres_rxn_ind)
        jline += ' * ('
    else:
        jline += '('

    reac_nu = 0
    prod_nu = 0
    if rxn.thd_body_eff and not rxn.pdep:
        reac_nu = 1
        if rxn.rev:
            prod_nu = 1

    # get reac and prod nu sums
    reac_nu += sum(rxn.reac_nu)

    if rxn.rev:
        prod_nu += sum(rxn.prod_nu)

    if reac_nu != 0:
        if reac_nu != 1:
            jline += f'{float(reac_nu)} * '
        jline += '' + get_array(lang, 'fwd_rates', rxn_ind)

    if prod_nu != 0:
        if prod_nu == 1:
            jline += ' - '
        else:
            jline += f' - {float(prod_nu)} * '
        jline += '' + get_array(lang, 'rev_rates', rev_reacs.index(rxn_ind))

    if rxn.pdep and (rxn.pdep_sp or rxn.thd_body_eff):
        jline += ' + pres_mod_temp'
    jline += ')'

    file.write(jline + utils.line_end[lang])

    if rxn.pdep and (rxn.pdep_sp or rxn.thd_body_eff):
        jline = ''

        if rxn.low:
            k0 = rxn.low
            kinf = [rxn.A, rxn.b, rxn.E]
        else:
            k0 = [rxn.A, rxn.b, rxn.E]
            kinf = rxn.high
        jline = utils.line_start + 'pres_mod_temp *= '
        # k0 / kinf
        jline += rate.rxn_rate_const(k0[0] / kinf[0], k0[1] - kinf[1], k0[2] - kinf[2])
        # Fi
        if rxn.troe:
            jline += ' * pow(Fcent, 1.0 / (1 + A * A / (B * B)))'
        elif rxn.sri:
            jline += f'* pow({rxn.sri_par[0]:.6} * '
            # Need to check for negative parameters, and
            # skip "-" sign if so.
            if rxn.sri_par[1] > 0.0:
                jline += f'exp(-{rxn.sri_par[1]:.6} / T)'
            else:
                jline += f'exp({abs(rxn.sri_par[1]):.6} / T)'

            if rxn.sri_par[2] > 0.0:
                jline += f' + exp(-T / {rxn.sri_par[2]:.6}), X) '
            else:
                jline += f' + exp(T / {abs(rxn.sri_par[2]):.6}), X) '

            if (
                len(rxn.sri_par) == 5
                and rxn.sri_par[3] != 1.0
                and rxn.sri_par[4] != 0.0
            ):
                jline += f'* {rxn.sri_par[3]:.8e} * ' + f'pow(T, {rxn.sri_par[4]:.6}) '
        jline += ' / (1.0 + Pr)'
        file.write(jline + utils.line_end[lang])


def write_rates(file, lang, rxn):
    """Write evaluation of the forward/reverse rate constant

    Parameters
    ----------
    file : ``file`` object
        The open file object to write to
    lang : str
        The Programming language
    rxn : `ReacInfo`
        The reaction to consider

    Returns
    -------
    None

    """

    if not (rxn.cheb or rxn.plog):
        file.write(
            '  kf = ' + rate.rxn_rate_const(rxn.A, rxn.b, rxn.E) + utils.line_end[lang]
        )
    elif rxn.plog:
        vals = rxn.plog_par[0]
        file.write(f'  if (pres <= {vals[0]:.4e}) {{\n')
        line = '    kf = ' + rate.rxn_rate_const(vals[1], vals[2], vals[3])
        file.write(line + utils.line_end[lang])

        for idx, vals in enumerate(rxn.plog_par[:-1]):
            vals2 = rxn.plog_par[idx + 1]

            line = (
                f'  }} else if ((pres > {vals[0]:.4e}) '
                + f'&& (pres <= {vals2[0]:.4e})) {{\n'
            )
            file.write(line)

            line = (
                '    kf = log(' + rate.rxn_rate_const(vals[1], vals[2], vals[3]) + ')'
            )
            file.write(line + utils.line_end[lang])
            line = (
                '    kf2 = log('
                + rate.rxn_rate_const(vals2[1], vals2[2], vals2[3])
                + ')'
            )
            file.write(line + utils.line_end[lang])

            pres_log_diff = math.log(vals2[0]) - math.log(vals[0])
            line = (
                '    kf = exp(kf + (kf2 - kf) * (log(pres) - '
                + f'{math.log(vals[0]):.16e}) / '
                + f'{pres_log_diff:.16e})'
            )
            file.write(line + utils.line_end[lang])

        vals = rxn.plog_par[-1]
        file.write(f'  }} else if (pres > {vals[0]:.4e}) {{\n')
        line = '    kf = ' + rate.rxn_rate_const(vals[1], vals[2], vals[3])
        file.write(line + utils.line_end[lang])
        file.write('  }\n')
    elif rxn.cheb:
        file.write(rate.get_cheb_rate(lang, rxn, False))
    if rxn.rev and not rxn.rev_par:
        file.write('  kr = kf / Kc' + utils.line_end[lang])
    elif rxn.rev_par:
        file.write(
            '  kr = '
            + rate.rxn_rate_const(rxn.rev_par[0], rxn.rev_par[1], rxn.rev_par[2])
            + utils.line_end[lang]
        )


def get_dr_dy_species(
    lang, specs, rxn, pres_rxn_ind, j_sp, sp_j, rxn_ind, rev_reacs, get_array
):
    """Returns string for evaluation of the (non-pressure dependent part) of the
    reaction rate R with respect to a species ``j``

    Parameters
    ----------
    lang : str
        The Programming language
    specs : list of `SpecInfo`
        The species in the mechanism
    rxn : `ReacInfo`
        The reaction to consider
    pres_rxn_ind : int
        The index of the reaction in the pressure dependent reactions
    j_sp : int
        The species index
    sp_j : `SpecInfo`
        The species to consider
    rxn_ind : int
        The index of the reaction in the mechanism
    rev_reacs : list of `ReacInfo`
        The list of reverisble reactions
    get_array : function
        The SMM binded get_array function (or utils.get_array) as required

    Returns
    -------
    jline : str
        Jacobian evaluation line with non-pressure-dependent part of \
        species derivative added.

    """
    jline = 'j_temp'
    last_spec = len(specs) - 1
    mw_frac = sp_j.mw / specs[last_spec].mw
    jline += f' * {1.0 - mw_frac:.16e}'
    if ((rxn.pdep and rxn.pdep_sp is None) or (rxn.thd_body)) and rxn.thd_body_eff:
        alphaij = next((thd[1] for thd in rxn.thd_body_eff if thd[0] == j_sp), 1.0)
        alphai_nspec = next(
            (thd[1] for thd in rxn.thd_body_eff if thd[0] == last_spec), 1.0
        )
        if alphai_nspec != 0:
            alphaij -= alphai_nspec * mw_frac
        if alphaij != 0:
            if alphaij != 1:
                if alphaij == -1:
                    jline += ' - pres_mod_temp'
                else:
                    jline += f' + {alphaij:.16e} * pres_mod_temp'
            else:
                jline += ' + pres_mod_temp'
    elif rxn.pdep_sp == j_sp or rxn.pdep_sp == last_spec:
        if rxn.pdep_sp == j_sp:
            jline += ' + pres_mod_temp'
        else:
            jline += f' - pres_mod_temp * {sp_j.mw / specs[rxn.pdep_sp].mw:.16e}'

    s_term = ''
    if (rxn.pdep or rxn.thd_body) and (
        (j_sp in rxn.reac or last_spec in rxn.reac)
        or (rxn.rev and (j_sp in rxn.prod or last_spec in rxn.prod))
    ):
        s_term += ' + ' + get_array(lang, 'pres_mod', pres_rxn_ind)
        s_term += ' * ('

    def __get_s_term(rxn, j_sp, reac=True):
        jline = 'kf' if reac else 'kr'
        if reac:
            nu = rxn.reac_nu[rxn.reac.index(j_sp)]
        else:
            nu = rxn.prod_nu[rxn.prod.index(j_sp)]
        if nu != 1:
            jline += f' * {float(nu)}'

        if (nu - 1) > 0:
            if utils.is_integer(nu):
                # integer, so just use multiplication
                for _ in range(int(nu) - 1):
                    if jline:
                        jline += ' * '
                    jline += get_array(lang, 'conc', j_sp)
            else:
                if jline:
                    jline += ' * '
                jline += 'pow(' + get_array(lang, 'conc', j_sp) + f', {nu - 1})'

        the_list = rxn.reac if reac else rxn.prod
        # loop through remaining reactants
        for i, isp in enumerate(the_list):
            if isp == j_sp:
                continue

            nu = rxn.reac_nu[i] if reac else rxn.prod_nu[i]
            if utils.is_integer(nu):
                # integer, so just use multiplication
                for _ in range(int(nu)):
                    if jline:
                        jline += ' * '
                    jline += get_array(lang, 'conc', isp)
            else:
                if jline:
                    jline += ' * '
                jline += 'pow(' + get_array(lang, 'conc', isp) + ', ' + str(nu) + ')'
        return jline

    j_sp_add = False
    if j_sp in rxn.reac or (rxn.rev and j_sp in rxn.prod):
        j_sp_add = True
        add = ''
        if j_sp in rxn.reac:
            if not s_term:
                add += ' + '
            add += __get_s_term(rxn, j_sp, True)
        if rxn.rev and j_sp in rxn.prod:
            if not s_term:
                add += ' - '
            elif s_term[-1] == '(':
                add += '-'
            add += __get_s_term(rxn, j_sp, False)
        s_term += add

    if last_spec in rxn.reac or (rxn.rev and last_spec in rxn.prod):
        pre = f'{mw_frac:.16e}'
        add = ''
        if j_sp_add:
            s_term += ' - '
        else:
            if s_term and s_term[-1] == '(':
                pre = '-' + pre
            else:
                pre = ' - ' + pre
        if last_spec in rxn.reac:
            add += __get_s_term(rxn, last_spec, True)
        if rxn.rev and last_spec in rxn.prod:
            if add:
                add += ' - '
            else:
                add += '-'
            add += __get_s_term(rxn, last_spec, False)
        s_term += pre + ' * (' + add + ')'

    if (rxn.pdep or rxn.thd_body) and s_term:
        s_term += ')'

    return jline + s_term


def write_kc(file, lang, specs, rxn):
    """Write evaluation of the reaction rate equilibrium constant

    Parameters
    ----------
    file : ``file`` object
        The open file object to write to
    lang : str
        The Programming language
    specs : list of `SpecInfo`
        The species in the mechanism
    rxn : `ReacInfo`
        The reaction to consider

    Returns
    -------
    None

    """
    sum_nu = 0
    coeffs = {}
    for isp in set(rxn.reac + rxn.prod):
        sp = specs[isp]
        nu = utils.get_nu(isp, rxn)

        if nu == 0:
            continue

        sum_nu += nu

        lo_array = [nu] + [
            sp.lo[6],
            sp.lo[0],
            sp.lo[0] - 1.0,
            sp.lo[1] / 2.0,
            sp.lo[2] / 6.0,
            sp.lo[3] / 12.0,
            sp.lo[4] / 20.0,
            sp.lo[5],
        ]

        lo_array = [x * lo_array[0] for x in [lo_array[1] - lo_array[2]] + lo_array[3:]]

        hi_array = [nu] + [
            sp.hi[6],
            sp.hi[0],
            sp.hi[0] - 1.0,
            sp.hi[1] / 2.0,
            sp.hi[2] / 6.0,
            sp.hi[3] / 12.0,
            sp.hi[4] / 20.0,
            sp.hi[5],
        ]

        hi_array = [x * hi_array[0] for x in [hi_array[1] - hi_array[2]] + hi_array[3:]]
        if sp.Trange[1] not in coeffs:
            coeffs[sp.Trange[1]] = lo_array, hi_array
        else:
            coeffs[sp.Trange[1]] = (
                [
                    lo_array[i] + coeffs[sp.Trange[1]][0][i]
                    for i in range(len(lo_array))
                ],
                [
                    hi_array[i] + coeffs[sp.Trange[1]][1][i]
                    for i in range(len(hi_array))
                ],
            )

    isFirst = True
    for T_mid in coeffs:
        # need temperature conditional for equilibrium constants
        line = utils.line_start + f'if (T <= {T_mid})'
        if lang in ['c', 'cuda']:
            line += ' {\n'
        elif lang == 'fortran':
            line += ' then\n'
        elif lang == 'matlab':
            line += '\n'
        file.write(line)

        lo_array, hi_array = coeffs[T_mid]

        if isFirst:
            line = utils.line_start + '  Kc = '
        else:
            if lang in ['cuda', 'c']:
                line = utils.line_start + '  Kc += '
            else:
                line = utils.line_start + '  Kc = Kc + '
        line += (
            f'({lo_array[0]:.16e} + ' + f'{lo_array[1]:.16e} * ' + 'logT + T * ('
            f'{lo_array[2]:.16e} + T * ('
            + f'{lo_array[3]:.16e} + T * ('
            + f'{lo_array[4]:.16e} + '
            + f'{lo_array[5]:.16e} * T))) - '
            + f'{lo_array[6]:.16e} / T)'
            + utils.line_end[lang]
        )
        file.write(line)

        if lang in ['c', 'cuda']:
            file.write('  } else {\n')
        elif lang in ['fortran', 'matlab']:
            file.write('  else\n')

        if isFirst:
            line = utils.line_start + '  Kc = '
        else:
            if lang in ['cuda', 'c']:
                line = utils.line_start + '  Kc += '
            else:
                line = utils.line_start + '  Kc = Kc + '
        line += (
            f'({hi_array[0]:.16e} + ' + f'{hi_array[1]:.16e} * ' + 'logT + T * ('
            f'{hi_array[2]:.16e} + T * ('
            + f'{hi_array[3]:.16e} + T * ('
            + f'{hi_array[4]:.16e} + '
            + f'{hi_array[5]:.16e} * T))) - '
            + f'{hi_array[6]:.16e} / T)'
            + utils.line_end[lang]
        )
        file.write(line)

        if lang in ['c', 'cuda']:
            file.write('  }\n\n')
        elif lang == 'fortran':
            file.write('  end if\n\n')
        elif lang == 'matlab':
            file.write('  end\n\n')
        isFirst = False

    line = utils.line_start + 'Kc = '
    if sum_nu != 0:
        num = (chem.PA / chem.RU) ** sum_nu
        line += f'{num:.16e} * '
    line += 'exp(Kc)' + utils.line_end[lang]
    file.write(line)


def get_infs(rxn):
    """Returns the reaction rate parameters for a pressure-dependent reaction

    Parameters
    ----------
    rxn : `ReacInfo`
        Reaction object for pressure-dependent reaction (falloff or
        chemically activated bimolecular)

    Returns
    -------
    beta_0minf : float
        Low-pressure limit temperature exponent minus high-pressure value.
    E_0minf : float
        Low-pressure limit activation energy minus high-pressure value.
    k0kinf : float
        Low-pressure limit reaction coefficient divided by high-pressure value.

    """
    if rxn.low:
        # unimolecular/recombination fall-off
        beta_0minf = rxn.low[1] - rxn.b
        E_0minf = rxn.low[2] - rxn.E
        k0kinf = rate.rxn_rate_const(rxn.low[0] / rxn.A, beta_0minf, E_0minf)
    elif rxn.high:
        # chem-activated bimolecular rxn
        beta_0minf = rxn.b - rxn.high[1]
        E_0minf = rxn.E - rxn.high[2]
        k0kinf = rate.rxn_rate_const(rxn.A / rxn.high[0], beta_0minf, E_0minf)
    return beta_0minf, E_0minf, k0kinf


def write_dt_comment(file, lang, rxn_ind):
    """Writes comment line for temperature partial derivatives of reactions

    Parameters
    ----------
    file : ``file`` object
        Open file object
    lang : str
        Programming language
    rxn_ind : int
        Index of reaction

    Returns
    -------
    None

    """
    line = utils.line_start + utils.comment[lang]
    line += 'partial of rxn ' + str(rxn_ind) + ' wrt T' + '\n'
    file.write(line)


def write_dy_comment(file, lang, rxn_ind):
    """Writes comment line for species mass fraction partial derivatives

    Parameters
    ----------
    file : ``file`` object
        Open file object
    lang : str
        Programming language
    rxn_ind : int
        Index of reaction

    Returns
    -------
    None

    """
    line = utils.line_start + utils.comment[lang]
    line += 'partial of rxn ' + str(rxn_ind) + ' wrt species' + '\n'
    file.write(line)


def write_dy_y_finish_comment(file, lang):
    """Writes comment line for finishing species mass fraction derivatives

    Parameters
    ----------
    file : ``file`` object
        Open file object
    lang : str
        Programming language

    Returns
    -------
    None

    """
    line = utils.line_start + utils.comment[lang]
    line += "Finish dYk / Yj's\n"
    line += utils.line_start + utils.comment[lang]
    line += "And dT/dYj's\n"
    file.write(line)


def get_rxn_params_dt(rxn, rev=False):
    """Write evaluation of the forward/reverse reaction rate constant

    Parameters
    ----------
    rxn : `ReacInfo`
        The reaction to consider
    rev : bool, optional
        If true, get the reverse constant rate constant derivative

    Returns
    -------
    jline : str
        String containing evaluation of forward/reverse reaction rate constant

    """
    jline = ''
    if rev:
        if abs(rxn.rev_par[1]) > 1.0e-90 and abs(rxn.rev_par[2]) > 1.0e-90:
            jline += f'{rxn.rev_par[1]:.16e} + ' + f'({rxn.rev_par[2]:.16e} / T)'
        elif abs(rxn.rev_par[1]) > 1.0e-90:
            jline += f'{rxn.rev_par[1]:.16e}'
        elif abs(rxn.rev_par[2]) > 1.0e-90:
            jline += f'({rxn.rev_par[2]:.16e} / T)'
    else:
        if (abs(rxn.b) > 1.0e-90) and (abs(rxn.E) > 1.0e-90):
            jline += f'{rxn.b:.16e} + ({rxn.E:.16e} / T)'
        elif abs(rxn.b) > 1.0e-90:
            jline += f'{rxn.b:.16e}'
        elif abs(rxn.E) > 1.0e-90:
            jline += f'({rxn.E:.16e} / T)'
    return jline


def write_db_dt_def(file, lang, specs, reacs, rev_reacs, dBdT_flag, do_unroll):
    """Write definition of dB/dT terms for each species

    Parameters
    ----------
    file : ``file`` object
        The open file object to write to
    lang : {'c', 'cuda'}
        The programming language
    specs : list of `SpecInfo`
        The species in the mechanism
    reacs : list of `ReacInfo`
        The reactions in the mechanism
    rev_reacs : list of `ReacInfo`
        The reversible reactions in the mechanism
    dBdT_flag : list of bool
        Upon completion of this method this list contains ``True`` for
        the index of all species with non-zero dB/dT entries
    do_unroll : bool
        If ``True``, turn on Jacobian unrolling

    Returns
    -------
    None

    """
    if len(rev_reacs):
        if lang == 'c':
            file.write(f'  double dBdT[{len(specs)}]' + utils.line_end[lang])
        else:
            file.write(
                utils.line_start
                + f'double * {utils.restrict[lang]} '
                + 'dBdT = d_mem->dBdT'
                + utils.line_end[lang]
            )
    t_mid = {}
    for i_rxn in rev_reacs:
        rxn = reacs[i_rxn]
        # only reactions with no reverse Arrhenius parameters
        if rxn.rev_par:
            continue

        # all participating species
        for sp_ind in rxn.reac + rxn.prod:
            # skip if already printed
            if dBdT_flag[sp_ind]:
                continue

            dBdT_flag[sp_ind] = True
            if specs[sp_ind].Trange[1] not in t_mid:
                t_mid[specs[sp_ind].Trange[1]] = []
            t_mid[specs[sp_ind].Trange[1]].append(sp_ind)

    for mid_temp in t_mid:
        # dB/dT evaluation (with temperature conditional)
        line = utils.line_start + f'if (T <= {mid_temp})'
        if lang in ['c', 'cuda']:
            line += ' {\n'
        elif lang == 'fortran':
            line += ' then\n'
        elif lang == 'matlab':
            line += '\n'
        file.write(line)
        for sp_ind in sorted(t_mid[mid_temp]):
            dBdT = utils.get_array(lang, 'dBdT', sp_ind)
            line = (
                utils.line_start * 2
                + dBdT
                + f' = ({specs[sp_ind].lo[0] - 1.0:.16e}'
                + f' + {specs[sp_ind].lo[5]:.16e} / T) / T'
                + f' + {specs[sp_ind].lo[1] / 2.0:.16e} + T'
                + f' * ({specs[sp_ind].lo[2] / 3.0:.16e}'
                + f' + T * ({specs[sp_ind].lo[3] / 4.0:.16e}'
                + f' + {specs[sp_ind].lo[4] / 5.0:.16e} * T))'
                + utils.line_end[lang]
            )
            file.write(line)

        if lang in ['c', 'cuda']:
            file.write('  } else {\n')
        elif lang in ['fortran', 'matlab']:
            file.write('  else\n')

        for sp_ind in sorted(t_mid[mid_temp]):
            dBdT = utils.get_array(lang, 'dBdT', sp_ind)
            line = (
                utils.line_start * 2
                + dBdT
                + f' = ({specs[sp_ind].hi[0] - 1.0:.16e}'
                + f' + {specs[sp_ind].hi[5]:.16e} / T) / T'
                + f' + {specs[sp_ind].hi[1] / 2.0:.16e} + T'
                + f' * ({specs[sp_ind].hi[2] / 3.0:.16e}'
                + f' + T * ({specs[sp_ind].hi[3] / 4.0:.16e}'
                + f' + {specs[sp_ind].hi[4] / 5.0:.16e} * T))'
                + utils.line_end[lang]
            )
            file.write(line)

        if lang in ['c', 'cuda']:
            file.write('  }\n\n')
        elif lang == 'fortran':
            file.write('  end if\n\n')
        elif lang == 'matlab':
            file.write('  end\n\n')


def get_db_dt(lang, specs, rxn, do_unroll):
    """Write evaluation of dB/dT term

    Parameters
    ----------
    lang : {'c', 'cuda'}
        The programming language
    specs : list of `SpecInfo`
        The species in the mechanism
    rxn : `ReacInfo`
        The reaction to consider
    do_unroll : bool
        If ``True``, Jacobian unrolling is turned on

    Returns
    -------
    jline : str
        String containing evaluation of dB/dT term

    """
    jline = ''
    notfirst = False
    # contribution from dBdT terms from
    # all participating species
    for sp_ind in rxn.prod:
        if sp_ind in rxn.reac:
            nu = (
                rxn.prod_nu[rxn.prod.index(sp_ind)]
                - rxn.reac_nu[rxn.reac.index(sp_ind)]
            )
        else:
            nu = rxn.prod_nu[rxn.prod.index(sp_ind)]

        if nu == 0:
            continue

        dBdT = utils.get_array(lang, 'dBdT', sp_ind)

        if not notfirst:
            # first entry
            if nu == 1:
                jline += dBdT
            elif nu == -1:
                jline += '-' + dBdT
            else:
                jline += f'{float(nu)} * ' + dBdT
        else:
            # not first entry
            if nu == 1:
                jline += ' + '
            elif nu == -1:
                jline += ' - '
            else:
                if nu > 0:
                    jline += f' + {float(nu)}'
                else:
                    jline += f' - {float(abs(nu))}'
                jline += ' * '
            jline += dBdT
        notfirst = True

    for sp_ind in rxn.reac:
        # skip species also in products, already counted
        if sp_ind in rxn.prod:
            continue

        nu = -rxn.reac_nu[rxn.reac.index(sp_ind)]

        dBdT = utils.get_array(lang, 'dBdT', sp_ind)

        # not first entry
        if nu == 1:
            jline += ' + '
        elif nu == -1:
            jline += ' - '
        else:
            if nu > 0:
                jline += f' + {float(nu)}'
            else:
                jline += f' - {float(abs(nu))}'
            jline += ' * '
        jline += dBdT

    return jline


def write_pr(file, lang, specs, reacs, pdep_reacs, rxn, get_array, last_conc_temp=None):
    """Write the evaluation of the reduced pressure

    Parameters
    ----------
    file : ``file`` object
        The open file object to write to
    lang : str
        The Programming language
    specs : list of `SpecInfo`
        The species in the mechanism
    reacs : list of `ReacInfo`
        The reactions in the mechanism
    pdep_reacs : list of `ReacInfo`
        The pressure dependent reactions (not including PLOG/Chebyshev) \
        reactions in the mechanism
    rxn : `ReacInfo`
        The reactio to consider
    get_array : function
        The SMM binded get_array function (or utils.get_array) as required
    last_conc_temp : list of tuples
        If specified, the non-unity third body efficiencies and corresponding species for the last
        pressure dependent reaction

    Returns
    -------
    conc_temp_log : list of (int, float)
        List with (species index, efficiency) for third-body efficiencies.

    """
    # print lines for necessary pressure-dependent variables
    line = utils.line_start + 'conc_temp = '
    conc_temp_log = None
    if rxn.pdep_sp is not None:
        line += get_array(lang, 'conc', rxn.pdep_sp)
    elif not rxn.thd_body_eff:
        line += 'm'
    else:
        # take care of the conc_temp collapsing
        conc_temp_log = []
        line += '(m'

        for isp, eff in rxn.thd_body_eff:
            if eff > 1.0:
                line += f' + {eff - 1.0} * '
            elif eff < 1.0:
                line += f' - {1.0 - eff} * '
            if eff != 1.0:
                line += get_array(lang, 'conc', isp)
                if conc_temp_log is not None:
                    conc_temp_log.append((isp, eff - 1.0))
        line += ')'

        if last_conc_temp is not None:
            # need to update based on the last
            new_conc_temp = []
            for species, alpha in conc_temp_log:
                match = next((sp for sp in last_conc_temp if sp[0] == species), None)
                if match is not None:
                    coeff = alpha - match[1]
                else:
                    coeff = alpha
                if coeff != 0.0:
                    new_conc_temp.append((species, coeff))
            for species, alpha in last_conc_temp:
                match = next((sp for sp in conc_temp_log if sp[0] == species), None)
                if match is None:
                    new_conc_temp.append((species, -alpha))

            use_conc = (
                new_conc_temp
                if len(new_conc_temp) < len(conc_temp_log)
                else conc_temp_log
            )
            if len(use_conc):
                # remake the line with the updated numbers
                line = utils.line_start + 'conc_temp {}= ({}'.format(
                    '+' if use_conc == new_conc_temp else '',
                    'm + ' if use_conc != new_conc_temp else '',
                )

                for i, thd_sp in enumerate(use_conc):
                    isp = thd_sp[0]
                    if i > 0:
                        line += ' {}{} * '.format(
                            '- ' if thd_sp[1] < 0 else '+ ', abs(thd_sp[1])
                        )
                    else:
                        line += f'{thd_sp[1]} * '
                    line += get_array(lang, 'conc', isp)
                line += ')'
            else:
                line = ''

    if len(line):
        file.write(line + utils.line_end[lang])

    if rxn.pdep:
        line = utils.line_start + 'Pr = conc_temp'
        beta_0minf, E_0minf, k0kinf = get_infs(rxn)
        # finish writing P_ri
        line += ' * (' + k0kinf + ')' + utils.line_end[lang]
        file.write(line)

    return conc_temp_log


def write_troe(file, lang, rxn):
    """Write the evaluation of the Troe falloff terms

    Parameters
    ----------
    file : ``file`` object
        The open file object to write to
    lang : {'c', 'cuda'}
        The programming language
    rxn : `ReacInfo`
        The reaction to consider

    Returns
    -------
    None

    """
    line = (
        '  Fcent = '
        f'{1.0 - rxn.troe_par[0]:.16e} * '
        + f'exp(T / {-rxn.troe_par[1]:.16e})'
        + f' + {rxn.troe_par[0]:.16e} * exp(T / '
        + f'{-rxn.troe_par[2]:.16e})'
    )
    if len(rxn.troe_par) == 4 and rxn.troe_par[3] != 0.0:
        line += f' + exp({-rxn.troe_par[3]:.16e} / T)'
    line += utils.line_end[lang]
    file.write(line)

    line = (
        '  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * '
        'log10(fmax(Fcent, 1.0e-300)) - 0.4' + utils.line_end[lang]
    )
    file.write(line)

    line = (
        '  B = 0.806 - 1.1762 * log10(fmax(Fcent, 1.0e-300)) - '
        '0.14 * log10(fmax(Pr, 1.0e-300))' + utils.line_end[lang]
    )
    file.write(line)

    line = (
        '  lnF_AB = 2.0 * log(fmax(Fcent, 1.0e-300)) * '
        'A / (B * B * B * (1.0 + A * A / (B * B)) * '
        '(1.0 + A * A / (B * B)))' + utils.line_end[lang]
    )
    file.write(line)


def write_sri(file, lang):
    """Write the valuation of the SRI exponent

    Parameters
    ----------
    file : ``file`` object
        The open file object to write to
    lang : str
        The programming language

    Returns
    -------
    None

    """
    line = (
        '  X = 1.0 / (1.0 + log10(fmax(Pr, 1.0e-300)) * '
        'log10(fmax(Pr, 1.0e-300)))' + utils.line_end[lang]
    )
    file.write(line)


def get_pdep_dt(lang, rxn, rev_reacs, rxn_ind, pres_rxn_ind, get_array):
    """Write contribution from temperature derivative of reaction rate for
    a pressure dependent reaction

    Parameters
    ----------
    lang : str
        The Programming language
    rxn : `ReacInfo`
        The reaction to consider
    rev_reacs : list of `ReacInfo`
        The list of reverisble reactions
    rxn_ind : int
        The index of the reaction in the reaction list
    pres_rxn_ind : int
        The index of the reaction in the pressure dependent reaction list
    get_array : function
        The SMM binded get_array function (or utils.get_array) as required

    Returns
    -------
    None

    """
    beta_0minf, E_0minf, k0kinf = get_infs(rxn)
    jline = utils.line_start + 'j_temp = (' + get_array(lang, 'pres_mod', pres_rxn_ind)
    # high -> chem-activated bimolecular rxn
    jline += ' * ((' + ('-Pr * ' if rxn.high else '')

    # dPr/dT
    jline += (
        f'({beta_0minf:.4e} + ('
        + f'{E_0minf:.16e} / T) - 1.0) / '
        + '(T * (1.0 + Pr)))'
    )

    if rxn.sri:
        jline += get_sri_dt(lang, rxn, beta_0minf, E_0minf, k0kinf)
    elif rxn.troe:
        jline += get_troe_dt(lang, rxn, beta_0minf, E_0minf, k0kinf)

    jline += ') * '

    if rxn.rev:
        # forward and reverse reaction rates
        jline += '(' + get_array(lang, 'fwd_rates', rxn_ind)
        jline += ' - ' + get_array(lang, 'rev_rates', rev_reacs.index(rxn_ind))
        jline += ')'
    else:
        # forward reaction rate only
        jline += '' + get_array(lang, 'fwd_rates', rxn_ind)

    jline += ' + (' + get_array(lang, 'pres_mod', pres_rxn_ind)

    return jline


def get_sri_dt(lang, rxn, beta_0minf, E_0minf, k0kinf):
    """Returns section of line for temperature partial derivative of SRI falloff.

    Parameters
    ----------
    lang : str
        Programming language, {'c', 'cuda'}
    rxn : `ReacInfo`
        Reaction of interest; pressure dependence expressed with SRI falloff
    beta_0minf : float

    E_0minf : float

    k0kinf : float

    Returns
    -------
    jline : str
        Line fragment with SRI temperature derivative

    """
    jline = (
        ' + X * ((('
        f'{rxn.sri_par[0] * rxn.sri_par[1]:.16} / ' + '(T * T)) * exp('
        f'{-rxn.sri_par[1]:.16} / T) - '
        + f'{1.0 / rxn.sri_par[2]:.16e} * '
        + f'exp(T / {-rxn.sri_par[2]:.16})) / '
        + f'({rxn.sri_par[0]:.16} * '
        + f'exp({-rxn.sri_par[1]:.16} / T) + '
        + f'exp(T / {-rxn.sri_par[2]:.16})) - '
        + f'X * {2.0 / math.log(10.0):.16} * '
        + 'log10(fmax(Pr, 1.0e-300)) * ('
        f'{beta_0minf:.16e} + ('
        + f'{E_0minf:.16e} / T) - 1.0) * '
        + f'log({rxn.sri_par[0]:.16} * exp('
        + f'{-rxn.sri_par[1]:.16} / T) + '
        + 'exp(T / '
        f'{-rxn.sri_par[2]:.16})) / T)'
    )

    if len(rxn.sri_par) == 5 and rxn.sri_par[4] != 0.0:
        jline += f' + ({rxn.sri_par[4]:.16} / T)'

    return jline


def get_troe_dt(lang, rxn, beta_0minf, E_0minf, k0kinf):
    """Returns section of line for temperature partial derivative of Troe falloff.

    Parameters
    ----------
    lang : str
        Programming language, {'c', 'cuda'}
    rxn : `ReacInfo`
        Reaction of interest; pressure dependence expressed with Troe falloff.
    beta_0minf : float
        Low-pressure limit temperature exponent minus high-pressure value.
    E_0minf : float
        Low-pressure limit activation energy minus high-pressure value.
    k0kinf : float
        Low-pressure limit reaction coefficient divided by high-pressure value.

    Returns
    -------
    jline : str
        Line fragment with Troe temperature derivative

    """
    jline = (
        ' + (((1.0 / '
        '(Fcent * (1.0 + A * A / (B * B)))) - '
        'lnF_AB * ('
        f'-{0.67 / math.log(10.0):.16e}' + ' * B + '
        f'{1.1762 / math.log(10.0):.16e} * '
        + f'A) / Fcent) * ({-(1.0 - rxn.troe_par[0]) / rxn.troe_par[1]:.16e}'
        + ' * exp(T / '
        f'{-rxn.troe_par[1]:.16e}) - '
        + f'{rxn.troe_par[0] / rxn.troe_par[2]:.16e} * '
        + 'exp(T / '
        f'{-rxn.troe_par[2]:.16e})'
    )
    if len(rxn.troe_par) == 4 and rxn.troe_par[3] != 0.0:
        jline += (
            f' + ({rxn.troe_par[3]:.16e} / ' + '(T * T)) * exp('
            f'{-rxn.troe_par[3]:.16e} / T)'
        )
    jline += '))'

    jline += (
        ' - lnF_AB * ('
        f'{1.0 / math.log(10.0):.16e}' + ' * B + '
        f'{0.14 / math.log(10.0):.16e}' + ' * A) * '
        f'({beta_0minf:.16e} + (' + f'{E_0minf:.16e} / T) - 1.0) / T'
    )

    return jline


def write_dcp_dt(file, lang, specs):
    """Write derivative of cp w.r.t. temperature for each species

    Parameters
    ----------
    file : ``file`` object
        The open file object to write to
    lang : str
        The Programming language
    specs : list of `SpecInfo`
        The species in the mechanism

    Returns
    -------
    None

    """
    T_mid_buckets = {}
    # put all of the same T_mids together
    for isp, sp in enumerate(specs):
        if sp.Trange[1] not in T_mid_buckets:
            T_mid_buckets[sp.Trange[1]] = []
        T_mid_buckets[sp.Trange[1]].append(isp)

    first = True
    for T_mid in sorted(T_mid_buckets):
        # write the if statement
        line = utils.line_start + f'if (T <= {T_mid})'
        if lang in ['c', 'cuda']:
            line += ' {\n'
        elif lang == 'fortran':
            line += ' then\n'
        elif lang == 'matlab':
            line += '\n'
        file.write(line)
        # and the update line
        line = utils.line_start + '  working_temp'
        if lang in ['c', 'cuda']:
            line += ' {}= '.format('+' if not first else '')
        elif lang in ['fortran', 'matlab']:
            line += ' = {}'.format('working_temp + ' if not first else '')
        for i, isp in enumerate(sorted(T_mid_buckets[T_mid])):
            sp = specs[isp]
            if i:
                line += '\n    + '

            y_str = (
                utils.get_array(lang, 'y', isp + 1) if isp + 1 != len(specs) else 'y_N'
            )
            line += '(' + y_str
            line += (
                f' * {chem.RU / sp.mw:.16e} * ('
                + f'{sp.lo[1]:.16e} + '
                + f'T * ({2.0 * sp.lo[2]:.16e} + '
                + f'T * ({3.0 * sp.lo[3]:.16e} + '
                + f'{4.0 * sp.lo[4]:.16e} * T)))'
                + ')'
            )
        line += utils.line_end[lang]
        file.write(line)

        # now do the high temperature side
        if lang in ['c', 'cuda']:
            file.write('  } else {\n')
        elif lang in ['fortran', 'matlab']:
            file.write('  else\n')
        # and the update line
        line = utils.line_start + '  working_temp'
        if lang in ['c', 'cuda']:
            line += ' {}= '.format('+' if not first else '')
        elif lang in ['fortran', 'matlab']:
            line += ' = {}'.format('working_temp + ' if not first else '')

        for i, isp in enumerate(sorted(T_mid_buckets[T_mid])):
            sp = specs[isp]
            if i:
                line += '\n    + '

            y_str = (
                utils.get_array(lang, 'y', isp + 1) if isp + 1 != len(specs) else 'y_N'
            )
            line += '(' + y_str
            line += (
                f' * {chem.RU / sp.mw:.16e} * ('
                + f'{sp.hi[1]:.16e} + '
                + f'T * ({2.0 * sp.hi[2]:.16e} + '
                + f'T * ({3.0 * sp.hi[3]:.16e} + '
                + f'{4.0 * sp.hi[4]:.16e} * T)))'
                + ')'
            )
        line += utils.line_end[lang]
        file.write(line)
        # and finish the if
        if lang in ['c', 'cuda']:
            file.write('  }\n\n')
        elif lang == 'fortran':
            file.write('  end if\n\n')
        elif lang == 'matlab':
            file.write('  end\n\n')

        first = False


def get_elementary_rxn_dt(lang, specs, rxn, rxn_ind, rev_idx, get_array, do_unroll):
    """Write contribution from temperature derivative of reaction rate for
    elementary reaction.

    Parameters
    ----------
    lang : str
        The programming language
    rxn : `ReacInfo`
        The reaction to consider
    rxn_ind : int
        The reaction index
    rev_idx : int
        The index of the reaction in the reverse reaction list (if applicable)
    get_array : function
        The SMM binded get_array function (or `utils.get_array`) as required
    do_unroll : bool
        If true, Jacobian unrolling is turned on

    Returns
    -------
    jline : str
        Jacobian entry substring with temperature derivative contribution.

    """

    jline = ''
    if rxn.rev and rxn.rev_par:
        dk_dt = get_rxn_params_dt(rxn, rev=False)
        nu = sum(rxn.reac_nu)

        if dk_dt or nu != 1.0:
            # we actually need to do the dk/dt for both
            jline = get_array(lang, 'fwd_rates', rxn_ind)
            jline += ' * ('
            if dk_dt:
                jline += dk_dt

            # loop over reactants
            if nu != 1.0:
                if dk_dt and jline:
                    jline += ' + '
                jline += f'{1.0 - float(nu)}'
            jline += ')'

        dk_dt = get_rxn_params_dt(rxn, rev=True)
        nu = sum(rxn.prod_nu)
        if dk_dt or nu != 1.0:
            jline += ' - ' + get_array(lang, 'rev_rates', rev_idx) + ' * ('

            if dk_dt:
                jline += dk_dt

            # product nu sum
            if nu != 1.0:
                if dk_dt and jline:
                    jline += ' + '
                jline += f'{1.0 - float(nu)}'
            jline += ')'
    elif rxn.rev:
        # we don't need the dk/dt for both,
        # so write different to not calculate twice, and instead
        # rely on loading fwd/rev rates again, as they should
        # be cached

        dk_dt = get_rxn_params_dt(rxn, rev=False)
        if dk_dt:
            jline += '('
            jline += get_array(lang, 'fwd_rates', rxn_ind)
            if rxn.rev:
                jline += ' - ' + get_array(lang, 'rev_rates', rev_idx)
            jline += ')'
            jline += ' * ('

            jline += dk_dt
            jline += ')'

        # loop over reactants
        nu = sum(rxn.reac_nu)
        if nu != 1.0:
            if jline:
                jline += ' + '
            jline += get_array(lang, 'fwd_rates', rxn_ind)
            jline += f' * {1.0 - float(nu)}'

        dbdt = get_db_dt(lang, specs, rxn, do_unroll)
        nu = sum(rxn.prod_nu)
        if dbdt or nu != 1.0:
            if jline:
                jline += ' - '
            else:
                jline += '-'
            jline += get_array(lang, 'rev_rates', rev_idx)
            jline += ' * ('
            # product nu sum
            nu = sum(rxn.prod_nu)
            if nu != 1.0:
                jline += f'{1.0 - float(nu)} + '
            if dbdt:
                jline += '-T * ('

                # product nu sum
                jline += dbdt
                jline += '))'
    else:
        # forward only, combine dk/dt and nu sum
        dk_dt = get_rxn_params_dt(rxn, rev=False)
        nu = sum(rxn.reac_nu)
        if dk_dt or nu != 1.0:
            jline += get_array(lang, 'fwd_rates', rxn_ind)
            jline += ' * ('
            jline += dk_dt

            # loop over reactants
            nu = sum(rxn.reac_nu)
            if nu != 1.0:
                if jline:
                    jline += ' + '
                jline += f'{1.0 - float(nu)}'

            jline += ')'

    # print line for reaction
    if jline:
        jline += ')) * rho_inv' + utils.line_end[lang]
    return jline


def write_cheb_ut(file, lang, rxn):
    """
    Computes the derivative of the chebyshev polynomial recursively w.r.t T

    Parameters
    ----------
    file : ``file`` object
        The open file object to write to
    lang : str
        The Programming language
    rxn : `ReacInfo`
        The reaction to consider

    Returns
    -------
    None

    """
    line_list = []
    line_list.append('cheb_temp_0 = 1')
    line_list.append('cheb_temp_1 = Pred')
    # start pressure dot product
    for i in range(1, rxn.cheb_n_temp):
        line_list.append(
            utils.get_array(lang, 'dot_prod', i)
            + f'= {i * rxn.cheb_par[i, 0]:.16e} + Pred * {i * rxn.cheb_par[i, 1]:.16e}'
        )

    # finish pressure dot product
    update_one = True
    for j in range(2, rxn.cheb_n_pres):
        if update_one:
            new = 1
            old = 0
        else:
            new = 0
            old = 1
        line = f'cheb_temp_{old}'
        line += f' = 2 * Pred * cheb_temp_{new}'
        line += f' - cheb_temp_{old}'
        line_list.append(line)
        for i in range(1, rxn.cheb_n_temp):
            line_list.append(
                utils.get_array(lang, 'dot_prod', i)
                + f' += {i * rxn.cheb_par[i, j]:.16e} * cheb_temp_{old}'
            )

        update_one = not update_one

    line_list.append('cheb_temp_0 = 1.0')
    line_list.append('cheb_temp_1 = 2.0 * Tred')
    # finally, do the temperature portion
    # d/dTred of the Chebyshev series is a series in Chebyshev polynomials of
    # the second kind, with U_0 = 1 and U_1 = 2 * Tred. dot_prod is only filled
    # for indices 1 .. cheb_n_temp - 1, so emit only the terms that exist.
    terms = []
    if rxn.cheb_n_temp > 1:
        terms.append(utils.get_array(lang, 'dot_prod', 1))
    if rxn.cheb_n_temp > 2:
        terms.append('2.0 * Tred * ' + utils.get_array(lang, 'dot_prod', 2))
    line_list.append('kf = ' + (' + '.join(terms) if terms else '0.0'))

    update_one = True
    for i in range(3, rxn.cheb_n_temp):
        if update_one:
            new = 1
            old = 0
        else:
            new = 0
            old = 1
        line = f'cheb_temp_{old}'
        line += f' = 2.0 * Tred * cheb_temp_{new}'
        line += f' - cheb_temp_{old}'
        line_list.append(line)
        line_list.append(
            'kf += ' + utils.get_array(lang, 'dot_prod', i) + ' * ' + f'cheb_temp_{old}'
        )

        update_one = not update_one

    line_list = [utils.line_start + line + utils.line_end[lang] for line in line_list]
    file.write(''.join(line_list))


def write_cheb_rxn_dt(
    file, lang, jline, rxn, rxn_ind, rev_idx, specs, get_array, do_unroll
):
    """
    Writes the code for the temperature derivative of Chebyshev reactions

    Parameters
    ----------

    file : ``file`` object
        The open file object to write to
    lang : str
        The Programming language
    jline : str
        The current jacobian line (containing the non-Chebyshev part of the derivative)
    specs : list of `SpecInfo`
        The species in the mechanism
    rxn : `ReacInfo`
        The reaction to consider
    rxn_ind : int
        The reaction index
    rev_idx : int
        The index of the reaction in the reverse reaction list (if applicable)
    get_array : function
        The SMM binded get_array function (or utils.get_array) as required
    do_unroll : bool
        If true, Jacobian unrolling is turned on
    """
    # Chebyshev reaction
    tlim_inv_sum = 1.0 / rxn.cheb_tlim[0] + 1.0 / rxn.cheb_tlim[1]
    tlim_inv_sub = 1.0 / rxn.cheb_tlim[1] - 1.0 / rxn.cheb_tlim[0]
    file.write(
        utils.line_start
        + 'Tred = ((2.0 / T) - '
        + f'{tlim_inv_sum:.16e}) / {tlim_inv_sub:.16e}'
        + utils.line_end[lang]
    )

    plim_log_sum = math.log10(rxn.cheb_plim[0]) + math.log10(rxn.cheb_plim[1])
    plim_log_sub = math.log10(rxn.cheb_plim[1]) - math.log10(rxn.cheb_plim[0])
    file.write(
        utils.line_start
        + 'Pred = (2.0 * log10(pres) - '
        + f'{plim_log_sum:.16e}) / {plim_log_sub:.16e}'
        + utils.line_end[lang]
    )

    # do U(T) sum
    write_cheb_ut(file, lang, rxn)

    jline += f'kf * ({-2.0 * math.log(10) / tlim_inv_sub:.16e} / T)'

    jline += ' * (' + get_array(lang, 'fwd_rates', rxn_ind)

    if rxn.rev:
        # reverse reaction rate also
        jline += ' - ' + get_array(lang, 'rev_rates', rev_idx)

    jline += ')'
    nu = sum(rxn.reac_nu)
    if nu != 1.0:
        jline += ' + ' + get_array(lang, 'fwd_rates', rxn_ind)
        jline += f' * {1.0 - float(nu)}'

    if rxn.rev:
        jline += ' - ' + get_array(lang, 'rev_rates', rev_idx) + ' * ('
        nu = sum(rxn.prod_nu)
        if nu != 1.0:
            jline += f'{1.0 - float(nu)} + '
        jline += '-T * (' + get_db_dt(lang, specs, rxn, do_unroll)
        jline += '))'

    jline += ')) * rho_inv'
    # print line for reaction
    file.write(jline + utils.line_end[lang])


def write_plog_rxn_dt(
    file, lang, jline, specs, rxn, rxn_ind, rev_idx, get_array, do_unroll
):
    """
    Writes the code for the temperature derivative of PLog reactions

    Parameters
    ----------

    file : ``file`` object
        The open file object to write to
    lang : str
        The Programming language
    jline : str
        The current jacobian line (containing the non-PLog part of the derivative)
    specs : list of `SpecInfo`
        The species in the mechanism
    rxn : `ReacInfo`
        The reaction to consider
    rxn_ind : int
        The reaction index
    rev_idx : int
        The index of the reaction in the reverse reaction list (if applicable)
    get_array : function
        The SMM binded get_array function (or utils.get_array) as required
    do_unroll : bool
        If true, Jacobian unrolling is turned on
    """
    # Plog reactions have conditional contribution,
    # depends on pressure range

    (p1, A_p1, b_p1, E_p1) = rxn.plog_par[0]

    # For pressure below the first pressure given, use standard
    # Arrhenius expression.

    # Make copy, but with specific pressure Arrhenius coefficients
    rxn_p = chem.ReacInfo(
        rxn.rev, rxn.reac, rxn.reac_nu, rxn.prod, rxn.prod_nu, A_p1, b_p1, E_p1
    )

    dkdt = get_elementary_rxn_dt(
        lang, specs, rxn_p, rxn_ind, rev_idx, get_array, do_unroll
    )
    have_prev = False
    if dkdt:
        file.write(utils.line_start + f'if (pres <= {p1:.4e}) {{\n')
        file.write(utils.line_start + jline + dkdt)
        have_prev = True

    for idx, vals in enumerate(rxn.plog_par[:-1]):
        (p1, A_p1, b_p1, E_p1) = vals
        (p2, A_p2, b_p2, E_p2) = rxn.plog_par[idx + 1]

        jline_p = ''
        if A_p2 / A_p1 < 0:
            # MIT mechanisms occasionally have (for some unknown reason)
            # negative A's, so we need to handle the
            # log(K2) - log(K1) term differently
            raise NotImplementedError
        else:
            assert b_p1 != 0.0 or E_p1 != 0.0 or b_p2 != 0.0 or E_p2 != 0.0, (
                'PLOG Derivative undefined'
            )
            if b_p1 != 0.0:
                jline_p += f'{b_p1:.16e}'
            if E_p1 != 0.0:
                if jline_p:
                    jline_p += ' + '
                jline_p += f'{E_p1:.16e} / T'
            if b_p2 - b_p1 != 0.0 or E_p2 - E_p1 != 0.0:
                if jline_p:
                    jline_p += ' + '

                jline_p += '('
                if b_p2 - b_p1 != 0.0:
                    jline_p += f'{b_p2 - b_p1:.16e} + '
                if E_p2 - E_p1 != 0.0:
                    jline_p += f'{E_p2 - E_p1:.16e} / T) * (log(pres)'
                    if p1 != 1.0:
                        jline_p += f' - {math.log(p1):.16e}'
                    jline_p += ') / '
                    assert p1 != p2, 'Cannot have equal pressures in PLOG'
                    jline_p += f'{math.log(p2) - math.log(p1):.16e})'
            else:
                jline_p += ')'

            jline_p += ' * (' + get_array(lang, 'fwd_rates', rxn_ind)
            if rxn.rev:
                # reverse reaction rate also
                jline_p += ' - ' + get_array(lang, 'rev_rates', rev_idx)
            jline_p += ')'

        nu = sum(rxn.reac_nu)
        if nu != 1.0:
            if jline_p:
                jline_p += ' + '
            jline_p += get_array(lang, 'fwd_rates', rxn_ind) + f' * {1.0 - nu}'

        if rxn.rev:
            nu = sum(rxn.prod_nu)
            dbdt = get_db_dt(lang, specs, rxn, do_unroll)
            if nu != 1.0 or dbdt:
                jline_p += (
                    '{}'.format(' - ' if jline_p else '-')
                    + get_array(lang, 'rev_rates', rev_idx)
                    + ' * ('
                )
                if nu != 1.0:
                    jline_p += f'{1.0 - nu} + '
                dbdt = get_db_dt(lang, specs, rxn, do_unroll)
                if dbdt:
                    jline_p += '-T * (' + get_db_dt(lang, specs, rxn, do_unroll) + ')'
                jline_p += ')'

        if jline_p:
            jline_p = jline + '(' + jline_p + ')) * rho_inv'
        else:
            jline_p = jline + '0.0e0)'

        if have_prev:
            file.write(
                utils.line_start
                + f'}} else if ((pres > {p1:.4e}) '
                + f'&& (pres <= {p2:.4e})) {{\n'
            )
        else:
            file.write(
                utils.line_start
                + f'if ((pres > {p1:.4e}) '
                + f'&& (pres <= {p2:.4e})) {{\n'
            )
        have_prev = True
        # print line for reaction
        file.write(utils.line_start + jline_p + utils.line_end[lang])

    (pn, A_pn, b_pn, E_pn) = rxn.plog_par[-1]

    # For pressure above the final pressure given, use standard
    # Arrhenius expression.

    # Make copy, but with specific pressure Arrhenius coefficients
    rxn_p = chem.ReacInfo(
        rxn.rev, rxn.reac, rxn.reac_nu, rxn.prod, rxn.prod_nu, A_pn, b_pn, E_pn
    )
    dkdt = get_elementary_rxn_dt(
        lang, specs, rxn_p, rxn_ind, rev_idx, get_array, do_unroll
    )
    if dkdt:
        if have_prev:
            file.write(utils.line_start + f'}} else if (pres > {pn:.4e}) {{\n')
        else:
            file.write(utils.line_start + 'j_temp = 0' + utils.line_end[lang])
            file.write(utils.line_start + f'if (pres > {pn:.4e}) {{\n')
        file.write(utils.line_start + jline + dkdt)

        file.write(utils.line_start + '}\n')


def write_dt_completion(file, lang, specs, J_nplusone_touched, get_array):
    r"""Finishes calculation of d(\partial T / \partial t)/dT

    Parameters
    ----------
    file : ``file`` object
        Open file object to write to
    lang : str
        The Programming language
    specs : list of `SpecInfo`
        The species in this mechanism
    J_nplusone_touched : bool
        If true, the last species has a non-zero contribution to d(\partial T / \partial t)
    get_array : function
        The SMM binded get_array function (or utils.get_array) as required
    """

    line = utils.line_start + utils.comment[lang]
    line += 'Complete dT wrt T calculations\n'
    file.write(line)
    line = utils.line_start
    if lang in ['c', 'cuda']:
        line += get_array(lang, 'jac', 0)
    elif lang in ['fortran', 'matlab']:
        line += get_array(lang, 'jac', 0, twod=0)

    line += ' = -('

    for k_sp, sp_k in enumerate(specs):
        if k_sp:
            line += utils.line_start + '  + '
        line += get_array(lang, 'spec_rates', k_sp) + f' * {sp_k.mw:.8e}' + ' * '
        line += (
            '(-working_temp * '
            + get_array(lang, 'h', k_sp)
            + ' / cp_avg + '
            + ''
            + get_array(lang, 'cp', k_sp)
            + ')'
        )
        if k_sp + 1 == len(specs):
            j_str = 'J_nplusone'
        else:
            j_str = get_array(lang, 'jac', k_sp + 1, twod=0)
        if k_sp + 1 < len(specs) or J_nplusone_touched:
            line += ' + ' + j_str + ' * ' + get_array(lang, 'h', k_sp) + ' * rho'
        if k_sp != len(specs) - 1:
            if lang == 'fortran':
                line += ' &'
            line += '\n'

    line += ') / (rho * cp_avg)'
    line += utils.line_end[lang]
    file.write(line)


def write_sub_intro(
    path,
    lang,
    number,
    rate_list,
    this_rev,
    this_pdep,
    have_pres_mod_temp,
    batch_has_m,
    this_thd,
    this_troe,
    this_sri,
    this_cheb,
    cheb_dim,
    this_plog,
    no_shared,
    has_nsp,
):
    """
    Writes the header and definitions for the Jacobian reaction update subfiles

    Parameters
    ----------
    path : str
        Path to build directory for file.
    lang : str {'c', 'cuda'}
        Programming language
    number : int
        Jacobian subfile number
    rate_list : str
        The required reaction/species rate strings needed for the intro
    this_rev : bool
        If ``True``, batch contains a reversible reaction
    this_pdep : bool
        If ``True``, batch contains pressure dependent (falloff/bimolecular) reaction
    have_pres_mod_temp : bool
        If ``True``, batch requires definition of ``pres_mod_temp``
    batch_has_m : bool
        If ``True``, batch requires the overall concentration 'm'
    this_thd : bool
        If ``True``, batch has a third-body reaction
    this_troe : bool
        If ``True``, batch contains a Troe reaction
    this_sri : bool
        If ``True``, batch contains an SRI reaction
    this_cheb : bool
        If ``True``, batch contains a Chebyshev reaction
    cheb_dim : int
        If ``this_cheb`` is ``True``, the largest Chebyshev dimension required
    this_plog : bool
        If ``True``, batch contains a PLOG reaction
    no_shared : bool
        If ``True``, do not use CUDA shared memory
    has_nsp : bool
        If ``True``, >=1 reaction has nonzero contribution from the last species

    Returns
    -------
    file : ``file`` object
        Opened Jacobian file

    """
    with open(
        os.path.join(path, 'jacob_' + str(number) + utils.header_ext[lang]), 'w'
    ) as file:
        file.write(
            f'#ifndef JACOB_HEAD_{number}\n' + f'#define JACOB_HEAD_{number}\n' + '\n'
            f'#include "header{utils.header_ext[lang]}"\n'
            + '\n'
            + ('__device__ ' if lang == 'cuda' else '')
            + ''
            f'void eval_jacob_{number} ('
        )
        line = 'const double, const double * {0}'
        for _ in rate_list:
            line += ', const double * {0}'
        if batch_has_m:
            line += ', const double'
        line += (
            ', const double, const double'
            + ('' if not this_rev else ', const double * {0}')
            + ', const double, double * {0}'
            + (', double * {0}, double* {0}' if has_nsp else '')
            + (', double * {0}' if this_cheb and lang == 'cuda' else '')
            + ');\n'
            '\n'
            '#endif\n'
        )
        file.write(line.format(utils.restrict[lang]))
    file = open(os.path.join(path, 'jacob_' + str(number) + utils.file_ext[lang]), 'w')
    file.write(f'#include <math.h>\n#include "header{utils.header_ext[lang]}"\n' + '\n')

    line = '__device__ ' if lang == 'cuda' else ''

    line += f'void eval_jacob_{number} (const double pres, ' + 'const double * {0} conc'
    for rate_name in rate_list:
        line += ', const double * {0} ' + rate_name
    if batch_has_m:
        line += ', const double m'
    line += ', const double mw_avg, const double rho'
    if this_rev:
        line += ', const double * {0} dBdT'
    line += ', const double T, double * {0} jac'

    if has_nsp:
        line += ', double * {0} J_nplusone, double * {0} J_nplusjplus'
    if this_cheb and lang == 'cuda':
        line += ', double * {0} dot_prod'
    line += ') {{'
    file.write(line.format(utils.restrict[lang]) + '\n')

    if not no_shared and lang == 'cuda':
        file.write(
            utils.line_start
            + 'extern volatile __shared__ double shared_temp[]'
            + utils.line_end[lang]
        )
        # third-body variable needed for reactions
    if this_pdep:
        line = utils.line_start
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'conc_temp' + utils.line_end[lang]
        file.write(line)

    # log(T)
    line = utils.line_start
    if lang == 'c':
        line += 'double '
    elif lang == 'cuda':
        line += 'register double '
    line += 'logT = log(T)' + utils.line_end[lang]
    file.write(line)

    line = utils.line_start
    if lang == 'c':
        line += 'double '
    elif lang == 'cuda':
        line += 'register double '
    line += 'kf = 0.0' + utils.line_end[lang]
    file.write(line)

    line = utils.line_start
    if lang == 'c':
        line += 'double '
    elif lang == 'cuda':
        line += 'register double '
    line += 'j_temp = 0.0' + utils.line_end[lang]
    file.write(line)

    if have_pres_mod_temp:
        line = utils.line_start
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'pres_mod_temp = 0.0' + utils.line_end[lang]
        file.write(line)

    # if any reverse reactions, will need Kc
    if this_rev:
        line = utils.line_start
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        file.write(line + 'Kc = 0.0' + utils.line_end[lang])
        file.write(line + 'kr = 0.0' + utils.line_end[lang])

    # pressure-dependence variables
    if this_pdep:
        line = utils.line_start
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'Pr = 0.0' + utils.line_end[lang]
        file.write(line)

    if this_troe:
        line = ''.join(
            [
                utils.line_start + f'double {x} = 0.0{utils.line_end[lang]}'
                for x in ['Fcent', 'A', 'B', 'lnF_AB']
            ]
        )
        file.write(line)

    if this_sri:
        line = utils.line_start + 'double X = 0.0' + utils.line_end[lang]
        file.write(line)

    if this_cheb:
        file.write(utils.line_start + 'double Tred, Pred' + utils.line_end[lang])
        file.write(
            utils.line_start + 'double cheb_temp_0, cheb_temp_1' + utils.line_end[lang]
        )
        if lang == 'c':
            file.write(
                utils.line_start + f'double dot_prod[{cheb_dim}]' + utils.line_end[lang]
            )

    if this_plog:
        file.write(utils.line_start + 'double kf2' + utils.line_end[lang])

    file.write(utils.line_start + 'double rho_inv = 1.0 / rho' + utils.line_end[lang])

    return file


def write_dy_intros(path, lang, number, have_jnplus_jplus):
    """
    Writes the header and definitions for the various Jacobian species update subfiles

    Parameters
    ----------
    path : str
        The path to place the file in
    lang : str {'c', 'cuda'}
        The programming language
    number : int
        The jacobian subfile index
    have_jnplus_jplus : bool
        If ``True``, the last species has non-zero contributions to the Jacobian

    Returns
    -------
    file : ``file`` object
        Jacobian file object

    """
    with open(
        os.path.join(path, 'jacob_' + str(number) + utils.header_ext[lang]), 'w'
    ) as file:
        file.write(
            f'#ifndef JACOB_HEAD_{number}\n' + f'#define JACOB_HEAD_{number}\n' + '\n'
            f'#include "header{utils.header_ext[lang]}"\n'
            + '\n'
            + ('__device__ ' if lang == 'cuda' else '')
            + f'void eval_jacob_{number} ('
        )
        file.write(
            'const double, const double, const double, const double*, '
            'const double*, const double*, double*'
            + (', double*' if have_jnplus_jplus else '')
            + ');\n'
            '\n'
            '#endif\n'
        )
    file = open(os.path.join(path, 'jacob_' + str(number) + utils.file_ext[lang]), 'w')
    file.write(f'#include "header{utils.header_ext[lang]}"\n' + '\n')

    line = '__device__ ' if lang == 'cuda' else ''

    line += (
        f'void eval_jacob_{number} ' + '(const double mw_avg, const double rho, '
        'const double cp_avg, const double* spec_rates, '
        'const double* h, const double* cp, double* jac'
        + (', double* J_nplusjplus' if have_jnplus_jplus else '')
        + ') '
    )
    line += '{\n'
    line += utils.line_start
    if lang == 'cuda':
        line += 'register '
    line += 'double rho_inv = 1.0 / rho'
    file.write(line + utils.line_end[lang])

    file.write(
        utils.line_start + 'double working_temp = (1.0 / cp_avg)' + utils.line_end[lang]
    )
    file.write(
        utils.line_start + 'double j_temp = 1.0 / '
        '(rho * cp_avg * cp_avg)' + utils.line_end[lang]
    )

    return file


def write_jacobian(path, lang, specs, reacs, seen_sp, smm=None):
    """Write Jacobian subroutine in desired language.

    Parameters
    ----------
    path : str
        Path to build directory for file.
    lang : str {'c', 'cuda', 'fortran', 'matlab'}
        Programming language.
    specs : list of SpecInfo
        List of species in the mechanism.
    reacs : list of ReacInfo
        List of reactions in the mechanism.
    seen_sp : list of bool
        List of `bool`; ``False`` if species has (identically) zero rate
    smm : shared_memory_manager, optional
        If not ``None``, use this to manage shared memory optimization

    Returns
    -------
    None

    """

    if lang == 'cuda':
        do_unroll = len(reacs) > CUDAParams.Jacob_Unroll
        unroll_len = CUDAParams.Jacob_Unroll
        limit = CUDAParams.Max_Lines
    elif lang == 'c':
        do_unroll = len(reacs) > CParams.Jacob_Unroll
        unroll_len = CParams.Jacob_Unroll
        limit = CParams.Max_Lines
    if do_unroll:
        # make paths for separate jacobian files
        utils.create_dir(os.path.join(path, 'jacobs'))

    # first write header file
    file = open(os.path.join(path, 'jacob' + utils.header_ext[lang]), 'w')
    file.write(
        '#ifndef JACOB_HEAD\n'
        '#define JACOB_HEAD\n'
        '\n'
        f'#include "header{utils.header_ext[lang]}"\n'
        + (
            f'#include "jacobs/jac_include{utils.header_ext[lang]}"\n'
            if do_unroll
            else ''
        )
        + f'#include "chem_utils{utils.header_ext[lang]}"\n'
        f'#include "rates{utils.header_ext[lang]}"\n'
    )
    if lang == 'cuda':
        file.write('#include "gpu_memory.cuh"\n\n__device__ ')
    file.write(
        'void eval_jacob (const double, const double, '
        'const double * {0}, double * {0}{1});\n'
        '\n'
        '#endif\n'.format(
            utils.restrict[lang],
            f', const mechanism_memory * {utils.restrict[lang]}'
            if lang == 'cuda'
            else '',
        )
    )
    file.close()

    # numbers of species and reactions
    num_s = len(specs)
    num_r = len(reacs)
    rev_reacs = [i for i, rxn in enumerate(reacs) if rxn.rev]
    num_rev = len(rev_reacs)

    pdep_reacs = []
    for i, reac in enumerate(reacs):
        if reac.thd_body or reac.pdep:
            # add reaction index to list
            pdep_reacs.append(i)
    num_pdep = len(pdep_reacs)

    # create file depending on language
    filename = 'jacob' + utils.file_ext[lang]
    file = open(os.path.join(path, filename), 'w')

    # header files
    file.write(f'#include "jacob{utils.header_ext[lang]}"\n\n')

    line = ''
    if lang == 'cuda':
        line = '__device__ '

    if lang in ['c', 'cuda']:
        line += (
            'void eval_jacob (const double t, const double pres, '
            'const double * {0} y, double * {0} jac{1}) {{\n\n'.format(
                utils.restrict[lang],
                f', const mechanism_memory * {utils.restrict[lang]} d_mem'
                if lang == 'cuda'
                else '',
            )
        )
    elif lang == 'fortran':
        line += 'subroutine eval_jacob (t, pres, y, jac)\n\n'

        # fortran needs type declarations
        line += (
            '  implicit none\n'
            '  integer, parameter :: wp = kind(1.0d0)'
            f'  real(wp), intent(in) :: t, pres, y({num_s})\n'
            + f'  real(wp), intent(out) :: jac({num_s},{num_s})\n'
            + '  \n'
            '  real(wp) :: T, rho, cp_avg, logT\n'
        )
        if any(reacs[rxn].thd_body for rxn in rev_reacs):
            line += '  real(wp) :: m\n'
        line += (
            f'  real(wp), dimension({num_s}) :: ' + 'conc, cp, h, dy\n'
            f'  real(wp), dimension({num_r}) :: rxn_rates\n'
            + f'  real(wp), dimension({num_pdep}) :: pres_mod\n'
        )
    elif lang == 'matlab':
        line += 'function jac = eval_jacob (T, pres, y)\n\n'
    file.write(line)

    get_array = utils.get_array
    if lang == 'cuda' and smm is not None:
        smm.reset()
        get_array = smm.get_array
        smm.write_init(file, indent=2)

    # get temperature
    if lang in ['c', 'cuda']:
        line = utils.line_start + 'double T = ' + get_array(lang, 'y', 0)
    elif lang in ['fortran', 'matlab']:
        line = utils.line_start + 'T = ' + get_array(lang, 'y', 0)
    line += utils.line_end[lang]
    file.write(line)

    file.write('\n')

    file.write(utils.line_start + utils.comment[lang] + ' average molecular weight\n')
    # calculation of average molecular weight
    if lang in ['c', 'cuda']:
        file.write(utils.line_start + 'double mw_avg;\n')

    file.write(utils.line_start + utils.comment[lang] + ' mass-averaged density\n')
    if lang in ['c', 'cuda']:
        file.write(utils.line_start + 'double rho;\n')

    # evaluate species molar concentrations
    file.write(
        utils.line_start + utils.comment[lang] + ' species molar concentrations\n'
    )
    if lang == 'c':
        file.write(utils.line_start + f'double conc[{num_s}];\n')
    elif lang == 'cuda':
        file.write(
            utils.line_start
            + f'double * {utils.restrict[lang]}'
            + ' conc = d_mem->conc'
            + utils.line_end[lang]
        )
    elif lang == 'matlab':
        file.write(utils.line_start + f'conc = zeros({num_s},1);\n')
    file.write(utils.line_start + 'double y_N' + utils.line_end[lang])
    file.write(
        utils.line_start
        + 'eval_conc('
        + utils.get_array(lang, 'y', 0)
        + ', pres, &'
        + (utils.get_array(lang, 'y', 1) if lang != 'cuda' else 'y[GRID_DIM]')
        + ', &y_N, &mw_avg, &rho, conc)'
        + utils.line_end[lang]
        + '\n'
    )

    rate_list = ['fwd_rates']
    if len(rev_reacs):
        rate_list.append('rev_rates')
    if len(pdep_reacs):
        rate_list.append('pres_mod')
    rate_list.append('spec_rates')
    file.write(utils.line_start + utils.comment[lang] + ' evaluate reaction rates\n')

    cuda_cheb = any(rxn.cheb for rxn in reacs) and lang == 'cuda'
    # evaluate forward and reverse reaction rates
    if lang in ['c', 'cuda']:
        if lang == 'cuda':
            file.write(
                utils.line_start
                + f'double * {utils.restrict[lang]} fwd_rates = d_mem->fwd_rates'
                + utils.line_end[lang]
            )
        else:
            file.write(utils.line_start + f'double fwd_rates[{num_r}];\n')
        if num_rev == 0:
            file.write(utils.line_start + 'double* rev_rates = 0;\n')
        elif lang == 'cuda':
            file.write(
                utils.line_start
                + f'double * {utils.restrict[lang]} rev_rates = d_mem->rev_rates'
                + utils.line_end[lang]
            )
        else:
            file.write(utils.line_start + f'double rev_rates[{num_rev}];\n')
        if cuda_cheb:
            file.write(
                f'  double * {utils.restrict[lang]} dot_prod'
                + ' = d_mem->dot_prod'
                + utils.line_end[lang]
            )

        file.write(
            utils.line_start
            + 'eval_rxn_rates (T, pres, conc, fwd_rates, rev_rates{});\n'.format(
                ', dot_prod' if cuda_cheb else ''
            )
        )
    elif lang == 'fortran':
        file.write(
            utils.line_start + 'call eval_rxn_rates (T, pres, conc, fwd_rates, '
            'rev_rates)\n'
        )
    elif lang == 'matlab':
        file.write(
            utils.line_start + '[fwd_rates, rev_rates] = eval_rxn_rates '
            '(T, pres, conc);\n'
        )
    file.write('\n')

    if num_pdep == 0:
        file.write(utils.line_start + 'double* pres_mod = 0;\n')
    elif lang == 'c':
        file.write(utils.line_start + f'double pres_mod[{num_pdep}];\n')
    else:
        file.write(
            utils.line_start
            + f'double * {utils.restrict[lang]} pres_mod = d_mem->pres_mod{utils.line_end[lang]}'
        )

    if len(pdep_reacs):
        file.write(
            utils.line_start
            + utils.comment[lang]
            + 'get pressure modifications to reaction rates\n'
        )
        # evaluate third-body and pressure-dependence reaction modifications
        if lang in ['c', 'cuda']:
            file.write(
                utils.line_start + 'get_rxn_pres_mod (T, pres, conc, pres_mod);\n'
            )
        elif lang == 'fortran':
            file.write(
                utils.line_start + 'call get_rxn_pres_mod (T, pres, conc, pres_mod)\n'
            )
        elif lang == 'matlab':
            file.write(
                utils.line_start + 'pres_mod = get_rxn_pres_mod (T, pres, conc, '
                'pres_mod);\n'
            )
    file.write('\n')

    # evaluate species rates
    file.write(
        utils.line_start
        + utils.comment[lang]
        + ' evaluate rate of change of species molar concentration\n'
    )
    if lang == 'c':
        file.write(utils.line_start + f'double spec_rates[{num_s}] = {{0}};\n')
        file.write(
            utils.line_start + 'eval_spec_rates (fwd_rates, rev_rates, '
            f'pres_mod, spec_rates, &spec_rates[{num_s - 1}]);\n'
        )
    elif lang == 'cuda':
        file.write(
            utils.line_start
            + f'double * {utils.restrict[lang]} spec_rates = d_mem->spec_rates{utils.line_end[lang]}'
        )
        file.write(
            utils.line_start
            + 'eval_spec_rates (fwd_rates, rev_rates, '
            'pres_mod, spec_rates, &{}){}'.format(
                utils.get_array(lang, 'spec_rates', num_s - 1), utils.line_end[lang]
            )
        )
    elif lang == 'fortran':
        file.write(
            utils.line_start + 'call eval_spec_rates (fwd_rates, rev_rates, '
            f'pres_mod, spec_rates, spec_rates({num_s - 1}))\n'
        )
    elif lang == 'matlab':
        file.write(
            utils.line_start + 'spec_rates = eval_spec_rates(fwd_rates, '
            'rev_rates, pres_mod);\n'
        )
    file.write('\n')

    # third-body variable needed for reactions
    if any((rxn.pdep and rxn.pdep_sp is None) or rxn.thd_body for rxn in reacs):
        line = utils.line_start
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += f'm = pres / ({chem.RU:.8e} * T)' + utils.line_end[lang]
        file.write(line)

        if not do_unroll:
            line = utils.line_start
            if lang == 'c':
                line += 'double '
            elif lang == 'cuda':
                line += 'register double '
            line += 'conc_temp' + utils.line_end[lang]
            file.write(line)

    # log(T)
    line = utils.line_start
    if lang == 'c':
        line += 'double '
    elif lang == 'cuda':
        line += 'register double '
    line += 'logT = log(T)' + utils.line_end[lang]
    file.write(line)

    if not do_unroll:
        line = utils.line_start
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'j_temp = 0.0' + utils.line_end[lang]
        file.write(line)

        line = utils.line_start
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'kf = 0.0' + utils.line_end[lang]
        file.write(line)

        if any(
            (rxn.pdep or rxn.thd_body) and (rxn.thd_body_eff or rxn.pdep_sp)
            for rxn in reacs
        ):
            line = utils.line_start
            if lang == 'c':
                line += 'double '
            elif lang == 'cuda':
                line += 'register double '
            line += 'pres_mod_temp = 0.0' + utils.line_end[lang]
            file.write(line)

        # if any reverse reactions, will need Kc
        if rev_reacs:
            line = utils.line_start
            if lang == 'c':
                line += 'double '
            elif lang == 'cuda':
                line += 'register double '
            file.write(line + 'Kc = 0.0' + utils.line_end[lang])
            file.write(line + 'kr = 0' + utils.line_end[lang])

        # pressure-dependence variables
        if any(rxn.pdep for rxn in reacs):
            line = utils.line_start
            if lang == 'c':
                line += 'double '
            elif lang == 'cuda':
                line += 'register double '
            line += 'Pr = 0.0' + utils.line_end[lang]
            file.write(line)

        if any(rxn.troe for rxn in reacs):
            line = ''.join(
                [
                    f'  double {x} = 0.0{utils.line_end[lang]}'
                    for x in ['Fcent', 'A', 'B', 'lnF_AB']
                ]
            )
            file.write(line)

        if any(rxn.sri for rxn in reacs):
            line = utils.line_start + 'double X = 0.0' + utils.line_end[lang]
            file.write(line)

        if any(rxn.cheb for rxn in reacs):
            file.write(utils.line_start + 'double Tred, Pred' + utils.line_end[lang])
            file.write(
                utils.line_start
                + 'double cheb_temp_0, cheb_temp_1'
                + utils.line_end[lang]
            )
            dim = max(rxn.cheb_n_temp for rxn in reacs if rxn.cheb)
            file.write(
                utils.line_start
                + (
                    f'double dot_prod[{dim}]'
                    if lang == 'c'
                    else f'double * {utils.restrict[lang]} dot_prod = d_mem->dot_prod'
                )
                + utils.line_end[lang]
            )

        if any(rxn.plog for rxn in reacs):
            file.write(utils.line_start + 'double kf2' + utils.line_end[lang])

        line = utils.line_start
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'rho_inv = 1.0 / rho' + utils.line_end[lang]
        file.write(line)

    if any(
        len(specs) - 1 in set(reac.reac + reac.prod)
        and utils.get_nu(len(specs) - 1, reac)
        for reac in reacs
    ):
        file.write(utils.line_start + 'double J_nplusone = 0' + utils.line_end[lang])
        if lang == 'c':
            file.write(
                utils.line_start + 'double J_nplusjplus[NSP]' + utils.line_end[lang]
            )
        else:
            file.write(
                utils.line_start
                + f'double * {utils.restrict[lang]} J_nplusjplus = d_mem->J_nplusjplus'
                + utils.line_end[lang]
            )

    # variables for equilibrium constant derivatives, if needed
    dBdT_flag = [False for sp in specs]

    # define dB/dT's
    write_db_dt_def(file, lang, specs, reacs, rev_reacs, dBdT_flag, do_unroll)

    line = ''

    ###################################
    # now begin Jacobian evaluation
    ###################################
    ###################################
    # partial derivatives of reactions
    ###################################
    success = False
    retry = False
    while not success:
        if lang == 'cuda' and smm is not None:
            smm.reset()
        # whether this jacobian index has been modified
        touched = [False for i in range(len(specs) * len(specs))]
        J_nplusone_touched = False
        J_nplusjplus_touched = [False for i in range(len(specs))]

        last_conc_temp = None

        jac_count = 0
        next_fn_index = 0
        for rxn_ind, rxn in enumerate(reacs):
            if do_unroll and (rxn_ind == next_fn_index):
                # clear conc temp
                last_conc_temp = None
                if not retry:
                    file_store = file
                retry = False
                # get next index
                next_fn_index = min(rxn_ind + unroll_len, len(reacs))
                # get flags
                rev = False
                pdep = False
                thd = False
                troe = False
                sri = False
                cheb = False
                plog = False
                has_jnplus_one = False
                batch_has_m = False
                have_pres_mod_temp = False
                for ind_next in range(rxn_ind, next_fn_index):
                    if reacs[ind_next].rev:
                        rev = True
                    if reacs[ind_next].pdep:
                        pdep = True
                        if reacs[ind_next].pdep_sp is None:
                            batch_has_m = True
                    if (reacs[ind_next].pdep or reacs[ind_next].thd_body) and (
                        reacs[ind_next].thd_body_eff or reacs[ind_next].pdep_sp
                    ):
                        have_pres_mod_temp = True
                    if reacs[ind_next].thd_body:
                        thd = True
                    if reacs[ind_next].troe:
                        troe = True
                    if reacs[ind_next].sri:
                        sri = True
                    if reacs[ind_next].cheb:
                        cheb = True
                    if reacs[ind_next].plog:
                        plog = True
                    reac = reacs[ind_next]
                    if len(specs) - 1 in set(reac.reac + reac.prod) and utils.get_nu(
                        len(specs) - 1, reac
                    ):
                        has_jnplus_one = True

                dim = None
                if cheb:
                    dim = max(rxn.cheb_n_temp for rxn in reacs if rxn.cheb)
                # write the specific evaluator for this reaction
                file = write_sub_intro(
                    os.path.join(path, 'jacobs'),
                    lang,
                    jac_count,
                    rate_list,
                    rev,
                    pdep,
                    have_pres_mod_temp,
                    batch_has_m,
                    thd,
                    troe,
                    sri,
                    cheb,
                    dim,
                    plog,
                    smm is None,
                    has_jnplus_one,
                )

            if lang == 'cuda' and smm is not None:
                variable_list, usages = calculate_shared_memory(
                    rxn_ind, rxn, specs, reacs, rev_reacs, pdep_reacs
                )
                smm.load_into_shared(file, variable_list, usages)

            ######################################
            # with respect to temperature
            ######################################

            write_dt_comment(file, lang, rxn_ind)

            # first we need any pres mod terms
            jline = ''
            pres_rxn_ind = None
            if rxn.pdep:
                pres_rxn_ind = pdep_reacs.index(rxn_ind)
                last_conc_temp = write_pr(
                    file, lang, specs, reacs, pdep_reacs, rxn, get_array, last_conc_temp
                )

                # dF/dT
                if rxn.troe:
                    write_troe(file, lang, rxn)
                elif rxn.sri:
                    write_sri(file, lang)

                jline = get_pdep_dt(
                    lang, rxn, rev_reacs, rxn_ind, pres_rxn_ind, get_array
                )

            elif rxn.thd_body:
                # third body reaction
                pres_rxn_ind = pdep_reacs.index(rxn_ind)

                jline = (
                    utils.line_start
                    + 'j_temp = ((-'
                    + get_array(lang, 'pres_mod', pres_rxn_ind)
                    + ' * '
                )

                if rxn.rev:
                    # forward and reverse reaction rates
                    jline += '(' + get_array(lang, 'fwd_rates', rxn_ind)
                    jline += ' - ' + get_array(
                        lang, 'rev_rates', rev_reacs.index(rxn_ind)
                    )
                    jline += ')'
                else:
                    # forward reaction rate only
                    jline += get_array(lang, 'fwd_rates', rxn_ind)

                jline += ' / T) + (' + get_array(lang, 'pres_mod', pres_rxn_ind)

            else:
                if lang in ['c', 'cuda', 'matlab']:
                    jline += '  j_temp = ((1.0'
                elif lang in ['fortran']:
                    jline += '  j_temp = ((1.0_wp'

            jline += ' / T) * ('

            doT = True
            if rxn.plog:
                write_plog_rxn_dt(
                    file,
                    lang,
                    jline,
                    specs,
                    rxn,
                    rxn_ind,
                    rev_reacs.index(rxn_ind) if rxn.rev else None,
                    get_array,
                    do_unroll,
                )

            elif rxn.cheb:
                write_cheb_rxn_dt(
                    file,
                    lang,
                    jline,
                    rxn,
                    rxn_ind,
                    rev_reacs.index(rxn_ind) if rxn.rev else None,
                    specs,
                    get_array,
                    do_unroll,
                )

            else:
                dkdt = get_elementary_rxn_dt(
                    lang,
                    specs,
                    rxn,
                    rxn_ind,
                    rev_reacs.index(rxn_ind) if rxn.rev else None,
                    get_array,
                    do_unroll,
                )
                if dkdt:
                    file.write(jline + dkdt)
                else:
                    doT = False

            if doT:
                for k_sp in set(rxn.reac + rxn.prod):
                    sp_k = specs[k_sp]
                    line = utils.line_start
                    nu = utils.get_nu(k_sp, rxn)
                    if nu == 0:
                        continue
                    if lang in ['c', 'cuda']:
                        j_str = (
                            '{}J_nplusone'.format('*' if do_unroll else '')
                            if k_sp + 1 == num_s
                            else get_array(lang, 'jac', k_sp + 1)
                        )
                        line += j_str + ' {}= {}j_temp{} * {:.16e}'.format(
                            '+' if touched[k_sp + 1] else '',
                            '' if nu == 1 else ('-' if nu == -1 else ''),
                            f' * {float(nu)}' if nu != 1 and nu != -1 else '',
                            sp_k.mw,
                        )
                    elif lang in ['fortran', 'matlab']:
                        # NOTE: I believe there was a bug here w/ the previous
                        # fortran/matlab code (as it looks like it would be zero
                        # indexed)
                        j_str = (
                            'J_nplusone'
                            if k_sp + 1 == num_s
                            else get_array(lang, 'jac', k_sp + 1, twod=0)
                        )
                        line += (
                            j_str
                            + ' = '
                            + (j_str + ' + ' if touched[k_sp + 1] else '')
                            + ' {}j_temp{} * {:.16e}'.format(
                                '' if nu == 1 else ('-' if nu == -1 else ''),
                                f' * {float(nu)}' if nu != 1 and nu != -1 else '',
                                sp_k.mw,
                            )
                        )
                    file.write(line + utils.line_end[lang])
                    if k_sp + 1 == num_s:
                        J_nplusone_touched = True
                    else:
                        touched[k_sp + 1] = True

                file.write('\n')

            ######################################
            # with respect to species
            ######################################
            write_dy_comment(file, lang, rxn_ind)

            if rxn.rev and not rxn.rev_par:
                # need to find Kc
                write_kc(file, lang, specs, rxn)

            # need to write the dr/dy parts (independent of any species)
            write_dr_dy(file, lang, rev_reacs, rxn, rxn_ind, pres_rxn_ind, get_array)

            # write the forward / backwards rates:
            write_rates(file, lang, rxn)

            # now loop through each species
            for j_sp, sp_j in enumerate(specs[:-1]):
                dr_dyj = get_dr_dy_species(
                    lang,
                    specs,
                    rxn,
                    pres_rxn_ind,
                    j_sp,
                    sp_j,
                    rxn_ind,
                    rev_reacs,
                    get_array,
                )
                for k_sp in set(rxn.reac + rxn.prod):
                    sp_k = specs[k_sp]

                    nu = utils.get_nu(k_sp, rxn)

                    if nu == 0:
                        continue

                    jline = utils.line_start
                    if k_sp + 1 < num_s:
                        lin_index = k_sp + 1 + (num_s) * (j_sp + 1)
                        # if not rxn_ind in thelist and lin_index == 30608:
                        #    thelist.add(rxn_ind)
                        # sparse indexes
                        if lang in ['c', 'cuda']:
                            jline += get_array(lang, 'jac', lin_index) + ' {}= '.format(
                                '+' if touched[lin_index] else ''
                            )
                        elif lang in ['fortran', 'matlab']:
                            jline += (
                                get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1)
                                + (
                                    ' = '
                                    + get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1)
                                    if touched[k_sp + 1]
                                    else ''
                                )
                                + ' + '
                            )

                        touched[lin_index] = True
                    else:
                        if lang in ['c', 'cuda']:
                            jline += get_array(
                                lang, 'J_nplusjplus', j_sp
                            ) + ' {}= '.format(
                                '+' if J_nplusjplus_touched[j_sp] else ''
                            )
                        elif lang in ['fortran', 'matlab']:
                            jline += (
                                get_array(lang, 'J_nplusjplus', j_sp)
                                + (
                                    ' = ' + get_array(lang, 'J_nplusjplus', j_sp)
                                    if J_nplusjplus_touched[j_sp]
                                    else ''
                                )
                                + ' + '
                            )

                        J_nplusjplus_touched[j_sp] = True

                    working_temp = ''
                    mw_frac = (sp_k.mw / sp_j.mw) * float(nu)
                    if mw_frac == -1.0:
                        working_temp += ' -'
                    elif mw_frac != 1.0:
                        working_temp += f' {mw_frac:.16e} * '
                    else:
                        working_temp += ' '

                    working_temp += '('

                    working_temp += dr_dyj

                    working_temp += ')'

                    jline += working_temp
                    jline += utils.line_end[lang]

                    file.write(jline)
                    jline = ''

            file.write('\n')

            if lang == 'cuda' and smm is not None:
                evictable = [x for x in variable_list if not x.base == 'conc']
                smm.mark_for_eviction(evictable)

            if do_unroll and (
                rxn_ind == next_fn_index - 1 or rxn_ind == len(reacs) - 1
            ):
                # switch back
                file.write('}\n\n')
                file.close()
                file = file_store
                # test file size for CUDA
                # to avoid killing nvcc
                if jac_count == 0:
                    with open(
                        os.path.join(
                            path, 'jacobs', f'jacob_{jac_count}{utils.file_ext[lang]}'
                        )
                    ) as readfile:
                        num_lines = sum(1 for line in readfile)
                    if num_lines > limit:
                        unroll_len = int(unroll_len / 2)
                        retry = True
                        break

                file.write(f'  eval_jacob_{jac_count}(')
                jac_count += 1
                line = 'pres, conc'
                for rate in rate_list:
                    line += ', ' + rate
                if batch_has_m:
                    line += ', m'
                line += ', mw_avg, rho'
                if rev:
                    line += ', dBdT'
                line += ', T, jac'
                if has_jnplus_one:
                    line += ', &J_nplusone, J_nplusjplus'
                if cheb and lang == 'cuda':
                    line += ', dot_prod'
                line += ')'
                file.write(line + utils.line_end[lang])
        success = rxn_ind == len(reacs) - 1

    ###################################
    # Partial derivatives of temperature (energy equation)
    ###################################

    # evaluate enthalpy
    if lang == 'c':
        file.write(
            f'  // species enthalpies\n  double h[{num_s}];\n' + '  eval_h(T, h);\n'
        )
    elif lang == 'cuda':
        file.write(
            '  // species enthalpies\n'
            f'  double * {utils.restrict[lang]} h = d_mem->h;\n' + '  eval_h(T, h);\n'
        )
    elif lang == 'fortran':
        file.write('  ! species enthalpies\n  call eval_h(T, h)\n')
    elif lang == 'matlab':
        file.write('  % species enthalpies\n  h = eval_h(T);\n')
    file.write('\n')

    # evaluate specific heat
    if lang == 'c':
        file.write(
            '  // species specific heats\n'
            f'  double cp[{num_s}];\n' + '  eval_cp(T, cp);\n'
        )
    elif lang == 'cuda':
        file.write(
            '  // species specific heats\n'
            f'  double * {utils.restrict[lang]} cp = d_mem->cp;\n'
            + '  eval_cp(T, cp);\n'
        )
    elif lang == 'fortran':
        file.write('  ! species specific heats\n  call eval_cp(T, cp)\n')
    elif lang == 'matlab':
        file.write('  % species specific heats\n  cp = eval_cp(T);\n')
    file.write('\n')

    # average specific heat
    if lang == 'c':
        file.write('  // average specific heat\n  double cp_avg;\n')
    elif lang == 'cuda':
        file.write('  // average specific heat\n  register double cp_avg;\n')
    elif lang == 'fortran':
        file.write('  ! average specific heat\n')
    elif lang == 'matlab':
        file.write('  % average specific heat\n')

    line = utils.line_start + 'cp_avg = '
    for sp in specs[:-1]:
        if len(line) > 70:
            if lang in ['c', 'cuda']:
                line += '\n'
            elif lang == 'fortran':
                line += ' &\n'
            elif lang == 'matlab':
                line += ' ...\n'
            file.write(line)
            line = utils.line_start + '   '

        isp = specs.index(sp)
        line += (
            '('
            + get_array(lang, 'y', isp + 1)
            + ' * '
            + get_array(lang, 'cp', isp)
            + ')'
            ' + '
        )

    line += '(' + 'y_N' + ' * ' + get_array(lang, 'cp', len(specs) - 1) + ')'
    line += utils.line_end[lang]
    file.write(line)

    # set jac[0] = 0
    # set to zero
    line = utils.line_start
    if lang in ['c', 'cuda']:
        line += get_array(lang, 'jac', 0) + ' = 0.0'
    elif lang == 'fortran':
        line += get_array(lang, 'jac', 0, twod=0) + ' = 0.0_wp'
    elif lang == 'matlab':
        line += get_array(lang, 'jac', 0, twod=0) + ' = 0.0'
    touched[0] = True
    line += utils.line_end[lang]
    file.write(line)

    if not do_unroll:
        line = utils.line_start
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'working_temp = (1.0 / cp_avg)' + utils.line_end[lang]
        file.write(line)

        file.write(
            utils.line_start
            + 'j_temp = 1.0 / (rho * cp_avg * cp_avg)'
            + utils.line_end[lang]
        )
    else:
        file.write(utils.line_start + 'double working_temp = 0' + utils.line_end[lang])

    # need to finish the dYk/dYj's
    write_dy_y_finish_comment(file, lang)
    unroll_len = (
        CParams.Jacob_Spec_Unroll if lang == 'c' else CUDAParams.Jacob_Spec_Unroll
    )
    limit = CParams.Max_Spec_Lines if lang == 'c' else CUDAParams.Max_Spec_Lines

    touched_copy = touched[:]
    success = False
    while not success:
        touched = touched_copy[:]
        next_fn_index = 0
        for k_sp, sp_k in enumerate(specs):
            if do_unroll and k_sp == next_fn_index:
                next_fn_index += min(unroll_len, len(specs) - k_sp)

                have_jnplus_jplus = next_fn_index >= len(specs) and any(
                    J_nplusjplus_touched
                )

                file = write_dy_intros(
                    os.path.join(path, 'jacobs'), lang, jac_count, have_jnplus_jplus
                )

            for j_sp, sp_j in enumerate(specs):
                lin_index = k_sp + 1 + (num_s) * (j_sp + 1)
                # the num_s + 1 row is zero
                # so skip
                if j_sp + 1 == num_s:
                    continue

                if k_sp + 1 < num_s and touched[lin_index]:
                    # still in the actual jacobian
                    # and this combo matters
                    line = utils.line_start
                    # need to finish
                    if lang in ['c', 'cuda']:
                        line += get_array(lang, 'jac', lin_index) + ' += '
                    elif lang in ['fortran', 'matlab']:
                        line += (
                            get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1)
                            + ' = '
                            + get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1)
                            + ' + '
                        )

                    line += (
                        '('
                        + get_array(lang, 'spec_rates', k_sp)
                        + f' * mw_avg * {(sp_k.mw / sp_j.mw) * (1.0 - sp_j.mw / specs[-1].mw):.16e}'
                        + ' * rho_inv)'
                        + utils.line_end[lang]
                    )
                    file.write(line)
                elif k_sp + 1 == num_s and J_nplusjplus_touched[j_sp]:
                    # out of bounds in the Jnplusjplus ones
                    # and this combo matters
                    line = utils.line_start
                    # need to finish
                    if lang in ['c', 'cuda']:
                        line += get_array(lang, 'J_nplusjplus', j_sp) + ' += '
                    elif lang in ['fortran', 'matlab']:
                        line += (
                            get_array(lang, 'J_nplusjplus', j_sp)
                            + ' = '
                            + get_array(lang, 'jac', j_sp)
                            + ' + '
                        )

                    line += (
                        '('
                        + get_array(lang, 'spec_rates', k_sp)
                        + f' * mw_avg * {(sp_k.mw / sp_j.mw) * (1.0 - sp_j.mw / specs[-1].mw):.16e}'
                        + ' * rho_inv)'
                        + utils.line_end[lang]
                    )
                    file.write(line)

                ######################################
                # Derivative with respect to species
                ######################################
                line = utils.line_start
                my_index = (num_s) * (j_sp + 1)
                if lang in ['c', 'cuda']:
                    line += get_array(lang, 'jac', my_index)
                elif lang in ['fortran', 'matlab']:
                    line += get_array(lang, 'jac', 0, twod=j_sp + 1)
                if lang in ['fortran', 'matlab']:
                    line += (
                        ' = '
                        + (
                            get_array(lang, 'jac', 0, twod=j_sp + 1) + ' +'
                            if touched[my_index]
                            else ''
                        )
                        + ' -('
                    )
                else:
                    line += ' {}= {}('.format(
                        '-' if touched[my_index] else '',
                        '' if touched[my_index] else '-',
                    )
                touched[my_index] = True

                jac_part = ''
                if k_sp + 1 < num_s:
                    # still in the actual jacobian
                    if touched[lin_index]:
                        if lang in ['c', 'cuda']:
                            jac_part = (
                                'working_temp * '
                                + get_array(lang, 'jac', lin_index)
                                + ' - '
                            )
                        if lang in ['fortran', 'matlab']:
                            jac_part = (
                                'working_temp * '
                                + get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1)
                                + ' - '
                            )
                    else:
                        jac_part = '-'
                else:
                    # out of bounds, so need to check the
                    # Jnplusjplus ones
                    if J_nplusjplus_touched[j_sp]:
                        if lang in ['c', 'cuda']:
                            jac_part = (
                                'working_temp * '
                                + get_array(lang, 'J_nplusjplus', j_sp)
                                + ' - '
                            )
                        if lang in ['fortran', 'matlab']:
                            jac_part = (
                                'working_temp * '
                                + get_array(lang, 'J_nplusjplus', j_sp + 1)
                                + ' - '
                            )
                    else:
                        jac_part = '-'

                sp_part = (
                    '(j_temp * ('
                    + get_array(lang, 'cp', j_sp)
                    + ' - '
                    + get_array(lang, 'cp', num_s - 1)
                    + ')'
                    + ' * '
                    + get_array(lang, 'spec_rates', k_sp)
                    + f' * {sp_k.mw:.8e}))'
                )

                line += (
                    get_array(lang, 'h', k_sp)
                    + ' * ('
                    + jac_part
                    + sp_part
                    + ')'
                    + utils.line_end[lang]
                )
                if jac_part != '-' or seen_sp[k_sp]:
                    file.write(line)

            if do_unroll and k_sp == next_fn_index - 1:
                # switch back
                file.write('}\n\n')
                file = file_store
                # check that file length is under limit
                with open(
                    os.path.join(
                        path, 'jacobs', f'jacob_{jac_count}{utils.file_ext[lang]}'
                    )
                ) as readfile:
                    num_lines = sum(1 for line in readfile)
                if num_lines > limit:
                    unroll_len = int(unroll_len / 2)
                    break
                file.write(f'  eval_jacob_{jac_count}(')
                jac_count += 1
                line = 'mw_avg, rho, cp_avg, spec_rates, h, cp, jac'
                if have_jnplus_jplus:
                    line += ', J_nplusjplus'
                line += ')'
                file.write(line + utils.line_end[lang])
        success = k_sp == len(specs) - 1

    ######################################
    # Derivatives with respect to temperature
    ######################################
    write_dcp_dt(file, lang, specs)

    ######################################
    # Derivative with respect to species
    ######################################
    file.write('\n')

    # finish the dT entry
    write_dt_completion(file, lang, specs, J_nplusone_touched, get_array)

    if lang in ['c', 'cuda']:
        file.write('} // end eval_jacob\n\n')
    elif lang == 'fortran':
        file.write('end subroutine eval_jacob\n\n')
    elif lang == 'matlab':
        file.write('end\n\n')

    file.close()

    # create include file
    if do_unroll:
        with open(
            os.path.join(path, 'jacobs', 'jac_include' + utils.header_ext[lang]), 'w'
        ) as tempfile:
            tempfile.write('#ifndef JAC_INCLUDE_H\n#define JAC_INCLUDE_H\n')
            for i in range(jac_count):
                tempfile.write(f'#include "jacob_{i}{utils.header_ext[lang]}"\n')
            tempfile.write('#endif\n\n')

        with open(os.path.join(path, 'jacobs', f'jac_list_{lang}'), 'w') as tempfile:
            tempfile.write(
                ' '.join([f'jacob_{i}{utils.file_ext[lang]}' for i in range(jac_count)])
            )
    return touched


def write_sparse_multiplier(path, lang, touched, nvars):
    """Write a subroutine that multiplies the non-zero entries of the
    Jacobian with a column 'j' of another matrix.

    Parameters
    ----------
    path : str
        Path to build directory for file.
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Programming language.
    inidicies : list
        A list of indicies where the Jacobian is non-zero
    nvars : int
        Number of variables in the Jacobian matrix

    Returns
    -------
    None

    """

    sparse_indicies = [x for x in range(nvars * nvars) if touched[nvars]]

    # first write header file
    file = open(os.path.join(path, f'sparse_multiplier{utils.header_ext[lang]}'), 'w')
    file.write('#ifndef SPARSE_HEAD\n#define SPARSE_HEAD\n')
    file.write(f'\n#define N_A {len(sparse_indicies)}')
    file.write(
        '\n'
        f'#include "header{utils.header_ext[lang]}"\n'
        + '\n'
        + ('__device__\n' if lang == 'cuda' else '')
        + 'void sparse_multiplier (const double *, const double *, double*);\n'
        '\n'
        '#ifdef COMPILE_TESTING_METHODS\n'
        '  int test_sparse_multiplier();\n'
        '#endif\n'
        '\n'
        '#endif\n'
    )
    file.close()

    # create file depending on language
    filename = 'sparse_multiplier' + utils.file_ext[lang]
    file = open(os.path.join(path, filename), 'w')

    file.write(f'#include "sparse_multiplier{utils.header_ext[lang]}"\n\n')

    if lang == 'cuda':
        file.write('__device__\n')

    file.write(
        'void sparse_multiplier(const double * A, const double * Vm, double* w) {\n'
    )

    if lang == 'cuda':
        """optimize for cache reusing"""
        touched = [False for i in range(nvars)]
        for i in range(nvars):
            # get all indicies that belong to row i
            i_list = [x for x in sparse_indicies if int(x / nvars) == i]
            for index in i_list:
                file.write(
                    ' '
                    + utils.get_array(lang, 'w', index % nvars)
                    + ' {}= '.format('+' if touched[index % nvars] else '')
                )
                file.write(
                    ' '
                    + utils.get_array(lang, 'A', index)
                    + ' * '
                    + utils.get_array(lang, 'Vm', i)
                    + utils.line_end[lang]
                )
                touched[index % nvars] = True
        zero_out = [i for i, t in enumerate(touched) if not t]
        for i in zero_out:
            file.write(
                ' ' + utils.get_array(lang, 'w', i) + ' = 0' + utils.line_end[lang]
            )
        file.write('}\n')
    else:
        for i in range(nvars):
            # get all indicies that belong to row i
            i_list = [x for x in sparse_indicies if x % nvars == i]
            if not len(i_list):
                file.write(
                    '  ' + utils.get_array(lang, 'w', i) + ' = 0' + utils.line_end[lang]
                )
                continue
            file.write('  ' + utils.get_array(lang, 'w', i) + ' = ')
            for index in i_list:
                if i_list.index(index):
                    file.write(' + ')
                file.write(
                    ' '
                    + utils.get_array(lang, 'A', index)
                    + ' * '
                    + utils.get_array(lang, 'Vm', int(index / nvars))
                )
            file.write(';\n')
        file.write('}\n')

    file.close()


def create_jacobian(
    lang,
    mech_name=None,
    therm_name=None,
    gas=None,
    optimize_cache=False,
    initial_state='',
    num_blocks=8,
    num_threads=64,
    no_shared=False,
    L1_preferred=True,
    multi_thread=None,
    force_optimize=False,
    build_path='./out/',
    last_spec=None,
    skip_jac=False,
    auto_diff=False,
):
    """Create Jacobian subroutine from mechanism.

    Parameters
    ----------
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Language type.
    mech_name : str, optional
        Reaction mechanism filename (e.g. 'mech.dat').
        This or gas must be specified
    therm_name : str, optional
        Thermodynamic database filename (e.g. 'therm.dat')
        or nothing if info in mechanism file.
    gas : cantera.Solution, optional
        The mechanism to generate the Jacobian for.  This or ``mech_name`` must be specified
    optimize_cache : bool, optional
        If ``True``, use the greedy optimizer to attempt to improve cache hit rates
    initial_state : str, optional
        A comma separated list of the initial conditions to use in form
        T,P,X (e.g. '800,1,H2=1.0,O2=0.5'). Temperature in K, P in atm
    num_blocks : int, optional
        The target number of blocks / sm to achieve for cuda
    num_threads : int, optional
        The target number of threads / block to achieve for cuda
    no_shared : bool, optional
        If ``True``, do not use the shared_memory_manager
        to attempt to optimize for CUDA
    L1_preferred : bool, optional
        If ``True``, prefer a larger L1 cache and a smaller shared memory size for CUDA
    multi_thread : int, optional
        The number of threads to use during optimization
    force_optimize : bool, optional
        If ``True``, redo the cache optimization even if the same mechanism
    build_path : str, optional
        The output directory for the jacobian files
    last_spec : str, optional
        If specified, the species to assign to the last index.
        Typically should be N2, Ar, He or another inert bath gas
    skip_jac : bool, optional
        If ``True``, only the reaction rate subroutines will be generated
    auto_diff : bool, optional
        If ``True``, generate files for use with the Adept autodifferention library.

    Returns
    -------
    None

    """
    if lang != 'c' and auto_diff:
        print('Error: autodifferention only supported for C')
        sys.exit(2)

    if auto_diff:
        skip_jac = True

    lang = lang.lower()
    if lang not in utils.langs:
        print('Error: language needs to be one of: ')
        for supported in utils.langs:
            print(supported)
        sys.exit(2)

    # Reject the incomplete backends before any output is written.
    if lang not in utils.supported_langs:
        raise NotImplementedError(
            '{} output is not implemented. The {} backend was never completed '
            'and does not produce usable source. Supported languages are: '
            '{}.'.format(lang, lang, ', '.join(utils.supported_langs))
        )

    # create output directory if none exists
    utils.create_dir(build_path)

    if auto_diff:
        with open(os.path.join(build_path, 'ad_jacob.h'), 'w') as file:
            file.write(
                '#ifndef AD_JAC_H\n'
                '#define AD_JAC_H\n'
                'void eval_jacob (const double t, const double pres, '
                'const double* y, double* jac);\n'
                '#endif\n'
            )

    assert mech_name is not None or gas is not None, 'No mechanism specified!'

    # Interpret reaction mechanism file, depending on Cantera or
    # Chemkin format.
    if mech_name is not None and mech_name.endswith(('.cti', '.xml')):
        raise NotImplementedError(
            f'{mech_name} uses a legacy Cantera format. Both CTI and the XML '
            'format were removed in Cantera 3.0; convert the mechanism with '
            '`python -m cantera.cti2yaml` or `python -m cantera.ctml2yaml` '
            'and pass the resulting YAML file.'
        )

    if gas is not None or mech_name.endswith(('.yaml', '.yml')):
        elems, specs, reacs = mech.read_mech_ct(mech_name, gas)
    else:
        elems, specs, reacs = mech.read_mech(mech_name, therm_name)

    if not specs:
        print(f'No species found in file: {mech_name}')
        sys.exit(3)

    if not reacs:
        print(f'No reactions found in file: {mech_name}')
        sys.exit(3)

    # check to see if the last_spec is specified
    if last_spec is not None:
        # find the index if possible
        isp = next(
            (
                i
                for i, sp in enumerate(specs)
                if sp.name.lower() == last_spec.lower().strip()
            ),
            None,
        )
        if isp is None:
            print(
                f'Warning: User specified last species {last_spec} '
                'not found in mechanism.'
                '  Attempting to find a default species.'
            )
            last_spec = None
        else:
            last_spec = isp
    else:
        print(
            'User specified last species not found or not specified.  '
            'Attempting to find a default species'
        )
    if last_spec is None:
        wt = chem.get_elem_wt()
        # check for N2, Ar, He, etc.
        candidates = [('N2', wt['n'] * 2.0), ('Ar', wt['ar']), ('He', wt['he'])]
        for sp in candidates:
            match = next(
                (
                    isp
                    for isp, spec in enumerate(specs)
                    if sp[0].lower() == spec.name.lower() and sp[1] == spec.mw
                ),
                None,
            )
            if match is not None:
                last_spec = match
                break
        if last_spec is not None:
            print(f'Default last species {specs[last_spec].name} found.')
    if last_spec is None:
        print(
            'Warning: Neither a user specified or default last species '
            'could be found. Proceeding using the last species in the '
            f'base mechanism: {specs[-1].name}'
        )
        last_spec = len(specs) - 1

    optimize_cache = optimize_cache and cache.have_bitarray
    if optimize_cache:
        (
            specs,
            reacs,
            fwd_spec_mapping,
            fwd_rxn_mapping,
            reverse_spec_mapping,
            reverse_rxn_mapping,
        ) = cache.optimize_cache(
            specs, reacs, multi_thread, force_optimize, build_path, last_spec
        )
    else:
        fwd_rxn_mapping = list(range(len(reacs)))

        fwd_spec_mapping, reverse_spec_mapping = utils.get_species_mappings(
            len(specs), last_spec
        )

        # pick up the last_spec and drop it at the end
        temp = specs[:]
        for i in range(len(specs)):
            specs[i] = temp[fwd_spec_mapping[i]]

    # remove old file which potentially could corrupt library generation
    if not auto_diff:
        try:
            os.remove(os.path.join(build_path, 'jacobs', f'jac_list_{lang}'))
        except OSError:
            pass

        try:
            os.remove(os.path.join(build_path, 'rates', f'rate_list_{lang}'))
        except OSError:
            pass

    if lang == 'cuda':
        CUDAParams.write_launch_bounds(
            build_path, num_blocks, num_threads, L1_preferred, no_shared
        )
    smm = None
    if lang == 'cuda' and not no_shared:
        smm = shared.shared_memory_manager(num_blocks, num_threads, L1_preferred)

    # reassign the reaction's product / reactant / third body list
    # to integer indexes for speed
    utils.reassign_species_lists(reacs, specs)

    ## now begin writing subroutines

    # print reaction rate subroutine
    rate.write_rxn_rates(
        build_path, lang, specs, reacs, fwd_rxn_mapping, smm, auto_diff
    )

    # if third-body/pressure-dependent reactions,
    # print modification subroutine
    if next((r for r in reacs if (r.thd_body or r.pdep)), None):
        rate.write_rxn_pressure_mod(
            build_path, lang, specs, reacs, fwd_rxn_mapping, smm, auto_diff
        )

    # write species rates subroutine
    seen_sp = rate.write_spec_rates(
        build_path,
        lang,
        specs,
        reacs,
        fwd_spec_mapping,
        fwd_rxn_mapping,
        smm,
        auto_diff,
    )

    # write chem_utils subroutines
    rate.write_chem_utils(build_path, lang, specs, auto_diff)

    # write derivative subroutines
    rate.write_derivs(build_path, lang, specs, reacs, seen_sp, auto_diff)

    # write mass-mole fraction conversion subroutine
    rate.write_mass_mole(build_path, lang, specs)

    # write header file
    aux.write_header(build_path, lang)

    # write mechanism initializers and testing methods
    aux.write_mechanism_initializers(
        build_path,
        lang,
        specs,
        reacs,
        fwd_spec_mapping,
        reverse_spec_mapping,
        initial_state,
        optimize_cache,
        last_spec,
        auto_diff,
    )

    if not skip_jac:
        # write Jacobian subroutine
        touched = write_jacobian(build_path, lang, specs, reacs, seen_sp, smm)

        write_sparse_multiplier(build_path, lang, touched, len(specs))

    return 0
