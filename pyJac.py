#! /usr/bin/env python
"""Creates source code for calculating analytical Jacobian matrix.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import sys
import math
from argparse import ArgumentParser
import os

# Local imports
import chem_utilities as chem
import mech_interpret as mech
import rate_subs as rate
import utils
import mech_auxiliary as aux
import CUDAParams
import CParams
import cache_optimizer as cache
import shared_memory as shared


def calculate_shared_memory(rind, rxn, specs, reacs, rev_reacs, pdep_reacs):
    # need to figure out shared memory stuff
    variable_list = []
    usages = []
    fwd_usage = 2
    rev_usage = 0 if not rxn.rev else 2
    pres_mod_usage = 0 if not (rxn.pdep or rxn.thd_body) else (4 if rxn.thd_body else 2)
    reac_usages = [0 for i in range(len(rxn.reac))]
    prod_usages = [0 for i in range(len(rxn.prod))]
    # add variables
    variable_list.append(utils.get_array('cuda', 'fwd_rates', rind))
    if rxn.rev:
        variable_list.append(utils.get_array('cuda', 'rev_rates', rev_reacs.index(rind)))
    if rxn.pdep or rxn.thd_body:
        variable_list.append(utils.get_array('cuda', 'pres_mod', pdep_reacs.index(rind)))
    for sp in set(rxn.reac + rxn.prod + [x[0] for x in rxn.thd_body_eff]):
        variable_list.append(utils.get_array('cuda', 'conc', sp))

    alphaij_count = 0
    # calculate usages
    if rxn.thd_body or rxn.pdep:
        fwd_usage += 1
        if rxn.rev:
            rev_usage += 1
        for i, thd in enumerate(rxn.thd_body_eff):
            # check alphaij
            alphaij = thd[1]
            if alphaij is not None and alphaij != 1.0:
                alphaij_count += 1
        fwd_usage += alphaij_count
        if rxn.rev:
            rev_usage += alphaij_count

    for i, sp in enumerate(rxn.reac):
        nu = rxn.reac_nu[i]
        if nu - 1 > 0:
            reac_usages[i] += 1
            if rxn.thd_body:
                pres_mod_usage += 1
        for i2, sp2 in enumerate(rxn.reac):
            if sp == sp2:
                continue
            reac_usages[i2] += 1

    if rxn.rev:
        for i, sp in enumerate(rxn.prod):
            nu = rxn.prod_nu[i]
            if nu - 1 > 0:
                prod_usages[i] += 1
            for i2, sp2 in enumerate(rxn.prod):
                if sp == sp2:
                    continue
                prod_usages[i2] += 1

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

    if CUDAParams.JacRateStrat == CUDAParams.JacRatesCacheStrat.Exclude:
        if rxn.rev:
            usages[0] = 0
            usages[1] = 0
        else:
            usages[0] = 0

    return variable_list, usages


def write_dr_dy(file, lang, rev_reacs, rxn, rind, pind, nspec, get_array):
    # write the T_Pr and T_Fi terms if needed
    if rxn.pdep:
        jline = utils.line_start + 'pres_mod_temp = '
        jline += get_array(lang, 'pres_mod', pind) + ' * ('
        # dPr/dYj contribution
        if rxn.low:
            # unimolecular/recombination
            jline += '(1.0 / (1.0 + Pr))'
        elif rxn.high:
            # chem-activated bimolecular
            jline += '(-Pr / (1.0 + Pr))'
        if rxn.troe:
            jline += (' - log(Fcent) * 2.0 * A * (B * '
                      '{:.16}'.format(1.0 / math.log(10.0)) +
                      ' + A * '
                      '{:.16}) / '.format(0.14 / math.log(10.0)) +
                      '(B * B * B * (1.0 + A * A / (B * B)) '
                      '* (1.0 + A * A / (B * B)))'
                      )
        elif rxn.sri:
            jline += ('- X * X * '
                      '{:.16} * '.format(2.0 / math.log(10.0)) +
                      'log10(Pr) * '
                      'log({:.4} * '.format(rxn.sri_par[0]) +
                      'exp({:.4} / T) + '.format(-rxn.sri_par[1]) +
                      'exp(T / {:.4}))'.format(-rxn.sri_par[2])
                      )

        jline += ') * '
        if rxn.rev:
            jline += '(' + get_array(lang, 'fwd_rates', rind)
            jline += ' - ' + \
                     get_array(lang, 'rev_rates', rev_reacs.index(rind))
            jline += ')'
        else:
            jline += get_array(lang, 'fwd_rates', rind)
        file.write(jline + utils.line_end[lang])

    file.write('  j_temp = -mw_avg * rho_inv * (')
    jline = ''
    # next, contribution from dR/dYj
    # namely the T_dy independent term
    if rxn.pdep or rxn.thd_body:
        jline += get_array(lang, 'pres_mod', pind)
        jline += ' * ('

    reac_nu = 0
    prod_nu = 0
    if rxn.thd_body and not rxn.pdep:
        reac_nu = 1
        if rxn.rev:
            prod_nu = 1

    # get reac and prod nu sums
    reac_nu += sum(rxn.reac_nu)

    if rxn.rev:
        prod_nu += sum(rxn.prod_nu)

    if reac_nu != 0:
        if reac_nu == -1:
            jline += '-'
        elif reac_nu != 1:
            jline += '{} * '.format(float(reac_nu))
        jline += '' + get_array(lang, 'fwd_rates', rind)

    if prod_nu != 0:
        if prod_nu == 1:
            jline += ' - '
        elif prod_nu == -1:
            jline += ' + '
        else:
            jline += ' - {} * '.format(float(prod_nu))
        jline += '' + get_array(lang, 'rev_rates', rev_reacs.index(rind))

    # find alphaij_hat
    alphaij_hat = None
    counter = {}
    counter[1.0] = 0
    if rxn.thd_body_eff:
        for spec, efficiency in rxn.thd_body_eff:
            if not efficiency in counter:
                counter[efficiency] = 0
            counter[efficiency] += 1
        counter[1.0] += (nspec - len(rxn.thd_body_eff))
        alphaij_hat = max(counter.keys(), key=lambda x: counter[x])

    # now handle third body / pdep parts if needed
    if rxn.thd_body:
        jline += '))'
        if alphaij_hat is not None:
            if alphaij_hat == 1.0:
                jline += ' + '
            elif alphaij_hat == -1.0:
                jline += ' - '
            else:
                jline += ' + {} * '.format(alphaij_hat)
            if rxn.rev:
                jline += '(' + get_array(lang, 'fwd_rates', rind)
                jline += ' - ' + \
                         get_array(lang, 'rev_rates', rev_reacs.index(rind))
                jline += ')'
            else:
                jline += get_array(lang, 'fwd_rates', rind)
    elif rxn.pdep:
        jline += ') + pres_mod_temp'
        jline += ')'
        if alphaij_hat is not None:
            if alphaij_hat == 1.0:
                jline += ' + '
            elif alphaij_hat == -1.0:
                jline += ' - '
            else:
                jline += ' + {} * '.format(alphaij_hat)
            jline += '(pres_mod_temp / conc_temp)'
    else:
        jline += ')'
    file.write(jline + utils.line_end[lang])

    if rxn.pdep and rxn.pdep_sp == '':
        file.write('  pres_mod_temp /= conc_temp' + utils.line_end[lang])
    return alphaij_hat


def write_rates(file, lang, rxn):
    if not (rxn.cheb or rxn.plog):
        file.write('  kf = ' + rate.rxn_rate_const(rxn.A, rxn.b, rxn.E) +
                   utils.line_end[lang])
    elif rxn.plog:
        vals = rxn.plog_par[0]
        file.write('  if (pres <= {:.4e}) {{\n'.format(vals[0]))
        line = ('    kf = ' + rate.rxn_rate_const(vals[1], vals[2], vals[3]))
        file.write(line + utils.line_end[lang])

        for idx, vals in enumerate(rxn.plog_par[:-1]):
            vals2 = rxn.plog_par[idx + 1]

            line = ('  }} else if ((pres > {:.4e}) '.format(vals[0]) +
                    '&& (pres <= {:.4e})) {{\n'.format(vals2[0]))
            file.write(line)

            line = ('    kf = log(' +
                    rate.rxn_rate_const(vals[1], vals[2], vals[3]) + ')'
                    )
            file.write(line + utils.line_end[lang])
            line = ('    kf2 = log(' +
                    rate.rxn_rate_const(vals2[1], vals2[2], vals2[3]) + ')'
                    )
            file.write(line + utils.line_end[lang])

            pres_log_diff = math.log(vals2[0]) - math.log(vals[0])
            line = ('    kf = exp(kf + (kf2 - kf) * (log(pres) - ' +
                    '{:.16e}) / '.format(math.log(vals[0])) +
                    '{:.16e})'.format(pres_log_diff)
                    )
            file.write(line + utils.line_end[lang])

        vals = rxn.plog_par[-1]
        file.write('  }} else if (pres > {:.4e}) {{\n'.format(vals[0]))
        line = ('    kf = ' + rate.rxn_rate_const(vals[1], vals[2], vals[3]))
        file.write(line + utils.line_end[lang])
        file.write('  }\n')
    elif rxn.cheb:
        file.write(rate.get_cheb_rate(lang, rxn, False))
    if rxn.rev and not rxn.rev_par:
        file.write('  kr = kf / Kc' + utils.line_end[lang])
    elif rxn.rev_par:
        file.write('  kr = ' +
                   rate.rxn_rate_const(rxn.rev_par[0],
                                       rxn.rev_par[1],
                                       rxn.rev_par[2]
                                       ) +
                   utils.line_end[lang]
                   )


def write_dr_dy_species(lang, specs, rxn, pind, j_sp, sp_j, alphaij_hat, rind, rev_reacs, get_array):
    jline = 'j_temp'
    if rxn.pdep and rxn.pdep_sp == '' and alphaij_hat is not None:
        alphaij = next((thd[1] for thd in rxn.thd_body_eff
                        if thd[0] == j_sp and thd[1] != 1.0), None)
        if alphaij is None:
            alphaij = 1.0
        if alphaij != alphaij_hat:
            diff = alphaij - alphaij_hat
            if diff != 1:
                if diff == -1:
                    jline += ' - pres_mod_temp'
                else:
                    jline += ' + {} * pres_mod_temp'.format(diff)
            else:
                jline += ' + pres_mod_temp'
    elif rxn.pdep and rxn.pdep_sp == j_sp:
        jline += ' + pres_mod_temp / (rho * {})'.format(get_array(lang, 'y', j_sp))
    elif rxn.thd_body and not rxn.pdep and alphaij_hat is not None:
        alphaij = next((thd[1] for thd in rxn.thd_body_eff
                        if thd[0] == j_sp and thd[1] != 1.0), None)
        if alphaij is None:
            alphaij = 1.0
        if alphaij != alphaij_hat:
            diff = alphaij - alphaij_hat
            if diff != 0:
                if diff != 1:
                    if diff == -1:
                        jline += ' - '
                    else:
                        jline += ' + {} * '.format(diff)
                else:
                    jline += ' + '

                if rxn.rev:
                    jline += '(' + get_array(lang, 'fwd_rates', rind)
                    jline += ' - ' + \
                             get_array(lang, 'rev_rates', rev_reacs.index(rind))
                    jline += ')'
                else:
                    jline += get_array(lang, 'fwd_rates', rind)
    if (rxn.pdep or rxn.thd_body) and (j_sp in rxn.reac or (rxn.rev and j_sp in rxn.prod)):
        jline += ' + ' + get_array(lang, 'pres_mod', pind)
        jline += ' * '

    if j_sp in rxn.reac:
        if not rxn.pdep and not rxn.thd_body:
            jline += ' + '
        nu = rxn.reac_nu[rxn.reac.index(j_sp)]
        if nu != 1:
            if nu == -1:
                jline += '-'
            else:
                jline += '{} * '.format(float(nu))
        jline += 'kf'
        if (nu - 1) > 0:
            if utils.is_integer(nu):
                # integer, so just use multiplication
                for i in range(int(nu) - 1):
                    jline += ' * ' + get_array(lang, 'conc', j_sp)
            else:
                jline += (' * pow(' + get_array(lang, 'conc', j_sp) +
                          ', {})'.format(nu - 1)
                          )

        # loop through remaining reactants
        for i, isp in enumerate(rxn.reac):
            if isp == j_sp:
                continue

            nu = rxn.reac_nu[i]
            if utils.is_integer(nu):
                # integer, so just use multiplication
                for i in range(int(nu)):
                    jline += ' * ' + get_array(lang, 'conc', isp)
            else:
                jline += (' * pow(' + get_array(lang, 'conc', isp) +
                          ', ' + str(nu) + ')'
                          )

    if rxn.rev and j_sp in rxn.prod:
        if not rxn.pdep and not rxn.thd_body and not j_sp in rxn.reac:
            jline += ' + '
        elif j_sp in rxn.reac:
            jline += ' + '

        nu = rxn.prod_nu[rxn.prod.index(j_sp)]
        if nu != -1:
            if nu == 1:
                jline += '-'
            else:
                jline += '{} * '.format(float(-1 * nu))

        jline += 'kr'
        if (nu - 1) > 0:
            if utils.is_integer(nu):
                # integer, so just use multiplication
                for i in range(int(nu) - 1):
                    jline += ' * ' + get_array(lang, 'conc', j_sp)
            else:
                jline += (' * pow(' + get_array(lang, 'conc', j_sp) +
                          ', {})'.format(nu - 1)
                          )

        # loop through remaining products
        for i, isp in enumerate(rxn.prod):
            if isp == j_sp:
                continue

            nu = rxn.prod_nu[i]
            if utils.is_integer(nu):
                # integer, so just use multiplication
                jline += (''.join([' * ' +
                          get_array(lang, 'conc', isp) for i in range(int(nu))])
                          )
            else:
                jline += ' * pow(' + get_array(lang, 'conc', isp)
                jline += ', {})'.format(nu)

    return jline


def write_kc(file, lang, specs, rxn):
    sum_nu = 0
    coeffs = {}
    for isp in set(rxn.reac + rxn.prod):
        sp = specs[isp]
        nu = get_nu(isp, rxn)

        if nu == 0:
            continue

        sum_nu += nu

        lo_array = [nu] + [sp.lo[6], sp.lo[0], sp.lo[0] - 1.0, sp.lo[1] / 2.0,
                           sp.lo[2] / 6.0, sp.lo[3] / 12.0, sp.lo[4] / 20.0,
                           sp.lo[5]
                           ]

        lo_array = [x * lo_array[0] for x in [lo_array[1] - lo_array[2]] +
                    lo_array[3:]
                    ]

        hi_array = [nu] + [sp.hi[6], sp.hi[0], sp.hi[0] - 1.0, sp.hi[1] / 2.0,
                           sp.hi[2] / 6.0, sp.hi[3] / 12.0, sp.hi[4] / 20.0,
                           sp.hi[5]
                           ]

        hi_array = [x * hi_array[0] for x in [hi_array[1] - hi_array[2]] +
                    hi_array[3:]
                    ]
        if not sp.Trange[1] in coeffs:
            coeffs[sp.Trange[1]] = lo_array, hi_array
        else:
            coeffs[sp.Trange[1]] = [lo_array[i] + coeffs[sp.Trange[1]][0][i]
                                    for i in range(len(lo_array))
                                    ], \
                                    [hi_array[i] + coeffs[sp.Trange[1]][1][i]
                                    for i in range(len(hi_array))
                                    ]

    isFirst = True
    for T_mid in coeffs:
        # need temperature conditional for equilibrium constants
        line = utils.line_start + 'if (T <= {:})'.format(T_mid)
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
        line += ('({:.16e} + '.format(lo_array[0]) +
                 '{:.16e} * '.format(lo_array[1]) +
                 'logT + T * ('
                 '{:.16e} + T * ('.format(lo_array[2]) +
                 '{:.16e} + T * ('.format(lo_array[3]) +
                 '{:.16e} + '.format(lo_array[4]) +
                 '{:.16e} * T))) - '.format(lo_array[5]) +
                 '{:.16e} / T)'.format(lo_array[6]) +
                 utils.line_end[lang]
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
        line += ('({:.16e} + '.format(hi_array[0]) +
                 '{:.16e} * '.format(hi_array[1]) +
                 'logT + T * ('
                 '{:.16e} + T * ('.format(hi_array[2]) +
                 '{:.16e} + T * ('.format(hi_array[3]) +
                 '{:.16e} + '.format(hi_array[4]) +
                 '{:.16e} * T))) - '.format(hi_array[5]) +
                 '{:.16e} / T)'.format(hi_array[6]) +
                 utils.line_end[lang]
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
        line += '{:.16e} * '.format(num)
    line += 'exp(Kc)' + utils.line_end[lang]
    file.write(line)


def get_nu(isp, rxn):
    if isp in rxn.prod and isp in rxn.reac:
        nu = (rxn.prod_nu[rxn.prod.index(isp)] -
              rxn.reac_nu[rxn.reac.index(isp)])
        # check if net production zero
        if nu == 0:
            return 0
    elif isp in rxn.prod:
        nu = rxn.prod_nu[rxn.prod.index(isp)]
    elif isp in rxn.reac:
        nu = -rxn.reac_nu[rxn.reac.index(isp)]
    else:
        # doesn't participate in reaction
        return 0
    return nu


def get_infs(rxn):
    if rxn.low:
        # unimolecular/recombination fall-off
        beta_0minf = rxn.low[1] - rxn.b
        E_0minf = rxn.low[2] - rxn.E
        k0kinf = rate.rxn_rate_const(rxn.low[0] / rxn.A,
                                     beta_0minf, E_0minf
                                     )
    elif rxn.high:
        # chem-activated bimolecular rxn
        beta_0minf = rxn.b - rxn.high[1]
        E_0minf = rxn.E - rxn.high[2]
        k0kinf = rate.rxn_rate_const(rxn.A / rxn.high[0],
                                     beta_0minf, E_0minf
                                     )
    return beta_0minf, E_0minf, k0kinf


def write_dt_comment(file, lang, rind):
    line = utils.line_start + utils.comment[lang]
    line += ('partial of rxn ' + str(rind) + ' wrt T' + '\n')
    file.write(line)


def write_dy_comment(file, lang, rind):
    line = utils.line_start + utils.comment[lang]
    line += ('partial of rxn ' + str(rind) + ' wrt species' + '\n')
    file.write(line)


def write_dy_y_finish_comment(file, lang):
    line = utils.line_start + utils.comment[lang]
    line += 'Finish dYk / Yj\'s\n'
    file.write(line)


def write_dt_t_comment(file, lang, sp):
    line = utils.line_start + utils.comment[lang]
    line += 'partial of dT wrt T for spec {}'.format(sp.name) + '\n'
    file.write(line)


def write_dt_y_comment(file, lang, sp):
    line = utils.line_start + utils.comment[lang]
    line += 'partial of T wrt Y_{}'.format(sp.name) + '\n'
    file.write(line)

def get_rxn_params_dt(rxn, rev=False):
    jline = ''
    if rev:
        if (abs(rxn.rev_par[1]) > 1.0e-90
            and abs(rxn.rev_par[2]) > 1.0e-90):
            jline += ('{:.16e} + '.format(rxn.rev_par[1]) +
                      '({:.16e} / T)'.format(rxn.rev_par[2])
                      )
        elif abs(rxn.rev_par[1]) > 1.0e-90:
            jline += '{:.16e}'.format(rxn.rev_par[1])
        elif abs(rxn.rev_par[2]) > 1.0e-90:
            jline += '({:.16e} / T)'.format(rxn.rev_par[2])
    else:
        if (abs(rxn.b) > 1.0e-90) and (abs(rxn.E) > 1.0e-90):
            jline += '{:.16e} + ({:.16e} / T)'.format(rxn.b, rxn.E)
        elif abs(rxn.b) > 1.0e-90:
            jline += '{:.16e}'.format(rxn.b)
        elif abs(rxn.E) > 1.0e-90:
            jline += '({:.16e} / T)'.format(rxn.E)
    return jline


def write_db_dt_def(file, lang, specs, reacs, rev_reacs, dBdT_flag, do_unroll):
    if lang == 'cuda' or do_unroll:
        if len(rev_reacs):
            file.write('  double dBdT[{}]'.format(len(specs)) + utils.line_end[lang])
        template = 'dBdT[{}]'
    else:
        template = 'dBdT_{}'
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

            if lang in ['c', 'cuda']:
                dBdT = template.format(sp_ind)
            elif lang in ['fortran', 'matlab']:
                dBdT = template.format(sp_ind + 1)
            # declare dBdT
            if not (lang == 'cuda' or do_unroll):
                file.write('  double ' + dBdT + utils.line_end[lang])

            # dB/dT evaluation (with temperature conditional)
            line = utils.line_start + 'if (T <= {:})'.format(specs[sp_ind].Trange[1])
            if lang in ['c', 'cuda']:
                line += ' {\n'
            elif lang == 'fortran':
                line += ' then\n'
            elif lang == 'matlab':
                line += '\n'
            file.write(line)

            line = ('    ' + dBdT +
                    ' = ({:.16e}'.format(specs[sp_ind].lo[0] - 1.0) +
                    ' + {:.16e} / T) / T'.format(specs[sp_ind].lo[5]) +
                    ' + {:.16e} + T'.format(specs[sp_ind].lo[1] / 2.0) +
                    ' * ({:.16e}'.format(specs[sp_ind].lo[2] / 3.0) +
                    ' + T * ({:.16e}'.format(specs[sp_ind].lo[3] / 4.0) +
                    ' + {:.16e} * T))'.format(specs[sp_ind].lo[4] / 5.0) +
                    utils.line_end[lang]
                    )
            file.write(line)

            if lang in ['c', 'cuda']:
                file.write('  } else {\n')
            elif lang in ['fortran', 'matlab']:
                file.write('  else\n')

            line = ('    ' + dBdT +
                    ' = ({:.16e}'.format(specs[sp_ind].hi[0] - 1.0) +
                    ' + {:.16e} / T) / T'.format(specs[sp_ind].hi[5]) +
                    ' + {:.16e} + T'.format(specs[sp_ind].hi[1] / 2.0) +
                    ' * ({:.16e}'.format(specs[sp_ind].hi[2] / 3.0) +
                    ' + T * ({:.16e}'.format(specs[sp_ind].hi[3] / 4.0) +
                    ' + {:.16e} * T))'.format(specs[sp_ind].hi[4] / 5.0) +
                    utils.line_end[lang]
                    )
            file.write(line)

            if lang in ['c', 'cuda']:
                file.write('  }\n\n')
            elif lang == 'fortran':
                file.write('  end if\n\n')
            elif lang == 'matlab':
                file.write('  end\n\n')

def get_db_dt(lang, specs, rxn, do_unroll):
    if lang == 'cuda' or do_unroll:
        template = 'dBdT[{}]'
    else:
        template = 'dBdT_{}'
    jline = ''
    notfirst = False
    # contribution from dBdT terms from
    # all participating species
    for sp_ind in rxn.prod:
        if sp_ind in rxn.reac:
            nu = (rxn.prod_nu[rxn.prod.index(sp_ind)] -
                  rxn.reac_nu[rxn.reac.index(sp_ind)]
                  )
        else:
            nu = rxn.prod_nu[rxn.prod.index(sp_ind)]

        if (nu == 0):
            continue

        if lang in ['c', 'cuda']:
            dBdT = template.format(sp_ind)
        elif lang in ['fortran', 'matlab']:
            dBdT = template.format(sp_ind + 1)

        if not notfirst:
            # first entry
            if nu == 1:
                jline += dBdT
            elif nu == -1:
                jline += '-' + dBdT
            else:
                jline += '{} * '.format(float(nu)) + dBdT
        else:
            # not first entry
            if nu == 1:
                jline += ' + '
            elif nu == -1:
                jline += ' - '
            else:
                if (nu > 0):
                    jline += ' + {}'.format(float(nu))
                else:
                    jline += ' - {}'.format(float(abs(nu)))
                jline += ' * '
            jline += dBdT
        notfirst = True

    for sp_ind in rxn.reac:
        # skip species also in products, already counted
        if sp_ind in rxn.prod:
            continue

        nu = -rxn.reac_nu[rxn.reac.index(sp_ind)]

        if lang in ['c', 'cuda']:
            dBdT = template.format(sp_ind)
        elif lang in ['fortran', 'matlab']:
            dBdT = template.format(sp_ind + 1)

        # not first entry
        if nu == 1:
            jline += ' + '
        elif nu == -1:
            jline += ' - '
        else:
            if (nu > 0):
                jline += ' + {}'.format(float(nu))
            else:
                jline += ' - {}'.format(float(abs(nu)))
            jline += ' * '
        jline += dBdT

    return jline


def write_pr(file, lang, specs, reacs, pdep_reacs, rxn, get_array, last_conc_temp=None):
    # print lines for necessary pressure-dependent variables
    line = utils.line_start + '{} = '.format('conc_temp' if rxn.thd_body_eff > 0 else 'Pr')
    conc_temp_log = None
    if rxn.thd_body_eff:
        # take care of the conc_temp collapsing
        conc_temp_log = []
    if rxn.pdep_sp != '':
        line += get_array(lang, 'conc', rxn.pdep_sp)
    else:
        line += '(m'

        for isp, eff in rxn.thd_body_eff:
            if eff > 1.0:
                line += ' + {} * '.format(eff - 1.0)
            elif eff < 1.0:
                line += ' - {} * '.format(1.0 - eff)
            if eff != 1.0:
                line += get_array(lang, 'conc', isp)
                if conc_temp_log is not None:
                    conc_temp_log.append((isp, eff - 1.0))
        line += ')'

        if not rxn.thd_body_eff:
            # only depends on +m
            conc_temp_log = []

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

            use_conc = new_conc_temp if len(new_conc_temp) < len(conc_temp_log) else conc_temp_log
            if len(use_conc):
                # remake the line with the updated numbers
                line = utils.line_start + 'conc_temp {}= ({}'.format(
                    '+' if use_conc == new_conc_temp else '',
                    'm + ' if use_conc != new_conc_temp else '')

                for i, thd_sp in enumerate(use_conc):
                    isp = thd_sp[0]
                    if i > 0:
                        line += ' {}{} * '.format('- ' if thd_sp[1] < 0 else '+ ', abs(thd_sp[1]))
                    else:
                        line += '{} * '.format(thd_sp[1])
                    line += get_array(lang, 'conc', isp)
                line += ')'
            else:
                line = ''

    if rxn.thd_body_eff and len(line):
        file.write(line + utils.line_end[lang])

    if rxn.pdep:
        if rxn.thd_body_eff:
            line = utils.line_start + 'Pr = conc_temp'
        beta_0minf, E_0minf, k0kinf = get_infs(rxn)
        # finish writing P_ri
        line += (' * (' + k0kinf + ')' +
                 utils.line_end[lang]
                 )
        file.write(line)

    return conc_temp_log


def write_troe(file, lang, rxn):
    line = ('  Fcent = '
            '{:.16e} * '.format(1.0 - rxn.troe_par[0]) +
            'exp(T / {:.16e})'.format(-rxn.troe_par[1]) +
            ' + {:.16e} * exp(T / '.format(rxn.troe_par[0]) +
            '{:.16e})'.format(-rxn.troe_par[2])
            )
    if len(rxn.troe_par) == 4 and rxn.troe_par[3] != 0.0:
        line += ' + exp({:.16e} / T)'.format(-rxn.troe_par[3])
    line += utils.line_end[lang]
    file.write(line)

    line = ('  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4' +
            utils.line_end[lang]
            )
    file.write(line)

    line = ('  B = 0.806 - 1.1762 * log10(Fcent) - '
            '0.14 * log10(Pr)' +
            utils.line_end[lang]
            )
    file.write(line)

    line = ('  lnF_AB = 2.0 * log(Fcent) * '
            'A / (B * B * B * (1.0 + A * A / (B * B)) * '
            '(1.0 + A * A / (B * B)))' +
            utils.line_end[lang]
            )
    file.write(line)


def write_sri(file, lang):
    line = ('  X = 1.0 / (1.0 + log10(Pr) * log10(Pr))' +
            utils.line_end[lang]
            )
    file.write(line)

def get_pdep_dt(lang, rxn, rev_reacs, rind, pind, get_array):
    beta_0minf, E_0minf, k0kinf = get_infs(rxn)
    jline = utils.line_start + 'j_temp = (' + get_array(lang, 'pres_mod', pind)
    jline += ' * ((' + ('-Pr' if rxn.high else '')  # high -> chem-activated bimolecular rxn

    # dPr/dT
    jline += ('({:.4e} + ('.format(beta_0minf) +
              '{:.16e} / T) - 1.0) / '.format(E_0minf) +
              '(T * (1.0 + Pr)))'
              )

    if rxn.sri:
        jline += write_sri_dt(lang, rxn, beta_0minf, E_0minf, k0kinf)
    elif rxn.troe:
        jline += write_troe_dt(lang, rxn, beta_0minf, E_0minf, k0kinf)

    jline += ') * '

    if rxn.rev:
        # forward and reverse reaction rates
        jline += '(' + get_array(lang, 'fwd_rates', rind)
        jline += ' - ' + \
                 get_array(lang, 'rev_rates', rev_reacs.index(rind))
        jline += ')'
    else:
        # forward reaction rate only
        jline += '' + get_array(lang, 'fwd_rates', rind)

    jline += ' + (' + get_array(lang, 'pres_mod', pind)

    return jline


def write_sri_dt(lang, rxn, beta_0minf, E_0minf, k0kinf):
    jline = (' + X * ((('
             '{:.16} / '.format(rxn.sri_par[0] * rxn.sri_par[1]) +
             '(T * T)) * exp('
             '{:.16} / T) - '.format(-rxn.sri_par[1]) +
             '{:.16e} * '.format(1.0 / rxn.sri_par[2]) +
             'exp(T / {:.16})) / '.format(-rxn.sri_par[2]) +
             '({:.16} * '.format(rxn.sri_par[0]) +
             'exp({:.16} / T) + '.format(-rxn.sri_par[1]) +
             'exp(T / {:.16})) - '.format(-rxn.sri_par[2]) +
             'X * {:.16} * '.format(2.0 / math.log(10.0)) +
             'log10(Pr) * ('
             '{:.16e} + ('.format(beta_0minf) +
             '{:.16e} / T) - 1.0) * '.format(E_0minf) +
             'log({:.16} * exp('.format(rxn.sri_par[0]) +
             '{:.16} / T) + '.format(-rxn.sri_par[1]) +
             'exp(T / '
             '{:.16})) / T)'.format(-rxn.sri_par[2])
             )

    if len(rxn.sri_par) == 5 and rxn.sri_par[4] != 0.0:
        jline += ' + ({:.16} / T)'.format(rxn.sri_par[4])

    return jline


def write_troe_dt(lang, rxn, beta_0minf, E_0minf, k0kinf):
    jline = (' + (((1.0 / '
             '(Fcent * (1.0 + A * A / (B * B)))) - '
             'lnF_AB * ('
             '-{:.16e}'.format(0.67 / math.log(10.0)) +
             ' * B + '
             '{:.16e} * '.format(1.1762 / math.log(10.0)) +
             'A) / Fcent)'
             ' * ({:.16e}'.format(-(1.0 - rxn.troe_par[0]) /
                                 rxn.troe_par[1]) +
             ' * exp(T / '
             '{:.16e}) - '.format(-rxn.troe_par[1]) +
             '{:.16e} * '.format(rxn.troe_par[0] /
                                rxn.troe_par[2]) +
             'exp(T / '
             '{:.16e})'.format(-rxn.troe_par[2])
             )
    if len(rxn.troe_par) == 4 and rxn.troe_par[3] != 0.0:
        jline += (' + ({:.16e} / '.format(rxn.troe_par[3]) +
                  '(T * T)) * exp('
                  '{:.16e} / T)'.format(-rxn.troe_par[3])
                  )
    jline += '))'

    jline += (' - lnF_AB * ('
              '{:.16e}'.format(1.0 / math.log(10.0)) +
              ' * B + '
              '{:.16e}'.format(0.14 / math.log(10.0)) +
              ' * A) * '
              '({:.16e} + ('.format(beta_0minf) +
              '{:.16e} / T) - 1.0) / T'.format(E_0minf)
              )

    return jline


def write_dcp_dt(file, lang, specs, sparse_indicies):
    T_mid_buckets = {}
    # put all of the same T_mids together
    for isp, sp in enumerate(specs):
        if sp.Trange[1] not in T_mid_buckets:
            T_mid_buckets[sp.Trange[1]] = []
        T_mid_buckets[sp.Trange[1]].append(isp)

    first = True
    for T_mid in T_mid_buckets:
        # write the if statement
        line = utils.line_start + 'if (T <= {:})'.format(T_mid)
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
        for isp in T_mid_buckets[T_mid]:
            sp = specs[isp]
            if T_mid_buckets[T_mid].index(isp):
                line += '\n    + '
            line += '(' + utils.get_array(lang, 'y', isp + 1)
            line += (' * {:.16e} * ('.format(chem.RU / sp.mw) +
                     '{:.16e} + '.format(sp.lo[1]) +
                     'T * ({:.16e} + '.format(2.0 * sp.lo[2]) +
                     'T * ({:.16e} + '.format(3.0 * sp.lo[3]) +
                     '{:.16e} * T)))'.format(4.0 * sp.lo[4]) +
                     ')'
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

        for isp in T_mid_buckets[T_mid]:
            sp = specs[isp]
            if T_mid_buckets[T_mid].index(isp):
                line += '\n    + '
            line += '(' + utils.get_array(lang, 'y', isp + 1)
            line += (' * {:.16e} * ('.format(chem.RU / sp.mw) +
                     '{:.16e} + '.format(sp.hi[1]) +
                     'T * ({:.16e} + '.format(2.0 * sp.hi[2]) +
                     'T * ({:.16e} + '.format(3.0 * sp.hi[3]) +
                     '{:.16e} * T)))'.format(4.0 * sp.hi[4]) +
                     ')'
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

def get_elementary_rxn_dt(lang, specs, rxn, rind, rev_idx, get_array, do_unroll):
    """Write contribution from temperature derivative of reaction rate for
    elementary reaction.

    """

    jline = ''
    if rxn.rev and rxn.rev_par:
        dk_dt = get_rxn_params_dt(rxn, rev=False)
        nu = sum(rxn.reac_nu)

        if dk_dt or nu != 1.0:
            #we actually need to do the dk/dt for both
            line = get_array(lang, 'fwd_rates', rind)
            jline += ' * ('
            if dk_dt:
                jline += dk_dt

            # loop over reactants
            nu = sum(rxn.reac_nu)
            if nu != 1.0:
                if dk_dt and jline:
                    jline += ' + '
                jline += '{}'.format(1. - float(nu))
            jline += ')'

        jline += ' - ' + \
            get_array(lang, 'rev_rates', rev_idx) + \
            ' * ('

        dk_dt = get_rxn_params_dt(rxn, rev=True)
        nu = sum(rxn.prod_nu)
        if dk_dt or nu != 1.0:
            if dk_dt:
                jline += dk_dt

            # product nu sum
            nu = sum(rxn.prod_nu)
            if nu != 1.0:
                if dk_dt and jline:
                    jline += ' + '
                jline += '{}'.format(1. - float(nu))
            jline += ')'
    elif rxn.rev:
        #we don't need the dk/dt for both,
        #so write different to not calculate twice, and instead
        #rely on loading fwd/rev rates again, as they should
        #be cached

        dk_dt = get_rxn_params_dt(rxn, rev=False)
        if dk_dt:
            jline += '('
            jline += get_array(lang, 'fwd_rates', rind)
            if rxn.rev:
                jline += ' - ' + \
                get_array(lang, 'rev_rates', rev_idx)
            jline += ')'
            jline += ' * ('

            jline += dk_dt
            jline += ')'

        # loop over reactants
        nu = sum(rxn.reac_nu)
        if nu != 1.0:
            if jline:
                jline += ' + '
            jline += get_array(lang, 'fwd_rates', rind)
            jline += ' * {}'.format(1. - float(nu))

        if jline:
            jline += ' - '
        jline += get_array(lang, 'rev_rates', rev_idx)
        jline += ' * ('
        # product nu sum
        nu = sum(rxn.prod_nu)
        if nu != 1.0:
            jline += '{} + '.format(1. - float(nu))
        jline += '-T * ('

        # product nu sum
        jline += get_db_dt(lang, specs, rxn, do_unroll)
        jline += '))'
    else:
        #forward only, combine dk/dt and nu sum
        dk_dt = get_rxn_params_dt(rxn, rev=False)
        nu = sum(rxn.reac_nu)
        if dk_dt or nu != 1.0:
            jline += get_array(lang, 'fwd_rates', rind)
            jline += ' * ('
            jline += dk_dt

            # loop over reactants
            nu = sum(rxn.reac_nu)
            if nu != 1.0:
                if jline:
                    jline += ' + '
                jline += '{}'.format(1. - float(nu))

            jline += ')'


    # print line for reaction
    return jline + ')) * rho_inv' + utils.line_end[lang]

def write_cheb_ut(file, lang, rxn):
    line_list = []
    line_list.append('cheb_temp_0 = 1')
    line_list.append('cheb_temp_1 = Pred')
    #start pressure dot product
    for i in range(1, rxn.cheb_n_temp):
        line_list.append(utils.get_array(lang, 'dot_prod', i) +
          '= {:.16e} + Pred * {:.16e}'.format(i * rxn.cheb_par[i, 0],
            i * rxn.cheb_par[i, 1]))

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
        for i in range(1, rxn.cheb_n_temp):
            line_list.append(utils.get_array(lang, 'dot_prod', i)  +
              ' += {:.16e} * cheb_temp_{}'.format(
                i * rxn.cheb_par[i, j], old))

        update_one = not update_one

    line_list.append('cheb_temp_0 = 1.0')
    line_list.append('cheb_temp_1 = 2.0 * Tred')
    #finally, do the temperature portion
    line_list.append('kf = ' + utils.get_array(lang, 'dot_prod', 1) +
                     ' + 2.0 * Tred * ' + utils.get_array(lang, 'dot_prod', 2))

    update_one = True
    for i in range(3, rxn.cheb_n_temp):
        if update_one:
            new = 1
            old = 0
        else:
            new = 0
            old = 1
        line = 'cheb_temp_{}'.format(old)
        line += ' = 2.0 * Tred * cheb_temp_{}'.format(new)
        line += ' - cheb_temp_{}'.format(old)
        line_list.append(line)
        line_list.append('kf += ' + utils.get_array(lang, 'dot_prod', i) +
                         ' * ' + 'cheb_temp_{}'.format(old))

        update_one = not update_one

    line_list = [utils.line_start + line + utils.line_end[lang] for
                  line in line_list]
    file.write(''.join(line_list))

def write_cheb_rxn_dt(file, lang, jline, rxn, rind, rev_idx, specs, get_array, do_unroll):
    # Chebyshev reaction
    tlim_inv_sum = 1.0 / rxn.cheb_tlim[0] + 1.0 / rxn.cheb_tlim[1]
    tlim_inv_sub = 1.0 / rxn.cheb_tlim[1] - 1.0 / rxn.cheb_tlim[0]
    file.write(utils.line_start +
            'Tred = ((2.0 / T) - ' +
            '{:.16e}) / {:.16e}'.format(tlim_inv_sum, tlim_inv_sub) +
            utils.line_end[lang]
            )

    plim_log_sum = (math.log10(rxn.cheb_plim[0]) +
                    math.log10(rxn.cheb_plim[1])
                    )
    plim_log_sub = (math.log10(rxn.cheb_plim[1]) -
                    math.log10(rxn.cheb_plim[0])
                    )
    file.write(utils.line_start +
            'Pred = (2.0 * log10(pres) - ' +
            '{:.16e}) / {:.16e}'.format(plim_log_sum, plim_log_sub) +
            utils.line_end[lang]
            )

    #do U(T) sum
    write_cheb_ut(file, lang, rxn)

    jline += 'kf * ({:.16e} / T)'.format(-2.0 * math.log(10) / tlim_inv_sub)

    jline += ' * (' + get_array(lang, 'fwd_rates', rind)

    if rxn.rev:
        # reverse reaction rate also
        jline += ' - ' + get_array(lang, 'rev_rates', rev_idx)

    jline += ')'
    nu = sum(rxn.reac_nu)
    if nu != 1.0:
        jline += ' + ' + get_array(lang, 'fwd_rates', rind)
        jline += ' * {}'.format(1. - float(nu))

    if rxn.rev:
        jline += ' - ' + get_array(lang, 'rev_rates', rev_idx) + ' * ('
        nu = sum(rxn.prod_nu)
        if nu != 1.0:
            jline += '{} + '.format(1. - float(nu))
        jline += '-T * (' + get_db_dt(lang, specs, rxn, do_unroll)
        jline += '))'

    jline += ')) * rho_inv'
    # print line for reaction
    file.write(jline + utils.line_end[lang])

def write_plog_rxn_dt(file, lang, jline, specs, rxn, rind, rev_idx, get_array, do_unroll):
    # Plog reactions have conditional contribution,
    # depends on pressure range

    (p1, A_p1, b_p1, E_p1) = rxn.plog_par[0]
    file.write(utils.line_start + 'if (pres <= {:.4e}) {{\n'.format(p1))

    # For pressure below the first pressure given, use standard
    # Arrhenius expression.

    # Make copy, but with specific pressure Arrhenius coefficients
    rxn_p = chem.ReacInfo(rxn.rev, rxn.reac, rxn.reac_nu,
                          rxn.prod, rxn.prod_nu,
                          A_p1, b_p1, E_p1
                          )

    file.write(utils.line_start + jline + get_elementary_rxn_dt(lang, specs, rxn_p, rind, rev_idx, get_array, do_unroll))

    for idx, vals in enumerate(rxn.plog_par[:-1]):
        (p1, A_p1, b_p1, E_p1) = vals
        (p2, A_p2, b_p2, E_p2) = rxn.plog_par[idx + 1]

        file.write(utils.line_start + '}} else if ((pres > {:.4e}) '.format(p1) +
                   '&& (pres <= {:.4e})) {{\n'.format(p2)
                   )

        if A_p2 / A_p1 < 0:
            #MIT mechanisms occaisionally have (for some unknown reason)
            #negative A's, so we need to handle the log(K2) - log(K1) term differently
            raise NotImplementedError
        else:
            jline_p = (jline + '({:.16e} + '.format(b_p1) +
                       '{:.16e} / T + '.format(E_p1) +
                       '({:.16e} + '.format(b_p2 - b_p1) +
                       '{:.16e} / T) * '.format(E_p2 - E_p1) +
                       '(log(pres) - {:.16e}) /'.format(math.log(p1)) +
                       ' {:.16e}'.format(math.log(p2) - math.log(p1)) +
                       ')'
                       )

        jline_p += ' * (' + get_array(lang, 'fwd_rates', rind)

        if rxn.rev:
            # reverse reaction rate also
            jline_p += ' - ' + get_array(lang, 'rev_rates', rev_idx)

        jline_p += ') '
        nu = sum(rxn.reac_nu)
        if nu != 1.0:
            jline_p += ' + ' + get_array(lang, 'fwd_rates', rind)
            jline_p += ' * {}'.format(1. - nu)

        if rxn.rev:
            jline_p += ' - ' + get_array(lang, 'rev_rates', rev_idx)
            jline_p += ' * ('
            nu = sum(rxn.prod_nu)
            if nu != 1.0:
                jline_p+= '{} + '.format(1. - nu)
            jline_p += '-T * (' + get_db_dt(lang, specs, rxn, do_unroll)
            jline_p += '))'

        jline_p += ')) * rho_inv'
        #jline_p += '))'
        # print line for reaction
        file.write(utils.line_start + jline_p + utils.line_end[lang])

    (pn, A_pn, b_pn, E_pn) = rxn.plog_par[-1]
    file.write(utils.line_start + '}} else if (pres > {:.4e}) {{\n'.format(pn))

    # For pressure above the final pressure given, use standard
    # Arrhenius expression.

    # Make copy, but with specific pressure Arrhenius coefficients
    rxn_p = chem.ReacInfo(rxn.rev, rxn.reac, rxn.reac_nu,
                          rxn.prod, rxn.prod_nu,
                          A_pn, b_pn, E_pn
                          )

    file.write(utils.line_start + jline + get_elementary_rxn_dt(lang, specs, rxn_p, rind, rev_idx, get_array, do_unroll))

    file.write(utils.line_start + '}\n')


def write_dt_y(file, lang, specs, sp, isp, num_s, touched, sparse_indicies, offset, get_array):
    for k_sp, sp_k in enumerate(specs):
        line = utils.line_start
        lin_index = (num_s + 1) * (k_sp + 1)
        if lang in ['c', 'cuda']:
            line += get_array(lang, 'jac', lin_index)
            if not lin_index in sparse_indicies:
                sparse_indicies.append(lin_index)
        elif lang in ['fortran', 'matlab']:
            line += get_array(lang, 'jac', 0, twod=k_sp + 1)
            if not (1, k_sp + 1) in sparse_indicies:
                sparse_indicies.append((1, k_sp + 1))
        if lang in ['fortran', 'matlab']:
            line += ' = ' + (get_array(lang, 'jac', 0, twod=k_sp + 1) + ' +' if touched[lin_index] else '') + ' -('
        else:
            line += ' {}= -('.format('+' if touched[lin_index] else '')
        touched[lin_index] = True

        if lang in ['c', 'cuda']:
            line += ('' + get_array(lang, 'h', isp) + ' * ('
                     '' + get_array(lang, 'jac', isp + 1 + (num_s + 1) * (k_sp + 1)) +
                     ' * cp_avg * rho' +
                     ' - (' + get_array(lang, 'cp', k_sp) + ' * ' + get_array(lang, 'dy', isp + offset) +
                     ' * {:.8e}))'.format(sp.mw)
                     )
        elif lang in ['fortran', 'matlab']:
            line += ('' + get_array(lang, 'h', isp) + ' * ('
                     '' + get_array(lang, 'jac', isp + 1, twod=k_sp + 1) +
                     ' * cp_avg * rho' +
                     ' - (' + get_array(lang, 'cp', k_sp) + ' * ' + get_array(lang, 'dy', isp + offset) +
                     ' * {:.8e}))'.format(sp.mw)
                     )
        line += ')' + utils.line_end[lang]
        file.write(line)


def write_dt_y_division(file, lang, specs, num_s, get_array):
    line = utils.line_start + utils.comment[lang]
    line += ('Complete dT/dy calculations\n')
    file.write(line)
    file.write(utils.line_start + 'j_temp = 1.0 / (rho * cp_avg * cp_avg)' + utils.line_end[lang])
    for k_sp, sp_k in enumerate(specs):
        line = utils.line_start
        lin_index = (num_s + 1) * (k_sp + 1)
        if lang in ['c', 'cuda']:
            line += get_array(lang, 'jac', lin_index)
            line += ' *= (j_temp)'
        elif lang in ['fortran', 'matlab']:
            line += get_array(lang, 'jac', 0, twod=k_sp + 1)
            line += '= ' + get_array(lang, 'jac', 0, twod=k_sp + 1)
            line += ' * (j_temp)'

        line += utils.line_end[lang]
        file.write(line)
    file.write('\n')


def write_dt_completion(file, lang, specs, offset, get_array):
    line = utils.line_start + utils.comment[lang]
    line += ('Complete dT wrt T calculations\n')
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
        line += ('' + get_array(lang, 'dy', k_sp + offset) +
                 ' * {:.8e}'.format(sp_k.mw) + ' * '
                 )
        line += ('(-working_temp * ' + get_array(lang, 'h', k_sp) +
                 ' / cp_avg + ' + '' + get_array(lang, 'cp', k_sp) + ')'
                 )
        line += (' + ' + get_array(lang, 'jac', k_sp + 1, twod=0) + ' * ' +
                 get_array(lang, 'h', k_sp) + ' * rho'
                 )
        if k_sp != len(specs) - 1:
            if lang == 'fortran':
                line += ' &'
            line += '\n'

    line += ') / (rho * cp_avg)'
    line += utils.line_end[lang]
    file.write(line)


def write_sub_intro(path, lang, number, rate_list, this_rev, this_pdep, this_pdep_has_thd_eff, this_thd,
                     this_troe, this_sri, this_cheb, cheb_dim, this_plog, no_shared
                     ):
    """
    Writes the header and definitions for of any of the various sub-functions

    Returns the opened file
    """
    with open(os.path.join(path, 'jacob_' + str(number) + utils.header_ext[lang]), 'w') as file:
        file.write('#ifndef JACOB_HEAD_{}\n'.format(number) +
                   '#define JACOB_HEAD_{}\n'.format(number) +
                   '\n'
                   '#include "../header.h"\n'
                   '\n' + ('__device__ ' if lang == 'cuda' else '') +
                   ''
                   'void eval_jacob_{} ('.format(number)
                   )
        file.write('const double, const double*')
        for rate in rate_list:
            file.write(', const double*')
        if this_pdep and this_pdep_has_thd_eff:
            file.write(', const double')
        file.write(', const double, const double, const double*, const double, double*'
                   ');\n'
                   '\n'
                   '#endif\n'
                   )
    file = open(os.path.join(path, 'jacob_' + str(number) + utils.file_ext[lang]), 'w')
    file.write('#include <math.h>\n'
               '#include "../header.h"\n'
               '\n'
               )

    line =  '__device__ ' if lang == 'cuda' else ''

    line += ('void eval_jacob_{} (const double pres, '.format(number) +
             'const double * conc')
    for rate in rate_list:
        line += ', const double* ' + rate
    if this_pdep and this_pdep_has_thd_eff:
        line += ', const double m'
    line += ', const double mw_avg, const double rho, const double* dBdT, const double T, double* jac) {'
    file.write(line + '\n')

    if not no_shared and lang == 'cuda':
        file.write(utils.line_start + 'extern __shared__ double shared_temp[]' + utils.line_end[lang])
        # third-body variable needed for reactions
    if this_pdep and this_pdep_has_thd_eff:
        line = utils.line_start
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += ('conc_temp' +
                 utils.line_end[lang]
                 )
        file.write(line)


    # log(T)
    line = utils.line_start
    if lang == 'c':
        line += 'double '
    elif lang == 'cuda':
        line += 'register double '
    line += ('logT = log(T)' +
             utils.line_end[lang]
             )
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

    if this_pdep:
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
        line = ''.join([utils.line_start + 'double {} = 0.0{}'.format(x, utils.line_end[lang]) for x in 'Fcent', 'A', 'B', 'lnF_AB'])
        file.write(line)

    if this_sri:
        line = utils.line_start + 'double X = 0.0' + utils.line_end[lang]
        file.write(line)

    if this_cheb:
        file.write(utils.line_start + 'double Tred, Pred' + utils.line_end[lang])
        file.write(utils.line_start + 'double cheb_temp_0, cheb_temp_1' + utils.line_end[lang])
        file.write(utils.line_start + 'double dot_prod[{}]'.format(cheb_dim) + utils.line_end[lang])

    if this_plog:
        file.write(utils.line_start + 'double kf2' + utils.line_end[lang])

    file.write(utils.line_start + 'double rho_inv = 1.0 / rho' + utils.line_end[lang])

    return file


def write_dy_intros(path, lang, number):
    with open(os.path.join(path, 'jacob_' + str(number) + utils.header_ext[lang]), 'w') as file:
        file.write('#ifndef JACOB_HEAD_{}\n'.format(number) +
                   '#define JACOB_HEAD_{}\n'.format(number) +
                   '\n'
                   '#include "../header.h"\n'
                   '\n' +
                   ('__device__ ' if lang == 'cuda' else '') +
                   'void eval_jacob_{} ('.format(number)
                   )
        file.write('const double, const double, const double, const double*, const double*, const double*, double*);\n'
                   '\n'
                   '#endif\n'
                   )
    file = open(os.path.join(path, 'jacob_' + str(number) + utils.file_ext[lang]), 'w')
    file.write('#include "../header.h"\n'
               '\n'
               )

    line = '__device__ ' if lang == 'cuda' else ''

    line += (
        'void eval_jacob_{} (const double mw_avg, const double rho, const double cp_avg, const double* dy, '
        'const double* h, const double* cp, double* jac) '.format(
            number))
    line += '{\n'
    line += utils.line_start
    if lang == 'cuda':
        line += 'register '
    line += 'double rho_inv = 1.0 / rho;'
    file.write(line + utils.line_end[lang])
    return file


def write_jacobian(path, lang, specs, reacs, splittings=None, smm=None):
    """Write Jacobian subroutine in desired language.

    Parameters
    ----------
    path : str
        Path to build directory for file.
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Programming language.
    specs : list of SpecInfo
        List of species in the mechanism.
    reacs : list of ReacInfo
        List of reactions in the mechanism.
    splittings : list of int
        If not None and lang == 'cuda', this will be used to partition the sub jacobian routines
    smm : shared_memory_manager, optional
        If not None, use this to manage shared memory optimization

    Returns
    -------
    None

    """

    if lang == 'cuda':
        do_unroll = len(specs) > CUDAParams.Jacob_Unroll
    elif lang == 'c':
        do_unroll = len(specs) > CParams.C_Jacob_Unroll
    if do_unroll:
        # make paths for separate jacobian files
        utils.create_dir(os.path.join(path, 'jacobs'))

    # first write header file
    if lang == 'c':
        file = open(path + 'jacob.h', 'w')
        file.write('#ifndef JACOB_HEAD\n'
                   '#define JACOB_HEAD\n'
                   '\n'
                   '#include "header.h"\n' + 
                    ('#include "jacobs/jac_include.h"\n' if
                    do_unroll else '') +
                   '#include "chem_utils.h"\n'
                   '#include "rates.h"\n'
                   '\n'
                   'void eval_jacob (const double, const double, '
                   'const double*, double*);\n'
                   '\n'
                   '#endif\n'
                   )
        file.close()
    elif lang == 'cuda':
        file = open(path + 'jacob.cuh', 'w')
        file.write('#ifndef JACOB_HEAD\n'
                   '#define JACOB_HEAD\n'
                   '\n'
                   '#include "header.h"\n' +
                   ('#include "jacobs/jac_include.cuh"\n' if
                    do_unroll else '') +
                   '#include "chem_utils.cuh"\n'
                   '#include "rates.cuh"\n'
                   '#include "gpu_macros.cuh"\n'
                   '\n'
                   '__device__ void eval_jacob (const double, const double, '
                   'const double*, double*);\n'
                   '\n'
                   '#endif\n'
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
    file = open(path + filename, 'w')

    # header files
    if lang == 'c':
        file.write('#include "jacob.h"\n'
                   '\n'
                   )
    elif lang == 'cuda':
        file.write('#include "jacob.cuh"\n'
                   '\n'
                   )

    line = ''
    if lang == 'cuda':
        line = '__device__ '

    offset = 0

    if lang in ['c', 'cuda']:
        line += ('void eval_jacob (const double t, const double pres, '
                 'const double * y, double * jac) {\n\n')
    elif lang == 'fortran':
        line += 'subroutine eval_jacob (t, pres, y, jac)\n\n'

        # fortran needs type declarations
        line += ('  implicit none\n'
                 '  integer, parameter :: wp = kind(1.0d0)'
                 '  real(wp), intent(in) :: t, pres, y({})\n'.format(num_s) +
                 '  real(wp), intent(out) :: jac({0},{0})\n'.format(num_s) +
                 '  \n'
                 '  real(wp) :: T, rho, cp_avg, logT\n'
                 )
        if any(reacs[rxn].thd_body for rxn in rev_reacs):
            line += '  real(wp) :: m\n'
        line += ('  real(wp), dimension({}) :: '.format(num_s) +
                 'conc, cp, h, dy\n'
                 '  real(wp), dimension({}) :: rxn_rates\n'.format(num_r) +
                 '  real(wp), dimension({}) :: pres_mod\n'.format(num_pdep)
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
    file.write(utils.line_start + utils.comment[lang] + ' species molar concentrations\n')
    if lang in ['c', 'cuda']:
        file.write(utils.line_start + 'double conc[{}];\n'.format(num_s))
    elif lang == 'matlab':
        file.write(utils.line_start + 'conc = zeros({},1);\n'.format(num_s)
                   )
    file.write(utils.line_start + 'eval_conc (y[0], pres, &y[1], &mw_avg, &rho, conc);\n\n')

    rate_list = ['fwd_rates']
    if len(rev_reacs):
        rate_list.append('rev_rates')
    if len(pdep_reacs):
        rate_list.append('pres_mod')
    rate_list.append('dy')
    file.write(utils.line_start + utils.comment[lang] + ' evaluate reaction rates\n')
    # evaluate forward and reverse reaction rates
    if lang in ['c', 'cuda']:
        file.write(utils.line_start + 'double fwd_rates[{}];\n'.format(num_r))
        if lang == 'cuda' and num_rev == 0:
            file.write(utils.line_start + 'double* rev_rates = 0;\n')
        else:
            file.write(utils.line_start + 'double rev_rates[{}];\n'.format(num_rev))
        file.write(utils.line_start + 'eval_rxn_rates (T, pres, conc, fwd_rates, '
                   'rev_rates);\n')
    elif lang == 'fortran':
            file.write(utils.line_start +
                       'call eval_rxn_rates (T, pres, conc, fwd_rates, '
                       'rev_rates)\n'
                       )
    elif lang == 'matlab':
        file.write(utils.line_start + '[fwd_rates, rev_rates] = eval_rxn_rates '
                   '(T, pres, conc);\n'
                   )
    file.write('\n')

    if lang == 'c' or (lang == 'cuda' and num_pdep != 0):
        file.write(utils.line_start + 'double pres_mod[{}];\n'.format(num_pdep))
    elif lang == 'cuda':
        file.write(utils.line_start + 'double* pres_mod = 0;\n')


    if len(pdep_reacs):
        file.write(utils.line_start + utils.comment[lang] + 'get pressure modifications to reaction rates\n')
        # evaluate third-body and pressure-dependence reaction modifications
        if lang in ['c', 'cuda']:
            file.write(utils.line_start + 'get_rxn_pres_mod (T, pres, conc, pres_mod);\n')
        elif lang == 'fortran':
            file.write(utils.line_start + 'call get_rxn_pres_mod (T, pres, conc, pres_mod)\n')
        elif lang == 'matlab':
            file.write(utils.line_start + 'pres_mod = get_rxn_pres_mod (T, pres, conc, '
                       'pres_mod);\n'
                       )
    file.write('\n')

    # evaluate species rates
    file.write(utils.line_start + utils.comment[lang] + ' evaluate rate of change of species molar '
                   'concentration\n')
    if lang in ['c', 'cuda']:
        file.write(utils.line_start + 'double dy[{}];\n'.format(num_s))
        file.write(utils.line_start + 'eval_spec_rates (fwd_rates, rev_rates, '
                   'pres_mod, dy);\n'
                   )
    elif lang == 'fortran':
        file.write(utils.line_start + 'call eval_spec_rates (fwd_rates, rev_rates, '
                   'pres_mod, dy)\n'
                   )
    elif lang == 'matlab':
        file.write(utils.line_start + 'dy = eval_spec_rates(fwd_rates, '
                   'rev_rates, pres_mod);\n'
                   )
    file.write('\n')

    # third-body variable needed for reactions
    if any((rxn.pdep and rxn.pdep_sp == '') or rxn.thd_body for rxn in reacs):
        line = utils.line_start
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += ('m = pres / ({:.8e} * T)'.format(chem.RU) +
                 utils.line_end[lang]
                 )
        file.write(line)

        if not do_unroll:
            line = utils.line_start
            if lang == 'c':
                line += 'double '
            elif lang == 'cuda':
                line += 'register double '
            line += ('conc_temp' +
                     utils.line_end[lang]
                     )
            file.write(line)


    # log(T)
    line = utils.line_start
    if lang == 'c':
        line += 'double '
    elif lang == 'cuda':
        line += 'register double '
    line += ('logT = log(T)' +
             utils.line_end[lang]
             )
    file.write(line)

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
    if not do_unroll:
        file.write(line)
    else:
        line = ''

    if any(rxn.pdep for rxn in reacs) and not do_unroll:
        line = utils.line_start
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'pres_mod_temp = 0.0' + utils.line_end[lang]
        file.write(line)

    # if any reverse reactions, will need Kc
    if rev_reacs and not do_unroll:
        line = utils.line_start
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        file.write(line + 'Kc = 0.0' + utils.line_end[lang])
        file.write(line + 'kr = 0' + utils.line_end[lang])

    # pressure-dependence variables
    if any(rxn.pdep for rxn in reacs) and not do_unroll:
        line = utils.line_start
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'Pr = 0.0' + utils.line_end[lang]
        file.write(line)

    if any(rxn.troe for rxn in reacs) and not do_unroll:
        line = ''.join(['  double {} = 0.0{}'.format(x, utils.line_end[lang]) for x in 'Fcent', 'A', 'B', 'lnF_AB'])
        file.write(line)

    if any(rxn.sri for rxn in reacs) and not do_unroll:
        line = utils.line_start + 'double X = 0.0' + utils.line_end[lang]
        file.write(line)

    if any(rxn.cheb for rxn in reacs) and not do_unroll:
        file.write(utils.line_start + 'double Tred, Pred' + utils.line_end[lang])
        file.write(utils.line_start + 'double cheb_temp_0, cheb_temp_1' + utils.line_end[lang])
        dim = max(rxn.cheb_n_temp for rxn in reacs if rxn.cheb)
        file.write(utils.line_start + 'double dot_prod[{}]'.format(dim) + utils.line_end[lang])

    if any(rxn.plog for rxn in reacs) and not do_unroll:
        file.write(utils.line_start + 'double kf2' + utils.line_end[lang])

    if not do_unroll:
        line = utils.line_start
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'rho_inv = 1.0 / rho' + utils.line_end[lang]
        file.write(line)

    # variables for equilibrium constant derivatives, if needed
    dBdT_flag = [False for sp in specs]

    # define dB/dT's
    write_db_dt_def(file, lang, specs, reacs, rev_reacs, dBdT_flag, do_unroll)

    line = ''

    ###################################
    # now begin Jacobian evaluation
    ###################################

    # whether this jacobian index has been modified
    touched = [False for i in range((len(specs) + 1) * (len(specs) + 1))]
    sparse_indicies = []

    last_split_index = None
    jac_count = 0
    next_fn_index = 0
    batch_has_thd = False
    last_conc_temp = None
    ###################################
    # partial derivatives of reactions
    ###################################
    for rind, rxn in enumerate(reacs):
        if do_unroll and (rind == next_fn_index):
            # clear conc temp
            last_conc_temp = None
            file_store = file
            # get next index
            next_fn_index = rind + splittings[0]
            splittings = splittings[1:]
            # get flags
            rev = False
            pdep = False
            thd = False
            troe = False
            sri = False
            cheb = False
            plog = False
            pdep_thd_eff = False
            for ind_next in range(rind, next_fn_index):
                if reacs[ind_next].rev:
                    rev = True
                if reacs[ind_next].pdep:
                    pdep = True
                    if reacs[ind_next].thd_body_eff:
                        pdep_thd_eff = True
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
            batch_has_m = pdep and pdep_thd_eff

            dim = None
            if cheb:
                dim = max(rxn.cheb_n_temp for rxn in reacs if rxn.cheb)
            # write the specific evaluator for this reaction
            file = write_sub_intro(os.path.join(path, 'jacobs'), lang, jac_count,
                                    rate_list, rev, pdep, pdep_thd_eff, 
                                    thd, troe, sri,
                                    cheb, dim, plog, smm is None
                                    )

        if lang == 'cuda' and smm is not None:
            variable_list, usages = calculate_shared_memory(rind, rxn, specs,
                                                            reacs, rev_reacs,
                                                            pdep_reacs
                                                            )
            smm.load_into_shared(file, variable_list, usages)


        ######################################
        # with respect to temperature
        ######################################

        write_dt_comment(file, lang, rind)

        # first we need any pres mod terms
        jline = ''
        pind = None
        if rxn.pdep:
            pind = pdep_reacs.index(rind)
            last_conc_temp = write_pr(file, lang, specs, reacs, pdep_reacs,
                                      rxn, get_array, last_conc_temp
                                      )

            # dF/dT
            if rxn.troe:
                write_troe(file, lang, rxn)
            elif rxn.sri:
                write_sri(file, lang)

            jline = get_pdep_dt(lang, rxn, rev_reacs, rind, pind, get_array)

        elif rxn.thd_body:
            # third body reaction
            pind = pdep_reacs.index(rind)

            jline = utils.line_start + 'j_temp = ((-' + get_array(lang, 'pres_mod', pind)
            jline += ' * '

            if rxn.rev:
                # forward and reverse reaction rates
                jline += '(' + get_array(lang, 'fwd_rates', rind)
                jline += ' - ' + \
                    get_array(lang, 'rev_rates', rev_reacs.index(rind))
                jline += ')'
            else:
                # forward reaction rate only
                jline += get_array(lang, 'fwd_rates', rind)

            jline += ' / T) + (' + get_array(lang, 'pres_mod', pind)

        else:
            if lang in ['c', 'cuda', 'matlab']:
                jline += '  j_temp = ((1.0'
            elif lang in ['fortran']:
                jline += '  j_temp = ((1.0_wp'

        jline += ' / T) * ('

        if rxn.plog:

            write_plog_rxn_dt(file, lang, jline, specs, rxn, rind,
                rev_reacs.index(rind) if rxn.rev else None, get_array, do_unroll)

        elif rxn.cheb:
            write_cheb_rxn_dt(file, lang, jline, rxn, rind,
                rev_reacs.index(rind) if rxn.rev else None,
                specs, get_array, do_unroll)

        else:
            jline += get_elementary_rxn_dt(lang, specs, rxn, rind,
                rev_reacs.index(rind) if rxn.rev else None, get_array, do_unroll)
            file.write(jline)


        for k_sp in set(rxn.reac + rxn.prod):
            sp_k = specs[k_sp]
            line = utils.line_start
            nu = get_nu(k_sp, rxn)
            if nu == 0:
                continue
            if lang in ['c', 'cuda']:
                line += (get_array(lang, 'jac', k_sp + 1) +
                         ' {}= {}j_temp{} * {:.8e}'.format('+' if touched[k_sp + 1] else '',
                                                           '' if nu == 1 else ('-' if nu == -1 else ''),
                                                           ' * {}'.format(float(nu)) if nu != 1 and nu != -1 else '',
                                                           sp_k.mw)
                         )
            elif lang in ['fortran', 'matlab']:
                # NOTE: I believe there was a bug here w/ the previous
                # fortran/matlab code (as it looks like it would be zero
                # indexed)
                line += (get_array(lang, 'jac', k_sp + 1, twod=0) + ' = ' +
                         (get_array(lang, 'jac', k_sp + 1, twod=0) + ' + ' if touched[k_sp + 1] else '') +
                         ' {}j_temp{} * {:.8e}'.format('' if nu == 1 else ('-' if nu == -1 else ''),
                                                       ' * {}'.format(float(nu)) if nu != 1 and nu != -1 else '',
                                                       sp_k.mw)
                         )
            file.write(line + utils.line_end[lang])
            touched[k_sp + 1] = True
            if lang in ['c', 'cuda']:
                if k_sp + 1 not in sparse_indicies:
                    sparse_indicies.append(k_sp + 1)
            elif lang in ['fortran', 'matlab']:
                if (k_sp + 1, 1) not in sparse_indicies:
                    sparse_indicies.append((k_sp + 1, 1))

        file.write('\n')

        ######################################
        # with respect to species
        ######################################
        write_dy_comment(file, lang, rind)

        if rxn.rev and not rxn.rev_par:
            # need to find Kc
            write_kc(file, lang, specs, rxn)

        # need to write the dr/dy parts (independent of any species)
        alphaij_hat = write_dr_dy(file, lang, rev_reacs, rxn, rind, pind, len(specs), get_array)

        # write the forward / backwards rates:
        write_rates(file, lang, rxn)

        # now loop through each species
        for j_sp, sp_j in enumerate(specs):
            for k_sp in set(rxn.reac + rxn.prod):
                sp_k = specs[k_sp]

                nu = get_nu(k_sp, rxn)

                # sparse indexes
                if lang in ['c', 'cuda']:
                    if k_sp + 1 + (num_s + 1) * (j_sp + 1) not in sparse_indicies:
                        sparse_indicies.append(k_sp + 1 + (num_s + 1) * (j_sp + 1))
                elif lang in ['fortran', 'matlab']:
                    if (k_sp + 1, j_sp + 1) not in sparse_indicies:
                        sparse_indicies.append((k_sp + 1, j_sp + 1))

                if nu == 0:
                    continue

                working_temp = ''
                mw_frac = (sp_k.mw / sp_j.mw)* float(nu)
                if mw_frac == -1.0:
                    working_temp += ' -'
                elif mw_frac != 1.0:
                    working_temp += ' {:.16e} * '.format(mw_frac)
                else:
                    working_temp += ' '

                working_temp += '('

                working_temp += write_dr_dy_species(lang, specs, rxn, pind, j_sp, sp_j, alphaij_hat, rind, rev_reacs,
                                                    get_array)

                working_temp += ')'

                lin_index = k_sp + 1 + (num_s + 1) * (j_sp + 1)
                jline = utils.line_start
                if lang in ['c', 'cuda']:
                    jline += (get_array(lang, 'jac', lin_index) +
                              ' {}= '.format('+' if touched[lin_index] else '')
                              )
                elif lang in ['fortran', 'matlab']:
                    jline += (get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) +
                              (' = ' + get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) if touched[k_sp + 1] else '')
                              + ' + ')

                if not touched[lin_index]:
                    touched[lin_index] = True

                jline += working_temp
                jline += utils.line_end[lang]

                file.write(jline)
                jline = ''

        file.write('\n')

        if lang == 'cuda' and smm is not None:
            smm.mark_for_eviction(variable_list)

        if do_unroll and (rind == next_fn_index - 1 or rind == len(reacs) - 1):
            # switch back
            file.write('}\n\n')
            file = file_store
            file.write('  eval_jacob_{}('.format(jac_count))
            jac_count += 1
            line = ('pres, conc')
            for rate in rate_list:
                line += ', ' + rate
            if batch_has_m:
                line += ', m'
            line += ', mw_avg, rho, dBdT, T, jac)'
            file.write(line + utils.line_end[lang])

    ###################################
    # Partial derivatives of temperature (energy equation)
    ###################################

    # evaluate enthalpy
    if lang in ['c', 'cuda']:
        file.write('  // species enthalpies\n'
                   '  double h[{}];\n'.format(num_s) +
                   '  eval_h(T, h);\n')
    elif lang == 'fortran':
        file.write('  ! species enthalpies\n'
                   '  call eval_h(T, h)\n'
                   )
    elif lang == 'matlab':
        file.write('  % species enthalpies\n'
                   '  h = eval_h(T);\n'
                   )
    file.write('\n')

    # evaluate specific heat
    if lang in ['c', 'cuda']:
        file.write('  // species specific heats\n'
                   '  double cp[{}];\n'.format(num_s) +
                   '  eval_cp(T, cp);\n')
    elif lang == 'fortran':
        file.write('  ! species specific heats\n'
                   '  call eval_cp(T, cp)\n'
                   )
    elif lang == 'matlab':
        file.write('  % species specific heats\n'
                   '  cp = eval_cp(T);\n'
                   )
    file.write('\n')

    # average specific heat
    if lang == 'c':
        file.write('  // average specific heat\n'
                   '  double cp_avg;\n'
                   )
    elif lang == 'cuda':
        file.write('  // average specific heat\n'
                   '  register double cp_avg;\n'
                   )
    elif lang == 'fortran':
        file.write('  ! average specific heat\n')
    elif lang == 'matlab':
        file.write('  % average specific heat\n')

    line = utils.line_start + 'cp_avg = '
    isfirst = True
    for sp in specs:
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
        if not isfirst:
            line += ' + '
        line += ('(' + get_array(lang, 'y', isp + 1) + ' * ' + '' + get_array(lang, 'cp', isp) + ')')

        isfirst = False
    line += utils.line_end[lang]
    file.write(line)

    # set jac[0] = 0
    # set to zero
    line = utils.line_start
    if lang in ['c', 'cuda']:
        line += get_array(lang, 'jac', 0) + ' = 0.0'
        sparse_indicies.append(0)
    elif lang == 'fortran':
        line += get_array(lang, 'jac', 0, twod=0) + ' = 0.0_wp'
        sparse_indicies.append((1, 1))
    elif lang == 'matlab':
        line += get_array(lang, 'jac', 0, twod=0) + ' = 0.0'
        sparse_indicies.append((1, 1))
    touched[0] = True
    line += utils.line_end[lang]
    file.write(line)

    line = utils.line_start
    if lang == 'c':
        line += 'double '
    elif lang == 'cuda':
        line += 'register double '
    line += 'working_temp = 0.0' + utils.line_end[lang]
    file.write(line)

    next_fn_index = 0
    # need to finish the dYk/dYj's
    write_dy_y_finish_comment(file, lang)
    for k_sp, sp_k in enumerate(specs):
        if do_unroll and k_sp == next_fn_index:
            store_file = file
            file = write_dy_intros(os.path.join(path, 'jacobs'), lang, jac_count)
            unroll_len = CParams.C_Jacob_Unroll if lang == 'c' else CUDAParams.Jacob_Unroll
            next_fn_index += min(unroll_len, len(specs) - k_sp)

        for j_sp, sp_j in enumerate(specs):
            lin_index = k_sp + 1 + (num_s + 1) * (j_sp + 1)
            # see if this combo matters
            if touched[lin_index]:
                line = utils.line_start
                # zero out if unused
                if lang in ['c', 'cuda'] and not lin_index in sparse_indicies:
                    file.write(get_array(lang, 'jac', lin_index) + ' = 0.0' + utils.line_end[lang])
                elif lang in ['fortran', 'matlab'] and not (k_sp + 1, j_sp + 1) in sparse_indicies:
                    file.write(get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) + ' = 0.0' + utils.line_end[lang])
                else:
                    # need to finish
                    if lang in ['c', 'cuda']:
                        line += get_array(lang, 'jac', lin_index) + ' += '
                    elif lang in ['fortran', 'matlab']:
                        line += (get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) +
                                 ' = ' + get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) + ' + ')
                    line += '(' + get_array(lang, 'dy', k_sp + offset)
                    line += ' * mw_avg * {:.16e} * rho_inv)'.format(sp_k.mw / sp_j.mw)
                    line += utils.line_end[lang]
                    file.write(line)


            ######################################
            # Derivative with respect to species
            ######################################
            if touched[lin_index]:
                line = utils.line_start
                lin_index = (num_s + 1) * (j_sp + 1)
                if lang in ['c', 'cuda']:
                    line += get_array(lang, 'jac', lin_index)
                    if not lin_index in sparse_indicies:
                        sparse_indicies.append(lin_index)
                elif lang in ['fortran', 'matlab']:
                    line += get_array(lang, 'jac', 0, twod=j_sp + 1)
                    if not (1, j_sp + 1) in sparse_indicies:
                        sparse_indicies.append((1, j_sp + 1))
                if lang in ['fortran', 'matlab']:
                    line += ' = ' + (get_array(lang, 'jac', 0, twod=j_sp + 1)
                                     + ' +' if touched[lin_index] else ''
                                     ) + ' -('
                else:
                    line += ' {}= -('.format('+' if touched[lin_index] else '')
                touched[lin_index] = True

                if lang in ['c', 'cuda']:
                    line += ('' + get_array(lang, 'h', k_sp) + ' * ('
                             '' + get_array(lang, 'jac',
                             k_sp + 1 + (num_s + 1) * (j_sp + 1)) +
                             ' * cp_avg * rho' +
                             ' - (' + get_array(lang, 'cp', j_sp) +
                             ' * ' + get_array(lang, 'dy', k_sp + offset) +
                             ' * {:.8e}))'.format(sp_k.mw)
                             )
                elif lang in ['fortran', 'matlab']:
                    line += ('' + get_array(lang, 'h', k_sp) + ' * ('
                             '' + get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) +
                             ' * cp_avg * rho' +
                             ' - (' + get_array(lang, 'cp', j_sp) + ' * ' +
                             get_array(lang, 'dy', k_sp + offset) +
                             ' * {:.8e}))'.format(sp_k.mw)
                             )
                line += ')' + utils.line_end[lang]
                file.write(line)

        if do_unroll and k_sp == next_fn_index - 1:
            # switch back
            file.write('}\n\n')
            file = file_store
            file.write('  eval_jacob_{}('.format(jac_count))
            jac_count += 1
            line = 'mw_avg, rho, cp_avg, dy, h, cp, jac)'
            file.write(line + utils.line_end[lang])

    ######################################
    # Derivatives with respect to temperature
    ######################################
    write_dcp_dt(file, lang, specs, sparse_indicies)

    ######################################
    # Derivative with respect to species
    ######################################
    # write_dt_y_comment(file, lang, sp)
    # write_dt_y(file, lang, specs, sp, isp, num_s, touched, sparse_indicies, offset, get_array)

    file.write('\n')

    # next need to divide everything out
    write_dt_y_division(file, lang, specs, num_s, get_array)

    # finish the dT entry
    write_dt_completion(file, lang, specs, offset, get_array)

    if lang in ['c', 'cuda']:
        file.write('} // end eval_jacob\n\n')
    elif lang == 'fortran':
        file.write('end subroutine eval_jacob\n\n')
    elif lang == 'matlab':
        file.write('end\n\n')

    file.close()

    # remove any duplicates
    sparse_indicies = list(set(sparse_indicies))

    # create include file
    if do_unroll:
        with open(os.path.join(path, 'jacobs', 'jac_include' + utils.header_ext[lang]), 'w') as tempfile:
            tempfile.write('#ifndef JAC_INCLUDE_H\n'
                           '#define JAC_INCLUDE_H\n')
            for i in range(jac_count):
                tempfile.write('#include "jacob_{}{}"\n'.format(i, utils.header_ext[lang]))
            tempfile.write('#endif\n\n')

        with open(os.path.join(path, 'jacobs', 'jac_list_{}'.format(lang)), 'w') as tempfile:
            tempfile.write(' '.join(['jacob_{}{}'.format(i, utils.file_ext[lang]) for i in range(jac_count)]))
    return sparse_indicies


def write_sparse_multiplier(path, lang, sparse_indicies, nvars):
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
        How many variables in the Jacobian matrix

    Returns
    -------
    None

    """

    sorted_and_cleaned = sorted(list(set(sparse_indicies)))
    # first write header file
    if lang == 'c':
        file = open(path + 'sparse_multiplier.h', 'w')
        file.write('#ifndef SPARSE_HEAD\n'
                   '#define SPARSE_HEAD\n')
        file.write('\n#define N_A {}'.format(len(sorted_and_cleaned)))
        file.write(
            '\n'
            '#include "header.h"\n'
            '\n'
            'void sparse_multiplier (const double *, const double *, double*);\n'
            '\n'
            '#ifdef COMPILE_TESTING_METHODS\n'
            '  int test_sparse_multiplier();\n'
            '#endif\n'
            '\n'
            '#endif\n'
        )
        file.close()
    elif lang == 'cuda':
        file = open(path + 'sparse_multiplier.cuh', 'w')
        file.write('#ifndef SPARSE_HEAD\n'
                   '#define SPARSE_HEAD\n')
        file.write('\n#define N_A {}'.format(len(sorted_and_cleaned)))
        file.write(
            '\n'
            '#include "header.h"\n'
            '\n'
            '__device__ void sparse_multiplier (const double *, const double *, double*);\n'
            '#ifdef COMPILE_TESTING_METHODS\n'
            '  __device__ int test_sparse_multiplier();\n'
            '#endif\n'
            '\n'
            '#endif\n'
        )
        file.close()
    else:
        raise NotImplementedError

    # create file depending on language
    filename = 'sparse_multiplier' + utils.file_ext[lang]
    file = open(path + filename, 'w')

    file.write('#include "sparse_multiplier.{}h"\n\n'.format('cu' if lang == 'cuda' else ''))

    if lang == 'cuda':
        file.write('__device__\n')

    file.write(
        "void sparse_multiplier(const double * A, const double * Vm, double* w) {\n")

    if lang == 'cuda':
        """optimize for cache reusing"""
        touched = [False for i in range(nvars)]
        for i in range(nvars):
            # get all indicies that belong to row i
            i_list = [x for x in sorted_and_cleaned if int(round(x / nvars)) == i]
            for index in i_list:
                file.write(' ' + utils.get_array(lang, 'w', index % nvars) + ' {}= '.format(
                    '+' if touched[index % nvars] else ''))
                file.write(' ' + utils.get_array(lang, 'A', index) + ' * '
                           + utils.get_array(lang, 'Vm', i) + utils.line_end[lang])
                touched[index % nvars] = True
        file.write("}\n")
    else:
        for i in range(nvars):
            # get all indicies that belong to row i
            i_list = [x for x in sorted_and_cleaned if x % nvars == i]
            if not len(i_list):
                file.write(
                    '  ' + utils.get_array(lang, 'w', i) + ' = 0' + utils.line_end[lang])
                continue
            file.write('  ' + utils.get_array(lang, 'w', i) + ' = ')
            for index in i_list:
                if i_list.index(index):
                    file.write(" + ")
                file.write(' ' + utils.get_array(lang, 'A', index) + ' * '
                           + utils.get_array(lang, 'Vm', int(index / nvars)))
            file.write(";\n")
        file.write("}\n")

    file.close()


def create_jacobian(lang, mech_name, therm_name=None, optimize_cache=False,
                    initial_state="", num_blocks=8, num_threads=64,
                    no_shared=False, L1_preferred=True, multi_thread=1,
                    force_optimize=False, build_path='./out/'
                    ):
    """Create Jacobian subroutine from mechanism.

    Parameters
    ----------
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Language type.
    mech_name : str
        Reaction mechanism filename (e.g. 'mech.dat').
    therm_name : str, optional
        Thermodynamic database filename (e.g. 'therm.dat')
        or nothing if info in mechanism file.
    optimize_cache : bool, optional
        If true, use the greedy optimizer to attempt to
        improve cache hit rates
    initial_state : str, optional
        A comma separated list of the initial conditions to use in form T,P,X (e.g. 800,1,H2=1.0,O2=0.5)
        Temperature in K, P in atm
    num_blocks : int, optional
        The target number of blocks / sm to achieve for cuda
    num_threads : int, optional
        The target number of threads / blck to achieve for cuda
    no_shared : bool, optional
        If true, do not use the shared_memory_manager to attempt to optimize for CUDA
    L1_preferred : bool, optional
        If true, prefer a larger L1 cache and a smaller shared memory size for CUDA
    multi_thread : int, optional
        The number of threads to use during optimization
    force_optimize : bool, optional
        If true, redo the cache optimization even if the same mechanism
    build_path : str, optional
        The output directory for the jacobian files

    Returns
    -------
    None

    """

    lang = lang.lower()
    if lang not in utils.langs:
        print('Error: language needs to be one of: ')
        for l in utils.langs:
            print(l)
        sys.exit(2)

    # create output directory if none exists
    utils.create_dir(build_path)

    # Interpret reaction mechanism file, depending on Cantera or
    # Chemkin format.
    if mech_name.endswith(tuple(['.cti', '.xml'])):
        [elems, specs, reacs] = mech.read_mech_ct(mech_name)
    else:
        [elems, specs, reacs] = mech.read_mech(mech_name, therm_name)

    if optimize_cache:
        specs, reacs, rxn_rate_order, pdep_rate_order, \
        spec_rate_order, old_spec_order, \
        old_rxn_order = cache.greedy_optimizer(lang, specs, reacs,
                                               multi_thread, force_optimize,
                                               build_path
                                               )
    else:
        spec_rate_order = [(range(len(specs)), range(len(reacs)))]
        rxn_rate_order = range(len(reacs))
        if any(r.pdep or r.thd_body for r in reacs):
            pdep_rate_order = [x for x in range(len(reacs))
                               if reacs[x].pdep or reacs[x].thd_body
                               ]
        else:
            pdep_rate_order = None
        old_spec_order = range(len(specs))
        old_rxn_order = range(len(reacs))

    the_len = len(reacs)
    splittings = []
    unroll_len = CParams.C_Jacob_Unroll if lang == 'c' else CUDAParams.Jacob_Unroll
    while the_len > 0:
        splittings.append(min(unroll_len, the_len))
        the_len -= unroll_len

    if lang == 'cuda':
        CUDAParams.write_launch_bounds(build_path, num_blocks, num_threads,
                                       L1_preferred, no_shared
                                       )
    smm = None
    if lang == 'cuda' and not no_shared:
        smm = shared.shared_memory_manager(num_blocks, num_threads,
                                           L1_preferred
                                           )

    #reassign the reaction's product / reactant / third body list
    # to integer indexes for speed
    utils.reassign_species_lists(reacs, specs)

    ## now begin writing subroutines

    # print reaction rate subroutine
    rate.write_rxn_rates(build_path, lang, specs, reacs, rxn_rate_order, smm)

    # if third-body/pressure-dependent reactions,
    # print modification subroutine
    if next((r for r in reacs if (r.thd_body or r.pdep)), None):
        rate.write_rxn_pressure_mod(build_path, lang, specs, reacs,
                                    pdep_rate_order, smm
                                    )

    # write species rates subroutine
    rate.write_spec_rates(build_path, lang, specs, reacs, spec_rate_order, smm)

    # write chem_utils subroutines
    rate.write_chem_utils(build_path, lang, specs)

    # write derivative subroutines
    rate.write_derivs(build_path, lang, specs, reacs)

    # write mass-mole fraction conversion subroutine
    rate.write_mass_mole(build_path, lang, specs)

    # write header file
    aux.write_header(build_path, lang)

    # write mechanism initializers and testing methods
    aux.write_mechanism_initializers(build_path, lang, specs, reacs, initial_state, old_spec_order, old_rxn_order,
                                     optimize_cache)

    # write Jacobian subroutine
    sparse_indicies = write_jacobian(build_path, lang, specs, reacs, splittings, smm)

    write_sparse_multiplier(build_path, lang, sparse_indicies, len(specs) + 1)

    return 0


if __name__ == "__main__":
    # command line arguments
    parser = ArgumentParser(description='pyJac: Generates source code '
                                        'for analytical Jacobian.')
    parser.add_argument('-l', '--lang',
                        type=str,
                        choices=utils.langs,
                        required=True,
                        help='Programming language for output '
                             'source files.')
    parser.add_argument('-i', '--input',
                        type=str,
                        required=True,
                        help='Input mechanism filename (e.g., mech.dat).')
    parser.add_argument('-t', '--thermo',
                        type=str,
                        default=None,
                        help='Thermodynamic database filename (e.g., '
                             'therm.dat), or nothing if in mechanism.')
    parser.add_argument('-ic', '--initial-conditions',
                        type=str,
                        dest='initial_conditions',
                        default='',
                        required=False,
                        help='A comma separated list of initial initial conditions to set in the '
                             'set_same_initial_conditions method. \
                                Expected Form: T,P,Species1=...,Species2=...,...\n\
                                Temperature in K\n\
                                Pressure in Atm\n\
                                Species in moles')
    # cuda specific
    parser.add_argument('-nco', '--no-cache-optimizer',
                        dest='cache_optimizer',
                        action='store_false',
                        default=True,
                        help='Do not attempt to optimize cache store/loading via use '
                             'of a greedy selection algorithm.')
    parser.add_argument('-nosmem', '--no-shared-memory',
                        dest='no_shared',
                        action='store_true',
                        default=False,
                        help='Use this option to turn off attempted shared memory acceleration for CUDA')
    parser.add_argument('-pshare', '--prefer-shared',
                        dest='L1_preferred',
                        action='store_false',
                        default=True,
                        help='Use this option to allocate more space for shared memory than the L1 cache for CUDA')
    parser.add_argument('-nb', '--num-blocks',
                        type=int,
                        dest='num_blocks',
                        default=8,
                        required=False,
                        help='The target number of blocks / sm to achieve for CUDA.')
    parser.add_argument('-nt', '--num-threads',
                        type=int,
                        dest='num_threads',
                        default=64,
                        required=False,
                        help='The target number of threads / block to achieve for CUDA.')
    parser.add_argument('-mt', '--multi-threaded',
                        type=int,
                        dest='multi_thread',
                        default=1,
                        required=False,
                        help='The number of threads to use during the optimization process')
    parser.add_argument('-fopt', '--force-optimize',
                        dest='force_optimize',
                        action='store_true',
                        default=False,
                        help='Use this option to force a reoptimization of the mechanism (usually only happens when '
                             'generating for a different mechanism)')

    args = parser.parse_args()

    create_jacobian(args.lang, args.input, args.thermo, args.cache_optimizer,
                    args.initial_conditions, args.num_blocks,
                    args.num_threads, args.no_shared, args.L1_preferred,
                    args.multi_thread, args.force_optimize
                    )
