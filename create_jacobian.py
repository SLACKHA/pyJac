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
import cache_optimizer_alt as cache
import shared_memory as shared

def calculate_shared_memory(rind, rxn, specs, reacs, rev_reacs, pdep_reacs):
    #need to figure out shared memory stuff
    variable_list = []
    usages = []
    fwd_usage = 2
    rev_usage = 0 if not rxn.rev else 2
    pres_mod_usage = 0 if not (rxn.pdep or rxn.thd) else (4 if rxn.thd else 2)
    reac_usages = [0 for i in range(len(rxn.reac))]
    prod_usages = [0 for i in range(len(rxn.prod))]
    #add variables
    variable_list.append(utils.get_array('cuda', 'fwd_rates', rind))
    if rxn.rev:
        variable_list.append(utils.get_array('cuda', 'rev_rates', rev_reacs.index(rxn)))
    if rxn.pdep or rxn.thd:
        variable_list.append(utils.get_array('cuda', 'pres_mod', pdep_reacs.index(rind)))
    for sp in set(rxn.reac + rxn.prod + [x[0] for x in rxn.thd_body]):
        sp_real = next(spec for spec in specs if spec.name == sp)
        variable_list.append(utils.get_array('cuda', 'conc', specs.index(sp_real)))

    alphaij_count = 0
    #calculate usages
    if rxn.thd or rxn.pdep:
        fwd_usage +=1
        if rxn.rev:
            rev_usage += 1
        for i, thd in enumerate(rxn.thd_body):
            #check alphaij
            alphaij = thd[1]
            if alphaij is not None and alphaij != 1.0:
                alphaij_count += 1
        fwd_usage += alphaij_count
        if rxn.rev:
            rev_usage += alphaij_count

    for sp_name in rxn.reac:
        nu = rxn.reac_nu[rxn.reac.index(sp_name)]
        if nu - 1 > 0:
            reac_usages[rxn.reac.index(sp_name)] += 1
            if rxn.thd:
                pres_mod_usage += 1
        for sp_name_2 in rxn.reac:
            if sp_name == sp_name_2:
                continue
            reac_usages[rxn.reac.index(sp_name_2)] += 1

    if rxn.rev:
        for sp_name in rxn.prod:
            nu = rxn.prod_nu[rxn.prod.index(sp_name)]
            if nu - 1 > 0:
                prod_usages[rxn.prod.index(sp_name)] += 1
            for sp_name_2 in rxn.prod:
                if sp_name == sp_name_2:
                    continue
                prod_usages[rxn.prod.index(sp_name_2)] += 1

    usages.append(fwd_usage)
    if rxn.rev:
        usages.append(rev_usage)
    if rxn.pdep or rxn.thd:
        usages.append(pres_mod_usage)
    for sp in set(rxn.reac + rxn.prod + [x[0] for x in rxn.thd_body]):
        u = 0
        if sp in rxn.reac:
            u += reac_usages[rxn.reac.index(sp)]
        if sp in rxn.prod:
            u += prod_usages[rxn.prod.index(sp)]
        if sp in rxn.thd_body:
            u += 1
        usages.append(u)

    if CUDAParams.JacRateStrat == CUDAParams.JacRatesCacheStrat.Exclude:
        if rxn.rev:
            usages[0] = 0
            usages[1] = 0
        else:
            usages[0] = 0

    return variable_list, usages  

def get_net_rate_string(lang, rxn, rind, rev_reacs, get_array):
    jline = ''
    if rxn.rev:
        jline += '(' + get_array(lang, 'fwd_rates', rind)
        jline += ' - ' + \
            get_array(lang, 'rev_rates', rev_reacs.index(rxn))
        jline += ')'
    else:
        jline += get_array(lang, 'fwd_rates', rind)
    return jline

def write_dr_dy(file, lang, rev_reacs, rxn, rind, pind, nspec, get_array):
    #write the T_Pr and T_Fi terms if needed
    if rxn.pdep:
        jline = '  pres_mod_temp = '
        jline += get_array(lang, 'pres_mod', pind) + ' * ('
        # dPr/dYj contribution
        if rxn.low:
            # unimolecular/recombination
            jline += '(1.0 / (1.0 + Pr))'
        elif rxn.high:
            # chem-activated bimolecular
            jline += '(-Pr / (1.0 + Pr))'
        if rxn.troe:
            jline += (' - Pr * log(Fcent) * 2.0 * A * (B * '
                      '{:.6}'.format(1.0 / math.log(10.0)) +
                      ' + A * '
                      '{:.6}) / '.format(0.14 / math.log(10.0)) +
                      '(B * B * B * (1.0 + A * A / (B * B)) '
                      '* (1.0 + A * A / (B * B)))'
                      )
        elif rxn.sri:
            jline += ('- Pr * X * X * '
                      '{:.6} * '.format(2.0 / math.log(10.0)) +
                      'log10(Pr) * '
                      'log({:.4} * '.format(rxn.sri[0]) +
                      'exp(-{:4} / T) + '.format(rxn.sri[1]) +
                      'exp(-T / {:.4}))'.format(rxn.sri[2])
                      )

        jline += ') * ' + get_net_rate_string(lang, rxn, rind, rev_reacs, get_array)
        file.write(jline + utils.line_end[lang])

    file.write('  j_temp = -mw_avg * rho_inv * (')
    jline = ''
    # next, contribution from dR/dYj
    #namely the T_dy independent term
    if rxn.pdep or rxn.thd:
        jline += get_array(lang, 'pres_mod', pind)
        jline += ' * ('

    reac_nu = 0
    prod_nu = 0
    if rxn.thd and not rxn.pdep:
        reac_nu = 1
        if rxn.rev:
            prod_nu = 1

    #get reac and prod nu sums
    for sp_name in rxn.reac:
        nu = rxn.reac_nu[rxn.reac.index(sp_name)]
        if nu == 0:
            continue
        reac_nu += nu

    if rxn.rev:
        for sp_name in rxn.prod:
            nu = rxn.prod_nu[rxn.prod.index(sp_name)]
            if nu == 0:
                continue
            prod_nu += nu

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
        jline += '' + get_array(lang, 'rev_rates', rev_reacs.index(rxn))
  
    #find alphaij_hat
    alphaij_hat = None
    counter = {}
    counter[1.0] = 0
    if rxn.thd_body:
        for spec, efficiency in rxn.thd_body:
            if not efficiency in counter:
                counter[efficiency] = 0
            counter[efficiency] += 1
        counter[1.0] += (nspec - sum(counter.values()))
        alphaij_hat = max(counter.keys(), key=lambda x:counter[x])

    #now handle third body / pdep parts if needed
    if rxn.thd and not rxn.pdep:
        jline += '))'
        if alphaij_hat is not None:
            if alphaij_hat == 1.0:
                jline += ' + '
            elif alphaij_hat == -1.0:
                jline += ' - '
            else:
                jline += ' + {} * '.format(alphaij_hat) 
            jline += get_net_rate_string(lang, rxn, rind, rev_reacs, get_array)
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

    return alphaij_hat

def write_rates(file, lang, rxn):
    file.write('  kf = ' + rate.rxn_rate_const(rxn.A, rxn.b, rxn.E) +
                utils.line_end[lang])
    if rxn.rev and not rxn.rev_par:
        file.write('  kr = kf / Kc' + utils.line_end[lang])
    elif rxn.rev_par:
        file.write('  kr = ' +
        rate.rxn_rate_const(rxn.rev_par[0],
                                         rxn.rev_par[1],
                                         rxn.rev_par[2]
                                         ) +
        utils.line_end[lang])
                                      
def write_dr_dy_species(lang, specs, rxn, pind, j_sp, sp_j, alphaij_hat, rind, rev_reacs, get_array):
    jline = 'j_temp'
    if rxn.pdep and not rxn.pdep_sp:
        alphaij = next((thd[1] for thd in rxn.thd_body
                        if thd[0] == sp_j.name), None)
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
    elif rxn.pdep and rxn.pdep_sp and rxn.pdep_sp == sp_j.name:
        jline += ' + pres_mod_temp / (rho * {})'.format(get_array(lang, 'y', j_sp))
    elif rxn.thd and not rxn.pdep:
        alphaij = next((thd[1] for thd in rxn.thd_body
                        if thd[0] == sp_j.name), None)
        if alphaij is None:
            alphaij = 1.0
        if alphaij != alphaij_hat:
            diff = alphaij - alphaij_hat
            if diff != 1:
                if diff == -1:
                    jline += ' - '
                else:
                    jline += ' + {} * '.format(diff)
            else:
                jline += ' + '
            jline += get_net_rate_string(lang, rxn, rind, rev_reacs, get_array)

    if (rxn.pdep or rxn.thd) and (sp_j.name in rxn.reac or (rxn.rev and sp_j.name in rxn.prod)):
        jline += ' + ' + get_array(lang, 'pres_mod', pind)
        jline += ' * '

    if sp_j.name in rxn.reac:
        if not rxn.pdep and not rxn.thd:
            jline += ' + '
        nu = rxn.reac_nu[rxn.reac.index(sp_j.name)]
        if nu != 1:
            if nu == -1:
                jline += '-'
            else:
                jline += '{} * '.format(float(nu))
        jline += 'kf'
        nu_temp = rxn.reac_nu[rxn.reac.index(sp_j.name)]
        if (nu_temp - 1) > 0:
            if isinstance(nu_temp - 1, float):
                jline += ' * pow(' + \
                    get_array(lang, 'conc', j_sp)
                jline += ', {})'.format(nu_temp - 1)
            else:
                # integer, so just use multiplication
                for i in range(nu_temp - 1):
                    jline += ' * ' + \
                        get_array(lang, 'conc', j_sp)

        # loop through remaining reactants
        for sp_reac in rxn.reac:
            if sp_reac == sp_j.name:
                continue

            nu_temp = rxn.reac_nu[rxn.reac.index(sp_reac)]
            isp = next(i for i in range(len(specs))
                       if specs[i].name == sp_reac)
            if isinstance(nu_temp, float):
                jline += ' * pow(conc' + \
                    get_array(lang, isp)
                jline += ', ' + str(nu_temp) + ')'
            else:
                # integer, so just use multiplication
                for i in range(nu_temp):
                    jline += ' * ' + \
                        get_array(lang, 'conc', isp)

    if rxn.rev and sp_j.name in rxn.prod:
        if not rxn.pdep and not rxn.thd and not sp_j.name in rxn.reac:
            jline += ' + '
        elif sp_j.name in rxn.reac:
            jline += ' + '

        nu = rxn.prod_nu[rxn.prod.index(sp_j.name)]
        if nu != -1:
            if nu == 1:
                jline += '-'
            else:
                jline += '{} * '.format(float(-1 * nu))


        jline += 'kr'
        temp_nu = rxn.prod_nu[rxn.prod.index(sp_j.name)]
        if (temp_nu - 1) > 0:
            if isinstance(temp_nu - 1, float):
                jline += ' * pow(' + \
                    get_array(lang, 'conc', j_sp)
                jline += ', {})'.format(temp_nu - 1)
            else:
                # integer, so just use multiplication
                for i in range(temp_nu - 1):
                    jline += ' * ' + \
                        get_array(lang, 'conc', j_sp)

        # loop through remaining products
        for sp_prod in rxn.prod:
            if sp_prod == sp_j.name:
                continue

            temp_nu = rxn.prod_nu[rxn.prod.index(sp_prod)]
            isp = next(i for i in range(len(specs))
                       if specs[i].name == sp_prod)
            if isinstance(temp_nu, float):
                jline += ' * pow(' + \
                    get_array(lang, 'conc', isp)
                jline += ', {})'.format(temp_nu)
            else:
                # integer, so just use multiplication
                jline += ''.join([' * ' + get_array(lang, 'conc', isp) for i in range(temp_nu)])
    
    return jline

def __round_sig(x, sig=8):
    from math import log10, floor
    if x == 0:
        return 0
    return round(x, sig-int(floor(log10(abs(x))))-1)

def write_kc(file, lang, specs, rxn):
    sum_nu = 0
    coeffs = {}
    for sp_name in set(rxn.reac + rxn.prod):
        spec = next((spec for spec in specs
            if spec.name == sp_name), None)
        nu = get_nu(spec, rxn)

        if nu == 0:
            continue

        sum_nu += nu

        lo_array = [__round_sig(nu, 3)] + [__round_sig(x, 9) for x in [
                    spec.lo[6], spec.lo[0], spec.lo[0] - 1.0, spec.lo[1] / 2.0,
                    spec.lo[2] / 6.0, spec.lo[3] / 12.0, spec.lo[4] / 20.0,
                    spec.lo[5]]
                ]
        lo_array = [x * lo_array[0] for x in [lo_array[1] - lo_array[2]] + lo_array[3:]]

        hi_array = [__round_sig(nu, 3)] + [__round_sig(x, 9) for x in [
                    spec.hi[6], spec.hi[0], spec.hi[0] - 1.0, spec.hi[1] / 2.0,
                    spec.hi[2] / 6.0, spec.hi[3] / 12.0, spec.hi[4] / 20.0,
                    spec.hi[5]]
                ]
        hi_array = [x * hi_array[0] for x in [hi_array[1] - hi_array[2]] + hi_array[3:]]                     
        if not spec.Trange[1] in coeffs:
            coeffs[spec.Trange[1]] = lo_array, hi_array
        else:
            coeffs[spec.Trange[1]] = [lo_array[i] + coeffs[spec.Trange[1]][0][i] for i in range(len(lo_array))], \
                                    [hi_array[i] + coeffs[spec.Trange[1]][1][i] for i in range(len(hi_array))]

    isFirst = True
    for T_mid in coeffs:
        # need temperature conditional for equilibrium constants
        line = '  if (T <= {:})'.format(T_mid)
        if lang in ['c', 'cuda']:
            line += ' {\n'
        elif lang == 'fortran':
            line += ' then\n'
        elif lang == 'matlab':
            line += '\n'
        file.write(line)

        lo_array, hi_array = coeffs[T_mid]
        
        if isFirst:
            line = '    Kc = '
        else:
            if lang in ['cuda', 'c']:
                line = '    Kc += '
            else:
                line = '    Kc = Kc + '
        line += ('({:.8e} + '.format(lo_array[0]) + 
                 '{:.8e} * '.format(lo_array[1]) + 
                 'logT + T * ('
                 '{:.8e} + T * ('.format(lo_array[2]) + 
                 '{:.8e} + T * ('.format(lo_array[3]) + 
                 '{:.8e} + '.format(lo_array[4]) + 
                 '{:.8e} * T))) - '.format(lo_array[5]) + 
                 '{:.8e} / T)'.format(lo_array[6]) + 
                 utils.line_end[lang]
                 )
        file.write(line)
        
        if lang in ['c', 'cuda']:
            file.write('  } else {\n')
        elif lang in ['fortran', 'matlab']:
            file.write('  else\n')
        
        if isFirst:
            line = '    Kc = '
        else:
            if lang in ['cuda', 'c']:
                line = '    Kc += '
            else:
                line = '    Kc = Kc + '
        line += ('({:.8e} + '.format(hi_array[0]) + 
                 '{:.8e} * '.format(hi_array[1]) + 
                 'logT + T * ('
                 '{:.8e} + T * ('.format(hi_array[2]) + 
                 '{:.8e} + T * ('.format(hi_array[3]) + 
                 '{:.8e} + '.format(hi_array[4]) + 
                 '{:.8e} * T))) - '.format(hi_array[5]) + 
                 '{:.8e} / T)'.format(hi_array[6]) + 
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

    line = '  Kc = '
    if sum_nu != 0:
        num = (chem.PA / chem.RU) ** sum_nu
        line += '{:.8e} * '.format(num)
    line += 'exp(Kc)' + utils.line_end[lang]
    file.write(line)

def get_nu(sp_k, rxn):
    if sp_k.name in rxn.prod and sp_k.name in rxn.reac:
        nu = (rxn.prod_nu[rxn.prod.index(sp_k.name)] -
              rxn.reac_nu[rxn.reac.index(sp_k.name)])
        # check if net production zero
        if nu == 0:
            return 0
    elif sp_k.name in rxn.prod:
        nu = rxn.prod_nu[rxn.prod.index(sp_k.name)]
    elif sp_k.name in rxn.reac:
        nu = -rxn.reac_nu[rxn.reac.index(sp_k.name)]
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
    if lang in ['c', 'cuda']:
        line = '  //'
    elif lang == 'fortran':
        line = '  !'
    elif lang == 'matlab':
        line = '  %'
    line += ('partial of rxn ' + str(rind) + ' wrt T' +
             utils.line_end[lang]
             )
    file.write(line)

def write_dy_comment(file, lang, rind):
    if lang in ['c', 'cuda']:
        line = '  //'
    elif lang == 'fortran':
        line = '  !'
    elif lang == 'matlab':
        line = '  %'
    line += ('partial of rxn ' + str(rind) + ' wrt species' +
             utils.line_end[lang]
             )
    file.write(line)

def write_dy_y_finish_comment(file, lang):
    if lang in ['c', 'cuda']:
        line = '  //'
    elif lang == 'fortran':
        line = '  !'
    elif lang == 'matlab':
        line = '  %'
    line += 'Finish dYk / Yj\'s\n'
    file.write(line)

def write_dt_t_comment(file, lang, sp):
    if lang in ['c', 'cuda']:
        line = '  //'
    elif lang == 'fortran':
        line = '  !'
    elif lang == 'matlab':
        line = '  %'
    line += 'partial of dT wrt T for spec {}'.format(sp.name) + '\n'
    file.write(line)

def write_dt_y_comment(file, lang, sp):
    if lang in ['c', 'cuda']:
        line = '  //'
    elif lang == 'fortran':
        line = '  !'
    elif lang == 'matlab':
        line = '  %'
    line += 'partial of T wrt Y_{}'.format(sp.name) + '\n'
    file.write(line)

def write_rxn_params_dt(file, rxn, rev=False):
    jline = ''
    if rev:
        if (abs(rxn.rev_par[1]) > 1.0e-90
            and abs(rxn.rev_par[2]) > 1.0e-90):
            jline += ('{:.8e} + '.format(rxn.rev_par[1]) +
                      '({:.8e} / T)'.format(rxn.rev_par[2])
                      )
        elif abs(rxn.rev_par[1]) > 1.0e-90:
            jline += '{:.8e}'.format(rxn.rev_par[1])
        elif abs(rxn.rev_par[2]) > 1.0e-90:
            jline += '({:.8e} / T)'.format(rxn.rev_par[2])
        jline += '{}1.0 - '.format(' + ' if (abs(rxn.rev_par[1]) > 1.0e-90) or (
            abs(rxn.rev_par[2]) > 1.0e-90) else '')
    else:
        if (abs(rxn.b) > 1.0e-90) and (abs(rxn.E) > 1.0e-90):
            jline += '{:.8e} + ({:.8e} / T)'.format(rxn.b, rxn.E)
        elif abs(rxn.b) > 1.0e-90:
            jline += '{:.8e}'.format(rxn.b)
        elif abs(rxn.E) > 1.0e-90:
            jline += '({:.8e} / T)'.format(rxn.E)
        jline += '{}1.0 - '.format(' + ' if (abs(rxn.b)
                                             > 1.0e-90) or (abs(rxn.E) > 1.0e-90) else '')
    file.write(jline)

def write_db_dt_def(file, lang, specs, reacs, rev_reacs, dBdT_flag):
    if lang == 'cuda':
        if len(rev_reacs):
            file.write('  double dBdT[{}]'.format(len(specs)) + utils.line_end[lang])
        template = 'dBdT[{}]'
    else:
        template = 'dBdT_{}'
    for rxn in rev_reacs:
        # only reactions with no reverse Arrhenius parameters
        if rxn.rev_par:
            continue

        # all participating species
        for rxn_sp in (rxn.reac + rxn.prod):
            sp_ind = next((specs.index(s) for s in specs
                           if s.name == rxn_sp), None)

            # skip if already printed
            if dBdT_flag[sp_ind]:
                continue

            dBdT_flag[sp_ind] = True

            if lang in ['c', 'cuda']:
                dBdT = template.format(sp_ind)
            elif lang in ['fortran', 'matlab']:
                dBdT = template.format(sp_ind + 1)
            # declare dBdT
            if lang != 'cuda':
                file.write('  double ' + dBdT + utils.line_end[lang])

            # dB/dT evaluation (with temperature conditional)
            line = '  if (T <= {:})'.format(specs[sp_ind].Trange[1])
            if lang in ['c', 'cuda']:
                line += ' {\n'
            elif lang == 'fortran':
                line += ' then\n'
            elif lang == 'matlab':
                line += '\n'
            file.write(line)

            line = ('    ' + dBdT +
                    ' = ({:.8e}'.format(specs[sp_ind].lo[0] - 1.0) +
                    ' + {:.8e} / T) / T'.format(specs[sp_ind].lo[5]) +
                    ' + {:.8e} + T'.format(specs[sp_ind].lo[1] / 2.0) +
                    ' * ({:.8e}'.format(specs[sp_ind].lo[2] / 3.0) +
                    ' + T * ({:.8e}'.format(specs[sp_ind].lo[3] / 4.0) +
                    ' + {:.8e} * T))'.format(specs[sp_ind].lo[4] / 5.0) +
                    utils.line_end[lang]
                    )
            file.write(line)

            if lang in ['c', 'cuda']:
                file.write('  } else {\n')
            elif lang in ['fortran', 'matlab']:
                file.write('  else\n')

            line = ('    ' + dBdT +
                    ' = ({:.8e}'.format(specs[sp_ind].hi[0] - 1.0) +
                    ' + {:.8e} / T) / T'.format(specs[sp_ind].hi[5]) +
                    ' + {:.8e} + T'.format(specs[sp_ind].hi[1] / 2.0) +
                    ' * ({:.8e}'.format(specs[sp_ind].hi[2] / 3.0) +
                    ' + T * ({:.8e}'.format(specs[sp_ind].hi[3] / 4.0) +
                    ' + {:.8e} * T))'.format(specs[sp_ind].hi[4] / 5.0) +
                    utils.line_end[lang]
                    )
            file.write(line)

            if lang in ['c', 'cuda']:
                file.write('  }\n\n')
            elif lang == 'fortran':
                file.write('  end if\n\n')
            elif lang == 'matlab':
                file.write('  end\n\n')

def write_db_dt(file, lang, specs, rxn):
    if lang == 'cuda':
        template = 'dBdT[{}]'
    else:
        template = 'dBdT_{}'
    jline = ''
    notfirst = False
    # contribution from dBdT terms from
    # all participating species
    for sp in rxn.prod:
        if sp in rxn.reac:
            nu = (rxn.prod_nu[rxn.prod.index(sp)] -
                  rxn.reac_nu[rxn.reac.index(sp)]
                  )
        else:
            nu = rxn.prod_nu[rxn.prod.index(sp)]

        if (nu == 0):
            continue

        sp_ind = next((specs.index(s) for s in specs
                       if s.name == sp), None)

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

    for sp in rxn.reac:
        # skip species also in products, already counted
        if sp in rxn.prod:
            continue

        nu = -rxn.reac_nu[rxn.reac.index(sp)]

        sp_ind = next((specs.index(s) for s in specs
                       if s.name == sp), None)

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

    jline += '))'

    file.write(jline)

def write_pr(file, lang, specs, reacs, pdep_reacs, rxn, get_array, last_conc_temp=None):
    # print lines for necessary pressure-dependent variables
    line = '  {} = '.format('conc_temp' if rxn.thd_body else 'Pr')
    conc_temp_log = None
    if rxn.thd_body:
        #take care of the conc_temp collapsing
        conc_temp_log = []
    if rxn.pdep_sp:
        line += get_array(lang, 'conc', specs.index(rxn.pdep_sp))
    else:
        line += '(m'

        for thd_sp in rxn.thd_body:
            isp = specs.index(next((s for s in specs
                                    if s.name == thd_sp[0]), None))
            if thd_sp[1] > 1.0:
                line += ' + {} * '.format(thd_sp[1] - 1.0)
            elif thd_sp[1] < 1.0:
                line += ' - {} * '.format(1.0 - thd_sp[1])
            if thd_sp[1] != 1.0:
                line += get_array(lang, 'conc', isp)
                if conc_temp_log is not None:
                    conc_temp_log.append((thd_sp[0], thd_sp[1] - 1.0))
        line += ')'

        if last_conc_temp is not None:
            #need to update based on the last
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

            if len(new_conc_temp):
                #remake the line with the updated numbers
                line = '  conc_temp += ('

                for i, thd_sp in enumerate(new_conc_temp):
                    isp = specs.index(next((s for s in specs
                                            if s.name == thd_sp[0]), None))
                    if i > 0:
                        line += ' {}{} * '.format('- ' if thd_sp[1] < 0 else '+ ', abs(thd_sp[1]))
                    else:
                        line += '{} * '.format(thd_sp[1])
                    line += get_array(lang, 'conc', isp)
                line += ')'
            else:
                line = ''

    if rxn.thd_body and len(line):
        file.write(line + utils.line_end[lang])

    if rxn.pdep:
        if rxn.thd_body:
            line = '  Pr = conc_temp'
        beta_0minf, E_0minf, k0kinf = get_infs(rxn)
        # finish writing P_ri
        line += (' * (' + k0kinf + ')' +
                 utils.line_end[lang]
                 )
        file.write(line)

    return conc_temp_log


def write_troe(file, lang, rxn):
    line = ('  Fcent = '
            '{:.8e} * '.format(1.0 - rxn.troe_par[0]) +
            'exp(-T / {:.8e})'.format(rxn.troe_par[1]) +
            ' + {:.8e} * exp(T / '.format(rxn.troe_par[0]) +
            '{:.8e})'.format(rxn.troe_par[2])
            )
    if len(rxn.troe_par) == 4:
        line += ' + exp(-{:.8e} / T)'.format(rxn.troe_par[3])
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

def write_pdep_dt(file, lang, rxn, rev_reacs, rind, pind, get_array):
    beta_0minf, E_0minf, k0kinf = get_infs(rxn)
    jline = '  j_temp = (' + get_array(lang, 'pres_mod', pind)
    jline += ' * ((' + ('-Pr' if rxn.high else '') #high -> chem-activated bimolecular rxn

    # dPr/dT
    jline += ('({:.4e} + ('.format(beta_0minf) +
              '{:.8e} / T) - 1.0) / '.format(E_0minf) +
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
            get_array(lang, 'rev_rates', rev_reacs.index(rxn))
        jline += ')'
    else:
        # forward reaction rate only
        jline += '' + get_array(lang, 'fwd_rates', rind)

    jline += ' + (' + get_array(lang, 'pres_mod', pind)

    file.write(jline)

def write_sri_dt(lang, rxn, beta_0minf, E_0minf, k0kinf):
    jline = (' + X * ((('
          '{:.8} / '.format(rxn.sri[0] * rxn.sri[1]) +
          '(T * T)) * exp(-'
          '{:.8} / T) - '.format(rxn.sri[1]) +
          '{:.8e} * '.format(1.0 / rxn.sri[2]) +
          'exp(-T / {:.8})) / '.format(rxn.sri[2]) +
          '({:.8} * '.format(rxn.sri[0]) +
          'exp(-{:.8} / T) + '.format(rxn.sri[1]) +
          'exp(-T / {:.8})) - '.format(rxn.sri[2]) +
          'X * {:.8} * '.format(2.0 / math.log(10.0)) +
          'log10(Pr) * ('
          '{:.8e} + ('.format(beta_0minf) +
          '{:.8e} / T) - 1.0) * '.format(E_0minf) +
          'log({:8} * exp('.format(rxn.sri[0]) +
          '-{:.8} / T) + '.format(rxn.sri[1]) +
          'exp(-T / '
          '{:8})) / T)'.format(rxn.sri[2])
          )

    if len(rxn.sri) == 5:
        jline += ' + ({:.8} / T)'.format(rxn.sri[4])

    return jline

def write_troe_dt(lang, rxn, beta_0minf, E_0minf, k0kinf):
    jline = (' + (((1.0 / '
              '(Fcent * (1.0 + A * A / (B * B)))) - '
              'lnF_AB * ('
              '-{:.8e}'.format(0.67 / math.log(10.0)) +
              ' * B + '
              '{:.8e} * '.format(1.1762 / math.log(10.0)) +
              'A) / Fcent)'
              ' * ({:.8e}'.format(-(1.0 - rxn.troe_par[0]) /
                                  rxn.troe_par[1]) +
              ' * exp(-T / '
              '{:.8e}) - '.format(rxn.troe_par[1]) +
              '{:.8e} * '.format(rxn.troe_par[0] /
                                 rxn.troe_par[2]) +
              'exp(-T / '
              '{:.8e})'.format(rxn.troe_par[2])
              )
    if len(rxn.troe_par) == 4:
        jline += (' + ({:.8e} / '.format(rxn.troe_par[3]) +
                  '(T * T)) * exp('
                  '-{:.8e} / T)'.format(rxn.troe_par[3])
                  )
    jline += '))'

    jline += (' - lnF_AB * ('
              '{:.8e}'.format(1.0 / math.log(10.0)) +
              ' * B + '
              '{:.8e}'.format(0.14 / math.log(10.0)) +
              ' * A) * '
              '({:.8e} + ('.format(beta_0minf) +
              '{:.8e} / T) - 1.0) / T'.format(E_0minf)
              )

    return jline

def write_dcp_dt(file, lang, specs, sparse_indicies):
    T_mid_buckets = {}
    #put all of the same T_mids together
    for isp, sp in enumerate(specs):
        if sp.Trange[1] not in T_mid_buckets:
            T_mid_buckets[sp.Trange[1]] = []
        T_mid_buckets[sp.Trange[1]].append(isp)


    first = True
    for T_mid in T_mid_buckets:
        #write the if statement
        line = '  if (T <= {:})'.format(T_mid)
        if lang in ['c', 'cuda']:
            line += ' {\n'
        elif lang == 'fortran':
            line += ' then\n'
        elif lang == 'matlab':
            line += '\n'
        file.write(line)
        #and the update line
        line = '    working_temp'
        if lang in ['c', 'cuda']:
            line += ' {}= '.format('+' if not first else '')
        elif lang in ['fortran', 'matlab']:
            line += ' = {}'.format('working_temp + ' if not first else '')
        for isp in T_mid_buckets[T_mid]:
            sp = specs[isp]
            if T_mid_buckets[T_mid].index(isp):
                line += '\n    + '
            line += '(' + utils.get_array(lang, 'y', isp + 1)
            line += (' * {:.8e} * ('.format(chem.RU / sp.mw) +
                     '{:.8e} + '.format(sp.lo[1]) +
                     'T * ({:.8e} + '.format(2.0 * sp.lo[2]) +
                     'T * ({:.8e} + '.format(3.0 * sp.lo[3]) +
                     '{:.8e} * T)))'.format(4.0 * sp.lo[4]) +
                     ')'
                     )
        line += utils.line_end[lang]
        file.write(line)

        #now do the high temperature side
        if lang in ['c', 'cuda']:
            file.write('  } else {\n')
        elif lang in ['fortran', 'matlab']:
            file.write('  else\n')
        #and the update line
        line = '    working_temp'
        if lang in ['c', 'cuda']:
            line += ' {}= '.format('+' if not first else '')
        elif lang in ['fortran', 'matlab']:
            line += ' = {}'.format('working_temp + ' if not first else '')

        for isp in T_mid_buckets[T_mid]:
            sp = specs[isp]
            if T_mid_buckets[T_mid].index(isp):
                line += '\n    + '
            line += '(' + utils.get_array(lang, 'y', isp + 1)
            line += (' * {:.8e} * ('.format(chem.RU / sp.mw) +
                     '{:.8e} + '.format(sp.hi[1]) +
                     'T * ({:.8e} + '.format(2.0 * sp.hi[2]) +
                     'T * ({:.8e} + '.format(3.0 * sp.hi[3]) +
                     '{:.8e} * T)))'.format(4.0 * sp.hi[4]) +
                     ')'
                     )
        line += utils.line_end[lang]
        file.write(line)
        #and finish the if
        if lang in ['c', 'cuda']:
            file.write('  }\n\n')
        elif lang == 'fortran':
            file.write('  end if\n\n')
        elif lang == 'matlab':
            file.write('  end\n\n')

        first = False

def write_dt_y(file, lang, specs, sp, isp, num_s, touched, sparse_indicies, offset, get_array):
    for k_sp, sp_k in enumerate(specs):
        line = '  '
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
    if lang in ['c', 'cuda']:
        line = '  //'
    elif lang == 'fortran':
        line = '  !'
    elif lang == 'matlab':
        line = '  %'
    line += ('Complete dT/dy calculations\n')
    file.write(line)
    file.write('  j_temp = 1.0 / (rho * cp_avg * cp_avg)' + utils.line_end[lang])
    for k_sp, sp_k in enumerate(specs):
        line = '  '
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
    if lang in ['c', 'cuda']:
        line = '  //'
    elif lang == 'fortran':
        line = '  !'
    elif lang == 'matlab':
        line = '  %'
    line += ('Complete dT wrt T calculations\n')
    file.write(line)
    line = '  '
    if lang in ['c', 'cuda']:
        line += '' + get_array(lang, 'jac', 0)
    elif lang in ['fortran', 'matlab']:
        line += '' + get_array(lang, 'jac', 0, twod=0)

    line += ' = -('

    for k_sp, sp_k in enumerate(specs):
        if k_sp:
            line += '    + '
        line += '' + get_array(lang, 'dy', k_sp + offset) + ' * {:.8e}'.format(sp_k.mw) + ' * '
        line += '(-working_temp * ' + get_array(lang, 'h', k_sp) + ' / cp_avg + ' + '' + get_array(lang, 'cp', k_sp) + ')'
        line += ' + ' + get_array(lang, 'jac', k_sp + 1, twod=0) + ' * ' + get_array(lang, 'h', k_sp) + ' * rho'
        if k_sp != len(specs) - 1:
            if lang == 'fortran':
                line += ' &'
            line += '\n'

    line += ') / (rho * cp_avg)'
    line += utils.line_end[lang]
    file.write(line)

def write_cuda_intro(path, number, rate_list, this_rev, this_pdep, this_thd, this_troe, this_sri, no_shared):
    """
    Writes the header and definitions for of any of the various sub-functions for CUDA

    Returns the opened file
    """
    lang = 'cuda'
    with open(os.path.join(path, 'jacob_' + str(number) + '.h'), 'w') as file:
        file.write('#ifndef JACOB_HEAD_{}\n'.format(number) + 
                   '#define JACOB_HEAD_{}\n'.format(number) + 
                   '\n'
                   '#include "../header.h"\n'
                   '\n'
                   '__device__ void eval_jacob_{} ('.format(number)
                  )
        file.write('const double, const double*')
        for rate in rate_list:
            file.write(', const double*')
        if this_thd:
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

    line = '__device__ '

    line += ('void eval_jacob_{} (const double pres, '.format(number) + 
            'const double * conc')
    for rate in rate_list:
        line += ', const double* ' + rate
    if this_thd:
        line += ', const double m'
    line += ', const double mw_avg, const double rho, const double* dBdT, const double T, double* jac) {'
    file.write(line + '\n')

    if not no_shared:
        file.write('  extern __shared__ double shared_temp[]' + utils.line_end[lang])
     # third-body variable needed for reactions
    if this_pdep and this_thd:
        line = '  '
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += ('conc_temp' +
                 utils.line_end[lang]
                 )
        file.write(line)


    # log(T)
    line = '  '
    if lang == 'c':
        line += 'double '
    elif lang == 'cuda':
        line += 'register double '
    line += ('logT = log(T)' +
             utils.line_end[lang]
             )
    file.write(line)

    line = '  '
    if lang == 'c':
        line += 'double '
    elif lang == 'cuda':
        line += 'register double '
    line += 'j_temp = 0.0' + utils.line_end[lang]
    file.write(line)

    if this_pdep:
        line = '  '
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'pres_mod_temp = 0.0' + utils.line_end[lang]
        file.write(line)

    if this_thd:
        line = '  '
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'thd_temp = 0.0' + utils.line_end[lang]
        file.write(line)

    # if any reverse reactions, will need Kc
    if this_rev:
        line = '  '
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'Kc = 0.0' + utils.line_end[lang]
        file.write(line)

    # pressure-dependence variables
    if this_pdep:
        line = '  '
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'Pr = 0.0' + utils.line_end[lang]
        file.write(line)

    if this_troe:
        line = ''.join(['  double {} = 0.0{}'.format(x, utils.line_end[lang]) for x in 'Fcent', 'A', 'B', 'lnF_AB'])
        file.write(line)

    if this_sri:
        line = '  double X = 0.0' + utils.line_end[lang]
        file.write(line)

    return file

def write_dy_intros(path, number):
    lang = 'cuda'
    with open(os.path.join(path, 'jacob_' + str(number) + '.h'), 'w') as file:
        file.write('#ifndef JACOB_HEAD_{}\n'.format(number) + 
                   '#define JACOB_HEAD_{}\n'.format(number) + 
                   '\n'
                   '#include "../header.h"\n'
                   '\n'
                   '__device__ void eval_jacob_{} ('.format(number)
                  )
        file.write('const double, const double, const double, const double*, const double*, const double*, double*);\n'
                   '\n'
                   '#endif\n'
                   )
    file = open(os.path.join(path, 'jacob_' + str(number) + utils.file_ext[lang]), 'w')
    file.write('#include "../header.h"\n'
               '\n'
               )

    line = '__device__ '

    line += ('void eval_jacob_{} (const double mw_avg, const double rho, const double cp_avg, const double* dy, const double* h, const double* cp, double* jac) '.format(number))
    line += '{\n'
    line += '  register double rho_inv = 1.0 / rho;'
    file.write(line + utils.line_end['cuda'])
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

    do_unroll = len(specs) > CUDAParams.Jacob_Unroll
    if lang == 'cuda' and do_unroll:
        #make paths for separate jacobian files
        utils.create_dir(os.path.join(path, 'jacobs'))

    # first write header file
    if lang == 'c':
        file = open(path + 'jacob.h', 'w')
        file.write('#ifndef JACOB_HEAD\n'
                   '#define JACOB_HEAD\n'
                   '\n'
                   '#include "header.h"\n'
                   '\n'
                   'void eval_jacob (const double, const double, double*, '
                   'double*);\n'
                   '\n'
                   '#endif\n'
                   )
        file.close()
    elif lang == 'cuda':
        file = open(path + 'jacob.cuh', 'w')
        file.write('#ifndef JACOB_HEAD\n'
                   '#define JACOB_HEAD\n'
                   '\n'
                   '#include "header.h"\n'
                   '\n'
                   '__device__ void eval_jacob (const double, const double, '
                   'double*, double*);\n'
                   '\n'
                   '#endif\n'
                   )
        file.close()

    # numbers of species and reactions
    num_s = len(specs)
    num_r = len(reacs)
    rev_reacs = [rxn for rxn in reacs if rxn.rev]
    num_rev = len(rev_reacs)

    pdep_reacs = []
    for reac in reacs:
        if reac.thd or reac.pdep:
            # add reaction index to list
            pdep_reacs.append(reacs.index(reac))
    num_pdep = len(pdep_reacs)

    # create file depending on language
    filename = 'jacob' + utils.file_ext[lang]
    file = open(path + filename, 'w')

    # header files
    if lang == 'c':
        file.write('#include <math.h>\n'
                   '#include "header.h"\n'
                   '#include "chem_utils.h"\n'
                   '#include "rates.h"\n'
                   '\n'
                   )
    elif lang == 'cuda':
        file.write('#include <math.h>\n'
                   '#include "jacobs/jac_include.h"\n'
                   '#include "header.h"\n'
                   '#include "chem_utils.cuh"\n'
                   '#include "rates.cuh"\n'
                   '#include "gpu_macros.cuh"\n'
                   '\n'
                   )

    line = ''
    if lang == 'cuda':
        line = '__device__ '

    offset = 0

    if lang in ['c', 'cuda']:
        line += ('void eval_jacob (const double t, const double pres, '
                 'double * y, double * jac) {\n\n')
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
        if any(rxn.thd for rxn in rev_reacs):
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
        smm.write_init(file, indent = 2)

    # get temperature
    if lang in ['c', 'cuda']:
        line = '  double T = ' + get_array(lang, 'y', 0)
    elif lang in ['fortran', 'matlab']:
        line = '  T = ' + get_array(lang, 'y', 0)
    line += utils.line_end[lang]
    file.write(line)

    file.write('\n')

    # calculation of average molecular weight
    if lang in ['c', 'cuda']:
        file.write('  // average molecular weight\n'
                   '  double mw_avg;\n'
                   )
    elif lang == 'fortran':
        file.write('  ! average molecular weight\n')
    elif lang == 'matlab':
        file.write('  % average molecular weight\n')

    if lang in ['c', 'cuda']:
        file.write('  // mass-averaged density\n'
                   '  double rho;\n'
                   )
    elif lang == 'fortran':
        file.write('  ! mass-averaged density\n')
    elif lang == 'matlab':
        file.write('  % mass-averaged density\n')

    # evaluate species molar concentrations
    if lang in ['c', 'cuda']:
            file.write('  // species molar concentrations\n'
                       '  double conc[{}];\n'.format(num_s)
                       )
    elif lang == 'fortran':
        file.write('  ! species molar concentrations\n')
    elif lang == 'matlab':
        file.write('  % species molar concentrations\n'
                   '  conc = zeros({},1);\n'.format(num_s)
                   )
    file.write('  eval_conc (y[0], pres, &y[1], mw_avg, rho, conc);\n\n')

    rate_list = ['fwd_rates']
    if len(rev_reacs):
        rate_list.append('rev_rates')
    if len(pdep_reacs):
        rate_list.append('pres_mod')
    rate_list.append('dy')
    # evaluate forward and reverse reaction rates
    if lang in ['c', 'cuda']:
        file.write('  // evaluate reaction rates\n'
                   '  double fwd_rates[{}];\n'.format(num_r)
                   )
        if rev_reacs:
            file.write('  double rev_rates[{}];\n'.format(num_rev) +
                       '  eval_rxn_rates (T, conc, fwd_rates, '
                       'rev_rates);\n'
                       )
        else:
            file.write('  eval_rxn_rates (T, conc, fwd_rates);\n')
    elif lang == 'fortran':
        file.write('  ! evaluate reaction rates\n')
        if rev_reacs:
            file.write('  call eval_rxn_rates (T, conc, fwd_rates, '
                       'rev_rates)\n'
                       )
        else:
            file.write('  eval_rxn_rates (T, conc, fwd_rates)\n')
    elif lang == 'matlab':
        file.write('  % evaluate reaction rates\n')
        if rev_reacs:
            file.write('  [fwd_rates, rev_rates] = eval_rxn_rates '
                       '(T, conc);\n'
                       )
        else:
            file.write('  fwd_rates = eval_rxn_rates (T, conc);\n')
    file.write('\n')

    # evaluate third-body and pressure-dependence reaction modifications
    if lang in ['c', 'cuda']:
        file.write('  // get pressure modifications to reaction rates\n'
                   '  double pres_mod[{}];\n'.format(num_pdep) +
                   '  get_rxn_pres_mod (T, pres, conc, pres_mod);\n')
    elif lang == 'fortran':
        file.write('  ! get and evaluate pressure modifications to '
                   'reaction rates\n'
                   '  call get_rxn_pres_mod (T, pres, conc, pres_mod)\n'
                   )
    elif lang == 'matlab':
        file.write('  % get and evaluate pressure modifications to '
                   'reaction rates\n'
                   '  pres_mod = get_rxn_pres_mod (T, pres, conc, '
                   'pres_mod);\n'
                   )
    file.write('\n')

    # evaluate species rates
    if lang in ['c', 'cuda']:
        file.write('  // evaluate rate of change of species molar '
                   'concentration\n')
        file.write('  double dy[{}];\n'.format(num_s))
        if rev_reacs:
            file.write('  eval_spec_rates (fwd_rates, rev_rates, '
                       'pres_mod, dy);\n'
                       )
        else:
            file.write('  eval_spec_rates (fwd_rates, pres_mod, '
                       'dy);\n'
                       )
    elif lang == 'fortran':
        file.write('  ! evaluate rate of change of species molar '
                   'concentration\n'
                   )
        if rev_reacs:
            file.write('  call eval_spec_rates (fwd_rates, rev_rates, '
                       'pres_mod, dy)\n'
                       )
        else:
            file.write('  eval_spec_rates (fwd_rates, pres_mod, '
                       'dy)\n'
                       )
    elif lang == 'matlab':
        file.write('  % evaluate rate of change of species molar '
                   'concentration\n'
                   )
        if rev_reacs:
            file.write('  dy = eval_spec_rates(fwd_rates, '
                       'rev_rates, pres_mod);\n'
                       )
        else:
            file.write('  dy = eval_spec_rates(fwd_rates, '
                       'pres_mod);\n'
                       )
    file.write('\n')

    # third-body variable needed for reactions
    if any(rxn.pdep and rxn.thd for rxn in reacs):
        line = '  '
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += ('m = pres / ({:4e} * T)'.format(chem.RU) +
                 utils.line_end[lang]
                 )
        file.write(line)

        if not (lang == 'cuda' and do_unroll):
            line = '  '
            if lang == 'c':
                line += 'double '
            elif lang == 'cuda':
                line += 'register double '
            line += ('conc_temp' +
                     utils.line_end[lang]
                     )
            file.write(line)


    # log(T)
    line = '  '
    if lang == 'c':
        line += 'double '
    elif lang == 'cuda':
        line += 'register double '
    line += ('logT = log(T)' +
             utils.line_end[lang]
             )
    file.write(line)

    line = '  '
    if lang == 'c':
        line += 'double '
    elif lang == 'cuda':
        line += 'register double '
    line += 'j_temp = 0.0' + utils.line_end[lang]
    file.write(line)

    line = '  '
    if lang == 'c':
        line += 'double '
    elif lang == 'cuda':
        line += 'register double '
    line += 'kf = 0.0' + utils.line_end[lang]
    file.write(line)

    if any(rxn.pdep for rxn in reacs) and not (lang == 'cuda' and do_unroll):
        line = '  '
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'pres_mod_temp = 0.0' + utils.line_end[lang]
        file.write(line)

    # if any reverse reactions, will need Kc
    if rev_reacs and not (lang == 'cuda' and do_unroll):
        line = '  '
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        file.write(line + 'Kc = 0.0' + utils.line_end[lang])
        file.write(line + 'kr = 0' + utils.line_end[lang])

    # pressure-dependence variables
    if any(rxn.pdep for rxn in reacs) and not (lang == 'cuda' and do_unroll):
        line = '  '
        if lang == 'c':
            line += 'double '
        elif lang == 'cuda':
            line += 'register double '
        line += 'Pr = 0.0' + utils.line_end[lang]
        file.write(line)

    if any(rxn.troe for rxn in reacs) and not (lang == 'cuda' and do_unroll):
        line = ''.join(['  double {} = 0.0{}'.format(x, utils.line_end[lang]) for x in 'Fcent', 'A', 'B', 'lnF_AB'])
        file.write(line)

    if any(rxn.sri for rxn in reacs) and not (lang == 'cuda' and do_unroll):
        line = '  double X = 0.0' + utils.line_end[lang]
        file.write(line)

    if lang != 'cuda':
        line = '  '
        if lang == 'c':
            line += 'double '
        line += 'rho_inv = 1.0 / rho' + utils.line_end[lang]
        file.write(line)

    # variables for equilibrium constant derivatives, if needed
    dBdT_flag = [False for sp in specs]

    #define dB/dT's
    write_db_dt_def(file, lang, specs, reacs, rev_reacs, dBdT_flag)

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
        if lang == 'cuda' and do_unroll and (rind == next_fn_index):
            #clear conc temp
            last_conc_temp = None
            file_store = file
            #get next index
            next_fn_index = rind + splittings[0]
            splittings = splittings[1:]
            #get flags
            rev = False
            pdep = False
            thd = False
            troe = False
            sri = False
            for ind_next in range(rind, next_fn_index):
                if reacs[ind_next].rev:
                    rev = True
                if reacs[ind_next].pdep:
                    pdep = True
                if reacs[ind_next].thd:
                    thd = True
                if reacs[ind_next].troe:
                    troe = True
                if reacs[ind_next].sri:
                    sri = True
            batch_has_thd = thd
            #write the specific evaluator for this reaction
            file = write_cuda_intro(os.path.join(path, 'jacobs'), jac_count, rate_list, rev, pdep, thd, troe, sri, smm is None)

        if lang == 'cuda' and smm is not None:
            variable_list, usages = calculate_shared_memory(rind, rxn, specs, reacs, rev_reacs, pdep_reacs)
            smm.load_into_shared(file, variable_list, usages)

        
        ######################################
        # with respect to temperature
        ######################################

        write_dt_comment(file, lang, rind)

        #first we need any pres mod terms
        jline = ''
        pind = None
        if rxn.pdep:
            pind = pdep_reacs.index(rind)
            last_conc_temp = write_pr(file, lang, specs, reacs, pdep_reacs, rxn, get_array, last_conc_temp)

            #dF/dT
            if rxn.troe:
                write_troe(file, lang, rxn)
            elif rxn.sri:
                write_sri(file, lang)
            
            write_pdep_dt(file, lang, rxn, rev_reacs, rind, pind, get_array)

        else:
            # not pressure dependent

            # third body reaction
            if rxn.thd_body:
                pind = pdep_reacs.index(rind)

                #going to need conc_temp
                #write_pr(file, lang, specs, reacs, pdep_reacs, rxn, get_array)

                jline = '  j_temp = ((-' + get_array(lang, 'pres_mod', pind)
                jline += ' * '

                if rxn.rev:
                    # forward and reverse reaction rates
                    jline += '(' + get_array(lang, 'fwd_rates', rind)
                    jline += ' - ' + \
                        get_array(lang, 'rev_rates', rev_reacs.index(rxn))
                    jline += ')'
                else:
                    # forward reaction rate only
                    jline += '' + get_array(lang, 'fwd_rates', rind)

                jline += ' / T) + (' + get_array(lang, 'pres_mod', pind)

            else:
                if lang in ['c', 'cuda', 'matlab']:
                    jline += '  j_temp = ((1.0'
                elif lang in ['fortran']:
                    jline += '  j_temp = ((1.0_wp'

            file.write(jline)

        jline = ' / T) * ('

        # contribution from temperature derivative of forward reaction rate
        jline += '' + get_array(lang, 'fwd_rates', rind)
        jline += ' * ('
        file.write(jline)

        write_rxn_params_dt(file, rxn, rev=False)

        # loop over reactants
        jline = ''
        nu = 0
        for sp in rxn.reac:
            nu += rxn.reac_nu[rxn.reac.index(sp)]
        jline += '{})'.format(float(nu))

        # contribution from temperature derivative of reaction rates
        if rxn.rev:
            # reversible reaction

            jline += ' - ' + \
                get_array(lang, 'rev_rates', rev_reacs.index(rxn)) + \
                ' * ('

            file.write(jline)
            jline = ''

            if rxn.rev_par:
                write_rxn_params_dt(file, rxn, rev=True)

                nu = 0
                # loop over products
                for sp in rxn.prod:
                    nu += rxn.prod_nu[rxn.prod.index(sp)]
                jline += '{})'.format(float(nu))
                file.write(jline)
            else:
                write_rxn_params_dt(file, rxn, rev=False)

                nu = 0
                # loop over products
                for sp in rxn.prod:
                    nu += rxn.prod_nu[rxn.prod.index(sp)]
                jline += '{} - T * ('.format(float(nu))
                file.write(jline)
                jline = ''
                write_db_dt(file, lang, specs, rxn)
                file.write(')')
        else:
            jline += ')'

        # print line for reaction
        file.write(jline + ') / rho' + utils.line_end[lang])

        for rxn_sp in set(rxn.reac + rxn.prod):
            sp_k, k_sp = next(((s, specs.index(s)) for s in specs
                           if s.name == rxn_sp), None)
            line = '  '
            nu = get_nu(sp_k, rxn)
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
            #need to find Kc
            write_kc(file, lang, specs, rxn)

        #need to write the dr/dy parts (independent of any species)
        alphaij_hat = write_dr_dy(file, lang, rev_reacs, rxn, rind, pind, len(specs), get_array)

        #write the forward / backwards rates:
        write_rates(file, lang, rxn)

        #now loop through each species
        for j_sp, sp_j in enumerate(specs):
            for rxn_sp_k in set(rxn.reac + rxn.prod):
                sp_k, k_sp = next(((s, specs.index(s)) for s in specs
                               if s.name == rxn_sp_k), None)

                nu = get_nu(sp_k, rxn)

                #sparse indexes
                if lang in ['c', 'cuda']:
                    if k_sp + 1 + (num_s + 1) * (j_sp + 1) not in sparse_indicies:
                        sparse_indicies.append(k_sp + 1 + (num_s + 1) * (j_sp + 1))
                elif lang in ['fortran', 'matlab']:
                    if (k_sp + 1, j_sp + 1) not in sparse_indicies:
                        sparse_indicies.append((k_sp + 1, j_sp + 1))

                if nu == 0:
                    continue

                working_temp = '('

                working_temp += write_dr_dy_species(lang, specs, rxn, pind, j_sp, sp_j, alphaij_hat, rind, rev_reacs, get_array)

                working_temp += ')'
                mw_frac = __round_sig(sp_k.mw / sp_j.mw, 9)
                if mw_frac != 1.0:
                    working_temp += ' * {:.8e}'.format(mw_frac)

                lin_index = k_sp + 1 + (num_s + 1) * (j_sp + 1)
                jline = '  '
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

                jline += '' if nu == 1 else ('-' if nu == -1 else '{} * '.format(float(nu)))
                jline += working_temp
                jline += utils.line_end[lang]

                file.write(jline)
                jline = ''

        file.write('\n')

        if lang == 'cuda' and smm is not None:
            smm.mark_for_eviction(variable_list)

        if lang == 'cuda' and do_unroll and (rind == next_fn_index - 1 or rind == len(reacs) - 1):
            #switch back
            file.write('}\n\n')
            file = file_store
            file.write('  eval_jacob_{}('.format(jac_count))
            jac_count += 1
            line = ('pres, conc')
            for rate in rate_list:
                line += ', ' + rate
            if batch_has_thd:
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

    line = '  cp_avg = '
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
            line = '     '

        isp = specs.index(sp)
        if not isfirst:
            line += ' + '
        line += ('(' + get_array(lang, 'y', isp + 1) + ' * ' + '' + get_array(lang, 'cp', isp) + ')')

        isfirst = False
    line += utils.line_end[lang]
    file.write(line)

    #set jac[0] = 0
    # set to zero
    line = '  '
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

    line = '  '
    if lang == 'c':
        line += 'double '
    elif lang == 'cuda':
        line += 'register double '
    line += 'working_temp = 0.0' + utils.line_end[lang]
    file.write(line)
               
    next_fn_index = 0
    #need to finish the dYk/dYj's
    write_dy_y_finish_comment(file, lang)
    for k_sp, sp_k in enumerate(specs):
        if lang == 'cuda' and do_unroll and k_sp == next_fn_index:
            store_file = file
            file = write_dy_intros(os.path.join(path, 'jacobs'), jac_count)
            next_fn_index += min(10, len(specs) - k_sp)

        for j_sp, sp_j in enumerate(specs):
            lin_index = k_sp + 1 + (num_s + 1) * (j_sp + 1)
            #see if this combo matters
            if touched[lin_index]:
                line = '  '
                #zero out if unused
                if lang in ['c', 'cuda'] and not lin_index in sparse_indicies:
                    file.write(get_array(lang, 'jac', lin_index) + ' = 0.0' + utils.line_end[lang])
                elif lang in ['fortran', 'matlab'] and not (k_sp + 1, j_sp + 1) in sparse_indicies:
                    file.write(get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) + ' = 0.0' + utils.line_end[lang])
                else:
                    #need to finish
                    if lang in ['c', 'cuda']:
                        line += get_array(lang, 'jac', lin_index) + ' += '
                    elif lang in ['fortran', 'matlab']:
                        line += (get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) +
                                 ' = ' + get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) + ' + ')
                    line += '(' + get_array(lang, 'dy', k_sp + offset)
                    line += ' * mw_avg * {} * rho_inv)'.format(__round_sig(sp_k.mw / sp_j.mw, 9))
                    line += utils.line_end[lang]
                    file.write(line)


            ######################################
            # Derivative with respect to species
            ######################################
            if touched[lin_index]:
                line = '  '
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
                    line += ' = ' + (get_array(lang, 'jac', 0, twod=j_sp + 1) + ' +' if touched[lin_index] else '') + ' -('
                else:
                    line += ' {}= -('.format('+' if touched[lin_index] else '')
                touched[lin_index] = True

                if lang in ['c', 'cuda']:
                    line += ('' + get_array(lang, 'h', k_sp) + ' * ('
                             '' + get_array(lang, 'jac', k_sp + 1 + (num_s + 1) * (j_sp + 1)) +
                             ' * cp_avg * rho' +
                             ' - (' + get_array(lang, 'cp', j_sp) + ' * ' + get_array(lang, 'dy', k_sp + offset) +
                             ' * {:.8e}))'.format(sp_k.mw)
                             )
                elif lang in ['fortran', 'matlab']:
                    line += ('' + get_array(lang, 'h', k_sp) + ' * ('
                             '' + get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) +
                             ' * cp_avg * rho' +
                             ' - (' + get_array(lang, 'cp', j_sp) + ' * ' + get_array(lang, 'dy', k_sp + offset) +
                             ' * {:.8e}))'.format(sp_k.mw)   
                             )
                line += ')' + utils.line_end[lang]
                file.write(line)

        if lang == 'cuda' and do_unroll and k_sp == next_fn_index - 1:
            #switch back
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
    #write_dt_y_comment(file, lang, sp)
    #write_dt_y(file, lang, specs, sp, isp, num_s, touched, sparse_indicies, offset, get_array)

    file.write('\n')

    #next need to divide everything out
    write_dt_y_division(file, lang, specs, num_s, get_array)

    #finish the dT entry
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

    #create include file
    if lang == 'cuda' and do_unroll:
        with open(os.path.join(path, 'jacobs', 'jac_include.h'), 'w') as tempfile:
            tempfile.write('#ifndef JAC_INCLUDE_H\n'
                       '#define JAC_INCLUDE_H\n')
            for i in range(jac_count):
                tempfile.write('#include "jacob_{}.h"\n'.format(i))
            tempfile.write('#endif\n\n')

        with open(os.path.join(path, 'jacobs', 'jac_list'), 'w') as tempfile:
            tempfile.write(' '.join(['jacob_{}.cu'.format(i) for i in range(jac_count)]))
    return sparse_indicies


def write_sparse_multiplier(path, lang, sparse_indicies, nvars):
    """Write a subroutine that multiplies the non-zero entries of the Jacobian with a column 'j' of another matrix

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
                file.write(' ' + utils.get_array(lang, 'w', index % nvars) + ' {}= '.format('+' if touched[index % nvars] else ''))
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


def create_jacobian(lang, mech_name, therm_name=None, optimize_cache=True, initial_state = "", num_blocks=8, 
    num_threads=64, no_shared=False, L1_preferred=True, multi_thread=1, force_optimize=False):
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
    force_optimize : boo, optional
        If true, redo the cache optimization even if the same mechanism

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
    build_path = './out/'
    utils.create_dir(build_path)

    # Interpret reaction mechanism file, depending on Cantera or
    # Chemkin format.
    if mech_name.endswith(tuple(['.cti','.xml'])):
        [elems, specs, reacs] = mech.read_mech_ct(mech_name)
    else:
        [elems, specs, reacs] = mech.read_mech(mech_name, therm_name)

    if optimize_cache:
        splittings, specs, reacs, rxn_rate_order, pdep_rate_order, spec_rate_order, \
        old_spec_order, old_rxn_order = cache.greedy_optimizer(lang, specs, reacs, multi_thread, force_optimize, build_path)
    else:
        spec_rate_order = [(range(len(specs)), range(len(reacs)))]
        rxn_rate_order = range(len(reacs))
        if any(r.pdep or r.thd for r in reacs): 
            pdep_rate_order = [x for x in range(len(reacs)) if reacs[x].pdep or reacs[x].thd]
        else:
            pdep_rate_order = None
        the_len = len(reacs)
        splittings = []
        while the_len > 0:
            splittings.append(min(CUDAParams.Jacob_Unroll, the_len))
            the_len -= CUDAParams.Jacob_Unroll
        old_spec_order = range(len(specs))
        old_rxn_order = range(len(reacs))
    
    if lang == 'cuda':
        CUDAParams.write_launch_bounds(build_path, num_blocks, num_threads, L1_preferred, no_shared)
    smm = None
    if lang == 'cuda' and not no_shared:
        smm = shared.shared_memory_manager(num_blocks, num_threads, L1_preferred)

    ## now begin writing subroutines

    # print reaction rate subroutine
    rate.write_rxn_rates(build_path, lang, specs, reacs, rxn_rate_order, smm)

    # if third-body/pressure-dependent reactions, 
    # print modification subroutine
    if next((r for r in reacs if (r.thd or r.pdep)), None):
        rate.write_rxn_pressure_mod(build_path, lang, specs, reacs, pdep_rate_order, smm)

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
    aux.write_mechanism_initializers(build_path, lang, specs, reacs, initial_state, old_spec_order, old_rxn_order, optimize_cache)

    if skip_jacob:
        return

    # write Jacobian subroutine
    sparse_indicies = write_jacobian(build_path, lang, specs, reacs, splittings, smm)

    write_sparse_multiplier(build_path, lang, sparse_indicies, len(specs) + 1)

    return None


if __name__ == "__main__":

    # command line arguments
    parser = ArgumentParser(description='Generates source code '
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
                        default = '',
                        required=False,
                        help = 'A comma separated list of initial initial conditions to set in the set_same_initial_conditions method. \
                                Expected Form: T,P,Species1=...,Species2=...,...\n\
                                Temperature in K\n\
                                Pressure in Atm\n\
                                Species in moles')
    #cuda specific
    parser.add_argument('-nco', '--no-cache-optimizer',
                        dest = 'cache_optimizer',
                        action = 'store_false',
                        default = True,
                        help = 'Attempt to optimize cache store/loading via use '
                        'of a greedy selection algorithm.')
    parser.add_argument('-nosmem', '--no-shared-memory',
                        dest='no_shared',
                        action='store_true',
                        default=False,
                        help = 'Use this option to turn off attempted shared memory acceleration for CUDA')
    parser.add_argument('-pshare', '--prefer-shared',
                        dest='L1_preferred',
                        action='store_false',
                        default=True,
                        help = 'Use this option to allocate more space for shared memory than the L1 cache for CUDA')
    parser.add_argument('-nb', '--num-blocks',
                        type=int,
                        dest='num_blocks',
                        default = 8,
                        required=False,
                        help = 'The target number of blocks / sm to achieve for CUDA.')
    parser.add_argument('-nt', '--num-threads',
                        type=int,
                        dest='num_threads',
                        default = 64,
                        required=False,
                        help = 'The target number of threads / block to achieve for CUDA.')
    parser.add_argument('-mt', '--multi-threaded',
                        type=int,
                        dest='multi_thread',
                        default=1,
                        required=False,
                        help = 'The number of threads to use during the optimization process')
    parser.add_argument('-fopt', '--force-optimize',
                        dest='force_optimize',
                        action='store_true',
                        default=False,
                        help='Use this option to force a reoptimization of the mechanism (usually only happens when generating for a different mechanism)')

    args = parser.parse_args()

    create_jacobian(args.lang, args.input, args.thermo, args.cache_optimizer, args.initial_conditions, args.num_blocks, args.num_threads\
                   , args.no_shared, args.L1_preferred, args.multi_thread, args.force_optimize)
