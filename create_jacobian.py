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
        reac_usages[rxn.reac.index(sp_name)] += 1
        if nu - 1 > 0:
            reac_usages[rxn.reac.index(sp_name)] += 1
            if rxn.thd:
                pres_mod_usage += 1


    for sp_name in rxn.prod:
        nu = rxn.prod_nu[rxn.prod.index(sp_name)]
        prod_usages[rxn.prod.index(sp_name)] += 1
        if nu - 1 > 0:
            prod_usages[rxn.prod.index(sp_name)] += 1

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

    return variable_list, usages   

def write_dr_dy(file, lang, rev_reacs, rxn, rind, pind, get_array):
    file.write('  j_temp = ')
    jline = ''
    # next, contribution from dR/dYj
    if rxn.pdep or rxn.thd:
        jline += '' + get_array(lang, 'pres_mod', pind)
        jline += ' * ('

    jline += '-mw_avg * ('
    #get reac and prod nu sums
    reac_nu = 0
    for sp_name in rxn.reac:
        nu = rxn.reac_nu[rxn.reac.index(sp_name)]
        if nu == 0:
            continue
        reac_nu += nu

    prod_nu = 0
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

    jline += ')'
    
    if rxn.pdep or rxn.thd:
        jline += ')'
    file.write(jline + utils.line_end[lang])

def write_pdep_dy(file, lang, rev_reacs, rxn, rind, pind, get_array):
    jline = ''
    if rxn.thd and not rxn.pdep:
        jline = '  thd_temp = '
        jline += '(-mw_avg * ' + get_array(lang, 'pres_mod', pind) + ' + rho'
        jline += ') * '     
    else:
        beta_0minf, E_0minf, k0kinf = get_infs(rxn)
        file.write('  pres_mod_temp = ')
        jline += '' + get_array(lang, 'pres_mod', pind)
        jline += ' * ('

         # dPr/dYj contribution
        if rxn.low:
            # unimolecular/recombination
            jline += '(1.0 / (1.0 + Pr))'
        elif rxn.high:
            # chem-activated bimolecular
            jline += '(-Pr / (1.0 + Pr))'

        if rxn.troe:
            jline += (' - log(Fcent) * 2.0 * A * (B * '
                      '{:.6}'.format(1.0 / math.log(10.0)) +
                      ' + A * '
                      '{:.6}) / '.format(0.14 / math.log(10.0)) +
                      '(B * B * B * (1.0 + A * A / (B * B)) '
                      '* (1.0 + A * A / (B * B)))'
                      )
        elif rxn.sri:
            jline += (' - X * X * '
                      '{:.6} * '.format(2.0 / math.log(10.0)) +
                      'log10(Pr) * '
                      'log({:.4} * '.format(rxn.sri[0]) +
                      'exp(-{:4} / T) + '.format(rxn.sri[1]) +
                      'exp(-T / {:.4}))'.format(rxn.sri[2])
                      )

        jline += ')'
        file.write(jline + utils.line_end[lang])
        jline = '  thd_temp = pres_mod_temp * (-mw_avg + (rho / conc_temp)) * '

    if rxn.rev:
        jline += '(' + get_array(lang, 'fwd_rates', rind)
        jline += ' - ' + \
            get_array(lang, 'rev_rates', rev_reacs.index(rxn))
        jline += ')'
    else:
        jline += get_array(lang, 'fwd_rates', rind)

    file.write(jline + utils.line_end[lang])


def write_dr_dy_species(lang, specs, rxn, pind, j_sp, sp_j, get_array):
    jline = ''
    if rxn.pdep or rxn.thd:
        jline += ' + '
    jline += 'j_temp'
    if (rxn.pdep or rxn.thd) and (sp_j.name in rxn.reac or (rxn.rev and sp_j.name in rxn.prod)):
        jline += ' + ' + get_array(lang, 'pres_mod', pind)
        jline += ' * ('

    if sp_j.name in rxn.reac:
        if not rxn.pdep and not rxn.thd:
            jline += ' + '
        nu = rxn.reac_nu[rxn.reac.index(sp_j.name)]
        if nu != 1:
            if nu == -1:
                jline += '-'
            else:
                jline += '{} * '.format(float(nu))
        jline += (rate.rxn_rate_const(rxn.A, rxn.b,
                                      rxn.E
                                      ) +
                  ' * rho '
                  )
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
            isp = next(i for i in xrange(len(specs))
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


        if not rxn.rev_par:
            jline += ('(' +
                      rate.rxn_rate_const(rxn.A,
                                          rxn.b,
                                          rxn.E
                                          ) +
                      ' / Kc)'
                      )
        else:
            # explicit reverse coefficients
            jline += rate.rxn_rate_const(rxn.rev_par[0],
                                         rxn.rev_par[1],
                                         rxn.rev_par[2]
                                         )

        jline += ' * rho'

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
        for sp_reac in rxn.prod:
            if sp_reac == sp_j.name:
                continue

            temp_nu = rxn.prod_nu[rxn.prod.index(sp_reac)]
            isp = next(i for i in xrange(len(specs))
                       if specs[i].name == sp_reac)
            if isinstance(temp_nu, float):
                jline += ' * pow(' + \
                    get_array(lang, 'conc', isp)
                jline += ', {})'.format(temp_nu)
            else:
                # integer, so just use multiplication
                jline += ''.join([' * ' + get_array(lang, 'conc', isp) for i in range(temp_nu)])
    
    if (rxn.pdep or rxn.thd) and (sp_j.name in rxn.reac or (rxn.rev and sp_j.name in rxn.prod)):
        jline += ')'
    return jline

def write_pdep_dy_species(lang, j_sp, sp_j, rev_reacs, rxn, rind, get_array):
    jline = 'thd_temp'
    mult = False
    if rxn.thd_body and not rxn.pdep:
        #check alphaij
        alphaij = next((thd[1] for thd in rxn.thd_body
                        if thd[0] == sp_j.name), None)
        if alphaij is not None and alphaij != 1.0:
            #we already baked the implicit one into there
            jline += ' + '
            jline += '{} * '.format(float(alphaij - 1.0))
            # default is 1.0
            jline += 'rho'
            mult = True
            
    elif rxn.pdep and not rxn.pdep_sp:
        #check alphaij
        alphaij = next((thd[1] for thd in rxn.thd_body
                        if thd[0] == sp_j.name), None)
        if alphaij is not None and alphaij != 1.0:
            #thd_temp needs to be multiplied still
            jline += ' + ('
            jline += 'pres_mod_temp * '
            jline += '{} * '.format(float(alphaij - 1.0))
            # default is 1.0
            jline += '(rho / conc_temp)'
            mult = True
            jline += ')'

    elif rxn.pdep and rxn.pdep_sp == sp_j.name:
        # NOTE: This Y array previously seems to be a bug, I can't find a reference to it anywhere else
        #      changing it to y
        jline += ' + (1.0 / ' + get_array(lang, 'y', j_sp)
        jline += ')'

    if mult:
        jline += ' * '
        if rxn.rev:
            jline += '(' + get_array(lang, 'fwd_rates', rind)
            jline += ' - ' + \
                get_array(lang, 'rev_rates', rev_reacs.index(rxn))
            jline += ')'
        else:
            jline += '' + get_array(lang, 'fwd_rates', rind)

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
            file.write('  Real dBdT[{}]'.format(len(specs)) + utils.line_end[lang])
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
                file.write('  Real ' + dBdT + utils.line_end[lang])

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

def write_dcp_dt(file, lang, sp, isp, sparse_indicies, get_array):
    line = '  if (T <= {:})'.format(sp.Trange[1])
    if lang in ['c', 'cuda']:
        line += ' {\n'
    elif lang == 'fortran':
        line += ' then\n'
    elif lang == 'matlab':
        line += '\n'
    file.write(line)

    line = '    working_temp'

    if lang in ['c', 'cuda']:
        line += ' {}= '.format('+' if isp > 0 else '')
    elif lang in ['fortran', 'matlab']:
        line += ' = {}'.format('working_temp + ' if isp > 0 else '')
    line += '' + get_array(lang, 'y', isp + 1)
    line += (' * {:.8e} * ('.format(chem.RU / sp.mw) +
             '{:.8e} + '.format(sp.lo[1]) +
             'T * ({:.8e} + '.format(2.0 * sp.lo[2]) +
             'T * ({:.8e} + '.format(3.0 * sp.lo[3]) +
             '{:.8e} * T)))'.format(4.0 * sp.lo[4]) +
             utils.line_end[lang]
             )
    file.write(line)

    if lang in ['c', 'cuda']:
        file.write('  } else {\n')
    elif lang in ['fortran', 'matlab']:
        file.write('  else\n')

    line = '    working_temp'

    if lang in ['c', 'cuda']:
        line += ' {}= '.format('+' if isp > 0 else '')
    elif lang in ['fortran', 'matlab']:
        line += ' = {}'.format('working_temp + ' if isp > 0 else '')
    line += '' + get_array(lang, 'y', isp + 1)
    line += (' * {:.8e} * ('.format(chem.RU / sp.mw) +
             '{:.8e} + '.format(sp.hi[1]) +
             'T * ({:.8e} + '.format(2.0 * sp.hi[2]) +
             'T * ({:.8e} + '.format(3.0 * sp.hi[3]) +
             '{:.8e} * T)))'.format(4.0 * sp.hi[4]) +
             utils.line_end[lang]
             )
    file.write(line)

    if lang in ['c', 'cuda']:
        file.write('  }\n\n')
    elif lang == 'fortran':
        file.write('  end if\n\n')
    elif lang == 'matlab':
        file.write('  end\n\n')

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

def write_cuda_intro(path, number, rate_list, this_rev, this_pdep, this_thd, this_troe, this_sri):
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
        file.write('const Real, const Real*')
        for rate in rate_list:
            file.write(', const Real*')
        if this_thd:
            file.write(', const Real')
        file.write(', const Real, const Real, const Real*, const Real, Real*'
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

    line += ('void eval_jacob_{} (const Real pres, '.format(number) + 
            'const Real * conc')
    for rate in rate_list:
        line += ', const Real* ' + rate
    if this_thd:
        line += ', const Real m'
    line += ', const Real mw_avg, const Real rho, const Real* dBdT, const Real T, Real* jac) {'
    file.write(line + '\n')

    file.write('  extern __shared__ double shared_temp[]' + utils.line_end[lang])
     # third-body variable needed for reactions
    if this_pdep and this_thd:
        line = '  '
        if lang == 'c':
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += ('conc_temp' +
                 utils.line_end[lang]
                 )
        file.write(line)


    # log(T)
    line = '  '
    if lang == 'c':
        line += 'Real '
    elif lang == 'cuda':
        line += 'register Real '
    line += ('logT = log(T)' +
             utils.line_end[lang]
             )
    file.write(line)

    line = '  '
    if lang == 'c':
        line += 'Real '
    elif lang == 'cuda':
        line += 'register Real '
    line += 'j_temp = 0.0' + utils.line_end[lang]
    file.write(line)

    if this_pdep:
        line = '  '
        if lang == 'c':
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += 'pres_mod_temp = 0.0' + utils.line_end[lang]
        file.write(line)

    if this_thd:
        line = '  '
        if lang == 'c':
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += 'thd_temp = 0.0' + utils.line_end[lang]
        file.write(line)

    # if any reverse reactions, will need Kc
    if this_rev:
        line = '  '
        if lang == 'c':
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += 'Kc = 0.0' + utils.line_end[lang]
        file.write(line)

    # pressure-dependence variables
    if this_pdep:
        line = '  '
        if lang == 'c':
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += 'Pr = 0.0' + utils.line_end[lang]
        file.write(line)

    if this_troe:
        line = ''.join(['  Real {} = 0.0{}'.format(x, utils.line_end[lang]) for x in 'Fcent', 'A', 'B', 'lnF_AB'])
        file.write(line)

    if this_sri:
        line = '  Real X = 0.0' + utils.line_end[lang]
        file.write(line)

    return file




def write_jacobian_alt(path, lang, specs, reacs, splittings=None, smm=None):
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
                   'void eval_jacob (const Real, const Real, Real*, '
                   'Real*);\n'
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
                   '__device__ void eval_jacob (const Real, const Real, '
                   'Real*, Real*);\n'
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

        if CUDAParams.is_global():
            file.write('extern __device__ double* conc;\n')
            if len(rev_reacs):
                file.write('extern __device__ Real* fwd_rates;\n'
                           'extern __device__ Real* rev_rates;\n'
                           )
            else:
                file.write('extern __device__ Real* rates;\n')
            if len(pdep_reacs):
                file.write('extern __device__ Real* pres_mod;\n')
            file.write('extern __device__ Real* dy;\n')
            file.write('#ifdef CONP\n'
                       'extern __device__ Real* h;\n'
                       'extern __device__ Real* cp;\n'
                       '#else\n'
                       'extern __device__ Real* u;\n'
                       'extern __device__ Real* cv;\n'
                       '#endif\n')

    line = ''
    if lang == 'cuda':
        line = '__device__ '

    offset = 0
    if lang == 'cuda' and CUDAParams.is_global():
        offset = 1

    if lang in ['c', 'cuda']:
        line += ('void eval_jacob (const Real t, const Real pres, '
                 'Real * y, Real * jac) {\n\n')
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
        line = '  Real T = ' + get_array(lang, 'y', 0)
    elif lang in ['fortran', 'matlab']:
        line = '  T = ' + get_array(lang, 'y', 0)
    line += utils.line_end[lang]
    file.write(line)

    file.write('\n')

    # calculation of average molecular weight
    if lang in ['c', 'cuda']:
        file.write('  // average molecular weight\n'
                   '  Real mw_avg;\n'
                   )
    elif lang == 'fortran':
        file.write('  ! average molecular weight\n')
    elif lang == 'matlab':
        file.write('  % average molecular weight\n')
    line = '  mw_avg = '
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

        if not isfirst:
            line += ' + '
        line += '(' + get_array(lang, 'y', specs.index(sp) + 1) + \
            ' / {:})'.format(sp.mw)
        isfirst = False

    line += utils.line_end[lang]
    file.write(line)
    line = '  mw_avg = 1.0 / mw_avg' + utils.line_end[lang]
    file.write(line)

    if lang in ['c', 'cuda']:
        file.write('  // mass-averaged density\n'
                   '  Real rho;\n'
                   )
    elif lang == 'fortran':
        file.write('  ! mass-averaged density\n')
    elif lang == 'matlab':
        file.write('  % mass-averaged density\n')
    line = '  rho = pres * mw_avg / ({:.8e} * T)'.format(chem.RU)
    line += utils.line_end[lang]
    file.write(line)

    file.write('\n')

    # evaluate species molar concentrations
    if lang in ['c', 'cuda']:
        if lang != 'cuda' or not CUDAParams.is_global():
            file.write('  // species molar concentrations\n'
                       '  Real conc[{}];\n'.format(num_s)
                       )
    elif lang == 'fortran':
        file.write('  ! species molar concentrations\n')
    elif lang == 'matlab':
        file.write('  % species molar concentrations\n'
                   '  conc = zeros({},1);\n'.format(num_s)
                   )
    # loop through species
    for sp in specs:
        isp = specs.index(sp)
        line = ('  ' + get_array(lang, 'conc', isp) +
                ' = rho * ' +
                '' + get_array(lang, 'y', isp + 1) +
                ' / {}'.format(sp.mw)
                )
        line += utils.line_end[lang]
        file.write(line)
    file.write('\n')

    rate_list = ['fwd_rates']
    if len(rev_reacs):
        rate_list.append('rev_rates')
    if len(pdep_reacs):
        rate_list.append('pres_mod')
    rate_list.append('dy')
    # evaluate forward and reverse reaction rates
    if lang in ['c', 'cuda']:
        if lang != 'cuda' or not CUDAParams.is_global():
            file.write('  // evaluate reaction rates\n'
                       '  Real fwd_rates[{}];\n'.format(num_r)
                       )
        if rev_reacs:
            if lang != 'cuda' or not CUDAParams.is_global():
                file.write('  Real rev_rates[{}];\n'.format(num_rev))
            file.write('  eval_rxn_rates (T, conc, fwd_rates, '
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
        file.write('  // get pressure modifications to reaction rates\n')
        if lang != 'cuda' or not CUDAParams.is_global():
            file.write('  Real pres_mod[{}];\n'.format(num_pdep))
        file.write('  get_rxn_pres_mod (T, pres, conc, pres_mod);\n')
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
        if lang != 'cuda' or not CUDAParams.is_global():
            file.write('  Real dy[{}];\n'.format(num_s))
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
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += ('m = pres / ({:4e} * T)'.format(chem.RU) +
                 utils.line_end[lang]
                 )
        file.write(line)

        if not (lang == 'cuda' and do_unroll):
            line = '  '
            if lang == 'c':
                line += 'Real '
            elif lang == 'cuda':
                line += 'register Real '
            line += ('conc_temp' +
                     utils.line_end[lang]
                     )
            file.write(line)


    # log(T)
    line = '  '
    if lang == 'c':
        line += 'Real '
    elif lang == 'cuda':
        line += 'register Real '
    line += ('logT = log(T)' +
             utils.line_end[lang]
             )
    file.write(line)

    line = '  '
    if lang == 'c':
        line += 'Real '
    elif lang == 'cuda':
        line += 'register Real '
    line += 'j_temp = 0.0' + utils.line_end[lang]
    file.write(line)

    if any(rxn.pdep for rxn in reacs) and not (lang == 'cuda' and do_unroll):
        line = '  '
        if lang == 'c':
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += 'pres_mod_temp = 0.0' + utils.line_end[lang]
        file.write(line)

    if any(rxn.thd for rxn in reacs) and not (lang == 'cuda' and do_unroll):
        line = '  '
        if lang == 'c':
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += 'thd_temp = 0.0' + utils.line_end[lang]
        file.write(line)

    # if any reverse reactions, will need Kc
    if rev_reacs and not (lang == 'cuda' and do_unroll):
        line = '  '
        if lang == 'c':
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += 'Kc = 0.0' + utils.line_end[lang]
        file.write(line)

    # pressure-dependence variables
    if any(rxn.pdep for rxn in reacs) and not (lang == 'cuda' and do_unroll):
        line = '  '
        if lang == 'c':
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += 'Pr = 0.0' + utils.line_end[lang]
        file.write(line)

    if any(rxn.troe for rxn in reacs) and not (lang == 'cuda' and do_unroll):
        line = ''.join(['  Real {} = 0.0{}'.format(x, utils.line_end[lang]) for x in 'Fcent', 'A', 'B', 'lnF_AB'])
        file.write(line)

    if any(rxn.sri for rxn in reacs) and not (lang == 'cuda' and do_unroll):
        line = '  Real X = 0.0' + utils.line_end[lang]
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
            file = write_cuda_intro(os.path.join(path, 'jacobs'), jac_count, rate_list, rev, pdep, thd, troe, sri)

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

        if rxn.pdep or rxn.thd:
            #need to write the pdep parts (independent of any species)
            write_pdep_dy(file, lang, rev_reacs, rxn, rind, pind, get_array)

        #need to write the dr/dy parts (independent of any species)
        write_dr_dy(file, lang, rev_reacs, rxn, rind, pind, get_array)

        #now loop through each species
        for j_sp, sp_j in enumerate(specs):
            isFirst = True
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

                if isFirst:
                    working_temp = '('

                    #check if there is any species specific stuff we need to do
                    if rxn.thd or rxn.pdep:
                        working_temp += write_pdep_dy_species(lang, j_sp, sp_j, rev_reacs, rxn, rind, get_array)
                    
                    working_temp += write_dr_dy_species(lang, specs, rxn, pind, j_sp, sp_j, get_array)

                    working_temp += ') * {:.8e}'.format(1.0 / __round_sig(sp_j.mw, 9))

                    isFirst = False

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


    if lang == 'cuda' and do_unroll:
        #create include file
        with open(os.path.join(path, 'jacobs', 'jac_include.h'), 'w') as tempfile:
            tempfile.write('#ifndef JAC_INCLUDE_H\n'
                       '#define JAC_INCLUDE_H\n')
            for i in range(jac_count):
                tempfile.write('#include "jacob_{}.h"\n'.format(i))
            tempfile.write('#endif\n\n')

        with open(os.path.join(path, 'jacobs', 'jac_list'), 'w') as tempfile:
            tempfile.write(' '.join(['jacob_{}.cu'.format(i) for i in range(jac_count)]))


    line = '  '
    if lang == 'c':
        line += 'Real '
    elif lang == 'cuda':
        line += 'register Real '
    line += 'rho_inv = 1.0 / rho' + utils.line_end[lang]
    file.write(line)

    ###################################
    # Partial derivatives of temperature (energy equation)
    ###################################

    # evaluate enthalpy
    if lang in ['c', 'cuda']:
        file.write('  // species enthalpies\n')
        if lang != 'cuda' or not CUDAParams.is_global():
            file.write('  Real h[{}];\n'.format(num_s))
        file.write('  eval_h(T, h);\n')
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
        file.write('  // species specific heats\n')
        if lang != 'cuda' or not CUDAParams.is_global():
            file.write('  Real cp[{}];\n'.format(num_s))
        file.write('  eval_cp(T, cp);\n')
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
                   '  Real cp_avg;\n'
                   )
    elif lang == 'cuda':
        file.write('  // average specific heat\n'
                   '  register Real cp_avg;\n'
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
        line += 'Real '
    elif lang == 'cuda':
        line += 'register Real '
    line += 'working_temp = 0.0' + utils.line_end[lang]
    file.write(line)
               
    #need to finish the dYk/dYj's
    write_dy_y_finish_comment(file, lang)
    for k_sp, sp_k in enumerate(specs):
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
                        line += get_array(lang, 'jac', lin_index) + ' = ('
                        line += get_array(lang, 'jac', lin_index) + ' + '
                    elif lang in ['fortran', 'matlab']:
                        line += (get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) +
                                 ' = (' + get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) + ' + ')
                    line += get_array(lang, 'dy', k_sp + offset)
                    line += ' * mw_avg * {}'.format(1.0 / __round_sig(sp_j.mw, 9))
                    line += ') * {:.8e} * rho_inv'.format(sp_k.mw)
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

    ######################################
    # Derivatives with respect to temperature
    ######################################

    for isp, sp in enumerate(specs):
        write_dt_t_comment(file, lang, sp)
        #write dcp/dT for this species
        write_dcp_dt(file, lang, sp, isp, sparse_indicies, get_array)

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
    return sparse_indicies



def write_jacobian(path, lang, specs, reacs):
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

    Returns
    -------
    None

    """

    # first write header file
    if lang == 'c':
        file = open(path + 'jacob.h', 'w')
        file.write('#ifndef JACOB_HEAD\n'
                   '#define JACOB_HEAD\n'
                   '\n'
                   '#include "header.h"\n'
                   '\n'
                   'void eval_jacob (const Real, const Real, Real*, '
                   'Real*);\n'
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
                   '__device__ void eval_jacob (const Real, const Real, '
                   'Real*, Real*);\n'
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
                   '#include "header.h"\n'
                   '#include "chem_utils.cuh"\n'
                   '#include "rates.cuh"\n'
                   '#include "gpu_macros.cuh"\n'
                   '\n'
                   )

        if CUDAParams.is_global():
            file.write('extern __device__ double* conc;\n')
            if len(rev_reacs):
                file.write('extern __device__ Real* fwd_rates;\n'
                           'extern __device__ Real* rev_rates;\n'
                           )
            else:
                file.write('extern __device__ Real* rates;\n')
            if len(pdep_reacs):
                file.write('extern __device__ Real* pres_mod;\n')
            file.write('extern __device__ Real* dy;\n')
            file.write('#ifdef CONP\n'
                       'extern __device__ Real* h;\n'
                       'extern __device__ Real* cp;\n'
                       '#else\n'
                       'extern __device__ Real* u;\n'
                       'extern __device__ Real* cv;\n'
                       '#endif\n')

    line = ''
    if lang == 'cuda':
        line = '__device__ '

    offset = 0
    if lang == 'cuda' and CUDAParams.is_global():
        offset = 1

    if lang in ['c', 'cuda']:
        line += ('void eval_jacob (const Real t, const Real pres, '
                 'Real * y, Real * jac) {\n\n')
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

    # get temperature
    if lang in ['c', 'cuda']:
        line = '  Real T = ' + utils.get_array(lang, 'y', 0)
    elif lang in ['fortran', 'matlab']:
        line = '  T = ' + utils.get_array(lang, 'y', 0)
    line += utils.line_end[lang]
    file.write(line)

    file.write('\n')

    # calculation of average molecular weight
    if lang in ['c', 'cuda']:
        file.write('  // average molecular weight\n'
                   '  Real mw_avg;\n'
                   )
    elif lang == 'fortran':
        file.write('  ! average molecular weight\n')
    elif lang == 'matlab':
        file.write('  % average molecular weight\n')
    line = '  mw_avg = '
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

        if not isfirst:
            line += ' + '
        line += '(' + utils.get_array(lang, 'y', specs.index(sp) + 1) + \
            ' / {:})'.format(sp.mw)
        isfirst = False

    line += utils.line_end[lang]
    file.write(line)
    line = '  mw_avg = 1.0 / mw_avg' + utils.line_end[lang]
    file.write(line)

    if lang in ['c', 'cuda']:
        file.write('  // mass-averaged density\n'
                   '  Real rho;\n'
                   )
    elif lang == 'fortran':
        file.write('  ! mass-averaged density\n')
    elif lang == 'matlab':
        file.write('  % mass-averaged density\n')
    line = '  rho = pres * mw_avg / ({:.8e} * T)'.format(chem.RU)
    line += utils.line_end[lang]
    file.write(line)

    file.write('\n')

    # evaluate species molar concentrations
    if lang in ['c', 'cuda']:
        if lang != 'cuda' or not CUDAParams.is_global():
            file.write('  // species molar concentrations\n'
                       '  Real conc[{}];\n'.format(num_s)
                       )
    elif lang == 'fortran':
        file.write('  ! species molar concentrations\n')
    elif lang == 'matlab':
        file.write('  % species molar concentrations\n'
                   '  conc = zeros({},1);\n'.format(num_s)
                   )
    # loop through species
    for sp in specs:
        isp = specs.index(sp)
        line = ('  ' + utils.get_array(lang, 'conc', isp) +
                ' = rho * ' +
                '' + utils.get_array(lang, 'y', isp + 1) +
                ' / {}'.format(sp.mw)
                )
        line += utils.line_end[lang]
        file.write(line)
    file.write('\n')

    # evaluate forward and reverse reaction rates
    if lang in ['c', 'cuda']:
        if lang != 'cuda' or not CUDAParams.is_global():
            file.write('  // evaluate reaction rates\n'
                       '  Real fwd_rates[{}];\n'.format(num_r)
                       )
        if rev_reacs:
            if lang != 'cuda' or not CUDAParams.is_global():
                file.write('  Real rev_rates[{}];\n'.format(num_rev))
            file.write('  eval_rxn_rates (T, conc, fwd_rates, '
                       'rev_rates);\n'
                       )
        else:
            file.write('  eval_rxn_rates (T, conc, fwd_rates);\n')
    elif lang == 'fortran':
        file.write('  ! evaluate reaction rates\n')
        if rev_reacs:
            file.write('  eval_rxn_rates (T, conc, fwd_rates, '
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
        file.write('  // get pressure modifications to reaction rates\n')
        if lang != 'cuda' or not CUDAParams.is_global():
            file.write('  Real pres_mod[{}];\n'.format(num_pdep))
        file.write('  get_rxn_pres_mod (T, pres, conc, pres_mod);\n')
    elif lang == 'fortran':
        file.write('  ! get and evaluate pressure modifications to '
                   'reaction rates\n'
                   '  get_rxn_pres_mod (T, pres, conc, pres_mod)\n'
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
        if lang != 'cuda' or not CUDAParams.is_global():
            file.write('  Real dy[{}];\n'.format(num_s))
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
            file.write('  eval_spec_rates (fwd_rates, rev_rates, '
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
    if any(rxn.thd for rxn in reacs):
        line = '  '
        if lang == 'c':
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += ('m = pres / ({:4e} * T)'.format(chem.RU) +
                 utils.line_end[lang]
                 )
        file.write(line)

    # log(T)
    line = '  '
    if lang == 'c':
        line += 'Real '
    elif lang == 'cuda':
        line += 'register Real '
    line += ('logT = log(T)' +
             utils.line_end[lang]
             )
    file.write(line)

    # if any reverse reactions, will need Kc
    if rev_reacs:
        line = ('  Real Kc' +
                utils.line_end[lang]
                )
        file.write(line)

    # pressure-dependence variables
    if any(rxn.pdep for rxn in reacs):
        line = '  '
        if lang == 'c':
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += 'Pr' + utils.line_end[lang]
        file.write(line)

    if any(rxn.troe for rxn in reacs):
        line = '  Real Fcent, A, B, lnF_AB' + utils.line_end[lang]
        file.write(line)

    if any(rxn.sri for rxn in reacs):
        line = '  Real X' + utils.line_end[lang]
        file.write(line)

    # variables for equilibrium constant derivatives, if needed
    dBdT_flag = []
    for sp in specs:
        dBdT_flag.append(False)

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

            dBdT = 'dBdT_'
            if lang in ['c', 'cuda']:
                dBdT += str(sp_ind)
            elif lang in ['fortran', 'matlab']:
                dBdT += str(sp_ind + 1)
            # declare dBdT
            file.write('  Real ' + dBdT + utils.line_end[lang])

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

    line = ''

    ###################################
    # now begin Jacobian evaluation
    ###################################

    # index list for sparse multiplier
    sparse_indicies = []

    ###################################
    # partial derivatives of species
    ###################################
    for sp_k in specs:

        ######################################
        # with respect to temperature
        ######################################
        k_sp = specs.index(sp_k)

        if lang in ['c', 'cuda']:
            line = '  //'
        elif lang == 'fortran':
            line = '  !'
        elif lang == 'matlab':
            line = '  %'
        line += ('partial of omega_' + sp_k.name + ' wrt T' +
                 utils.line_end[lang]
                 )
        file.write(line)

        isfirst = True

        for rxn in reacs:
            rind = reacs.index(rxn)

            if sp_k.name in rxn.prod and sp_k.name in rxn.reac:
                nu = (rxn.prod_nu[rxn.prod.index(sp_k.name)] -
                      rxn.reac_nu[rxn.reac.index(sp_k.name)])
                # check if net production zero
                if nu == 0:
                    continue
            elif sp_k.name in rxn.prod:
                nu = rxn.prod_nu[rxn.prod.index(sp_k.name)]
            elif sp_k.name in rxn.reac:
                nu = -rxn.reac_nu[rxn.reac.index(sp_k.name)]
            else:
                # doesn't participate in reaction
                continue

            # start contribution to Jacobian entry for reaction
            jline = '  ' + utils.get_array(lang, 'jac', k_sp + 1, twod=1)
            sparse_indicies.append(k_sp + 1)

            # first reaction for this species
            if isfirst:
                jline += ' = '

                if nu != 1:
                    jline += '{} * '.format(float(nu))
                isfirst = False
            else:
                if lang in ['c', 'cuda']:
                    jline += ' += '
                elif lang in ['fortran', 'matlab']:
                    jline += ' = ' + utils.get_array(lang, 'jac', k_sp, twod=1) + ' + '

                if nu != 1:
                    jline += '{} * '.format(float(nu))

            jline += '('

            if rxn.pdep:
                # print lines for necessary pressure-dependent variables
                line = '  Pr = '
                pind = pdep_reacs.index(rind)

                if rxn.pdep_sp:
                    line += utils.get_array(lang, 'conc', specs.index(rxn.pdep_sp))
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
                            line += utils.get_array(lang, 'conc', isp)
                    line += ')'

                jline += '' + utils.get_array(lang, 'pres_mod', pind)
                jline += '* (('

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

                    #NOTE: I think this was a bug
                    jline += '-Pr * '

                # finish writing P_ri
                line += (' * (' + k0kinf + ')' +
                         utils.line_end[lang]
                         )
                file.write(line)

                # dPr/dT
                jline += ('({:.4e} + ('.format(beta_0minf) +
                          '{:.8e} / T) - 1.0) / '.format(E_0minf) +
                          '(T * (1.0 + Pr)))'
                          )

                # dF/dT
                if rxn.troe:
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

                    jline += (' + (((1.0 / '
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

                elif rxn.sri:
                    line = ('  X = 1.0 / (1.0 + log10(Pr) * log10(Pr))' +
                            utils.line_end[lang]
                            )
                    file.write(line)

                    jline += (' + X * ((('
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

                # lindemann, dF/dT = 0

                jline += ') * '

                if rxn.rev:
                    # forward and reverse reaction rates
                    jline += '(' + utils.get_array(lang, 'fwd_rates', rind)
                    jline += ' - ' + \
                        utils.get_array(lang, 'rev_rates', rev_reacs.index(rxn))
                    jline += ')'
                else:
                    # forward reaction rate only
                    jline += '' + utils.get_array(lang, 'fwd_rates', rind)

                jline += ' + (' + utils.get_array(lang, 'pres_mod', pind)

            else:
                # not pressure dependent

                # third body reaction
                if rxn.thd:
                    pind = pdep_reacs.index(rind)

                    jline += '(-' + utils.get_array(lang, 'pres_mod', pind)
                    jline += ' * '

                    if rxn.rev:
                        # forward and reverse reaction rates
                        jline += '(' + utils.get_array(lang, 'fwd_rates', rind)
                        jline += ' - ' + \
                            utils.get_array(lang, 'rev_rates', rev_reacs.index(rxn))
                        jline += ')'
                    else:
                        # forward reaction rate only
                        jline += '' + utils.get_array(lang, 'fwd_rates', rind)

                    jline += ' / T) + (' + utils.get_array(lang, 'pres_mod', pind)

                else:
                    if lang in ['c', 'cuda', 'matlab']:
                        jline += '(1.0'
                    elif lang in ['fortran']:
                        jline += '(1.0_wp'

            jline += ' / T) * ('

            # contribution from temperature derivative of forward reaction rate
            jline += '' + utils.get_array(lang, 'fwd_rates', rind)

            jline += ' * ('
            if (abs(rxn.b) > 1.0e-90) and (abs(rxn.E) > 1.0e-90):
                jline += '{:.8e} + ({:.8e} / T)'.format(rxn.b, rxn.E)
            elif abs(rxn.b) > 1.0e-90:
                jline += '{:.8e}'.format(rxn.b)
            elif abs(rxn.E) > 1.0e-90:
                jline += '({:.8e} / T)'.format(rxn.E)
            jline += '{}1.0 - '.format(' + ' if (abs(rxn.b)
                                                 > 1.0e-90) or (abs(rxn.E) > 1.0e-90) else '')

            # loop over reactants
            nu = 0
            for sp in rxn.reac:
                nu += rxn.reac_nu[rxn.reac.index(sp)]
            jline += '{})'.format(float(nu))

            # contribution from temperature derivative of reaction rates
            if rxn.rev:
                # reversible reaction

                jline += ' - ' + \
                    utils.get_array(lang, 'rev_rates', rev_reacs.index(rxn))

                if rxn.rev_par:
                    # explicit reverse parameters

                    jline += ' * ('
                    # check which parameters are > 0 (effectively)
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

                    nu = 0
                    # loop over products
                    for sp in rxn.prod:
                        nu += rxn.prod_nu[rxn.prod.index(sp)]
                    jline += '{})'.format(float(nu))

                else:
                    # reverse rate constant from forward and equilibrium

                    jline += ' * ('
                    if abs(rxn.b) > 1.0e-90 and abs(rxn.E) > 1.0e-90:
                        jline += '{:.8e} + ({:.8e} / T)'.format(rxn.b, rxn.E)
                    elif abs(rxn.b) > 1.0e-90:
                        jline += '{:.8e}'.format(rxn.b)
                    elif abs(rxn.E) > 1.0e-90:
                        jline += '({:.8e} / T)'.format(rxn.E)
                    jline += '{}1.0 - '.format(
                        ' + ' if (abs(rxn.b) > 1.0e-90) or (abs(rxn.E) > 1.0e-90) else '')

                    nu = 0
                    # loop over products
                    for sp in rxn.prod:
                        nu += rxn.prod_nu[rxn.prod.index(sp)]
                    jline += '{} - T * ('.format(float(nu))

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

                        dBdT = 'dBdT_'
                        if lang in ['c', 'cuda']:
                            dBdT += str(sp_ind)
                        elif lang in ['fortran', 'matlab']:
                            dBdT += str(sp_ind + 1)

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

                        dBdT = 'dBdT_'
                        if lang in ['c', 'cuda']:
                            dBdT += str(sp_ind)
                        elif lang in ['fortran', 'matlab']:
                            dBdT += str(sp_ind + 1)

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

            # else:
                # irreversible reaction
                #jline += ')'

            jline += '))'

            # print line for reaction
            jline += utils.line_end[lang]
            file.write(jline)

        if isfirst:
            # not participating in any reactions,
            # or at least no net production

            # NOTE: I believe there was a bug here w/ the previous
            # fortran/matlab code (as it looks like it would be zero indexed)
            line = '  ' + utils.get_array(lang, 'jac', k_sp + 1)
            line += ' = 0.0'
        else:
            line = '  '
            if lang in ['c', 'cuda']:
                line += (utils.get_array(lang, 'jac', k_sp + 1) +
                         ' *= {:.8e} / rho'.format(sp_k.mw)
                         )
            elif lang in ['fortran', 'matlab']:
                # NOTE: I believe there was a bug here w/ the previous
                # fortran/matlab code (as it looks like it would be zero
                # indexed)
                line += (utils.get_array(lang, k_sp + 1, twod=1) +
                         ' = ' + utils.get_array(lang, 'jac', k_sp + 1, twod=1) +
                         ' * {:.8e} / rho'.format(sp_k.mw)
                         )

        line += utils.line_end[lang]
        file.write(line)
        file.write('\n')

        ###############################
        # Derivatives with respect to species mass fractions
        ###############################
        for sp_j in specs:
            j_sp = specs.index(sp_j)

            if lang in ['c', 'cuda']:
                line = '  //'
            elif lang == 'fortran':
                line = '  !'
            elif lang == 'matlab':
                line = '  %'
            line += ('partial of omega_' + sp_k.name + ' wrt Y_' +
                     sp_j.name + utils.line_end[lang]
                     )
            file.write(line)

            isfirst = True

            for rxn in reacs:
                rind = reacs.index(rxn)

                if sp_k.name in rxn.prod and sp_k.name in rxn.reac:
                    nu = (rxn.prod_nu[rxn.prod.index(sp_k.name)] -
                          rxn.reac_nu[rxn.reac.index(sp_k.name)])
                    # check if net production zero
                    if nu == 0:
                        continue
                elif sp_k.name in rxn.prod:
                    nu = rxn.prod_nu[rxn.prod.index(sp_k.name)]
                elif sp_k.name in rxn.reac:
                    nu = -rxn.reac_nu[rxn.reac.index(sp_k.name)]
                else:
                    # doesn't participate in reaction
                    continue

                # start contribution to Jacobian entry for reaction
                jline = '  '
                if lang in ['c', 'cuda']:
                    jline += utils.get_array(lang, 'jac',
                                             k_sp + 1 + (num_s + 1) * (j_sp + 1))
                    sparse_indicies.append(k_sp + 1 + (num_s + 1) * (j_sp + 1))
                elif lang in ['fortran', 'matlab']:
                    jline += utils.get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1)
                    sparse_indicies.append((k_sp + 1, j_sp + 1))

                if isfirst:
                    jline += ' = '
                    isfirst = False
                else:
                    if lang in ['c', 'cuda']:
                        jline += ' += '
                    elif lang in ['fortran', 'matlab']:
                        jline += utils.get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1)

                if nu != 1:
                    jline += '{} * '.format(float(nu))

                # start reaction
                jline += '('

                if rxn.thd and not rxn.pdep:
                    # third-body reaction
                    pind = pdep_reacs.index(rind)

                    jline += '(-mw_avg * ' + \
                        utils.get_array(lang, 'pres_mod', pind)
                    jline += ' / {:.8e}'.format(sp_j.mw)

                    # check if species of interest is third body in reaction
                    alphaij = next((thd[1] for thd in rxn.thd_body
                                    if thd[0] == sp_j.name), None)
                    if alphaij != 0.0:
                        jline += ' + '
                        if alphaij:
                            jline += '{} * '.format(float(alphaij))
                        # default is 1.0
                        jline += 'rho / {:.8e}'.format(sp_j.mw)
                    jline += ') * '

                    if rxn.rev:
                        jline += '(' + utils.get_array(lang, 'fwd_rates', rind)
                        jline += ' - ' + \
                            utils.get_array(lang, 'rev_rates', rev_reacs.index(rxn))
                        jline += ')'
                    else:
                        jline += '' + utils.get_array(lang, 'fwd_rates', rind)

                    jline += ' + '
                elif rxn.pdep:
                    # pressure-dependent reaction

                    line = '  Pr = '
                    pind = pdep_reacs.index(rind)

                    if rxn.pdep_sp:
                        line += utils.get_array(lang, 'conc', specs.index(rxn.pdep_sp))
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
                                line += utils.get_array(lang, 'conc', isp)
                        line += ')'

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
                    line += (' * (' + k0kinf + ')' +
                             utils.line_end[lang]
                             )
                    file.write(line)

                    jline += '' + utils.get_array(lang, 'pres_mod', pind)
                    jline += ' * ('

                    # dPr/dYj contribution
                    if rxn.low:
                        # unimolecular/recombination
                        jline += '(1.0 / (1.0 + Pr))'
                    elif rxn.high:
                        # chem-activated bimolecular
                        jline += '(-Pr / (1.0 + Pr))'

                    # dF/dYj contribution
                    if rxn.troe:
                        line = ('  Fcent = '
                                '{:.4e} * '.format(1.0 - rxn.troe_par[0]) +
                                'exp(-T / {:.4e})'.format(rxn.troe_par[1]) +
                                ' + {:.4e} * '.format(rxn.troe_par[0]) +
                                'exp(T / {:.4e})'.format(rxn.troe_par[2])
                                )
                        if len(rxn.troe_par) == 4:
                            line += (' + exp(-'
                                     '{:.4e} / T)'.format(rxn.troe_par[3])
                                     )
                        line += utils.line_end[lang]
                        file.write(line)

                        line = ('  A = log10(Pr) - 0.67 * '
                                'log10(Fcent) - 0.4' +
                                utils.line_end[lang]
                                )
                        file.write(line)

                        line = ('  B = 0.806 - 1.1762 * log10(Fcent) - '
                                '0.14 * log10(Pr)' +
                                utils.line_end[lang]
                                )
                        file.write(line)

                        jline += (' - log(Fcent) * 2.0 * A * (B * '
                                  '{:.6}'.format(1.0 / math.log(10.0)) +
                                  ' + A * '
                                  '{:.6}) / '.format(0.14 / math.log(10.0)) +
                                  '(B * B * B * (1.0 + A * A / (B * B)) '
                                  '* (1.0 + A * A / (B * B)))'
                                  )

                    elif rxn.sri:
                        file.write('  X = 1.0 / (1.0 + log10(Pr) * '
                                   'log10(Pr))' +
                                   utils.line_end[lang]
                                   )

                        jline += (' - X * X * '
                                  '{:.6} * '.format(2.0 / math.log(10.0)) +
                                  'log10(Pr) * '
                                  'log({:.4} * '.format(rxn.sri[0]) +
                                  'exp(-{:4} / T) + '.format(rxn.sri[1]) +
                                  'exp(-T / {:.4}))'.format(rxn.sri[2])
                                  )
                    jline += ') * '

                    # dPr/dYj part
                    jline += '(-mw_avg / {:.8e}'.format(sp_j.mw)

                    if rxn.thd_body:

                        alphaij = next((thd[1] for thd in rxn.thd_body
                                        if thd[0] == sp_j.name), None)

                        # need to make sure alpha isn't 0.0
                        if alphaij != 0.0:
                            jline += ' + rho'
                            if alphaij:
                                jline += ' * {}'.format(float(alphaij))

                            jline += ' / ((m'
                            for thd_sp in rxn.thd_body:
                                i = next((s for s in specs if s.name ==
                                          thd_sp[0]), None)
                                isp = specs.index(i)
                                if thd_sp[1] > 1.0:
                                    line += ' + {} * '.format(thd_sp[1] - 1.0)
                                elif thd_sp[1] < 1.0:
                                    line += ' - {} * '.format(1.0 - thd_sp[1])
                                if thd_sp[1] != 1.0:
                                    line += utils.get_array(lang, 'conc', isp)
                            jline += ') * {:.8e})'.format(sp_j.mw)
                    elif rxn.pdep and rxn.pdep_sp == sp_j.name:
                        # NOTE: This Y array previously seems to be a bug, I can't find a reference to it anywhere else
                        #      changing it to y
                        jline += ' + (1.0 / ' + utils.get_array(lang, 'y', j_sp)
                        jline += ')'
                    jline += ') * '

                    if rxn.rev:
                        jline += '(' + utils.get_array(lang, 'fwd_rates', rind)
                        jline += ' - ' + \
                            utils.get_array(lang, 'rev_rates', rev_reacs.index(rxn))
                        jline += ')'
                    else:
                        jline += '' + utils.get_array(lang, 'fwd_rates', rind)

                    jline += ' + '

                # next, contribution from dR/dYj
                if rxn.pdep or rxn.thd:
                    jline += '' + utils.get_array(lang, 'pres_mod', pind)
                    jline += ' * ('

                firstSp = True
                for sp_l in specs:
                    l_sp = specs.index(sp_l)

                    # contribution from forward
                    if sp_l.name in rxn.reac:

                        if not firstSp:
                            jline += ' + '
                        firstSp = False

                        nu = rxn.reac_nu[rxn.reac.index(sp_l.name)]
                        if nu != 1:
                            jline += '{} * '.format(float(nu))
                        jline += '('

                        jline += '-mw_avg * ' + \
                            utils.get_array(lang, 'fwd_rates', rind)
                        jline += ' / {:.8e}'.format(sp_j.mw)

                        # only contribution from 2nd part of sp_l is sp_j
                        if sp_l is sp_j:
                            jline += (' + ' +
                                      rate.rxn_rate_const(rxn.A, rxn.b,
                                                          rxn.E
                                                          ) +
                                      ' * (rho / {:.8e})'.format(sp_l.mw)
                                      )

                            if (nu - 1) > 0:
                                if isinstance(nu - 1, float):
                                    jline += ' * pow(' + \
                                        utils.get_array(lang, 'conc', l_sp)
                                    jline += ', {})'.format(nu - 1)
                                else:
                                    # integer, so just use multiplication
                                    for i in range(nu - 1):
                                        jline += ' * ' + \
                                            utils.get_array(lang, 'conc', l_sp)

                            # loop through remaining reactants
                            for sp_reac in rxn.reac:
                                if sp_reac == sp_l.name:
                                    continue

                                nu = rxn.reac_nu[rxn.reac.index(sp_reac)]
                                isp = next(i for i in xrange(len(specs))
                                           if specs[i].name == sp_reac)
                                if isinstance(nu, float):
                                    jline += ' * pow(' + \
                                        utils.get_array(lang, 'conc', isp)
                                    jline += ', ' + str(nu) + ')'
                                else:
                                    # integer, so just use multiplication
                                    for i in range(nu):
                                        jline += ' * ' + \
                                            utils.get_array(lang, 'conc', isp)
                        # end reactant section
                        jline += ')'

                    if rxn.rev and sp_l.name in rxn.prod:
                        # need to calculate reverse rate constant

                        if firstSp:
                            jline += '-'
                        else:
                            jline += ' - '
                        firstSp = False

                        nu = rxn.prod_nu[rxn.prod.index(sp_l.name)]
                        if nu != 1:
                            jline += '{} * '.format(float(nu))
                        jline += '('

                        jline += '-mw_avg * ' + \
                            utils.get_array(lang, 'rev_rates', rev_reacs.index(rxn))
                        jline += ' / {:.8e}'.format(sp_j.mw)

                        # only contribution from 2nd part of sp_l is sp_j
                        if sp_l is sp_j:

                            jline += ' + '
                            if not rxn.rev_par:
                                line = '  Kc = 0.0' + utils.line_end[lang]
                                file.write(line)

                                # sum of stoichiometric coefficients
                                sum_nu = 0

                                # go through product species
                                for sp in rxn.prod:

                                    # check if also in reactants
                                    if sp in rxn.reac:
                                        nu = (rxn.prod_nu[rxn.prod.index(sp)]
                                              -
                                              rxn.reac_nu[rxn.reac.index(sp)]
                                              )
                                    else:
                                        nu = rxn.prod_nu[rxn.prod.index(sp)]

                                    # skip if overall stoich coefficient is
                                    # zero
                                    if (nu == 0):
                                        continue

                                    sum_nu += nu

                                    # get species object
                                    spec = next((spec for spec in specs
                                                 if spec.name == sp), None)
                                    if not spec:
                                        i = reacs.index(rxn)
                                        print('Error: species ' + sp +
                                              ' in reaction {}'.format(i) +
                                              ' not found.\n'
                                              )
                                        sys.exit(2)

                                    # need temperature conditional for
                                    # equilibrium constants
                                    line = ('  if (T <= '
                                            '{})'.format(spec.Trange[1])
                                            )
                                    if lang in ['c', 'cuda']:
                                        line += ' {\n'
                                    elif lang == 'fortran':
                                        line += ' then\n'
                                    elif lang == 'matlab':
                                        line += '\n'
                                    file.write(line)

                                    line = '    Kc'
                                    if lang in ['c', 'cuda']:
                                        if nu < 0:
                                            line += ' -= '
                                        elif nu > 0:
                                            line += ' += '
                                    elif lang in ['fortran', 'matlab']:
                                        if nu < 0:
                                            line += ' = Kc - '
                                        elif nu > 0:
                                            line += ' = Kc + '
                                    line += (
                                        '{:.2f} * '.format(abs(nu)) +
                                        '({:.8e} - '.format(spec.lo[6]) +
                                        '{:.8e} + '.format(spec.lo[0]) +
                                        '{:.8e}'.format(spec.lo[0] -
                                                        1.0) +
                                        ' * logT + T * ('
                                        '{:.8e}'.format(spec.lo[1] /
                                                        2.0) +
                                        ' + T * ('
                                        '{:.8e}'.format(spec.lo[2] /
                                                        6.0) +
                                        ' + T * ('
                                        '{:.8e}'.format(spec.lo[3] /
                                                        12.0) +
                                        ' + '
                                        '{:.8e}'.format(spec.lo[4] /
                                                        20.0) +
                                        ' * T))) - '
                                        '{:.8e} / T)'.format(spec.lo[5]) +
                                        utils.line_end[lang]
                                    )
                                    file.write(line)

                                    if lang in ['c', 'cuda']:
                                        file.write('  } else {\n')
                                    elif lang in ['fortran', 'matlab']:
                                        file.write('  else\n')

                                    line = '    Kc'
                                    if lang in ['c', 'cuda']:
                                        if nu < 0:
                                            line += ' -= '
                                        elif nu > 0:
                                            line += ' += '
                                    elif lang in ['fortran', 'matlab']:
                                        if nu < 0:
                                            line += ' = Kc - '
                                        elif nu > 0:
                                            line += ' = Kc + '
                                    line += (
                                        '{:.2f} * '.format(abs(nu)) +
                                        '({:.8e} - '.format(spec.hi[6]) +
                                        '{:.8e} + '.format(spec.hi[0]) +
                                        '{:.8e}'.format(spec.hi[0] -
                                                        1.0) +
                                        ' * logT + T * ('
                                        '{:.8e}'.format(spec.hi[1] /
                                                        2.0) +
                                        ' + T * ('
                                        '{:.8e}'.format(spec.hi[2] /
                                                        6.0) +
                                        ' + T * ('
                                        '{:.8e}'.format(spec.hi[3] /
                                                        12.0) +
                                        ' + '
                                        '{:.8e}'.format(spec.hi[4] /
                                                        20.0) +
                                        ' * T))) - '
                                        '{:.8e} / T)'.format(spec.hi[5]) +
                                        utils.line_end[lang]
                                    )
                                    file.write(line)

                                    if lang in ['c', 'cuda']:
                                        file.write('  }\n\n')
                                    elif lang == 'fortran':
                                        file.write('  end if\n\n')
                                    elif lang == 'matlab':
                                        file.write('  end\n\n')

                                # now loop through reactants
                                for sp in rxn.reac:
                                    # check if species also in products
                                    # (if so, already considered)
                                    if sp in rxn.prod:
                                        continue

                                    nu = -rxn.reac_nu[rxn.reac.index(sp)]

                                    # skip if overall stoich coefficient
                                    # is zero
                                    if (nu == 0):
                                        continue

                                    sum_nu += nu

                                    # get species object
                                    spec = next((spec for spec in specs
                                                 if spec.name == sp), None)
                                    if not spec:
                                        i = reacs.index(rxn)
                                        print('Error: species ' + sp +
                                              ' in reaction {}'.format(i) +
                                              ' not found.\n')
                                        sys.exit(2)

                                    # Write temperature conditional for
                                    # equilibrium constants
                                    line = ('  if (T <= '
                                            '{})'.format(spec.Trange[1])
                                            )
                                    if lang in ['c', 'cuda']:
                                        line += ' {\n'
                                    elif lang == 'fortran':
                                        line += ' then\n'
                                    elif lang == 'matlab':
                                        line += '\n'
                                    file.write(line)

                                    line = '    Kc'
                                    if lang in ['c', 'cuda']:
                                        if nu < 0:
                                            line += ' -= '
                                        elif nu > 0:
                                            line += ' += '
                                    elif lang in ['fortran', 'matlab']:
                                        if nu < 0:
                                            line += ' = Kc - '
                                        elif nu > 0:
                                            line += ' = Kc + '
                                    line += (
                                        '{:.2f} * '.format(abs(nu)) +
                                        '({:.8e} - '.format(spec.lo[6]) +
                                        '{:.8e} + '.format(spec.lo[0]) +
                                        '{:.8e}'.format(spec.lo[0] -
                                                        1.0) +
                                        ' * logT + T * ('
                                        '{:.8e}'.format(spec.lo[1] /
                                                        2.0) +
                                        ' + T * ('
                                        '{:.8e}'.format(spec.lo[2] /
                                                        6.0) +
                                        ' + T * ('
                                        '{:.8e}'.format(spec.lo[3] /
                                                        12.0) +
                                        ' + '
                                        '{:.8e}'.format(spec.lo[4] /
                                                        20.0) +
                                        ' * T))) - '
                                        '{:.8e} / T)'.format(spec.lo[5]) +
                                        utils.line_end[lang]
                                    )
                                    file.write(line)

                                    if lang in ['c', 'cuda']:
                                        file.write('  } else {\n')
                                    elif lang in ['fortran', 'matlab']:
                                        file.write('  else\n')

                                    line = '    Kc'
                                    if lang in ['c', 'cuda']:
                                        if nu < 0:
                                            line += ' -= '
                                        elif nu > 0:
                                            line += ' += '
                                    elif lang in ['fortran', 'matlab']:
                                        if nu < 0:
                                            line += ' = Kc - '
                                        elif nu > 0:
                                            line += ' = Kc + '
                                    line += (
                                        '{:.2f} * '.format(abs(nu)) +
                                        '({:.8e} - '.format(spec.hi[6]) +
                                        '{:.8e} + '.format(spec.hi[0]) +
                                        '{:.8e}'.format(spec.hi[0] -
                                                        1.0) +
                                        ' * logT + T * ('
                                        '{:.8e}'.format(spec.hi[1] /
                                                        2.0) +
                                        ' + T * ('
                                        '{:.8e}'.format(spec.hi[2] /
                                                        6.0) +
                                        ' + T * ('
                                        '{:.8e}'.format(spec.hi[3] /
                                                        12.0) +
                                        ' + '
                                        '{:.8e}'.format(spec.hi[4] /
                                                        20.0) +
                                        ' * T))) - '
                                        '{:.8e} / T)'.format(spec.hi[5]) +
                                        utils.line_end[lang]
                                    )
                                    file.write(line)

                                    if lang in ['c', 'cuda']:
                                        file.write('  }\n\n')
                                    elif lang == 'fortran':
                                        file.write('  end if\n\n')
                                    elif lang == 'matlab':
                                        file.write('  end\n\n')

                                line = '  Kc = '
                                if sum_nu != 0:
                                    num = (chem.PA / chem.RU) ** sum_nu
                                    line += '{:.8e} * '.format(num)
                                line += 'exp(Kc)' + utils.line_end[lang]
                                file.write(line)

                                jline += ('(' +
                                          rate.rxn_rate_const(rxn.A,
                                                              rxn.b,
                                                              rxn.E
                                                              ) +
                                          ' / Kc)'
                                          )

                            else:
                                # explicit reverse coefficients
                                jline += rate.rxn_rate_const(rxn.rev_par[0],
                                                             rxn.rev_par[1],
                                                             rxn.rev_par[2]
                                                             )

                            jline += ' * (rho / {:.8e})'.format(sp_l.mw)

                            nu = rxn.prod_nu[rxn.prod.index(sp_l.name)]
                            if (nu - 1) > 0:
                                if isinstance(nu - 1, float):
                                    jline += ' * pow(conc' + \
                                        utils.get_array(lang, l_sp)
                                    jline += ', {})'.format(nu - 1)
                                else:
                                    # integer, so just use multiplication
                                    for i in range(nu - 1):
                                        jline += ' * ' + \
                                            utils.get_array(lang, 'conc', l_sp)

                            # loop through remaining products
                            for sp_reac in rxn.prod:
                                if sp_reac == sp_l.name:
                                    continue

                                nu = rxn.prod_nu[rxn.prod.index(sp_reac)]
                                isp = next(i for i in xrange(len(specs))
                                           if specs[i].name == sp_reac)
                                if isinstance(nu, float):
                                    jline += ' * pow(conc' + \
                                        utils.get_array(lang, isp)
                                    jline += ', {})'.format(nu)
                                else:
                                    # integer, so just use multiplication
                                    for i in range(nu):
                                        jline += ' * ' + \
                                            utils.get_array(lang, 'conc', isp)
                        # end product section
                        jline += ')'
                # done with species loop

                if rxn.pdep or rxn.thd:
                    jline += ')'

                # done with this reaction
                jline += ')' + utils.line_end[lang]
                file.write(jline)

            if isfirst:
                # not participating in any reactions, or at least no net
                # production
                line = '  '
                if lang in ['c', 'cuda']:
                    line += utils.get_array(lang, 'jac',
                                            k_sp + 1 + (num_s + 1) * (j_sp + 1))
                    sparse_indicies.append(k_sp + 1 + (num_s + 1) * (j_sp + 1))
                elif lang in ['fortran', 'matlab']:
                    line += utils.get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1)
                    sparse_indicies.append((k_sp + 1, j_sp + 1))
                line += ' = 0.0'
            else:
                line = '  '
                if lang in ['c', 'cuda']:
                    i = k_sp + 1 + (num_s + 1) * (j_sp + 1)
                    line += utils.get_array(lang, 'jac', i) + ' += '
                    sparse_indicies.append(i)
                elif lang in ['fortran', 'matlab']:
                    line += (utils.get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) +
                             '= ' + utils.get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) + ' + ')
                    sparse_indicies.append((k_sp + 1, j_sp + 1))
                line += '' + utils.get_array(lang, 'dy', k_sp + offset)
                line += (' * mw_avg / {:.8e}'.format(sp_j.mw) +
                         utils.line_end[lang]
                         )
                file.write(line)

                if lang in ['c', 'cuda']:
                    line += utils.get_array(lang, 'jac', k_sp +
                                            1 + (num_s + 1) * (j_sp + 1)) + ' *= '
                    sparse_indicies.append(k_sp + 1 + (num_s + 1) * (j_sp + 1))
                elif lang in ['fortran', 'matlab']:
                    line += (utils.get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) + ' = ' +
                             utils.get_array(lang, 'jac', k_sp + 1, twod=j_sp + 1) + ' * ')
                    sparse_indicies.append((k_sp + 1, j_sp + 1))
                line += '{:.8e} / rho'.format(sp_k.mw)
            line += utils.line_end[lang]
            file.write(line)
            file.write('\n')

    file.write('\n')

    ###################################
    # Partial derivatives of temperature (energy equation)
    ###################################

    # evaluate enthalpy
    if lang in ['c', 'cuda']:
        file.write('  // species enthalpies\n')
        if lang != 'cuda' or not CUDAParams.is_global():
            file.write('  Real h[{}];\n'.format(num_s))
        file.write('  eval_h(T, h);\n')
        file.write(line)
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
        file.write('  // species specific heats\n')
        if lang != 'cuda' or not CUDAParams.is_global():
            file.write('  Real cp[{}];\n'.format(num_s))
        file.write('  eval_cp(T, cp);\n')
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
                   '  Real cp_avg;\n'
                   )
    elif lang == 'cuda':
        file.write('  // average specific heat\n'
                   '  register Real cp_avg;\n'
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
        line += ('(' + utils.get_array(lang, 'y', isp + 1) + ' * ' + utils.get_array(lang, 'cp', isp) + ')')

        isfirst = False
    line += utils.line_end[lang]
    file.write(line)

    # sum of enthalpy * species rate * molecular weight for all species
    if lang == 'c':
        file.write('  // sum of enthalpy * species rate * molecular weight '
                   'for all species\n'
                   '  Real sum_hwW;\n'
                   )
    elif lang == 'cuda':
        file.write('  // sum of enthalpy * species rate * molecular weight '
                   'for all species\n'
                   '  register Real sum_hwW;\n'
                   )
    elif lang == 'fortran':
        file.write('  ! sum of enthalpy * species rate * molecular weight '
                   'for all species\n'
                   )
    elif lang == 'matlab':
        file.write('  % sum of enthalpy * species rate * molecular weight '
                   'for all species\n'
                   )
    file.write('\n')
    line = '  sum_hwW = '

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
        line += ('(' + utils.get_array(lang, 'h', isp) + ' * ' + utils.get_array(lang, 'dy', isp + offset) + ' * '
                 '{1:.6})'.format(isp, sp.mw))

        isfirst = False
    line += utils.line_end[lang]
    file.write(line)

    file.write('\n')

    ######################################
    # Derivatives with respect to temperature
    ######################################

    if lang in ['c', 'cuda']:
        line = '  //'
    elif lang == 'fortran':
        line = '  !'
    elif lang == 'matlab':
        line = '  %'
    line += 'partial of dT wrt T' + utils.line_end[lang]
    file.write(line)

    # set to zero
    line = '  '
    if lang in ['c', 'cuda']:
        line += utils.get_array(lang, 'jac', 0) + ' = 0.0'
        sparse_indicies.append(0)
    elif lang == 'fortran':
        line += utils.get_array(lang, 'jac', 0, twod=0) + ' = 0.0_wp'
        sparse_indicies.append((1, 1))
    elif lang == 'matlab':
        line += utils.get_array(lang, 'jac', 0, twod=0) + ' = 0.0'
        sparse_indicies.append((1, 1))
    line += utils.line_end[lang]
    file.write(line)

    # need dcp/dT
    for sp in specs:
        isp = specs.index(sp)

        line = '  if (T <= {:})'.format(sp.Trange[1])
        if lang in ['c', 'cuda']:
            line += ' {\n'
        elif lang == 'fortran':
            line += ' then\n'
        elif lang == 'matlab':
            line += '\n'
        file.write(line)

        line = '    ' + utils.get_array(lang, 'jac', 0, twod=0)

        if lang in ['c', 'cuda']:
            line += ' += '
            sparse_indicies.append(0)
        elif lang in ['fortran', 'matlab']:
            line += ' = ' + utils.get_array(lang, 'jac', 0, twod=0) + ' + '
            sparse_indicies.append((0, 0))
        line += '' + utils.get_array(lang, 'y', isp + 1)
        line += (' * {:.8e} * ('.format(chem.RU / sp.mw) +
                 '{:.8e} + '.format(sp.lo[1]) +
                 'T * ({:.8e} + '.format(2.0 * sp.lo[2]) +
                 'T * ({:.8e} + '.format(3.0 * sp.lo[3]) +
                 '{:.8e} * T)))'.format(4.0 * sp.lo[4]) +
                 utils.line_end[lang]
                 )
        file.write(line)

        if lang in ['c', 'cuda']:
            file.write('  } else {\n')
        elif lang in ['fortran', 'matlab']:
            file.write('  else\n')

        line = '    ' + utils.get_array(lang, 'jac', 0, twod=0)

        if lang in ['c', 'cuda']:
            line += ' += '
        elif lang in ['fortran', 'matlab']:
            line += ' = ' + utils.get_array(lang, 'jac', 0, twod=0) + ' + '
        line += '' + utils.get_array(lang, 'y', isp + 1)
        line += (' * {:.8e} * ('.format(chem.RU / sp.mw) +
                 '{:.8e} + '.format(sp.hi[1]) +
                 'T * ({:.8e} + '.format(2.0 * sp.hi[2]) +
                 'T * ({:.8e} + '.format(3.0 * sp.hi[3]) +
                 '{:.8e} * T)))'.format(4.0 * sp.hi[4]) +
                 utils.line_end[lang]
                 )
        file.write(line)

        if lang in ['c', 'cuda']:
            file.write('  }\n\n')
        elif lang == 'fortran':
            file.write('  end if\n\n')
        elif lang == 'matlab':
            file.write('  end\n\n')

    line = '  '
    if lang in ['c', 'cuda']:
        line += '' + utils.get_array(lang, 'jac', 0) + ' *= (-1.0'
        sparse_indicies.append(0)
    elif lang in ['fortran', 'matlab']:
        line += '' + utils.get_array(lang, 'jac', 0, twod=0) + ' = (-jac' + \
            utils.get_array(lang, 0, twod=0)
        sparse_indicies.append((0, 0))
    line += ' / (rho * cp_avg)) * ('

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
            line = '        '

        isp = specs.index(sp)
        if not isfirst:
            line += ' + '
        line += ' + ' + \
            utils.get_array(
                lang, 'h', isp) + ' * ' + utils.get_array(lang, 'dy', isp + offset) + ' * {:.8e}'.format(sp.mw)
        isfirst = False
    line += ')' + utils.line_end[lang]
    file.write(line)

    line = '  '
    if lang in ['c', 'cuda']:
        line += '' + utils.get_array(lang, 'jac', 0) + ' += ('
        sparse_indicies.append(0)
    elif lang in ['fortran', 'matlab']:
        line += 'jac' + \
            utils.get_array(
                lang, 0, twod=0) + ' = ' + utils.get_array(lang, 'jac', 0, twod=0) + ' + ('
        sparse_indicies.append((0, 0))

    isfirst = True
    for sp in specs:
        if len(line) > 65:
            if lang in ['c', 'cuda']:
                line += '\n'
            elif lang == 'fortran':
                line += ' &\n'
            elif lang == 'matlab':
                line += ' ...\n'
            file.write(line)
            line = '         '

        isp = specs.index(sp)
        if not isfirst:
            line += ' + '
        line += '(' + utils.get_array(lang, 'cp', isp) + ' * ' + utils.get_array(lang, 'dy', isp + offset) + \
            ' * {:.8e} / rho + '.format(sp.mw) + '' + utils.get_array(lang, 'h', isp) + ' * ' + \
            utils.get_array(lang, 'jac', isp + 1, twod=0) + ')'
        isfirst = False
    line += ')' + utils.line_end[lang]
    file.write(line)

    line = '  ' + utils.get_array(lang, 'jac', 0, twod=0)
    if lang in ['c', 'cuda']:
        line += ' /= '
        sparse_indicies.append(0)
    elif lang in ['fortran', 'matlab']:
        line += '' + utils.get_array(lang, 'jac', 0, twod=0) + ' / '
        sparse_indicies.append((0, 0))
    line += '(-cp_avg)' + utils.line_end[lang]
    file.write(line)

    file.write('\n')

    ######################################
    # Derivative with respect to species
    ######################################
    for sp in specs:
        isp = specs.index(sp)

        if lang in ['c', 'cuda']:
            line = '  //'
        elif lang == 'fortran':
            line = '  !'
        elif lang == 'matlab':
            line = '  %'
        line += ('partial of dT wrt Y_' + sp.name +
                 utils.line_end[lang])
        file.write(line)

        line = '  '
        if lang in ['c', 'cuda']:
            line += utils.get_array(lang, 'jac', (num_s + 1) * (isp + 1))
            sparse_indicies.append((num_s + 1) * (isp + 1))
        elif lang in ['fortran', 'matlab']:
            line += utils.get_array(lang, 1, twod=isp + 1)
            sparse_indicies.append((1, isp + 2))
        line += ' = -('

        isfirst = True
        for sp_k in specs:
            if len(line) > 70:
                if lang in ['c', 'cuda']:
                    line += '\n'
                elif lang == 'fortran':
                    line += ' &\n'
                elif lang == 'matlab':
                    line += ' ...\n'
                file.write(line)
                line = '        '

            k_sp = specs.index(sp_k)
            if not isfirst:
                line += ' + '
            if lang in ['c', 'cuda']:
                line += ('' + utils.get_array(lang, 'h', k_sp) + ' * ('
                         '' + utils.get_array(lang, 'jac', k_sp + 1 + (num_s + 1) * (isp + 1)) +
                         ' - (' + utils.get_array(lang, 'cp', isp) + ' * ' + utils.get_array(lang, 'dy', k_sp + offset) +
                         ' * {:.8e} / (rho * cp_avg)))'.format(sp_k.mw)
                         )
                sparse_indicies.append(k_sp + 1 + (num_s + 1) * (isp + 1))
            elif lang in ['fortran', 'matlab']:
                line += ('' + utils.get_array(lang, 'h', k_sp) + ' * ('
                         '' + utils.get_array(lang, 'jac', k_sp + 1, twod=isp + 1) +
                         ' - (' + utils.get_array(lang, 'cp', isp) + ' * ' + utils.get_array(lang, 'dy', k_sp + offset)
                         )
                sparse_indicies.append((k_sp + 1, isp + 1))
            isfirst = False

        line += ') / cp_avg' + utils.line_end[lang]
        file.write(line)

    if lang in ['c', 'cuda']:
        file.write('} // end eval_jacob\n\n')
    elif lang == 'fortran':
        file.write('end subroutine eval_jacob\n\n')
    elif lang == 'matlab':
        file.write('end\n\n')

    file.close()

    # remove any duplicates
    sparse_indicies = list(set(sparse_indicies))
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
            'void sparse_multiplier (const Real *, const Real *, Real*);\n'
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
            '__device__ void sparse_multiplier (const Real *, const Real *, Real*);\n'
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
        "void sparse_multiplier(const Real * A, const Real * Vm, Real* w) {\n")

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
    build_path = os.path.join(os.getcwd(), 'out/')
    utils.create_dir(build_path)

    # interpret reaction mechanism file
    [elems, specs, reacs, units] = mech.read_mech(mech_name)

    # interpret thermodynamic database file (if it exists & needed)
    therm_flag = True
    if therm_name:
        # check for any species missing molecular weight
        for sp in specs:
            if not sp.mw:
                therm_flag = False
                break
        if not therm_flag:
            # need to read thermo file
            file = open(therm_name, 'r')
            mech.read_thermo(file, elems, specs)
            file.close()

    # convert activation energy units to K (if needed)
    if 'kelvin' not in units:
        efac = 1.0

        if 'kcal/mole' in units:
            efac = 4184.0 / chem.RU_JOUL
        elif 'cal/mole' in units:
            efac = 4.184 / chem.RU_JOUL
        elif 'kjoule' in units:
            efac = 1000.0 / chem.RU_JOUL
        elif 'joules' in units:
            efac = 1.00 / chem.RU_JOUL
        elif 'evolt' in units:
            efac = 11595.
        else:
            # default is cal/mole
            efac = 4.184 / chem.RU_JOUL

        for rxn in reacs:
            rxn.E *= efac

        for rxn in [rxn for rxn in reacs if rxn.low]:
            rxn.low[2] *= efac

        for rxn in [rxn for rxn in reacs if rxn.high]:
            rxn.high[2] *= efac

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
        splittings = None
        old_spec_order = range(len(specs))
        old_rxn_order = range(len(reacs))
    
    if lang == 'cuda':
        CUDAParams.write_launch_bounds(build_path, num_blocks, num_threads, L1_preferred)
    smm = None
    if lang == 'cuda' and not no_shared:
        smm = shared.shared_memory_manager(num_blocks, num_threads, L1_preferred)

    # now begin writing subroutines
    
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

    # write mechanism initializers and testing methods
    aux.write_mechanism_initializers(build_path, lang, specs, reacs, initial_state, old_spec_order, old_rxn_order)

    # write Jacobian subroutine
    sparse_indicies = write_jacobian_alt(build_path, lang, specs, reacs, splittings, smm)

    write_sparse_multiplier(build_path, lang, sparse_indicies, len(specs) + 1)


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
