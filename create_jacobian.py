#! /usr/bin/env python

import math
from chem_utilities import *
from mech_interpret import *
from rate_subs import *

def write_jacobian(lang, specs, reacs):
    """Write Jacobian subroutine.
    
    Input
    lang:   programming language ('c', 'cuda', 'fortran', 'matlab')
    specs:  list of species objects
    reacs:  list of reaction objects
    """
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
    filename = file_lang_app('jacob', lang)
    file = open(filename, 'w')
    
    # header files
    if lang == 'c':
        file.write('#include <stdlib.h>\n')
        file.write('#include <math.h>\n')
        file.write('#include "header.h"\n')
        file.write('#include "chem_utils.h"\n')
        file.write('#include "rates.h"\n')
        file.write('\n')
    elif lang == 'cuda':
        file.write('#include <stdlib.h>\n')
        file.write('#include <math.h>\n')
        file.write('#include "header.h"\n')
        file.write('#include "chem_utils.cuh"\n')
        file.write('#include "rates.cuh"\n')
        file.write('\n')
        
    line = ''
    if lang == 'cuda': line = '__device__ '
    
    if lang in ['c', 'cuda']:
        line += 'void eval_jacob (Real t, Real pres, Real * y, Real * jac) {\n\n'
    elif lang == 'fortran':
        line += 'subroutine eval_jacob (t, pres, y, jac)\n\n'
        
        # fortran needs type declarations
        line += '  implicit none\n'
        line += '  double precision, intent(in) :: t, pres, y(' + str(num_s) + ')\n'
        line += '  double precision, intent(out) :: jac(' + str(num_s) + ',' + str(num_s) + ')\n'
        line += '  \n'
        line += '  double precision :: T, rho, cp_avg, logT\n'
        if any(rxn.thd for rxn in rev_reacs):
            line += '  double precision :: m\n'
        line += '  double precision, dimension(' + str(num_s) + ') :: conc, cp, h, sp_rates\n'
        line += '  double precision, dimension(' + str(num_r) + ') :: rxn_rates\n'
        line += '  double precision, dimension(' + str(num_pdep) + ') :: pres_mod\n'
    elif lang == 'matlab':
        line += 'function jac = eval_jacob (T, pres, y)\n\n'
    file.write(line)
    
    
    # get temperature
    if lang in ['c', 'cuda']:
        line = '  Real T = y[0]'
    elif lang in ['fortran', 'matlab']:
        line = '  T = y(1)'
    line += line_end(lang)
    file.write(line)
    
    file.write('\n')
    
    # calculation of average molecular weight
    if lang in ['c', 'cuda']:
        file.write('  // average molecular weight\n')
        file.write('  Real mw_avg;\n')
        line = '  mw_avg = '
    elif lang == 'fortran':
        file.write('  ! average molecular weight\n')
        line = '  mw_avg = '
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
        
        if not isfirst: line += ' + '
        if lang in ['c', 'cuda']:
            line += '(y[' + str(specs.index(sp) + 1) + '] / {:})'.format(sp.mw)
        elif lang in ['fortran', 'matlab']:
            line += '(y(' + str(specs.index(sp) + 2) + ') / {:})'.format(sp.mw)
        
        isfirst = False
    
    line += line_end(lang)
    file.write(line)
    line = '  mw_avg = 1.0 / mw_avg' + line_end(lang)
    file.write(line)
    
    if lang in ['c', 'cuda']:
        file.write('  // mass-averaged density\n')
        file.write('  Real rho;\n')
    elif lang == 'fortran':
        file.write('  ! mass-averaged density\n')
    elif lang == 'matlab':
        file.write('  % mass-averaged density\n')
    line = '  rho = pres * mw_avg / ({:.8e} * T)'.format(RU)
    line += line_end(lang)
    file.write(line)
    
    file.write('\n')
    
    # evaluate species molar concentrations
    if lang in ['c', 'cuda']:
        file.write('  // species molar concentrations\n')
        file.write('  Real conc[' + str(num_s) + '];\n')
    elif lang == 'fortran':
        file.write('  ! species molar concentrations\n')
    elif lang == 'matlab':
        file.write('  % species molar concentrations\n')
        file.write('  conc = zeros(' + str(num_s) + ',1);\n')
    # loop through species
    for sp in specs:
        isp = specs.index(sp)
        if lang in ['c', 'cuda']:
            line = '  conc[{:}] = rho * y[{:}] / {:}'.format(isp, isp + 1, sp.mw)
        elif lang in ['fortran', 'matlab']:
            line = '  conc({:}) = rho * y({:}) / {:}'.format(isp + 1, isp + 2, sp.mw)
        line += line_end(lang)
        file.write(line)
    file.write('\n')
    
    
    # evaluate forward and reverse reaction rates
    if lang in ['c', 'cuda']:
        file.write('  // evaluate reaction rates\n')
        file.write('  Real fwd_rxn_rates[' + str(num_r) + '];\n')
        if rev_reacs:
            file.write('  Real rev_rxn_rates[' + str(num_rev) + '];\n')
            file.write('  eval_rxn_rates (T, conc, fwd_rxn_rates, rev_rxn_rates);\n')
        else:
            file.write('  eval_rxn_rates (T, conc, fwd_rxn_rates);\n')
    elif lang == 'fortran':
        file.write('  ! evaluate reaction rates\n')
        if rev_reacs:
            file.write('  eval_rxn_rates (T, conc, fwd_rxn_rates, rev_rxn_rates)\n')
        else:
            file.write('  eval_rxn_rates (T, conc, fwd_rxn_rates)\n')
    elif lang == 'matlab':
        file.write('  % evaluate reaction rates\n')
        if rev_reacs:
            file.write('  [fwd_rxn_rates, rev_rxn_rates] = eval_rxn_rates (T, conc);\n')
        else:
            file.write('  fwd_rxn_rates = eval_rxn_rates (T, conc);\n')
    file.write('\n')
    
    
    # evaluate third-body and pressure-dependence reaction modifications
    if lang in ['c', 'cuda']:
        file.write('  // get pressure modifications to reaction rates\n')
        file.write('  Real pres_mod[' + str(num_pdep) + '];\n')
        file.write('  get_rxn_pres_mod (T, pres, conc, pres_mod);\n')
    elif lang == 'fortran':
        file.write('  ! get and evaluate pressure modifications to reaction rates\n')
        file.write('  get_rxn_pres_mod (T, pres, conc, pres_mod)\n')
    elif lang == 'matlab':
        file.write('  % get and evaluate pressure modifications to reaction rates\n')
        file.write('  pres_mod = get_rxn_pres_mod (T, pres, conc, pres_mod);\n')
    file.write('\n')
    
    
    # evaluate species rates
    if lang in ['c', 'cuda']:
        file.write('  // evaluate rate of change of species molar concentration\n')
        file.write('  Real sp_rates[' + str(num_s) + '];\n')
        if rev_reacs:
            file.write('  eval_spec_rates (fwd_rxn_rates, rev_rxn_rates, pres_mod, sp_rates);\n')
        else:
            file.write('  eval_spec_rates (fwd_rxn_rates, pres_mod, sp_rates);\n')
    elif lang == 'fortran':
        file.write('  ! evaluate rate of change of species molar concentration\n')
        if rev_reacs:
            file.write('  eval_spec_rates (fwd_rxn_rates, rev_rxn_rates, pres_mod, sp_rates)\n')
        else:
            file.write('  eval_spec_rates (fwd_rxn_rates, pres_mod, sp_rates)\n')
    elif lang == 'matlab':
        file.write('  % evaluate rate of change of species molar concentration\n')
        if rev_reacs:
            file.write('  sp_rates = eval_spec_rates(fwd_rxn_rates, rev_rxn_rates, pres_mod);\n')
        else:
            file.write('  sp_rates = eval_spec_rates(fwd_rxn_rates, pres_mod);\n')
    file.write('\n')
    
    
    # third-body variable needed for reactions
    if any(rxn.thd for rxn in reacs):
        line = '  '
        if lang == 'c':
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += 'm = pres / ({:4e} * T)'.format(RU)
        line += line_end(lang)
        file.write(line)
    
    # log(T)
    line = '  '
    if lang == 'c':
        line += 'Real '
    elif lang == 'cuda':
        line += 'register Real '
    line += 'logT = log(T)'
    line += line_end(lang)
    file.write(line)
    
    # if any reverse reactions, will need Kc
    if rev_reacs:
        line = '  Real Kc'
        line += line_end(lang)
        file.write(line)
    
    # pressure-dependence variables
    if any(rxn.pdep for rxn in reacs):
        line = '  '
        if lang == 'c':
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += 'Pr'
        line += line_end(lang)
        file.write(line)
    
    if any(rxn.troe for rxn in reacs):
        line = '  Real Fcent, A, B, lnF_AB' + line_end(lang)
        file.write(line)
    
    if any(rxn.sri for rxn in reacs):
        line = '  Real X' + line_end(lang)
        file.write(line)
    
    # variables for equilibrium constant derivatives, if needed
    dBdT_flag = []
    for sp in specs:
        dBdT_flag.append(False)
    
    for rxn in rev_reacs:
        # only reactions with no reverse Arrhenius parameters
        if rxn.rev_par: continue
        
        # all participating species
        for rxn_sp in (rxn.reac + rxn.prod):
            sp_ind = next((specs.index(s) for s in specs if s.name == rxn_sp), None)
            
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
            file.write('  Real ' + dBdT + line_end(lang))
            
            # dB/dT evaluation (with temperature conditional)
            line = '  if (T <= {:})'.format(specs[sp_ind].Trange[1])
            if lang in ['c', 'cuda']:
                line += ' {\n'
            elif lang == 'fortran':
                line += ' then\n'
            elif lang == 'matlab':
                line += '\n'
            file.write(line)
            
            line = '    ' + dBdT + ' = ({:.8e} + {:.8e} / T) / T + {:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T))'.format(specs[sp_ind].lo[0] - 1.0, specs[sp_ind].lo[5], specs[sp_ind].lo[1] / 2.0, specs[sp_ind].lo[2] / 3.0, specs[sp_ind].lo[3] / 4.0, specs[sp_ind].lo[4] / 5.0)
            line += line_end(lang)
            file.write(line)
            
            if lang in ['c', 'cuda']:
                file.write('  } else {\n')
            elif lang in ['fortran', 'matlab']:
                file.write('  else\n')
            
            line = '    ' + dBdT + ' = ({:.8e} + {:.8e} / T) / T + {:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T))'.format(specs[sp_ind].hi[0] - 1.0, specs[sp_ind].hi[5], specs[sp_ind].hi[1] / 2.0, specs[sp_ind].hi[2] / 3.0, specs[sp_ind].hi[3] / 4.0, specs[sp_ind].hi[4] / 5.0)
            line += line_end(lang)
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
    
    ###################################
    # partial derivatives of species
    ###################################
    for sp_k in specs:
        
        ######################################
        # w.r.t. temperature
        ######################################
        k_sp = specs.index(sp_k)
        
        if lang in ['c', 'cuda']:
            line = '  //'
        elif lang == 'fortran':
            line = '  !'
        elif lang == 'matlab':
            line = '  %'
        line += 'partial of dT wrt Y_' + sp_k.name
        line += line_end(lang)
        file.write(line)
        
        isfirst = True
        
        for rxn in reacs:
            rind = reacs.index(rxn)
            
            if sp_k.name in rxn.prod and sp_k.name in rxn.reac:
                nu = rxn.prod_nu[rxn.prod.index(sp_k.name)] - rxn.reac_nu[rxn.reac.index(sp_k.name)]
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
            jline = '  jac'
            if lang in ['c', 'cuda']:
                jline += '[' + str(k_sp + 1) + ']'
            elif lang in ['fortran', 'matlab']:
                jline += '(' + str(k_sp + 2) + ', ' + str(1) + ')'
            
            # first reaction for this species
            if isfirst:
                jline += ' = '
                
                if nu != 1:
                    jline += str(float(nu)) + ' * '
                isfirst = False
            else:
                if lang in ['c', 'cuda']:
                    jline += ' += '
                elif lang in ['fortran', 'matlab']:
                    jline += ' = jac(' + str(k_sp) + ', ' + str(1) + ') + '
                
                if nu != 1:
                    jline += str(float(nu)) + ' * '
            
            jline += '('
            
            if rxn.pdep:
                # print lines for necessary pressure-dependent variables
                line = '  Pr = '
                pind = pdep_reacs.index(rind)
                
                if rxn.pdep_sp:
                    line += 'conc'
                    if lang in ['c', 'cuda']:
                        line += '[' + str(specs.index(rxn.pdep_sp)) + ']'
                    elif lang in ['fortran', 'matlab']:
                        line += '(' + str(specs.index(rxn.pdep_sp) + 1) + ')'
                else:
                    line += '(m'
                    
                    for thd_sp in rxn.thd_body:
                        isp = specs.index(next((s for s in specs if s.name == thd_sp[0]), None))
                        if thd_sp[1] > 1.0:
                            line += ' + ' + str(thd_sp[1] - 1.0) + ' * conc'
                            if lang in ['c', 'cuda']:
                                line += '[' + str(isp) + ']'
                            elif lang in ['fortran', 'matlab']:
                                line += '(' + str(isp + 1) + ')'
                        elif thd_sp[1] < 1.0:
                            line += ' - ' + str(1.0 - thd_sp[1]) + ' * conc'
                            if lang in ['c', 'cuda']:
                                line += '[' + str(isp) + ']'
                            elif lang in ['fortran', 'matlab']:
                                line += '(' + str(isp + 1) + ')'
                    line += ')'
                
                #jline += '(('
                
                jline += 'pres_mod'
                if lang in ['c', 'cuda']:
                    jline += '[' + str(pind) + ']'
                elif lang in ['fortran', 'cuda']:
                    jline += '(' + str(pind) + ')'
                jline += '* (('
                
                if rxn.low:
                    # unimolecular/recombination fall-off
                    beta_0minf = rxn.low[1] - rxn.b
                    E_0minf = rxn.low[2] - rxn.E
                    k0kinf = rxn_rate_const(rxn.low[0] / rxn.A, beta_0minf, E_0minf)
                elif rxn.high:
                    # chem-activated bimolecular rxn
                    beta_0minf = rxn.b - rxn.high[1]
                    E_0minf = rxn.E - rxn.high[2]
                    k0kinf = rxn_rate_const(rxn.A / rxn.high[0], beta_0minf, E_0minf)
                    
                    jline += '-Pr'
                
                # finish writing P_ri
                line += ' * (' + k0kinf + ')'
                line += line_end(lang)
                file.write(line)
                
                # dPr/dT
                jline += '({:.4e} + ({:.4e} / T) - 1.0) / (T * (1.0 + Pr)))'.format(beta_0minf, E_0minf)
                
                # dF/dT
                if rxn.troe:
                    line = '  Fcent = {:.4e} * exp(-T / {:.4e})'.format(1.0 - rxn.troe_par[0], rxn.troe_par[1])
                    line += ' + {:.4e} * exp(T / {:.4e})'.format(rxn.troe_par[0], rxn.troe_par[2])
                    if len(rxn.troe_par) == 4:
                        line += ' + exp(-{:.4e} / T)'.format(rxn.troe_par[3])
                    line += line_end(lang)
                    file.write(line)
                    
                    line = '  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4'
                    line += line_end(lang)
                    file.write(line)
                
                    line = '  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr)'
                    line += line_end(lang)
                    file.write(line)
                    
                    line = '  lnF_AB = ' + str(2.0) + ' * log(Fcent) * A / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))'
                    line += line_end(lang)
                    file.write(line)
                    
                    jline += ' + (((1.0 / (Fcent * (1.0 + A * A / (B * B)))) - lnF_AB * (-{:.4e} * B + {:.4e} * A) / Fcent)'.format(0.67 / math.log(10.0), 1.1762 / math.log(10.0))
                    jline += ' * ({:.4e} * exp(-T / {:.4e}) - {:.4e} * exp(-T / {:.4e})'.format( -(1.0 - rxn.troe_par[0]) / rxn.troe_par[1], rxn.troe_par[1], rxn.troe_par[0] / rxn.troe_par[2], rxn.troe_par[2])
                    if len(rxn.troe_par) == 4:
                        line += ' + ({:.4e} / (T * T)) * exp(-{:.4e} / T)'.format(rxn.troe_par[3], rxn.troe_par[3])
                    jline += '))'
                    
                    #jline += ' - (lnF_AB * ({:.4e} * B + {:.4e} * A) / Pr) * '.format(1.0 / math.log(10.0), 0.14 / math.log(10.0))
                    jline += ' - lnF_AB * ({:.4e} * B + {:.4e} * A) * '.format(1.0 / math.log(10.0), 0.14 / math.log(10.0))
                    jline += '({:.4e} + ({:.4e} / T) - 1.0) / T'.format(beta_0minf, E_0minf)
                    
                elif rxn.sri:
                    line = '  X = 1.0 / (1.0 + log10(Pr) * log10(Pr))' + line_end(lang)
                    file.write(line)
                    
                    jline += ' + X * ((({:.4} / (T * T)) * exp(-{:.4} / T) - {:.4e} * exp(-T / {:.4})) / '.format(rxn.sri[0] * rxn.sri[1], rxn.sri[1], 1.0 / rxn.sri[2], rxn.sri[2])
                    jline += '({:.4} * exp(-{:.4} / T) + exp(-T / {:.4}))'.format(rxn.sri[0], rxn.sri[1], rxn.sri[2])
                    jline += ' - X * {:.6} * log10(Pr) * ({:.4e} + ({:.4e} / T) - 1.0) * log({:4} * exp(-{:.4} / T) + exp(-T / {:4})) / T)'.format(2.0 / math.log(10.0), beta_0minf, E_0minf, rxn.sri[0], rxn.sri[1], rxn.sri[2])
                    
                    if len(rxn.sri) == 5:
                        jline += ' + ({:.4} / T)'.format(rxn.sri[4])
                # lindemann, dF/dT = 0
                
                jline += ') * '
                
                if rxn.rev:
                    # forward and reverse reaction rates
                    jline += '(fwd_rxn_rates'
                    if lang in ['c', 'cuda']:
                        jline += '[' + str(rind) + ']'
                    elif lang in ['fortran', 'cuda']:
                        jline += '(' + str(rind + 1) + ')'
                    
                    jline += '- rev_rxn_rates'
                    if lang in ['c', 'cuda']:
                        jline += '[' + str(rev_reacs.index(rxn)) + ']'
                    elif lang in ['fortran', 'cuda']:
                        jline += '(' + str(rev_reacs.index(rxn) + 1) + ')'
                    jline += ')'
                else:
                    # forward reaction rate only
                    jline += 'fwd_rxn_rates'
                    if lang in ['c', 'cuda']:
                        jline += '[' + str(rind) + ']'
                    elif lang in ['fortran', 'cuda']:
                        jline += '(' + str(rind + 1) + ')'
                
                jline += ' + (pres_mod'
                if lang in ['c', 'cuda']:
                    jline += '[' + str(pind) + ']'
                elif lang in ['fortran', 'cuda']:
                    jline += '(' + str(pind) + ')'
                
            else:
                # not pressure dependent
                
                # third body reaction
                if rxn.thd:
                    pind = pdep_reacs.index(rind)
                    
                    jline += '(-pres_mod'
                    if lang in ['c', 'cuda']:
                        jline += '[' + str(pind) + ']'
                    elif lang in ['fortran', 'cuda']:
                        jline += '(' + str(pind) + ')'
                    jline += ' * '
                    
                    if rxn.rev:
                        # forward and reverse reaction rates
                        jline += '(fwd_rxn_rates'
                        if lang in ['c', 'cuda']:
                            jline += '[' + str(rind) + ']'
                        elif lang in ['fortran', 'cuda']:
                            jline += '(' + str(rind + 1) + ')'
                    
                        jline += ' - rev_rxn_rates'
                        if lang in ['c', 'cuda']:
                            jline += '[' + str(rev_reacs.index(rxn)) + ']'
                        elif lang in ['fortran', 'cuda']:
                            jline += '(' + str(rev_reacs.index(rxn) + 1) + ')'
                        jline += ')'
                    else:
                        # forward reaction rate only
                        jline += 'fwd_rxn_rates'
                        if lang in ['c', 'cuda']:
                            jline += '[' + str(rind) + ']'
                        elif lang in ['fortran', 'cuda']:
                            jline += '(' + str(rind + 1) + ')'
                    
                    jline += ' / T) + (pres_mod'
                    if lang in ['c', 'cuda']:
                        jline += '[' + str(pind) + ']'
                    elif lang in ['fortran', 'cuda']:
                        jline += '(' + str(pind) + ')'
                    
                else:
                    if lang in ['c', 'cuda', 'matlab']:
                        jline += '(1.0'
                    elif lang in ['fortran']:
                        jline += '(1.0d0'
            
            jline += ' / T) * ('
            
            # contribution from temperature derivative of forward reaction rate
            jline += 'fwd_rxn_rates'
            if lang in ['c', 'cuda']:
                jline += '[' + str(rind) + ']'
            elif lang in ['fortran', 'matlab']:
                jline += '(' + str(rind + 1) + ')'
            
            jline += ' * ('
            if (abs(rxn.b) > 1.0e-90) and (rxn.E > 1.0e-90):
                jline += '{:.4e} + ({:.4e} / T)'.format(rxn.b, rxn.E)
            elif abs(rxn.b) > 1.0e-90:
                jline += '{:.4e}'.format(rxn.b)
            elif abs(rxn.E) > 1.0e-90:
                jline += '({:.4e} / T)'.format(rxn.E)
            jline += ' + 1.0 - ('
            
            # loop over reactants
            notfirst = False
            for sp in rxn.reac:
                sp_ind = next((specs.index(s) for s in specs if s.name == sp), None)
                nu = rxn.reac_nu[rxn.reac.index(sp)]
                
                if notfirst:
                    jline += ' + '
                jline += str(float(nu))
                
                notfirst = True
            jline += '))'
            
            # contribution from temperature derivative of reaction rates
            if rxn.rev:
                # reversible reaction
                
                jline += ' - rev_rxn_rates'
                if lang in ['c', 'cuda']:
                    jline += '[' + str(rev_reacs.index(rxn)) + ']'
                elif lang in ['fortran', 'matlab']:
                    jline += '(' + str(rev_reacs.index(rxn) + 1) + ')'
                
                if rxn.rev_par:
                    # explicit reverse parameters
                                        
                    jline += ' * ('
                    if abs(rxn.rev_par[1]) > 1.0e-90 and rxn.rev_par[2] > 1.0e-90:
                        jline += '{:.4e} + ({:.4e} / T)'.format(rxn.rev_par[1], rxn.rev_par[2])
                    elif abs(rxn.rev_par[1]) > 1.0e-90:
                        jline += '{:.4e}'.format(rxn.rev_par[1])
                    elif abs(rxn.rev_par[2]) > 1.0e-90:
                        jline += '({:.4e} / T)'.format(rxn.rev_par[2])
                    jline += ' + 1.0 - ('
                    
                    notfirst = False
                    # loop over products
                    for sp in rxn.prod:
                        sp_ind = next((specs.index(s) for s in specs if s.name == sp), None)
                        nu = rxn.prod_nu[rxn.prod.index(sp)]
                        
                        if notfirst:
                            jline += ' + '
                        jline += str(float(nu))
                                                
                        notfirst = True
                    jline += '))'
                    
                else:
                    # reverse rate constant from forward and equilibrium
                    
                    jline += ' * ('
                    if abs(rxn.b) > 1.0e-90 and rxn.E > 1.0e-90:
                        jline += '{:.4e} + ({:.4e} / T)'.format(rxn.b, rxn.E)
                    elif abs(rxn.b) > 1.0e-90:
                        jline += '{:.4e}'.format(rxn.b)
                    elif abs(rxn.E) > 1.0e-90:
                        jline += '({:.4e} / T)'.format(rxn.E)
                    jline += ' + 1.0 - ('
                    
                    notfirst = False
                    # loop over products
                    for sp in rxn.prod:
                        sp_ind = next((specs.index(s) for s in specs if s.name == sp), None)
                        nu = rxn.prod_nu[rxn.prod.index(sp)]
                        
                        if notfirst:
                            jline += ' + '
                        jline += str(float(nu))
                        
                        notfirst = True
                    jline += ') - T * ('
                    
                    notfirst = False
                    # contribution from dBdT terms from all participating species
                    for sp in (rxn.prod + rxn.reac):
                        sp_ind = next((specs.index(s) for s in specs if s.name == sp), None)
                        
                        # get stoichiometric coefficient
                        # if in both reactants and products
                        if sp in rxn.prod and sp in rxn.reac:
                            isp = rxn.reac.index(sp)
                            isp2 = rxn.prod.index(sp)
                            nu = rxn.prod_nu[isp2] - rxn.reac_nu[isp]
                        elif sp in rxn.prod:
                            isp = rxn.prod.index(sp)
                            nu = rxn.prod_nu[isp]
                        else:
                            isp = rxn.reac.index(sp)
                            nu = -rxn.reac_nu[isp]
                        
                        dBdT = 'dBdT_'
                        if lang in ['c', 'cuda']:
                            dBdT += str(sp_ind)
                        elif lang in ['fortran', 'matlab']:
                            dBdT += str(sp_ind + 1)
                        
                        if notfirst:
                            jline += ' + '
                        if nu == 1:
                            jline += dBdT
                        elif nu == -1:
                            jline += '-' + dBdT
                        else:
                            jline += '(' + str(float(nu)) + ' * ' + dBdT + ')'
                        notfirst = True
                    
                    jline += '))'
                   
            #else:
                # irreversible reaction
                #jline += ')'
            
            jline += '))'
            
            # print line for reaction
            jline += line_end(lang)
            file.write(jline)
                
            
        if isfirst:
            # not participating in any reactions, or at least no net production
            line = '  jac'
            if lang in ['c', 'cuda']:
                line += '[' + str(k_sp + 1) + ']'
            elif lang in ['fortran', 'matlab']:
                line += '(' + str(k_sp) + ', ' + str(1) + ')'
            line += ' = 0.0'
        else:
            line = '  jac'
            if lang in ['c', 'cuda']:
                line += '[' + str(k_sp + 1) + ']'
                line += ' *= {:.8e} / rho'.format(sp_k.mw)
            elif lang in ['fortran', 'matlab']:
                line += '(' + str(k_sp) + ', ' + str(1) + ')'
                line += ' = jac(' + str(k_sp) + ', ' + str(1) + ') * {:.8e} / rho'.format(sp_k.mw)
        
        line += line_end(lang)
        file.write(line)
        file.write('\n')
        
        ###############################
        # w.r.t. species mass fractions
        ###############################
        for sp_j in specs:
            j_sp = specs.index(sp_j)
            
            if lang in ['c', 'cuda']:
                line = '  //'
            elif lang == 'fortran':
                line = '  !'
            elif lang == 'matlab':
                line = '  %'
            line += 'partial of omega_' + sp_k.name + ' wrt Y_' + sp_j.name
            line += line_end(lang)
            file.write(line)
            
            isfirst = True
            
            for rxn in reacs:
                rind = reacs.index(rxn)
                
                if sp_k.name in rxn.prod and sp_k.name in rxn.reac:
                    nu = rxn.prod_nu[rxn.prod.index(sp_k.name)] - rxn.reac_nu[rxn.reac.index(sp_k.name)]
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
                jline = '  jac'
                if lang in ['c', 'cuda']:
                    jline += '[' + str(k_sp + 1 + (num_s + 1) * (j_sp + 1)) + ']'
                elif lang in ['fortran', 'matlab']:
                    jline += '(' + str(k_sp + 2) + ', ' + str(j_sp + 2) + ')'
                
                if isfirst:
                    jline += ' = '
                    isfirst = False
                else:
                    if lang in ['c', 'cuda']:
                        jline += ' += '
                    elif lang in ['fortran', 'matlab']:
                        jline += 'jac(' + str(k_sp + 2) + ', ' + str(j_sp + 2) + ') + '
                
                if nu != 1:
                    jline += str(float(nu)) + ' * '
                
                # start reaction
                jline += '('
                
                if rxn.thd and not rxn.pdep:
                    # third-body reaction
                    pind = pdep_reacs.index(rind)
                    
                    jline += '(-mw_avg * pres_mod'
                    if lang in ['c', 'cuda']:
                        jline += '[' + str(pind) + ']'
                    elif lang in ['fortran', 'cuda']:
                        jline += '(' + str(pind + 1) + ')'
                    jline += ' / {:.8e} + '.format(sp_j.mw)
                    
                    # check if species of interest is third body in reaction
                    alphaij = next((thd[1] for thd in rxn.thd_body if thd[0] == sp_j.name), None)
                    if alphaij:
                        jline += str(float(alphaij)) + ' * '
                    # default is 1.0
                    jline += 'rho / {:.8e}'.format(sp_j.mw)
                    jline += ') * '
                    
                    if rxn.rev:
                        jline += '(fwd_rxn_rates'
                        if lang in ['c', 'cuda']:
                            jline += '[' + str(rind) + ']'
                        elif lang in ['fortran', 'cuda']:
                            jline += '(' + str(rind + 1) + ')'
                        
                        jline += ' - rev_rxn_rates'
                        if lang in ['c', 'cuda']:
                            jline += '[' + str(rev_reacs.index(rxn)) + ']'
                        elif lang in ['fortran', 'cuda']:
                            jline += '(' + str(rev_reacs.index(rxn) + 1) + ')'
                        jline += ')'
                    else:
                        jline += 'fwd_rxn_rates'
                        if lang in ['c', 'cuda']:
                            jline += '[' + str(rind) + ']'
                        elif lang in ['fortran', 'cuda']:
                            jline += '(' + str(rind + 1) + ')'
                    
                    #if (sp_j.name in rxn.reac) or (rxn.rev and sp_j.name in rxn.prod):
                    jline += ' + '
                elif rxn.pdep:
                    # pressure-dependent reaction
                    
                    line = '  Pr = '
                    pind = pdep_reacs.index(rind)
                    
                    if rxn.pdep_sp:
                        line += 'conc'
                        if lang in ['c', 'cuda']:
                            line += '[' + str(specs.index(rxn.pdep_sp)) + ']'
                        elif lang in ['fortran', 'matlab']:
                            line += '(' + str(specs.index(rxn.pdep_sp) + 1) + ')'
                    else:
                        line += '(m'
                        for thd_sp in rxn.thd_body:
                            isp = specs.index( next((s for s in specs if s.name == thd_sp[0]), None) )
                            if thd_sp[1] > 1.0:
                                line += ' + ' + str(thd_sp[1] - 1.0) + ' * conc'
                                if lang in ['c', 'cuda']:
                                    line += '[' + str(isp) + ']'
                                elif lang in ['fortran', 'matlab']:
                                    line += '(' + str(isp + 1) + ')'
                            elif thd_sp[1] < 1.0:
                                line += ' - ' + str(1.0 - thd_sp[1]) + ' * conc'
                                if lang in ['c', 'cuda']:
                                    line += '[' + str(isp) + ']'
                                elif lang in ['fortran', 'matlab']:
                                    line += '(' + str(isp + 1) + ')'
                        line += ')'
                    
                    if rxn.low:
                        # unimolecular/recombination fall-off
                        beta_0minf = rxn.low[1] - rxn.b
                        E_0minf = rxn.low[2] - rxn.E
                        k0kinf = rxn_rate_const(rxn.low[0] / rxn.A, beta_0minf, E_0minf)
                    elif rxn.high:
                        # chem-activated bimolecular rxn
                        beta_0minf = rxn.b - rxn.high[1]
                        E_0minf = rxn.E - rxn.high[2]
                        k0kinf = rxn_rate_const(rxn.A / rxn.high[0], beta_0minf, E_0minf)
                    line += ' * (' + k0kinf + ')'
                    line += line_end(lang)
                    file.write(line)
                    
                    jline += 'pres_mod'
                    if lang in ['c', 'cuda']:
                        jline += '[' + str(pind) + ']'
                    elif lang in ['fortran', 'cuda']:
                        jline += '(' + str(pind + 1) + ')'
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
                        line = '  Fcent = {:.4e} * exp(-T / {:.4e})'.format(1.0 - rxn.troe_par[0], rxn.troe_par[1])
                        line += ' + {:.4e} * exp(T / {:.4e})'.format(rxn.troe_par[0], rxn.troe_par[2])
                        if len(rxn.troe_par) == 4:
                            line += ' + exp(-{:.4e} / T)'.format(rxn.troe_par[3])
                        line += line_end(lang)
                        file.write(line)
                        
                        line = '  A = log10(Pr) - 0.67 * log10(Fcent) - 0.4'
                        line += line_end(lang)
                        file.write(line)
                        
                        line = '  B = 0.806 - 1.1762 * log10(Fcent) - 0.14 * log10(Pr)'
                        line += line_end(lang)
                        file.write(line)
                        
                        jline += ' - log(Fcent) * 2.0 * A * (B * {:.6} + A * {:.6}) / '.format(1.0 / math.log(10.0), 0.14 / math.log(10.0))
                        jline += '(B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))'
                        
                    elif rxn.sri:
                        file.write('  X = 1.0 / (1.0 + log10(Pr) * log10(Pr))' + line_end(lang))
                        
                        jline += ' - X * X * {:.6} * log10(Pr) * log({:.4} * exp(-{:4} / T) + exp(-T / {:.4}))'.format(2.0 / math.log(10.0), rxn.sri[0], rxn.sri[1], rxn.sri[2])                                    
                    jline += ') * '
                    
                    # dPr/dYj part
                    #jline += '(-rho * {:.8e} * T / (pres * {:.8e})'.format(RU, sp_j.mw)
                    jline += '(-mw_avg / {:.8e}'.format(sp_j.mw)
                    
                    if rxn.thd_body:
                        jline += ' + rho'
                        alphaij = next((thd[1] for thd in rxn.thd_body if thd[0] == sp_j.name), None)
                        if alphaij:
                            jline += ' * ' + str(float(alphaij))
                        
                        jline += ' / ((m'
                        for thd_sp in rxn.thd_body:
                            isp = specs.index( next((s for s in specs if s.name == thd_sp[0]), None) )
                            if thd_sp[1] > 1.0:
                                jline += ' + ' + str(thd_sp[1] - 1.0) + ' * conc'
                                if lang in ['c', 'cuda']:
                                    jline += '[' + str(isp) + ']'
                                elif lang in ['fortran', 'matlab']:
                                    jline += '(' + str(isp + 1) + ')'
                            elif thd_sp[1] < 1.0:
                                jline += ' - ' + str(1.0 - thd_sp[1]) + ' * conc'
                                if lang in ['c', 'cuda']:
                                    jline += '[' + str(isp) + ']'
                                elif lang in ['fortran', 'matlab']:
                                    jline += '(' + str(isp + 1) + ')'
                        jline += ') * {:.8e})'.format(sp_j.mw)
                    elif rxn.pdep and rxn.pdep_sp == sp_j.name:
                        jline += ' + (1.0 / Y'
                        if lang in ['c', 'cuda']:
                            jline += '[' + str(j_sp) + ']'
                        elif lang in ['fortran', 'cuda']:
                            jline += '(' + str(j_sp + 1) + ')'
                        jline += ')'
                    jline += ') * '
                    
                    if rxn.rev:
                        jline += '(fwd_rxn_rates'
                        if lang in ['c', 'cuda']:
                            jline += '[' + str(rind) + ']'
                        elif lang in ['fortran', 'cuda']:
                            jline += '(' + str(rind + 1) + ')'
                        
                        jline += ' - rev_rxn_rates'
                        if lang in ['c', 'cuda']:
                            jline += '[' + str(rev_reacs.index(rxn)) + ']'
                        elif lang in ['fortran', 'cuda']:
                            jline += '(' + str(rev_reacs.index(rxn) + 1) + ')'
                        jline += ')'
                    else:
                        jline += 'fwd_rxn_rates'
                        if lang in ['c', 'cuda']:
                            jline += '[' + str(rind) + ']'
                        elif lang in ['fortran', 'cuda']:
                            jline += '(' + str(rind + 1) + ')'
                    #jline += ')'
                    
                    #if (sp_j.name in rxn.reac) or (rxn.rev and sp_j.name in rxn.prod):
                    jline += ' + '
                
                # next, contribution from dR/dYj
                if rxn.pdep or rxn.thd:
                    jline += 'pres_mod'
                    pind = pdep_reacs.index(rind)
                    if lang in ['c', 'cuda']:
                        jline += '[' + str(pind) + ']'
                    elif lang in ['fortran', 'cuda']:
                        jline += '(' + str(pind + 1) + ')'
                    jline += ' * ('
                
                firstSp = True
                for sp_l in specs:
                    l_sp = specs.index(sp_l)
                    
                    # contribution from forward
                    if sp_l.name in rxn.reac:
                        
                        if not firstSp: jline += ' + '
                        firstSp = False
                        
                        nu = rxn.reac_nu[rxn.reac.index(sp_l.name)]
                        if nu != 1:
                            jline += str(float(nu)) + ' * '
                        jline += '('
                        
                        jline += '-mw_avg * fwd_rxn_rates'
                        if lang in ['c', 'cuda']:
                            jline += '[' + str(rind) + ']'
                        elif lang in ['fortran', 'cuda']:
                            jline += '(' + str(rind + 1) + ')'
                        jline += ' / {:.8e}'.format(sp_j.mw)
                        
                        # only contribution from 2nd part of sp_l is sp_j
                        if sp_l is sp_j:
                            jline += ' + ' + rxn_rate_const(rxn.A, rxn.b, rxn.E)
                            jline += ' * (rho / {:.8e})'.format(sp_l.mw)
                            
                            if (nu - 1) > 0:
                                if isinstance(nu - 1, float):
                                    jline += ' * pow(conc'
                                    if lang in ['c', 'cuda']:
                                        jline += '[' + str(l_sp) + ']'
                                    elif lang in ['fortran', 'matlab']:
                                        jline += '(' + str(l_sp + 1) + ')'
                                    jline += ', ' + str(nu - 1) + ')'
                                else:
                                    # integer, so just use multiplication
                                    for i in range(nu - 1):
                                        jline += ' * conc'
                                        if lang in ['c', 'cuda']:
                                            jline += '[' + str(l_sp) + ']'
                                        elif lang in ['fortran', 'matlab']:
                                            jline += '(' + str(l_sp) + ')'
                            
                            # loop through remaining reactants
                            for sp_reac in rxn.reac:
                                if sp_reac == sp_l.name: continue
                                
                                nu = rxn.reac_nu[rxn.reac.index(sp_reac)]
                                isp = next(i for i in xrange(len(specs)) if specs[i].name == sp_reac)
                                if isinstance(nu, float):
                                    jline += ' * pow(conc'
                                    if lang in ['c', 'cuda']:
                                        jline += '[' + str(isp) + ']'
                                    elif lang in ['fortran', 'matlab']:
                                        jline += '(' + str(isp + 1) + ')'
                                    jline += ', ' + str(nu) + ')'
                                else:
                                    # integer, so just use multiplication
                                    for i in range(nu):
                                        jline += ' * conc'
                                        if lang in ['c', 'cuda']:
                                            jline += '[' + str(isp) + ']'
                                        elif lang in ['fortran', 'matlab']:
                                            jline += '(' + str(isp) + ')'
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
                            jline += str(float(nu)) + ' * '
                        jline += '('
                        
                        jline += '-mw_avg * rev_rxn_rates'
                        if lang in ['c', 'cuda']:
                            jline += '[' + str(rev_reacs.index(rxn)) + ']'
                        elif lang in ['fortran', 'cuda']:
                            jline += '(' + str(rev_reacs.index(rxn) + 1) + ')'
                        jline += ' / {:.8e}'.format(sp_j.mw)
                        
                        # only contribution from 2nd part of sp_l is sp_j
                        if sp_l is sp_j:
                            
                            jline += ' + '
                            if not rxn.rev_par:
                                line = '  Kc = 0.0'
                                line += line_end(lang)
                                file.write(line)
                                
                                # sum of stoichiometric coefficients
                                sum_nu = 0
                                
                                # go through species in reaction
                                for sp in (rxn.reac + rxn.prod):
                                    if sp in rxn.reac and sp in rxn.prod:
                                        nu = rxn.prod_nu[rxn.prod.index(sp)] - rxn.reac_nu[rxn.reac.index(sp)]
                                    elif sp in rxn.reac:
                                        nu = -rxn.reac_nu[rxn.reac.index(sp)]
                                    elif sp in rxn.prod:
                                        nu = rxn.prod_nu[rxn.prod.index(sp)]
                                    
                                    sum_nu += nu
                                    
                                    spec = next((spec for spec in specs if spec.name == sp), None)
                                    
                                    # need temperature conditional for equilibrium constants
                                    line = '  if (T <= {:})'.format(spec.Trange[1])
                                    if lang in ['c', 'cuda']:
                                        line += ' {\n'
                                    elif lang == 'fortran':
                                        line += ' then\n'
                                    elif lang == 'matlab':
                                        line += '\n'
                                    file.write(line)
                                    
                                    if nu < 0:
                                        line = '    Kc = Kc - {:.2f} * '.format(abs(nu))
                                    elif nu > 0:
                                        line = '    Kc = Kc + {:.2f} * '.format(nu)
                                    line += '({:.8e} - {:.8e} + {:.8e} * logT + T * ({:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T))) - {:.8e} / T)'.format(spec.lo[6], spec.lo[0], spec.lo[0] - 1.0, spec.lo[1]/2.0, spec.lo[2]/6.0, spec.lo[3]/12.0, spec.lo[4]/20.0, spec.lo[5])
                                    line += line_end(lang)
                                    file.write(line)
                                    
                                    if lang in ['c', 'cuda']:
                                        file.write('  } else {\n')
                                    elif lang in ['fortran', 'matlab']:
                                        file.write('  else\n')
                                    
                                    if nu < 0:
                                        line = '    Kc = Kc - {:.2f} * '.format(abs(nu))
                                    elif nu > 0:
                                        line = '    Kc = Kc + {:.2f} * '.format(nu)
                                    line += '({:.8e} - {:.8e} + {:.8e} * logT + T * ({:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T))) - {:.8e} / T)'.format(spec.hi[6], spec.hi[0], spec.hi[0] - 1.0, spec.hi[1]/2.0, spec.hi[2]/6.0, spec.hi[3]/12.0, spec.hi[4]/20.0, spec.hi[5])
                                    line += line_end(lang)
                                    file.write(line)
                                    
                                    if lang in ['c', 'cuda']:
                                        file.write('  }\n\n')
                                    elif lang == 'fortran':
                                        file.write('  end if\n\n')
                                    elif lang == 'matlab':
                                        file.write('  end\n\n')
                                
                                line = '  Kc = '
                                if sum_nu != 0:
                                    line += '{:.8e} * '.format((PA / RU)**sum_nu)
                                line += 'exp(Kc)'
                                line += line_end(lang)
                                file.write(line)
                                
                                jline += '(' + rxn_rate_const(rxn.A, rxn.b, rxn.E) + ' / Kc)'
                            
                            else:
                                # explicit reverse coefficients
                                jline += rxn_rate_const(rxn.rev_par[0], rxn.rev_par[1], rxn.rev_par[2])
                            
                            jline += ' * (rho / {:.8e})'.format(sp_l.mw)
                            
                            if (nu - 1) > 0:
                                if isinstance(nu - 1, float):
                                    jline += ' * pow(conc'
                                    if lang in ['c', 'cuda']:
                                        jline += '[' + str(l_sp) + ']'
                                    elif lang in ['fortran', 'matlab']:
                                        jline += '(' + str(l_sp + 1) + ')'
                                    jline += ', ' + str(nu - 1) + ')'
                                else:
                                    # integer, so just use multiplication
                                    for i in range(nu - 1):
                                        jline += ' * conc'
                                        if lang in ['c', 'cuda']:
                                            jline += '[' + str(l_sp) + ']'
                                        elif lang in ['fortran', 'matlab']:
                                            jline += '(' + str(l_sp) + ')'
                            
                            # loop through remaining products
                            for sp_reac in rxn.prod:
                                if sp_reac == sp_l.name: continue
                                
                                nu = rxn.prod_nu[rxn.prod.index(sp_reac)]
                                isp = next(i for i in xrange(len(specs)) if specs[i].name == sp_reac)
                                if isinstance(nu, float):
                                    jline += ' * pow(conc'
                                    if lang in ['c', 'cuda']:
                                        jline += '[' + str(isp) + ']'
                                    elif lang in ['fortran', 'matlab']:
                                        jline += '(' + str(isp + 1) + ')'
                                    jline += ', ' + str(nu) + ')'
                                else:
                                    # integer, so just use multiplication
                                    for i in range(nu):
                                        jline += ' * conc'
                                        if lang in ['c', 'cuda']:
                                            jline += '[' + str(isp) + ']'
                                        elif lang in ['fortran', 'matlab']:
                                            jline += '(' + str(isp) + ')'
                        # end product section
                        jline += ')'
                # done with species loop
                
                if rxn.pdep or rxn.thd:    
                    jline += ')'
                
                # done with this reaction
                jline += ')'
                jline += line_end(lang)
                file.write(jline)
            
            if isfirst:
                # not participating in any reactions, or at least no net production
                line = '  jac'
                if lang in ['c', 'cuda']:
                    line += '[' + str(k_sp + 1 + (num_s + 1) * (j_sp + 1)) + ']'
                elif lang in ['fortran', 'matlab']:
                    line += '(' + str(k_sp + 2) + ', ' + str(j_sp + 2) + ')'
                line += ' = 0.0'
            else:
                line = '  jac'
                if lang in ['c', 'cuda']:
                    line += '[' + str(k_sp + 1 + (num_s + 1) * (j_sp + 1)) + '] += '
                elif lang in ['fortran', 'matlab']:
                    line += '(' + str(k_sp + 2) + ', ' + str(j_sp + 2) + ') = '
                    line += 'jac(' + str(k_sp + 2) + ', ' + str(j_sp + 2) + ') + '
                line += 'sp_rates'
                if lang in ['c', 'cuda']:
                    line += '[' + str(k_sp) + ']'
                elif lang in ['fortran', 'matlab']:
                    line += '(' + str(k_sp + 1) + ')'
                line += ' * mw_avg / {:.8e}'.format(sp_j.mw)
                line += line_end(lang)
                file.write(line)
                
                line = '  jac'
                if lang in ['c', 'cuda']:
                    line += '[' + str(k_sp + 1 + (num_s + 1) * (j_sp + 1)) + '] *= '
                elif lang in ['fortran', 'matlab']:
                    line += '(' + str(k_sp + 2) + ', ' + str(j_sp + 2) + ') = '
                    line += 'jac(' + str(k_sp + 2) + ', ' + str(j_sp + 2) + ') * '
                line += '{:.8e} / rho'.format(sp_k.mw)
            line += line_end(lang)
            file.write(line)
            file.write('\n')
    
    file.write('\n')
    
    ###################################
    # partial derivatives of temperature (energy equation)
    ###################################
    
    # evaluate enthalpy
    if lang in ['c', 'cuda']:
        file.write('  // species enthalpies\n')
        file.write('  Real h[' + str(num_s) + '];\n')
        file.write('  eval_h(T, h);\n')
    elif lang == 'fortran':
        file.write('  ! species enthalpies\n')
        file.write('  call eval_h(T, h)\n')
    elif lang == 'matlab':
        file.write('  % species enthalpies\n')
        file.write('  h = eval_h(T);\n')
    file.write('\n')
    
    # evaluate specific heat
    if lang in ['c', 'cuda']:
        file.write('  // species specific heats\n')
        file.write('  Real cp[' + str(num_s) + '];\n')
        file.write('  eval_cp(T, cp);\n')
    elif lang == 'fortran':
        file.write('  ! species specific heats\n')
        file.write('  call eval_cp(T, cp)\n')
    elif lang == 'matlab':
        file.write('  % species specific heats\n')
        file.write('  cp = eval_cp(T);\n')
    file.write('\n')
    
    # average specific heat
    if lang == 'c':
        file.write('  // average specific heat\n')
        file.write('  Real cp_avg;\n')
    elif lang == 'cuda':
        file.write('  // average specific heat\n')
        file.write('  register Real cp_avg;\n')
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
        
        if not isfirst: line += ' + '
        if lang in ['c', 'cuda']:
            line += '(y[' + str(specs.index(sp) + 1) + '] * cp[' + str(specs.index(sp)) + '])'
        elif lang in ['fortran', 'matlab']:
            line += '(y(' + str(specs.index(sp) + 2) + ') * cp(' + str(specs.index(sp) + 1) + '))'
        
        isfirst = False
    line += line_end(lang)
    file.write(line)
    
    # sum of enthalpy * species rate * molecular weight for all species
    if lang == 'c':
        file.write('  // sum of enthalpy * species rate * molecular weight for all species\n')
        file.write('  Real sum_hwW;\n')
    elif lang == 'cuda':
        file.write('  // sum of enthalpy * species rate * molecular weight for all species\n')
        file.write('  register Real sum_hwW;\n')
    elif lang == 'fortran':
        file.write('  ! sum of enthalpy * species rate * molecular weight for all species\n')
    elif lang == 'matlab':
        file.write('  % sum of enthalpy * species rate * molecular weight for all species\n')
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
        if not isfirst: line += ' + '
        if lang in ['c', 'cuda']:
            line += '(h[' + str(isp) + '] * sp_rates[' + str(isp) + '] * {:.6})'.format(sp.mw)
        elif lang in ['fortran', 'matlab']:
            line += '(h[' + str(isp + 1) + '] * sp_rates[' + str(isp + 1) + '] * {:.6})'.format(sp.mw)
        
        isfirst = False
    line += line_end(lang)
    file.write(line)
    
    file.write('\n')
    
    ######################################
    # w.r.t. temperature
    ######################################
    
    if lang in ['c', 'cuda']:
        line = '  //'
    elif lang == 'fortran':
        line = '  !'
    elif lang == 'matlab':
        line = '  %'
    line += 'partial of dT wrt T'
    line += line_end(lang)
    file.write(line)
    
    # set to zero
    line = '  jac'
    if lang in ['c', 'cuda']:
        line += '[0] = 0.0'
    elif lang == 'fortran':
        line += '(1, 1) = 0.0d0'
    elif lang == 'matlab':
        line += '(1, 1) = 0.0'
    line += line_end(lang)
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
        
        line = '    jac'
        if lang in ['c', 'cuda']:
            line += '[0]'
        elif lang in ['fortran', 'matlab']:
            line += '(1, 1)'
            
        if lang in ['c', 'cuda']:
            line += ' += '
        elif lang in ['fortran', 'matlab']:
            line += ' = jac(1, 1) +'
        line += 'y'
        if lang in ['c', 'cuda']:
            line += '[' + str(isp + 1) + ']'
        elif lang in ['fortran', 'matlab']:
            line += '(' + str(isp + 2) + ')'
        line += ' * {:.8e} * '.format(RU / sp.mw)
        line += '({:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T)))'.format(sp.lo[1], 2.0 * sp.lo[2], 3.0 * sp.lo[3], 4.0 * sp.lo[4])
        line += line_end(lang)
        file.write(line)
        
        if lang in ['c', 'cuda']:
            file.write('  } else {\n')
        elif lang in ['fortran', 'matlab']:
            file.write('  else\n')
        
        line = '    jac'
        if lang in ['c', 'cuda']:
            line += '[0]'
        elif lang in ['fortran', 'matlab']:
            line += '(1, 1)'
        
        if lang in ['c', 'cuda']:
            line += ' += '
        elif lang in ['fortran', 'matlab']:
            line += ' = jac(1, 1) +'
        line += 'y'
        if lang in ['c', 'cuda']:
            line += '[' + str(isp + 1) + ']'
        elif lang in ['fortran', 'matlab']:
            line += '(' + str(isp + 2) + ')'
        line += ' * {:.8e} * '.format(RU / sp.mw)
        line += '({:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T)))'.format(sp.hi[1], 2.0 * sp.hi[2], 3.0 * sp.hi[3], 4.0 * sp.hi[4])
        line += line_end(lang)
        file.write(line)
        
        if lang in ['c', 'cuda']:
            file.write('  }\n\n')
        elif lang == 'fortran':
            file.write('  end if\n\n')
        elif lang == 'matlab':
            file.write('  end\n\n')
    
    line = '  '
    if lang in ['c', 'cuda']:
        line += 'jac[0] *= (-1.0'
    elif lang in ['fortran', 'matlab']:
        line += 'jac(1, 1) = (-jac(1, 1)'
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
        if not isfirst: line += ' + '
        if lang in ['c', 'cuda']:
            line += 'h[' + str(isp) + '] * sp_rates[' + str(isp) + '] * {:.8e}'.format(sp.mw)
        elif lang in ['fortran', 'matlab']:
            line += 'h(' + str(isp + 1) + ') * sp_rates(' + str(isp + 1) + ') * {:.8e}'.format(sp.mw)
        isfirst = False
    line += ')'
    line += line_end(lang)
    file.write(line)
    
    line = '  '
    if lang in ['c', 'cuda']:
        line += 'jac[0] += ('
    elif lang in ['fortran', 'matlab']:
        line += 'jac(1, 1) = jac(1, 1) + ('

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
        if not isfirst: line += ' + '
        if lang in ['c', 'cuda']:
            line += '(cp[' + str(isp) + '] * sp_rates[' + str(isp) + '] * {:.8e} / rho + '.format(sp.mw)
            line += 'h[' + str(isp) + '] * jac[' + str(isp + 1) + '])'
        elif lang in ['fortran', 'matlab']:
            line += '(cp(' + str(isp + 1) + ') * sp_rates(' + str(isp + 1) + ') * {:.8e} / rho + '.format(sp.mw)
            line += 'h(' + str(isp + 1) + ') * jac(' + str(isp + 2) + ', ' + str(1) + '))'
        isfirst = False
    line += ')'
    line += line_end(lang)
    file.write(line)
    
    line = '  '
    if lang in ['c', 'cuda']:
        line += 'jac[0] /= '
    elif lang in ['fortran', 'matlab']:
        line += 'jac(1, 1) = jac(1,1) / '
    line += '(-cp_avg)'
    line += line_end(lang)
    file.write(line)
    
    file.write('\n')
    
    ######################################
    # w.r.t. species
    ######################################
    for sp in specs:
        isp = specs.index(sp)
        
        if lang in ['c', 'cuda']:
            line = '  //'
        elif lang == 'fortran':
            line = '  !'
        elif lang == 'matlab':
            line = '  %'
        line += 'partial of dT wrt Y_' + sp.name
        line += line_end(lang)
        file.write(line)
        
        line = '  jac'
        if lang in ['c', 'cuda']:
            line += '[' + str((num_s + 1) * (isp + 1)) + ']'
        elif lang in ['fortran', 'matlab']:
            line += '(' + str(1) + ', ' + str(isp + 2) + ')'
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
            if not isfirst: line += ' + '
            if lang in ['c', 'cuda']:
                line += 'h[' + str(k_sp) + '] * (jac[' + str(k_sp + 1 + (num_s + 1) * (isp + 1)) + ']'
                line += ' - (cp[' + str(isp) + '] * sp_rates[' + str(k_sp) + ']'
            elif lang in ['fortran', 'matlab']:
                line += 'h(' + str(k_sp + 1) + ') * (jac(' + str(k_sp + 2) + ', ' + str(isp + 2) + ')'
                line += ' - (cp(' + str(isp + 1) + ') * sp_rates(' + str(k_sp + 1) + ')'
            line += ' * {:.8e} / (rho * cp_avg)))'.format(sp_k.mw)
            isfirst = False
        
        line += ') / cp_avg'
        line += line_end(lang)
        file.write(line)
    
    if lang in ['c', 'cuda']:
        file.write('} // end eval_jacob\n\n')
    elif lang == 'fortran':
        file.write('end subroutine eval_jacob\n\n')
    elif lang == 'matlab':
        file.write('end\n\n')
    
    file.close()
    
    return


def create_jacobian(lang, mech_name, therm_name = None):
    """Create Jacobian subroutine from mechanism.
    
    Input
    lang: language type (e.g., C, CUDA, fortran, matlab)
    mech_name: string with reaction mechanism filename (e.g. 'mech.dat')
    therm_name: string with thermodynamic database filename (e.g. 'therm.dat') or nothing if info in mech_name
    """
    import sys
    
    elems = []
    specs = []
    reacs = []
    
    # supported languages
    langs = ['c', 'cuda', 'fortran', 'matlab']
    
    lang = lang.lower()
    if lang not in langs:
        print 'Error: language needs to be one of: '
        for l in langs:
            print l
        sys.exit()
    
    # interpret reaction mechanism file
    [num_e, num_s, num_r, units] = read_mech(mech_name, elems, specs, reacs)
    
    # interpret thermodynamic database file (if it exists)
    if therm_name:
        file = open(therm_name, 'r')
        read_thermo(file, elems, specs)
        file.close()
    
    # convert activation energy units to K (if needed)
    if 'kelvin' not in units:
        efac = 1.0
        
        if 'kcal/mole' in units:
            efac = 4184.0 / RU_JOUL
        elif 'cal/mole' in units:
            efac = 4.184 / RU_JOUL
        elif 'kjoule' in units:
            efac = 1000.0 / RU_JOUL
        elif 'joules' in units:
            efac = 1.00 / RU_JOUL
        elif 'evolt' in units:
            efac = 11595.
        else:
            # default is cal/mole
            efac = 4.184 / RU_JOUL
        
        for rxn in reacs:
            rxn.E *= efac
        
        for rxn in [rxn for rxn in reacs if rxn.low]:
            rxn.low[2] *= efac
        
        for rxn in [rxn for rxn in reacs if rxn.high]:
            rxn.high[2] *= efac
    
    # now begin writing subroutines
    
    # print reaction rate subroutine
    write_rxn_rates(lang, specs, reacs)
    
    # if third-body/pressure-dependent reactions, print modification subroutine
    if next((r for r in reacs if (r.thd or r.pdep)), None):
        write_rxn_pressure_mod(lang, specs, reacs)
    
    # write species rates subroutine
    write_spec_rates(lang, specs, reacs)
    
    # write chem_utils subroutines
    write_chem_utils(lang, specs)
    
    # write derivative subroutines
    write_derivs(lang, specs, num_r)
    
    # write mass-mole fraction conversion subroutine
    write_mass_mole(specs)
    
    # write Jacobian subroutine
    write_jacobian(lang, specs, reacs)
    
    return


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 3:
        create_jacobian(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        create_jacobian(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print 'Incorrect number of arguments'