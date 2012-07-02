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
    
    line = ''
    if lang == 'cuda': line = '__device__ '
    
    if lang in ['c', 'cuda']:
        line += 'void eval_jacob ( Real t, Real pres, Real * y, Real * jac ) {\n\n'
    elif lang == 'fortran':
        line += 'subroutine eval_jacob ( t, pres, y, jac )\n\n'
        
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
        line += 'function jac = eval_jacob ( T, pres, y )\n\n'
    file.write(line)
    
    
    # get temperature
    if lang in ['c', 'cuda']:
        line = '  Real T = y[0]'
    elif lang in ['fortran', 'matlab']:
        line = '  T = y(1)'
    line += line_end(lang)
    file.write(line)
    
    file.write('\n')
    
    # calculation of density
    if lang in ['c', 'cuda']:
        file.write('  // mass-averaged density\n')
        file.write('  Real rho;\n')
        line = '  rho = '
    elif lang == 'fortran':
        file.write('  ! mass-averaged density\n')
        line = '  rho = '
    elif lang == 'matlab':
        file.write('  % mass-averaged density\n')
        line = '  rho = '
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
            line += '( y[' + str(specs.index(sp) + 1) + '] / {:} )'.format(sp.mw)
        elif lang in ['fortran', 'matlab']:
            line += '( y(' + str(specs.index(sp) + 2) + ') / {:} )'.format(sp.mw)
        
        isfirst = False
    
    line += line_end(lang)
    file.write(line)
    
    line = '  rho = pres / ({:e} * T * rho)'.format(RU)
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
            file.write('  eval_rxn_rates ( T, pres, conc, fwd_rxn_rates, rev_rxn_rates );\n')
        else:
            file.write('  eval_rxn_rates ( T, pres, conc, fwd_rxn_rates );\n')
    elif lang == 'fortran':
        file.write('  ! evaluate reaction rates\n')
        if rev_reacs:
            file.write('  eval_rxn_rates ( T, pres, conc, fwd_rxn_rates, rev_rxn_rates )\n')
        else:
            file.write('  eval_rxn_rates ( T, pres, conc, fwd_rxn_rates )\n')
    elif lang == 'matlab':
        file.write('  % evaluate reaction rates\n')
        if rev_reacs:
            file.write('  [fwd_rxn_rates, rev_rxn_rates] = eval_rxn_rates ( T, pres, conc );\n')
        else:
            file.write('  fwd_rxn_rates = eval_rxn_rates ( T, pres, conc );\n')
    file.write('\n')
    
    
    # evaluate third-body and pressure-dependence reaction modifications
    if lang in ['c', 'cuda']:
        file.write('  // get pressure modifications to reaction rates\n')
        file.write('  Real pres_mod[' + str(num_pdep) + '];\n')
        file.write('  get_rxn_pres_mod ( T, pres, C, pres_mod );\n')
    elif lang == 'fortran':
        file.write('  ! get and evaluate pressure modifications to reaction rates\n')
        file.write('  get_rxn_pres_mod ( T, pres, conc, pres_mod )\n')
    elif lang == 'matlab':
        file.write('  % get and evaluate pressure modifications to reaction rates\n')
        file.write('  pres_mod = get_rxn_pres_mod ( T, pres, C, pres_mod );\n')
    file.write('\n')
    
    
    # evaluate species rates
    if lang in ['c', 'cuda']:
        file.write('  // evaluate rate of change of species molar concentration\n')
        file.write('  Real sp_rates[' + str(num_s) + '];\n')
        if rev_reacs:
            file.write('  eval_spec_rates ( fwd_rxn_rates, rev_rxn_rates, pres_mod, sp_rates );\n')
        else:
            file.write('  eval_spec_rates ( fwd_rxn_rates, pres_mod, sp_rates );\n')
    elif lang == 'fortran':
        file.write('  ! evaluate rate of change of species molar concentration\n')
        if rev_reacs:
            file.write('  eval_spec_rates ( fwd_rxn_rates, rev_rxn_rates, pres_mod, sp_rates )\n')
        else:
            file.write('  eval_spec_rates ( fwd_rxn_rates, pres_mod, sp_rates )\n')
    elif lang == 'matlab':
        file.write('  % evaluate rate of change of species molar concentration\n')
        if rev_reacs:
            file.write('  sp_rates = eval_spec_rates( fwd_rxn_rates, rev_rxn_rates, pres_mod );\n')
        else:
            file.write('  sp_rates = eval_spec_rates( fwd_rxn_rates, pres_mod );\n')
    file.write('\n')
    
    
    # third-body variable needed for reactions
    if any(rxn.thd for rxn in rev_reacs):
        line = '  '
        if lang == 'c':
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += 'm = p / ({:4e} * T)'.format(RU)
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
    
    # pressure-dependence variables
    if any(rxn.pdep for rxn in rev_reacs):
        line = '  '
        if lang == 'c':
            line += 'Real '
        elif lang == 'cuda':
            line += 'register Real '
        line += 'Pr'
        line += line_end(lang)
        file.write(line)
    
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
        
        isfirst = True
        
        for rxn in reacs:
            rind = reacs.index(rxn)
            
            if sp_k.name in rxn.prod and sp_k.name in rxn.reac:
                nu = rxn.prod_nu[rxn.prod.index(sp_k.name)] - rxn.reac_nu[rxn.reac.index(sp_k.name)]
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
            
            if isfirst:
                jline += ' = ' + str(float(nu)) + ' * '
                isfirst = False
            else:
                jline += ' jac'
                if lang in ['c', 'cuda']:
                    jline += '[' + str(k_sp + 1) + ']'
                elif lang in ['fortran', 'matlab']:
                    jline += '(' + str(k_sp) + ', ' + str(1) + ')'
                jline += ' * ' + str(float(nu)) + ' * '
            
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
                    
                    for thd_sp in reac.thd_body:
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
                    #line += ' * ( ' + rxn_rate_const(rxn.low[0], rxn.low[1], rxn.low[2]) + ' )'
                    #line += ' / ( ' + rxn_rate_const(rxn.A, rxn.b, rxn.E) + ' )'
                    k0kinf = rxn_rate_const(rxn.low[0] / rxn.A, rxn.low[1] - rxn.b, rxn.low[2] - rxn.E)
                elif rxn.high:
                    # chem-activated bimolecular rxn
                    #line += ' * ( ' + rxn_rate_const(rxn.A, rxn.b, rxn.E) + ' )'
                    #line += ' / ( ' + rxn_rate_const(rxn.high[0], rxn.high[1], rxn.high[2]) + ' )'
                    k0kinf = rxn_rate_const(rxn.A / rxn.high[0], rxn.b - rxn.high[1], rxn.E - rxn.high[2])
                line += ' * ( ' + k0kinf + ' )'
                line += line_end(lang)
                file.write(line)
                
                if rxn.troe:
                    file.write('  logPr = log10(Pr)' + line_end_lang)
                    
                    line = '  Fcent = {:.4e} * exp(-T / {:.4e})'.format(1.0 - rxn.troe_par[0], rxn.troe_par[1])
                    line += ' + {:.4e} * exp(T / {:.4e})'.format(rxn.troe_par[0], rxn.troe_par[2])
                    if len(rxn.troe_par) == 4:
                        line += ' + exp(-{:.4e} / T)'.format(rxn.troe_par[3])
                    line += line_end(lang)
                    file.write(line)
                    
                    line = '  logFcent = log10(Fcent)'
                    line += line_end(lang)
                    file.write(line)
                    
                    line = '  A = logPr - 0.67 * logFcent - 0.4'
                    line += line_end(lang)
                    file.write(line)
                
                    line = '  B = 0.806 - 1.1762 * logFcent - 0.14 * logPr'
                    line += line_end(lang)
                    file.write(line)
                    
                    line = '  FlnF_AB = {:.8e} * logFcent * A / (B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))'.format(2.0 * log(10.0))
                    line += line_end(lang)
                    file.write(line)
                    
                elif rxn.sri:
                    file.write('  logPr = log10(Pr)' + line_end(lang))
                    
                    file.write('  X = 1.0 / (1.0 + logPr * logPr)' + line_end(lang))
                    
                    line = '  aexp_bT = {:.4} * exp(-{:.4} / T)'.format(rxn.sri[0], rxn.sri[1])
                    line += line_end(lang)
                    file.write(line)
                    
                    line = '  exp_Tc = exp(-T / {:.4})'.format(rxn.sri[2])
                    line += line_end(lang)
                    file.write(line)
                    
                jline += 'pres_mod'
                if lang in ['c', 'cuda']:
                    jline += '[' + str(pind) + ']'
                elif lang in ['fortran', 'cuda']:
                    jline += '(' + str(pind + 1) + ')'
                jline += ' * ( ( '
                if rxn.low:
                    # unimolecular/recombination fall-off
                    beta_0minf = rxn.low[1] - rxn.b
                    E_0minf = rxn.low[2] - rxn.E
                elif rxn.high:
                    # chem-activated bimolecular rxn
                    beta_0minf = rxn.b - rxn.high[1]
                    E_0minf = rxn.E - rxn.high[2]
                    jline += '-'
                jline += '({:.4e} + ({:.4e} / T)) / (T * (1.0 + Pr))'.format(beta_0minf, E_0minf)
                
                if rxn.troe:
                    jline += ' + ( ( (1.0 / (Fcent * (1.0 + A * A / (B * B)))) - FlnF_AB * (-{:.4e} * B + {:.4e} * A) / Fcent )'.format(0.67 / log(10.0), 1.1762 / log(10.0))
                    jline += ' * ( {:.4e} * exp(-T / {:.4e}) - {:.4e} * exp(T / {:.4e})'.format( -(1.0 - rxn.troe_par[0]) / rxn.troe_par[1], rxn.troe_par[1], rxn.troe_par[0] / rxn.troe_par[2], rxn.troe_par[2])
                    if len(reac.troe_par) == 4:
                        line += ' + ({:.4e} / (T * T)) * exp(-{:.4e} / T)'.format(rxn.troe_par[3], rxn.troe_par[3])
                    jline += ' ) )'
                    
                    jline += ' - FlnF_AB * ({:.4e} * B + {:.4e} * A) * '
                    jline += '({:.4e} + ({:.4e} / T)) / T'.format(beta_0minf, E_0minf)
                    
                elif rxn.sri:
                    jline += ' + X * ( (({:.4} / (T * T)) * aexp_bT - {:.4e} * exp_Tc) / (aexp_bT + exp_Tc)'.format(rxn.sri[1], 1.0 / rxn.sri[2])
                    jline += ' - log(aexp_bT + exp_Tc) * {:.6} * X * logPr * ({:.4e} + ({:.4e} / T)) / T )'.format(2.0 / log(10.0), beta_0minf, E_0minf)
                    
                    if len(rxn.sri) == 5:
                        jline += ' + {:.4} / T'
                
                # lindemann, dF/dT = 0
                
                jline += ' ) * '
            else:
                # not pressure dependent
                jline += '('
                
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
            jline += ' + '
            
            # contribution from temperature derivative of reaction rates
            if rxn.rev:
                # reversible reaction
                
                if rxn.rev_par:
                    # explicit reverse parameters
                    jline += '(fwd_rxn_rates'
                    if lang in ['c', 'cuda']:
                        jline += '[' + str(rind) + ']'
                    elif lang in ['fortran', 'cuda']:
                        jline += '(' + str(rind + 1) + ')'

                    jline += ' * ({:.4e} + {:.4e} / T)'.format(rxn.b, rxn.E)
                    
                    jline += ' - rev_rxn_rates'
                    if lang in ['c', 'cuda']:
                        jline += '[' + str(rev_reacs.index(rxn)) + ']'
                    elif lang in ['fortran', 'cuda']:
                        jline += '(' + str(rev_reacs.index(rxn) + 1) + ')'
                    
                    jline += ' * ({:.4e} + {:.4e} / T)'.format(rxn.rev_par[1], rxn.rev_par[2])
                    jline += ') / T'
                    
                else:
                    # reverse rate constant from forward and equilibrium
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

                    jline += ') * ({:.4e} + {:.4e} / T)'.format(rxn.b, rxn.E)
                    jline += ' / T'
                    
                    jline += ' + rev_rxn_rates'
                    if lang in ['c', 'cuda']:
                        jline += '[' + str(rev_reacs.index(rxn)) + ']'
                    elif lang in ['fortran', 'cuda']:
                        jline += '(' + str(rev_reacs.index(rxn) + 1) + ')'
                    jline += ' * ('
                    for rxn_sp in rxn.reac:
                        sp_ind = next((s for s in specs if s.name == rxn_sp), None)
                        
                        if rxn_sp in rxn.prod:
                            nu = rxn.prod_nu[rxn.prod.index(rxn_sp)] - rxn.reac_nu[rxn.reac.index(rxn_sp)]
                        else:
                            nu = -rxn.reac_nu[rxn.reac.index(rxn_sp)]
                        
                        dBdT = 'dBdT_'
                        if lang in ['c', 'cuda']:
                            dBdT += str(sp_ind)
                        elif lang in ['fortran', 'matlab']:
                            dBdT += str(sp_ind + 1)
                        
                        jline += str(float(nu)) + ' * ' + dBdT
                        
                        # print dB/dT evaluation (with temperature conditional)
                        line = '  if (T <= {:})'.format(sp.Trange[1])
                        if lang in ['c', 'cuda']:
                            line += ' {\n'
                        elif lang == 'fortran':
                            line += ' then\n'
                        elif lang == 'matlab':
                            line += '\n'
                        file.write(line)
                        
                        line = '  ' + dBdT + ' = ({:.8e} + {:.8e} / T) / T + {:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T))'.format(specs[sp_ind].lo[0] - 1.0, specs[sp_ind].lo[5], specs[sp_ind].lo[1] / 2.0, specs[sp_ind].lo[2] / 3.0, specs[sp_ind].lo[3] / 4.0, specs[sp_ind].lo[4] / 5.0)
                        line += line_end(lang)
                        file.write(line)
                        
                        if lang in ['c', 'cuda']:
                            file.write('  } else {\n')
                        elif lang in ['fortran', 'matlab']:
                            file.write('  else\n')
                        
                        line = '  ' + dBdT + ' = ({:.8e} + {:.8e} / T) / T + {:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T))'.format(specs[sp_ind].hi[0] - 1.0, specs[sp_ind].hi[5], specs[sp_ind].hi[1] / 2.0, specs[sp_ind].hi[2] / 3.0, specs[sp_ind].hi[3] / 4.0, specs[sp_ind].hi[4] / 5.0)
                        line += line_end(lang)
                        file.write(line)
                        
                        if lang in ['c', 'cuda']:
                            file.write('  }\n\n')
                        elif lang == 'fortran':
                            file.write('  end if\n\n')
                        elif lang == 'matlab':
                            file.write('  end\n\n')
                        
                    for rxn_sp in rxn.prod:
                        sp_ind = next((s for s in specs if s.name == rxn_sp), None)
                        
                        if rxn_sp in rxn.reac:
                            # skip, already done
                            continue
                        else:
                            nu = rxn.prod_nu[rxn.prod.index(rxn_sp)]

                        dBdT = 'dBdT_'
                        if lang in ['c', 'cuda']:
                            dBdT += str(sp_ind)
                        elif lang in ['fortran', 'matlab']:
                            dBdT += str(sp_ind + 1)

                        jline += str(float(nu)) + ' * ' + dBdT

                        # print dB/dT evaluation (with temperature conditional)
                        line = '  if (T <= {:})'.format(sp.Trange[1])
                        if lang in ['c', 'cuda']:
                            line += ' {{\n'
                        elif lang == 'fortran':
                            line += ' then\n'
                        elif lang == 'matlab':
                            line += '\n'
                        file.write(line)

                        line = '  ' + dBdT + ' = ({:.8e} + {:.8e} / T) / T + {:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T))'.format(specs[sp_ind].lo[0] - 1.0, specs[sp_ind].lo[5], specs[sp_ind].lo[1] / 2.0, specs[sp_ind].lo[2] / 3.0, specs[sp_ind].lo[3] / 4.0, specs[sp_ind].lo[4] / 5.0)
                        line += line_end(lang)
                        file.write(line)

                        if lang in ['c', 'cuda']:
                            file.write('  } else {\n')
                        elif lang in ['fortran', 'matlab']:
                            file.write('  else\n')

                        line = '  ' + dBdT + ' = ({:.8e} + {:.8e} / T) / T + {:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T))'.format(specs[sp_ind].hi[0] - 1.0, specs[sp_ind].hi[5], specs[sp_ind].hi[1] / 2.0, specs[sp_ind].hi[2] / 3.0, specs[sp_ind].hi[3] / 4.0, specs[sp_ind].hi[4] / 5.0)
                        line += line_end(lang)
                        file.write(line)

                        if lang in ['c', 'cuda']:
                            file.write('  }\n\n')
                        elif lang == 'fortran':
                            file.write('  end if\n\n')
                        elif lang == 'matlab':
                            file.write('  end\n\n')
                        
                    jline += ')'
            else:
                # irreversible reaction
                jline += 'fwd_rxn_rates'
                if lang in ['c', 'cuda']:
                    jline += '[' + str(rind) + ']'
                elif lang in ['fortran', 'cuda']:
                    jline += '(' + str(rind + 1) + ')'
                
                jline += ' * ({:.4e} + {:.4e} / T) / T'.format(rxn.b, rxn.E)
                        
            jline += ')'
            
            # print line for reaction
            line += line_end(lang)
            file.write(line)
                
            
        if isfirst:
            # not participating in any reactions, or at least no net production
            line = '  jac'
            if lang in ['c', 'cuda']:
                line += '[' + str(k_sp + 1) + ']'
            elif lang in ['fortran', 'matlab']:
                line += '(' + str(k_sp) + ', ' + str(1) + ')'
            line += '0.0'
            line += line_end(lang)
            file.write(line)
        else:
            line = '  jac'
            if lang in ['c', 'cuda']:
                line += '[' + str(k_sp + 1) + ']'
                line += ' *= {:.8e} / rho'.format(sp_k.mw)
            elif lang in ['fortran', 'matlab']:
                line += '(' + str(k_sp) + ', ' + str(1) + ')'
                line += ' = jac(' + str(k_sp) + ', ' + str(1) + ') * {:.8e} / rho'.format(sp_k.mw)
        
        file.write('\n')
        
        ###############################
        # w.r.t. species mass fractions
        ###############################
        for sp_j in specs:
            j_sp = specs.index(sp_j)
            
            isfirst = True
            
            for rxn in reacs:
                rind = reacs.index(rxn)
                
                if sp_k.name in rxn.prod and sp_k.name in rxn.reac:
                    nu = rxn.prod_nu[rxn.prod.index(sp_k.name)] - rxn.reac_nu[rxn.reac.index(sp_k.name)]
                elif sp_k.name in rxn.prod:
                    nu = rxn.prod_nu[rxn.prod.index(sp_k.name)]
                elif sp_k.name in rxn.reac:
                    nu = -rxn.reac_nu[rxn.reac.index(sp_k.name)]
                else:
                    # doesn't participate in reaction
                    continue
                
                if not rxn.thd or not rxn.pdep or sp_j.name not in rxn.prod or sp_j.name not in rxn.reac:
                    # no contribution from this reaction
                    continue
                
                # start contribution to Jacobian entry for reaction
                jline = '  jac'
                if lang in ['c', 'cuda']:
                    jline += '[' + str(k_sp + 1 + (num_s + 1) * (j_sp + 1)) + ']'
                elif lang in ['fortran', 'matlab']:
                    jline += '(' + str(k_sp + 2) + ', ' + str(j_sp + 2) + ')'
                
                if isfirst:
                    jline += ' = ' + str(float(nu)) + ' * '
                    isfirst = False
                else:
                    jline += ' jac'
                    if lang in ['c', 'cuda']:
                        jline += '[' + str(k_sp + 1 + (num_s + 1) * (j_sp + 1)) + ']'
                    elif lang in ['fortran', 'matlab']:
                        jline += '(' + str(k_sp + 2) + ', ' + str(j_sp + 2) + ')'
                    jline += ' * ' + str(float(nu)) + ' * '
                
                if rxn.thd:
                    # third-body reaction
                    
                    # check if species of interest is third body in reaction
                    alphaij = next((thd[1] for thd in rxn.thd_body if thd[0] == sp_j.name), None)
                    if alphaij:
                        jline += '(' + str(float(alphaij)) + ' * '
                    else:
                        # default is 1.0
                        jline += '('
                    
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
                        
                        for thd_sp in reac.thd_body:
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
                        #line += ' * ( ' + rxn_rate_const(rxn.low[0], rxn.low[1], rxn.low[2]) + ' )'
                        #line += ' / ( ' + rxn_rate_const(rxn.A, rxn.b, rxn.E) + ' )'
                        k0kinf = rxn_rate_const(rxn.low[0] / rxn.A, rxn.low[1] - rxn.b, rxn.low[2] - rxn.E)
                    elif rxn.high:
                        # chem-activated bimolecular rxn
                        #line += ' * ( ' + rxn_rate_const(rxn.A, rxn.b, rxn.E) + ' )'
                        #line += ' / ( ' + rxn_rate_const(rxn.high[0], rxn.high[1], rxn.high[2]) + ' )'
                        k0kinf = rxn_rate_const(rxn.A / rxn.high[0], rxn.b - rxn.high[1], rxn.E - rxn.high[2])
                    line += ' * ( ' + k0kinf + ' )'
                    line += line_end(lang)
                    file.write(line)
                    
                    if rxn.troe:
                        file.write('  logPr = log10(Pr)' + line_end_lang)
                        
                        line = '  Fcent = {:.4e} * exp(-T / {:.4e})'.format(1.0 - rxn.troe_par[0], rxn.troe_par[1])
                        line += ' + {:.4e} * exp(T / {:.4e})'.format(rxn.troe_par[0], rxn.troe_par[2])
                        if len(rxn.troe_par) == 4:
                            line += ' + exp(-{:.4e} / T)'.format(rxn.troe_par[3])
                        line += line_end(lang)
                        file.write(line)
                        
                        line = '  logFcent = log10(Fcent)'
                        line += line_end(lang)
                        file.write(line)
                        
                        line = '  A = logPr - 0.67 * logFcent - 0.4'
                        line += line_end(lang)
                        file.write(line)
                        
                        line = '  B = 0.806 - 1.1762 * logFcent - 0.14 * logPr'
                        line += line_end(lang)
                        file.write(line)
                        
                    elif rxn.sri:
                        file.write('  logPr = log10(Pr)' + line_end(lang))
                        
                        file.write('  X = 1.0 / (1.0 + logPr * logPr)' + line_end(lang))
                    
                    jline += 'pres_mod'
                    if lang in ['c', 'cuda']:
                        jline += '[' + str(pind) + ']'
                    elif lang in ['fortran', 'cuda']:
                        jline += '(' + str(pind + 1) + ')'
                    jline += ' * ( '
                    
                    if rxn.thd_body or (rxn.pdep_sp == sp_j.name):
                        jline += k0kinf + ' * '
                        
                        if rxn.thd_body:
                            alphaij = next((thd[1] for thd in rxn.thd_body if thd[0] == sp_j.name), None)
                            if alphaij:
                                jline += str(float(alphaij)) + ' * '
                        
                        jline += '('
                        
                        if rxn.high: jline += '-'
                        jline += '1.0 / (1.0 + Pr)'
                        
                        if rxn.troe:
                            jline += ' - log(Fcent) * 2.0 * A * (B * {:.6} + A * {:.6}) / '
                            if rxn.high:
                                # bimolecular
                                jline += '(Pr'
                            jline += ' * B * B * B * (1.0 + A * A / (B * B)) * (1.0 + A * A / (B * B)))'.format(1.0 / log(10.0), 0.14 / log(10.0))
                            
                        elif rxn.sri:
                            jline += ' - X * X * {:.6} * logPr * log({:.4} * exp(-{:.4} / T) + exp(-T / {:.4}))'.format(2.0 / log(10.0), rxn.sri[0], rxn.sri[1], rxn.sri[2])
                        
                        jline += ') * '
                        
                else:
                    # not pressure dependent
                    jline += '('
                
                if rxn.thd or rxn.pdep:
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
                
                if sp_j.name in rxn.prod or sp_j.name in rxn.reac:
                    jline += ' + ('
                    if sp_j.name in rxn.reac:
                        jline += str(float(rxn.reac_nu[rxn.reac.index(sp_j.name)])) + ' * fwd_rxn_rates'
                        if lang in ['c', 'cuda']:
                            jline += '[' + str(rind) + ']'
                        elif lang in ['fortran', 'cuda']:
                            jline += '(' + str(rind + 1) + ')'
                        
                    if rxn.rev and sp_j.name in rxn.prod:
                        jline += str(float(rxn.prod_nu[rxn.prod.index(sp_j.name)])) + ' * rev_rxn_rates'
                        if lang in ['c', 'cuda']:
                            jline += '[' + str(rev_reacs.index(rxn)) + ']'
                        elif lang in ['fortran', 'cuda']:
                            jline += '(' + str(rev_reacs.index(rxn) + 1) + ')'
                    
                    jline += ') / conc'
                    if lang in ['c', 'cuda']:
                        line += '[' + str(specs.index(sp_j)) + ']'
                    elif lang in ['fortran', 'matlab']:
                        line += '(' + str(specs.index(sp_j) + 1) + ')'
            
            if isfirst:
                # not participating in any reactions, or at least no net production
                line = '  jac'
                if lang in ['c', 'cuda']:
                    line += '[' + str(k_sp + 1 + (num_s + 1) * (j_sp + 1)) + ']'
                elif lang in ['fortran', 'matlab']:
                    line += '(' + str(k_sp + 2) + ', ' + str(j_sp + 2) + ')'
                line += '0.0'
                line += line_end(lang)
                file.write(line)
            else:
                line = '  jac'
                if lang in ['c', 'cuda']:
                    line += '[' + str(k_sp + 1 + (num_s + 1) * (j_sp + 1)) + ']'
                    line += ' *= {:.8e} / rho'.format(sp_k.mw)
                elif lang in ['fortran', 'matlab']:
                    line += '(' + str(k_sp + 2) + ', ' + str(j_sp + 2) + ')'
                    line += ' = jac(' + str(k_sp) + ', ' + str(1) + ') * {:.8e}'.format(sp_k.mw / sp_j.mw)
                    line += line_end(lang)
                    file.write(line)
    
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
    
    line += line_end(lang)
    file.write(line)
    
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
            line += '( y[' + str(specs.index(sp) + 1) + '] * cp[' + str(specs.index(sp)) + '] )'
        elif lang in ['fortran', 'matlab']:
            line += '( y(' + str(specs.index(sp) + 2) + ') * cp(' + str(specs.index(sp) + 1) + ') )'
        
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
            line += '( h[' + str(isp) + '] * sp_rates[' + str(isp) + '] * {:.6} )'.format(sp.mw)
        elif lang in ['fortran', 'matlab']:
            line += '( h[' + str(isp + 1) + '] * sp_rates[' + str(isp + 1) + '] * {:.6} )'.format(sp.mw)
        
        isfirst = False
    line += line_end(lang)
    file.write(line)
    
    file.write('\n')
    
    ######################################
    # w.r.t. temperature
    ######################################
    
    # need dcp/dT
    for sp in specs:
        isp = specs.index(sp)
        
        line = '  if (T <= {:})'.format(sp.Trange[1])
        if lang in ['c', 'cuda']:
            line += ' {{\n'
        elif lang == 'fortran':
            line += ' then\n'
        elif lang == 'matlab':
            line += '\n'
        file.write(line)
        
        line = '  jac'
        if lang in ['c', 'cuda']:
            line += '[0]'
        elif lang in ['fortran', 'matlab']:
            line += '(1, 1)'
            
        line = ' = jac'
        if lang in ['c', 'cuda']:
            line += '[0]'
        elif lang in ['fortran', 'matlab']:
            line += '(1, 1)'
        line += ' + y'
        if lang in ['c', 'cuda']:
            line += '[' + str(isp + 1) + ']'
        elif lang in ['fortran', 'matlab']:
            line += '(' + str(isp + 2) + ')'
        line += ' * {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) )'.format(sp.lo[1], 2.0 * sp.lo[2], 3.0 * sp.lo[3], 4.0 * sp.lo[4])
        line += line_end(lang)
        file.write(line)
        
        if lang in ['c', 'cuda']:
            file.write('  } else {\n')
        elif lang in ['fortran', 'matlab']:
            file.write('  else\n')
        
        line = '  jac'
        if lang in ['c', 'cuda']:
            line += '[0]'
        elif lang in ['fortran', 'matlab']:
            line += '(1, 1)'
        
        line = ' = jac'
        if lang in ['c', 'cuda']:
            line += '[0]'
        elif lang in ['fortran', 'matlab']:
            line += '(1, 1)'
        line += ' + y'
        if lang in ['c', 'cuda']:
            line += '[' + str(isp + 1) + ']'
        elif lang in ['fortran', 'matlab']:
            line += '(' + str(isp + 2) + ')'
        line += ' * {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) )'.format(sp.hi[1], 2.0 * sp.hi[2], 3.0 * sp.hi[3], 4.0 * sp.hi[4])
        line += line_end(lang)
        file.write(line)
        
        if lang in ['c', 'cuda']:
            file.write('  }\n\n')
        elif lang == 'fortran':
            file.write('  end if\n\n')
        elif lang == 'matlab':
            file.write('  end\n\n')
    
    line = '  jac'
    if lang in ['c', 'cuda']:
        line += '[0]'
    elif lang in ['fortran', 'matlab']:
        line += '(1, 1)'
    line += ' = jac'
    if lang in ['c', 'cuda']:
        line += '[0]'
    elif lang in ['fortran', 'matlab']:
        line += '(1, 1)'
    line += ' * sum_hwW - ('
    
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
            line += 'cp[' + str(isp) + '] * sp_rates[' + str(isp) + '] * {:e}'.format(sp.mw)
        elif lang in ['fortran', 'matlab']:
            line += 'cp(' + str(isp + 1) + ') * sp_rates(' + str(isp + 1) + ') * {:e}'.format(sp.mw)
    line += ')'
    line += line_end(lang)
    file.write(line)
    
    line = '  jac'
    if lang in ['c', 'cuda']:
        line += '[0]'
    elif lang in ['fortran', 'matlab']:
        line += '(1, 1)'
    line += ' = (jac'
    if lang in ['c', 'cuda']:
        line += '[0]'
    elif lang in ['fortran', 'matlab']:
        line += '(1, 1)'
    line += ' / rho) - '

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
            line += 'h[' + str(isp) + '] * jac[' + str(isp + 1) + ']'
        elif lang in ['fortran', 'matlab']:
            line += 'h(' + str(isp + 1) + ') * jac(' + str(isp + 2) + ', ' + str(1) + ')'
        isfirst = False
    line += ')'
    line += line_end(lang)
    file.write(line)
    
    line = '  jac'
    if lang in ['c', 'cuda']:
        line += '[0]'
    elif lang in ['fortran', 'matlab']:
        line += '(1, 1)'
    line += ' = jac'
    if lang in ['c', 'cuda']:
        line += '[0]'
    elif lang in ['fortran', 'matlab']:
        line += '(1, 1)'
    line += ' / cp_avg'
    line += line_end(lang)
    file.write(line)
    
    file.write('\n')
    
    ######################################
    # w.r.t. species
    ######################################
    for sp in specs:
        isp = specs.index(sp)
        
        line = '  jac'
        if lang in ['c', 'cuda']:
            line += '[' + str((num_s + 1) * (isp + 1)) + ']'
        elif lang in ['fortran', 'matlab']:
            line += '(' + str(1) + ', ' + str(isp + 2) + ')'
        line += ' = (cp'
        if lang in ['c', 'cuda']:
            line += '[' + str(isp) + ']'
        elif lang in ['fortran', 'matlab']:
            line += '(' + str(isp + 1) + ')'
        line += ' * sum_hwW / (rho * cp_avg)'
        
        line += ' - ('
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
                line += 'h[' + str(isp) + '] * jac[' + str(k_sp + 1 + (num_s + 1) * (isp + 1)) + ']'
            elif lang in ['fortran', 'matlab']:
                line += 'h(' + str(isp + 1) + ') * jac(' + str(k_sp + 2) + ', ' + str(isp + 2) + ')'
            isfirst = False
            
        line += ')'
        
        line += ') / cp_avg'
        line += line_end(lang)
        file.write(line)
    
    
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
            print lan
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
    if next((r for r in reacs if (r.thd or r.pdep), None):
        write_rxn_pressure_mod(lang, specs, reacs)
    
    # write species rates subroutine
    write_spec_rates(lang, specs, reacs)
    
    # write chem_utils subroutines
    write_chem_utils(lang, specs)
    
    # write derivative subroutines
    write_derivs(lang, specs, num_r)
    
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