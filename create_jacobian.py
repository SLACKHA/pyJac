#! /usr/bin/env python

import math
from chem_utilities import *
from mech_interpret import *
from rate_subs import *

def rxn_rate_const(A, b, E):
    """Returns line with reaction rate calculation (after = sign).
    """
    
    # form of the reaction rate constant (from e.g. Lu and Law POCS 2009):
    # kf = A, if b = 0 & E = 0
    # kf = exp(logA + b logT), if b !=  0 & E = 0
    # kf = exp(logA + b logT - E/RT), if b != 0 & E != 0
    # kf = exp(logA - E/RT), if b = 0 & E != 0
    # kf = A T *b* T, if E = 0 & b is integer
    
    line = ''
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
                line += 'exp({:.4e} + '.format(logA) + str(b) + ' * logT)'
    else:
        # E != 0
        if not b:
            # b = 0
            line += 'exp({:.4e}'.format(logA) + ' - ({:.4e} / T) )'.format(E)
        else:
            # b!= 0
            line += 'exp({:.4e} + '.format(logA) + str(b) + ' * logT - ({:.4e} / T) )'.format(E)
    
    return line


def write_rxn_rates(lang, specs, reacs):
    """Write reaction rate subroutine.
    
    Input
    lang: processor type, either CPU or GPU
    specs: list of species objects
    reacs: list of reaction objects
    """
    if lang == 'cpu':
        filename = 'rxn_rates_cpu.c'
        line = ''
    elif lang == 'gpu':
        filename = 'rxn_rates_gpu.cu'
        line = '__device__ '
    
    file = open(filename, 'w')
    
    line += 'void eval_rxn_rates ( Real T, Real p, Real * C, Real * rates ) {\n'
    file.write(line)
    
    thd_flag = False
    
    if lang == 'cpu':
        file.write('  Real logT = log(T);\n')
        file.write('  Real m = p / ({:4e} * T);\n'.format(RU))
        file.write('\n')
        
        if next((r for r in reacs if r.thd == True), None):
            # third body variables
            file.write('  // third body variable declarations\n')
            file.write('  Real thd;\n')
            file.write('\n')
            thd_flag = True
        
        if next((r for r in reacs if r.thd == True), None):
            # pressure dependence variables
            file.write('  // pressure dependence variable declarations\n')
            if not thd_flag: file.write('  Real thd;\n')
            file.write('  Real k0;\n')
            file.write('  Real kinf;\n')
            file.write('  Real F;\n')
            file.write('  Real Pr;\n')
            file.write('\n')
            
            if next((r for r in reacs if r.troe == True), None):
                # troe variables
                file.write('  // troe variable declarations\n')
                file.write('  Real logPr;\n')
                file.write('  Real logFcent;\n')
                file.write('  Real logPrc;\n')
                file.write('  Real n;\n')
                file.write('  Real Fterm;\n')
                file.write('\n')
                
                troe_flag = True
            
            if next((r for r in reacs if r.sri == True), None):
                # sri variables
                file.write('  // sri variable declarations\n')
                if not troe_flag: file.write('  Real logPr;\n')
                file.write('  Real x;\n')
                file.write('\n')
        
    else:
        file.write('  register Real logT = log(T);\n')
        file.write('  register Real m = p / ({:4e} * T);\n'.format(RU))
        file.write('\n')
        
        if next((r for r in reacs if r.thd == True), None):
            # third body variables
            file.write('  // third body variable declarations\n')
            file.write('  register Real thd;\n')
            file.write('\n')
            thd_flag = True
        
        if next((r for r in reacs if r.thd == True), None):
            # pressure dependence variables
            file.write('  // pressure dependence variable declarations\n')
            if not thd_flag: file.write('  register Real thd;\n')
            file.write('  register Real k0;\n')
            file.write('  register Real kinf;\n')
            file.write('  register Real F;\n')
            file.write('  register Real Pr;\n')
            file.write('\n')
            
            if next((r for r in reacs if r.troe == True), None):
                # troe variables
                file.write('  // troe variable declarations\n')
                file.write('  register Real logPr;\n')
                file.write('  register Real logFcent;\n')
                file.write('  register Real logPrc;\n')
                file.write('  register Real n;\n')
                file.write('  register Real Fterm;\n')
                file.write('\n')
                
                troe_flag = True
            
            if next((r for r in reacs if r.sri == True), None):
                # sri variables
                file.write('  // sri variable declarations\n')
                if not troe_flag: file.write('  register Real logPr;\n')
                file.write('  register Real x;\n')
                file.write('\n')
    
    file.write('\n')
    
    for rxn in reacs:
        
        # third bodies
        if rxn.thd:
            line = '  thd = m'
            for sp in rxn.thd_body:
                isp = specs.index( next((s for s in specs if s.name == sp[0]), None) )
                if sp[1] > 1.0:
                    line += ' + ' + str(sp[1] - 1.0) + ' * C[' + str(isp) + ']'
                elif sp[1] < 1.0:
                    line += ' - ' + str(1.0 - sp[1]) + ' * C[' + str(isp) + ']'
            
            line += ';\n'
            file.write(line)
        
        # pressure dependence
        if rxn.pdep:
            if rxn.pdep_sp.lower() == 'm':
                line = '  thd = m'
                for sp in rxn.thd_body:
                    isp = specs.index( next((s for s in specs if s.name == sp[0]), None) )
                    if sp[1] > 1.0:
                        line += ' + ' + str(sp[1] - 1.0) + ' * C[' + str(isp) + ']'
                    elif sp[1] < 1.0:
                        line += ' - ' + str(1.0 - sp[1]) + ' * C[' + str(isp) + ']'
                
            else:
                isp = next(i for i in xrange(len(specs)) if specs[i].name == rxn.pdep_sp)
                line = '  thd = C[' + str(isp) + ']'
            
            line += ';\n'
            file.write(line)
            
            # low-pressure limit rate:
            line = '  k0 = '
            if rxn.low:
                line += rxn_rate_const(rxn.low[0], rxn.low[1], rxn.low[2])
            else:
                line += rxn_rate_const(rxn.A, rxn.b, rxn.E)
            line += ';\n'
            file.write(line)
            
            # high-pressure limit rate:
            line = '  kinf = '
            if rxn.high:
                line += rxn_rate_const(rxn.high[0], rxn.high[1], rxn.high[2])
            else:
                line += rxn_rate_const(rxn.A, rxn.b, rxn.E)
            line += ';\n'
            file.write(line)
            
            # reduced pressure
            file.write('  Pr = k0 * thd / kinf;\n')
            
            if rxn.troe:
                # troe form
                file.write('  logPr = log10(Pr);\n')
                line = '  logFcent = log10( {:4e} * exp(-T / {:4e})'.format(1.0 - rxn.troe_par[0], rxn.troe_par[1])
                line += ' + {:4e} * exp(T / {:4e})'.format(rxn.troe_par[0], rxn.troe_par[2])
                if len(rxn.troe_par) == 4:
                    line += ' + exp(-{:4e} / T)'.format(rxn.troe_par[3])
                
                line += ' );\n'
                file.write(line)
                
                #file.write('  c = -0.4 - 0.67 * logFcent;\n')
                file.write('  logPrc = logPr - (0.4 + 0.67 * logFcent);\n')
                file.write('  n = 0.75 - 1.27 * logFcent;\n')
                #file.write('  Fterm = (logPr + c) / (n - 0.14 * (logPr + c));\n')
                #file.write('  Fterm = logPrc / (n - 0.14 * logPrc);\n')
                file.write('  Fterm = 1.0 / ( (n / logPrc) - 0.14 );\n')
                file.write('  F = exp10( logFcent / (1.0 + Fterm * Fterm) );\n')
                
            elif rxn.sri:
                # sri form
                file.write('  logPr = log10(Pr);\n')
                file.write('  x = 1.0 / (1.0 + logPr * logPr);\n')
                
                line = '  F = pow({:4} * exp(-{:4} / T) + exp(-T / {:4}), x)'.format(rxn.sri[0], rxn.sri[1], rxn.sri[2])
                if len(rxn.sri) == 5:
                    line += ' * {:4e} * pow(T, {:4})'.format(rxn.sri[3], rxn.sri[4])
                line += ';\n'
                file.write(line)
                
            else:
                # lindemann form
                file.write('  F = 1.0;\n')
        
        
        line = '  rates[' + str(reacs.index(rxn)) + '] = '
        
        if rxn.thd:
            line += 'thd * '
        
        # reactants
        for sp in rxn.reac:
            isp = next(i for i in xrange(len(specs)) if specs[i].name == sp)
            nu = rxn.reac_nu[rxn.reac.index(sp)]
            
            # check if stoichiometric coefficient is real or integer
            if isinstance(nu, float):
                line += 'pow(C[' + str(isp) + '], ' + str(nu) + ') * '
            else:
                # integer, so just use multiplication
                for i in range(nu):
                    line += 'C[' + str(isp) + '] * '
        
        # rate constant
        if not rxn.pdep:
            # no pressure dependence, normal rate constant
            line += rxn_rate_const(rxn.A, rxn.b, rxn.E)
        else:
            line += 'kinf * F * Pr / (1.0 + Pr)'
        
        line += ';\n'
        file.write(line)
    
    file.write('} // end eval_rxn_rates\n')
    
    file.close()
    
    return


def write_spec_rates(lang, specs, reacs):
    """
    
    
    """
    
    if lang == 'cpu':
        filename = 'spec_rates_cpu.c'
        line = ''
    elif lang == 'gpu':
        filename = 'spec_rates_gpu.cu'
        line = '__device__ '
    
    file = open(filename, 'w')
    
    line += 'void eval_spec_rates ( Real * rates, Real * sp_rates ) {\n\n'
    file.write(line)
    
    # loop through species
    for sp in specs:
        line = '  sp_rates[' + str(specs.index(sp)) + '] = '
        # continuation line
        cline = ' ' * ( len(line) - 2)
        
        isfirst = True
        
        inreac = False
        
        # loop through reactions
        for rxn in reacs:
            
            # move to new line if current line is too long
            if len(line) > 85:
                line += '\n'
                file.write(line)
                line = cline
            
            # first check to see if in both products and reactants
            if sp.name in rxn.prod and sp.name in rxn.reac:
                inreac = True
                pisp = rxn.prod.index(sp.name)
                risp = rxn.reac.index(sp.name)
                nu = rxn.prod_nu[pisp] - rxn.reac_nu[risp]
                
                if nu > 0.0:
                    if not isfirst: line += ' + '
                    if nu > 1:
                        if isinstance(nu, int):
                            line += str(nu) + '.0 * '
                        else:
                            line += '{:3} * '.format(nu)
                    elif nu < 1.0:
                        line += str(nu) + ' * '
                    
                    line += 'rates[' + str(reacs.index(rxn)) + ']'
                elif nu < 0.0:
                    if isfirst:
                        line += '-'
                    else:
                        line += ' - '
                    
                    if nu < -1:
                        if isinstance(nu, int):
                            line += str(abs(nu)) + '.0 * '
                        else:
                            line += '{:3} * '.format(abs(nu))    
                    elif nu > -1:
                        line += str(abs(nu)) + ' * '
                    
                    line += 'rates[' + str(reacs.index(rxn)) + ']'
                else:
                    inreac = False
                    continue
                
                if isfirst: isfirst = False
                
            # check products
            elif sp.name in rxn.prod:
                inreac = True
                isp = rxn.prod.index(sp.name)
                nu = rxn.prod_nu[isp]
                
                if not isfirst: line += ' + '
                
                if nu > 1:
                    if isinstance(nu, int):
                        line += str(nu) + '.0 * '
                    else:
                        line += '{:3} * '.format(nu)
                elif nu < 1.0:
                    line += str(nu) + ' * '
                
                line += 'rates[' + str(reacs.index(rxn)) + ']'
                
                if isfirst: isfirst = False
                
            # check reactants
            elif sp.name in rxn.reac:
                inreac = True
                isp = rxn.reac.index(sp.name)
                nu = rxn.reac_nu[isp]
                
                if isfirst:
                    line += '-'
                else:
                    line += ' - '
                
                if nu > 1:
                    if isinstance(nu, int):
                        line += str(nu) + '.0 * '
                    else:
                        line += '{:3} * '.format(nu)
                elif nu < 1.0:
                    line += str(nu) + ' * '
                
                line += 'rates[' + str(reacs.index(rxn)) + ']'
                
                if isfirst: isfirst = False
        
        # species not participate in any reactions
        if not inreac: line += '0.0'
        
        # done with this species
        line += ';\n\n'
        file.write(line)
    
    
    file.write('} // end eval_spec_rates\n')
    file.close()
    
    return

def write_chem_utils(lang, specs):
    """
    
    """
    if lang == 'cpu':
        filename = 'chem_utils_cpu.h'
        pre = ''
    elif lang == 'gpu':
        filename = 'chem_utils_gpu.cuh'
        pre = '__device__ '
    
    file = open(filename, 'w')
    
    # enthalpy subroutine
    line = pre + 'void eval_h ( Real T, Real * h ) {\n\n'
    file.write(line)
    
    # loop through species
    for sp in specs:
        line = '  if (T <= {:}) {{\n'.format(sp.Trange[1])
        file.write(line)
        
        line = '    h[' + str(specs.index(sp)) + '] = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) ) );\n'.format(sp.lo[5], sp.lo[0], sp.lo[1] / 2.0, sp.lo[2] / 3.0, sp.lo[3] / 4.0, sp.lo[4] / 5.0)
        file.write(line)
        
        file.write('  } else {\n')
        
        line = '    h[' + str(specs.index(sp)) + '] = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) ) );\n'.format(sp.hi[5], sp.hi[0], sp.hi[1] / 2.0, sp.hi[2] / 3.0, sp.hi[3] / 4.0, sp.hi[4] / 5.0)
        file.write(line)
        
        file.write('  }\n\n')
    
    file.write('} // end eval_h\n\n')
    
    # internal energy subroutine
    line = pre + 'void eval_u ( Real T, Real * u ) {\n\n'
    file.write(line)
    
    # loop through species
    for sp in specs:
        line = '  if (T <= {:}) {{\n'.format(sp.Trange[1])
        file.write(line)
        
        line = '    u[' + str(specs.index(sp)) + '] = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} + T * ( {:.8e} - 1.0 + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) ) );\n'.format(sp.lo[5], sp.lo[0], sp.lo[1] / 2.0, sp.lo[2] / 3.0, sp.lo[3] / 4.0, sp.lo[4] / 5.0)
        file.write(line)
        
        file.write('  } else {\n')
        
        line = '    u[' + str(specs.index(sp)) + '] = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} + T * ( {:.8e} - 1.0 + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) ) );\n'.format(sp.hi[5], sp.hi[0], sp.hi[1] / 2.0, sp.hi[2] / 3.0, sp.hi[3] / 4.0, sp.hi[4] / 5.0)
        file.write(line)
        
        file.write('  }\n\n')
    
    file.write('} // end eval_u\n\n')
    
    # cv subroutine
    line = pre + 'void eval_cv ( Real T, Real * cv ) {\n\n'
    file.write(line)
    
    # loop through species
    for sp in specs:
        line = '  if (T <= {:}) {{\n'.format(sp.Trange[1])
        file.write(line)
        
        line = '    cv[' + str(specs.index(sp)) + '] = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} - 1.0 + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) );\n'.format(sp.lo[0], sp.lo[1], sp.lo[2], sp.lo[3], sp.lo[4])
        file.write(line)
        
        file.write('  } else {\n')
        
        line = '    cv[' + str(specs.index(sp)) + '] = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} - 1.0 + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) );\n'.format(sp.hi[0], sp.hi[1], sp.hi[2], sp.hi[3], sp.hi[4])
        file.write(line)
        
        file.write('  }\n\n')
    
    file.write('} // end eval_cv\n\n')
    
    # cp subroutine 
    line = pre + 'void eval_cp ( Real T, Real * cp ) {\n\n'
    file.write(line)
    
    # loop through species
    for sp in specs:
        line = '  if (T <= {:}) {{\n'.format(sp.Trange[1])
        file.write(line)
        
        line = '    cp[' + str(specs.index(sp)) + '] = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) );\n'.format(sp.lo[0], sp.lo[1], sp.lo[2], sp.lo[3], sp.lo[4])
        file.write(line)
        
        file.write('  } else {\n')
        
        line = '    cp[' + str(specs.index(sp)) + '] = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) );\n'.format(sp.hi[0], sp.hi[1], sp.hi[2], sp.hi[3], sp.hi[4])
        file.write(line)
        
        file.write('  }\n\n')
    
    file.write('} // end eval_cp\n\n')
    
    file.close()
    
    return

def write_derivs(lang, specs, num_r):
    """
    
    
    """
    if lang == 'cpu':
        filename = 'dydt_cpu.c'
        pre = ''
    elif lang == 'gpu':
        filename = 'dydt_gpu.cu'
        pre = '__device__ '
    
    file = open(filename, 'w')
    
    # constant pressure
    file.write('#if defined(CONP)\n\n')
    
    line = pre + 'void dydt ( Real t, Real pres, Real * y, Real * dy ) {\n\n'
    file.write(line)
    
    file.write('  Real T = y[0];\n\n')
    
    # calculation of density
    file.write('  // mass-averaged density\n')
    file.write('  Real rho;\n')
    line = '  rho = '
    isfirst = True
    for sp in specs:
        if len(line) > 70:
            line += '\n'
            file.write(line)
            line = '     '
        
        if not isfirst: line += ' + '
        line += '( y[' + str(specs.index(sp) + 1) + '] / {:} )'.format(sp.mw)
        
        isfirst = False
    
    line += ';\n'
    file.write(line)
    line = '  rho = pres / ({:e} * T * rho);\n\n'.format(RU)
    file.write(line)
    
    # calculation of species molar concentrations
    file.write('  // species molar concentrations\n')
    file.write('  Real conc[{:}];\n'.format(len(specs)) )
    # loop through species
    for sp in specs:
        isp = specs.index(sp)
        line = '  conc[{:}] = rho * y[{:}] / {:};\n'.format(isp, isp + 1, sp.mw)
        file.write(line)
    
    file.write('\n')
    
    # evaluate reaction rates
    file.write('  // local array holding reaction rates\n')
    file.write('  Real rates[{:}];\n'.format(num_r) )
    file.write('  eval_rxn_rates ( T, pres, conc, rates );\n\n')
    
    # species rate of change of molar concentration
    file.write('  // evaluate rate of change of species molar concentration\n')
    file.write('  eval_spec_rates ( rates, &dy[1] );\n\n')
    
    # evaluate specific heat
    file.write('  // local array holding constant pressure specific heat\n')
    file.write('  Real cp[{:}];\n'.format(len(specs)) )
    file.write('  eval_cp ( T, cp );\n\n')
    
    file.write('  // constant pressure mass-average specific heat\n')
    line = '  Real cp_avg = '
    isfirst = True
    for sp in specs:
        if len(line) > 70:
            line += '\n'
            file.write(line)
            line = '             '
        
        if not isfirst: line += ' + '
        
        isp = specs.index(sp)
        line += '( cp[{:}] * y[{:}] )'.format(isp, isp + 1)
        
        isfirst = False
    
    line += ';\n\n'
    file.write(line)
    
    # evaluate enthalpy
    file.write('  // local array for species enthalpies\n')
    file.write('  Real h[{:}];\n'.format(len(specs)) )
    file.write('  eval_h ( T, h );\n\n')
    
    # energy equation
    file.write('  // rate of change of temperature\n')
    line = '  dy[0] = ( -1.0 / ( rho * cp_avg ) ) * ( '
    isfirst = True
    for sp in specs:
        if len(line) > 70:
            line += '\n'
            file.write(line)
            line = '      '
        
        if not isfirst: line += ' + '
        
        isp = specs.index(sp)
        line += '( dy[{:}] * h[{:}] * {:} )'.format(isp + 1, isp, sp.mw)
        
        isfirst = False
    
    line += ' );\n\n'
    file.write(line)
    
    # rate of change of species mass fractions
    file.write('  // calculate rate of change of species mass fractions\n')
    for sp in specs:
        isp = specs.index(sp)
        line = '  dy[{:}] = dy[{:}] * {:} / rho;\n'.format(isp + 1, isp + 1, sp.mw)
        file.write(line)
    
    file.write('\n')
    file.write('} // end dydt\n\n')
    
    
    # constant volume
    file.write('#elif defined(CONV)\n\n')
    
    line = pre + 'void dydt ( Real t, Real rho, Real * y, Real * dy ) {\n\n'
    file.write(line)
    
    file.write('  Real T = y[0];\n\n')
    
    # calculation of pressure
    file.write('  // pressure\n')
    file.write('  Real pres;\n')
    line = '  pres = '
    isfirst = True
    for sp in specs:
        if len(line) > 70:
            line += '\n'
            file.write(line)
            line = '      '
        
        if not isfirst: line += ' + '
        line += '( y[' + str(specs.index(sp) + 1) + '] / {:} )'.format(sp.mw)
        
        isfirst = False
    
    line += ';\n'
    file.write(line)
    line = '  pres = rho * {:e} * T * pres;\n\n'.format(RU)
    file.write(line)
    
    # calculation of species molar concentrations
    file.write('  // species molar concentrations\n')
    file.write('  Real conc[{:}];\n'.format(len(specs)) )
    # loop through species
    for sp in specs:
        isp = specs.index(sp)
        line = '  conc[{:}] = rho * y[{:}] / {:};\n'.format(isp, isp + 1, sp.mw)
        file.write(line)
    
    file.write('\n')
    
    # evaluate reaction rates
    file.write('  // local array holding reaction rates\n')
    file.write('  Real rates[{:}];\n'.format(num_r) )
    file.write('  eval_rxn_rates ( T, pres, conc, rates );\n\n')
    
    # species rate of change of molar concentration
    file.write('  // evaluate rate of change of species molar concentration\n')
    file.write('  eval_spec_rates ( rates, &dy[1] );\n\n')
    
    # evaluate specific heat
    file.write('  // local array holding constant volume specific heat\n')
    file.write('  Real cv[{:}];\n'.format(len(specs)) )
    file.write('  eval_cv ( T, cv );\n\n')
    
    file.write('  // constant volume mass-average specific heat\n')
    line = '  Real cv_avg = '
    isfirst = True
    for sp in specs:
        if len(line) > 70:
            line += '\n'
            file.write(line)
            line = '             '
        
        if not isfirst: line += ' + '
        
        isp = specs.index(sp)
        line += '( cv[{:}] * y[{:}] )'.format(isp, isp + 1)
        
        isfirst = False
    
    line += ';\n\n'
    file.write(line)
    
    # evaluate internal energy
    file.write('  // local array for species internal energies\n')
    file.write('  Real u[{:}];\n'.format(len(specs)) )
    file.write('  eval_u ( T, u );\n\n')
    
    # energy equation
    file.write('  // rate of change of temperature\n')
    line = '  dy[0] = ( -1.0 / ( rho * cv_avg ) ) * ( '
    isfirst = True
    for sp in specs:
        if len(line) > 70:
            line += '\n'
            file.write(line)
            line = '      '
        
        if not isfirst: line += ' + '
        
        isp = specs.index(sp)
        line += '( dy[{:}] * u[{:}] * {:} )'.format(isp + 1, isp, sp.mw)
        
        isfirst = False
    
    line += ' );\n\n'
    file.write(line)
    
    # rate of change of species mass fractions
    file.write('  // calculate rate of change of species mass fractions\n')
    for sp in specs:
        isp = specs.index(sp)
        line = '  dy[{:}] = dy[{:}] * {:} / rho;\n'.format(isp + 1, isp + 1, sp.mw)
        file.write(line)
    
    file.write('\n')
    file.write('} // end dydt\n\n')
    
    file.write('#endif\n')
    
    file.close()
    
    return

def write_int(lang, specs):
    """
    
    """
    if lang == 'cpu':
        filename = 'RK4_cpu.c'
        pre = ''
    elif lang == 'gpu':
        filename = 'RK4_gpu.cu'
        pre = '__device__ '
    
    nn = len(specs) + 1
    
    file = open(filename, 'w')
    
    file.write(pre + 'void RK4 ( Real t, Real pr, Real h, Real * y0, Real * y ) {\n\n')
    
    file.write('  // variables holding various fractions of time step\n')
    file.write('  Real h2 = h / 2.0;\n')
    file.write('  Real onesixh = h / 6.0;\n')
    file.write('  Real onethirdh = h / 3.0;\n')
    file.write('\n')
    
    file.write('  // local array holding derivatives\n')
    file.write('  Real k[NN];\n')
    file.write('  // local array holding intermediate y values\n')
    file.write('  Real ym[NN];\n')
    file.write('\n')
    
    file.write('  // calculate k1\n')
    file.write('  dydt ( t, pr, y0, k );\n')
    file.write('\n')
    
    file.write('  // calculate midpoint values\n')
    for i in xrange(nn):
        file.write('  ym[{:}] = y0[{:}] + h2 * k[{:}];\n'.format(i, i, i))
    file.write('\n')
    
    file.write('  // add first contribution to integrated data\n')
    for i in xrange(nn):
        file.write('  y[{:}] = y0[{:}] + onesixh * k[{:}];\n'.format(i, i, i))
    file.write('\n')
    
    file.write('  // calculate k2\n')
    file.write('  dydt ( t + h2, pr, ym, k );\n')
    file.write('\n')
    
    file.write('  // calculate next midpoint values\n')
    for i in xrange(nn):
        file.write('  ym[{:}] = y0[{:}] + h2 * k[{:}];\n'.format(i, i, i))
    file.write('\n')
    
    file.write('  // calculate second contribution to integrated data\n')
    for i in xrange(nn):
        file.write('  y[{:}] += onethirdh * k[{:}];\n'.format(i, i))
    file.write('\n')
    
    file.write('  // calculate k3\n')
    file.write('  dydt ( t + h2, pr, ym, k );\n')
    file.write('\n')
    
    file.write('  // calculate next midpoint values\n')
    for i in xrange(nn):
        file.write('  ym[{:}] = y0[{:}] + h * k[{:}];\n'.format(i, i, i))
    file.write('\n')
    
    file.write('  // calculate third contribution to integrated data\n')
    for i in xrange(nn):
        file.write('  y[{:}] += onethirdh * k[{:}];\n'.format(i, i))
    file.write('\n')
    
    file.write('  // calculate k4\n')
    file.write('  dydt ( t + h, pr, ym, k );\n')
    file.write('\n')
    
    file.write('  // add final contribution to integrated data\n')
    for i in xrange(nn):
        file.write('  y[{:}] += onesixh * k[{:}];\n'.format(i, i))
    file.write('\n')
    
    file.write('} // end RK4\n')
    
    file.close()
    
    return


def write_main(lang, specs):
    """
    
    """
    if lang == 'cpu':
        filename = 'main_cpu.c'
        pre = ''
    elif lang == 'gpu':
        filename = 'main_gpu.cu'
        pre = '__global__ '
    
    nn = len(specs) + 1
    
    file = open(filename, 'w')
    
    # include other subroutine files
    file.write('#include "header.h"\n\n')
    
    if lang == 'cpu':
        file.write('#include "chem_utils_cpu.h"\n')
        file.write('#include "rxn_rates_cpu.c"\n')
        file.write('#include "spec_rates_cpu.c"\n')
        file.write('#include "dydt_cpu.c"\n')
        file.write('#include "RK4_cpu.c"\n\n')
    else:
        file.write('/** CUDA libraries */\n')
        file.write('#include <cuda.h>\n')
        file.write('#include <cutil.h>\n')
        file.write('\n')
        
        file.write('#include "chem_utils_gpu.cuh"\n')
        file.write('#include "rxn_rates_gpu.cu"\n')
        file.write('#include "spec_rates_gpu.cu"\n')
        file.write('#include "dydt_gpu.cu"\n')
        file.write('#include "RK4_gpu.cu"\n\n')
    
    file.write('///////////////////////////////////////////////////////\n\n')
    
    file.write(pre + 'void intDriver ( Real t, Real h, Real pr, Real * y_global ) {\n\n')
    
    if lang == 'cpu':
        file.write('  // loop over all "threads"\n')
        file.write('  for ( uint tid = 0; tid < NUM; ++tid ) {\n\n')
        tab = '    '
    else:
        file.write('  // unique thread ID, based on local ID in block and block ID\n')
        file.write('  uint tid = threadIdx.x + ( blockDim.x * blockIdx.x );\n\n')
        tab = '  '
    
    file.write(tab + '// local array with initial values\n')
    file.write(tab + 'Real y0_local[' + str(nn) + '];\n')
    file.write(tab + '// local array with integrated values\n')
    file.write(tab + 'Real yn_local[' + str(nn) + '];\n\n')
    
    file.write(tab + '// load local array with initial values from global array\n')
    for i in xrange(nn):
        line = tab + 'y0_local[' + str(i) + '] = y_global[tid + NUM * ' + str(i) + '];\n'
        file.write(line)
    file.write('\n')
    
    file.write(tab + '// call integrator for one time step\n')
    file.write(tab + 'RK4 ( t, pr, h, y0_local, yn_local );\n\n')
    
    file.write(tab + '// update global array with integrated values\n')
    for i in xrange(nn):
        line = tab + 'y_global[tid + NUM * ' + str(i) + '] = yn_local[' + str(i) + '];\n'
        file.write(line)
    file.write('\n')
    
    if lang == 'cpu':
        file.write('  } // end tid loop\n\n')
    
    file.write('} // end intDriver\n\n')
    
    file.write('///////////////////////////////////////////////////////\n\n')
    
    file.write('int main ( void ) {\n\n')
    
    if lang == 'cpu':
        file.write('  // print number of threads\n')
        file.write('  printf ("# threads: %d\\n", NUM);\n\n')
    else:
        file.write('  // print number of threads and block size\n')
        file.write('  printf ("# threads: %d \\t block size: %d\\n", NUM, BLOCK);\n\n')
    
    file.write('  // starting time (usually 0.0), units [s]\n')
    file.write('  Real t0 = 0.0;\n')
    file.write('  // ending time of integration, units [s]\n')
    file.write('  Real tend = 1.0e-7;\n')
    file.write('  // time step size, units [s]\n')
    file.write('  Real h = 1.0e-8;\n')
    file.write('  // number of steps, based on time range and step size\n')
    file.write('  uint steps = (tend - t0)/h;\n\n')
    
    file.write('  // species indices:\n')
    for sp in specs:
        file.write('  // ' + str(specs.index(sp)) + ' ' + sp.name + '\n')
    file.write('\n')
    
    file.write('  // initial mole fractions\n')
    file.write('  Real Xi[{:}];\n'.format(nn - 1))
    file.write('  for ( int j = 0; j < {:}; ++ j ) {{\n'.format(nn - 1))
    file.write('    Xi[j] = 0.0;\n')
    file.write('  }\n')
    file.write('\n')
    
    file.write('  //\n  // set initial mole fractions here\n  //\n\n')
    file.write('  // normalize mole fractions to sum to 1\n')
    file.write('  Real Xsum = 0.0;\n')
    file.write('  for ( int j = 0; j < {:}; ++ j ) {{\n'.format(nn - 1))
    file.write('    Xsum += Xi[j];\n')
    file.write('  }\n')
    file.write('  for ( int j = 0; j < {:}; ++ j ) {{\n'.format(nn - 1))
    file.write('    Xi[j] /= Xsum;\n')
    file.write('  }\n\n')
    
    file.write('  // initial mass fractions\n')
    file.write('  Real Yi[{:}];\n'.format(nn - 1))
    file.write('  mole2mass ( Xi, Yi );\n\n')
    
    file.write('  // size of data array in bytes\n')
    file.write('  uint size = NUM * sizeof(Real) * {:};\n\n'.format(nn))
    
    file.write('  // pointer to data on host memory\n')
    file.write('  Real *y_host;\n')
    file.write('  // allocate memory for all data on host\n')
    file.write('  y_host = (Real *) malloc (size);\n\n')
    
    file.write('  // set initial pressure, units [dyn/cm^2]\n')
    file.write('  // 1 atm = 101325 dyn/cm^2\n')
    file.write('  Real pres = 101325e0;\n\n')
    file.write('  // set initial temperature, units [K]\n')
    file.write('  Real T0 = 1600.0;\n\n')
    
    file.write('  // load temperature and mass fractions for all threads (cells)\n')
    file.write('  for ( int i = 0; i < NUM; ++i ) {\n')
    file.write('    y_host[i] = T0;\n')
    file.write('    // loop through species\n')
    file.write('    for ( int j = 1; j < {:}; ++j) {{\n'.format(nn))
    file.write('      y_host[i + NUM * j] = Yi[j - 1];\n')
    file.write('    }\n')
    file.write('  }\n\n')
    
    file.write('#ifdef CONV\n')
    file.write('  // if constant volume, calculate density\n')
    file.write('  Real rho = 0.0;\n')
    for sp in specs:
        file.write('  rho += Xi[{:}] * {:};\n'.format(specs.index(sp), sp.mw))
    file.write('  rho = pres * rho / ( {:} * T0 );\n'.format(RU))
    file.write('#endif\n\n')
    
    file.write('#ifdef IGN\n')
    file.write('  // flag for ignition\n')
    file.write('  bool ign_flag = false;\n')
    file.write('  // ignition delay time, units [s]\n')
    file.write('  Real t_ign = 0.0;\n')
    file.write('#endif\n\n')
    
    file.write('  // set time to initial time\n')
    file.write('  Real t = t0;\n\n')
    
    if lang == 'cpu':
        file.write('  // timer start point\n')
        file.write('  clock_t t_start;\n')
        file.write('  // timer end point\n')
        file.write('  clock_t t_end;\n\n')
        
        file.write('  // start timer\n')
        file.write('  t_start = clock();\n\n')
    else:
        file.write('  // set GPU card to one other than primary\n')
        file.write('  cudaSetDevice (1);\n\n')
        
        file.write('  // integer holding timer time\n')
        file.write('  uint timer_compute = 0;\n\n')
        file.write('  // create timer object\n')
        file.write('  CUT_SAFE_CALL ( cutCreateTimer ( &timer_compute ) );\n')
        file.write('  // start timer\n')
        file.write('  CUT_SAFE_CALL ( cutStartTimer ( timer_compute ) );\n\n')
    
    file.write('  // pointer to memory used for integration\n')
    file.write('  Real *y_device;\n')
    
    if lang == 'cpu':
        file.write('  // allocate memory\n')
        file.write('  y_device = (Real *) malloc ( size );\n\n')
    else:
        file.write('  // allocate memory on device\n')
        file.write('  CUDA_SAFE_CALL ( cudaMalloc ( (void**) &y_device, size ) );\n\n')
    
    # time integration loop
    file.write('  // time integration loop\n')
    file.write('  while ( t < tend ) {\n\n')
    if lang == 'cpu':
        file.write('    // copy local array to "global" array\n')
        file.write('    memcpy ( y_device, y_host, size );\n\n')
        
        file.write('#if defined(CONP)\n')
        file.write('    // constant pressure case\n')
        file.write('    intDriver ( t, h, pres, y_device );\n')
        file.write('#elif defined(CONV)\n')
        file.write('    // constant volume case\n')
        file.write('    intDriver ( t, h, rho, y_device );\n')
        file.write('#endif\n\n')
        
        file.write('    // transfer integrated data back to local array\n')
        file.write('    memcpy ( y_host, y_device, size );\n\n')
    else:
        file.write('    // copy data on host to device\n')
        file.write('    CUDA_SAFE_CALL ( cudaMemcpy ( y_device, y_host, size, cudaMemcpyHostToDevice ) );\n\n')
        file.write('    //\n    // kernel invocation\n    //\n\n')
        file.write('    // block size\n')
        file.write('    dim3 dimBlock ( BLOCK, 1 );\n')
        file.write('    // grid size\n')
        file.write('    dim3 dimGrid ( NUM / BLOCK, 1 );\n\n')
        
        file.write('#if defined(CONP)\n')
        file.write('    // constant pressure case\n')
        file.write('    intDriver <<< dimGrid, dimBlock >>> ( t, h, pres, y_device );\n')
        file.write('#elif defined(CONV)\n')
        file.write('    // constant volume case\n')
        file.write('    intDriver <<< dimGrid, dimBlock >>> ( t, h, rho, y_device );\n')
        file.write('#endif\n\n')
        
        file.write('#ifdef DEBUG\n')
        file.write('    // barrier thread synchronization\n')
        file.write('    CUDA_SAFE_CALL ( cudaThreadSynchronize() );\n')
        file.write('#endif\n\n')
        
        file.write('    // transfer integrated data from device back to host\n')
        file.write('    CUDA_SAFE_CALL ( cudaMemcpy ( y_host, y_device, size, cudaMemcpyDeviceToHost ) );\n\n')
    
    # check for ignition
    file.write('#ifdef IGN\n')
    file.write('    // determine if ignition has occurred\n')
    file.write('    if ( ( y_host[0] >= (T0 + 400.0) ) && !(ign_flag) ) {\n')
    file.write('      ign_flag = true;\n')
    file.write('      t_ign = t;\n')
    file.write('    }\n')
    file.write('#endif\n\n')
    
    file.write('    // increase time by one step\n')
    file.write('    t += h;\n\n')
    file.write('  } // end time loop\n\n')
    
    # after integration, free memory and stop timer
    if lang == 'cpu':
        file.write('  // free data array from global memory\n')
        file.write('  free ( y_device );\n\n')
        
        file.write('  // stop timer\n')
        file.write('  t_end = clock();\n\n')
        
        file.write('  // get clock tiem in seconds\n')
        file.write('  Real tim = ( t_end - t_start ) / ( (Real)(CLOCKS_PER_SEC) );\n')
    else:
        file.write('  // free data array from device memory\n')
        file.write('  CUDA_SAFE_CALL ( cudaFree ( y_device ) );\n\n')
        
        file.write('  // stop timer\n')
        file.write('  CUT_SAFE_CALL ( cutStopTimer ( timer_compute ) );\n\n')
        
        file.write('  // get clock time in seconds; cutGetTimerValue() returns ms\n')
        file.write('  Real tim = cutGetTimerValue ( timer_compute ) / 1000.0;\n')
    file.write('  tim = tim / ( (Real)(steps) );\n')
    
    # print time
    file.write('  // print time per step and time per step per thread\n')
    file.write('  printf("' + lang.upper() + ' time per step: %e (s)\\t%e (s/thread)\\n", tim, tim / NUM);\n\n')
    
    file.write('#ifdef CONV\n')
    file.write('  // calculate final pressure for constant volume case\n')
    file.write('  pres = 0.0;\n')
    for sp in specs:
        file.write('  pres += y_host[1 + NUM * {:}] / {:};\n'.format(specs.index(sp), sp.mw))
    file.write('  pres = rho * {:} * y_host[0] * pres;\n'.format(RU))
    file.write('#endif\n\n')
    
    file.write('#ifdef DEBUG\n')
    file.write('  // if debugging/testing, print temperature and first species mass fraction of last thread\n')
    file.write('  printf ("T[NUM-1]: %f, Yh: %e\\n", y_host[NUM-1], y_host[NUM-1+NUM]);\n')
    file.write('#endif\n\n')
    
    file.write('#ifdef IGN\n')
    file.write('  // if calculating ignition delay, print ign delay; units [s]\n')
    file.write('  printf ( "Ignition delay: %le\\n", t_ign );\n')
    file.write('#endif\n\n')
    
    file.write('  // free local data array\n')
    file.write('  free ( y_host );\n\n')
    
    file.write('  return 0;\n')
    file.write('} // end main\n')
    
    file.close()
    return


def write_header(specs):
    """
    
    """
    nsp = len(specs)
    nn = nsp + 1
    
    file = open('header.h', 'w')
    
    file.write('#include <stdlib.h>\n')
    file.write('#include <stdio.h>\n')
    file.write('#include <assert.h>\n')
    file.write('#include <time.h>\n')
    file.write('#include <math.h>\n')
    file.write('#include <string.h>\n')
    file.write('#include <stdbool.h>\n')
    file.write('\n')
    
    file.write('/** number of threads */\n')
    file.write('#define NUM 65536\n')
    file.write('/** GPU block size */\n')
    file.write('#define BLOCK 128\n')
    file.write('\n')
    
    file.write(
    '/** Sets precision as double or float. */\n' + 
    '#define DOUBLE\n' + 
    '#ifdef DOUBLE\n' + 
    '  /** Define Real as double. */\n' + 
    '  #define Real double\n' + 
    '\n' + 
    '  /** Double precision ONE. */\n' + 
    '  #define ONE 1.0\n' + 
    '  /** Double precision TWO. */\n' + 
    '  #define TWO 2.0\n' + 
    '  /** Double precision THREE. */\n' + 
    '  #define THREE 3.0\n' + 
    '  /** Double precision FOUR. */\n' + 
    '  #define FOUR 4.0\n' + 
    '#else\n' + 
    '  /** Define Real as float. */\n' + 
    '  #define Real float\n' + 
    '\n' + 
    '  /** Single precision ONE. */\n' + 
    '  #define ONE 1.0f\n' + 
    '  /** Single precision (float) TWO. */\n' + 
    '  #define TWO 2.0f\n' + 
    '  /** Single precision THREE. */\n' + 
    '  #define THREE 3.0f\n' + 
    '  /** Single precision FOUR. */\n' + 
    '  #define FOUR 4.0f\n' + 
    '#endif\n' + 
    '\n' + 
    '/** DEBUG definition. Used for barrier synchronization after kernel in GPU code. */\n' + 
    '#define DEBUG\n' + 
    '\n' + 
    '/** IGN definition. Used to flag ignition delay calculation. */\n' + 
    '//#define IGN\n' + 
    '\n' + 
    '/** PRINT definition. Used to flag printing of output values. */\n' + 
    '//#define PRINT\n' + 
    '\n' + 
    '/** Definition of problem type.\n' + 
    ' * CONV is constant volume.\n' + 
    ' * CONP is constant pressure.\n' + 
    ' */\n' + 
    '#define CONV\n\n')
    
    file.write('/** Number of species.\n')
    for sp in specs:
        file.write(' * {:} {:}\n'.format(specs.index(sp), sp.name))
    file.write(' */\n')
    file.write('#define NSP {:}\n'.format(nsp))
    
    file.write('/** Number of variables. NN = NSP + 1 (temperature). */\n')
    file.write('#define NN {:}\n'.format(nn))
    file.write('\n')
    
    file.write('/** Unsigned int typedef. */\n')
    file.write('typedef unsigned int uint;\n')
    file.write('/** Unsigned short int typedef. */\n')
    file.write('typedef unsigned short int usint;\n')
    file.write('\n')
    
    file.write('/** Function converting species mole fractions to mass fractions.\n')
    file.write(' *\n')
    file.write(' * \param[in]  X  array of species mole fractions\n')
    file.write(' * \param[out] Y  array of species mass fractions\n')
    file.write(' */\n')
    file.write('void mole2mass ( Real * X, Real * Y ) {\n\n')
    file.write('  // average molecular weight\n')
    file.write('  Real mw_avg = 0.0;\n')
    for sp in specs:
        file.write('  mw_avg += X[{:}] * {:};\n'.format(specs.index(sp), sp.mw))
    file.write('\n')
    
    file.write('  // calculate mass fractions\n')
    for sp in specs:
        isp = specs.index(sp)
        file.write('  Y[{:}] = X[{:}] * {:} / mw_avg;\n'.format(isp, isp, sp.mw))
    file.write('\n')
    file.write('} // end mole2mass\n\n')
    
    file.write('/** Function converting species mass fractions to mole fractions.\n')
    file.write(' *\n')
    file.write(' * \param[in]  Y  array of species mass fractions\n')
    file.write(' * \param[out] X  array of species mole fractions\n')
    file.write(' */\n')
    file.write('void mass2mole ( Real * Y, Real * X ) {\n\n')
    file.write('  // average molecular weight\n')
    file.write('  Real mw_avg = 0.0;\n')
    for sp in specs:
        file.write('  mw_avg += Y[{:}] / {:};\n'.format(specs.index(sp), sp.mw))
    file.write('  mw_avg = 1.0 / mw_avg;\n')
    file.write('\n')
    
    file.write('  // calculate mass fractions\n')
    for sp in specs:
        isp = specs.index(sp)
        file.write('  X[{:}] = Y[{:}] * mw_avg / {:};\n'.format(isp, isp, sp.mw))
    file.write('\n')
    file.write('} // end mass2mole\n\n')
    
    file.close()
    
    return


def create_jacobian(lang, mech_name, therm_name = None):
    """Create Jacobian subroutine from mechanism.
    
    Input
    lang_type: language type (C, CUDA, fortran, matlab)
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