import math
from chem_utilities import *
from mech_interpret import *

def file_lang_app(filename, lang):
    """Append filename with extension based on programming language.
    """
    if lang == 'c':
        filename += '.c'
    elif lang == 'cuda':
        filename += '.cu'
    elif lang == 'fortran':
        filename += '.f90'
    elif lang == 'matlab':
        filename += '.m'
    
    return filename


def line_end(lang):
    """Return appropriate line ending for langauge.
    """
    end = ''
    if lang in ['c', 'cuda', 'matlab']:
        end = ';\n'
    elif lang == 'fortran':
        end = '\n'
    
    return end


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
    """Write reaction rate subroutine. Conditionals for reverse reaction.
    
    Input
    lang: language type
    specs: list of species objects
    reacs: list of reaction objects
    """
    
    filename = file_lang_app('rxn_rates', lang)
    file = open(filename, 'w')
    
    num_s = len(specs)
    num_r = len(reacs)
    rev_reacs = [rxn for rxn in reacs if rxn.rev]
    num_rev = len(rev_reacs)
    
    line = ''
    if lang == 'cuda': line = '__device__ '
    
    if lang in ['c', 'cuda']:
        if rev_reacs:
            line += 'void eval_rxn_rates ( Real T, Real * C, Real * fwd_rxn_rates, Real * rev_rxn_rates ) {\n'
        else:
            line += 'void eval_rxn_rates ( Real T, Real * C, Real * fwd_rxn_rates ) {\n'
    elif lang == 'fortran':
        if rev_reacs:
            line += 'subroutine eval_rxn_rates ( T, C, fwd_rxn_rates, rev_rxn_rates )\n\n'
        else:
            line += 'subroutine eval_rxn_rates ( T, C, fwd_rxn_rates )\n\n'
        
        # fortran needs type declarations
        line += '  implicit none\n'
        line += '  double precision, intent(in) :: T, C(' + str(num_s) + ')\n'
        if rev_reacs:
            line += '  double precision, intent(out) :: fwd_rxn_rates(' + str(num_r) + '), rev_rxn_rates(' + str(num_rev) + ')\n'
        else:
            line += '  double precision, intent(out) :: fwd_rxn_rates(' + str(num_r) + ')\n'
        line += '  \n'
        line += '  double precision :: logT\n'
        if rev_reacs and any(rxn.rev_par != [] for rxn in rev_reacs):
                line += '  double precision :: kf, Kc\n'
        line += '\n'
    elif lang == 'matlab':
        if rev_reacs:
            line += 'function [fwd_rxn_rates, rev_rxn_rates] = eval_rxn_rates ( T, C )\n\n'
            line += '  fwd_rxn_rates = zeros(' + str(num_r) + ', 1);\n'
            line += '  rev_rxn_rates = fwd_rxn_rates;\n'
        else:
            line += 'function fwd_rxn_rates = eval_rxn_rates ( T, C )\n\n'
            line += '  fwd_rxn_rates = zeros(' + str(num_r) + ', 1);\n'
    file.write(line)
    
    file = open(filename, 'w')
    
    pre = '  '
    if lang == 'c':
        pre += 'Real '
    elif lang == 'cuda':
        pre += 'register Real '
    line = pre + 'logT = log(T)'
    line += line_end(lang)
    file.write(line)
    file.write('\n')
    
    if rev_reacs and any(rxn.rev_par != [] for rxn in rev_reacs):
        if lang == 'c':
            file.write('Real kf;\n')
            file.write('Real Kc;\n')
        elif lang == 'cuda':
            file.write('register Real kf;\n')
            file.write('register Real Kc;\n')
    
    for rxn in reacs:
        
        # if reversible, save forward rate constant for use
        if not rxn.rev_par:
            line = '  kf = ' + rxn_rate_const(rxn.A, rxn.b, rxn.E)
            line += line_end(lang)
            file.write(line)
        
        line = '  fwd_rxn_rates'
        if lang in ['c', 'cuda']:
            line += '[' + str(reacs.index(rxn)) + '] = '
        elif lang in ['fortran', 'matlab']:
            line += '(' + str(reacs.index(rxn) + 1) + ') = '
        
        # reactants
        for sp in rxn.reac:
            isp = next(i for i in xrange(len(specs)) if specs[i].name == sp)
            nu = rxn.reac_nu[rxn.reac.index(sp)]
            
            # check if stoichiometric coefficient is real or integer
            if isinstance(nu, float):
                if lang in ['c', 'cuda']:
                    line += 'pow(C[' + str(isp) + '], ' + str(nu) + ') * '
                elif lang in ['fortran', 'matlab']:
                    line += 'pow(C(' + str(isp + 1) + '), ' + str(nu) + ') * '
            else:
                # integer, so just use multiplication
                for i in range(nu):
                    if lang in ['c', 'cuda']:
                        line += 'C[' + str(isp) + '] * '
                    elif lang in ['fortran', 'matlab']:
                        line += 'C(' + str(isp) + ') * '
        
        # rate constant (print if not reversible, or reversible but with explicit reverse parameters)
        if not rxn.rev or rxn.rev_par:
            line += rxn_rate_const(rxn.A, rxn.b, rxn.E)
        else:
            line += 'kf'
        
        line += line_end(lang)
        file.write(line)
        
        if rxn.rev:
            
            if not rxn.rev_par:
                
                line = '  Kc = 0.0'
                line += line_end(lang)
                
                # sum of stoichiometric coefficients
                sum_nu = 0
                
                # go through product species
                for sp in rxn.prod:
                    isp = rxn.prod.index(sp)
                    
                    # check if species also in reactants
                    if sp in rxn.reac:
                        isp2 = rxn.reac.index(sp)
                        nu = rxn.prod_nu[isp] - rxn.reac_nu[isp2]
                    else:
                        nu = rxn.prod_nu[isp]
                    sum_nu += nu
                        
                    # need temperature conditional for equilibrium constants
                    line = '  if (T <= {:})'.format(sp.Trange[1])
                    if lang in ['c', 'cuda']:
                        line += ' {\n'
                    elif lang == 'fortran':
                        line += ' then\n'
                    elif lang == 'matlab':
                        line += '\n'
                    file.write(line)
                    
                    if nu < 0:
                        line = '  Kc = Kc - {:.2f} * '.format(abs(nu))
                    elif nu > 0:
                        line = '  Kc = Kc + {:.2f} * '.format(nu)
                    line += '({:.8e} - {:.8e} + {:.8e} * logT + T * ({:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T) ) ) - {:.8e} / T )'.format(sp.lo[6], sp.lo[0], sp.lo[0] - 1.0, sp.lo[1]/2.0, sp.lo[2]/6.0, sp.lo[3]/12.0, sp.lo[4]/20.0, sp.lo[5])
                    
                    if lang in ['c', 'cuda']:
                        file.write('  } else {\n')
                    elif lang in ['fortran', 'matlab']:
                        file.write('  else\n')
                    
                    if nu < 0:
                        line = '  Kc = Kc - {:.2f} * '.format(abs(nu))
                    elif nu > 0:
                        line = '  Kc = Kc + {:.2f} * '.format(nu)
                    line += '({:.8e} - {:.8e} + {:.8e} * logT + T * ({:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T) ) ) - {:.8e} / T )'.format(sp.hi[6], sp.hi[0], sp.hi[0] - 1.0, sp.hi[1]/2.0, sp.hi[2]/6.0, sp.hi[3]/12.0, sp.hi[4]/20.0, sp.hi[5])
                    
                    if lang in ['c', 'cuda']:
                        file.write('  }\n\n')
                    elif lang == 'fortran':
                        file.write('  end if\n\n')
                    elif lang == 'matlab':
                        file.write('  end\n\n')
                
                # now loop through reactants
                for sp in rxn.reac:
                    isp = rxn.prod.index(sp)
                    
                    # check if species also in products (if so, already considered)
                    if sp in rxn.prod: continue
                    
                    nu = rxn.reac_nu[isp]
                    sum_nu -= nu
                        
                    # need temperature conditional for equilibrium constants
                    line = '  if (T <= {:})'.format(sp.Trange[1])
                    if lang in ['c', 'cuda']:
                        line += ' {{\n'
                    elif lang == 'fortran':
                        line += ' then\n'
                    elif lang == 'matlab':
                        line += '\n'
                    file.write(line)
                    
                    line = '  Kc = Kc - {:.2f} * '.format(nu)
                    line += '({:.8e} - {:.8e} + {:.8e} * logT + T * ({:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T) ) ) - {:.8e} / T )'.format(sp.lo[6], sp.lo[0], sp.lo[0] - 1.0, sp.lo[1]/2.0, sp.lo[2]/6.0, sp.lo[3]/12.0, sp.lo[4]/20.0, sp.lo[5])
                    
                    if lang in ['c', 'cuda']:
                        file.write('  } else {\n')
                    elif lang in ['fortran', 'matlab']:
                        file.write('  else\n')
                    
                    line = '  Kc = Kc - {:.2f} * '.format(nu)
                    line += '({:.8e} - {:.8e} + {:.8e} * logT + T * ({:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T) ) ) - {:.8e} / T )'.format(sp.hi[6], sp.hi[0], sp.hi[0] - 1.0, sp.hi[1]/2.0, sp.hi[2]/6.0, sp.hi[3]/12.0, sp.hi[4]/20.0, sp.hi[5])
                    
                    if lang in ['c', 'cuda']:
                        file.write('  }\n\n')
                    elif lang == 'fortran':
                        file.write('  end if\n\n')
                    elif lang == 'matlab':
                        file.write('  end\n\n')
                
                line = '  Kc = {:.8e} * exp(Kc)'.format((PA / RU)**sum_nu)
            
            line = '  rev_rxn_rates'
            if lang in ['c', 'cuda']:
                line += '[' + str(rev_reacs.index(rxn)) + '] = '
            elif lang in ['fortran', 'matlab']:
                line += '(' + str(rev_reacs.index(rxn) + 1) + ') = '
            
            # reactants (products from forward reaction)
            for sp in rxn.reac:
                isp = next(i for i in xrange(len(specs)) if specs[i].name == sp)
                nu = rxn.reac_nu[rxn.reac.index(sp)]
            
                # check if stoichiometric coefficient is real or integer
                if isinstance(nu, float):
                    if lang in ['c', 'cuda']:
                        line += 'pow(C[' + str(isp) + '], ' + str(nu) + ') * '
                    elif lang in ['fortran', 'matlab']:
                        line += 'pow(C(' + str(isp + 1) + '), ' + str(nu) + ') * '
                else:
                    # integer, so just use multiplication
                    for i in range(nu):
                        if lang in ['c', 'cuda']:
                            line += 'C[' + str(isp) + '] * '
                        elif lang in ['fortran', 'matlab']:
                            line += 'C(' + str(isp) + ') * '
        
            # rate constant
            if rxn.rev_par:
                # explicit reverse Arrhenius parameters
                line += rxn_rate_const(rxn.rev_par[0], rxn.rev_par[1], rxn.rev_par[2])
            else:
                # use equilibrium constant
                line += 'kf / Kc'
            line += line_end(lang)
            file.write(line)
            
    
    if lang in ['c', 'cuda']:
        file.write('} // end eval_rxn_rates\n\n')
    elif lang == 'fortran':
        file.write('end subroutine eval_rxn_rates\n\n')
    elif lang == 'matlab':
        file.write('end\n\n')
    
    file.close()
    
    return


def write_rxn_pressure_mod(lang, specs, reacs):
    """Write subroutine to calculate pressure dependence modifications for reactions.
    
    Input
    lang: language type
    specs: list of species objects
    reacs: list of reaction objects
    """
    filename = file_lang_app('rxn_rates_pres_mod', lang)
    file = open(filename, 'w')
    
    # list of reactions with third-body or pressure-dependence
    pdep_reacs = []
    thd_flag = False
    pdep_flag = False
    troe_flag = False
    sri_flag = False
    for reac in reacs:
        if reac.thd:
            # add reaction index to list
            thd_flag = True
            pdep_reacs.append(reacs.index(reac))
        if reac.pdep:
            # add reaction index to list
            pdep_flag = True
            if not reac.thd: pdep_reacs.append(reacs.index(reac))
            
            if reac.troe and not troe_flag: troe_flag = True
            if reac.sri and not sri_flag: sri_flag = True
    
    line = ''
    if lang == 'cuda': line = '__device__ '
    
    if lang in ['c', 'cuda']:
        line += 'void get_rxn_pres_mod ( Real T, Real pres, Real * C, Real * pres_mod ) {\n'
    elif lang == 'fortran':
        line += 'subroutine get_rxn_pres_mod ( T, pres, C, pres_mod )\n\n'
        
        # fortran needs type declarations
        line += '  implicit none\n'
        line += '  double precision, intent(in) :: T, pres, C(' + str(len(specs)) + ')\n'
        line += '  double precision, intent(out) :: pres_mod(' + str(len(pdep_reacs)) + ')\n'
        line += '  \n'
        line += '  double precision :: logT, m\n'
    elif lang == 'matlab':
        line += 'function pres_mod = get_rxn_pres_mod ( T, pres, C )\n\n'
        line += '  pres_mod = zeros(' + str(len(pdep_reacs)) + ', 1);\n'
    file.write(line)
    
    # declarations for third-body variables
    if thd_flag:
        if lang == 'c':
            file.write('  // third body variable declaration\n')
            file.write('  Real thd;\n')
            file.write('\n')
        elif lang == 'cuda':
            file.write('  // third body variable declaration\n')
            file.write('  register Real thd;\n')
            file.write('\n')
        elif lang == 'fortran':
            file.write('  ! third body variable declaration\n')
            file.write('  double precision :: thd\n')
    
    # declarations for pressure-dependence variables
    if pdep_flag:
        if lang == 'c':
            file.write('  // pressure dependence variable declarations\n')
#            if not thd_flag: file.write('  Real thd;\n')
            file.write('  Real k0;\n')
            file.write('  Real kinf;\n')
            file.write('  Real Pr;\n')
            file.write('\n')
        
            if troe_flag:
                # troe variables
                file.write('  // troe variable declarations\n')
                file.write('  Real logPr;\n')
                file.write('  Real logFcent;\n')
                file.write('  Real A;\n')
                file.write('  Real B;\n')
                file.write('\n')
        
            if sri_flag:
                # sri variables
                file.write('  // sri variable declarations\n')
                if not troe_flag: file.write('  Real logPr;\n')
                file.write('  Real x;\n')
                file.write('\n')
        elif lang == 'cuda':
            file.write('  // pressure dependence variable declarations\n')
#            if not thd_flag: file.write('  register Real thd;\n')
            file.write('  register Real k0;\n')
            file.write('  register Real kinf;\n')
            file.write('  register Real Pr;\n')
            file.write('\n')
        
            if troe_flag:
                # troe variables
                file.write('  // troe variable declarations\n')
                file.write('  register Real logPr;\n')
                file.write('  register Real logFcent;\n')
                file.write('  register Real A;\n')
                file.write('  register Real B;\n')
                file.write('\n')
        
            if sri_flag:
                # sri variables
                file.write('  // sri variable declarations\n')
                if not troe_flag: file.write('  register Real logPr;\n')
                file.write('  register Real x;\n')
                file.write('\n')
        elif lang == 'fortran':
            file.write('  ! pressure dependence variable declarations\n')
#            if not thd_flag: file.write('  double precision :: thd\n')
            file.write('  double precision :: k0, kinf, Pr\n')
            file.write('\n')
        
            if troe_flag:
                # troe variables
                file.write('  ! troe variable declarations\n')
                file.write('  double precision :: logPr, logFcent, A, B\n')
                file.write('\n')
        
            if sri_flag:
                # sri variables
                file.write('  ! sri variable declarations\n')
                if not troe_flag: file.write('  double precision :: logPr\n')
                file.write('  double precision :: X\n')
                file.write('\n')
    
    if lang == 'c':
        file.write('  Real logT = log(T);\n')
        file.write('  Real m = p / ({:4e} * T);\n'.format(RU))
    elif lang == 'cuda':
        file.write('  register Real logT = log(T);\n')
        file.write('  register Real m = p / ({:4e} * T);\n'.format(RU))
    elif lang == 'fortran':
        file.write('  logT = log(T)\n')
        file.write('  m = p / ({:4e} * T)\n'.format(RU))
    elif lang == 'matlab':
        file.write('  logT = log(T);\n')
        file.write('  m = p / ({:4e} * T);\n'.format(RU))
    
    file.write('\n')
    
    # loop through third-body and pressure-dependent reactions
    for rind in pdep_reacs:
        reac = reacs[rind]              # index in reaction list
        pind = pdep_reacs.index(rind)   # index in list of third/pressure-dep reactions
        
        # print reaction index
        if lang in ['c', 'cuda']:
            line = '  // reaction ' + str(rind)
        elif lang == 'fortran':
            line = '  ! reaction ' + str(rind + 1)
        elif lang == 'matlab':
            line = '  % reaction ' + str(rind + 1)
        line += line_end(lang)
        file.write(line)
        
        # third-body reaction
        if reac.thd:
            
            if reac.pdep:
                line = '  thd = m'
            else:
                if lang in ['c', 'cuda']:
                    line = '  pres_mod(' + str(pind) + ') = m'
                elif lang in ['fortran', 'matlab']:
                    line = '  pres_mod(' + str(pind + 1) + ') = m'
            
            for sp in reac.thd_body:
                isp = specs.index( next((s for s in specs if s.name == sp[0]), None) )
                if sp[1] > 1.0:
                    if lang in ['c', 'cuda']:
                        line += ' + ' + str(sp[1] - 1.0) + ' * C[' + str(isp) + ']'
                    elif lang in ['fortran', 'matlab']:
                        line += ' + ' + str(sp[1] - 1.0) + ' * C(' + str(isp + 1) + ')'
                elif sp[1] < 1.0:
                    if lang in ['c', 'cuda']:
                        line += ' - ' + str(1.0 - sp[1]) + ' * C[' + str(isp) + ']'
                    elif lang in ['fortran', 'matlab']:
                        line += ' - ' + str(1.0 - sp[1]) + ' * C(' + str(isp + 1) + ')'
            
            line += line_end(lang)
            file.write(line)
        
        # pressure dependence
        if reac.pdep:
            
            # low-pressure limit rate
            line = '  k0 = '
            if reac.low:
                line += rxn_rate_const(reac.low[0], reac.low[1], reac.low[2])
            else:
                line += rxn_rate_const(reac.A, reac.b, reac.E)
            
            line += line_end(lang)
            file.write(line)
            
            # high-pressure limit rate
            line = '  kinf = '
            if reac.high:
                line += rxn_rate_const(reac.high[0], reac.high[1], reac.high[2])
            else:
                line += rxn_rate_const(reac.A, reac.b, reac.E)
            
            line += line_end(lang)
            file.write(line)
            
            # reduced pressure
            if reac.thd:
                line = '  Pr = k0 * thd / kinf'
            else:
                isp = next(i for i in xrange(len(specs)) if specs[i].name == reac.pdep_sp)
                if lang in ['c', 'cuda']:
                    line = '  Pr = k0 * C[' + str(isp) + '] / kinf'
                elif lang in ['fortran', 'matlab']:
                    line = '  Pr = k0 * C(' + str(isp + 1) + ') / kinf'
            line += line_end(lang)
            file.write(line)
            
            # log10 of Pr needed in both Troe and SRI formulation
            line = '  logPr = log10(Pr)'
            line += line_end(lang)
            file.write(line)
            
            if reac.troe:
                # Troe form
                
                line = '  logFcent = log10( {:.4e} * exp(-T / {:.4e})'.format(1.0 - reac.troe_par[0], reac.troe_par[1])
                line += ' + {:.4e} * exp(T / {:.4e})'.format(reac.troe_par[0], reac.troe_par[2])
                if len(reac.troe_par) == 4:
                    line += ' + exp(-{:.4e} / T)'.format(reac.troe_par[3])
                line += ' )'
                line += line_end(lang)
                file.write(line)
                
                line = '  A = logPr - 0.67 * logFcent - 0.4'
                line += line_end(lang)
                file.write(line)
                
                line = '  B = 0.806 - 1.1762 * logFcent - 0.14 * logPr'
                line += line_end(lang)
                file.write(line)
                
                line = '  pres_mod('
                if lang in ['c', 'cuda']:
                    line += str(pind) + ') = exp10( logFcent / ( 1.0 + A * A / (B * B) ) ) '
                elif lang in ['fortran', 'matlab']:
                    # fortran & matlab don't have exp10
                    line += str(pind + 1) + ') = exp( log(10.0) * logFcent / ( 1.0 + A * A / (B * B) ) ) '
                
            elif reac.sri:
                # SRI form
                
                line = '  X = 1.0 / (1.0 + logPr * logPr)'
                line += line_end(lang)
                file.write(line)
                
                line = '  pres_mod('
                if lang in ['c', 'cuda']:
                    line += str(pind)
                elif lang in ['fortran', 'matlab']:
                    line += str(pind + 1)
                
                line += ') = pow({:.4} * exp(-{:.4} / T) + exp(-T / {:.4}), X) '.format(rxn.sri[0], rxn.sri[1], rxn.sri[2])
                if len(rxn.sri) == 5:
                    line += '* {:.4e} * pow(T, {:.4}) '.format(rxn.sri[3], rxn.sri[4])
            
            # regardless of F formulation
            if reac.low:
                # unimolecular/recombination fall-off reaction
                line += '* Pr / (1.0 + Pr)'
            elif reac.high:
                # chemically-activated bimolecular reaction
                line += '/ (1.0 + Pr)'
            
            line += line_end(lang)
            file.write(line)
    
    if lang in ['c', 'cuda']:
        file.write('} // end get_rxn_pres_mod\n\n')
    elif lang == 'fortran':
        file.write('end subroutine get_rxn_pres_mod\n\n')
    elif lang == 'matlab':
        file.write('end\n\n')
    
    file.close()
    
    return


def write_spec_rates(lang, specs, reacs):
    """Write language-specific subroutine to evaluate species rates of production.
    
    Input
    lang:   programming language ('c', 'cuda', 'fortran', 'matlab')
    specs:  list of species objects
    reacs:  list of reaction objects
    """     
    filename = file_lang_app('spec_rates', lang)
    file = open(filename, 'w')
    
    num_s = len(specs)
    num_r = len(reacs)
    rev_reacs = [rxn for rxn in reacs if rxn.rev]
    num_rev = len(rev_reacs)
    
    # pressure dependent reactions
    pdep_reacs = []
    for reac in reacs:
        if reac.thd or reac.pdep:
            # add reaction index to list
            pdep_reacs.append(reacs.index(reac))
    num_pdep = len(pdep_reacs)
    
    line = ''
    if lang == 'cuda': line = '__device__ '
    
    if lang in ['c', 'cuda']:
        if rev_reacs:
            line += 'void eval_spec_rates ( Real * fwd_rates, Real * rev_rates, Real * pres_mod, Real * sp_rates ) {\n'
        else:
            line += 'void eval_spec_rates ( Real * fwd_rates, Real * pres_mod, Real * sp_rates ) {\n'
    elif lang == 'fortran':
        if rev_reacs:
            line += 'subroutine eval_spec_rates ( fwd_rates, rev_rates, pres_mod, sp_rates )\n\n'
        else:
            line += 'subroutine eval_spec_rates ( fwd_rates, pres_mod, sp_rates )\n\n'
        
        # fortran needs type declarations
        line += '  implicit none\n'
        if rev_reacs:
            line += '  double precision, intent(in) :: fwd_rates(' + str(num_r) + '), rev_rates(' + str(num_r) + '), pres_mod(' + str(num_pdep) + ')\n'
        else:
            line += '  double precision, intent(in) :: fwd_rates(' + str(num_r) + '), pres_mod(' + str(num_pdep) + ')\n'
        line += '  double precision, intent(out) :: sp_rates(' + str(num_s) + ')\n'
        line += '\n'
    elif lang == 'matlab':
        if rev_reacs:
            line += 'function sp_rates = eval_spec_rates ( fwd_rates, rev_rates, pres_mod )\n\n'
        else:
            line += 'function sp_rates = eval_spec_rates ( fwd_rates, pres_mod )\n\n'
        line += '  sp_rates = zeros(' + str(len(specs)) + ', 1);\n'
    file.write(line)
    
    file = open(filename, 'w')
    
    # loop through species
    for sp in specs:
        line = '  sp_rates'
        if lang in ['c', 'cuda']:
            line += '[' + str(specs.index(sp)) + '] = '
        elif lang in ['fortran', 'matlab']:
            line += '(' + str(specs.index(sp) + 1) + ') = '
        
        # continuation line
        cline = ' ' * ( len(line) - 2)
        
        isfirst = True
        
        inreac = False
        
        # loop through reactions
        for rxn in reacs:
            
            rind = reacs.index(rxn)
            
            pdep = False
            if rxn.thd or rxn.pdep: pdep = True
            
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
                    
                    if rxn.rev:
                        line += '(fwd_rates'
                        if lang in ['c', 'cuda']:
                            line += '[' + str(rind) + ']'
                        elif lang in ['fortran', 'matlab']:
                            line += '(' + str(rind + 1) + ')'
                        line += ' - rev_rates'
                        if lang in ['c', 'cuda']:
                            line += '[' + str(rev_reacs.index(rxn)) + ']'
                        elif lang in ['fortran', 'matlab']:
                            line += '(' + str(rev_reacs.index(rxn) + 1) + ')'
                        line += ')'
                    else:
                        line += 'fwd_rates'
                        if lang in ['c', 'cuda']:
                            line += '[' + str(rind) + ']'
                        elif lang in ['fortran', 'matlab']:
                            line += '(' + str(rind + 1) + ')'
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
                    
                    if rxn.rev:
                        line += '(fwd_rates'
                        if lang in ['c', 'cuda']:
                            line += '[' + str(rind) + ']'
                        elif lang in ['fortran', 'matlab']:
                            line += '(' + str(rind + 1) + ')'
                        line += ' - rev_rates'
                        if lang in ['c', 'cuda']:
                            line += '[' + str(rev_reacs.index(rxn)) + ']'
                        elif lang in ['fortran', 'matlab']:
                            line += '(' + str(rev_reacs.index(rxn) + 1) + ')'
                        line += ')'
                    else:
                        line += 'fwd_rates'
                        if lang in ['c', 'cuda']:
                            line += '[' + str(rind) + ']'
                        elif lang in ['fortran', 'matlab']:
                            line += '(' + str(rind + 1) + ')'
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
                
                if lang in ['c', 'cuda']:
                    line += 'rates[' + str(rind) + ']'
                elif lang in ['fortran', 'matlab']:
                    line += 'rates(' + str(rind + 1) + ')'
                
                if rxn.rev:
                        line += '(fwd_rates'
                        if lang in ['c', 'cuda']:
                            line += '[' + str(rind) + ']'
                        elif lang in ['fortran', 'matlab']:
                            line += '(' + str(rind + 1) + ')'
                        line += ' - rev_rates'
                        if lang in ['c', 'cuda']:
                            line += '[' + str(rev_reacs.index(rxn)) + ']'
                        elif lang in ['fortran', 'matlab']:
                            line += '(' + str(rev_reacs.index(rxn) + 1) + ')'
                        line += ')'
                else:
                    line += 'fwd_rates'
                    if lang in ['c', 'cuda']:
                        line += '[' + str(rind) + ']'
                    elif lang in ['fortran', 'matlab']:
                        line += '(' + str(rind + 1) + ')'
                
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
                
                if rxn.rev:
                        line += '(fwd_rates'
                        if lang in ['c', 'cuda']:
                            line += '[' + str(rind) + ']'
                        elif lang in ['fortran', 'matlab']:
                            line += '(' + str(rind + 1) + ')'
                        line += ' - rev_rates'
                        if lang in ['c', 'cuda']:
                            line += '[' + str(rev_reacs.index(rxn)) + ']'
                        elif lang in ['fortran', 'matlab']:
                            line += '(' + str(rev_reacs.index(rxn) + 1) + ')'
                        line += ')'
                else:
                    line += 'fwd_rates'
                    if lang in ['c', 'cuda']:
                        line += '[' + str(rind) + ']'
                    elif lang in ['fortran', 'matlab']:
                        line += '(' + str(rind + 1) + ')'
                
                if isfirst: isfirst = False
            else:
                continue
            
            # pressure dependence modification
            if pdep:
                pind = pdep_reacs.index(rind)
                if lang in ['c', 'cuda']:
                    line += ' * pres_mod[' + str(pind) + ']'
                elif lang in ['fortran', 'matlab']:
                    line += ' * pres_mod(' + str(pind) + ')'
        
        # species not participate in any reactions
        if not inreac: line += '0.0'
        
        # done with this species
        line += line_end(lang)
        line += '\n'
        file.write(line)
    
    if lang in ['c', 'cuda']:
        file.write('} // end eval_spec_rates\n\n')
    elif lang == 'fortran':
        file.write('end subroutine eval_spec_rates\n\n')
    elif lang == 'matlab':
        file.write('end\n\n')
    
    file.close()
    
    return

def write_chem_utils(lang, specs):
    """Write language-specific subroutine to evaluate species thermodynamic properties (enthalpy, energy, specific heat)
    
    Input
    lang:   programming language ('c', 'cuda', 'fortran', 'matlab')
    specs:  list of species objects
    """
    filename = file_lang_app('chem_utils', lang)
    file = open(filename, 'w')
    
    pre = ''
    if lang == 'cuda': pre = '__device__ '
    
    ######################
    # enthalpy subroutine
    ######################
    if lang in ['c', 'cuda']:
        line += pre + 'void eval_h ( Real T, Real * h ) {\n\n'
    elif lang == 'fortran':
        line += 'subroutine eval_h ( T, h )\n\n'
        
        # fortran needs type declarations
        line += '  implicit none\n'
        line += '  double precision, intent(in) :: T\n'
        line += '  double precision, intent(out) :: h(' + str(len(specs)) + ')\n'
        line += '\n'
    elif lang == 'matlab':
        line += 'function h = eval_h ( T )\n\n'
    file.write(line)
    
    # loop through species
    for sp in specs:
        line = '  if (T <= {:})'.format(sp.Trange[1])
        if lang in ['c', 'cuda']:
            line += ' {{\n'
        elif lang == 'fortran':
            line += ' then\n'
        elif lang == 'matlab':
            line += '\n'
        file.write(line)
        
        if lang in ['c', 'cuda']:
            line = '    h[' + str(specs.index(sp)) + ']'
        elif lang in ['fortran', 'matlab']:
            line = '    h(' + str(specs.index(sp) + 1) + ')'
        line += ' = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) ) )'.format(sp.lo[5], sp.lo[0], sp.lo[1] / 2.0, sp.lo[2] / 3.0, sp.lo[3] / 4.0, sp.lo[4] / 5.0)
        line += line_end(lang)
        file.write(line)
        
        if lang in ['c', 'cuda']:
            file.write('  } else {\n')
        elif lang in ['fortran', 'matlab']:
            file.write('  else\n')
        
        if lang in ['c', 'cuda']:
            line = '    h[' + str(specs.index(sp)) + ']'
        elif lang in ['fortran', 'matlab']:
            line = '    h(' + str(specs.index(sp) + 1) + ')'
        line += ' = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) ) )'.format(sp.hi[5], sp.hi[0], sp.hi[1] / 2.0, sp.hi[2] / 3.0, sp.hi[3] / 4.0, sp.hi[4] / 5.0)
        line += line_end(lang)
        file.write(line)
        
        if lang in ['c', 'cuda']:
            file.write('  }\n\n')
        elif lang == 'fortran':
            file.write('  end if\n\n')
        elif lang == 'matlab':
            file.write('  end\n\n')
    
    if lang in ['c', 'cuda']:
        file.write('} // end eval_h\n\n')
    elif lang == 'fortran':
        file.write('end subroutine eval_h\n\n')
    elif lang == 'matlab':
        file.write('end\n\n')
    
    #################################
    # internal energy subroutine
    #################################
    if lang in ['c', 'cuda']:
        line += pre + 'void eval_u ( Real T, Real * u ) {\n\n'
    elif lang == 'fortran':
        line += 'subroutine eval_u ( T, u )\n\n'
        
        # fortran needs type declarations
        line += '  implicit none\n'
        line += '  double precision, intent(in) :: T\n'
        line += '  double precision, intent(out) :: u(' + str(len(specs)) + ')\n'
        line += '\n'
    elif lang == 'matlab':
        line += 'function u = eval_u ( T )\n\n'
    file.write(line)
    
    # loop through species
    for sp in specs:
        line = '  if (T <= {:})'.format(sp.Trange[1])
        if lang in ['c', 'cuda']:
            line += ' {{\n'
        elif lang == 'fortran':
            line += ' then\n'
        elif lang == 'matlab':
            line += '\n'
        file.write(line)
        
        if lang in ['c', 'cuda']:
            line = '    u[' + str(specs.index(sp)) + ']'
        elif lang in ['fortran', 'matlab']:
            line = '    u(' + str(specs.index(sp) + 1) + ')'
        line += ' = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} + T * ( {:.8e} - 1.0 + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) ) )'.format(sp.lo[5], sp.lo[0], sp.lo[1] / 2.0, sp.lo[2] / 3.0, sp.lo[3] / 4.0, sp.lo[4] / 5.0)
        line += line_end(lang)
        file.write(line)
        
        if lang in ['c', 'cuda']:
            file.write('  } else {\n')
        elif lang in ['fortran', 'matlab']:
            file.write('  else\n')
        
        if lang in ['c', 'cuda']:
            line = '    u[' + str(specs.index(sp)) + ']'
        elif lang in ['fortran', 'matlab']:
            line = '    u(' + str(specs.index(sp) + 1) + ')'
        line += ' = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} + T * ( {:.8e} - 1.0 + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) ) )'.format(sp.hi[5], sp.hi[0], sp.hi[1] / 2.0, sp.hi[2] / 3.0, sp.hi[3] / 4.0, sp.hi[4] / 5.0)
        line += line_end(lang)
        file.write(line)
        
        if lang in ['c', 'cuda']:
            file.write('  }\n\n')
        elif lang == 'fortran':
            file.write('  end if\n\n')
        elif lang == 'matlab':
            file.write('  end\n\n')
    
    if lang in ['c', 'cuda']:
        file.write('} // end eval_u\n\n')
    elif lang == 'fortran':
        file.write('end subroutine eval_u\n\n')
    elif lang == 'matlab':
        file.write('end\n\n')
    
    ##################################
    # cv subroutine
    ##################################
    if lang in ['c', 'cuda']:
        line += pre + 'void eval_cv ( Real T, Real * cv ) {\n'
    elif lang == 'fortran':
        line += 'subroutine eval_cv ( T, cv )\n\n'
        
        # fortran needs type declarations
        line += '  implicit none\n'
        line += '  double precision, intent(in) :: T\n'
        line += '  double precision, intent(out) :: cv(' + str(len(specs)) + ')\n'
        line += '\n'
    elif lang == 'matlab':
        line += 'function cv = eval_cv ( T )\n\n'
    file.write(line)
    
    # loop through species
    for sp in specs:
        line = '  if (T <= {:})'.format(sp.Trange[1])
        if lang in ['c', 'cuda']:
            line += ' {{\n'
        elif lang == 'fortran':
            line += ' then\n'
        elif lang == 'matlab':
            line += '\n'
        file.write(line)
        
        if lang in ['c', 'cuda']:
            line = '    cv[' + str(specs.index(sp)) + ']'
        elif lang in ['fortran', 'matlab']:
            line = '    cv(' + str(specs.index(sp) + 1) + ')'
        line += ' = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} - 1.0 + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) )'.format(sp.lo[0], sp.lo[1], sp.lo[2], sp.lo[3], sp.lo[4])
        line += line_end(lang)
        file.write(line)
        
        if lang in ['c', 'cuda']:
            file.write('  } else {\n')
        elif lang in ['fortran', 'matlab']:
            file.write('  else\n')
        
        if lang in ['c', 'cuda']:
            line = '    cv[' + str(specs.index(sp)) + ']'
        elif lang in ['fortran', 'matlab']:
            line = '    cv(' + str(specs.index(sp) + 1) + ')'
        line += ' = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} - 1.0 + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) )'.format(sp.hi[0], sp.hi[1], sp.hi[2], sp.hi[3], sp.hi[4])
        line += line_end(lang)
        file.write(line)
        
        if lang in ['c', 'cuda']:
            file.write('  }\n\n')
        elif lang == 'fortran':
            file.write('  end if\n\n')
        elif lang == 'matlab':
            file.write('  end\n\n')
    
    if lang in ['c', 'cuda']:
        file.write('} // end eval_cv\n\n')
    elif lang == 'fortran':
        file.write('end subroutine eval_cv\n\n')
    elif lang == 'matlab':
        file.write('end\n\n')
    
    ###############################
    # cp subroutine 
    ###############################
    if lang in ['c', 'cuda']:
        line += pre + 'void eval_cp ( Real T, Real * cp ) {\n\n'
    elif lang == 'fortran':
        line += 'subroutine eval_cp ( T, cp )\n\n'
        
        # fortran needs type declarations
        line += '  implicit none\n'
        line += '  double precision, intent(in) :: T\n'
        line += '  double precision, intent(out) :: cp(' + str(len(specs)) + ')\n'
        line += '\n'
    elif lang == 'matlab':
        line += 'function cp = eval_cp ( T )\n\n'
    file.write(line)
    
    # loop through species
    for sp in specs:
        line = '  if (T <= {:})'.format(sp.Trange[1])
        if lang in ['c', 'cuda']:
            line += ' {{\n'
        elif lang == 'fortran':
            line += ' then\n'
        elif lang == 'matlab':
            line += '\n'
        file.write(line)
        
        if lang in ['c', 'cuda']:
            line = '    cp[' + str(specs.index(sp)) + ']'
        elif lang in ['fortran', 'matlab']:
            line = '    cp(' + str(specs.index(sp) + 1) + ')'
        line += ' = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) )'.format(sp.lo[0], sp.lo[1], sp.lo[2], sp.lo[3], sp.lo[4])
        line += line_end(lang)
        file.write(line)
        
        if lang in ['c', 'cuda']:
            file.write('  } else {\n')
        elif lang in ['fortran', 'matlab']:
            file.write('  else\n')
        
        if lang in ['c', 'cuda']:
            line = '    cp[' + str(specs.index(sp)) + ']'
        elif lang in ['fortran', 'matlab']:
            line = '    cp(' + str(specs.index(sp) + 1) + ')'
        line += ' = {:e} * '.format(RU / sp.mw)
        line += '( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + T * ( {:.8e} + {:.8e} * T ) ) ) )'.format(sp.hi[0], sp.hi[1], sp.hi[2], sp.hi[3], sp.hi[4])
        line += line_end(lang)
        file.write(line)
        
        if lang in ['c', 'cuda']:
            file.write('  }\n\n')
        elif lang == 'fortran':
            file.write('  end if\n\n')
        elif lang == 'matlab':
            file.write('  end\n\n')
    
    if lang in ['c', 'cuda']:
        file.write('} // end eval_cp\n\n')
    elif lang == 'fortran':
        file.write('end subroutine eval_cp\n\n')
    elif lang == 'matlab':
        file.write('end\n\n')
    
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
