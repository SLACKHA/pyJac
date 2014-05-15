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
                line += 'exp({:.8e}'.format(logA)
                if b > 0:
                    line += ' + ' + str(b)
                else:
                    line += ' - ' + str(abs(b))
                line += ' * logT)'
    else:
        # E != 0
        if not b:
            # b = 0
            line += 'exp({:.8e}'.format(logA) + ' - ({:.8e} / T))'.format(E)
        else:
            # b!= 0
            line += 'exp({:.8e}'.format(logA)
            if b > 0:
                line += ' + ' + str(b)
            else:
                line += ' - ' + str(abs(b))
            line += ' * logT - ({:.8e} / T))'.format(E)
    
    return line


def write_rxn_rates(path, lang, specs, reacs):
    """Write reaction rate subroutine. Conditionals for reverse reaction.
    
    Keyword arguments:
    path  -- path to build directory for file
    lang  -- language type
    specs -- list of species objects
    reacs -- list of reaction objects
    """
    
    num_s = len(specs)
    num_r = len(reacs)
    rev_reacs = [rxn for rxn in reacs if rxn.rev]
    num_rev = len(rev_reacs)
    
    pdep_reacs = []
    for reac in reacs:
        if reac.thd or reac.pdep:
            # add reaction index to list
            pdep_reacs.append(reacs.index(reac))
    
    # first write header file
    if lang == 'c':
        file = open(path + 'rates.h', 'w')
        file.write('#ifndef RATES_HEAD\n')
        file.write('#define RATES_HEAD\n')
        file.write('\n')
        file.write('#include "header.h"\n')
        file.write('\n')
        if rev_reacs:
            file.write('void eval_rxn_rates (const Real, const Real*, Real*, Real*);\n')
            file.write('void eval_spec_rates (const Real*, const Real*, const Real*, Real*);\n')
        else:
            file.write('void eval_rxn_rates (const Real, const Real*, Real*);\n')
            file.write('void eval_spec_rates (const Real*, const Real*, Real*);\n')
        
        if pdep_reacs:
            file.write('void get_rxn_pres_mod (const Real, const Real, const Real*, Real*);\n')
        
        file.write('\n')
        file.write('#endif\n')
        file.close()
    elif lang == 'cuda':
        file = open(path + 'rates.cuh', 'w')
        file.write('#ifndef RATES_HEAD\n')
        file.write('#define RATES_HEAD\n')
        file.write('\n')
        file.write('#include "header.h"\n')
        file.write('\n')
        if rev_reacs:
            file.write('__device__ void eval_rxn_rates (const Real, const Real*, Real*, Real*);\n')
            file.write('__device__ void eval_spec_rates (const Real*, const Real*, const Real*, Real*);\n')
        else:
            file.write('__device__ void eval_rxn_rates (const Real, const Real*, Real*);\n')
            file.write('__device__ void eval_spec_rates (const Real*, const Real*, Real*);\n')
        
        if pdep_reacs:
            file.write('__device__ void get_rxn_pres_mod (const Real, const Real, const Real*, Real*);\n')
        
        file.write('\n')
        file.write('#endif\n')
        file.close()
    
    filename = file_lang_app('rxn_rates', lang)
    file = open(path + filename, 'w')
    
    if lang in ['c', 'cuda']:
        file.write('#include <math.h>\n')
        file.write('#include "header.h"\n')
        file.write('\n')
    
    line = ''
    if lang == 'cuda': line = '__device__ '
    
    if lang in ['c', 'cuda']:
        if rev_reacs:
            line += 'void eval_rxn_rates (const Real T, const Real * C, Real * fwd_rxn_rates, Real * rev_rxn_rates) {\n'
        else:
            line += 'void eval_rxn_rates (const Real T, const Real * C, Real * fwd_rxn_rates) {\n'
    elif lang == 'fortran':
        if rev_reacs:
            line += 'subroutine eval_rxn_rates (T, C, fwd_rxn_rates, rev_rxn_rates)\n\n'
        else:
            line += 'subroutine eval_rxn_rates (T, C, fwd_rxn_rates)\n\n'
        
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
            line += 'function [fwd_rxn_rates, rev_rxn_rates] = eval_rxn_rates (T, C)\n\n'
            line += '  fwd_rxn_rates = zeros(' + str(num_r) + ', 1);\n'
            line += '  rev_rxn_rates = fwd_rxn_rates;\n'
        else:
            line += 'function fwd_rxn_rates = eval_rxn_rates (T, C)\n\n'
            line += '  fwd_rxn_rates = zeros(' + str(num_r) + ', 1);\n'
    file.write(line)
    
    pre = '  '
    if lang == 'c':
        pre += 'Real '
    elif lang == 'cuda':
        pre += 'register Real '
    line = pre + 'logT = log(T)'
    line += line_end(lang)
    file.write(line)
    file.write('\n')
    
    if rev_reacs and any(rxn.rev_par == [] for rxn in rev_reacs):
        if lang == 'c':
            file.write('  Real kf;\n')
            file.write('  Real Kc;\n')
        elif lang == 'cuda':
            file.write('  register Real kf;\n')
            file.write('  register Real Kc;\n')
    
    file.write('\n')
    
    for rxn in reacs:
        
        # if reversible, save forward rate constant for use
        if rxn.rev and not rxn.rev_par:
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
                file.write(line)
                
                # sum of stoichiometric coefficients
                sum_nu = 0
                
                # go through product species
                for prod_sp in rxn.prod:
                    isp = rxn.prod.index(prod_sp)
                    
                    # check if species also in reactants
                    if prod_sp in rxn.reac:
                        isp2 = rxn.reac.index(prod_sp)
                        nu = rxn.prod_nu[isp] - rxn.reac_nu[isp2]
                    else:
                        nu = rxn.prod_nu[isp]
                    
                    # skip species with zero overall stoichiometric coefficient
                    if (nu == 0):
                        continue
                    
                    sum_nu += nu
                    
                    # get species object
                    sp = next((sp for sp in specs if sp.name == prod_sp), None)
                    if not sp:
                        print 'Error: species ' + prod_sp + ' in reaction ' + str(reacs.index(rxn)) + ' not found.\n'
                        sys.exit()
                    
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
                        line = '    Kc = Kc - {:.2f} * '.format(abs(nu))
                    elif nu > 0:
                        line = '    Kc = Kc + {:.2f} * '.format(nu)
                    line += '({:.8e} - {:.8e} + {:.8e} * logT + T * ({:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T) ) ) - {:.8e} / T )'.format(sp.lo[6], sp.lo[0], sp.lo[0] - 1.0, sp.lo[1]/2.0, sp.lo[2]/6.0, sp.lo[3]/12.0, sp.lo[4]/20.0, sp.lo[5])
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
                    line += '({:.8e} - {:.8e} + {:.8e} * logT + T * ({:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T) ) ) - {:.8e} / T )'.format(sp.hi[6], sp.hi[0], sp.hi[0] - 1.0, sp.hi[1]/2.0, sp.hi[2]/6.0, sp.hi[3]/12.0, sp.hi[4]/20.0, sp.hi[5])
                    line += line_end(lang)
                    file.write(line)
                    
                    if lang in ['c', 'cuda']:
                        file.write('  }\n\n')
                    elif lang == 'fortran':
                        file.write('  end if\n\n')
                    elif lang == 'matlab':
                        file.write('  end\n\n')
                
                # now loop through reactants
                for reac_sp in rxn.reac:
                    isp = rxn.reac.index(reac_sp)
                    
                    # check if species also in products (if so, already considered)
                    if reac_sp in rxn.prod: continue
                    
                    nu = rxn.reac_nu[isp]
                    sum_nu -= nu
                    
                    # get species object
                    sp = next((sp for sp in specs if sp.name == reac_sp), None)
                    if not sp:
                        print 'Error: species ' + reac_sp + ' in reaction ' + str(reacs.index(rxn)) + ' not found.\n'
                        sys.exit()
                        
                    # need temperature conditional for equilibrium constants
                    line = '  if (T <= {:})'.format(sp.Trange[1])
                    if lang in ['c', 'cuda']:
                        line += ' {\n'
                    elif lang == 'fortran':
                        line += ' then\n'
                    elif lang == 'matlab':
                        line += '\n'
                    file.write(line)
                    
                    line = '    Kc = Kc - {:.2f} * '.format(nu)
                    line += '({:.8e} - {:.8e} + {:.8e} * logT + T * ({:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T) ) ) - {:.8e} / T )'.format(sp.lo[6], sp.lo[0], sp.lo[0] - 1.0, sp.lo[1]/2.0, sp.lo[2]/6.0, sp.lo[3]/12.0, sp.lo[4]/20.0, sp.lo[5])
                    line += line_end(lang)
                    file.write(line)
                    
                    if lang in ['c', 'cuda']:
                        file.write('  } else {\n')
                    elif lang in ['fortran', 'matlab']:
                        file.write('  else\n')
                    
                    line = '    Kc = Kc - {:.2f} * '.format(nu)
                    line += '({:.8e} - {:.8e} + {:.8e} * logT + T * ({:.8e} + T * ({:.8e} + T * ({:.8e} + {:.8e} * T) ) ) - {:.8e} / T )'.format(sp.hi[6], sp.hi[0], sp.hi[0] - 1.0, sp.hi[1]/2.0, sp.hi[2]/6.0, sp.hi[3]/12.0, sp.hi[4]/20.0, sp.hi[5])
                    line += line_end(lang)
                    file.write(line)
                    
                    if lang in ['c', 'cuda']:
                        file.write('  }\n\n')
                    elif lang == 'fortran':
                        file.write('  end if\n\n')
                    elif lang == 'matlab':
                        file.write('  end\n\n')
                
                line = '  Kc = {:.8e} * exp(Kc)'.format((PA / RU)**sum_nu)
                line += line_end(lang)
                file.write(line)
            
            line = '  rev_rxn_rates'
            if lang in ['c', 'cuda']:
                line += '[' + str(rev_reacs.index(rxn)) + '] = '
            elif lang in ['fortran', 'matlab']:
                line += '(' + str(rev_reacs.index(rxn) + 1) + ') = '
            
            # reactants (products from forward reaction)
            for sp in rxn.prod:
                isp = next(i for i in xrange(len(specs)) if specs[i].name == sp)
                nu = rxn.prod_nu[rxn.prod.index(sp)]
            
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


def write_rxn_pressure_mod(path, lang, specs, reacs):
    """Write subroutine to calculate pressure dependence modifications for reactions.
    
    Keyword arguments:
    path  -- path to build directory for file
    lang  -- language type
    specs -- list of species objects
    reacs -- list of reaction objects
    """
    filename = file_lang_app('rxn_rates_pres_mod', lang)
    file = open(path + filename, 'w')
    
    # headers
    if lang in ['c', 'cuda']:
        file.write('#include <math.h>\n')
        file.write('#include "header.h"\n')
        file.write('\n')
    
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
        line += 'void get_rxn_pres_mod (const Real T, const Real pres, const Real * C, Real * pres_mod) {\n'
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
        file.write('  Real m = pres / ({:.8e} * T);\n'.format(RU))
    elif lang == 'cuda':
        file.write('  register Real logT = log(T);\n')
        file.write('  register Real m = pres / ({:.8e} * T);\n'.format(RU))
    elif lang == 'fortran':
        file.write('  logT = log(T)\n')
        file.write('  m = pres / ({:.8e} * T)\n'.format(RU))
    elif lang == 'matlab':
        file.write('  logT = log(T);\n')
        file.write('  m = pres / ({:.8e} * T);\n'.format(RU))
    
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
            
            if reac.pdep and not reac.pdep_sp:
                line = '  thd = m'
            else:
                if lang in ['c', 'cuda']:
                    line = '  pres_mod[' + str(pind) + '] = m'
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
                line = '  logFcent = log10( {:.8e} * '.format(1.0 - reac.troe_par[0])
                if reac.troe_par[1] > 0.0:
                    line += 'exp(-T / {:.8e})'.format(reac.troe_par[1])
                else:
                    line += 'exp(T / {:.8e})'.format(abs(reac.troe_par[1]))
                
                if reac.troe_par[2] > 0.0:
                    line += ' + {:.8e} * exp(-T / {:.8e})'.format(reac.troe_par[0], reac.troe_par[2])
                else:
                    line += ' + {:.8e} * exp(T / {:.8e})'.format(reac.troe_par[0], abs(reac.troe_par[2]))
                
                if len(reac.troe_par) == 4:
                    if reac.troe_par[3] > 0.0:
                        line += ' + exp(-{:.8e} / T)'.format(reac.troe_par[3])
                    else:
                        line += ' + exp({:.8e} / T)'.format(abs(reac.troe_par[3]))
                line += ' )' + line_end(lang)
                file.write(line)
                
                
                line = '  A = logPr - 0.67 * logFcent - 0.4'
                line += line_end(lang)
                file.write(line)
                
                line = '  B = 0.806 - 1.1762 * logFcent - 0.14 * logPr'
                line += line_end(lang)
                file.write(line)
                
                line = '  pres_mod'
                if lang in ['c', 'cuda']:
                    #line += '[' + str(pind) + '] = exp10(logFcent / (1.0 + A * A / (B * B))) '
                    line += '[' + str(pind) + ']'
                elif lang in ['fortran', 'matlab']:
                    # fortran & matlab don't have exp10
                    line += '(' + str(pind + 1) + ')'
                line += ' = exp(log(10.0) * logFcent / (1.0 + A * A / (B * B))) '
                
            elif reac.sri:
                # SRI form
                
                line = '  X = 1.0 / (1.0 + logPr * logPr)'
                line += line_end(lang)
                file.write(line)
                
                line = '  pres_mod'
                if lang in ['c', 'cuda']:
                    line += '[' + str(pind) + ']'
                elif lang in ['fortran', 'matlab']:
                    line += '(' + str(pind + 1) + ')'
                line += ' = pow({:4} * '.format(reac.sri[0])
                # need to check for negative parameters, and skip "-" sign if so
                if reac.sri[1] > 0.0:
                    line += 'exp(-{:.4} / T)'.format(reac.sri[1])
                else:
                    line += 'exp({:.4} / T)'.format(abs(reac.sri[1]))
                
                if reac.sri[2] > 0.0:
                    line += ' + exp(-T / {:.4}), X) '.format(reac.sri[2])
                else:
                    line += ' + exp(T / {:.4}), X) '.format(abs(reac.sri[2]))
                    
                if len(reac.sri) == 5:
                    line += '* {:.8e} * pow(T, {:.4}) '.format(reac.sri[3], reac.sri[4])
            
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


def write_spec_rates(path, lang, specs, reacs):
    """Write language-specific subroutine to evaluate species rates of production.
    
    Keyword arguments:
    path  -- path to build directory for file
    lang  -- programming language ('c', 'cuda', 'fortran', 'matlab')
    specs -- list of species objects
    reacs -- list of reaction objects
    """     
    filename = file_lang_app('spec_rates', lang)
    file = open(path + filename, 'w')
    
    if lang in ['c', 'cuda']:
        file.write('#include "header.h"\n')
        file.write('\n')
    
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
            line += 'void eval_spec_rates (const Real * fwd_rates, const Real * rev_rates, const Real * pres_mod, Real * sp_rates) {\n'
        else:
            line += 'void eval_spec_rates (const Real * fwd_rates, const Real * pres_mod, Real * sp_rates) {\n'
    elif lang == 'fortran':
        if rev_reacs:
            line += 'subroutine eval_spec_rates (fwd_rates, rev_rates, pres_mod, sp_rates)\n\n'
        else:
            line += 'subroutine eval_spec_rates (fwd_rates, pres_mod, sp_rates)\n\n'
        
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
                # record position
                lastPos = file.tell()
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
                
#                if lang in ['c', 'cuda']:
#                    line += 'rates[' + str(rind) + ']'
#                elif lang in ['fortran', 'matlab']:
#                    line += 'rates(' + str(rind + 1) + ')'
                
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

def write_chem_utils(path, lang, specs):
    """Write language-specific subroutine to evaluate species thermodynamic properties (enthalpy, energy, specific heat)
    
    Keyword arguments:
    path  -- path to build directory for file
    lang  -- programming language ('c', 'cuda', 'fortran', 'matlab')
    specs -- list of species objects
    """
    
    # first write header file
    if lang == 'c':
        file = open(path + 'chem_utils.h', 'w')
        file.write('#ifndef CHEM_UTILS_HEAD\n')
        file.write('#define CHEM_UTILS_HEAD\n')
        file.write('\n')
        file.write('#include "header.h"\n')
        file.write('\n')
        file.write('void eval_h (const Real, Real*);\n')
        file.write('void eval_u (const Real, Real*);\n')
        file.write('void eval_cv (const Real, Real*);\n')
        file.write('void eval_cp (const Real, Real*);\n')
        file.write('\n')
        file.write('#endif\n')
        file.close()
    elif lang == 'cuda':
        file = open(path + 'chem_utils.cuh', 'w')
        file.write('#ifndef CHEM_UTILS_HEAD\n')
        file.write('#define CHEM_UTILS_HEAD\n')
        file.write('\n')
        file.write('#include "header.h"\n')
        file.write('\n')
        file.write('__device__ void eval_h (const Real, Real*);\n')
        file.write('__device__ void eval_u (const Real, Real*);\n')
        file.write('__device__ void eval_cv (const Real, Real*);\n')
        file.write('__device__ void eval_cp (const Real, Real*);\n')
        file.write('\n')
        file.write('#endif\n')
        file.close()
    
    filename = file_lang_app('chem_utils', lang)
    file = open(path + filename, 'w')
    
    if lang in ['c', 'cuda']:
        file.write('#include "header.h"\n\n')
    
    pre = ''
    if lang == 'cuda': pre = '__device__ '
    
    ######################
    # enthalpy subroutine
    ######################
    line = pre
    if lang in ['c', 'cuda']:
        line += 'void eval_h (const Real T, Real * h) {\n\n'
    elif lang == 'fortran':
        line += 'subroutine eval_h (T, h)\n\n'
        
        # fortran needs type declarations
        line += '  implicit none\n'
        line += '  double precision, intent(in) :: T\n'
        line += '  double precision, intent(out) :: h(' + str(len(specs)) + ')\n'
        line += '\n'
    elif lang == 'matlab':
        line += 'function h = eval_h (T)\n\n'
    file.write(line)
    
    # loop through species
    for sp in specs:
        line = '  if (T <= {:})'.format(sp.Trange[1])
        if lang in ['c', 'cuda']:
            line += ' {\n'
        elif lang == 'fortran':
            line += ' then\n'
        elif lang == 'matlab':
            line += '\n'
        file.write(line)
        
        if lang in ['c', 'cuda']:
            line = '    h[' + str(specs.index(sp)) + ']'
        elif lang in ['fortran', 'matlab']:
            line = '    h(' + str(specs.index(sp) + 1) + ')'
        line += ' = {:.8e} * '.format(RU / sp.mw)
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
        line += ' = {:.8e} * '.format(RU / sp.mw)
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
    line = pre
    if lang in ['c', 'cuda']:
        line += 'void eval_u (const Real T, Real * u) {\n\n'
    elif lang == 'fortran':
        line += 'subroutine eval_u (T, u)\n\n'
        
        # fortran needs type declarations
        line += '  implicit none\n'
        line += '  double precision, intent(in) :: T\n'
        line += '  double precision, intent(out) :: u(' + str(len(specs)) + ')\n'
        line += '\n'
    elif lang == 'matlab':
        line += 'function u = eval_u (T)\n\n'
    file.write(line)
    
    # loop through species
    for sp in specs:
        line = '  if (T <= {:})'.format(sp.Trange[1])
        if lang in ['c', 'cuda']:
            line += ' {\n'
        elif lang == 'fortran':
            line += ' then\n'
        elif lang == 'matlab':
            line += '\n'
        file.write(line)
        
        if lang in ['c', 'cuda']:
            line = '    u[' + str(specs.index(sp)) + ']'
        elif lang in ['fortran', 'matlab']:
            line = '    u(' + str(specs.index(sp) + 1) + ')'
        line += ' = {:.8e} * '.format(RU / sp.mw)
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
        line += ' = {:.8e} * '.format(RU / sp.mw)
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
        line = pre + 'void eval_cv (const Real T, Real * cv) {\n\n'
    elif lang == 'fortran':
        line = 'subroutine eval_cv (T, cv)\n\n'
        
        # fortran needs type declarations
        line += '  implicit none\n'
        line += '  double precision, intent(in) :: T\n'
        line += '  double precision, intent(out) :: cv(' + str(len(specs)) + ')\n'
        line += '\n'
    elif lang == 'matlab':
        line = 'function cv = eval_cv (T)\n\n'
    file.write(line)
    
    # loop through species
    for sp in specs:
        line = '  if (T <= {:})'.format(sp.Trange[1])
        if lang in ['c', 'cuda']:
            line += ' {\n'
        elif lang == 'fortran':
            line += ' then\n'
        elif lang == 'matlab':
            line += '\n'
        file.write(line)
        
        if lang in ['c', 'cuda']:
            line = '    cv[' + str(specs.index(sp)) + ']'
        elif lang in ['fortran', 'matlab']:
            line = '    cv(' + str(specs.index(sp) + 1) + ')'
        line += ' = {:.8e} * '.format(RU / sp.mw)
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
        line += ' = {:.8e} * '.format(RU / sp.mw)
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
        line = pre + 'void eval_cp (const Real T, Real * cp) {\n\n'
    elif lang == 'fortran':
        line = 'subroutine eval_cp (T, cp)\n\n'
        
        # fortran needs type declarations
        line += '  implicit none\n'
        line += '  double precision, intent(in) :: T\n'
        line += '  double precision, intent(out) :: cp(' + str(len(specs)) + ')\n'
        line += '\n'
    elif lang == 'matlab':
        line = 'function cp = eval_cp (T)\n\n'
    file.write(line)
    
    # loop through species
    for sp in specs:
        line = '  if (T <= {:})'.format(sp.Trange[1])
        if lang in ['c', 'cuda']:
            line += ' {\n'
        elif lang == 'fortran':
            line += ' then\n'
        elif lang == 'matlab':
            line += '\n'
        file.write(line)
        
        if lang in ['c', 'cuda']:
            line = '    cp[' + str(specs.index(sp)) + ']'
        elif lang in ['fortran', 'matlab']:
            line = '    cp(' + str(specs.index(sp) + 1) + ')'
        line += ' = {:.8e} * '.format(RU / sp.mw)
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
        line += ' = {:.8e} * '.format(RU / sp.mw)
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

def write_derivs(path, lang, specs, reacs):
    """Writes derivative function file and header.
    
    Keyword arguments:
    path  -- path to build directory for file
    lang  -- programming language ('c', 'cuda', 'fortran', 'matlab')
    specs -- list of species objects
    reacs -- list of reaction objects
    """
    
    # first write header file
    if lang == 'c':
        file = open(path + 'dydt.h', 'w')
        file.write('#ifndef DYDT_HEAD\n')
        file.write('#define DYDT_HEAD\n')
        file.write('\n')
        file.write('#include "header.h"\n')
        file.write('\n')
        file.write('void dydt (const Real, const Real, const Real*, Real*);\n')
        file.write('\n')
        file.write('#endif\n')
        file.close()
    elif lang == 'cuda':
        file = open(path + 'dydt.cuh', 'w')
        file.write('#ifndef DYDT_HEAD\n')
        file.write('#define DYDT_HEAD\n')
        file.write('\n')
        file.write('#include "header.h"\n')
        file.write('\n')
        file.write('__device__ void dydt (const Real, const Real, const Real*, Real*);\n')
        file.write('\n')
        file.write('#endif\n')
        file.close()
    
    filename = file_lang_app('dydt', lang)
    file = open(path + filename, 'w')

    pre = ''
    if lang == 'cuda': pre = '__device__ '
    
    file.write('#include "header.h"\n')
    if lang == 'c':
        file.write('#include "chem_utils.h"\n')
        file.write('#include "rates.h"\n')
    elif lang == 'cuda':
        file.write('#include "chem_utils.cuh"\n')
        file.write('#include "rates.cuh"\n')
    file.write('\n')
    
    # constant pressure
    file.write('#if defined(CONP)\n\n')
    
    line = pre + 'void dydt (const Real t, const Real pres, const Real * y, Real * dy) {\n\n'
    file.write(line)
    
    # avoid T variable, just use y[0]
    #file.write('  Real T = y[0];\n\n')
    
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
    line = '  rho = pres / ({:.8e} * y[0] * rho);\n\n'.format(RU)
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
    rev_reacs = [rxn for rxn in reacs if rxn.rev]
    if rev_reacs:
        file.write('  // local arrays holding reaction rates\n')
        file.write('  Real fwd_rates[{:}];\n'.format(len(reacs)) )
        file.write('  Real rev_rates[{:}];\n'.format(len(rev_reacs)) )
        file.write('  eval_rxn_rates (y[0], conc, fwd_rates, rev_rates);\n\n')
    else:
        file.write('  // local array holding reaction rates\n')
        file.write('  Real rates[{:}];\n'.format(len(reacs)) )
        file.write('  eval_rxn_rates (y[0], conc, rates);\n\n')
    
    # reaction pressure dependence
    pdep_reacs = []
    for reac in reacs:
        if reac.thd or reac.pdep:
            # add reaction index to list
            pdep_reacs.append(reacs.index(reac))
    num_pdep = len(pdep_reacs)
    if pdep_reacs:
        if lang in ['c', 'cuda']:
            file.write('  // get pressure modifications to reaction rates\n')
            file.write('  Real pres_mod[' + str(num_pdep) + '];\n')
            file.write('  get_rxn_pres_mod (y[0], pres, conc, pres_mod);\n')
        elif lang == 'fortran':
            file.write('  ! get and evaluate pressure modifications to reaction rates\n')
            file.write('  get_rxn_pres_mod (y[0], pres, conc, pres_mod)\n')
        elif lang == 'matlab':
            file.write('  % get and evaluate pressure modifications to reaction rates\n')
            file.write('  pres_mod = get_rxn_pres_mod (y[0], pres, conc, pres_mod);\n')
        file.write('\n')
    
    # species rate of change of molar concentration
    file.write('  // evaluate rate of change of species molar concentration\n')
    if rev_reacs and pdep_reacs:
        file.write('  eval_spec_rates (fwd_rates, rev_rates, pres_mod, &dy[1]);\n\n')
    elif rev_reacs:
        file.write('  eval_spec_rates (fwd_rates, rev_rates, &dy[1]);\n\n')
    else:
        file.write('  eval_spec_rates (rates, &dy[1] );\n\n')
    
    # evaluate specific heat
    file.write('  // local array holding constant pressure specific heat\n')
    file.write('  Real cp[{:}];\n'.format(len(specs)) )
    file.write('  eval_cp (y[0], cp);\n\n')
    
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
    file.write('  eval_h (y[0], h);\n\n')
    
    # energy equation
    file.write('  // rate of change of temperature\n')
    line = '  dy[0] = (-1.0 / ( rho * cp_avg )) * ( '
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
    
    line = pre + 'void dydt (const Real t, const Real rho, const Real * y, Real * dy) {\n\n'
    file.write(line)
    
    # just use y[0] for temperature
    #file.write('  Real T = y[0];\n\n')
    
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
    line = '  pres = rho * {:.8e} * y[0] * pres;\n\n'.format(RU)
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
    file.write('  Real rates[{:}];\n'.format(len(reacs)) )
    file.write('  eval_rxn_rates (y[0], pres, conc, rates);\n\n')
    
    # species rate of change of molar concentration
    file.write('  // evaluate rate of change of species molar concentration\n')
    file.write('  eval_spec_rates (rates, &dy[1]);\n\n')
    
    # evaluate specific heat
    file.write('  // local array holding constant volume specific heat\n')
    file.write('  Real cv[{:}];\n'.format(len(specs)) )
    file.write('  eval_cv (y[0], cv);\n\n')
    
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
    file.write('  eval_u (y[0], u);\n\n')
    
    # energy equation
    file.write('  // rate of change of temperature\n')
    line = '  dy[0] = (-1.0 / ( rho * cv_avg )) * ( '
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

def write_mass_mole(path, lang, specs):
    """Writes file and header for mass/molar concentration and density conversion utility.
    
    Keyword arguments:
    path -- path to build directory for file
    lang  -- programming language ('c', 'cuda', 'fortran', 'matlab')
    specs -- list of species objects
    """
    
    # Create header file
    if lang in ['c', 'cuda']:
        file = open(path + 'mass_mole.h', 'w')
    
        file.write('#ifndef MASS_MOLE_H\n')
        file.write('#define MASS_MOLE_H\n\n')
    
        file.write('#ifdef __cplusplus\n  extern "C" {\n#endif\n')
    
        file.write('void mole2mass (const Real*, Real*);\n')
        file.write('void mass2mole (const Real*, Real*);\n')
        file.write('Real getDensity (Real, Real, Real*);\n')
    
        file.write('#ifdef __cplusplus\n  }\n#endif\n')
    
        file.write('#endif\n')
        file.close()
    
    # Open file; both C and CUDA programs use C file (only used on host)
    if lang in ['c', 'cuda']:
        filename = 'mass_mole.c'
    elif lang == 'fortran':
        filename = 'mass_mole.f90'
    elif lang == 'matlab':
        filename = 'mass_mole.m'
    file = open(path + filename, 'w')
    
    if lang in ['c', 'cuda']:
        file.write('#include "header.h"\n\n')
    
    ###################################################
    # Documentation and function/subroutine initializaton for mole2mass
    if lang in ['c', 'cuda']:
        file.write('/** Function converting species mole fractions to mass fractions.\n')
        file.write(' *\n')
        file.write(' * \param[in]  X  array of species mole fractions\n')
        file.write(' * \param[out] Y  array of species mass fractions\n')
        file.write(' */\n')
        file.write('void mole2mass (const Real * X, Real * Y) {\n\n')
    elif lang == 'fortran':
        file.write('!-----------------------------------------------------------------\n')
        file.write('!> Subroutine converting species mole fractions to mass fractions.\n')
        file.write('!! @param[in]  X  array of species mole fractions\n')
        file.write('!! @param[out] Y  array of species mass fractions\n')
        file.write('!-----------------------------------------------------------------\n')
        file.write('subroutine mole2mass (X, Y)\n\n')
        file.write('  implicit none\n')
        file.write('  double, dimension(:), intent(in) :: X\n')
        file.write('  double, dimension(:), intent(out) :: X\n')
        file.write('  double :: mw_avg\n\n')
    
    # calculate molecular weight
    if lang in ['c', 'cuda']:
        file.write('  // average molecular weight\n')
        file.write('  Real mw_avg = 0.0;\n')
        for sp in specs:
            file.write('  mw_avg += X[{:}] * {:};\n'.format(specs.index(sp), sp.mw))
    elif lang == 'fortran':
        file.write('  ! average molecular weight\n')
        file.write('  mw_avg = 0.0\n')
        for sp in specs:
            file.write('  mw_avg = mw_avg + X({:}) * {:}\n'.format(specs.index(sp) + 1, sp.mw))
    file.write('\n')
    
    # calculate mass fractions
    if lang in ['c', 'cuda']:
        file.write('  // calculate mass fractions\n')
        for sp in specs:
            isp = specs.index(sp)
            file.write('  Y[{:}] = X[{:}] * {:} / mw_avg;\n'.format(isp, isp, sp.mw))
        file.write('\n')
        file.write('} // end mole2mass\n\n')
    elif lang == 'fortran':
        file.write('  ! calculate mass fractions\n')
        for sp in specs:
            isp = specs.index(sp) + 1
            file.write('  Y({:}) = X({:}) * {:} / mw_avg\n'.format(isp, isp, sp.mw))
        file.write('\n')
        file.write('end subroutine mole2mass\n\n')
    
    ################################
    # Documentation and function/subroutine initialization for mass2mole
    
    if lang in ['c', 'cuda']:
        file.write('/** Function converting species mass fractions to mole fractions.\n')
        file.write(' *\n')
        file.write(' * \param[in]  Y  array of species mass fractions\n')
        file.write(' * \param[out] X  array of species mole fractions\n')
        file.write(' */\n')
        file.write('void mass2mole (const Real * Y, Real * X) {\n\n')
    elif lang == 'fortran':
        file.write('!-----------------------------------------------------------------\n')
        file.write('!> Subroutine converting species mass fractions to mole fractions.\n')
        file.write('!! @param[in]  Y  array of species mass fractions\n')
        file.write('!! @param[out] X  array of species mole fractions\n')
        file.write('!-----------------------------------------------------------------\n')
        file.write('subroutine mass2mole (Y, X)\n\n')
        file.write('  implicit none\n')
        file.write('  double, dimension(:), intent(in) :: Y\n')
        file.write('  double, dimension(:), intent(out) :: X\n')
        file.write('  double :: mw_avg\n\n')
    
    # calculate average molecular weight
    if lang in ['c', 'cuda']:
        file.write('  // average molecular weight\n')
        file.write('  Real mw_avg = 0.0;\n')
        for sp in specs:
            file.write('  mw_avg += Y[{:}] / {:};\n'.format(specs.index(sp), sp.mw))
        file.write('  mw_avg = 1.0 / mw_avg;\n')
    elif lang == 'fortran':
        file.write('  ! average molecular weight\n')
        file.write('  mw_avg = 0.0\n')
        for sp in specs:
            file.write('  mw_avg = mw_avg + Y({:}) / {:}\n'.format(specs.index(sp) + 1, sp.mw))
    file.write('\n')
    
    # calculate mass fractions
    if lang in ['c', 'cuda']:
        file.write('  // calculate mass fractions\n')
        for sp in specs:
            isp = specs.index(sp)
            file.write('  X[{:}] = Y[{:}] * mw_avg / {:};\n'.format(isp, isp, sp.mw))
        file.write('\n')
        file.write('} // end mass2mole\n\n')
    elif lang == 'fortran':
        file.write('  ! calculate mass fractions\n')
        for sp in specs:
            isp = specs.index(sp) + 1
            file.write('  X({:}) = Y({:}) * mw_avg / {:}\n'.format(isp, isp, sp.mw))
        file.write('\n')
        file.write('end subroutine mass2mole\n\n')
    
    
    ###############################
    # Documentation and subroutine/function initialization for getDensity
    
    if lang in ['c', 'cuda']:
        file.write('/** Function calculating density from mole fractions.\n')
        file.write(' *\n')
        file.write(' * \param[in]  temp  temperature\n')
        file.write(' * \param[in]  pres  pressure\n')
        file.write(' * \param[in]  X     array of species mole fractions\n')
        file.write(r' * \return     rho  mixture mass density' + '\n')
        file.write(' */\n')
        file.write('Real getDensity (const Real temp, const real pres, Real * X) {\n\n')
    elif lang == 'fortran':
        file.write('!-----------------------------------------------------------------\n')
        file.write('!> Function calculating density from mole fractions.\n')
        file.write('!! @param[in]  temp  temperature\n')
        file.write('!! @param[in]  pres  pressure\n')
        file.write('!! @param[in]  X     array of species mole fractions\n')
        file.write('!! @return     rho   mixture mass density' + '\n')
        file.write('!-----------------------------------------------------------------\n')
        file.write('function mass2mole (temp, pres, X) result(rho)\n\n')
        file.write('  implicit none\n')
        file.write('  double, intent(in) :: temp, pres\n')
        file.write('  double, dimension(:), intent(in) :: X\n')
        file.write('  double :: mw_avg, rho\n\n')
    
    # get molecular weight
    if lang in ['c', 'cuda']:
        file.write('  // average molecular weight\n')
        file.write('  Real mw_avg = 0.0;\n')
        for sp in specs:
            file.write('  mw_avg += X[{:}] * {:};\n'.format(specs.index(sp), sp.mw))
        file.write('\n')
    elif lang == 'fortran':
        file.write('  ! average molecular weight\n')
        file.write('  mw_avg = 0.0\n')
        for sp in specs:
            file.write('  mw_avg = mw_avg + X({:}) * {:}\n'.format(specs.index(sp), sp.mw))
        file.write('\n')
    
    # calculate density
    if lang in ['c', 'cuda']:
        file.write('  rho = pres * mw_avg / ({:.8e} * temp);\n'.format(RU))
        file.write('  return rho;\n')
    elif lang == 'fortran':
        file.write('  rho = pres * mw_avg / ({:.8e} * temp)\n'.format(RU))
    file.write('\n')
    
    if lang in ['c', 'cuda']:
        file.write('} // end getDensity\n\n')
    elif lang == 'fortran':
        file.write('end function getDensity\n\n')
    
    file.close()
    return

