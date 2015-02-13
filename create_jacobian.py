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

# Local imports
import chem_utilities as chem
import mech_interpret as mech
import rate_subs as rate
import utils
import mech_auxiliary as aux
import CUDAParams


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
        line = '  Real T = y' + utils.get_array(lang, 0)
    elif lang in ['fortran', 'matlab']:
        line = '  T = y' + utils.get_array(lang, 0)
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
        line += '(y' + utils.get_array(lang, specs.index(sp) + 1) + \
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
        line = ('  conc' + utils.get_array(lang, isp) +
                ' = rho * ' +
                'y' + utils.get_array(lang, isp + 1) +
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
            jline = '  jac' + utils.get_array(lang, k_sp + 1, twod=1)
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
                    jline += ' = jac' + utils.get_array(lang, k_sp, twod=1)

                if nu != 1:
                    jline += '{} * '.format(float(nu))

            jline += '('

            if rxn.pdep:
                # print lines for necessary pressure-dependent variables
                line = '  Pr = '
                pind = pdep_reacs.index(rind)

                if rxn.pdep_sp:
                    line += 'conc' + \
                        utils.get_array(lang, specs.index(rxn.pdep_sp))
                else:
                    line += '(m'

                    for thd_sp in rxn.thd_body:
                        isp = specs.index(next((s for s in specs
                                                if s.name == thd_sp[0]), None))
                        if thd_sp[1] > 1.0:
                            line += ' + {} * conc'.format(thd_sp[1] - 1.0)
                        elif thd_sp[1] < 1.0:
                            line += ' - {} * conc'.format(1.0 - thd_sp[1])
                        if thd_sp[1] != 1.0:
                            line += utils.get_array(lang, isp)
                    line += ')'

                jline += 'pres_mod' + utils.get_array(lang, pind)
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

                    jline += '-Pr'

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
                    jline += '(fwd_rates' + utils.get_array(lang, rind)
                    jline += ' - rev_rates' + \
                        utils.get_array(lang, rev_reacs.index(rxn))
                    jline += ')'
                else:
                    # forward reaction rate only
                    jline += 'fwd_rates' + utils.get_array(lang, rind)

                jline += ' + (pres_mod' + utils.get_array(lang, pind)

            else:
                # not pressure dependent

                # third body reaction
                if rxn.thd:
                    pind = pdep_reacs.index(rind)

                    jline += '(-pres_mod' + utils.get_array(lang, pind)
                    jline += ' * '

                    if rxn.rev:
                        # forward and reverse reaction rates
                        jline += '(fwd_rates' + utils.get_array(lang, rind)
                        jline += ' - rev_rates' + \
                            utils.get_array(lang, rev_reacs.index(rxn))
                        jline += ')'
                    else:
                        # forward reaction rate only
                        jline += 'fwd_rates' + utils.get_array(lang, rind)

                    jline += ' / T) + (pres_mod' + utils.get_array(lang, pind)

                else:
                    if lang in ['c', 'cuda', 'matlab']:
                        jline += '(1.0'
                    elif lang in ['fortran']:
                        jline += '(1.0_wp'

            jline += ' / T) * ('

            # contribution from temperature derivative of forward reaction rate
            jline += 'fwd_rates' + utils.get_array(lang, rind)

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

                jline += ' - rev_rates' + \
                    utils.get_array(lang, rev_reacs.index(rxn))

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
            line = '  jac' + utils.get_array(lang, k_sp + 1)
            line += ' = 0.0'
        else:
            line = '  jac'
            if lang in ['c', 'cuda']:
                line += (utils.get_array(lang, k_sp + 1) +
                         ' *= {:.8e} / rho'.format(sp_k.mw)
                         )
            elif lang in ['fortran', 'matlab']:
                # NOTE: I believe there was a bug here w/ the previous
                # fortran/matlab code (as it looks like it would be zero
                # indexed)
                line += (utils.get_array(lang, k_sp + 1, twod=1) +
                         ' = jac' + utils.get_array(lang, k_sp + 1, twod=1) +
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
                jline = '  jac'
                if lang in ['c', 'cuda']:
                    jline += utils.get_array(lang,
                                             k_sp + 1 + (num_s + 1) * (j_sp + 1))
                    sparse_indicies.append(k_sp + 1 + (num_s + 1) * (j_sp + 1))
                elif lang in ['fortran', 'matlab']:
                    jline += utils.get_array(lang, k_sp + 1, twod=j_sp + 1)
                    sparse_indicies.append((k_sp + 1, j_sp + 1))

                if isfirst:
                    jline += ' = '
                    isfirst = False
                else:
                    if lang in ['c', 'cuda']:
                        jline += ' += '
                    elif lang in ['fortran', 'matlab']:
                        jline += utils.get_array(lang, k_sp + 1, twod=j_sp + 1)

                if nu != 1:
                    jline += '{} * '.format(float(nu))

                # start reaction
                jline += '('

                if rxn.thd and not rxn.pdep:
                    # third-body reaction
                    pind = pdep_reacs.index(rind)

                    jline += '(-mw_avg * pres_mod' + \
                        utils.get_array(lang, pind)
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
                        jline += '(fwd_rates' + utils.get_array(lang, rind)
                        jline += ' - rev_rates' + \
                            utils.get_array(lang, rev_reacs.index(rxn))
                        jline += ')'
                    else:
                        jline += 'fwd_rates' + utils.get_array(lang, rind)

                    jline += ' + '
                elif rxn.pdep:
                    # pressure-dependent reaction

                    line = '  Pr = '
                    pind = pdep_reacs.index(rind)

                    if rxn.pdep_sp:
                        line += 'conc' + \
                            utils.get_array(lang, specs.index(rxn.pdep_sp))
                    else:
                        line += '(m'
                        for thd_sp in rxn.thd_body:
                            isp = specs.index(next((s for s in specs
                                                    if s.name == thd_sp[0]), None))
                            if thd_sp[1] > 1.0:
                                line += ' + {} * conc'.format(thd_sp[1] - 1.0)
                            elif thd_sp[1] < 1.0:
                                line += ' - {} * conc'.format(1.0 - thd_sp[1])
                            if thd_sp[1] != 1.0:
                                line += utils.get_array(lang, isp)
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

                    jline += 'pres_mod' + utils.get_array(lang, pind)
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
                                    jline += (' + '
                                              '{}'.format(thd_sp[1] - 1.0) +
                                              ' * conc'
                                              )
                                    jline += utils.get_array(lang, isp)
                                elif thd_sp[1] < 1.0:
                                    jline += (' - '
                                              '{}'.format(1.0 - thd_sp[1]) +
                                              ' * conc'
                                              )
                                    jline += utils.get_array(lang, isp)
                            jline += ') * {:.8e})'.format(sp_j.mw)
                    elif rxn.pdep and rxn.pdep_sp == sp_j.name:
                        # NOTE: This Y array previously seems to be a bug, I can't find a reference to it anywhere else
                        #      changing it to y
                        jline += ' + (1.0 / y' + utils.get_array(lang, j_sp)
                        jline += ')'
                    jline += ') * '

                    if rxn.rev:
                        jline += '(fwd_rates' + utils.get_array(lang, rind)
                        jline += ' - rev_rates' + \
                            utils.get_array(lang, rev_reacs.index(rxn))
                        jline += ')'
                    else:
                        jline += 'fwd_rates' + utils.get_array(lang, rind)

                    jline += ' + '

                # next, contribution from dR/dYj
                if rxn.pdep or rxn.thd:
                    jline += 'pres_mod' + utils.get_array(lang, pind)
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

                        jline += '-mw_avg * fwd_rates' + \
                            utils.get_array(lang, rind)
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
                                    jline += ' * pow(conc' + \
                                        utils.get_array(lang, l_sp)
                                    jline += ', {})'.format(nu - 1)
                                else:
                                    # integer, so just use multiplication
                                    for i in range(nu - 1):
                                        jline += ' * conc' + \
                                            utils.get_array(lang, l_sp)

                            # loop through remaining reactants
                            for sp_reac in rxn.reac:
                                if sp_reac == sp_l.name:
                                    continue

                                nu = rxn.reac_nu[rxn.reac.index(sp_reac)]
                                isp = next(i for i in xrange(len(specs))
                                           if specs[i].name == sp_reac)
                                if isinstance(nu, float):
                                    jline += ' * pow(conc' + \
                                        utils.get_array(lang, isp)
                                    jline += ', ' + str(nu) + ')'
                                else:
                                    # integer, so just use multiplication
                                    for i in range(nu):
                                        jline += ' * conc' + \
                                            utils.get_array(lang, isp)
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

                        jline += '-mw_avg * rev_rates' + \
                            utils.get_array(lang, rev_reacs.index(rxn))
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
                                        jline += ' * conc' + \
                                            utils.get_array(lang, l_sp)

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
                                        jline += ' * conc' + \
                                            utils.get_array(lang, isp)
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
                line = '  jac'
                if lang in ['c', 'cuda']:
                    line += utils.get_array(lang,
                                            k_sp + 1 + (num_s + 1) * (j_sp + 1))
                    sparse_indicies.append(k_sp + 1 + (num_s + 1) * (j_sp + 1))
                elif lang in ['fortran', 'matlab']:
                    line += utils.get_array(lang, k_sp + 1, twod=j_sp + 1)
                    sparse_indicies.append((k_sp + 1, j_sp + 1))
                line += ' = 0.0'
            else:
                line = '  jac'
                if lang in ['c', 'cuda']:
                    i = k_sp + 1 + (num_s + 1) * (j_sp + 1)
                    line += utils.get_array(lang, i) + ' += '
                    sparse_indicies.append(i)
                elif lang in ['fortran', 'matlab']:
                    line += (utils.get_array(lang, k_sp + 1, twod=j_sp + 1) +
                             '= jac' + utils.get_array(lang, k_sp + 1, twod=j_sp + 1) + ' + ')
                    sparse_indicies.append((k_sp + 1, j_sp + 1))
                line += 'dy' + utils.get_array(lang, k_sp)
                line += (' * mw_avg / {:.8e}'.format(sp_j.mw) +
                         utils.line_end[lang]
                         )
                file.write(line)

                line = '  jac'
                if lang in ['c', 'cuda']:
                    line += utils.get_array(lang, k_sp +
                                            1 + (num_s + 1) * (j_sp + 1)) + ' *= '
                    sparse_indicies.append(k_sp + 1 + (num_s + 1) * (j_sp + 1))
                elif lang in ['fortran', 'matlab']:
                    line += (utils.get_array(lang, k_sp + 1, twod=j_sp + 1) + ' = jac' +
                             utils.get_array(lang, k_sp + 1, twod=j_sp + 1) + ' * ')
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
        line += ('(y' + utils.get_array(lang, isp + 1) + ' * ' + 'cp' +
                 utils.get_array(lang, isp) + ')')

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
        line += ('(h' + utils.get_array(lang, isp) + ' * dy' + utils.get_array(lang, isp) + ' * '
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
    line = '  jac'
    if lang in ['c', 'cuda']:
        line += utils.get_array(lang, 0) + ' = 0.0'
        sparse_indicies.append(0)
    elif lang == 'fortran':
        line += utils.get_array(lang, 0, twod=0) + ' = 0.0_wp'
        sparse_indicies.append((1, 1))
    elif lang == 'matlab':
        line += utils.get_array(lang, 0, twod=0) + ' = 0.0'
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

        line = '    jac' + utils.get_array(lang, 0, twod=0)

        if lang in ['c', 'cuda']:
            line += ' += '
            sparse_indicies.append(0)
        elif lang in ['fortran', 'matlab']:
            line += ' = jac' + utils.get_array(lang, 0, twod=0) + ' + '
            sparse_indicies.append((0, 0))
        line += 'y' + utils.get_array(lang, isp + 1)
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

        line = '    jac' + utils.get_array(lang, 0, twod=0)

        if lang in ['c', 'cuda']:
            line += ' += '
        elif lang in ['fortran', 'matlab']:
            line += ' = jac' + utils.get_array(lang, 0, twod=0) + ' + '
        line += 'y' + utils.get_array(lang, isp + 1)
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
        line += 'jac' + utils.get_array(lang, 0) + ' *= (-1.0'
        sparse_indicies.append(0)
    elif lang in ['fortran', 'matlab']:
        line += 'jac' + utils.get_array(lang, 0, twod=0) + ' = (-jac' + \
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
        line += ' + h' + \
            utils.get_array(
                lang, isp) + ' * dy' + utils.get_array(lang, isp) + ' * {:.8e}'.format(sp.mw)
        isfirst = False
    line += ')' + utils.line_end[lang]
    file.write(line)

    line = '  '
    if lang in ['c', 'cuda']:
        line += 'jac' + utils.get_array(lang, 0) + ' += ('
        sparse_indicies.append(0)
    elif lang in ['fortran', 'matlab']:
        line += 'jac' + \
            utils.get_array(
                lang, 0, twod=0) + ' = jac' + utils.get_array(lang, 0, twod=0) + ' + ('
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
        line += '(cp' + utils.get_array(lang, isp) + ' * dy' + utils.get_array(lang, isp) + \
            '{:.8e} / rho + '.format(sp.mw) + 'h' + utils.get_array(lang, isp) + ' * jac' + \
            utils.get_array(lang, isp + 1, twod=0) + ')'
        isfirst = False
    line += ')' + utils.line_end[lang]
    file.write(line)

    line = '  jac' + utils.get_array(lang, 0, twod=0)
    if lang in ['c', 'cuda']:
        line += ' /= '
        sparse_indicies.append(0)
    elif lang in ['fortran', 'matlab']:
        line += 'jac' + utils.get_array(lang, 0, twod=0) + ' / '
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

        line = '  jac'
        if lang in ['c', 'cuda']:
            line += utils.get_array(lang, (num_s + 1) * (isp + 1))
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
                line += ('h' + utils.get_array(lang, k_sp) + ' * ('
                         'jac' + utils.get_array(lang, k_sp + 1 + (num_s + 1) * (isp + 1)) +
                         ' - (cp' + utils.get_array(lang, isp) + ' * dy' +
                         utils.get_array(lang, k_sp) +
                         ' * {:.8e} / (rho * cp_avg)))'.format(sp_k.mw)
                         )
                sparse_indicies.append(k_sp + 1 + (num_s + 1) * (isp + 1))
            elif lang in ['fortran', 'matlab']:
                line += ('h' + utils.get_array(lang, k_sp) + ' * ('
                         'jac' + utils.get_array(lang, k_sp + 1, twod=isp + 1) +
                         ' - (cp' + utils.get_array(lang, isp) + ' * dy' +
                         utils.get_array(lang, k_sp)
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

    file.write('#include "sparse_multiplier.h"\n\n')

    if lang == 'cuda':
        file.write('__device__\n')

    file.write(
        "void sparse_multiplier(const Real * A, const Real * Vm, Real* w) {\n")

    for i in range(nvars):
        # get all indicies that belong to row i
        i_list = [x for x in sorted_and_cleaned if x % nvars == i]
        if not len(i_list):
            file.write(
                '  w' + utils.get_array(lang, i) + ' = 0' + utils.line_end[lang])
            continue
        file.write('  w' + utils.get_array(lang, i) + ' = ')
        for index in i_list:
            if i_list.index(index):
                file.write(" + ")
            file.write(' A' + utils.get_array(lang, index) + ' * '
                       'Vm' + utils.get_array(lang, int(index / nvars)))
        file.write(";\n")
    file.write("}\n")

    """
    if lang == 'cuda':
        file.write(
        '#ifdef COMPILE_TESTING_METHODS\n'
        '  __device__ bool test_sparse_multiplier() {\n'
        )
    else:
        file.write(
        '\n#ifdef COMPILE_TESTING_METHODS\n'
        '  int test_sparse_multiplier(){\n'
        )
    #write test method
    file.write(
        '    Real A[NN * NN] = {ZERO};\n'
        '    Real v[NN] = {ZERO};\n'
        '    //do not zero, tests sparse_multiplier\'s fill of non touched entries\n'
        '    Real w[NN];\n'
        '    Real w2[NN] = {ZERO};\n'
    )
    for i in range(nvars):
        #get all indicies that belong to row i
        i_list = [x for x in sorted_and_cleaned if x % nvars == i]
        file.write('    v[{:}] = {:};\n'.format(i, i))
        if not len(i_list):
            continue
        for index in i_list:
            file.write('    A[{:}] = {:};\n'.format(index, index))

    file.write(
        '    for (uint i = 0; i < NN; ++i) {\n'               
        '       for (uint j = 0; j < NN; ++j) {\n'
        '          w2[i] += A[i + (j * NN)] * v[j];\n'
        '       }\n'
        '    }\n'
        )
    file.write('    sparse_multiplier(A, v, w);\n')
    file.write('    for (uint i = 0; i < NN; ++i) {\n'
               '       if (w[i] != w2[i]) \n'
               '         return 0;\n'
               '    }\n'
               '    return 1;\n'
               '  }\n'
               '#endif\n'
        )"""
    file.close()


def create_jacobian(lang, mech_name, therm_name=None):
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

    # now begin writing subroutines

    # print reaction rate subroutine
    rate.write_rxn_rates(build_path, lang, specs, reacs)

    # if third-body/pressure-dependent reactions,
    # print modification subroutine
    if next((r for r in reacs if (r.thd or r.pdep)), None):
        rate.write_rxn_pressure_mod(build_path, lang, specs, reacs)

    # write species rates subroutine
    rate.write_spec_rates(build_path, lang, specs, reacs)

    # write chem_utils subroutines
    rate.write_chem_utils(build_path, lang, specs)

    # write derivative subroutines
    rate.write_derivs(build_path, lang, specs, reacs)

    # write mass-mole fraction conversion subroutine
    rate.write_mass_mole(build_path, lang, specs)

    # write mechanism initializers and testing methods
    aux.write_mechanism_initializers(build_path, lang, specs, reacs)

    # write Jacobian subroutine
    sparse_indicies = write_jacobian(build_path, lang, specs, reacs)

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

    args = parser.parse_args()

    create_jacobian(args.lang, args.input, args.thermo)
