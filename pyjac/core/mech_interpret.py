# -*- coding: utf-8 -*-
"""Chemkin-format mechanism interpreter module.
"""

# Python 2 compatibility
from __future__ import division

# Standard libraries
import sys
import math
import re
from copy import deepcopy
import logging

import numpy as np

# Local imports
from .. import utils
from . import chem_utilities as chem

# Related module
CANTERA_FLAG = False
try:
    import cantera as ct
    version = ct.__version__.split('.')
    if int(version[0]) < 2 or int(version[1]) < 3:
        print('Parsing of Cantera mechanisms requires at least version 2.3.0 in order to access species thermo properties...')
        print('Detected version is only {}'.format(ct.__version__))
        sys.exit(1)
    CANTERA_FLAG = True
except ImportError:
    CANTERA_FLAG = False

pre_units = ['moles', 'molecules']
"""list(`str`): Supported units list for pre-exponential factor"""

act_energy_units = ['kelvins', 'evolts', 'cal/mole', 'joules/kmole',
                    'kcal/mole', 'joules/mole', 'kjoules/mole'
                    ]
"""list(`str`): Supported units list for activation energy"""

act_energy_fact = dict({'kelvins': 1.0,
                        'evolts': 11595.,
                        'cal/mole': 4.184 / chem.RU_JOUL,
                        'kcal/mole': 4184. / chem.RU_JOUL,
                        'joules/mole': 1. / chem.RU_JOUL,
                        'kjoules/mole': 1000.0 / chem.RU_JOUL,
                        'joules/kmole': 1. / (chem.RU_JOUL * 1000.)
                        })
"""dict: Activation energy conversion factor"""

# get local element atomic weight dict
elem_wt = chem.get_elem_wt()


def read_mech(mech_filename, therm_filename):
    """Read and interpret mechanism file for elements, species, and reactions.

    Parameters
    ----------
    mech_filename : str
        Reaction mechanism filename (e.g. 'mech.dat')
    therm_filename : str, optional
        Thermodynamic database filename (e.g., 'therm.dat')

    Returns
    -------
    elems : list of str
        List of elements in mechanism.
    specs : list of `SpecInfo`
        List of species in mechanism.
    reacs : list of `ReacInfo`
        List of reactions in mechanism.
    units : str
        Units of reactions' Arrhenius coefficients

    Notes
    -----
    Doesn't support element names with digits.

    """

    elems = []
    reacs = []
    specs = []

    units = ''
    key = ''

    # By default, need to read thermo database if file given.
    if therm_filename:
        therm_flag = True

    with open(mech_filename, 'r') as file:
        # start line reading loop
        while True:
            # remember last line position
            last_line = file.tell()

            line = file.readline()

            # end of file
            if not line: break

            # skip blank or commented lines
            if re.search('^\s*$', line) or re.search('^\s*!', line): continue

            # don't convert to lowercase, since thermo
            # needs to match (for Chemkin)

            # Remove trailing and leading whitespace, tabs, newline
            line = line.strip()

            # remove any comments from end of line
            ind = line.find('!')
            if ind > 0: line = line[0:ind]

            # now determine key
            if line[0:4].lower() == 'elem':
                key = 'elem'

                # check for any entries on this line
                line_split = line.split()
                if len(line_split) > 1:
                    ind = line.index(line_split[1])
                    line = line[ind:]
                else:
                    continue

            elif line[0:4].lower() == 'spec':
                key = 'spec'

                # check for any entries on this line
                line_split = line.split()
                if len(line_split) > 1:
                    ind = line.index(line_split[1])
                    line = line[ind:]
                else:
                    continue

            elif line[0:4].lower() == 'reac':
                key = 'reac'

                # default units from Chemkin
                units_E = 'cal/mole'
                units_A = 'moles'

                # get Arrhenius coefficient units (if specified)
                for unit in line.split()[1:]:
                    if unit.lower() in pre_units:
                        units_A = unit.lower()
                    elif unit.lower() in act_energy_units:
                        units_E = unit.lower()
                    else:
                        print('Error: unsupported units on REACTION line.')
                        print('For pre-exponential factor, choose from: ' +
                              pre_units
                              )
                        print('For activation energy, choose from: ' +
                              act_energy_units
                              )
                        print('Otherwise leave blank for moles and cal/mole.')
                        sys.exit(1)

                if units_A == 'molecules':
                    raise NotImplementedError('Molecules units not '
                                              'supported, sorry.'
                                              )
                continue
            elif line[0:4].lower() == 'ther':
                # thermo data is in mechanism file
                read_thermo(mech_filename, elems, specs)
                continue
            elif line[0:3].lower() == 'end':
                key = ''
                continue

            if key == 'elem':
                # if any atomic weight declarations, replace / with spaces
                line = line.replace('/', ' ')

                line_split = line.split()
                e_last = ''
                for e in line_split:
                    if e.isalpha():
                        if e[0:3] == 'end': continue
                        if e not in elems:
                            elems.append(e)
                        e_last = e
                    else:
                        # either add new element or update existing
                        # atomic weight
                        elem_wt[e_last.lower()] = float(e)

            elif key == 'spec':
                line_split = line.split()
                for s in line_split:
                    if s[0:3] == 'end': continue
                    if not next((sp for sp in specs if sp.name == s), None):
                        specs.append(chem.SpecInfo(s))

            elif key == 'reac':
                # determine if reaction or auxiliary info line

                if '=' in line:
                    # new reaction

                    cheb_flag = False

                    # get Arrhenius coefficients
                    line_split = line.split()
                    n = len(line_split)
                    reac_A = float(line_split[n - 3])
                    reac_b = float(line_split[n - 2])
                    reac_E = float(line_split[n - 1])

                    ind = line.index(line_split[n - 3])
                    line = line[0:ind].strip()

                    if '<=>' in line:
                        ind = line.index('<=>')
                        reac_rev = True
                        reac_str = line[0:ind].strip()
                        prod_str = line[ind + 3:].strip()
                    elif '=>' in line:
                        ind = line.index('=>')
                        reac_rev = False
                        reac_str = line[0:ind].strip()
                        prod_str = line[ind + 2:].strip()
                    else:
                        ind = line.index('=')
                        reac_rev = True
                        reac_str = line[0:ind].strip()
                        prod_str = line[ind + 1:].strip()

                    thd = False
                    pdep = False
                    pdep_sp = ''

                    reac_spec = []
                    reac_nu = []
                    prod_spec = []
                    prod_nu = []

                    # reactants

                    # look for third-body species
                    sub_str = reac_str
                    while '(' in sub_str:
                        ind1 = sub_str.find('(')
                        ind2 = sub_str.find(')')

                        # Need to check if '+' is first character inside
                        # parentheses and not embedded within parentheses
                        # (e.g., '(+)').
                        # If not, part of species name.
                        inParen = sub_str[ind1 + 1: ind2].strip()
                        if inParen is '+':
                            # '+' embedded within parentheses
                            sub_str = sub_str[ind2 + 1:]
                        elif inParen[0] is '+':
                            pdep = True

                            # either 'm' or a specific species
                            pdep_sp = sub_str[ind1 + 1: ind2].replace('+', ' ')
                            pdep_sp = pdep_sp.strip()

                            if pdep_sp.lower() == 'm':
                                thd = True
                                pdep_sp = ''

                            # now remove from string
                            ind = reac_str.find(sub_str)
                            reac_str = (reac_str[0: ind1 + ind] +
                                        reac_str[ind2 + ind + 1:]
                                        )
                            break
                        else:
                            # Part of species name, remove from substring
                            # and look at rest of reactant line.
                            sub_str = sub_str[ind2 + 1:]

                    reac_list = reac_str.split('+')

                    # Check for empty list elements, meaning there were
                    # multiple '+' in a row, which indicates species'
                    # name ended in '+'.
                    while '' in reac_list:
                        ind = reac_list.index('')
                        reac_list[ind - 1] = reac_list[ind - 1] + '+'
                        del reac_list[ind]

                    # check for any species with '(+)' that was split apart
                    for sp in reac_list:
                        ind = reac_list.index(sp)

                        # ensure not last entry
                        if (ind < len(reac_list) - 1):
                            spNext = reac_list[ind + 1]
                            if sp[len(sp) - 1] is '(' and spNext[0] is ')':
                                reac_list[ind] = sp + '+' + spNext
                                del reac_list[ind + 1]

                    for sp in reac_list:

                        sp = sp.strip()

                        # look for coefficient
                        if sp[0:1].isdigit():
                            # starts with number (coefficient)

                            # search for first letter
                            for i in range(len(sp)):
                                if sp[i: i + 1].isalpha(): break

                            nu = sp[0:i]
                            if '.' in nu:
                                # float
                                nu = float(nu)
                            else:
                                # integer
                                nu = int(nu)

                            sp = sp[i:].strip()
                        else:
                            # no coefficient given
                            nu = 1

                        # check for third body
                        if sp.lower() == 'm':
                            thd = True
                            continue

                        # check if species already in reaction
                        if sp not in reac_spec:
                            # new reactant
                            reac_spec.append(sp)
                            reac_nu.append(nu)
                        else:
                            # existing reactant
                            i = reac_spec.index(sp)
                            reac_nu[i] += nu

                    # products

                    # look for third-body species
                    sub_str = prod_str
                    while '(' in sub_str:
                        ind1 = sub_str.find('(')
                        ind2 = sub_str.find(')')

                        # Need to check if '+' is first character inside
                        # parentheses and not embedded within parentheses
                        # (e.g., '(+)'). If not, part of species name.
                        inParen = sub_str[ind1 + 1: ind2].strip()
                        if inParen is '+':
                            # '+' embedded within parentheses
                            sub_str = sub_str[ind2 + 1:]
                        elif inParen[0] is '+':
                            pdep = True

                            # either 'm' or a specific species
                            pdep_sp = sub_str[ind1 + 1: ind2].replace('+', ' ')
                            pdep_sp = pdep_sp.strip()

                            if pdep_sp.lower() == 'm':
                                thd = True
                                pdep_sp = ''

                            # now remove from string
                            ind = prod_str.find(sub_str)
                            prod_str = (prod_str[0: ind1 + ind] +
                                        prod_str[ind2 + ind + 1:]
                                        )
                            break
                        else:
                            # Part of species name, remove from substring and
                            # look at rest of product line.
                            sub_str = sub_str[ind2 + 1:]

                    prod_list = prod_str.split('+')

                    # Check for empty list elements, meaning there were
                    # multiple '+' in a row, which indicates species
                    # name ended in '+'.
                    while '' in prod_list:
                        ind = prod_list.index('')
                        prod_list[ind - 1] = prod_list[ind - 1] + '+'
                        del prod_list[ind]

                    # check for any species with '(+)' that was split apart
                    for sp in prod_list:
                        ind = prod_list.index(sp)

                        # ensure not last entry
                        if (ind < len(prod_list) - 1):
                            spNext = prod_list[ind + 1]
                            if sp[len(sp) - 1] is '(' and spNext[0] is ')':
                                prod_list[ind] = sp + '+' + spNext
                                del prod_list[ind + 1]

                    for sp in prod_list:

                        sp = sp.strip()

                        # look for coefficient
                        if sp[0:1].isdigit():
                            # starts with number (coefficient)

                            # search for first letter
                            for i in range(len(sp)):
                                if sp[i: i + 1].isalpha(): break

                            nu = sp[0:i]
                            if '.' in nu:
                                # float
                                nu = float(nu)
                            else:
                                # integer
                                nu = int(nu)

                            sp = sp[i:].strip()
                        else:
                            # no coefficient given
                            nu = 1

                        # check for third body
                        if sp in ['m', 'M']:
                            thd = True
                            continue

                        # check if species already in reaction
                        if sp not in prod_spec:
                            # new product
                            prod_spec.append(sp)
                            prod_nu.append(nu)
                        else:
                            # existing product
                            i = prod_spec.index(sp)
                            prod_nu[i] += nu

                    # Don't want to confuse third-body and pressure-dependent
                    # reactions... they are different!
                    if pdep:
                        thd = False

                    # Convert given activation energy units to internal units
                    reac_E *= act_energy_fact[units_E]

                    # Convert given pre-exponential units to internal units
                    if units_A == 'moles':
                        reac_ord = sum(reac_nu)
                        if thd:
                            reac_A /= 1000. ** reac_ord
                        elif pdep:
                            # Low- (chemically activated bimolecular reaction) or
                            # high-pressure (fall-off reaction) limit parameters
                            reac_A /= 1000. ** (reac_ord - 1.)
                        else:
                            # Elementary reaction
                            reac_A /= 1000. ** (reac_ord - 1.)

                    # add reaction to list
                    reac = chem.ReacInfo(reac_rev, reac_spec, reac_nu,
                                         prod_spec, prod_nu, reac_A, reac_b,
                                         reac_E
                                         )
                    reac.thd_body = thd
                    reac.pdep = pdep
                    if pdep: reac.pdep_sp = pdep_sp

                    reacs.append(reac)

                else:
                    # auxiliary reaction info

                    aux = line[0:3].lower()
                    if aux == 'dup':
                        reacs[-1].dup = True

                    elif aux == 'rev':
                        line = line.replace('/', ' ')
                        line = line.replace(',', ' ')
                        line_split = line.split()
                        par1 = float(line_split[1])
                        par2 = float(line_split[2])
                        par3 = float(line_split[3])

                        # Convert reverse activation energy units
                        par3 *= act_energy_fact[units_E]

                        # Convert reverse pre-exponential factor
                        if units_A == 'moles':
                            reac_ord = sum(reacs[-1].prod_nu)
                            if thd:
                                par1 /= 1000. ** reac_ord
                            elif pdep:
                                # Low- (chemically activated bimolecular reaction) or
                                # high-pressure (fall-off reaction) limit parameters
                                par1 /= 1000. ** (reac_ord - 1.)
                            else:
                                # Elementary reaction
                                par1 /= 1000. ** (reac_ord - 1.)

                        # Ensure nonzero reverse coefficients
                        if par1 != 0.0:
                            reacs[-1].rev_par.append(par1)
                            reacs[-1].rev_par.append(par2)
                            reacs[-1].rev_par.append(par3)
                        else:
                            reacs[-1].rev = False

                    elif aux == 'low':
                        line = line.replace('/', ' ')
                        line = line.replace(',', ' ')
                        line_split = line.split()
                        par1 = float(line_split[1])
                        par2 = float(line_split[2])
                        par3 = float(line_split[3])

                        # Convert low-pressure activation energy units
                        par3 *= act_energy_fact[units_E]

                        # Convert low-pressure pre-exponential factor
                        if units_A == 'moles':
                            par1 /= 1000. ** sum(reacs[-1].reac_nu)

                        reacs[-1].low.append(par1)
                        reacs[-1].low.append(par2)
                        reacs[-1].low.append(par3)

                    elif aux == 'hig':
                        line = line.replace('/', ' ')
                        line = line.replace(',', ' ')
                        line_split = line.split()
                        par1 = float(line_split[1])
                        par2 = float(line_split[2])
                        par3 = float(line_split[3])

                        # Convert high-pressure activation energy units
                        par3 *= act_energy_fact[units_E]

                        # Convert high-pressure pre-exponential factor
                        if units_A == 'moles':
                            par1 /= 1000. ** (sum(reacs[-1].reac_nu) - 2.)

                        reacs[-1].high.append(par1)
                        reacs[-1].high.append(par2)
                        reacs[-1].high.append(par3)

                    elif aux == 'tro':
                        line = line.replace('/', ' ')
                        line = line.replace(',', ' ')
                        line_split = line.split()
                        reacs[-1].troe = True
                        par1 = float(line_split[1])
                        par2 = float(line_split[2])
                        par3 = float(line_split[3])

                        do_warn = False
                        if par2 == 0:
                            do_warn=True
                            par2 = 1e-30
                        if par3 == 0:
                            do_warn=True
                            par3 = 1e-30
                        if do_warn:
                            logging.warn('Troe parameters in reaction {} modified to avoid'
                                ' division by zero!.'.format(len(reacs)))

                        reacs[-1].troe_par.append(par1)
                        reacs[-1].troe_par.append(par2)
                        reacs[-1].troe_par.append(par3)

                        # optional fourth parameter
                        if len(line_split) > 4:
                            par4 = float(line_split[4])
                            reacs[-1].troe_par.append(par4)

                    elif aux == 'sri':
                        line = line.replace('/', ' ')
                        line = line.replace(',', ' ')
                        line_split = line.split()
                        reacs[-1].sri = True
                        par1 = float(line_split[1])
                        par2 = float(line_split[2])
                        par3 = float(line_split[3])
                        reacs[-1].sri_par.append(par1)
                        reacs[-1].sri_par.append(par2)
                        reacs[-1].sri_par.append(par3)

                        # optional fourth and fifth parameters
                        if len(line_split) > 4:
                            par4 = float(line_split[4])
                            par5 = float(line_split[5])
                            reacs[-1].sri_par.append(par4)
                            reacs[-1].sri_par.append(par5)
                    elif aux == 'che':
                        reacs[-1].cheb = True
                        line = line.replace('/', ' ')
                        line_split = line.split()
                        if cheb_flag:
                            for par in line_split[1:]:
                                reacs[-1].cheb_par.append(float(par))
                        else:
                            # first CHEB line
                            cheb_flag = True
                            # Don't want Cheb reactions lumped in with
                            # standard falloff.
                            reacs[-1].pdep = False
                            reacs[-1].cheb_n_temp = int(line_split[1])
                            reacs[-1].cheb_n_pres = int(line_split[2])
                            reacs[-1].cheb_par = []
                            for par in line_split[3:]:
                                reacs[-1].cheb_par.append(float(par))
                    elif aux == 'pch':
                        line = line.replace('/', ' ')
                        line_split = line.split()
                        # Convert pressure from atm to Pa
                        reacs[-1].cheb_plim = [float(line_split[1]) * chem.PA,
                                               float(line_split[2]) * chem.PA
                                               ]

                        # Look for temperature limits on same line:
                        if line_split[3].lower() == 'tcheb':
                            reacs[-1].cheb_tlim = [float(line_split[4]),
                                                   float(line_split[5])
                                                   ]
                    elif aux == 'tch':
                        line = line.replace('/', ' ')
                        line_split = line.split()
                        reacs[-1].cheb_tlim = [float(line_split[1]),
                                               float(line_split[2])
                                               ]
                        # Look for pressure limits on same line:
                        if line_split[3].lower() == 'pcheb':
                            reacs[-1].cheb_plim = [float(line_split[4]) * chem.PA,
                                                   float(line_split[5]) * chem.PA
                                                   ]
                    elif aux == 'plo':
                        line = line.replace('/', ' ')
                        line_split = line.split()
                        if not reacs[-1].plog:
                            reacs[-1].plog = True
                            # Don't want Plog reactions lumped in with
                            # standard falloff.
                            reacs[-1].pdep = False
                            reacs[-1].plog_par = []
                        pars = [float(n) for n in line_split[1:5]]

                        # Convert pressure from atm to Pa
                        pars[0] *= 101325.0

                        # Convert given activation energy units to internal units
                        pars[3] *= act_energy_fact[units_E]

                        # Convert given pre-exponential units to internal units
                        if units_A == 'moles':
                            reac_ord = sum(reacs[-1].reac_nu)
                            # Looks like elementary reaction
                            pars[1] /= 1000. ** (reac_ord - 1.)

                        reacs[-1].plog_par.append(pars)
                    else:
                        # enhanced third body efficiencies
                        line = line.replace('/', ' ')
                        line_split = line.split()
                        for i in range(0, len(line_split), 2):
                            pair = [line_split[i], float(line_split[i + 1])]
                            reacs[-1].thd_body_eff.append(pair)

    # process some reaction auxiliary info
    for idx, reac in enumerate(reacs):
        if reac.cheb:
            # check for correct number
            n = reac.cheb_n_temp
            m = reac.cheb_n_pres
            if len(reac.cheb_par) != n * m:
                print('Error: incorrect number of CHEB coefficients in '
                      'reaction ' + repr(idx)
                      )
                sys.exit(1)
            else:
                # Convert units of first Chebyshev parameter
                order = sum(reac.reac_nu)
                if units_A == 'moles':
                    reac.cheb_par[0] += math.log10(0.001 ** (order - 1.))

                reacs[idx].cheb_par = np.reshape(reac.cheb_par, (n, m))

    # Split reversible reactions with explicit reverse parameters into
    # two irreversible reactions to match Cantera's behavior
    for reac in reacs[:]:
        if reac.rev_par:
            new_reac = deepcopy(reac)

            idx = reacs.index(reac)
            reacs[idx].rev = False
            reacs[idx].rev_par = []

            new_reac.A = new_reac.rev_par[0]
            new_reac.b = new_reac.rev_par[1]
            new_reac.E = new_reac.rev_par[2]
            new_reac.rev = False
            new_reac.rev_par = []
            new_reac.prod = reac.reac[:]
            new_reac.prod_nu = reac.reac_nu[:]
            new_reac.reac = reac.prod[:]
            new_reac.reac_nu = reac.prod_nu[:]

            reacs.insert(idx + 1, new_reac)

    # Read seperate thermo file if present and needed
    if any([not sp.mw for sp in specs]):
        if therm_filename:
            read_thermo(therm_filename, elems, specs)
        else:
            print('Error: no thermo file specified, but species missing \n'
                  'data. Either specify file, or ensure complete data in\n'
                  'mechanism file with THERMO option.'
                  )
            sys.exit(1)

    # Check for missing thermo data again
    missing_mw = [sp.name for sp in specs if not sp.mw]
    if missing_mw:
        print('Error: missing thermo data for ' + ', '.join(missing_mw))
        sys.exit(1)

    return (elems, specs, reacs)


def read_thermo(filename, elems, specs):
    """Read and interpret thermodynamic database for species data.

    Reads the thermodynamic file and returns the species thermodynamic
    coefficients as well as the species-specific temperature range
    values (if given).

    Parameters
    ----------
    filename : str
        Name of thermo database file.
    elems : list of str
        List of element names in mechanism.
    specs : list of `SpecInfo`
        List of species in mechanism.

    Returns
    -------
    None

    """

    with open(filename, 'r') as file:

        # loop through intro lines
        while True:
            line = file.readline()

            # skip blank or commented lines
            if re.search('^\s*$', line) or re.search('^\s*!', line): continue

            # skip 'thermo' at beginning
            if 'thermo' in line.lower(): break

        # next line either has common temperature ranges or first species
        last_line = file.tell()
        line = file.readline()

        line_split = line.split()
        if line_split[0][0:1].isdigit():
            T_ranges = utils.read_str_num(line)
        else:
            # no common temperature info
            file.seek(last_line)
            # default

        # now start reading species thermo info
        while True:
            # first line of species info
            line = file.readline()

            # don't convert to lowercase, needs to match thermo for Chemkin

            # break if end of file
            if line is None or line[0:3].lower() == 'end': break

            # skip blank/commented line
            if re.search('^\s*$', line) or re.search('^\s*!', line): continue

            # species name, columns 0:18
            spec = line[0:18].strip()

            # Apparently, in some cases, notes are in the
            # columns of shorter species names, so make
            # sure no spaces.
            if spec.find(' ') > 0:
                spec = spec[0: spec.find(' ')]

            # now need to determine if this species is in mechanism
            if next((sp for sp in specs if sp.name == spec), None):
                sp_ind = next(i for i in range(len(specs))
                              if specs[i].name == spec
                              )
            else:
                # not in mechanism, read next three lines and continue
                line = file.readline()
                line = file.readline()
                line = file.readline()
                continue

            # set species to the one matched
            spec = specs[sp_ind]

            # ensure not reading the same species more than once...
            if spec.mw:
                # already done! skip next three lines
                line = file.readline()
                line = file.readline()
                line = file.readline()
                continue

            # now get element composition of species, columns 24:44
            # each piece of data is 5 characters long (2 for element, 3 for #)
            elem_str = utils.split_str(line[24:44], 5)

            for e_str in elem_str:
                e = e_str[0:2].strip()
                # skip if blank
                if e == '' or e == '0': continue
                # may need to convert to float first, in case of e.g. "1."
                e_num = float(e_str[2:].strip())
                e_num = int(e_num)

                spec.elem.append([e, e_num])

                # calculate molecular weight
                spec.mw += e_num * elem_wt[e.lower()]

            # temperatures for species
            T_spec = utils.read_str_num(line[45:73])
            T_low = T_spec[0]
            T_high = T_spec[1]
            if len(T_spec) == 3:
                T_com = T_spec[2]
            else:
                T_com = T_ranges[1]

            spec.Trange = [T_low, T_com, T_high]

            # second species line
            line = file.readline()
            coeffs = utils.split_str(line[0:75], 15)
            spec.hi[0] = float(coeffs[0])
            spec.hi[1] = float(coeffs[1])
            spec.hi[2] = float(coeffs[2])
            spec.hi[3] = float(coeffs[3])
            spec.hi[4] = float(coeffs[4])

            # third species line
            line = file.readline()
            coeffs = utils.split_str(line[0:75], 15)
            spec.hi[5] = float(coeffs[0])
            spec.hi[6] = float(coeffs[1])
            spec.lo[0] = float(coeffs[2])
            spec.lo[1] = float(coeffs[3])
            spec.lo[2] = float(coeffs[4])

            # fourth species line
            line = file.readline()
            coeffs = utils.split_str(line[0:75], 15)
            spec.lo[3] = float(coeffs[0])
            spec.lo[4] = float(coeffs[1])
            spec.lo[5] = float(coeffs[2])
            spec.lo[6] = float(coeffs[3])

            # stop reading if all species in mechanism accounted for
            if not next((sp for sp in specs if sp.mw == 0.0), None): break

    return None


def read_mech_ct(filename=None, gas=None):
    """Read and interpret Cantera-format mechanism file.

    Parameters
    ----------
    filename : str
        Reaction mechanism filename (e.g. 'mech.cti'). Optional.
    gas : `cantera.Solution` object
        Existing Cantera Solution object to be used. Optional.

    Returns
    -------
    elems : list of str
        List of elements in mechanism.
    specs : list of `SpecInfo`
        List of species in mechanism.
    reacs : list of `ReacInfo`
        List of reactions in mechanism.
    units : str
        Units of reactions' Arrhenius coefficients

    """

    if not CANTERA_FLAG:
        print('Error: Cantera not installed. Cannot interpret '
              'Cantera-format mechanism.')
        sys.exit(1)

    if filename:
        gas = ct.Solution(filename)
    elif not gas:
        print('Error: need either filename or Cantera Solution object.')
        sys.exit(1)

    # Elements
    elems = gas.element_names
    for e, wt in zip(elems, gas.atomic_weights):
        if e.lower() not in elem_wt:
            elem_wt[e.lower()] = wt

    # Species
    specs = []
    for i, sp in enumerate(gas.species_names):
        spec = chem.SpecInfo(sp)

        spec.mw = gas.molecular_weights[i]

        # Get Species object
        species = gas.species(i)

        # Species elemental composition
        for e in species.composition:
            spec.elem.append([e, species.composition[e]])

        # Species thermodynamic properties
        coeffs = species.thermo.coeffs
        spec.Trange = [species.thermo.min_temp, coeffs[0],
                       species.thermo.max_temp
                       ]
        if isinstance(species.thermo, ct.NasaPoly2):
            spec.hi = coeffs[1:8]
            spec.lo = coeffs[8:15]
        else:
            print('Error: unsupported thermo form for species ' + sp)
            sys.exit(1)

        specs.append(spec)

    # Reactions
    reacs = []

    # Cantera internally uses joules/kmol for activation energy
    E_fac = act_energy_fact['joules/kmole']

    for rxn in gas.reactions():

        if isinstance(rxn, ct.ThreeBodyReaction):
            # Instantiate internal reaction based on Cantera Reaction data.
            reac = chem.ReacInfo(rxn.reversible,
                                 list(rxn.reactants.keys()),
                                 list(rxn.reactants.values()),
                                 list(rxn.products.keys()),
                                 list(rxn.products.values()),
                                 rxn.rate.pre_exponential_factor,
                                 rxn.rate.temperature_exponent,
                                 rxn.rate.activation_energy * E_fac
                                 )
            reac.thd_body = True
            for thd_body in rxn.efficiencies:
                reac.thd_body_eff.append([thd_body,
                                          rxn.efficiencies[thd_body]
                                          ])

        elif isinstance(rxn, ct.FalloffReaction) and \
             not isinstance(rxn, ct.ChemicallyActivatedReaction):
            reac = chem.ReacInfo(rxn.reversible,
                                 list(rxn.reactants.keys()),
                                 list(rxn.reactants.values()),
                                 list(rxn.products.keys()),
                                 list(rxn.products.values()),
                                 rxn.high_rate.pre_exponential_factor,
                                 rxn.high_rate.temperature_exponent,
                                 rxn.high_rate.activation_energy * E_fac
                                 )
            reac.pdep = True
            # See if single species acts as third body
            if rxn.default_efficiency == 0.0:
                reac.pdep_sp = list(rxn.efficiencies.keys())[0]
            else:
                for sp in rxn.efficiencies:
                    reac.thd_body_eff.append([sp, rxn.efficiencies[sp]])

            reac.low = [rxn.low_rate.pre_exponential_factor,
                        rxn.low_rate.temperature_exponent,
                        rxn.low_rate.activation_energy * E_fac
                        ]

            if rxn.falloff.type == 'Troe':
                reac.troe = True
                reac.troe_par = rxn.falloff.parameters.tolist()
                do_warn = False
                if reac.troe_par[1] == 0:
                    reac.troe_par[1] = 1e-30
                    do_warn = True
                if reac.troe_par[2] == 0:
                    reac.troe_par[2] = 1e-30
                    do_warn = True
                if do_warn:
                    logging.warn('Troe parameters in reaction {} modified to avoid'
                                    ' division by zero!.'.format(len(reacs)))
            elif rxn.falloff.type == 'SRI':
                reac.sri = True
                reac.sri_par = rxn.falloff.parameters.tolist()

        elif isinstance(rxn, ct.ChemicallyActivatedReaction):
            reac = chem.ReacInfo(rxn.reversible,
                                 list(rxn.reactants.keys()),
                                 list(rxn.reactants.values()),
                                 list(rxn.products.keys()),
                                 list(rxn.products.values()),
                                 rxn.low_rate.pre_exponential_factor,
                                 rxn.low_rate.temperature_exponent,
                                 rxn.low_rate.activation_energy * E_fac
                                 )
            reac.pdep = True
            # See if single species acts as third body
            if rxn.default_efficiency == 0.0:
                reac.pdep_sp = list(rxn.efficiencies.keys())[0]
            else:
                for sp in rxn.efficiencies:
                    reac.thd_body_eff.append([sp, rxn.efficiencies[sp]])

            reac.high = [rxn.high_rate.pre_exponential_factor,
                         rxn.high_rate.temperature_exponent,
                         rxn.high_rate.activation_energy * E_fac
                         ]

            if rxn.falloff.type == 'Troe':
                reac.troe = True
                reac.troe_par = rxn.falloff.parameters.tolist()
            elif rxn.falloff.type == 'SRI':
                reac.sri = True
                reac.sri_par = rxn.falloff.parameters.tolist()

        elif isinstance(rxn, ct.PlogReaction):
            reac = chem.ReacInfo(rxn.reversible,
                                 list(rxn.reactants.keys()),
                                 list(rxn.reactants.values()),
                                 list(rxn.products.keys()),
                                 list(rxn.products.values()),
                                 0.0, 0.0, 0.0
                                 )
            reac.plog = True
            reac.plog_par = []
            for rate in rxn.rates:
                pars = [rate[0], rate[1].pre_exponential_factor,
                        rate[1].temperature_exponent,
                        rate[1].activation_energy * E_fac
                        ]
                reac.plog_par.append(pars)

        elif isinstance(rxn, ct.ChebyshevReaction):
            reac = chem.ReacInfo(rxn.reversible,
                                 list(rxn.reactants.keys()),
                                 list(rxn.reactants.values()),
                                 list(rxn.products.keys()),
                                 list(rxn.products.values()),
                                 0.0, 0.0, 0.0
                                 )
            reac.cheb = True
            reac.cheb_n_temp = rxn.nTemperature
            reac.cheb_n_pres = rxn.nPressure
            reac.cheb_plim = [rxn.Pmin, rxn.Pmax]
            reac.cheb_tlim = [rxn.Tmin, rxn.Tmax]
            reac.cheb_par = rxn.coeffs

        elif isinstance(rxn, ct.ElementaryReaction):
            # Instantiate internal reaction based on Cantera Reaction data.

            # Ensure no reactions with zero pre-exponential factor allowed
            if rxn.rate.pre_exponential_factor == 0.0:
                continue

            reac = chem.ReacInfo(rxn.reversible,
                                 list(rxn.reactants.keys()),
                                 list(rxn.reactants.values()),
                                 list(rxn.products.keys()),
                                 list(rxn.products.values()),
                                 rxn.rate.pre_exponential_factor,
                                 rxn.rate.temperature_exponent,
                                 rxn.rate.activation_energy * E_fac
                                 )

        else:
            print('Error: unsupported reaction.')
            sys.exit(1)

        reac.dup = rxn.duplicate

        # No reverse reactions with explicit coefficients in Cantera.

        reacs.append(reac)

    return (elems, specs, reacs)
