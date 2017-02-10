#! /usr/bin/env python
"""Creates source code for calculating analytical Jacobian matrix.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import sys
import math
import os

# Local imports
from .. import utils
from . import chem_model as chem
from . import mech_interpret as mech
from . import rate_subs as rate
from . import mech_auxiliary as aux


def create_jacobian(lang, mech_name=None, therm_name=None, gas=None,
                    simd_width=4, build_path='./out/', last_spec=None,
                    skip_jac=False, auto_diff=False, platform=''
                    ):
    """Create Jacobian subroutine from mechanism.

    Parameters
    ----------
    lang : {'c', 'cuda', 'fortran', 'matlab'}
        Language type.
    mech_name : str, optional
        Reaction mechanism filename (e.g. 'mech.dat').
        This or gas must be specified
    therm_name : str, optional
        Thermodynamic database filename (e.g. 'therm.dat')
        or nothing if info in mechanism file.
    gas : cantera.Solution, optional
        The mechanism to generate the Jacobian for.  This or ``mech_name`` must be specified
    simd_width : int
        The SIMD vector width to use.  If the targeted platform is a GPU, this is the GPU block size
    build_path : str, optional
        The output directory for the jacobian files
    last_spec : str, optional
        If specified, the species to assign to the last index.
        Typically should be N2, Ar, He or another inert bath gas
    skip_jac : bool, optional
        If ``True``, only the reaction rate subroutines will be generated
    auto_diff : bool, optional
        If ``True``, generate files for use with the Adept autodifferention library.
    platform : str, optional
        If specified, generate code for this platform.  May be ['CPU', 'GPU'] or a specific vendor name,
        e.g. 'AMD'

    Returns
    -------
    None

    """
    if auto_diff or not skip_jac:
        raise NotImplementedException()


    if lang != 'c' and auto_diff:
        print('Error: autodifferention only supported for C')
        sys.exit(2)

    if auto_diff:
        skip_jac = True

    lang = lang.lower()
    if lang not in utils.langs:
        print('Error: language needs to be one of: ')
        for l in utils.langs:
            print(l)
        sys.exit(2)

    # create output directory if none exists
    utils.create_dir(build_path)

    if auto_diff:
        with open(os.path.join(build_path, 'ad_jacob.h'), 'w') as file:
            file.write('#ifndef AD_JAC_H\n'
                       '#define AD_JAC_H\n'
                       'void eval_jacob (const double t, const double pres, '
                       'const double* y, double* jac);\n'
                       '#endif\n'
                       )

    assert mech_name is not None or gas is not None, 'No mechanism specified!'

    # Interpret reaction mechanism file, depending on Cantera or
    # Chemkin format.
    if gas is not None or mech_name.endswith(tuple(['.cti', '.xml'])):
        elems, specs, reacs = mech.read_mech_ct(mech_name, gas)
    else:
        elems, specs, reacs = mech.read_mech(mech_name, therm_name)

    if not specs:
        print('No species found in file: {}'.format(mech_name))
        sys.exit(3)

    if not reacs:
        print('No reactions found in file: {}'.format(mech_name))
        sys.exit(3)

    #check to see if the last_spec is specified
    if last_spec is not None:
        #find the index if possible
        isp = next((i for i, sp in enumerate(specs)
                   if sp.name.lower() == last_spec.lower().strip()),
                   None
                   )
        if isp is None:
            print('Warning: User specified last species {} '
                  'not found in mechanism.'
                  '  Attempting to find a default species.'.format(last_spec)
                  )
            last_spec = None
        else:
            last_spec = isp
    else:
        print('User specified last species not found or not specified.  '
              'Attempting to find a default species')
    if last_spec is None:
        wt = chem.get_elem_wt()
        #check for N2, Ar, He, etc.
        candidates = [('N2', wt['n'] * 2.), ('Ar', wt['ar']),
                        ('He', wt['he'])]
        for sp in candidates:
            match = next((isp for isp, spec in enumerate(specs)
                          if sp[0].lower() == spec.name.lower() and
                          sp[1] == spec.mw),
                            None)
            if match is not None:
                last_spec = match
                break
        if last_spec is not None:
            print('Default last species '
                  '{} found.'.format(specs[last_spec].name)
                  )
    if last_spec is None:
        print('Warning: Neither a user specified or default last species '
              'could be found. Proceeding using the last species in the '
              'base mechanism: {}'.format(specs[-1].name))
        last_spec = len(specs) - 1

    #pick up the last_spec and drop it at the end
    temp = specs[:]
    specs[-1] = temp[last_spec]
    specs[last_spec] = temp[-1]


    the_len = len(reacs)

    #reassign the reaction's product / reactant / third body list
    # to integer indexes for speed
    utils.reassign_species_lists(reacs, specs)

    ## now begin writing subroutines

    # write species rates subroutine
    rate.write_specrates_kernel(path, eqs, reacs, specs, loopy_opts, test_size, auto_diff)

    # write chem_utils subroutines
    rate.write_chem_utils(build_path, lang, specs, auto_diff)

    # write derivative subroutines
    rate.write_derivs(build_path, lang, specs, reacs, seen_sp, auto_diff)

    # write mass-mole fraction conversion subroutine
    rate.write_mass_mole(build_path, lang, specs)

    # write header file
    aux.write_header(build_path, lang)

    # write mechanism initializers and testing methods
    aux.write_mechanism_initializers(build_path, lang, specs, reacs,
                                     fwd_spec_mapping, reverse_spec_mapping,
                                     initial_state, optimize_cache,
                                     last_spec, auto_diff
                                     )

    if skip_jac == False:
        # write Jacobian subroutine
        touched = write_jacobian(build_path, lang, specs,
                                         reacs, seen_sp, smm)

        write_sparse_multiplier(build_path, lang, touched, len(specs))

    return 0


if __name__ == "__main__":
    args = utils.get_parser()

    create_jacobian(lang=args.lang,
                    mech_name=args.input,
                    therm_name=args.thermo,
                    optimize_cache=args.cache_optimizer,
                    initial_state=args.initial_conditions,
                    num_blocks=args.num_blocks,
                    num_threads=args.num_threads,
                    no_shared=args.no_shared,
                    L1_preferred=args.L1_preferred,
                    multi_thread=args.multi_thread,
                    force_optimize=args.force_optimize,
                    build_path=args.build_path,
                    last_spec=args.last_species,
                    auto_diff=args.auto_diff
                    )
