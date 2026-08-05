"""Command-line interface for pyJac."""

import multiprocessing
import os
import sys
from argparse import ArgumentParser

from . import utils
from ._version import __version__
from .core.create_jacobian import create_jacobian


def build_parser():
    """Construct the pyJac argument parser.

    Returns
    -------
    parser : `argparse.ArgumentParser`
        Parser for the pyJac command line.

    """
    parser = ArgumentParser(
        prog='pyjac',
        description='pyJac: Generates source code for analytical chemical '
                    'Jacobians.'
    )
    parser.add_argument('-v', '--version',
                        action='version',
                        version=f'pyJac {__version__}',
                        help="Show pyJac's version number and exit."
                        )
    parser.add_argument('-l', '--lang',
                        type=str,
                        choices=utils.langs,
                        required=True,
                        help='Programming language for output source files. '
                             'Implemented: '
                             f'{", ".join(utils.supported_langs)}. The fortran '
                             'and matlab backends are incomplete and will '
                             'report an error.'
                        )
    parser.add_argument('-i', '--input',
                        type=str,
                        required=True,
                        help='Input mechanism filename (e.g., mech.dat).'
                        )
    parser.add_argument('-t', '--thermo',
                        type=str,
                        default=None,
                        help='Thermodynamic database filename (e.g., '
                             'therm.dat), or nothing if in mechanism.'
                        )
    parser.add_argument('-ic', '--initial-conditions',
                        type=str,
                        dest='initial_conditions',
                        default='',
                        required=False,
                        help='A comma separated list of initial initial '
                             'conditions to set in the '
                             'set_same_initial_conditions method.\n'
                             '   Expected Form: T,P,Species1=...,Species2=...,...\n'
                             '   Temperature in K\n'
                             '   Pressure in Atm\n'
                             '   Species in moles'
                        )
    # cuda specific
    parser.add_argument('-co', '--cache-optimizer',
                        dest='cache_optimizer',
                        action='store_true',
                        default=False,
                        help='Attempt to optimize cache store/loading '
                             'via use of a greedy selection algorithm. (Experimental)'
                        )
    parser.add_argument('-nosmem', '--no-shared-memory',
                        dest='no_shared',
                        action='store_true',
                        default=False,
                        help='Use this option to turn off attempted shared '
                             'memory acceleration for CUDA.'
                        )
    parser.add_argument('-pshare', '--prefer-shared',
                        dest='L1_preferred',
                        action='store_false',
                        default=True,
                        help='Use this option to allocate more space for '
                             'shared memory than the L1 cache for CUDA '
                             '(not recommended).'
                        )
    parser.add_argument('-nb', '--num-blocks',
                        type=int,
                        dest='num_blocks',
                        default=8,
                        required=False,
                        help='The target number of blocks / sm for CUDA.'
                        )
    parser.add_argument('-nt', '--num-threads',
                        type=int,
                        dest='num_threads',
                        default=64,
                        required=False,
                        help='The target number of threads / block for CUDA.'
                        )
    parser.add_argument('-mt', '--multi-threaded',
                        type=int,
                        dest='multi_thread',
                        default=multiprocessing.cpu_count(),
                        required=False,
                        help='The number of threads to use during the '
                             'optimization process.'
                        )
    parser.add_argument('-fopt', '--force-optimize',
                        dest='force_optimize',
                        action='store_true',
                        default=False,
                        help='Use this option to force a reoptimization of '
                             'the mechanism (usually only happens when '
                             'generating for a different mechanism).'
                        )
    parser.add_argument('-b', '--build_path',
                        required=False,
                        default='./out/',
                        help='The folder to generate the Jacobian and rate subroutines in.'
                        )
    parser.add_argument('-ls', '--last_species',
                        required=False,
                        type=str,
                        default=None,
                        help='The name of the species to set as the last in '
                             'the mechanism. If not specifed, defaults to '
                             'the first of N2, AR, and HE in the mechanism.'
                        )
    parser.add_argument('-ad', '--auto_diff',
                        default=False,
                        action='store_true',
                        help='Use this option to generate file for use with the '
                             'Adept autodifferentiation library.')
    parser.add_argument('-sj', '--skip_jac',
                        required=False,
                        default=False,
                        action='store_true',
                        help='If specified, this option turns off Jacobian generation '
                             '(only rate subs are generated)')
    return parser


def get_parser():
    """Parse and validate the pyJac command line.

    Validation failures exit through the parser, so the user gets a plain
    message and a conventional exit status instead of a traceback from deep
    inside generation.

    Returns
    -------
    args : `argparse.Namespace`
        Validated command line arguments for running pyJac.

    """
    parser = build_parser()
    args = parser.parse_args()

    if args.lang not in utils.supported_langs:
        parser.exit(2, f'{parser.prog}: {args.lang} output is not implemented. '
                       f'The {args.lang} backend was never completed and does '
                       'not produce usable source. Supported languages are: '
                       f'{", ".join(utils.supported_langs)}.\n')

    if not os.path.isfile(args.input):
        parser.exit(2, f'{parser.prog}: mechanism file not found: '
                       f'{args.input}\n')

    if args.thermo is not None and not os.path.isfile(args.thermo):
        parser.exit(2, f'{parser.prog}: thermodynamic database not found: '
                       f'{args.thermo}\n')

    return args


def main(args=None):
    """Run pyJac.

    Parameters
    ----------
    args : `argparse.Namespace`, optional
        Parsed arguments. If omitted, they are read from the command line.

    Returns
    -------
    int
        Process exit status.

    """
    if args is None:
        args = get_parser()
    try:
        create_jacobian(
                    lang=args.lang,
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
                    skip_jac=args.skip_jac,
                    last_spec=args.last_species,
                    auto_diff=args.auto_diff
                    )
    except NotImplementedError as err:
        print(f'Error: {err}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
