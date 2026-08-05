from argparse import ArgumentParser
from . import test
from .. import utils
import os

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Tests pyJac versus a finite difference'
                    ' Cantera jacobian\n'
        )
    parser.add_argument('-l', '--lang',
                        type=str,
                        choices=utils.langs,
                        required=True,
                        help='Programming language for output '
                             'source files'
                        )
    parser.add_argument('-b', '--build_dir',
                        type=str,
                        default='out' + os.path.sep,
                        help='The directory the jacob/rates/tester'
                             ' files will be generated and built in'
                        )
    parser.add_argument('-m', '--mech',
                        type=str,
                        required=True,
                        help='Mechanism filename (e.g., mech.dat)'
                        )
    parser.add_argument('-t', '--thermo',
                        type=str,
                        default=None,
                        help='Thermodynamic database filename (e.g., '
                             'therm.dat), or nothing if in mechanism'
                        )
    parser.add_argument('-i', '--input',
                        type=str,
                        default='pasr_input.yaml',
                        help='Partially stirred reactor input file for '
                             'generating test data (e.g, pasr_input.yaml)'
                        )
    parser.add_argument('-dng', '--do_not_generate',
                        action='store_false',
                        dest='generate_jacob',
                        default=True,
                        help='Use this option to have the tester use '
                             'existing Jacobian files'
                        )
    parser.add_argument('-s', '--seed',
                        type=int,
                        default=None,
                        help='The seed to be used for random numbers'
                        )
    parser.add_argument('-p', '--pasr_output',
                        type=str,
                        default=None,
                        help='An optional saved .npy file that has the '
                             'resulting PaSR data (to speed testing)'
                             )
    parser.add_argument('-ls', '--last_spec',
                        type=str,
                        default=None,
                        help='The last species, to pass to pyJac'
                        )
    parser.add_argument('-co', '--cache_optimization',
                        default=False,
                        action='store_true',
                        help='Use to enable cache optimization in pyJac'
                        )
    parser.add_argument('-nosmem', '--no_shared',
                        default=False,
                        action='store_true',
                        help='Use to disable shared memory usage in pyJac '
                             '(CUDA only)'
                        )
    parser.add_argument('-tc', '--tchem',
                        default=False,
                        action='store_true',
                        help='Activate TChem comparison (false by default)'
                        )
    parser.add_argument('-dnc', '--do_not_compile',
                        action='store_false',
                        dest='compile_jacob',
                        default=True,
                        help='Use this option to force the tester to use '
                             'previously compiled files'
                        )
    parser.add_argument('-orxn', '--only_reaction',
                        type=str,
                        default=None,
                        help='A comma separated list of reactions to test.'
                        )
    parser.add_argument('-dnr', '--do_not_remove',
                        default=False,
                        action='store_true',
                        help='Do not remove old pyjacob module. '
                        'Useful for debugging.'
                        )
    parser.add_argument('-cn', '--condition_numbers',
                        default=None,
                        type=str,
                        help='Comma separated list of conditions to test,'
                             ' useful for debugging.'
                        )
    args = parser.parse_args()
    test.test(args.lang, os.path.dirname(os.path.abspath(test.__file__)),
              args.build_dir, args.mech, args.thermo, args.input,
              args.generate_jacob, args.compile_jacob, args.seed,
              args.pasr_output, args.last_spec, args.cache_optimization,
              args.no_shared, args.tchem, args.only_reaction,
              args.do_not_remove, args.condition_numbers
              )
