import sys
import os

from . import performance_tester as pt
from argparse import ArgumentParser

def main(args=None):
    if args is None:
        # command line arguments
        parser = ArgumentParser(description='performance_tester.py: '
                                            'tests pyJac performance'
                                            )
        parser.add_argument('-w', '--working_directory',
                            type=str,
                            default='performance',
                            help='Directory storing the mechanisms / data.'
                            )
        parser.add_argument('-uoo', '--use_old_opt',
                            action='store_true',
                            default=False,
                            required=False,
                            help='If True, allows performance_tester to use '
                                 'any old optimization files found'
                            )
        parser.add_argument('-cl', '--compute_level',
                            default=20,
                            required=False,
                            choices=[20, 21, 30, 32, 35, 37, 50, 52, 53, 60, 61,
                                     62, 70, 71],
                            help='If True, allows performance_tester to use '
                                 'any old optimization files found'
                            )
        args = parser.parse_args()
        pt.performance_tester(os.path.dirname(os.path.abspath(pt.__file__)),
                              args.working_directory,
                              args.use_old_opt,
                              args.compute_level)

if __name__ == '__main__':
    sys.exit(main())
