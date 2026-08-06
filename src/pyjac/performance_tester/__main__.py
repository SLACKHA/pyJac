import sys
from argparse import ArgumentParser
from pathlib import Path

from . import performance_tester as pt


def main(args=None):
    if args is None:
        # command line arguments
        parser = ArgumentParser(
            description='performance_tester.py: tests pyJac performance'
        )
        parser.add_argument(
            '-w',
            '--working_directory',
            type=str,
            default='performance',
            help='Directory storing the mechanisms / data.',
        )
        parser.add_argument(
            '-uoo',
            '--use_old_opt',
            action='store_true',
            default=False,
            required=False,
            help='If True, allows performance_tester to use '
            'any old optimization files found',
        )
        args = parser.parse_args()
        pt.performance_tester(
            str(Path(pt.__file__).resolve().parent),
            args.working_directory,
            args.use_old_opt,
        )


if __name__ == '__main__':
    sys.exit(main())
