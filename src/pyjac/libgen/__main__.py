from argparse import ArgumentParser

from .libgen import generate_library
from .. import utils

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Generates a shared/static library '
                    'from previously generated pyJac files using gcc/nvcc.'
        )
    parser.add_argument('-l', '--lang',
                        type=str,
                        choices=utils.langs,
                        required=True,
                        help='Programming language for source files'
                        )
    parser.add_argument('-so', '--source_dir',
                        type=str,
                        required=True,
                        help='Path of directory with existing pyJac files.'
                        )
    parser.add_argument('-ob', '--obj_dir',
                        type=str,
                        required=False,
                        default=None,
                        help='Path of directory for generated object files.'
                        )
    parser.add_argument('-out', '--out_dir',
                        type=str,
                        required=False,
                        default=None,
                        help='Path of directory for generated library'
                        )
    parser.add_argument('-st', '--static',
                        required=False,
                        default=False,
                        action='store_true',
                        help='If specified, the generated library will be'
                             'a static library (required for CUDA).'
                        )

    args = parser.parse_args()
    generate_library(args.lang, args.source_dir, args.obj_dir,
                     args.out_dir, not args.static
                     )
