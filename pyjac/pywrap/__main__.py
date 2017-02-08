"""Main module for pywrap module.
"""
from argparse import ArgumentParser
from .pywrap_gen import generate_wrapper

from .. import utils

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Generates a python wrapper for pyJac via Cython'
        )
    parser.add_argument('-l', '--lang',
                        type=str,
                        choices=utils.langs,
                        required=True,
                        help='Programming language for output '
                             'source files'
                        )
    parser.add_argument('-so', '--source_dir',
                        type=str,
                        required=True,
                        help='The folder that contains the generated pyJac '
                             'files.')
    parser.add_argument('-out', '--out_dir',
                        type=str,
                        required=False,
                        default=None,
                        help='The folder to place the generated library in')

    args = parser.parse_args()
    generate_wrapper(args.lang, args.source_dir, args.out_dir)
