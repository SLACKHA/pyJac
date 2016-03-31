from argparse import ArgumentParser
from libgen import generate_library
from .. import utils

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
    parser.add_argument('-so', '--source_dir',
                        type=str,
                        required=True,
                        help='The folder that contains the generated pyJac '
                             'files.')
    parser.add_argument('-ob', '--obj_dir',
                        type=str,
                        required=False,
                        default=None,
                        help='The folder to store the generated object '
                             'files in')
    parser.add_argument('-out', '--out_dir',
                        type=str,
                        required=False,
                        default=None,
                        help='The folder to place the generated library in')
    parser.add_argument('-st', '--static',
                        required=False,
                        default=False,
                        action='store_true',
                        help='If specified, the generated library will be'
                             'a static library')

    args = parser.parse_args()
    generate_library(args.lang, args.source_dir, args.obj_dir,
        args.out_dir, not args.static)
    