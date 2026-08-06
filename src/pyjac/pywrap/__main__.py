"""Main module for pywrap module."""

from argparse import ArgumentParser

from .. import utils
from ..core.CUDAParams import DEFAULT_ARCH
from .pywrap_gen import generate_wrapper

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Generates a python wrapper for pyJac via Cython'
    )
    parser.add_argument(
        '-l',
        '--lang',
        type=str,
        choices=utils.langs,
        required=True,
        help='Programming language for output source files',
    )
    parser.add_argument(
        '-so',
        '--source_dir',
        type=str,
        required=True,
        help='The folder that contains the generated pyJac files.',
    )
    parser.add_argument(
        '-out',
        '--out_dir',
        type=str,
        required=False,
        default=None,
        help='The folder to place the generated library in',
    )

    parser.add_argument(
        '-ca',
        '--cuda-arch',
        dest='cuda_arch',
        type=str,
        default=DEFAULT_ARCH,
        help='CUDA compute capability to compile for, e.g. sm_80. Defaults to '
        f'{DEFAULT_ARCH}; use "native" to target the GPU in the build machine '
        '(CUDA 11.5 and later). Ignored for non-CUDA languages.',
    )

    args = parser.parse_args()
    generate_wrapper(args.lang, args.source_dir, args.out_dir, cuda_arch=args.cuda_arch)
