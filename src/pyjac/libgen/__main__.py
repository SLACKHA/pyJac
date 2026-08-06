from argparse import ArgumentParser

from .. import utils
from ..core.CUDAParams import DEFAULT_ARCH
from .libgen import generate_library

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Generates a shared/static library '
        'from previously generated pyJac files using gcc/nvcc.'
    )
    parser.add_argument(
        '-l',
        '--lang',
        type=str,
        choices=utils.langs,
        required=True,
        help='Programming language for source files',
    )
    parser.add_argument(
        '-so',
        '--source_dir',
        type=str,
        required=True,
        help='Path of directory with existing pyJac files.',
    )
    parser.add_argument(
        '-ob',
        '--obj_dir',
        type=str,
        required=False,
        default=None,
        help='Path of directory for generated object files.',
    )
    parser.add_argument(
        '-out',
        '--out_dir',
        type=str,
        required=False,
        default=None,
        help='Path of directory for generated library',
    )
    parser.add_argument(
        '-st',
        '--static',
        required=False,
        default=False,
        action='store_true',
        help='If specified, the generated library will be'
        'a static library (required for CUDA).',
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
    generate_library(
        args.lang,
        args.source_dir,
        args.obj_dir,
        args.out_dir,
        not args.static,
        cuda_arch=args.cuda_arch,
    )
