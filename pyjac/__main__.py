import sys

from . import utils
from .core.create_jacobian import create_jacobian


def main(args=None):
    if args is None:
        args = utils.get_parser()
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

if __name__ == '__main__':
    sys.exit(main())
