import multiprocessing
from multiprocessing.pool import ThreadPool
import distutils.ccompiler

N = num_threads=multiprocessing.cpu_count()

# monkey-patch for parallel compilation
def parallelCompile(self, sources, output_dir=None, macros=None,
                     include_dirs=None, debug=0, extra_preargs=None,
                     extra_postargs=None, depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs
        )
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    # number of parallel compilations

    def _single_compile(obj):
        try: src, ext = build[obj]
        except KeyError: return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # convert to list, imap is evaluated on-demand
    list(ThreadPool(N).imap(_single_compile, objects))
    return objects