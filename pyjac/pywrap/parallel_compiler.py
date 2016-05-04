"""Module for performing parallel compilation of source code files.
"""

import multiprocessing
from multiprocessing.pool import ThreadPool
import distutils.ccompiler

N = multiprocessing.cpu_count()

# monkey-patch for parallel compilation
def parallel_compile(self, sources, output_dir=None, macros=None,
                     include_dirs=None, debug=False, extra_preargs=None,
                     extra_postargs=None, depends=None
                     ):
    """Compile source files in parallel.

    Parameters
    ----------
    sources : list of `str`
        List of source files
    output_dir : str
        Optional; path to directory for object files
    macros : list of `tuple`
        Optional; list of macro definitions, like (name, value) or (name,)
    include_dirs : list of `str`
        Optional; list of directories to add to default include file search path
    debug : bool
        Optional; if ``True``, instruct compiler to output debug signals
    extra_preargs : list of `str`
        Optional; extra command-line arguments to prepend to compiler command
    extra_postargs : list of `str`
        Optional; extra command-line arguments to append to compiler command
    depends : list of `str`
        Optional; list of filenames that target depends on

    Returns
    -------
    objects : list of `str`
        List of object files generated

    """
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs
        )
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    # number of parallel compilations

    def _single_compile(obj):
        """Compile single file.
        """
        try: src, ext = build[obj]
        except KeyError: return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # convert to list, imap is evaluated on-demand
    list(ThreadPool(N).imap(_single_compile, objects))
    return objects
