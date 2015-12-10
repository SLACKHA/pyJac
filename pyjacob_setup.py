#! /usr/bin/env python2.7
from distutils.core import setup, Extension
import distutils.ccompiler

from Cython.Distutils import build_ext
from multiprocessing.pool import ThreadPool
import numpy
import os

# monkey-patch for parallel compilation
def parallelCCompile(self, sources, output_dir=None, macros=None,
                     include_dirs=None, debug=0, extra_preargs=None,
                     extra_postargs=None, depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs
        )
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    # number of parallel compilations
    N = 8

    def _single_compile(obj):
        try: src, ext = build[obj]
        except KeyError: return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # convert to list, imap is evaluated on-demand
    list(ThreadPool(N).imap(_single_compile,objects))
    return objects

distutils.ccompiler.CCompiler.compile = parallelCCompile

sources = ['pyjacob_wrapper.pyx',
           'out/dydt.c',
           'out/rxn_rates.c',
           'out/rxn_rates_pres_mod.c',
           'out/spec_rates.c',
           'out/chem_utils.c',
           'out/jacob.c'
           ]
includes = ['out/']

# Look for file with list of Jacobian files
if os.path.exists('out/jacobs') and os.path.isfile('out/jacobs/jac_list_c'):
    with open('out/jacobs/jac_list_c', 'r') as f:
        line = f.readline()
    files = line.split()
    sources += ['out/jacobs/' + f for f in files]
    includes += ['out/jacobs/']

ext_modules=[Extension("pyjacob",
     sources=sources,
     include_dirs=includes + [numpy.get_include()],
     extra_compile_args=['-frounding-math', '-fsignaling-nans'],
     language='c',
     )]

setup(
    name='pyjacob',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext}
)
