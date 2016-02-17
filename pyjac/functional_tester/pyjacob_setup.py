#! /usr/bin/env python2.7
from distutils.core import setup, Extension
import distutils.ccompiler

from Cython.Distutils import build_ext
import parallel_compiler as pcc
import numpy
import os

sources = ['pyjacob_wrapper.pyx',
           'out/dydt.c',
           'out/rxn_rates.c',
           'out/rxn_rates_pres_mod.c',
           'out/spec_rates.c',
           'out/chem_utils.c',
           'out/jacob.c'
           ]
includes = ['out/']

distutils.ccompiler.CCompiler.compile = pcc.parallelCompile

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
