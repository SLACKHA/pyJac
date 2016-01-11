#! /usr/bin/env python2.7
from distutils.core import setup, Extension
import distutils.ccompiler

from Cython.Distutils import build_ext
import parallel_compiler as pcc
import numpy
import os

sources = ['adjacob_wrapper.pyx',
           'out/ad_dydt.c',
           'out/ad_rxn_rates.c',
           'out/ad_rxn_rates_pres_mod.c',
           'out/ad_spec_rates.c',
           'out/ad_chem_utils.c',
           'out/ad_jac.c'
           ]
includes = ['out/']

distutils.ccompiler.CCompiler.compile = pcc.parallelCompile

ext_modules=[Extension("adjacob",
     sources=sources,
     include_dirs=includes + [numpy.get_include()],
     extra_compile_args=['-frounding-math', '-fsignaling-nans'],
     language='c',
     )]

setup(
    name='adjacob',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext}
)
