#! /usr/bin/env python2.7
from distutils.core import setup, Extension
import distutils.ccompiler

from Cython.Distutils import build_ext
from Cython.Build import cythonize
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

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

ext = [Extension("adjacob",
     sources=sources,
     include_dirs=includes + [numpy.get_include()],
     extra_compile_args=['-frounding-math', '-fsignaling-nans', 
                         '-DADEPT_STACK_THREAD_UNSAFE', '-fopenmp'],
     language='c++',
     libraries=['adept'],
     extra_link_args=['-fopenmp']
     )]

setup(
    name='adjacob',
    ext_modules=ext,
    cmdclass={'build_ext': build_ext},
)
