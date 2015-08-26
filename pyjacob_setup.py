from distutils.core import setup, Extension
#! /usr/bin/env python2.7

from Cython.Distutils import build_ext
import numpy
ext_modules=[Extension("pyjacob", 
	sources=["pyjacob_wrapper.pyx", 
			'out/dydt.c', 
			'out/rxn_rates.c',
			'out/rxn_rates_pres_mod.c',
			'out/spec_rates.c',
			'out/chem_utils.c',
			'out/jacob.c'],
    include_dirs=['out/', numpy.get_include()],
    language='c',
    )]

setup(
	name='pyjacob',
	ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext}
)