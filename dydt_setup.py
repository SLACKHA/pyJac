from distutils.core import setup, Extension
#! /usr/bin/env python2.7

from Cython.Distutils import build_ext
import numpy
ext_modules=[Extension("py_dydt", 
	sources=["dydt_wrapper.pyx", 
			'out/dydt.c', 
			'out/rxn_rates.c',
			'out/rxn_rates_pres_mod.c',
			'out/spec_rates.c',
			'out/chem_utils.c'],
    include_dirs=['out/', numpy.get_include()],
    language='c',
    )]

setup(
	name='py_dydt',
	ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext}
)