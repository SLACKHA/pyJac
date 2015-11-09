import os
import shutil
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

tchem_home = '/Users/Kyle/work/TChem_v0.2/'

# Need to copy periodic table file into local directory... for some reason
shutil.copy(os.path.join(tchem_home, 'data', 'periodictable.dat'),
            'periodictable.dat'
            )

sources = ['pytchem_wrapper.pyx',
           'py_tchem.c',
           'out/chem_utils.c',
           'out/dydt.c',
           'out/rxn_rates.c',
           'out/rxn_rates_pres_mod.c',
           'out/spec_rates.c',
           'out/jacob.c'
           ]
includes = ['out/', './']

# Look for file with list of Jacobian files
if os.path.exists('out/jacobs') and os.path.isfile('out/jacobs/jac_list_c'):
    with open('out/jacobs/jac_list_c', 'r') as f:
        line = f.readline()
    files = line.split()
    sources += ['out/jacobs/' + f for f in files]
    includes += ['out/jacobs/']

ext = Extension('py_tchem',
                sources=sources,
                library_dirs=[os.path.join(tchem_home, 'lib')],
                libraries=['tchem'],
                language='c',
                include_dirs=includes + [numpy.get_include(),
                                         os.path.join(tchem_home, 'include')
                                         ],
                extra_compile_args=['-frounding-math', '-fsignaling-nans']
                )

setup(name='py_tchem',
      ext_modules=[ext],
      cmdclass={'build_ext': build_ext},
      # since the package has c code, the egg cannot be zipped
      zip_safe=False
      )
