import os
import shutil
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

#os.environ['CC'] = 'gcc-5'
#os.environ['CXX'] = 'g++-5'

if os.getenv('TCHEM_HOME'):
    tchem_home = os.getenv('TCHEM_HOME')
else:
    raise SystemError('TCHEM_HOME environment variable not set.')

# Need to copy periodic table file into local directory... for some reason
shutil.copy(os.path.join(tchem_home, 'data', 'periodictable.dat'),
            'periodictable.dat'
            )

sources = ['pytchem_wrapper.pyx', 'py_tchem.c']
includes = ['out/', './']

ext = Extension('py_tchem',
                sources=sources,
                library_dirs=[os.path.join(tchem_home, 'lib')],
                libraries=['c', 'tchem'],
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
