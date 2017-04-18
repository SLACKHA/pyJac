"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup
from codecs import open
from os import path
import sys

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'pyjac', '_version.py')) as version_file:
    exec(version_file.read())

# Get the long description from the relevant files
with open(path.join(here, 'README.md')) as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'CHANGELOG.md')) as changelog_file:
    changelog = changelog_file.read()

with open(path.join(here, 'CITATION.md')) as citation_file:
    citation = citation_file.read()

desc = readme + '\n\n' + changelog + '\n\n' + citation
try:
    import pypandoc
    long_description = pypandoc.convert_text(desc, 'rst', format='md')
    with open(path.join(here, 'README.rst'), 'w') as rst_readme:
        rst_readme.write(long_description)
except (ImportError, OSError, IOError):
    long_description = desc

install_requires = [
    'numpy>=1.12.0',
    'bitarray>=0.8.1',
    'optionloop>1.0.3',
    'Cython>=0.23.1',
    'pyyaml>=3.11',
]

tests_require = [
    'pytest>=3.0.1',
]

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
setup_requires = ['pytest-runner'] if needs_pytest else []

setup(
    name='pyjac',
    version=__version__,
    description='Create analytical Jacobian matrix source code for chemical kinetics',
    long_description=long_description,
    url='https://github.com/slackha/pyJac',
    author='Kyle E. Niemeyer',
    author_email='kyle.niemeyer@gmail.com',
    license='MIT License',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='chemical_kinetics analytical_Jacobian',

    packages=['pyjac', 'pyjac.core', 'pyjac.functional_tester', 'pyjac.libgen',
              'pyjac.performance_tester', 'pyjac.pywrap', 'pyjac.tests',
              ],
    package_dir={'pyjac': 'pyjac'},
    install_requires=install_requires,
    package_data={
        'pyjac.pywrap': ['*.pyx', '*.c', '*.h', '*.cu', '*.cuh', '*.in'],
        'pyjac.functional_tester' : ['*.yaml'],
        'pyjac.performance_tester' : ['*.pyx', '*.c', '*.h', '*.cu',
                                      '*.cuh', '*.in'
                                      ],
        },
    include_package_data=True,
    tests_require=tests_require,
    setup_requires=setup_requires,
    zip_safe=False,

    entry_points={
        'console_scripts': [
            'pyjac=pyjac.__main__:main',
        ],
    },
)
