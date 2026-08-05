"""Module for generating Python wrapper around pyJac code.
"""
import sys
import os
import subprocess
from string import Template

from ..libgen import generate_library

def generate_setup(setupfile, home_dir, build_dir, out_dir, libname):
    """Helper method to fill in the template .in files

    Parameters
    ----------
    setupfile : str
        Filename of existing setup file
    home_dir : str
        Home directory path
    build_dir : str
        Build directory path
    out_dir : str
        Output directory path
    libname : str
        Library name

    Returns
    -------
    None

    """
    with open(setupfile, 'r') as file:
        src = Template(file.read())

    file_data = {'homepath' : home_dir,
                 'buildpath' : build_dir,
                 'libname' : libname,
                 'outpath' : out_dir
                 }
    src = src.safe_substitute(file_data)
    with open(setupfile[:setupfile.rindex('.in')], 'w') as file:
        file.write(src)


def distutils_dir_name(dname):
    """Returns the name of a distutils build directory

    Parameters
    ----------
    dname : str
        Base directory name

    Returns
    -------
    Name of a distutils build directory

    """
    import sys
    import sysconfig
    f = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return f.format(dirname=dname,
                    platform=sysconfig.get_platform(),
                    version=sys.version_info
                    )


def generate_wrapper(lang, source_dir, out_dir=None, auto_diff=False):
    """Generates a Python wrapper for the given language and source files

    Parameters
    ----------
    lang : {'cuda', 'c', 'tchem'}
        Programming language of pyJac (cuda, c) or TChem
    source_dir : str
        Directory path of source files.
    out_dir : Optional[str]
        Directory path for output files
    auto_diff : Optional[bool]
        Optional; if ``True``, build autodifferentiation library

    Returns
    -------
    None

    """

    source_dir = os.path.normpath(source_dir)
    home_dir = os.path.abspath(os.path.dirname(__file__))

    if out_dir is None:
        out_dir = os.getcwd()

    distutils_build = os.path.join('build', distutils_dir_name('temp'))

    shared = False
    ext = '.so' if shared else '.a'
    lib = None
    if lang != 'tchem':
        #first generate the library
        lib = generate_library(lang, source_dir, out_dir=distutils_build,
                               shared=shared, auto_diff=auto_diff
                               )
        lib = os.path.normpath(lib)
        if shared:
            lib = lib[lib.index('lib') + len('lib'):lib.index(ext)]

    setupfile = None
    if lang == 'c':
        setupfile = 'pyjacob_setup.py.in'
        if auto_diff:
            setupfile = 'adjacob_setup.py.in'
    elif lang == 'cuda':
        setupfile = 'pyjacob_cuda_setup.py.in'
    elif lang == 'tchem':
        setupfile = 'pytchem_setup.py.in'
    else:
        print('Language {} not recognized'.format(lang))
        sys.exit(-1)

    generate_setup(os.path.join(home_dir, setupfile), home_dir, source_dir,
                   distutils_build, lib
                   )

    python_str = 'python{}.{}'.format(sys.version_info[0], sys.version_info[1])

    subprocess.check_call([python_str, os.path.join(home_dir,
                           setupfile[:setupfile.index('.in')]),
                           'build_ext', '--build-lib', out_dir
                           ])
