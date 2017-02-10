"""Module for generating Python wrapper around pyJac code.
"""
import sys
import os
import subprocess
from string import Template

from ..libgen import generate_library
from .. import site_conf as site

def generate_setup(setupfile, home_dir, build_dir, out_dir, libname,
    extra_include_dirs=[], libraries=[], libdirs=[]):
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
    extra_include_dirs : Optional[list of str]
        Optional; if supplied, extra include directions for the python wrapper
    libraries : Optional[list of str]
        Optional; if supplied extra libraries to use
    libdirs : Optional[list of str]
        Optional; if supplied, library directories

    Returns
    -------
    None

    """
    with open(setupfile, 'r') as file:
        src = Template(file.read())

    def __arr_create(arr):
        return ', '.join(["'{}'".format(x) for x in arr])

    file_data = {'homepath' : home_dir,
                 'buildpath' : build_dir,
                 'libname' : libname,
                 'outpath' : out_dir,
                 'extra_include_dirs' : __arr_create(extra_include_dirs),
                 'libs' : __arr_create(libraries),
                 'libdirs' : __arr_create(libdirs)
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


def generate_wrapper(lang, source_dir, build_dir=None, out_dir=None, auto_diff=False,
    platform=''):
    """Generates a Python wrapper for the given language and source files

    Parameters
    ----------
    lang : {'cuda', 'c', 'tchem'}
        Programming language of pyJac (cuda, c) or TChem
    source_dir : str
        Directory path of source files.
    build_dir : str
        Directory path of the generated c/cuda/opencl library
    out_dir : Optional[str]
        Directory path for the output python library
    auto_diff : Optional[bool]
        Optional; if ``True``, build autodifferentiation library
    platform : Optional[str]
        Optional; if specified, the platform for OpenCL execution
    Returns
    -------
    None

    """

    source_dir = os.path.normpath(source_dir)
    home_dir = os.path.abspath(os.path.dirname(__file__))

    if out_dir is None:
        out_dir = os.getcwd()

    if build_dir is None:
        build_dir = os.path.join(os.path.getcwd(), 'build', distutils_dir_name('temp'))

    shared = False
    ext = '.so' if shared else '.a'
    lib = None
    if lang != 'tchem':
        #first generate the library
        lib = generate_library(lang, source_dir, out_dir=build_dir,
                               shared=shared, auto_diff=auto_diff
                               )
        lib = os.path.normpath(lib)
        if shared:
            lib = lib[lib.index('lib') + len('lib'):lib.index(ext)]

    extra_include_dirs = []
    libraries = []
    libdirs = []
    rpath = ''
    if lang == 'opencl':
        extra_include_dirs.extend(site.CL_INC_DIR)
        libraries.extend(site.CL_LIBNAME)
        rpath = next(x for x in site.CL_PATHS if
            platform.lower() in x)
        rpath = site.CL_PATHS[rpath]
        libdirs.extend([rpath])

    setupfile = None
    if lang == 'c':
        setupfile = 'pyjacob_setup.py.in'
        if auto_diff:
            setupfile = 'adjacob_setup.py.in'
    elif lang == 'cuda':
        setupfile = 'pyjacob_cuda_setup.py.in'
    elif lang == 'tchem':
        setupfile = 'pytchem_setup.py.in'
    elif lang == 'opencl':
        setupfile = 'pyocl_setup.py.in'
    else:
        print('Language {} not recognized'.format(lang))
        sys.exit(-1)

    generate_setup(os.path.join(home_dir, setupfile), home_dir, source_dir,
                   build_dir, lib, extra_include_dirs, libraries, libdirs
                   )

    python_str = 'python{}.{}'.format(sys.version_info[0], sys.version_info[1])

    call = [python_str, os.path.join(home_dir,
                           setupfile[:setupfile.index('.in')]),
                           'build_ext', '--build-lib', out_dir
                           ]
    if rpath:
        call += ['--rpath', rpath]

    subprocess.check_call(call)
