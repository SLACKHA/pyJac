"""Module for generating Python wrapper around pyJac code."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from string import Template

from ..libgen import generate_library


def generate_setup(setupfile, home_dir, build_dir, out_dir, libname, setup_path):
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
    setup_path : str
        Path to write the filled-in setup script to. This is kept out of the
        package directory, which is typically read-only once installed.

    Returns
    -------
    None

    """
    with open(setupfile) as file:
        src = Template(file.read())

    file_data = {
        'homepath': home_dir,
        'buildpath': build_dir,
        'libname': libname,
        'outpath': out_dir,
    }
    src = src.safe_substitute(file_data)

    Path(setup_path).parent.mkdir(parents=True, exist_ok=True)
    with open(setup_path, 'w') as file:
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

    f = '{dirname}.{platform}-{version[0]}.{version[1]}'
    return f.format(
        dirname=dname, platform=sysconfig.get_platform(), version=sys.version_info
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
    package_dir = Path(__file__).resolve().parent

    if out_dir is None:
        out_dir = os.getcwd()

    distutils_build = os.path.join('build', distutils_dir_name('temp'))

    # Cython writes its generated .c next to the .pyx it compiles, so the
    # wrapper sources are staged into the build directory rather than compiled
    # in place; the package directory is read-only in a normal installation.
    home_dir = os.path.join(distutils_build, 'pywrap_src')
    Path(home_dir).mkdir(parents=True, exist_ok=True)
    for pattern in ('*.pyx', '*.pxd', '*.c', '*.h', '*.cu', '*.cuh'):
        for src in package_dir.glob(pattern):
            shutil.copy2(src, Path(home_dir) / src.name)

    shared = False
    ext = '.so' if shared else '.a'
    lib = None
    if lang != 'tchem':
        # first generate the library
        lib = generate_library(
            lang,
            source_dir,
            out_dir=distutils_build,
            shared=shared,
            auto_diff=auto_diff,
        )
        lib = os.path.normpath(lib)
        if shared:
            lib = lib[lib.index('lib') + len('lib') : lib.index(ext)]

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
        print(f'Language {lang} not recognized')
        sys.exit(-1)

    setup_path = os.path.join(distutils_build, setupfile[: setupfile.index('.in')])
    generate_setup(
        os.path.join(package_dir, setupfile),
        home_dir,
        source_dir,
        distutils_build,
        lib,
        setup_path,
    )

    # sys.executable, not a reconstructed pythonX.Y name: the latter resolves
    # against PATH and so escapes the active virtual environment, where Cython,
    # NumPy and setuptools are installed.
    subprocess.check_call(
        [sys.executable, setup_path, 'build_ext', '--build-lib', out_dir]
    )
