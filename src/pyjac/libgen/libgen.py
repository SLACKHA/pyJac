"""Module used to create a shared/static library from pyJac files.
"""
from __future__ import print_function

import shutil
import re
import os
import subprocess
import sys
import multiprocessing
import platform

from .. import utils

def lib_ext(shared):
    """Returns the appropriate library extension based on the shared flag"""
    return '.a' if not shared else '.so'


cmd_compile = dict(c='gcc',
                   icc='icc',
                   cuda='nvcc'
                   )


def cmd_lib(lang, shared):
    """Returns the appropriate compilation command for creation of the library based on the
    language and shared flag"""
    if lang == 'c':
        return ['ar', 'rcs'] if not shared else ['gcc', '-shared']
    elif lang == 'cuda':
        return ['nvcc', '-lib'] if not shared else ['nvcc', '-shared']
    elif lang == 'icc':
        return ['ar', 'rcs'] if not shared else ['icc', '-shared']


includes = dict(c=['/usr/local/include/'], icc=['/usr/local/include/'],
                cuda=['/usr/local/cuda/include/',
                      '/usr/local/cuda/samples/common/inc/'
                      ]
                )

flags = dict(c=['-std=c99', '-O3', '-mtune=native'],
             icc=['-std=c99', '-O3', '-xhost', '-fp-model', 'precise', '-ipo'],
             cuda=['-O3', '-arch=sm_20']
             )

shared_flags = dict(c=['-fPIC'],
                    icc=['-fPIC'],
                    cuda=['-Xcompiler', '"-fPIC"']
                    )

libs = dict(c=['-lm', '-std=c99'],
            cuda=['-lcudart'],
            icc=['-m64', '-ipo', '-lm', '-std=c99']
            )


def which(file):
    """A substitute for the `which` command, searches the PATH for
    a given file"""
    for path in os.environ["PATH"].split(os.pathsep):
        if os.path.exists(os.path.join(path, file)):
                return os.path.join(path, file)

    return None


def compiler(fstruct):
    """Given a file structure, this method will compile the source file for the
    language and options specified

    Parameters
    ----------
    fstruct : `file_struct`
        An information struct that holds the various compilation options

    Returns
    -------
    success : int
        0 if the compilation process was sucessful, -1 otherwise

    Notes
    -----
    Designed to work with a multiprocess compilation workflow
    """
    args = [cmd_compile[fstruct.build_lang]]
    if fstruct.auto_diff:
        args = ['g++']
    args.extend(flags[fstruct.build_lang])
    if fstruct.auto_diff:
        args = [x for x in args if 'std=c99' not in x]

    #always use fPIC in case we're building wrapper
    args.extend(shared_flags[fstruct.build_lang])
    args.extend(fstruct.args)
    include = ['-I{}'.format(d) for d in fstruct.i_dirs +
               includes[fstruct.build_lang]
               ]
    args.extend(include)
    args.extend([
        '-{}c'.format('d' if fstruct.lang == 'cuda' else ''),
        os.path.join(fstruct.source_dir, fstruct.filename +
                     utils.file_ext[fstruct.build_lang]
                     ),
        '-o', os.path.join(fstruct.obj_dir, os.path.basename(fstruct.filename) + '.o')
        ])
    args = [val for val in args if val.strip()]
    try:
        print(' '.join(args))
        subprocess.check_call(args)
    except OSError:
        print('Error: Compiler {} not found, generation of pyjac library failed.'.format(args[0]))
        sys.exit(-1)
    except subprocess.CalledProcessError:
        print('Error: compilation failed for ' + fstruct.filename +
              utils.file_ext[fstruct.build_lang]
              )
        return -1
    return 0


def get_cuda_path():
    """Returns location of CUDA (nvcc) on the system.

    Parameters
    ----------
    None

    Returns
    -------
    cuda_path : str
        Path where CUDA (nvcc) is found on the system.

    """
    cuda_path = which('nvcc')
    if cuda_path is None:
        print('nvcc not found!')
        sys.exit(-1)

    sixtyfourbit = platform.architecture()[0] == '64bit'
    cuda_path = os.path.dirname(os.path.dirname(cuda_path))
    cuda_path = os.path.join(cuda_path,
                             'lib{}'.format('64' if sixtyfourbit else '')
                             )
    return cuda_path


def libgen(lang, obj_dir, out_dir, filelist, shared, auto_diff):
    """Create a library from a list of compiled files

    Parameters
    ----------
    Parameters
    ----------
    obj_dir : str
        Path with object files
    out_dir : str
        Path to place the library in
    lang : {'c', 'cuda'}
        Programming language
    filelist : List of `str`
        The list of object files to include in the library
    auto_diff : Optional[bool]
        Optional; if ``True``, include autodifferentiation

    """
    command = cmd_lib(lang, shared)

    if lang == 'cuda':
        desc = 'cu'
    elif lang == 'c':
        if auto_diff:
            desc = 'ad'
        else:
            desc = 'c'

    libname = 'lib{}_pyjac'.format(desc)

    #remove the old library
    if os.path.exists(os.path.join(out_dir, libname + lib_ext(shared))):
        os.remove(os.path.join(out_dir, libname + lib_ext(shared)))
    if os.path.exists(os.path.join(out_dir, libname + lib_ext(not shared))):
        os.remove(os.path.join(out_dir, libname + lib_ext(not shared)))

    libname += lib_ext(shared)

    if not shared and lang == 'c':
        command += [os.path.join(out_dir, libname)]

    #add the files
    command.extend([os.path.join(obj_dir, os.path.basename(f) + '.o') for f in filelist])

    if shared:
        command.extend(shared_flags[lang])

    if shared or lang == 'cuda':
        command += ['-o']
        command += [os.path.join(out_dir, libname)]

        if lang == 'cuda':
            command += ['-L{}'.format(get_cuda_path())]
        command.extend(libs[lang])

    try:
        print(' '.join(command))
        subprocess.check_call(command)
    except OSError:
        print('Error: Compiler {} not found, generation of pyjac library failed.'.format(args[0]))
        sys.exit(-1)
    except subprocess.CalledProcessError:
        print('Error: Generation of pyjac library failed.')
        sys.exit(-1)

    return libname


class file_struct(object):
    """A simple structure designed to enable multiprocess compilation
    """
    def __init__(self, lang, build_lang, filename, i_dirs, args,
                 source_dir, obj_dir, shared
                 ):
        """
        Parameters
        ----------
        lang : str
            Compiler to use
        build_lang : {'c', 'cuda'}
            Programming language
        file_name : str
            The file to compile
        i_dirs : List of str
            List of include directorys for compilation
        args : List of str
            List of additional arguements
        source_dir : str
            The directory the file is located in
        obj_dir : str
            The directory to place the compiled object file in
        shared : bool
            If true, this is creating a shared library
        """

        self.lang = lang
        self.build_lang = build_lang
        self.filename = filename
        self.i_dirs = i_dirs
        self.args = args
        self.source_dir = source_dir
        self.obj_dir = obj_dir
        self.shared = shared
        self.auto_diff=False


def get_file_list(source_dir, pmod, lang, FD=False, AD=False):
    """

    Parameters
    ----------
    source_dir : str
        Path with source files
    pmod : bool
        ``True`` if pressure dependent reactions present in mechanism
    lang : {'c', 'cuda'}
        Programming language
    FD : Optional[bool]
        Optional; if ``True``, include finite difference
    AD : Optional[bool]
        Optional; if ``True``, include autodifferentiation

    Returns
    -------
    i_dirs : list of `str`
        List of include directories
    files : list of `str`
        List of files

    """
    i_dirs = [source_dir]
    if AD:
        files = ['ad_dydt', 'ad_rxn_rates', 'ad_spec_rates',
                'ad_chem_utils', 'ad_jac'
                ]
        if pmod:
            files += ['ad_rxn_rates_pres_mod']
        return i_dirs, files

    files = ['chem_utils', 'dydt', 'spec_rates',
             'rxn_rates', 'mechanism', 'mass_mole'
             ]
    if pmod:
        files += ['rxn_rates_pres_mod']

    if FD:
        files += ['fd_jacob']
        flists = []
    else:
        files += ['jacob']
        flists = [('jacobs', 'jac_list_{}')]

    flists += [('rates', 'rate_list_{}')]
    for flist in flists:
        try:
            with open(os.path.join(source_dir,
                      flist[0], flist[1].format(lang))
                      ) as file:
                vals = file.readline().strip().split(' ')
                vals = [os.path.join(flist[0],
                        f[:f.index(utils.file_ext[lang])]) for f in vals
                        ]
                files += vals
                i_dirs.append(os.path.join(source_dir, flist[0]))
        except:
            pass
    if lang == 'cuda':
        files += ['gpu_memory']

    return i_dirs, files


def generate_library(lang, source_dir, obj_dir=None,
                     out_dir=None, shared=None,
                     finite_difference=False, auto_diff=False
                     ):
    """Generate shared/static library for pyJac files.

    Parameters
    ----------
    lang : {'c', 'cuda'}
        Programming language
    source_dir : str
        Path of folder with pyJac files
    obj_dir : Optional[str]
        Optional; path of folder to store generated object files
    shared : bool
        If ``True``, generate shared library (vs. static)
    finite_difference : Optional[bool]
        If ``True``, include finite differences
    auto_diff : bool
        If ``True``, include autodifferentiation

    Returns
    -------
    Location of generated library

    """
    #check lang
    if lang not in flags.keys():
        print('Cannot generate library for unknown language {}'.format(lang))
        sys.exit(-1)

    shared = shared and lang != 'cuda'

    if lang == 'cuda' and shared:
        print('CUDA does not support linking of shared device libraries.')
        sys.exit(-1)

    build_lang = lang if lang != 'icc' else 'c'

    source_dir = os.path.abspath(os.path.normpath(source_dir))
    if obj_dir is None:
        obj_dir = os.path.join(os.getcwd(), 'obj')
    else:
        obj_dir = os.path.abspath(os.path.normpath(obj_dir))
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)
    if out_dir is None:
        out_dir = os.getcwd()
    else:
        out_dir = os.path.abspath(os.path.normpath(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    obj_dir = os.path.abspath(obj_dir)
    out_dir = os.path.abspath(out_dir)

    pmod = False
    #figure out whether there's pressure mod reactions or not
    with open(os.path.join(source_dir,
              'mechanism{}'.format(utils.header_ext[build_lang])), 'r'
              ) as file:
        for line in file.readlines():
            line = line.strip()
            match = re.search(r'\s*#define PRES_MOD_RATES (\d+)', line)
            if match is not None:
                pmod = int(match.group(1)) > 0
                break

    #get file lists
    i_dirs, files = get_file_list(source_dir, pmod, build_lang,
                                  FD=finite_difference, AD=auto_diff
                                  )

    # Compile generated source code
    structs = [file_struct(lang, build_lang, f, i_dirs,
               (['-DFINITE_DIFF'] if finite_difference else []),
               source_dir, obj_dir, shared) for f in files
               ]
    for x in structs:
        x.auto_diff=auto_diff

    pool = multiprocessing.Pool()
    results = pool.map(compiler, structs)
    pool.close()
    pool.join()
    if any(r == -1 for r in results):
       sys.exit(-1)

    libname = libgen(lang, obj_dir, out_dir, files, shared, auto_diff)
    return os.path.join(out_dir, libname)
