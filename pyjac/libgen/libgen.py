import shutil
import re
import os
import subprocess
import sys
import multiprocessing

from .. import utils

def lib_ext(shared):
    return '.a' if not shared else '.so'

cmd_compile = dict(c='gcc',
                   icc='icc',
                   cuda='nvcc'
                   )

def cmd_lib(lang, shared):
    if lang == 'c':
        return ['ar', 'rcs'] if not shared else ['gcc', '-shared']
    elif lang == 'cuda':
        return ['ar', 'rcs'] if not shared else ['nvcc', '-shared']
    elif lang == 'icc':
        return ['ar', 'rcs'] if not shared else ['icc', '-shared']

flags = dict(c=['-std=c99', '-O3', '-mtune=native',
                '-fopenmp'],
             icc=['-std=c99', '-O3', '-xhost', '-fp-model', 'precise', '-ipo'],
             cuda=['-O3', '-arch=sm_20',
                   '-I/usr/local/cuda/include/',
                   '-I/usr/local/cuda/samples/common/inc/',
                   '-dc']
             )

shared_flags = dict(c=['-fPIC'],
                    icc=['-fPIC'],
                    cuda=['-Xcompiler', '"-fPIC"'])

libs = dict(c=['-lm', '-std=c99', '-fopenmp'],
            cuda=['-arch=sm_20'],
            icc=['-m64', '-ipo', '-lm', '-std=c99']
            )

def getf(x):
    return os.path.basename(x)

def compiler(fstruct):
    args = [cmd_compile[fstruct.lang]]
    args.extend(fstruct.args)
    include = ['-I ' + d for d in fstruct.i_dirs]
    args.extend(include)
    args.extend([
        '-c', os.path.join(fstruct.source_dir, fstruct.filename +
                    utils.file_ext[fstruct.build_lang]),
        '-o', os.path.join(fstruct.obj_dir, getf(fstruct.filename) + '.o')
        ])
    args = [val for val in args if val.strip()]
    try:
        subprocess.check_call(args)
    except subprocess.CalledProcessError:
        print('Error: compilation failed for ' + fstruct.filename +
                utils.file_ext[fstruct.build_lang])
        return -1
    return 0

def libgen(lang, obj_dir, out_dir, filelist, shared):
    command = cmd_lib(lang, shared)
    desc = 'c' if lang != 'cuda' else 'cu'
    libname = 'lib{}_pyjac'.format(desc) + lib_ext(shared)

    #remove the old library
    if os.path.exists(os.path.join(out_dir, libname)):
        os.remove(libname)

    if shared:
        command += ['-o']
    command += [os.path.join(out_dir, libname)]
    if shared:
        command.extend(libs[lang])
    command.extend([os.path.join(obj_dir, getf(f) + '.o') for f in filelist])

    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError:
        print('Error: Generation of pyjac library failed.')
        sys.exit(-1)

    return libname

class file_struct(object):
    def __init__(self, lang, build_lang, filename, i_dirs, args, source_dir, obj_dir):
        self.lang = lang
        self.build_lang = build_lang
        self.filename = filename
        self.i_dirs = i_dirs
        self.args = args
        self.source_dir = source_dir
        self.obj_dir = obj_dir

def get_file_list(source_dir, pmod, lang, FD=False, tchem=False):
    i_dirs = [source_dir]
    files = ['chem_utils', 'dydt', 'spec_rates',
         'rxn_rates', 'mechanism', 'mass_mole']
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
            with open(os.path.join(source_dir, flist[0], flist[1].format(lang))) as file:
                vals = file.readline().strip().split(' ')
                vals = [os.path.join(flist[0],
                            f[:f.index(utils.file_ext[lang])]) for f in vals]
                files += vals
                i_dirs.append(os.path.join(source_dir, flist[0]))
        except:
            pass
    if lang == 'cuda':
        files += ['gpu_memory']

    return i_dirs, files

def generate_library(lang, source_dir, obj_dir=None,
                        out_dir=None, shared=True,
                        finite_difference=False):
    #check lang
    if lang not in flags.keys():
        print 'Cannot generate library for unknown language {}'.format(lang)
        sys.exit(-1)

    build_lang = lang if lang != 'icc' else 'c'

    source_dir = os.path.abspath(source_dir)
    if obj_dir is None:
        obj_dir = os.path.join(os.path.dirname(__file__), 'obj')
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)
    if out_dir is None:
        out_dir = os.getcwd()

    obj_dir = os.path.abspath(obj_dir)
    out_dir = os.path.abspath(out_dir)

    pmod = False
    #figure out whether there's pressure mod reactions or not
    with open(os.path.join(source_dir, 'mechanism{}'.format(utils.header_ext[build_lang])), 'r') as file:
        for line in file.readlines():
            line = line.strip()
            match = re.search(r'\s*#define PRES_MOD_RATES (\d+)', line)
            if match is not None:
                pmod = int(match.group(1)) > 0
                break

    #get file lists
    i_dirs, files = get_file_list(source_dir, pmod, build_lang, FD=finite_difference)

    sflag = [] if not shared else shared_flags[lang]
    # Compile generated source code
    structs = [file_struct(lang, build_lang, f, i_dirs,
                    (['-DFINITE_DIFF'] if finite_difference else []) +
                    flags[lang] + sflag,
                    source_dir, obj_dir) for f in files]

    pool = multiprocessing.Pool()
    results = pool.map(compiler, structs)
    pool.close()
    pool.join()
    if any(r == -1 for r in results):
       sys.exit(-1)

    libname = libgen(lang, obj_dir, out_dir, files, shared)
    return os.path.join(out_dir, libname)