#! /usr/bin/env python2.7

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import sys
import subprocess
import re
from argparse import ArgumentParser
import multiprocessing
import shutil

# More Python 2 compatibility
if sys.version_info.major == 3:
    from itertools import zip
elif sys.version_info.major == 2:
    from itertools import izip as zip

from itertools import permutations, product, chain
from string import Template

# Related modules
import numpy as np

try:
    import cantera as ct
    from cantera import ck2cti
except ImportError:
    print('Error: Cantera must be installed.')
    raise

# Local imports
import utils
from pyJac import create_jacobian
import partially_stirred_reactor as pasr

# Compiler based on language
cmd_compile = dict(c='gcc',
                   icc='icc',
                   cuda='nvcc'
                   )


# Flags based on language
flags = dict(c=['-std=c99', '-O3', '-mtune=native',
                '-fopenmp'],
             icc=['-std=c99', '-O3', '-xhost', '-fp-model', 'precise', '-ipo'],
             cuda=['-O3', '-arch=sm_20',
                   '-I/usr/local/cuda/include/',
                   '-I/usr/local/cuda/samples/common/inc/',
                   '-dc']
             )

libs = dict(c=['-lm', '-std=c99', '-fopenmp'],
            cuda=['-arch=sm_20'],
            icc=['-m64', '-ipo', '-lm', '-std=c99']
            )

def is_pdep(rxn):
    return (isinstance(rxn, ct.ThreeBodyReaction) or
    isinstance(rxn, ct.FalloffReaction) or
    isinstance(rxn, ct.ChemicallyActivatedReaction))

def check_step_file(filename, steplist):
    #checks file for existing data
    #and returns number of runs left to do
    #for each # of does in steplist
    runs = {}
    for step in steplist:
        runs[step] = 0
    if not 'cuda' in filename:
        raise Exception(filename)

    try:
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        for line in lines:
            try:
                vals = line.split(',')
                if len(vals) == 2:
                    vals = [float(v) for v in vals]
                    runs[vals[0]] += 1
            except:
                pass
        return runs
    except:
        return runs

def check_file(filename):
        #checks file for existing data
        #and returns number of runs left to do
        try:
            with open(filename, 'r') as file:
                lines = [line.strip() for line in file.readlines()]
            num_completed = 0
            to_find = 2
            for line in lines:
                try:
                    vals = line.split(',')
                    if len(vals) == to_find:
                        i = int(vals[0])
                        f = float(vals[1])
                        num_completed += 1
                except:
                    pass
            return num_completed
        except:
            return 0

def getf(x):
    return os.path.basename(x)

def compiler(fstruct):
    args = [cmd_compile[fstruct.lang]]
    args.extend(flags[fstruct.lang])
    include = ['-I./' + d for d in fstruct.i_dirs]
    args.extend(include)
    args.extend([
        '-I.' + os.path.sep + fstruct.build_dir,
        '-c', os.path.join(fstruct.build_dir, fstruct.filename + 
                    utils.file_ext[fstruct.lang]),
        '-o', os.path.join(fstruct.test_dir, getf(fstruct.filename) + '.o')
        ])
    args = [val for val in args if val.strip()]
    try:
        subprocess.check_call(args)
    except subprocess.CalledProcessError:
        print('Error: compilation failed for ' + fstruct.filename +
                utils.file_ext[fstruct.lang])
        return -1
    return 0

def linker(lang, temp_lang, test_dir, filelist):
    args = [cmd_compile[temp_lang]]
    args.extend([os.path.join(test_dir, getf(f) + '.o') for f in filelist])
    args.extend(['-o', os.path.join(test_dir, 'speedtest')])
    args.extend(libs[temp_lang])
    if lang == 'tchem':
        if os.getenv('TCHEM_HOME'):
            tchem_home = os.getenv('TCHEM_HOME')
        else:
            raise SystemError('TCHEM_HOME environment variable not set.')
        args.extend(['-L' + os.path.join(tchem_home, 'lib'), '-ltchem'])

    try:
        subprocess.check_call(args)
    except subprocess.CalledProcessError:
        print('Error: linking of test program failed.')
        sys.exit(1)

class file_struct(object):
    def __init__(self, lang, filename, i_dirs, args, build_dir, test_dir):
        self.lang = lang
        self.filename = filename
        self.i_dirs = i_dirs
        self.args = args
        self.build_dir = build_dir
        self.test_dir = test_dir

def performance_tester():
    pdir = 'performance'
    home = os.getcwd()
    build_dir = 'out'
    test_dir = 'test'

    def get_file_list(pmod, gpu=False, FD=False, tchem=False):
        i_dirs = [build_dir]
        if tchem:
            files = ['test', 'read_initial_conditions', 
                        'mechanism', 'mass_mole']
            return i_dirs, files
        files = ['chem_utils', 'dydt', 'spec_rates',
             'rxn_rates', 'test', 'read_initial_conditions',
             'mechanism', 'mass_mole'
             ]
        if pmod:
            files += ['rxn_rates_pres_mod']
        if FD:
            files += ['fd_jacob']
        else:
            files += ['jacob']
            test_lang = 'c' if not gpu else 'cuda'
            flists = [('rates', 'rate_list_{}'), ('jacobs', 'jac_list_{}')]
            for flist in flists:
                try:
                    with open(os.path.join(build_dir, flist[0], flist[1].format(test_lang))) as file:
                        vals = file.readline().strip().split(' ')
                        vals = [os.path.join(flist[0],
                                    f[:f.index(utils.file_ext[test_lang])]) for f in vals]
                        files += vals
                        i_dirs.append(os.path.join(build_dir, flist[0]))
                except:
                    pass
        if gpu:
            files += ['gpu_memory']

        return i_dirs, files

    #find the mechanisms to test
    mechanism_list = {}
    for name in os.listdir(pdir):
        if os.path.isdir(os.path.join(pdir, name)):
            #check for cti
            files = [f for f in os.listdir(os.path.join(pdir, name)) if 
                        os.path.isfile(os.path.join(pdir, name, f))]
            for f in files:
                if f.endswith('.cti'):
                    mechanism_list[name] = {}
                    mechanism_list[name]['mech'] = f
                    mechanism_list[name]['chemkin'] = f.replace('.cti', '.dat')
                    gas = ct.Solution(os.path.join(pdir, name, f))
                    mechanism_list[name]['ns'] = gas.n_species

                    thermo = next((tf for tf in files if 'therm' in tf), None)
                    if thermo is not None:
                        mechanism_list[name]['thermo'] = thermo

    if os.getenv('TCHEM_HOME'):
        tchem_home = os.getenv('TCHEM_HOME')
    else:
        raise SystemError('TCHEM_HOME environment variable not set.')


    cache_opt_base = [False, True]
    shared_base = [True, False]
    finite_diffs_base = [False, True]

    cpu_repeats = 10
    gpu_repeats = 10

    #langs=['c', 'tchem']
    for mech_name, mech_info in sorted(mechanism_list.items(), key=lambda x:x[1]['ns']):
        #get the cantera object
        gas = ct.Solution(os.path.join(home, pdir, mech_name, mech_info['mech']))
        pmod = any([is_pdep(rxn) for rxn in gas.reactions()])

        #ensure directory structure is valid
        os.chdir(os.path.join(home, pdir, mech_name))
        subprocess.check_call(['mkdir', '-p', build_dir])
        subprocess.check_call(['mkdir', '-p', test_dir])

        #clear old data
        with open(os.path.join('data.bin'), 'wb') as file:
            pass

        npy_files = [f for f in os.listdir(os.path.join(home, pdir, mech_name))
                        if f.endswith('.npy')
                        and os.path.isfile(f)]
        num_conditions = 0
        #load PaSR data for different pressures/conditions, and save to binary c file
        for npy in npy_files:
            state_data = np.load(npy)
            state_data = state_data.reshape(state_data.shape[0] *
                                state_data.shape[1],
                                state_data.shape[2]
                                )
            with open(os.path.join('data.bin'), "ab") as file:
                    state_data.tofile(file)

            num_conditions += state_data.shape[0]
            print(num_conditions)

        #figure out gpu steps
        step_size = 1
        steplist = []
        while step_size < num_conditions:
            steplist.append(step_size)
            step_size *= 2
        if step_size / 2 != num_conditions:
            steplist.append(num_conditions)

        the_path = os.getcwd()
        first_run = True

        for lang in langs:
            temp_lang = 'c' if lang != 'cuda' else 'cuda'
            if lang == 'cuda':
                shared = shared_base
                finite_diffs = finite_diffs_base
                cache_opt = cache_opt_base
            elif lang == 'c':
                shared = [False]
                finite_diffs = finite_diffs_base
                cache_opt = cache_opt_base
            elif lang == 'tchem':
                finite_diffs = [False]
                cache_opt = [False]
                shared = [False]

            for FD in finite_diffs:
                if FD:
                    shutil.copy(os.path.join(home, 'fd_jacob{}'.format(utils.file_ext[temp_lang])),
                                os.path.join(build_dir, 'fd_jacob{}'.format(utils.file_ext[temp_lang])))
                    shared = [False]
                    cache_opt = [False]

                for opt in cache_opt:
                    for smem in shared:

                        data_output = '{}_{}_{}_{}_output.txt'.format(lang, 'co' if opt else 'nco',
                                                                    'smem' if smem else 'nosmem',
                                                                    'fd' if FD else 'ajac')
                        data_output = os.path.join(the_path, data_output)
                        if lang != 'cuda':
                            repeats = cpu_repeats
                            num_completed = check_file(data_output)
                            if num_completed >= repeats:
                                continue
                            todo = {num_conditions: repeats - num_completed}
                        else:
                            repeats = gpu_repeats
                            todo = check_step_file(data_output, steplist)
                            if all(todo[x] >= repeats for x in todo):
                                continue
                            for x in todo:
                                todo[x] = repeats - todo[x]

                        if lang != 'tchem':
                            create_jacobian(lang, mech_info['mech'],
                                            optimize_cache=opt,
                                            build_path=build_dir,
                                            no_shared=not smem,
                                            num_blocks=8, num_threads=64,
                                            force_optimize=first_run
                                            )
                            if opt:
                                first_run = False

                        #now we need to write the reader
                        shutil.copy(os.path.join(home, 'static_files', 'read_initial_conditions{}'.format(utils.file_ext[temp_lang])),
                                    os.path.join(os.getcwd(), build_dir, 'read_initial_conditions{}'.format(utils.file_ext[temp_lang])))

                        #write the tester
                        file_data = {'datafile' : os.path.join(the_path, 'data.bin')}
                        if lang == 'c' or lang == 'cuda':
                            with open(os.path.join(home, 'static_files', 
                                                    'tester{}.in'.format(utils.file_ext[temp_lang]))
                                                   , 'r') as file:
                                src = Template(file.read())
                            src = src.substitute(file_data)
                        else:
                            file_data['mechfile'] = mech_info['chemkin']
                            file_data['thermofile'] = mech_info['thermo']
                            with open(os.path.join(home, 'static_files', 
                                                   'tc_tester.c.in'), 'r') as file:
                                src = Template(file.read())
                            src = src.substitute(file_data)
                        with open(os.path.join(build_dir, 
                                              'test{}'.format(utils.file_ext[temp_lang]))
                                              , 'w') as file:
                            file.write(src)

                        #copy timer
                        shutil.copy(os.path.join(home, 'static_files', 'timer.h'),
                                    os.path.join(os.getcwd(), build_dir, 'timer.h'))

                        #get file lists
                        i_dirs, files = get_file_list(pmod, gpu=lang=='cuda', FD=FD, tchem=lang=='tchem')
                        
                        # Compile generated source code
                        structs = [file_struct(temp_lang, f, flags[temp_lang], i_dirs, build_dir, test_dir) for f in files]

                        pool = multiprocessing.Pool()
                        results = pool.map(compiler, structs)
                        pool.close()
                        pool.join()
                        if any(r == -1 for r in results):
                            sys.exit(-1)

                        linker(lang, temp_lang, test_dir, files)

                        if lang == 'tchem':
                            #copy periodic table and mechanisms in
                            shutil.copy(os.path.join(tchem_home, 'data', 'periodictable.dat'), 
                                                    'periodictable.dat')

                        with open(data_output, 'a+') as file:
                            for stepsize in todo:
                                for i in range(todo[stepsize]):
                                    print(i, "/", todo[stepsize])
                                    subprocess.check_call([os.path.join(the_path, test_dir, 'speedtest'),
                                    str(num_conditions)], stdout=file)


if __name__=='__main__':
    # command line arguments
    performance_tester()