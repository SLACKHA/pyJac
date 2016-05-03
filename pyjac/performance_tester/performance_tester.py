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
from collections import defaultdict

# More Python 2 compatibility
if sys.version_info.major == 3:
    from itertools import zip
elif sys.version_info.major == 2:
    from itertools import izip as zip

from string import Template

# Related modules
import numpy as np

try:
    import cantera as ct
    from cantera import ck2cti
except ImportError:
    print('Error: Cantera must be installed.')
    raise

try:
    from optionloop import optionloop
except ImportError:
    print('Error: optionloop must be installed.')
    raise

# Local imports
from .. import utils
from ..core.create_jacobian import create_jacobian
from ..libgen import generate_library, libs, compiler, file_struct, get_cuda_path, flags

#cuda only works for static libraries
STATIC = True

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


def cmd_link(lang, shared):
    if lang == 'icc':
        return ['icc']
    elif lang == 'c':
        return ['gcc']
    elif lang == 'cuda':
        return ['nvcc'] if not shared else ['g++']


def linker(lang, temp_lang, test_dir, filelist, lib=None):
    args = cmd_link(temp_lang, not STATIC)
    if lang == 'cuda' or (not STATIC):
        args.extend(flags[temp_lang])
    args.extend([os.path.join(test_dir, getf(f) + '.o') for f in filelist])
    args.extend(['-o', os.path.join(test_dir, 'speedtest')])
    if temp_lang == 'cuda':
        args.append('-L{}'.format(get_cuda_path()))
    args.extend(libs[temp_lang])
    if lang == 'tchem':
        if os.getenv('TCHEM_HOME'):
            tchem_home = os.getenv('TCHEM_HOME')
        else:
            raise SystemError('TCHEM_HOME environment variable not set.')
        args.extend(['-L{}'.format(os.path.join(tchem_home, 'lib')), '-ltchem'])

    if lib is not None:
        if STATIC:
            args += ['-L{}'.format(os.getcwd())]
            args += ['-l{}'.format(lib)]
        else:
            args += [lib]

    args.append('-lm')

    try:
        print(' '.join(args))
        subprocess.check_call(args)
    except subprocess.CalledProcessError:
        print('Error: linking of test program failed.')
        sys.exit(1)

def performance_tester(home, pdir, use_old_opt, num_threads):
    build_dir = 'out'
    test_dir = 'test'

    pdir = os.path.abspath(pdir)

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

    def false_factory():
        return False

    c_params = {'lang' : 'c',
                'cache_opt' : [False, True],
                'finite_diffs' : [False, True]}
    cuda_params = {'lang' : 'cuda',
                    'cache_opt' : [False, True],
                    'shared' : [False, True],
                    'finite_diffs' : [False, True]}
    tchem_params = {'lang' : 'tchem'}

    #set up testing environment
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(num_threads)
    env['MKL_NUM_THREADS'] = str(num_threads)

    for mech_name, mech_info in sorted(mechanism_list.items(), key=lambda x:x[1]['ns']):
        #get the cantera object
        gas = ct.Solution(os.path.join(pdir, mech_name, mech_info['mech']))
        pmod = any([is_pdep(rxn) for rxn in gas.reactions()])

        #ensure directory structure is valid
        os.chdir(os.path.join(pdir, mech_name))
        subprocess.check_call(['mkdir', '-p', build_dir])
        subprocess.check_call(['mkdir', '-p', test_dir])

        #clear old data
        with open(os.path.join('data.bin'), 'wb') as file:
            pass

        npy_files = [f for f in os.listdir(os.path.join(pdir, mech_name))
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
        op = optionloop(c_params, false_factory)
        op = op + optionloop(cuda_params, false_factory)
        op = op + optionloop(tchem_params, false_factory)

        haveOpt = False
        if os.path.isfile(os.path.join(os.getcwd(), build_dir, 'optimized.pickle')):
            haveOpt = True

        for state in op:
            lang = state['lang']
            temp_lang = 'c' if lang != 'cuda' else 'cuda'
            FD = state['finite_diffs']
            if FD:
                shutil.copy(os.path.join(home, 'fd_jacob{}'.format(utils.file_ext[temp_lang])),
                            os.path.join(build_dir, 'fd_jacob{}'.format(utils.file_ext[temp_lang])))

            opt = state['cache_opt']
            smem = state['shared']

            data_output = '{}_{}_{}_{}_output.txt'.format(lang, 'co' if opt else 'nco',
                                                        'smem' if smem else 'nosmem',
                                                        'fd' if FD else 'ajac')

            data_output = os.path.join(the_path, data_output)
            if lang != 'cuda':
                repeats = cpu_repeats
                num_completed = check_file(data_output)
                todo = {num_conditions: repeats - num_completed}
            else:
                repeats = gpu_repeats
                todo = check_step_file(data_output, steplist)
                for x in todo:
                    todo[x] = repeats - todo[x]
            if not any(todo[x] > 0 for x in todo):
                continue

            if opt and haveOpt and not use_old_opt:
                raise Exception('Previous optimization file found... exiting')

            if lang != 'tchem':
                create_jacobian(lang, mech_info['mech'],
                                optimize_cache=opt,
                                build_path=build_dir,
                                no_shared=not smem,
                                num_blocks=8, num_threads=64,
                                multi_thread=multiprocessing.cpu_count()
                                )

            #now we need to write the reader
            shutil.copy(os.path.join(home, 'read_initial_conditions{}'.format(utils.file_ext[temp_lang])),
                        os.path.join(os.getcwd(), build_dir, 'read_initial_conditions{}'.format(utils.file_ext[temp_lang])))

            #write the tester
            file_data = {'datafile' : os.path.join(the_path, 'data.bin')}
            if lang == 'c' or lang == 'cuda':
                with open(os.path.join(home,
                                        'tester{}.in'.format(utils.file_ext[temp_lang]))
                                       , 'r') as file:
                    src = Template(file.read())
                src = src.substitute(file_data)
            else:
                file_data['mechfile'] = mech_info['chemkin']
                file_data['thermofile'] = mech_info['thermo']
                with open(os.path.join(home,
                                       'tc_tester.c.in'), 'r') as file:
                    src = Template(file.read())
                src = src.substitute(file_data)
            with open(os.path.join(build_dir,
                                  'test{}'.format(utils.file_ext[temp_lang]))
                                  , 'w') as file:
                file.write(src)

            #copy timer
            shutil.copy(os.path.join(home, 'timer.h'),
                        os.path.join(os.getcwd(), build_dir, 'timer.h'))

            #get file lists
            i_dirs = [build_dir]
            files = ['test', 'read_initial_conditions']

            lib = None
            #now build the library
            if lang != 'tchem':
                lib = generate_library(lang, build_dir, test_dir, finite_difference=FD, shared=not STATIC)
                if not STATIC:
                    lib = os.path.normpath(lib)
                    lib = lib[lib.index('lib') + len('lib'):lib.index('.so' if not STATIC else '.a')]
            else:
                files += ['mechanism', 'mass_mole']

            # Compile generated source code
            structs = [file_struct(lang, temp_lang, f, i_dirs,
                            (['-DFINITE_DIFF'] if FD else []),
                            build_dir, test_dir, not STATIC) for f in files]

            pool = multiprocessing.Pool()
            results = pool.map(compiler, structs)
            pool.close()
            pool.join()
            if any(r == -1 for r in results):
               sys.exit(-1)

            linker(lang, temp_lang, test_dir, files, lib)

            if lang == 'tchem':
                #copy periodic table and mechanisms in
                shutil.copy(os.path.join(tchem_home, 'data', 'periodictable.dat'),
                                        'periodictable.dat')

            with open(data_output, 'a+') as file:
                for stepsize in todo:
                    for i in range(todo[stepsize]):
                        print(i, "/", todo[stepsize])
                        subprocess.check_call([os.path.join(the_path, test_dir, 'speedtest'),
                        str(stepsize)], stdout=file, env=env)
