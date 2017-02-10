"""Module for performance testing of pyJac and related tools.
"""

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
    from optionloop import OptionLoop
except ImportError:
    print('Error: optionloop must be installed.')
    raise

# Local imports
from .. import utils
from ..core.create_jacobian import create_jacobian
from ..libgen import (generate_library, libs, compiler, file_struct,
                      get_cuda_path, flags, lib_dirs
                      )
from .. import site_config as site

STATIC = False
"""bool: CUDA only works for static libraries"""

def check_step_file(filename, steplist):
    """Checks file for existing data, returns number of runs left

    Parameters
    ----------
    filename : str
        Name of file with data
    steplist : list of int
        List of different numbers of steps

    Returns
    -------
    runs : dict
        Dictionary with number of runs left for each step

    """
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
    """Checks file for existing data, returns number of completed runs

    Parameters
    ----------
    filename : str
        Name of file with data

    Returns
    -------
    num_completed : int
        Number of completed runs

    """
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
    """Return linker command.

    Parameters
    ----------
    lang : {'icc', 'c', 'cuda'}
        Programming language
    shared : bool
        ``True`` if shared

    Returns
    -------
    cmd : list of `str`
        List with linker command

    """
    cmd = None
    if lang == 'opencl':
        cmd = ['gcc']
    elif lang == 'c':
        cmd = ['gcc']
    elif lang == 'cuda':
        cmd = ['nvcc'] if not shared else ['g++']
    else:
        print('Lang must be one of {opecl, c}')
        raise
    return cmd


def linker(lang, temp_lang, test_dir, filelist, lib=None, platform=''):
    args = cmd_link(temp_lang, not STATIC)
    args.extend(flags[lang])
    args.extend([os.path.join(test_dir, getf(f) + '.o') for f in filelist])
    args.extend(['-o', os.path.join(test_dir, 'speedtest')])
    args.extend(libs[lang])
    rpath = ''
    if lang == 'opencl':
        rpath = next(x for x in site.CL_PATHS if
            platform.lower() in x)
        rpath = site.CL_PATHS[rpath]
        libdirs.extend([rpath])

    args.append('-lm')

    try:
        print(' '.join(args))
        subprocess.check_call(args)
    except subprocess.CalledProcessError:
        print('Error: linking of test program failed.')
        sys.exit(1)


def performance_tester(home, work_dir):
    """Runs performance testing for pyJac, TChem, and finite differences.

    Parameters
    ----------
    home : str
        Directory of source code files
    work_dir : str
        Working directory with mechanisms and for data
    use_old_opt : bool
        If ``True``, use old optimization files found
    num_threads : int
        Number of OpenMP threads to parallelize performance testing

    Returns
    -------
    None

    """
    build_dir = 'out'
    test_dir = 'test'

    work_dir = os.path.abspath(work_dir)

    #find the mechanisms to test
    mechanism_list = {}
    if not os.path.exists(work_dir):
        print ('Error: work directory {} for '.format(work_dir) +
               'performance testing not found, exiting...')
        sys.exit(-1)
    for name in os.listdir(work_dir):
        if os.path.isdir(os.path.join(work_dir, name)):
            #check for cti
            files = [f for f in os.listdir(os.path.join(work_dir, name)) if
                        os.path.isfile(os.path.join(work_dir, name, f))]
            for f in files:
                if f.endswith('.cti'):
                    mechanism_list[name] = {}
                    mechanism_list[name]['mech'] = f
                    mechanism_list[name]['chemkin'] = f.replace('.cti', '.dat')
                    gas = ct.Solution(os.path.join(work_dir, name, f))
                    mechanism_list[name]['ns'] = gas.n_species

                    thermo = next((tf for tf in files if 'therm' in tf), None)
                    if thermo is not None:
                        mechanism_list[name]['thermo'] = thermo

    if len(mechanism_list) == 0:
        print('No mechanisms found for performance testing in '
              '{}, exiting...'.format(work_dir)
              )
        sys.exit(-1)

    repeats = 10

    def false_factory():
        return False

    #c_params = {'lang' : 'c',
    #            'cache_opt' : [False, True],
    #            'finite_diffs' : [False, True]
    #            }
    #cuda_params = {'lang' : 'cuda',
    #               'cache_opt' : [False, True],
    #               'shared' : [False, True],
    #               'finite_diffs' : [False, True]
    #               }
    #tchem_params = {'lang' : 'tchem'}
    vec_widths = [4, 8, 16]
    wide_ocl_params = {'lang' : 'opencl',
                  'vecsize' : vec_widths,
                  'order' : 'F'}

    for mech_name, mech_info in sorted(mechanism_list.items(),
                                       key=lambda x:x[1]['ns']
                                       ):
        #get the cantera object
        gas = ct.Solution(os.path.join(work_dir, mech_name, mech_info['mech']))

        #ensure directory structure is valid
        os.chdir(os.path.join(work_dir, mech_name))
        subprocess.check_call(['mkdir', '-p', build_dir])
        subprocess.check_call(['mkdir', '-p', test_dir])

        num_conditions = 0
        npy_files = [f for f in os.listdir(os.path.join(work_dir, mech_name))
                        if f.endswith('.npy')
                        and os.path.isfile(f)]
        data = None
        with open('data.bin', 'wb') as file:
            #load PaSR data for different pressures/conditions,
            # and save to binary C file
            for npy in sorted(npy_files):
                state_data = np.load(npy)
                state_data = state_data.reshape(state_data.shape[0] *
                                    state_data.shape[1],
                                    state_data.shape[2]
                                    )
                num_conditions += state_data.shape[0]

                use_data = state_data.T
                out_data = np.zeros((gas.n_species + 2, state_data.shape[0]))
                for i in range(state_data.shape[0]):
                    #convert to T, P, C
                    gas.TPY = use_data[i, 1], use_data[i, 2], use_data[i, 3:]
                    out_data[i, 0] = gas.T
                    out_data[i, 1] = gas.P
                    out_data[i, 2] = gas.concentrations[:]

                if data is None:
                    data = state_data
                else:
                    data = np.vstack((data, state_data))
                print(num_conditions, data.shape)
            if num_conditions == 0:
                print('No data found in folder {}, continuing...'.format(mech_name))
                continue
            data.tofile(file)

        #figure out gpu steps
        step_size = 1
        steplist = []
        while step_size < num_conditions:
            steplist.append(step_size)
            step_size *= vec_widths[-1]

        the_path = os.getcwd()
        first_run = True
        op = OptionLoop(wide_ocl_params, false_factory)

        for state in op:
            lang = state['lang']
            vecsize = state['vecsize']
            order = state['order']
            temp_lang = 'c'

            data_output = ('{}_{}_{}'.format(lang, vecsize, order) +
                           '_output.txt'
                           )

            data_output = os.path.join(the_path, data_output)
            todo = check_step_file(data_output, steplist)
            for x in todo:
                todo[x] = repeats - todo[x]
            if not any(todo[x] > 0 for x in todo):
                continue

            if lang != 'tchem':
                create_jacobian(lang,
                    mech_name=mech_info['mech'],
                    vector_size=vector_size,
                    wide=True,
                    build_path=build_dir,
                    skip_jac=True,
                    auto_diff=False,
                    platform='intel',
                    data_filename='data.bin'
                    )


            #get file lists
            i_dirs, files = get_file_list(build_dir, lang)

            structs = [file_struct(lang, temp_lang, f, i_dirs,
               [], build_dir, test_dir, not STATIC) for f in files]

            pool = multiprocessing.Pool()
            results = pool.map(compiler, structs)
            pool.close()
            pool.join()
            if any(r == -1 for r in results):
               sys.exit(-1)

            linker(lang, temp_lang, test_dir, files, lib)

            with open(data_output, 'a+') as file:
                for stepsize in todo:
                    for i in range(todo[stepsize]):
                        print(i, "/", todo[stepsize])
                        subprocess.check_call(
                            [os.path.join(the_path,
                            test_dir, 'speedtest'),
                            str(stepsize), str(num_threads)], stdout=file
                            )
