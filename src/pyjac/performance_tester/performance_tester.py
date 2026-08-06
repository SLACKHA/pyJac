"""Module for performance testing of pyJac and related tools."""

# Standard libraries
import itertools
import multiprocessing
import os
import shutil
import subprocess
import sys
from pathlib import Path
from string import Template

import cantera as ct
import numpy as np

# Local imports
from .. import utils
from ..core.create_jacobian import create_jacobian
from ..libgen import compiler, file_struct, flags, generate_library, get_cuda_path, libs

STATIC = True
"""bool: CUDA only works for static libraries"""


def option_cases(*param_sets):
    """Expand parameter dictionaries into concrete option cases.

    Each dictionary maps an option name to either a fixed value or a list of
    values to sweep over. One case is yielded per combination within a set, and
    the sets are visited in the order given. Empty sets are skipped, so a
    backend that is unavailable simply contributes nothing.

    Options missing from a given set read back as ``False``.

    Parameters
    ----------
    *param_sets : dict
        Option dictionaries, values either scalars or lists.

    Yields
    ------
    dict
        One fully-populated set of options per combination.

    """
    all_keys = {key for params in param_sets for key in params}
    for params in param_sets:
        if not params:
            continue
        names = sorted(params)
        values = [
            params[name] if isinstance(params[name], list) else [params[name]]
            for name in names
        ]
        for combination in itertools.product(*values):
            case = dict.fromkeys(all_keys, False)
            case.update(zip(names, combination, strict=True))
            yield case


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
    # checks file for existing data
    # and returns number of runs left to do
    # for each # of does in steplist
    runs = {}
    for step in steplist:
        runs[step] = 0
    if 'cuda' not in filename:
        raise Exception(filename)

    try:
        with open(filename) as file:
            lines = [line.strip() for line in file.readlines()]
        for line in lines:
            try:
                vals = line.split(',')
                if len(vals) == 2:
                    vals = [float(v) for v in vals]
                    runs[vals[0]] += 1
            except (ValueError, IndexError, KeyError):
                pass
        return runs
    except OSError:
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
        with open(filename) as file:
            lines = [line.strip() for line in file.readlines()]
        num_completed = 0
        to_find = 2
        for line in lines:
            try:
                vals = line.split(',')
                if len(vals) == to_find:
                    # parsing both fields confirms the line is well formed
                    int(vals[0])
                    float(vals[1])
                    num_completed += 1
            except (ValueError, IndexError):
                pass
        return num_completed
    except OSError:
        return 0


def getf(x):
    return Path(x).name


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
    if lang == 'icc':
        cmd = ['icc']
    elif lang == 'c':
        cmd = ['gcc']
    elif lang == 'cuda':
        cmd = ['nvcc'] if not shared else ['g++']
    else:
        print('Lang must be one of {icc, c, cuda}')
        raise
    return cmd


def linker(lang, temp_lang, test_dir, filelist, lib=None):
    args = cmd_link(temp_lang, not STATIC)
    if lang == 'cuda' or (not STATIC):
        args.extend(flags[temp_lang])
    args.extend([os.path.join(test_dir, getf(f) + '.o') for f in filelist])
    args.extend(['-o', os.path.join(test_dir, 'speedtest')])
    if temp_lang == 'cuda':
        args.append(f'-L{get_cuda_path()}')
    args.extend(libs[temp_lang])
    if temp_lang != 'cuda':
        args.append('-fopenmp')
    if lang == 'tchem':
        if os.getenv('TCHEM_HOME'):
            tchem_home = os.getenv('TCHEM_HOME')
        else:
            raise SystemError('TCHEM_HOME environment variable not set.')
        args.extend(['-L{}'.format(os.path.join(tchem_home, 'lib')), '-ltchem'])

    if lib is not None:
        if STATIC:
            args += [f'-L{os.getcwd()}']
            args += [f'-l{lib}']
        else:
            args += [lib]

    args.append('-lm')

    try:
        print(' '.join(args))
        subprocess.check_call(args)
    except subprocess.CalledProcessError:
        print('Error: linking of test program failed.')
        sys.exit(1)


def performance_tester(home, work_dir, use_old_opt):
    """Runs performance testing for pyJac, TChem, and finite differences.

    Parameters
    ----------
    home : str
        Directory of source code files
    work_dir : str
        Working directory with mechanisms and for data
    use_old_opt : bool
        If ``True``, use old optimization files found

    Returns
    -------
    None

    """
    build_dir = 'out'
    test_dir = 'test'

    work_dir = os.path.abspath(work_dir)

    # find the mechanisms to test
    mechanism_list = {}
    if not os.path.exists(work_dir):
        print(
            f'Error: work directory {work_dir} for '
            + 'performance testing not found, exiting...'
        )
        sys.exit(-1)
    for name in os.listdir(work_dir):
        if os.path.isdir(os.path.join(work_dir, name)):
            # look for a Cantera YAML mechanism
            files = [
                f
                for f in os.listdir(os.path.join(work_dir, name))
                if os.path.isfile(os.path.join(work_dir, name, f))
            ]
            for f in files:
                if f.endswith(('.yaml', '.yml')):
                    mechanism_list[name] = {}
                    mechanism_list[name]['mech'] = f
                    mechanism_list[name]['chemkin'] = str(Path(f).with_suffix('.dat'))
                    gas = ct.Solution(os.path.join(work_dir, name, f))
                    mechanism_list[name]['ns'] = gas.n_species

                    thermo = next((tf for tf in files if 'therm' in tf), None)
                    if thermo is not None:
                        mechanism_list[name]['thermo'] = thermo

    if len(mechanism_list) == 0:
        print(f'No mechanisms found for performance testing in {work_dir}, exiting...')
        sys.exit(-1)

    if os.getenv('TCHEM_HOME'):
        tchem_home = os.getenv('TCHEM_HOME')
    else:
        raise SystemError('TCHEM_HOME environment variable not set.')

    cpu_repeats = 10
    gpu_repeats = 10

    max_cpu = multiprocessing.cpu_count()
    # thread counts to sweep over, one test case per count
    thread_counts = [1]
    while thread_counts[-1] < max_cpu:
        thread_counts.append(min(max_cpu, thread_counts[-1] * 2))
    c_params = {
        'lang': 'c',
        'cache_opt': [False],
        'finite_diffs': [False, True],
        'num_threads': thread_counts,
    }

    # check that nvcc installed
    cuda_params = {}
    try:
        subprocess.check_call(['nvcc', '--version'])
        # if we have NVCC, assume we can execute CUDA
        cuda_params = {
            'lang': 'cuda',
            'cache_opt': [False],
            'shared': [False, True],
            'finite_diffs': [False, True],
        }
    except OSError:
        # otherwise simply skip cuda
        pass
    # tchem seems not to be openmp parallelizable, nor do we care
    tchem_params = {'lang': 'tchem', 'num_threads': [1]}

    for mech_name, mech_info in sorted(
        mechanism_list.items(), key=lambda x: x[1]['ns']
    ):
        # get the cantera object
        gas = ct.Solution(os.path.join(work_dir, mech_name, mech_info['mech']))

        # ensure directory structure is valid
        os.chdir(os.path.join(work_dir, mech_name))
        subprocess.check_call(['mkdir', '-p', build_dir])
        subprocess.check_call(['mkdir', '-p', test_dir])

        num_conditions = 0
        npy_files = [
            f
            for f in os.listdir(os.path.join(work_dir, mech_name))
            if f.endswith('.npy') and os.path.isfile(f)
        ]
        data = None
        with open('data.bin', 'wb') as file:
            # load PaSR data for different pressures/conditions,
            # and save to binary C file
            for npy in sorted(npy_files):
                state_data = np.load(npy)
                state_data = state_data.reshape(
                    state_data.shape[0] * state_data.shape[1], state_data.shape[2]
                )
                if data is None:
                    data = state_data
                else:
                    data = np.vstack((data, state_data))
                num_conditions += state_data.shape[0]
                print(num_conditions, data.shape)
            if num_conditions == 0:
                print(f'No data found in folder {mech_name}, continuing...')
                continue
            data.tofile(file)

        # figure out gpu steps
        step_size = 1
        steplist = []
        while step_size < num_conditions:
            steplist.append(step_size)
            step_size *= 2
        if step_size / 2 != num_conditions:
            steplist.append(num_conditions)

        the_path = os.getcwd()
        op = option_cases(c_params, cuda_params, tchem_params)

        haveOpt = False
        if os.path.isfile(os.path.join(os.getcwd(), build_dir, 'optimized.pickle')):
            haveOpt = True

        for state in op:
            lang = state['lang']
            temp_lang = 'c' if lang != 'cuda' else 'cuda'
            FD = state['finite_diffs']
            if FD:
                filename = f'fd_jacob{utils.file_ext[temp_lang]}'
                shutil.copy(
                    os.path.join(home, filename), os.path.join(build_dir, filename)
                )

            opt = state['cache_opt']
            smem = state['shared']

            # CUDA cases do not sweep thread counts; -1 selects the test
            # binary's own default
            num_threads = state['num_threads'] or -1

            if lang == 'tchem' and any(
                utils.is_plog_or_cheb(rxn) for rxn in gas.reactions()
            ):
                print(
                    'TChem performance evaluation disabled; '
                    'not compatible with Plog or Chebyshev reactions.'
                )
                continue

            data_output = (
                '{}_{}_{}_{}_{}'.format(
                    lang,
                    'co' if opt else 'nco',
                    'smem' if smem else 'nosmem',
                    'fd' if FD else 'ajac',
                    num_threads,
                )
                + '_output.txt'
            )

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
                create_jacobian(
                    lang,
                    mech_info['mech'],
                    optimize_cache=opt,
                    build_path=build_dir,
                    no_shared=not smem,
                    num_blocks=8,
                    num_threads=64,
                    multi_thread=multiprocessing.cpu_count(),
                )

            # now we need to write the reader
            filename = f'read_initial_conditions{utils.file_ext[temp_lang]}'
            shutil.copy(
                os.path.join(home, filename),
                os.path.join(os.getcwd(), build_dir, filename),
            )

            # write the tester
            file_data = {'datafile': os.path.join(the_path, 'data.bin')}
            if lang == 'c' or lang == 'cuda':
                filename = f'tester{utils.file_ext[temp_lang]}.in'
                with open(os.path.join(home, filename)) as file:
                    src = Template(file.read())
                src = src.substitute(file_data)
            else:
                file_data['mechfile'] = mech_info['chemkin']
                if 'thermo' in mech_info:
                    file_data['thermofile'] = mech_info['thermo']
                else:
                    # it's the same file
                    file_data['thermofile'] = mech_info['chemkin']
                with open(os.path.join(home, 'tc_tester.c.in')) as file:
                    src = Template(file.read())
                src = src.substitute(file_data)
            filename = f'test{utils.file_ext[temp_lang]}'
            with open(os.path.join(build_dir, filename), 'w') as file:
                file.write(src)

            # copy timer
            shutil.copy(
                os.path.join(home, 'timer.h'),
                os.path.join(os.getcwd(), build_dir, 'timer.h'),
            )

            # get file lists
            i_dirs = [build_dir]
            files = ['test', 'read_initial_conditions']

            lib = None
            # now build the library
            if lang != 'tchem':
                lib = generate_library(
                    lang, build_dir, test_dir, finite_difference=FD, shared=not STATIC
                )

                lib = os.path.normpath(lib)
                lib = lib[
                    lib.index('lib') + len('lib') : lib.index(
                        '.so' if not STATIC else '.a'
                    )
                ]
            else:
                files += ['mechanism', 'mass_mole']

            # Compile generated source code
            structs = [
                file_struct(
                    lang,
                    temp_lang,
                    f,
                    i_dirs,
                    (['-DFINITE_DIFF'] if FD else []),
                    build_dir,
                    test_dir,
                    not STATIC,
                )
                for f in files
            ]
            if lang != 'cuda':
                for s in structs:
                    s.args.append('-fopenmp')

            pool = multiprocessing.Pool()
            results = pool.map(compiler, structs)
            pool.close()
            pool.join()
            if any(r == -1 for r in results):
                sys.exit(-1)

            linker(lang, temp_lang, test_dir, files, lib)

            if lang == 'tchem':
                # copy periodic table and mechanisms in
                shutil.copy(
                    os.path.join(tchem_home, 'data', 'periodictable.dat'),
                    'periodictable.dat',
                )

            with open(data_output, 'a+') as file:
                for stepsize in todo:
                    for i in range(todo[stepsize]):
                        print(i, '/', todo[stepsize])
                        subprocess.check_call(
                            [
                                os.path.join(the_path, test_dir, 'speedtest'),
                                str(stepsize),
                                str(num_threads),
                            ],
                            stdout=file,
                        )
