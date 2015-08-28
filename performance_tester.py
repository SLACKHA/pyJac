#! /usr/bin/env python2.7

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import sys
import subprocess
from argparse import ArgumentParser
import multiprocessing
import shutil

# More Python 2 compatibility
if sys.version_info.major == 3:
    from itertools import zip
elif sys.version_info.major == 2:
    from itertools import izip as zip

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

def __is_pdep(rxn):
    return (isinstance(rxn, ct.ThreeBodyReaction) or
    isinstance(rxn, ct.FalloffReaction) or
    isinstance(rxn, ct.ChemicallyActivatedReaction))

def run_pasr(pasr_input, mech_filename, pasr_output_file):
    #try to load output file
    try:
        state_data = np.load(pasr_output_file)
    except:
        # Run PaSR to get data
        state_data = pasr.run_simulation(
                        mech_filename,
                        pasr_input['case'],
                        pasr_input['temperature'],
                        pasr_input['pressure'],
                        pasr_input['equivalence ratio'],
                        pasr_input['fuel'],
                        pasr_input['oxidizer'],
                        pasr_input['complete products'],
                        pasr_input['number of particles'],
                        pasr_input['residence time'],
                        pasr_input['mixing time'],
                        pasr_input['pairing time'],
                        pasr_input['number of residence times']
                        )
        if pasr_output_file:
            np.save(pasr_output_file, state_data)
    return state_data

def write_timer():
    with open('out/timer.h', 'w') as file:
        file.write(
"""
#ifndef TIMER_H
#define TIMER_H

#include <stdlib.h>

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
 #ifndef __USE_BSD
 #define __USE_BSD
 #endif
#include <time.h>
#include <sys/time.h>
#endif

#ifdef WIN32
double PCFreq = 0.0;
__int64 timerStart = 0;
#else
struct timeval timerStart;
#endif

void StartTimer()
{
#ifdef WIN32
    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
        printf("QueryPerformanceFrequency failed!\\n");

    PCFreq = (double)li.QuadPart/1000.0;

    QueryPerformanceCounter(&li);
    timerStart = li.QuadPart;
#else
    gettimeofday(&timerStart, NULL);
#endif
}

// time elapsed in ms
double GetTimer()
{
#ifdef WIN32
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (double)(li.QuadPart-timerStart)/PCFreq;
#else
    struct timeval timerStop, timerElapsed;
    gettimeofday(&timerStop, NULL);
    timersub(&timerStop, &timerStart, &timerElapsed);
    return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
#endif
}

#endif // TIMER_H
        """
            )

def write_cuda_reader(file):
    file.write(
    """
#include "header.h"
#include "gpu_memory.cuh"
#include "gpu_macros.cuh"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

 int read_initial_conditions(const char* filename, int NUM, int block_size, int grid_size, double** y_host, double** y_device, double** variable_host, double** variable_device) {
    int padded = initialize_gpu_memory(NUM, block_size, grid_size, y_device, variable_device);
    (*y_host) = (double*)malloc(padded * NN * sizeof(double));
    (*variable_host) = (double*)malloc(padded * sizeof(double));
    FILE *fp = fopen (filename, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "Could not open file: %s\\n", filename);
        exit(1);
    }
    double buffer[NN + 2];

    // load temperature and mass fractions for all threads (cells)
    for (int i = 0; i < NUM; ++i)
    {
        // read line from data file
        int count = fread(buffer, sizeof(double), NN + 2, fp);
        if (count != (NN + 2))
        {
            fprintf(stderr, "File (%s) is incorrectly formatted, %d doubles were expected but only %d were read.\\n", filename, NN + 1, count);
            exit(-1);
        }
        //apply mask if necessary
        apply_mask(&buffer[3]);
        //put into y_host
        (*y_host)[i] = buffer[1];
#ifdef CONP
        (*variable_host)[i] = buffer[2];
#elif CONV
        double pres = buffer[2];
#endif
        for (int j = 2; j <= NN; j++)
            (*y_host)[i + (j - 1) * padded] = buffer[j + 1];

        // if constant volume, calculate density
#ifdef CONV
        double Yi[NSP];
        double Xi[NSP];

        for (int j = 1; j < NN; ++j)
        {
            Yi[j - 1] = (*y_host)[i + j * padded];
        }

        mass2mole (Yi, Xi);
        (*variable_host)[i] = getDensity ((*y_host)[i], pres, Xi);
#endif
    }
    fclose (fp);
    return padded;
}
    """
    )

def write_c_reader(file):
    file.write(
    """
#include "header.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

 void read_initial_conditions(const char* filename, int NUM, double** y_host, double** variable_host) {
    (*y_host) = (double*)malloc(NUM * NN * sizeof(double));
    (*variable_host) = (double*)malloc(NUM * sizeof(double));
    FILE *fp = fopen (filename, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "Could not open file: %s\\\n", filename);
        exit(-1);
    }
    double buffer[NN + 2];

    // load temperature and mass fractions for all threads (cells)
    for (int i = 0; i < NUM; ++i)
    {
        // read line from data file
        int count = fread(buffer, sizeof(double), NN + 2, fp);
        if (count != (NN + 2))
        {
            fprintf(stderr, "File (%s) is incorrectly formatted, %d doubles were expected but only %d were read.\\n", filename, NN + 1, count);
            exit(-1);
        }
        //apply mask if necessary
        apply_mask(&buffer[3]);
        //put into y_host
        (*y_host)[i] = buffer[1];
#ifdef CONP
        (*variable_host)[i] = buffer[2];
#elif CONV
        double pres = buffer[2];
#endif
        for (int j = 2; j <= NN; j++)
            (*y_host)[i + (j - 1) * NUM] = buffer[j + 1];

        // if constant volume, calculate density
#ifdef CONV
        double Yi[NSP];
        double Xi[NSP];

        for (int j = 1; j < NN; ++j)
        {
            Yi[j - 1] = (*y_host)[i + j * NUM];
        }

        mass2mole (Yi, Xi);
        (*variable_host)[i] = getDensity ((*y_host)[i], pres, Xi);
#endif
    }
    fclose (fp);
}
    """
    )

def write_tc_tester(file, path, mechfile, thermofile):
    the_file = path + "data.bin"
    if thermofile == None:
        thermofile = mechfile
    file.write(
    """

    #include <stdio.h>
    #include "header.h"
    #include "timer.h"

    void read_initial_conditions(const char* filename, int NUM, double** y_host, double** variable_host);
   

    void chemjac(double t, double* y, double *jac) 
    {

      int i, j, iJac, Nv, Ns ;
      double sumY ;
      unsigned int useJacAnl = 1 ; /* use analytic Jacobian */

      TC_getJacTYN ( y, NSP, jac, useJacAnl ) ;
    }

    int main(int argc, char *argv[])
    {
    """
    )
    file.write("""
      char mechfile[100] = "{}";
      char thermofile[100] = "{}";
      """.format(mechfile, thermofile)
        )
    file.write(
    """
      int num_odes = 1;
      if (sscanf(argv[1], "%i", &num_odes) !=1 || (num_odes <= 0))
      {
          exit(-1);
      }

      double* y_host;
      double* var_host;
      """
    )
    file.write(
    """
      read_initial_conditions("{}", num_odes, &y_host, &var_host);
    """.format(the_file))
    file.write(
    """

      /* Initialize TC library */
      TC_initChem( mechfile, thermofile, (int) 0, 1.0) ; 
      
      StartTimer();


      for(int tid = 0; tid < num_odes; ++tid)
      {
          TC_setThermoPres(var_host[tid]) ;
          double y_local[NN];
          double jac[NN * NN] = {0};
          #pragma unroll
          for (int i = 0; i < NN; ++i)
          {
              y_local[i] = y_host[tid + i * num_odes];
          }
          chemjac(0, y_local, jac);
          for (int i = 0; i < NN * NN; ++i)
          {
              printf("%.15le\\n", jac[i]);
          }
      }
      double runtime = GetTimer();
      printf("%d,%.15le\\n", num_odes, runtime);
      free(y_host);
      free(var_host);
      return 0;
    }
    """
    )

def write_c_tester(file, path):
    the_file = path + "data.bin"
    file.write(
    """
    void read_initial_conditions(const char* filename, int NUM, double** y_host, double** variable_host);
    #include "jacob.h"
    #include "timer.h"
    #include "header.h"
    #include <stdio.h>
    int main (int argc, char *argv[])
    {
        int num_odes = 1;
        if (sscanf(argv[1], "%i", &num_odes) !=1 || (num_odes <= 0))
        {
            exit(-1);
        }
        double* y_host;
        double* var_host;
    """
    )
    file.write("""
        read_initial_conditions("{}", num_odes, &y_host, &var_host);
        """.format(the_file))
    file.write(
    """
        StartTimer();
        for(int tid = 0; tid < num_odes; ++tid)
        {
            double y_local[NN];
            double jac[NN * NN] = {0};
            #pragma unroll
            for (int i = 0; i < NN; ++i)
            {
                y_local[i] = y_host[tid + i * num_odes];
            }
            eval_jacob(0, var_host[tid], y_local, jac);
        }
        double runtime = GetTimer();
        printf("%d,%.15le\\n", num_odes, runtime);
        free(y_host);
        free(var_host);
        return 0;
    }
    """
        )
def write_cuda_tester(file, path):
    the_file = path + "data.bin"
    file.write(
    """
    #include <stdlib.h>
    #include <stdio.h>
    #include <string.h>
    #include "jacob.cuh"


    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <helper_cuda.h>
    #include "header.h"
    #include "timer.h"
    #include "gpu_memory.cuh"
    #include "launch_bounds.cuh"

    int read_initial_conditions(const char* filename, int NUM, int block_size, int grid_size, double** y_host, double** y_device, double** variable_host, double** variable_device);

    #define T_ID (threadIdx.x + (blockDim.x * blockIdx.x))
    #define GRID_SIZE (blockDim.x * gridDim.x)
    __global__ 
    void jac_driver(int NUM, double* pres, double* y)
    {
        if (T_ID < NUM)
        {
            double y_local[NN];
            double pr_local = pres[T_ID];
            double jac_local[NN * NN] = {0};

        #pragma unroll
            for (int i = 0; i < NN; i++)
            {
                y_local[i] = y[T_ID + i * GRID_SIZE];
            }

            eval_jacob (0, pr_local, y_local, jac_local);
        }
    }

    int main (int argc, char *argv[])
    {
        int num_odes = 1;
        if (sscanf(argv[1], "%i", &num_odes) !=1 || (num_odes <= 0))
        {
            exit(1);
        }

        cudaErrorCheck (cudaSetDevice (0) );

        cudaErrorCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

        int g_num = (int)ceil(((double)num_odes) / ((double)TARGET_BLOCK_SIZE));
        if (g_num == 0)
            g_num = 1;

        double* y_device;
        double* y_host;
        double* var_device;
        double* var_host;
            """
    )
    file.write("""
        int padded = read_initial_conditions("{}", num_odes, TARGET_BLOCK_SIZE, g_num, &y_host, &y_device, &var_host, &var_device);
        """.format(the_file))
    file.write(
    """
        g_num = (int)ceil(((double)padded) / ((double)TARGET_BLOCK_SIZE));
        if (g_num == 0)
            g_num = 1;
        dim3 dimGrid (g_num, 1 );
        dim3 dimBlock(TARGET_BLOCK_SIZE, 1);
        cudaErrorCheck( cudaMemcpy (var_device, var_host, padded * sizeof(double), cudaMemcpyHostToDevice));
        cudaErrorCheck( cudaMemcpy (y_device, y_host, padded * NN * sizeof(double), cudaMemcpyHostToDevice));
        StartTimer();
        #ifdef SHARED_SIZE
            jac_driver <<< dimGrid, dimBlock, SHARED_SIZE >>> (num_odes, var_device, y_device);
        #else
            jac_driver <<< dimGrid, dimBlock >>> (num_odes, var_device, y_device);
        #endif
        cudaDeviceSynchronize();
        double runtime = GetTimer();
        printf("%d,%.15le\\n", num_odes, runtime);
        cudaErrorCheck( cudaPeekAtLastError() );
        cudaErrorCheck( cudaDeviceSynchronize() );
        free_gpu_memory(y_device, var_device);
        free(y_host);
        free(var_host);
        cudaErrorCheck( cudaDeviceReset() );
        return 0;
    }
    """
    )

# Compiler based on language
cmd_compile = dict(c='gcc',
                   cuda='nvcc',
                   fortran='gfortran'
                   )

# Flags based on language
flags = dict(c=['-std=c99', '-O3', '-mtune=native',
                '-fopenmp'],
             cuda=['-O3', '-arch=sm_20',
                   '-I/usr/local/cuda/include/',
                   '-I/usr/local/cuda/samples/common/inc/',
                   '-dc'])

libs = dict(c=['-lm', '-std=c99', '-fopenmp'],
            cuda=['-arch=sm_20'])

mechanism_dir = '/home/nick/mechs/'
mechanism_list = [{'name':'H2', 'mech':'chem.cti', 'input':'pasr_input_h2.yaml', 'chemkin':'h2.dat', 'thermo':'h2therm.dat'},
                  {'name':'GRI', 'mech':'grimech30.cti', 'input':'pasr_input_ch4.yaml', 'chemkin':'grimech30.dat', 'thermo':'thermo30.dat'},
                  {'name':'USC', 'mech':'uscmech.cti', 'input':'pasr_input_c2h4.yaml', 'chemkin':'uscmech.dat', 'thermo':'usctherm.dat'}]

pressure_list = [1, 10, 25]
temp_list = [400, 600, 800]
#premixed = ['premixed', 'non-premixed']

cache_opt = [True, False]
shared = [True, False]
num_threads = [1]#[1, 12]

repeats = 20
gpu_repeats = 100
home = os.getcwd() + os.path.sep
build_dir = 'out/'
test_dir = 'test/'

def check_step_file(filename, steplist):
    #checks file for existing data
    #and returns number of runs left to do 
    #for each # of does in steplist
    runs = {}
    for step in steplist:
        runs[step] = 0
    if not 'gpu' in filename:
        raise Exception

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


#make sure the performance directory exists
subprocess.check_call(['mkdir', '-p', 'performance'])

for mechanism in mechanism_list:
    os.chdir(home + 'performance')
    subprocess.check_call(['mkdir', '-p', mechanism['name']])
    os.chdir(mechanism['name'])
    subprocess.check_call(['mkdir', '-p', build_dir])
    subprocess.check_call(['mkdir', '-p', test_dir])
    #get input
    pasr_input = pasr.parse_input_file(home + mechanism['input'])
    
    with open("test/data.bin", "wb") as file:
        pass

    num_conditions = 0
    #generate PaSR data for different pressures, and save to binary c file
    index = 0
    for pressure in pressure_list:
        for temperature in temp_list:
            #for premix in premixed:
            pasr_input['pressure'] = pressure
            pasr_input['temperature'] = temperature
            #pasr_input['premixed'] = premix
            state_data = run_pasr(pasr_input, mechanism_dir+mechanism['mech'], 'pasr_out_{}.npy'.format(index))
            state_data = state_data.reshape(state_data.shape[0] * state_data.shape[1],
                                state_data.shape[2]
                                )
            with open("test/data.bin", "ab") as file:
                state_data.tofile(file)
            num_conditions += state_data.shape[0]
            print(num_conditions)
            index += 1

    the_path = os.path.join(os.getcwd(), test_dir)
    #do c
    #next we need to start writing the jacobians
    for opt in cache_opt:
        for thread in num_threads:
            data_output = 'cpu_{}_{}_output.txt'.format('co' if opt else 'nco', thread)
            data_output = os.path.join(os.getcwd(), data_output)
            num_completed = check_file(data_output)
            if num_completed >= repeats:
                continue
            create_jacobian('c', mechanism_dir+mechanism['mech'],
                            optimize_cache=opt, multi_thread=12, build_path=build_dir)

            #now we need to write the reader and the tester
            with open(build_dir + 'read_initial_conditions.c', 'w') as file:
                write_c_reader(file)

            with open(build_dir + 'test.c', 'w') as file:
                write_c_tester(file, the_path)

            write_timer()

            #get the cantera object
            gas = ct.Solution(mechanism_dir+mechanism['mech'])
            pmod = any([__is_pdep(rxn) for rxn in gas.reactions()])

            # Compile generated source code
            files = ['chem_utils', 'dydt', 'jacob', 'spec_rates',
                     'rxn_rates', 'test', 'read_initial_conditions',
                     'mechanism', 'mass_mole'
                     ]
            if pmod:
                files += ['rxn_rates_pres_mod']

            i_dirs = [build_dir]
            try:
                with open('out/jacobs/jac_list_c') as file:
                    vals = file.readline().strip().split(' ')
                    vals = ['jacobs/' + f[:f.index('.c')] for f in vals]
                    files = vals + files
                    i_dirs.append('out/jacobs/')
            except:
                pass


            getf = lambda x: x[x.index('/') + 1:] \
                                if 'jacobs/' in x else x

            def compiler(f):
                args = [cmd_compile['c']]
                args.extend(flags['c'])
                include = ['-I./' + d for d in i_dirs]
                args.extend(include)
                args.extend([
                    '-I.' + os.path.sep + build_dir,
                    '-c', os.path.join(build_dir, f + utils.file_ext['c']),
                    '-o', os.path.join(test_dir, getf(f) + '.o')
                    ])
                args = [val for val in args if val.strip()]
                try:
                    subprocess.check_call(args)
                except subprocess.CalledProcessError:
                    print('Error: compilation failed for ' + f + utils.file_ext['c'])
                    return -1
                return 0

            pool = multiprocessing.Pool()
            results = pool.map(compiler, files)
            pool.close()
            pool.join()
            if any(r == -1 for r in results):
                sys.exit(-1)

            # Link into executable
            args = [cmd_compile['c']]
            args.extend([os.path.join(test_dir, getf(f) + '.o') for f in files])
            args.extend(['-o', os.path.join(test_dir, 'speedtest')])
            args.extend(libs['c'])

            try:
                subprocess.check_call(args)
            except subprocess.CalledProcessError:
                print('Error: linking of test program failed.')
                sys.exit(1)

            with open(data_output, 'a+') as file:
                for i in range(repeats - num_completed):
                    print(i, '/', repeats - num_completed)
                    subprocess.check_call([os.path.join(the_path, 'speedtest'),
                     str(num_conditions)], stdout=file)

    #do cuda
    #next we need to start writing the jacobians
    for opt in cache_opt:
        for smem in shared:

            #do steps
            step_size = 1
            steplist = []
            while step_size < num_conditions:
                steplist.append(step_size)
                step_size *= 2
            if step_size / 2 != num_conditions:
                steplist.append(num_conditions)

            data_output = 'gpu_{}_{}_output_steps.txt'.format('co' if opt else 'nco',
                'sm' if smem else 'nsm')
            data_output = os.path.join(os.getcwd(), data_output)
            todo = check_step_file(data_output, steplist)
            if all(todo[x] >= gpu_repeats for x in todo):
                continue

            create_jacobian('cuda', mechanism_dir+mechanism['mech'],
                            optimize_cache=opt, multi_thread=12, 
                            build_path=build_dir,
                            num_blocks=8, num_threads=64)

            with open('out/regcount') as file:
                regcount = int(file.readline())

            #now we need to write the reader and the tester
            with open(build_dir + 'read_initial_conditions.cu', 'w') as file:
                write_cuda_reader(file)

            with open(build_dir + 'test.cu', 'w') as file:
                write_cuda_tester(file, the_path)

            write_timer()

            #get the cantera object
            gas = ct.Solution(mechanism_dir+mechanism['mech'])
            pmod = any([__is_pdep(rxn) for rxn in gas.reactions()])

            # Compile generated source code
            files = ['chem_utils', 'dydt', 'jacob', 'spec_rates',
                     'rxn_rates', 'test', 'read_initial_conditions',
                     'mechanism', 'mass_mole', 'gpu_memory'
                     ]
            i_dirs = [build_dir]
            try:
                with open('out/jacobs/jac_list_cuda') as file:
                    vals = file.readline().strip().split(' ')
                    vals = ['jacobs/' + f[:f.index('.cu')] for f in vals]
                    files = vals + files
                    i_dirs.append('out/jacobs/')
            except:
                pass
            if pmod:
                files += ['rxn_rates_pres_mod']

            ext = lambda x: utils.file_ext['cuda'] if x != 'mass_mole' else \
                                utils.file_ext['c']
            getf = lambda x: x[x.index('/') + 1:] \
                                if 'jacobs/' in x else x

            def compiler(f):
                args = [cmd_compile['cuda']]
                args.extend(['-maxrregcount', str(regcount)])
                args.extend(flags['cuda'])
                include = ['-I./' + d for d in i_dirs]
                for i in include:
                    args.insert(-1, i)
                args.extend([
                    '-o', os.path.join(test_dir, getf(f)) + '.o',
                    os.path.join(build_dir, f + ext(f))
                    ])
                args = [val for val in args if val.strip()]
                try:
                    subprocess.check_call(args)
                except subprocess.CalledProcessError:
                    print('Error: compilation failed for ' + f + utils.file_ext['cuda'])
                    return -1
                return 0
            
            pool = multiprocessing.Pool()
            results = pool.map(compiler, files)
            pool.close()
            pool.join()
            if any(r == -1 for r in results):
                sys.exit(-1)

            # Link into executable
            args = [cmd_compile['cuda']]
            args.extend([os.path.join(os.getcwd(), test_dir, getf(f) + '.o') for f in files])
            args.extend(['-o', os.path.join(test_dir, 'speedtest')])
            args.extend(libs['cuda'])

            try:
                subprocess.check_call(args)
            except subprocess.CalledProcessError:
                print('Error: linking of test program failed.')
                sys.exit(1)

            with open(data_output, 'a+') as file:
                for stepsize in todo:
                    for i in range(gpu_repeats - todo[stepsize]):
                        print(i, "/", gpu_repeats - todo[stepsize])
                        subprocess.check_call([os.path.join(the_path, 'speedtest'),
                         str(stepsize)], stdout=file)
    #finally tc tester
    data_output = 'tchem_output.txt'
    data_output = os.path.join(os.getcwd(), data_output)
    num_completed = check_file(data_output)
    #now we need to write the reader and the tester
    with open(build_dir + 'read_initial_conditions.c', 'w') as file:
        write_c_reader(file)

    with open(build_dir + 'tc_test.c', 'w') as file:
        write_tc_tester(file, the_path, mechanism['chemkin'], mechanism['thermo'])

    write_timer()

    # Compile generated source code
    files = ['tc_test', 'read_initial_conditions', 'mechanism', 'mass_mole']

    i_dirs = [build_dir]

    def compiler(f):
        args = [cmd_compile['c']]
        args.extend(flags['c'])
        include = ['-I./' + d for d in i_dirs]
        args.extend(include)
        args.extend([
            '-I.' + os.path.sep + build_dir,
            '-c', os.path.join(build_dir, f + utils.file_ext['c']),
            '-o', os.path.join(test_dir, f + '.o')
            ])
        args = [val for val in args if val.strip()]
        try:
            subprocess.check_call(args)
        except subprocess.CalledProcessError:
            print('Error: compilation failed for ' + f + utils.file_ext['c'])
            return -1
        return 0

    pool = multiprocessing.Pool()
    results = pool.map(compiler, files)
    pool.close()
    pool.join()
    if any(r == -1 for r in results):
        sys.exit(-1)

    # Link into executable
    args = [cmd_compile['c']]
    args.extend([os.path.join(test_dir, f+ '.o') for f in files])
    args.extend(['-o', os.path.join(test_dir, 'speedtest')])
    args.extend(libs['c'] + ['-L' + os.path.join(home, 'TChem_v0.2', 'lib'), '-ltchem'])

    try:
        subprocess.check_call(args)
    except subprocess.CalledProcessError:
        print('Error: linking of test program failed.')
        sys.exit(1)

    #copy periodic table and mechanisms in
    shutil.copy(os.path.join(home, 'TChem_v0.2', 'data', 'periodictable.dat'), os.path.join(test_dir, 'periodictable.dat'))
    shutil.copy(os.path.join(mechanism_dir, mechanism['chemkin']), mechanism['chemkin'])
    shutil.copy(os.path.join(mechanism_dir, mechanism['thermo']), mechanism['thermo'])

    with open(data_output, 'a+') as file:
        for i in range(repeats - num_completed):
            print(i, '/', repeats - num_completed)
            subprocess.check_call([os.path.join(the_path, 'speedtest'),
             str(num_conditions)], stdout=file)