#! /usr/bin/env python2.7

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import sys
import subprocess
from argparse import ArgumentParser

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
from create_jacobian import create_jacobian
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

def write_c_tester(file):
    file.write(
    """
    void read_initial_conditions(const char* filename, int NUM, double** y_host, double** variable_host);
    #include "jacob.h"
    #include "timer.h"
    #include "header.h"
    #include <stdio.h>
    int main (int argc, char *argv[])
    {
        int max_threads = omp_get_max_threads ();
        int num_threads = 1;
        if (sscanf(argv[1], "%i", &num_threads) !=1 || (num_threads <= 0) || (num_threads > max_threads)) {
                exit(1);
        }
        omp_set_num_threads (num_threads);
        int num_odes = 1;
        if (sscanf(argv[2], "%i", &num_odes) !=1 || (num_odes <= 0))
        {
                exit(1);
        }
        double* y_host;
        double* var_host;
        read_initial_conditions("data.bin", num_odes, &y_host, &var_host);
        StartTimer();
        #pragma omp parallel for shared(y_host, var_host)
        for(int tid = 0; tid < num_odes; ++tid)
        {
            double y_local[NN] = {0};
            double jac[NN * NN] = {0};
            #pragma unroll
            for (int i = 0; i < NN; ++i)
            {
                y_local[i] = y_host[tid + i * num_odes];
            }
            eval_jacob(0, var_host[tid], y_local, jac);
        }
        double runtime = GetTimer();
        free(y_host);
        free(var_host);
        printf("%d,%d,%.15le", num_threads, num_odes, runtime);
        return 0;
    }
    """
        )
def write_cuda_tester(file):
    file.write(
    """
    #include <stdlib.h>
    #include <stdio.h>
    #include <math.h>
    #include <string.h>

    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <helper_cuda.h>
    #include "header.h"
    #include "timer.h"
    //get our solver stuff
    #include "solver.cuh"
    #include "gpu_memory.cuh"
    #include "read_initial_conditions.cuh"
    #include "launch_bounds.cuh"

    omp_set_num_threads (num_threads);
    int num_odes = 1;
    if (sscanf(argv[1], "%i", &num_odes) !=1 || (num_odes <= 0))
    {
        exit(1);
    }

    cudaErrorCheck (cudaSetDevice (0) );
    cudaErrorCheck (cudaGetDeviceProperties(&devProp, 0));

    #ifdef PREFERL1
        cudaErrorCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    #endif

    int g_num = (int)ceil(((double)NUM) / ((double)TARGET_BLOCK_SIZE));
    if (g_num == 0)
        g_num = 1;

    double* y_device;
    double* y_host;
    double* vaa_device;
    double* var_host;
    int padded = read_initial_conditions("data.bin", num_odes, TARGET_BLOCK_SIZE, g_num, &y_host, &y_device, &var_host, &var_device);
    dim3 dimGrid (g_num, 1 );
    dim3 dimBlock(TARGET_BLOCK_SIZE, 1);
    StartTimer();
    cudaErrorCheck( cudaMemcpy (var_device, var_host, padded * sizeof(double), cudaMemcpyHostToDevice));
#ifdef SHARED_SIZE
        intDriver <<< dimGrid, dimBlock, SHARED_SIZE >>> (NUM, t, t_next, var_device, y_device);
#else
        intDriver <<< dimGrid, dimBlock >>> (NUM, t, t_next, var_device, y_device);
#endif
    // transfer memory back to CPU
    cudaErrorCheck( cudaMemcpy (y_host, y_device, padded * NN * sizeof(double), cudaMemcpyDeviceToHost) );
    double runtime = GetTimer();
    cudaErrorCheck( cudaPeekAtLastError() );
    cudaErrorCheck( cudaDeviceSynchronize() );
    free_gpu_memory(y_device, var_device);
    free(y_host);
    free(var_host);
    cudaErrorCheck( cudaDeviceReset() );
    printf("%d,%.15le", num_odes, runtime);
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

libs = dict(c=['-O3', '-lm', '-std=c99', '-fopenmp'],
            cuda=['-arch=sm_20', '-03'])

mechanism_dir = '~/mechs/'
mechanism_list = [{'name':'H2', 'mech':'chem.cti', 'input':'pasr_input_h2.yaml'},
                  {'name':'GRI', 'mech':'grimech30.cti', 'input':'pasr_input_ch4.yaml'},
                  {'name':'USC', 'mech':'uscmech.cti', 'input':'pasr_input_usc.yaml'}]

pressure_list = [1, 10, 25]
temp_list = [400, 600, 800]
#premixed = ['premixed', 'non-premixed']

cache_opt = [True, False]
shared = [True, False]
num_threads = [1, 12]

repeats = 1000
home = os.getcwd() + os.path.sep
build_dir = 'out/'
test_dir = 'test/'

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

    #do c
    #next we need to start writing the jacobians
    for opt in cache_opt:
        for thread in num_threads:
            create_jacobian('c', mechanism_dir+mechanism['mech'],
                            optimize_cache=opt, multi_thread=12, build_path=build_dir)

            #now we need to write the reader and the tester
            with open(build_dir + 'read_initial_conditions.c', 'w') as file:
                write_c_reader(file)

            with open(build_dir + 'test.c', 'w') as file:
                write_c_tester(file)

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

            for f in files:
                args = [cmd_compile['c']]
                args.extend(flags['c'])
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
                    sys.excreate_jacobian('c', mechanism_dir+mechanism['mech'],
                            optimize_cache=opt, multi_thread=12, build_path=build_dir)

            # Link into executable
            args = [cmd_compile['c']]
            args.extend([os.path.join(test_dir, f + '.o') for f in files])
            args.extend(['-o', os.path.join(test_dir, 'speedtest')])
            args.extend(libs['c'])

            try:
                subprocess.check_call(args)
            except subprocess.CalledProcessError:
                print('Error: linking of test program failed.')
                sys.exit(1)

            with open('{}output.txt'.format('co_' if cache_opt else 'nco_'), 'w') as file:
                for i in range(repeats):
                    print(i, "/", repeats)
                    subprocess.check_call([os.path.join(os.path.join(os.getcwd(), test_dir), 'speedtest'),
                     str(thread), str(num_conditions)], stdout=file)

    #do cuda
    #next we need to start writing the jacobians
    for opt in cache_opt:
        for smem in shared:
            create_jacobian('cuda', mechanism_dir+mechanism['mech'],
                            optimize_cache=opt, multi_thread=12, 
                            build_path=build_dir,
                            num_blocks=8 num_threads=64)

            with open('out/regcount') as file:
                regcount = int(file.readline())

            #now we need to write the reader and the tester
            with open(build_dir + 'read_initial_conditions.cu', 'w') as file:
                write_cuda_reader(file)

            with open(build_dir + 'test.cu', 'w') as file:
                write_cuda_tester(file)

            write_timer()

            #get the cantera object
            gas = ct.Solution(mechanism_dir+mechanism['mech'])
            pmod = any([__is_pdep(rxn) for rxn in gas.reactions()])

            # Compile generated source code
            files = ['chem_utils', 'dydt', 'jacob', 'spec_rates',
                     'rxn_rates', 'test', 'read_initial_conditions',
                     'mechanism', 'mass_mole', 'gpu_memory'
                     ]
            if pmod:
                files += ['rxn_rates_pres_mod']

            for f in files:
                args = [cmd_compile['cuda']]
                args.extend(flags['cuda'])
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
                    sys.excreate_jacobian('c', mechanism_dir+mechanism['mech'],
                            optimize_cache=opt, multi_thread=12, build_path=build_dir)

            # Link into executable
            args = [cmd_compile['cuda']]
            args.extend([os.path.join(test_dir, f + '.o') for f in files])
            args.extend(['-o', os.path.join(test_dir, 'speedtest')])
            args.extend(libs['cuda'])

            try:
                subprocess.check_call(args)
            except subprocess.CalledProcessError:
                print('Error: linking of test program failed.')
                sys.exit(1)

            with open('gpu_{}_{}_output.txt'.format('co_' if cache_opt else 'nco_',
                'sm' if smem else 'nsm'), 'w') as file:
                for i in range(repeats):
                    print(i, "/", repeats)
                    subprocess.check_call([os.path.join(os.path.join(os.getcwd(), test_dir), 'speedtest'),
                     str(thread), str(num_conditions)], stdout=file)