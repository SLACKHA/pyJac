/* wrapper to translate to cuda arrays */

#include "header.cuh"
#include "gpu_memory.cuh"
#include "launch_bounds.cuh"
#include "chem_utils.cuh"
#include "rates.cuh"
#include "jacob.cuh"
#include "dydt.cuh"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#ifndef SHARED_SIZE
    #define SHARED_SIZE (0)
#endif

#define T_ID (threadIdx.x + (blockDim.x * blockIdx.x))

//#define ECHECK

__global__
void k_dydt(const int num, const double* pres, const double* y, double * dy, const mechanism_memory* d_mem)
{
    if (T_ID < num)
    {
        dydt(0, pres[T_ID], y, dy, d_mem);
    }   
}


__global__
void k_eval_jacob(const int num, const double* pres, const double* y, double * jac, const mechanism_memory* d_mem)
{
    if (T_ID < num)
    {
        eval_jacob(0, pres[T_ID], y, jac, d_mem);
    }   
}

inline void check_err()
{
#ifdef ECHECK
    cudaErrorCheck( cudaPeekAtLastError() );
    cudaErrorCheck( cudaDeviceSynchronize() );
#endif
}

inline void memcpy2D_out(double* dst, const int pitch_dst, double const * src, const int pitch_src,
                                      const int offset, const size_t width, const int height) {
    for (int i = 0; i < height; ++i)
    {
        memcpy(&dst[offset], src, width);
        dst += pitch_dst;
        src += pitch_src;
    }
}

inline void memcpy2D_in(double* dst, const int pitch_dst, double const * src, const int pitch_src,
                                     const int offset, const size_t width, const int height) {
    for (int i = 0; i < height; ++i)
    {
        memcpy(dst, &src[offset], width);
        dst += pitch_dst;
        src += pitch_src;
    }
}

mechanism_memory * d_mem = 0;
mechanism_memory * h_mem = 0;
double* y_device = 0;
double* var_device = 0;
double* y_temp = 0;
double* pres_temp = 0;
double* conc_temp = 0;
double* fwd_temp = 0;
#if REV_RATES > 0
    double* rev_temp = 0;
#endif
#if PRES_MOD_RATES > 0
    double* pres_mod_temp = 0;
#endif
double* spec_temp = 0;
double* dy_temp = 0;
double* jac_temp = 0;
int device = 0;

int init(int num)
{
    cudaErrorCheck( cudaSetDevice(device) );
    //reset device
    cudaErrorCheck( cudaDeviceReset() );
    //determine maximum # of threads for this mechanism
    //bytes per thread
    size_t mech_size = get_required_size();
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaErrorCheck( cudaMemGetInfo (&free_mem, &total_mem) );
    //conservatively estimate the maximum allowable threads
    int max_threads = int(floor(0.8 * ((double)free_mem) / ((double)mech_size)));
    int padded = min(num, max_threads);
    padded = padded - padded % TARGET_BLOCK_SIZE;
    if (padded == 0)
    {
        printf("Mechanism is too large to fit into global CUDA memory... exiting.");
        exit(-1);
    }

    padded = 1;
    h_mem = (mechanism_memory*)malloc(sizeof(mechanism_memory));
    initialize_gpu_memory(padded, &h_mem, &d_mem, &y_device, &var_device);

    size_t pitch_device = padded * sizeof(double);
    //now the temp memory on the cpu
    pres_temp = (double*)malloc(pitch_device);
    y_temp = (double*)malloc(NSP * pitch_device);
    conc_temp = (double*)malloc(NSP * pitch_device);
    fwd_temp = (double*)malloc(FWD_RATES * pitch_device);
#if REV_RATES != 0
    rev_temp = (double*)malloc(REV_RATES * pitch_device);
#endif
#if PRES_MOD_RATES != 0
    pres_mod_temp = (double*)malloc(PRES_MOD_RATES * pitch_device);
#endif
    spec_temp = (double*)malloc(NSP * pitch_device);
    dy_temp = (double*)malloc(NSP * pitch_device);
    jac_temp = (double*)malloc(NSP * NSP * pitch_device);

    return padded;
}

void cleanup()
{
    free(pres_temp);
    free(y_temp);
    free(conc_temp);
    free(fwd_temp);
#if REV_RATES != 0
    free(rev_temp);
#endif
#if PRES_MOD_RATES != 0
    free(pres_mod_temp);
#endif
    free(spec_temp);
    free(dy_temp);
    free(jac_temp);
    
    //clean up
    free_gpu_memory(&h_mem, &d_mem, &y_device, &var_device);
    free(h_mem);

    //reset device
    cudaErrorCheck( cudaDeviceReset() );
}

void run(int num, int padded, int offset, const double* pres, const double* mass_frac,
            double* conc, double* fwd_rxn_rates, double* rev_rxn_rates,
            double* pres_mod, double* spec_rates, double* dy, double* jac)
{   
    int grid_num = padded / TARGET_BLOCK_SIZE;
    size_t pitch_host = num * sizeof(double);
    size_t pitch_device = padded * sizeof(double);

    //copy over our data
    memcpy(pres_temp, &pres[offset], pitch_host);
    cudaErrorCheck( cudaMemcpy(var_device, pres_temp, pitch_host, cudaMemcpyHostToDevice) );
    memcpy2D_in(y_temp, padded, mass_frac, num, offset, pitch_host, NSP);
    cudaErrorCheck( cudaMemcpy2D(y_device, pitch_device, y_temp,
                    pitch_device, pitch_host, NSP, cudaMemcpyHostToDevice) );

    //eval dydt
    //this gets us all arrays but the Jacobian
    k_dydt<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, var_device, y_device, h_mem->dy, d_mem);

    check_err();

    //copy back
    cudaErrorCheck( cudaMemcpy2D(conc_temp, pitch_device, h_mem->conc, pitch_device,
                    pitch_host, NSP, cudaMemcpyDeviceToHost) );
    memcpy2D_out(conc, num, conc_temp, padded, offset, pitch_host, NSP);

    cudaErrorCheck( cudaMemcpy2D(fwd_temp, pitch_device, h_mem->fwd_rates, pitch_device,
                                 pitch_host, FWD_RATES, cudaMemcpyDeviceToHost) );
    memcpy2D_out(fwd_rxn_rates, num, fwd_temp, padded, offset, pitch_host, FWD_RATES);

    #if REV_RATES != 0
        cudaErrorCheck( cudaMemcpy2D(rev_temp, pitch_device, h_mem->rev_rates,
                                pitch_device, pitch_host, REV_RATES, cudaMemcpyDeviceToHost) );
        memcpy2D_out(rev_rxn_rates, num, rev_temp, padded, offset, pitch_host, REV_RATES);
    #endif

    #if PRES_MOD_RATES != 0
    cudaErrorCheck( cudaMemcpy2D(pres_mod_temp, pitch_device, h_mem->pres_mod, pitch_device, pitch_host,
                                    PRES_MOD_RATES, cudaMemcpyDeviceToHost) );
    memcpy2D_out(pres_mod, num, pres_mod_temp, padded, offset, pitch_host, PRES_MOD_RATES);
    #endif

    cudaErrorCheck( cudaMemcpy2D(spec_temp, pitch_device, h_mem->spec_rates, pitch_device,
                                    pitch_host, NSP, cudaMemcpyDeviceToHost) );
    memcpy2D_out(spec_rates, num, spec_temp, padded, offset, pitch_host, NSP);

    cudaErrorCheck( cudaMemcpy2D(dy_temp, pitch_device, h_mem->dy, pitch_device, pitch_host,
                                    NSP, cudaMemcpyDeviceToHost) );
    memcpy2D_out(dy, num, dy_temp, padded, offset, pitch_host, NSP);

    //jacobian
    k_eval_jacob<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, var_device, y_device, h_mem->jac, d_mem);

    check_err();

    //copy back
    cudaErrorCheck( cudaMemcpy2D(jac_temp, pitch_device, h_mem->jac, pitch_device,
                                    pitch_host, NSP * NSP, cudaMemcpyDeviceToHost) );
    memcpy2D_out(jac, num, jac_temp, padded, offset, pitch_host, NSP * NSP);
}