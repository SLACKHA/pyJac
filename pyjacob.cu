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

#define T_ID (threadIdx.x + (blockDim.x * blockIdx.x))

#define ECHECK

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

#define USE_MEM (0.8)

int init(int num)
{
	cudaErrorCheck( cudaSetDevice(device) );
	//reset device
	cudaErrorCheck( cudaDeviceReset() );
#ifdef PREFERL1
	//prefer L1 for speed
	cudaErrorCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	cudaFuncCache L1type;
	cudaErrorCheck(cudaDeviceGetCacheConfig(&L1type));
	assert(L1type == cudaFuncCachePreferL1);
	printf("L1 Cache size increased...\n");
#endif
	//determine maximum # of threads for this mechanism
	//bytes per thread
    size_t mech_size = required_mechanism_size();
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaErrorCheck( cudaMemGetInfo (&free_mem, &total_mem) );
    //conservatively estimate the maximum allowable threads
    int max_threads = int(floor(USE_MEM * ((double)free_mem) / ((double)mech_size)));
    int padded = min(num, max_threads);
    //padded is next factor of block size up
    padded = int(ceil(padded / float(TARGET_BLOCK_SIZE)) * TARGET_BLOCK_SIZE);
    if (padded == 0)
    {
    	printf("Mechanism is too large to fit into global CUDA memory... exiting.");
    	exit(-1);
    }
	
	printf("Initializing CUDA interface...\n");
    printf("%ld free bytes of memory found on Device 0.\n", free_mem);
    printf("%ld bytes required per mechanism thread\n", mech_size);
    printf("Setting up memory to work on kernels of %d threads, with blocksize %d\n", padded, TARGET_BLOCK_SIZE);

    h_mem = (mechanism_memory*)malloc(sizeof(mechanism_memory));
    initialize_gpu_memory(padded, &h_mem, &d_mem);
    return padded;
}

void cleanup()
{
	//clean up
	free_gpu_memory(&h_mem, &d_mem);
	free(h_mem);

	//reset device
	cudaErrorCheck( cudaDeviceReset() );
}

void run(int num, int padded, const double* pres, const double* mass_frac,
			double* conc, double* fwd_rxn_rates, double* rev_rxn_rates,
			double* pres_mod, double* spec_rates, double* dy, double* jac)
{	
	int grid_num = padded / TARGET_BLOCK_SIZE;
	size_t pitch_host = num * sizeof(double);
	size_t pitch_device = padded * sizeof(double);

	//copy over our data
	cudaErrorCheck( cudaMemcpy(h_mem->var, pres, pitch_host, cudaMemcpyHostToDevice) );
	cudaErrorCheck( cudaMemcpy2D(h_mem->y, pitch_device, mass_frac,
					pitch_host, pitch_host, NSP, cudaMemcpyHostToDevice) );

	size_t smem = 0;
	#ifdef SHARED_SIZE
		smem = SHARED_SIZE;
	#endif
	//eval dydt
	//this gets us all arrays but the Jacobian
	k_dydt<<<grid_num, TARGET_BLOCK_SIZE, smem>>>(num, h_mem->var, h_mem->y, h_mem->dy, d_mem);

	check_err();

	//copy back
	cudaErrorCheck( cudaMemcpy2D(conc, pitch_host, h_mem->conc, pitch_device,
					pitch_host, NSP, cudaMemcpyDeviceToHost) );

	cudaErrorCheck( cudaMemcpy2D(fwd_rxn_rates, pitch_host, h_mem->fwd_rates, pitch_device,
								 pitch_host, FWD_RATES, cudaMemcpyDeviceToHost) );

	#if REV_RATES != 0
		cudaErrorCheck( cudaMemcpy2D(rev_rxn_rates, pitch_host, h_mem->rev_rates,
								pitch_device, pitch_host, REV_RATES, cudaMemcpyDeviceToHost) );
	#endif

	#if PRES_MOD_RATES != 0
	cudaErrorCheck( cudaMemcpy2D(pres_mod, pitch_host, h_mem->pres_mod, pitch_device, pitch_host,
									PRES_MOD_RATES, cudaMemcpyDeviceToHost) );
	#endif

	cudaErrorCheck( cudaMemcpy2D(spec_rates, pitch_host, h_mem->spec_rates, pitch_device,
									pitch_host, NSP, cudaMemcpyDeviceToHost) );

	cudaErrorCheck( cudaMemcpy2D(dy, pitch_host, h_mem->dy, pitch_device, pitch_host,
									NSP, cudaMemcpyDeviceToHost) );

	//jacobian
	k_eval_jacob<<<grid_num, TARGET_BLOCK_SIZE, smem>>>(num, h_mem->var, h_mem->y, h_mem->jac, d_mem);

	check_err();

	//copy back
	cudaErrorCheck( cudaMemcpy2D(jac, pitch_host, h_mem->jac, pitch_device,
									pitch_host, NSP * NSP, cudaMemcpyDeviceToHost) );
}