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
int init()
{
	//determine maximum # of threads for this mechanism
	//bytes per thread
    size_t mech_size = get_required_size();
    cudaDeviceProp props;
    cudaErrorCheck( cudaGetDeviceProperties(&props, device) );
    //memory size in bytes
    size_t mem_avail = props.totalGlobalMem;
    //conservatively estimate the maximum allowable threads
    int max_threads = int(floor(0.8 * float(mem_avail) / float(mech_size)));
    int padded = max_threads * block_size;

    initialize_gpu_memory(padded, &h_mem, &d_mem, &y_device, &var_device);
    return padded;
}

void cleanup()
{

	//reset device
	cudaErrorCheck( cudaDeviceReset() );

	//clean up
	free_gpu_memory(&h_mem, &d_mem, &y_device, &var_device);
	free(h_mem);
}

bool run(int num, int padded, int offset, const double* pres, const double* mass_frac,
			double* conc, double* fwd_rxn_rates, double* rev_rxn_rates,
			double* pres_mod, double* spec_rates, double* dy, double* jac,
			bool eval_rates)
{	
	int pitch_host = num * sizeof(double);
	int pitch_device = padded * sizeof(double);

	if (padded > 0)
	{
		//copy over state data
		cudaErrorCheck( cudaMemcpy(var_device, pres, pitch_host, cudaMemcpyHostToDevice) );
		cudaErrorCheck( cudaMemcpy2D(y_device, pitch_device, mass_frac,
						pitch_host, pitch_host, NSP, cudaMemcpyHostToDevice) );

		if (eval_rates)
		{
			//eval dydt
			//this gets us all arrays but the Jacobian
			k_dydt<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, var_device, y_device, h_mem->dy, d_mem);

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
		}

		//jacobian
		k_eval_jacob<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, var_device, y_device, h_mem->jac, d_mem);

		check_err();

		//copy back
		cudaErrorCheck( cudaMemcpy2D(jac, pitch_host, h_mem->jac, pitch_device,
										pitch_host, NSP * NSP, cudaMemcpyDeviceToHost) );

	}
}