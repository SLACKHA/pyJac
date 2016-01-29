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

void run(int num, const double* pres, const double* mass_frac,
			double* conc, double* fwd_rxn_rates, double* rev_rxn_rates,
			double* pres_mod, double* spec_rates, double* dy, double* jac,
			bool eval_rates)
{
	int grid_num = (int)ceil(((double)num) / ((double)TARGET_BLOCK_SIZE));
	mechanism_memory * d_mem = 0;
	mechanism_memory * h_mem = (mechanism_memory*)malloc(sizeof(mechanism_memory));
	double* dMass = 0;
	double* dPres = 0;
	int padded = initialize_gpu_memory(num, TARGET_BLOCK_SIZE, grid_num, &h_mem, &d_mem, &dMass, &dPres);
	int pitch_host = num * sizeof(double);
	int pitch_device = padded * sizeof(double);

	//copy over state data
	cudaErrorCheck( cudaMemcpy(dPres, pres, pitch_host, cudaMemcpyHostToDevice) );
	cudaErrorCheck( cudaMemcpy2D(dMass, pitch_device, mass_frac,
					pitch_host, pitch_host, NSP, cudaMemcpyHostToDevice) );

	if (eval_rates)
	{
		//eval dydt
		//this gets us all arrays but the Jacobian
		k_dydt<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, dPres, dMass, h_mem->dy, d_mem);

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

		cudaErrorCheck( cudaMemcpy2D(pres_mod, pitch_host, h_mem->pres_mod, pitch_device, pitch_host,
										PRES_MOD_RATES, cudaMemcpyDeviceToHost) );

		cudaErrorCheck( cudaMemcpy2D(spec_rates, pitch_host, h_mem->spec_rates, pitch_device,
										pitch_host, NSP, cudaMemcpyDeviceToHost) );

		cudaErrorCheck( cudaMemcpy2D(dy, pitch_host, h_mem->dy, pitch_device, pitch_host,
										NSP, cudaMemcpyDeviceToHost) );
	}

	//jacobian
	k_eval_jacob<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, dPres, dMass, h_mem->jac, d_mem);

	check_err();

	//copy back
	cudaErrorCheck( cudaMemcpy2D(jac, pitch_host, h_mem->jac, pitch_device,
									pitch_host, NSP * NSP, cudaMemcpyDeviceToHost) );

	//clean up
	free_gpu_memory(&h_mem, &d_mem, &dMass, &dPres);
	free(h_mem);

}