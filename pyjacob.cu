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

#define DEBUG

__global__
void k_eval_conc(const int num, const double* pres, 
	const double* dMass, double* dConc)
{
	if (T_ID < num)
	{
		double mw_avg = 0;
		double rho = 0;
		double y_last = 0;

		eval_conc(dMass[T_ID], pres[T_ID], &dMass[GRID_DIM], &y_last, 
					&mw_avg, &rho, dConc);
	}
}

__global__
void k_eval_rxn_rates(const int num, const double* T, const double* pres, const double * C,
	double * fwd_rxn_rates, double * rev_rxn_rates)
{
	if (T_ID < num) {
		eval_rxn_rates(T[T_ID], pres[T_ID], C, fwd_rxn_rates, rev_rxn_rates);
	} 
}

__global__
void k_get_rxn_pres_mod(const int num, const double* T, const double* pres, const double * C, double * pres_mod)
{
	if (T_ID < num)
	{
		get_rxn_pres_mod(T[T_ID], pres[T_ID], C, pres_mod);
	}	
}

__global__
void k_eval_spec_rates(const int num, const double* fwd_rates, const double * rev_rates,
	const double* pres_mod, double* spec_rates)
{
	if (T_ID < num)
	{
		eval_spec_rates(fwd_rates, rev_rates, pres_mod, spec_rates, &spec_rates[(NSP - 1) * GRID_DIM]);
	}
}

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

void check_err()
{
#ifdef DEBUG
	cudaErrorCheck( cudaPeekAtLastError() );
	cudaErrorCheck( cudaDeviceSynchronize() );
#endif
}

void run(int num, const double* pres, const double* mass_frac,
			double* conc, double* fwd_rxn_rates, double* rev_rxn_rates,
			double* pres_mod, double* spec_rates, double* dy, double* jac)
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

	//get concentrations
	k_eval_conc<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, dPres,
		dMass, h_mem->conc);

	check_err();
	
	//copy back
	cudaErrorCheck( cudaMemcpy2D(conc, pitch_host, h_mem->conc, pitch_device,
					pitch_host, NSP, cudaMemcpyDeviceToHost) );

	//reaction rates
	k_eval_rxn_rates<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, dMass, dPres,
						h_mem->conc, h_mem->fwd_rates, h_mem->rev_rates);

	check_err();

	//copy back
	cudaErrorCheck( cudaMemcpy2D(fwd_rxn_rates, pitch_host, h_mem->fwd_rates, pitch_device,
								 pitch_host, FWD_RATES, cudaMemcpyDeviceToHost) );

	#if REV_RATES != 0
		cudaErrorCheck( cudaMemcpy2D(rev_rxn_rates, pitch_host, h_mem->rev_rates,
								pitch_device, pitch_host, REV_RATES, cudaMemcpyDeviceToHost) );
	#endif

	//pres mod rates
	k_get_rxn_pres_mod<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, dMass, dPres,
								h_mem->conc, h_mem->pres_mod);

	check_err();

	//copy back
	cudaErrorCheck( cudaMemcpy2D(pres_mod, pitch_host, h_mem->pres_mod, pitch_device, pitch_host,
									PRES_MOD_RATES, cudaMemcpyDeviceToHost) );
	
	//species rates
	k_eval_spec_rates<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, h_mem->fwd_rates, h_mem->rev_rates,
						h_mem->pres_mod, h_mem->spec_rates);

	check_err();

	//copy back
	cudaErrorCheck( cudaMemcpy2D(spec_rates, pitch_host, h_mem->spec_rates, pitch_device,
									pitch_host, NSP, cudaMemcpyDeviceToHost) );

	//dydt
	k_dydt<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, dPres, dMass, h_mem->dy, d_mem);

	check_err();

	//copy back
	cudaErrorCheck( cudaMemcpy2D(dy, pitch_host, h_mem->dy, pitch_device, pitch_host,
									NSP, cudaMemcpyDeviceToHost) );

	//jacobian
	k_eval_jacob<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, dPres, dMass, h_mem->jac, d_mem);

	check_err();

	//copy back
	cudaErrorCheck( cudaMemcpy2D(jac, pitch_host, h_mem->jac, pitch_device, pitch_host, NSP * NSP, cudaMemcpyDeviceToHost) );

	//clean up
	free_gpu_memory(&h_mem, &d_mem, &dMass, &dPres);
	free(h_mem);

}