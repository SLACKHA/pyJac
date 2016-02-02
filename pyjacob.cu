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

inline void memcpy2D_out(double* dst, const int pitch_dst, const double* src, const int pitch_src, 
									  const int offset, const size_t width, const int height) {
	for (int i = 0; i < height; ++i)
	{
		memcpy(&dst[i * pitch_dst + offset], &src[i * pitch_src], width);
	}
}

inline void memcpy2D_in(double* dst, const int pitch_dst, const double* src, const int pitch_src, 
									 const int offset, const size_t width, const int height) {
	for (int i = 0; i < height; ++i)
	{
		memcpy(&dst[i * pitch_dst], &src[i * pitch_src + offset], width);
	}
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
	int num_kernels = 1;
	int num_per_kernel = num;
	int padded = -1;
	while (padded == -1)
	{
		padded = initialize_gpu_memory(num_per_kernel, TARGET_BLOCK_SIZE, grid_num, &h_mem, &d_mem, &dMass, &dPres);
		if (padded == -1)
		{
			num_kernels *= 2;
			num_per_kernel /= 2;
		}
	}
	int offset = 0;
	int pitch_host = -1;
	int pitch_device = padded * sizeof(double);

	if (num_per_kernel != num) {
		//special handling for kernels that would require too much memory

		grid_num = (int)ceil(((double)num_per_kernel) / ((double)TARGET_BLOCK_SIZE));
		int pitch_host_real = num * sizeof(double);
		//first we need temporary copies on the host memory for memory transfer
		double* pres_temp = (double*)malloc(pitch_device);
		double* mass_temp = (double*)malloc(NSP * pitch_device);
		double* conc_temp = 0;
		double* fwd_temp = 0;
		double* rev_temp = 0;
		double* pres_mod_temp = 0;
		double* spec_temp = 0;
		double* dy_temp = 0;
		if (eval_rates)
		{
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
		}
		double* jac_temp = (double*)malloc(NSP * NSP * pitch_device);

		//now exectute each kernel
		for (int i = 0; i < num_kernels; ++i)
		{
			//figure out how many conditions we have in this kernel, and the memory size
			int num_cond = min(num_per_kernel, num - offset);
			pitch_host = num_cond * sizeof(double);

			//copy over our data
			memcpy(pres_temp, &pres[offset], pitch_host);
			cudaErrorCheck( cudaMemcpy(dPres, pres_temp, pitch_host, cudaMemcpyHostToDevice) );
			memcpy2D_in(mass_temp, pitch_device, mass_frac, pitch_host_real, offset, pitch_host, NSP);
			cudaErrorCheck( cudaMemcpy2D(dMass, pitch_device, mass_frac,
							pitch_host, pitch_host, NSP, cudaMemcpyHostToDevice) );

			if (eval_rates)
			{
				//eval dydt
				//this gets us all arrays but the Jacobian
				k_dydt<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, dPres, dMass, h_mem->dy, d_mem);

				check_err();

				//copy back
				cudaErrorCheck( cudaMemcpy2D(conc_temp, pitch_host, h_mem->conc, pitch_device,
								pitch_host, NSP, cudaMemcpyDeviceToHost) );
				memcpy2D_out(conc, pitch_host_real, conc_temp, pitch_host, offset, pitch_host, NSP);

				cudaErrorCheck( cudaMemcpy2D(fwd_rxn_rates, pitch_host, h_mem->fwd_rates, pitch_device,
											 pitch_host, FWD_RATES, cudaMemcpyDeviceToHost) );
				memcpy2D_out(fwd_rxn_rates, pitch_host_real, fwd_temp, pitch_host, offset, pitch_host, FWD_RATES);

				#if REV_RATES != 0
					cudaErrorCheck( cudaMemcpy2D(rev_rxn_rates, pitch_host, h_mem->rev_rates,
											pitch_device, pitch_host, REV_RATES, cudaMemcpyDeviceToHost) );
					memcpy2D_out(rev_rxn_rates, pitch_host_real, rev_temp, pitch_host, offset, pitch_host, REV_RATES);
				#endif

				#if PRES_MOD_RATES != 0
				cudaErrorCheck( cudaMemcpy2D(pres_mod, pitch_host, h_mem->pres_mod, pitch_device, pitch_host,
												PRES_MOD_RATES, cudaMemcpyDeviceToHost) );
				memcpy2D_out(pres_mod, pitch_host_real, pres_mod_temp, pitch_host, offset, pitch_host, PRES_MOD_RATES);
				#endif

				cudaErrorCheck( cudaMemcpy2D(spec_rates, pitch_host, h_mem->spec_rates, pitch_device,
												pitch_host, NSP, cudaMemcpyDeviceToHost) );
				memcpy2D_out(spec_rates, pitch_host_real, spec_temp, pitch_host, offset, pitch_host, NSP);

				cudaErrorCheck( cudaMemcpy2D(dy, pitch_host, h_mem->dy, pitch_device, pitch_host,
												NSP, cudaMemcpyDeviceToHost) );
				memcpy2D_out(dy, pitch_host_real, dy_temp, pitch_host, offset, pitch_host, NSP);
			}

			//jacobian
			k_eval_jacob<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, dPres, dMass, h_mem->jac, d_mem);

			check_err();

			//copy back
			cudaErrorCheck( cudaMemcpy2D(jac, pitch_host, h_mem->jac, pitch_device,
											pitch_host, NSP * NSP, cudaMemcpyDeviceToHost) );
			memcpy2D_out(jac, pitch_host_real, jac_temp, pitch_host, offset, pitch_host, NSP * NSP);

			offset += num_per_kernel;
		}
		//and finally clean up our allocated memory
		free(pres_temp);
		free(mass_temp);
		if (eval_rates) {
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
		}
		free(jac_temp);
	}
	else {
		pitch_host = num * sizeof(double);

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
		k_eval_jacob<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, dPres, dMass, h_mem->jac, d_mem);

		check_err();

		//copy back
		cudaErrorCheck( cudaMemcpy2D(jac, pitch_host, h_mem->jac, pitch_device,
										pitch_host, NSP * NSP, cudaMemcpyDeviceToHost) );

	}

	//clean up
	free_gpu_memory(&h_mem, &d_mem, &dMass, &dPres);
	free(h_mem);

}