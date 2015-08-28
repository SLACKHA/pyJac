/* wrapper to translate to cuda arrays */

#include "header.h"
#include "gpu_memory.cuh"
#include "launch_bounds.cuh"
#include "chem_utils.cuh"
#include "rates.cuh"
#include "jacob.cuh"

#ifndef SHARED_SIZE
	#define SHARED_SIZE (0)
#endif

#define T_ID (threadIdx.x + (blockDim.x * blockIdx.x))

__global__
void k_eval_conc(const int num, const double* T, const double* pres, 
	const double* dMass, size_t pitch1, 
	const double* dMw,
	const double* dRho,
	const double* dConc, size_t pitch2);
{
	if (T_ID < num)
	{
		double mass_local[NSP];
		#pragma unroll
		for (int i = 0; i < NSP; ++i)
		{
			mass_local[i] = *((double*)((char*)dMass + i * pitch1) + T_ID);
		}

		double T_local = T[T_ID]
		double pres_local = pres[T_ID];
		double mw_avg = 0;
		double rho = 0;
		double conc_local[NSP] = {0};

		eval_conc(T_local, pres_local, mass_local, &mw_avg, &rho, conc_local);

		dMw[T_ID] = mw_avg;
		dRho[T_ID] = rho;
		#pragma unroll
		for (int i = 0; i < FWD_RATES; ++i)
		{
			*((double*)((char*)dConc + i * pitch2) + T_ID) = conc_local[i];
		}
	}
}

void cu_eval_conc (const int num, const double * T, const double * pres, const double * mass_frac, double * mw_avg, double * rho, double * conc) {
	int grid_num = num / TARGET_BLOCK_SIZE;
	//allocate device memory
	double* dT;
	double* dPres;
	double* dMass;
	double* dMw;
	double* dRho = 0;
	double* dC = 0;
	size_t pitch1 = 0, pitch2 = 0, pitch3 = 0, pitch4 = 0, pitch5 = 0, pitch6 = 0;
	cudaErrorCheck( cudaMallocPitch((void**)&dT, &pitch1, num * sizeof(double), 1) );
	cudaErrorCheck( cudaMallocPitch((void**)&dPres, &pitch2, num * sizeof(double), 1) );
	cudaErrorCheck( cudaMallocPitch((void**)&dMass, &pitch3, num * sizeof(double), NSP) );
	cudaErrorCheck( cudaMallocPitch((void**)&dMw, &pitch4, num * sizeof(double), 1) );
	cudaErrorCheck( cudaMallocPitch((void**)&dRho, &pitch5, num * sizeof(double), 1) );
	cudaErrorCheck( cudaMallocPitch((void**)&dC, &pitch6, num * sizeof(double), NSP) );

	//copy over
	cudaErrorCheck( cudaMemcpy2D(dT, pitch1, T, num, num * sizeof(double), 1, cudaMemcpyHostToDevice) );
	cudaErrorCheck( cudaMemcpy2D(dPres, pitch2, pres, num, num * sizeof(double), 1, cudaMemcpyHostToDevice) );
	cudaErrorCheck( cudaMemcpy2D(dMass, pitch3, mass_frac, num, num * sizeof(double), NSP, cudaMemcpyHostToDevice) );

	//run
	k_eval_conc<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, T, pres, dMass, pitch3, dMw, dRho, dConc, pitch6);
	//copy back
	cudaErrorCheck( cudaMemcpy2D(mw_avg, num, dMw, pitch4, num * sizeof(double), 1, cudaMemcpyDeviceToHost) );
	cudaErrorCheck( cudaMemcpy2D(rho, num, dRho, pitch5, num * sizeof(double), 1, cudaMemcpyDeviceToHost) );
	cudaErrorCheck( cudaMemcpy2D(conc, num, dC, pitch6, num * sizeof(double), NSP, cudaMemcpyDeviceToHost) );

	cudaErrorCheck( cudaFree(dT) );
	cudaErrorCheck( cudaFree(dPres) );
	cudaErrorCheck( cudaFree(dMass) );
	cudaErrorCheck( cudaFree(dMw) );
	cudaErrorCheck( cudaFree(dRho) );
	cudaErrorCheck( cudaFree(dC) );
}


__global__
void k_eval_rxn_rates(const int num, const double T, const double pres, const double * C,
	size_t pitch1, double * fwd_rxn_rates, size_t pitch2, double * rev_rxn_rates,
	size_t pitch3)
{
	if (T_ID < num) {
		double conc_local[NSP];
		#pragma unroll
		for (int i = 0; i < NSP; ++i)
		{
			conc_local[i] = *((double*)((char*)C + i * pitch1) + T_ID);
		}

		double fwd_local[FWD_RATES];

		#if REV_RATES == 0
			double* rev = 0;
		#else
			double rev_local[REV_RATES];
		#endif

		eval_rxn_rates(T, pres, conc_local, fwd_local, rev_local);

		#pragma unroll
		for (int i = 0; i < FWD_RATES; ++i)
		{
			*((double*)((char*)fwd_rxn_rates + i * pitch2) + T_ID) = fwd_local[i];
		}

		#if REV_RATES != 0
			#pragma unroll
			for (int i = 0; i < REV_RATES; ++i)
			{
				*((double*)((char*)rev_rxn_rates + i * pitch3) + T_ID) = rev_local[i];
			}
		#endif
	} 
}

void cu_eval_rxn_rates (const int num, const double T, const double pres, const double * C, double * fwd_rxn_rates, double * rev_rxn_rates) {
	int grid_num = num / TARGET_BLOCK_SIZE;
	//allocate device memory
	double* dC;
	double* dFwd;
	double* dRev = 0;
	size_t pitch1 = 0, pitch2 = 0, pitch3 = 0;
	cudaErrorCheck( cudaMallocPitch((void**)&dC, &pitch1, num * sizeof(double), NSP) );
	cudaErrorCheck( cudaMallocPitch((void**)&dFwd, &pitch2, num * sizeof(double), FWD_RATES) );
	#if REV_RATES != 0
		cudaErrorCheck( cudaMallocPitch((void**)&dRev, &pitch3, num * sizeof(double), REV_RATES) );
	#endif

	//copy over
	cudaErrorCheck( cudaMemcpy2D(dC, pitch1, C, num, num * sizeof(double), NSP, cudaMemcpyHostToDevice) );

	//run
	k_eval_rxn_rates<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(T, pres, C, pitch1, dFwd, pitch2, dRev, pitch3);

	//copy back
	cudaErrorCheck( cudaMemcpy2D(fwd_rxn_rates, num, dFwd, pitch2, num * sizeof(double), FWD_RATES, cudaMemcpyDeviceToHost) );

	#if REV_RATES != 0
		cudaErrorCheck( cudaMemcpy2D(rev_rxn_rates, num, dRev, pitch3, num * sizeof(double), REV_RATES, cudaMemcpyDeviceToHost) );
	#endif

	cudaErrorCheck( cudaFree(dC) );
	cudaErrorCheck( cudaFree(dFwd) );
	cudaErrorCheck( cudaFree(dRev) );
}

__global__
void k_get_rxn_pres_mod(const int num, const double T, const double pres, const double * C, size_t pitch1, double * pres_mod,
	size_t pitch2)
{
	if (T_ID < num)
	{
		double conc_local[NSP];
		double pres_mod_local[PRES_MOD_RATES];
		#pragma unroll
		for (int i = 0; i < NSP; ++i)
		{
			conc_local[i] = *((double*)((char*)C + i * pitch1) + T_ID);
		}

		get_rxn_pres_mod(T, pres, conc_local, pres_mod_local);

		#pragma unroll
		for (int i = 0; i < PRES_MOD_RATES; ++i)
		{
			*((double*)((char*)pres_mod + i * pitch2) + T_ID) = pres_mod_local[i];
		}
	}	
}

void cu_get_rxn_pres_mod (const int num, const double T, const double pres, const double * C, double * pres_mod) {
	#if PRES_MOD_RATES != 0
		int grid_num = num / TARGET_BLOCK_SIZE;
		//allocate device memory
		double* dC;
		double* dPres;
		size_t pitch1, pitch2;
		
		cudaErrorCheck( cudaMallocPitch((void**)&dC, &pitch1, num * sizeof(double), NSP) );
		cudaErrorCheck( cudaMallocPitch((void**)&dPres, &pitch2, num * sizeof(double), PRES_MOD_RATES) );

		//copy over
		cudaErrorCheck( cudaMemcpy2D(dC, pitch1, C, num, num * sizeof(double), NSP, cudaMemcpyHostToDevice) );

		//run
		k_get_rxn_pres_mod<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, T, pres, C, fwd_rxn_rates, rev_rxn_rates);

		//copy back
		cudaErrorCheck( cudaMemcpy2D(pres_mod, num, dPres, pitch2, num * sizeof(double), PRES_MOD_RATES, cudaMemcpyHostToDevice) );
	
		cudaErrorCheck(cudaFree(dC));
		cudaErrorCheck(cudaFree(dPres));
	#endif
}


__global__
void k_eval_spec_rates(const int num, const double* fwd_rates, size_t pitch1, const double * rev_rates,
	size_t pitch2, const double* pres_mod, size_t pitch3, double* spec_rates, size_t pitch4)
{
	if (T_ID < num)
	{
		double fwd_local[FWD_RATES];
		#pragma unroll
		for (int i = 0; i < FWD_RATES; ++i)
		{
			fwd_local[i] = *((double*)((char*)fwd_rates + i * pitch1) + T_ID);
		}

		#if REV_RATES != 0
			double rev_local[REV_RATES];
			#pragma unroll
			for (int i = 0; i < REV_RATES; ++i)
			{
				rev_local[i] = *((double*)((char*)rev_rates + i * pitch2) + T_ID);
			}
		#else
			double* rev_local = 0;
		#endif

		#if PRES_MOD_RATES != 0
			double pres_mod_local[PRES_MOD_RATES];
			#pragma unroll
			for (int i = 0; i < PRES_MOD_RATES; ++i)
			{
				pres_mod_local[i] = *((double*)((char*)pres_mod + i * pitch3) + T_ID);
			}
		#else
			double* pres_mod_local = 0;
		#endif

		double spec_rates_local[NSP];
		eval_spec_rates(fwd_local, rev_local, pres_mod_local, spec_rates_locals);

		#pragma unroll
		for (int i = 0; i < NSP; ++i)
		{
			*((double*)((char*)spec_rates + i * pitch4) + T_ID) = spec_rates_local[i];
		}
	}	
}

void cu_eval_spec_rates (const int num, const double * fwd_rates, const double * rev_rates, const double * pres_mod, double * spec_rates) {
	int grid_num = num / TARGET_BLOCK_SIZE;
	//allocate device memory
	double* dFwd;
	double* dRev = 0;
	double* dPres = 0;
	double* dSpec;
	size_t pitch1, pitch2 = 0, pitch3 = 0, pitch4;
	
	cudaErrorCheck( cudaMallocPitch((void**)&dFwd, &pitch1, num * sizeof(double), FWD_RATES) );
	cudaErrorCheck( cudaMemcpy2D(dFwd, pitch1, fwd_rates, num, num * sizeof(double), FWD_RATES, cudaMemcpyHostToDevice) );
	#if REV_RATES != 0
		cudaErrorCheck( cudaMallocPitch((void**)&dRev, &pitch2, num * sizeof(double), REV_RATES) );
		cudaErrorCheck( cudaMemcpy2D(dRev, pitch2, rev_rates, num, num * sizeof(double), REV_RATES, cudaMemcpyHostToDevice) );
	#endif

	#if PRES_MOD_RATES != 0
		cudaErrorCheck( cudaMallocPitch((void**)&dPres, &pitch3, num * sizeof(double), PRES_MOD_RATES) );
		cudaErrorCheck( cudaMemcpy2D(dPres, pitch3, pres_mod, num, num * sizeof(double), PRES_MOD_RATES, cudaMemcpyHostToDevice) );
	#endif

	cudaErrorCheck( cudaMallocPitch((void**)&dSpec, &pitch4, num * sizeof(double), NSP) );
	//run
	k_eval_spec_rates<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, dFwd, pitch1, dRev, pitch2, dPres, pitch3, dSpec, pitch4);

	//copy back
	cudaErrorCheck( cudaMemcpy2D(spec_rates, num, dSpec, pitch4, num * sizeof(double), NSP, cudaMemcpyHostToDevice) );

	cudaErrorCheck(cudaFree(dFwd));
	#if REV_RATES != 0
		cudaErrorCheck(cudaFree(dRev));
	#endif

	#if PRES_MOD_RATES != 0
		cudaErrorCheck(cudaFree(dPres));
	#endif

	cudaErrorCheck(cudaFree(dSpec));
}


__global__
void k_dydt(const int num, const double pres, const double* y, size_t pitch1, double * dy,
	size_t pitch2)
{
	if (T_ID < num)
	{
		double y_local[NN];
		#pragma unroll
		for (int i = 0; i < NN; ++i)
		{
			y_local[i] = *((double*)((char*)y + i * pitch1) + T_ID);
		}

		double dy_local[NN];
		dydt(0, pres, y_local, dy_local);

		#pragma unroll
		for (int i = 0; i < NN; ++i)
		{
			*((double*)((char*)dy + i * pitch2) + T_ID) = dy_local[i];
		}
	}	
}

void cu_dydt (const int num, const double pres, const double* y, double* dy) {
	int grid_num = num / TARGET_BLOCK_SIZE;
	//allocate device memory
	double* dY;
	double* dDy;
	size_t pitch1, pitch2;
	
	cudaErrorCheck( cudaMallocPitch((void**)&dY, &pitch1, num * sizeof(double), NN) );
	cudaErrorCheck( cudaMemcpy2D(dY, pitch1, y, num, num * sizeof(double), NN, cudaMemcpyHostToDevice) );

	cudaErrorCheck( cudaMallocPitch((void**)&dDy, &pitch2, num * sizeof(double), NN) );
	//run
	k_dydt<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, dY, pitch1, dDy, pitch2);

	//copy back
	cudaErrorCheck( cudaMemcpy2D(dy, num, dDy, pitch2, num * sizeof(double), NN, cudaMemcpyHostToDevice) );

	cudaErrorCheck(cudaFree(dY));
	cudaErrorCheck(cudaFree(dDy));
}

__global__
void k_eval_jacob(const int num, const double pres, const double* y, size_t pitch1, double * jac,
	size_t pitch2)
{
	if (T_ID < num)
	{
		double y_local[NN];
		#pragma unroll
		for (int i = 0; i < NN; ++i)
		{
			y_local[i] = *((double*)((char*)y + i * pitch1) + T_ID);
		}

		double jac_local[NN * NN] = {0};
		eval_jacob(0, pres, y_local, jac_local);

		#pragma unroll
		for (int i = 0; i < NN * NN; ++i)
		{
			*((double*)((char*)jac + i * pitch2) + T_ID) = jac_local[i];
		}
	}	
}

void cu_eval_jacob (const int num, const double t, const double pres, const double* y, double* jac) {
	int grid_num = num / TARGET_BLOCK_SIZE;
	//allocate device memory
	double* dY;
	double* dJac;
	size_t pitch1, pitch2;
	
	cudaErrorCheck( cudaMallocPitch((void**)&dY, &pitch1, num * sizeof(double), NN) );
	cudaErrorCheck( cudaMemcpy2D(dY, pitch1, y, num, num * sizeof(double), NN, cudaMemcpyHostToDevice) );

	cudaErrorCheck( cudaMallocPitch((void**)&dJac, &pitch2, num * sizeof(double), NN * NN) );
	//run
	k_eval_jacob<<<grid_num, TARGET_BLOCK_SIZE, SHARED_SIZE>>>(num, dY, pitch1, dJac, pitch2);

	//copy back
	cudaErrorCheck( cudaMemcpy2D(jac, num, dJac, pitch2, num * sizeof(double), NN * NN, cudaMemcpyHostToDevice) );

	cudaErrorCheck(cudaFree(dY));
	cudaErrorCheck(cudaFree(dJac));
}