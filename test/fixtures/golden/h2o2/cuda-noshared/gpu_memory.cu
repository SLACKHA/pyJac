#include "gpu_memory.cuh"

size_t required_mechanism_size() {
  //returns the total required size for the mechanism per thread
  size_t mech_size = 0;
  //y
  mech_size += NSP;
  //dy
  mech_size += NSP;
  //conc
  mech_size += NSP;
  //fwd_rates
  mech_size += FWD_RATES;
  //rev_rates
  mech_size += REV_RATES;
  //spec_rates
  mech_size += NSP;
  //cp
  mech_size += NSP;
  //h
  mech_size += NSP;
  //dBdT
  mech_size += NSP;
  //jac
  mech_size += NSP * NSP;
  //var
  mech_size += 1;
  //pres_mod
  mech_size += PRES_MOD_RATES;
  //y_device
  mech_size += NSP;
  //pres_device
  mech_size += 1;
  return mech_size * sizeof(double);
}
void initialize_gpu_memory(int padded, mechanism_memory** h_mem, mechanism_memory** d_mem)
{
  //init vectors
  // Allocate storage for the device struct
  cudaErrorCheck( cudaMalloc(d_mem, sizeof(mechanism_memory)) );
  //allocate the device arrays on the host pointer
  cudaErrorCheck( cudaMalloc(&((*h_mem)->y), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->dy), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->conc), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->fwd_rates), FWD_RATES * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->rev_rates), REV_RATES * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->spec_rates), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->cp), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->h), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->dBdT), NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->jac), NSP * NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->var), 1 * padded * sizeof(double)) );
  cudaErrorCheck( cudaMalloc(&((*h_mem)->pres_mod), PRES_MOD_RATES * padded * sizeof(double)) );
  cudaErrorCheck( cudaMemset((*h_mem)->spec_rates, 0, NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMemset((*h_mem)->dy, 0, NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMemset((*h_mem)->jac, 0, NSP * NSP * padded * sizeof(double)) );
  cudaErrorCheck( cudaMemcpy(*d_mem, *h_mem, sizeof(mechanism_memory), cudaMemcpyHostToDevice) );
  //zero out required values
}
void free_gpu_memory(mechanism_memory** h_mem, mechanism_memory** d_mem)
{
  cudaErrorCheck(cudaFree((*h_mem)->y));
  cudaErrorCheck(cudaFree((*h_mem)->dy));
  cudaErrorCheck(cudaFree((*h_mem)->conc));
  cudaErrorCheck(cudaFree((*h_mem)->fwd_rates));
  cudaErrorCheck(cudaFree((*h_mem)->rev_rates));
  cudaErrorCheck(cudaFree((*h_mem)->spec_rates));
  cudaErrorCheck(cudaFree((*h_mem)->cp));
  cudaErrorCheck(cudaFree((*h_mem)->h));
  cudaErrorCheck(cudaFree((*h_mem)->dBdT));
  cudaErrorCheck(cudaFree((*h_mem)->jac));
  cudaErrorCheck(cudaFree((*h_mem)->var));
  cudaErrorCheck(cudaFree((*h_mem)->pres_mod));
  cudaErrorCheck(cudaFree(*d_mem));
}
