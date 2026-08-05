#include "header.cuh"
#include "rates.cuh"

__device__ void eval_spec_rates (const double * __restrict__ fwd_rates, const double * __restrict__ rev_rates, const double * __restrict__ pres_mod, double * __restrict__ sp_rates, double * __restrict__ dy_N) {
  extern volatile __shared__ double shared_temp[];
  //rxn 0
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] = -2.0 * (fwd_rates[INDEX(0)] - rev_rates[INDEX(0)]) * pres_mod[INDEX(0)];
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] = (fwd_rates[INDEX(0)] - rev_rates[INDEX(0)]) * pres_mod[INDEX(0)];

  //rxn 1
  //sp 1
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(1)] - rev_rates[INDEX(1)]) * pres_mod[INDEX(1)];
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(1)] - rev_rates[INDEX(1)]) * pres_mod[INDEX(1)];
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] = (fwd_rates[INDEX(1)] - rev_rates[INDEX(1)]) * pres_mod[INDEX(1)];

  //rxn 2
  //sp 0
  sp_rates[INDEX(0)] = -(fwd_rates[INDEX(2)] - rev_rates[INDEX(2)]);
  //sp 1
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(2)] - rev_rates[INDEX(2)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(2)] - rev_rates[INDEX(2)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(2)] - rev_rates[INDEX(2)]);

  //rxn 3
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(3)] - rev_rates[INDEX(3)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(3)] - rev_rates[INDEX(3)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(3)] - rev_rates[INDEX(3)]);
  //sp 6
  sp_rates[INDEX(6)] = -(fwd_rates[INDEX(3)] - rev_rates[INDEX(3)]);

  //rxn 4
  sp_rates[INDEX(1)] = shared_temp[threadIdx.x];
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(4)] - rev_rates[INDEX(4)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(4)] - rev_rates[INDEX(4)]);
  //sp 6
  shared_temp[threadIdx.x] = (fwd_rates[INDEX(4)] - rev_rates[INDEX(4)]);
  //sp 7
  sp_rates[INDEX(7)] = -(fwd_rates[INDEX(4)] - rev_rates[INDEX(4)]);

  //rxn 5
  //sp 1
  sp_rates[INDEX(1)] -= (fwd_rates[INDEX(5)] - rev_rates[INDEX(5)]) * pres_mod[INDEX(2)];
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(5)] - rev_rates[INDEX(5)]) * pres_mod[INDEX(2)];
  //sp 6
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(5)] - rev_rates[INDEX(5)]) * pres_mod[INDEX(2)];

  //rxn 6
  sp_rates[INDEX(2)] = shared_temp[threadIdx.x + 3 * blockDim.x];
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] = -(fwd_rates[INDEX(6)] - rev_rates[INDEX(6)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(6)] - rev_rates[INDEX(6)]);
  //sp 6
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(6)] - rev_rates[INDEX(6)]);

  //rxn 7
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(7)] - rev_rates[INDEX(7)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(7)] - rev_rates[INDEX(7)]);
  //sp 6
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(7)] - rev_rates[INDEX(7)]);

  //rxn 8
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(8)] - rev_rates[INDEX(8)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(8)] - rev_rates[INDEX(8)]);
  //sp 6
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(8)] - rev_rates[INDEX(8)]);

  //rxn 9
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(9)] - rev_rates[INDEX(9)]);
  //sp 2
  sp_rates[INDEX(2)] += (fwd_rates[INDEX(9)] - rev_rates[INDEX(9)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(9)] - rev_rates[INDEX(9)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(9)] - rev_rates[INDEX(9)]);

  //rxn 10
  sp_rates[INDEX(6)] += shared_temp[threadIdx.x];
  //sp 0
  shared_temp[threadIdx.x] = (fwd_rates[INDEX(10)] - rev_rates[INDEX(10)]) * pres_mod[INDEX(3)];
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(10)] - rev_rates[INDEX(10)]) * pres_mod[INDEX(3)];

  //rxn 11
  //sp 0
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(11)] - rev_rates[INDEX(11)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(11)] - rev_rates[INDEX(11)]);

  //rxn 12
  //sp 0
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(12)] - rev_rates[INDEX(12)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(12)] - rev_rates[INDEX(12)]);

  //rxn 13
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(13)] - rev_rates[INDEX(13)]) * pres_mod[INDEX(4)];
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(13)] - rev_rates[INDEX(13)]) * pres_mod[INDEX(4)];
  //sp 5
  sp_rates[INDEX(5)] = (fwd_rates[INDEX(13)] - rev_rates[INDEX(13)]) * pres_mod[INDEX(4)];

  //rxn 14
  sp_rates[INDEX(3)] = shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(14)] - rev_rates[INDEX(14)]);
  //sp 2
  sp_rates[INDEX(2)] += (fwd_rates[INDEX(14)] - rev_rates[INDEX(14)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(14)] - rev_rates[INDEX(14)]);
  //sp 6
  shared_temp[threadIdx.x + 2 * blockDim.x] = -(fwd_rates[INDEX(14)] - rev_rates[INDEX(14)]);

  //rxn 15
  //sp 0
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(15)] - rev_rates[INDEX(15)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(15)] - rev_rates[INDEX(15)]);
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(15)] - rev_rates[INDEX(15)]);
  //sp 6
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(15)] - rev_rates[INDEX(15)]);

  //rxn 16
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(16)] - rev_rates[INDEX(16)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += 2.0 * (fwd_rates[INDEX(16)] - rev_rates[INDEX(16)]);
  //sp 6
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(16)] - rev_rates[INDEX(16)]);

  //rxn 17
  //sp 0
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(17)] - rev_rates[INDEX(17)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(17)] - rev_rates[INDEX(17)]);
  //sp 6
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(17)] - rev_rates[INDEX(17)]);
  //sp 7
  sp_rates[INDEX(7)] -= (fwd_rates[INDEX(17)] - rev_rates[INDEX(17)]);

  //rxn 18
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(18)] - rev_rates[INDEX(18)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(18)] - rev_rates[INDEX(18)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(18)] - rev_rates[INDEX(18)]);
  //sp 7
  sp_rates[INDEX(7)] -= (fwd_rates[INDEX(18)] - rev_rates[INDEX(18)]);

  //rxn 19
  //sp 0
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(19)] - rev_rates[INDEX(19)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(19)] - rev_rates[INDEX(19)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(19)] - rev_rates[INDEX(19)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(19)] - rev_rates[INDEX(19)]);

  //rxn 20
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(20)] - rev_rates[INDEX(20)]) * pres_mod[INDEX(5)];
  //sp 7
  sp_rates[INDEX(7)] += (fwd_rates[INDEX(20)] - rev_rates[INDEX(20)]) * pres_mod[INDEX(5)];

  //rxn 21
  sp_rates[INDEX(6)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 2
  sp_rates[INDEX(2)] += (fwd_rates[INDEX(21)] - rev_rates[INDEX(21)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(21)] - rev_rates[INDEX(21)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] = (fwd_rates[INDEX(21)] - rev_rates[INDEX(21)]);

  //rxn 22
  sp_rates[INDEX(1)] += shared_temp[threadIdx.x + 3 * blockDim.x];
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(22)] - rev_rates[INDEX(22)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(22)] - rev_rates[INDEX(22)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(22)] - rev_rates[INDEX(22)]);
  //sp 6
  shared_temp[threadIdx.x + 3 * blockDim.x] = -(fwd_rates[INDEX(22)] - rev_rates[INDEX(22)]);

  //rxn 23
  sp_rates[INDEX(0)] += shared_temp[threadIdx.x];
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(23)] - rev_rates[INDEX(23)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(23)] - rev_rates[INDEX(23)]);
  //sp 6
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(23)] - rev_rates[INDEX(23)]);
  //sp 7
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(23)] - rev_rates[INDEX(23)]);

  //rxn 24
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(24)] - rev_rates[INDEX(24)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(24)] - rev_rates[INDEX(24)]);
  //sp 6
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(24)] - rev_rates[INDEX(24)]);
  //sp 7
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(24)] - rev_rates[INDEX(24)]);

  //rxn 25
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(25)] - rev_rates[INDEX(25)]);
  //sp 6
  shared_temp[threadIdx.x + 3 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(25)] - rev_rates[INDEX(25)]);
  //sp 7
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(25)] - rev_rates[INDEX(25)]);

  //rxn 26
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(26)] - rev_rates[INDEX(26)]);
  //sp 6
  shared_temp[threadIdx.x + 3 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(26)] - rev_rates[INDEX(26)]);
  //sp 7
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(26)] - rev_rates[INDEX(26)]);

  //rxn 27
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(27)] - rev_rates[INDEX(27)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(27)] - rev_rates[INDEX(27)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(27)] - rev_rates[INDEX(27)]);
  //sp 6
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(27)] - rev_rates[INDEX(27)]);

  //sp 8
  (*dy_N) = 0.0;
  sp_rates[INDEX(4)] = shared_temp[threadIdx.x + 1 * blockDim.x];
  sp_rates[INDEX(5)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  sp_rates[INDEX(6)] += shared_temp[threadIdx.x + 3 * blockDim.x];
  sp_rates[INDEX(7)] += shared_temp[threadIdx.x];
} // end eval_spec_rates

