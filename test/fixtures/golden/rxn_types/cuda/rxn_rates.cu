  extern volatile __shared__ double shared_temp[];
#include "rates.cuh"
__device__ void eval_rxn_rates (const double T, const double pres, const double * __restrict__ C, double * __restrict__ fwd_rxn_rates, double * __restrict__ rev_rxn_rates, double * __restrict__ dot_prod) {
  extern volatile __shared__ double shared_temp[];
  register double logT = log(T);

  register double kf;
  register double Kc;
  register double Tred;
  register double Pred;
  double cheb_temp_0, cheb_temp_1;
  register double kf2;

  //rxn 0
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(1)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(4)];
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(2)];
  shared_temp[threadIdx.x] = C[INDEX(3)];
  kf = exp(3.0905899130912033e+01 - 0.671 * logT - (8.5753640703524688e+03 / T));
  fwd_rxn_rates[INDEX(0)] = shared_temp[threadIdx.x + 3 * blockDim.x] * shared_temp[threadIdx.x] * kf;
  if (T <= 1000.0) {
    Kc = (-2.1408110450000000e+00 + 8.7782617000000007e-01 * logT + T * (-1.3419511003526661e-03 + T * (2.3561672699931997e-07 + T * (-2.7325373525067921e-11 + 1.1652302046386620e-14 * T))) - 8.3276234199999999e+03 / T);
  } else {
    Kc = (4.3747157639999994e+00 + -1.2022939999999949e-01 * logT + T * (-5.1031595730785138e-04 + T * (1.5440338995730085e-07 + T * (-2.5619540480403969e-11 + 1.7320676705901320e-15 * T))) - 8.6910339200000035e+03 / T);
  }

  Kc = 1.0000000000000000e+00 * exp(Kc);
  rev_rxn_rates[INDEX(0)] = shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 2 * blockDim.x] * kf / Kc;

  //rxn 1
  fwd_rxn_rates[INDEX(1)] = shared_temp[threadIdx.x + 1 * blockDim.x] * C[INDEX(0)] * exp(3.8262472284067424e+00 + 2.7 * logT - (3.1501542797022739e+03 / T));
  //rxn 2
  fwd_rxn_rates[INDEX(2)] = C[INDEX(0)] * shared_temp[threadIdx.x + 2 * blockDim.x] * exp(1.2063356343328579e+01 + 1.51 * logT - (1.7260429999007667e+03 / T));
  //rxn 3
  fwd_rxn_rates[INDEX(3)] = C[INDEX(5)] * shared_temp[threadIdx.x + 3 * blockDim.x] * exp(1.1723186390453424e+01 + 1.4 * logT - (9.0579516029777842e+03 / T));
  //rxn 4
  kf = exp(2.9181557920442486e+01 - 1.0 * logT);
  fwd_rxn_rates[INDEX(4)] = shared_temp[threadIdx.x + 3 * blockDim.x] * shared_temp[threadIdx.x + 1 * blockDim.x] * kf;
  if (T <= 1000.0) {
    Kc = (-3.2924395000000217e-02 + -6.7625167000000008e-01 * logT + T * (4.3900065964733344e-04 + T * (-3.3752092466734680e-07 + T * (1.8724440897493195e-10 + -3.7427250453613386e-14 * T))) - -5.0980838539999997e+04 / T);
  } else {
    Kc = (2.1155734939999999e+00 + -9.7653312000000003e-01 * logT + T * (3.1720192639214862e-04 + T * (1.4092792157300863e-08 + T * (-6.4940309137373039e-12 + 5.2564500959013221e-16 * T))) - -5.0832581999999995e+04 / T);
  }

  Kc = 8.2057366080959690e-02 * exp(Kc);
  rev_rxn_rates[INDEX(1)] = shared_temp[threadIdx.x + 2 * blockDim.x] * kf / Kc;

  //rxn 5
  kf = exp(2.2355638720663009e+01 + 0.44 * logT);
  fwd_rxn_rates[INDEX(5)] = shared_temp[threadIdx.x + 3 * blockDim.x] * shared_temp[threadIdx.x] * kf;
  if (T <= 1000.0) {
    Kc = (2.4863279230000002e+00 + -9.8065835000000012e-01 * logT + T * (-8.7619317535266614e-04 + T * (1.8851645153326528e-06 + T * (-1.2162578593584016e-09 + 3.0242614354638670e-13 * T))) - -2.4114908299999999e+04 / T);
  } else {
    Kc = (5.4388072400000009e-01 + -7.6532694999999951e-01 * logT + T * (3.7836630654214875e-04 + T * (2.0718083807300866e-08 + T * (-7.9353483554039681e-12 + 5.4404627009013211e-16 * T))) - -2.4273345466999999e+04 / T);
  }

  Kc = 8.2057366080959690e-02 * exp(Kc);
  rev_rxn_rates[INDEX(2)] = C[INDEX(6)] * kf / Kc;

  //rxn 6
  kf = exp(3.3318335397877441e+01 - (2.4370922007345227e+04 / T));
  fwd_rxn_rates[INDEX(6)] = C[INDEX(7)] * kf;
  if (T <= 1000.0) {
    Kc = (-7.3508198260000004e+00 + 2.7079181700000001e+00 * logT + T * (-2.1299063115000003e-03 + T * (-1.2496155466666669e-06 + T * (1.1512345533333333e-09 + -2.9481571150000003e-13 * T))) - 2.4932743220000000e+04 / T);
  } else {
    Kc = (4.0164630899999985e+00 + 1.0207724900000006e+00 * logT + T * (-1.9057287540000000e-03 + T * (3.5906711766666662e-07 + T * (-4.5589858100000000e-11 + 2.6136652849999998e-15 * T))) - 2.5579101699999999e+04 / T);
  }

  Kc = 1.2186596374704216e+01 * exp(Kc);
  rev_rxn_rates[INDEX(3)] = shared_temp[threadIdx.x + 2 * blockDim.x] * shared_temp[threadIdx.x + 2 * blockDim.x] * kf / Kc;

  //rxn 7
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(6)];
  if (pres <= 1.0132e+04) {
    kf = exp(2.2680398491738462e+01 - (1.4844976238213590e+02 / T));
  } else if ((pres > 1.0132e+04) && (pres <= 1.0132e+05)) {
    kf = log(exp(2.2680398491738462e+01 - (1.4844976238213590e+02 / T)));
    kf2 = log(exp(2.4982983584732509e+01 - (1.4844976238213590e+02 / T)));
    kf = exp(kf + (kf2 - kf) * (log(pres) - 9.2235033585024642e+00) / 2.3025850929940450e+00);
  } else if ((pres > 1.0132e+05) && (pres <= 1.0132e+06)) {
    kf = log(exp(2.4982983584732509e+01 - (1.4844976238213590e+02 / T)));
    kf2 = log(exp(2.7285568677726555e+01 - (1.4844976238213590e+02 / T)));
    kf = exp(kf + (kf2 - kf) * (log(pres) - 1.1526088451496509e+01) / 2.3025850929940450e+00);
  } else if (pres > 1.0132e+06) {
    kf = exp(2.7285568677726555e+01 - (1.4844976238213590e+02 / T));
  }
  fwd_rxn_rates[INDEX(7)] = shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 3 * blockDim.x] * kf;
  if (T <= 1000.0) {
    Kc = (-4.6600633630000008e+00 + 1.1822328500000001e+00 * logT + T * (-2.6757265352666734e-05 + T * (-1.9870687130006801e-06 + T * (1.3761768948082653e-09 + -3.2820109195361340e-13 * T))) - -1.8538306819999998e+04 / T);
  } else {
    Kc = (5.9464085339999997e+00 + -3.3143557000000001e-01 * logT + T * (-5.7148033745785135e-04 + T * (1.4777809830730085e-07 + T * (-2.4178223038737304e-11 + 1.7136664100901320e-15 * T))) - -1.7868202612999998e+04 / T);
  }

  Kc = 1.0000000000000000e+00 * exp(Kc);
  rev_rxn_rates[INDEX(4)] = shared_temp[threadIdx.x + 2 * blockDim.x] * shared_temp[threadIdx.x + 2 * blockDim.x] * kf / Kc;

  //rxn 8
  Tred = ((2.0 / T) - 3.73333333e-03) / -2.93333333e-03;
  Pred = (2.0 * log10(pres) - 1.00114332e+01) / 4.00000000e+00;
  cheb_temp_0 = 1;
  cheb_temp_1 = Pred;
  dot_prod[INDEX(0)]= 7.00000000e+00 + Pred * 1.00000000e-01;
  dot_prod[INDEX(1)]= 5.00000000e-01 + Pred * 2.00000000e-02;
  dot_prod[INDEX(2)]= 1.00000000e-01 + Pred * -5.00000000e-03;
  dot_prod[INDEX(3)]= -2.00000000e-02 + Pred * 1.00000000e-03;
  cheb_temp_0 = 2 * Pred * cheb_temp_1 - cheb_temp_0;
  dot_prod[INDEX(0)] += 1.00000000e-02 * cheb_temp_0;
  dot_prod[INDEX(1)] += -1.00000000e-03 * cheb_temp_0;
  dot_prod[INDEX(2)] += 2.00000000e-04 * cheb_temp_0;
  dot_prod[INDEX(3)] += -5.00000000e-05 * cheb_temp_0;
  cheb_temp_0 = 1;
  cheb_temp_1 = Tred;
  kf = dot_prod[INDEX(0)] + Tred * dot_prod[INDEX(1)];
  cheb_temp_0 = 2 * Tred * cheb_temp_1 - cheb_temp_0;
  kf += dot_prod[INDEX(2)] * cheb_temp_0;
  cheb_temp_1 = 2 * Tred * cheb_temp_0 - cheb_temp_1;
  kf += dot_prod[INDEX(3)] * cheb_temp_1;
  kf = exp10(kf);
  fwd_rxn_rates[INDEX(8)] = shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 2 * blockDim.x] * kf;
  if (T <= 1000.0) {
    Kc = (-4.9137695000000026e-01 + -3.1271651999999905e-01 * logT + T * (1.0586348849999999e-03 + T * (-1.5680872316666664e-06 + T * (1.0823547516666667e-09 + -2.8203297000000007e-13 * T))) - -3.5267558859999997e+04 / T);
  } else {
    Kc = (2.9517713799999998e+00 + -7.9356824000000037e-01 * logT + T * (4.3587786700000021e-04 + T * (-6.9147710833333339e-08 + T * (7.1773627999999976e-12 + -2.9010321499999993e-16 * T))) - -3.5063268533000002e+04 / T);
  }

  Kc = 1.0000000000000000e+00 * exp(Kc);
  rev_rxn_rates[INDEX(5)] = C[INDEX(5)] * shared_temp[threadIdx.x] * kf / Kc;

  //rxn 9
  kf = exp(1.8683045008419857e+01 - (-8.2024783960298828e+02 / T));
  fwd_rxn_rates[INDEX(9)] = shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 1 * blockDim.x] * kf;
  if (T <= 1000.0) {
    Kc = (2.0442853999999988e-01 + -5.4502697000000033e-01 * logT + T * (2.9793422214999996e-03 + T * (-2.6226176816666659e-06 + T * (1.4412002008333336e-09 + -3.3581152400000006e-13 * T))) - -1.9356141739999999e+04 / T);
  } else {
    Kc = (1.3860647200000007e+00 + -5.8688111000000109e-01 * logT + T * (9.5588210999999991e-04 + T * (-2.3200710316666662e-07 + T * (2.9346983416666660e-11 + -1.4440451450000000e-15 * T))) - -1.9173958845999998e+04 / T);
  }

  Kc = 1.0000000000000000e+00 * exp(Kc);
  rev_rxn_rates[INDEX(6)] = C[INDEX(7)] * shared_temp[threadIdx.x] * kf / Kc;

  //rxn 10
  kf = exp(2.6763520548223823e+01 - (6.0386344019851895e+03 / T));
  fwd_rxn_rates[INDEX(10)] = shared_temp[threadIdx.x + 1 * blockDim.x] * shared_temp[threadIdx.x + 1 * blockDim.x] * kf;
  if (T <= 1000.0) {
    Kc = (2.0442853999999988e-01 + -5.4502697000000033e-01 * logT + T * (2.9793422214999996e-03 + T * (-2.6226176816666659e-06 + T * (1.4412002008333336e-09 + -3.3581152400000006e-13 * T))) - -1.9356141739999999e+04 / T);
  } else {
    Kc = (1.3860647200000007e+00 + -5.8688111000000109e-01 * logT + T * (9.5588210999999991e-04 + T * (-2.3200710316666662e-07 + T * (2.9346983416666660e-11 + -1.4440451450000000e-15 * T))) - -1.9173958845999998e+04 / T);
  }

  Kc = 1.0000000000000000e+00 * exp(Kc);
  rev_rxn_rates[INDEX(7)] = C[INDEX(7)] * shared_temp[threadIdx.x] * kf / Kc;

} // end eval_rxn_rates

