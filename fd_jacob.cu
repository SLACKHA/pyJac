#include <math.h>
#include <float.h>
#include "header.cuh"
#include "dydt.cuh"
#include "gpu_macros.cuh"

#define ATOL (1e-15)
#define RTOL (1e-8)
#define FD_ORD 1

// Finite difference coefficients
#if FD_ORD == 2
  __constant__ double x_coeffs[FD_ORD] = {-1.0, 1.0};
  __constant__ double y_coeffs[FD_ORD] = {-0.5, 0.5};
#elif FD_ORD == 4
  __constant__ double x_coeffs[FD_ORD] = {-2.0, -1.0, 1.0, 2.0};
  __constant__ double y_coeffs[FD_ORD] = {1.0 / 12.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 12.0};
#elif FD_ORD == 6
  __constant__ double x_coeffs[FD_ORD] = {-3.0, -2.0, - 1.0, 1.0, 2.0, 3.0};
  __constant__ double y_coeffs[FD_ORD] = {-1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};
#endif

__device__
void eval_jacob (const double t, const double pres, const double * cy, double * jac) {
  double y[NSP];
  double dy[NSP];
  double ewt[NSP];
  
  #pragma unroll
  for (int i = 0; i < NSP; ++i) {
    ewt[i] = ATOL + (RTOL * fabs(cy[i]));
    y[i] = cy[i];
  }

  dydt (t, pres, y, dy);
  
  // unit roundoff of machine
  double srur = sqrt(DBL_EPSILON);
  
  double sum = 0.0;
  #pragma unroll
  for (int i = 0; i < NSP; ++i) {
    sum += (ewt[i] * dy[i]) * (ewt[i] * dy[i]);
  }
  double fac = sqrt(sum / ((double)(NSP)));
  double r0 = 1000.0 * RTOL * DBL_EPSILON * ((double)(NSP)) * fac;
  double f_temp[NSP];
  
  #pragma unroll
  for (int j = 0; j < NSP; ++j) {
    double yj_orig = y[j];
    double r = fmax(srur * fabs(yj_orig), r0 / ewt[j]);
    
    #if FD_ORD == 1
      y[j] = yj_orig + r;
      dydt (t, pres, y, f_temp);
        
      #pragma unroll
      for (int i = 0; i < NSP; ++i) {
        jac[i + NSP*j] = (f_temp[i] - dy[i]) / r;
      }
    #else
      #pragma unroll
      for (int i = 0; i < NSP; ++i) {
        jac[i + NSP*j] = 0.0;
      }
      #pragma unroll
      for (int k = 0; k < FD_ORD; ++k) {
        y[j] = yj_orig + x_coeffs[k] * r;
        dydt (t, pres, y, f_temp);
        
        #pragma unroll
        for (int i = 0; i < NSP; ++i) {
          jac[i + NSP*j] += y_coeffs[k] * f_temp[i];
        }
      }
      #pragma unroll
      for (int i = 0; i < NSP; ++i) {
        jac[i + NSP*j] /= r;
      }
    #endif
    
    y[j] = yj_orig;
  }
  
}