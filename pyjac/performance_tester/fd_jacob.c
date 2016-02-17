#include "header.h"
#include "dydt.h"
#include <math.h>
#include <float.h>

#define ATOL (1e-15)
#define RTOL (1e-8)
#define FD_ORD 1

void eval_jacob (const double t, const double pres, const double * cy, double * jac) {
  
  double y[NSP];
  memcpy(y, cy, NSP * sizeof(double));
  double dy[NSP];
  dydt (t, pres, y, dy);
  
  // Finite difference coefficients
  #if FD_ORD != 1
  double x_coeffs[FD_ORD];
  double y_coeffs[FD_ORD];

  if (FD_ORD == 2) {
    // 2nd order central difference
    x_coeffs[0] = -1.0;
    x_coeffs[1] = 1.0;
    y_coeffs[0] = -0.5;
    y_coeffs[1] = 0.5;
  } else if (FD_ORD == 4) {
    // 4th order central difference
    x_coeffs[0] = -2.0;
    x_coeffs[1] = -1.0;
    x_coeffs[2] = 1.0;
    x_coeffs[3] = 2.0;
    y_coeffs[0] = 1.0 / 12.0;
    y_coeffs[1] = -2.0 / 3.0;
    y_coeffs[2] = 2.0 / 3.0;
    y_coeffs[3] = -1.0 / 12.0;
  } else if (FD_ORD == 6) {
    // 6th order central difference
    x_coeffs[0] = -3.0;
    x_coeffs[1] = -2.0;
    x_coeffs[2] = -1.0;
    x_coeffs[3] = 1.0;
    x_coeffs[4] = 2.0;
    x_coeffs[5] = 3.0;
    
    y_coeffs[0] = -1.0 / 60.0;
    y_coeffs[1] = 3.0 / 20.0;
    y_coeffs[2] = -3.0 / 4.0;
    y_coeffs[3] = 3.0 / 4.0;
    y_coeffs[4] = -3.0 / 20.0;
    y_coeffs[5] = 1.0 / 60.0;
  }
  #endif
  
  double ewt[NSP];
  #pragma unroll
  for (int i = 0; i < NSP; ++i) {
    ewt[i] = ATOL + (RTOL * fabs(y[i]));
  }
  
  // unit roundoff of machine
  double srur = sqrt(DBL_EPSILON);
  
  double sum = 0.0;
  #pragma unroll
  for (int i = 0; i < NSP; ++i) {
    sum += (ewt[i] * dy[i]) * (ewt[i] * dy[i]);
  }
  double fac = sqrt(sum / ((double)(NSP)));
  double r0 = 1000.0 * RTOL * DBL_EPSILON * ((double)(NSP)) * fac;
  
  double ftemp[NSP];
  
  #pragma unroll
  for (int j = 0; j < NSP; ++j) {
    double yj_orig = y[j];
    double r = fmax(srur * fabs(yj_orig), r0 / ewt[j]);
    
    #if FD_ORD==1
      y[j] = yj_orig + r;
      dydt (t, pres, y, ftemp);
        
      #pragma unroll
      for (int i = 0; i < NSP; ++i) {
        jac[i + NSP*j] = (ftemp[i] - dy[i]) / r;
      }
    #else
      #pragma unroll
      for (int i = 0; i < NSP; ++i) {
        jac[i + NSP*j] = 0.0;
      }
      #pragma unroll
      for (int k = 0; k < FD_ORD; ++k) {
        y[j] = yj_orig + x_coeffs[k] * r;
        dydt (t, pres, y, ftemp);
        
        #pragma unroll
        for (int i = 0; i < NSP; ++i) {
          jac[i + NSP*j] += y_coeffs[k] * ftemp[i];
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
