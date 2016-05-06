import numpy as np
# distutils: language = c++

cimport numpy as np

cdef extern from "ad_jacob.h":
    void eval_jacob (const double t, const double pres, const double* y, double* jac)

def ad_eval_jacobian(np.double_t t,
            np.double_t pres,
            np.ndarray[np.double_t] y,
            np.ndarray[np.double_t] jac):
    eval_jacob(t, pres, &y[0], &jac[0])