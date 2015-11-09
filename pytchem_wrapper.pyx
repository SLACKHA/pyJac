#! /usr/bin/env python2.7

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "py_tchem.h":
    void tc_eval_jacob (const char* mechname, const char* thermoname,
                        const int num, const double t, const double* pres,
                        const double* y,
                        double* jac
                        )

@cython.boundscheck(False)
def py_eval_jacobian(str mechname, str thermoname, int num,
                     np.double_t t,
                     np.ndarray[np.double_t, mode='c'] pres,
                     np.ndarray[np.double_t, mode='c'] y,
                     np.ndarray[np.double_t, mode='c'] jac
                     ):
    cdef bytes py_bytes_mech = mechname.encode()
    cdef char* c_mech = py_bytes_mech
    cdef bytes py_bytes_therm = thermoname.encode()
    cdef char* c_therm = py_bytes_therm

    tc_eval_jacob(c_mech, c_therm, num, t, &pres[0], &y[0], &jac[0])
