#! /usr/bin/env python2.7

import numpy as np
cimport numpy as np


cdef extern from "out/dydt.h":
    void dydt(double t, double pres, double* y, double* dy)

def py_dydt(np.double_t t,
            np.double_t pres,
            np.ndarray[np.double_t] y,
            np.ndarray[np.double_t] dy):
    dydt(t, pres, &y[0], &dy[0])