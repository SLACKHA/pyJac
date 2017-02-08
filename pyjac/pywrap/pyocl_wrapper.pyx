import cython
import numpy as np
cimport numpy as np

cdef extern from "species_rates_kernel.h":
    void species_rates_kernel(np.uint_t problem_size, np.uint_t num_devices, double* T, double* P,
                        double* conc, double* wdot)
    void finalize()
    void compiler()

@cython.boundscheck(False)
@cython.wraparound(False)
def species_rates(np.uint_t problem_size,
            np.uint_t num_devices,
            np.ndarray[np.float64_t] T,
            np.ndarray[np.float64_t] P,
            np.ndarray[np.float64_t] conc,
            np.ndarray[np.float64_t] wdot):
    species_rates_kernel(problem_size, num_devices, &T[0], &P[0], &conc[0], &wdot[0])
    return None

def __cinit__(self):
    compiler()

def __dealloc__(self):
    finalize()