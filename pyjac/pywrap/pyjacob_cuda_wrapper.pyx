import numpy as np
cimport numpy as np
cimport cython

cdef extern from "pyjacob.cuh":
    void run(int num, int padded, const double* pres, const double* mass_frac,
            double* conc, double* fwd_rxn_rates, double* rev_rxn_rates,
            double* pres_mod, double* spec_rates, double* dy, double* jac);
    int init(int);
    void cleanup();

@cython.boundscheck(False)
def py_cuinit(int num):
    return init(num)

@cython.boundscheck(False)
def py_cuclean():
    cleanup()


@cython.boundscheck(False)
def py_cujac(int num,
            int padded,
            np.ndarray[np.double_t, mode='c'] pres,
            np.ndarray[np.double_t, mode='c'] y,
            np.ndarray[np.double_t, mode='c'] conc,
            np.ndarray[np.double_t, mode='c'] fwd_rates,
            np.ndarray[np.double_t, mode='c'] rev_rates,
            np.ndarray[np.double_t, mode='c'] pres_mod,
            np.ndarray[np.double_t, mode='c'] spec_rates,
            np.ndarray[np.double_t, mode='c'] dy,
            np.ndarray[np.double_t, mode='c'] jac):
    run(num, padded, &pres[0], &y[0], &conc[0], &fwd_rates[0], &rev_rates[0],
             &pres_mod[0], &spec_rates[0], &dy[0], &jac[0])