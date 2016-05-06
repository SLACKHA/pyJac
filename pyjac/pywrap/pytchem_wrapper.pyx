import numpy as np
cimport numpy as np
cimport cython

cdef extern from "py_tchem.h":
    void tc_eval_jacob (char* mechname, char* thermoname,
                        const int num, const double* pres,
                        double* y, double* conc, double* fwd_rates,
                        double* rev_rates, double* spec_rates,
                        double* dydt, double* jac
                        )

@cython.boundscheck(False)
def py_eval_jacobian(str mechname, str thermoname, int num,
                     np.ndarray[np.double_t, mode='c'] pres,
                     np.ndarray[np.double_t, mode='c'] y,
                     np.ndarray[np.double_t, mode='c'] conc,
                     np.ndarray[np.double_t, mode='c'] fwd_rates,
                     np.ndarray[np.double_t, mode='c'] rev_rates,
                     np.ndarray[np.double_t, mode='c'] spec_rates,
                     np.ndarray[np.double_t, mode='c'] dydt,
                     np.ndarray[np.double_t, mode='c'] jac
                     ):
    cdef bytes py_bytes_mech = mechname.encode()
    cdef char* c_mech = py_bytes_mech
    cdef bytes py_bytes_therm = thermoname.encode()
    cdef char* c_therm = py_bytes_therm

    tc_eval_jacob(c_mech, c_therm, num, &pres[0], &y[0], &conc[0],
                  &fwd_rates[0], &rev_rates[0], &spec_rates[0],
                  &dydt[0], &jac[0]
                  )
