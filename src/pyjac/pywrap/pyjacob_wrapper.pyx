import numpy as np
cimport numpy as np

cdef extern from "dydt.h":
    void dydt(double t, double pres, double* y, double* dy)

cdef extern from "jacob.h":
    void eval_jacob (const double t, const double pres, const double* y, double* jac)

cdef extern from "rates.h":
    void eval_rxn_rates (const double T, const double pres, const double* C, double* fwd_rxn_rates, double* rev_rxn_rates)
    void eval_spec_rates (const double * fwd_rates, const double * rev_rates, const double * pres_mod, double * sp_rates, double* dy_N)
    void get_rxn_pres_mod (const double T, const double pres, const double * C, double * pres_mod)

cdef extern from "chem_utils.h":
    void eval_conc (const double T, const double pres, const double * mass_frac, double * y_N, double * mw_avg, double * rho, double * conc)

def py_dydt(np.double_t t,
            np.double_t pres,
            np.ndarray[np.double_t] y,
            np.ndarray[np.double_t] dy):
    dydt(t, pres, &y[0], &dy[0])

def py_eval_jacobian(np.double_t t,
            np.double_t pres,
            np.ndarray[np.double_t] y,
            np.ndarray[np.double_t] jac):
    eval_jacob(t, pres, &y[0], &jac[0])

def py_eval_rxn_rates(np.double_t T,
            np.double_t pres,
            np.ndarray[np.double_t] C,
            np.ndarray[np.double_t] fwd_rxn_rates,
            np.ndarray[np.double_t] rev_rxn_rates):
    eval_rxn_rates(T, pres, &C[0], &fwd_rxn_rates[0], &rev_rxn_rates[0])

def py_eval_spec_rates(np.ndarray[np.double_t] fwd_rxn_rates,
            np.ndarray[np.double_t] rev_rxn_rates,
            np.ndarray[np.double_t] pres_mod,
            np.ndarray[np.double_t] sp_rates):
    eval_spec_rates(&fwd_rxn_rates[0], &rev_rxn_rates[0], &pres_mod[0], &sp_rates[0], &sp_rates[sp_rates.shape[0] - 1])

def py_get_rxn_pres_mod(np.double_t T,
            np.double_t pres,
            np.ndarray[np.double_t] C,
            np.ndarray[np.double_t] pres_mod):
    get_rxn_pres_mod(T, pres, &C[0], &pres_mod[0])

def py_eval_conc(np.double_t T,
            np.double_t pres,
            np.ndarray[np.double_t] mass_frac,
            np.double_t mw_avg,
            np.double_t rho,
            np.ndarray[np.double_t] conc):
    eval_conc(T, pres, &mass_frac[0], &mass_frac[-1], &mw_avg, &rho, &conc[0])
