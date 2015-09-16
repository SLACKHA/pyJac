#! /usr/bin/env python2.7

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "pyjacob.cuh":
    void cu_dydt(const int num, const double* pres, const double* y, double* dy)
    void cu_eval_jacob (const int num, const double* pres, const double* y, double* jac)
    void cu_eval_rxn_rates (const int num, const double* T, const double* pres, const double * C, double * fwd_rxn_rates, double * rev_rxn_rates)
    void cu_eval_spec_rates (const int num, const double* fwd_rates, const double* rev_rates, const double* pres_mod, double * sp_rates)
    void cu_get_rxn_pres_mod (const int num, const double* T, const double* pres, const double* C, double* pres_mod)
    void cu_eval_conc (const int num, const double * T, const double * pres, const double * mass_frac, double * mw_avg, double * rho, double * conc)

@cython.boundscheck(False)
def py_dydt(int num,
            np.ndarray[np.double_t, mode='c'] pres,
            np.ndarray[np.double_t, mode='c'] y,
            np.ndarray[np.double_t, mode='c'] dy):
    cu_dydt(num, &pres[0], &y[0], &dy[0])

@cython.boundscheck(False)
def py_eval_jacobian(int num,
            np.ndarray[np.double_t, mode='c'] pres,
            np.ndarray[np.double_t, mode='c'] y,
            np.ndarray[np.double_t, mode='c'] jac):
    cu_eval_jacob(num, &pres[0], &y[0], &jac[0])

@cython.boundscheck(False)
def py_eval_rxn_rates(int num,
            np.ndarray[np.double_t, mode='c'] T,
            np.ndarray[np.double_t, mode='c'] pres,
            np.ndarray[np.double_t, mode='c'] C,
            np.ndarray[np.double_t, mode='c'] fwd_rxn_rates,
            np.ndarray[np.double_t, mode='c'] rev_rxn_rates):
    cu_eval_rxn_rates(num, &T[0], &pres[0], &C[0], &fwd_rxn_rates[0], &rev_rxn_rates[0])

@cython.boundscheck(False)
def py_eval_spec_rates(int num,
            np.ndarray[np.double_t, mode='c'] fwd_rxn_rates,
            np.ndarray[np.double_t, mode='c'] rev_rxn_rates,
            np.ndarray[np.double_t, mode='c'] pres_mod,
            np.ndarray[np.double_t, mode='c'] sp_rates):
    cu_eval_spec_rates(num, &fwd_rxn_rates[0], &rev_rxn_rates[0], &pres_mod[0], &sp_rates[0])

@cython.boundscheck(False)
def py_get_rxn_pres_mod(int num,
            np.ndarray[np.double_t, mode='c'] T,
            np.ndarray[np.double_t, mode='c'] pres,
            np.ndarray[np.double_t, mode='c'] C,
            np.ndarray[np.double_t, mode='c'] pres_mod):
    cu_get_rxn_pres_mod(num, &T[0], &pres[0], &C[0], &pres_mod[0])

@cython.boundscheck(False)
def py_eval_conc(int num,
            np.ndarray[np.double_t, mode='c'] T,
            np.ndarray[np.double_t, mode='c'] pres,
            np.ndarray[np.double_t, mode='c'] mass_frac,
            np.ndarray[np.double_t, mode='c'] mw_avg,
            np.ndarray[np.double_t, mode='c'] rho,
            np.ndarray[np.double_t, mode='c'] conc):
    cu_eval_conc(num, &T[0], &pres[0], &mass_frac[0], &mw_avg[0], &rho[0], &conc[0])