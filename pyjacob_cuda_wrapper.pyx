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
def py_dydt(np.ndarray[np.double_t] pres,
            np.ndarray[np.double_t, ndim=2] y,
            np.ndarray[np.double_t, ndim=2] dy):
    cdef int num
    try:
        num = y.shape[1]
    except:
        num = 1
    cdef np.ndarray[np.double_t] temp_p = pres.T
    cdef np.ndarray[np.double_t, ndim=2] temp_y = y.T
    cdef np.ndarray[np.double_t, ndim=2] temp_dy = dy.T
    cu_dydt(num, &temp_p[0], &temp_y[0, 0], &temp_dy[0, 0])
    dy = temp_dy.T

@cython.boundscheck(False)
def py_eval_jacobian(np.ndarray[np.double_t] pres,
            np.ndarray[np.double_t, ndim=2] y,
            np.ndarray[np.double_t, ndim=2] jac):
    cdef int num
    try:
        num = y.shape[1]
    except:
        num = 1
    cdef np.ndarray[np.double_t] temp_p = pres.T
    cdef np.ndarray[np.double_t, ndim=2] temp_y = y.T
    cdef np.ndarray[np.double_t, ndim=2] temp_jac = jac.T
    cu_eval_jacob(num, &temp_p[0], &temp_y[0, 0], &temp_jac[0, 0])
    jac = temp_jac.T

@cython.boundscheck(False)
def py_eval_rxn_rates(np.ndarray[np.double_t] T,
            np.ndarray[np.double_t] pres,
            np.ndarray[np.double_t, ndim=2] C,
            np.ndarray[np.double_t, ndim=2] fwd_rxn_rates,
            np.ndarray[np.double_t, ndim=2] rev_rxn_rates):
    cdef int num
    try:
        num = C.shape[1]
    except:
        num = 1
    cdef np.ndarray[np.double_t] temp_t = T.T
    cdef np.ndarray[np.double_t] temp_p = pres.T
    cdef np.ndarray[np.double_t, ndim=2] temp_c = C.T
    cdef np.ndarray[np.double_t, ndim=2] temp_fwd = fwd_rxn_rates.T
    cdef np.ndarray[np.double_t, ndim=2] temp_rev = rev_rxn_rates.T
    cu_eval_rxn_rates(num, &temp_t[0], &temp_p[0], &temp_c[0, 0], &temp_fwd[0, 0], &temp_rev[0, 0])
    fwd_rxn_rates = temp_fwd.T
    rev_rxn_rates = temp_rev.T

@cython.boundscheck(False)
def py_eval_spec_rates(np.ndarray[np.double_t, ndim=2] fwd_rxn_rates,
            np.ndarray[np.double_t, ndim=2] rev_rxn_rates,
            np.ndarray[np.double_t, ndim=2] pres_mod,
            np.ndarray[np.double_t, ndim=2] sp_rates):
    cdef int num
    try:
        num = fwd_rxn_rates.shape[1]
    except:
        num = 1
    cdef np.ndarray[np.double_t, ndim=2] temp_fwd = fwd_rxn_rates.T
    cdef np.ndarray[np.double_t, ndim=2] temp_rev = rev_rxn_rates.T
    cdef np.ndarray[np.double_t, ndim=2] temp_pres = pres_mod.T
    cdef np.ndarray[np.double_t, ndim=2] temp_sp = sp_rates.T
    cu_eval_spec_rates(num, &temp_fwd[0, 0], &temp_rev[0, 0], &temp_pres[0, 0], &temp_sp[0, 0])
    sp_rates = temp_sp.T

@cython.boundscheck(False)
def py_get_rxn_pres_mod(np.ndarray[np.double_t] T,
            np.ndarray[np.double_t] pres,
            np.ndarray[np.double_t, ndim=2] C,
            np.ndarray[np.double_t, ndim=2] pres_mod):
    cdef int num
    try:
        num = C.shape[1]
    except:
        num = 1
    cdef np.ndarray[np.double_t] temp_t = T.T
    cdef np.ndarray[np.double_t] temp_p = pres.T 
    cdef np.ndarray[np.double_t, ndim=2] temp_c = C.T
    cdef np.ndarray[np.double_t, ndim=2] temp_pres = pres_mod.T
    cu_get_rxn_pres_mod(num, &temp_t[0], &temp_p[0], &temp_c[0, 0], &temp_pres[0, 0])
    pres_mod = temp_pres.T

@cython.boundscheck(False)
def py_eval_conc(np.ndarray[np.double_t, mode='c'] T,
            np.ndarray[np.double_t, mode='c'] pres,
            np.ndarray[np.double_t, ndim=2, mode='c'] mass_frac,
            np.ndarray[np.double_t, mode='c'] mw_avg,
            np.ndarray[np.double_t, mode='c'] rho,
            np.ndarray[np.double_t, ndim=2, mode='c'] conc):
    cdef int num
    try:
        num = T.shape[0]
    except:
        num = 1
    cdef np.ndarray[np.double_t] temp_mass = mass_frac.flatten(order='F')
    print(mass_frac)
    print(temp_mass)
    cdef np.ndarray[np.double_t] temp_conc = conc.flatten(order='F')
    cu_eval_conc(num, &T[0], &pres[0], &temp_mass[0], &mw_avg[0], &rho[0], &temp_conc[0])
    conc = temp_conc.reshape((num, -1), order='F').astype(np.dtype('d'), order='C')
    print(conc, rho, mw_avg)