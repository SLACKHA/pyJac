#! /usr/bin/env python2.7

import numpy as np
cimport numpy as np

ctypedef np.int_t size_type

cdef extern from "pyjacob.cuh":
    void cu_dydt(const int num, const double* pres, const double* y, double* dy)
    void cu_eval_jacob (const int num, const double* pres, const double* y, double* jac)
    void cu_eval_rxn_rates (const int num, const double* T, const double pres, const double* C, double* fwd_rxn_rates, double* rev_rxn_rates)
    void cu_eval_spec_rates (const int num, const double* fwd_rates, const double* rev_rates, const double* pres_mod, double * sp_rates)
    void cu_get_rxn_pres_mod (const int num, const double* T, const double* pres, const double* C, double* pres_mod)
    void cu_eval_conc (const int num, const double * T, const double * pres, const double * mass_frac, double * mw_avg, double * rho, double * conc)

def py_dydt(np.ndarray[np.double_t] t,
            np.ndarray[np.double_t] pres,
            np.ndarray[np.double_t] y,
            np.ndarray[np.double_t] dy):
    cdef size_type num
    try:
        num = y.shape[1]
    except:
        num = 1
    cdef np.ndarray[np.double_t] temp_p = pres.T
    cdef np.ndarray[np.double_t] temp_y = y.T
    cdef np.ndarray[np.double_t] temp_dy = dy.T
    cu_dydt(num, &temp_p[0], &temp_y[0], &temp_dy[0])
    dy = temp_dy.T

def py_eval_jacobian(np.ndarray[np.double_t] t,
            np.ndarray[np.double_t] pres,
            np.ndarray[np.double_t] y,
            np.ndarray[np.double_t] jac):
    cdef size_type num
    try:
        num = y.shape[1]
    except:
        num = 1
    cdef np.ndarray[np.double_t] temp_p = pres.T
    cdef np.ndarray[np.double_t] temp_y = y.T
    cdef np.ndarray[np.double_t] temp_jac = jac.T
    cu_eval_jacob(num, &temp_p[0], &temp_y[0], &temp_jac[0])
    jac = temp_jac.T

def py_eval_rxn_rates(np.ndarray[np.double_t] T,
            np.ndarray[np.double_t] pres,
            np.ndarray[np.double_t] C,
            np.ndarray[np.double_t] fwd_rxn_rates,
            np.ndarray[np.double_t] rev_rxn_rates):
    cdef size_type num
    try:
        num = C.shape[1]
    except:
        num = 1
    cdef np.ndarray[np.double_t] temp_t = T.T
    cdef np.ndarray[np.double_t] temp_p = pres.T
    cdef np.ndarray[np.double_t] temp_c = C.T
    cdef np.ndarray[np.double_t] temp_fwd = fwd_rxn_rates.T
    cdef np.ndarray[np.double_t] temp_rev = rev_rxn_rates.T
    cu_eval_rxn_rates(num, &temp_t[0], &temp_p[0], &temp_c[0], &temp_fwd[0], &temp_rev[0])
    fwd_rxn_rates = temp_fwd.T
    rev_rxn_rates = temp_rev.T

def py_eval_spec_rates(np.ndarray[np.double_t] fwd_rxn_rates,
            np.ndarray[np.double_t] rev_rxn_rates,
            np.ndarray[np.double_t] pres_mod,
            np.ndarray[np.double_t] sp_rates):
    cdef size_type num
    try:
        num = fwd_rxn_rates.shape[1]
    except:
        num = 1
    cdef np.ndarray[np.double_t] temp_fwd = fwd_rxn_rates.T
    cdef np.ndarray[np.double_t] temp_rev = rev_rxn_rates.T
    cdef np.ndarray[np.double_t] temp_pres = pres_mod.T
    cdef np.ndarray[np.double_t] temp_sp = sp_rates.T
    cu_eval_spec_rates(num, &temp_fwd[0], &temp_rev[0], &temp_pres[0], &temp_sp[0])
    sp_rates = temp_sp.T

def py_get_rxn_pres_mod(np.ndarray[np.double_t] T,
            np.ndarray[np.double_t] pres,
            np.ndarray[np.double_t] C,
            np.ndarray[np.double_t] pres_mod):
    cdef size_type num
    try:
        num = C.shape[1]
    except:
        num = 1
    cdef np.ndarray[np.double_t] temp_t = T.T
    cdef np.ndarray[np.double_t] temp_p = pres.T 
    cdef np.ndarray[np.double_t] temp_c = C.T
    cdef np.ndarray[np.double_t] temp_pres = pres_mod.T
    cu_get_rxn_pres_mod(num, &temp_t[0], &temp_p[0], &temp_c[0], &temp_pres[0])
    pres_mod = temp_pres.T

def py_eval_conc(np.ndarray[np.double_t] T,
            np.ndarray[np.double_t] pres,
            np.ndarray[np.double_t] mass_frac,
            np.ndarray[np.double_t] mw_avg,
            np.ndarray[np.double_t] rho,
            np.ndarray[np.double_t] conc):
    cdef size_type num
    try:
        num = T.shape[0]
    except:
        num = 1
    cdef np.ndarray[np.double_t] temp_mass = mass_frac.T
    cdef np.ndarray[np.double_t] temp_conc = conc.T
    cdef np.ndarray[np.double_t] temp_t = T.T
    cdef np.ndarray[np.double_t] temp_p = pres.T
    cdef np.ndarray[np.double_t] temp_mw = mw_avg.T
    cdef np.ndarray[np.double_t] temp_rho = rho.T
    cu_eval_conc(num, &temp_t[0], &temp_p[0], &temp_mass[0], &temp_mw[0], &temp_rho[0], &temp_conc[0])
    conc = temp_conc.T
    mw_avg = temp_mw.T
    temp_rho = rho.T