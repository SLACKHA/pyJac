#! /usr/bin/env python2.7

import numpy as np
cimport numpy as np

ctypedef np.int_t size_type

cdef extern from "cu_pyjacob.cu":
    void cu_dydt(const int num, const double pres, const double* y, double* dy)
    void cu_eval_jacob (const int num, const double t, const double pres, const double* y, double* jac)
    void cu_eval_rxn_rates (const int num, const double T, const double pres, const double* C, double* fwd_rxn_rates, double* rev_rxn_rates)
    void cu_eval_spec_rates (const int num, const double * fwd_rates, const double * rev_rates, const double * pres_mod, double * sp_rates)
    void cu_get_rxn_pres_mod (const int num, const double T, const double pres, const double * C, double * pres_mod)
    void cu_eval_conc (const double * T, const double * pres, const double * mass_frac, double * mw_avg, double * rho, double * conc)


def py_dydt(np.double_t t,
            np.double_t pres,
            np.ndarray[np.double_t] y,
            np.ndarray[np.double_t] dy):
    def size_type num
    try:
        num = y.shape[1]
    except:
        num = 1
    temp_y = y.T
    temp_dy = dy.T
    cu_dydt(num, t, pres, &temp_y[0], &temp_dy[0])
    dy = temp_dy.T

def py_eval_jacobian(np.double_t t,
            np.double_t pres,
            np.ndarray[np.double_t] y,
            np.ndarray[np.double_t] jac):
    def size_type num
    try:
        num = y.shape[1]
    except:
        num = 1
    temp_y = y.T
    temp_jac = dy.T
    cu_eval_jacob(num, t, pres, &temp_y[0], &temp_jac[0])
    jac = temp_jac.T

def py_eval_rxn_rates(np.double_t T,
            np.double_t pres,
            np.ndarray[np.double_t] C,
            np.ndarray[np.double_t] fwd_rxn_rates,
            np.ndarray[np.double_t] rev_rxn_rates):
    def size_type num
    try:
        num = C.shape[1]
    except:
        num = 1
    temp_c = C.T
    temp_fwd = fwd_rxn_rates.T
    temp_rev = rev_rxn_rates.T
    cu_eval_rxn_rates(num, T, pres, &temp_c[0], &temp_fwd[0], &temp_rev[0])
    fwd_rxn_rates = temp_fwd.T
    rev_rxn_rates = temp_rev.T

def py_eval_spec_rates(np.ndarray[np.double_t] fwd_rxn_rates,
            np.ndarray[np.double_t] rev_rxn_rates,
            np.ndarray[np.double_t] pres_mod,
            np.ndarray[np.double_t] sp_rates):
    def size_type num
    try:
        num = fwd_rxn_rates.shape[1]
    except:
        num = 1
    temp_fwd = fwd_rxn_rates.T
    temp_rev = rev_rxn_rates.T
    temp_pres = pres_mod.T
    temp_sp = sp_rates.T
    cu_eval_spec_rates(num, &temp_fwd[0], &temp_rev[0], &temp_pres[0], &temp_sp[0])
    sp_rates = temp_sp.T

def py_get_rxn_pres_mod(np.double_t T,
            np.double_t pres,
            np.ndarray[np.double_t] C,
            np.ndarray[np.double_t] pres_mod):
    def size_type num
    try:
        num = C.shape[1]
    except:
        num = 1
    temp_c = C.T
    temp_pres = pres_mod.T
    cu_get_rxn_pres_mod(nym, T, pres, &temp_c[0], &temp_pres[0])
    pres_mod = temp_pres.T

def py_eval_conc(np.double_t T,
            np.double_t pres,
            np.ndarray[np.double_t] mass_frac,
            np.double_t mw_avg,
            np.double_t rho,
            np.ndarray[np.double_t] conc):
    def size_type num
    num = 1
    temp_mass = mass_frac.T
    temp_conc = conc.T
    cu_eval_conc(num, &T, &pres, &temp_mass[0], &mw_avg, &rho, &temp_conc[0])
    conc = temp_conc.T

def py_eval_conc(np.ndarray[np.double_t] T,
            np.ndarray[np.double_t] pres,
            np.ndarray[np.double_t] mass_frac,
            np.ndarray[np.double_t] mw_avg,
            np.ndarray[np.double_t] rho,
            np.ndarray[np.double_t] conc):
        def size_type num
    try:
        num = T.shape[0]
    except:
        num = 1
    temp_mass = mass_frac.T
    temp_conc = conc.T
    temp_t = T.T
    temp_p = pres.T
    temp_mw = mw_avg.T
    temp_rho = rho.T
    cu_eval_conc(num, &temp_t[0], &temp_p[0], &temp_mass[0], &temp_mw[0], &temp_rho[0], &temp_conc[0])
    conc = temp_conc.T
    mw_avg = temp_mw.T
    temp_rho = rho.T