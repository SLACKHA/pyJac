/* wrapper to interface with TChem */

#include "header.h"
#include "TC_interface.h"

void tc_eval_jacob (char* mechfile, char* thermofile, const int num,
                    const double* pres, double* y, double* conc,
                    double* fwd_rates, double* rev_rates, double* spec_rates,
                    double* dydt, double* jac) {

    /* Initialize TC library */
    int withtab = 0;
    TC_initChem (mechfile, thermofile, withtab, 1.0);

    for (int tid = 0; tid < num; ++tid) {
        // set pressure
        TC_setThermoPres (pres[tid]);

        // get concentration
        TC_getMs2Cc (&y[tid * NN], NN, &conc[tid*NSP]);

        // get reaction rates of progress
        double rxn_rates[2 * FWD_RATES];
        TC_getRfrb (&y[tid * NN], NN, rxn_rates);
        for (int i = 0; i < FWD_RATES; ++i) {
            fwd_rates[tid*FWD_RATES + i] = rxn_rates[i];
            rev_rates[tid*FWD_RATES + i] = rxn_rates[FWD_RATES + i];
        }

        // get species production rates
        TC_getTY2RRml (&y[tid * NN], NN, &spec_rates[tid*NSP]);

        // get derivative
        TC_getSrc (&y[tid * NN], NN, &dydt[tid*NN]);

        // get reduced Jacobian matrix
        TC_getJacTYNm1anl (&y[tid * NN], NSP, &jac[tid*NSP*NSP]);
    }

    TC_reset();

    return;
}
