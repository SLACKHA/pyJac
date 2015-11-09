/* wrapper to interface with TChem */

#include "header.h"
#include "TC_interface.h"

void tc_eval_jacob (const char* mechfile, const char* thermofile, const int num,
                    const double t, const double* pres, const double* y,
                    double* jac) {

    /* Initialize TC library */
    int withtab = 0;
    TC_initChem (mechfile, thermofile, withtab, 1.0);

    for (int tid = 0; tid < num; ++tid) {
        TC_setThermoPres (pres[tid]) ;
        TC_getJacTYNm1anl ( &y[tid * NN], NSP, &jac[tid*NSP*NSP] );
    }

    return;
}
