/* wrapper to translate to cuda arrays */

#ifndef CU_PYJAC_HEAD
#define CU_PYJAC_HEAD

void run(int, int, const double*, const double*,
			double*, double*, double*,
			double*, double*, double*, double*);
int init(int);
void cleanup();

#endif