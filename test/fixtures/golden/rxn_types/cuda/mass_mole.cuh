#ifndef MASS_MOLE_HEAD
#define MASS_MOLE_HEAD

#include "header.cuh"

void mole2mass (const double*, double*);
void mass2mole (const double*, double*);
double getDensity (const double, const double, const double*);

#endif
