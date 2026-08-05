
#include <vector>
#include "adept.h"
#include "header.h"
#include "ad_dydt.h"
void eval_jacob(const double t, const double p, const double* y,
                         double* jac) {
    using adept::adouble; // Import Stack and adouble from adept
    adept::Stack stack; // Where the derivative information is stored
    std::vector<adouble> in(NSP); // Vector of active input variables
    adept::set_values(&in[0], NSP, y); // Initialize adouble inputs
    adouble pres = p;
    stack.new_recording(); // Start recording
    std::vector<adouble> out(NSP); // Create vector of active output variables
    dydt(t, pres, &in[0], &out[0]); // Run algorithm
    stack.independent(&in[0], NSP); // Identify independent variables
    stack.dependent(&out[0], NSP); // Identify dependent variables
    stack.jacobian(jac); // Compute & store Jacobian in jac
}
            