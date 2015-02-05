"""Writes mechanism header and output testing files"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import sys
import math

# Local imports
import chem_utilities as chem
import utils

def write_mechanism_initializers(path, lang, specs, reacs):
    if lang in ['matlab', 'fortran']:
        raise NotImplementedError

    #the mechanism header defines a number of useful preprocessor defines, as well as defining method stubs for setting initial conditions
    with open(path + 'mechanism.h', 'w') as file:

        file.write('#ifndef MECHANISM_H\n'
                   '#define MECHANISM_H\n\n')

        #convience: write species indexes
        file.write('/* Species Indexes\n')
        file.write('\n'.join('{}  {}'.format(i, spec.name) for i, spec in enumerate(specs)))
        file.write('*/\n\n')

        #defines
        file.write("//Number of species\n")
        file.write('#define NSP {}\n'.format(len(specs)))
        file.write("//Number of variables. NN = NSP + 1 (temperature)\n")
        file.write('#define NN {}\n'.format(len(specs) + 1))
        file.write('//Number of forward reactions\n')
        file.write('#define FWD_RATES {}\n'.format(len(reacs)))
        file.write('//Number of reversible reactions\n')
        file.write('#define REV_RATES {}\n'.format(len([reac for reac in reacs if reac.rev])))
        file.write('//Number of reactions with pressure modified rates\n')
        file.write('#define PRES_MOD_RATES {}\n\n'.format(len([reac for reac in reacs if reac.pdep or reac.thd])))

        #to work with both c and cuda
        file.write('#ifdef __cplusplus\n'
                   'extern "C" {\n'
                   '#endif\n')

        file.write('    //Must be implemented by user on a per mechanism basis in mechanism.c\n'
                   '    #ifdef CONP\n'
                   '    void set_same_initial_conditions(int NUM, double* y_host, double* pres_host);\n'
                   '    #elif CONV\n'
                   '    void set_same_initial_conditions(int NUM, double* y_host, double* pres_host, double* rho_host);\n'
                   '    #endif\n'
                   '    #ifdef RATES_TEST\n'
                   '    void write_jacobian_and_rates_output();\n'
                   '    #endif\n'
                   )

        #close previous extern
        file.write('#ifdef __cplusplus\n'
                   '}\n'
                   '#endif\n\n')

        file.write('#endif\n\n')

    #now the mechanism file
    with open(path + 'mechanism.c', 'w') as file:
        file.write('#include <stdio.h>\n'
                   '#include "mass_mole.h"\n'
                   '#include "mechanism.h"\n'
                   '#ifdef RATES_TEST\n'
                   '    #include "rates.h"\n'
                   '    #include "derivs.h"\n'
                   '#endif\n')
        file.write('#ifdef CONP\n'
                   '    void set_same_initial_conditions(int NUM, double* y_host, double* pres_host) {\n'
                   '#elif CONV\n'
                   '    void set_same_initial_conditions(int NUM, double* y_host, double* pres_host, double* rho_host) {\n'
                   '#endif\n'
                   '        double Xi [NSP] = {0.0};\n'
                   '        //set initial mole fractions here\n\n'
                   '        //Normalize mole fractions to sum to one\n'
                   '        double Xsum = 0.0;\n'
                   '        for (int j = 0; j < NSP; ++ j) {\n'
                   '            Xsum += Xi[j];\n'
                   '        }\n'
                   '        if (Xsum == 0.0) {\n'
                   '            printf("Use of the set initial conditions function requires user implementation!");\n'
                   '            exit(-1);\n'
                   '        }\n'
                   '        for (int j = 0; j < NSP; ++ j) {\n'
                   '            Xi[j] /= Xsum;\n'
                   '        }\n\n'
                   '        //convert to mass fractions\n'
                   '        double Yi[NSP];\n'
                   '        mole2mass(Xi, Yi);\n\n'
                   '        //set initial pressure, units [dyn/cm^2]\n'
                   '        double pres = 1.01325e6;\n'
                   '        // set intial temperature, units [K]\n'
                   '        double T0 = 1600;\n\n'
                   '        //load temperature and mass fractions for all threads (cells)\n'
                   '        for (int i = 0; i < NUM; ++i) {\n'
                   '            y_host[i] = T0;\n'
                   '            //loop through species\n'
                   '            for (int j = 1; j < NN; ++j) {\n'
                   '                y_host[i + NUM * j] = Yi[j - 1];\n'
                   '            }\n'
                   '        }\n\n'
                   '        #ifdef CONV\n'
                   '        //calculate density\n'
                   '        double rho = getDensity(T0, pres, Xi);\n'
                   '        #endif\n\n'
                   '        for (int i = 0; i < NUM; ++i) {\n'
                   '            #ifdef CONV\n'
                   '            rho_host[i] = rho;\n'
                   '            #endif\n'
                   '            pres_host[i] = pres;\n'
                   '        }\n'
                   '    }\n\n'
                   )
        file.write('#ifdef RATES_TEST\n')
        #write utility function that finds concentrations at a given state
        file.write('    void get_concentrations(double pres, const double* y, double* conc) {\n'
                   '        double rho;\n'
                   )

        line = '        rho = '
        isfirst = True
        for sp in specs:
            if len(line) > 70:
                line += '\n'
                file.write(line)
                line = '         '
            
            if not isfirst: line += ' + '
            line += '(y[{}] / {})'.format(specs.index(sp) + 1, sp.mw)
            isfirst = False

        line += ';\n'
        file.write(line)
        line = '        rho = pres / ({:.8e} * y[0] * rho);\n\n'.format(chem.RU)
        file.write(line)


        # calculation of species molar concentrations
        file.write('        // species molar concentrations\n')
        # loop through species
        for sp in specs:
            isp = specs.index(sp)
            line = '        conc[{}] = rho * y[{}] / '.format(isp, isp)
            line += '{}'.format(sp.mw) + utils.line_end[lang]
            file.write(line)
        
        file.write('    }\n\n')


        #convience method to write rates to file
        file.write('    void write_rates(FILE* fp, const double* fwd_rates, const double* rev_rates, const double* pres_mod, const double* dy) {\n'
                   '        for(int i = 0; i < FWD_RATES; ++i) {\n'
                   '            fprintf(fp, "%.15le\\n", fwd_rates[i]);\n'
                   '        }\n'
                   '        for(int i = 0; i < REV_RATES; ++i) {\n'
                   '            fprintf(fp, "%.15le\\n", rev_rates[i]);\n'
                   '        }\n'
                   '        for(int i = 0; i < PRES_MOD_RATES; i++) {\n'
                   '            fprintf(fp, "%.15le\\n", pres_mod[i]);\n'
                   '        }\n'
                   '        for(int i = 0; i < NSP; i++) {\n'
                   '            fprintf(fp, "%.15le\\n", dy[i]);\n'
                   '        }\n'
                   '    }\n\n'
                  )

        file.write('    void write_jacob(FILE* fp, const double* jacob) {\n'
                   '        for (int i = 0; i < NN * NN; ++i) {\n'
                   '            fprintf(fp, "%.15le\\n", jacob[i]);\n'
                   '        }\n'
                   '    }\n\n'
                   )


        file.write('    void write_jacobian_and_rates_output() {\n'
                   '        //set mass fractions to unity to turn on all reactions\n'
                   '        double Yi[NN];\n'
                   '        for (int i = 1; i < NN; ++i) {\n'
                   '            Yi[i] = 1.0 / ((double)NSP);\n'
                   '        }\n'
                   '        double conc[NSP];\n'
                   '        double fwd_rates[FWD_RATES];\n'
                   '        double rev_rates[REV_RATES];\n'
                   '        double pres_mod[PRES_MOD_RATES];\n'
                   '        double dy[NSP];\n'
                   '        double jacob[NN * NN];\n'
                   '        //evaluate and write rates for various conditions\n'
                   '        FILE* fp = fopen ("rates_data.txt", "w");\n'
                   '        fprintf(fp, "800K, 1 atm\\n");\n'
                   '        Yi[0] = 800;\n'
                   '        get_concentrations(1.01325e6, Yi, conc);\n'
                   '        eval_rxn_rates(800, conc, fwd_rates, rev_rates);\n'
                   '        get_rxn_pres_mod (800, 1.01325e6, conc, pres_mod);\n'
                   '        eval_spec_rates (fwd_rates, rev_rates, pres_mod, dy);\n'
                   '        write_rates(fp, fwd_rates, rev_rates, pres_mod, dy);\n'
                   '        eval_jacob(0, 1.01325e6, Yi, jacob);\n'
                   '        write_jacob(fp, jacob);\n'
                   '        fprintf(fp, "1600K, 1 atm\\n");\n'
                   '        Yi[0] = 1600;\n'
                   '        get_concentrations(1.01325e6, Yi, conc);\n'
                   '        eval_rxn_rates(1600, conc, fwd_rates, rev_rates);\n'
                   '        get_rxn_pres_mod (1600, 1.01325e6, conc, pres_mod);\n'
                   '        eval_spec_rates (fwd_rates, rev_rates, pres_mod, dy);\n'
                   '        write_rates(fp, fwd_rates, rev_rates, pres_mod, dy);\n'
                   '        eval_jacob(0, 1.01325e6, Yi, jacob);\n'
                   '        write_jacob(fp, jacob);\n'
                   '        fprintf(fp, "800K, 10 atm\\n");\n'
                   '        Yi[0] = 800;\n'
                   '        get_concentrations(1.01325e7, Yi, conc);\n'
                   '        eval_rxn_rates(800, conc, fwd_rates, rev_rates);\n'
                   '        get_rxn_pres_mod (800, 1.01325e7, conc, pres_mod);\n'
                   '        eval_spec_rates (fwd_rates, rev_rates, pres_mod, dy);\n'
                   '        write_rates(fp, fwd_rates, rev_rates, pres_mod, dy);\n'
                   '        eval_jacob(0, 1.01325e7, Yi, jacob);\n'
                   '        write_jacob(fp, jacob);\n'
                   '        fclose(fp);\n'
                   '    }\n'
                   )
        file.write('#endif')




