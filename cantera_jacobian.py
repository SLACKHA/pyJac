import cantera as ct
import numpy as np
from argparse import ArgumentParser

def get_str(arr):
    return '\n'.join('{:.15e}'.format(x) for x in arr) + '\n'

def get_y_dot(gas):
    rates = gas.net_production_rates
    dT = np.sum(-gas.partial_molar_enthalpies * rates / (gas.density * gas.cp_mass))
    rates = gas.molecular_weights * rates / gas.density
    vals = np.zeros(gas.n_species + 1)
    vals[0] = dT
    vals[1:] = rates
    return vals

def eval_jacobian(gas, T, P, conp):
    ATOL = 1e-15
    RTOL = 1e-9

    FD_ORD = 6
    x_coeffs = np.zeros(FD_ORD)
    #6th order central difference
    x_coeffs[0] = -3.0
    x_coeffs[1] = -2.0
    x_coeffs[2] = -1.0
    x_coeffs[3] = 1.0
    x_coeffs[4] = 2.0
    x_coeffs[5] = 3.0
    
    y_coeffs = np.zeros(FD_ORD)
    y_coeffs[0] = -1.0 / 60.0
    y_coeffs[1] = 3.0 / 20.0
    y_coeffs[2] = -3.0 / 4.0
    y_coeffs[3] = 3.0 / 4.0
    y_coeffs[4] = -3.0 / 20.0
    y_coeffs[5] = 1.0 / 60.0

    Y = np.zeros(gas.n_species + 1)
    Y[0] = gas.T
    Y[1:] = gas.Y[:]

    ewt = ATOL + (RTOL * np.abs(Y))

    dY = get_y_dot(gas)

    srur = np.sqrt(np.finfo(float).eps)
    the_sum = np.sum(np.power(dY * ewt, 2))
    fac = np.sqrt(the_sum / (float)(gas.n_species + 1))
    r0 = 1000.0 * RTOL * np.finfo(float).eps * float(gas.n_species + 1) * fac
    jac = np.zeros((gas.n_species + 1, gas.n_species + 1))

    for j in range(gas.n_species + 1):
        yj_orig = Y[j]
        r = np.max(srur * np.abs(yj_orig), r0 / ewt[j])

        jac[:, j] = 0.0

        for k in range(FD_ORD):
            Y[j] = yj_orig + x_coeffs[k] * r
            gas.set_unnormalized_mass_fractions(Y[1:])
            gas.TP = Y[0], P
            dy = get_y_dot(gas)
            jac[:, j] += y_coeffs[k] * dy

        jac[:, j] /= r
        Y[j] = yj_orig

    return jac

def jacob_gen(mechanism, conp=True):
    gas = ct.Solution(mechanism)

    T_list = [800, 1600]
    atm_list = [1, 10, 50]
    P_list = [v * ct.one_atm for v in atm_list]
    Y = np.arange(1, gas.n_species + 1)

    with open('cantera_jacobian.txt', 'w') as file:
        for T in T_list:
            for P in P_list:
                gas.TPY = T, P, Y
                file.write('{}K, {} atm\n'.format(T, atm_list[P_list.index(P)]))
                file.write('Net Rates of Progess\n')
                file.write(get_str(gas.net_rates_of_progress))
                file.write('Spec Rates\n')
                file.write(get_str(gas.net_production_rates))
                file.write('dy\n')
                dy = get_y_dot(gas)
                file.write(get_str(dy))
                file.write('Jacob\n')
                jac = eval_jacobian(gas, T, P, conp)
                for i in range(gas.n_species + 1):
                    file.write(get_str(jac[:, i]))



if __name__ == '__main__':
    parser = ArgumentParser(description='Generates baseline data for checking jacobian output')
    parser.add_argument('-m', '--mech',
        type=str,
        required=True,
        help='The cantera formatted mechanism')

    args = parser.parse_args()
    jacob_gen(args.mech)