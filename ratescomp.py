#! /usr/bin/env python2.7
#ratescomp.py
import pickle
import numpy as np

def parse_file(file):
    value_dict = {}
    lines = [line.strip() for line in file.readlines()]
    for line in lines:
        if "k" in line.lower() and "atm" in line.lower():
            state = line
            value_dict[line] = {}
        else:
            try:
                val = float(line)
                value_dict[state][method].append(val)
            except:
                method = line
                value_dict[state][method] = []
    return value_dict

def run_comp(new, baseline):
    with open(new) as file:
        new_vals = parse_file(file)
    with open(baseline) as file:
        baseline_vals = parse_file(file)

    #first make sure we have all the same states
    for state in baseline_vals:
        assert state in new_vals, "State {} missing from new file".format(state)

    #next make sure all the methods in the baseline vals are in the new vals
    for state in baseline_vals:
        for method in baseline_vals[state]:
            assert method in new_vals[state], "Method {} missing from new file".format(method)

    error_dict = {}
    zerror_dict = {}
    for state in baseline_vals:
        print
        print state
        for method in baseline_vals[state]:
            assert len(baseline_vals[state][method]) == len(new_vals[state][method]), "Different number of values for State {} and Method {}, different mechanisms?".format(state, method)
            error = (-1, 0)
            zerror = (-1, 0)
            
            for i in range(len(baseline_vals[state][method])):
                if np.abs(baseline_vals[state][method][i]) < 1e-10:
                    zero_err = np.abs(baseline_vals[state][method][i] - new_vals[state][method][i])
                    if zero_err > zerror[1]:
                        zerror = (i, zero_err)
                else:
                    err = 100.0 * np.abs((baseline_vals[state][method][i] - new_vals[state][method][i])
                                                / baseline_vals[state][method][i])
                    if err > error[1]:
                        error = (i, err)
            print method
            print "{}% @ index {}".format(error[1], error[0]), "\t\t{} @ index {}".format(zerror[1], zerror[0])



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Compares rates between various versions of the create_jacobian framework')
    parser.add_argument('-n', '--new-file',
                        dest='new',
                        type=str,
                        required=True,
                        help='The new rates_data file')
    parser.add_argument('-b', '--baseline-file',
                        type=str,
                        dest = 'base',
                        required=True,
                        help='The baseline rates_data file')

    args = parser.parse_args()
    run_comp(args.new, args.base)