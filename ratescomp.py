#! /usr/bin/env python2.7
#ratescomp.py
import pickle
import math

def run_comp(new, baseline):
    with open(new) as file:
        new_lines = [line.strip() for line in file.readlines()]
    with open(baseline) as file:
        baseline_lines = [line.strip() for line in file.readlines()]
    #first make sure they're the same length
    assert len(new_lines) == len(baseline_lines), "Input does not match, are you using the same mechanism?"

    #load the old order

    with open('out/optimized.pickle', 'rb') as file:
        dummy = pickle.load(file)
        specs = pickle.load(file)
        dummy = pickle.load(file)
        dummy = pickle.load(file)
        dummy = pickle.load(file)
        spec_ordering = pickle.load(file)
        num_specs = len(specs)

    state = None
    error_dict = {}
    method = None
    start = 0
    for i in range(len(new_lines)):
        if "k" in new_lines[i].lower() and "atm" in new_lines[i].lower():
            state = new_lines[i]
            assert new_lines[i] == baseline_lines[i], "State strings do not match!"
            error_dict[state] = {}
        else:
            try:
                n_val = float(new_lines[i])
                b_val = float(baseline_lines[i])
                if b_val == 0:
                    err = abs(n_val - b_val)
                else:
                    err = 100.0 * abs(n_val - b_val) / b_val
                if err > error_dict[state][method][0]:
                    error_dict[state][method] = (err, i - start)
            except Exception, e:
                method = new_lines[i]
                error_dict[state][method] = (0.0, -1)
                assert new_lines[i] == baseline_lines[i], "Method strings do not match! {} != {}".format(new_lines[i], baseline_lines[i])
                start = i
    for state in error_dict:
        print state
        for method in error_dict[state]:
            err = error_dict[state][method][0]
            ind = error_dict[state][method][1]
            if "Jacob" in method:
                #find out if the ind is a multiple of the # of species
                j_index = ind % num_specs
                i_index = int(math.floor(ind / num_specs))
                i_index = 0 if i_index == 0 else spec_ordering.index(i_index - 1) + 1
                j_index = 0 if j_index == 0 else spec_ordering.index(j_index - 1) + 1
                print method, "{}%".format(err), "index: {}".format(ind), "new index: {}".format(i_index * num_specs + j_index)
            else:
                print method, "{}%".format(err), "index: {}".format(ind)

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