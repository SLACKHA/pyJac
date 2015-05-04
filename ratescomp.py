#! /usr/bin/env python2.7
#ratescomp.py

def run_comp(new, baseline):
	with open(new) as file:
		new_lines = [line.strip() for line in file.readlines()]
	with open(baseline) as file:
		baseline_lines = [line.strip() for line in file.readlines()]
	#first make sure they're the same length
	assert len(new_lines) == len(baseline_lines), "Input does not match, are you using the same mechanism?"

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
			print method, "{}%".format(error_dict[state][method][0]), "index: {}".format(error_dict[state][method][1])

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