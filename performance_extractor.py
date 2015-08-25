#! /usr/bin/env python2.7

import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
d=os.path.join('.', 'performance')
dirs = [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

#ok, now we have a list of data directories, open each and look for the text files
data = {}
step_data = {}
for directory in dirs:
	data[directory] = {}
	step_data[directory] = {}

	thedir = os.path.join(d,directory)

	#get text files
	files = [os.path.join(thedir, o) for o in os.listdir(thedir) if os.path.isfile(os.path.join(thedir,o)) and o.endswith('.txt')]
	for filename in files:
		with open(filename) as file:
			lines = [l.strip() for l in file.readlines() if l.strip()]
		if 'cpu' in filename:
			#format is num_threads, num_odes, runtime (ms)
			basename = 'CPU'
			if '_co_output' in filename:
				basename += ' (cache-opt)'
			for line in lines:
				vals = [float(f) for f in line.split(',')]
				name = basename + ' {:d} thread(s)'.format(int(vals[0]))
				if not name in data[directory]:
					data[directory][name] = []
				data[directory][name].append(vals[2] / vals[1])
		else:
			basename = 'GPU'
			if '_co_' in filename:
				basename += '(cache-opt, '
			if '_sm_' in filename:
				if not '_co_' in filename:
					basename += '('
				basename == 'shared-mem)'
			if '_steps' in filename:
				#format is num_odes, runtime (ms)
				for line in lines:
					vals = [float(f) for f in line.split(',')]
					if not name in step_data[directory]:
						step_data[directory][name] = {}
					if not vals[0] in step_data[directory][name]:
						step_data[directory][name][vals[0]] = []
					step_data[directory][name][vals[0]].append(vals[1])
			
for thedir in data:
	for name in data[thedir]:
		avg = np.mean(data[thedir][name])
		dev = np.std(data[thedir][name])
		data[thedir][name] = (avg, dev)

for thedir in step_data:
	for name in step_data[thedir]:
		avg = np.mean(data[thedir][name])
		dev = np.std(data[thedir][name])
		data[thedir][name] = (avg, dev)

print data
print step_data

