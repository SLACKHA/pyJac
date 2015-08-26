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
			if '_co_' in filename:
				basename += ' (cache-opt)'
			for line in lines:
				vals = [float(f) for f in line.split(',')]
				name = basename + ' {:d} thread(s)'.format(int(vals[0]))
				if not name in data[directory]:
					data[directory][name] = []
				data[directory][name].append(vals[2] / vals[1])
		else:
			name = 'GPU'
			if '_co_' in filename:
				name += '(cache-opt'
			if '_sm_' in filename:
				if not '_co_' in filename:
					name += '('
				else:
					name += ', '
				name += 'shared-mem)'
			elif '_co_' in filename:
				name += ')'
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

gpu_data = {}
for thedir in step_data:
	gpu_data[thedir] = {}
	for name in step_data[thedir]:
		thelist = []
		for point in step_data[thedir][name]:
			avg = np.mean(step_data[thedir][name][point])
			dev = np.std(step_data[thedir][name][point])
			thelist.append((point, avg, dev))
		gpu_data[thedir][name] = zip(*thelist)

def barplot(data):
	N = len(data)

	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig, ax = plt.subplots()
	offset = 0
	rects = []
	name_count = None
	name_list = [thedir for thedir in data]
	legend_list = [thename for thename in data[thedir]]
	count = 0
	color_wheel = ['r', 'b']
	for thedir in data:
		if name_count is None:
			name_count = len(data[thedir])
		name_count = 0
		for name in data[thedir]:
			means, devs = data[thedir][name]
			rects.extend(ax.bar(ind[count] + offset, means, width, color=color_wheel[name_count], yerr=devs, label=name))
			offset += width
			name_count += 1
		count += 1

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Mean evaulation time per condition (ms / condition)')
	ax.set_title('CPU Jacobian Evaluation Performance')
	x_loc = [ind[0] + (name_count / 2) * width]
	for i in range(1, name_count + 1):
		x_loc.append(ind[i] + (name_count / 2) * width + 2 * i * width)
	ax.set_xticks(x_loc)
	ax.set_xticklabels( name_list )

	ax.legend(rects[:len(legend_list)], legend_list, loc=0)
	#autolabel(rects)
	plt.savefig('cpu.pdf')
	plt.close()

def line_plot(data):
	for thedir in data:
		fig, ax = plt.subplots()
		for name in data[thedir]:
			x, y, z = data[thedir][name]
			order, x = zip(*sorted(enumerate(x), key=lambda k:k[1]))
			order = list(order)
			y = np.array(y)[order]
			z = np.array(z)[order]
			plt.errorbar(x, y, yerr=z, label=name)

		# add some text for labels, title and axes ticks
		ax.set_ylabel('Mean evaluation time (ms)')
		ax.set_title('GPU Jacobian Evaluation Performance for {} mechanism'.format(thedir))
		ax.set_xlabel('# of conditions')
		ax.legend(loc=0)
		plt.savefig('gpu_{}.pdf'.format(thedir))
		plt.close()

barplot(data)
line_plot(gpu_data)	