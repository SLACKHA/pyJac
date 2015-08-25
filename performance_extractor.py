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
	data[thedir][name] = zip(*data[thedir][name])

gpu_data = {}
for thedir in step_data:
	gpu_data[thedir] = {}
	for name in step_data[thedir]:
		gpu_data[thedir][name] = []
		for point in step_data[thedir][name]:
			avg = np.mean(step_data[thedir][name][point])
			dev = np.std(step_data[thedir][name][point])
			gpu_data[thedir][name].append(point, avg, dev)
	gpu_data[thedir][name] = zip(*gpu_data[thedir][name])

print data
print gpu_data

barplot(data)
line_plot(gpu_data)

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
	for thedir in data:
		if name_count is None:
			name_count = len(data[thedir])
		for name in data[thedir]:
			means, devs = data[thedir][name]
			rects.append(ax.bar(ind + offset, means, width, yerr=devs, label=name))
			offset += width

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Mean evaulation time per condition (ms / condition)')
	ax.set_title('CPU Jacobian Evaluation Performance')
	ax.set_xticks(ind+(name_count / 2) * width)
	ax.set_xticklabels( name_list )

	ax.legend( reacs[:len(legend_list)], legend_list )
	for rect in rects:
		autolabel(rects)

	def autolabel(rects):
	    # attach some text labels
	    for rect in rects:
	        height = rect.get_height()
	        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
	                ha='center', va='bottom')
	plt.save('cpu.png')
	plt.close()

def line_plot(data):
	for thedir in data:
		fig, ax = plt.subplots()
		legend_list = [thename for thename in data[thedir]]
		for name in data[thedir]:
			x, y, z = data[thedir][name]
			plt.plot(x, y, yerr=z, label=name)

		# add some text for labels, title and axes ticks
		ax.set_ylabel('Mean evaluation time (ms)')
		ax.set_title('GPU Jacobian Evaluation Performance for {} mechanism'.format(thedir))
		ax.set_xlabel('# of conditions')
		ax.legend(legend_list, loc=0)
		plt.save('gpu_{}.png'.format(thedir))
		plt.close()	