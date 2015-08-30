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
		if 'nvcc' not in filename:
			#format is num_threads, num_odes, runtime (ms)
			if '_co_' in filename:
				continue
			if 'tchem' in filename:
				basename = 'TChem'
			else:
				basename = 'pyJac'
				if not '447' in filename:
					continue
				#vals = filename.split('_')
				#compiler = vals[0][vals[0].rindex('/') + 1:]
				#if compiler == 'icc':
				#	basename += '(' + vals[0][vals[0].rindex('/') + 1:] + ' - ' + vals[1][0:2] + '.' + vals[1][2] + '.' + vals[1][3] + ')' 
				#else:
			#		basename += '(' + vals[0][vals[0].rindex('/') + 1:] + ' - ' + vals[1][0] + '.' + vals[1][1] + '.' + vals[1][2:] + ')' 
			for line in lines:
				try:
					vals = [float(f) for f in line.split(',')]
				except:
					continue
				if not basename in data[directory]:
					data[directory][basename] = []
				data[directory][basename].append(vals[1] / vals[0])
		else:
			name = 'pyJac (GPU)'
			if not '_nco_' in filename and not '_nsm_' in filename:
				continue
			#format is num_odes, runtime (ms)
			for line in lines:
				try:
					vals = [float(f) for f in line.split(',')]
				except:
					continue
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
	print(thedir)
	for name in data[thedir]:
		if name == 'TChem': continue
		print(name)
		print(data[thedir]['TChem'][0] / data[thedir][name][0])

gpu_data = {}
norm_gpu_data = {}
for thedir in step_data:
	gpu_data[thedir] = {}
	norm_gpu_data[thedir] = {}
	for name in step_data[thedir]:
		thelist = []
		maxpoint = 0 
		for point in step_data[thedir][name]:
			avg = np.mean(step_data[thedir][name][point])
			dev = np.std(step_data[thedir][name][point])
			thelist.append((point, avg, dev))

			if point > maxpoint:
				vals = [x / point for x in step_data[thedir][name][point]]
				mavg =  np.mean(vals)
				mdev = np.std(vals)
		gpu_data[thedir][name] = zip(*thelist)
		norm_gpu_data[thedir][name] = (mavg, mdev)

mechanism_sizes = {'USC':{'ns':111, 'nr':784},
				   'H2':{'ns':13, 'nr':27},
				   'GRI':{'ns':53, 'nr':325}}

key = {'H2':r'H$_2$/CO',
	   'GRI':'GRI',
	   'USC':'USC'}

ls = ''
ms = 'o'

def barplot(data):
	N = len(data)

	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig, ax = plt.subplots()
	offset = 0
	rects = []
	name_count = None
	name_list = [key[thedir] for thedir in data]
	legend_list = [thename for thename in data[thedir]]
	count = 0
	color_wheel = ['r', 'b', 'g', 'k', 'y']
	for thedir in data:
		if name_count is None:
			name_count = len(data[thedir])
		name_count = 0
		for name in data[thedir]:
			means, devs = data[thedir][name]
			rects.extend(ax.bar(ind[count] + offset, means, width, color=color_wheel[name_count]
				, yerr=devs, label=name))
			offset += width
			name_count += 1
		count += 1

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Mean evaulation time per condition (ms)')
	ax.set_title('Jacobian Evaluation Performance')
	x_loc = [ind[0] + (name_count / 2) * width]
	for i in range(1, N):
		x_loc.append(ind[i] + (name_count / 2) * width + 2 * i * width)
	ax.set_xticks(x_loc)
	ax.set_xticklabels( name_list )
	ax.set_yscale('log')

	ax.legend(rects[:len(legend_list)], legend_list, loc=0)
	#autolabel(rects)
	plt.savefig('cpu.pdf')
	plt.close()

	fig, ax = plt.subplots()
	miny = None
	for name in legend_list:
		thex = []
		they = []
		thez = []
		for thedir in data:
			y, z = data[thedir][name]
			thex.append(mechanism_sizes[thedir]['nr'])
			they.append(y)
			thez.append(z)
		miny = they[0] if miny is None else they[0] if they[0] < miny else miny
		thex, they, thez = zip(*sorted(zip(thex, they, thez), key=lambda x:x[0]))
		plt.errorbar(thex, they, thez, linestyle=ls, marker=ms, label=name)

	#draw a slope line
	x = np.arange(thex[-1] + 1)
	y = np.arange(thex[-1] + 1)
	y = y / float(y[-1])
	plt.plot(x, y, '--k')
	props = {'width':0.1,
			 'frac':0.05,
			 'shrink':0.01,
			 'color':'k'}
	ann = (x[len(x) / 4.], y[len(y) / 4.])
	plt.annotate(r'$y\/\alpha\/N_r$', xy=ann, xytext=(ann[0]-150, ann[1]+0.125), arrowprops=props,
		fontsize=16)

	ax.set_yscale('log')
	ax.set_ylim(ymin=miny)
	ax.legend(loc=0)
	# add some text for labels, title and axes ticks
	ax.set_ylabel('Mean evaluation time / condition (ms)')
	#ax.set_title('GPU Jacobian Evaluation Performance for {} mechanism'.format(thedir))
	ax.set_xlabel('Number of reactions in mechanism')
	#ax.legend(loc=0)
	plt.savefig('cpu_norm.pdf')
	plt.close()	


def line_plot(data):
	fig, ax = plt.subplots()
	for thedir in data:
		for name in data[thedir]:
			x, y, z = data[thedir][name]
			order, x = zip(*sorted(enumerate(x), key=lambda k:k[1]))
			order = list(order)
			y = np.array(y)[order]
			z = np.array(z)[order]
			plt.errorbar(x, y, yerr=z, label=key[thedir], linestyle=ls, marker=ms)

	ax.set_yscale('log')
	# add some text for labels, title and axes ticks
	ax.set_ylabel('Mean evaluation time (ms)')
	#ax.set_title('GPU Jacobian Evaluation Performance for {} mechanism'.format(thedir))
	ax.set_xlabel('Number of conditions')
	ax.legend(loc=0)
	plt.savefig('gpu.pdf')
	plt.close()

	thex = []
	they = []
	thez = []
	fig, ax = plt.subplots()
	for thedir in norm_gpu_data:
		for name in norm_gpu_data[thedir]:
			y, z = norm_gpu_data[thedir][name]
			thex.append(mechanism_sizes[thedir]['nr'])
			they.append(y)
			thez.append(z)

	thex, they, thez = zip(*sorted(zip(thex, they, thez), key=lambda x:x[0]))
	plt.errorbar(thex, they, thez, linestyle=ls, marker=ms)

	#draw a slope line
	x = np.arange(thex[-1] + 1)
	y = np.arange(thex[-1] + 1)
	y = 0.1 * y / float(y[-1])
	plt.plot(x, y, '--k')
	props = {'width':0.1,
			 'frac':0.05,
			 'shrink':0.01,
			 'color':'k'}
	ann = (x[len(x) / 4.], y[len(y) / 4.])
	plt.annotate(r'$y\/\alpha\/0.1\times\/N_r$', xy=ann, xytext=(ann[0]-150, ann[1]+0.075), arrowprops=props,
				fontsize=16)

	ax.set_yscale('log')
	ax.set_ylim(ymin=they[0])
	# add some text for labels, title and axes ticks
	ax.set_ylabel('Mean evaluation time / condition (ms / condition)')
	#ax.set_title('GPU Jacobian Evaluation Performance for {} mechanism'.format(thedir))
	ax.set_xlabel('Number of reactions in mechanism')
	#ax.legend(loc=0)
	plt.savefig('gpu_norm.pdf')
	plt.close()	

barplot(data)
line_plot(gpu_data)	