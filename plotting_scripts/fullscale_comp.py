#! /usr/bin/env python2.7

import sys
from performance_extractor import get_data
from general_plotting import legend_key
from general_plotting import plot
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

FD_marker = '>'
pj_marker = 'o'
tc_marker = 's'
def nice_names(x):
	if x.finite_difference:
		name = 'Finite Difference'
		marker = FD_marker
	elif x.lang == 'tchem':
		name = 'TChem'
		marker=tc_marker
	else:
		name = 'pyJac'
		marker = pj_marker
	if x.lang == 'cuda':
		name += ' (gpu)'
	return name, marker

def get_fullscale(data):
	mechanisms = set([x.mechanism for x in data])
	#need to filter out runs on subsets
	for mech in mechanisms:
		max_x = max(x.x for x in [y for y in data if y.mechanism == mech])
		data = [x for x in data if x.mechanism != mech or 
					(x.x == max_x)]
	return data

def fullscale_comp(lang, plot_std=True, homedir=None,
						cache_opt_default=False,
						smem_default=True):
	if lang == 'c':
		langs = ['c', 'tchem']
		desc = 'cpu'
		smem_default=False
	elif lang == 'cuda':
		langs = ['cuda']
		desc = 'gpu'
	else:
		raise Exception('unknown lang {}'.format(lang))

	def thefilter(x):
		return x.cache_opt==cache_opt_default and x.smem == smem_default
	
	data = get_data(homedir)
	data = [x for x in data if x.lang in langs]
	data = filter(thefilter, data)
	data = get_fullscale(data)
	if not len(data):
		print 'no data found... exiting'
		sys.exit(-1)

	fig, ax = plt.subplots()
	miny = None

	linestyle = ''

	#FD
	plotdata = [x for x in data if x.finite_difference]
	if plotdata:
		miny = plot(plotdata, FD_marker, 'Finite Difference', miny)

	for lang in langs:
		plotdata = [x for x in data if not x.finite_difference
		                and x.lang == lang]
		if plotdata:
			name, marker = nice_names(plotdata[0])
			miny = plot(plotdata, marker, name, miny)

	ax.set_yscale('log')
	ax.set_ylim(ymin=miny*0.95)
	ax.legend(loc=0, numpoints=1)
	# add some text for labels, title and axes ticks
	ax.set_ylabel('Mean evaluation time / condition (ms)')
	#ax.set_title('GPU Jacobian Evaluation Performance for {} mechanism'.format(thedir))
	ax.set_xlabel('Number of reactions')
	#ax.legend(loc=0)
	plt.savefig('{}_norm.pdf'.format(desc))
	plt.close() 