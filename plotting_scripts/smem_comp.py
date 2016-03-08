#! /usr/bin/env python2.7

import sys
from performance_extractor import get_data
from general_plotting import legend_key
from general_plotting import plot
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from fullscale_comp import get_fullscale

data = get_data()
data = get_fullscale(data)

def plotsm(lang):
	desc = 'gpu' if lang == 'cuda' else 'cpu'

	fig, ax = plt.subplots()
	miny = None

	linestyle = ''
	plot_std = False
	nco_marker = 'o'
	co_marker = 's'

	#pyjac
	plotdata1 = [x for x in data if not x.finite_difference
	                and x.lang == lang
	                and not x.cache_opt
	                and x.smem]
	miny = plot(plotdata1, nco_marker, 'Shared Memory Caching', miny)

	plotdata2 = [x for x in data if not x.finite_difference
	                and x.lang == lang
	                and not x.cache_opt
	                and not x.smem]
	miny = plot(plotdata2, co_marker, 'No Shared Memory Caching', miny)

	ax.set_yscale('log')
	ax.set_ylim(ymin=miny*0.95)
	ax.legend(loc=0, numpoints=1)
	# add some text for labels, title and axes ticks
	ax.set_ylabel('Mean evaluation time / condition (ms)')
	#ax.set_title('GPU Jacobian Evaluation Performance for {} mechanism'.format(thedir))
	ax.set_xlabel('Number of reactions')
	#ax.legend(loc=0)
	plt.savefig('smem_comp_{}.pdf'.format(desc))
	plt.close() 

plotsm('cuda')