#! /usr/bin/env python2.7

import sys
from performance_extractor import get_data
from general_plotting import legend_key
from general_plotting import plot_cpu as plot
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

data = get_data()
data = [x for x in data if x.lang in ['c', 'tchem']]

fig, ax = plt.subplots()
miny = None

linestyle = ''
plot_std = False
FD_marker = '>'
pj_marker = 'o'
tc_marker = 's'

#FD
plotdata = [x for x in data if x.finite_difference]
miny = plot(plotdata, FD_marker, 'Finite Difference', miny)

#pyjac
plotdata = [x for x in data if not x.finite_difference
                and x.lang == 'c'
                and not x.cache_opt]
miny = plot(plotdata, pj_marker, 'pyJac', miny)

#tchem

plotdata = [x for x in data if not x.finite_difference
                and x.lang == 'tchem']
miny = plot(plotdata, tc_marker, 'TChem', miny)

ax.set_yscale('log')
ax.set_ylim(ymin=miny*0.95)
ax.legend(loc=0, numpoints=1)
# add some text for labels, title and axes ticks
ax.set_ylabel('Mean evaluation time / condition (ms)')
#ax.set_title('GPU Jacobian Evaluation Performance for {} mechanism'.format(thedir))
ax.set_xlabel('Number of species')
#ax.legend(loc=0)
plt.savefig('cpu_norm.pdf')
plt.close() 