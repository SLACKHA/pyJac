import os
os.path.append('../')
from performance_extractor import get_data
from general_plotting import legend_key
import numpy as np

data, mechanism_sizes = get_data('../')
data = [x for x in data if x.lang in ['c', 'tchem']]

fig, ax = plt.subplots()
miny = None

linestyle = ''
plot_std = False
FD_marker = '>'
pj_marker = 'o'
tc_marker = 's'

def plot(plotdata, marker, name):
    plotdata = sorted(plotdata, key=lambda x: x.num_reactions)
    thex = [x.num_reactions for x in plotdata]
    they = [x.y for x in plotdata]
    they = [float(x.y) / float(x.x) for x in plotdata]
    thez = np.std(they)
    miny = they[0] if miny is None else they[0] if they[0] < miny else miny
    if plot_std:
        plt.errorbar(thex, they, yerr=z, marker=marker, label=name)
    else:
        plt.plot(thex, they, marker=marker, label=name)

#FD
plotdata = [x for x in data if x.finite_difference]
plot(plotdata, FD_marker, 'Finite Difference')

#pyjac
plotdata = [x for x in data if not x.finite_difference
                and x.lang == 'c'
                and not x.cache_opt]
plot(plotdata, pj_marker, 'pyJac')

#tchem

plotdata = [x for x in data if not x.finite_difference
                and x.lang == 'tchem']
plot(plotdata, tc_marker, 'TChem')

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