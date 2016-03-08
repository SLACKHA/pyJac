from performance_extractor import get_data
from general_plotting import legend_key
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_scaling(plotdata, markerlist, miny, plot_std=True, hollow=False):
    mset = list(set(x.mechanism for x in plotdata))
    mechs = sorted(mset, key=lambda mech:next(x for x in plotdata if x.mechanism == mech).num_specs) 
    for i, mech in enumerate(mechs):
        name = legend_key[mech]
        data = [x for x in plotdata if x.mechanism == mech]
        thex = sorted(list(set(x.x for x in data)))
        they = [next(x.y for x in data if x.x == xval) for xval in thex]
        thez = [np.std(x) for x in they]
        they = [np.mean(x) for x in they]
        miny = they[0] if miny is None else they[0] if they[0] < miny else miny
        color = next(ax._get_lines.color_cycle)
        argdict = {	'x':thex,
        			'y':they,
        			'linestyle':'',
        			'marker':markerlist[i],
        			'markeredgecolor':color,
        			'color':color,
        			'label':name}
        if hollow:
        	argdict['markerfacecolor'] = 'None'
        	argdict['label'] += ' (smem)'
        if plot_std:
        	argdict['yerr'] = thez
        if plot_std:
            line=plt.errorbar(**argdict)
        else:
            line=plt.plot(**argdict)
    return miny

legend_markers = ['o', 'v', 's', '>']

data = get_data()
plotdata = [x for x in data if x.lang == 'cuda'
		and not x.cache_opt
		and not x.smem
		and not x.finite_difference]

fig, ax = plt.subplots()
miny = None

miny = plot_scaling(plotdata, legend_markers, miny)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(ymin=miny*0.95)
ax.legend(loc=0, numpoints=1)
# add some text for labels, title and axes ticks
ax.set_ylabel('Mean evaluation time')
#ax.set_title('GPU Jacobian Evaluation Performance for {} mechanism'.format(thedir))
ax.set_xlabel('Number of conditions')
#ax.legend(loc=0)
plt.savefig('gpu_scaling.pdf')
plt.close()

data = get_data()
plotdata = [x for x in data if x.lang == 'cuda'
		and not x.cache_opt
		and not x.smem
		and not x.finite_difference]

fig, ax = plt.subplots()
miny = None

miny = plot_scaling(plotdata, legend_markers, miny)
plt.gca().set_color_cycle(None)

plotdata = [x for x in data if x.lang == 'cuda'
		and not x.cache_opt
		and x.smem
		and not x.finite_difference]

miny = plot_scaling(plotdata, legend_markers, miny, hollow=True)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(ymin=miny*0.95)
ax.legend(loc=0, numpoints=1)
# add some text for labels, title and axes ticks
ax.set_ylabel('Mean evaluation time')
#ax.set_title('GPU Jacobian Evaluation Performance for {} mechanism'.format(thedir))
ax.set_xlabel('Number of conditions')
#ax.legend(loc=0)
plt.savefig('gpu_scaling_smem.pdf')
plt.close()

