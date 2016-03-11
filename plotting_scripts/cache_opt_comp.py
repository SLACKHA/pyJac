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

def plotco(lang, smem_default=False, norm=True):
    desc = 'gpu' if lang == 'cuda' else 'cpu'
    if lang == 'c':
        smem_default=False

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
                    and x.smem==smem_default]

    plotdata2 = [x for x in data if not x.finite_difference
                    and x.lang == lang
                    and x.cache_opt
                    and x.smem==smem_default]

    if norm:
        xvals = sorted(list(set([p.num_reacs for p in plotdata1] + [p.num_reacs for p in plotdata2])))
        def get_y(plotdata, x):
            return np.mean(next(pd for pd in plotdata if pd.num_reacs == x).y)
        ymax = [max(get_y(plotdata1, x), get_y(plotdata2, x)) for x in xvals]
        for i, x in enumerate(xvals):
            for pd in [pdtest for pdtest in plotdata1 + plotdata2 if pdtest.num_reacs == x]:
                pd.y = np.array(pd.y / ymax[i])

    def fake_norm(x):
        return np.array(x.y)

    thenorm = None if not norm else fake_norm
    miny = plot(plotdata1, nco_marker, 'Non Cache-Optimized', miny, norm=thenorm)
    miny = plot(plotdata2, nco_marker, 'Cache-Optimized', miny, norm=thenorm)

    if not norm:
        ax.set_yscale('log')
    ax.set_ylim(ymin=miny*0.95)
    ax.legend(loc=0, numpoints=1)
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Mean evaluation time / condition (ms)')
    #ax.set_title('GPU Jacobian Evaluation Performance for {} mechanism'.format(thedir))
    ax.set_xlabel('Number of reactions')
    #ax.legend(loc=0)
    plt.savefig('cache_opt_{}.pdf'.format(desc))
    plt.close() 

plotco('c')
plotco('cuda')