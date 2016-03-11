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
from scipy import optimize

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
        name += ' (GPU)'
    return name, marker

def get_fullscale(data):
    mechanisms = set([x.mechanism for x in data])
    #need to filter out runs on subsets
    for mech in mechanisms:
        max_x = max(x.x for x in [y for y in data if y.mechanism == mech])
        data = [x for x in data if x.mechanism != mech or 
                    (x.x == max_x)]
    return data

def fit_order(plotdata, y, std, order, color='k', text_loc=None, fontsize=10):
    x = sorted([x.num_specs for x in plotdata])
    maxx = int(x[-1])

    def fit_func(p, x):
        n = order
        c = p[0]
        if order is None:
            n = p[0]
            c = p[1]
        return n * np.log10(x) + np.log10(c)

    def errfunc(p, x, y):
        return (fit_func(p, x) - np.log10(y)) # Distance to the target function

    if order is None:
        p0 = [1, 1] # Initial guess for the parameters
        p1, success = optimize.leastsq(errfunc, p0[:], args=(x, y))
        order = p1[0]
        c = p1[1]
    else:
        p0 = [1] # Initial guess for the parameters
        p1, success = optimize.leastsq(errfunc, p0[:], args=(x, y))
        c = p1[0]

    fi = c * x ** order
    ss_res = np.sum(np.abs(y - fi)**2)
    ss_tot = np.sum(np.abs(y - np.mean(y))**2)
    Rsq = 1 - ss_res / ss_tot

    name = nice_names(plotdata[0])[0]

    if text_loc is not None:
        point, xoffset, yoffset = text_loc
        xbase = x[point]
        ybase = y[point]
        xbase += xoffset
        ybase += yoffset
        if order != 1:
            label = r'{:.1e}$N_S^{{{:.2}}}$'.format(c, order)
        elif order == 1:
            label = r'{:.1e}$N_S$'.format(c, order)

        plt.text(xbase, ybase, label, fontsize=fontsize)
    else:
        if order != 1:
            label = r'{:.1e}$N_S^{{{:.2}}}$'.format(c, order)
        elif order == 1:
            label = r'{:.1e}$N_S$'.format(c, order)

    plt.plot(range(maxx + 1), c * np.array(range(maxx + 1))**order,
                'k-', color=color)

    return name, Rsq, c, order

def fullscale_comp(lang, plot_std=True, homedir=None,
                        cache_opt_default=False,
                        smem_default=False,
                        loc_override=None,
                        text_loc=None, fontsize=10,
                        color_list=['b', 'g', 'r', 'k']):
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
    
    fit_vals = []
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

    fitvals = []
    retdata = []
    color_ind = 0
    text_ind = None
    if text_loc:
        text_ind = 0

    #FD
    plotdata = [x for x in data if x.finite_difference]
    if plotdata:
        color = color_list[color_ind]
        color_ind += 1
        miny, they, thez = plot(plotdata, FD_marker, 'Finite Difference', miny, return_y=True, color=color)
        theloc = None
        if text_ind is not None:
            theloc = text_loc[text_ind]
            text_ind += 1
        fitvals.append(
            fit_order(plotdata, they, thez, None, color, text_loc=theloc, fontsize=fontsize)
            )
        retdata.append(they)

    for lang in langs:
        plotdata = [x for x in data if not x.finite_difference
                        and x.lang == lang]
        color = color_list[color_ind]
        color_ind += 1
        if plotdata:
            name, marker = nice_names(plotdata[0])
            miny, they, thez = plot(plotdata, marker, name, miny, return_y=True, color=color)
            theloc = None
            if text_ind is not None:
                theloc = text_loc[text_ind]
                text_ind += 1
            fitvals.append(
                fit_order(plotdata, they, thez, None, color, text_loc=theloc, fontsize=fontsize)
            )
            retdata.append(they)

    ax.set_yscale('log')
    ax.set_ylim(ymin=miny*0.95)
    loc = 0 if loc_override is None else loc_override
    ax.legend(loc=loc, numpoints=1, fontsize=10)
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Mean evaluation time / condition (ms)')
    #ax.set_title('GPU Jacobian Evaluation Performance for {} mechanism'.format(thedir))
    ax.set_xlabel('Number of species')
    #ax.legend(loc=0)
    plt.savefig('{}_norm.pdf'.format(desc))
    plt.close()

    return fitvals, retdata