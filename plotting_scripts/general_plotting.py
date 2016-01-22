legend_key = {'H2':r'H$_2$/CO',
              'GRI':'GRI',
              'USC':'USC'}

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
def plot_cpu(plotdata, marker, name, miny, plot_std=True):
    plotdata = sorted(plotdata, key=lambda x: x.num_reacs)
    thex = [x.num_reacs for x in plotdata]
    they = [x.y for x in plotdata]
    they = [np.array(x.y) / x.x for x in plotdata]
    thez = np.std(they)
    they = [np.mean(y) for y in they]
    miny = they[0] if miny is None else they[0] if they[0] < miny else miny
    if plot_std:
        plt.errorbar(thex, they, yerr=thez, marker=marker, label=name)
    else:
        plt.plot(thex, they, marker=marker, label=name)
    return miny