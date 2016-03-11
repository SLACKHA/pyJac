legend_key = {'H2':r'H$_2$/CO',
              'CH4':r'GRI-Mech 3.0',
              'C2H4':r'USC-Mech II',
              'IC5H11OH':r'IC$_5$H$_{11}$OH'}

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
def plot(plotdata, marker, name, miny, plot_std=True, norm=None, return_y=False, reacs_as_x=True,
          color=None, hollow=False):
    if reacs_as_x:
      plotdata = sorted(plotdata, key=lambda x: x.num_reacs)
      thex = [x.num_reacs for x in plotdata]
    else:
      plotdata = sorted(plotdata, key=lambda x: x.num_specs)
      thex = [x.num_specs for x in plotdata]
    if norm is None:
      they = [np.array(x.y) / x.x for x in plotdata]
    else:
      they = [norm(x) for x in plotdata]

    thez = [np.std(y) for y in they]
    they = [np.mean(y) for y in they]
    argdict = { 'x':thex,
          'y':they,
          'linestyle':'',
          'marker':marker,
          'label':name}
    if color:
      argdict['markeredgecolor'] = color
      argdict['color'] = color
    elif hollow:
      argdict['markerfacecolor'] = 'None'
      argdict['label'] += ' (smem)'
    if plot_std:
      argdict['yerr'] = thez
    if plot_std:
        line=plt.errorbar(**argdict)
    else:
        line=plt.plot(**argdict)

    miny = they[0] if miny is None else they[0] if they[0] < miny else miny
    retval = miny
    if return_y:
      retval = (retval, they, thez)
    return retval