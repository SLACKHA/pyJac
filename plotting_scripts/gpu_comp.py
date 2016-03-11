#! /usr/bin/env python2.7

from fullscale_comp import fullscale_comp
import numpy as np
text_loc = [(3, -55, 0.4), #fd
			(3, -55, -0.25), #pyjac
			]

fit_params = fullscale_comp('cuda', text_loc=text_loc, loc_override=2)

for x in fit_params:
	print x[0] + '\tR^2:{}'.format(x[1])

pJ = next(val for val in fit_params if 'pyJac' in val[0])
fD = next(val for val in fit_params if 'Finite' in val[0])

#c1 x^n1 = c2 x^n2
#x^(n1 - n2) = c2 / c1
#x = (c2 / c1) ^ (1. / (n1 - n2))

print('The x intercept of pyJac with FD is:  {}'.format(
	np.power(pJ[2] / fD[2], 1.0 / (fD[3] - pJ[3]))))
