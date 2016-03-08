import cantera as ct
import numpy as np
mech = '../mechs/Sarathy_ic5_mech_rev.cti'
gas = ct.Solution(mech)
import subprocess

condition = 190
arr = [1027, 15, 16, 17, 18, 19, 20, 21, 1558, 1559, 1560, 1561, 2074, 545, 546, 547, 2092, 1073, 1074, 74, 1612, 1619, 85, 86, 1625, 95, 1632, 100, 2032, 1638, 1644, 1141, 1142, 1143, 121, 1557, 129, 363, 22, 23, 654, 24, 1180, 1187, 1188, 1189, 1204, 1212, 193, 1226, 1227, 2083, 1750, 1751, 1752, 1753, 1754, 234, 748, 2002, 2003, 756, 249, 260, 780, 781, 1807, 1298, 1815, 1823, 1318, 1319, 831, 832, 851, 1374, 362, 1387, 364, 1901, 1902, 1903, 1904, 1393, 1394, 1402, 391, 403, 920, 414, 1441, 930, 2033, 428, 1466, 1469, 1475, 2000, 2001, 466, 467, 2004, 2005, 2006, 2011, 2018, 2019, 2020, 1517, 495, 496, 1009, 2034, 2035, 1015, 1021]
arr = sorted(arr)
for i in arr:
	if i < 2060:
		continue
	with open('dummy', 'w') as outfile:
		args = ['/usr/local/bin/python2.7', 'test.py',
					'-m', mech, '-p', 'data/ic5h11oh/pasr_output_0.npy',
					'-l', 'c', '-orxn', str(i),
					'-cn', str(condition)]
		try:
			subprocess.check_call(args, stdout=outfile, stderr=outfile)
		except Exception, e:
			print ' '.join(args)
			raise e

	err = np.load('error_arrays.npz')
	err = np.max(err['err_jac_thr'])
	print i, err
