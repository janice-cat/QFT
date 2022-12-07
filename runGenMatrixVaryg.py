#! /usr/bin/env python3
import os
from matplotlib import pyplot as plt

# g = 0, L = 6
gArr = [0, 0.001, 0.01, 
	  0.1, 0.2, 0.3, 0.4, 0.5,
	  0.6, 0.7, 0.8, 0.9, 1]

for g in gArr:
	Emax = 20
	print("Run Emax={}, g={} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>".format(Emax, g))
	# os.system("python genMatrix.py 6 {}".format(Emax+2))
	os.system("python phi4eigs.py Emax={}.0_L=6.0.npz {} {}".format(Emax+2, g, Emax))
