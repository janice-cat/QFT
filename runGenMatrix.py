#! /usr/bin/env python3
import os

# g = 0, L = 6
for Emax in [25, 30,
	     35, 40, 45]:

	print("Run Emax={}".format(Emax))
	os.system("python genMatrix.py 6 {}".format(Emax+2))
	os.system("python phi4eigs.py Emax={}.0_L=6.0.npz 0 {}".format(Emax+2, Emax))