import phi1234
from runphi4 import calcPhi4
import os, sys
import numpy as np
import multiprocessing as mp
from functools import partial

def main():

    Emax= 22.0
    L   =  6.0
    m   = 1 

    a   = phi1234.Phi1234()
    fstr= "data/Emax=" + str(Emax) + "_L=" + str(L) + ".npz"
    
    print('Reading', fstr)
    #check if file already exists
    if not os.path.exists(fstr):
        print('{} not found. Generating ...'.format(fstr))
        a.buildFullBasis(k=1, Emax=Emax, L=L, m=m)
        a.buildFullBasis(k=-1, Emax=Emax, L=L, m=m)
        a.buildMatrix()
        a.saveMatrix(fstr)

    a.loadMatrix(fstr)

    fpartial = partial(f, Emax=Emax, L=L, m=m, a=a)
    with mp.Pool(16) as p:
        garr = np.linspace(0, 5, 26)
        results = p.map(fpartial, garr)
        
    print(results)

def f(g4, Emax, L, m, a):

    E0, _, _ = calcPhi4(a, g4, Emax, neigs=3, g2=0, L=L, m=m,
                    ren=True, printout=True)

    return( [ 
        a.vacuumE(ren="raw"), a.spectrum(k=1, ren="raw"), a.spectrum(k=-1, ren="raw"),
        a.vacuumE(ren="renlocal"), a.spectrum(k=1, ren="renlocal"), a.spectrum(k=-1, ren="renlocal"),
        a.vacuumE(ren="rensubl"), a.spectrum(k=1, ren="rensubl"), a.spectrum(k=-1, ren="rensubl")
        ] )

if __name__ == '__main__':
    main()
