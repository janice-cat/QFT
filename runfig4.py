######################################################
#
# Fock space Hamiltonian truncation for phi^4 theory in 2 dimensions
# Authors: Slava Rychkov (slava.rychkov@lpt.ens.fr) and Lorenzo Vitale (lorenzo.vitale@epfl.ch)
# December 2014
#
######################################################

import phi1234
import renorm
import sys
import scipy
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import utility as u


def main():
    Emax= 22.0
    L   = 6.0
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

    plot_figure4(a, Emax=Emax, L=L, m=m)

def calcPhi4(a, g4, Emax, neigs=3, g2=0.0, L=2*np.pi, m=1, printout=False):
    """ Build basis and compute energy spectrum.

    Returns
    -------
    e0 : ground state energy
    k1 : spectrum for Kparity == 1
    k2 : spectrum for Kparity == -1
    """
    sigma = -30. #hard coded parameter

    a.buildBasis(k=1, Emax=Emax)
    a.buildBasis(k=-1, Emax=Emax)
    if printout:
        print('K=1 full basis size = ', a.fullBasis[1].size)
        print('K=-1 full basis size = ', a.fullBasis[-1].size)
        print('K=1 basis size = ', a.basis[1].size)
        print('K=-1 basis size = ', a.basis[-1].size)

        print("Computing raw eigenvalues for g4 = ", g4)
    a.setcouplings(g4=g4, g2=g2)
    a.computeHamiltonian(k=1, ren=False)
    a.computeHamiltonian(k=-1, ren=False)

    a.computeEigval(k=1, sigma=sigma, n=neigs, ren=False)
    a.computeEigval(k=-1, sigma=sigma, n=neigs, ren=False)

    if printout:
        print("Raw vacuum energy: ", a.vacuumE(ren="raw"))
        print("K=1 Raw spectrum: ", a.spectrum(k=1, ren="raw"))
        print("K=-1 Raw spectrum: ", a.spectrum(k=-1, ren="raw"))

    return a.vacuumE(ren="raw"), a.spectrum(k=1, ren="raw"), a.spectrum(k=-1, ren="raw")


def plot_figure4(a, Emax=20, L=10, m=1):
    """ Plot vacuum energy vs g for phi^2 theory.
    TODO: Code numerical integration to get E(L) and compare to exact result.
    """
    u.plotStyle()
    garr = np.linspace(0, 5, 25)
    e0_arr = []
    for g4 in garr:
        print('g4 = {:.3f} ...'.format(g4), end='\r')
        E0, _, _ = calcPhi4(a, g4, Emax, neigs=3, g2=0, L=L, m=m)
        e0_arr.append(E0)
        print('g4 = {:.3f}, E0 = {:.3f}'.format(g4, E0))
    print(e0_arr)
    fig, ax = plt.subplots()
    ax.plot(garr, e0_arr, 'o-', markersize=3)
    ax.set_xlabel('$g_4$')
    ax.set_ylabel('$E_0$')
    ax.set_title(f"m={m}, L={L}, Emax = {Emax}")
    fig.tight_layout()
    plt.savefig('plots/reproduce_fig4_raw_phi4.pdf')


if __name__ == '__main__':
    main()

