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
import subprocess
import pickle

# Plotting parameters
# PRL Font preference: computer modern roman (cmr), medium weight (m), normal shape
cm_in_inch = 2.54
# column size is 8.6 cm
col_size = 8.6 / cm_in_inch
default_width = 1.0*col_size
aspect_ratio = 5/7
default_height = aspect_ratio*default_width
plot_params = {
    'backend': 'pdf',
    'savefig.format': 'pdf',
    'text.usetex': True,
    'font.size': 7,

    'figure.figsize': [default_width, default_height],
    'figure.facecolor': 'white',

    'axes.grid': False,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'white',

    'axes.titlesize': 8.0,
    'axes.titlepad' : 5,
    'axes.labelsize': 8,
    'legend.fontsize': 6.5,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'axes.linewidth': 0.75,

    'xtick.top': False,
    'xtick.bottom': True,
    'xtick.direction': 'out',
    'xtick.minor.size': 2,
    'xtick.minor.width': 0.5,
    'xtick.major.pad': 2,
    'xtick.major.size': 4,
    'xtick.major.width': 1,

    'ytick.left': True,
    'ytick.right': False,
    'ytick.direction': 'out',
    'ytick.minor.size': 2,
    'ytick.minor.width': 0.5,
    'ytick.major.pad': 2,
    'ytick.major.size': 4,
    'ytick.major.width': 1,

    'lines.linewidth': 1
}
plt.rcParams.update(plot_params)

def run(g4, Emax, neigs=3, g2=0.0, L=2*np.pi, m=1, printout=False, save=True):
    """ Build basis and compute energy spectrum.

    Returns
    -------
    e0 : ground state energy
    k1 : spectrum for Kparity == 1
    k2 : spectrum for Kparity == -1
    """
    sigma = -30. #hard coded parameter
    a = phi1234.Phi1234()
    fstr = "data/Emax=" + str(Emax) + "_L=" + str(L) + ".npz"

    #check if file already exists
    if not os.path.exists(fstr):
        a.buildFullBasis(k=1, Emax=Emax, L=L, m=m)
        a.buildFullBasis(k=-1, Emax=Emax, L=L, m=m)
        a.buildMatrix()
        a.saveMatrix(fstr)

    #check if this spectrum computation has already been done
    if os.path.exists(f'data/Emax={Emax}_L={L}_g2={g2}_g4={g4}_spectrum.pkl'):
        with open(f'data/Emax={Emax}_L={L}_g2={g2}_g4={g4}_spectrum.pkl', 'rb') as handle:
            data_dict = pickle.load(handle)
            return data_dict['E0'], data_dict['K1spectrum'], data_dict['K-1spectrum']

    a.loadMatrix(fstr)
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

    vacuumE = a.vacuumE(ren="raw")
    K1spectrum = a.spectrum(k=1, ren="raw")
    Km1spectrum = a.spectrum(k=-1, ren="raw")

    if save:
        data_dict = {}
        data_dict['E0'] = vacuumE
        data_dict['K1spectrum'] = K1spectrum
        data_dict['K-1spectrum'] = Km1spectrum

        with open(f'data/Emax={Emax}_L={L}_g2={g2}_g4={g4}_spectrum.pkl', 'wb') as f:
            pickle.dump(data_dict, f)

    if printout:
        print("Raw vacuum energy: ", vacuumE)
        print("K=1 Raw spectrum: ", K1spectrum)
        print("K=-1 Raw spectrum: ", Km1spectrum)

    return vacuumE, K1spectrum, Km1spectrum

def plot_figure2(Emax=12, L=10, m=1):
    """ Plot vacuum energy vs g for phi^2 theory.
    TODO: Code numerical integration to get E(L) and compare to exact result.
    """
    garr = np.linspace(-0.4, 0.8, 24)
    e0_arr = []
    for g2 in garr:
        print(g2)
        E0, _, _ = run(0.0, Emax, neigs=3, g2=g2, L=L, m=m)
        e0_arr.append(E0)
    print(e0_arr)
    fig, ax = plt.subplots()
    ax.plot(garr, e0_arr, 'o-', markersize=3)
    ax.set_xlabel('$g_2$')
    ax.set_ylabel('$E_0$')
    ax.set_title(f"m={m}, L={L}, Emax = {Emax}")
    fig.tight_layout()
    plt.savefig('plots/reproduce_fig2_raw_phi2.pdf')
    #permission error?
    #subprocess.Popen(['plots/reproduce_fig2_raw_phi2.pdf'], shell=True)

if __name__ == "__main__":
    a, b, c = run(0.0, 10.0, neigs=3, g2=0.0, L=2*np.pi, m=1, printout=False, save=True)
    #plot_figure2()
