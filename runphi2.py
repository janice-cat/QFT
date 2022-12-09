######################################################
#
# Fock space Hamiltonian truncation for phi^2 theory in 2 dimensions
# Authors: Slava Rychkov (slava.rychkov@lpt.ens.fr) and Lorenzo Vitale (lorenzo.vitale@epfl.ch)
# December 2014
#
# Compute energy spectrum with phi^2 coupling as a test for truncation method.
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
textwidth = 6.47699
slide_width = 11.5
half_slide_width = 5.67
aspect_ratio = 5/7
pres_params = {'axes.edgecolor': 'black',
                  'axes.facecolor':'white',
                  'axes.grid': False,
                  'axes.linewidth': 0.5,
                  'backend': 'ps',
                  'savefig.format': 'pdf',
                  'axes.titlesize': 24,
                  'axes.labelsize': 20,
                  'legend.fontsize': 20,
                  'xtick.labelsize': 18,
                  'ytick.labelsize': 18,
                  'text.usetex': True,
                  'figure.figsize': [half_slide_width, half_slide_width * aspect_ratio],
                  'font.family': 'sans-serif',
                  #'mathtext.fontset': 'cm',
                  'xtick.bottom':True,
                  'xtick.top': False,
                  'xtick.direction': 'out',
                  'xtick.major.pad': 3,
                  'xtick.major.size': 3,
                  'xtick.minor.bottom': False,
                  'xtick.major.width': 0.2,

                  'ytick.left':True,
                  'ytick.right':False,
                  'ytick.direction':'out',
                  'ytick.major.pad': 3,
                  'ytick.major.size': 3,
                  'ytick.major.width': 0.2,
                  'ytick.minor.right':False,
                  'lines.linewidth':2}

params = {'axes.edgecolor': 'black',
                  'axes.facecolor':'white',
                  'axes.grid': False,
                  'axes.linewidth': 0.5,
                  'backend': 'ps',
                  'savefig.format': 'ps',
                  'axes.titlesize': 11,
                  'axes.labelsize': 9,
                  'legend.fontsize': 9,
                  'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'text.usetex': True,
                  'figure.figsize': [7, 5],
                  'font.family': 'sans-serif',
                  #'mathtext.fontset': 'cm',
                  'xtick.bottom':True,
                  'xtick.top': False,
                  'xtick.direction': 'out',
                  'xtick.major.pad': 3,
                  'xtick.major.size': 3,
                  'xtick.minor.bottom': False,
                  'xtick.major.width': 0.2,

                  'ytick.left':True,
                  'ytick.right':False,
                  'ytick.direction':'out',
                  'ytick.major.pad': 3,
                  'ytick.major.size': 3,
                  'ytick.major.width': 0.2,
                  'ytick.minor.right':False,
                  'lines.linewidth':2}
plt.rcParams.update(pres_params)

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

    # check if this spectrum computation has already been done
    if os.path.exists(f'data/Emax={Emax}_L={L}_g2={g2}_g4={g4}_spectrum.pkl'):
        with open(f'data/Emax={Emax}_L={L}_g2={g2}_g4={g4}_spectrum.pkl', 'rb') as handle:
            data_dict = pickle.load(handle)
            return data_dict['E0'], data_dict['K1spectrum'], data_dict['K-1spectrum']

    #check if file already exists
    if not os.path.exists(fstr):
        a.buildFullBasis(k=1, Emax=Emax, L=L, m=m)
        a.buildFullBasis(k=-1, Emax=Emax, L=L, m=m)
        a.buildMatrix()
        a.saveMatrix(fstr)

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

def casimir_energy(L, m=1):
    """ Calculate the Casimir energy of the free scalar field"""
    def E(x):
        denom = np.sqrt(m**2 * L**2 + x**2)
        return (-1/(np.pi * L)) * x**2 /(denom * (np.exp(denom) - 1))

    result = scipy.integrate.quad(E, 0.0, np.inf)
    return result[0]

def phisquared_e0(g2, L, m=1):
    """ Calculate the energy spectrum of phi^2 theory exactly. """
    E0 = casimir_energy(L, m)
    musquared = m**2 + 2*g2
    lam = (musquared * (1 - np.log(musquared/m**2)) - m**2)/(8 * np.pi)
    return E0 + lam*L

def plot_figure2(Emax=12, L=10, m=1):
    """ Plot vacuum energy vs g for phi^2 theory.
    TODO: Code numerical integration to get E(L) and compare to exact result.
    """
    garr = np.linspace(-0.4, 0.8, 24)
    Emaxs = [5, 7, 12, 15]
    e0_arr = np.zeros((len(Emaxs), len(garr)))
    e0_exact = np.zeros((len(Emaxs), len(garr)))

    for i, Emax in enumerate(Emaxs):
        for j, g2 in enumerate(garr):
            print(f'Emax={Emax}, g2={g2}')
            E0, _, _ = run(0.0, Emax, neigs=3, g2=g2, L=L, m=m)
            e0_arr[i, j] = E0
            e0_exact[i, j] = phisquared_e0(g2, L, m)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    #plot ground state energy vs g2, compare exact and truncated
    ax.plot(garr, e0_exact[2, :], 'k-', label='Exact')
    ax.plot(garr, e0_arr[2, :], 'kx', markersize=5, label='Truncated')
    ax.set_xlabel('$g_2$')
    ax.set_ylabel('$E_0$')
    ax.set_title(f"m={m}, L={L}, Emax = {Emaxs[2]}")
    ax.legend()

    #plot difference between exact and truncated as a function of g2 for varying Emax
    for k in range(len(Emaxs)):
        ax2.plot(garr, e0_arr[k, :] - e0_exact[k, :], label=r'$E_{max}$' + f'= {Emaxs[k]}')
    ax2.set_xlabel('$g_2$')
    ax2.set_ylabel('$E_0$ - exact')
    ax2.set_title(f"m={m}, L={L}")
    ax2.legend()
    fig.tight_layout()
    plt.savefig('plots/reproduce_fig2_raw_phi2.pdf')
    #permission error?
    #subprocess.Popen(['plots/reproduce_fig2_raw_phi2.pdf'], shell=True)

if __name__ == "__main__":
    #a, b, c = run(0.0, 10.0, neigs=3, g2=0.0, L=2*np.pi, m=1, printout=True, save=True)
    plot_figure2()
    #print(casimir_energy(10.0))
