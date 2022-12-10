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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import utility as u
import pickle


def main():
    global doMinimal, doRen
    doMinimal = int(sys.argv[1])
    doRen     = int(sys.argv[2])
    Emax= 20.0 if not doMinimal else 10.0
    L   = 10.0 if not doMinimal else  6.283185307179586
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

    plot_figure4a(a, Emax=Emax, L=L, m=m, ren=doRen)
    plot_figure4b(a, Emax=Emax, L=L, m=m, ren=doRen)

def calcPhi4(a, g4, Emax, neigs=3, g2=0.0, L=2*np.pi, m=1, printout=False,
             save=True,
             ren=False  # False: only run 'raw'
                        # True : 'raw' + 'ren'
             ):
    """ Build basis and compute energy spectrum.

    Returns
    -------
    e0 : ground state energy
    k1 : spectrum for Kparity == 1
    k2 : spectrum for Kparity == -1
    """
    sigma = -g4*1.9-1e-2 #hard coded parameter


    # check if this spectrum computation has already been done
    if os.path.exists(f'data/Emax={Emax}_L={L}_g2={g2}_g4={g4}_spectrum.pkl'):
        with open(f'data/Emax={Emax}_L={L}_g2={g2}_g4={g4}_spectrum.pkl', 'rb') as handle:
            data_dict = pickle.load(handle)
            if not ren: 
                return data_dict['E0'], data_dict['K1spectrum'], data_dict['K-1spectrum']
            else:
                return [ data_dict['E0_renlocal'], data_dict['E0_rensubl'] ], \
                       [ data_dict['K1spectrum_renlocal'], data_dict['K1spectrum_rensubl'] ], \
                       [ data_dict['K-1spectrum_renlocal'], data_dict['K-1spectrum_rensubl'] ]

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

    if printout:
        print("Raw vacuum energy: ", vacuumE)
        print("K=1 Raw spectrum: ", K1spectrum)
        print("K=-1 Raw spectrum: ", Km1spectrum)

    if not ren: 
        if save:
            data_dict = {}
            data_dict['E0'] = vacuumE
            data_dict['K1spectrum'] = K1spectrum
            data_dict['K-1spectrum'] = Km1spectrum

            with open(f'data/Emax={Emax}_L={L}_g2={g2}_g4={g4}_spectrum.pkl', 'wb') as f:
                pickle.dump(data_dict, f)
        return vacuumE, K1spectrum, Km1spectrum
    else:    
        a.renlocal(Er=a.vacuumE(ren="raw"))
        if printout: print("Computing renormalized eigenvalues for g0r,g2r,g4r = ", a.g0r,a.g2r,a.g4r)
            
        a.computeHamiltonian(k=1, ren=True)
        a.computeHamiltonian(k=-1, ren=True)

        a.computeEigval(k=1, sigma=sigma, n=neigs, ren=True, corr=True, printout=printout)
        a.computeEigval(k=-1, sigma=sigma, n=neigs, ren=True, corr=True, printout=printout)

        vacuumE_renlocal = a.vacuumE(ren="renlocal")
        K1spectrum_renlocal = a.spectrum(k=1, ren="renlocal")
        Km1spectrum_renlocal = a.spectrum(k=-1, ren="renlocal")

        vacuumE_rensubl = a.vacuumE(ren="rensubl")
        K1spectrum_rensubl = a.spectrum(k=1, ren="rensubl")
        Km1spectrum_rensubl = a.spectrum(k=-1, ren="rensubl")

        if save:
            data_dict = {}
            data_dict['E0'] = vacuumE
            data_dict['K1spectrum'] = K1spectrum
            data_dict['K-1spectrum'] = Km1spectrum
            data_dict['E0_renlocal'] = vacuumE_renlocal
            data_dict['K1spectrum_renlocal'] = K1spectrum_renlocal
            data_dict['K-1spectrum_renlocal'] = Km1spectrum_renlocal
            data_dict['E0_rensubl'] = vacuumE_rensubl
            data_dict['K1spectrum_rensubl'] = K1spectrum_rensubl
            data_dict['K-1spectrum_rensubl'] = Km1spectrum_rensubl

            with open(f'data/Emax={Emax}_L={L}_g2={g2}_g4={g4}_spectrum.pkl', 'wb') as f:
                pickle.dump(data_dict, f)

        if printout:
            print("Renlocal vacuum energy: ", vacuumE_renlocal)
            print("K=1 renlocal spectrum: ", K1spectrum_renlocal)
            print("K=-1 renlocal spectrum: ", Km1spectrum_renlocal)
            
            print("Rensubl vacuum energy: ", vacuumE_rensubl)
            print("K=1 rensubl spectrum: ", K1spectrum_rensubl)
            print("K=-1 rensubl spectrum: ", Km1spectrum_rensubl)

        return [ vacuumE_renlocal, vacuumE_rensubl ], \
               [ K1spectrum_renlocal, K1spectrum_rensubl ], \
               [ Km1spectrum_renlocal, Km1spectrum_rensubl ]

def plot_figure4a(a, Emax=20, L=10, m=1, ren=False):
    """ Plot vacuum energy vs g for phi^4 theory.
    """
    u.plotStyle()
    g2 = 0
    garr = np.linspace(0, 5, 26)
    e0_arr = []
    for g4 in garr:
        print('g4 = {:.3f} ...'.format(g4), end='\r')
        E0, _, _ = calcPhi4(a, g4, Emax, neigs=3, g2=g2, L=L, m=m,
                            save=True,
                            ren=ren)
        e0_arr.append(E0) 
        if not ren: 
            print('g4 = {:.3f}, E0 = {:.3f}'.format(g4, E0))
        else: 
            ### renlocal, rensubl
            print('g4 = {:.3f}, E0_renlocal = {:.3f}, E0_rensubl = {:.3f}'.format(g4, *E0))
    
    fig, ax = plt.subplots()
    if not ren: 
        ax.plot(garr, e0_arr, 'o-', markersize=3)
    else:
        e0_arr = np.array(e0_arr)
        ax.plot(garr, e0_arr[:,0], '-', markersize=3)
        ax.plot(garr, e0_arr[:,1], '--', markersize=3)

    ax.set_xlabel('$g_4$')
    ax.set_ylabel('$E_0$')
    ax.set_xlim([0,5])
    if ren: 
        leg = plt.legend([r'$\rm ren.$',r'$\rm subl.$'])
        leg.get_frame().set_linewidth(0.)
    
    ax.set_title(r"$m={}$, $L={}$, $E_{{\rm max}} = {}$".format(
        m, 
        L if abs(L-2*np.pi) > 1e-3 else 6.28, 
        Emax))
    
    fig.tight_layout()
    plt.savefig('plots/reproduce_fig4a_{}_phi4{}.pdf'.format(
        'raw'   if not ren else 'ren',
        ''      if not doMinimal else '_minimal'))
    os.system('dropbox_uploader.sh upload plots/reproduce_fig4a_{}_phi4{}.pdf /tmp/'.format(
        'raw'   if not ren else 'ren',
        ''      if not doMinimal else '_minimal'))


def plot_figure4b(a, Emax=20, L=10, m=1, ren=False):
    """ Plot excited state energy - vacuum energy vs g for phi^4 theory.
    """
    u.plotStyle()
    g2 = 0
    garr = np.linspace(0, 5, 26)
    
    if not ren: 
        deltaE_dict = {i: [] for i in range(5)}
    else:
        deltaE_dict = {i: 
                        {'renlocal': [],
                         'rensubl' : []
                        } for i in range(5) 
                      }

    for g4 in garr:
        print('g4 = {:.3f} ...'.format(g4), end='\r')
        E0, K1spectrum, Km1spectrum = calcPhi4(a, g4, Emax, neigs=3, g2=g2, L=L, m=m,
                            save=True,
                            ren=ren)
        
        #### fill out the excited state energy against the ground state energy
        #### for E_{I} where I: 1 -> 5
        if not ren: 
            deltaE_dict[0].append( Km1spectrum[0] )
            deltaE_dict[1].append( K1spectrum[0]  )
            deltaE_dict[2].append( K1spectrum[1]  )
            deltaE_dict[3].append( Km1spectrum[1] )
            deltaE_dict[4].append( Km1spectrum[2] )
            print('g4 = {:.3f}, dE1 = {:.3f}, dE2 = {:.3f}, dE3 = {:.3f}, dE4 = {:.3f}, dE5 = {:.3f}'.format(
                   g4, deltaE_dict[0][-1], \
                       deltaE_dict[1][-1], \
                       deltaE_dict[2][-1], \
                       deltaE_dict[3][-1], \
                       deltaE_dict[4][-1] ))
        else:
            # print(E0)
            # print(K1spectrum)
            # print(Km1spectrum)
            deltaE_dict[0]['renlocal'].append( Km1spectrum[0][0] )
            deltaE_dict[1]['renlocal'].append( K1spectrum[0][0]  )
            deltaE_dict[2]['renlocal'].append( K1spectrum[0][1]  )
            deltaE_dict[3]['renlocal'].append( Km1spectrum[0][1] )
            deltaE_dict[4]['renlocal'].append( Km1spectrum[0][2] )

            deltaE_dict[0]['rensubl'].append( Km1spectrum[1][0] )
            deltaE_dict[1]['rensubl'].append( K1spectrum[1][0]  )
            deltaE_dict[2]['rensubl'].append( K1spectrum[1][1]  )
            deltaE_dict[3]['rensubl'].append( Km1spectrum[1][1] )
            deltaE_dict[4]['rensubl'].append( Km1spectrum[1][2] )

            # print(deltaE_dict)

            print('g4 = {:.3f}, ren local:\n'
                  'dE1 = {:.3f}, dE2 = {:.3f}, dE3 = {:.3f}, dE4 = {:.3f}, dE5 = {:.3f}'.format(
                   g4, deltaE_dict[0]['renlocal'][-1], \
                       deltaE_dict[1]['renlocal'][-1], \
                       deltaE_dict[2]['renlocal'][-1], \
                       deltaE_dict[3]['renlocal'][-1], \
                       deltaE_dict[4]['renlocal'][-1] ))
            print('g4 = {:.3f}, ren subl.:\n'
                  'dE1 = {:.3f}, dE2 = {:.3f}, dE3 = {:.3f}, dE4 = {:.3f}, dE5 = {:.3f}'.format(
                   g4, deltaE_dict[0]['rensubl'][-1], \
                       deltaE_dict[1]['rensubl'][-1], \
                       deltaE_dict[2]['rensubl'][-1], \
                       deltaE_dict[3]['rensubl'][-1], \
                       deltaE_dict[4]['rensubl'][-1] ))

    
    fig, ax = plt.subplots()
    if not ren: 
        ax.plot(garr, deltaE_dict[0], '-', markersize=3, color='red')
        ax.plot(garr, deltaE_dict[1], '-', markersize=3, color='blue')
        ax.plot(garr, deltaE_dict[2], '-', markersize=3, color='blue')
        ax.plot(garr, deltaE_dict[3], '-', markersize=3, color='red')
        ax.plot(garr, deltaE_dict[4], '-', markersize=3, color='red')
    else:
        ax.plot(garr, deltaE_dict[0]['renlocal'], '-', markersize=3, color='red')
        ax.plot(garr, deltaE_dict[0]['rensubl'], '--', markersize=3, color='red')
        ax.plot(garr, deltaE_dict[1]['renlocal'], '-', markersize=3, color='blue')
        ax.plot(garr, deltaE_dict[1]['rensubl'], '--', markersize=3, color='blue')
        ax.plot(garr, deltaE_dict[2]['renlocal'], '-', markersize=3, color='blue')
        ax.plot(garr, deltaE_dict[2]['rensubl'], '--', markersize=3, color='blue')
        ax.plot(garr, deltaE_dict[3]['renlocal'], '-', markersize=3, color='red')
        ax.plot(garr, deltaE_dict[3]['rensubl'], '--', markersize=3, color='red')
        ax.plot(garr, deltaE_dict[4]['renlocal'], '-', markersize=3, color='red')
        ax.plot(garr, deltaE_dict[4]['rensubl'], '--', markersize=3, color='red')
    
    ax.set_xlabel('$g_4$')
    ax.set_ylabel('$E_I - E_0$')
    ax.set_xlim([0,5])
    ax.set_ylim([0,8 if doMinimal else 6])
    if not ren: 
        leg = plt.legend(['$Z_2 = -$','$Z_2 = +$'],
            bbox_to_anchor = (0.99, 0.99), loc='upper right')
        leg.get_frame().set_linewidth(0.)
    else:
        leg = plt.legend([r'${\rm ren.~}Z_2 = -$',
                          r'${\rm subl.~}Z_2 = -$',
                          r'${\rm ren.~}Z_2 = +$',
                          r'${\rm subl.~}Z_2 = +$'],
            bbox_to_anchor = (0.99, 0.99), loc='upper right')
        leg.get_frame().set_linewidth(0.)

    ax.set_title(r"$m={}$, $L={}$, $E_{{\rm max}} = {}$".format(
        m, 
        L if abs(L-2*np.pi) > 1e-3 else 6.28, 
        Emax))
    
    fig.tight_layout()
    plt.savefig('plots/reproduce_fig4b_{}_phi4{}.pdf'.format(
        'raw'   if not ren else 'ren',
        ''      if not doMinimal else '_minimal'))
    os.system('dropbox_uploader.sh upload plots/reproduce_fig4b_{}_phi4{}.pdf /tmp/'.format(
        'raw'   if not ren else 'ren',
        ''      if not doMinimal else '_minimal'))


    #### do fitting ###
    deltaE_arr = deltaE_dict[0] if not ren else \
                 deltaE_dict[0]['rensubl']

    def mph_fit(g, C, gc, Delta): return C * np.power( abs(g - gc), 1/(2-Delta) )

    ### fitting region from 1.4 to 2.4 => index 7-12

    p0_gc  = garr[ np.argmin(deltaE_arr) ]
    p0_arr = [-1/p0_gc, p0_gc, 1]
    print("Initializing fitting, with C: {:.3f}, gc: {:.3f}, Delta: {:.3f}".format(*p0_arr))
    param, param_cov = scipy.optimize.curve_fit(mph_fit, 
                        garr[7:13], deltaE_arr[7:13],
                        p0 = p0_arr )

    print("Fitted params:")
    print("C: {:.3f}, gc: {:.3f}, Delta: {:.3f}".format(*param))
    print("Covariance matrix:")
    print(param_cov)

    gc      = param[1]
    gc_err  = np.sqrt(param_cov[1][1])
    Delta   = param[2]
    Delta_err = np.sqrt(param_cov[2][2])


    x       = np.linspace(1.4, 2.4, 30)
    y       = mph_fit(x, *param)
    x_extrap= np.linspace(2.4, gc, 30)
    y_extrap= mph_fit(x_extrap, *param)
    
    plt.plot(x, y, '-',  color ='black', linewidth = 1)
    plt.plot(x_extrap, y_extrap, '--',  color ='black', linewidth = 1)

    plt.text(0.1, 7.3/8*ax.get_ylim()[1], 
        r'$g_{{4,c}}{{\rm~from~fit:~}}{:.2f} \pm {:.4f}$'.format( gc, gc_err ),
        fontsize=20)
    plt.text(0.1, 6.4/8*ax.get_ylim()[1], 
        r'$\Delta_{{\epsilon}}{{\rm~from~fit:~}}{:.2f} \pm {:.4f}$'.format( Delta, Delta_err ),
        fontsize=20)


    plt.savefig('plots/reproduce_fig4b_{}_phi4_fit{}.pdf'.format(
        'raw'   if not ren else 'ren',
        ''      if not doMinimal else '_minimal'))
    os.system('dropbox_uploader.sh upload plots/reproduce_fig4b_{}_phi4_fit{}.pdf /tmp/'.format(
        'raw'   if not ren else 'ren',
        ''      if not doMinimal else '_minimal'))

if __name__ == '__main__':
    main()

