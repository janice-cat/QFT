## Fock space Hamiltonian truncation for phi^4 theory in 2 dimensions
- Authors: Slava Rychkov (slava.rychkov@lpt.ens.fr) and Lorenzo Vitale (lorenzo.vitale@epfl.ch)                                      
- December 2014                                                             
---

EXAMPLE:

Create and save to file the potential matrices, for L=6 and Emax=22:
```bash
python genMatrix.py 6 22
```
Calculate the spectrum for g=1, L=6 and Emax=20:
```bash
python phi4eigs.py Emax=22.0_L=6.0.npz 1 20
```

### [Fig. 2](https://github.com/janice-cat/QFT/blob/master/plots/reproduce_fig2_raw_phi2.pdf)
```bash
python3 runphi2.py
```

### [Fig. 4a](https://github.com/janice-cat/QFT/blob/master/plots/reproduce_fig4a_raw_phi4.pdf), [Fig. 4b](https://github.com/janice-cat/QFT/blob/master/plots/reproduce_fig4b_raw_phi4_fit.pdf), [Fig. 13a](https://github.com/janice-cat/QFT/blob/master/plots/reproduce_fig13a_raw_phi4.pdf), [Fig. 13b](https://github.com/janice-cat/QFT/blob/master/plots/reproduce_fig13b_raw_phi4.pdf)
```bash
#### paper version (L=2π, Emax=10)
python3 runphi4.py 0 0
python3 runphi4.py 0 1
```


```bash
#### minimal version (L=2π, Emax=10)
python3 runphi4.py 1 0
python3 runphi4.py 1 1
```