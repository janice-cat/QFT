######################################################
# 
# Fock space Hamiltonian truncation for phi^4 theory in 2 dimensions
# Authors: Slava Rychkov (slava.rychkov@lpt.ens.fr) and Lorenzo Vitale (lorenzo.vitale@epfl.ch)
# December 2014
#
######################################################

import scipy
from scipy import sqrt, pi
from operator import attrgetter
import math

tol = 0.00001

def omega(n,L,m):
    """ computes one particle energy from wavenumber"""
    return sqrt(m**2.+((2.*pi/L)*n)**2.)
def k(n,L):
    """ computes momentum from wavenumber"""
    return (2.*pi/L)*n

class State():
    def __init__(self, occs, nmax, L=None, m=None, fast=False, checkAtRest=True):
        """ occs: occupation number list
            nmax: wave number of the last element in occs """
        self.occs = occs
        self.size = len(self.occs)
        self.nmax = nmax
        self.fast = fast
        
        if fast == True:
            return

        wavenum = scipy.vectorize(lambda i: i-self.size+self.nmax+1)(range(self.size))
        energies = scipy.vectorize(lambda k : omega(k,L,m))(wavenum)

        self.totalWN = (wavenum*self.occs).sum()

        if checkAtRest:
            if self.totalWN != 0:            
                raise ValueError("State not at rest")

        if self.size == 2*self.nmax+1 and self.occs[::-1] == self.occs:
            self.__parityEigenstate = True
        else:
            self.__parityEigenstate = False
            
        self.L = L
        self.m = m
        self.energy = sum(energies*self.occs)
        self.momentum = (2.*pi/self.L)*self.totalWN
    
    def isParityEigenstate(self):
        """ Returns True if the Fock space state is a P-parity eigenstate """
        return self.__parityEigenstate        

    def Kparity(self):
        """ Returns the K-parity quantum number """
        return (-1)**sum(self.occs)

    def __repr__(self):
        return str(self.occs)
    
    def __eq__(self, other):
       return (self.occs == other.occs) or (self.occs == other.occs[::-1])
       # check also if the P-reversed is the same!

    def __hash__(self):
        return hash(tuple(self.occs))

    def __setitem__(self, wn, n):
        """ Sets the occupation number corresponding to a wave number """
        if self.fast==False:
            self.energy += (n-self[wn])*omega(wn,self.L,self.m)
            self.totalWN += (n-self[wn])*wn
            self.momentum = (2.*pi/self.L)*self.totalWN
        
        self.occs[wn+self.size-self.nmax-1] = n

    def __getitem__(self, wn):
        """ Returns the occupation number corresponding to a wave number"""
        return self.occs[wn+self.size-self.nmax-1]
    
    def parityReversed(self):
        """ Reverse under P parity """
        if not self.size == 2*self.nmax+1:
            raise ValueError("attempt to reverse asymmetric occupation list")
        return State(self.occs[::-1],self.nmax,L=self.L,m=self.m)

class NotInBasis(LookupError):
    """ Exception class """
    pass
    
class Basis():
    """ Generic list of basis elements sorted in energy. """
    def __init__(self, L, Emax, m, K, nmax=None):
        """ nmax: if not None, forces the state vectors to have length 2nmax+1
            K: field parity
        """
        self.L = L
        self.Emax = Emax
        self.m = m
        self.K = K
        
        if nmax == None:
            self.nmax = int(math.floor(sqrt((Emax/2.)**2.-m**2.)*self.L/(2.*pi)))
        else:
            self.nmax=nmax
        
        self.stateList = sorted(self.__buildBasis(), key=attrgetter('energy'))
        # Collection of Fock space states, possibly sorted in energy

        self.reversedStateList = [state.parityReversed() for state in self.stateList]
        # P-parity reversed collection of Fock-space states

        self.statePos = { state : i for i, state in enumerate(self.stateList) }
        self.reversedStatePos = { state : i for i, state in enumerate(self.reversedStateList) }

        self.size = len(self.stateList)

    def __repr__(self):
        return str(self.stateList)

    def __getitem__(self,index):
        return self.stateList[index]

    def lookup(self, state):
        # Now this is implemented only for P-even states. Generalization for P-odd states is straightforward
        """looks up the index of a state. If this is not present, tries to look up for its parity-reversed """
        try:
            i = self.statePos[state]

            c=1.
            if(self.stateList[i].isParityEigenstate()):
                c=scipy.sqrt(2.)
                # Required for state normalization
            return (c, i)
        
        except KeyError:
            # In case the state is not found
            try:
                return (1., self.reversedStatePos[state])
            except KeyError:
                raise NotInBasis()

    def __buildRMlist(self):
        """ sets list of all right -moving states with particles of individual wave number 
        <= nmax, total momentum <= Emax/2 and total energy <= Emax
        This function works by first filling in n=1 mode in all possible ways, then n=2 mode
        in all possible ways assuming the occupation of n=1 mode, etc"""

        if self.nmax == 0:
            self.__RMlist = [State([],0,L=self.L,m=self.m,checkAtRest=False)]
            return
        
        kmax = max(0., scipy.sqrt((self.Emax/2.)**2.-self.m**2.))
                
        maxN1 = int(math.floor(
            min(kmax/k(1,self.L), self.Emax/omega(1,self.L,self.m))
            )) #maximal occupation number of n=1 mode
        
        RMlist0 = [State([N],1,L=self.L,m=self.m,checkAtRest=False) for N in range(maxN1+1)]
        # seed list of RM states,all possible n=1 mode occupation numbers
        
        for n in range(2,self.nmax+1): #go over all other modes
            RMlist1=[] #we will take states out of RMlist0, augment them and add to RMlist1
            for RMstate in RMlist0: # cycle over all RMstates
                p0 = RMstate.momentum
                e0 = RMstate.energy
                maxNn = int(math.floor(
                    min((kmax-p0)/k(n,self.L), (self.Emax-scipy.sqrt(self.m**2+p0**2)-e0)/omega(n,self.L,self.m))
                    ))#maximal occupation number of mode n given the occupation numbers of all previous modes
                                    
                for N in range(maxNn+1):
                    longerstate=RMstate.occs[:]
                    longerstate.append(N) #add all possible occupation numbers for mode n 
                    RMlist1.append(State(longerstate,len(longerstate),L=self.L,m=self.m, checkAtRest=False))
            #RMlist1 created, copy it back to RMlist0
            RMlist0 = RMlist1
            
        self.__RMlist = RMlist0 #save list of RMstates in an internal variable 

    def __divideRMlist(self):
        """ divides the list of RMstates into a list of lists, RMdivided,
        so that two states in each list have a fixed total RM wavenumber,
        also each sublist is ordered in energy"""
        
        self.__nRMmax=max([RMstate.totalWN for RMstate in self.__RMlist])
        self.__RMdivided = [[] for ntot in range(self.__nRMmax+1)] #initialize list of lists
        for RMstate in self.__RMlist: #go over RMstates and append them to corresponding sublists
            self.__RMdivided[RMstate.totalWN].append(RMstate)
        
        #now sort each sublist in energy
        for RMsublist in self.__RMdivided:
            RMsublist.sort(key=attrgetter('energy'))
        
    # finally function which builds the basis        
    def __buildBasis(self):
        """ creates basis of states of total momentum zero and energy <=Emax """
        self.__buildRMlist()
        self.__divideRMlist()

        statelist = []

        for nRM,RMsublist in enumerate(self.__RMdivided):
            for i, RMstate in enumerate(RMsublist):
                ERM = RMstate.energy
                for LMstate in RMsublist[i:]: # LM part of the state will come from the same sublist. We take the position of LMState to be greater or equal to the position of RMstate
                    #we will just have to reverse it
                    ELM = LMstate.energy
                    deltaE = self.Emax - ERM - ELM
                    if deltaE < 0: #if this happens, we can break since subsequent LMstates have even higherenergy (RMsublist is ordered in energy)
                        break
                    
                    maxN0 = int(math.floor(deltaE/self.m))
                                        
                    for N0 in range(maxN0+1):
                        #possible values for the occupation value at rest

                        state = State(LMstate.occs[::-1]+[N0]+RMstate.occs, self.nmax, L=self.L,m=self.m,checkAtRest=True)
                        
                        if self.K == state.Kparity():
                            statelist.append(state)

        return statelist
