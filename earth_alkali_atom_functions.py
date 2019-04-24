
from __future__ import division, print_function, absolute_import

from math import exp,log,sqrt
# for web-server execution, uncomment the following two lines
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re
import shutil

from .wigner import Wigner6j,Wigner3j,wignerD,CG,wignerDmatrix
from scipy.constants import physical_constants, pi , epsilon_0, hbar
from scipy.constants import k as C_k
from scipy.constants import c as C_c
from scipy.constants import h as C_h
from scipy.constants import e as C_e
from scipy.constants import m_e as C_m_e
from scipy.optimize import curve_fit

# for matrices
from numpy import zeros,savetxt, complex64,complex128
from numpy.linalg import eigvalsh,eig,eigh
from numpy.ma import conjugate
from numpy.lib.polynomial import real

from scipy.sparse import csr_matrix
from scipy.sparse import kron as kroneckerp
from scipy.sparse.linalg import eigsh
from scipy.special.specfun import fcoef
from scipy import floor

import sys, os
if sys.version_info > (2,):
    xrange = range

try:
    import cPickle as pickle   # fast, C implementation of the pickle
except:
    import pickle   # Python 3 already has efficient pickle (instead of cPickle)
import gzip
import csv
import mpmath


#taken directly from alkali_atom_functions
DPATH = os.path.join(os.path.expanduser('~'), '.arc-data')

def setup_data_folder():
    """ Setup the data folder in the users home directory.

    """
    if not os.path.exists(DPATH):
        os.makedirs(DPATH)
        dataFolder = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data")
        for fn in os.listdir(dataFolder):
            if os.path.isfile(os.path.join(dataFolder, fn)):
                shutil.copy(os.path.join(dataFolder, fn), DPATH)

class EarthAlkaliAtom(object):

    #these are the funtions we want to implement
    #we need to initalise all the values

    #read in the energies and find a way to store them

    #does this change with the atom?
    alpha = physical_constants["fine-structure constant"][0]

    a1,a2,a3,a4,rc = [0],[0],[0],[0],[0]
    """
        Model potential parameters fitted from experimental observations for
        different l (electron angular momentum)
    """
    alphaC = 0.0    #: Core polarizability
    Z = 0.0       #: Atomic number

    # state energies from different
    #store the energy levels as a dictionary pointing to an (n x 3 )array,
    # [n  0], n Values
    #[n, 1], energy for that n
    #[n, 2] who it came from

    #values are read in in increasing l, s, but decreasing j aka 1S0, 3S0,1P1, 3P2, 3P1, 3P0...
    #utill all values up to 3F2 are read in

    #Spectroscopic term notation for all spectral lines to be read in
    level_labels = ["1S0","3S1", "1P1","3P2","3P1", "3P0","1D2","3D3","3D2","3D1","1F3","3F4","3F3","3F2"]
    sEnergy = dict()
    NISTdataLevels = dict()     #number of energy levels for all 14 different transition lines
    scaledRydbergConstant = 0 #: in eV

    quantumDefect = dict()

    """ Contains dictionary pointing to array  of modified Rydberg-Ritz coefficients for calculating
        quantum defects. The array is (m = 14, n = 5), where m is the number of transition lines we deal with,
        [m,0] -  the min n for which these defect can be applied
        [m,1] - the max n for which these defect can be applied
        [m,2] - \delta0
        [m,3] - \delta2
        [m,4] - \delta4
    """


    levelDataFromNIST = [""]*14             #: Store list of filenames to read in
    quantumDefectData = ""                  #: Store list of quantum defect names to read in
    dipoleMatrixElementFile = ""            #: location of hard-disk stored dipole matrix elements
    quadrupoleMatrixElementFile = ""        #: location of hard-disk stored dipole matrix elements

    dataFolder = DPATH

    # now additional literature sources of dipole matrix elements

    literatureDMEfilename = ""
    """
        Filename of the additional literature source values of dipole matrix
        elements.

        These additional values should be saved as reduced dipole matrix elements
        in J basis.

    """
    refs = dict()
    errs = dict()

    #: levels that are for smaller principal quantum number (n) than ground level, but are above in energy due to angular part
    extraLevels = dict()

    #: principal quantum number for the ground state
    groundStateN = 0

    #: swich - should the wavefunction be calculated with Numerov algorithm implemented in C++
    #LIZZY have turned this to false bc I don't want to deal with it
    cpp_numerov = False

    mass = 0.  #: atomic mass in kg
    abundance = 1.0  #: relative isotope abundance

    elementName = "elementName"  #: Human-readable element name

    preferQuantumDefects = False  #:would the user rather use quantum defects overall?
    minQuantumDefectN =dict(zip(level_labels,[10] *14))  #: minimal quantum number for which quantum defects can be used; uses measured energy levels otherwise
    maxQuantumDefectN =dict(zip(level_labels,[20] *14))

    extrapolate_data = True #should we extrapolate the Rydberg-Ritz values to n larger than that for which it was originally fitted? (This will only extrapolate
    #for n larger than the one fitted)
    # SQLite connection and cursor
    conn = False
    c = False

    def __init__(self,preferQuantumDefects=True,cpp_numerov=False, uncertainty=False):
        #nikolai does lots with loading and saving stuff to and from a database (ignore for now)

        #TODO LIZZY

        # Always load NIST data on measured energy levels;
        # Even when user wants to use quantum defects, qunatum defects for
        # lowest lying state are not always so accurate, so below the
        # minQuantumDefectN cut-off (defined for each element separately)
        # getEnergy(...) will always return measured, not calculated energy levels

        #loop over all the different spectral lines
        deltas = []
        if(self.quantumDefectData == ""):
            preferQuantumDefects = False
            print("Quantum defect data file not specified, only measured values will be used ")#work out the lowest values for which we can use
        else:
            deltas = self._parse_QuantumDefects(os.path.join(self.dataFolder,self.quantumDefectData), uncertainty)

        for i in range(len(self.level_labels)):
            if (self.levelDataFromNIST[i] == ""):
                preferQuantumDefects = True
                print("NIST level data file for "+self.level_labels[i]+" not specified. Only quantum defects will be used for this energy level.")
            else:
                levels, ref, err = self._parseLevelsFromNIST(os.path.join(self.dataFolder,\
                                                   self.levelDataFromNIST[i]), self.NISTdataLevels[self.level_labels[i]])

            self._addEnergy(levels, self.level_labels[i], ref, err, uncertainty)
            #set up the quantum defects
            #self.quantumDefect[self.level_labels[i]] = deltas[i]


    def _parseLevelsFromNIST(self,fileData,levels):
        """
            Parses the level energies from CSV file we have of the energy values

            args:
                fileData : string - filename
                levels : int - number of levels
            returns:
                array (levels, 3) - array to be passed to the sEnergy dictionary.
            """
        f = open(fileData,"r")

        temp = np.full((levels+self.groundStateN+1,1), 999999999.0)
        r = [""]*(levels+self.groundStateN +1)
        err = np.zeros(((levels+self.groundStateN +1),1))
        count = 0
        for line in f:
            split = line.split(",")
            split = [x.strip() for x in split]
            n = int(split[0])   #pricipal quantum number n
            if (split[1] == ""):
                temp[n, 0] = 999999999.0 #set it really high
            else:
                temp[n, 0] = float(split[1])      #corresponding energy cm^-1
            #can include references in the information if you care enough
            r[n] = split[2]

            if(split[3] == ""):
                err[n] = 0
            else:
                err[n] = float(split[3])
            count += 1

        return temp, r,err

    def _parse_QuantumDefects(self, fileName, uncert):
        """
            Reads in the quantum defect file, stores all the \delta values in an array
        """

        #deltas = np.zeros((14,3))
        f = open(fileName, 'r')
        counter = 0
        for line in f:
            split = line.split(",")
            split = [str(x.strip()) for x in split]
            #remove the bracket
            if uncert == True:
                split[3] = float(split[3])+ float(split[7])
                split[4] = float(split[4])+ float(split[8])
            self.minQuantumDefectN[split[0]] = int(split[1])
            self.maxQuantumDefectN[split[0]] = int(split[2])

            defects = [float(x) for x in split[3:6]]
            #deltas[counter] = defects
            self.quantumDefect[split[0]]  = defects
        return

    def _addEnergy(self, energyNIST, label, ref, err, uncert):
        """
            Adding energy levels and references to ref

            Accepts energy level relative to **ground state**, and
            saves energy levels, relative to the **ionization treshold**.

            Args:
                energyNIST: groundStateN + NISTdataLevels[label]
                    - [n,0] - energy value relative to nonexcited level (= 0 eV)
        """
        #this is slow so make it more pythonic?
        if label in self.extraLevels:
            for n in range(self.extraLevels[label],energyNIST.shape[0]):
                if uncert == True:
                    energyNIST[n,0] = (energyNIST[n,0] +err[n] -self.ionisationEnergycm)*2/219475#*4.55633e-6#  - self.ionisationEnergy #turned to to hartrees# - self.ionisationEnergy
                else:
                    energyNIST[n,0] = (energyNIST[n,0] -self.ionisationEnergycm)*2/219475#*4.55633e-6#  - self.ionisationEnergy #turned to to hartrees# - self.ionisationEnergy

            self.sEnergy[label] = energyNIST[:-1]
            self.refs[label] = ref[:-1]
            self.errs[label] = err[:-1]
        else:
            for n in range(self.groundStateN, energyNIST.shape[0]):
                if uncert == True:
                    energyNIST[n,0] = (energyNIST[n,0] +err[n] -self.ionisationEnergycm)*2/219475#*4.55633e-6#  - self.ionisationEnergy #turned to to hartrees# - self.ionisationEnergy

                else:
                    energyNIST[n,0] = (energyNIST[n,0] -self.ionisationEnergycm)*2/219475#*4.55633e-6#ionisationEnergy
                #print(energyNIST[n,0])
            self.sEnergy[label] = energyNIST
            self.refs[label] = ref
            self.errs[label] = err

    def getEnergy(self,n,L,S,J):
        """
            Energy of the level relative to the ionisation level (in eV)
            It will be negative for all valid energy levels. If the energy is positive, no measured values were found.

            Returned energies are with respect to the center of gravity of the
            hyperfine-split states.
            If `preferQuantumDefects` =False (set during initialization) program
            will try use NIST energy value, if such exists, falling back to energy
            calculation with quantum defects if the measured value doesn't exist.
            For `preferQuantumDefects` =True, program will always calculate
            energies from quantum defects (useful for comparing quantum defect
            calculations with measured energy level values).


            Returns:
                float: state energy (eV)
        """
        #creates the string term.
        term = self.getTerm(L,S,J)
        #gets the index for use with the corresponding lists
        idx = self.getIndex(term)

        energy_arr = self.sEnergy[term]


        if (not self.preferQuantumDefects and
            (n <= self.NISTdataLevels[term]+5 and energy_arr[n] <= self.ionisationEnergycm)):

            return energy_arr[n]

        elif((n>=self.minQuantumDefectN[term]) and\
                                (n<=self.maxQuantumDefectN[term] or (self.extrapolate_data ==True and n>self.maxQuantumDefectN[term] ))):
            #print('got here',n,self.NISTdataLevels[term]+5, self.maxQuantumDefectN[term])
            if(n>self.maxQuantumDefectN[term]):
                print("CAUTION: The Rydberg-Ritz values have been extrapolated beyond their fitted values")
            defect = self.getQuantumDefect(n, L,S,J)
            return - self.scaledRydbergConstant//219475/((n-defect)**2)# self.scaledRydbergConstant
        else:
            raise ValueError("We there are no measured energy levels, or Rydberg-Ritz coefficients fitted for the n you have specified.")

    def getTerm(self,L,S,J):
        '''This returns the correct key for the given values and checks that the values inputted are valid L,S and J
            Args:
                L: total orbital angular quantum numbers
                S: total spin number
                J: total angular quantum number
        '''
        if(S != 0 and S !=1):
            print('That is not a valid total spin number (S)')
        elif(L< 0 or L>4):
            print('That is not a vaild orbital angular quantum number (L)')
        elif(J < ( L-1) or J > (L+1)):
            print('That is not a vaild total angular quantum number (J)')
        else:
            if L == 0:
                return str(S*2+1)+'S'+str(J)
            elif L == 1:
                return str(S*2+1)+'P'+str(J)
            elif L == 2:
                return str(S*2+1)+'D'+str(J)
            elif L == 3:
                return str(S*2+1)+'F'+str(J)

    def getIndex(self,term):
        '''This gets the index where all data stored in lists will be kept. for use with:

        levelDataFromNIST, level_labels

        Args:
        term - string which corresponds to a key for the sEnergy and quantumDefect dictionaries
        returns idx - int corresponding to its index
        '''
        return self.level_labels.index(term)
    def getQuantumNumbers(self,term):
        """Takes a term scheme and turns it into its component L,S,J values."""
        s = (int(term[0]) -1.)/2.
        j = int(term[2])
        l = None
        if term[1] == 'S':
            l = 0
        elif term[1] == 'P':
            l = 1
        elif term[1] == 'D':
            l = 2
        elif term[1] == 'F':
            l = 3
        return l,s,j

    def getRef(self,n,L,S,J):
        term = self.getTerm(L,S,J)
        return self.refs[term][n]

    def getQuantumDefect(self, n, L,S,J):
        """
            Quantum defect of the level.

            For an example, see `Rydberg energy levels example snippet`_.

            .. _`Rydberg energy levels example snippet`:
                ./Rydberg_atoms_a_primer.html#Rydberg-Atom-Energy-Levels


            Returns:
                float: quantum defect
        """
        term = self.getTerm(L,S,J)
        if(self.minQuantumDefectN[term]>n):
            print('no quantum defects fitted for this range')
            defect = 0
        elif(self.maxQuantumDefectN[term]< n):
            #print('Rydberg ritz values have been extrapolted beyond their fitted values')
            defect = self.quantumDefect[term][0]+\
                self.quantumDefect[term][1]/((n-self.quantumDefect[term][0])**2)+\
                self.quantumDefect[term][2]/((n-self.quantumDefect[term][0])**4)
        else:
            defect = self.quantumDefect[term][0]+\
                self.quantumDefect[term][1]/((n-self.quantumDefect[term][0])**2)+\
                self.quantumDefect[term][2]/((n-self.quantumDefect[term][0])**4)

        return defect

    def getRadialCoupling(self,n,l,s,j,n1,l1,s1,j1):
        """
            Returns radial part of the coupling between two states (dipole and
            quadrupole interactions only)

            Args:
                n1 (int): principal quantum number
                l1 (int): orbital angular momentum
                j1 (float): total angular momentum
                n2 (int): principal quantum number
                l2 (int): orbital angular momentum
                j2 (float): total angular momentum

            Returns:
                float:  radial coupling strength (in a.u.), or zero for forbidden
                transitions in dipole and quadrupole approximation.

        """
        ##LIZZY Does this condition still apply?
        dl = abs(l-l1)
        if (dl == 1 and abs(j-j1)<1.1):
            #print(n," ",l," ",j," ",n1," ",l1," ",j1)
            return self.getRadialMatrixElementSemiClassical(n,l,s,j,n1,l1,s1,j1)
        elif (dl==0 or dl==1 or dl==2) and(abs(j-j1)<2.1):
            # quadrupole coupling
            return 0.
            #LIZZY: need to impelment
            #return self.getQuadrupoleMatrixElement(n,l,j,n1,l1,j1)
        else:
            # neglect octopole coupling and higher
            #print("NOTE: Neglecting couplings higher then quadrupole")
            return 0

    def getAngularMatrixElement(self, l, s, j, m, l1, s1, j1, m1):
        ''' p - this is the electric field polarisation
           I have passed these in as an argument for now can worry about working them out later.
         '''
        sum = 0
        for p in range(-1,2):                  #sum over all polariasations
            sign = (-1)**(s-m-l-j)
            elem = sign * sqrt((2*j+1)*(2*l+1))
            elem = elem * CG(l,0,1,0,l1,0) * CG(j,m,1,p,j1,m1)
            elem = elem * Wigner6j(j, 1, j1, l1, s, l)
            sum += elem

        return sum
    def getDoubleAngME(self, l, s, j, m, l1, s1, j1, m1):
        elem =(2*j1 + 1)*l1* Wigner6j(j, 1, j1, l1, s, l)
        return elem
    def getRadialMatrixElement(self,n1,l1,j1,n2,l2,j2,useLiterature=True):
         """
             Radial part of the dipole matrix element

             Calculates :math:`\\int \\mathbf{d}r~R_{n_1,l_1,j_1}(r)\cdot \
                 R_{n_1,l_1,j_1}(r) \cdot r^3`.

             Args:
                 n1 (int): principal quantum number of state 1
                 l1 (int): orbital angular momentum of state 1
                 j1 (float): total angular momentum of state 1
                 n2 (int): principal quantum number of state 2
                 l2 (int): orbital angular momentum of state 2
                 j2 (float): total angular momentum of state 2

             Returns:
                 float: dipole matrix element (:math:`a_0 e`).
         """
         dl = abs(l1-l2)
         dj = abs(j2-j2)
         if not(dl==1 and (dj<1.1)):
             return 0

         if (self.getEnergy(n1, l1, j1)>self.getEnergy(n2, l2, j2)):
             temp = n1
             n1 = n2
             n2 = temp
             temp = l1
             l1 = l2
             l2 = temp
             temp = j1
             j1 = j2
             j2 = temp

         n1 = int(n1)
         n2 = int(n2)
         l1 = int(l1)
         l2 = int(l2)
         j1_x2 = int(round(2*j1))
         j2_x2 = int(round(2*j2))

         if useLiterature:
             # is there literature value for this DME? If there is, use the best one (smalles error)
             self.c.execute('''SELECT dme FROM literatureDME WHERE
              n1= ? AND l1 = ? AND j1_x2 = ? AND
              n2 = ? AND l2 = ? AND j2_x2 = ?
              ORDER BY errorEstimate ASC''',(n1,l1,j1_x2,n2,l2,j2_x2))
             answer = self.c.fetchone()
             if (answer):
                 # we did found literature value
                 return answer[0]


         # was this calculated before? If it was, retrieve from memory
         self.c.execute('''SELECT dme FROM dipoleME WHERE
          n1= ? AND l1 = ? AND j1_x2 = ? AND
          n2 = ? AND l2 = ? AND j2_x2 = ?''',(n1,l1,j1_x2,n2,l2,j2_x2))
         dme = self.c.fetchone()
         if (dme):
             return dme[0]

         step = 0.001
         r1,psi1_r1 = self.radialWavefunction(l1,0.5,j1,\
                                                self.getEnergy(n1, l1, j1)/27.211,\
                                                self.alphaC**(1/3.0),\
                                                 2.0*n1*(n1+15.0), step)
         r2,psi2_r2 = self.radialWavefunction(l2,0.5,j2,\
                                                self.getEnergy(n2, l2, j2)/27.211,\
                                                self.alphaC**(1/3.0),\
                                                 2.0*n2*(n2+15.0), step)

         upTo = min(len(r1),len(r2))

         # note that r1 and r2 change in same staps, starting from the same value
         dipoleElement = np.trapz(np.multiply(np.multiply(psi1_r1[0:upTo],psi2_r2[0:upTo]),\
                                            r1[0:upTo]), x = r1[0:upTo])

         self.c.execute(''' INSERT INTO dipoleME VALUES (?,?,?, ?,?,?, ?)''',\
                        [n1,l1,j1_x2,n2,l2,j2_x2, dipoleElement] )
         self.conn.commit()

         return dipoleElement

    def getRadialMatrixElementSemiClassical(self,n,l,s,j,n1,l1,s1,j1):
        #get the effective principal number of both states
        nu = n - self.getQuantumDefect(n,l,s,j)
        nu1 = n1 - self.getQuantumDefect(n1,l1,s1,j1)

        #get the parameters required to calculate the sum
        l_c = (l+l1+1.)/2.
        nu_c = sqrt(nu*nu1)

        delta_nu = nu- nu1
        delta_l = l1 -l

        gamma  = (delta_l*l_c)/nu_c

        g0 = (1./(3.*delta_nu))*(mpmath.angerj(delta_nu-1.,-delta_nu) - mpmath.angerj(delta_nu+1,-delta_nu))
        g1 = -(1./(3.*delta_nu))*(mpmath.angerj(delta_nu-1.,-delta_nu) + mpmath.angerj(delta_nu+1,-delta_nu))
        g2 = g0 - mpmath.sin(pi*delta_nu)/(pi*delta_nu)
        g3 = (delta_nu/2.)*g0 + g1

        radial_ME = (3/2)*nu_c**2*(1-(l_c/nu_c)**(2))**0.5*(g0 + gamma*g1 + gamma**2*g2 + gamma**3*g3)

        return radial_ME

    def getC6term(self,n,l,s,j,n1,l1,s1,j1,n2,l2,s2,j2):
        """
            C6 interaction term for the given two pair-states

            Calculates :math:`C_6` intaraction term for :math:`|n,l,j,n,l,j\\rangle\
            \\leftrightarrow |n_1,l_1,j_1,n_2,l_2,j_2\\rangle`. For details
            of calculation see Ref. [#c6r1]_.

            Args:
                n (int): principal quantum number
                l (int): orbital angular momenutum
                j (float): total angular momentum
                n1 (int): principal quantum number
                l1 (int): orbital angular momentum
                j1 (float): total angular momentum
                n2 (int): principal quantum number
                l2 (int): orbital angular momentum
                j2 (float): total angular momentum

            Returns:
                float:  :math:`C_6 = \\frac{1}{4\\pi\\varepsilon_0} \
                    \\frac{|\\langle n,l,j |er|n_1,l_1,j_1\\rangle|^2|\
                    \\langle n,l,j |er|n_2,l_2,j_2\\rangle|^2}\
                    {E(n_1,l_1,j_2,n_2,j_2,j_2)-E(n,l,j,n,l,j)}`
                (:math:`h` Hz m :math:`{}^6`).

            Example:
                We can reproduce values from Ref. [#c6r1]_ for C3 coupling
                to particular channels. Taking for example channels described
                by the Eq. (50a-c) we can get the values::

                    from arc import *

                    channels = [[70,0,0.5, 70, 1,1.5, 69,1, 1.5],\\
                                [70,0,0.5, 70, 1,1.5, 69,1, 0.5],\\
                                [70,0,0.5, 69, 1,1.5, 70,1, 0.5],\\
                                [70,0,0.5, 70, 1,0.5, 69,1, 0.5]]

                    print(" = = = Caesium = = = ")
                    atom = Caesium()
                    for channel in channels:
                        print("%.0f  GHz (mu m)^6" % ( atom.getC6term(*channel) / C_h * 1.e27 ))

                    print("\\n = = = Rubidium  = = =")
                    atom = Rubidium()
                    for channel in channels:
                        print("%.0f  GHz (mu m)^6" % ( atom.getC6term(*channel) / C_h * 1.e27 ))

                Returns::

                     = = = Caesium = = =
                    722  GHz (mu m)^6
                    316  GHz (mu m)^6
                    383  GHz (mu m)^6
                    228  GHz (mu m)^6

                     = = = Rubidium  = = =
                    799  GHz (mu m)^6
                    543  GHz (mu m)^6
                    589  GHz (mu m)^6
                    437  GHz (mu m)^6

                which is in good agreement with the values cited in the Ref. [#c6r1]_.
                Small discrepancies for Caesium originate from slightly different
                quantum defects used in calculations.


            References:
                .. [#c6r1] T. G. Walker, M. Saffman, PRA **77**, 032723 (2008)
                    https://doi.org/10.1103/PhysRevA.77.032723

        """
        d1 = self.getRadialMatrixElementSemiClassical(n,l,s,j,n1,l1,s1,j1)
        d2 = self.getRadialMatrixElementSemiClassical(n,l,s,j,n2,l2,s2,j2)
        a1 = self.getAngularMatrixElement(l,s,j,0,l1,s1,j1,0)           #set both Mj= Mj' = 0
        a2 = self.getAngularMatrixElement(l,s,j,0,l2,s2,j2,0)           #set both Mj= Mj'' = 0

        #a1_new =self.getAngularMatrixElement(l,s,j,0,l,s,j,0)
        #a2_new = self.getAngularMatrixElement(l1,s1,j1,0,l2,s2,j2,0)
        #dou = self.getDoubleAngME(l,s,j,0,l2,s2,j2,0)

        print(a1,a2,a1*a2)
        #print(dou)
        d1d2 = d1*d2#*a1+a2

        return d1d2**2/(self.getEnergy(n1,l1,s1,j1)+\
                                     self.getEnergy(n2,l2,s2,j2)-\
                                     2*self.getEnergy(n,l,s,j))


    def radialWavefunction(self,L,S,J,stateEnergy,innerLimit,outerLimit,step):
        """
        Radial part of electron wavefunction

        Calculates radial function with Numerov (from outside towards the core).
        Note that wavefunction might not be calculated all the way to the requested
        `innerLimit` if the divergence occurs before. In that case third returned
        argument gives nonzero value, corresponding to the first index in the array
        for which wavefunction was calculated. For quick example see
        `Rydberg wavefunction calculation snippet`_.

        .. _`Rydberg wavefunction calculation snippet`:
            ./Rydberg_atoms_a_primer.html#Rydberg-atom-wavefunctions



        Args:
            l (int): orbital angular momentum
            s (float): spin angular momentum
            j (float): total angular momentum
            stateEnergy (float): state energy, relative to ionization threshold,
                should be given in atomic units (Hatree)
            innerLimit (float): inner limit at which wavefunction is requested
            outerLimit (float): outer limit at which wavefunction is requested
            step (flaot): radial step for integration mesh (a.u.)
        Returns:
            List[float], List[flaot], int:
                :math:`r`

                :math:`R(r)\cdot r`

        .. note::
            Radial wavefunction is not scaled to unity! This normalization
            condition means that we are using spherical harmonics which are
            normalized such that
            :math:`\\int \\mathrm{d}\\theta~\\mathrm{d}\\psi~Y(l,m_l)^* \\times \
            Y(l',m_{l'})  =  \\delta (l,l') ~\\delta (m_l, m_{l'})`.

        Note:
            Alternative calculation methods can be added here (potenatial
            package expansion).

        """
        innerLimit = max(4. * step, innerLimit)  # prevent divergence due to hitting 0
        if self.cpp_numerov:
            # efficiant implementation in C
            if (l<4):
                d = self.NumerovWavefunction(innerLimit,outerLimit,\
                                        step,0.01,0.01,\
                                        L,S,J,stateEnergy,self.alphaC,self.alpha,\
                                        self.Z,
                                        self.a1[l],self.a2[l],self.a3[l],self.a4[l],\
                                        self.rc[l],\
                                        (self.mass-C_m_e)/self.mass)
            else:
                d = self.NumerovWavefunction(innerLimit,outerLimit,\
                                        step,0.01,0.01,\
                                        L,S,J,stateEnergy,self.alphaC,self.alpha,\
                                        self.Z,0.,0.,0.,0.,0.,\
                                        (self.mass-C_m_e)/self.mass)

            psi_r  = d[0]
            r = d[1]
            suma = np.trapz(psi_r**2, x=r)
            psi_r = psi_r/(sqrt(suma))
        else:
            # full implementation in Python

            #remove mu as the single atom approximation.
            #mu = (self.mass-C_m_e)/self.mass
            def potential(x):
                r = x*x
                #THIS MAY NOT BE CORRECT. we may have to use little L, in which case we are gonna have to add a new feature.
                return -2*C_m_e*stateEnergy- (L*(L+1.))/r**2 - (2*C_m_e)/r

            r,psi_r = NumerovBack(innerLimit,outerLimit,potential,\
                                         step,0.01,0.01)

            suma = np.trapz(psi_r**2, x=r)
            psi_r = psi_r/(sqrt(suma))

        return r,psi_r
    def getC6earth_Alkali():
        """need to be put in its own class"""
        pass
    def getTransitionFrequency(self,n1,l1,s1,j1,n2,l2,s2,j2):
        """
            Calculated transition frequency in Hz

            Returned values is given relative to the centre of gravity of the
            hyperfine-split states.

            Args:
                n1 (int): principal quantum number of the state **from** which we are going
                l1 (int): orbital angular momentum of the state **from** which we are going
                j1 (float): total angular momentum of the state **from** which we are going
                n2 (int): principal quantum number of the state **to** which we are going
                l2 (int): orbital angular momentum of the state **to** which we are going
                j2 (float): total angular momentum of the state **to** which we are going

            Returns:
                float:
                    transition frequency (in Hz). If the returned value is negative,
                    level from which we are going is **above** the level to which we are
                    going.
        """
        return (self.getEnergy(n2, l2, s2,j2)-self.getEnergy(n1, l1,s1, j1))*C_e/C_h
    def getTransitionWavelength(self,n1,l1,s1,j1,n2,l2,s2,j2):
        """
            Calculated transition wavelength (in vacuum) in m.

            Returned values is given relative to the centre of gravity of the
            hyperfine-split states.

            Args:
                n1 (int): principal quantum number of the state **from** which we are going
                l1 (int): orbital angular momentum of the state **from** which we are going
                j1 (float): total angular momentum of the state **from** which we are going
                n2 (int): principal quantum number of the state **to** which we are going
                l2 (int): orbital angular momentum of the state **to** which we are going
                j2 (float): total angular momentum of the state **to** which we are going

            Returns:
                float:
                    transition wavelength (in m). If the returned value is negative,
                    level from which we are going is **above** the level to which we are
                    going.
        """
        return (C_h*C_c)/((self.getEnergy(n2, l2,s2, j2)-self.getEnergy(n1, l1,s1 ,j1))*C_e)

    def getEnergyDefect(self,n,l,s,j,n1,l1,s1,j1,n2,l2,s2,j2):
        """
            Energy defect for the given two pair-states (one of the state has
            two atoms in the same state)

            Energy difference between the states
            :math:`E(n_1,l_1,j_1,n_2,l_2,j_2) - E(n,l,j,n,l,j)`

            Args:
                n (int): principal quantum number
                l (int): orbital angular momenutum
                j (float): total angular momentum
                n1 (int): principal quantum number
                l1 (int): orbital angular momentum
                j1 (float): total angular momentum
                n2 (int): principal quantum number
                l2 (int): orbital angular momentum
                j2 (float): total angular momentum

            Returns:
                float:  energy defect (SI units: J)
        """
        return (self.getEnergy(n1,l1,s1,j1)+self.getEnergy(n2,l2,s2,j2)-\
                           2*self.getEnergy(n,l,s,j))
    def getEnergyDefect2(self,n,l,s,j,nn,ll,ss,jj,n1,l1,s1,j1,n2,l2,s2,j2):
        """
            Energy defect for the given two pair-states

            Energy difference between the states
            :math:`E(n_1,l_1,j_1,n_2,l_2,j_2) - E(n,l,j,nn,ll,jj)`

            See `pair-state energy defects example snippet`_.

            .. _`pair-state energy defects example snippet`:
                ./Rydberg_atoms_a_primer.html#Rydberg-atom-interactions


            Args:
                n (int): principal quantum number
                l (int): orbital angular momenutum
                j (float): total angular momentum
                nn (int): principal quantum number
                ll (int): orbital angular momenutum
                jj (float): total angular momentum
                n1 (int): principal quantum number
                l1 (int): orbital angular momentum
                j1 (float): total angular momentum
                n2 (int): principal quantum number
                l2 (int): orbital angular momentum
                j2 (float): total angular momentum

            Returns:
                float:  energy defect (SI units: J)
        """
        return C_e*(self.getEnergy(n1,l1,s1,j1)+self.getEnergy(n2,l2,s2,j2)-\
                           self.getEnergy(n,l,s,j)-self.getEnergy(nn,ll,ss,jj))
    def NumerovBack(innerLimit,outerLimit,kfun,step,init1,init2):
        """
            Full Python implementation of Numerov integration

            Calculates solution function :math:`rad(r)` with descrete step in
            :math:`r` size of `step`, integrating from `outerLimit` towards the
            `innerLimit` (from outside, inwards) equation
            :math:`\\frac{\\mathrm{d}^2 rad(r)}{\\mathrm{d} r^2} = kfun(r)\\cdot rad(r)`.




            Args:
                innerLimit (float): inner limit of integration
                outerLimit (flaot): outer limit of integration
                kfun (function(double)): pointer to function used in equation (see
                    longer explanation above)
                step: descrete step size for integration
                init1 (float): initial value, `rad`(`outerLimit`+`step`)
                init2 (float): initial value, `rad`(`outerLimit`+:math:`2\\cdot` `step`)

            Returns:
                numpy array of float , numpy array of float, int : :math:`r` (a.u),
                :math:`rad(r)`;

            Note:
                Returned function is not normalized!

            Note:
                If :obj:`AlkaliAtom.cpp_numerov` swich is set to True (default option),
                much faster C implementation of the algorithm will be used instead.
                That is recommended option. See documentation installation
                instructions for more details.

        """
        br = int((sqrt(outerLimit)-sqrt(innerLimit))/step)
        sol = np.zeros(br,dtype=np.dtype('d'))  # integrated wavefunction R(r)*r^{3/4}
        rad = np.zeros(br,dtype=np.dtype('d'))  # radial coordinate for integration \sqrt(r)

        br = br-1
        x = sqrt(innerLimit)+step*(br-1)
        sol[br] = (2.*(1.-5.0/12.0*step**2*kfun(x))*init1-\
                   (1.+1./12.0*step**2*kfun(x+step))*init2)/\
                   (1+1/12.0*step**2*kfun(x-step))
        rad[br] = x

        x = x-step
        br = br-1

        sol[br] = (2.*(1.-5.0/12.0*step**2*kfun(x))*sol[br+1]-\
                   (1.+1./12.0*step**2*kfun(x+step))*init1)/\
                   (1+1/12.0*step**2*kfun(x-step))
        rad[br] = x

        # check if the function starts diverging  before the innerLimit
        # -> in that case break integration earlier

        maxValue = 0.

        checkPoint = 0
        fromLastMax = 0

        while br>checkPoint:
            br = br-1
            x = x-step
            sol[br] = (2.*(1.-5.0/12.0*step**2*kfun(x))*sol[br+1]-\
                       (1.+1./12.0*step**2*kfun(x+step))*sol[br+2])/\
                       (1.+1./12.0*step**2*kfun(x-step))
            rad[br] = x
            if abs(sol[br]*sqrt(x))>maxValue:
                maxValue = abs(sol[br]*sqrt(x))
            else:
                fromLastMax += 1
                if fromLastMax>50:
                    checkPoint = br
        # now proceed with caution - checking if the divergence starts
        # - if it does, cut earlier

        divergencePoint = 0

        while (br>0)and(divergencePoint==0):
            br = br-1
            x = x-step
            sol[br] = (2.*(1.-5.0/12.0*step**2*kfun(x))*sol[br+1]-\
                       (1.+1./12.0*step**2*kfun(x+step))*sol[br+2])/\
                       (1.+1./12.0*step**2*kfun(x-step))
            rad[br] = x
            if (divergencePoint==0)and(abs(sol[br]*sqrt(x))>maxValue):
                divergencePoint = br
                while( abs(sol[divergencePoint])>abs(sol[divergencePoint+1])) and \
                    (divergencePoint<checkPoint):
                    divergencePoint +=1
                if divergencePoint>checkPoint:
                    print("Numerov error")
                    exit()

        br = divergencePoint;
        while (br>0):
            rad[br]=rad[br+1]-step;
            sol[br]=0;
            br -= 1;

        # convert R(r)*r^{3/4} to  R(r)*r
        sol = np.multiply(sol,np.sqrt(rad))
        # convert \sqrt(r) to r
        rad = np.multiply(rad,rad)

        return rad,sol
