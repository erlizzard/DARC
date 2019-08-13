
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
from .alkali_atom_functions import AlkaliAtom
from scipy.constants import physical_constants, pi , epsilon_0, hbar
from scipy.constants import k as C_k
from scipy.constants import c as C_c
from scipy.constants import h as C_h
from scipy.constants import e as C_e
from scipy.constants import m_e as C_m_e
from scipy.optimize import curve_fit
from scipy.special import sph_harm

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
from scipy.special import factorial


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
import sqlite3

sqlite3.register_adapter(np.float64, float)
sqlite3.register_adapter(np.float32, float)
sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.int32, int)

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

class EarthAlkaliAtom(AlkaliAtom):

    #these are the funtions we want to implement
    #we need to initalise all the values

    #read in the energies and find a way to store them

    #does this change with the atom?
    alpha = physical_constants["fine-structure constant"][0]

    modelPotential_coef = dict()
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
    
    preferMeasuredEnergies = False
    semi = True
    
    # SQLite connection and cursor
    conn = False
    c = False

    def __init__(self,preferQuantumDefects=True,cpp_numerov=False, uncertainty=False):
        #nikolai does lots with loading and saving stuff to and from a database (ignore for now)
        print(DPATH)
        self._databaseInit()

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

        data=[]
        if (self.dipoleMatrixElementFile != ""):
            if (preferQuantumDefects == False):
                self.dipoleMatrixElementFile  = "NIST_"+self.dipoleMatrixElementFile

            try:
                data = np.load(os.path.join(self.dataFolder,\
                                            self.dipoleMatrixElementFile),\
                               encoding = 'latin1')
            except IOError as e:
                print("Error reading dipoleMatrixElement File "+\
                    os.path.join(self.dataFolder,self.dipoleMatrixElementFile))
                print(e)
        # save to SQLite database
        try:
            self.c.execute('''SELECT COUNT(*) FROM sqlite_master
                            WHERE type='table' AND name='dipoleME';''')
            if (self.c.fetchone()[0] == 0):
                # create table
                self.c.execute('''CREATE TABLE IF NOT EXISTS dipoleME
                 (n1 TINYINT UNSIGNED, l1 TINYINT UNSIGNED, j1_x2 TINYINT UNSIGNED,
                 n2 TINYINT UNSIGNED, l2 TINYINT UNSIGNED, j2_x2 TINYINT UNSIGNED,
                 dme DOUBLE, s TINYINT UNSIGNED, semi TINYINT UNSIGNED,
                 PRIMARY KEY (n1,l1,j1_x2,n2,l2,j2_x2,s,semi)
                ) ''')
                if (len(data)>0):
                    self.c.executemany('INSERT INTO dipoleME VALUES (?,?,?,?,?,?,?,?,?)', data)
                self.conn.commit()
        except sqlite3.Error as e:
            print("Error while loading precalculated values into the database")
            print(e)
            exit()

        # load quadrupole matrix elements previously calculated
        data=[]
        if (self.quadrupoleMatrixElementFile != ""):
            if (preferQuantumDefects == False):
                self.quadrupoleMatrixElementFile  = "NIST_"+self.quadrupoleMatrixElementFile
            try:
                data = np.load(os.path.join(self.dataFolder,\
                                            self.quadrupoleMatrixElementFile),\
                               encoding = 'latin1')

            except IOError as e:
                print("Error reading quadrupoleMatrixElementFile File "+\
                    os.path.join(self.dataFolder,self.quadrupoleMatrixElementFile))
                print(e)
        # save to SQLite database
        try:
            self.c.execute('''SELECT COUNT(*) FROM sqlite_master
                            WHERE type='table' AND name='quadrupoleME';''')
            if (self.c.fetchone()[0] == 0):
                # create table
                self.c.execute('''CREATE TABLE IF NOT EXISTS quadrupoleME
                 (n1 TINYINT UNSIGNED, l1 TINYINT UNSIGNED, j1_x2 TINYINT UNSIGNED,
                 n2 TINYINT UNSIGNED, l2 TINYINT UNSIGNED, j2_x2 TINYINT UNSIGNED,
                 qme DOUBLE, s TINYINT UNSIGNED,
                 PRIMARY KEY (n1,l1,j1_x2,n2,l2,j2_x2,s)
                ) ''')
                if (len(data)>0):
                    self.c.executemany('INSERT INTO quadrupoleME VALUES (?,?,?,?,?,?,?,?)', data)
                self.conn.commit()
        except sqlite3.Error as e:
            print("Error while loading precalculated values into the database")
            print(e)
            exit()

        self._readLiteratureValues()

        return

    def _databaseInit(self):
        print(self.dataFolder)
        print(self.precalculatedDB)
        self.conn = sqlite3.connect(os.path.join(self.dataFolder,\
                                                 self.precalculatedDB))
        self.c = self.conn.cursor()
        
    def _parseLevelsFromNIST(self,fileData,levels):
        """
            Parses the level energies from CSV file we have of the energy values

            args:
                fileData : string - filename
                levels : int - number of levels
            returns:
                array (levels, 3) - array to be passed to the sEnergy dictionary.
            """
            
        full_filepath = os.path.join(self.dataFolder,fileData)
        
        f = open(full_filepath,"r")

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
        full_filepath = os.path.join(self.dataFolder,fileName)
        
        f = open(full_filepath, 'r')
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
        
        #/8065.544 comes from cm-1 to eV
        if label in self.extraLevels:
            for n in range(self.extraLevels[label],energyNIST.shape[0]):
                if uncert == True:
                    energyNIST[n,0] = (energyNIST[n,0] +err[n] -self.ionisationEnergycm)/8065.544#/219475#*4.55633e-6#  - self.ionisationEnergy #turned to to hartrees# - self.ionisationEnergy
                else:
                    energyNIST[n,0] = (energyNIST[n,0] -self.ionisationEnergycm)/8065.544#219475#*4.55633e-6#  - self.ionisationEnergy #turned to to hartrees# - self.ionisationEnergy

            self.sEnergy[label] = energyNIST[:-1]
            self.refs[label] = ref[:-1]
            self.errs[label] = err[:-1]
        else:
            for n in range(self.groundStateN, energyNIST.shape[0]):
                if uncert == True:
                    energyNIST[n,0] = (energyNIST[n,0] +err[n] -self.ionisationEnergycm)/8065.544#219475#*4.55633e-6#  - self.ionisationEnergy #turned to to hartrees# - self.ionisationEnergy

                else:
                    energyNIST[n,0] = (energyNIST[n,0] -self.ionisationEnergycm)/8065.544# 219475#*4.55633e-6#ionisationEnergy
                #print(energyNIST[n,0])
            self.sEnergy[label] = energyNIST
            self.refs[label] = ref
            self.errs[label] = err

    def getEnergy(self,n,L,J,S):
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
       
        #energy_arr = self.sEnergy[term]

       


        #If we havent prefered quantum defect but have chosen measured energy levels OR we are below defect fitting range
        if ( L < 4 and (self.preferMeasuredEnergies or n<self.minQuantumDefectN[term] ) and 
            n <= self.NISTdataLevels[term]+5 and self.sEnergy[term][n] <= self.ionisationEnergycm ):
            print(term)
            return self.sEnergy[term][n]

        else:
            #if(n>self.maxQuantumDefectN[term]):
                #LIZZY SORT OUT THIS WARNING PROPERLY
            #    print("CAUTION: The Rydberg-Ritz values have been extrapolated beyond their fitted values")
            defect = self.getQuantumDefect(n, L,J,S)
            return - self.scaledRydbergConstant/8065.544 /((n-defect)**2)# self.scaledRydbergConstant//219475


    def getTerm(self,L,S,J):
        '''This returns the correct key for the given values and checks that the values inputted are valid L,S and J
            Args:
                L: total orbital angular quantum numbers
                S: total spin number
                J: total angular quantum number
        '''
        if(S != 0 and S !=1):
            print('That is not a valid total spin number (S)')
        elif(L< 0):
            print('That is not a vaild orbital angular quantum number (L)')
        elif(J < ( L-1) or J > (L+1)):
            print('That is not a vaild total angular quantum number (J)')
        else:
            if L == 0:
                return str(S*2+1)+'S'+str(int(J))
            elif L == 1:
                return str(S*2+1)+'P'+str(int(J))
            elif L == 2:
                return str(S*2+1)+'D'+str(int(J))
            elif L == 3:
                return str(S*2+1)+'F'+str(int(J))
            else:
                #This is used for L>3, for which we have no measured values and qd's are 0
                return str(S*2+1)+'X'+str(int(J))

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

    def getQuantumDefect(self, n, L,J,S):
        """
            Quantum defect of the level.

            For an example, see `Rydberg energy levels example snippet`_.

            .. _`Rydberg energy levels example snippet`:
                ./Rydberg_atoms_a_primer.html#Rydberg-Atom-Energy-Levels


            Returns:
                float: quantum defect
        """
        
        if(L>3):
            defect = 0
        else:
            term = self.getTerm(L,S,J)
            out = np.where(self.minQuantumDefectN[term]>n, )
            if(self.minQuantumDefectN[term]>n):
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
    
    def potential(self,l,s,j,r):
        """ returns total potential that electron feels

            Total potential = core potential + Spin-Orbit interaction

            Args:
                l (int): orbital angular momentum
                s (float): spin angular momentum
                j (float): total angular momentum
                r (float): distance from the nucleus (in a.u.)
            Returns:
                float: potential (in a.u.)
        """
        #we have no spin orbit coupling and hence no extra term
        term = self.getTerm(l,s,j)
        if (l>3): 
            a,b,c = 0,0,0
        else:
            a,b,c = self.modelPotential_coef[term]
        
        return -1/r*(1.0+(self.Z-1)*exp(-a*r)) +b*exp(-c*r)
        
    

    #def getRadialCoupling(self,n,l,j,n1,l1,j1,s,semi):
    #    """
    #        Returns radial part of the coupling between two states (dipole and
    #        quadrupole interactions only)#
#            Args:
#                n1 (int): principal quantum number
#                l1 (int): orbital angular momentum
#                j1 (float): total angular momentum
#                n2 (int): principal quantum number
#                l2 (int): orbital angular momentum
#                j2 (float): total angular momentum

#            Returns:
#                float:  radial coupling strength (in a.u.), or zero for forbidden
#                transitions in dipole and quadrupole approximation.
#
 #       """
        ##LIZZY Does this condition still apply?
  #      dl = abs(l-l1)
   #     if (dl == 1 and abs(j-j1)<1.1):
    #        #print(n," ",l," ",j," ",n1," ",l1," ",j1)
     #       return self.getRadialMatrixElement(n,l,s,j,n1,l1,s,j1)
      #  elif (dl==0 or dl==1 or dl==2) and(abs(j-j1)<2.1):
            # quadrupole coupling
       #     return 0.
            #LIZZY: need to impelment
            #return self.getQuadrupoleMatrixElement(n,l,j,n1,l1,j1)
       # else:
            # neglect octopole coupling and higher
            #print("NOTE: Neglecting couplings higher then quadrupole")
        #    return 0

    def getAngularMatrixElement(self, l, s, j, m, l1, s1, j1, m1,p):
        ''' 
        Angular matrix element.
        
        ***INSERT CHRISTOPHES CITATION
        '''
                         #sum over all polariasations
        sign = (-1)**(s-m-l-j)
        elem = sign * sqrt((2*j+1)*(2*l+1))
        elem = elem * CG(l,0,1,0,l1,0) * CG(j,m,1,p,j1,m1)
        elem = elem * Wigner6j(j, 1, j1, l1, s, l)
       
        return elem


    #def getRadialMatrixElement(self,n,l,s,j,n1,l1,s1,j1):
   #    
   #    '''This has been using the semiClassical method, this will differ from the inherited Alkali Atom method '''
   #    #get the effective principal number of both states
   #    nu = n - self.getQuantumDefect(n,l,j,s)
   #    nu1 = n1 - self.getQuantumDefect(n1,l1,j1,s1)#

        #get the parameters required to calculate the sum
   #     l_c = (l+l1+1.)/2.
   #     nu_c = sqrt(nu*nu1)

#        delta_nu = nu- nu1
 #       delta_l = l1 -l

 #       gamma  = (delta_l*l_c)/nu_c

 #       g0 = (1./(3.*delta_nu))*(mpmath.angerj(delta_nu-1.,-delta_nu) - mpmath.angerj(delta_nu+1,-delta_nu))
 #       g1 = -(1./(3.*delta_nu))*(mpmath.angerj(delta_nu-1.,-delta_nu) + mpmath.angerj(delta_nu+1,-delta_nu))
 #       g2 = g0 - mpmath.sin(pi*delta_nu)/(pi*delta_nu)
 #       g3 = (delta_nu/2.)*g0 + g1

  #      radial_ME = (3/2)*nu_c**2*(1-(l_c/nu_c)**(2))**0.5*(g0 + gamma*g1 + gamma**2*g2 + gamma**3*g3)

   #     return radial_ME
    
    
    def getDipoleMatrixElement(self,n,l,j,m,n1,l1,j1,m1,p,s):
        """
        Dipole matrix element
            :math:`\\langle n_1 l_1 j_1 m_{j_1} |e\\mathbf{r}|n_2 l_2 j_2 m_{j_2}\\rangle`
            in units of :math:`a_0 e`

            Returns:
                float: dipole matrix element( :math:`a_0 e`)
        NOTE: The angular matrix element used here is for a singular EarthAlkali atom, For the angular matrix element of 
        two intracting strontium atoms, use the angular matrix element provied in Pairstate interactions. 

        """
        if self.semi == True:  
            return self.getRadialMatrixElementSemiClassical(n,l,j,n1,l1,j1,s)*self.getAngularMatrixElement(l,s,j ,m,l1,s,j1,m1,p)
        else:
            return self.getRadialMatrixElement(n,l,j,n1,l1,j1,s)*self.getAngularMatrixElement(l,s,j ,m,l1,s,j1,m1,p)

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


    def getAngularMatrixElementSrSr(self,l,j,m, l1, j1, m1, ll, jj, mm, ll1,jj1, mm1,s,c1,c2):
        elem = (-1)**(j+j1+l+l1+s*2)
        #elem *= sqrt((4*pi*4*3*2*(2*l+1)*(2*l1+1))/(2*2*5))

        elem *= sqrt((4*pi*factorial(2*c1+2*c2)*(2*l+1)*(2*l1+1))\
                                    /(factorial(2*c1)*factorial(2*c2)*(2*c1+2*c2+1)))
        elem *= sqrt((2*j+1)*(2*j1+1))*CG(l,0,c1,0,ll,0)*CG(l1,0,c2,0,ll1,0)

        #print('cg',CG(l,0,1,0,ll,0),CG(l1,0,1,0,ll1,0))
        elem *= Wigner6j(j,c1,jj,ll,s,l)*Wigner6j(j1,c2,jj1,ll1,s,l1)

        def getSpherical(p,p1,p2):
           # print('p,c1+c2', p, c1+c2)
            spher = sph_harm(p, c1+c2, 0, 0)
            #print('Spherical harmmonic', sph_harm(p, c1+c2, 0, 0))
            spher *= CG( c1, p1, c2, p2,c1+c2,p)
            #print('clebsch',CG( c1, p1, c2, p2,c1+c2,p))
            spher *= CG(j,m,c2,p1,jj,mm)
            spher *= CG(j1, m1, c2, p2,jj1,mm1)
            return spher

        out = [getSpherical(p,p1,p2) for p in np.arange(-2,3,1) \
                                     for p1 in np.arange(-1,2,1) \
                                     for p2 in np.arange(-1,2,1)]
        return -1*elem*np.sum(np.array(out))
    #def getEnergyDefect2(self, n,l,s,j,nn,ll,ss,jj,n1,l1,s1,j1,nn1,ll1,ss1,jj1):
    #    return (self.getEnergy(n1,l1,s1,j1)+\
    #                                 self.getEnergy(nn1,ll1,ss1,jj1)-\
    #                                 self.getEnergy(n,l,s,j)- self.getEnergy(nn,ll,ss,jj))
