# -*- coding: utf-8 -*-

"""
    Pair-state basis level diagram calculations

    Calculates Rydberg spaghetti of level diagrams, as well as pertubative C6
    and similar properties. It also allows calculation of Foster resonances
    tuned by DC electric fields.

    Example:
        Calculation of the Rydberg eigenstates in pair-state basis for Rubidium
        in the vicinity of the
        :math:`|60~S_{1/2}~m_j=1/2,~60~S_{1/2}~m_j=1/2\\rangle` state. Colour
        highlights coupling strength from state :math:`6~P_{1/2}~m_j=1/2` with
        :math:`\\pi` (:math:`q=0`) polarized light.
        eigenstates::

            from arc import *
            calc1 = PairStateInteractions(Rubidium(), 60, 0, 0.5, 60, 0, 0.5,0.5, 0.5)
            calc1.defineBasis( 0., 0., 4, 5,10e9)
            # optionally we can save now results of calculation for future use
            saveCalculation(calc1,"mycalculation.pkl")
            calculation1.diagonalise(linspace(1,10.0,30),250,progressOutput = True,drivingFromState=[6,1,0.5,0.5,0])
            calc1.plotLevelDiagram()
            calc1.ax.set_xlim(1,10)
            calc1.ax.set_ylim(-2,2)
            calc1.showPlot()

"""

from __future__ import division, print_function, absolute_import

from math import exp,log,sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['font.family'] = 'serif'

import numpy as np
import re
from scipy.constants import physical_constants, pi , epsilon_0, hbar
from scipy.constants import k as C_k
from scipy.constants import c as C_c
from scipy.constants import h as C_h
from scipy.constants import e as C_e
from scipy.optimize import curve_fit

# for matrices
from numpy import zeros,savetxt, complex64,complex128
from numpy.linalg import eigvalsh,eig,eigh
from numpy.ma import conjugate
from numpy.lib.polynomial import real
from scipy.sparse import lil_matrix,csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.special.specfun import fcoef
from scipy import floor
from scipy.special import factorial

from .wigner import Wigner6j,Wigner3j,CG,wignerDmatrix
from .alkali_atom_functions import *
from .alkali_atom_functions import _EFieldCoupling,_atomLightAtomCoupling
from .calculations_atom_single import StarkMap
from .earth_alkali_atom_functions import EarthAlkaliAtom
import itertools

from matplotlib.colors import LinearSegmentedColormap
import matplotlib

import datetime

import sys
if sys.version_info > (2,):
    xrange = range

import gzip

DPATH = os.path.join(os.path.expanduser('~'), '.arc-data')

class PairStateInteractions:
    """
        Calculates Rydberg level diagram (spaghetti) for the given pair state

        Initializes Rydberg level spaghetti calculation for the given atom in
        the vicinity of the given pair state. For details of calculation see
        Ref. [1]_. For a quick start point example see
        `interactions example snippet`_.

        .. _`interactions example snippet`:
            ./Rydberg_atoms_a_primer.html#Short-range-interactions

        Args:
            atom (:obj:`AlkaliAtom`): ={ :obj:`alkali_atom_data.Lithium6`,
                :obj:`alkali_atom_data.Lithium7`,
                :obj:`alkali_atom_data.Sodium`,
                :obj:`alkali_atom_data.Potassium39`,
                :obj:`alkali_atom_data.Potassium40`,
                :obj:`alkali_atom_data.Potassium41`,
                :obj:`alkali_atom_data.Rubidium85`,
                :obj:`alkali_atom_data.Rubidium87`,
                :obj:`alkali_atom_data.Caesium` }
                Select the alkali metal for energy level
                diagram calculation
            n (int): principal quantum number for the *first* atom
            l (int): orbital angular momentum for the *first* atom
            j (float): total angular momentum for the *first* atom
            nn (int): principal quantum number for the *second* atom
            ll (int): orbital angular momentum for the *second* atom
            jj (float): total angular momentum for the *second* atom
            m1 (float): projection of the total angular momentum on z-axis
                for the *first* atom
            m2 (float): projection of the total angular momentum on z-axis
                for the *second* atom
            interactionsUpTo (int): Optional. If set to 1, includes only
                dipole-dipole interactions. If set to 2 includes interactions
                up to quadrupole-quadrupole. Default value is 1.

        References:
            .. [1] T. G Walker, M. Saffman, PRA **77**, 032723 (2008)
                https://doi.org/10.1103/PhysRevA.77.032723

        Examples:
            **Advanced interfacing of pair-state interactions calculations
            (PairStateInteractions class).** This
            is an advanced example intended for building up extensions to the
            existing code. If you want to directly access the pair-state
            interaction matrix, constructed by :obj:`defineBasis`,
            you can assemble it easily from diagonal part
            (stored in :obj:`matDiagonal` ) and off-diagonal matrices whose
            spatial dependence is :math:`R^{-3},R^{-4},R^{-5}` stored in that
            order in :obj:`matR`. Basis states are stored in :obj:`basisStates` array.

            >>> from arc import *
            >>> calc = PairStateInteractions(Rubidium(), 60,0,0.5, 60,0,0.5, 0.5,0.5,interactionsUpTo = 1)
            >>> # theta=0, phi = 0, range of pqn, range of l, deltaE = 25e9
            >>> calc.defineBasis(0 ,0 , 5, 5, 25e9, progressOutput=True)
            >>> # now calc stores interaction matrix and relevant basis
            >>> # we can access this directly and generate interaction matrix
            >>> # at distance rval :
            >>> rval = 4  # in mum
            >>> matrix = calc.matDiagonal
            >>> rX = (rval*1.e-6)**3
            >>> for matRX in self.matR:
            >>>     matrix = matrix + matRX/rX
            >>>     rX *= (rval*1.e-6)
            >>> # matrix variable now holds full interaction matrix for
            >>> # interacting atoms at distance rval calculated in
            >>> # pair-state basis states can be accessed as
            >>> basisStates = calc.basisStates
    """

    dataFolder = DPATH

    # =============================== Methods ===============================

    def __init__(self,atom,n,l,j,nn,ll,jj,m1,m2,s=0.5,interactionsUpTo=1,m1f = 0, m2f = 0):
        # alkali atom type, principal quantum number, orbital angular momentum,
        #  total angular momentum projections of the angular momentum on z axis
        self.atom = atom  #: atom type
        self.n = n #: pair-state definition: principal quantum number of the first atom
        self.l = l #: pair-state definition: orbital angular momentum of the first atom
        self.j = j #: pair-state definition: total angular momentum of the first atom
        self.nn = nn #: pair-state definition: principal quantum number of the second atom
        self.ll = ll #: pair-state definition: orbital angular momentum of the second atom
        self.jj = jj #: pair-state definition: total angular momentum of the second atom
        self.m1 = m1 #: pair-state definition: projection of the total ang. momentum for the *first* atom
        self.m2 = m2 #: pair-state definition: projection of the total angular momentum for the *second* atom
        self.s = s #: spin of both atoms
        self.interactionsUpTo = interactionsUpTo
        self.m1f = m1f
        self.m2f = m2f
        
        """"
            Specifies up to which approximation we include in pair-state interactions.
            By default value is 1, corresponding to pair-state interactions up to
            dipole-dipole coupling. Value of 2 is also supported, corresponding
            to pair-state interactions up to quadrupole-quadrupole coupling.
        """

        # ====================== J basis (not resolving mj) ======================

        self.coupling = []
        """
            List of matrices defineing coupling strengths between the states in J basis (not
            resolving :math:`m_j` ). Basis is given by :obj:`channel`. Used as
            intermediary for full interaction matrix calculation by
            :obj:`defineBasis`.
        """
        self.channel = []
        """
            states relevant for calculation, defined in J basis (not resolving
            :math:`m_j`. Used as intermediary for full interaction matrix
            calculation by :obj:`defineBasis`.
        """

        # ======================= Full basis (resolving mj) =======================

        self.basisStates = []
        """
            List of pair-states for calculation. In the form
            [[n1,l1,j1,mj1,n2,l2,j2,mj2], ...].
            Each state is an array [n1,l1,j1,mj1,n2,l2,j2,mj2] corresponding to
            :math:`|n_1,l_1,j_1,m_{j1},n_2,l_2,j_2,m_{j2}\\rangle` state.
            Calculated by :obj:`defineBasis`.
        """
        self.matrixElement = []
        """
            `matrixElement[i]` gives index of state in :obj:`channel` basis (that
            doesn't resolve :obj:`m_j` states), for the given index `i` of the
            state in :obj:`basisStates`  ( :math:`m_j` resolving) basis.
        """

        # variuos parts of interaction matrix in pair-state basis
        self.matDiagonal = []
        """
            Part of interaction matrix in pair-state basis that doesn't depend
            on inter-atomic distance. E.g. diagonal elements of the interaction
            matrix, that describe energies of the pair states in unperturbed basis,
            will be stored here. Basis states are stored in :obj:`basisStates`.
            Calculated by :obj:`defineBasis`.
        """
        self.matR = []
        """
            Stores interaction matrices in pair-state basis
            that scale as :math:`1/R^3`, :math:`1/R^4` and :math:`1/R^5`
            with distance in  :obj:`matR[0]`, :obj:`matR[1]` and :obj:`matR[2]`
            respectively. These matrices correspond to dipole-dipole
            ( :math:`C_3`), dipole-quadrupole ( :math:`C_4`) and
            quadrupole-quadrupole ( :math:`C_5`) interactions
            coefficients. Basis states are stored in :obj:`basisStates`.
            Calculated by :obj:`defineBasis`.
        """
        self.originalPairStateIndex = 0
        """
            index of the original n,l,j,m1,nn,ll,jj,m2 pair-state in the
            :obj:`basisStates` basis.
        """

        self.matE = []
        self.matB_1 = []
        self.matB_2 = []

        # ====================== Eigen states and plotting ======================

        # finding perturbed energy levels
        self.r = []    # detuning scale
        self.y = []    # energy levels
        self.highlight = []

        # pointers towards figure
        self.fig = 0
        self.ax = 0

        # for normalization of the maximum coupling later
        self.maxCoupling = 0.

        self.drivingFromState = [0,0,0,0,0]  # n,l,j,mj, drive polarization q

        #sam = saved angular matrix metadata
        self.angularMatrixFile = "angularMatrix.npy"
        self.angularMatrixFile_meta = "angularMatrix_meta.npy"
        #self.sam = []
        self.savedAngularMatrix_matrix = []

        # intialize precalculated values for factorial term in __getAngularMatrix_M
        fcoef = lambda l1,l2,m: factorial(l1+l2)/(factorial(l1+m)*factorial(l1-m)*factorial(l2+m)*factorial(l2-m))**0.5
        x = self.interactionsUpTo
        self.fcp = np.zeros((x+1,x+1,2*x+1))
        for c1 in range(1,x+1):
            for c2 in range(1,x+1):
                for p in range(-min(c1,c2),min(c1,c2)+1):
                    self.fcp[c1,c2,p+x] = fcoef(c1,c2,p)

        self.conn = False
        self.c = False


    def _getAngularMatrixElementAlkali(self,l,j,m,ll,jj,mm,l1,j1,m1,l2,j2,m2, c1, c2):
        #This will get the corret matrix elements for the Alkali atoms
        #This is just a level of abstraction for what is already written in the
        #current function, and so we can write one for strontium.

        # angular matrix element from Sa??mannshausen, Heiner, Merkt, Fr??d??ric, Deiglmayr, Johannes
        # PRA 92: 032505 (2015)

        #Written by Nikola
        elem = (-1.0)**(j+jj+1.0+l1+l2)*CG(l,0,c1,0,l1,0)*CG(ll,0,c2,0,l2,0)
        elem = elem*sqrt((2.0*l+1.0)*(2.0*ll+1.0))*sqrt((2.0*j+1.0)*(2.0*jj+1.0))
        elem = elem*Wigner6j(l, 0.5, j, j1, c1, l1)*Wigner6j(ll,0.5,jj,j2,c2,l2)

        sumPol = 0.0  # sum over polarisations
        limit = min(c1,c2)
        for p in xrange(-limit,limit+1):
            sumPol = sumPol + \
                     self.fcp[c1,c2,p + self.interactionsUpTo] * \
                     CG(j,m,c1,p,j1,m1) *\
                     CG(jj,mm,c2,-p,j2,m2)
        return elem*sumPol

    def _getAngularMatrixElementEarthAlkali(self,l,j,m,ll,jj,mm,l1,j1,m1,l2,j2,m2, c1, c2, s):
        return self.atom.getAngularMatrixElementSrSr(l,j,m,ll,jj,mm,l1,j1,m1,l2,j2,m2,c1,c2,s)

    def __getAngularMatrix_M(self,l,j,ll,jj,l1,j1,l2,j2,atom):
        # did we already calculated this matrix?
        self.c.execute('''SELECT ind FROM pair_angularMatrix WHERE
             l1 = ? AND j1_x2 = ? AND
             l2 = ? AND j2_x2 = ? AND

             l3 = ? AND j3_x2 = ? AND
             l4 = ? AND j4_x2 = ? AND 
             s  = ?
             ''',(l,j*2,ll,jj*2,l1,j1*2,l2,j2*2, self.s*2))

        index = self.c.fetchone()
        if (index):
            return np.array(self.savedAngularMatrix_matrix[index[0]])

        # determine coupling
        dl = abs(l-l1)
        dj = abs(j-j1)
        c1 = 0
        if dl==1 and (dj<1.1):
            c1 = 1  # dipole coupling
        elif (dl==0 or dl==2 or dl==1):
            c1 = 2  # quadrupole coupling
        else:
            raise ValueError("error in __getAngularMatrix_M")
            exit()
        dl = abs(ll-l2)
        dj = abs(jj-j2)
        c2 = 0
        if dl==1 and (dj<1.1):
            c2 = 1  # dipole coupling
        elif (dl==0 or dl==2 or dl==1):
            c2 = 2  # quadrupole coupling
        else:
            raise ValueError("error in __getAngularMatrix_M")
            exit()


        am = zeros((int(round((2*j1+1)*(2*j2+1),0)),\
                    int(round((2*j+1)*(2*jj+1),0))),dtype=np.float64)


        j1range = np.linspace(-j1,j1,round(2*j1)+1)
        j2range = np.linspace(-j2,j2,round(2*j2)+1)
        jrange = np.linspace(-j,j,int(2*j)+1)
        jjrange = np.linspace(-jj,jj,int(2*jj)+1)

        for m1 in j1range:
            for m2 in j2range:
                # we have chosen the first index
                index1 = int(round(m1*(2.0*j2+1.0)+m2+(j1*(2.0*j2+1.0)+j2),0))
                for m in jrange:
                    for mm in jjrange:

                        # we have chosen the second index
                        index2 = int(round(m*(2.0*jj+1.0)+mm+(j*(2.0*jj+1.0)+jj),0))
                        if issubclass(type(self.atom), EarthAlkaliAtom):
                            am[index1,index2] = real(self._getAngularMatrixElementEarthAlkali(l1,j1,m1,l2,j2,m2,l,j,m,ll,jj,mm,self.s ,c1, c2))
                        else:
                            am[index1,index2] = self._getAngularMatrixElementAlkali(l,j,m,ll,jj,mm,l1,j1,m1,l2,j2,m2, c1, c2)


        index = len(self.savedAngularMatrix_matrix)
        self.c.execute(''' INSERT INTO pair_angularMatrix
                            VALUES (?,?, ?,?, ?,?, ?,?, ?,?)''',\
                       (l,j*2,ll,jj*2,l1,j1*2,l2,j2*2, index, self.s*2) )
        self.conn.commit()
        self.savedAngularMatrix_matrix.append(am)

        self.savedAngularMatrixChanged = True
        return am

    def __updateAngularMatrixElementsFile(self):
        if not (self.savedAngularMatrixChanged):
            return

        try:
            self.c.execute('''SELECT * FROM pair_angularMatrix ''')
            data = []
            for v in self.c.fetchall():
                data.append(v)

            data = np.array(data,dtype = np.float32)

            data[:,1] /= 2.  # 2 r j1 -> j1
            data[:,3] /= 2.  # 2 r j2 -> j2
            data[:,5] /= 2.  # 2 r j3 -> j3
            data[:,7] /= 2.  # 2 r j4 -> j4
            data[:,9] /= 2.

            fileHandle = gzip.GzipFile(os.path.join(self.dataFolder,\
                                                    self.angularMatrixFile_meta), 'wb')
            np.save(fileHandle,data)
            fileHandle.close()
        except IOError as e:
            print("Error while updating angularMatrix \
                data meta (description) File "+self.angularMatrixFile_meta)

        try:
            fileHandle = gzip.GzipFile(os.path.join(self.dataFolder,\
                                                    self.angularMatrixFile),'wb')
            np.save(fileHandle,self.savedAngularMatrix_matrix)
            fileHandle.close()
        except IOError as e:
            print("Error while updating angularMatrix \
                    data File "+self.angularMatrixFile)
            print(e)

    def __loadAngularMatrixElementsFile(self):
        try:
            fileHandle =  gzip.GzipFile(os.path.join(self.dataFolder,\
                                                     self.angularMatrixFile_meta),'rb')
            data = np.load(fileHandle, encoding = 'latin1')
            fileHandle.close()
        except :
            print("Note: No saved angular matrix metadata files to be loaded.")
            print(sys.exc_info())
            return

        data[:,1] *= 2  # j1 -> 2 r j1
        data[:,3] *= 2  # j2 -> 2 r j2
        data[:,5] *= 2  # j3 -> 2 r j3
        data[:,7] *= 2  # j4 -> 2 r j4
        data[:,9] *= 2  
        data = np.array(np.rint(data),dtype=np.int)
        try:

            self.c.executemany('''INSERT INTO pair_angularMatrix
                (l1, j1_x2 ,
                 l2 , j2_x2 ,
                 l3, j3_x2,
                 l4 , j4_x2,
                 ind,s)
                      VALUES (?,?,?,?,?,?,?,?,?,?)''', data)

            self.conn.commit()

        except sqlite3.Error as e:
            print("Error while loading precalculated values into the database!")
            print(e)
            exit()
        if len(data)==0:
            print("error")
            return

        try:
            fileHandle =  gzip.GzipFile(os.path.join(self.dataFolder,\
                                                    self.angularMatrixFile),'rb')
            self.savedAngularMatrix_matrix = np.load(fileHandle,\
                                                     encoding = 'latin1').tolist()
            fileHandle.close()
        except :
            print("Note: No saved angular matrix files to be loaded.")
            print(sys.exc_info())


    def __isCoupled(self,n,l,j,nn,ll,jj,n1,l1,j1,n2,l2,j2,limit,s = 0.5):
        #LIZZY: what is the point of this, surely we know the coupling by
        # allowing the user to input the interactionUpTo?s
        if (abs(self.atom.getEnergyDefect2(n,l,j,nn,ll,jj,n1,l1,j1,n2,l2,j2,s))/C_h<limit) and\
                not (n==n1 and nn==n2 and l==l1 and ll==l2 and j==j1 and jj==j2) \
                and not ((abs(l1-l)!=1 and abs(j-s)<0.1 and abs(j1-s)<0.1) or
                         (abs(l2-ll)!=1 and abs(jj-s)<0.1 and abs(j2-s)<0.1)):
                            # determine coupling
                dl = abs(l-l1)
                dj = abs(j-j1)
                c1 = 0
                if dl==1 and (dj<1.1):
                    c1 = 1  # dipole coupling
                elif (dl==0 or dl==2 or dl==1)and (dj<2.1) and \
                    (2 <= self.interactionsUpTo):
                    c1 = 2  # quadrupole coupling
                else:
                    return False
                dl = abs(ll-l2)
                dj = abs(jj-j2)
                c2 = 0
                if dl==1 and (dj<1.1):
                    c2 = 1  # dipole coupling
                elif (dl==0 or dl==2 or dl==1) and (dj<2.1) and \
                    (2 <= self.interactionsUpTo):
                    c2 = 2  # quadrupole coupling
                else:
                    return False
                return c1+c2
        else:
            return False

    def __makeRawMatrix2(self,n,l,j,nn,ll,jj,k,lrange,limit,limitBasisToMj,\
                         progressOutput = False,debugOutput=False):
        # limit = limit in Hz on energy defect
        # k defines range of n' = [n-k, n+k]
        
        dimension = 0

        # which states/channels contribute significantly in the second order perturbation?
        states = []

        # original pairstate index
        opi = 0

        # this numbers are conserved if we use only dipole-dipole interactions
        Lmod2 = ((l+ll) % 2)

        l1start = l-1
        if l == 0: l1start=0

        l2start = ll-1
        if ll==0: l2start=0

        if debugOutput:
            print("\n ======= Relevant states =======\n")

        for n1 in xrange(max(n-k,1),n+k+1):
            for n2 in xrange(max(nn-k,1),nn+k+1):
                l1max = max(l+self.interactionsUpTo,lrange)+1
                l1max = min(l1max,n1-1)
                for l1 in xrange(l1start,l1max):
                    l2max = max(ll+self.interactionsUpTo,lrange)+1
                    l2max = min(l2max,n2-1)
                    for l2 in xrange(l2start,l2max):
                        j1 = l1-self.s
                        if l1 == 0:
                            j1 = self.s
                        while j1 <= l1+self.s+0.1:
                            j2 = l2-self.s
                            if l2 == 0:
                                j2 = self.s

                            while j2 <= l2+self.s+0.1:
                                ed = self.atom.getEnergyDefect2(n,l,j,\
                                                               nn,ll,jj,\
                                                               n1,l1,j1,\
                                                               n2,l2,j2,self.s)/C_h
                                if  (abs(ed)<limit  \
                                    and (not (self.interactionsUpTo==1) or\
                                         (Lmod2 == ((l1+l2)%2) ) )
                                    and ((not limitBasisToMj) or \
                                         (j1+j2+0.1>self.m1+self.m2) )    ):

                                    if debugOutput:
                                        pairState = "|"+printStateString(n1,l1,j1,self.s)+\
                                                ","+printStateString(n2,l2,j2,self.s)+">"
                                        print(pairState+("\t EnergyDefect = %.3f GHz" % (ed*1.e-9)))

                                    #LIZZY: this has been edited so that the 7th element is the spin for both
                                    states.append([n1,l1,j1,n2,l2,j2,self.s])


                                    if (n==n1 and nn==n2 and l==l1 and\
                                         ll==l2 and j==j1 and jj==j2):
                                        opi = dimension

                                    dimension = dimension +1
                                j2 = j2+1.0
                            j1 = j1+1.0

        if debugOutput:
            print("\tMatrix dimension\t=\t",dimension)
        m = np.zeros((dimension,dimension),dtype=np.float64)

        # mat_value, mat_row, mat_column for each sparce matrix describing
        # dipole-dipole, dipole-quadrupole (and quad-dipole) and quadrupole-quadrupole
        couplingMatConstructor = [ [[],[],[]] \
                                  for i in xrange(2*self.interactionsUpTo-1) ]

        # original pair-state (i.e. target pair state) Zeeman Shift
        #LIZZY We need to see if this is correct first before trusting the result 
        #opZeemanShift = (self.atom.getZeemanEnergyShift(
        #                      self.l, self.j, self.m1,
        #                      self.Bz) + \
        #                 self.atom.getZeemanEnergyShift(
        #                      self.ll, self.jj, self.m2,
        #                      self.Bz)) / C_h * 1.0e-9  # in GHz
                                        
        #temp
        opZeemanShift =0

        if debugOutput:
            print("\n ======= Coupling strengths (radial part only) =======\n")

        maxCoupling = "quadrupole-quadrupole"
        if (self.interactionsUpTo == 1):
            maxCoupling = "dipole-dipole"
        if debugOutput:
            print("Calculating coupling (up to ",maxCoupling,") between the pair states")

        for i in xrange(dimension):

            ed =  self.atom.getEnergyDefect2(states[opi][0],states[opi][1],states[opi][2],
                                          states[opi][3],states[opi][4],states[opi][5],
                                          states[i][0],states[i][1],states[i][2],
                                          states[i][3],states[i][4],states[i][5],states[i][6]) / C_h * 1.0e-9\
                 - opZeemanShift

            pairState1 = "|"+printStateString(states[i][0],states[i][1],states[i][2],states[i][6])+\
                        ","+printStateString(states[i][3],states[i][4],states[i][5],states[i][6])+">"
            #LIZZY: this means that the indexing of th energy defect in the 
            #function which calls this will have changed, TODO [6] -> [7]
            # states = [n,l,j,n1,l1,j1,s,ed]
            states[i].append(ed)  # energy defect of given state

            for j in xrange(i+1,dimension):
                
                #Work out if these states are coupled and by which means they are coupled?
                #Returns false if not coupled 2 if dipole dipole, 3 if quad- dipole, and 4 for quad-quad
                coupled = self.__isCoupled(states[i][0],states[i][1],states[i][2],
                                            states[i][3],states[i][4],states[i][5],
                                            states[j][0],states[j][1],states[j][2],
                                             states[j][3],states[j][4],states[j][5], limit, states[i][6])

                if (states[i][0]==24 and states[j][0]==18):
                    print("\n")
                    print(states[i])
                    print(states[j])
                    print(coupled)

                #if there is some coupling and the ns are within k range
                if coupled and (abs(states[i][0]-states[j][0])<=k and\
                                abs(states[i][3]-states[j][3])<=k ):
                    pairState2 = "|"+printStateString(states[j][0],states[j][1],states[j][2],states[j][6])+\
                                ","+printStateString(states[j][3],states[j][4],states[j][5],states[j][6])+">"
                    if debugOutput:
                        print(pairState1+" <---> "+pairState2)

                    #work out the coupling strength between the two. 
                    couplingStregth = _atomLightAtomCoupling(states[i][0],states[i][1],states[i][2],
                                            states[i][3],states[i][4],states[i][5],
                                            states[j][0],states[j][1],states[j][2],
                                             states[j][3],states[j][4],states[j][5],self.atom,states[j][6])/C_h*1.0e-9

                    couplingMatConstructor[coupled-2][0].append(float(couplingStregth))
                    couplingMatConstructor[coupled-2][1].append(i)
                    couplingMatConstructor[coupled-2][2].append(j)

                    exponent = coupled+1
                    if debugOutput:
                        print(("\tcoupling (C_%d/R^%d) = %.5f" %
                            (exponent,exponent,couplingStregth*(1e6)**(exponent))),\
                            "/R^",exponent," GHz  (mu m)^",exponent,"\n")

        # coupling = [1,1] dipole-dipole, [2,1]  quadrupole dipole, [2,2] quadrupole quadrupole
        
        length = len(couplingMatConstructor)
        couplingMatArray = [csr_matrix((couplingMatConstructor[i][0], \
                                (couplingMatConstructor[i][1], couplingMatConstructor[i][2])),\
                                       shape=(dimension, dimension))\
                    for i in xrange(length)]
        return states, couplingMatArray


    def __initializeDatabaseForMemoization(self):
        # memoization of angular parts
        self.conn = sqlite3.connect(os.path.join(self.dataFolder,\
                                                 "precalculated_pair.db"))
        self.c = self.conn.cursor()


        ### ANGULAR PARTS
        self.c.execute('''DROP TABLE IF EXISTS pair_angularMatrix''')
        self.c.execute('''SELECT COUNT(*) FROM sqlite_master
                            WHERE type='table' AND name='pair_angularMatrix';''')
        if (self.c.fetchone()[0] == 0):
            # create table
            print('gothere')
            try:
                self.c.execute('''CREATE TABLE IF NOT EXISTS pair_angularMatrix
                 (l1 TINYINT UNSIGNED, j1_x2 TINYINT UNSIGNED,
                 l2 TINYINT UNSIGNED, j2_x2 TINYINT UNSIGNED,
                 l3 TINYINT UNSIGNED, j3_x2 TINYINT UNSIGNED,
                 l4 TINYINT UNSIGNED, j4_x2 TINYINT UNSIGNED,
                 ind INTEGER, s TINYINT UNSIGNED,
                 PRIMARY KEY (l1,j1_x2, l2,j2_x2, l3,j3_x2, l4,j4_x2,s)
                ) ''')
            except sqlite3.Error as e:
                print(e)
            self.conn.commit()
        self.__loadAngularMatrixElementsFile()
        self.savedAngularMatrixChanged = False

    def __closeDatabaseForMemoization(self):
        self.conn.commit()
        self.conn.close()
        self.conn = False
        self.c = False


    def getLeRoyRadius(self):
        """
            Returns Le Roy radius for initial pair-state.

            Le Roy radius [#leroy]_ is defined as
            :math:`2(\\langle r_1^2 \\rangle^{1/2} + \\langle r_2^2 \\rangle^{1/2})`,
            where :math:`r_1` and :math:`r_2` are electron coordinates for the
            first and the second atom in the initial pair-state.
            Below this radius, calculations are not valid since electron
            wavefunctions start to overlap.

            Returns:
                float: LeRoy radius measured in :math:`\\mu m`

            References:
                .. [#leroy] R.J. Le Roy, Can. J. Phys. **52**, 246 (1974)
                    http://www.nrcresearchpress.com/doi/abs/10.1139/p74-035
        """
        step = 0.001
        r1,psi1_r1 = self.atom.radialWavefunction(self.l,self.s,self.j,\
                                               self.atom.getEnergy(self.n, self.l, self.j,self.s)/27.211,\
                                               15**(1/3.0),\
                                               2.0*self.n*(self.n+15.0), step)
        ###LIZZY :note temporary fix to the core polarizabilites
        sqrt_r1_on2 = np.trapz(np.multiply(np.multiply(psi1_r1,psi1_r1),\
                                               np.multiply(r1,r1)),\
                                     x = r1)

        r2,psi2_r2 = self.atom.radialWavefunction(self.ll,self.s,self.jj,\
                                               self.atom.getEnergy(self.nn, self.ll, self.jj,self.s)/27.211,\
                                               15**(1/3.0),\
                                               2.0*self.nn*(self.nn+15.0), step)

        sqrt_r2_on2 = np.trapz(np.multiply(np.multiply(psi2_r2,psi2_r2),\
                                               np.multiply(r2,r2)),\
                                     x = r2)



        return 2.*(sqrt(sqrt_r1_on2)+sqrt(sqrt_r2_on2))\
                *(physical_constants["Bohr radius"][0]*1.e6)
                
                
    def getC6perturbatively(self,theta,phi,nRange,energyDelta,semi):
        if isinstance(self.atom, EarthAlkaliAtom):
            return self._getC6perturbativelyEarthAlkali( self.m1, self.m2, self.m1f, self.m2f, theta, phi, nRange)

        else:
            return self._getC6perturbativelyAlkali(theta,phi,nRange,energyDelta,semi)
        
    def _getC6perturbativelyAlkali(self,theta,phi,nRange,energyDelta,semi):
        """
            Calculates :math:`C_6` from second order perturbation theory.

            Calculates
            :math:`C_6=\\sum_{\\rm r',r''}|\\langle {\\rm r',r''}|V|\
            {\\rm r1,r2}\\rangle|^2/\\Delta_{\\rm r',r''}`, where
            :math:`\\Delta_{\\rm r',r''}\\equiv E({\\rm r',r''})-E({\\rm r1, r2})`

            This calculation is faster then full diagonalization, but it is valid
            only far from the so called spaghetti region that occurs when atoms
            are close to each other. In that region multiple levels are strongly
            coupled, and one needs to use full diagonalization. In region where
            perturbative calculation is correct, energy level shift can be
            obtained as :math:`V(R)=-C_6/R^6`

            See `perturbative C6 calculations example snippet`_.

            .. _`perturbative C6 calculations example snippet`:
                ./Rydberg_atoms_a_primer.html#Dispersion-Coefficients

            Args:
                theta (float): orientation of inter-atomic axis with respect
                    to quantization axis (:math:`z`) in Euler coordinates
                    (measured in units of radian)
                phi (float): orientation of inter-atomic axis with respect
                    to quantization axis (:math:`z`) in Euler coordinates
                    (measured in units of radian)
                nRange (int): how much below and above the given principal quantum number
                    of the pair state we should be looking
                energyDelta (float): what is maximum energy difference ( :math:`\\Delta E/h` in Hz)
                    between the original pair state and the other pair states that we are including in
                    calculation

            Returns:larFactor = d
                                                #else:
                float: :math:`C_6` measured in :math:`\\text{GHz }\\mu\\text{m}^6`

            Example:
                If we want to quickly calculate :math:`C_6` for two Rubidium
                atoms in state :math:`62 D_{3/2} m_j=3/2`, positioned in space
                along the shared quantization axis::    print('angular part', D11(L1, L2, J1, J2, M1a, M2a, L1_i, L2_i, J1_i, J2_i, M1_i, M2_i, S, theta, phi))


                    from arc import *
                    calculation = PairStateInteractions(Rubidium(), 62, 2, 1.5, 62, 2, 1.5, 1.5, 1.5)
                    c6 = calculation.getC6perturbatively(0,0, 5, 25e9)
                    print "C_6 = %.0f GHz (mu m)^6" % c6

                Which returns::

                    C_6 = 767 GHz (mu m)^6

                Quick calculation of angular anisotropy of for Rubidium
                :math:`D_{2/5},m_j=5/2` states::

                    # Rb 60 D_{2/5}, mj=2.5 , 60 D_{2/5}, mj=2.5 pair state
                    calculation1 = PairStateInteractions(Rubidium(), 60, 2, 2.5, 60, 2, 2.5, 2.5, 2.5)
                    # list of atom orientations
                    thetaList = np.linspace(0,pi,30)
                    # do calculation of C6 pertubatively for all atom orientations
                    c6 = []
                    for theta in thetaList:
                        value = calculation1.getC6perturbatively(theta,0,5,25e9)
                        c6.append(value)
                        print ("theta = %.2f * pi \tC6 = %.2f GHz  mum^6" % (theta/pi,value))
                    # plot results                                                angularFactor = 0

                    plot(thetaList/pi,c6,"b-")
                    title("Rb, pairstate  60 $D_{5/2},m_j = 5/2$, 60 $D_{5/2},m_j = 5/2$")
                    xlabel(r"$\Theta /\pi$")
                    ylabel(r"$C_6$ (GHz $\mu$m${}^6$")
                    show()

        """
        self.__initializeDatabaseForMemoization()    

        # ========= START OF THE MAIN CODE ===========
        C6 = 0.
        # wigner D matrix allows calculations with arbitrary orientation of
        # the two atoms
        wgd = wignerDmatrix(theta,phi)
        # state that we are coupling
        statePart1 = singleAtomState(self.j, self.m1)
        statePart2 = singleAtomState(self.jj, self.m2)
        
        #wigner rotations are included in the calcuation of the matrix element
        #if not isinstance(self.atom,EarthAlkaliAtom):
        # rotate individual states
        dMatrix = wgd.get(self.j)
        statePart1 = dMatrix.dot(statePart1)

        dMatrix = wgd.get(self.jj)
        statePart2 = dMatrix.dot(statePart2)
        stateCom = compositeState(statePart1, statePart2)

        # any conservation?
        limitBasisToMj = False
        if theta<0.001:
            limitBasisToMj = True  # Mj will be conserved in calculations
        originalMj = self.m1+self.m2
        # this numbers are conserved if we use only dipole-dipole interactions
        Lmod2 = ((self.l+self.ll) % 2)

        # find nearby states

        lmin1 = self.l-1
        if lmin1 < -0.1:
            lmin1 = 1
        lmin2 = self.ll-1
        if lmin2 < -0.1:
            lmin2 = 1
        counter = 0

        
        
        #getting the intermediate pairstates 
        for n1 in xrange(max(self.n-nRange,1),self.n+nRange+1):
            for n2 in xrange(max(self.nn-nRange,1),self.nn+nRange+1):
                for l1 in xrange(lmin1,self.l+2,2):
                    for l2 in xrange(lmin2,self.ll+2,2):
                        j1 = l1-self.s
                        if l1 == 0:
                            j1 = self.s
                        while j1 <= l1+self.s+0.1:
                            j2 = l2-self.s
                            if l2 == 0:
                                j2 = self.s

                            while j2 <= l2+self.s+0.1:

                                getEnergyDefect = self.atom.getEnergyDefect2(self.n,self.l,self.j,\
                                                                  self.nn,self.ll,self.jj,\
                                                                  n1,l1,j1,\
                                                                  n2,l2,int(j2),self.s)/C_h
                                if abs(getEnergyDefect)<energyDelta  \
                                    and (not (self.interactionsUpTo==1) or\
                                         (Lmod2 == ((l1+l2)%2) )) :
                                    getEnergyDefect = getEnergyDefect*1.0e-9 # GHz

                                    # calculate radial part
                                    couplingStregth = _atomLightAtomCoupling(self.n,self.l,self.j,
                                            self.nn,self.ll,self.jj,
                                            n1,l1,j1,
                                            n2,l2,j2,self.atom,self.s)*(1.0e-9*(1.0e6)**3/C_h) # GHz / mum^3

                                    pairState2 = "|"+printStateString(n1,l1,j1,self.s)+\
                                        ","+printStateString(n2,l2,j2,self.s)+">"

                                    # include relevant mj and add contributions[]
                                    for m1c in np.linspace(j1,-j1,round(1+2*j1)):
                                        for m2c in np.linspace(j2,-j2,round(1+2*j2)):
                                            #I should get rid of this for the EarthAlkali atoms 
                                            if ((not limitBasisToMj) or (abs(originalMj-m1c-m2c)==0) ):
                                                # find angular part
                                                statePart1 = singleAtomState(j1, m1c)
                                                statePart2 = singleAtomState(j2, m2c)
                                                # rotate individual states
                                                #if not  isinstance(self.atom,EarthAlkaliAtom):
                                                
                                                
                                                dMatrix = wgd.get(j1)
                                                statePart1 = dMatrix.dot(statePart1)
                                                dMatrix = wgd.get(j2)
                                                statePart2 = dMatrix.dot(statePart2)
                                                # composite state of two atoms
                                                stateCom2 = compositeState(statePart1, statePart2)
    
                                                d = self.__getAngularMatrix_M(self.l,self.j,
                                                                               self.ll,self.jj,
                                                                               l1,j1,
                                                                               l2,j2,
                                                                               self.atom)
                                                #check to see if there is already the anglular parts included
                                                #if  isinstance(self.atom,EarthAlkaliAtom):
                                                #    angularFactor = d
                                                #else:
                                                
                                                angularFactor = conjugate(stateCom2.T).dot(d.dot(stateCom))
                                                #print('angualar factor, coupling strength', angularFactor, couplingStregth)
                                                angularFactor = real(angularFactor[0,0])
                                                C6 += -((couplingStregth*angularFactor)**2/getEnergyDefect)
                              
                                j2 = j2+1.0
                            j1 = j1+1.0

        # ========= END OF THE MAIN CODE ===========
        self.__closeDatabaseForMemoization()
        
        return C6

    #def getC6perturbively()
    #should take an M1 and M2 and M1' and M2' value for us to return the value of 
    def getC6term(self,n1,l1,j1,n2,l2,j2,m1_ini,m1_fini,m2_ini,m2_fini, m1_pairstate, m2_pairstate,theta,phi):
        wgd = wignerDmatrix(theta,phi)
        
        
        #HAVE TO MAKE TWO SETS OF PAIRSTATES WE NEED to get the angular momentutm twice
        ed = self.atom.getEnergyDefect2(self.n,self.l,self.j,\
                                  self.nn,self.ll,self.jj,\
                                  n1,l1,j1,\
                                  n2,l2,int(j2),self.s)/C_h* 1.0e-9 
                                    
        d = self.__getAngularMatrix_M(self.l,self.j,self.ll,self.jj,\
                                      l1,j1,l2,j2,self.atom)
        
        #get the first angular part 
        statePart1 = singleAtomState(self.j, m1_ini)
        dMatrix = wgd.get(self.j)
        statePart1 = dMatrix.dot(statePart1)
        
        statePart1_final = singleAtomState(self.j,m1_fini)
        statePart1_final = dMatrix.dot(statePart1_final)
        
        
        #get the rotate states for the final 
        statePart2 = singleAtomState(self.jj, m2_ini)
        dMatrix = wgd.get(self.jj)
        statePart2 = dMatrix.dot(statePart2)
        
        statePart2_final = singleAtomState(self.jj,m2_fini)
        statePart2_final = dMatrix.dot(statePart2_final)
        
    
        #get the rotated states for the pairstate
        statePart1_pair = singleAtomState(j1,m1_pairstate)
        dMatrix = wgd.get(j1)
        statePart1_pair = dMatrix.dot(statePart1_pair)
        
        statePart2_pair = singleAtomState(j2,m2_pairstate)
        dMatrix = wgd.get(j2)
        statePart2_pair = dMatrix.dot(statePart2_pair)
        
        stateCom = compositeState(statePart1, statePart2)

        stateCom_final = compositeState(statePart1_final, statePart2_final)

        
        stateCom_pair = compositeState(statePart1_pair, statePart2_pair)
       
        #angular Calcuation     LIZZY cHECK THIS IS CORECT
        #print('inital paistate composite',stateCom.shape)
        #print('Final paistate composite',stateCom_final.shape)

        
        angularFactor1 = conjugate(stateCom_pair.T).dot(d.dot(stateCom))
        angularFactor1 = real(angularFactor1[0,0])
        
        angularFactor2 = conjugate(stateCom_pair.T).dot(d.dot(stateCom_final))
        angularFactor2 = real(angularFactor2[0,0])
        #get the light atom coupling here 
        
        #print('angular',angularFactor1)
        #print('agular2',angularFactor2)
        
        couplingStrength = _atomLightAtomCoupling(self.n,self.l,self.j,\
                                            self.nn,self.ll,self.jj,\
                                            n1,l1,j1,\
                                            n2,l2,j2,self.atom,self.s)*(1.0e-9*(1.0e6)**3/C_h) # GHz / mum^3

        #write
       #have to loop over all pairstates, Work out the small C6 elements.

    #defi,
        #if abs(ed) < 1.: 
        #    print(n1,l1,j1,n2,l2,j2, m1_fini, m2_fini,m1_pairstate, m2_pairstate)
        #    print('radial', couplingStrength**2)
        #    print('angular',angularFactor1*angularFactor2)
        #    print('ed',ed)

        #   print('inter',self.atom.getQuantumDefect(n1,l1,j1,self.s))

        #    print('inter',self.atom.getQuantumDefect(n2,l2,j2,self.s))
        #    print(self.atom.getEnergy(n1,l1,j1,self.s)*C_e/C_h *1e-9)
        #    print(self.atom.getEnergy(n2,l2,j2,self.s)*C_e/C_h *1e-9)
        #    print(self.atom.getEnergy(self.n,self.l,self.j,self.s)*C_e/C_h *1e-9)
        #    print(self.atom.getEnergy(self.nn,self.ll,self.jj,self.s)*C_e/C_h *1e-9)
            

        #    print()
        #    print()
        return -(couplingStrength**2*angularFactor1*angularFactor2)/ed
    
    def commonStates(self, arr1, arr2):
        nrows, ncols = arr1.shape
        dtype={'names':['f{}'.format(i) for i in range(ncols)],
       'formats':ncols * [arr1.dtype]}

        common = np.intersect1d(arr1.view(dtype), arr2.view(dtype))

        return common
    
    def getIntermediateQuantumNumbers(self,n,l,s,j,m,rangeOfN):
        '''This is going to take some initial state and some final state'''
        #first we get the range of
        nRange = np.arange(n-rangeOfN, n+rangeOfN+1,1)
        LRange = np.delete(np.array([l-1, l+1]), np.where(np.array([l-1, l+1])<0))
        MRange = np.array([m-1,m,m+1])
        if j == 0:
            JRange = np.array([1])
        else:
            JRange = np.array([j-1,j,j+1])

        #create a matrix of n,l,s,j,m values, 0:n, 1:l, 2:s, 3:j, 4:m
        out =np.array(list(itertools.product(nRange,LRange,np.array([s]), JRange, MRange)))
        #now remove where Mj > J
        out = np.delete(out[:], np.where(abs(out[:,4])>out[:,3]),axis = 0).reshape(-1,5)
        #remove where J> L+S  J< abs(l-S)
        out = np.delete(out[:], np.where(abs(out[:,3])>(out[:,2]+out[:,1])),axis = 0).reshape(-1,5)
        out = np.delete(out[:], np.where(abs(out[:,3])<abs(out[:,1]-out[:,2])),axis = 0).reshape(-1,5)

        #out = np.delete(out, [n,l,s,j,m], axis = 0).shape
        return out
    def getIntermediatePairstates(self,n1,l1,s1,j1, m1, n2, l2,s2,j2,m2,nRange):
        '''This is going to take quantum numbers and return
        an array consisiting of all possible combinations of the valid pairstates

        This is going to be used to work out intermediate pairstates between inital and final values of M,
        this means that the spin
        '''
        pairstates_a = self.getIntermediateQuantumNumbers(n1,l1,s1,j1,m1,nRange)
        pairstates_b = self.getIntermediateQuantumNumbers(n2,l2,s2,j2,m2,nRange)
        out = np.array(list(itertools.product(pairstates_a, pairstates_b))).reshape(-1,10)
        return out

    def _getC6perturbativelyEarthAlkali(self,m1, m2, m1f, m2f, theta, phi, nRange):
        #this should facilitate for the provision of a M1, M2, M1' and an M2'. For which we return the small C6
        #the m1', m2' should be provided in the construtor
        #get the intermeidate pairstates, 
        #
         

        states_a = self.getIntermediatePairstates(self.n,self.l,self.s,self.j,m1,self.nn,self.ll,self.s,self.jj, m2,nRange)
        states_b = self.getIntermediatePairstates(self.n,self.l,self.s,self.j,m1f, self.n,self.ll,self.s,self.jj, m2f, nRange)

        common = self.commonStates(states_a, states_b)
        #print(common.shape)
        terms = [self.getC6term( n_i, l_i, j_i,  n1_i, l1_i, j1_i,m1,m1f, m2, m2f, m_i,m1_i,theta, phi) \
                for n_i, l_i, s_i, j_i, m_i, n1_i, l1_i, s1_i, j1_i, m1_i in common]
        c6 = np.sum(terms, axis = 0)
        
        return c6
    
    def getC6Diagonalisation(self, theta, phi, nRange):
        '''This carries out a full diagonalistaion of the C6 matrix 
        
        Need to add the matrix saving option and the progress output
        Need to add matrix loading and saving
        
        '''
        self.__initializeDatabaseForMemoization() 
        
        
        dim = (2*self.jj +1)*(2*self.j+1)
        mat = np.zeros((dim,dim), dtype = 'complex')
        m_list = np.arange(-self.j, self.j+1)
        m1_list = np.arange(-self.jj, self.jj+1)

        #Make a list of all possible combinations of ms
        pairs = np.array(list(itertools.product(m_list, m1_list)))
        #sum these to work out the omegas
        omegas = np.sum(pairs, axis = 1 ).reshape((-1,1))
        #print(omegas)

        concat = np.concatenate((omegas, pairs), axis = 1)
        
        #print(concat)

        sorted_pairs = concat[np.lexsort((concat[:,1],concat[:,0])),:]
        sorted_pairs = sorted_pairs[:,1:]
        #print(sorted_pairs)

        #the symmetric states were taken from Angelas
        if self.j==1: #e.g.3S1
            symindices = [[0,0],[1,1],[1,2],[3,3],[3,4],[3,5],[4,4]]
        elif self.j==2: #e.g. 3D2
            symindices = [[0,0],[1,1],[1,2],[3,3],[3,4],[3,5],[4,4],[6,6],[6,7],[6,8],[6,9],\
            [7,7],[7,9],[10,10],[10,11],[10,12],[10,13],[10,14],[11,11],[11,12],[11,13],[12,12]]

        if(self.j == 0 ): symindices= [[0,0]]

        #my code again
        #for i,k in symindices:
        for i in range(dim):
            for k in range(dim):
                m1, m2 = sorted_pairs[i]
                m1f, m2f = sorted_pairs[k]
                

                #print('['+str(i)+','+str(k)+']',m1,m2,m1f,m2f)
                
                mat[i,k] = self._getC6perturbativelyEarthAlkali(m1, m2, m1f, m2f, theta, phi, nRange)
                #if mat[i,k] != 0.:
                #    print(i,k)
                #    print(m1, m2, m1f, m2f)
                #    print(mat[i,k])
                
        #if self.j==1:
        #    mat[8,8] = mat[0,0]
        #    mat[2,2], mat[6,6], mat[7,7] = np.repeat(mat[1,1], 3)
        #    mat[2,1], mat[6,7], mat[7,6] = np.repeat(mat[1,2], 3)
        #    mat[5,5] = mat[3,3]
        #    mat[4,3], mat[4,5], mat[5,4] = np.repeat(mat[3,4], 3)
        #    mat[5,3] = mat[3,5]
        #elif self.j==2:
        #    mat[24,24] = mat[0,0]
        #    mat[2,2], mat[22,22], mat[23,23] = np.repeat(mat[1,1], 3)
        #    mat[2,1], mat[22,23], mat[23,22] = np.repeat(mat[1,2], 3)
        #    mat[5,5], mat[19,19], mat[21,21] = np.repeat(mat[3,3], 3)
        #    mat[4,3], mat[4,5], mat[5,4], mat[19,20], mat[20,19], mat[20,21], mat[21,20] = np.repeat(mat[3,4], 7)
        #    mat[5,3], mat[19,21], mat[21,19] = np.repeat(mat[3,5], 3)
        #    mat[20,20] = mat[4,4]
        #    mat[8,8], mat[16,16], mat[18,18] = np.repeat(mat[6,6], 3)
        #    mat[7,6], mat[8,9], mat[9,8], mat[15,16], mat[16,15], mat[18,17], mat[17,18] = np.repeat(mat[6,7], 7)
        #    mat[8,6], mat[16,18], mat[18,16] = np.repeat(mat[6,8], 3)
        #    mat[7,8], mat[8,7], mat[9,6], mat[15,18], mat[16,17], mat[17,16], mat[18,15] = np.repeat(mat[6,9], 7)
        #    mat[9,9], mat[15,15], mat[17,17] = np.repeat(mat[7,7], 3)
        #    mat[9,7], mat[15,17], mat[17,15] = np.repeat(mat[7,9], 3)
        #    mat[14,14] = mat[10,10]
        #    mat[11,10], mat[13,14], mat[14,13] = np.repeat(mat[10,11], 3)
        #    mat[12,10], mat[12,14], mat[14,12] = np.repeat(mat[10,12], 3)
        #    mat[11,14], mat[13,10], mat[14,11] = np.repeat(mat[10,13], 3)
        #    mat[14,10] = mat[10,14]
        #    mat[13,13] = mat[11,11]
        #    mat[12,11], mat[12,13], mat[13,12] = np.repeat(mat[11,12], 3)
        #    mat[13,11] = mat[11,13]      
            
        self.__updateAngularMatrixElementsFile()

        self.__closeDatabaseForMemoization()

        
        return np.linalg.eigh(mat)

    def defineBasis(self,theta,phi,nRange,lrange,energyDelta,\
                         Bz=0, progressOutput=False,debugOutput=False):
        """
            Finds relevant states in the vicinity of the given pair-state

            Finds relevant pair-state basis and calculates interaction matrix.
            Pair-state basis is saved in :obj:`basisStates`.
            Interaction matrix is saved in parts depending on the scaling with
            distance. Diagonal elements :obj:`matDiagonal`, correponding to
            relative energy defects of the pair-states, don't change with
            interatomic separation. Off diagonal elements can depend
            on distance as :math:`R^{-3}, R^{-4}` or :math:`R^{-5}`, corresponding
            to dipole-dipole (:math:`C_3` ), dipole-qudrupole (:math:`C_4` ) and
            quadrupole-quadrupole coupling (:math:`C_5` ) respectively. These
            parts of the matrix are stored in :obj:`matR` in that order. I.e.
            :obj:`matR[0]` stores dipole-dipole coupling (:math:`\propto R^{-3}`),
            :obj:`matR[0]` stores dipole-quadrupole couplings etc.

            Args:
                theta (float):  relative orientation of the two atoms
                    (see figure on top of the page), range 0 to :math:`\pi`
                phi (float): relative orientation of the two atoms (see figure
                    on top of the page), range 0 to :math:`2\pi`
                nRange (int): how much below and above the given principal quantum number
                    of the pair state we should be looking?
                lrange (int): what is the maximum angular orbital momentum state that we are including
                    in calculation
                energyDelta (float): what is maximum energy difference ( :math:`\\Delta E/h` in Hz)
                    between the original pair state and the other pair states that we are including in
                    calculation
                Bz (float): optional, magnetic field directed along z-axis in
                    units of Tesla. Calculation will be correct only for weak
                    magnetic fields, where paramagnetic term is much stronger
                    then diamagnetic term. Diamagnetic term is neglected.
                progressOutput (bool): optional, False by default. If true, prints
                    information about the progress of the calculation.
                debugOutput (bool): optional, False by default. If true, similar
                    to progressOutput=True, this will print information about the
                    progress of calculations, but with more verbose output.

            See also:
                :obj:`alkali_atom_functions.saveCalculation` and
                :obj:`alkali_atom_functions.loadSavedCalculation` for information
                on saving intermediate results of calculation for later use.
        """

        self.__initializeDatabaseForMemoization()

        # save call parameters
        self.theta = theta; self.phi = phi; self.nRange = nRange;
        self.lrange = lrange; self.energyDelta = energyDelta
        self.Bz = Bz

        # wignerDmatrix
        wgd = wignerDmatrix(theta,phi)

        limitBasisToMj = False
        if (theta<0.001):
            limitBasisToMj = True  # Mj will be conserved in calculations

        originalMj = self.m1+self.m2

        self.channel, self.coupling= self.__makeRawMatrix2(
                                                    self.n,self.l,self.j,
                                                    self.nn,self.ll,self.jj,
                                                    nRange,lrange,energyDelta,
                                                    limitBasisToMj,
                                                    progressOutput=progressOutput,
                                                    debugOutput=debugOutput)

        #self.atom.updateDipoleMatrixElementsFile()
        # generate all the states (with mj principal quantum number)
        
        # opi = original pairstate index
        opi = 0
        #print(self.coupling[0].data)

        # NEW FOR SPACE MATRIX
        self.index = np.zeros(len(self.channel)+1,dtype=np.int16)

        #For all the channels
        for i in xrange(len(self.channel)):
            self.index[i] = len(self.basisStates)

            stateCoupled=self.channel[i]
            #for all M1 M2 pairs 
            for m1c in np.linspace(stateCoupled[2],-stateCoupled[2],\
                                   round(1+2*stateCoupled[2])):
                for m2c in np.linspace(stateCoupled[5],-stateCoupled[5],\
                                       round(1+2*stateCoupled[5])):
                    #if we do not have to conserve the M = m1 +m2 quantum number 
                    if ((not limitBasisToMj) or (abs(originalMj-m1c-m2c)==0) ):
                        
                        #Then we add it to this list of  basis states 
                        self.basisStates.append([stateCoupled[0],stateCoupled[1],stateCoupled[2],m1c,
                                        stateCoupled[3],stateCoupled[4],stateCoupled[5],m2c,stateCoupled[6]])
                        #then we add channel number to matrix element (should have m1*m2))
                        self.matrixElement.append(i)

                        #check to see if it is the orginal basis state.
                        if (abs(stateCoupled[0]-self.n)<0.1 and \
                            abs(stateCoupled[1]-self.l)<0.1 and \
                            abs(stateCoupled[2]-self.j)<0.1 and \
                            abs(m1c-self.m1)<0.1 and \
                            abs(stateCoupled[3]-self.nn)<0.1 and \
                            abs(stateCoupled[4]-self.ll)<0.1 and \
                            abs(stateCoupled[5]-self.jj)<0.1 and \
                            abs(m2c-self.m2)<0.1):
                            
                            #get the index for the original basis state
                            opi = len(self.basisStates)-1
            #if nothing has been added to the basis states then print?                
            if (self.index[i] == len(self.basisStates)):
                print(stateCoupled)
        #the last index should be the length of all basis states        
        self.index[-1] = len(self.basisStates)

        if progressOutput or debugOutput:
            print("\nCalculating Hamiltonian matrix...\n")

        #The dimension of the interaction matrix
        dimension = len(self.basisStates)
        if progressOutput or debugOutput:
            print("\n\tmatrix (dimension ",dimension,")\n")

        # INITIALIZING MATICES
        # all (sparce) matrices will be saved in csr format
        # value, row, column
        matDiagonalConstructor = [[],[],[]]

        matRConstructor = [ [[],[],[]] \
                           for i in xrange(self.interactionsUpTo*2-1) ]

        matRIndex = 0
        #coupling csr matrixces
        for c in self.coupling:
            progress = 0.
            for ii in xrange(len(self.channel)):
                if progressOutput:
                    dim  = len(self.channel)
                    progress += ((dim-ii)*2-1)
                    sys.stdout.write( "\rMatrix R%d %.1f %% (state %d of %d)"%\
                                      (matRIndex+3,float(progress)/float(dim**2)*100.,\
                                                   ii+1,len(self.channel)))
                    sys.stdout.flush()

                #LIZZY index change from 6 to 7
                ed = self.channel[ii][7]
                #print(self.channel)
                #need to check if that is right? Maybe get using the debugging tools
                # solves problems with exactly degenerate basisStates
                degeneracyOffset = 0.00000001

                i = self.index[ii]
                dMatrix1 = wgd.get(self.basisStates[i][2])
                dMatrix2 = wgd.get(self.basisStates[i][6])

                for i in xrange(self.index[ii],self.index[ii+1]):
                    statePart1 = singleAtomState(self.basisStates[i][2], self.basisStates[i][3])
                    statePart2 = singleAtomState(self.basisStates[i][6], self.basisStates[i][7])
                    # rotate individual states

                    statePart1 = dMatrix1.dot(statePart1)
                    statePart2 = dMatrix2.dot(statePart2)

                    stateCom = compositeState(statePart1, statePart2)

                    if (matRIndex==0):
                    #    zeemanShift = (self.atom.getZeemanEnergyShift(
                     #                      self.basisStates[i][1],
                      #                     self.basisStates[i][2],
                       #                    self.basisStates[i][3],
                    #                       self.Bz) + \
                    #                   self.atom.getZeemanEnergyShift(
                    #                       self.basisStates[i][5],
                    #                       self.basisStates[i][6],
                    #                       self.basisStates[i][7],
                    #                       self.Bz)) / C_h * 1.0e-9   # in GHz
                        
                        zeemanShift = 0 
                        matDiagonalConstructor[0].append(ed + zeemanShift +
                                                         degeneracyOffset)
                        degeneracyOffset +=  0.00000001
                        matDiagonalConstructor[1].append(i)
                        matDiagonalConstructor[2].append(i)

                    for dataIndex in xrange(c.indptr[ii],c.indptr[ii+1]):

                        jj = c.indices[dataIndex]
                        radialPart = c.data[dataIndex]

                        j = self.index[jj]
                        dMatrix3 = wgd.get(self.basisStates[j][2])
                        dMatrix4 = wgd.get(self.basisStates[j][6])
                        
                        #print('basis states i', self.basisStates[i],i)
                        #print('basis states j', self.basisStates[j],j)

                        #print()

                        if (self.index[jj]!=self.index[jj+1]):
                            d = self.__getAngularMatrix_M(self.basisStates[i][1],self.basisStates[i][2],
                                                        self.basisStates[i][5],self.basisStates[i][6],
                                                        self.basisStates[j][1],self.basisStates[j][2],
                                                        self.basisStates[j][5],self.basisStates[j][6],
                                                        self.atom)
                            secondPart = d.dot(stateCom)
                        else:
                            print(" - - - ",self.channel[jj])


                        for j in xrange(self.index[jj],self.index[jj+1]):
                            statePart1 = singleAtomState(self.basisStates[j][2], self.basisStates[j][3])
                            statePart2 = singleAtomState(self.basisStates[j][6], self.basisStates[j][7])
                            # rotate individual states

                            statePart1 = dMatrix3.dot(statePart1)
                            statePart2 = dMatrix4.dot(statePart2)
                            # composite state of two atoms
                            stateCom2 = compositeState(statePart1, statePart2)

                            angularFactor = conjugate(stateCom2.T).dot(secondPart)
                            angularFactor = real(angularFactor[0,0])

                            if (abs(angularFactor)>1.e-5):
                                matRConstructor[matRIndex][0].append(radialPart*angularFactor)
                               #print(radialPart*angularFactor)
                                matRConstructor[matRIndex][1].append(i)
                                matRConstructor[matRIndex][2].append(j)

                                matRConstructor[matRIndex][0].append(radialPart*angularFactor)
                                matRConstructor[matRIndex][1].append(j)
                                matRConstructor[matRIndex][2].append(i)
                    
            matRIndex += 1
            if progressOutput or debugOutput:
                print("\n")

        self.matDiagonal = csr_matrix((matDiagonalConstructor[0], \
                                    (matDiagonalConstructor[1], matDiagonalConstructor[2])),
                                       shape=(dimension, dimension))


        self.matR = [csr_matrix((matRConstructor[i][0], \
                                 (matRConstructor[i][1], matRConstructor[i][2])),
                                shape=(dimension, dimension))
                     for i in xrange(self.interactionsUpTo*2-1)
                     ]


        self.originalPairStateIndex = opi

        self.__updateAngularMatrixElementsFile()
        self.__closeDatabaseForMemoization()


    def diagonalise(self,rangeR,noOfEigenvectors,
                         drivingFromState = [0,0,0,0,0],
                         eigenstateDetuning = 0.,
                         sortEigenvectors = False,
                         progressOutput = False,
                         debugOutput = False):
        """
            Finds eigenstates in atom pair basis.

            ARPACK ( :obj:`scipy.sparse.linalg.eigsh`) calculation of the
            `noOfEigenvectors` eigenvectors closest to the original state. If
            `drivingFromState` is specified as `[n,l,j,mj,q]` coupling between
            the pair-states and the situation where one of the atoms in the pair
            state basis is in :math:`|n,l,j,m_j\\rangle` state due to driving
            with a laser field that drives :math:`q` transition (+1,0,-1 for
            :math:`\\sigma^-`, :math:`\pi` and :math:`\\sigma^+` transitions
            respectively)  is calculated and marked by the colourmaping these
            values on the obtained eigenvectors.

            Args:
                rangeR ( :obj:`array`): Array of values for distance between the
                    atoms (in :math:`\mu` m) for which we want to calculate
                    eigenstates.
                noOfEigenvectors (int): number of eigen vectors closest to the
                    energy of the original (unperturbed) pair state. Has to be
                    smaller then the total number of states.
                eigenstateDetuning (float, optional): Default is 0. This
                    specifies detuning from the initial pair-state (in Hz) around which we want to find `noOfEigenvectors` eigenvectors.
                    This is useful when looking only for couple of off-resonant features.
                drivingFromState ([int,int,float,float,int]): Optional. State
                    of the one of the atoms from the original pair-state basis
                    from which we try to dribe to the excited pair-basis
                    manifold. By default, program will calculate just
                    contribution of the original pair-state in the eigenstates
                    obtained by diagonalization, and will highlight it's
                    admixure by colour mapping the obtained eigenstates plot.
                sortEigenvectors(bool): optional, False by default. Tries to sort
                    eigenvectors so that given eigen vector index corresponds
                    to adiabatically changing eigenstate, as detirmined by
                    maximising overlap between old and new eigenvectors.
                progressOutput (bool): optional, False by default. If true, prints
                    information about the progress of the calculation.
                debugOutput (bool): optional, False by default. If true, similar
                    to progressOutput=True, this will print information about the
                    progress of calculations, but with more verbose output.
        """

        self.r = np.sort(rangeR)
        dimension = len(self.basisStates)

        self.noOfEigenvectors = noOfEigenvectors

        # energy of the state - to be calculated
        self.y = []
        # how much original state is contained in this eigenvector
        self.highlight = []
        # what are the dominant contributing states?
        self.composition = []

        if (noOfEigenvectors>=dimension-1):
            noOfEigenvectors=dimension-1
            print("Warning: Requested number of eigenvectors >=dimension-1\n \
                 ARPACK can only find up to dimension-1 eigenvectors, where\
                dimension is matrix dimension.\n");
            if noOfEigenvectors<1:
                return

        coupling = []
        self.maxCoupling = 0.
        self.maxCoupledStateIndex = 0
        if (drivingFromState[0] != 0):
            self.drivingFromState = drivingFromState
            if progressOutput: print("Finding coupling strengths")
            # get first what was the state we are calculating coupling with
            state1 = drivingFromState
            n1 = int(round(state1[0]))
            l1 = int(round(state1[1]))
            j1 = state1[2]
            m1 = state1[3]
            q = state1[4]

            for i in xrange(dimension):
                thisCoupling = 0.
                #if progressOutput:
                #    sys.stdout.write("\r%d%%" %  (i/float(dimension)*100.))
                #    sys.stdout.flush()
                if int(abs(self.basisStates[i][1]-l1))==1 and \
                   abs(self.basisStates[i][4]-self.basisStates[self.originalPairStateIndex][4])<0.1 and \
                   abs(self.basisStates[i][5]-self.basisStates[self.originalPairStateIndex][5])<0.1 and \
                   abs(self.basisStates[i][6]-self.basisStates[self.originalPairStateIndex][6])<0.1 and \
                   abs(self.basisStates[i][7]-self.basisStates[self.originalPairStateIndex][7])<0.1 :
                    state2 = self.basisStates[i]
                    n2 = int(state2[0])
                    l2 = int(state2[1])
                    j2 = state2[2]
                    m2 = state2[3]
                    if debugOutput:
                        print(n1," ",l1," ",j1," ",m1," ",n2," ",l2," ",j2," ",m2," q=",q)
                        print(self.basisStates[i])
                    dme = self.atom.getDipoleMatrixElement(n1, l1,j1,m1,\
                                                            n2,l2,j2,m2,\
                                                            q,self.s)
                    thisCoupling += dme

                if int(abs(self.basisStates[i][5]-l1))==1 and \
                    abs(self.basisStates[i][0]-self.basisStates[self.originalPairStateIndex][0])<0.1 and\
                    abs(self.basisStates[i][1]-self.basisStates[self.originalPairStateIndex][1])<0.1 and \
                    abs(self.basisStates[i][2]-self.basisStates[self.originalPairStateIndex][2])<0.1 and \
                    abs(self.basisStates[i][3]-self.basisStates[self.originalPairStateIndex][3])<0.1 :
                    state2 = self.basisStates[i]
                    n2 = int(state2[0+4])
                    l2 = int(state2[1+4])
                    j2 = state2[2+4]
                    m2 = state2[3+4]
                    if debugOutput:
                        print(n1," ",l1," ",j1," ",m1," ",n2," ",l2," ",j2," ",m2," q=",q)
                        print(self.basisStates[i])
                    dme = self.atom.getDipoleMatrixElement(n1, l1,j1,m1,\
                                                            n2,l2,j2,m2,\
                                                            q,self.s)
                    thisCoupling += dme

                thisCoupling = abs(thisCoupling)**2
                if thisCoupling > self.maxCoupling:
                    self.maxCoupling = thisCoupling
                    self.maxCoupledStateIndex = i
                if (thisCoupling >0.000001) and debugOutput:
                    print("original pairstate index = ",self.originalPairStateIndex)
                    print("this pairstate index = ",i)
                    print("state itself ", self.basisStates[i])
                    print("coupling = ",thisCoupling)
                coupling.append(thisCoupling)

            print("Maximal coupling from a state")
            print("is to a state ",self.basisStates[self.maxCoupledStateIndex])
            print("is equal to %.3e a_0 e" % self.maxCoupling)

        if progressOutput:
            print("\n\nDiagonalizing interaction matrix...\n")

        rvalIndex = 0.
        previousEigenvectors = []

        for rval in self.r:
            if progressOutput:
                sys.stdout.write("\r%d%%" %  (rvalIndex/len(self.r-1)*100.))
                sys.stdout.flush()
            rvalIndex += 1.

            # calculate interaction matrix

            m = self.matDiagonal

            rX = (rval*1.e-6)**3
            for matRX in self.matR:
                m = m + matRX/rX
                rX *= (rval*1.e-6)

            # uses ARPACK algorithm to find only noOfEigenvectors eigenvectors
            # sigma specifies center frequency (in GHz)
            ev, egvector = eigsh(m, noOfEigenvectors, sigma= eigenstateDetuning*1.e-9, which='LM',tol=1E-6)

            if sortEigenvectors:
                # Find which eigenvectors overlap most with eigenvectors from
                # previous diagonalisatoin, in order to find "adiabatic"
                # continuation for the respective states

                if previousEigenvectors==[]:
                    previousEigenvectors = np.copy(egvector)
                    previousEigenvalues = np.copy(ev)
                rowPicked = [False for i in range(len(ev))]
                columnPicked = [False for i in range(len(ev))]

                stateOverlap = np.zeros((len(ev),len(ev)))
                for i in range(len(ev)):
                    for j in range(len(ev)):
                        stateOverlap[i,j] = np.vdot(egvector[:, i],
                                                    previousEigenvectors[:, j])**2

                sortedOverlap = np.dstack(np.unravel_index(np.argsort(stateOverlap.ravel()),
                                                           (len(ev), len(ev))))[0]

                sortedEigenvaluesOrder = np.zeros(len(ev), dtype=np.int32)
                j = len(ev)**2 - 1
                for i in range(len(ev)):
                    while rowPicked[sortedOverlap[j, 0]] or \
                        columnPicked[sortedOverlap[j, 1]]:
                        j -= 1
                    rowPicked[sortedOverlap[j, 0]] = True
                    columnPicked[sortedOverlap[j, 1]] = True
                    sortedEigenvaluesOrder[sortedOverlap[j, 1]] = sortedOverlap[j, 0]

                egvector = egvector[:,sortedEigenvaluesOrder]
                ev = ev[sortedEigenvaluesOrder]
                previousEigenvectors = np.copy(egvector)

            self.y.append(ev)

            if (drivingFromState[0] < 0.1):
                # if we've defined from which state we are driving
                sh = []
                comp = []
                for i in xrange(len(ev)):
                    sh.append(abs(egvector[self.originalPairStateIndex,i])**2)
                    comp.append(self._stateComposition(egvector[:,i]))
                self.highlight.append(sh)
                self.composition.append(comp)
            else:
                sh = []
                comp = []
                for i in xrange(len(ev)):
                    sumCoupledStates = 0.
                    for j in xrange(dimension):
                        sumCoupledStates += abs(coupling[j]/self.maxCoupling)*\
                                                abs(egvector[j,i])**2
                    comp.append(self._stateComposition(egvector[:,i]))
                    sh.append(sumCoupledStates)
                self.highlight.append(sh)
                self.composition.append(comp)

        # end of FOR loop over inter-atomic dinstaces


    def exportData(self,fileBase,exportFormat = "csv"):
        """
            Exports PairStateInteractions calculation data.

            Only supported format (selected by default) is .csv in a
            human-readable form with a header that saves details of calculation.
            Function saves three files: 1) `filebase` _r.csv;
            2) `filebase` _energyLevels
            3) `filebase` _highlight

            For more details on the format, see header of the saved files.

            Args:
                filebase (string): filebase for the names of the saved files
                    without format extension. Add as a prefix a directory path
                    if necessary (e.g. saving outside the current working directory)
                exportFormat (string): optional. Format of the exported file. Currently
                    only .csv is supported but this can be extended in the future.
        """
        fmt='on %Y-%m-%d @ %H:%M:%S'
        ts = datetime.datetime.now().strftime(fmt)

        commonHeader = "Export from Alkali Rydberg Calculator (ARC) %s.\n" % ts
        commonHeader += ("\n *** Pair State interactions for %s %s m_j = %d/2 , %s m_j = %d/2 pair-state. ***\n\n" %\
                          (self.atom.elementName,
                        printStateString(self.n, self.l, self.j,self.s), int(round(2.*self.m1)),\
                        printStateString(self.nn, self.ll, self.jj,self.s), int(round(2.*self.m2)) ) )
        if (self.interactionsUpTo==1):
            commonHeader += " - Pair-state interactions included up to dipole-dipole coupling.\n"
        elif (self.interactionsUpTo==2):
            commonHeader += " - Pair-state interactions included up to quadrupole-quadrupole coupling.\n"

        if hasattr(self, 'theta'):
            commonHeader += " - Atom orientation:\n"
            commonHeader += "      theta (polar angle) = %.5f x pi\n" % (self.theta/pi)
            commonHeader += "      phi (azimuthal angle) = %.5f x pi\n" % (self.phi/pi)
            commonHeader += " - Calculation basis includes:\n"
            commonHeader += "      States with principal quantum number in range [%d-%d]x[%d-%d],\n"%\
                            (self.n-self.nRange,self.n+self.nRange,\
                             self.nn-self.nRange,self.nn+self.nRange)
            commonHeader += "      AND whose orbital angular momentum (l) is in range [%d-%d] (i.e. %s-%s),\n"%\
                            (0,self.lrange,printStateLetter(0),printStateLetter(self.lrange))
            commonHeader += "      AND whose pair-state energy difference is at most %.3f GHz\n" %\
                             (self.energyDelta/1.e9)
            commonHeader += "      (energy difference is measured relative to original pair-state).\n"
        else:
            commonHeader += " ! Atom orientation and basis not yet set (this is set in defineBasis method).\n"

        if hasattr(self,"noOfEigenvectors"):
            commonHeader += " - Finding %d eigenvectors closest to the given pair-state\n"%\
                            self.noOfEigenvectors

            if self.drivingFromState[0]<0.1:
                commonHeader += " - State highlighting based on the relative contribution \n"+\
                "   of the original pair-state in the eigenstates obtained by diagonalization.\n"
            else:
                commonHeader += (" - State highlighting based on the relative driving strength \n"+\
                "   to a given energy eigenstate (energy level) from state\n"+\
                "   %s m_j =%d/2 with polarization q=%d.\n"%\
                 ( printStateString(*self.drivingFromState[0:3]),\
                 int(round(2.*self.drivingFromState[3])),
                 self.drivingFromState[4]))

        else:
            commonHeader += " ! Energy levels not yet found (this is done by calling diagonalise method).\n"



        if exportFormat=="csv":
            print("Exporting StarkMap calculation results as .csv ...")

            commonHeader += " - Export consists of three (3) files:\n"
            commonHeader += ("       1) %s,\n" % (fileBase+"_r."+exportFormat))
            commonHeader += ("       2) %s,\n" % (fileBase+"_energyLevels."+exportFormat))
            commonHeader += ("       3) %s.\n\n" % (fileBase+"_highlight."+exportFormat))

            filename = fileBase+"_r."+exportFormat
            np.savetxt(filename, \
                self.r, fmt='%.18e', delimiter=', ',\
                newline='\n', \
                header=(commonHeader + " - - - Interatomic distance, r (\mu m) - - -"),\
                comments='# ')
            print("   Interatomic distances (\mu m) saved in %s" % filename)

            filename = fileBase+"_energyLevels."+exportFormat
            headerDetails = " NOTE : Each row corresponds to eigenstates for a single specified interatomic distance"
            np.savetxt(filename, \
                self.y, fmt='%.18e', delimiter=', ',\
                newline='\n', \
                header=(commonHeader + ' - - - Energy (GHz) - - -\n' + headerDetails),\
                comments='# ')
            print("   Lists of energies (in GHz relative to the original pair-state energy)"+\
                  (" saved in %s" % filename))

            filename = fileBase+"_highlight."+exportFormat
            np.savetxt(filename, \
                self.highlight, fmt='%.18e', delimiter=', ',\
                newline='\n', \
                header=(commonHeader + ' - - - Highlight value (rel.units) - - -\n'+ headerDetails),\
                comments='# ')
            print("   Highlight values saved in %s" % filename)

            print("... data export finished!")
        else:
            raise ValueError("Unsupported export format (.%s)." % format)


    def _stateComposition(self,stateVector):
        contribution = np.absolute(stateVector)
        order = np.argsort(contribution,kind='heapsort')
        index = -1
        totalContribution = 0
        value = "$"
        while (index>-5) and (totalContribution<0.95):
            i = order[index]
            if (index!=-1 and stateVector[i]>0):
                value+= "+"
            value = value+ ("%.2f" % stateVector[i])+self._addState(*self.basisStates[i])
            totalContribution += contribution[i]**2
            index -= 1

        if totalContribution<0.999:
            value+="+\\ldots"
        return value+"$"

    def _addState(self,n1,l1,j1,mj1,n2,l2,j2,mj2,s):
        return "|%s %d/2,%s %d/2\\rangle" %\
             (printStateStringLatex(n1, l1, j1),int(2*mj1),\
              printStateStringLatex(n2, l2, j2),int(2*mj2))

    def plotLevelDiagram(self,  highlightColor='red'):
        """
            Plots pair state level diagram

            Call :obj:`showPlot` if you want to display a plot afterwards.

            Args:
                highlightColor (string): optional, specifies the colour used
                    for state highlighting

        """

        rvb = LinearSegmentedColormap.from_list('mymap',\
                                                 ['0.9', highlightColor])

        cNorm  = matplotlib.colors.Normalize(vmin=0., vmax=1.)

        print(" Now we are plotting...")
        self.fig, self.ax = plt.subplots(1, 1,figsize=(11.5,5.0))

        self.y = np.array(self.y)
        self.highlight = np.array(self.highlight)


        colorfulX = []
        colorfulY = []
        colorfulState = []

        for i in xrange(len(self.r)):

            for j in xrange(len(self.y[i])):
                colorfulX.append(self.r[i])
                colorfulY.append(self.y[i][j])
                colorfulState.append(self.highlight[i][j])

        colorfulState = np.array(colorfulState)
        sortOrder = colorfulState.argsort(kind='heapsort')
        colorfulX = np.array(colorfulX)
        colorfulY = np.array(colorfulY)

        colorfulX = colorfulX[sortOrder]
        colorfulY = colorfulY[sortOrder]
        colorfulState = colorfulState[sortOrder]

        self.ax.scatter(colorfulX,colorfulY,s=10,c=colorfulState,linewidth=0,\
                        norm=cNorm, cmap=rvb,zorder=2,picker=5)
        cax = self.fig.add_axes([0.91, 0.1, 0.02, 0.8])
        cb = matplotlib.colorbar.ColorbarBase(cax, cmap=rvb, norm=cNorm)

        if (self.drivingFromState[0] == 0):
            # colouring is based on the contribution of the original pair state here
            cb.set_label(r"$|\langle %s m_j=%d/2 , %s m_j=%d/2 | \mu \rangle |^2$" % \
                                 (printStateStringLatex(self.n, self.l,self.j),\
                                  int(round(2.*self.m1,0)),\
                                  printStateStringLatex(self.nn, self.ll,self.jj),\
                                  int(round(2.*self.m2,0)) ) )
        else:
            # colouring is based on the coupling to different states
            cb.set_label(r"$(\Omega_\mu/\Omega)^2$" )

        self.ax.set_xlabel(r"Interatomic distance, $R$ ($\mu$m)")
        self.ax.set_ylabel(r"Pair-state relative energy, $\Delta E/h$ (GHz)")

    def savePlot(self,filename="PairStateInteractions.pdf"):
        """
            Saves plot made by :obj:`plotLevelDiagram`

            Args:
                filename (:obj:`str`, optional): file location where the plot
                    should be saved
        """
        if (self.fig != 0):
            self.fig.savefig(filename,bbox_inches='tight')
        else:
            print("Error while saving a plot: nothing is plotted yet")
        return 0

    def showPlot(self, interactive = True):
        """
            Shows level diagram printed by
            :obj:`PairStateInteractions.plotLevelDiagram`

            By default, it will output interactive plot, which means that
            clicking on the state will show the composition of the clicked
            state in original basis (dominant elements)

            Args:
                interactive (bool): optional, by default it is True. If true,
                    plotted graph will be interactive, i.e. users can click
                    on the state to identify the state composition

            Note:
                interactive=True has effect if the graphs are explored in usual
                matplotlib pop-up windows. It doesn't have effect on inline
                plots in Jupyter (IPython) notebooks.


        """
        if interactive:
            self.ax.set_title("Click on state to see state composition")
            self.clickedPoint = 0
            self.fig.canvas.draw()
            self.fig.canvas.mpl_connect('pick_event', self._onPick)

        plt.show()
        return 0

    def _onPick(self,event):
        if isinstance(event.artist, matplotlib.collections.PathCollection):
            x = event.mouseevent.xdata
            y = event.mouseevent.ydata

            i = np.searchsorted(self.r,x)
            if i == len(self.r):
                i -= 1
            if ((i>0) and (abs(self.r[i-1]-x)<abs(self.r[i]-x))):
                i -=1

            j = 0
            for jj in xrange(len(self.y[i])):
                if (abs(self.y[i][jj]-y) < abs(self.y[i][j]-y)):
                    j = jj

            # now choose the most higlighted state in this area
            distance = abs(self.y[i][j]-y)*1.5
            for jj in xrange(len(self.y[i])):
                if (abs(self.y[i][jj]-y) < distance and \
                    (abs(self.highlight[i][jj])>abs(self.highlight[i][j]))):
                    j = jj

            if (self.clickedPoint!=0):
                self.clickedPoint.remove()

            self.clickedPoint, = self.ax.plot([self.r[i]], [self.y[i][j]],"bs",\
                                                 linewidth=0,zorder=3)


            self.ax.set_title("State = "+self.composition[i][j]+\
                              ("   Colourbar = %.2f"% self.highlight[i][j]),fontsize=11)

            event.canvas.draw()

    def getC6fromLevelDiagram(self,rStart,rStop,showPlot = False,\
                              minStateContribution=0.0):
        """
            Finds :math:`C_6` coefficient for original pair state.

            Function first finds for each distance in the range
            [ `rStart` , `rStop` ] the eigen state with highest contribution of
            the original state. One can set optional parameter
            `minStateContribution` to value in the range [0,1), so that function
            finds only states if they have contribution of the original state
            that is bigger then `minStateContribution`.

            Once original pair-state is found in the range of interatomic
            distances, from smallest `rStart` to the biggest `rStop`, function
            will try to perform fitting of the corresponding state energy
            :math:`E(R)` at distance :math:`R` to the function
            :math:`A+C_6/R^6` where :math:`A` is some offset.

            Args:
                rStart (float): smallest inter-atomic distance to be used for fitting
                rStop (float): maximum inter-atomic distance to be used for fitting
                showPlot (bool): If set to true, it will print the plot showing
                    fitted energy level and the obtained best fit. Default is
                    False
                minStateContribution (float): valid values are in the range [0,1).
                    It specifies minimum amount of the original state in the given
                    energy state necessary for the state to be considered for
                    the adiabatic continuation of the original unperturbed
                    pair state.

            Returns:
                float:
                    :math:`C_6` measured in :math:`\\text{GHz }\\mu\\text{m}^6`
                    on success; If unsuccessful returns False.

            Note:
                In order to use this functions, highlighting in
                :obj:`diagonalise` should be based on the original pair
                state contribution of the eigenvectors (that this,
                `drivingFromState` parameter should not be set, which
                corresponds to `drivingFromState` = [0,0,0,0,0]).
        """
        initialStateDetuning = []
        initialStateDetuningX = []

        fromRindex = -1
        toRindex = -1
        for br in xrange(len(self.r)):
            if (fromRindex == -1) and (self.r[br]>=rStart):
                fromRindex = br
            if (self.r[br]>rStop):
                toRindex = br-1
                break
        if (fromRindex != -1) and (toRindex == -1):
            toRindex = len(self.r)-1
        if (fromRindex == -1):
            print("\nERROR: could not find data for energy levels for interatomic")
            print("distances between %2.f and %.2f mu m.\n\n" % (rStart,rStop))
            return 0

        for br in xrange(fromRindex,toRindex+1):
            index = -1
            maxPortion = minStateContribution
            for br2 in xrange(len(self.y[br])):
                if (abs(self.highlight[br][br2])>maxPortion):
                    index = br2
                    maxPortion = abs(self.highlight[br][br2])
            if (index != -1) :
                initialStateDetuning.append(abs(self.y[br][index]))
                initialStateDetuningX.append(self.r[br])

        initialStateDetuning = np.log(np.array(initialStateDetuning))
        initialStateDetuningX = np.array(initialStateDetuningX)

        def c6fit(r,c6,offset):
            return np.log(c6/r**6+offset)

        try:
            popt,pcov = curve_fit(c6fit,\
                              initialStateDetuningX,\
                              initialStateDetuning,\
                              [1,0])
        except:
            print("ERROR: unable to find a fit for C6.")
            return False
        print("c6 = ",popt[0]," GHz /R^6 (mu m)^6")
        print("offset = ",popt[1])

        y_fit = []
        for val in initialStateDetuningX:
            y_fit.append(c6fit(val,popt[0],popt[1]))
        y_fit = np.array(y_fit)

        if showPlot:
            fig,ax = plt.subplots(1,1,figsize=(8.0,5.0))
            ax.loglog(initialStateDetuningX,np.exp(initialStateDetuning),\
                      "b-",lw=2,zorder=1)
            ax.loglog(initialStateDetuningX,np.exp(y_fit),\
                      "r--",lw=2,zorder=2)

            ax.legend(("calculated energy level","fitted model function"),\
                      loc=1,fontsize=10)

            ax.set_xlim(np.min(self.r),\
                        np.max(self.r) )
            ymin = np.min(initialStateDetuning)
            ymax = np.max(initialStateDetuning)
            ax.set_ylim( exp(ymin),exp(ymax))

            minorLocator = mpl.ticker.MultipleLocator(1)
            minorFormatter = mpl.ticker.FormatStrFormatter('%d')
            ax.xaxis.set_minor_locator(minorLocator)
            ax.xaxis.set_minor_formatter(minorFormatter)
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.set_xlabel(r"Interatomic distance, $r$ ($\mu$m)")
            ax.set_ylabel(r"Pair-state energy, $|E|$ (GHz)")
            ax.set_title(r"$C_6$ fit")
            plt.show()

        self.fitX = initialStateDetuningX
        self.fitY = initialStateDetuning
        self.fittedCurveY = y_fit

        return popt[0]

    def getC3fromLevelDiagram(self,rStart,rStop,showPlot = False,\
                              minStateContribution=0.0,\
                              resonantBranch = +1):
        """
            Finds :math:`C_3` coefficient for original pair state.

            Function first finds for each distance in the range
            [`rStart` , `rStop`] the eigen state with highest contribution of
            the original state. One can set optional parameter
            `minStateContribution` to value in the range [0,1), so that function
            finds only states if they have contribution of the original state
            that is bigger then `minStateContribution`.

            Once original pair-state is found in the range of interatomic
            distances, from smallest `rStart` to the biggest `rStop`, function
            will try to perform fitting of the corresponding state energy
            :math:`E(R)` at distance :math:`R` to the function
            :math:`A+C_3/R^3` where :math:`A` is some offset.

            Args:
                rStart (float): smallest inter-atomic distance to be used for fitting
                rStop (float): maximum inter-atomic distance to be used for fitting
                showPlot (bool): If set to true, it will print the plot showing
                    fitted energy level and the obtained best fit. Default is
                    False
                minStateContribution (float): valid values are in the range [0,1).
                    It specifies minimum amount of the original state in the given
                    energy state necessary for the state to be considered for
                    the adiabatic continuation of the original unperturbed
                    pair state.
                resonantBranch (int): optional, default +1. For resonant
                    interactions we have two branches with identical
                    state contributions. In this case, we will select only
                    positively detuned branch (for resonantBranch = +1)
                    or negatively detuned branch (fore resonantBranch = -1)
                    depending on the value of resonantBranch optional parameter
            Returns:
                float:
                    :math:`C_3` measured in :math:`\\text{GHz }\\mu\\text{m}^6`
                    on success; If unsuccessful returns False.

            Note:
                In order to use this functions, highlighting in
                :obj:`diagonalise` should be based on the original pair
                state contribution of the eigenvectors (that this,
                `drivingFromState` parameter should not be set, which
                corresponds to `drivingFromState` = [0,0,0,0,0]).
        """


        selectBranch = False
        if (abs(self.l-self.ll)==1):
            selectBranch = True
            resonantBranch = float(resonantBranch)

        initialStateDetuning = []
        initialStateDetuningX = []

        fromRindex = -1
        toRindex = -1
        for br in xrange(len(self.r)):
            if (fromRindex == -1) and (self.r[br]>=rStart):
                fromRindex = br
            if (self.r[br]>rStop):
                toRindex = br-1
                break
        if (fromRindex != -1) and (toRindex == -1):
            toRindex = len(self.r)-1
        if (fromRindex == -1):
            print("\nERROR: could not find data for energy levels for interatomic")
            print("distances between %2.f and %.2f mu m.\n\n" % (rStart,rStop))
            return False

        discontinuityDetected = False
        for br in xrange(toRindex,fromRindex-1,-1):
            index = -1
            maxPortion = minStateContribution
            for br2 in xrange(len(self.y[br])):
                if (abs(self.highlight[br][br2])>maxPortion) \
                    and (not selectBranch or (self.y[br][br2]*selectBranch>0.)):
                    index = br2
                    maxPortion = abs(self.highlight[br][br2])

            if (len(initialStateDetuningX)>2):
                slope1 = (initialStateDetuning[-1]-initialStateDetuning[-2])/\
                        (initialStateDetuningX[-1]-initialStateDetuningX[-2])
                slope2 =  (abs(self.y[br][index])-initialStateDetuning[-1])/\
                        (self.r[br]-initialStateDetuningX[-1])
                if abs(slope2)>3.*abs(slope1):
                    discontinuityDetected = True
            if (index != -1)and (not discontinuityDetected):
                initialStateDetuning.append(abs(self.y[br][index]))
                initialStateDetuningX.append(self.r[br])


        initialStateDetuning = np.log(np.array(initialStateDetuning))##*1e9
        initialStateDetuningX = np.array(initialStateDetuningX)


        def c3fit(r,c3,offset):
            return np.log(c3/r**3+offset)

        try:
            popt,pcov = curve_fit(c3fit,\
                              initialStateDetuningX,\
                              initialStateDetuning,\
                              [1,0])
        except:
            print("ERROR: unable to find a fit for C3.")
            return False
        print("c3 = ",popt[0]," GHz /R^3 (mu m)^3")
        print("offset = ",popt[1])

        y_fit = []

        for val in initialStateDetuningX:
            y_fit.append(c3fit(val,popt[0],popt[1]))
        y_fit = np.array(y_fit)

        if showPlot:
            fig,ax = plt.subplots(1,1,figsize=(8.0,5.0))
            ax.loglog(initialStateDetuningX,np.exp(initialStateDetuning),\
                      "b-",lw=2,zorder=1)
            ax.loglog(initialStateDetuningX,np.exp(y_fit),\
                      "r--",lw=2,zorder=2)

            ax.legend(("calculated energy level","fitted model function"),\
                      loc=1, fontsize=10)
            ax.set_xlim(np.min(self.r),\
                        np.max(self.r) )
            ymin = np.min(initialStateDetuning)
            ymax = np.max(initialStateDetuning)
            ax.set_ylim( exp(ymin),exp(ymax))

            minorLocator = mpl.ticker.MultipleLocator(1)
            minorFormatter = mpl.ticker.FormatStrFormatter('%d')
            ax.xaxis.set_minor_locator(minorLocator)
            ax.xaxis.set_minor_formatter(minorFormatter)
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.set_xlabel(r"Interatomic distance, $r$ ($\mu$m)")
            ax.set_ylabel(r"Pair-state energy, $|E|$ (GHz)")

            locatorStep = 1.
            while (locatorStep>(ymax-ymin)) and locatorStep>1.e-4:
                locatorStep /= 10.

            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(locatorStep))
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(locatorStep/10.))
            #ax.yaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter('%.3f'))

            ax.set_title(r"$C_3$ fit")

            plt.show()


        self.fitX = initialStateDetuningX
        self.fitY = initialStateDetuning
        self.fittedCurveY = y_fit

        return popt[0]

    def getVdwFromLevelDiagram(self,rStart,rStop,showPlot = False,\
                              minStateContribution=0.0):
        """
            Finds :math:`r_{\\rm vdW}` coefficient for original pair state.

            Function first finds for each distance in the range [`rStart`,`rStop`]
            the eigen state with highest contribution of the original state.
            One can set optional parameter `minStateContribution` to value in
            the range [0,1), so that function finds only states if they have
            contribution of the original state that is bigger then
            `minStateContribution`.

            Once original pair-state is found in the range of interatomic
            distances, from smallest `rStart` to the biggest `rStop`, function
            will try to perform fitting of the corresponding state energy
            :math:`E(R)` at distance :math:`R` to the function
            :math:`A+B\\frac{1-\\sqrt{1+(r_{\\rm vdW}/r)^6}}{1-\\sqrt{1+r_{\\rm vdW}^6}}`
             where :math:`A` and :math:`B` are some offset.

            Args:
                rStart (float): smallest inter-atomic distance to be used for fitting
                rStop (float): maximum inter-atomic distance to be used for fitting
                showPlot (bool): If set to true, it will print the plot showing
                    fitted energy level and the obtained best fit. Default is
                    False
                minStateContribution (float): valid values are in the range [0,1).
                    It specifies minimum amount of the original state in the given
                    energy state necessary for the state to be considered for
                    the adiabatic continuation of the original unperturbed
                    pair state.
            Returns:
                float:
                    :math:`r_{\\rm vdW}`  measured in :math:`\\mu\\text{m}`
                    on success; If unsuccessful returns False.

            Note:
                In order to use this functions, highlighting in
                :obj:`diagonalise` should be based on the original pair
                state contribution of the eigenvectors (that this,
                `drivingFromState` parameter should not be set, which
                corresponds to `drivingFromState` = [0,0,0,0,0]).
        """

        initialStateDetuning = []
        initialStateDetuningX = []

        fromRindex = -1
        toRindex = -1
        for br in xrange(len(self.r)):
            if (fromRindex == -1) and (self.r[br]>=rStart):
                fromRindex = br
            if (self.r[br]>rStop):
                toRindex = br-1
                break
        if (fromRindex != -1) and (toRindex == -1):
            toRindex = len(self.r)-1
        if (fromRindex == -1):
            print("\nERROR: could not find data for energy levels for interatomic")
            print("distances between %2.f and %.2f mu m.\n\n" % (rStart,rStop))
            return False

        discontinuityDetected = False;
        for br in xrange(toRindex,fromRindex-1,-1):
            index = -1
            maxPortion = minStateContribution
            for br2 in xrange(len(self.y[br])):
                if (abs(self.highlight[br][br2])>maxPortion):
                    index = br2
                    maxPortion = abs(self.highlight[br][br2])
            if (len(initialStateDetuningX)>2):
                slope1 = (initialStateDetuning[-1]-initialStateDetuning[-2])/\
                        (initialStateDetuningX[-1]-initialStateDetuningX[-2])
                slope2 =  (abs(self.y[br][index])-initialStateDetuning[-1])/\
                        (self.r[br]-initialStateDetuningX[-1])
                if abs(slope2)>3.*abs(slope1):
                    discontinuityDetected = True
            if (index != -1)and (not discontinuityDetected):
                initialStateDetuning.append(abs(self.y[br][index]))
                initialStateDetuningX.append(self.r[br])


        initialStateDetuning = np.log(abs(np.array(initialStateDetuning)))
        initialStateDetuningX = np.array(initialStateDetuningX)

        def vdwFit(r,offset,scale,vdw):
            return np.log(abs(offset+scale*(1.-np.sqrt(1.+(vdw/r)**6))/(1.-np.sqrt(1+vdw**6))))


        noOfPoints = len(initialStateDetuningX)
        print("Data points to fit = ",noOfPoints)

        try:
            popt,pcov = curve_fit(vdwFit,\
                              initialStateDetuningX,\
                              initialStateDetuning,\
                              [0,initialStateDetuning[noOfPoints//2],\
                               initialStateDetuningX[noOfPoints//2]])
        except:
            print("ERROR: unable to find a fit for van der Waals distance.")
            return False

        if (initialStateDetuningX[0]<popt[2]) or (popt[2]<initialStateDetuningX[-1]):
            print("WARNING: vdw radius seems to be outside the fitting range!")
            print("It's estimated to be around %.2f mu m from the current fit."%popt[2])

        print("Rvdw =  ",popt[2]," mu m")
        print("offset = ",popt[0],"\n scale = ",popt[1])

        y_fit = []

        for val in initialStateDetuningX:
            y_fit.append(vdwFit(val,popt[0],popt[1],popt[2]))
        y_fit = np.array(y_fit)

        if showPlot:
            fig,ax = plt.subplots(1,1,figsize=(8.0,5.0))
            ax.loglog(initialStateDetuningX,np.exp(initialStateDetuning),\
                      "b-",lw=2,zorder=1)
            ax.loglog(initialStateDetuningX,np.exp(y_fit),\
                      "r--",lw=2,zorder=2)

            ax.set_xlim(np.min(self.r),\
                        np.max(self.r) )
            ymin = np.min(initialStateDetuning)
            ymax = np.max(initialStateDetuning)
            ax.set_ylim( exp(ymin),exp(ymax))

            ax.axvline(x=popt[2],color="k")
            ax.text(popt[2],exp((ymin+ymax)/2.),r"$R_{vdw} = %.1f$ $\mu$m" % popt[2])

            minorLocator = mpl.ticker.MultipleLocator(1)
            minorFormatter = mpl.ticker.FormatStrFormatter('%d')
            ax.xaxis.set_minor_locator(minorLocator)
            ax.xaxis.set_minor_formatter(minorFormatter)
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.set_xlabel(r"Interatomic distance, $r$ ($\mu$m)")
            ax.set_ylabel(r"Pair-state energy, $|E|$ (GHz)")
            ax.legend(("calculated energy level","fitted model function"),\
                      loc=1,fontsize=10)

            plt.show()


        self.fitX = initialStateDetuningX
        self.fitY = initialStateDetuning
        self.fittedCurveY = y_fit

        return popt[2]

class StarkMapResonances:
    """
        Calculates pair state Stark maps for finding resonances

        Tool for finding conditions for Foster resonances. For a given pair
        state, in a given range of the electric fields, looks for the pair-state
        that are close in energy and coupled via dipole-dipole interactions
        to the original pair-state.

        See `Stark resonances example snippet`_.

        .. _`Stark resonances example snippet`:
                ././Rydberg_atoms_a_primer.html#Tuning-the-interaction-strength-with-electric-fields

        Args:
            atom (:obj:`AlkaliAtom`): ={ :obj:`alkali_atom_data.Lithium6`,
                :obj:`alkali_atom_data.Lithium7`,
                :obj:`alkali_atom_data.Sodium`,
                :obj:`alkali_atom_data.Potassium39`,
                :obj:`alkali_atom_data.Potassium40`,
                :obj:`alkali_atom_data.Potassium41`,
                :obj:`alkali_atom_data.Rubidium85`,
                :obj:`alkali_atom_data.Rubidium87`,
                :obj:`alkali_atom_data.Caesium` }
                 the first atom in the pair-state
            state1 ([int,int,float,float]): specification of the state
                of the first state as an array of values :math:`[n,l,j,m_j]`
            atom (:obj:`AlkaliAtom`): ={ :obj:`alkali_atom_data.Lithium6`,
                :obj:`alkali_atom_data.Lithium7`,
                :obj:`alkali_atom_data.Sodium`,
                :obj:`alkali_atom_data.Potassium39`,
                :obj:`alkali_atom_data.Potassium40`,
                :obj:`alkali_atom_data.Potassium41`,
                :obj:`alkali_atom_data.Rubidium85`,
                :obj:`alkali_atom_data.Rubidium87`,
                :obj:`alkali_atom_data.Caesium` }
                 the second atom in the pair-state
            state2 ([int,int,float,float]): specification of the state
                of the first state as an array of values :math:`[n,l,j,m_j]`

        Note:
            In checking if certain state is dipole coupled to the original
            state, only the highest contributing state is checked for dipole
            coupling. This should be fine if one is interested in resonances
            in weak fields. For stronger fields, one might want to include
            effect of coupling to other contributing base states.



    """

    def __init__(self,atom1,state1,atom2,state2):

        self.atom1 = atom1
        self.state1 = state1
        self.atom2 = atom2
        self.state2 = state2

        self.pairStateEnergy = (atom1.getEnergy(*state1[0:3])+\
                                atom2.getEnergy(*state2[0:3]))\
                            *C_e/C_h*1e-9

    def findResonances(self,nMin,nMax,maxL,eFieldList,energyRange=[-5.e9,+5.e9],\
             Bz=0, progressOutput=False):
        """
            Finds near-resonant dipole-coupled pair-states

            For states in range of principal quantum numbers [`nMin`,`nMax`]
            and orbital angular momentum [0,`maxL`], for a range of electric fields
            given by `eFieldList` function will find near-resonant pair states.

            Only states that are in the range given by `energyRange` will be
            extracted from the pair-state Stark maps.

            Args:
                nMin (int): minimal principal quantum number of the state to be
                    included in the StarkMap calculation
                nMax (int): maximal principal quantum number of the state to be
                    included in the StarkMap calculation
                maxL (int): maximum value of orbital angular momentum for the states
                    to be included in the calculation
                eFieldList ([float]): list of the electric fields (in V/m) for
                    which to calculate level diagram (StarkMap)
                Bz (float): optional, magnetic field directed along z-axis in
                    units of Tesla. Calculation will be correct only for weak
                    magnetic fields, where paramagnetic term is much stronger
                    then diamagnetic term. Diamagnetic term is neglected.
                energyRange ([float,float]): optinal argument. Minimal and maximal
                    energy of that some dipole-coupled state should have in order
                    to keep it in the plot (in units of Hz). By default it finds
                    states that are :math:`\pm 5` GHz
                progressOutput (:obj:`bool`, optional): if True prints the
                    progress of calculation; Set to false by default.
        """

        self.eFieldList = eFieldList
        self.Bz = Bz
        eMin = energyRange[0]*1.e-9  # in GHz
        eMax = energyRange[1]*1.e-9

        # find where is the original pair state

        sm1 = StarkMap(self.atom1)
        sm1.defineBasis(self.state1[0],self.state1[1],self.state1[2],\
                        self.state1[3], nMin, nMax, maxL, Bz=self.Bz,\
                        progressOutput = progressOutput)
        sm1.diagonalise(eFieldList,  progressOutput = progressOutput)
        if (self.atom2 is self.atom1) and \
            (self.state1[0]==self.state2[0]) and \
            (self.state1[1]==self.state2[1]) and \
            (abs(self.state1[2]-self.state2[2])<0.1) and \
            (abs(self.state1[3]-self.state2[3])<0.1):
            sm2 = sm1
        else:
            sm2 = StarkMap(self.atom2)
            sm2.defineBasis(self.state2[0],self.state2[1],self.state2[2],\
                            self.state2[3], nMin, nMax, maxL, Bz=self.Bz,\
                            progressOutput = progressOutput)
            sm2.diagonalise(eFieldList,  progressOutput = progressOutput)


        self.originalStateY = []
        self.originalStateContribution = []
        for i in xrange(len(sm1.eFieldList)):
            jmax1 = 0
            jmax2 = 0
            for j in xrange(len(sm1.highlight[i])):
                if (sm1.highlight[i][j]>sm1.highlight[i][jmax1]):
                    jmax1 = j
            for j in xrange(len(sm2.highlight[i])):
                if (sm2.highlight[i][j]>sm2.highlight[i][jmax2]):
                    jmax2 = j

            self.originalStateY.append(sm1.y[i][jmax1]+sm2.y[i][jmax2]-\
                                        self.pairStateEnergy)
            self.originalStateContribution.append((sm1.highlight[i][jmax1]+\
                                                   sm2.highlight[i][jmax2])/2.)

        # M= mj1+mj2 is conserved with dipole-dipole interaction

        dmlist1 = [1,0]
        if self.state1[3] != 0.5:
            dmlist1.append(-1)
        dmlist2 = [1,0]
        if self.state2[3] != 0.5:
            dmlist2.append(-1)

        n1 = self.state1[0]
        l1 = self.state1[1]+1
        j1 = self.state1[2]+1
        mj1 = self.state1[3]

        n2 = self.state2[0]
        l2 = self.state2[1]+1
        j2 = self.state2[2]+1
        mj2 = self.state2[3]

        self.fig, self.ax = plt.subplots(1,1,figsize=(9.,6))
        cm = LinearSegmentedColormap.from_list('mymap', ['0.9', 'red','black'])
        cNorm  = matplotlib.colors.Normalize(vmin=0., vmax=1.)

        self.r = []
        self.y = []
        self.composition = []

        for dm1 in dmlist1:
            sm1.defineBasis(n1,l1,j1,mj1+dm1, nMin, nMax, maxL, Bz=self.Bz,\
                        progressOutput = progressOutput)
            sm1.diagonalise(eFieldList,  progressOutput = progressOutput)

            for dm2 in dmlist2:
                sm2.defineBasis(n2,l2,j2,mj2+dm2, nMin, nMax, maxL, Bz=self.Bz,\
                            progressOutput = progressOutput)
                sm2.diagonalise(eFieldList,  progressOutput = progressOutput)

                for i in xrange(len(sm1.eFieldList)):
                    yList = []
                    compositionList = []
                    if progressOutput:
                        sys.stdout.write("\rE=%.2f V/m " % sm1.eFieldList[i])
                        sys.stdout.flush()
                    for j in xrange(len(sm1.y[i])):
                        for jj in xrange(len(sm2.y[i])):
                            energy = sm1.y[i][j]+sm2.y[i][jj]\
                                    -self.pairStateEnergy
                            statec1 = sm1.basisStates[sm1.composition[i][j][0][1]]
                            statec2 = sm2.basisStates[sm2.composition[i][jj][0][1]]
                            if (energy>eMin) and (energy<eMax) and\
                                (abs(statec1[1]-self.state1[1])==1) and\
                                (abs(statec2[1]-self.state2[1])==1):
                                # add this to PairStateMap
                                yList.append(energy)
                                compositionList.append([
                                    sm1._stateComposition(sm1.composition[i][j]),
                                    sm2._stateComposition(sm2.composition[i][jj])])

                    if (len(self.y)<=i):
                        self.y.append(yList)
                        self.composition.append(compositionList)
                    else:
                        self.y[i].extend(yList)
                        self.composition[i].extend(compositionList)

                if progressOutput:
                    print("\n")


        for i in xrange(len(sm1.eFieldList)):
            self.y[i] = np.array(self.y[i])
            self.composition[i] = np.array(self.composition[i])
            self.ax.scatter([sm1.eFieldList[i]/100.]*len(self.y[i]),\
                            self.y[i],c="k",\
                                s=5,norm=cNorm, cmap=cm,lw=0,picker=5)
        self.ax.plot(sm1.eFieldList/100.,\
                            self.originalStateY,"r-",lw=1)
        self.ax.set_ylim(eMin,eMax)
        self.ax.set_xlim(min(self.eFieldList)/100.,\
                         max(self.eFieldList)/100.)
        self.ax.set_xlabel("Electric field (V/cm)")
        self.ax.set_ylabel("Pair-state relative energy, $\Delta E/h$ (GHz)")

    def showPlot(self,interactive = True):
        """
            Plots initial state Stark map and its dipole-coupled resonances

            Args:

                interactive (optional, bool): if True (by default) points on plot
                    will be clickable so that one can find the state labels
                    and their composition (if they are heavily admixed).
        """

        if (self.fig != 0):
            if interactive:
                self.ax.set_title("Click on state to see state composition")
                self.clickedPoint = 0
                self.fig.canvas.draw()
                self.fig.canvas.mpl_connect('pick_event', self._onPick)
            plt.show()
        else:
            print("Error while showing a plot: nothing is plotted yet")

    def _onPick(self,event):
        if isinstance(event.artist, matplotlib.collections.PathCollection):

            x = event.mouseevent.xdata*100.
            y = event.mouseevent.ydata

            i = np.searchsorted(self.eFieldList,x)
            if i == len(self.eFieldList):
                i -= 1
            if ((i>0) and (abs(self.eFieldList[i-1]-x)<\
                           abs(self.eFieldList[i]-x))):
                i -=1

            j = 0
            for jj in xrange(len(self.y[i])):
                if (abs(self.y[i][jj]-y) < abs(self.y[i][j]-y)):
                    j = jj

            if (self.clickedPoint!=0):
                self.clickedPoint.remove()

            self.clickedPoint, = self.ax.plot([self.eFieldList[i]/100.],\
                                               [self.y[i][j]],"bs",\
                                                 linewidth=0,zorder=3)

            atom1 = self.atom1.elementName
            atom2 = self.atom2.elementName
            composition1 = str(self.composition[i][j][0])
            composition2 = str(self.composition[i][j][1])
            self.ax.set_title(("[%s,%s]=[" %(atom1,atom2))+\
                              composition1+","+\
                              composition2+"]",
                             fontsize=10)

            event.canvas.draw()

    def _onPick2(self,xdata,ydata):
        if True:

            x = xdata*100.
            y = ydata

            i = np.searchsorted(self.eFieldList,x)
            if i == len(self.eFieldList):
                i -= 1
            if ((i>0) and (abs(self.eFieldList[i-1]-x)<\
                           abs(self.eFieldList[i]-x))):
                i -=1

            j = 0
            for jj in xrange(len(self.y[i])):
                if (abs(self.y[i][jj]-y) < abs(self.y[i][j]-y)):
                    j = jj

            if (self.clickedPoint!=0):
                self.clickedPoint.remove()

            self.clickedPoint, = self.ax.plot([self.eFieldList[i]/100.],\
                                               [self.y[i][j]],"bs",\
                                                 linewidth=0,zorder=3)

            atom1 = self.atom1.elementName
            atom2 = self.atom2.elementName
            composition1 = str(self.composition[i][j][0])
            composition2 = str(self.composition[i][j][1])
            self.ax.set_title(("[%s,%s]=[" %(atom1,atom2))+\
                              composition1+","+\
                              composition2+"]",
                             fontsize=10)

            #event.canvas.draw()
