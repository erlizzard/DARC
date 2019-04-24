
from .alkali_atom_functions import *
from .earth_alkali_atom_functions import *
from scipy.constants import physical_constants, pi , epsilon_0, hbar
from scipy.constants import e as C_e
from scipy.constants import h as C_h
import numpy as np
from scipy.special import factorial


class PairStateInteractionDifferentSpecies:

    def __init__(self,atom1, atom2, n,l,j,nn,ll,ss,jj,m1,m2,interactionsUpTo=1):

        '''
        HOW MANY m values do we need to include?
        Can we make this as general as possible?
        How do interactions with atoms with many electrons in the outer shell interact?
        # AKA do we need a 1-1, 1-2, 2-2, 1-many, 2- many, many-many interactions?
        Will the models change enough that we will have to create lots of the same methods for each series
        '''
        self.atom1 = atom1  #the AlkaliAtom
        self.atom2 = atom2  #the EarthAlkaliAtom
        self.n = n #: pair-state definition: principal quantum number of the first atom
        self.l = l #: pair-state definition: orbital angular momentum of the first atom
        self.j = j #: pair-state definition: total angular momentum of the first atom
        self.nn = nn #: pair-state definition: principal quantum number of the second atom
        self.ll = ll #: pair-state definition: orbital angular momentum of the second atom
        self.ss = ss
        self.jj = jj #: pair-state definition: total angular momentum of the second atom
        self.m1 = m1 #: pair-state definition: projection of the total ang. momentum for the *first* atom
        self.m2 = m2 #: pair-state definition: projection of the total angular momentum for the *second* atom
        self.interactionsUpTo = interactionsUpTo


        fcoef = lambda l1,l2,m: factorial(l1+l2)/(factorial(l1+m)*factorial(l1-m)*factorial(l2+m)*factorial(l2-m))**0.5
        x = self.interactionsUpTo
        self.fcp = np.zeros((x+1,x+1,2*x+1))
        for c1 in range(1,x+1):
            for c2 in range(1,x+1):
                for p in range(-min(c1,c2),min(c1,c2)+1):
                    self.fcp[c1,c2,p+x] = fcoef(c1,c2,p)

    def getEnergyDefect2(self,n1,l1,j1,nn1,ll1,ss1,jj1):
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
        return ((self.atom1.getEnergy(n1,l1,j1)+ self.atom1.ionisationEnergy)/(27.21138602)+\
                                     ( self.atom2.getEnergy(nn1,ll1,ss1,jj1))-\
                                    ((self.atom1.getEnergy(self.n,self.l,self.j)+ self.atom1.ionisationEnergy)/(27.21138602) +(self.atom2.getEnergy(self.nn, self.ll, self.ss, self.jj))))#

    def _atomLightAtomCoupling(self,n,l,j,nn,ll,ss,jj,n1,l1,j1,nn1,ll1,ss1,jj1):
        """
            Calculates radial part of atom-light coupling

            This function might seem redundant, since similar function exist for
            each of the atoms. However, function that is not connected to specific
            atomic species is provided in order to provides route to implement
            inter-species coupling in the future.
        """
        # determine coupling
        dl = abs(l-l1)
        dj = abs(j-j1)
        c1 = 0
        if dl==1 and (dj<1.1):
            c1 = 1  # dipole coupling
        elif (dl==0 or dl==2 or dl==1) and(dj<2.1):
            c1 = 2  # quadrupole coupling
        else:
            return False
        dl = abs(ll-l2)
        dj = abs(jj-j2)
        c2 = 0
        if dl==1 and (dj<1.1):
            c2 = 1  # dipole coupling
        elif (dl==0 or dl==2 or dl==1) and(dj<2.1):
            c2 = 2  # quadrupole coupling
        else:
            return False

        radial1 = self.atom1.getRadialCoupling(n,l,j,n1,l1,j1)
        radial2 = self.atom2.getRadialCoupling(nn,ll,ss,jj,nn1,ll1,ss1,jj1)

        ## TO-DO: check exponent of the Boht radius (from where it comes?!)

        coupling = C_e**2/(4.0*pi*epsilon_0)*radial1*radial2*\
                    (physical_constants["Bohr radius"][0])**(c1+c2)
        return coupling


    def getC6term(self,n1,l1,j1, nn1,ll1,ss1,jj1):
        d1 = self.atom1.getRadialMatrixElementSemiClassical(self.n,self.l,self.j,n1,l1,j1)
        d2 = self.atom2.getRadialMatrixElementSemiClassical(self.nn,self.ll,self.ss,self.jj,nn1,ll1,ss1,jj1)


        d1d2 =d1*d2
        #print('d1d2',d1d2)
        #
        print('the energy in ev',(self.atom1.getEnergy(n1,l1,j1)), self.atom2.getEnergy(nn1,ll1,ss1,jj1),\
                (self.atom1.getEnergy(self.n,self.l,self.j)), self.atom2.getEnergy(self.nn, self.ll, self.ss, self.jj))

        print('the energy in hartrees',(self.atom1.getEnergy(n1,l1,j1)+ self.atom1.ionisationEnergy)/(27.21138602),( self.atom2.getEnergy(nn1,ll1,ss1,jj1)+self.atom2.ionisationEnergycm/219475),\
                (self.atom1.getEnergy(self.n,self.l,self.j)+ self.atom1.ionisationEnergy)/(27.21138602), (self.atom2.getEnergy(self.nn, self.ll, self.ss, self.jj)+self.atom2.ionisationEnergycm/219475))#+ self.atom1.ionisationEnergy
        #d1d2**2
        return -d1d2**2/((self.atom1.getEnergy(n1,l1,j1)+ self.atom1.ionisationEnergy)/(27.21138602)+\
                                     ( self.atom2.getEnergy(nn1,ll1,ss1,jj1))-\
                                    ((self.atom1.getEnergy(self.n,self.l,self.j)+ self.atom1.ionisationEnergy)/(27.21138602) +(self.atom2.getEnergy(self.nn, self.ll, self.ss, self.jj))))#+self.atom2.ionisationEnergycm/219475)))  #+self.atom2.ionisationEnergycm/219475)-\

    def anotherC6(self,n1,l1,j1,nn1,ll1,ss1,jj1, mj1, mj2,q ):
        dipole_cs = self.atom1.getDipoleMatrixElement(self.n, self.l, self.j, self.m1,n1,l1,j1,mj1,q, True)
        dipole_sr = self.atom2.getRadialMatrixElementSemiClassical( self.nn,self.ll,self.ss,self.jj, nn1,ll1,ss1,jj1)*self.atom2.getAngularMatrixElement(self.ll,self.ss,self.jj,self.m2,ll1,ss1,jj1,mj2)
        print(self.getEnergyDefect2(n1,l1,j1,nn1,ll1,ss1,jj1))
        print(((self.atom1.getEnergy(n1,l1,j1)+ self.atom1.ionisationEnergy)/(27.21138602)+\
                                     ( self.atom2.getEnergy(nn1,ll1,ss1,jj1))-\
                                    ((self.atom1.getEnergy(self.n,self.l,self.j)+ self.atom1.ionisationEnergy)/(27.21138602) +(self.atom2.getEnergy(self.nn, self.ll, self.ss, self.jj)))))

        print(abs(dipole_cs*dipole_sr))
        print(-(abs(dipole_cs*dipole_sr))**2)
        return -(abs(dipole_cs*dipole_sr))**2/ ((self.atom1.getEnergy(n1,l1,j1)+ self.atom1.ionisationEnergy)/(27.21138602)+\
                                     ( self.atom2.getEnergy(nn1,ll1,ss1,jj1))-\
                                    ((self.atom1.getEnergy(self.n,self.l,self.j)+ self.atom1.ionisationEnergy)/(27.21138602) +(self.atom2.getEnergy(self.nn, self.ll, self.ss, self.jj))))#

        #self.getEnergyDefect2(n1,l1,j1,nn1,ll1,ss1,jj1)

    def __getAngularMatrix_M(self,l,j,ll,jj,l1,j1,ll1,jj1):
        # did we already calculated this matrix?

        #self.c.execute('''SELECT ind FROM pair_angularMatrix WHERE
        #     l1 = ? AND j1_x2 = ? AND
        #     l2 = ? AND j2_x2 = ? AND
        #     l3 = ? AND j3_x2 = ? AND
        #     l4 = ? AND j4_x2 = ?
        #     ''',(l,j*2,ll,jj*2,l1,j1*2,l2,j2*2))

        #index = self.c.fetchone()
        #if (index):
        #    return self.savedAngularMatrix_matrix[index[0]]

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
        dl = abs(ll-ll1)
        dj = abs(jj-jj1)
        c2 = 0
        if dl==1 and (dj<1.1):
            c2 = 1  # dipole coupling
        elif (dl==0 or dl==2 or dl==1):
            c2 = 2  # quadrupole coupling
        else:
            raise ValueError("error in __getAngularMatrix_M")
            exit()


        am = np.zeros((int(round((2*j1+1)*(2*jj1+1),0)),\
                    int(round((2*j+1)*(2*jj+1),0))),dtype=np.float64)


        j1range = np.linspace(-j1,j1,round(2*j1)+1)
        jj1range = np.linspace(-jj1,jj1,round(2*jj1)+1)
        jrange = np.linspace(-j,j,int(2*j)+1)
        jjrange = np.linspace(-jj,jj,int(2*jj)+1)

        for m1 in j1range:
            for m2 in jj1range:
                # we have chosen the first index
                index1 = int(round(m1*(2.0*jj1+1.0)+m2+(j1*(2.0*jj1+1.0)+jj1),0))
                for m in jrange:
                    for mm in jjrange:
                        # we have chosen the second index
                        index2 = int(round(m*(2.0*jj+1.0)+mm+(j*(2.0*jj+1.0)+jj),0))


                        # angular matrix element from Sa??mannshausen, Heiner, Merkt, Fr??d??ric, Deiglmayr, Johannes
                        # PRA 92: 032505 (2015)
                        elem = (-1.0)**(j+jj+1.0+l1+ll1)*CG(l,0,c1,0,l1,0)*CG(ll,0,c2,0,ll1,0)
                        elem = elem*sqrt((2.0*l+1.0)*(2.0*ll+1.0))*sqrt((2.0*j+1.0)*(2.0*jj+1.0))
                        elem = elem*Wigner6j(l, 0.5, j, j1, c1, l1)*Wigner6j(ll,0.5,jj,jj1,c2,ll1)

                        #------LIZZY: why are we summing over the polarisations--------------
                        sumPol = 0.0  # sum over polarisations
                        limit = min(c1,c2)
                        for p in xrange(-limit,limit+1):
                            sumPol = sumPol + \
                                     self.fcp[c1,c2,p + self.interactionsUpTo] * \
                                     CG(j,m,c1,p,j1,m1) *\
                                     CG(jj,mm,c2,-p,jj1,m2)
                        am[index1,index2] = elem*sumPol

        #index = len(self.savedAngularMatrix_matrix)

        #self.c.execute(''' INSERT INTO pair_angularMatrix
        #                    VALUES (?,?, ?,?, ?,?, ?,?, ?)''',\
        #               (l,j*2,ll,jj*2,l1,j1*2,l2,j2*2,index) )
        #self.conn.commit()

        #self.savedAngularMatrix_matrix.append(am)
        #self.savedAngularMatrixChanged = True

        return am

    def getC6perturbatively(self,theta,phi,n1,l1,j1,nn1,ll1,ss1,jj1,energyDelta):
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

             Returns:
                 float: :math:`C_6` measured in :math:`\\text{GHz }\\mu\\text{m}^6`

             Example:
                 If we want to quickly calculate :math:`C_6` for two Rubidium
                 atoms in state :math:`62 D_{3/2} m_j=3/2`, positioned in space
                 along the shared quantization axis::

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
                     # plot results
                     plot(thetaList/pi,c6,"b-")
                     title("Rb, pairstate  60 $D_{5/2},m_j = 5/2$, 60 $D_{5/2},m_j = 5/2$")
                     xlabel(r"$\Theta /\pi$")
                     ylabel(r"$C_6$ (GHz $\mu$m${}^6$")
                     show()

         """
         #self.__initializeDatabaseForMemoization()

         # ========= START OF THE MAIN CODE ===========
         C6 = 0.

         # wigner D matrix allows calculations with arbitrary orientation of
         # the two atoms
         wgd = wignerDmatrix(theta,phi)
         # state that we are coupling
         statePart1 = singleAtomState(self.j, self.m1)
         statePart2 = singleAtomState(self.jj, self.m2)
         # rotate individual states
         dMatrix = wgd.get(self.j)
         statePart1 = dMatrix.dot(statePart1)

         dMatrix = wgd.get(self.jj)
         statePart2 = dMatrix.dot(statePart2)
         stateCom = compositeState(statePart1, statePart2)


         #---------------LIZZY: WHAT DOES THE LIMITING BASIS ON Mj mean?---------------------------------
         # any conservation?
         limitBasisToMj = False
         if theta<0.001:
             limitBasisToMj = True  # Mj will be conserved in calculations
         originalMj = self.m1+self.m2
         # this numbers are conserved if we use only dipole-dipole interactions
         Lmod2 = ((self.l+self.ll) % 2)

         # find nearby states -------------------LIZZY: Do we need to find the nearby states here?--------------------------------
         lmin1 = self.l-1
         if lmin1 < -0.1:
             lmin1 = 1
         lmin2 = self.ll-1
         if lmin2 < -0.1:
             lmin2 = 1

        #---------------------LIZZY: Have removed the sum and the checks just going to do a single calcualtion -----------------
        #for n1 in xrange(max(self.n-nRange,1),self.n+nRange+1):
        #     for n2 in xrange(max(self.nn-nRange,1),self.nn+nRange+1):
        #         for l1 in xrange(lmin1,self.l+2,2):
        #             for l2 in xrange(lmin2,self.ll+2,2):
        #                 j1 = l1-0.5
        #                 if l1 == 0:
        #                     j1 = 0.5
        #                 while j1 <= l1+0.5+0.1:
        #                     j2 = l2-0.5
        #                     if l2 == 0:
        #                         j2 = 0.5

        #                     while j2 <= l2+0.5+0.1:

        #LIZZY: Have to change this because it is only dependent on one atom
         getEnergyDefect = self.getEnergyDefect2(self.atom1, self.atom2, self.n,self.l,self.j,\
                                           self.nn,self.ll,self.ss,self.jj,\
                                           n1,l1,j1,\
                                           nn1,ll1,ss1,jj1)
        #LIZZY: Can I get rid of this condition? A
        # if abs(getEnergyDefect)<energyDelta  \
        #     and (not (self.interactionsUpTo==1) or\
        #          (Lmod2 == ((l1+l2)%2) )) :
        #     getEnergyDefect = getEnergyDefect*1.0e-9 # GHz

             # calculate radial part
         couplingStregth = self.atom1.getRadialMatrixElement(self.n, self.l, self.j,n1,l1,j1)*self.atom2.getRadialMatrixElementSemiClassical( self.nn,self.ll,self.ss,self.jj, nn1,ll1,ss1,jj1)

            #pairState2 = "|"+printStateString(n1,l1,j1)+\
            #     ","+printStateString(n2,l2,j2)+">"

             # include relevant mj and add contributions
         for m1c in np.linspace(j1,-j1,round(1+2*j1)):
             for m2c in np.linspace(jj1,-jj1,round(1+2*jj1)):
                 if ((not limitBasisToMj) or (abs(originalMj-m1c-m2c)==0) ):
                     # find angular part
                     statePart1 = singleAtomState(j1, m1c)
                     statePart2 = singleAtomState(j2, m2c)
                     # rotate individual states
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

                     angularFactor = conjugate(stateCom2.T).dot(d.dot(stateCom))
                     angularFactor = real(angularFactor[0,0])

                     C6 += (couplingStregth**2*angularFactor)*(atom2.getAngularMatrixElement(self.ll,self.ss,self.jj,0,self.ll1,self.ss1,self.jj1,0)/getEnergyDefect)

                         #do I get fo the square
                        # I think h



         # ========= END OF THE MAIN CODE ===========
        # self.__closeDatabaseForMemoization()
         return C6
