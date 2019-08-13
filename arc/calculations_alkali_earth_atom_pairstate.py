'''This is TEMPORARY. This is going to be properly integreated into the paristate
but I am not sure how to do this yet. Nikola said he would be willing to help.
This is just to get something to compare to Andeas work
'''
import numpy as np
import itertools
import scipy
import sys
import inspect, os #for obtaining current file's path
np.set_printoptions(threshold=sys.maxsize)
#from .earth_alkali_atom_functions import EarthAlkaliAtom
from .earth_alkali_atom_data import StrontiumI


class EarthAlkaliPairstateInteraction:

    def __init__(self, atom):
        self.atom = atom  #: atom type

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
    def getIntermediatePairstates(self,n1,l1,s1,j1, m1, n2, l2,s2,j2,m2 ):
        '''This is going to take quantum numbers and return
        an array consisiting of all possible combinations of the valid pairstates

        This is going to be used to work out intermediate pairstates between inital and final values of M,
        this means that the spin
        '''
        pairstates_a = self.getIntermediateQuantumNumbers(n1,l1,s1,j1,m1,3)
        pairstates_b = self.getIntermediateQuantumNumbers(n2,l2,s2,j2,m2,3)
        out = np.array(list(itertools.product(pairstates_a, pairstates_b))).reshape(-1,10)
        return out
    def commonStates(self, arr1, arr2):
        nrows, ncols = arr1.shape
        dtype={'names':['f{}'.format(i) for i in range(ncols)],
       'formats':ncols * [arr1.dtype]}

        common = np.intersect1d(arr1.view(dtype), arr2.view(dtype))

        return common
    def getC6term(self,n1,l1,s1,j1,m1_ini, m1_fini, n2,l2,s2,j2, m2_ini,m2_fini, n1_p,l1_p,s1_p, j1_p, m1_p, n2_p,l2_p, s2_p,j2_p,m2_p, theta, phi):
        
        print('n,l,j,m', n1,l1,j1, m1_ini)
        print('nn,ll,jj,mm',n2,l2,j2,m2_ini)
        print('n1,l1,j1,m1', n1_P,l1_p,j1_p,m1_fini)
        print('n2,l2,j2,m2', n1,L2_i,J2_i,M2b)
    
          
        forster_defect = E(n1_i, L1_i, J1_i, S) + E(n2_i, L2_i, J2_i, S) - E(n1, L1, J1, S) - E(n2, L2, J2, S)
       
        #radial matrix element
        rad = (self.atom.getRadialMatrixElement(n1,l1,s1,j1,n1_p,l1_p,s1_p,j1_p)*self.atom.getRadialMatrixElement(n2,l2,s2,j2,n2_p,l2_p,s2_p,j2_p))**2
        #angular part for the first atom and intermeidate state
        rad *= self.atom.getAngularMatrixElementSrSr( l1_p, j1_p, m1_p, l2_p, j2_p, m2_p,l1,j1,m1_ini,l2,j2,m2_ini,s1,1,1 ,theta, phi )
        #print('Angular Part1', self.atom.getAngularMatrixElementSrSr( l1_p, s1_p, j1_p, m1_p, l2_p, s2_p, j2_p, m2_p,l1,s1,j1,m1_ini,l2,s2,j2,m2_ini, theta, phi ))
         #angular part for the second atom and
        rad *= self.atom.getAngularMatrixElementSrSr( l1, j1, m1_fini, l2, j2, m2_fini, l1_p, j1_p, m1_p, l2_p, j2_p, m2_p,s1,1,1,theta, phi)

        return rad/(self.atom.getEnergyDefect2(n1,l1,j1,n2,l2,j2,n1_p,l1_p,j1_p, n2_p, l2_p, j2_p, s1 )/2) #times 2 because of au being 2Rb
    def getC6Matrix(self,n,l,s,j,n1,l1,s1,j1,theta,phi):
        dim = (2*j1 +1)*(2*j+1)
        mat = np.zeros((dim,dim), dtype = 'complex')
        m_list = np.arange(-j, j+1)
        m1_list = np.arange(-j1, j1+1)

        #Make a list of all possible combinations of ms
        pairs = np.array(list(itertools.product(m_list, m1_list)))
        #sum these to work out the omegas
        omegas = np.sum(pairs, axis = 1 ).reshape((-1,1))

        concat = np.concatenate((omegas, pairs), axis = 1)

        sorted_pairs = concat[concat[:,0].argsort()][:,1:]

        #the symmetric states were taken from Angelas
        if j==1: #e.g.3S1
            symindices = [[0,0],[1,1],[1,2],[3,3],[3,4],[3,5],[4,4]]
        elif j==2: #e.g. 3D2
            symindices = [[0,0],[1,1],[1,2],[3,3],[3,4],[3,5],[4,4],[6,6],[6,7],[6,8],[6,9],\
            [7,7],[7,9],[10,10],[10,11],[10,12],[10,13],[10,14],[11,11],[11,12],[11,13],[12,12]]

        if(s ==0 and j == 0 ): symindices= [[0,0]]

        #my code again
        for i,k in symindices:
            print('i,j',i,k)
            m_fin, mm_fin = sorted_pairs[i]
            m_ini, mm_ini = sorted_pairs[k]

            print('Inital m', m_ini, mm_ini)
            print('Final m',m_fin, mm_fin)
            states_a = self.getIntermediatePairstates(n,l,s,j,m_ini, n1,l1,s1,j1, mm_ini)
            states_b = self.getIntermediatePairstates(n,l,s,j,m_fin, n1,l1,s1,j1, mm_fin)

            common = self.commonStates(states_a, states_b)
            print('Number of common states',common.shape)
            terms = [-self.getC6term(n,l,s,j,m_ini,m_fin, n1,l1,s1,j1,mm_ini, mm_fin, n_i, l_i, s_i, j_i, m_i, n1_i, l1_i, s1_i, j1_i, m1_i,theta, phi) \
                    for n_i, l_i, s_i, j_i, m_i, n1_i, l1_i, s1_i, j1_i, m1_i in common]
            mat[i,k] = np.sum(terms, axis = 0)

            #print(mat[i,k])
        #work out the states common to eachother
         #fill in other elements by symmetry
         #FROM ANDREA
        if j==1:
            mat[8,8] = mat[0,0]
            mat[2,2], mat[6,6], mat[7,7] = np.repeat(mat[1,1], 3)
            mat[2,1], mat[6,7], mat[7,6] = np.repeat(mat[1,2], 3)
            mat[5,5] = mat[3,3]
            mat[4,3], mat[4,5], mat[5,4] = np.repeat(mat[3,4], 3)
            mat[5,3] = mat[3,5]
        elif j==2:
            mat[24,24] = mat[0,0]
            mat[2,2], mat[22,22], mat[23,23] = np.repeat(mat[1,1], 3)
            mat[2,1], mat[22,23], mat[23,22] = np.repeat(mat[1,2], 3)
            mat[5,5], mat[19,19], mat[21,21] = np.repeat(mat[3,3], 3)
            mat[4,3], mat[4,5], mat[5,4], mat[19,20], mat[20,19], mat[20,21], mat[21,20] = np.repeat(mat[3,4], 7)
            mat[5,3], mat[19,21], mat[21,19] = np.repeat(mat[3,5], 3)
            mat[20,20] = mat[4,4]
            mat[8,8], mat[16,16], mat[18,18] = np.repeat(mat[6,6], 3)
            mat[7,6], mat[8,9], mat[9,8], mat[15,16], mat[16,15], mat[18,17], mat[17,18] = np.repeat(mat[6,7], 7)
            mat[8,6], mat[16,18], mat[18,16] = np.repeat(mat[6,8], 3)
            mat[7,8], mat[8,7], mat[9,6], mat[15,18], mat[16,17], mat[17,16], mat[18,15] = np.repeat(mat[6,9], 7)
            mat[9,9], mat[15,15], mat[17,17] = np.repeat(mat[7,7], 3)
            mat[9,7], mat[15,17], mat[17,15] = np.repeat(mat[7,9], 3)
            mat[14,14] = mat[10,10]
            mat[11,10], mat[13,14], mat[14,13] = np.repeat(mat[10,11], 3)
            mat[12,10], mat[12,14], mat[14,12] = np.repeat(mat[10,12], 3)
            mat[11,14], mat[13,10], mat[14,11] = np.repeat(mat[10,13], 3)
            mat[14,10] = mat[10,14]
            mat[13,13] = mat[11,11]
            mat[12,11], mat[12,13], mat[13,12] = np.repeat(mat[11,12], 3)
            mat[13,11] = mat[11,13]

        print_results = True
        #path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
        #pathup = os.path.dirname(path) #one layer above
        pathup = os.getcwd()
        if print_results:
            print ('mat', mat)

        #saves C6 matrix into csv file
        if l==0 and j==1 and s==1: #3S1
            filename = 'Sr_n%i_3S1_C6_matrices' %(n1)
        elif l==2 and j==2 and s==1: #3D2
            filename = 'Sr_n%i_3D2_C6_matrices' %(n1)
        elif l==2 and j==1 and s==1: #3D1
            filename = 'Sr_n%i_3D1_C6_matrices' %(n1)
        elif l==2 and j==2 and s==0: #1D2
            filename = 'Sr_n%i_1D2_C6_matrices' %(n1)
        csvname = '%.4f,%.4f' %(theta,phi)

        #different computers have different slashes in paths...
        #if os.name=='posix':#mac - my laptop
        #    slash = '/'
        #elif os.name=='nt':#windows - library
        #    slash='\\'#double \\ so I don't confuse Python with my syntax
        #fullpath = '%s%s%s%s%s.csv' %(pathup,slash,filename,slash,csvname)

        #print(fullpath)
        #if os.path.exists(fullpath)==False:#only save if this file doesn't exist
            #this avoids clearing data by accident
        #    with open(fullpath, 'xb') as f:
        #        np.savetxt(f, mat, delimiter=",")
        #return mat

    def getEigenVals(self, n,l,s,j, n1,l1,s1,j1 ,theta, phi):
        print('Running')
        mat = self.getC6Matrix(n,l,s,j,n1,l1,s1,j1,theta,phi)
        print('Finishing Matrix')
        return np.linalg.eigvals(mat)
#print(EarthAlkaliPairstateInteraction(StrontiumI()).getEigenVals(50,0,0,0,50,0,0,0,0,0))