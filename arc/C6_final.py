from __future__ import division
import numpy as np
from sympy.physics.quantum.cg import CG
from sympy import I
import scipy
import scipy.special
from sympy.physics.wigner import wigner_6j
from mpmath import *
import matplotlib.pyplot as plt
import sys #sys.exit() is a useful tool for debugging
np.set_printoptions(threshold=np.nan) #print full output rather than giving '...'
from time import gmtime, strftime #to time the code
import time #to time the code
import inspect, os #for obtaining current file's path
import csv #for importing csv files
from scipy.constants import h as C_h
from scipy.constants import e as C_e
from scipy.constants import physical_constants, pi , epsilon_0, hbar

'''
References:
    - Vaillant et al., J. Phys. B: At. Mol. Opt. Phys. 45 (2012) 135004
    - Vaillant's PhD thesis, available at http://etheses.dur.ac.uk/10594/
    - Couturier et al., Phys. Rev. A 99, 022503 (2019)
'''

#print C6 matrix, eigenvalues and eigenvectors?
print_results = 0 

#range of n used in intermediate states
nlim = 1 #the value I used is 3, but the higher the better

#path of this file
path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
pathup = os.path.dirname(path) #one layer above

#=================================DEFINING A BUNCH OF FUNCTIONS=================================

def qd(n, L, J, S): #dimensionless
    '''
    ***Quantum Defect for Strontium - Vaillant paper Eqn (18)***
    Data found in Table 2 of Vaillant paper
    '''
    if S==1:
        if L==0 and J==1: #3S1
            '''
            #updated with Couturier's data
            del0 = 3.370778
            del2 = 0.418
            del4 = -0.3
            '''   
            
            #Vaillant's data, revert to these to reproduce their results
            del0 = 3.371
            del2 = 0.5
            del4 = -10
            
            
        elif L==1 and J==2: #3P2
            del0 = 2.8719
            del2 = 0.446
            del4 = -1.9
        elif L==1 and J==1: #3P1
            del0 = 2.8824
            del2 = 0.407
            del4 = -1.3
        elif L==1 and J==0: #3P0
            del0 = 2.8866
            del2 = 0.44
            del4 = -1.9
        elif L==2 and J==3: #3D3
            del0 = 2.63
            del2 = -42.3
            del4 = -18e3
        elif L==2 and J==2: #3D2
            #updated with Couturier's data
            #del0 = 2.66142
            #del2 = -16.77
            #del4 = -6.656e3
            
            #Vaillant's data, revert to these to reproduce their results
            del0 = 2.636
            del2 = -1
            del4 = -9.8e3
            
            
        elif L==2 and J==1: #3D1
            '''#updated with Couturier's data
            del0 = 2.67517
            del2 = -13.15
            del4 = -4.444e3
            '''
            #Vaillant's data, revert to these to reproduce their results
            del0 = 2.658
            del2 = 3
            del4 = -8.8e3
            
        elif L==3 and J==4: #3F4
            del0 = 0.120
            del2 = -2.4
            del4 = 120
        elif L==3 and J==3: #3F3
            del0 = 0.120
            del2 = -2.2
            del4 = 120
        elif L==3 and J==2: #3F2
            del0 = 0.120
            del2 = -2.2
            del4 = 120
    
    elif S==0:
        if L==0: #1S0 
            del0 = 3.26896
            del2 = -0.138
            del4 = 0.9
        elif L==1: #1P1
            del0 = 2.7295
            del2 = -4.67
            del4 = -157
        elif L==2: #1D2
            del0 = 2.3807
            del2 = -39.41
            del4 = -1090
        elif L==3: #1F3
            del0 = 0.089
            del2 = -2.0
            del4 = 30
    
    return del0 + del2/(n-del0)**2 + del4/(n-del0)**4

def nu(n, L, J, S): #dimensionless
    '''Effective Principal Quantum Number - Valliant thesis Eqn (2.2)'''
    return n - qd(n, L, J, S)
    
def E(n, L, J, S): #atomic units
    '''Energy level - Valliant thesis Eqn (2.2)'''
    #LIzzy Changed
    return -109736.627/8065.544 /((nu(n, L, J, S))**2)

def D11(L1b, L2b, J1b, J2b, M1b, M2b, L1a, L2a, J1a, J2a, M1a, M2a, S, theta, phi):
    '''
    ***Radial matrix element - Vallaint thesis Appendix A***
    - Variables ending in b denote the final state (alpha' in paper), those ending in a denote the original state (alpha in paper)
    - Spin is the same in the original and final states
    - phi & theta gives the angular part of atom 2 relative to atom 1   
   
    IMPORTANT CORRECTION TO VAILLAINT'S WORK: 
    - The right-hand side of Eqs. (A3) in paper and (A.3.15) in thesis should be multiplied by a phase factor (-1)**(J1+J2+L1+L2+2S)
    - This function contains the right equations
    '''
    factor0 = (-1)**(J1a+J2a+L1a+L2a+2*S)
    factor1 = np.sqrt(4*np.pi*4*3*2*1*(2*L1a+1)*(2*L2a+1)/20)
    factor2 = np.sqrt((2*J1a+1)*(2*J2a+1))
    factor3 = CG(L1a,0,1,0,L1b,0).doit()*CG(L2a,0,1,0,L2b,0).doit()
    factor4 = wigner_6j(J1a, 1, J1b, L1b, S, L1a)*wigner_6j(J2a, 1, J2b, L2b, S, L2a)
    factor5 = 0
    for p in np.linspace(-2,2,2-(-2)+1): #-2, -1, 0, 1, 2
        for p1 in np.linspace(-1,1,1-(-1)+1):
            for p2 in np.linspace(-1,1,1-(-1)+1):
                product = scipy.special.sph_harm(p, 2, phi, theta) #note: scipy's phi and theta are different from normal definitions
                product *= CG(1, p1, 1, p2, 2, p).doit() #note: sympy notations for CG coefficients are unconventional!
                product *= CG(J1a, M1a, 1, p1, J1b, M1b).doit()
                product *= CG(J2a, M2a, 1, p2, J2b, M2b).doit()
                factor5 += product
                '''
                #for debugging
                if CG(J1a, M1a, 1, p1, J1b, M1b).doit()*CG(J2a, M2a, 1, p2, J2b, M2b).doit()!=0:
                    print 'M1a, M1b', M1a, M1b, 'M2a, M2b', M2a, M2b
                    print 'CG*CG', CG(J1a, M1a, 1, p1, J1b, M1b).doit()*CG(J2a, M2a, 1, p2, J2b, M2b).doit()
                '''
 
    #print 'D11', (-factor0*factor1*factor2*factor3*factor4*factor5.evalf())
    return complex(-factor0*factor1*factor2*factor3*factor4*factor5.evalf()) #evaluate numerically and use j not I for imag part

'''
***g0, g1, g2 & g3 are needed to calculate Radial Dipole Matrix Element***
See Valliant thesis p.18 onwards

Divergences of g0, g1, g2 & g3 at dnu=0 are suppressed manually
This has been tested and gives the right results
'''

def g0(dnu):
    if dnu==0:
        print('DELTA MU = 0 ')
        print()
        ans=1
    else:
        ans=1/(3*dnu)*(angerj(dnu-1, -dnu)-angerj(dnu+1, -dnu))
    return ans
def g1(dnu):
    if dnu==0:
        ans=0
    else:
        ans=-1/(3*dnu)*(angerj(dnu-1,-dnu)+angerj(dnu+1, -dnu))
    return ans
def g2(dnu):
    if dnu==0:
        ans=0
    else:
        ans=g0(dnu) - np.sin(np.pi*dnu)/(np.pi*dnu)
    return ans
def g3(dnu):
    if dnu==0:
        ans=0
    else:
        ans=dnu/2*g0(dnu)+g1(dnu)    

    return ans

def rad_dip_mat_el(na, nb, La, Lb, Ja, Jb, S):
    '''
    ***Radial Dipole Matrix Element (semiclassical approach)***
    Valliant thesis p.18 onwards
    Variables ending in a (not primed in paper/thesis) - original state, ending in b (primed in paper/thesis) - final state
    '''
    nu_a, nu_b = nu(na, La, Ja, S), nu(nb, Lb, Jb, S)    
    dnu = nu_a - nu_b
    #print(dnu)
    dL = Lb - La
    nu_c = np.sqrt(nu_a*nu_b)
    L_c = 0.5*(La+Lb+1)
    gamma = dL*L_c/nu_c  
    return 3/2*(nu_c)**2*np.sqrt(1-(L_c/nu_c)**2)*(g0(dnu)+gamma*g1(dnu)+gamma**2*g2(dnu)+gamma**3*g3(dnu))

def R11(n1a, n2a, L1a, L2a, J1a, J2a, n1b, n2b, L1b, L2b, J1b, J2b, S):
    return rad_dip_mat_el(n1a,n1b,L1a,L1b,J1a,J1b,S)*rad_dip_mat_el(n2a,n2b,L2a,L2b,J2a,J2b, S)*C_e**2/(4.0*pi*epsilon_0)*\
                (physical_constants["Bohr radius"][0])**(2)
def intermediate_qns(n, L, M, J, S):
    '''Finds all sets of quantum numbers for dipole-allowed intermediate states for each atom'''
    #by the dipole selection rules:
    allowed_L = np.array([L-1, L+1])
    allowed_L = np.delete(allowed_L, np.where(allowed_L<0)) #remove any negative L values
    allowed_M = np.array([M-1, M, M+1])
    if J!=0:
        allowed_J = np.array([J-1, J, J+1])
    elif J==0:
        allowed_J = np.array([1])
    allowed_LMJ = []
    for Lj in allowed_L:
        for Mj in allowed_M:
            for Jj in allowed_J:
                if abs(Mj) <= Jj : #because M=-J, ..., J
                    if Jj <= Lj + S and Jj >= abs(Lj-S): #otherwise it's impossible 
                        allowed_LMJ.append([Lj, Mj, Jj])
      
    #constructing the list of n that will be used in the summation
    #theoretically it should be an infinite array but practically it's unecessary
    n_list = np.arange(n-nlim,n+nlim+1) #array form n-nlim to n+nlim. I used nlim=3.
    allowed_nLMJ = []
    for n_value in n_list:
        for LMJ_values in allowed_LMJ:
            L_value = LMJ_values[0]
            M_value = LMJ_values[1]
            J_value = LMJ_values[2]
            if L_value < n_value: #because l=0,...,n-1
                allowed_nLMJ.append([n_value, L_value, M_value, J_value])
    #remove intermediate states that are the same as the initial state
    del_index = []
    for i in range(np.shape(allowed_nLMJ)[0]):
        nLMJ = allowed_nLMJ[i]
        if nLMJ[0]==n and nLMJ[1]==L and nLMJ[3]==J:
            #print 'nLMJ', nLMJ
            del_index.append(i)
    allowed_nLMJ = np.delete(allowed_nLMJ, del_index, axis=0)
    return allowed_nLMJ #2D array with 4 columns

def intermediate_pairs(n1, L1, M1, J1, n2, L2, M2, J2, S):
    '''Finds all possible combinations of dipole-allowed intermediate states for the two atoms'''
    #separate arrays of dipole-allowed intermediate states for each atom
    atom1_states = intermediate_qns(n1, L1, M1, J1, S)
    atom2_states = intermediate_qns(n2, L2, M2, J2, S)
    pair_states = []
    for i in range(np.shape(atom1_states)[0]): #loop over each row, which represents one intermediate state
        for j in range(np.shape(atom2_states)[0]): 
            pair_states.append(np.concatenate((atom1_states[i], atom2_states[j])))
    return pair_states #2D array with 4x2=8 columns

def sum_term(n1, L1, n2, L2, M1a, M2a, M1b, M2b, J1, J2, n1_i, L1_i, M1_i, n2_i, L2_i, M2_i, J1_i, J2_i, S, theta, phi):
    '''
    ***Computes each term in the C6 summation***
    - n1, L1, n2, L2: physical quantum numbers of the two atoms
    - M1a, M2a, M1b, M2b: M values given by the column and row of the matrix element.
    - M1a/M2a -> M1'/M2' in paper/thesis; M1b/M2b -> M1/M2 in paper/thesis
    - Subscript i means intermediate state
    '''

      
    forster_defect = E(n1_i, L1_i, J1_i, S) + E(n2_i, L2_i, J2_i, S) - E(n1, L1, J1, S) - E(n2, L2, J2, S)
    #print('energy defect',forster_defect)
    product = D11(L1, L2, J1, J2, M1a, M2a, L1_i, L2_i, J1_i, J2_i, M1_i, M2_i, S, theta, phi)
    
    #print('Angular part1', D11(L1, L2, J1, J2, M1a, M2a, L1_i, L2_i, J1_i, J2_i, M1_i, M2_i, S, theta, phi))
    product *= R11(n1, n2, L1, L2, J1, J2, n1_i, n2_i, L1_i, L2_i, J1_i, J2_i, S)*(1.0e-9*(1.0e6)**3/C_h)
    product *= D11(L1_i, L2_i, J1_i, J2_i, M1_i, M2_i, L1, L2, J1, J2, M1b, M2b, S, theta, phi)
    #print('angualr part2',D11(L1_i, L2_i, J1_i, J2_i, M1_i, M2_i, L1, L2, J1, J2, M1b, M2b, S, theta, phi))
    product *= R11(n1_i, n2_i, L1_i, L2_i, J1_i, J2_i, n1, n2, L1, L2, J1, J2, S)*(1.0e-9*(1.0e6)**3/C_h)
    forster_defect =  forster_defect*C_e/C_h *1e-9

    if abs(forster_defect) < 1.:
        print(n1_i,L1_i,J1_i, n2_i, L2_i, J2_i, M1b,M2b,M1_i, M2_i)
        print('radial',R11(n1, n2, L1, L2, J1, J2, n1_i, n2_i, L1_i, L2_i, J1_i, J2_i, S)*(1.0e-9*(1.0e6)**3/C_h)*R11(n1_i, n2_i, L1_i, L2_i, J1_i, J2_i, n1, n2, L1, L2, J1, J2, S)*(1.0e-9*(1.0e6)**3/C_h))
        print('angular',D11(L1_i, L2_i, J1_i, J2_i, M1_i, M2_i, L1, L2, J1, J2, M1b, M2b, S, theta, phi)*D11(L1, L2, J1, J2, M1a, M2a, L1_i, L2_i, J1_i, J2_i, M1_i, M2_i, S, theta, phi))
        print('ed', forster_defect)

        print('int',qd(n1_i,L1_i,J1_i,S))
        print('int',qd(n2_i,L2_i,J2_i,S))

        print('int',E(n1_i, L1_i, J1_i, S)*C_e/C_h *1e-9)
        print('int', E(n2_i, L2_i, J2_i, S)*C_e/C_h *1e-9) 
        print(E(n1, L1, J1, S)*C_e/C_h *1e-9)
        print( E(n2, L2, J2, S)*C_e/C_h *1e-9)
        
    return 1/forster_defect*product #this does not include the negative sign before the summation


def c6_mat(n1, n2, L1, L2, J1, J2, S, theta, phi):
    '''Constructs the C6 matrix'''
    
    dim = (2*J1+1)*(2*J2+1) #dimensions of matrix
    mat = np.zeros((dim, dim), dtype='complex') #spherical harmonics can give complex values
    M1_list = np.arange(-J1,J1+1) #-J1,...,J1
    M2_list = np.arange(-J2,J2+1)
    M_pairs = [] #rows and columns of the matrix are labelled by pairs of M1, M2
    for i in range(len(M1_list)):
        for j in range(len(M2_list)):
            M_pairs.append([M1_list[i], M2_list[j]])

    M_pair_sums = [] 
    #these are the Omega quantum numbers in Vaillant's work, labelled as Q in my report
    for k in range(np.shape(M_pairs)[0]): #height
        M_pair_sums.append(np.sum(M_pairs[k]))
    M_pair_sums_sort = np.sort(M_pair_sums) 
    #firstly sorted by the sum
    #secondly sorted by value of M1 (because of the way M_pairs was constructed)
    indices = np.argsort(M_pair_sums)
    
    M_pairs_sort = []
    for index in indices:
        M_pairs_sort.append(M_pairs[index]) 
    
    #Elements that need to be computed. The rest can be filled by symmetry.
    if J1==1: #e.g.3S1
        symindices = [[0,0],[1,1],[1,2],[3,3],[3,4],[3,5],[4,4]]
    elif J1==2: #e.g. 3D2
        symindices = [[0,0],[1,1],[1,2],[3,3],[3,4],[3,5],[4,4],[6,6],[6,7],[6,8],[6,9],\
        [7,7],[7,9],[10,10],[10,11],[10,12],[10,13],[10,14],[11,11],[11,12],[11,13],[12,12]]

    if(J1 == 0 ): symindices= [[0,0]]
    for i in range(dim):
        for j in range(dim):
           #if i==3 and j==4: #COMMENT THIS OUT - for debugging only
           #print(i,j)
           M1a, M2a = M_pairs_sort[j]
           #print(M1a,M2a)

           if [i,j] in symindices: #only fill these matrix elements with the same omega = M1(')+M2(')
                #M1a/M2a -> M1'/M2'; M1b/M2b -> M1/M2
               
                #print(m1a,m2a)
                M1b, M2b = M_pairs_sort[i]
                #print('colunm',j)
               
                #print(M1a,M2a, M1b, M2b)
                #pick out intermediate states allowed by (M1b, M2b) & (M1a, M2a)
                states_a = intermediate_pairs(n1, L1, M1a, J1, n2, L2, M2a, J2, S)
                states_b = intermediate_pairs(n1, L1, M1b, J1, n2, L2, M2b, J2, S)
                
                #print(n1, L1, M1a, J1, n2, L2, M2a, J2, S)
                #print(n1, L1, M1b, J1, n2, L2, M2b, J2, S)                
                #picks out rows common to states_a and states_b
                def common_states(list1, list2):
                    nrows, ncols = list1.shape
                    dtype={'names':['f{}'.format(i) for i in range(ncols)],
                            'formats':ncols * [list1.dtype]}
                    c = np.intersect1d(list1.view(dtype), list2.view(dtype))
                    return c
                #change lists to np.arrays, otherwise common_states (found on Stack Exchange I think) doesn't work
                states_a, states_b = np.array(states_a), np.array(states_b)
                states_ab = common_states(states_a, states_b)
                #print 'np.shape(states_ab)', np.shape(states_ab)
                
                #print(states_ab.shape)
                element_sum = 0
                for n in range(np.shape(states_ab)[0]):
                    n1_i, L1_i, M1_i, J1_i, n2_i, L2_i, M2_i, J2_i = states_ab[n]#[0], states_ab[n][1], states_ab[n][2], states_ab[n][3], states_ab[n][4], states_ab[n][5]
                    term = -sum_term(n1, L1, n2, L2, M1a, M2a, M1b, M2b, J1, J2, n1_i, L1_i, M1_i, n2_i, L2_i, M2_i, J1_i, J2_i, S, theta, phi)
                    element_sum += term
                mat[i, j] = element_sum
                #if mat[i,j] !=0:
                #    print(i,j)
                #    print(M1a, M2a, M1b,M2b)
                #    print(mat[i,j])
    #fill in other elements by symmetry
    if J1==1: 
        mat[8,8] = mat[0,0]
        mat[2,2], mat[6,6], mat[7,7] = np.repeat(mat[1,1], 3)
        mat[2,1], mat[6,7], mat[7,6] = np.repeat(mat[1,2], 3)
        mat[5,5] = mat[3,3]
        mat[4,3], mat[4,5], mat[5,4] = np.repeat(mat[3,4], 3)
        mat[5,3] = mat[3,5]
    elif J1==2:
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
    if print_results:
        print ('mat', mat)
    
    #saves C6 matrix into csv file
    if L1==0 and J1==1 and S==1: #3S1
        filename = 'Sr_n%i_3S1_C6_matrices' %(n1)
    elif L1==2 and J1==2 and S==1: #3D2
        filename = 'Sr_n%i_3D2_C6_matrices' %(n1)
    elif L1==2 and J1==1 and S==1: #3D1
        filename = 'Sr_n%i_3D1_C6_matrices' %(n1)
    elif L1==2 and J1==2 and S==0: #1D2
        filename = 'Sr_n%i_1D2_C6_matrices' %(n1)
    csvname = '%.4f,%.4f' %(theta,phi)
    
    #different computers have different slashes in paths...
    #if os.name=='posix':#mac - my laptop
    #    slash = '/'
    #elif os.name=='nt':#windows - library
   #     slash='\\'#double \\ so I don't confuse Python with my syntax
    #fullpath = '%s%s%s%s%s.csv' %(pathup,slash,filename,slash,csvname)

    #if os.path.exists(fullpath)==0:#only save if this file doesn't exist
        #this avoids clearing data by accident
    #    with open(fullpath, 'wb') as f:
    #        np.savetxt(f, mat, delimiter=",") 
    return mat

def c6_coeffs(n1, n2, L1, L2, J1, J2, S, theta, phi):
    mat = c6_mat(n1, n2, L1, L2, J1, J2, S, theta, phi)
    #print 'c6_mat', mat
    return np.linalg.eigh(mat)

#================================ACTUALLY RUNNING THE CODE=========================================
#simplify angles and extrapolate later using symmetry of spherical harmonics
def run():
    theta_list = np.linspace(0,np.pi/2,8) #symmetric across theta=pi/2
    phi_list = [0]#I have found that there is no phi-dependence
    
    print ('Evaluating C6 matrix at various angles')
    print ('Matrices are saved in csv files')
    print ('')
    
    channels = [#[40,40,0,0,0,0,0,np.pi/2,np.pi/2], #1S0
                #[40,40,1,1,1,1,0,0,0]] #1P1
                [30,30,2,2,2,2,0,0,0]] #1D2
                #[40,40,0,0,1,1,1,np.pi/2,np.pi/2]] #3S1
                #[40,40,1,1,0,0,1,0,0], #3P0
                #[40,40,1,1,1,1,1,0,0], #3P1
                #[40,40,1,1,2,2,1,0,0], #3P2
                #[40,40,2,2,1,1,1,0,0], #3D1
                #[40,40,2,2,2,2,1,0,0]] #3D2
                #[40,2,3,40,2,3,0,0,1,1]]  #3D3
    
    output= [] 
    start_time = time.time()
    print ('Start running:')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    
    for channel in channels:
        ch_time = time.time()
        ev, _ = c6_coeffs(*channel)#*1.4448e-19)
        output.append(ev*(30)**(-11))
        ch_end = time.time()
        print('individual C6 time (mins)',(ch_end - ch_time)/60 )
    #ps = PairStateInteractions(atom,30,0,0,30,0,0,0,0,0,1)
    
    print ('Finish running:')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    
    end_time = time.time()
    print ('Finish running in %s minutes' %((end_time-start_time)/60))
    print ('')
    
    print(output)
#f, ax = plt.subplots(1,1, figsize=(12,8))

#get a list of the indicies 
#y = np.array(output).flatten()
#index = [[i]*len(output[i]) for i in range(len(output))]
#index = np.array(index).flatten()

#index =[item for sublist in index for item in sublist]
#y =[item for sublist in y for item in sublist]

#plt.scatter(index,y )

#draw the vertical lines
#xcoords = np.arange(0,9,1)
#for xc in xcoords:
#    plt.axvline(x = xc,linestyle = '--')
    
#ax.set_xticklabels(['0',r"$^1S_0$",r"$^1P_1$",r"$^1D_2$",r"$^3S_1$",r"$^3P_0$",\
#                   r"$^3P_1$",r"$^3P_2$",r"$^3D_1$",r"$^3D_2$"])
#ax.set_xlabel(r"Series")
#ax.set_ylabel(r"$C_6 n^{-11}$ (a.u)")
#plt.title(r"Scaled $C_6$ at $n=40$",fontsize=10)
#plt.savefig('Andrea_C6.png')
#plt.show()

#for theta in theta_list:
#    for phi in phi_list:
#        if theta==0 and phi==0 or theta!=0:
        #if theta=0, just need to consider phi=0 since the rest are the same point
#            count += 1
            