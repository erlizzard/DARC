#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:29:34 2019

@author: rbjv28
"""

import matplotlib.pyplot as plt  # Import library for direct plotting 
import numpy as np               # Import Numerical Python
from IPython.core.display import display, HTML #Import HTML for formatting output
from scipy.constants import h as C_h
from time import gmtime, strftime #to time the code
import time #to time the code

# NOTE: Uncomment following lines ONLY if you are not using installation via pip
import sys, os
rootDir = '/home/hudson/ug/rbjv28/My_Documents/ARC-Alkali-Rydberg-Calculator' # e.g. '/Users/Username/Desktop/ARC-Alkali-Rydberg-Calculator'
sys.path.insert(0,rootDir)

from arc import *
from C6_final import run


atom_sr = StrontiumI(cpp_numerov = False)
atom_sr.preferQuantumDefects = False
atom_sr.semi = True

darc = []
ps = PairStateInteractions(atom_sr, 30,2,2,30,2,2,0,0,0,1,0,0)
ev,_= ps.getC6Diagonalisation(0,0,3)
#darc.append((ev*40**(-11)) /1.4448e-19)
#print(zero_zero*(30**(-11)) /1.4448e-19)

#ps = PairStateInteractions(atom_sr, 40,1,1,40,1,1,0,0,0,1,0,0)
#darc.append((ps.getC6Diagonalisation(np.pi/2,np.pi/2,3)*40**(-11)) /1.4448e-19)

#ps = PairStateInteractions(atom_sr, 40,0,1,40,0,1,0,0,1,1,0,0)
#darc.append((ps.getC6Diagonalisation(np.pi/2,np.pi/2,3)*40**(-11)) /1.4448e-19)
print('got here')
print(darc)

#run()

#n =7



#step = 0.001
#a1,b1 = atom_sr.radialWavefunction(0,0,0,atom_sr.getEnergy(n,0,0,0)/27.11,17**(1/3.0),2.0*n*(n+15.0), step )

#plt.plot(a1,(b1)*(b1),"-" )

#plt.xlabel(r"Distance from nucleus $r$ ($a_0$)")
#plt.ylabel(r"$\vert rR(r)\vert^2$")
#plt.show()

#leroy =  PairStateInteractions(atom_sr, 30,2,2,30,2,2,0,0,1,1,0,0).getLeRoyRadius()
#print(leroy)
#n1,l1,j1,mj1,n2,l2,j2,q,electricFieldAmplitude,s=0.5
#rabi = atom_sr.getRabiFrequency(30,0,0,0,30,1,1,0,1,1,0)
#print(rabi)
        