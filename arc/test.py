import matplotlib.pyplot as plt  # Import library for direct plotting 
import numpy as np               # Import Numerical Python
from IPython.core.display import display, HTML #Import HTML for formatting output
from scipy.constants import h as C_h
from scipy.constants import e as C_e


# NOTE: Uncomment following lines ONLY if you are not using installation via pip
import sys, os
rootDir  = '/home/erlizzard/Documents/Uni_Year_4/DARC/' # e.g. '/Users/Username/Desktop/ARC-Alkali-Rydberg-Calculator'
sys.path.insert(0,rootDir)

from arc import *
atom= StrontiumI()

     
def fromLit(a,b,c,n ):
    return(a*n**2 + b*n +c)

atom.semi = False
atom.preferQuantumDefects = True
calc = StarkMap(atom)
#Target state
n0=80;l0=2;j0=2;mj0=0;  
#Define max/min n values in basis
nmax=85
nmin=75
#Maximum value of l to include (l~20 gives good convergence for states with l<5)
lmax=40

#Initialise Basis States for Solver : progressOutput=True gives verbose output
calc.defineBasis(n0, l0, j0, mj0, nmin, nmax, lmax, 0,progressOutput=True,debugOutput = True)

Emin=0. #Min E field (V/m)
Emax=30. #Max E field (V/m)
N=1001 #Number of Points

#Generate Stark Map
calc.diagonalise(np.linspace(Emin,Emax,N),sub_e = atom.getEnergy(n0,l0,j0,0)* C_e/C_h*1e-9, progressOutput=False)
#Show Sark Map
calc.plotLevelDiagram(progressOutput=True,units=2,highlighState = True,s = 0)
calc.savePlot('StarkMap801D2.png')
calc.showPlot(interactive = True)
#print("%.5f MHz cm^2 / V^2 " % calc.getPolarizability(showPlot=True, minStateContribution=0.9,s = 0))
#Return Polarizability of target state    
#ps.plotLevelDiagram()

#ps.ax.set_ylim(-4,4)

#ps.showPlot()
