import matplotlib.pyplot as plt  # Import library for direct plotting functions
import numpy as np               # Import Numerical Python
from IPython.core.display import display, HTML #Import HTML for formatting output
from scipy.constants import h as C_h

# NOTE: Uncomment following lines ONLY if you are not using installation via pip
import sys, os
rootDir = '/home/hudson/ug/rbjv28/My_Documents/ARC-Alkali-Rydberg-Calculator' # e.g. '/Users/Username/Desktop/ARC-Alkali-Rydberg-Calculator'
sys.path.insert(0,rootDir)

from arc import *
atom_sr = StrontiumI()
atom_sr.preferQuantumDefects = True
atom_cs = Caesium(cpp_numerov = False)

pairstate_interaction = PairStateInteractions(atom_sr, 50,1,1,50,1,1,1,1,1)
print(pairstate_interaction.getC6perturbatively(np.pi/2,0, 3,25e9,True))
