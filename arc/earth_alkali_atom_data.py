# coding=utf-8
"""
Data sources
-------------

.. [#c1] J. A. Armstrong, J. J. Wynne, and P. Esherick,
        "Bound, odd-parity J = 1 spectra of the alkaline earths: Ca, Sr, and Ba,"
        J. Opt. Soc. Am. 69, 211-230 (1979)


.. [#c2]R.Beigang K.Lücke A.Timmermann P.J.West D.Frölich
        Determination of absolute level energies of 5sns1S0 and 5snd1D2 Rydberg series of Sr
        Opt. Commun. 42, 19 1982.

.. [#c3] J E Sansonetti and G Nave. Wavelengths,
        Transition Probabilities, and Energy Levels for the Spectrum of Neutral Strontium (Sr I).
        Journal of Physical and Chemical Reference Data, 39:033103, 2010.

.. [#c4]Baig M Yaseen M Nadeem A Ali R Bhatti S
        Three-photon excitation of strontium Rydberg levels
        Optics Communications, vol: 156 (4-6) pp: 279-284, 1998

.. [#c5] P Esherick, J J Wynne, and J A Armstrong.
        Spectroscopy of 3P0 states of alkaline earths.
        Optics Letters, 1:19, 1977.

.. [#c6] P Esherick.
        Bound, even-parity J = 0 and J = 2 spectra of Sr.
        PhysicalReview A, 15:1920, 1977.

.. [#c7] R Beigang and D Schmidt.
        Two-Channel MQDT Analysis of Bound 5snd 3D1,3 Rydberg States of Strontium.
        Physica Scripta, 27:172, 1983.

.. [#c8]J R Rubbmark and S A Borgstr¨om.
        Rydberg Series in Strontium Found in Absorption by Selectively Laser-Excited Atoms.
        Physica Scripta, 18:196,1978

.. [#c9] Beigang R, Lucke K, Schmidt D, Timmermann A and West P J ¨
        One-Photon Laser Spectroscopy of Rydberg Series from Metastable Levels in Calcium and Strontium
        Phys. Scr. 26 183, 1982

.. [c10] L. Couturier, I. Nosske, F. Hu, C. Tan, C. Qiao, Y. H. Jiang, P. Chen, and M. Weidemüller.
        Measurement of the strontium triplet Rydberg series by depletion spectroscopy of ultracold atoms
        http://arxiv.org/abs/1810.07611

"""
from __future__ import division, print_function, absolute_import

from .earth_alkali_atom_functions import *

from scipy.constants import physical_constants, pi , epsilon_0, hbar
from scipy.constants import Rydberg as C_Rydberg
from scipy.constants import m_e as C_m_e

class StrontiumI(EarthAlkaliAtom):
    """Properties of Strontium atoms"""

    ionisationEnergy = 5.69486740  #eV  ref. [#c3]
    ionisationEnergycm = 45932.2036 #cm-1  ref. [#c3]
    Z = 38
    scaledRydbergConstant = 109736.627# cm-1 *1.e2\
        #*physical_constants["inverse meter-electron volt relationship"][0] # ref. [#c2]

    levelDataFromNIST = ["sr_1S0.csv", "sr_3S1.csv", "sr_1P1.csv", "sr_3P2.csv", \
                         "sr_3P1.csv", "sr_3P0.csv", "sr_1D2.csv", "sr_3D3.csv", \
                         "sr_3D2.csv", "sr_3D1.csv", "sr_1F3.csv", "sr_3F4.csv", \
                         "sr_3F3.csv", "sr_3F2.csv"]  #: Store list of filenames to read in

    NISTdataLevels = {"1S0":65,"3S1":45, "1P1":79,"3P2":55,"3P1":17, "3P0":10,"1D2":65,"3D3":41,"3D2":45,"3D1":46,"1F3":25,"3F4":24,"3F3":24,"3F2":24}
    level_labels = ["1S0","3S1", "1P1","3P2","3P1", "3P0","1D2","3D3","3D2","3D1","1F3","3F4","3F3","3F2"]
    dataFolder = 'data/sr_data'
    quantumDefectData ='quantum_defect.csv'
    groundStateN = 5
    extraLevels = {"3D3":4, "3D1":4, "1F3":4, "3F4":4,"3F3":4, "3F2":4}
    preferQuantumDefects = False
