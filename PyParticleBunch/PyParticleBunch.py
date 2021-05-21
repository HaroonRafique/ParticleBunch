"""
#
#   PyParticleBunch
#   A module to generate matched particle distributions for accelerator 
#	physics beam dynamics simulations
#
#   Version : 1.0
#   Author  : H. Rafique
#   Contact : haroon .dot. rafique .at. stfc .dot. ac .dot. uk
#
"""

from __future__ import absolute_import, division, print_function, unicode_literals
try:
    import numpy as np
except ImportError:
    print("# PyParticleBunch : numpy module is required. ")
try:
    import sympy as sy
    import sympy.stats as stats
except ImportError:
    print("# PyParticleBunch : sympy module is required. ")
   try:
    import tfs
except ImportError:
    print("# PyParticleBunch : tfs module is required. ") 
    
    
__version   = 1.0
__PyVersion = ["2.7", "3.6+"]
__author    = ["Haroon Rafique"]
__contact   = ["haroon .dot. rafique .at. stfc .dot. ac .dot. uk"]

class PyParticleBunch(object):    
	"""
    class for the generation of a matched particle bunch distribution
    Returns: PySCRDT instance
    """

    def __init__(self, parameters=False, mode=None, twissFile=None, order=None):
        """
        Initialization function
        Input :  parameters : [bool|str]  Parameters needed for the calculations (default=False)
                                          if True the default values of [setParameters] are used 
                                          if str parameters are read in a file
                 mode       : [3|5]       Resonance description mode (default=None)
                 order      : [list]      Resonance order and harmonic (default=None)
                 twissFile  : [str]       MAD-X/PTC Twiss file (default=None)
        Returns: void
        """
'''
IDEAS:
1D generator: start in normalised space with action angle co-ordinates 
then transform to beam phase space?



TODO:
KV
Waterbag
Gaussian
Joho

Function to add closed orbit/initial kicks etc
Function to quickly modify bunch
Function to read bunch file (multiple format: MAD-X, PTC, PyORBIT)
Function to write bunch file (multiple format: MAD-X, PTC, PyORBIT)

'''
