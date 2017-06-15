'''
sirasi.py  (SImple RAte SImulator)
A simple simulator of rate units with delayed connections.

Author: Sergio Verduzco Flores

June 2017
'''

from enum import Enum  # enumerators for the synapse and unit types

# These enumeration classes provide human-readable and globally consistent constants
class unit_types(Enum):
    source = 1
    sigmoidal = 2
    linear = 3
    
class synapse_types(Enum):
    static = 1
    oja = 2  
    #axoaxo = auto()  # axo-axonic synapses will provide presynaptic inhibitionz


# Importing the classes used by the simulator
from network import *
from units import *
from synapses import *

