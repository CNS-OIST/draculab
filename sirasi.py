"""@package sirasi

sirasi.py  (SImple RAte SImulator)
A simple simulator of rate units with delayed connections.

\author     Sergio Verduzco Flores
\date       2017
"""

# Enumeration classes provide human-readable and globally consistent constants
from enum import Enum  # enumerators for the synapse and unit types

class unit_types(Enum):
    """ An enum class with all the implemented unit models. """
    source = 1
    sigmoidal = 2
    linear = 3
    
class synapse_types(Enum):
    """ An enum class with all the implemented synapse models. """
    static = 1
    oja = 2   # implements the Oja learning rule 
    antihebb = 3  # implements the anti-Hebbian learning rule
    cov = 4 # implements the covariance learning rule
    anticov = 5  # implements the anti-covariance rule 
    hebbsnorm = 6 # Hebbian rule with substractive normalization
    #axoaxo = auto()  # axo-axonic synapses will provide presynaptic inhibitionz

class plant_models(Enum):
    """ An enum class with all the implemented plant models. """
    pendulum = 1

class syn_reqs(Enum):
    """ This enum contains all the variables that a synapse model may ask a unit
        to maintain in order to support its learning function.
    """
    lpf_fast = 1  # postsynaptic activity low-pass filtered with a fast time constant
    lpf_mid = 2   # postsynaptic activity low-pass filtered with a medium time constant
    lpf_slow = 3  # postsynaptic activity low-pass filtered with a slow time constant
    pre_lpf_fast = 4  # presynaptic activity low-pass filtered with a fast time constant
    pre_lpf_mid = 5   # presynaptic activity low-pass filtered with a medium time constant
    pre_lpf_slow = 6  # presynaptic activity low-pass filtered with a slow time constant
    inp_avg = 7   # Sum of fast-LPF'd hebbsnorm inputs, divided by number of hebbsnorm inputs 
    '''
    sum_w = x     # The sum of the weights of all the synapses of the unit  << NOT IMPLEMENTED
    w_list = x    # A list with the weights of all synapses of the unit   << NOT IMPLEMENTED
    inputs = x    # A list with all the inputs of the unit   << NOT IMPLEMENTED
    inp_sum = x  # The sum of all presynaptic inputs   << NOT IMPLEMENTED
    sc_inp_sum = x # Scaled input sum. This is the sum of presynaptic inputs, 
                    # each multiplied by its synaptic weight. Same as the 
                    # steady-state output of a linear unit. << NOT IMPLEMENTED
    '''


# Importing the classes used by the simulator
from network import *
from units import *
from synapses import *
from plants import *

