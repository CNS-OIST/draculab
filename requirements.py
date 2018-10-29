"""
requirements.py
This file contains the classes implementing synaptic requirements for draculab units.
"""

from draculab import unit_types, syn_reqs
from units import *

class requirement():
    """ The parent class of requirement classes.  """
    
    def __init__(self, unit):
        """ The class constructor.

            Args:
                unit : a reference to the unit where the requirement is used.

        """
        self.val = None
        unit.functions.add(self.update)

    def update(self, time):
        pass

    def get(self, time):
        return val


class lpf_fast(requirement):
    """ Maintains a low-pass filtered version of the unit's activity.

        The name lpf_fast indicates that the time constant of the low-pass filter,
        whose name is 'tau_fast', should be relatively fast. In practice this is
        arbitrary.
        The user needs to set the value of 'tau_fast' in the parameter dictionary 
        that initializes the unit.

        An instance of this class is meant to be created by ini_pre_syn_update 
        whenever the unit has the 'lpf_fast' requiremnt. In this case, the update
        method should be included in the unit's 'functions' list, and called at
        each simulation step by pre_syn_update.

        Additionally, when the unit has the 'lpf_fast' requirement, the init_buff
        method will be invoked by the unit's init_buffers method.
    """
    def __init__(self, unit):
        """ The class' constructor.
            Args:
                unit : the unit containing the requirement.
        """
        if not hasattr(unit,'tau_fast'): 
            raise NameError( 'Synaptic plasticity requires unit parameter tau_fast, not yet set' )
        self.val = unit.init_val
        self.unit = unit
        unit.functions.add(self.update)
        self.init_buff()
 
    def update(self, time):
        """ Update the lpf_fast variable. """
        #assert time >= self.unit.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_fast updated backwards in time']
        cur_act = self.unit.get_act(time)
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.val = cur_act + ( (self.val - cur_act) * 
                                   np.exp( (self.unit.last_time-time)/self.unit.tau_fast ) )
        # update the buffer
        self.lpf_fast_buff = np.roll(self.lpf_fast_buff, -1)
        self.lpf_fast_buff[-1] = self.val

    def init_buff(self):
        """ Initialize the buffer with past values of lpf_fast. """
        self.lpf_fast_buff = np.array( [self.unit.init_val]*self.unit.steps, dtype=self.unit.bf_type)

    def get(self, steps):
        """ Get the fast low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_fast_buff[-1-steps]


class lpf(requirement):
    """ A low pass filter with a given time constant. """
    def __init__(self, unit):
        self.tau = unit.lpf_tau
        self.val = unit.init_val
        self.unit = unit
        unit.functions.add(self.update)
        self.init_buff()
 
    def update(self, time):
        """ Update the lpf_fast variable. """
        #assert time >= self.unit.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_fast updated backwards in time']
        cur_act = self.unit.get_act(time)
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.val = cur_act + ( (self.val - cur_act) * 
                                   np.exp( (self.unit.last_time-time)/self.tau ) )
        # update the buffer
        self.buff = np.roll(self.buff, -1)
        self.buff[-1] = self.val

    def init_buff(self):
        """ Initialize the buffer with past values. """
        self.buff = np.array( [self.unit.init_val]*self.unit.steps, dtype=self.unit.bf_type)

    def get(self, steps):
        """ Get the fast low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_fast_buff[-1-steps]



