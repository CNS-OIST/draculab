"""
requirements.py
This file contains the classes implementing synaptic requirements for draculab units.
"""

from draculab import unit_types, synapse_types, syn_reqs
from units import *
import numpy as np

def make_lambda(unit, fun):
    return lambda : fun(unit)

def make_lambda2(unit, fun):
    return lambda t : fun(unit, t)

def add_lpf_fast(unit):
    if not hasattr(unit,'tau_fast'): 
        raise NameError( 'Synaptic plasticity requires unit parameter tau_fast, not yet set' )

    def init_lpf_fast_buff(self):
        """ Initialize the buffer with past values of lpf_fast. """
        self.lpf_fast_buff = np.array( [self.init_val]*self.steps, dtype=self.bf_type)

#    def upd_lpf_fast(self, time):
#        """ Update the lpf_fast variable. """
#        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
#        #                                ' lpf_fast updated backwards in time']
#        cur_act = self.get_act(time)
#        # This updating rule comes from analytically solving 
#        # lpf_x' = ( x - lpf_x ) / tau
#        # and assuming x didn't change much between self.last_time and time.
#        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
#        self.lpf_fast = cur_act + ( (self.lpf_fast - cur_act) * 
#                                   np.exp( (self.last_time-time)/self.tau_fast ) )
#        # update the buffer
#        self.lpf_fast_buff = np.roll(self.lpf_fast_buff, -1)
#        self.lpf_fast_buff[-1] = self.lpf_fast

    setattr(unit, 'lpf_fast', unit.init_val)
    setattr(unit, 'init_lpf_fast_buff', make_lambda(unit, init_lpf_fast_buff))
    #setattr(unit, 'upd_lpf_fast', make_lambda2(unit, upd_lpf_fast))
    #unit.functions.add(make_lambda2(unit, upd_lpf_fast))


class requirement():
    """ The parent class of requirement classes.  """
    
    def __init__(self, unit):
        """ The class constructor.

            Args:
                unio : a reference to the unit where the requirement is used.

        """
        self.val = None

    def update(self, time):
        pass

    def get(self):
        return self.val


class lpf_fast(requirement):
    """ Maintains a low-pass filtered version of the unit's activity.

        The name lpf_fast indicates that the time constant of the low-pass filter,
        whose name is 'tau_fast', should be relatively fast. In practice this is
        arbitrary.
        The user needs to set the value of 'tau_fast' in the parameter dictionary 
        that initializes the unit.

        An instance of this class is meant to be created by init_pre_syn_update 
        whenever the unit has the 'lpf_fast' requiremnt. In this case, the update
        method of this class will be included in the unit's 'functions' list, and 
        called at each simulation step by pre_syn_update.

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


class lpf_mid(requirement):
    """ Maintains a low-pass filtered version of the unit's activity.

        The name lpf_mid indicates that the time constant of the low-pass filter,
        whose name is 'tau_mid', should have an intermediate value. In practice 
        this is arbitrary.
        The user needs to set the value of 'tau_mid' in the parameter dictionary 
        that initializes the unit.

        An instance of this class is meant to be created by init_pre_syn_update 
        whenever the unit has the 'lpf_mid' requiremnt. In this case, the update
        method of this class will be included in the unit's 'functions' list, and 
        called at each simulation step by pre_syn_update.

        Additionally, when the unit has the 'lpf_mid' requirement, the init_buff
        method will be invoked by the unit's init_buffers method.
    """
    def __init__(self, unit):
        """ The class' constructor.
            Args:
                unit : the unit containing the requirement.
        """
        if not hasattr(unit,'tau_mid'): 
            raise NameError( 'Synaptic plasticity requires unit parameter tau_mid, not yet set' )
        self.val = unit.init_val
        self.unit = unit
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
                                   np.exp( (self.unit.last_time-time)/self.unit.tau_mid ) )
        # update the buffer
        self.lpf_mid_buff = np.roll(self.lpf_mid_buff, -1)
        self.lpf_mid_buff[-1] = self.val

    def init_buff(self):
        """ Initialize the buffer with past values of lpf_fast. """
        self.lpf_mid_buff = np.array( [self.unit.init_val]*self.unit.steps, dtype=self.unit.bf_type)

    def get(self, steps):
        """ Get the fast low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_mid_buff[-1-steps]


class lpf_slow(requirement):
    """ Maintains a low-pass filtered version of the unit's activity.

        The name lpf_slow indicates that the time constant of the low-pass filter,
        whose name is 'tau_slow', should have a relatively large value. In practice 
        this is arbitrary.
        The user needs to set the value of 'tau_slow' in the parameter dictionary 
        that initializes the unit.

        An instance of this class is meant to be created by init_pre_syn_update 
        whenever the unit has the 'lpf_slow' requiremnt. In this case, the update
        method of this class will be included in the unit's 'functions' list, and 
        called at each simulation step by pre_syn_update.

        Additionally, when the unit has the 'lpf_slow' requirement, the init_buff
        method will be invoked by the unit's init_buffers method.
    """
    def __init__(self, unit):
        """ The class' constructor.
            Args:
                unit : the unit containing the requirement.
        """
        if not hasattr(unit,'tau_slow'): 
            raise NameError( 'Synaptic plasticity requires unit parameter tau_slow, not yet set' )
        self.val = unit.init_val
        self.unit = unit
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
                                   np.exp( (self.unit.last_time-time)/self.unit.tau_slow ) )
        # update the buffer
        self.lpf_slow_buff = np.roll(self.lpf_slow_buff, -1)
        self.lpf_slow_buff[-1] = self.val

    def init_buff(self):
        """ Initialize the buffer with past values of lpf_fast. """
        self.lpf_slow_buff = np.array( [self.unit.init_val]*self.unit.steps, dtype=self.unit.bf_type)

    def get(self, steps):
        """ Get the fast low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_slow_buff[-1-steps]


class lpf(requirement):
    """ A low pass filter with a given time constant. """
    def __init__(self, unit):
        self.tau = unit.lpf_tau
        self.val = unit.init_val
        self.unit = unit
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


class sq_lpf_slow(requirement):
    """ A low pass filtered version of the squared activity.
        
        As the name implies, the filter uses the "slow" time constant 'tau_slow'.
        The purpose of this is to have a mean value of the square of the activity,
        as used in some versions of the BCM learning rule. Accordingly, this 
        requirement is used byt the bcm_synapse class.
    """
    def __init__(self, unit):
        if not hasattr(unit,'tau_slow'): 
            raise NameError( 'sq_lpf_slow requires unit parameter tau_slow, not yet set' )
        self.tau = unit.tau_slow
        self.val = unit.init_val
        self.unit = unit

    def update(self,time):
        """ Update the sq_lpf_slow variable. """
        #assert time >= self.unit.last_time, ['Unit ' + str(self.unit.ID) + 
        #                                ' sq_lpf_slow updated backwards in time']
        cur_sq_act = self.unit.get_act(time)**2.  
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.val = cur_sq_act + ( (self.val - cur_sq_act) * 
                                  np.exp( (self.unit.last_time-time)/self.tau ) )

class inp_vector(requirement):
    """ A numpy array with the unit's inputs at the start of the current simulation step.

        The inputs are not multiplied by synaptic weights, and come with their 
        appropriate delays.

        This input vector is used by other synaptic requirements, namely lpf_mid_inp_sum,
        balance, and exp_scale. In this last requirement, it is used to obtain the 'mu'
        factor used by the rule.
    """
    def __init__(self, unit):
        self.unit = unit
        self.val = np.tile(unit.init_val, len(unit.net.syns[unit.ID]))
        self.uid = unit.ID

    def update(self, time):
        self.val = np.array([ fun(time - dely) for dely,fun in 
                   zip(self.unit.net.delays[self.uid], self.unit.net.act[self.uid]) ])


class mp_inputs(requirement):
    """ Maintains a list with all the inputs, in the format of get_mp_inputs method. 

        In fact, the list is updated using the get_mp_inputs method. Some requirements
        like mp_balance and lpf_slow_mp_inp_sum use the output of get_mp_inputs, and
        the mp_inputs requirement saves computations by ensuring that get_mp_inputs
        only gets called once per simulation step.

        Repeating the docsting of unit.get_mp_inputs:
        This method is for units where multiport = True, and that have a port_idx attribute.
        The i-th element of the returned list is a numpy array containing the raw (not multiplied
        by the synaptic weight) inputs at port i. The inputs include transmision delays.
    """
    def __init__(self, unit):
        if not hasattr(unit,'port_idx'): 
            raise NameError( 'the mp_inputs requirement is for multiport units with a port_idx list' )
        self.val = [] 
        for prt_lst in unit.port_idx:
            self.val.append(np.array([unit.init_val for _ in range(len(prt_lst))]))
        self.unit = unit

    def update(self, time):
        self.val = self.unit.get_mp_inputs(time)
        

class inp_avg_hsn(requirement):
    """ Provides an average of the inputs arriving at hebbsnorm synapses.

        More precisely, the sum of fast-LPF'd hebbsnorm inputs divided by the the number
        of hebbsnorm inputs.

        Since we are using the lpf_fast signals, all the presynaptic units with hebbsnorm
        synapses need to have the lpf_fast requirement. 
    """
    def __init__(self, unit):  
        self.snorm_list = []  # a list with all the presynaptic units
                              # providing hebbsnorm synapses
        self.snorm_dels = []  # a list with the delay steps for each connection from snorm_list
        for syn in unit.net.syns[unit.ID]:
            if syn.type is synapse_types.hebbsnorm:
                if not syn_reqs.lpf_fast in unit.net.units[syn.preID].syn_needs:
                    raise AssertionError('inp_avg_hsn needs lpf_fast on presynaptic units')
                self.snorm_list.append(unit.net.units[syn.preID])
                self.snorm_dels.append(syn.delay_steps)
        self.n_hebbsnorm = len(self.snorm_list) # number of hebbsnorm synapses received
        self.val = 0.2  # an arbitrary initialization of the average input value
        self.snorm_list_dels = list(zip(self.snorm_list, self.snorm_dels)) # both lists zipped

    def update(self, time):
        self.val = sum([u.get_lpf_fast(s) for u,s in self.snorm_list_dels]) / self.n_hebbsnorm


class pos_inp_avg_hsn(requirement):
    """ Provides an average of the inputs arriving at hebbsnorm synapses with positive weights.

        More precisely, the sum of fast-LPF'd hebbsnorm inputs divided by the the number
        of hebbsnorm inputs, but only considering inputs with positive synapses.

        Since we are using the lpf_fast signals, all the presynaptic units with hebbsnorm
        synapses need to have the lpf_fast requirement. 
    """
    def __init__(self, unit):  
        self.snorm_units = []  # a list with all the presynaptic units
                               # providing hebbsnorm synapses
        self.snorm_syns = []  # a list with the synapses for the list above
        self.snorm_delys = []  # a list with the delay steps for these synapses
        for syn in unit.net.syns[unit.ID]:
            if syn.type is synapse_types.hebbsnorm:
                if not syn_reqs.lpf_fast in unit.net.units[syn.preID].syn_needs:
                    raise AssertionError('pos_inp_avg_hsn needs lpf_fast on presynaptic units')
                self.snorm_syns.append(syn)
                self.snorm_units.append(unit.net.units[syn.preID])
                self.snorm_delys.append(syn.delay_steps)
        self.val = 0.2  # an arbitrary initialization of the average input value
        self.n_vec = np.ones(len(self.snorm_units)) # the 'n' vector from Pg.290 of Dayan&Abbott

    def update(self, time):
        # first, update the n vector from Eq. 8.14, pg. 290 in Dayan & Abbott
        self.n_vec = [ 1. if syn.w>0. else 0. for syn in self.snorm_syns ]
        
        self.val = sum([n*(u.get_lpf_fast(s)) for n,u,s in 
                                zip(self.n_vec, self.snorm_units, self.snorm_delys)]) / sum(self.n_vec)


