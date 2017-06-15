'''
units.py
This file contains all the unit models used in the sirasi simulator.
'''

from sirasi import unit_types, synapse_types  # the names of the models for units and synapses
from synapses import *
import numpy as np
from scipy.interpolate import interp1d # to interpolate values
from scipy.integrate import odeint # to integrate ODEs


class unit():
    def __init__(self, ID, params, network):
        self.ID = ID # unit's unique identifier
        
        # Copying parameters from dictionary
        # A dictionary is not used inside the class to avoid dictionary retrieval operations
        self.coordinates = params['coordinates'] # spatial coordinates of the unit [cm]
        self.delay = params['delay']             # maximum delay on sending projections [ms]
        self.type = params['type'] # an enum identifying the type of unit being instantiated
        self.init_val = params['init_val'] # initial value for the activation (for units that use buffers)
        self.net = network # the network where the unit lives
        
        # The delay of a unit is the maximum delay among the projections it sends. It
        # must be a multiple of min_delay. Next line checks that.
        assert (self.delay+1e-6)%self.net.min_delay < 2e-6, ['unit' + str(self.ID) + 
                                                             ': delay is not a multiple of min_delay']       
        self.init_buffers() # This will create the buffers that store states and times
     
    def init_buffers(self):
        # This method (re)initializes the buffer variables according to the current parameters.
        # It is useful because new connections may increase self.delay, and thus the size of the buffers.
    
        assert self.net.sim_time == 0., 'Buffers are being reset when the simulation time is not zero'
        
        # 'source' units don't use buffers, so for them the method ends here
        if self.type == unit_types.source:
            return
        
        min_del = self.net.min_delay  # just to have shorter lines below
        min_buff = self.net.min_buff_size
        self.steps = int(round(self.delay/min_del)) # delay, in units of the minimum delay
        self.offset = (self.steps-1)*min_buff # an index used in the update function of derived classes
        self.buff_size = int(round(self.steps*min_buff)) # number of activation values to store
        self.buffer = np.array( [self.init_val]*self.buff_size ) # numpy array with previous activation values
        self.times = np.linspace(-self.delay, 0., self.buff_size) # the corresponding times for the buffer values
        self.times_grid = np.linspace(0, min_del, min_buff+1) # used to create values for 'times'
        
    ''' get_inputs(self, time)
        Returns a list with the inputs received by the unit from all other units at time 'time'.
        The returned inputs already account for the transmission delays.
        To do this: in the network's activation tower the entry corresponding to the unit's ID
        (e.g. self.net.act[self.ID]) is a list; for each i-th entry (a function) retrieve
        the value at time "time - delays[ID][i]".  
    '''
    def get_inputs(self, time):
        # the one-liner
        return [ fun(time - self.net.delays[self.ID][idx]) for idx, fun in enumerate(self.net.act[self.ID]) ]
    
    ''' get_weights will return a list with the synaptic weights corresponding to the input
        list obtained with get_inputs
    '''
    def get_weights(self, time):
        # once you include axo-axonic connections you have to modify the list below
        return [ synapse.get_w(time) for synapse in self.net.syns[self.ID]]

    # The default update function does nothing
    def update(self, time):
        return
                                
class source(unit):
    # these guys are 'functional', as they may also have inputs, but no dynamics.
    # At some point I'll have units whose output depends on the plant's dynamics. 
    # Should I call them afferent units? transducer units?
    def __init__(self, ID, params, funct, network):
        super(source, self).__init__(ID, params, network)
        self.get_act = funct  # the function which returns activation given the time
        assert self.type is unit_types.source, ['Unit ' + str(self.ID) + 
                                                         ' instantiated with the wrong type']
    def set_function(self, function):
        self.get_act = function

class sigmoidal(unit): 
    def __init__(self, ID, params, network):
        super(sigmoidal, self).__init__(ID, params, network)
        self.slope = params['slope']    # slope of the sigmoidal function
        self.thresh = params['thresh']  # horizontal displacement of the sigmoidal
        self.tau = params['tau']  # the time constant of the dynamics
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        assert self.type is unit_types.sigmoidal, ['Unit ' + str(self.ID) + 
                                                            ' instantiated with the wrong type']
        
    ''' get_act(t)' gives you the activity at a previous time 't' (within buffer range)
    '''
    def get_act(self,time):
        # Sometimes the ode solver asks about values slightly out of bounds, so I set this to extrapolate
        return interp1d(self.times, self.buffer, kind='linear', bounds_error=False, copy=False,
                        fill_value="extrapolate", assume_sorted=True)(time)
        
    ''' This is the sigmoidal function. Could roughly think of it as an f-I curve
    '''
    def f(self, arg):
        return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
    
    ''' This function returns the derivatives of the state variables at a given point in time.
    '''
    def derivatives(self, y, t):
        # there is only one state variable (the activity)
        scaled_inputs = sum([inp*w for inp,w in zip(self.get_inputs(t), self.get_weights(t))])
        return ( self.f(scaled_inputs) - y[0] ) * self.rtau
    
    '''
        Everytime you update, you'll replace the values in the activation buffer 
        corresponding to the latest "min_delay" time units, introducing "min_buff_size" new values.
        In the future this may be done with a faster ring buffer (memoryview?)
    '''
    def update(self,time):
        # the 'time' argument is currently only used to ensure times buffer is in sync
        # Maybe there should be a single 'times' array in the network. This seems more parallelizable, though.
        assert (self.times[-1]-time) < 2e-6, 'unit' + str(self.ID) + ': update time is desynchronized'

        new_times = self.times[-1] + self.times_grid
        self.times = np.roll(self.times, -self.net.min_buff_size)
        self.times[self.offset:] = new_times[1:] 
        
        # odeint also returns the initial condition, so to produce min_buff_size new values
        # we need to provide min_buff_size+1 desired times, starting with the one for the initial condition
        new_buff = odeint(self.derivatives, [self.buffer[-1]], new_times)
        self.buffer = np.roll(self.buffer, -self.net.min_buff_size)
        self.buffer[self.offset:] = new_buff[1:,0] 


class linear(unit): 
    def __init__(self, ID, params, network):
        super(linear, self).__init__(ID, params, network)
        self.tau = params['tau']  # the time constant of the dynamics
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        assert self.type is unit_types.linear, ['Unit ' + str(self.ID) + 
                                                            ' instantiated with the wrong type']
        
    ''' get_act(t)' gives you the activity at a previous time 't' (within buffer range)
    '''
    def get_act(self,time):
        # Sometimes the ode solver asks about values slightly out of bounds, so I set this to extrapolate
        return interp1d(self.times, self.buffer, kind='linear', bounds_error=False, copy=False,
                        fill_value="extrapolate", assume_sorted=True)(time)
        
    
    ''' This function returns the derivatives of the state variables at a given point in time.
    '''
    def derivatives(self, y, t):
        # there is only one state variable (the activity)
        scaled_inputs = sum([inp*w for inp,w in zip(self.get_inputs(t), self.get_weights(t))])
        return ( scaled_inputs - y[0] ) * self.rtau
    
    '''
        Everytime you update, you'll replace the values in the activation buffer 
        corresponding to the latest "min_delay" time units, introducing "min_buff_size" new values.
        In the future this may be done with a faster ring buffer (memoryview?)
    '''
    def update(self,time):
        # the 'time' argument is currently only used to ensure times buffer is in sync
        # Maybe there should be a single 'times' array in the network. This seems more parallelizable, though.
        assert (self.times[-1]-time) < 2e-6, 'unit' + str(self.ID) + ': update time is desynchronized'

        new_times = self.times[-1] + self.times_grid
        self.times = np.roll(self.times, -self.net.min_buff_size)
        self.times[self.offset:] = new_times[1:] 
        
        # odeint also returns the initial condition, so to produce min_buff_size new values
        # we need to provide min_buff_size+1 desired times, starting with the one for the initial condition
        new_buff = odeint(self.derivatives, [self.buffer[-1]], new_times)
        self.buffer = np.roll(self.buffer, -self.net.min_buff_size)
        self.buffer[self.offset:] = new_buff[1:,0] 



