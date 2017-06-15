'''
sirasi.py  (SImple RAte SImulator)
A simple simulator of rate units with delayed connections.

Author: Sergio Verduzco Flores
June 2017
'''

import numpy as np
from scipy.interpolate import interp1d # to interpolate values
from scipy.integrate import odeint # to integrate ODEs
import pylab                      # for plotting tools
import matplotlib.pyplot as plt   # more plotting tools
from enum import Enum  # enumerators for the synapse and unit types
import random # to create random connections

# These enumeration classes provide human-readable and globally consistent constants
class unit_types(Enum):
    source = 1
    sigmoidal = 2
    
class synapse_types(Enum):
    static = 1
    oja = 2  
    #axoaxo = auto()  # axo-axonic synapses will provide presynaptic inhibition


''' This class has the tools to build a network.
    First, you create an instance of 'network'; 
    second, you use the create() method to add units;
    third, you set_function() for source units;
    fourth, you use the connect() method to connect the units, 
    finally you use the run() method to run a simulation.
'''
class network():
    def __init__(self, params):
        self.sim_time = 0.0  # current simulation time [ms]
        self.n_units = 0     # current number of units in the network
        self.units = []      # list with all the unit objects
        # The next 3 lists implement the connectivity of the network
        self.delays = [] # delays[i][j] is the delay of the j-th connection to unit i in [ms]
        self.act = []    # act[i][j] is the function from which unit i obtains its j-th input
        self.syns = []   # syns[i][j] is the synapse object for the j-th connection to unit i
        self.min_delay = params['min_delay'] # minimum transmission delay [ms]
        self.min_buff_size = params['min_buff_size']  # number of values stored during a minimum delay period
        
    def create(self, n, params):
        # create 'n' units of type 'params['type']' and parameters from 'params'
        assert (type(n) == int) and (n > 0), 'Number of units must be a positive integer'
        assert self.sim_time == 0., 'Units are being created when the simulation time is not zero'
        
        unit_list = list(range(self.n_units, self.n_units + n))
        if params['type'] == unit_types.source:
            default_fun = lambda x: 0.  # source units start with a null function
            for ID in unit_list:
                self.units.append(source(ID,params,default_fun,self))
        elif params['type'] == unit_types.sigmoidal:
            for ID in unit_list:
                self.units.append(sigmoidal(ID,params,self))
        else:
            printf('Attempting to create an unknown model type. Nothing done.')
            return []
        
        self.n_units += n
        # putting n new slots in the delays, act, and syns lists
        self.delays += [[] for i in range(n)]
        self.act += [[] for i in range(n)]
        self.syns += [[] for i in range(n)]
        # note:  [[]]*n causes all empty lists to be the same object 
        return unit_list
    
    def connect(self, from_list, to_list, conn_spec, syn_spec):
        # connect the units in the 'from_list' to the units in the 'to_list' using the
        # connection specifications in the 'conn_spec' dictionary, and the
        # synapse specfications in the 'syn_spec' dictionary
        # The current version only handles one single delay value
        
        # A quick test first
        if (np.amax(from_list + to_list) > self.n_units-1) or (np.amin(from_list + to_list) < 0):
            print('Attempting to connect units with an ID out of range. Nothing done.')
            return
       
        # Let's find out the synapse type
        if syn_spec['type'] == synapse_types.static:
            syn_class = static_synapse
        elif syn_spec['type'] == synapse_types.oja:
            syn_class = oja_synapse
        else:
            print('Attempting connect with an unknown synapse type. Nothing done.')
            return
        
        # To specify connectivity, you need to update 3 lists: delays, act, and syns
        # The units connected depend on the connectivity rule in conn_spec
        if conn_spec['rule'] == 'fixed_outdegree':
            sources = from_list
            targets = random.sample(conn_spec['outdegree'], to_list)
        elif conn_spec['rule'] == 'all_to_all':
            targets = to_list
            sources = from_list
        else:
            print('Attempting connect with an unknown rule. Nothing done.')
            return
            
        for source in sources:
            for target in targets:
                if source == target and conn_spec['allow_autapses'] == False:
                    continue     
                self.delays[target].append(conn_spec['delay'])
                self.act[target].append(self.units[source].get_act)
                syn_params = syn_spec # a copy of syn_spec just for this connection
                syn_params['preID'] = source
                syn_params['postID'] = target
                self.syns[target].append(syn_class(syn_params, self))
                if self.units[source].delay <= conn_spec['delay']: # this is the longest delay for this source
                    self.units[source].delay = conn_spec['delay']+self.min_delay
                    self.units[source].init_buffers()
    
    def run(self, total_time):
        # A basic runner.
        # It takes steps of 'min_delay' length, in which the units and synapses 
        # use their own methods to advance their state variables.
        Nsteps = int(total_time/self.min_delay)
        storage = [np.zeros(Nsteps) for i in range(self.n_units)]
        times = np.zeros(Nsteps)
        
        for step in range(Nsteps):
            self.sim_time = self.min_delay*step
            times[step] = self.sim_time
            
            # store current state
            for unit in range(self.n_units):
                storage[unit][step] = self.units[unit].get_act(self.sim_time)
                
            # update units
            for unit in range(self.n_units):
                self.units[unit].update(self.sim_time)

            # update synapses
            for pre_list in self.syns:
                for pre in pre_list:
                    pre.update(self.sim_time)
            
        return times, storage        

class unit():
    # I'll probably move a bunch of the __init__ code to sigmoidal, since source doesn't need it
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
    
    ''' This function returns the derivatives of the state variables at a given point in time.
    '''
    def derivatives(self, y, t):
        # at this point there is only one state variable (the activity)
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

class synapse():
    def __init__(self, params, network):
        self.preID = params['preID']   # the ID of the presynaptic unit 
        self.postID = params['postID'] # the ID of the postsynaptic unit
        self.w = params['init_w'] # initializing the synaptic weight
        self.type = params['type'] # assigning the synapse type
        self.net = network # the network where the synapse lives
        assert self.net.n_units >= self.preID, 'Synapse connected from non existing unit ' + str(self.preID)  
        assert self.net.n_units >= self.postID, 'Synapse connected to non existing unit ' + str(self.postID)  
        assert self.net.sim_time == 0., 'Synapse being created when sim_time is not zero'

    def get_w(self, time): 
        # If I'm going to use time, I need to use buffers and interpolation...
        return self.w

    def update(self,time):
        # The default update rule does nothing
        return

    
class static_synapse(synapse):
    def __init__(self, params, network):
        super(static_synapse, self).__init__(params, network)
        assert self.type is synapse_types.static, ['Synapse from ' + str(self.preID) + ' to ' +
                                                            str(self.postID) + ' instantiated with the wrong type']
          
class oja_synapse(synapse):
    ''' This class implements a continuous version of the Oja learning rule. 
        Since there are no discrete inputs, the presynaptic activity is tracked with a low-pass filter,
        and used to adapt the weight everytime the "update" function is called.
    '''
    def __init__(self, params, network):
        super(oja_synapse, self).__init__(params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.tau_x = params['tau_x'] # time constant used to track the presynaptic activity
        self.tau_y = params['tau_y'] # time constant used to track the postsynaptic activity
        self.last_time = self.net.sim_time # time of the last call to the update function
        self.lpf_x = self.net.units[self.preID].get_act(self.last_time) # low-pass filtered presynaptic activity
        self.lpf_y = self.net.units[self.postID].get_act(self.last_time) # low-pass filtered postsynaptic activity
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        assert self.type is synapse_types.oja, ['Synapse from ' + str(self.preID) + ' to ' +
                                                          str(self.postID) + ' instantiated with the wrong type']

    
    def update(self, time):
        assert time >= self.last_time, ['Synapse from ' + str(self.preID) + ' to ' +
                                         str(self.postID) + ' updated backwards in time']
        cur_x = self.net.units[self.preID].get_act(time) # current presynaptic activity
        cur_y = self.net.units[self.postID].get_act(time) # current postsynaptic activity
        
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.lpf_x = cur_x + (self.lpf_x - cur_x)*np.exp((self.last_time-time)/self.tau_x)
        self.lpf_y = cur_y + (self.lpf_y - cur_y)*np.exp((self.last_time-time)/self.tau_y)
        
        # The Oja learning rule
        self.w = self.w + self.alpha * self.lpf_y * ( self.lpf_x - self.lpf_y*self.w )
