'''
units.py
This file contains all the unit models used in the sirasi simulator.
'''

from sirasi import unit_types, synapse_types, syn_reqs  # names of models and requirements
from synapses import *
import numpy as np
from scipy.interpolate import interp1d # to interpolate values
from scipy.integrate import odeint # to integrate ODEs


class unit():
    def __init__(self, ID, params, network):
        self.ID = ID # unit's unique identifier
        # Copying parameters from dictionary
        # Inside the class there are no dictionaries to avoid their retrieval operations
        self.coordinates = params['coordinates'] # spatial coordinates of the unit [cm]
        self.delay = params['delay']             # maximum delay on sending projections [ms]
        self.type = params['type'] # an enum identifying the type of unit being instantiated
        self.init_val = params['init_val'] # initial value for the activation (for units that use buffers)
        # These are the time constants for the low-pass filters (used for plasticity).
        # They are optional. No default value is given.
        if 'tau_fast' in params: self.tau_fast = params['tau_fast']
        if 'tau_mid' in params: self.tau_mid = params['tau_mid']
        if 'tau_slow' in params: self.tau_slow = params['tau_slow']
        self.net = network # the network where the unit lives
        self.syn_needs = set() # the set of all variables required by synaptic dynamics
                               # It is initialized by the init_pre_syn_update function
        self.last_time = 0  # time of last call to the update function
        
        # The delay of a unit is the maximum delay among the projections it sends. It
        # must be a multiple of min_delay. Next line checks that.
        assert (self.delay+1e-6)%self.net.min_delay < 2e-6, ['unit' + str(self.ID) + 
                                                             ': delay is not a multiple of min_delay']       
        self.init_buffers() # This will create the buffers that store states and times
        
        self.pre_syn_update = lambda time : None # See init_pre_syn_update below

     
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

        
    ''' 
        get_inputs(self, time)
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
        return [ synapse.get_w(time) for synapse in self.net.syns[self.ID] ]

    '''
        This update function will replace the values in the activation buffer 
        corresponding to the latest "min_delay" time units, introducing "min_buff_size" new values.
        In addition, all the synapses of the unit are updated.
        source units replace this with an empty update function.
    '''
    def update(self,time):
        # the 'time' argument is currently only used to ensure the 'times' buffer is in sync
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

        self.pre_syn_update(time) # update any variables needed for the synapse to update
        # For each synapse on the unit, update its state
        for pre in self.net.syns[self.ID]:
            pre.update(time)

        self.last_time = time # last_time is used to update some pre_syn_update values


    ''' get_act(t)' gives you the activity at a previous time 't' (within buffer range)
        This version works for units that store their previous activity values in a buffer.
        Units without buffers (e.g. source units) have their own get_act function.
    '''
    def get_act(self,time):
        # Sometimes the ode solver asks about values slightly out of bounds, so I set this to extrapolate
        return interp1d(self.times, self.buffer, kind='linear', bounds_error=False, copy=False,
                        fill_value="extrapolate", assume_sorted=True)(time)
  
    '''
        Correlational learning rules require the pre- and post-synaptic activity, in this
        case low-pass filtered in order to implement a running average. Moreover, for
        heterosynaptic plasticity individual synapses need information about all the
        other synapses on the unit. It is inefficient for each synapse to maintain
        low-pass filtered versions of pre- and post-synaptic activity, as well as to
        obtain by itself all the values required for its update.
        The function pre_syn_update(), initialized by init_pre_syn_update(), 
        is tasked with updating all the unit variables used by the synapses to update.

        init_pre_syn_update creates the function pre_syn_update, and initializes all the
        variables it requires. To do this, it compiles the requirements of all the
        relevant synapses in the set self.syn_needs. 
        For each requirement, the unit class already has the function to calculate it,
        so all init_pre_syn_update needs to do is to ensure that a call to pre_syn_update
        executes all the right functions.

        init_pre_syn_update is called for a unit everytime network.connect() connects it, 
        which may be more than once.
    '''
    def init_pre_syn_update(self):
    
        assert self.net.sim_time == 0, ['Tried to run init_pre_syn_update for unit ' + 
                                         str(self.ID) + ' when simulation time is not zero']
        # For each synapse you receive, add its requirements
        for syn in self.net.syns[self.ID]:
            self.syn_needs.update(syn.upd_requirements)

        pre_reqs = set([syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_mid, syn_reqs.pre_lpf_slow])
        self.syn_needs.difference_update(pre_reqs) # the "pre_" requirements are handled below

        # For each projection you send, check if its synapse needs the lpf presynaptic activity
        for syn_list in self.net.syns:
            for syn in syn_list:
                if syn.preID == self.ID:
                    if syn_reqs.pre_lpf_fast in syn.upd_requirements:
                        self.syn_needs.add(syn_reqs.lpf_fast)
                    if syn_reqs.pre_lpf_mid in syn.upd_requirements:
                        self.syn_needs.add(syn_reqs.lpf_mid)
                    if syn_reqs.pre_lpf_slow in syn.upd_requirements:
                        self.syn_needs.add(syn_reqs.lpf_slow)

        # Create the pre_syn_update function and the associated variables
        functions = set() 
        for req in self.syn_needs:
            if req is syn_reqs.lpf_fast:
                if not hasattr(self,'tau_fast'): raise NameError(self.ID)
                self.lpf_fast = self.init_val
                functions.add(self.upd_lpf_fast)
            else:
                raise NotImplementedError('Asking for a requirement not implemented yet')

        self.pre_syn_update = lambda time: [f(time) for f in functions]


    def upd_lpf_fast(self,time):
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_fast updated backwards in time']
        cur_act = self.get_act(time) # current activity
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.lpf_fast = cur_act + ( (self.lpf_fast - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_fast ) )


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
        #print('Assignment to unit ' + str(self.ID) + ' has happened')
        self.get_act = function
        # What if you're doing this after the connections have already been made?
        # Then net.act has links to functions other than this get_act.
        # Thus, we need to reset all those net.act entries...
        for idx1, syn_list in enumerate(self.net.syns):
            for idx2, syn in enumerate(syn_list):
                if syn.preID == self.ID:
                    self.net.act[idx1][idx2] = self.get_act

        #print('My get_act('+str(self.net.sim_time)+') = ' + str(self.get_act(self.net.sim_time)))

    def update(self, time):
        self.pre_syn_update(time) # update any variables needed for the synapse to update
        self.last_time = time # last_time is used to update some pre_syn_update values


    
class sigmoidal(unit): 
    # An implementation of a typical sigmoidal unit. Its output is produced by linearly
    # summing the inputs (times the synaptic weights), and feeding the sum to a sigmoidal
    # function, which constraints the output to values beween zero and one.
    # Because this unit operates in real time, it updates its value gradually, with
    # a 'tau' time constant.
    def __init__(self, ID, params, network):
        super(sigmoidal, self).__init__(ID, params, network)
        self.slope = params['slope']    # slope of the sigmoidal function
        self.thresh = params['thresh']  # horizontal displacement of the sigmoidal
        self.tau = params['tau']  # the time constant of the dynamics
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        assert self.type is unit_types.sigmoidal, ['Unit ' + str(self.ID) + 
                                                            ' instantiated with the wrong type']
        
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
    

class linear(unit): 
    # An implementation of a linear unit.
    # The output is the sum of the inputs multiplied by their synaptic weights.
    # The output upates with time constant 'tau'.
    def __init__(self, ID, params, network):
        super(linear, self).__init__(ID, params, network)
        self.tau = params['tau']  # the time constant of the dynamics
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        assert self.type is unit_types.linear, ['Unit ' + str(self.ID) + 
                                                            ' instantiated with the wrong type']
        
    ''' This function returns the derivatives of the state variables at a given point in time.
    '''
    def derivatives(self, y, t):
        # there is only one state variable (the activity)
        scaled_inputs = sum([inp*w for inp,w in zip(self.get_inputs(t), self.get_weights(t))])
        return ( scaled_inputs - y[0] ) * self.rtau
   

