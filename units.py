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
    '''
    The parent class of all unit models.
    '''
    def __init__(self, ID, params, network):
        """ The class constructor.

        Args:
            ID: The unique integer identifier of the unit within the network;
                it is usually assigned by network.connect().
            params: A dictionary with parameters to initialize the unit.
                REQUIRED PARAMETERS
                'type' : A unit type from the unit_types enum.
                'init_val' : initial value for the activation (not used for source units).
                OPTIONAL PARAMETERS
                'delay': maximum delay among the projections sent by the unit.
                'coordinates' : a tuple specifying the spatial location of the unit.
                'tau_fast' : time constant for the fast low-pass filter.
                'tau_mid' : time constant for the medium-speed low-pass filter.
                'tau_slow' : time constant for the slow low-pass filter.
                'n_ports' : number of inputs ports. Defaults to 1.
            network: the network where the unit lives.

        Raises:
            AssertionError.
        """

        self.ID = ID # unit's unique identifier
        # Copying parameters from dictionary
        # Inside the class there are no dictionaries to avoid their retrieval operations
        self.type = params['type'] # an enum identifying the type of unit being instantiated
        self.init_val = params['init_val'] # initial value for the activation (for units that use buffers)
        self.net = network # the network where the unit lives
        # The delay of a unit is the maximum delay among the projections it sends. 
        # Its final value of 'delay' should be set by network.connect(), after the unit is created.
        if 'delay' in params: 
            self.delay = params['delay']
            # delay must be a multiple of net.min_delay. Next line checks that.
            assert (self.delay+1e-6)%self.net.min_delay < 2e-6, ['unit' + str(self.ID) + 
                                                                 ': delay is not a multiple of min_delay']       
        else:  # giving a temporary value
            self.delay = 2 * self.net.min_delay 
        # These are the optional parameters. 
        # Default values are sometimes omitted so an error can arise if the parameter was needed.
        if 'coordinates' in params: self.coordinates = params['coordinates']
        # These are the time constants for the low-pass filters (used for plasticity).
        if 'tau_fast' in params: self.tau_fast = params['tau_fast']
        if 'tau_mid' in params: self.tau_mid = params['tau_mid']
        if 'tau_slow' in params: self.tau_slow = params['tau_slow']
        if 'n_ports' in params: self.n_ports = params['n_ports']
        else: self.n_ports = 1

        self.syn_needs = set() # the set of all variables required by synaptic dynamics
                               # It is initialized by the init_pre_syn_update function
        self.last_time = 0  # time of last call to the update function
        
        self.init_buffers() # This will create the buffers that store states and times
        
        self.pre_syn_update = lambda time : None # See init_pre_syn_update below

     
    def init_buffers(self):
        """
        This method (re)initializes the buffer variables according to the current parameters.

        It is useful because new connections may increase self.delay, and thus the size of the buffers.
        """
    
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

        
    def get_inputs(self, time):
        ''' 
        Returns a list with the inputs received by the unit from all other units at time 'time'.

        The returned inputs already account for the transmission delays.
        To do this: in the network's activation tower the entry corresponding to the unit's ID
        (e.g. self.net.act[self.ID]) is a list; for each i-th entry (a function) retrieve
        the value at time "time - delays[ID][i]".  
        '''
        return [ fun(time - dely) for dely,fun in zip(self.net.delays[self.ID], self.net.act[self.ID]) ]
        # A possibly slower option:
        #return [ fun(time - self.net.delays[self.ID][idx]) for idx, fun in enumerate(self.net.act[self.ID]) ]
    

    def get_weights(self, time):
        ''' Returns a list with the weights corresponding to the input list obtained with get_inputs.
        '''
        # once you include axo-axonic connections you have to modify the list below
        return [ synapse.get_w(time) for synapse in self.net.syns[self.ID] ]

    def update(self,time):
        '''
        Advance the dynamics from time to time+min_delay.

        This update function will replace the values in the activation buffer 
        corresponding to the latest "min_delay" time units, introducing "min_buff_size" new values.
        In addition, all the synapses of the unit are updated.
        source units override this with a shorter update function.
        '''

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


    def get_act(self,time):
        ''' Gives you the activity at a previous time 't' (within buffer range).

        This version works for units that store their previous activity values in a buffer.
        Units without buffers (e.g. source units) have their own get_act function.
        '''
        # Sometimes the ode solver asks about values slightly out of bounds, so I set this to extrapolate
        return interp1d(self.times, self.buffer, kind='linear', bounds_error=False, copy=False,
                        fill_value="extrapolate", assume_sorted=True)(time)

  
    def init_pre_syn_update(self):
        '''
        Create a pre_syn_update function according to current synaptic requirements.

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
        For each requirement, the unit class should already have the function to calculate it,
        so all init_pre_syn_update needs to do is to ensure that a call to pre_syn_update
        executes all the right functions.

        init_pre_syn_update is called for a unit everytime network.connect() connects it, 
        which may be more than once.

        Raises:
            NameError, NotImplementedError.
        ''' 
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
                    if (syn_reqs.pre_lpf_fast or syn_reqs.inp_avg) in syn.upd_requirements:
                        self.syn_needs.add(syn_reqs.lpf_fast)
                    if syn_reqs.pre_lpf_mid in syn.upd_requirements:
                        self.syn_needs.add(syn_reqs.lpf_mid)
                    if syn_reqs.pre_lpf_slow in syn.upd_requirements:
                        self.syn_needs.add(syn_reqs.lpf_slow)

        # Create the pre_syn_update function and the associated variables
        if not hasattr(self, 'functions'): # so we don't erase previous requirements
            self.functions = set() 

        for req in self.syn_needs:
            if req is syn_reqs.lpf_fast:
                if not hasattr(self,'tau_fast'): 
                    raise NameError( 'Synaptic plasticity required unit parameter tau_fast, not yet set' )
                self.lpf_fast = self.init_val
                self.functions.add(self.upd_lpf_fast)
            elif req is syn_reqs.lpf_slow:
                if not hasattr(self,'tau_slow'): 
                    raise NameError( 'Synaptic plasticity required unit parameter tau_slow, not yet set' )
                self.lpf_slow = self.init_val
                self.functions.add(self.upd_lpf_slow)
            elif req is syn_reqs.inp_avg:
                self.snorm_list = []  # a list with all the presynaptic neurons 
                                      # providing hebbsnorm synapses
                for syn in self.net.syns[self.ID]:
                    if syn.type is synapse_types.hebbsnorm:
                        self.snorm_list.append(self.net.units[syn.preID])
                self.n_hebbsnorm = len(self.snorm_list) # number of hebbsnorm synapses received
                self.inp_avg = 0.2  # an arbitrary initialization of the average input value
                self.functions.add(self.upd_inp_avg)
            else:
                raise NotImplementedError('Asking for a requirement not implemented yet')

        self.pre_syn_update = lambda time: [f(time) for f in self.functions]


    def upd_lpf_fast(self,time):
        """ Update the lpf_fast variable. """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_fast updated backwards in time']
        cur_act = self.get_act(time) # current activity
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.lpf_fast = cur_act + ( (self.lpf_fast - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_fast ) )

    def upd_lpf_slow(self,time):
        """ Update the lpf_slow variable. """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_slow updated backwards in time']
        cur_act = self.get_act(time) # current activity
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.lpf_slow = cur_act + ( (self.lpf_slow - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_slow ) )

    def upd_inp_avg(self, time):
        """ Update the average of the inputs with hebbsnorm synapses. """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' inp_avg updated backwards in time']
        self.inp_avg = sum([u.lpf_fast for u in self.snorm_list]) / self.n_hebbsnorm
        


class source(unit):
    """ The class of units whose activity comes from some Python function.
    
        source units serve two main roles. First, they can provide inputs to the
        network. Second, they can receive the outputs from plants, and thus 
        communicate them to other units in the network.
    """
    
    def __init__(self, ID, params, funct, network):
        """ The class constructor.

        In practice, the function giving the activity of the unit is set after the
        constructor has been called, using source.set_function .

        Args:
            ID, params, network : same as in the parent class (unit).
            funct : Reference to a Python function that gives the activity of the unit.

        Raises:
            AssertionError.
        """
        super(source, self).__init__(ID, params, network)
        self.get_act = funct  # the function which returns activation given the time
        assert self.type is unit_types.source, ['Unit ' + str(self.ID) + 
                                                ' instantiated with the wrong type']

    def set_function(self, function):
        """ 
        Set the function determiing the unit's activity value.

        Args:
            function: a reference to a Python function.
        """
        self.get_act = function
        # What if you're doing this after the connections have already been made?
        # Then net.act has links to functions other than this get_act.
        # Thus, we need to reset all those net.act entries...
        for idx1, syn_list in enumerate(self.net.syns):
            for idx2, syn in enumerate(syn_list):
                if syn.preID == self.ID:
                    self.net.act[idx1][idx2] = self.get_act

        # The same goes when the connection is to a plant instead of a unit...
        for plant in self.net.plants:
            for syn_list, inp_list in zip(plant.inp_syns, plant.inputs):
                for idx, syn in enumerate(syn_list):
                    if syn.preID == self.ID:
                        inp_list[idx] = self.get_act
                        
                        

    def update(self, time):
        """ 
        Update the unit's state variables.

        In the case of source units, update() only updates any values being used by
        synapses where it is the presynaptic component.
        """

        self.pre_syn_update(time) # update any variables needed for the synapse to update
        self.last_time = time # last_time is used to update some pre_syn_update values


    
class sigmoidal(unit): 
    # An implementation of a typical sigmoidal unit. Its output is produced by linearly
    # summing the inputs (times the synaptic weights), and feeding the sum to a sigmoidal
    # function, which constraints the output to values beween zero and one.
    # Because this unit operates in real time, it updates its value gradually, with
    # a 'tau' time constant.
    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS
                'slope' : Slope of the sigmoidal function.
                'thresh' : Threshold of the sigmoidal function.
                'tau' : Time constant of the update dynamics.

        Raises:
            AssertionError.

        """

        super(sigmoidal, self).__init__(ID, params, network)
        self.slope = params['slope']    # slope of the sigmoidal function
        self.thresh = params['thresh']  # horizontal displacement of the sigmoidal
        self.tau = params['tau']  # the time constant of the dynamics
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        assert self.type is unit_types.sigmoidal, ['Unit ' + str(self.ID) + 
                                                            ' instantiated with the wrong type']
        
    def f(self, arg):
        ''' This is the sigmoidal function. Could roughly think of it as an f-I curve. '''
        return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
    
    def derivatives(self, y, t):
        ''' This function returns the derivatives of the state variables at a given point in time. '''
        # there is only one state variable (the activity)
        scaled_inputs = sum([inp*w for inp,w in zip(self.get_inputs(t), self.get_weights(t))])
        return ( self.f(scaled_inputs) - y[0] ) * self.rtau
    

class linear(unit): 
    # An implementation of a linear unit.
    # The output is the sum of the inputs multiplied by their synaptic weights.
    # The output upates with time constant 'tau'.
    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS
                'tau' : Time constant of the update dynamics.

        Raises:
            AssertionError.

        """
        super(linear, self).__init__(ID, params, network)
        self.tau = params['tau']  # the time constant of the dynamics
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        assert self.type is unit_types.linear, ['Unit ' + str(self.ID) + 
                                                            ' instantiated with the wrong type']
        
    def derivatives(self, y, t):
        ''' This function returns the derivatives of the state variables at a given point in time. '''
        # there is only one state variable (the activity)
        scaled_inputs = sum([inp*w for inp,w in zip(self.get_inputs(t), self.get_weights(t))])
        return ( scaled_inputs - y[0] ) * self.rtau
   

