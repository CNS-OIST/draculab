"""
units.py
This file contains all the unit models used in the draculab simulator.
"""

from draculab import unit_types, synapse_types, syn_reqs  # names of models and requirements
from synapses import *
import numpy as np
from scipy.integrate import odeint # to integrate ODEs
from cython_utils import cython_get_act  # the cythonized linear interpolation 
#from scipy.interpolate import interp1d # to interpolate values


class unit():
    """
    The parent class of all unit models.
    """
    def __init__(self, ID, params, network):
        """ The class constructor.

        Args:
            ID: The unique integer identifier of the unit within the network;
                it is usually assigned by network.connect().
            params: A dictionary with parameters to initialize the unit.
                REQUIRED PARAMETERS
                'type' : A unit type from the unit_types enum.
                'init_val' : initial value for the activation 
                OPTIONAL PARAMETERS
                'delay': maximum delay among the projections sent by the unit.
                'coordinates' : a numpy array specifying the spatial location of the unit.
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
        self.net = network # the network where the unit lives
        self.rtol = self.net.rtol # local copies of the rtol and atol tolerances
        self.atol = self.net.atol 
        self.init_val = params['init_val'] # initial value for the activation (for units that use buffers)
        self.min_buff_size = network.min_buff_size # a local copy just to avoid the extra reference
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

        if self.n_ports < 2:
            self.multiport = False # If True, the port_idx list is created in init_pre_syn_update in
                                    # order to support customized get_mp_input* functions 
        else:
            self.multiport = True

        self.syn_needs = set() # the set of all variables required by synaptic dynamics
                               # It is initialized by the init_pre_syn_update function
        self.last_time = 0  # time of last call to the update function
                            # Used by the upd_lpf_X functions
        self.init_buffers() # This will create the buffers that store states and times
        
        self.pre_syn_update = lambda time : None # See init_pre_syn_update below

     
    def init_buffers(self):
        """
        This method (re)initializes the buffer variables according to the current parameters.

        It is useful because new connections may increase self.delay, and thus the size of the buffers.

        """
    
        assert self.net.sim_time == 0., 'Buffers are being reset when the simulation time is not zero'

        min_del = self.net.min_delay  # just to have shorter lines below
        self.steps = int(round(self.delay/min_del)) # delay, in units of the minimum delay

        # The following buffers are for the low-pass filterd variables required by synaptic plasticity.
        # They only store one value per update. Updated by upd_lpf_X
        if syn_reqs.lpf_fast in self.syn_needs:
            self.lpf_fast_buff = np.array( [self.init_val]*self.steps )
        if syn_reqs.lpf_mid in self.syn_needs:
            self.lpf_mid_buff = np.array( [self.init_val]*self.steps )
        if syn_reqs.lpf_slow in self.syn_needs:
            self.lpf_slow_buff = np.array( [self.init_val]*self.steps )
        if syn_reqs.lpf_mid_inp_sum in self.syn_needs:
            self.lpf_mid_inp_sum_buff = np.array( [self.init_val]*self.steps )

        # 'source' units don't use activity buffers, so for them the method ends here
        if self.type == unit_types.source:
            return
        
        min_buff = self.min_buff_size
        self.offset = (self.steps-1)*min_buff # an index used in the update function of derived classes
        self.buff_size = int(round(self.steps*min_buff)) # number of activation values to store
        self.buffer = np.array( [self.init_val]*self.buff_size ) # numpy array with previous activation values
        self.times = np.linspace(-self.delay, 0., self.buff_size) # the corresponding times for the buffer values
        self.times_grid = np.linspace(0, min_del, min_buff+1) # used to create values for 'times'
        self.time_bit = self.times[1] - self.times[0] + 1e-9 # time interval used by get_act.
        
        
    def get_inputs(self, time):
        """ 
        Returns a list with the inputs received by the unit from all other units at time 'time'.

        The returned inputs already account for the transmission delays.
        To do this: in the network's activation tower the entry corresponding to the unit's ID
        (e.g. self.net.act[self.ID]) is a list; for each i-th entry (a function) retrieve
        the value at time "time - delays[ID][i]".  

        This function ignores input ports.
        """
        return [ fun(time - dely) for dely,fun in zip(self.net.delays[self.ID], self.net.act[self.ID]) ]


    def get_input_sum(self,time):
        """ Returns the sum of all inputs at the given time, each scaled by its synaptic weight. 
        
        The sum accounts for transmission delays. Input ports are ignored. 
        """
        # original implementation is below
        #return sum([ syn.w * fun(time-dely) for syn,fun,dely in zip(self.net.syns[self.ID], 
        #                self.net.act[self.ID], self.net.delays[self.ID]) ])
        # second implementation is below
        return sum( map( lambda x: (x[0].w) * (x[1](time-x[2])), 
                    zip(self.net.syns[self.ID], self.net.act[self.ID], self.net.delays[self.ID]) ) )


    def get_mp_inputs(self, time):
        """
        Returns a list with all the inputs, arranged by input port.

        This method is for units where multiport = True, and that have a port_idx attribute.

        The i-th element of the returned list is a numpy array containing the raw (not multiplied
        by the synaptic weight) inputs at port i. The inputs include transmision delays. 
        """
        return [ np.array([self.net.act[self.ID][idx](time - self.net.delays[self.ID][idx]) for idx in idx_list]) 
                 for idx_list in self.port_idx ]


    def get_sc_input_sum(self, time):
        """ 
        Returns the sum of inputs, each scaled by its synaptic weight and by a gain constant. 
        
        The sum accounts for transmission delays. Input ports are ignored. 

        The extra scaling factor is the 'gain' attribute of the synapses. This is useful when
        different types of inputs have different gains applied to them. You then need a unit
        model that calls get_sc_input_sum instead of get_input_sum. 
        """
        return sum([ syn.gain * syn.w * fun(time-dely) for syn,fun,dely in zip(self.net.syns[self.ID], 
                        self.net.act[self.ID], self.net.delays[self.ID]) ])


    def get_exp_sc_input_sum(self, time):
        """ 
        Returns the sum of inputs, each scaled by its weight and by a scale factor.
        
        The sum accounts for transmission delays. Input ports are ignored. 

        The scale factor is applied only to excitatory synapses, and it is the scale_facs value
        set by the upd_exp_scale function. This is the way that exp_dist_sigmoidal units get
        their total input.
        """
        # This accelerates the simulation, but inp_vector is incorrect (only updates every min_delay).
        #weights = np.array([ syn.w for syn in self.net.syns[self.ID] ])
        #return sum( self.scale_facs * weights * self.inp_vector )
        return sum( [ sc * syn.w * fun(time-dely) for sc,syn,fun,dely in zip(self.scale_facs, 
                      self.net.syns[self.ID], self.net.act[self.ID], self.net.delays[self.ID]) ] )


    def get_weights(self, time):
        """ Returns a list with the weights corresponding to the input list obtained with get_inputs.
        """
        # if you include axo-axonic connections you have to modify the list below
        return [ synapse.get_w(time) for synapse in self.net.syns[self.ID] ]


    def get_mp_weights(self, time):
        """ 
        Returns a list with the weights corresponding to the list obtained with get_mp_inputs.

        This method is for units where multiport = True, and that have a port_idx attribute.

        The i-th element of the returned list is a numpy array with the weights of the
        synapses where port = i.
        """
        return [ np.array([self.net.syns[self.ID][idx].w for idx in idx_list]) for idx_list in self.port_idx ]


    def update(self,time):
        """
        Advance the dynamics from time to time+min_delay.

        This update function will replace the values in the activation buffer 
        corresponding to the latest "min_delay" time units, introducing "min_buff_size" new values.
        In addition, all the synapses of the unit are updated.
        source and kwta units override this with shorter update functions.
        """
        # the 'time' argument is currently only used to ensure the 'times' buffer is in sync
        # Maybe there should be a single 'times' array in the network. This seems more parallelizable, though.
        #assert (self.times[-1]-time) < 2e-6, 'unit' + str(self.ID) + ': update time is desynchronized'

        new_times = self.times[-1] + self.times_grid
        self.times = np.roll(self.times, -self.min_buff_size)
        self.times[self.offset:] = new_times[1:] 
        
        # odeint also returns the initial condition, so to produce min_buff_size new values
        # we need to provide min_buff_size+1 desired times, starting with the one for the initial condition
        new_buff = odeint(self.derivatives, [self.buffer[-1]], new_times, rtol=self.rtol, atol=self.atol)
        self.buffer = np.roll(self.buffer, -self.min_buff_size)
        self.buffer[self.offset:] = new_buff[1:,0] 

        self.pre_syn_update(time) # Update any variables needed for the synapse to update.
                                  # It is important this is done after the buffer has been updated.
        # For each synapse on the unit, update its state
        for pre in self.net.syns[self.ID]:
            pre.update(time)

        self.last_time = time # last_time is used to update some pre_syn_update values


    def get_act(self,time):
        """ Gives you the activity at a previous time 't' (within buffer range).

        This version works for units that store their previous activity values in a buffer.
        Units without buffers (e.g. source units) have their own get_act function.

        This is the most time-consuming method in draculab (thus the various optimizations).
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Below is the more general (but slow) interpolation using interp1d
        # Sometimes the ode solver asks about values slightly out of bounds, so I set this to extrapolate
        """
        return interp1d(self.times, self.buffer, kind='linear', bounds_error=False, copy=False,
                        fill_value="extrapolate", assume_sorted=True)(time)
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Below the code for the second implementation.
        # This linear interpolation takes advantage of the ordered, regularly-spaced buffer.
        # Time values outside the buffer range receive the buffer endpoints.
        # The third implementation is faster, but for small buffer sizes in test2.ipynb this
        # gives more exact results. I can't figure out why.
        """
        time = min( max(time,self.times[0]), self.times[-1] ) # clipping 'time'
        frac = (time-self.times[0])/(self.times[-1]-self.times[0])
        base = int(np.floor(frac*(self.buff_size-1))) # biggest index s.t. times[index] <= time
        frac2 = ( time-self.times[base] ) / ( self.times[min(base+1,self.buff_size-1)] - self.times[base] + 1e-8 )
        return self.buffer[base] + frac2 * ( self.buffer[min(base+1,self.buff_size-1)] - self.buffer[base] )
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This is the third implementation. 
        # Takes advantage of the regularly spaced times using divmod.
        # Values outside the buffer range will fall between buffer[-1] and buffer[-2].
        """
        base, rem = divmod(time-self.times[0], self.time_bit)
        # because time_bit is slightly larger than times[1]-times[0], we can limit
        # base to buff_size-2, even if time = times[-1]
        base =  max( 0, min(int(base), self.buff_size-2) ) 
        frac2 = rem/self.time_bit
        return self.buffer[base] + frac2 * ( self.buffer[base+1] - self.buffer[base] )
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # The fourth implementation uses the same algorithm as the third, with Cython
        return cython_get_act(time, self.times[0], self.time_bit, self.buff_size, self.buffer)

  
    def init_pre_syn_update(self):
        """
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

        In addition, for each one of the unit's synapses, init_pre_syn_update will initialize 
        its delay value.

        An extra task done here is to prepare the 'port_idx' list used by units with multiple 
        input ports.

        init_pre_syn_update is called for a unit everytime network.connect() connects the unit, 
        which may be more than once.

        Raises:
            NameError, NotImplementedError, ValueError.
        """ 
        assert self.net.sim_time == 0, ['Tried to run init_pre_syn_update for unit ' + 
                                         str(self.ID) + ' when simulation time is not zero']

        # Each synapse should know the delay of its connection
        for syn, delay in zip(self.net.syns[self.ID], self.net.delays[self.ID]):
            # The -1 below is because get_lpf_fast etc. return lpf_fast_buff[-1-steps], corresponding to
            # the assumption that buff[-1] is the value zero steps back
            syn.delay_steps = min(self.net.units[syn.preID].steps-1, int(round(delay/self.net.min_delay)))

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

        # If we require support for multiple input ports, create the port_idx list.
        # port_idx is a list whose elements are lists of integers.
        # port_idx[i] contains the indexes in net.syns[self.ID] (or net.delays[self.ID], or net.act[self.ID])
        # of the synapses whose input port is 'i'.
        if self.multiport is True:
            self.port_idx = [ [] for _ in range(self.n_ports) ]
            for idx, syn in enumerate(self.net.syns[self.ID]):
                if syn.port >= self.n_ports:
                    raise ValueError('Found a synapse input port larger than or equal to the number of ports')
                else:
                    self.port_idx[syn.port].append(idx) 

        # Create the pre_syn_update function and the associated variables
        if not hasattr(self, 'functions'): # so we don't erase previous requirements
            self.functions = set() 

        for req in self.syn_needs:
            if req is syn_reqs.lpf_fast:  # <----------------------------------
                if not hasattr(self,'tau_fast'): 
                    raise NameError( 'Synaptic plasticity requires unit parameter tau_fast, not yet set' )
                self.lpf_fast = self.init_val
                self.functions.add(self.upd_lpf_fast)
            elif req is syn_reqs.lpf_mid:  # <----------------------------------
                if not hasattr(self,'tau_mid'): 
                    raise NameError( 'Synaptic plasticity requires unit parameter tau_mid, not yet set' )
                self.lpf_mid = self.init_val
                self.functions.add(self.upd_lpf_mid)
            elif req is syn_reqs.lpf_slow:  # <----------------------------------
                if not hasattr(self,'tau_slow'): 
                    raise NameError( 'Synaptic plasticity requires unit parameter tau_slow, not yet set' )
                self.lpf_slow = self.init_val
                self.functions.add(self.upd_lpf_slow)
            elif req is syn_reqs.sq_lpf_slow:  # <----------------------------------
                if not hasattr(self,'tau_slow'): 
                    raise NameError( 'Synaptic plasticity requires unit parameter tau_slow, not yet set' )
                self.sq_lpf_slow = self.init_val
                self.functions.add(self.upd_sq_lpf_slow)
            elif req is syn_reqs.inp_vector: # <----------------------------------
                self.inp_vector = np.tile(self.init_val, len(self.net.syns[self.ID]))
                self.functions.add(self.upd_inp_vector)
            elif req is syn_reqs.mp_inputs: # <----------------------------------
                if not hasattr(self,'port_idx'): 
                    raise NameError( 'the mp_inputs requirement is for multiport units with a port_idx list' )
                self.mp_inputs = [] 
                for prt_lst in self.port_idx:
                    self.mp_inputs.append(np.array([self.init_val for _ in range(len(prt_lst))]))
                self.functions.add(self.upd_mp_inputs)
            elif req is syn_reqs.inp_avg:  # <----------------------------------
                self.snorm_list = []  # a list with all the presynaptic units
                                      # providing hebbsnorm synapses
                self.snorm_dels = []  # a list with the delay steps for each connection from snorm_list
                for syn in self.net.syns[self.ID]:
                    if syn.type is synapse_types.hebbsnorm:
                        self.snorm_list.append(self.net.units[syn.preID])
                        self.snorm_dels.append(syn.delay_steps)
                self.n_hebbsnorm = len(self.snorm_list) # number of hebbsnorm synapses received
                self.inp_avg = 0.2  # an arbitrary initialization of the average input value
                self.snorm_list_dels = list(zip(self.snorm_list, self.snorm_dels)) # both lists zipped
                self.functions.add(self.upd_inp_avg)
            elif req is syn_reqs.pos_inp_avg:  # <----------------------------------
                self.snorm_units = []  # a list with all the presynaptic units
                                      # providing hebbsnorm synapses
                self.snorm_syns = []  # a list with the synapses for the list above
                self.snorm_delys = []  # a list with the delay steps for these synapses
                for syn in self.net.syns[self.ID]:
                    if syn.type is synapse_types.hebbsnorm:
                        self.snorm_syns.append(syn)
                        self.snorm_units.append(self.net.units[syn.preID])
                        self.snorm_delys.append(syn.delay_steps)
                self.pos_inp_avg = 0.2  # an arbitrary initialization of the average input value
                self.n_vec = np.ones(len(self.snorm_units)) # the 'n' vector from Pg.290 of Dayan&Abbott
                self.functions.add(self.upd_pos_inp_avg)
            elif req is syn_reqs.err_diff:  # <----------------------------------
                self.err_idx = [] # a list with the indexes of the units that provide errors
                self.pred_idx = [] # a list with the indexes of the units that provide predictors 
                self.err_dels = [] # a list with the delays for each unit in err_idx
                self.err_diff = 0. # approximation of error derivative, updated by upd_err_diff
                for syn in self.net.syns[self.ID]:
                    if syn.type is synapse_types.inp_corr:
                        if syn.input_type == 'error':
                            self.err_idx.append(syn.preID)
                            self.err_dels.append(syn.delay_steps)
                        elif syn.input_type == 'pred': 
                            self.pred_idx.append(syn.preID) # not currently using this list
                        else:
                            raise ValueError('Incorrect input_type ' + str(syn.input_type) + ' found in synapse')
                self.err_idx_dels = list(zip(self.err_idx, self.err_dels)) # both lists zipped
                self.functions.add(self.upd_err_diff)
            elif req is syn_reqs.sc_inp_sum: # <----------------------------------
                sq_snorm_units = [] # a list with all the presynaptic neurons providing
                                        # sq_hebbsnorm synapses
                sq_snorm_syns = []  # a list with all the sq_hebbsnorm synapses
                sq_snorm_dels = []  # a list with the delay steps for the sq_hebbsnorm synapses
                for syn in self.net.syns[self.ID]:
                    if syn.type is synapse_types.sq_hebbsnorm:
                        sq_snorm_syns.append(syn)
                        sq_snorm_units.append(self.net.units[syn.preID])
                        sq_snorm_dels.append(syn.delay_steps)
                self.u_d_syn = list(zip(sq_snorm_units, sq_snorm_dels, sq_snorm_syns)) # all lists zipped
                self.sc_inp_sum = 0.2  # an arbitrary initialization of the scaled input value
                self.functions.add(self.upd_sc_inp_sum)
            elif req is syn_reqs.diff_avg: # <----------------------------------
                self.dsnorm_list = [] # list with all presynaptic units providing
                                      # diff_hebb_subsnorm synapses
                self.dsnorm_dels = [] # list with delay steps for each connection in dsnorm_list
                for syn in self.net.syns[self.ID]:
                    if syn.type is synapse_types.diff_hebbsnorm:
                        self.dsnorm_list.append(self.net.units[syn.preID])
                        self.dsnorm_dels.append(syn.delay_steps)
                self.n_dhebbsnorm = len(self.dsnorm_list) # number of diff_hebbsnorm synapses received
                self.diff_avg = 0.2  # an arbitrary initialization of the input derivatives average 
                self.dsnorm_list_dels = list(zip(self.dsnorm_list, self.dsnorm_dels)) # both lists zipped
                self.functions.add(self.upd_diff_avg)
            elif req is syn_reqs.lpf_mid_inp_sum:  # <----------------------------------
                if not hasattr(self,'tau_mid'): 
                    raise NameError( 'Synaptic plasticity requires unit parameter tau_mid, not yet set' )
                if not syn_reqs.inp_vector in self.syn_needs:
                    raise AssertionError('lpf_mid_inp_sum requires the inp_vector requirement to be set')
                self.lpf_mid_inp_sum = self.init_val # this initialization is rather arbitrary
                self.functions.add(self.upd_lpf_mid_inp_sum)
            elif req is syn_reqs.n_erd:  # <---------------------------------- ONLY USED IN LEGACY CODE
                self.n_erd = len([s for s in self.net.syns[self.ID] if s.type is synapse_types.exp_rate_dist])
                # n_erd doesn't need to be updated :)
            elif req is syn_reqs.balance:  # <----------------------------------
                if not syn_reqs.inp_vector in self.syn_needs:
                    raise AssertionError('balance requirement has the inp_vector requirement as a prerequisite')
                self.below = 0.5 # this initialization is rather arbitrary
                self.above = 0.5 
                self.functions.add(self.upd_balance)
            elif req is syn_reqs.balance_mp:  # <----------------------------------
                if not syn_reqs.mp_inputs in self.syn_needs:
                    raise AssertionError('balance_mp requirement has the mp_inputs requirement as a prerequisite')
                self.below = 0.5 # this initialization is rather arbitrary
                self.above = 0.5 
                self.functions.add(self.upd_balance_mp)
            elif req is syn_reqs.exp_scale:  # <----------------------------------
                if not syn_reqs.balance in self.syn_needs:
                    raise AssertionError('exp_scale requires the balance requirement to be set')
                if not syn_reqs.lpf_fast in self.syn_needs:
                    raise AssertionError('exp_scale requires the lpf_fast requirement to be set')
                self.scale_facs= np.tile(1., len(self.net.syns[self.ID])) # array with scale factors
                # exc_idx = numpy array with index of all excitatory units in the input vector 
                self.exc_idx = [ idx for idx,syn in enumerate(self.net.syns[self.ID]) if syn.w >= 0]
                # ensure the integer data type; otherwise you can't index numpy arrays
                self.exc_idx = np.array(self.exc_idx, dtype='uint32')
                # inh_idx = numpy array with index of all inhibitory units 
                self.inh_idx = [idx for idx,syn in enumerate(self.net.syns[self.ID]) if syn.w < 0]
                self.inh_idx = np.array(self.inh_idx, dtype='uint32')
                self.functions.add(self.upd_exp_scale)
            elif req is syn_reqs.exp_scale_mp:  # <----------------------------------
                if not syn_reqs.balance_mp in self.syn_needs:
                    raise AssertionError('exp_scale_mp requires the balance_mp requirement to be set')
                if not syn_reqs.lpf_fast in self.syn_needs:
                    raise AssertionError('exp_scale_mp requires the lpf_fast requirement to be set')
                self.scale_facs_rdc = np.tile(1., len(self.port_idx[self.rdc_port])) # array with scale factors
                # exc_idx_rdc = numpy array with index of all excitatory units at rdc port
                self.exc_idx_rdc = [ idx for idx,syn in enumerate([self.net.syns[self.ID][i] 
                                     for i in self.port_idx[self.rdc_port]]) if syn.w >= 0 ]
                # ensure the integer data type; otherwise you can't index numpy arrays
                self.exc_idx_rdc = np.array(self.exc_idx_rdc, dtype='uint32')
                self.functions.add(self.upd_exp_scale_mp)
            elif req is syn_reqs.slide_thresh:  # <----------------------------------
                if (not syn_reqs.balance in self.syn_needs) and (not syn_reqs.balance_mp in self.syn_needs):
                    raise AssertionError('slide_thresh requires the balance(_mp) requirement to be set')
                self.functions.add(self.upd_thresh)
            elif req is syn_reqs.slide_thr_shrp:  # <----------------------------------
                if not syn_reqs.balance_mp in self.syn_needs:
                    raise AssertionError('slide_thr_shrp requires the balance_mp requirement to be set')
                if not self.multiport:
                    raise AssertionError('The slide_thr_shrp is for multiport units only')
                self.functions.add(self.upd_thr_shrp)
            elif req is syn_reqs.lpf_slow_mp_inp_sum:  # <----------------------------------
                if not hasattr(self,'tau_slow'): 
                    raise NameError( 'Requirement lpf_slow_mp_inp_sum requires parameter tau_slow, not yet set' )
                if not syn_reqs.mp_inputs in self.syn_needs:
                    raise AssertionError('lpf_slow_mp_inp_sum requires the mp_inputs requirement')
                self.lpf_slow_mp_inp_sum = [ 0.2 for _ in range(self.n_ports) ]
                self.functions.add(self.upd_lpf_slow_mp_inp_sum)
            elif req is syn_reqs.slide_thr_hr:  # <----------------------------------
                if not syn_reqs.mp_inputs in self.syn_needs:
                    raise AssertionError('slide_thr_hr requires the mp_inputs requirement to be set')
                if not self.multiport:
                    raise AssertionError('The slide_thr_hr requirment is for multiport units only')
                self.functions.add(self.upd_slide_thr_hr)
            elif req is syn_reqs.syn_scale_hr:  # <----------------------------------
                if not syn_reqs.mp_inputs in self.syn_needs:
                    raise AssertionError('syn_scale_hr requires the mp_inputs requirement to be set')
                if not self.multiport:
                    raise AssertionError('The syn_scale_hr requirment is for multiport units only')
                if not syn_reqs.lpf_fast in self.syn_needs:
                    raise AssertionError('syn_scale_hr requires the lpf_fast requirement to be set')
                # exc_idx_hr = numpy array with index of all excitatory units at hr port
                self.exc_idx_hr = [ idx for idx,syn in enumerate([self.net.syns[self.ID][i] 
                                     for i in self.port_idx[self.hr_port]]) if syn.w >= 0 ]
                # ensure the integer data type; otherwise you can't index numpy arrays
                self.exc_idx_hr = np.array(self.exc_idx_hr, dtype='uint32')
                #self.scale_facs = np.tile(1., len(self.net.syns[self.ID])) # array with scale factors
                self.scale_facs_hr = np.tile(1., len(self.port_idx[self.hr_port])) # array with scale factors
                self.functions.add(self.upd_syn_scale_hr)
            elif req is syn_reqs.exp_scale_shrp:  # <----------------------------------
                if not syn_reqs.balance_mp in self.syn_needs:
                    raise AssertionError('exp_scale_shrp requires the balance_mp requirement to be set')
                if not self.multiport:
                    raise AssertionError('The exp_scale_shrp requirement is for multiport units only')
                if not syn_reqs.lpf_fast in self.syn_needs:
                    raise AssertionError('exp_scale_shrp requires the lpf_fast requirement to be set')
                self.scale_facs_rdc = np.tile(1., len(self.port_idx[self.rdc_port])) # array with scale factors
                # exc_idx_rdc = numpy array with index of all excitatory units at rdc port
                self.exc_idx_rdc = [ idx for idx,syn in enumerate([self.net.syns[self.ID][i] 
                                     for i in self.port_idx[self.rdc_port]]) if syn.w >= 0 ]
                # ensure the integer data type; otherwise you can't index numpy arrays
                self.exc_idx_rdc = np.array(self.exc_idx_rdc, dtype='uint32')
                self.rdc_exc_ones = np.tile(1., len(self.exc_idx_rdc))
                self.functions.add(self.upd_exp_scale_shrp)
            elif req is syn_reqs.error:  # <----------------------------------
                if not syn_reqs.mp_inputs in self.syn_needs:
                    raise AssertionError('The error requirement needs the mp_inputs requirement.')
                self.error = 0.
                self.learning = 0.
                self.functions.add(self.upd_error)
            elif req is syn_reqs.inp_l2:  # <----------------------------------
                if not syn_reqs.mp_inputs in self.syn_needs:
                    raise AssertionError('The inp_l2 requirement needs the mp_inputs requirement.')
                self.inp_l2 = 1.
                self.functions.add(self.upd_inp_l2)
            else:  # <----------------------------------------------------------------------
                raise NotImplementedError('Asking for a requirement that is not implemented')

        self.pre_syn_update = lambda time: [f(time) for f in self.functions]

        

    def upd_lpf_fast(self,time):
        """ Update the lpf_fast variable. """
        # Source units have their own implementation
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_fast updated backwards in time']
        cur_act = self.buffer[-1] # This doesn't work for source units
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.lpf_fast = cur_act + ( (self.lpf_fast - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_fast ) )
        # update the buffer
        self.lpf_fast_buff = np.roll(self.lpf_fast_buff, -1)
        self.lpf_fast_buff[-1] = self.lpf_fast


    def get_lpf_fast(self, steps):
        """ Get the fast low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_fast_buff[-1-steps]


    def upd_lpf_mid(self,time):
        """ Update the lpf_mid variable. """
        # Source units have their own implementation
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_mid updated backwards in time']
        cur_act = self.buffer[-1] # This doesn't work for source units
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.lpf_mid = cur_act + ( (self.lpf_mid - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_mid) )
        # update the buffer
        self.lpf_mid_buff = np.roll(self.lpf_mid_buff, -1)
        self.lpf_mid_buff[-1] = self.lpf_mid


    def get_lpf_mid(self, steps):
        """ Get the mid-speed low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_mid_buff[-1-steps]


    def upd_lpf_slow(self,time):
        """ Update the lpf_slow variable. """
        # Source units have their own implementation
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_slow updated backwards in time']
        cur_act = self.buffer[-1] # This doesn't work for source units
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.lpf_slow = cur_act + ( (self.lpf_slow - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_slow ) )
        # update the buffer
        self.lpf_slow_buff = np.roll(self.lpf_slow_buff, -1)
        self.lpf_slow_buff[-1] = self.lpf_slow


    def get_lpf_slow(self, steps):
        """ Get the slow low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_slow_buff[-1-steps]


    def upd_sq_lpf_slow(self,time):
        """ Update the sq_lpf_slow variable. """
        # Source units have their own implementation
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' sq_lpf_slow updated backwards in time']
        cur_sq_act = self.buffer[-1]**2.  # This doesn't work for source units.
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.sq_lpf_slow = cur_sq_act + ( (self.sq_lpf_slow - cur_sq_act) * 
                                  np.exp( (self.last_time-time)/self.tau_slow ) )


    def upd_inp_avg(self, time):
        """ Update the average of the inputs with hebbsnorm synapses. 
        
            The actual value being averaged is lpf_fast of the presynaptic units.
        """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' inp_avg updated backwards in time']
        self.inp_avg = sum([u.get_lpf_fast(s) for u,s in self.snorm_list_dels]) / self.n_hebbsnorm
        

    def upd_pos_inp_avg(self, time):
        """ Update the average of the inputs with hebbsnorm synapses. 
        
            The actual value being averaged is lpf_fast of the presynaptic units.
            Inputs whose synaptic weight is zero or negative  are saturated to zero and
            excluded from the computation.
        """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' pos_inp_avg updated backwards in time']
        # first, update the n vector from Eq. 8.14, pg. 290 in Dayan & Abbott
        self.n_vec = [ 1. if syn.w>0. else 0. for syn in self.snorm_syns ]
        
        self.pos_inp_avg = sum([n*(u.get_lpf_fast(s)) for n,u,s in 
                                zip(self.n_vec, self.snorm_units, self.snorm_delys)]) / sum(self.n_vec)


    def upd_err_diff(self, time):
        """ Update an approximate derivative of the error inputs used for input correlation learning. 

            A very simple approach is taken, where the derivative is approximated as the difference
            between the fast and medium low-pass filtered inputs. Each input arrives with its
            corresponding transmission delay.
        """
        self.err_diff = ( sum([ self.net.units[i].get_lpf_fast(s) for i,s in self.err_idx_dels ]) -
                          sum([ self.net.units[i].get_lpf_mid(s) for i,s in self.err_idx_dels ]) )
       

    def upd_sc_inp_sum(self, time):
        """ Update the sum of the inputs multiplied by their synaptic weights.
        
            The actual value being summed is lpf_fast of the presynaptic units.
        """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' sc_inp_sum updated backwards in time']
        self.sc_inp_sum = sum([u.get_lpf_fast(d) * syn.w for u,d,syn in self.u_d_syn])
        
        
    def upd_diff_avg(self, time):
        """ Update the average of derivatives from inputs with diff_hebbsnorm synapses.

            The values being averaged are not the actual derivatives, but approximations
            which are roughly proportional to them, coming from the difference 
            lpf_fast - lpf_mid . 
        """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' diff_avg updated backwards in time']
        self.diff_avg = ( sum([u.get_lpf_fast(s) - u.get_lpf_mid(s) for u,s in self.dsnorm_list_dels]) 
                          / self.n_dhebbsnorm )

    def upd_lpf_mid_inp_sum(self,time):
        """ Update the lpf_mid_inp_sum variable. """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_mid_inp_sum updated backwards in time']
        cur_inp_sum = (self.inp_vector).sum()
        #cur_inp_sum = sum(self.get_inputs(time))
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.lpf_mid_inp_sum = cur_inp_sum + ( (self.lpf_mid_inp_sum - cur_inp_sum) * 
                                   np.exp( (self.last_time-time)/self.tau_mid) )
        # update the buffer
        self.lpf_mid_inp_sum_buff = np.roll(self.lpf_mid_inp_sum_buff, -1)
        self.lpf_mid_inp_sum_buff[-1] = self.lpf_mid_inp_sum


    def get_lpf_mid_inp_sum(self):
        """ Get the latest value of the mid-speed low-pass filtered sum of inputs. """
        return self.lpf_mid_inp_sum_buff[-1]


    def upd_lpf_slow_mp_inp_sum(self, time):
        """ Update the slow LPF'd scaled sum of inputs at individual ports, returning them in a list. """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_slow_mp_inp_sum updated backwards in time']
        inputs = self.mp_inputs  # updated due to the mp_inputs synaptic requirement
        weights = self.get_mp_weights(time)
        dots = [ np.dot(i, w) for i, w in zip(inputs, weights) ]
        # same update rule from other upd_lpf_X methods above, put in a list comprehension
        self.lpf_slow_mp_inp_sum = [dots[i] + (self.lpf_slow_mp_inp_sum[i] - dots[i]) * 
                                   np.exp( (self.last_time-time)/self.tau_slow ) for i in range(self.n_ports)]


    def upd_balance(self, time):
        """ Updates two numbers called  below, and above.

            below = fraction of inputs with rate lower than this unit.
            above = fraction of inputs with rate higher than this unit.

            Those numbers are useful to produce a given firing rate distribtuion.

            NOTICE: this version does not restrict inputs to exp_rate_dist synapses.
        """
        #inputs = np.array(self.get_inputs(time)) # current inputs
        inputs = self.inp_vector
        N = len(inputs)
        r = self.buffer[-1] # current rate
        self.above = ( 0.5 * (np.sign(inputs - r) + 1.).sum() ) / N
        self.below = ( 0.5 * (np.sign(r - inputs) + 1.).sum() ) / N
        #assert abs(self.above+self.below - 1.) < 1e-5, ['sum was not 1: ' + 
        #                            str(self.above + self.below)]


    def upd_balance_mp(self, time):
        """ Updates two numbers called  below, and above. Used in units with multiple input ports.

            below = fraction of inputs with rate lower than this unit.
            above = fraction of inputs with rate higher than this unit.

            Those numbers are useful to produce a given firing rate distribtuion among
            the population of units that connect to port 0.

            This is the same as upd_balance, but ports other than the rdc_port are ignored.
        """
        inputs = self.mp_inputs[self.rdc_port]
        N = len(inputs)
        r = self.buffer[-1] # current rate
        self.above = ( 0.5 * (np.sign(inputs - r) + 1.).sum() ) / N
        self.below = ( 0.5 * (np.sign(r - inputs) + 1.).sum() ) / N

    
    def upd_exp_scale(self, time):
        """ Updates the synaptic scaling factor used in exp_dist_sigmoidal units.

            The algorithm is a multiplicative version of the  one used in exp_rate_dist synapses.
            It scales all excitatory inputs.
        """
        #r = self.get_lpf_fast(0)
        r = self.buffer[-1] # current rate
        r = max( min( .995, r), 0.005 ) # avoids bad arguments and overflows
        exp_cdf = ( 1. - np.exp(-self.c*r) ) / ( 1. - np.exp(-self.c) )
        error = self.below - self.above - 2.*exp_cdf + 1. 

        # First APCTP version (12/13/17)
        ######################################################################
        #u = (np.log(r/(1.-r))/self.slope) + self.thresh  # this u lags, so I changed it
        weights = np.array([syn.w for syn in self.net.syns[self.ID]])
        #u = np.dot(self.inp_vector[self.exc_idx], weights[self.exc_idx])
        #u = self.get_input_sum(time)
        u = self.get_exp_sc_input_sum(time)
        I = np.sum( self.inp_vector[self.inh_idx] * weights[self.inh_idx] ) 
        #I = u - np.sum( self.inp_vector[self.exc_idx] * weights[self.exc_idx] ) 
        mu_exc = np.maximum( np.sum( self.inp_vector[self.exc_idx] ), 0.001 )
        #fpr = 1. / (self.c * r * (1. - r)) # reciprocal of the sigmoidal's derivative
        #ss_scale = (u - I + self.Kp * fpr * error) / mu_exc
        ss_scale = (u - I + self.Kp * error) / mu_exc
        #self.scale_facs[self.exc_idx] += self.tau_scale * (ss_scale/np.maximum(weights[self.exc_idx],.05) - 
        #                                                    self.scale_facs[self.exc_idx])
        # ------------ soft weight-bounded version ------------
        # Soft weight bounding implies that the scale factors follow the logistic differential 
        # equation. This equation has the form x' = r (a - x) x, and has a solution
        # x(t) = a x(0) / [ x(0) + (a - x(0))*exp(-a r t) ] .
        # In our case, r = tau_scale, and a = ss_scale / weights[self.ID] .
        # We can use this analytical solution to update the scale factors on each update.
        a = ss_scale / np.maximum(weights[self.exc_idx],.001)
        a = np.minimum( a, 10.) # hard bound above
        x0 = self.scale_facs[self.exc_idx]
        t = self.net.min_delay
        self.scale_facs[self.exc_idx] = (x0 * a) / ( x0 + (a - x0) * np.exp(-self.tau_scale * a * t) )
        #self.scale_facs[self.exc_idx] += self.tau_scale * self.scale_facs[self.exc_idx] * (
        #                                 ss_scale/np.maximum(weights[self.exc_idx],.05) - 
        #                                 self.scale_facs[self.exc_idx] )


    def upd_exp_scale_mp(self, time):
        """ Updates the synaptic scaling factor used in multiport ssrdc units.

            The algorithm is the same as upd_exp_scale, but only the inputs at the rdc_port
            are considered.
        """
        #r = self.get_lpf_fast(0)  # lpf'd rate
        r = self.buffer[-1] # current rate
        r = max( min( .995, r), 0.005 ) # avoids bad arguments and overflows
        exp_cdf = ( 1. - np.exp(-self.c*r) ) / ( 1. - np.exp(-self.c) )
        error = self.below - self.above - 2.*exp_cdf + 1. 

        # First APCTP version (12/13/17)
        ######################################################################
        #u = (np.log(r/(1.-r))/self.slope) + self.thresh
        rdc_inp = self.mp_inputs[self.rdc_port]
        rdc_w = self.get_mp_weights(time)[self.rdc_port]
        u = np.sum(self.scale_facs_rdc[self.exc_idx_rdc]*rdc_inp[self.exc_idx_rdc]*rdc_w[self.exc_idx_rdc])
        I = u - self.get_mp_input_sum(time)     
        mu_exc = np.maximum( np.sum( rdc_inp[self.exc_idx_rdc] ), 0.001 )
        #fpr = 1. / (self.c * r * (1. - r)) # reciprocal of the sigmoidal's derivative
        #ss_scale = (u - I + self.Kp * fpr * error) / mu_exc # adjusting for sigmoidal's slope
        ss_scale = (u - I + self.Kp * error) / mu_exc
        #self.scale_facs[self.exc_idxrdc] += self.tau_scale * (ss_scale/np.maximum(weights[self.exc_idx_rdc],.05) - 
        #                                                    self.scale_facs[self.exc_idx_rdc])
        # ------------ weight-bounded version ------------
        # Soft weight bounding implies that the scale factors follow the logistic differential 
        # equation. This equation has the form x' = r (a - x) x, and has a solution
        # x(t) = a x(0) / [ x(0) + (a - x(0))*exp(-a r t) ] .
        # In our case, r = tau_scale, and a = ss_scale / weights[self.ID] .
        # We can use this analytical solution to update the scale factors on each update.
        a = ss_scale / np.maximum(rdc_w[self.exc_idx_rdc],.001)
        a = np.minimum( a, 10.) # hard bound above
        x0 = self.scale_facs_rdc[self.exc_idx_rdc]
        t = self.net.min_delay
        self.scale_facs_rdc[self.exc_idx_rdc] = (x0 * a) / ( x0 + (a - x0) * np.exp(-self.tau_scale * a * t) )


    def upd_exp_scale_shrp(self, time):
        """ Updates the synaptic scaling factor used in ssrdc_sharp units.

            The algorithm is the same as upd_exp_scale_mp, but controlled by the inputs at the
            sharpening port. When the sum of inputs at this port is smaller than 0.5 the scale factors
            return to 1 with a time constant of tau_relax.
        """
        if np.dot(self.mp_inputs[self.sharpen_port], self.get_mp_weights(time)[self.sharpen_port]) < 0.5:
            a = self.rdc_exc_ones   
            exp_tau = self.tau_relax
        else: 
            #r = self.get_lpf_fast(0)  # lpf'd rate
            r = self.buffer[-1] # current rate
            r = max( min( .995, r), 0.005 ) # avoids bad arguments and overflows
            exp_cdf = ( 1. - np.exp(-self.c*r) ) / ( 1. - np.exp(-self.c) )
            error = self.below - self.above - 2.*exp_cdf + 1. 

            #u = (np.log(r/(1.-r))/self.slope) + self.thresh
            rdc_inp = self.mp_inputs[self.rdc_port]
            rdc_w = self.get_mp_weights(time)[self.rdc_port]
            u = np.sum(self.scale_facs_rdc[self.exc_idx_rdc]*rdc_inp[self.exc_idx_rdc]*rdc_w[self.exc_idx_rdc])
            I = u - self.get_mp_input_sum(time)     
            mu_exc = np.maximum( np.sum( rdc_inp[self.exc_idx_rdc] ), 0.001 )
            #fpr = 1. / (self.c * r * (1. - r)) # reciprocal of the sigmoidal's derivative
            #ss_scale = (u - I + self.Kp * fpr * error) / mu_exc # adjusting for sigmoidal's slope
            ss_scale = (u - I + self.Kp * error) / mu_exc
            a = ss_scale / np.maximum(rdc_w[self.exc_idx_rdc],.001)
            a = np.minimum( a, 10.) # hard bound above
            exp_tau = self.tau_scale
        
        x0 = self.scale_facs_rdc[self.exc_idx_rdc]
        self.scale_facs_rdc[self.exc_idx_rdc] = (x0*a) / (x0+(a-x0)*np.exp(-exp_tau*a*self.net.min_delay))

        
    def upd_inp_vector(self, time):
        """ Update a numpy array containing all the current synaptic inputs """
        self.inp_vector = np.array([ fun(time - dely) for dely,fun in 
                                     zip(self.net.delays[self.ID], self.net.act[self.ID]) ])


    def upd_mp_inputs(self, time):
        """ Update a copy of the results of get_mp_inputs. """
        self.mp_inputs = self.get_mp_inputs(time)


    def upd_thresh(self, time):
        """ Updates the threshold of exp_dist_sig_thr units and some other 'trdc' units.

            The algorithm is an adapted version of the  one used in exp_rate_dist synapses.
        """
        r = self.get_lpf_fast(0)
        r = max( min( .995, r), 0.005 ) # avoids bad arguments and overflows
        exp_cdf = ( 1. - np.exp(-self.c*r) ) / ( 1. - np.exp(-self.c) )
        error = self.below - self.above - 2.*exp_cdf + 1. 
        self.thresh -= self.tau_thr * error
        self.thresh = max(min(self.thresh, 10.), -10.) # clipping large/small values
        

    def upd_thr_shrp(self, time):
        """ Updates the threshold of trdc units when input at 'sharpen' port is larger than 0.5 .

            The algorithm is based on upd_thresh.
        """
        idxs = self.port_idx[self.sharpen_port]
        inps = self.mp_inputs[self.sharpen_port]
        ws = [self.net.syns[self.ID][idx].w for idx in idxs]
        sharpen = sum( [w*i for w,i in zip(ws, inps)] )
            
        if sharpen > 0.5:
            r = self.get_lpf_fast(0)
            r = max( min( .995, r), 0.005 ) # avoids bad arguments and overflows
            exp_cdf = ( 1. - np.exp(-self.c*r) ) / ( 1. - np.exp(-self.c) )
            error = self.below - self.above - 2.*exp_cdf + 1. 
            self.thresh -= self.tau_thr * error
        else:
            self.thresh += self.tau_fix * (self.thr_fix - self.thresh)
        self.thresh = max(min(self.thresh, 10.), -10.) # clipping large/small values


    def upd_slide_thr_hr(self, time):
        """ Updates the threshold of units with 'rate harmonization'. """
        inputs = self.mp_inputs[self.hr_port]
        mean_inp = np.mean(inputs) 
        r = self.get_lpf_fast(0)
        self.thresh -= self.tau_thr * ( mean_inp - r )
        self.thresh = max(min(self.thresh, 10.), -10.) # clipping large/small values


    def upd_syn_scale_hr(self, time):
        """ Update the synaptic scale for units with 'rate harmonization'. 
            
            The scaling factors in self.scale_facs_hr correspond to the excitatory synapses
            at the port hr_port. 
        """
        # Other than the 'error', it's all as in upd_exp_scale_mp
        inputs = self.mp_inputs[self.hr_port]
        mean_inp = np.mean(inputs) 
        r = self.get_lpf_fast(0)
        error =  mean_inp - r

        hr_inputs = self.mp_inputs[self.hr_port]
        hr_weights = self.get_mp_weights(time)[self.hr_port]
        u = np.dot(hr_inputs[self.exc_idx_hr], hr_weights[self.exc_idx_hr])
        I = u - self.get_mp_input_sum(time) 
        mu_exc = np.maximum( np.sum( hr_inputs[self.exc_idx_hr] ), 0.001 )
        ss_scale = (u - I + self.Kp * error) / mu_exc
        a = ss_scale / np.maximum(hr_weights[self.exc_idx_hr],.001)
        x0 = self.scale_facs_hr[self.exc_idx_hr]
        t = self.net.min_delay
        self.scale_facs_hr[self.exc_idx_hr] = (x0 * a) / ( x0 + (a - x0) * np.exp(-self.tau_scale * a * t) )


    def upd_error(self, time):
        """ update the error used by delta units."""
        # Reliance on mp_inputs would normally be discouraged, since it slows things
        # down, and can be replaced by the commented code below. However, because I also
        # have the inp_l2 requirement in delta units, mp_inputs is available anyway.
        inputs = self.mp_inputs
        weights = self.get_mp_weights(time)
        if self.learning > 0.5:
            port1 = np.dot(inputs[1], weights[1])
            #port1 = sum( [syn.w * act(time - dely) for syn, act, dely in zip(
            #             [self.net.syns[self.ID][i] for i in self.port_idx[1]],
            #             [self.net.act[self.ID][i] for i in self.port_idx[1]],
            #             [self.net.delays[self.ID][i] for i in self.port_idx[1]])] ) 
            self.error = port1 - self.get_lpf_fast(0)
            self.bias += self.bias_lrate * self.error
            # to update 'learning' we can use the known solution to e' = -tau*e, namely
            # e(t) = e(t0) * exp((t-t0)/tau), instead of the forward Euler rule
            self.learning *= np.exp((self.last_time-time)/self.tau_e)
        else: # learning <= 0.5
            self.error = 0.
            # input at port 2
            port2 = np.dot(inputs[2], weights[2])
            #port2 = sum( [syn.w * act(time - dely) for syn, act, dely in zip(
            #             [self.net.syns[self.ID][i] for i in self.port_idx[2]],
            #             [self.net.act[self.ID][i] for i in self.port_idx[2]],
            #             [self.net.delays[self.ID][i] for i in self.port_idx[2]])] )
            if port2 >= 0.5:
                self.learning = 1.
            else: 
                self.learning *= np.exp((self.last_time-time)/self.tau_e)


    def upd_inp_l2(self, time):
        """ Update the L2 norm of the inputs at port 0. """
        self.inp_l2 = np.linalg.norm(self.mp_inputs[0])


class source(unit):
    """ The class of units whose activity comes from some Python function.
    
        source units provide inputs to the network. They can be conceived as units
        whose activity at time 't' comes from a function f(t). This function is passed to
        the constructor as a parameter, or later specified with the set_function method.

        Source units are also useful to track the value of any simulation variable that
        can be retrieved with a function.
    """
    
    def __init__(self, ID, params, network):
        """ The class constructor.


        Args:
            ID, params, network : same as in the parent class (unit).
            In addition, the params dictionary must include:
            REQUIRED PARAMETERS
            'function' : Reference to a Python function that gives the activity of the unit.
                         Oftentimes the function giving the activity of the unit is set after 
                         the constructor has been called, using source.set_function . In this 
                         case it is good practice to set 'function' : lambda x: None

            Notice that 'init_val' is still required because it is used to initialize any
            low-pass filtered values the unit might be keeping.

        Raises:
            AssertionError.
        """
        super(source, self).__init__(ID, params, network)
        self.get_act = params['function'] # the function which returns activation given the time
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
        # Then net.act and syns_act_dels have links to functions other than this get_act.
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


    def upd_lpf_fast(self,time):
        """ Update the lpf_fast variable. 
        
            Same as unit.upd_lpf_fast, except for the line obtaining cur_act .
        """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_fast updated backwards in time']
        cur_act = self.get_act(time) # current activity
        self.lpf_fast = cur_act + ( (self.lpf_fast - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_fast ) )
        # update the buffer
        self.lpf_fast_buff = np.roll(self.lpf_fast_buff, -1)
        self.lpf_fast_buff[-1] = self.lpf_fast


    def upd_lpf_mid(self,time):
        """ Update the lpf_mid variable.

            Same as unit.upd_lpf_mid, except for the line obtaining cur_act .
        """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_mid updated backwards in time']
        cur_act = self.get_act(time) # current activity
        self.lpf_mid = cur_act + ( (self.lpf_mid - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_mid) )
        self.lpf_mid_buff = np.roll(self.lpf_mid_buff, -1)
        self.lpf_mid_buff[-1] = self.lpf_mid


    def upd_lpf_slow(self,time):
        """ Update the lpf_slow variable.

            Same as unit.upd_lpf_slow, except for the line obtaining cur_act .
        """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_slow updated backwards in time']
        cur_act = self.get_act(time) # current activity
        self.lpf_slow = cur_act + ( (self.lpf_slow - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_slow ) )
        self.lpf_slow_buff = np.roll(self.lpf_slow_buff, -1)
        self.lpf_slow_buff[-1] = self.lpf_slow


    def upd_sq_lpf_slow(self,time):
        """ Update the sq_lpf_slow variable. """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' sq_lpf_slow updated backwards in time']
        cur_sq_act = self.get_act(time)**2 # square of current activity
        self.sq_lpf_slow = cur_sq_act + ( (self.sq_lpf_slow - cur_sq_act) * 
                                  np.exp( (self.last_time-time)/self.tau_slow ) )




    
class sigmoidal(unit): 
    """
    An implementation of a typical sigmoidal unit. 
    
    Its output is produced by linearly suming the inputs (times the synaptic weights), 
    and feeding the sum to a sigmoidal function, which constraints the output to values 
    beween zero and one.

    Because this unit operates in real time, it updates its value gradualy, with
    a 'tau' time constant.
    """

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
        """ This is the sigmoidal function. Could roughly think of it as an f-I curve. """
        return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_input_sum(t)) - y[0] ) * self.rtau
    

class linear(unit): 
    """ An implementation of a linear unit.

    The output is the sum of the inputs multiplied by their synaptic weights.
    The output upates with time constant 'tau'.
    """

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
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return( self.get_input_sum(t) - y[0] ) * self.rtau
   

class mp_linear(unit):
    """ Same as the linear unit, but with several input ports; useful for tests.
    
        Current version is only configured to receive inputs from a plant, and to
        treat inputs at all ports equally.
    """

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
        super(mp_linear, self).__init__(ID, params, network)
        self.tau = params['tau']  # the time constant of the dynamics
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        self.multiport = True
        assert self.type is unit_types.mp_linear, ['Unit ' + str(self.ID) + 
                                                            ' instantiated with the wrong type']
        
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return (self.get_plant_mp_input_sum(t) - y[0]) * self.rtau
 

    def get_plant_mp_input_sum(self, time):
        """
        Returns the sum of inputs scaled by their weights, assuming there are multiple input ports.

        The sum accounts for transmission delays. All ports are treated identically. 
        The inputs should come from a plant.
        """
        #return sum([ syn.w * fun(time-dely, syn.plant_out) for syn,fun,dely in zip(self.net.syns[self.ID], 
        #                self.net.act[self.ID], self.net.delays[self.ID]) ])
        return sum([ syn.w * fun(time-dely) for syn,fun,dely in zip(self.net.syns[self.ID], 
                        self.net.act[self.ID], self.net.delays[self.ID]) ])


    def get_sc_input_sum(self, time):
        """ 
        Returns the sum of inputs, each scaled by its synaptic weight and by a gain constant. 
        
        The sum accounts for transmission delays. Input ports are ignored. 

        The extra scaling factor is the 'gain' attribute of the synapses. This is useful when
        different types of inputs have different gains applied to them. You then need a unit
        model that calls get_sc_input_sum instead of get_input_sum. 
        """
        return sum([ syn.gain * syn.w * fun(time-dely) for syn,fun,dely in zip(self.net.syns[self.ID], 
                        self.net.act[self.ID], self.net.delays[self.ID]) ])


    def get_exp_sc_input_sum(self, time):
        """ 
        Returns the sum of inputs, each scaled by its weight and by a scale factor.
        
        The sum accounts for transmission delays. Input ports are ignored. 

        The scale factor is applied only to excitatory synapses, and it is the scale_facs value
        set by the upd_exp_scale function. This is the way that exp_dist_sigmoidal units get
        their total input.
        """
        # This accelerates the simulation, but inp_vector is incorrect (only updates every min_delay).
        #weights = np.array([ syn.w for syn in self.net.syns[self.ID] ])
        #return sum( self.scale_facs * weights * self.inp_vector )
        return sum( [ sc * syn.w * fun(time-dely) for sc,syn,fun,dely in zip(self.scale_facs, 
                      self.net.syns[self.ID], self.net.act[self.ID], self.net.delays[self.ID]) ] )


    def get_weights(self, time):
        """ Returns a list with the weights corresponding to the input list obtained with get_inputs.
        """
        # if you include axo-axonic connections you have to modify the list below
        return [ synapse.get_w(time) for synapse in self.net.syns[self.ID] ]


    def get_mp_weights(self, time):
        """ 
        Returns a list with the weights corresponding to the list obtained with get_mp_inputs.

        This method is for units where multiport = True, and that have a port_idx attribute.

        The i-th element of the returned list is a numpy array with the weights of the
        synapses where port = i.
        """
        return [ np.array([self.net.syns[self.ID][idx].w for idx in idx_list]) for idx_list in self.port_idx ]


    def update(self,time):
        """
        Advance the dynamics from time to time+min_delay.

        This update function will replace the values in the activation buffer 
        corresponding to the latest "min_delay" time units, introducing "min_buff_size" new values.
        In addition, all the synapses of the unit are updated.
        source and kwta units override this with shorter update functions.
        """
        # the 'time' argument is currently only used to ensure the 'times' buffer is in sync
        # Maybe there should be a single 'times' array in the network. This seems more parallelizable, though.
        #assert (self.times[-1]-time) < 2e-6, 'unit' + str(self.ID) + ': update time is desynchronized'

        new_times = self.times[-1] + self.times_grid
        self.times = np.roll(self.times, -self.min_buff_size)
        self.times[self.offset:] = new_times[1:] 
        
        # odeint also returns the initial condition, so to produce min_buff_size new values
        # we need to provide min_buff_size+1 desired times, starting with the one for the initial condition
        new_buff = odeint(self.derivatives, [self.buffer[-1]], new_times, rtol=self.rtol, atol=self.atol)
        self.buffer = np.roll(self.buffer, -self.min_buff_size)
        self.buffer[self.offset:] = new_buff[1:,0] 

        self.pre_syn_update(time) # Update any variables needed for the synapse to update.
                                  # It is important this is done after the buffer has been updated.
        # For each synapse on the unit, update its state
        for pre in self.net.syns[self.ID]:
            pre.update(time)

        self.last_time = time # last_time is used to update some pre_syn_update values


    def get_act(self,time):
        """ Gives you the activity at a previous time 't' (within buffer range).

        This version works for units that store their previous activity values in a buffer.
        Units without buffers (e.g. source units) have their own get_act function.

        This is the most time-consuming method in draculab (thus the various optimizations).
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Below is the more general (but slow) interpolation using interp1d
        # Sometimes the ode solver asks about values slightly out of bounds, so I set this to extrapolate
        """
        return interp1d(self.times, self.buffer, kind='linear', bounds_error=False, copy=False,
                        fill_value="extrapolate", assume_sorted=True)(time)
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Below the code for the second implementation.
        # This linear interpolation takes advantage of the ordered, regularly-spaced buffer.
        # Time values outside the buffer range receive the buffer endpoints.
        # The third implementation is faster, but for small buffer sizes in test2.ipynb this
        # gives more exact results. I can't figure out why.
        """
        time = min( max(time,self.times[0]), self.times[-1] ) # clipping 'time'
        frac = (time-self.times[0])/(self.times[-1]-self.times[0])
        base = int(np.floor(frac*(self.buff_size-1))) # biggest index s.t. times[index] <= time
        frac2 = ( time-self.times[base] ) / ( self.times[min(base+1,self.buff_size-1)] - self.times[base] + 1e-8 )
        return self.buffer[base] + frac2 * ( self.buffer[min(base+1,self.buff_size-1)] - self.buffer[base] )
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This is the third implementation. 
        # Takes advantage of the regularly spaced times using divmod.
        # Values outside the buffer range will fall between buffer[-1] and buffer[-2].
        """
        base, rem = divmod(time-self.times[0], self.time_bit)
        # because time_bit is slightly larger than times[1]-times[0], we can limit
        # base to buff_size-2, even if time = times[-1]
        base =  max( 0, min(int(base), self.buff_size-2) ) 
        frac2 = rem/self.time_bit
        return self.buffer[base] + frac2 * ( self.buffer[base+1] - self.buffer[base] )
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # The fourth implementation uses the same algorithm as the third, with Cython
        return cython_get_act(time, self.times[0], self.time_bit, self.buff_size, self.buffer)

  
    def init_pre_syn_update(self):
        """
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

        In addition, for each one of the unit's synapses, init_pre_syn_update will initialize 
        its delay value.

        An extra task done here is to prepare the 'port_idx' list used by units with multiple 
        input ports.

        init_pre_syn_update is called for a unit everytime network.connect() connects the unit, 
        which may be more than once.

        Raises:
            NameError, NotImplementedError, ValueError.
        """ 
        assert self.net.sim_time == 0, ['Tried to run init_pre_syn_update for unit ' + 
                                         str(self.ID) + ' when simulation time is not zero']

        # Each synapse should know the delay of its connection
        for syn, delay in zip(self.net.syns[self.ID], self.net.delays[self.ID]):
            # The -1 below is because get_lpf_fast etc. return lpf_fast_buff[-1-steps], corresponding to
            # the assumption that buff[-1] is the value zero steps back
            syn.delay_steps = min(self.net.units[syn.preID].steps-1, int(round(delay/self.net.min_delay)))

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

        # If we require support for multiple input ports, create the port_idx list.
        # port_idx is a list whose elements are lists of integers.
        # port_idx[i] contains the indexes in net.syns[self.ID] (or net.delays[self.ID], or net.act[self.ID])
        # of the synapses whose input port is 'i'.
        if self.multiport is True:
            self.port_idx = [ [] for _ in range(self.n_ports) ]
            for idx, syn in enumerate(self.net.syns[self.ID]):
                if syn.port >= self.n_ports:
                    raise ValueError('Found a synapse input port larger than or equal to the number of ports')
                else:
                    self.port_idx[syn.port].append(idx) 

        # Create the pre_syn_update function and the associated variables
        if not hasattr(self, 'functions'): # so we don't erase previous requirements
            self.functions = set() 

        for req in self.syn_needs:
            if req is syn_reqs.lpf_fast:  # <----------------------------------
                if not hasattr(self,'tau_fast'): 
                    raise NameError( 'Synaptic plasticity requires unit parameter tau_fast, not yet set' )
                self.lpf_fast = self.init_val
                self.functions.add(self.upd_lpf_fast)
            elif req is syn_reqs.lpf_mid:  # <----------------------------------
                if not hasattr(self,'tau_mid'): 
                    raise NameError( 'Synaptic plasticity requires unit parameter tau_mid, not yet set' )
                self.lpf_mid = self.init_val
                self.functions.add(self.upd_lpf_mid)
            elif req is syn_reqs.lpf_slow:  # <----------------------------------
                if not hasattr(self,'tau_slow'): 
                    raise NameError( 'Synaptic plasticity requires unit parameter tau_slow, not yet set' )
                self.lpf_slow = self.init_val
                self.functions.add(self.upd_lpf_slow)
            elif req is syn_reqs.sq_lpf_slow:  # <----------------------------------
                if not hasattr(self,'tau_slow'): 
                    raise NameError( 'Synaptic plasticity requires unit parameter tau_slow, not yet set' )
                self.sq_lpf_slow = self.init_val
                self.functions.add(self.upd_sq_lpf_slow)
            elif req is syn_reqs.inp_vector: # <----------------------------------
                self.inp_vector = np.tile(self.init_val, len(self.net.syns[self.ID]))
                self.functions.add(self.upd_inp_vector)
            elif req is syn_reqs.mp_inputs: # <----------------------------------
                if not hasattr(self,'port_idx'): 
                    raise NameError( 'the mp_inputs requirement is for multiport units with a port_idx list' )
                self.mp_inputs = [] 
                for prt_lst in self.port_idx:
                    self.mp_inputs.append(np.array([self.init_val for _ in range(len(prt_lst))]))
                self.functions.add(self.upd_mp_inputs)
            elif req is syn_reqs.inp_avg:  # <----------------------------------
                self.snorm_list = []  # a list with all the presynaptic units
                                      # providing hebbsnorm synapses
                self.snorm_dels = []  # a list with the delay steps for each connection from snorm_list
                for syn in self.net.syns[self.ID]:
                    if syn.type is synapse_types.hebbsnorm:
                        self.snorm_list.append(self.net.units[syn.preID])
                        self.snorm_dels.append(syn.delay_steps)
                self.n_hebbsnorm = len(self.snorm_list) # number of hebbsnorm synapses received
                self.inp_avg = 0.2  # an arbitrary initialization of the average input value
                self.snorm_list_dels = list(zip(self.snorm_list, self.snorm_dels)) # both lists zipped
                self.functions.add(self.upd_inp_avg)
            elif req is syn_reqs.pos_inp_avg:  # <----------------------------------
                self.snorm_units = []  # a list with all the presynaptic units
                                      # providing hebbsnorm synapses
                self.snorm_syns = []  # a list with the synapses for the list above
                self.snorm_delys = []  # a list with the delay steps for these synapses
                for syn in self.net.syns[self.ID]:
                    if syn.type is synapse_types.hebbsnorm:
                        self.snorm_syns.append(syn)
                        self.snorm_units.append(self.net.units[syn.preID])
                        self.snorm_delys.append(syn.delay_steps)
                self.pos_inp_avg = 0.2  # an arbitrary initialization of the average input value
                self.n_vec = np.ones(len(self.snorm_units)) # the 'n' vector from Pg.290 of Dayan&Abbott
                self.functions.add(self.upd_pos_inp_avg)
            elif req is syn_reqs.err_diff:  # <----------------------------------
                self.err_idx = [] # a list with the indexes of the units that provide errors
                self.pred_idx = [] # a list with the indexes of the units that provide predictors 
                self.err_dels = [] # a list with the delays for each unit in err_idx
                self.err_diff = 0. # approximation of error derivative, updated by upd_err_diff
                for syn in self.net.syns[self.ID]:
                    if syn.type is synapse_types.inp_corr:
                        if syn.input_type == 'error':
                            self.err_idx.append(syn.preID)
                            self.err_dels.append(syn.delay_steps)
                        elif syn.input_type == 'pred': 
                            self.pred_idx.append(syn.preID) # not currently using this list
                        else:
                            raise ValueError('Incorrect input_type ' + str(syn.input_type) + ' found in synapse')
                self.err_idx_dels = list(zip(self.err_idx, self.err_dels)) # both lists zipped
                self.functions.add(self.upd_err_diff)
            elif req is syn_reqs.sc_inp_sum: # <----------------------------------
                sq_snorm_units = [] # a list with all the presynaptic neurons providing
                                        # sq_hebbsnorm synapses
                sq_snorm_syns = []  # a list with all the sq_hebbsnorm synapses
                sq_snorm_dels = []  # a list with the delay steps for the sq_hebbsnorm synapses
                for syn in self.net.syns[self.ID]:
                    if syn.type is synapse_types.sq_hebbsnorm:
                        sq_snorm_syns.append(syn)
                        sq_snorm_units.append(self.net.units[syn.preID])
                        sq_snorm_dels.append(syn.delay_steps)
                self.u_d_syn = list(zip(sq_snorm_units, sq_snorm_dels, sq_snorm_syns)) # all lists zipped
                self.sc_inp_sum = 0.2  # an arbitrary initialization of the scaled input value
                self.functions.add(self.upd_sc_inp_sum)
            elif req is syn_reqs.diff_avg: # <----------------------------------
                self.dsnorm_list = [] # list with all presynaptic units providing
                                      # diff_hebb_subsnorm synapses
                self.dsnorm_dels = [] # list with delay steps for each connection in dsnorm_list
                for syn in self.net.syns[self.ID]:
                    if syn.type is synapse_types.diff_hebbsnorm:
                        self.dsnorm_list.append(self.net.units[syn.preID])
                        self.dsnorm_dels.append(syn.delay_steps)
                self.n_dhebbsnorm = len(self.dsnorm_list) # number of diff_hebbsnorm synapses received
                self.diff_avg = 0.2  # an arbitrary initialization of the input derivatives average 
                self.dsnorm_list_dels = list(zip(self.dsnorm_list, self.dsnorm_dels)) # both lists zipped
                self.functions.add(self.upd_diff_avg)
            elif req is syn_reqs.lpf_mid_inp_sum:  # <----------------------------------
                if not hasattr(self,'tau_mid'): 
                    raise NameError( 'Synaptic plasticity requires unit parameter tau_mid, not yet set' )
                if not syn_reqs.inp_vector in self.syn_needs:
                    raise AssertionError('lpf_mid_inp_sum requires the inp_vector requirement to be set')
                self.lpf_mid_inp_sum = self.init_val # this initialization is rather arbitrary
                self.functions.add(self.upd_lpf_mid_inp_sum)
            elif req is syn_reqs.n_erd:  # <---------------------------------- ONLY USED IN LEGACY CODE
                self.n_erd = len([s for s in self.net.syns[self.ID] if s.type is synapse_types.exp_rate_dist])
                # n_erd doesn't need to be updated :)
            elif req is syn_reqs.balance:  # <----------------------------------
                if not syn_reqs.inp_vector in self.syn_needs:
                    raise AssertionError('balance requirement has the inp_vector requirement as a prerequisite')
                self.below = 0.5 # this initialization is rather arbitrary
                self.above = 0.5 
                self.functions.add(self.upd_balance)
            elif req is syn_reqs.balance_mp:  # <----------------------------------
                if not syn_reqs.mp_inputs in self.syn_needs:
                    raise AssertionError('balance_mp requirement has the mp_inputs requirement as a prerequisite')
                self.below = 0.5 # this initialization is rather arbitrary
                self.above = 0.5 
                self.functions.add(self.upd_balance_mp)
            elif req is syn_reqs.exp_scale:  # <----------------------------------
                if not syn_reqs.balance in self.syn_needs:
                    raise AssertionError('exp_scale requires the balance requirement to be set')
                if not syn_reqs.lpf_fast in self.syn_needs:
                    raise AssertionError('exp_scale requires the lpf_fast requirement to be set')
                self.scale_facs= np.tile(1., len(self.net.syns[self.ID])) # array with scale factors
                # exc_idx = numpy array with index of all excitatory units in the input vector 
                self.exc_idx = [ idx for idx,syn in enumerate(self.net.syns[self.ID]) if syn.w >= 0]
                # ensure the integer data type; otherwise you can't index numpy arrays
                self.exc_idx = np.array(self.exc_idx, dtype='uint32')
                # inh_idx = numpy array with index of all inhibitory units 
                self.inh_idx = [idx for idx,syn in enumerate(self.net.syns[self.ID]) if syn.w < 0]
                self.inh_idx = np.array(self.inh_idx, dtype='uint32')
                self.functions.add(self.upd_exp_scale)
            elif req is syn_reqs.exp_scale_mp:  # <----------------------------------
                if not syn_reqs.balance_mp in self.syn_needs:
                    raise AssertionError('exp_scale_mp requires the balance_mp requirement to be set')
                if not syn_reqs.lpf_fast in self.syn_needs:
                    raise AssertionError('exp_scale_mp requires the lpf_fast requirement to be set')
                self.scale_facs_rdc = np.tile(1., len(self.port_idx[self.rdc_port])) # array with scale factors
                # exc_idx_rdc = numpy array with index of all excitatory units at rdc port
                self.exc_idx_rdc = [ idx for idx,syn in enumerate([self.net.syns[self.ID][i] 
                                     for i in self.port_idx[self.rdc_port]]) if syn.w >= 0 ]
                # ensure the integer data type; otherwise you can't index numpy arrays
                self.exc_idx_rdc = np.array(self.exc_idx_rdc, dtype='uint32')
                self.functions.add(self.upd_exp_scale_mp)
            elif req is syn_reqs.slide_thresh:  # <----------------------------------
                if (not syn_reqs.balance in self.syn_needs) and (not syn_reqs.balance_mp in self.syn_needs):
                    raise AssertionError('slide_thresh requires the balance(_mp) requirement to be set')
                self.functions.add(self.upd_thresh)
            elif req is syn_reqs.slide_thr_shrp:  # <----------------------------------
                if not syn_reqs.balance_mp in self.syn_needs:
                    raise AssertionError('slide_thr_shrp requires the balance_mp requirement to be set')
                if not self.multiport:
                    raise AssertionError('The slide_thr_shrp is for multiport units only')
                self.functions.add(self.upd_thr_shrp)
            elif req is syn_reqs.lpf_slow_mp_inp_sum:  # <----------------------------------
                if not hasattr(self,'tau_slow'): 
                    raise NameError( 'Requirement lpf_slow_mp_inp_sum requires parameter tau_slow, not yet set' )
                if not syn_reqs.mp_inputs in self.syn_needs:
                    raise AssertionError('lpf_slow_mp_inp_sum requires the mp_inputs requirement')
                self.lpf_slow_mp_inp_sum = [ 0.2 for _ in range(self.n_ports) ]
                self.functions.add(self.upd_lpf_slow_mp_inp_sum)
            elif req is syn_reqs.slide_thr_hr:  # <----------------------------------
                if not syn_reqs.mp_inputs in self.syn_needs:
                    raise AssertionError('slide_thr_hr requires the mp_inputs requirement to be set')
                if not self.multiport:
                    raise AssertionError('The slide_thr_hr requirment is for multiport units only')
                self.functions.add(self.upd_slide_thr_hr)
            elif req is syn_reqs.syn_scale_hr:  # <----------------------------------
                if not syn_reqs.mp_inputs in self.syn_needs:
                    raise AssertionError('syn_scale_hr requires the mp_inputs requirement to be set')
                if not self.multiport:
                    raise AssertionError('The syn_scale_hr requirment is for multiport units only')
                if not syn_reqs.lpf_fast in self.syn_needs:
                    raise AssertionError('syn_scale_hr requires the lpf_fast requirement to be set')
                # exc_idx_hr = numpy array with index of all excitatory units at hr port
                self.exc_idx_hr = [ idx for idx,syn in enumerate([self.net.syns[self.ID][i] 
                                     for i in self.port_idx[self.hr_port]]) if syn.w >= 0 ]
                # ensure the integer data type; otherwise you can't index numpy arrays
                self.exc_idx_hr = np.array(self.exc_idx_hr, dtype='uint32')
                #self.scale_facs = np.tile(1., len(self.net.syns[self.ID])) # array with scale factors
                self.scale_facs_hr = np.tile(1., len(self.port_idx[self.hr_port])) # array with scale factors
                self.functions.add(self.upd_syn_scale_hr)
            elif req is syn_reqs.exp_scale_shrp:  # <----------------------------------
                if not syn_reqs.balance_mp in self.syn_needs:
                    raise AssertionError('exp_scale_shrp requires the balance_mp requirement to be set')
                if not self.multiport:
                    raise AssertionError('The exp_scale_shrp requirement is for multiport units only')
                if not syn_reqs.lpf_fast in self.syn_needs:
                    raise AssertionError('exp_scale_shrp requires the lpf_fast requirement to be set')
                self.scale_facs_rdc = np.tile(1., len(self.port_idx[self.rdc_port])) # array with scale factors
                # exc_idx_rdc = numpy array with index of all excitatory units at rdc port
                self.exc_idx_rdc = [ idx for idx,syn in enumerate([self.net.syns[self.ID][i] 
                                     for i in self.port_idx[self.rdc_port]]) if syn.w >= 0 ]
                # ensure the integer data type; otherwise you can't index numpy arrays
                self.exc_idx_rdc = np.array(self.exc_idx_rdc, dtype='uint32')
                self.rdc_exc_ones = np.tile(1., len(self.exc_idx_rdc))
                self.functions.add(self.upd_exp_scale_shrp)

            else:  # <----------------------------------------------------------------------
                raise NotImplementedError('Asking for a requirement that is not implemented')

        self.pre_syn_update = lambda time: [f(time) for f in self.functions]

        

    def upd_lpf_fast(self,time):
        """ Update the lpf_fast variable. """
        # Source units have their own implementation
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_fast updated backwards in time']
        cur_act = self.buffer[-1] # This doesn't work for source units
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.lpf_fast = cur_act + ( (self.lpf_fast - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_fast ) )
        # update the buffer
        self.lpf_fast_buff = np.roll(self.lpf_fast_buff, -1)
        self.lpf_fast_buff[-1] = self.lpf_fast


    def get_lpf_fast(self, steps):
        """ Get the fast low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_fast_buff[-1-steps]


    def upd_lpf_mid(self,time):
        """ Update the lpf_mid variable. """
        # Source units have their own implementation
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_mid updated backwards in time']
        cur_act = self.buffer[-1] # This doesn't work for source units
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.lpf_mid = cur_act + ( (self.lpf_mid - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_mid) )
        # update the buffer
        self.lpf_mid_buff = np.roll(self.lpf_mid_buff, -1)
        self.lpf_mid_buff[-1] = self.lpf_mid


    def get_lpf_mid(self, steps):
        """ Get the mid-speed low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_mid_buff[-1-steps]


    def upd_lpf_slow(self,time):
        """ Update the lpf_slow variable. """
        # Source units have their own implementation
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_slow updated backwards in time']
        cur_act = self.buffer[-1] # This doesn't work for source units
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.lpf_slow = cur_act + ( (self.lpf_slow - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_slow ) )
        # update the buffer
        self.lpf_slow_buff = np.roll(self.lpf_slow_buff, -1)
        self.lpf_slow_buff[-1] = self.lpf_slow


    def get_lpf_slow(self, steps):
        """ Get the slow low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_slow_buff[-1-steps]


    def upd_sq_lpf_slow(self,time):
        """ Update the sq_lpf_slow variable. """
        # Source units have their own implementation
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' sq_lpf_slow updated backwards in time']
        cur_sq_act = self.buffer[-1]**2.  # This doesn't work for source units.
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.sq_lpf_slow = cur_sq_act + ( (self.sq_lpf_slow - cur_sq_act) * 
                                  np.exp( (self.last_time-time)/self.tau_slow ) )


    def upd_inp_avg(self, time):
        """ Update the average of the inputs with hebbsnorm synapses. 
        
            The actual value being averaged is lpf_fast of the presynaptic units.
        """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' inp_avg updated backwards in time']
        self.inp_avg = sum([u.get_lpf_fast(s) for u,s in self.snorm_list_dels]) / self.n_hebbsnorm
        

    def upd_pos_inp_avg(self, time):
        """ Update the average of the inputs with hebbsnorm synapses. 
        
            The actual value being averaged is lpf_fast of the presynaptic units.
            Inputs whose synaptic weight is zero or negative  are saturated to zero and
            excluded from the computation.
        """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' pos_inp_avg updated backwards in time']
        # first, update the n vector from Eq. 8.14, pg. 290 in Dayan & Abbott
        self.n_vec = [ 1. if syn.w>0. else 0. for syn in self.snorm_syns ]
        
        self.pos_inp_avg = sum([n*(u.get_lpf_fast(s)) for n,u,s in 
                                zip(self.n_vec, self.snorm_units, self.snorm_delys)]) / sum(self.n_vec)


    def upd_err_diff(self, time):
        """ Update an approximate derivative of the error inputs used for input correlation learning. 

            A very simple approach is taken, where the derivative is approximated as the difference
            between the fast and medium low-pass filtered inputs. Each input arrives with its
            corresponding transmission delay.
        """
        self.err_diff = ( sum([ self.net.units[i].get_lpf_fast(s) for i,s in self.err_idx_dels ]) -
                          sum([ self.net.units[i].get_lpf_mid(s) for i,s in self.err_idx_dels ]) )
       

    def upd_sc_inp_sum(self, time):
        """ Update the sum of the inputs multiplied by their synaptic weights.
        
            The actual value being summed is lpf_fast of the presynaptic units.
        """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' sc_inp_sum updated backwards in time']
        self.sc_inp_sum = sum([u.get_lpf_fast(d) * syn.w for u,d,syn in self.u_d_syn])
        
        
    def upd_diff_avg(self, time):
        """ Update the average of derivatives from inputs with diff_hebbsnorm synapses.

            The values being averaged are not the actual derivatives, but approximations
            which are roughly proportional to them, coming from the difference 
            lpf_fast - lpf_mid . 
        """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' diff_avg updated backwards in time']
        self.diff_avg = ( sum([u.get_lpf_fast(s) - u.get_lpf_mid(s) for u,s in self.dsnorm_list_dels]) 
                          / self.n_dhebbsnorm )

    def upd_lpf_mid_inp_sum(self,time):
        """ Update the lpf_mid_inp_sum variable. """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_mid_inp_sum updated backwards in time']
        cur_inp_sum = (self.inp_vector).sum()
        #cur_inp_sum = sum(self.get_inputs(time))
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.lpf_mid_inp_sum = cur_inp_sum + ( (self.lpf_mid_inp_sum - cur_inp_sum) * 
                                   np.exp( (self.last_time-time)/self.tau_mid) )
        # update the buffer
        self.lpf_mid_inp_sum_buff = np.roll(self.lpf_mid_inp_sum_buff, -1)
        self.lpf_mid_inp_sum_buff[-1] = self.lpf_mid_inp_sum


    def get_lpf_mid_inp_sum(self):
        """ Get the latest value of the mid-speed low-pass filtered sum of inputs. """
        return self.lpf_mid_inp_sum_buff[-1]


    def upd_lpf_slow_mp_inp_sum(self, time):
        """ Update the slow LPF'd scaled sum of inputs at individual ports, returning them in a list. """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_slow_mp_inp_sum updated backwards in time']
        inputs = self.mp_inputs  # updated due to the mp_inputs synaptic requirement
        weights = self.get_mp_weights(time)
        dots = [ np.dot(i, w) for i, w in zip(inputs, weights) ]
        # same update rule from other upd_lpf_X methods above, put in a list comprehension
        self.lpf_slow_mp_inp_sum = [dots[i] + (self.lpf_slow_mp_inp_sum[i] - dots[i]) * 
                                   np.exp( (self.last_time-time)/self.tau_slow ) for i in range(self.n_ports)]


    def upd_balance(self, time):
        """ Updates two numbers called  below, and above.

            below = fraction of inputs with rate lower than this unit.
            above = fraction of inputs with rate higher than this unit.

            Those numbers are useful to produce a given firing rate distribtuion.

            NOTICE: this version does not restrict inputs to exp_rate_dist synapses.
        """
        #inputs = np.array(self.get_inputs(time)) # current inputs
        inputs = self.inp_vector
        N = len(inputs)
        r = self.buffer[-1] # current rate
        self.above = ( 0.5 * (np.sign(inputs - r) + 1.).sum() ) / N
        self.below = ( 0.5 * (np.sign(r - inputs) + 1.).sum() ) / N
        #assert abs(self.above+self.below - 1.) < 1e-5, ['sum was not 1: ' + 
        #                            str(self.above + self.below)]


    def upd_balance_mp(self, time):
        """ Updates two numbers called  below, and above. Used in units with multiple input ports.

            below = fraction of inputs with rate lower than this unit.
            above = fraction of inputs with rate higher than this unit.

            Those numbers are useful to produce a given firing rate distribtuion among
            the population of units that connect to port 0.

            This is the same as upd_balance, but ports other than the rdc_port are ignored.
        """
        inputs = self.mp_inputs[self.rdc_port]
        N = len(inputs)
        r = self.buffer[-1] # current rate
        self.above = ( 0.5 * (np.sign(inputs - r) + 1.).sum() ) / N
        self.below = ( 0.5 * (np.sign(r - inputs) + 1.).sum() ) / N

    
    def upd_exp_scale(self, time):
        """ Updates the synaptic scaling factor used in exp_dist_sigmoidal units.

            The algorithm is a multiplicative version of the  one used in exp_rate_dist synapses.
            It scales all excitatory inputs.
        """
        #r = self.get_lpf_fast(0)
        r = self.buffer[-1] # current rate
        r = max( min( .995, r), 0.005 ) # avoids bad arguments and overflows
        exp_cdf = ( 1. - np.exp(-self.c*r) ) / ( 1. - np.exp(-self.c) )
        error = self.below - self.above - 2.*exp_cdf + 1. 

        # First APCTP version (12/13/17)
        ######################################################################
        #u = (np.log(r/(1.-r))/self.slope) + self.thresh  # this u lags, so I changed it
        weights = np.array([syn.w for syn in self.net.syns[self.ID]])
        #u = np.dot(self.inp_vector[self.exc_idx], weights[self.exc_idx])
        #u = self.get_input_sum(time)
        u = self.get_exp_sc_input_sum(time)
        I = np.sum( self.inp_vector[self.inh_idx] * weights[self.inh_idx] ) 
        #I = u - np.sum( self.inp_vector[self.exc_idx] * weights[self.exc_idx] ) 
        mu_exc = np.maximum( np.sum( self.inp_vector[self.exc_idx] ), 0.001 )
        #fpr = 1. / (self.c * r * (1. - r)) # reciprocal of the sigmoidal's derivative
        #ss_scale = (u - I + self.Kp * fpr * error) / mu_exc
        ss_scale = (u - I + self.Kp * error) / mu_exc
        #self.scale_facs[self.exc_idx] += self.tau_scale * (ss_scale/np.maximum(weights[self.exc_idx],.05) - 
        #                                                    self.scale_facs[self.exc_idx])
        # ------------ soft weight-bounded version ------------
        # Soft weight bounding implies that the scale factors follow the logistic differential 
        # equation. This equation has the form x' = r (a - x) x, and has a solution
        # x(t) = a x(0) / [ x(0) + (a - x(0))*exp(-a r t) ] .
        # In our case, r = tau_scale, and a = ss_scale / weights[self.ID] .
        # We can use this analytical solution to update the scale factors on each update.
        a = ss_scale / np.maximum(weights[self.exc_idx],.001)
        a = np.minimum( a, 10.) # hard bound above
        x0 = self.scale_facs[self.exc_idx]
        t = self.net.min_delay
        self.scale_facs[self.exc_idx] = (x0 * a) / ( x0 + (a - x0) * np.exp(-self.tau_scale * a * t) )
        #self.scale_facs[self.exc_idx] += self.tau_scale * self.scale_facs[self.exc_idx] * (
        #                                 ss_scale/np.maximum(weights[self.exc_idx],.05) - 
        #                                 self.scale_facs[self.exc_idx] )


    def upd_exp_scale_mp(self, time):
        """ Updates the synaptic scaling factor used in multiport ssrdc units.

            The algorithm is the same as upd_exp_scale, but only the inputs at the rdc_port
            are considered.
        """
        #r = self.get_lpf_fast(0)  # lpf'd rate
        r = self.buffer[-1] # current rate
        r = max( min( .995, r), 0.005 ) # avoids bad arguments and overflows
        exp_cdf = ( 1. - np.exp(-self.c*r) ) / ( 1. - np.exp(-self.c) )
        error = self.below - self.above - 2.*exp_cdf + 1. 

        # First APCTP version (12/13/17)
        ######################################################################
        #u = (np.log(r/(1.-r))/self.slope) + self.thresh
        rdc_inp = self.mp_inputs[self.rdc_port]
        rdc_w = self.get_mp_weights(time)[self.rdc_port]
        u = np.sum(self.scale_facs_rdc[self.exc_idx_rdc]*rdc_inp[self.exc_idx_rdc]*rdc_w[self.exc_idx_rdc])
        I = u - self.get_mp_input_sum(time)     
        mu_exc = np.maximum( np.sum( rdc_inp[self.exc_idx_rdc] ), 0.001 )
        #fpr = 1. / (self.c * r * (1. - r)) # reciprocal of the sigmoidal's derivative
        #ss_scale = (u - I + self.Kp * fpr * error) / mu_exc # adjusting for sigmoidal's slope
        ss_scale = (u - I + self.Kp * error) / mu_exc
        #self.scale_facs[self.exc_idxrdc] += self.tau_scale * (ss_scale/np.maximum(weights[self.exc_idx_rdc],.05) - 
        #                                                    self.scale_facs[self.exc_idx_rdc])
        # ------------ weight-bounded version ------------
        # Soft weight bounding implies that the scale factors follow the logistic differential 
        # equation. This equation has the form x' = r (a - x) x, and has a solution
        # x(t) = a x(0) / [ x(0) + (a - x(0))*exp(-a r t) ] .
        # In our case, r = tau_scale, and a = ss_scale / weights[self.ID] .
        # We can use this analytical solution to update the scale factors on each update.
        a = ss_scale / np.maximum(rdc_w[self.exc_idx_rdc],.001)
        a = np.minimum( a, 10.) # hard bound above
        x0 = self.scale_facs_rdc[self.exc_idx_rdc]
        t = self.net.min_delay
        self.scale_facs_rdc[self.exc_idx_rdc] = (x0 * a) / ( x0 + (a - x0) * np.exp(-self.tau_scale * a * t) )


    def upd_exp_scale_shrp(self, time):
        """ Updates the synaptic scaling factor used in ssrdc_sharp units.

            The algorithm is the same as upd_exp_scale_mp, but controlled by the inputs at the
            sharpening port. When the sum of inputs at this port is smaller than 0.5 the scale factors
            return to 1 with a time constant of tau_relax.
        """
        if np.dot(self.mp_inputs[self.sharpen_port], self.get_mp_weights(time)[self.sharpen_port]) < 0.5:
            a = self.rdc_exc_ones   
            exp_tau = self.tau_relax
        else: 
            #r = self.get_lpf_fast(0)  # lpf'd rate
            r = self.buffer[-1] # current rate
            r = max( min( .995, r), 0.005 ) # avoids bad arguments and overflows
            exp_cdf = ( 1. - np.exp(-self.c*r) ) / ( 1. - np.exp(-self.c) )
            error = self.below - self.above - 2.*exp_cdf + 1. 

            #u = (np.log(r/(1.-r))/self.slope) + self.thresh
            rdc_inp = self.mp_inputs[self.rdc_port]
            rdc_w = self.get_mp_weights(time)[self.rdc_port]
            u = np.sum(self.scale_facs_rdc[self.exc_idx_rdc]*rdc_inp[self.exc_idx_rdc]*rdc_w[self.exc_idx_rdc])
            I = u - self.get_mp_input_sum(time)     
            mu_exc = np.maximum( np.sum( rdc_inp[self.exc_idx_rdc] ), 0.001 )
            #fpr = 1. / (self.c * r * (1. - r)) # reciprocal of the sigmoidal's derivative
            #ss_scale = (u - I + self.Kp * fpr * error) / mu_exc # adjusting for sigmoidal's slope
            ss_scale = (u - I + self.Kp * error) / mu_exc
            a = ss_scale / np.maximum(rdc_w[self.exc_idx_rdc],.001)
            a = np.minimum( a, 10.) # hard bound above
            exp_tau = self.tau_scale
        
        x0 = self.scale_facs_rdc[self.exc_idx_rdc]
        self.scale_facs_rdc[self.exc_idx_rdc] = (x0*a) / (x0+(a-x0)*np.exp(-exp_tau*a*self.net.min_delay))

        
    def upd_inp_vector(self, time):
        """ Update a numpy array containing all the current synaptic inputs """
        self.inp_vector = np.array([ fun(time - dely) for dely,fun in 
                                     zip(self.net.delays[self.ID], self.net.act[self.ID]) ])


    def upd_mp_inputs(self, time):
        """ Update a copy of the results of get_mp_inputs. """
        self.mp_inputs = self.get_mp_inputs(time)


    def upd_thresh(self, time):
        """ Updates the threshold of exp_dist_sig_thr units and some other 'trdc' units.

            The algorithm is an adapted version of the  one used in exp_rate_dist synapses.
        """
        r = self.get_lpf_fast(0)
        r = max( min( .995, r), 0.005 ) # avoids bad arguments and overflows
        exp_cdf = ( 1. - np.exp(-self.c*r) ) / ( 1. - np.exp(-self.c) )
        error = self.below - self.above - 2.*exp_cdf + 1. 
        self.thresh -= self.tau_thr * error
        self.thresh = max(min(self.thresh, 10.), -10.) # clipping large/small values
        

    def upd_thr_shrp(self, time):
        """ Updates the threshold of trdc units when input at 'sharpen' port is larger than 0.5 .

            The algorithm is based on upd_thresh.
        """
        idxs = self.port_idx[self.sharpen_port]
        inps = self.mp_inputs[self.sharpen_port]
        ws = [self.net.syns[self.ID][idx].w for idx in idxs]
        sharpen = sum( [w*i for w,i in zip(ws, inps)] )
            
        if sharpen > 0.5:
            r = self.get_lpf_fast(0)
            r = max( min( .995, r), 0.005 ) # avoids bad arguments and overflows
            exp_cdf = ( 1. - np.exp(-self.c*r) ) / ( 1. - np.exp(-self.c) )
            error = self.below - self.above - 2.*exp_cdf + 1. 
            self.thresh -= self.tau_thr * error
        else:
            self.thresh += self.tau_fix * (self.thr_fix - self.thresh)
        self.thresh = max(min(self.thresh, 10.), -10.) # clipping large/small values


    def upd_slide_thr_hr(self, time):
        """ Updates the threshold of units with 'rate harmonization'. """
        inputs = self.mp_inputs[self.hr_port]
        mean_inp = np.mean(inputs) 
        r = self.get_lpf_fast(0)
        self.thresh -= self.tau_thr * ( mean_inp - r )
        self.thresh = max(min(self.thresh, 10.), -10.) # clipping large/small values


    def upd_syn_scale_hr(self, time):
        """ Update the synaptic scale for units with 'rate harmonization'. 
            
            The scaling factors in self.scale_facs_hr correspond to the excitatory synapses
            at the port hr_port. 
        """
        # Other than the 'error', it's all as in upd_exp_scale_mp
        inputs = self.mp_inputs[self.hr_port]
        mean_inp = np.mean(inputs) 
        r = self.get_lpf_fast(0)
        error =  mean_inp - r

        hr_inputs = self.mp_inputs[self.hr_port]
        hr_weights = self.get_mp_weights(time)[self.hr_port]
        u = np.dot(hr_inputs[self.exc_idx_hr], hr_weights[self.exc_idx_hr])
        I = u - self.get_mp_input_sum(time) 
        mu_exc = np.maximum( np.sum( hr_inputs[self.exc_idx_hr] ), 0.001 )
        ss_scale = (u - I + self.Kp * error) / mu_exc
        a = ss_scale / np.maximum(hr_weights[self.exc_idx_hr],.001)
        x0 = self.scale_facs_hr[self.exc_idx_hr]
        t = self.net.min_delay
        self.scale_facs_hr[self.exc_idx_hr] = (x0 * a) / ( x0 + (a - x0) * np.exp(-self.tau_scale * a * t) )



class source(unit):
    """ The class of units whose activity comes from some Python function.
    
        source units provide inputs to the network. They can be conceived as units
        whose activity at time 't' comes from a function f(t). This function is passed to
        the constructor as a parameter, or later specified with the set_function method.

        Source units are also useful to track the value of any simulation variable that
        can be retrieved with a function.
    """
    
    def __init__(self, ID, params, network):
        """ The class constructor.


        Args:
            ID, params, network : same as in the parent class (unit).
            In addition, the params dictionary must include:
            REQUIRED PARAMETERS
            'function' : Reference to a Python function that gives the activity of the unit.
                         Oftentimes the function giving the activity of the unit is set after 
                         the constructor has been called, using source.set_function . In this 
                         case it is good practice to set 'function' : lambda x: None

            Notice that 'init_val' is still required because it is used to initialize any
            low-pass filtered values the unit might be keeping.

        Raises:
            AssertionError.
        """
        super(source, self).__init__(ID, params, network)
        self.get_act = params['function'] # the function which returns activation given the time
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
        # Then net.act and syns_act_dels have links to functions other than this get_act.
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


    def upd_lpf_fast(self,time):
        """ Update the lpf_fast variable. 
        
            Same as unit.upd_lpf_fast, except for the line obtaining cur_act .
        """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_fast updated backwards in time']
        cur_act = self.get_act(time) # current activity
        self.lpf_fast = cur_act + ( (self.lpf_fast - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_fast ) )
        # update the buffer
        self.lpf_fast_buff = np.roll(self.lpf_fast_buff, -1)
        self.lpf_fast_buff[-1] = self.lpf_fast


    def upd_lpf_mid(self,time):
        """ Update the lpf_mid variable.

            Same as unit.upd_lpf_mid, except for the line obtaining cur_act .
        """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_mid updated backwards in time']
        cur_act = self.get_act(time) # current activity
        self.lpf_mid = cur_act + ( (self.lpf_mid - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_mid) )
        self.lpf_mid_buff = np.roll(self.lpf_mid_buff, -1)
        self.lpf_mid_buff[-1] = self.lpf_mid


    def upd_lpf_slow(self,time):
        """ Update the lpf_slow variable.

            Same as unit.upd_lpf_slow, except for the line obtaining cur_act .
        """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_slow updated backwards in time']
        cur_act = self.get_act(time) # current activity
        self.lpf_slow = cur_act + ( (self.lpf_slow - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_slow ) )
        self.lpf_slow_buff = np.roll(self.lpf_slow_buff, -1)
        self.lpf_slow_buff[-1] = self.lpf_slow


    def upd_sq_lpf_slow(self,time):
        """ Update the sq_lpf_slow variable. """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' sq_lpf_slow updated backwards in time']
        cur_sq_act = self.get_act(time)**2 # square of current activity
        self.sq_lpf_slow = cur_sq_act + ( (self.sq_lpf_slow - cur_sq_act) * 
                                  np.exp( (self.last_time-time)/self.tau_slow ) )




    
class sigmoidal(unit): 
    """
    An implementation of a typical sigmoidal unit. 
    
    Its output is produced by linearly suming the inputs (times the synaptic weights), 
    and feeding the sum to a sigmoidal function, which constraints the output to values 
    beween zero and one.

    Because this unit operates in real time, it updates its value gradualy, with
    a 'tau' time constant.
    """

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
        """ This is the sigmoidal function. Could roughly think of it as an f-I curve. """
        return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_input_sum(t)) - y[0] ) * self.rtau
    

class linear(unit): 
    """ An implementation of a linear unit.

    The output is the sum of the inputs multiplied by their synaptic weights.
    The output upates with time constant 'tau'.
    """

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
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return( self.get_input_sum(t) - y[0] ) * self.rtau
   

class custom_fi(unit): 
    """
    A unit where the f-I curve is provided to the constructor. 
    
    The output of this unit is f( inp ), where f is the custom gain curve given to 
    the constructor (or set with the set_fi method), and inp is the sum of inputs 
    times their weights. 

    Because this unit operates in real time, it updates its value gradualy, with
    a 'tau' time constant.
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS
                'tau' : Time constant of the update dynamics.
                'function' : Reference to a Python function providing the f-I curve

        Raises:
            AssertionError.

        """
        super(custom_fi, self).__init__(ID, params, network)
        self.tau = params['tau']  # the time constant of the dynamics
        self.f = params['function'] # the f-I curve
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        assert self.type is unit_types.custom_fi, ['Unit ' + str(self.ID) + 
                                                            ' instantiated with the wrong type']
        
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_input_sum(t)) - y[0] ) * self.rtau
 

    def set_fi(self, fun):
        """ Set the f-I curve with the given function. """
        self.f = fun


class custom_scaled_fi(unit): 
    """
    A unit where the f-I curve is provided to the constructor, and each synapse has an extra gain.
    
    The output of this unit is f( inp ), where f is the custom f-I curve given to 
    the constructor (or set with the set_fi method), and inp is the sum of each input 
    times its weight, times the synapse's gain factor. The gain factor must be initialized for
    all the synapses in the unit.

    Because this unit operates in real time, it updates its value gradualy, with
    a 'tau' time constant.
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS
                'tau' : Time constant of the update dynamics.
                'function' : Reference to a Python function providing the f-I curve

        Raises:
            AssertionError.

        """
        super(custom_scaled_fi, self).__init__(ID, params, network)
        self.tau = params['tau']  # the time constant of the dynamics
        self.f = params['function'] # the f-I curve
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        assert self.type is unit_types.custom_sc_fi, ['Unit ' + str(self.ID) + 
                                                   ' instantiated with the wrong type']
        
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_sc_input_sum(t)) - y[0] ) * self.rtau
 

    def set_fi(self, fun):
        """ Set the f-I curve with the given function. """
        self.f = fun


class kWTA(unit):
    """
        This is a special type of unit, used to produce a k-winners-take-all layer.

        The kWTA unit implements something similar to the feedforward feedback inhibition
        used in Leabra, meaning that it inhibits all units by the same amount, which is
        just enough to have k units active. It is assumed that all units in the layer
        are sigmoidal, with the same slope and threshold.

        The way to use a kWTA unit is to set reciprocal connections between it and
        all the units it will inhibit, using static synapses with weight 1 for incoming 
        connections and -1 for outgoing connections. On each call to update(), the kWTA
        unit will sort the activity of all its input units, calculate the inhibition to
        have k active units, and set its own activation to that level (e.g. put that
        value in its buffer).

        If the units in the layer need positive stimulation rather than inhibition in
        order to have k 'winners', the default behavior of the kWTA unit is to not activate.
        The behavior can be changed by setting the flag 'neg_act' equal to 'True'.
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

            All the attributes that require knowledge about the units in the layer
            are initialized in init_buffers.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS
                'k' : number of units that should be active.
                OPTIONAL PARAMETERS
                'neg_act': Whether the unit can have negative activity. Default is 'False'.
                'des_act': Desired activity of the k-th unit. Default is 0.9 .
                
        Raises:
            AssertionError.
        """
        super(kWTA, self).__init__(ID, params, network)
        self.k = params['k']
        if 'neg_act' in params:
            self.neg_act = params['neg_act']
        else:
            self.neg_act = False
        if 'des_act' in params:
            self.des_act = params['des_act']
            assert (self.des_act < 1) and (self.des_act > 0), 'Invalid desired activity in kWTA' 
        else:
            self.des_act = 0.9
        assert self.type is unit_types.kwta, ['Unit ' + str(self.ID) + 
                                              ' instantiated with the wrong type']


    def update(self,time):
        """
        Set updated values in the buffer.

        This update function will replace the values in the activation buffer 
        corresponding to the latest "min_delay" time units, introducing "min_buff_size" new values.
        Unlike unit.update(), this function does not use dynamics. The activation
        produced is the one needed to set k active units in the layer. To avoid discontinuity
        in the activation, the values put in the buffer come from a linear interpolation between
        the value from the previous update, and the current value. Synapses are not updated.
        """

        # the 'time' argument is currently only used to ensure the 'times' buffer is in sync
        assert (self.times[-1]-time) < 2e-6, 'unit' + str(self.ID) + ': update time is desynchronized'

        # Calculate the required activation value
        ## First, get the activation for all the units
        for idx, act in enumerate(self.net.act[self.ID]):
            self.activs[idx] = act(time - self.net.delays[self.ID][idx])
        ## We sort the activity and find the unit with the k-th largest activity
        # TODO: find k largest instead of sorting. Needs a single pass.
        sort_ids = self.activs.argsort()
        kunit = sort_ids[-self.k]
        self.winner = self.inpIDs[sort_ids[-1]] # the unit with the largest activation
        ## We get the input required to set the k-th unit to des_act
        kact = self.activs[kunit]
        new_act = -(1./self.slope)*np.log((1./kact) - 1.) + self.thresh - self.des_inp
                  # Notice how this assumes a synaptic weight of -1
        if not self.neg_act: # if negative activation is not allowed
            new_act = max(new_act, 0.)

        # update the buffers with the new activity value
        new_times = self.times[-1] + self.times_grid
        self.times = np.roll(self.times, -self.min_buff_size)
        self.times[self.offset:] = new_times[1:] 
        new_buff = np.linspace(self.buffer[-1], new_act, self.min_buff_size)
        self.buffer = np.roll(self.buffer, -self.min_buff_size)
        self.buffer[self.offset:] = new_buff

        self.pre_syn_update(time) # update any variables needed for the synapse to update
        self.last_time = time # last_time is used to update some pre_syn_update values


    def init_buffers(self):
        """
        In addition to initializing buffers, this initializes variables needed for update.

        Raises:
            TypeError, ValueError.
        """
        super(kWTA, self).init_buffers() # the init_buffers of the unit parent class

        # If this is the first time init_buffers is called, end here
        try:
            self.net.units[self.ID]
        except IndexError: # when the unit is being created, it is not in net.units
            return

        # Get a list with the IDs of the units sending inputs
        inps = []
        for syn in self.net.syns[self.ID]:
            inps.append(syn.preID)    

        # If we have inputs, initialize the variables used by update()
        if len(inps) > 0:
            self.inpIDs = np.array(inps)
            self.activs = np.zeros(len(inps))
            self.slope = self.net.units[inps[0]].slope
            self.thresh = self.net.units[inps[0]].thresh
            self.des_inp = self.thresh - (1./self.slope)*np.log((1./self.des_act)-1.)
        else:
            return

        # Running some tests...
        ## Make sure all slopes and thresholds are the same
        for unit in [self.net.units[idx] for idx in inps]:
            if unit.type != unit_types.sigmoidal:
                raise TypeError('kWTA unit connected to non-sigmoidal units')
            if self.thresh != unit.thresh:
                raise ValueError('Not all threshold values are equal in kWTA layer')
            if self.slope != unit.slope:
                raise ValueError('Not all slope values are equal in kWTA layer')
        ## Make sure all incoming connections are static
        for syn in self.net.syns[self.ID]:
            if syn.type != synapse_types.static:
                raise TypeError('Non-static connection to a kWTA unit')
        ## Make sure all outgoing connections are static with weight -1
        for syn_list in self.net.syns:
            for syn in syn_list:
                if syn.preID == self.ID:
                    if syn.type != synapse_types.static:
                        raise TypeError('kWTA unit sends a non-static connection')
                    if syn.w != -1.:
                        raise ValueError('kWTA sends connection with invalid weight value')


class exp_dist_sigmoidal(unit): 
    """
    A unit where the synaptic weights are scaled to produce an exponential distribution.
    
    This unit has the same activation function as the sigmoidal unit, but the excitatory
    synaptic weights are scaled to produce an exponential distribution of the firing rates in
    the network, using the same approach as the exp_rate_dist_synapse.

    Whether an input is excitatory or inhibitory is decided by the sign of its initial value.
    Synaptic weights initialized to zero will be considered excitatory.
    """
    # The visible difference with sigmoidal units.derivatives is that this unit type
    # calls get_exp_sc_input_sum() instead of get_input_sum(), and this causes the 
    # excitatory inputs to be scaled using an 'exp_scale' factor. The exp_scale
    # factor is calculated by the upd_exp_scale function, which is called every update
    # thanks to the exp_scale synaptic requirement.

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS (use of parameters is in flux. Definitions may be incorrect)
                'slope' : Slope of the sigmoidal function.
                'thresh' : Threshold of the sigmoidal function.
                'tau' : Time constant of the update dynamics.
                'tau_scale' : sets the speed of change for the scaling factor
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
                'Kp' : Gain factor for the scaling of weights (makes changes bigger/smaller).

        Raises:
            AssertionError.

        The actual values used for scaling are calculated in unit.upd_exp_scale() .
        Values around Kp=0.1, tau_scale=0.1 are usually appropriate when c >= 1 .
        When c <= 0 the current implementation is not very stable.
        """

        super(exp_dist_sigmoidal, self).__init__(ID, params, network)
        self.slope = params['slope']    # slope of the sigmoidal function
        self.thresh = params['thresh']  # horizontal displacement of the sigmoidal
        self.tau = params['tau']  # the time constant of the dynamics
        self.tau_scale = params['tau_scale']  # the scaling time constant
        self.Kp = params['Kp']  # gain for synaptic scaling
        self.c = params['c']  # The coefficient in the exponential distribution
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        self.syn_needs.update([syn_reqs.balance, syn_reqs.exp_scale, 
                               syn_reqs.lpf_fast, syn_reqs.inp_vector])
        #assert self.type is unit_types.exp_dist_sigmoidal, ['Unit ' + str(self.ID) + 
        #                                                    ' instantiated with the wrong type']
        
    def f(self, arg):
        """ This is the sigmoidal function. Could roughly think of it as an f-I curve. """
        return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_exp_sc_input_sum(t)) - y[0] ) * self.rtau


class ssrdc_sharp_base(unit): 
    """
    A parent class for units with switchable synaptic scaling based rate distribution control. 
    
    There is an extra port that controls whether or not the synaptic scaling is applied, it is
    called the 'sharpen_port'. The input that is used for rate distribution control should
    be at the 'rdc_port'.

    Whether an input is excitatory or inhibitory is decided by the sign of its initial value.
    Synaptic weights initialized to zero will be considered excitatory.
    """
    # The issue of updating the scaling factors according to the inputs at the sharpen_port is 
    # handled by the unit.upd_exp_scale_shrp method, used by the exp_scale_shrp synaptic requirement.

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports and tau_fast are no longer optional.
                'n_ports' : number of inputs ports. Needs a valuer > 1
                'tau_fast' : Time constant for the fast low-pass filter.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS (use of parameters is in flux. Definitions may be incorrect)
                'tau_scale' : sets the speed of change for the scaling factor
                'tau_relax' : controls how fast the scaling factors return to 1 when rdc is off
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
                'Kp' : Gain factor for the scaling of weights (makes changes bigger/smaller).
                'rdc_port' : port ID of inputs used for rate distribution control.
                'sharpen_port' : port ID where the inputs controlling rdc arrive.

        Raises:
            ValueError

        The actual values used for scaling are calculated in unit.upd_exp_scale_mp() .
        """
        if hasattr(self, 'unit_initialized') and self.unit_initialized:
            pass
        else:    # if the parent constructor hasn't been called
            #self.extra_ports = 1 # Used by double sigma units
            unit.__init__(self, ID, params, network)

        if self.n_ports < 2:
            raise ValueError('ssrdc_sharp units are expected to have at least 2 ports')
        else:
            self.multiport = True # This causes the port_idx list to be created in init_pre_syn_update
        self.rdc_port = params['rdc_port'] # port for rate distribution control
        self.tau_scale = params['tau_scale']  # the scaling time constant
        self.tau_relax = params['tau_relax']  # the second scaling time constant
        self.Kp = params['Kp']  # gain for synaptic scaling
        self.c = params['c']  # The coefficient in the exponential distribution
        self.sharpen_port = params['sharpen_port']
        
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.balance_mp, syn_reqs.exp_scale_shrp, syn_reqs.lpf_fast]) 


class sig_ssrdc_sharp(ssrdc_sharp_base): 
    """
    The synaptic weights are scaled to produce an exponential distribution when input at port 1 > 0.5 .
    
    In other words, this is the same as the exp_dist_sigmoidal unit, but there is an extra port
    that controls whether or not the synaptic scaling is applied. Normal inputs must arrive at
    port 0, and the input that signals whether to apply rate distribution control should be 
    at port 1.

    Whether an input is excitatory or inhibitory is decided by the sign of its initial value.
    Synaptic weights initialized to zero will be considered excitatory.
    """
    # The scale_facs_rdc factors used in get_mp_input sum are updated by the upd_exp_scale
    # method, which is called because the parent class has the exp_scale_shrp requirement.
    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports and tau_fast are no longer optional.
                'n_ports' : number of inputs ports. Needs a valuer > 1
                'tau_fast' : Time constant for the fast low-pass filter.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS (use of parameters is in flux. Definitions may be incorrect)
                'slope' : Slope of the sigmoidal function.
                'thresh' : Threshold of the sigmoidal function.
                'tau' : Time constant of the update dynamics.
                'tau_scale' : sets the speed of change for the scaling factor
                'tau_relax' : controls how fast the scaling factors return to 1 when rdc is off
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
                'Kp' : Gain factor for the scaling of weights (makes changes bigger/smaller).
                'rdc_port' : port ID of inputs used for rate distribution control.
                'sharpen_port' : port ID where the inputs controlling rdc arrive.

        Raises:
            ValueError

        """
        ssrdc_sharp_base.__init__(self, ID, params, network)
        self.slope = params['slope']    # slope of the sigmoidal function
        self.thresh = params['thresh']  # horizontal displacement of the sigmoidal
        self.tau = params['tau']  # the time constant of the dynamics
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        self.nm_prts = list(range(self.n_ports)) # port list without the rdc and sharpen ports
        del self.nm_prts[max(self.rdc_port,self.sharpen_port)]
        del self.nm_prts[min(self.rdc_port,self.sharpen_port)]
        #self.nm_prts = np.array(self.nm_prts, dtype='uint32')
        
    def f(self, arg):
        """ This is the sigmoidal function. Could roughly think of it as an f-I curve. """
        return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_mp_input_sum(t)) - y[0] ) * self.rtau

    def get_mp_input_sum(self, time):
        """ The input function of the sig_ssrdc_sharp unit. """
        ws = self.get_mp_weights(time)
        inps = self.get_mp_inputs(time)
        acc_sum = np.sum(self.scale_facs_rdc * ws[self.rdc_port] * inps[self.rdc_port])
        acc_sum += sum([np.dot(ws[p],inps[p]) for p in self.nm_prts]) 
        return acc_sum 


class sig_ssrdc(unit):
    """
    A sigmoidal unit where the synaptic weights are scaled to produce an exponential distribution. 
    
    This is a multiport version of the exp_dist_sigmoidal unit. Only the inputs received at the
    rdc port will be considered for the purpose of rate distribution control, which allows to
    ignore inputs from source units.

    Synaptic scaling happens at excitatory inputs in the rdc port.
    Whether an input is excitatory or inhibitory is decided by the sign of its initial value.
    Synaptic weights initialized to zero will be considered excitatory.
    """
    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS (use of parameters is in flux. Definitions may be incorrect)
                'slope' : Slope of the sigmoidal function.
                'thresh' : Threshold of the sigmoidal function.
                'tau' : Time constant of the update dynamics.
                'tau_scale' : sets the speed of change for the scaling factor
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
                'Kp' : Gain factor for the scaling of weights (makes changes bigger/smaller).
                'rdc_port' : port ID of inputs used for rate distribution control.

        Raises:
            ValueError

        The actual values used for scaling are calculated in unit.upd_exp_scale_mp() .
        """
        unit.__init__(self, ID, params, network)
        if self.n_ports < 2:
            raise ValueError('sig_ssrdc units need at least 2 ports')
        self.multiport = True  # This causes the port_idx list to be created in init_pre_syn_update
        self.slope = params['slope']    # slope of the sigmoidal function
        self.thresh = params['thresh']  # horizontal displacement of the sigmoidal
        self.tau = params['tau']  # the time constant of the dynamics
        self.tau_scale = params['tau_scale']  # the scaling time constant
        self.Kp = params['Kp']  # gain for synaptic scaling
        self.c = params['c']  # The coefficient in the exponential distribution
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        self.rdc_port = params['rdc_port'] # port for rate distribution control
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.balance_mp, syn_reqs.exp_scale_mp, syn_reqs.lpf_fast]) 
        
    def f(self, arg):
        """ This is the sigmoidal function. Could roughly think of it as an f-I curve. """
        return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_mp_input_sum(t)) - y[0] ) * self.rtau

    def get_mp_input_sum(self, time):
        """ The input function of the sig_ssrdc unit. """
        # TODO: calculate with the approach of sig_ssrdc_sharp
        weights = self.get_mp_weights(time)
        inps = self.get_mp_inputs(time)
        acc_sum = 0.
        for port in range(self.n_ports):
            if port == self.rdc_port:
                acc_sum += np.sum(self.scale_facs_rdc * weights[port] * inps[port])
            else:
                acc_sum += np.dot(weights[port], inps[port]) 
        return acc_sum 

    
class exp_dist_sig_thr(unit): 
    """
    A sigmoidal unit where the threshold is moved to produce an exponential distribution.
    
    This unit has the same activation function as the sigmoidal unit, but the thresh
    parameter is continually adjusted produce an exponential distribution of the firing rates in
    the network, using the same approach as the exp_rate_dist_synapse and the 
    exp_dist_sigmoidal units.

    Whether an input is excitatory or inhibitory is decided by the sign of its initial value.
    Synaptic weights initialized to zero will be considered excitatory.
    """
    # The difference with sigmoidal units is the use of the slide_thresh requirement.
    # This will automatically adjust the threshold at each buffer update.

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS (use of parameters is in flux. Definitions may be incorrect)
                'slope' : Slope of the sigmoidal function.
                'thresh' : Threshold of the sigmoidal function. This value may adapt.
                'tau' : Time constant of the update dynamics.
                'tau_thr' : sets the speed of change for the threshold (reciprocal of time constant). 
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).

        Raises:
            AssertionError.

        When c <= 0 the current implementation is not very stable.
        """

        super(exp_dist_sig_thr, self).__init__(ID, params, network)
        self.slope = params['slope']    # slope of the sigmoidal function
        self.thresh = params['thresh']  # horizontal displacement of the sigmoidal
        self.tau = params['tau']  # the time constant of the dynamics
        self.tau_thr = params['tau_thr']  # the reciprocal of the threshold sliding time constant
        self.c = params['c']  # The coefficient in the exponential distribution
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        self.syn_needs.update([syn_reqs.balance, syn_reqs.slide_thresh,
                               syn_reqs.lpf_fast, syn_reqs.inp_vector])
        #assert self.type is unit_types.exp_dist_sig_thr, ['Unit ' + str(self.ID) + 
        #                                                    ' instantiated with the wrong type']
        
    def f(self, arg):
        """ This is the sigmoidal function. Could roughly think of it as an f-I curve. """
        return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_input_sum(t)) - y[0] ) * self.rtau
 

class sig_trdc(unit): 
    """
    A sigmoidal unit where the threshold is moved to produce an exponential distribution.
    
    The only difference with exp_dist_sig_thr is that this is a multiport unit, so that the
    only inputs considered for the purpose of rate distribution control are those in the 
    rdc port.

    This unit has the same activation function as the sigmoidal unit, but the thresh
    parameter is continually adjusted produce an exponential distribution of the firing rates in
    the network, using the same approach as the exp_rate_dist_synapse and the 
    exp_dist_sigmoidal units.

    Whether an input is excitatory or inhibitory is decided by the sign of its initial value.
    Synaptic weights initialized to zero will be considered excitatory.
    """
    # The difference with sigmoidal units is the use of the slide_thresh requirement.
    # This will automatically adjust the threshold at each buffer update.

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS (use of parameters is in flux. Definitions may be incorrect)
                'slope' : Slope of the sigmoidal function.
                'thresh' : Threshold of the sigmoidal function. This value may adapt.
                'tau' : Time constant of the update dynamics.
                'tau_thr' : sets the speed of change for the threshold (reciprocal of time constant). 
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
                'rdc_port' : port ID of inputs used for rate distribution control.

        Raises:
            ValueError

        """
        if params['n_ports'] < 2:
            raise ValueError('At least two ports whould be used with the sig_trdc units')
        else:
            self.multiport = True
        unit.__init__(self, ID, params, network)
        self.slope = params['slope']    # slope of the sigmoidal function
        self.thresh = params['thresh']  # horizontal displacement of the sigmoidal
        self.tau = params['tau']  # the time constant of the dynamics
        self.tau_thr = params['tau_thr']  # the reciprocal of the threshold sliding time constant
        self.c = params['c']  # The coefficient in the exponential distribution
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        self.rdc_port = params['rdc_port'] # port for rate distribution control
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.balance_mp, syn_reqs.slide_thresh, syn_reqs.lpf_fast]) 
        
    def f(self, arg):
        """ This is the sigmoidal function. Could roughly think of it as an f-I curve. """
        return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_mp_input_sum(t)) - y[0] ) * self.rtau
 
    def get_mp_input_sum(self, time):
        """ The input function of sig_trdc units. """
        weights = self.get_mp_weights(time)
        inps = self.get_mp_inputs(time)
        return sum([np.dot(weights[prt], inps[prt]) for prt in range(self.n_ports)])


class trdc_sharp_base(unit):
    """ 
    The parent class for units with switchable threshold-based rate distribution control. 
    
    One of the parameters is a port number, called "sharpen_port". When the scaled sum of inputs to the 
    sharpen port are larger than 0.5, based on the distribution of inputs at port 'rdc_port' this unit 
    will use threshold-based rate distribution control to produce an exponential distribution of firing 
    rates. When the inputs to the sharpen port are smaller than 0.5 the threshold will decay exponentially
    to a default value called "thr_fix", with time constant reciprocal to "tau_fix".
    
    """
    # The only difference with multiport_trdc_base is the use of the slide_thr_shrp instead of slide_thresh

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports and tau_fast are no longer optional.
                'n_ports' : number of inputs ports. Needs a valuer > 1
                'tau_fast' : Time constant for the fast low-pass filter.

            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                'rdc_port' : port ID of inputs used for rate distribution control.
                'tau_thr' : sets the speed of change for the threshold (reciprocal of time constant). 
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
                'thr_fix' : default value of the threshold (when no distribution control is applied)
                'tau_fix' : reciprocal of time constant for threshold's return to thr_fix.
                'sharpen_port' : port ID where the inputs controlling trdc arrive.
        Raises:
            ValueError

        """
        if hasattr(self, 'unit_initialized') and self.unit_initialized:
            pass
        else:    # if the parent constructor hasn't been called
            self.extra_ports = 1 # Used by the parent class' creator
            unit.__init__(self, ID, params, network)

        if self.n_ports < 2:
            raise ValueError('Multiple ports required for classes derived from trdc_sharp_base')
        else:
            self.multiport = True
        self.rdc_port = params['rdc_port'] # port for rate distribution control
        self.tau_thr = params['tau_thr']  # the reciprocal of the threshold sliding time constant
        self.c = params['c']  # The coefficient in the exponential distribution
        self.thr_fix = params['thr_fix']  # the value where the threshold returns
        self.tau_fix = params['tau_fix']  # the speed of the threshold's return to thr_fix
        self.sharpen_port = params['sharpen_port']
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.balance_mp, syn_reqs.slide_thr_shrp, syn_reqs.lpf_fast])
        # next line is for debugging
        #self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.balance_mp, syn_reqs.slide_thresh, syn_reqs.lpf_fast])


class sig_trdc_sharp(trdc_sharp_base): 
    """
    The threshold slides to produce a rate exp distro when input at port sharpen_port is larger than 0.5 .
    
    This unit has the same activation function as the sigmoidal unit, but the thresh
    parameter is continually adjusted produce an exponential distribution of the firing rates in
    the network, using the same approach as the exp_rate_dist_synapse and the 
    exp_dist_sigmoidal units.

    Whether an input is excitatory or inhibitory is decided by the sign of its initial value.
    Synaptic weights initialized to zero will be considered excitatory.

    Any number of ports larger than 2 is accepted. The port numbers should range from
    0 to n_ports-1. Inputs to ports outside that range will be ignored.
    """
    # The threshold is adjusted by the upd_thr_sharp, which is a requirement of the
    # trdc_sharp_base parent class
    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports and tau_fast are no longer optional.
                'n_ports' : number of inputs ports. Needs a valuer > 1
                'tau_fast' : Time constant for the fast low-pass filter.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS (use of parameters is in flux. Definitions may be incorrect)
                'slope' : Slope of the sigmoidal function.
                'tau' : Time constant of the update dynamics.
                'tau_thr' : sets the speed of change for the threshold (reciprocal of time constant). 
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
                'rdc_port' : port ID of inputs used for rate distribution control.
                'thr_fix' : default value of the threshold (when no distribution control is applied)
                'tau_fix' : reciprocal of time constant for threshold's return to thr_fix.
                'sharpen_port' : port ID where the inputs controlling trdc arrive.

        """
        trdc_sharp_base.__init__(self, ID, params, network) # parent's constructor
        self.slope = params['slope']    # slope of the sigmoidal function
        self.thresh = params['thr_fix']  # horizontal displacement of the sigmoidal
        self.tau = params['tau']  # the time constant of the dynamics
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        self.ports = [int(n) for n in range(self.n_ports) if n != self.sharpen_port]
        self.ports = np.array(self.ports)  # array without the sharpen port

    def f(self, arg):
        """ This is the sigmoidal function. Could roughly think of it as an f-I curve. """
        return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_mp_input_sum(t)) - y[0] ) * self.rtau
 
    def get_mp_input_sum(self, time):
        """ The input function of sig_trdc_sharp units. """
        weights = self.get_mp_weights(time)
        inps = self.get_mp_inputs(time)
        return sum([np.dot(weights[p], inps[p]) for p in self.ports])


class double_sigma_base(unit):
    """ The parent class for units in the double sigma family. """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports is no longer optional.
                'n_ports' : number of inputs ports. n_ports = len(branch_w) + extra_ports

            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                'slope' : Slope of the global sigmoidal function.
                'thresh' : Threshold of the global sigmoidal function. 
                'branch_params' : A dictionary with the following 3 entries:
                    'branch_w' : The "weights" for all branches. This is a list whose length is the number
                            of branches. Each entry is a positive number, and all entries must add to 1.
                            The input port corresponding to a branch is the index of its corresponding 
                            weight in this list, so len(branch_w) = n_ports.
                    'slopes' : Slopes of the branch sigmoidal functions. It can either be a float value
                            (resulting in all values being the same), it can be a list of length n_ports
                            specifying the slope for each branch, or it can be a dictionary specifying a
                            distribution. Currently only the uniform distribution is supported:
                            {..., 'slopes':{'distribution':'uniform', 'low':0.5, 'high':1.5}, ...}
                    'threshs' : Thresholds of the branch sigmoidal functions. It can either be a float
                            value (resulting in all values being the same), it can be a list of length 
                            n_ports specifying the threshold for each branch, or it can be a dictionary 
                            specifying a distribution. Currently only the uniform distribution is supported:
                            {..., 'threshs':{'distribution':'uniform', 'low':-0.1, 'high':0.5}, ...}
                        
                'tau' : Time constant of the update dynamics.
                'phi' : This value is substracted from the branches' contribution. In this manner a branch
                        can have a negative (inhibitory) contribution.

            For units where the number of ports exceeds the number of branches, the
            'extra_ports' variable should be created, so that n_ports = len(branch_w) + extra_ports.
        Raises:
            ValueError, NameError

        """
        # branch_w, slopes, and threshs are inside the branch_params dictionary because if they
        # were lists then network.create_units would interpret them as values to assign to separate units.

        unit.__init__(self, ID, params, network)
        if not hasattr(self,'n_ports'): 
            raise NameError( 'Number of ports should be included in the parameters' )
        self.slope = params['slope']    # slope of the global sigmoidal function
        self.thresh = params['thresh']  # horizontal displacement of the global sigmoidal
        self.tau = params['tau']  # the time constant of the dynamics
        self.phi = params['phi'] # substractive factor for the inner sigmoidals
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        br_pars = params['branch_params']  # to make the following lines shorter
        self.br_w = br_pars['branch_w'] # weight factors for all branches
        if not hasattr(self, 'extra_ports'): # for models with more ports than branches
            self.extra_ports = 0
        if self.n_ports > 1:
            self.multiport = True  # This causes the port_idx list to be created in init_pre_syn_update

        if self.extra_ports > 0:
            n_branches = self.n_ports - self.extra_ports
            txt = ' minus ' + str(self.extra_ports) + ' '
        else:
            n_branches = self.n_ports
            txt = ' '
     
        # testing and initializing the branch parameters
        if n_branches != len(self.br_w):
            raise ValueError('Number of ports'+txt+'should equal the length of the branch_w parameter')
        elif min(self.br_w) <= 0:
            raise ValueError('Elements of branch_w should be positive')
        elif sum(self.br_w) - 1. > 1e-4:
            raise ValueError('Elements of branch_w should add to 1')
        #
        if type(br_pars['slopes']) is list: 
            if len(br_pars['slopes']) == n_branches:
                self.slopes = br_pars['slopes']    # slope of the local sigmoidal functions
            else:
                raise ValueError('Number of ports'+txt+'should equal the length of the slopes parameter')
        elif type(br_pars['slopes']) == float or type(br_pars['slopes']) == int:
                self.slopes = [br_pars['slopes']]*n_branches
        elif type(br_pars['slopes']) == dict:
            if br_pars['slopes']['distribution'] == 'uniform':
                self.slopes = np.random.uniform(br_pars['slopes']['low'], br_pars['slopes']['high'], n_branches)
            else:
                raise ValueError('Unknown distribution used to specify branch slopes')
        else:
            raise ValueError('Invalid type for slopes parameter')
        #
        if type(br_pars['threshs']) is list: 
            if len(br_pars['threshs']) == n_branches:
                self.threshs= br_pars['threshs']    # threshold of the local sigmoidal functions
            else:
                raise ValueError('Number of ports'+txt+'should equal the length of the slopes parameter')
        elif type(br_pars['threshs']) == float or type(br_pars['threshs']) == int:
                self.threshs = [br_pars['threshs']]*n_branches
        elif type(br_pars['threshs']) == dict:
            if br_pars['threshs']['distribution'] == 'uniform':
                self.threshs= np.random.uniform(br_pars['threshs']['low'], br_pars['threshs']['high'], n_branches)  
            else:
                raise ValueError('Unknown distribution used to specify branch thresholds')
        else:
            raise ValueError('Invalid type for threshs parameter')

 
    def f(self, thresh, slope, arg):
        """ The sigmoidal function, with parameters given explicitly."""
        return 1. / (1. + np.exp(-slope*(arg - thresh)))

    def derivatives(self, y, t):
        """ This function returns the derivative of the activity given its current values. """
        return ( self.f(self.thresh, self.slope, self.get_mp_input_sum(t)) - y[0] ) * self.rtau


 
class double_sigma(double_sigma_base):
    """ 
    Sigmoidal unit with multiple dendritic branches modeled as sigmoidal units.

    This model is inspired by:
    Poirazi et al. 2003 "Pyramidal Neuron as Two-Layer Neural Network" Neuron 37,6:989-999

    Each input belongs to a particular "branch". All inputs from the same branch add 
    linearly, and the sum is fed into a sigmoidal function that produces the output of the branch. 
    The output of all the branches is added linearly to produce the total input to the unit, 
    which is fed into a sigmoidal function to produce the output of the unit.

    The equations can be seen in the "double_sigma_unit" tiddler of the programming notes wiki.

    These type of units implement a type of soft constraint satisfaction. According to how
    the parameters are set, they might activate only when certain input branches receive
    enough stimulation.
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports is no longer optional.
                'n_ports' : number of inputs ports. 

            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                'slope' : Slope of the global sigmoidal function.
                'thresh' : Threshold of the global sigmoidal function. 
                'branch_params' : A dictionary with the following 3 entries:
                    'branch_w' : The "weights" for all branches. This is a list whose length is the number
                            of branches. Each entry is a positive number, and all entries must add to 1.
                            The input port corresponding to a branch is the index of its corresponding 
                            weight in this list, so len(branch_w) = n_ports.
                    'slopes' : Slopes of the branch sigmoidal functions. It can either be a float value
                            (resulting in all values being the same), it can be a list of length n_ports
                            specifying the slope for each branch, or it can be a dictionary specifying a
                            distribution. Currently only the uniform distribution is supported:
                            {..., 'slopes':{'distribution':'uniform', 'low':0.5, 'high':1.5}, ...}
                    'threshs' : Thresholds of the branch sigmoidal functions. It can either be a float
                            value (resulting in all values being the same), it can be a list of length 
                            n_ports specifying the threshold for each branch, or it can be a dictionary 
                            specifying a distribution. Currently only the uniform distribution is supported:
                            {..., 'threshs':{'distribution':'uniform', 'low':-0.1, 'high':0.5}, ...}
                        
                'tau' : Time constant of the update dynamics.
                'phi' : This value is substracted from the branches' contribution.
        Raises:
            ValueError, NameError

        """
        super(double_sigma, self).__init__(ID, params, network)
    
    def get_mp_input_sum(self, time):
        """ The input function of the double sigma unit. """
        # The two versions below require almost identical times
        #
        # version 1. The big zip
        return sum( [ o*(self.f(th,sl,np.dot(w,i)) - self.phi) for o,th,sl,w,i in 
                      zip(self.br_w, self.threshs, self.slopes, 
                          self.get_mp_weights(time), self.get_mp_inputs(time))  ] )
        # version 2. The accumulator
        #ret = 0.
        #w = self.get_mp_weights(time)
        #inp = self.get_mp_inputs(time)
        #for i in range(self.n_ports):
        #    ret += self.br_w[i] * (self.f(self.threshs[i], self.slopes[i], np.dot(w[i],inp[i]))-self.phi)
        #return ret


class double_sigma_normal(double_sigma_base):
    """ A version of double_sigma units where the inputs are normalized.

    Double sigma units are inspired by:
    Poirazi et al. 2003 "Pyramidal Neuron as Two-Layer Neural Network" Neuron 37,6:989-999

    Each input belongs to a particular "branch". All inputs from the same branch
    add linearly, are normalized, and the normalized sum is fed into a sigmoidal function 
    that produces the output of the branch. The output of all the branches is added linearly 
    to produce the total input to the unit, which is fed into a sigmoidal function to produce 
    the output of the unit.

    The equations can be seen in the "double_sigma_unit" tiddler of the programming notes wiki.

    These type of units implement a type of soft constraint satisfaction. According to how
    the parameters are set, they might activate only when certain input branches receive
    enough stimulation.
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports and tau_slow are  no longer optional.
                'n_ports' : number of inputs ports. 
                'tau_slow' : time constant for the slow low-pass filter.

            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                'slope' : Slope of the global sigmoidal function.
                'thresh' : Threshold of the global sigmoidal function. 
                'branch_params' : A dictionary with the following 3 entries:
                    'branch_w' : The "weights" for all branches. This is a list whose length is the number
                            of branches. Each entry is a positive number, and all entries must add to 1.
                            The input port corresponding to a branch is the index of its corresponding 
                            weight in this list, so len(branch_w) = n_ports.
                    'slopes' : Slopes of the branch sigmoidal functions. It can either be a float value
                            (resulting in all values being the same), it can be a list of length n_ports
                            specifying the slope for each branch, or it can be a dictionary specifying a
                            distribution. Currently only the uniform distribution is supported:
                            {..., 'slopes':{'distribution':'uniform', 'low':0.5, 'high':1.5}, ...}
                    'threshs' : Thresholds of the branch sigmoidal functions. It can either be a float
                            value (resulting in all values being the same), it can be a list of length 
                            n_ports specifying the threshold for each branch, or it can be a dictionary 
                            specifying a distribution. Currently only the uniform distribution is supported:
                            {..., 'threshs':{'distribution':'uniform', 'low':-0.1, 'high':0.5}, ...}
                        
                'tau' : Time constant of the update dynamics.
                'phi' : This value is substracted from the branches' contribution.
        Raises:
            ValueError, NameError

        """
        # branch_w, slopes, and threshs are inside the branch_params dictionary because if they
        # were lists then network.create_units would interpret them as values to assign to separate units.

        super().__init__(ID, params, network)
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.lpf_slow_mp_inp_sum]) 
     
    
    def get_mp_input_sum(self, time):
        """ The input function of the normalized double sigma unit. """
        # Removing zero or near-zero values from lpf_slow_mp_inp_sum
        lpf_inp_sum = [np.sign(arry)*(np.maximum(np.abs(arry), 1e-3)) for arry in self.lpf_slow_mp_inp_sum]
        #
        # version 1. The big zip
        return sum( [ o*( self.f(th, sl, (np.dot(w,i)-y) / y) - self.phi ) for o,th,sl,y,w,i in 
                     zip(self.br_w, self.threshs, self.slopes, lpf_inp_sum,
                         self.get_mp_weights(time), self.get_mp_inputs(time))  ] )
        # version 2. The accumulator
        #ret = 0.
        #w = self.get_mp_weights(time)
        #inp = self.get_mp_inputs(time)
        #for i in range(self.n_ports):
        #    ret += self.br_w[i] * (self.f(self.threshs[i], self.slopes[i], 
        #            (np.dot(w[i],inp[i])-lpf_inp_sum[i])/lpf_inp_sum[i]) -  self.phi)
        #return ret
    


class sigma_double_sigma(double_sigma_base):
    """ 
    Sigmoidal with somatic inputs and dendritic branches modeled as sigmoidal units.

    This model is a combination of the sigmoidal and double_sigma models, in recognition that
    a neuron usually receives inputs directly to the soma.

    There are two types of inputs. Somatic inputs arrive at port 0 (the default port), and are
    just like inputs to the sigmoidal units. Dendritic inputs arrive at ports with with ID 
    larger than 0, and are just like inputs to double_sigma units.

    Each dendritic input belongs to a particular "branch". All inputs from the same branch add 
    linearly, and the sum is fed into a sigmoidal function that produces the output of the branch. 
    The output of all the branches is added linearly to produce the total input to the unit, 
    which is fed into a sigmoidal function to produce the output of the unit.

    The equations can be seen in the "double_sigma_unit" tiddler of the programming notes wiki.

    These type of units implement a type of soft constraint satisfaction. According to how
    the parameters are set, they might activate only when certain input branches receive
    enough stimulation.
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports is no longer optional.
                'n_ports' : number of inputs ports. 

            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                'slope' : Slope of the global sigmoidal function.
                'thresh' : Threshold of the global sigmoidal function. 
                'branch_params' : A dictionary with the following 3 entries:
                    'branch_w' : The "weights" for all branches. This is a list whose length is the number
                            of branches. Each entry is a positive number, and all entries must add to 1.
                            The input port corresponding to a branch is the index of its corresponding 
                            weight in this list, so len(branch_w) = n_ports.
                    'slopes' : Slopes of the branch sigmoidal functions. It can either be a float value
                            (resulting in all values being the same), it can be a list of length n_ports
                            specifying the slope for each branch, or it can be a dictionary specifying a
                            distribution. Currently only the uniform distribution is supported:
                            {..., 'slopes':{'distribution':'uniform', 'low':0.5, 'high':1.5}, ...}
                    'threshs' : Thresholds of the branch sigmoidal functions. It can either be a float
                            value (resulting in all values being the same), it can be a list of length 
                            n_ports specifying the threshold for each branch, or it can be a dictionary 
                            specifying a distribution. Currently only the uniform distribution is supported:
                            {..., 'threshs':{'distribution':'uniform', 'low':-0.1, 'high':0.5}, ...}
                        
                'tau' : Time constant of the update dynamics.
                'phi' : This value is substracted from the branches' contribution.  Since all branch
                        weights are positive, this allows to have negative contributions.
        Raises:
            ValueError, NameError

        """
        self.extra_ports = 1 # Used by the parent class' creator
        super().__init__(ID, params, network)
            
    def get_mp_input_sum(self, time):
        """ The input function of the sigma_double_sigma unit. """
        w = self.get_mp_weights(time)
        inp = self.get_mp_inputs(time)
        # version 1 is slightly faster
        #
        # version 1. The big zip
        return np.dot(w[0], inp[0]) + sum(
                    [ o*self.f(th,sl,np.dot(w,i)) for o,th,sl,w,i in 
                      zip(self.br_w, self.threshs, self.slopes, w[1:], inp[1:]) ] )
        # version 2. The accumulator
        #ret = np.dot(w[0], inp[0])
        #for i in range(1,self.n_ports):
        #    ret += self.br_w[i-1] * self.f(self.threshs[i-1], self.slopes[i-1], np.dot(w[i],inp[i]))
        #return ret
    

class sigma_double_sigma_normal(double_sigma_base):
    """ sigma_double_sigma unit with  normalized branch inputs.

    Double sigma units are inspired by:
    Poirazi et al. 2003 "Pyramidal Neuron as Two-Layer Neural Network" Neuron 37,6:989-999

    There are two types of inputs. Somatic inputs arrive at port 0 (the default port), and are
    just like inputs to the sigmoidal units. Dendritic inputs arrive at ports with with ID 
    larger than 0, and are just like inputs to double_sigma units.

    Each dendritic input belongs to a particular "branch". All inputs from the same branch
    add linearly, are normalized, and the normalized sum is fed into a sigmoidal function 
    that produces the output of the branch. The output of all the branches is added linearly 
    to the somatic input to produce the total input to the unit, which is fed into a sigmoidal 
    function to produce the output of the unit.

    ~~~~~~~~~~ In this version the somatic inputs are not normalized ~~~~~~~~~~

    The equations can be seen in the "double_sigma_unit" tiddler of the programming notes wiki.
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports, tau_fast and tau_slow are  no longer optional.
                'n_ports' : number of inputs ports. 
                'tau_slow' : time constant for the slow low-pass filter.

            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                'slope' : Slope of the global sigmoidal function.
                'thresh' : Threshold of the global sigmoidal function. 
                'branch_params' : A dictionary with the following 3 entries:
                    branch_w: The "weights" for all branches. This is a list whose length is the number
                            of branches. Each entry is a positive number, and all entries must add to 1.
                            The input port corresponding to a branch is the index of its corresponding 
                            weight in this list plus one. For example, if branch_w=[.2, .8], then
                            input port 0 is at the soma, input port 1 has weight .2, and port 2 weight .8 .
                    'slopes' : Slopes of the branch sigmoidal functions. It can either be a float value
                            (resulting in all values being the same), it can be a list of length n_ports
                            specifying the slope for each branch, or it can be a dictionary specifying a
                            distribution. Currently only the uniform distribution is supported:
                            {..., 'slopes':{'distribution':'uniform', 'low':0.5, 'high':1.5}, ...}
                    'threshs' : Thresholds of the branch sigmoidal functions. It can either be a float
                            value (resulting in all values being the same), it can be a list of length 
                            n_ports specifying the threshold for each branch, or it can be a dictionary 
                            specifying a distribution. Currently only the uniform distribution is supported:
                            {..., 'threshs':{'distribution':'uniform', 'low':-0.1, 'high':0.5}, ...}
                        
                'tau' : Time constant of the update dynamics.
                'phi' : This value is substracted from the branches' contribution.  Since all branch
                        weights are positive, this allows to have negative contributions.
                
        """
        self.extra_ports = 1 # The soma port        
        double_sigma_base.__init__(self, ID, params, network)
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.lpf_slow_mp_inp_sum]) 
     
    
    def get_mp_input_sum(self, time):
        """ The input function of the normalized double sigma unit. """
        # Removing zero or near-zero values from lpf_slow_mp_inp_sum
        lpf_inp_sum = [np.sign(arry)*(np.maximum(np.abs(arry), 1e-3)) for arry in self.lpf_slow_mp_inp_sum]
        w = self.get_mp_weights(time)
        inp = self.get_mp_inputs(time)
         
        return np.dot(w[0], inp[0]) + sum( [ o*( self.f(th, sl, (np.dot(w,i)-y) / y) - self.phi ) 
               for o,th,sl,y,w,i in zip(self.br_w, self.threshs, self.slopes, lpf_inp_sum[1:], w[1:], inp[1:]) ] )
 

class multiport_trdc_base(unit):
    """ 
    The parent class for multiport units with threshold-based rate distribution control. 
    
    Based on the distribution of inputs at port 'rdc_port', this unit will use threshold-based rate
    distribution control. If the unit is part of the population that projects to port 'rdc_port', this 
    will contribute to produce an exponential distribution of firing rates.
    
    """
    # The mechanism for firing-rate distribution control is the same as the one used in the
    # exp_dist_sig_thr units. The only difference is that the 'balance' synaptic requirement has
    # been replaced by 'balance_mp', which uses only the inputs at port 'rdc_port'.

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports and tau_fast are no longer optional.
                'n_ports' : number of inputs ports. Needs a valuer > 1
                'tau_fast' : Time constant for the fast low-pass filter.

            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                'rdc_port' : port ID of inputs used for rate distribution control.
                'tau_thr' : sets the speed of change for the threshold (reciprocal of time constant). 
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
        Raises:
            ValueError

        """
        if hasattr(self, 'unit_initialized') and self.unit_initialized:
            pass
        else:    # if the parent constructor hasn't been called
            unit.__init__(self, ID, params, network)

        if self.n_ports < 2:
            raise ValueError('Multiple ports required for classes derived from multiport_trdc_base')
        else:
            self.multiport = True
        self.rdc_port = params['rdc_port'] # port for rate distribution control
        self.tau_thr = params['tau_thr']  # the reciprocal of the threshold sliding time constant
        self.c = params['c']  # The coefficient in the exponential distribution
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.balance_mp, syn_reqs.slide_thresh, syn_reqs.lpf_fast])



class double_sigma_trdc(double_sigma_base, multiport_trdc_base):
    """ 
    double-sigma unit with threshold-based rate distribution control.

    The double-sigma model is inspired by:
    Poirazi et al. 2003 "Pyramidal Neuron as Two-Layer Neural Network" Neuron 37,6:989-999

    Each input belongs to a particular "branch". All inputs from the same branch add 
    linearly, and the sum is fed into a sigmoidal function that produces the output of the branch. 
    The output of all the branches is added linearly to produce the total input to the unit, 
    which is fed into a sigmoidal function to produce the output of the unit.

    Based on the distribution of inputs at port 'rdc_port', this unit will use threshold-based rate
    distribution control. If the unit is part of the population that projects to port 'rdc_port', this 
    will contribute to produce an exponential distribution of firing rates.

    The equations of the doulble-sigma unit can be seen in the "double_sigma_unit" tiddler of 
    the programming notes wiki.
    """
    # Inheritance for this unit has a "diamond" scheme, since both parents inherit from unit;
    # mutiport_trdc_base, however, only implements a constructor. This constructor will not
    # call the unit constructor if unit_initialized=True .

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports and tau_fast are no longer optional.
                'n_ports' : number of inputs ports. 
                'tau_fast' : Time constant for the fast low-pass filter.

            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                'slope' : Slope of the global sigmoidal function.
                'thresh' : Threshold of the global sigmoidal function. 
                'branch_params' : A dictionary with the following 3 entries:
                    'branch_w' : The "weights" for all branches. This is a list whose length is the number
                            of branches. Each entry is a positive number, and all entries must add to 1.
                            The input port corresponding to a branch is the index of its corresponding 
                            weight in this list, so len(branch_w) = n_ports.
                    'slopes' : Slopes of the branch sigmoidal functions. It can either be a float value
                            (resulting in all values being the same), it can be a list of length n_ports
                            specifying the slope for each branch, or it can be a dictionary specifying a
                            distribution. Currently only the uniform distribution is supported:
                            {..., 'slopes':{'distribution':'uniform', 'low':0.5, 'high':1.5}, ...}
                    'threshs' : Thresholds of the branch sigmoidal functions. It can either be a float
                            value (resulting in all values being the same), it can be a list of length 
                            n_ports specifying the threshold for each branch, or it can be a dictionary 
                            specifying a distribution. Currently only the uniform distribution is supported:
                            {..., 'threshs':{'distribution':'uniform', 'low':-0.1, 'high':0.5}, ...}
                        
                'tau' : Time constant of the update dynamics.
                'phi' : This value is substracted from the branches' contribution.  Since all branch
                        weights are positive, this allows to have negative contributions.
                'rdc_port' : port ID of inputs used for rate distribution control.
                'tau_thr' : sets the speed of change for the threshold (reciprocal of time constant).
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
        Raises:
            ValueError, NameError

        """
        double_sigma_base.__init__(self, ID, params, network)
        self.unit_initialized = True  # to avoid calling the unit constructor twice
        multiport_trdc_base.__init__(self, ID, params, network)

            
    def get_mp_input_sum(self, time):
        """ The input function of the double sigma unit. """
        # The two versions below require almost identical times
        #
        # version 1. The big zip
        return sum( [ o*(self.f(th,sl,np.dot(w,i)) - self.phi) for o,th,sl,w,i in 
                      zip(self.br_w, self.threshs, self.slopes, 
                          self.get_mp_weights(time), self.get_mp_inputs(time))  ] )
        # version 2. The accumulator
        #ret = 0.
        #w = self.get_mp_weights(time)
        #inp = self.get_mp_inputs(time)
        #for i in range(self.n_ports):
        #    ret += self.br_w[i] * (self.f(self.threshs[i], self.slopes[i], np.dot(w[i],inp[i]))-self.phi)
        #return ret
    


class sigma_double_sigma_trdc(double_sigma_base, multiport_trdc_base):
    """ 
    sigma_double_sigma unit with threshold-based rate distribution control.

    This model is a combination of the sigmoidal and sigma_double_sigma models, in recognition 
    that a neuron usually receives inputs directly to the soma.

    There are two types of inputs. Somatic inputs arrive at port 0 (the default port), and are
    just like inputs to the sigmoidal units. Dendritic inputs arrive at ports with with ID 
    larger than 0, and are just like inputs to double_sigma units.

    Each dendritic input belongs to a particular "branch". All inputs from the same branch add 
    linearly, and the sum is fed into a sigmoidal function that produces the output of the branch. 
    The output of all the branches is added linearly to produce the total input to the unit, 
    which is fed into a sigmoidal function to produce the output of the unit.

    Based on the distribution of inputs at port 'rdc_port', this unit will use threshold-based rate
    distribution control. If the unit is part of the population that projects to port 'rdc_port', this 
    will contribute to produce an exponential distribution of firing rates.

    The equations can be seen in the "double_sigma_unit" tiddler of the programming notes wiki.
    """
    # Inheritance for this unit has a "diamond" scheme, since both parents inherit from unit;
    # mutiport_trdc_base, however, only implements a constructor. This constructor will not
    # call the unit constructor if unit_initialized=True .

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports and tau_fast are no longer optional.
                'n_ports' : number of inputs ports. 
                'tau_fast' : Time constant for the fast low-pass filter.

            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                'slope' : Slope of the global sigmoidal function.
                'thresh' : Threshold of the global sigmoidal function. 
                'branch_params' : A dictionary with the following 3 entries:
                    'branch_w' : The "weights" for all branches. This is a list whose length is the number
                            of branches. Each entry is a positive number, and all entries must add to 1.
                            The input port corresponding to a branch is the index of its corresponding 
                            weight in this list, so len(branch_w) = n_ports.
                    'slopes' : Slopes of the branch sigmoidal functions. It can either be a float value
                            (resulting in all values being the same), it can be a list of length n_ports
                            specifying the slope for each branch, or it can be a dictionary specifying a
                            distribution. Currently only the uniform distribution is supported:
                            {..., 'slopes':{'distribution':'uniform', 'low':0.5, 'high':1.5}, ...}
                    'threshs' : Thresholds of the branch sigmoidal functions. It can either be a float
                            value (resulting in all values being the same), it can be a list of length 
                            n_ports specifying the threshold for each branch, or it can be a dictionary 
                            specifying a distribution. Currently only the uniform distribution is supported:
                            {..., 'threshs':{'distribution':'uniform', 'low':-0.1, 'high':0.5}, ...}
                        
                'tau' : Time constant of the update dynamics.
                'phi' : This value is substracted from the branches' contribution.  Since all branch
                        weights are positive, this allows to have negative contributions.
                'rdc_port' : port ID of inputs used for rate distribution control.
                'tau_thr' : sets the speed of change for the threshold (reciprocal of time constant). 
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).

        Raises:
            ValueError, NameError

        """
        self.extra_ports = 1 # Used by the parent class' creator
        double_sigma_base.__init__(self, ID, params, network)
        self.unit_initialized = True  # to avoid calling the unit constructor twice
        multiport_trdc_base.__init__(self, ID, params, network)

            
    def get_mp_input_sum(self, time):
        """ Identical to the input function of the sigma_double_sigma unit. """
        w = self.get_mp_weights(time)
        inp = self.get_mp_inputs(time)
        # version 1 is slightly faster
        #
        # version 1. The big zip
        return np.dot(w[0], inp[0]) + sum(
                    [ o*self.f(th,sl,np.dot(w,i)) for o,th,sl,w,i in 
                      zip(self.br_w, self.threshs, self.slopes, w[1:], inp[1:]) ] )
        # version 2. The accumulator
        #ret = np.dot(w[0], inp[0])
        #for i in range(1,self.n_ports):
        #    ret += self.br_w[i-1] * self.f(self.threshs[i-1], self.slopes[i-1], np.dot(w[i],inp[i]))
        #return ret
    

class double_sigma_normal_trdc(double_sigma_base, multiport_trdc_base):
    """ double_sigma units with normalized inputs, and threshold-based rate distro control.

    Double sigma units are inspired by:
    Poirazi et al. 2003 "Pyramidal Neuron as Two-Layer Neural Network" Neuron 37,6:989-999

    Each input belongs to a particular "branch". All inputs from the same branch
    add linearly, are normalized, and the normalized sum is fed into a sigmoidal function 
    that produces the output of the branch. The output of all the branches is added linearly 
    to produce the total input to the unit, which is fed into a sigmoidal function to produce 
    the output of the unit.

    The equations can be seen in the "double_sigma_unit" tiddler of the programming notes wiki.

    Based on the distribution of inputs at port 'rdc_port', this unit will use threshold-based rate
    distribution control. If the unit is part of the population that projects to port 'rdc_port', this 
    will contribute to produce an exponential distribution of firing rates.
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports, tau_fast, and tau_slow are  no longer optional.
                'n_ports' : number of inputs ports. 
                'tau_slow' : time constant for the slow low-pass filter.
                'tau_fast' : Time constant for the fast low-pass filter.

            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                'slope' : Slope of the global sigmoidal function.
                'thresh' : Threshold of the global sigmoidal function. 
                'branch_params' : A dictionary with the following 3 entries:
                    'branch_w' : The "weights" for all branches. This is a list whose length is the number
                            of branches. Each entry is a positive number, and all entries must add to 1.
                            The input port corresponding to a branch is the index of its corresponding 
                            weight in this list, so len(branch_w) = n_ports.
                    'slopes' : Slopes of the branch sigmoidal functions. It can either be a float value
                            (resulting in all values being the same), it can be a list of length n_ports
                            specifying the slope for each branch, or it can be a dictionary specifying a
                            distribution. Currently only the uniform distribution is supported:
                            {..., 'slopes':{'distribution':'uniform', 'low':0.5, 'high':1.5}, ...}
                    'threshs' : Thresholds of the branch sigmoidal functions. It can either be a float
                            value (resulting in all values being the same), it can be a list of length 
                            n_ports specifying the threshold for each branch, or it can be a dictionary 
                            specifying a distribution. Currently only the uniform distribution is supported:
                            {..., 'threshs':{'distribution':'uniform', 'low':-0.1, 'high':0.5}, ...}
                        
                'tau' : Time constant of the update dynamics.
                'phi' : This value is substracted from the branches' contribution.  Since all branch
                        weights are positive, this allows to have negative contributions.
                'rdc_port' : port ID of inputs used for rate distribution control.
                'tau_thr' : sets the speed of change for the threshold (reciprocal of time constant). 
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
        Raises:
            ValueError, NameError
        """
        # Inheritance for this unit has a "diamond" scheme, since both parents inherit from unit;
        # mutiport_trdc_base, however, only implements a constructor. This constructor will not
        # call the unit constructor if unit_initialized=True .
        double_sigma_base.__init__(self, ID, params, network)
        self.unit_initialized = True  # to avoid calling the unit constructor twice
        multiport_trdc_base.__init__(self, ID, params, network)
        self.syn_needs.update([syn_reqs.lpf_slow_mp_inp_sum]) 
     
    
    def get_mp_input_sum(self, time):
        """ The input function of the normalized double sigma unit. """
        # Removing zero or near-zero values from lpf_slow_mp_inp_sum
        lpf_inp_sum = [np.sign(arry)*(np.maximum(np.abs(arry), 1e-3)) for arry in self.lpf_slow_mp_inp_sum]
        #
        # version 1. The big zip
        return sum( [ o*( self.f(th, sl, (np.dot(w,i)-y) / y) - self.phi ) for o,th,sl,y,w,i in 
                     zip(self.br_w, self.threshs, self.slopes, lpf_inp_sum,
                         self.get_mp_weights(time), self.get_mp_inputs(time))  ] )
         

class double_sigma_sharp(double_sigma_base, trdc_sharp_base):
    """ 
    double-sigma unit with threshold-based rate distribution control controled by a 'sharpen' port.

    The double-sigma model is inspired by:
    Poirazi et al. 2003 "Pyramidal Neuron as Two-Layer Neural Network" Neuron 37,6:989-999

    Each input belongs to a particular "branch". All inputs from the same branch add 
    linearly, and the sum is fed into a sigmoidal function that produces the output of the branch. 
    The output of all the branches is added linearly to produce the total input to the unit, 
    which is fed into a sigmoidal function to produce the output of the unit.

    One of the parameters is a port number, called "sharpen_port". When the scaled sum of inputs to the 
    sharpen port are larger than 0.5, based on the distribution of inputs at port 'rdc_port' this unit 
    will use threshold-based rate distribution control to produce an exponential distribution of firing 
    rates. When the inputs to the sharpen port are smaller than 0.5 the threshold will decay exponentially
    to a default value called "thr_fix", with a rate set by "tau_fix".

    The inputs at the sharpen port will not contribute to the activation of the unit. 

    The equations of the doulble-sigma unit can be seen in the "double_sigma_unit" tiddler of 
    the programming notes wiki.
    """
    # Inheritance for this unit has a "diamond" scheme, since both parents inherit from unit;
    # trdc_sharp_base, however, only implements a constructor. This constructor will not
    # call the unit constructor if unit_initialized=True .

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports and tau_fast are no longer optional.
                'n_ports' : number of inputs ports. Should be larger than 1.
                'tau_fast' : Time constant for the fast low-pass filter.

            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                'slope' : Slope of the global sigmoidal function.
                'thresh' : Threshold of the global sigmoidal function. 
                'branch_params' : A dictionary with the following 3 entries:
                    'branch_w' : The "weights" for all branches. This is a list whose length is the number
                            of branches. Each entry is a positive number, and all entries must add to 1.
                            The input port corresponding to a branch is the index of its corresponding 
                            weight in this list, so len(branch_w) = n_ports.
                    'slopes' : Slopes of the branch sigmoidal functions. It can either be a float value
                            (resulting in all values being the same), it can be a list of length n_ports
                            specifying the slope for each branch, or it can be a dictionary specifying a
                            distribution. Currently only the uniform distribution is supported:
                            {..., 'slopes':{'distribution':'uniform', 'low':0.5, 'high':1.5}, ...}
                    'threshs' : Thresholds of the branch sigmoidal functions. It can either be a float
                            value (resulting in all values being the same), it can be a list of length 
                            n_ports specifying the threshold for each branch, or it can be a dictionary 
                            specifying a distribution. Currently only the uniform distribution is supported:
                            {..., 'threshs':{'distribution':'uniform', 'low':-0.1, 'high':0.5}, ...}
                        
                'tau' : Time constant of the update dynamics.
                'phi' : This value is substracted from the branches' contribution.  Since all branch
                        weights are positive, this allows to have negative contributions.
                'rdc_port' : port ID of inputs used for rate distribution control.
                'tau_thr' : sets the speed of change for the threshold (reciprocal of time constant). 
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
                'thr_fix' : default value of the threshold (when no distribution control is applied)
                'tau_fix' : reciprocal of time constant for threshold's return to thr_fix.
                'sharpen_port' : port ID where the inputs controlling trdc arrive.

        """
        self.extra_ports = 1 # Used by the double_sigma_base creator
        double_sigma_base.__init__(self, ID, params, network)
        self.unit_initialized = True  # to avoid calling the unit constructor twice
        trdc_sharp_base.__init__(self, ID, params, network)

            
    def get_mp_input_sum(self, time):
        """ The input function of the double sigma sharp unit. """
        # First we should remove the inputs at the 'sharpen' port
        weights = self.get_mp_weights(time)
        inputs = self.get_mp_inputs(time)
        del weights[self.sharpen_port]
        del inputs[self.sharpen_port]

        return sum( [ o*(self.f(th,sl,np.dot(w,i)) - self.phi) for o,th,sl,w,i in 
                      zip(self.br_w, self.threshs, self.slopes, weights, inputs) ] )
 


class sigma_double_sigma_sharp(double_sigma_base, trdc_sharp_base):
    """ 
    sigma_double_sigma unit with switchable threshold-based rate distribution control.

    There are two types of inputs. Somatic inputs arrive at port 0 (the default port), and are
    just like inputs to the sigmoidal units. Dendritic inputs arrive at ports with with ID 
    larger than 0, and are just like inputs to double_sigma units.

    Each dendritic input belongs to a particular "branch". All inputs from the same branch add 
    linearly, and the sum is fed into a sigmoidal function that produces the output of the branch. 
    The output of all the branches is added linearly to produce the total input to the unit, 
    which is fed into a sigmoidal function to produce the output of the unit.

    One of the parameters is a port number, called "sharpen_port". When the scaled sum of inputs to the 
    sharpen port are larger than 0.5, based on the distribution of inputs at port 'rdc_port' this unit 
    will use threshold-based rate distribution control to produce an exponential distribution of firing 
    rates. When the inputs to the sharpen port are smaller than 0.5 the threshold will decay exponentially
    to a default value called "thr_fix", with a rate set by "tau_fix".
   
    The equations can be seen in the "double_sigma_unit" tiddler of the programming notes wiki.
    """
    # Inheritance for this unit has a "diamond" scheme, since both parents inherit from unit;
    # trdc_sharp_base, however, only implements a constructor. This constructor will not
    # call the unit constructor if unit_initialized=True .

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports and tau_fast are no longer optional.
                'n_ports' : number of inputs ports. 
                'tau_fast' : time constant for the fast low-pass filter

            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                'slope' : Slope of the global sigmoidal function.
                'thresh' : Threshold of the global sigmoidal function. 
                'branch_params' : A dictionary with the following 3 entries:
                    'branch_w' : The "weights" for all branches. This is a list whose length is the number
                            of branches. Each entry is a positive number, and all entries must add to 1.
                            The input port corresponding to a branch is the index of its corresponding 
                            weight in this list, so len(branch_w) = n_ports.
                    'slopes' : Slopes of the branch sigmoidal functions. It can either be a float value
                            (resulting in all values being the same), it can be a list of length n_ports
                            specifying the slope for each branch, or it can be a dictionary specifying a
                            distribution. Currently only the uniform distribution is supported:
                            {..., 'slopes':{'distribution':'uniform', 'low':0.5, 'high':1.5}, ...}
                    'threshs' : Thresholds of the branch sigmoidal functions. It can either be a float
                            value (resulting in all values being the same), it can be a list of length 
                            n_ports specifying the threshold for each branch, or it can be a dictionary 
                            specifying a distribution. Currently only the uniform distribution is supported:
                            {..., 'threshs':{'distribution':'uniform', 'low':-0.1, 'high':0.5}, ...}
                        
                'tau' : Time constant of the update dynamics.
                'phi' : This value is substracted from the branches' contribution.  Since all branch
                        weights are positive, this allows to have negative contributions.
                'rdc_port' : port ID of inputs used for rate distribution control.
                'tau_thr' : sets the speed of change for the threshold (reciprocal of time constant). 
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
                'thr_fix' : default value of the threshold (when no distribution control is applied)
                'tau_fix' : reciprocal of time constant for threshold's return to thr_fix.
                'sharpen_port' : port ID where the inputs controlling trdc arrive.


        """
        self.extra_ports = 2 # One for the soma, one for 'sharpen_port'
        double_sigma_base.__init__(self, ID, params, network)
        self.unit_initialized = True  # to avoid calling the unit constructor twice
        trdc_sharp_base.__init__(self, ID, params, network)

            
    def get_mp_input_sum(self, time):
        """ Identical to the input function of the sigma_double_sigma unit. """
        w = self.get_mp_weights(time)
        inp = self.get_mp_inputs(time)
        del w[self.sharpen_port]
        del inp[self.sharpen_port]
        return np.dot(w[0], inp[0]) + sum(
                    [ o*self.f(th,sl,np.dot(w,i)) for o,th,sl,w,i in 
                      zip(self.br_w, self.threshs, self.slopes, w[1:], inp[1:]) ] )
 

class double_sigma_normal_sharp(double_sigma_base, trdc_sharp_base):
    """ double_sigma units with normalized inputs and switchable threshold-based rate distro control.

    Double sigma units are inspired by:
    Poirazi et al. 2003 "Pyramidal Neuron as Two-Layer Neural Network" Neuron 37,6:989-999

    Each input belongs to a particular "branch". All inputs from the same branch
    add linearly, are normalized, and the normalized sum is fed into a sigmoidal function 
    that produces the output of the branch. The output of all the branches is added linearly 
    to produce the total input to the unit, which is fed into a sigmoidal function to produce 
    the output of the unit.

    The equations can be seen in the "double_sigma_unit" tiddler of the programming notes wiki.

    One of the parameters is a port number, called "sharpen_port". When the scaled sum of inputs to the 
    sharpen port are larger than 0.5, based on the distribution of inputs at port 'rdc_port' this unit 
    will use threshold-based rate distribution control to produce an exponential distribution of firing 
    rates. When the inputs to the sharpen port are smaller than 0.5 the threshold will decay exponentially
    to a default value called "thr_fix", with a rate set by "tau_fix".
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports, tau_fast, and tau_slow are  no longer optional.
                'n_ports' : number of inputs ports. 
                'tau_slow' : time constant for the slow low-pass filter.
                'tau_fast' : time constant for the fast low-pass filter

            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                'slope' : Slope of the global sigmoidal function.
                'thresh' : Threshold of the global sigmoidal function. 
                'branch_params' : A dictionary with the following 3 entries:
                    'branch_w' : The "weights" for all branches. This is a list whose length is the number
                            of branches. Each entry is a positive number, and all entries must add to 1.
                            The input port corresponding to a branch is the index of its corresponding 
                            weight in this list, so len(branch_w) = n_ports.
                    'slopes' : Slopes of the branch sigmoidal functions. It can either be a float value
                            (resulting in all values being the same), it can be a list of length n_ports
                            specifying the slope for each branch, or it can be a dictionary specifying a
                            distribution. Currently only the uniform distribution is supported:
                            {..., 'slopes':{'distribution':'uniform', 'low':0.5, 'high':1.5}, ...}
                    'threshs' : Thresholds of the branch sigmoidal functions. It can either be a float
                            value (resulting in all values being the same), it can be a list of length 
                            n_ports specifying the threshold for each branch, or it can be a dictionary 
                            specifying a distribution. Currently only the uniform distribution is supported:
                            {..., 'threshs':{'distribution':'uniform', 'low':-0.1, 'high':0.5}, ...}
                        
                'tau' : Time constant of the update dynamics.
                'phi' : This value is substracted from the branches' contribution.  Since all branch
                        weights are positive, this allows to have negative contributions.
                'rdc_port' : port ID of inputs used for rate distribution control.
                'tau_thr' : sets the speed of change for the threshold (reciprocal of time constant). 
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
                'thr_fix' : default value of the threshold (when no distribution control is applied)
                'tau_fix' : reciprocal of time constant for threshold's return to thr_fix.
                'sharpen_port' : port ID where the inputs controlling trdc arrive.


        """
        # Inheritance for this unit has a "diamond" scheme, since both parents inherit from unit;
        # trdc_sharp_base, however, only implements a constructor. This constructor will not
        # call the unit constructor if unit_initialized=True .
        self.extra_ports = 1 # The sharpen port        
        double_sigma_base.__init__(self, ID, params, network)
        self.unit_initialized = True  # to avoid calling the unit constructor twice
        trdc_sharp_base.__init__(self, ID, params, network)
        self.syn_needs.update([syn_reqs.lpf_slow_mp_inp_sum]) 
     
    
    def get_mp_input_sum(self, time):
        """ The input function of the normalized double sigma unit. """
        # Removing zero or near-zero values from lpf_slow_mp_inp_sum
        lpf_inp_sum = [np.sign(arry)*(np.maximum(np.abs(arry), 1e-3)) for arry in self.lpf_slow_mp_inp_sum]
        w = self.get_mp_weights(time)
        inp = self.get_mp_inputs(time)
        del w[self.sharpen_port]
        del inp[self.sharpen_port]
         
        return sum( [ o*( self.f(th, sl, (np.dot(w,i)-y) / y) - self.phi ) for o,th,sl,y,w,i in 
                     zip(self.br_w, self.threshs, self.slopes, lpf_inp_sum, w, inp)  ] )
 


class sigma_double_sigma_normal_sharp(double_sigma_base, trdc_sharp_base):
    """ sigma_double_sigma units with  normalized branch inputs, and switchable threshold-based rate distro control.

    Double sigma units are inspired by:
    Poirazi et al. 2003 "Pyramidal Neuron as Two-Layer Neural Network" Neuron 37,6:989-999

    There are two types of inputs. Somatic inputs arrive at port 0 (the default port), and are
    just like inputs to the sigmoidal units. Dendritic inputs arrive at ports with with ID 
    larger than 0, and are just like inputs to double_sigma units.

    Each dendritic input belongs to a particular "branch". All inputs from the same branch
    add linearly, are normalized, and the normalized sum is fed into a sigmoidal function 
    that produces the output of the branch. The output of all the branches is added linearly 
    to produce the total input to the unit, which is fed into a sigmoidal function to produce 
    the output of the unit. 
    
    ~~~~~~~~~~ In this version the somatic inputs are not normalized ~~~~~~~~~~

    The equations can be seen in the "double_sigma_unit" tiddler of the programming notes wiki.

    One of the parameters is a port number, called "sharpen_port". When the scaled sum of inputs to the 
    sharpen port are larger than 0.5, based on the distribution of inputs at port 'rdc_port' this unit 
    will use threshold-based rate distribution control to produce an exponential distribution of firing 
    rates. When the inputs to the sharpen port are smaller than 0.5 the threshold will decay exponentially
    to a default value called "thr_fix", with a rate set by "tau_fix".

    The preferred way to assign the ports is: 0 -> soma; 1,...,n -> branches; n+1 -> sharpen. 
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports, tau_fast and tau_slow are  no longer optional.
                'n_ports' : number of inputs ports. 
                'tau_slow' : time constant for the slow low-pass filter.
                'tau_fast' : time constant for the fast low-pass filter

            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                'slope' : Slope of the global sigmoidal function.
                'thresh' : Threshold of the global sigmoidal function. 
                'branch_params' : A dictionary with the following 3 entries:
                    branch_w: The "weights" for all branches. This is a list whose length is the number
                            of branches. Each entry is a positive number, and all entries must add to 1.
                            The input port corresponding to a branch is the index of its corresponding 
                            weight in this list plus one. For example, if branch_w=[.2, .8], then
                            input port 0 is at the soma, input port 1 has weight .2, port 2 weight .8, and
                            port 3 is the sharpening port.
                    'slopes' : Slopes of the branch sigmoidal functions. It can either be a float value
                            (resulting in all values being the same), it can be a list of length n_ports
                            specifying the slope for each branch, or it can be a dictionary specifying a
                            distribution. Currently only the uniform distribution is supported:
                            {..., 'slopes':{'distribution':'uniform', 'low':0.5, 'high':1.5}, ...}
                    'threshs' : Thresholds of the branch sigmoidal functions. It can either be a float
                            value (resulting in all values being the same), it can be a list of length 
                            n_ports specifying the threshold for each branch, or it can be a dictionary 
                            specifying a distribution. Currently only the uniform distribution is supported:
                            {..., 'threshs':{'distribution':'uniform', 'low':-0.1, 'high':0.5}, ...}
                        
                'tau' : Time constant of the update dynamics.
                'phi' : This value is substracted from the branches' contribution.  Since all branch
                        weights are positive, this allows to have negative contributions.
                'rdc_port' : port ID of inputs used for rate distribution control.
                'tau_thr' : sets the speed of change for the threshold (reciprocal of time constant). 
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
                'thr_fix' : default value of the threshold (when no distribution control is applied)
                'tau_fix' : reciprocal of time constant for threshold's return to thr_fix.
                'sharpen_port' : port ID where the inputs controlling trdc arrive.


        """
        # Inheritance for this unit has a "diamond" scheme, since both parents inherit from unit;
        # trdc_sharp_base, however, only implements a constructor. This constructor will not
        # call the unit constructor if unit_initialized=True .
        self.extra_ports = 2 # The soma and the sharpen port        
        double_sigma_base.__init__(self, ID, params, network)
        self.unit_initialized = True  # to avoid calling the unit constructor twice
        trdc_sharp_base.__init__(self, ID, params, network)
        self.syn_needs.update([syn_reqs.lpf_slow_mp_inp_sum]) 
     
    
    def get_mp_input_sum(self, time):
        """ The input function of the normalized double sigma unit. """
        # Removing zero or near-zero values from lpf_slow_mp_inp_sum
        lpf_inp_sum = [np.sign(arry)*(np.maximum(np.abs(arry), 1e-3)) for arry in self.lpf_slow_mp_inp_sum]
        w = self.get_mp_weights(time)
        inp = self.get_mp_inputs(time)
        del w[self.sharpen_port]
        del inp[self.sharpen_port]
         
        return np.dot(w[0], inp[0]) + sum( [ o*( self.f(th, sl, (np.dot(w,i)-y) / y) - self.phi ) 
               for o,th,sl,y,w,i in zip(self.br_w, self.threshs, self.slopes, lpf_inp_sum[1:], w[1:], inp[1:]) ] )
 

class sliding_threshold_harmonic_rate_sigmoidal(unit):
    """ Sigmoidal unit that slides its threshold in order to produce a harmonic pattern of firing rates.

    What this means is that the threshold will adjust by following a gradient that comes from
    the discrete Laplacian operator applied to the directed graph where the nodes are the units, and
    the edges come from their connections.

    In practice this means that the firing rate of the unit will be adjusted so it approaches the mean 
    firing rate of its inputs. 

    It is expected that one input port will have the inputs to be used for the rate harmonization,
    usually the internal connections, and another port will have the external inputs, which are
    ignored for rate harmonization. Thus, the number of ports should be at least 2, and the port
    whose inputs are used to harmonize the rate hare in the hr_port parameter.
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports and tau_fast are  no longer optional.
                'n_ports' : number of inputs ports. At least two ports are required.
                'tau_fast' : time constant for the fast low-pass filter
            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                'slope' : Slope of the sigmoidal function.
                'thresh' : Initial threshold of the sigmoidal function
                'tau' : Time constant of the update dynamics.
                'hr_port' : port whose inputs will be used to harmonize the rate
                'tau_thr' : sets the speed of change for the threshold (reciprocal of time constant). 
        Raises:
            ValueError
        """
        if params['n_ports'] < 2:
            raise ValueError('At least two ports whould be used with the st_hr_sig units')
        else:
            self.multiport = True
        unit.__init__(self, ID, params, network)  # the parent's constructor
        self.hr_port = params['hr_port'] # port for rate distribution control
        self.tau_thr = params['tau_thr']  # the reciprocal of the threshold sliding time constant
        self.slope = params['slope']    # slope of the sigmoidal function
        self.thresh = params['thresh']  # horizontal displacement of the sigmoidal
        self.tau = params['tau']  # the time constant of the dynamics
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.slide_thr_hr, syn_reqs.lpf_fast])

    def f(self, arg):
        """ This is the sigmoidal function. Could roughly think of it as an f-I curve. """
        return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_mp_input_sum(t)) - y[0] ) * self.rtau

    def get_mp_input_sum(self, time):
        """ The input function of st_hr_sig unit. """
        weights = self.get_mp_weights(time)
        inps = self.get_mp_inputs(time)
        return sum([np.dot(weights[prt], inps[prt]) for prt in range(self.n_ports)])
     


class synaptic_scaling_harmonic_rate_sigmoidal(unit): 
    """
    The synaptic weights are scaled to produce a harmonic pattern of firing rates.
    
    The weights of all excitatory inputs at the 'hr_port' port will adjust by following a gradient 
    that comes from the discrete Laplacian operator applied to the directed graph where the nodes are 
    the units, and the edges come from their connections.

    In practice this means that ther firing rate of the unit will be adjusted so it approaches the mean 
    firing rate of its inputs. 

    It is expected that one input port will have the inputs to be used for the rate harmonization,
    usually the internal connections, and another port will have the external inputs, which are
    ignored for rate harmonization. Thus, the number of ports should be at least 2, and the port
    whose inputs are used to harmonize the rate hare in the hr_port parameter.
 
    Whether an input is excitatory or inhibitory is decided by the sign of its initial value.
    Synaptic weights initialized to zero will be considered excitatory.
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports and tau_fast are  no longer optional.
                'n_ports' : number of inputs ports. At least two ports are required.
                'tau_fast' : time constant for the fast low-pass filter
            In addition, params should have the following entries.
                REQUIRED PARAMETERS (use of parameters is in flux. Definitions may be incorrect)
                'slope' : Slope of the sigmoidal function.
                'thresh' : Threshold of the sigmoidal function.
                'tau' : Time constant of the update dynamics.
                'tau_scale' : sets the speed of change for the scaling factor
                'Kp' : Gain factor for the scaling of weights (makes changes bigger/smaller).
                'hr_port' : port whose inputs will be used to harmonize the rate

        Raises:
            ValueError

        The actual values used for scaling are calculated in unit.upd_syn_scale_hr(), which updates
        the array self.scale_facs .
        """
        if params['n_ports'] < 2:
            raise ValueError('At least two ports whould be used with the ss_hr_sig units')
        else:
            self.multiport = True
        unit.__init__(self, ID, params, network)
        self.multiport = True  # This causes the port_idx list to be created in init_pre_syn_update
        self.slope = params['slope']    # slope of the sigmoidal function
        self.thresh = params['thresh']  # horizontal displacement of the sigmoidal
        self.tau = params['tau']  # the time constant of the dynamics
        self.tau_scale = params['tau_scale']  # the scaling time constant
        self.Kp = params['Kp']  # gain for synaptic scaling
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        self.hr_port = params['hr_port'] # port for rate distribution control
        
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.syn_scale_hr, syn_reqs.lpf_fast])
        
    def f(self, arg):
        """ This is the sigmoidal function. Could roughly think of it as an f-I curve. """
        return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_mp_input_sum(t)) - y[0] ) * self.rtau

    def get_mp_input_sum(self, time):
        """ The input function of the ss_hr_sig unit. """
        # TODO: calculate with the approach of sig_ssrdc_sharp
        weights = self.get_mp_weights(time)
        inps = self.get_mp_inputs(time)
        acc_sum = 0.
        for port in range(self.n_ports):
            if port == self.hr_port:
                acc_sum += sum( [sc * w * i for sc, w, i in zip(self.scale_facs_hr, weights[port], inps[port]) ] ) 
            else:
                acc_sum += np.dot(weights[port], inps[port]) 
        return acc_sum 


class ds_ssrdc_sharp(double_sigma_base, ssrdc_sharp_base): 
    """
    Double-sigma unit with switchable synaptic-scaling rate distribution control.

    Each input belongs to a particular "branch". All inputs from the same branch add 
    linearly, and the sum is fed into a sigmoidal function that produces the output of the branch. 
    The output of all the branches is added linearly to produce the total input to the unit, 
    which is fed into a sigmoidal function to produce the output of the unit.

    One of the parameters is a port number, called "sharpen_port". When the scaled sum of inputs to the 
    sharpen port are larger than 0.5, based on the distribution of inputs at port 'rdc_port' this unit 
    will use synaptic-scaling based rate distribution control to produce an exponential distribution of firing 
    rates. When the inputs to the sharpen port are smaller than 0.5 the scale factors will decay exponentially
    to the default value 1, with a rate set by the "tau_relax" parameter.

    The inputs at the sharpen port will not contribute to the activation of the unit. 

    The equations of the doulble-sigma unit can be seen in the "double_sigma_unit" tiddler of 
    the programming notes wiki.

    """
    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports and tau_fast are no longer optional.
                n_ports: number of inputs ports. Needs a valuer > 1
                tau_fast: Time constant for the fast low-pass filter.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                slope: Slope of the sigmoidal function.
                thresh: Threshold of the sigmoidal function.
                tau: Time constant of the update dynamics.
                tau_scale: sets the speed of change for the scaling factor
                tau_relax: controls how fast the scaling factors return to 1 when rdc is off
                c: Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
                Kp' : Gain factor for the scaling of weights (makes changes bigger/smaller).
                rdc_port: port ID of inputs used for rate distribution control.
                sharpen_port: port ID where the inputs controlling rdc arrive.
                branch_params: A dictionary with the following 3 entries:
                    branch_w: The "weights" for all branches. This is a list whose length is the number
                            of branches. Each entry is a positive number, and all entries must add to 1.
                            The input port corresponding to a branch is the index of its corresponding 
                            weight in this list, so len(branch_w) = n_ports.
                    slopes: Slopes of the branch sigmoidal functions. It can either be a float value
                            (resulting in all values being the same), it can be a list of length n_ports
                            specifying the slope for each branch, or it can be a dictionary specifying a
                            distribution. Currently only the uniform distribution is supported:
                            {..., 'slopes':{'distribution':'uniform', 'low':0.5, 'high':1.5}, ...}
                    threshs: Thresholds of the branch sigmoidal functions. It can either be a float
                            value (resulting in all values being the same), it can be a list of length 
                            n_ports specifying the threshold for each branch, or it can be a dictionary 
                            specifying a distribution. Currently only the uniform distribution is supported:
                            {..., 'threshs':{'distribution':'uniform', 'low':-0.1, 'high':0.5}, ...}
                phi: This value is substracted from the branches' contribution. In this manner branches with
                     zero input will contribute (0.5 - phi) times the branch weight.

        """
        self.extra_ports = 1 # The sharpen port is an extra port for double_sigma_base
        double_sigma_base.__init__(self, ID, params, network)
        self.unit_initialized = True  # to avoid calling the unit constructor twice
        ssrdc_sharp_base.__init__(self, ID, params, network)
        self.nm_prts = list(range(self.n_ports)) # port list without the rdc and sharpen ports
        del self.nm_prts[max(self.rdc_port,self.sharpen_port)]
        del self.nm_prts[min(self.rdc_port,self.sharpen_port)]
        #self.nm_prts = np.array(self.nm_prts, dtype='uint32')
        
    def get_mp_input_sum(self, time):
        """ The input function of the ds_ssrdc_sharp unit. """
        ws = self.get_mp_weights(time)
        inps = self.get_mp_inputs(time)
        rdcp = self.rdc_port # to make lines shorter
        acc_sum = self.br_w[rdcp] * ( self.f( self.threshs[rdcp], self.slopes[rdcp], 
                                            np.sum(self.scale_facs_rdc * ws[rdcp] * inps[rdcp]) -self.phi ) )
        acc_sum += sum( [ self.br_w[p] * ( self.f( self.threshs[p], self.slopes[p], np.dot(ws[p],inps[p]) )
                                           - self.phi ) for p in self.nm_prts ] )
        return acc_sum 


class delta_linear(unit):
    """ A linear unit that modifies its weights using a continuous version of the delta rule.

    This unit scales its input vector so it has a unit L2 norm. Moreover, there is an implicit bias
    input that is always 1, and whose weight starts at 0 and changes with rate 'bias_lrate'.

    Units in this class have 3 input ports. 
    Port 0 is for the plastic synapses, which are of the delta_synapse class.
    Port 1 is for the desired value signal, which is used to produce the error.
    Port 2 is for the signal that indicates when to update the synaptic weights.

    The delta_linear units have a variable named 'learning', which decays to zero exponentially
    with a time constant given as a paramter. If 'e' stands for learning, e' = -tau_e * e .

    When learning > 0.5: 
        * The signal at port 2 is ignored. 
        * The error transmitted to the synapses is the the desired output (e.g. the signal at
          port 1) minus the current activity. 
        * Eligibility decays to 0 exponentially.
    When learning <= 0.5:
        * The error is 0 (which cancels plasticity at the synapses).
        * If the scaled sum of inputs at port 2 is smaller than 0.5, learning decays exponentially, 
          but if the input is larger than 0.5 then learning is set to the value 1.

    See unit.upd_error for details.
    After being set to 1, learning will take T = log(2)/tau_e time units to decay back to 0.5.
    During this time any non-zero error signal will cause plasticity at the synapses.
    """
    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports and tau_fast are no longer optional.
                n_ports: number of inputs ports. IT MUST BE SET TO 3.
                tau_fast: Time constant for the fast low-pass filter.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                gain: slope of the linear unit.
                tau: time constant of the update dynamics.
                tau_e: time constant for the decay of learning.
                bias_lrate: learning rate of the bias. Requires small values.
        Raises:
            ValueError
        """
        unit.__init__(self, ID, params, network)  # The parent class' constructor
        if self.n_ports != 3:
            raise ValueError("delta_linear units require 3 input ports.")
        self.gain = params['gain'] # a scalar that multiplies the output of the unit
        self.tau = params['tau'] # time constant for the unit's dynamics
        self.tau_e = params['tau_e'] # time constant for the decay of the learning
        self.bias_lrate = params['bias_lrate'] # learning rate of the bias
        self.bias = 0. # initial value of the bias input

        self.syn_needs.update([syn_reqs.lpf_fast, syn_reqs.error, syn_reqs.mp_inputs, syn_reqs.inp_l2])
        
    def get_mp_input_sum(self,time):
        """ The input function of the delta_linear unit. """
        
        return  sum( [syn.w * act(time - dely) for syn, act, dely in zip(
                     [self.net.syns[self.ID][i] for i in self.port_idx[0]],
                     [self.net.act[self.ID][i] for i in self.port_idx[0]],
                     [self.net.delays[self.ID][i] for i in self.port_idx[0]])] ) / self.inp_l2
        #return np.dot(self.get_mp_inputs(time)[0], self.get_mp_weights(time)[0]) / self.inp_l2
        
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        return ( self.gain * self.get_mp_input_sum(t) + self.bias  - y[0] ) / self.tau



class sds_n_ssrdc_sharp(double_sigma_base, ssrdc_sharp_base): 
    """
    Sigma double-sigma unit normalized branch inputs and switchable synaptic-scaling rate distribution control.

    Double sigma units are inspired by:
    Poirazi et al. 2003 "Pyramidal Neuron as Two-Layer Neural Network" Neuron 37,6:989-999

    There are two types of inputs. Somatic inputs arrive at port 0 (the default port), and are
    just like inputs to the sigmoidal units. Dendritic inputs arrive at ports with with ID 
    larger than 0, and are just like inputs to double_sigma units.

    Each dendritic input belongs to a particular "branch". All inputs from the same branch
    add linearly, are normalized, and the normalized sum is fed into a sigmoidal function 
    that produces the output of the branch. The output of all the branches is added linearly 
    to produce the total input to the unit, which is fed into a sigmoidal function to produce 
    the output of the unit.

    One of the parameters is a port number, called "sharpen_port". When the scaled sum of inputs to the 
    sharpen port are larger than 0.5, based on the distribution of inputs at port 'rdc_port' this unit 
    will use synaptic-scaling based rate distribution control to produce an exponential distribution of firing 
    rates. When the inputs to the sharpen port are smaller than 0.5 the scale factors will decay exponentially
    to the default value 1, with a rate set by the "tau_relax" parameter.

    In this version the inputs at the rdc_port are not normalized, since this interacts with the scaling.
    Also, notice that when the rdc_port is not 0 (e.g. at the soma), the ability to maintain the
    rate distribution will be compromised, although scaling should still bias things in the right direction.

    The inputs at the sharpen port will not contribute to the activation of the unit. 

    """
    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            n_ports and tau_fast are no longer optional.
                n_ports: number of inputs ports. Needs a valuer > 1
                tau_fast: Time constant for the fast low-pass filter.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS 
                slope: Slope of the sigmoidal function.
                thresh: Threshold of the sigmoidal function.
                tau: Time constant of the update dynamics.
                tau_scale: sets the speed of change for the scaling factor
                tau_relax: controls how fast the scaling factors return to 1 when rdc is off
                c: Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
                Kp' : Gain factor for the scaling of weights (makes changes bigger/smaller).
                rdc_port: port ID of inputs used for rate distribution control.
                sharpen_port: port ID where the inputs controlling rdc arrive.
                branch_params: A dictionary with the following 3 entries:
                    branch_w: The "weights" for all branches. This is a list whose length is the number
                            of branches. Each entry is a positive number, and all entries must add to 1.
                            The input port corresponding to a branch is the index of its corresponding 
                            weight in this list plus one. For example, if branch_w=[.2, .8], then
                            input port 0 is at the soma, input port 1 has weight .2, port 2 weight .8, and
                            port 3 is the sharpening port.
                    slopes: Slopes of the branch sigmoidal functions. It can either be a float value
                            (resulting in all values being the same), it can be a list of length n_ports
                            specifying the slope for each branch, or it can be a dictionary specifying a
                            distribution. Currently only the uniform distribution is supported:
                            {..., 'slopes':{'distribution':'uniform', 'low':0.5, 'high':1.5}, ...}
                    threshs: Thresholds of the branch sigmoidal functions. It can either be a float
                            value (resulting in all values being the same), it can be a list of length 
                            n_ports specifying the threshold for each branch, or it can be a dictionary 
                            specifying a distribution. Currently only the uniform distribution is supported:
                            {..., 'threshs':{'distribution':'uniform', 'low':-0.1, 'high':0.5}, ...}
                phi: This value is substracted from the branches' contribution. In this manner, a branch
                        with no inputs may not contribute to the unit's activation.

        """
        self.extra_ports = 2 # The soma and the sharpen port 
        double_sigma_base.__init__(self, ID, params, network)
        self.unit_initialized = True  # to avoid calling the unit constructor twice
        ssrdc_sharp_base.__init__(self, ID, params, network)
        self.syn_needs.update([syn_reqs.lpf_slow_mp_inp_sum])

        self.nm_prts = list(range(self.n_ports)) # port list without the 0, rdc, and sharpen ports
        if self.rdc_port != 0:
            del self.nm_prts[max(self.rdc_port,self.sharpen_port)]
            del self.nm_prts[min(self.rdc_port,self.sharpen_port)]
            self.nm_prts = self.nm_prts[1:] # removes port 0
        else:
            del self.nm_prts[self.sharpen_port]
            del self.nm_prts[0]
        #self.nm_prts = np.array(self.nm_prts, dtype='uint32')
        
    def get_mp_input_sum(self, time):
        """ The input function of the sds_ssrdc_sharp unit. """
        ws = self.get_mp_weights(time)
        inps = self.get_mp_inputs(time)
        rdcp = self.rdc_port # to make lines shorter
        # Removing zero or near-zero values from lpf_slow_mp_inp_sum
        lpf_i = [np.sign(arry)*(np.maximum(np.abs(arry), 1e-3)) for arry in self.lpf_slow_mp_inp_sum]
        acc_sum = 0
        # Adding input from soma
        if rdcp != 0:
            acc_sum += (np.dot(ws[0], inps[0]) - lpf_i[0]) / lpf_i[0]
        # Adding input from rdc port (not normalized)
        acc_sum += self.br_w[rdcp] * ( self.f( self.threshs[rdcp], self.slopes[rdcp], 
                   np.sum(self.scale_facs_rdc * ws[rdcp] * inps[rdcp] ) - self.phi ) )
        # Adding inputs from the other ports
        acc_sum += sum( [ self.br_w[p-1] * ( self.f( self.threshs[p-1], self.slopes[p-1], 
                          (np.dot(ws[p],inps[p])-lpf_i[p])/lpf_i[p] - self.phi ) ) for p in self.nm_prts ] )
        return acc_sum 


