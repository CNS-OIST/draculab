"""
units.py
This file contains the basic unit models in the draculab simulator.
"""

from draculab import unit_types, synapse_types, syn_reqs  # names of models and requirements
from synapses import *  # synapse models
from requirements import *
import numpy as np
from cython_utils import * # interpolation and integration methods including cython_get_act*,
                           # cython_sig, euler_int, euler_maruyama, exp_euler
from scipy.integrate import odeint # to integrate ODEs
from array import array # optionally used for the unit's buffer
#from scipy.integrate import solve_ivp # to integrate ODEs
#from scipy.interpolate import interp1d # to interpolate values


class unit():
    """ The parent class of all unit models.  """

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
        # Copying parameters from dictionary into unit's attributes
        self.type = params['type'] # an enum identifying the type of unit being instantiated
        self.net = network # the network where the unit lives
        self.rtol = self.net.rtol # local copies of the rtol and atol tolerances
        self.atol = self.net.atol
        self.init_val = params['init_val'] # initial value for the activation (for units that use buffers)
        self.min_buff_size = network.min_buff_size # a local copy just to avoid the extra reference
        # The delay of a unit is the maximum delay among the projections it sends.
        # The final value of 'delay' should be set by network.connect(), after the unit is created.
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
        self.last_time = 0.  # time of last call to the update function
                            # Used by the upd_lpf_X functions
        self.init_buffers() # This will create the buffers that store states and times
        #self.pre_syn_update = lambda time : None # See init_pre_syn_update below
        # functions will have all the functions invoked by pre_syn_update
        self.functions = set()

     
    def init_buffers(self):
        """
        This method (re)initializes the buffer variables according to the current parameters.

        It is useful because new connections may increase self.delay, and thus the size of the buffers.

        Raises:
            AssertionError.
        """
        assert self.net.sim_time == 0., 'Buffers are being reset when the simulation time is not zero'

        min_del = self.net.min_delay  # just to have shorter lines below
        self.steps = int(round(self.delay/min_del)) # delay, in units of the minimum delay
        self.bf_type = np.dtype('d') # data type of the buffer when it is a numpy array

        # The following buffers are for synaptic requirements.
        # They store one value per update, in the requirement's update method.
        if syn_reqs.lpf_fast in self.syn_needs:
            self.lpf_fast.init_buff()
        if syn_reqs.lpf_mid in self.syn_needs:
            self.lpf_mid_buff = np.array( [self.init_val]*self.steps, dtype=self.bf_type)
        if syn_reqs.lpf_slow in self.syn_needs:
            self.lpf_slow_buff = np.array( [self.init_val]*self.steps, dtype=self.bf_type)
        if syn_reqs.lpf_mid_inp_sum in self.syn_needs:
            self.lpf_mid_inp_sum_buff = np.array( [self.init_val]*self.steps, dtype=self.bf_type)

        # 'source' units don't use activity buffers, so for them the method ends here
        if self.type == unit_types.source:
            return
        
        min_buff = self.min_buff_size
        self.offset = (self.steps-1)*min_buff # an index used in the update function of derived classes
        self.buff_size = int(round(self.steps*min_buff)) # number of activation values to store
        # currently testing 3 ways of initializing the buffer:
        #self.buffer = np.ascontiguousarray( [self.init_val]*self.buff_size, dtype=self.bf_type) # numpy array with
                                                                             # previous activation values
        #self.buffer = np.array( [self.init_val]*self.buff_size, dtype=self.bf_type) # numpy array with
        self.buffer = array('d', [self.init_val]*self.buff_size)
        self.times = np.linspace(-self.delay, 0., self.buff_size, dtype=self.bf_type) # the corresponding
                                                                             # times #for the buffer values
        self.times_grid = np.linspace(0, min_del, min_buff+1, dtype=self.bf_type) # used to create
                                                                             # values for 'times'
        self.time_bit = self.times[1] - self.times[0] + 1e-10 # time interval used by get_act.
        
        
    def get_inputs(self, time):
        """
        Returns a list with the inputs received by the unit from all other units at time 'time'.

        The time argument should be within the range of values stored in the unit's buffer.

        The returned inputs already account for the transmission delays.
        To do this: in the network's act list the entry corresponding to the unit's ID
        (e.g. self.net.act[self.ID]) is a list; for each i-th entry (a function) retrieve
        the value at time "time - delays[ID][i]".


        This function ignores input ports.
        """
        return [ fun(time - dely) for dely,fun in zip(self.net.delays[self.ID], self.net.act[self.ID]) ]


    def get_input_sum(self,time):
        """ Returns the sum of all inputs at the given time, each scaled by its synaptic weight.
        
        The time argument should be within the range of values stored in the unit's buffer.
        The sum accounts for transmission delays. Input ports are ignored.
        """
        # original implementation is below
        return sum([ syn.w * fun(time-dely) for syn,fun,dely in zip(self.net.syns[self.ID],
                        self.net.act[self.ID], self.net.delays[self.ID]) ])
        # second implementation is below
        #return sum( map( lambda x: (x[0].w) * (x[1](time-x[2])),
        #            zip(self.net.syns[self.ID], self.net.act[self.ID], self.net.delays[self.ID]) ) )


    def get_mp_inputs(self, time):
        """
        Returns a list with all the inputs, arranged by input port.

        This method is for units where multiport = True, and that have a port_idx attribute.

        The i-th element of the returned list is a numpy array containing the raw (not multiplied
        by the synaptic weight) inputs at port i. The inputs include transmision delays.

        The time argument should be within the range of values stored in the unit's buffer.
        """
        return [ np.array([self.net.act[self.ID][idx](time - self.net.delays[self.ID][idx]) for idx in idx_list])
                 for idx_list in self.port_idx ]


    def get_sc_input_sum(self, time):
        """
        Returns the sum of inputs, each scaled by its synaptic weight and by a gain constant.
        
        The sum accounts for transmission delays. Input ports are ignored.
        The time argument should be within the range of values stored in the unit's buffer.

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
        The time argument should be within the range of values stored in the unit's buffer.

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

        # Maybe there should be a single 'times' array in the network, to avoid those rolls,
        # but they add very little to the simualation times
        #assert (self.times[-1]-time) < 2e-6, 'unit' + str(self.ID) + ': update time is desynchronized'
        new_times = self.times[-1] + self.times_grid
        self.times = np.roll(self.times, -self.min_buff_size)
        self.times[self.offset:] = new_times[1:]
        
        # TO USE A PARTICULAR SOLVER, UNCOMMENT ITS CODE, COMMENT OTHER SOLVERS
        #---------------------------------------------------------------------
        # odeint
        #"""
        # odeint also returns the initial condition, so to produce min_buff_size new values
        # we need to provide min_buff_size+1 desired times, starting with the one for the initial condition
        new_buff = odeint(self.derivatives, [self.buffer[-1]], new_times, rtol=self.rtol, atol=self.atol)
        self.buffer = np.roll(self.buffer, -self.min_buff_size)
        self.buffer[self.offset:] = new_buff[1:,0]
        #"""
        #---------------------------------------------------------------------
        # solve_ivp --- REQUIRES ADDITIONAL STEPS (see below)
        """
        # This is a reimplementation with solve_ivp. To make it work you need to change the order of the
        # arguments in all the derivatives functions, from derivatives(self, y, t) to derivatives(self, t, y).
        # In vi it takes one command: :%s/derivatives(self, y, t)/derivatives(self, t, y)/
        # Also, make sure that the import command at the top is uncommented for solve_ivp.
        # One more thing: to use the stiff solvers the derivatives must return a list or array, so
        # the returned value must be enclosed in square brackets in <unit_type>.derivatives.
        solution = solve_ivp(self.derivatives, (new_times[0], new_times[-1]), [self.buffer[-1]], method='LSODA',
                                                t_eval=new_times, rtol=self.rtol, atol=self.atol)
        self.buffer = np.roll(self.buffer, -self.min_buff_size)
        self.buffer[self.offset:] = solution.y[0,1:]
        """
        #---------------------------------------------------------------------
        # Euler
        """
        # This is an implementation with forward Euler integration. Although this may be
        # imprecise and unstable in some cases, it is a first step to implement the
        # Euler-Maruyama method. And it's also faster than odeint.
        # Notice the atol and rtol values are not used in this case.
        dt = new_times[1] - new_times[0]
        # euler_int is defined in cython_utils.pyx
        new_buff = euler_int(self.derivatives, self.buffer[-1], time, len(new_times), dt)
        self.buffer = np.roll(self.buffer, -self.min_buff_size)
        self.buffer[self.offset:] = new_buff[1:]
        """
        #---------------------------------------------------------------------
        # Euler-Maruyama
        """
        # This is the Euler-Maruyama implementation. Basically the same as forward Euler.
        # The atol and rtol values are meaningless in this case.
        # This solver does the buuffer's "rolling" by itself.
        # The unit needs to have 'mu' and 'sigma' attributes.
        ## self.mu = 0. # Mean of the white noise
        ## self.sigma = 0.0 # standard deviation of Wiener process.
        dt = new_times[1] - new_times[0]
        # euler_maruyama_ is defined in cython_utils.pyx
        euler_maruyama(self.derivatives, self.buffer, time,
                        self.buff_size-self.min_buff_size, dt, self.mu, self.sigma)
        """
        #---------------------------------------------------------------------

        self.pre_syn_update(time) # Update any variables needed for the synapse to update.
                                  # It is important this is done after the buffer has been updated.
        # For each synapse on the unit, update its state
        for pre in self.net.syns[self.ID]:
            pre.update(time)

        self.last_time = time # last_time is used to update some pre_syn_update values
        #"""
        # EXPERIMENTAL AND CURRENTLY INEFFICIENT CYTHON VERSION:
        #cython_update(self, time)


    def get_act(self,time):
        """ Gives you the activity at a previous time 't' (within buffer range).

        This version works for units that store their previous activity values in a buffer.
        Units without buffers (e.g. source units) have their own get_act function.

        This is the most time-consuming method in draculab (thus the various optimizations).
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Below is the more general (but slow) interpolation using interp1d
        # Make sure to import interp1d at the top of the file before using this.
        # Sometimes the ode solver asks about values slightly out of bounds, so I set this to extrapolate
        """
        return interp1d(self.times, self.buffer, kind='linear', bounds_error=False, copy=False,
                        fill_value="extrapolate", assume_sorted=True)(time)
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Below the code for the second implementation.
        # This linear interpolation takes advantage of the ordered, regularly-spaced buffer.
        # Time values outside the buffer range receive the buffer endpoints.
        # The third implementation is faster, but for small buffer sizes this gives more exact results.
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
        # This is the second implementation, written in Cython
        #return cython_get_act2(time, self.times[0], self.times[-1], self.times, self.buff_size, self.buffer)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This is the third implementation, written in Cython
        return cython_get_act3(time, self.times[0], self.time_bit, self.buff_size, self.buffer)

  
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
        is tasked with updating all the unit variables used by the synapses to update, or
        for other purposes of the unit's model.
        pre_syn_update() is called once per simulation step (i.e. every min_delay period).

        init_pre_syn_update creates the 'functions' list, which contains all the functions
        invoked by pre_syn_update, and initializes all the variables it requires. 
        To do this, it compiles the requirements of all the unit's synapses in the set 
        self.syn_needs, and creates an instance of each requirement, from the classes in
        the requirements.py file.

        Each requirement is a class whose __init__ and update methods initialize the necessary
        variables, and update them, respectively.  The __init__ method of the requirement also
        puts its 'update' methods in the unit's 'functions' list.

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

        # Create instances of all the requirements to be updated by
        # the pre_syn_update function. 
        """
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444

        # eventually this section should be reduced to:

        for req in self.syn_needs:
            if issubclass(type(req), syn_reqs):
                # You create an instance of the requirement. 
                # Its name is the requirement's name.
                setattr(self. req.name, eval(req.name+'(self)'))
            else:  
                raise NotImplementedError('Asking for a requirement that is not implemented')

        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444
        """

        for req in self.syn_needs:
            if req is syn_reqs.lpf_fast:  # <----------------------------------
                if issubclass(type(req), syn_reqs):
                    name = req.name # name of the requirement as a string
                    setattr(self, name, eval(name+'(self)')) # create an instance with that name
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
            elif req is syn_reqs.exp_scale_sort_mp:  # <----------------------------------
                if not self.multiport:
                    raise AssertionError('The exp_scale_sort_mp requirement is for multiport units only')
                self.scale_facs_rdc = np.tile(1., len(self.port_idx[self.rdc_port])) # array with scale factors
                # exc_idx_rdc = numpy array with index of all excitatory units at rdc port
                self.exc_idx_rdc = [ idx for idx,syn in enumerate([self.net.syns[self.ID][i] 
                                     for i in self.port_idx[self.rdc_port]]) if syn.w >= 0 ]
                self.autapse = False
                for idx, syn in enumerate([self.net.syns[self.ID][eir] for eir in self.exc_idx_rdc]):
                    if syn.preID == self.ID: # if there is an excitatory autapse at the rdc port
                        self.autapse_idx = idx
                        self.autapse = True
                        break
                # Gotta produce the array of ideal rates given the truncated exponential distribution
                if self.autapse:
                    N = len(self.exc_idx_rdc)
                else:
                    N = len(self.exc_idx_rdc) + 1 # an extra point for the fake autapse
                points = [ (k + 0.5) / (N + 1.) for k in range(N) ]
                self.ideal_rates = np.array([-(1./self.c) * np.log(1.-(1.-np.exp(-self.c))*pt) for pt in points])
                self.functions.add(self.upd_exp_scale_sort_mp)
            elif req is syn_reqs.exp_scale_sort_shrp:  # <----------------------------------
                if not self.multiport:
                    raise AssertionError('The exp_scale_sort_shrp requirement is for multiport units only')
                self.scale_facs_rdc = np.tile(1., len(self.port_idx[self.rdc_port])) # array with scale factors
                # exc_idx_rdc = numpy array with index of all excitatory units at rdc port
                self.exc_idx_rdc = [ idx for idx,syn in enumerate([self.net.syns[self.ID][i] 
                                     for i in self.port_idx[self.rdc_port]]) if syn.w >= 0 ]
                self.autapse = False
                for idx, syn in enumerate([self.net.syns[self.ID][eir] for eir in self.exc_idx_rdc]):
                    if syn.preID == self.ID: # if there is an excitatory autapse at the rdc port
                        self.autapse_idx = idx
                        self.autapse = True
                        break
                # Gotta produce the array of ideal rates given the truncated exponential distribution
                if self.autapse:
                    N = len(self.exc_idx_rdc)
                else:
                    N = len(self.exc_idx_rdc) + 1 # an extra point for the fake autapse
                points = [ (k + 0.5) / (N + 1.) for k in range(N) ]
                self.ideal_rates = np.array([-(1./self.c) * np.log(1.-(1.-np.exp(-self.c))*pt) for pt in points])
                self.functions.add(self.upd_exp_scale_sort_shrp)
            else:  # <----------------------------------------------------------------------
                raise NotImplementedError('Asking for a requirement that is not implemented')


    def pre_syn_update(self, time):
        """ Call the update functions for the requirements added in init_pre_syn_update. """
        for f in self.functions:
            f(time)
        

    def get_lpf_fast(self, steps):
        """ Get the fast low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_fast.get(steps)


    def upd_lpf_mid(self,time):
        """ Update the lpf_mid variable. """
        # Source units have their own implementation
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_mid updated backwards in time']
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
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_slow updated backwards in time']
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
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' sq_lpf_slow updated backwards in time']
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
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' inp_avg updated backwards in time']
        self.inp_avg = sum([u.get_lpf_fast(s) for u,s in self.snorm_list_dels]) / self.n_hebbsnorm
        

    def upd_pos_inp_avg(self, time):
        """ Update the average of the inputs with hebbsnorm synapses. 
        
            The actual value being averaged is lpf_fast of the presynaptic units.
            Inputs whose synaptic weight is zero or negative  are saturated to zero and
            excluded from the computation.
        """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' pos_inp_avg updated backwards in time']
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
        """ Updates the synaptic scaling factors used in multiport ssrdc units.

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

    
    def upd_exp_scale_sort_mp(self, time):
        """ Updates the synaptic scale factor optionally used in some multiport ssrdc units.

            This method implements the exp_scale_sort_mp requirement. It uses the 'ideal_rates' 
            array produced in init_pre_syn_update. 
        """
        # The equations come from the APCTP notebook, 8/28/18.
        exc_rdc_inp = self.mp_inputs[self.rdc_port][self.exc_idx_rdc] # Exc. inputs at the rdc port
        exc_rdc_w = self.get_mp_weights(time)[self.rdc_port][self.exc_idx_rdc] # Exc. weights at rdc port
        r = self.buffer[-1] # current rate
        if not self.autapse: # if unit has no autapses, put a fake one with zero weight
            exc_rdc_inp = np.concatenate((exc_rdc_inp, [r]))
            exc_rdc_w = np.concatenate((exc_rdc_w, [0]))
            self.autapse_idx = len(exc_rdc_inp) - 1
        rate_rank = np.argsort(exc_rdc_inp)  # array index that sorts exc_rdc_inp
        ideal_exc_inp = np.dot(exc_rdc_w[rate_rank], self.ideal_rates)
        my_ideal_rate = self.ideal_rates[np.where(rate_rank == self.autapse_idx)[0][0]]
        if self.autapse:
            u = np.sum(self.scale_facs_rdc[self.exc_idx_rdc] * exc_rdc_inp * exc_rdc_w)
        else:
            u = np.sum(self.scale_facs_rdc[self.exc_idx_rdc]*exc_rdc_inp[:-1]*exc_rdc_w[:-1])
        I = u - self.get_mp_input_sum(time)     
        syn_scale = (my_ideal_rate - I) / (ideal_exc_inp)
        # same weight bounding as upd_exp_scale_mp
        syn_scale = min(syn_scale, 10.) # hard_bound_above
        x0 = self.scale_facs_rdc[self.exc_idx_rdc][0] # x0 is scalar cuz all Exc. factors are equal
        t = self.net.min_delay
        x = (x0 * syn_scale) / (x0 + (syn_scale - x0) * np.exp(-self.tau_scale * syn_scale * t) )
        self.scale_facs_rdc[self.exc_idx_rdc] = x


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


    def upd_exp_scale_sort_shrp(self, time):
        """ Updates the synaptic scale factor optionally used in some ssrdc_sharp units.

            The algorithm is the same as upd_exp_scale_sort_mp, but controlled by the inputs at the
            sharpening port. When the sum of inputs at this port is smaller than 0.5 the scale factors
            return to 1 with a time constant of tau_relax.
        """
        if np.dot(self.mp_inputs[self.sharpen_port], self.get_mp_weights(time)[self.sharpen_port]) < 0.5:
            syn_scale = 1.
            exp_tau = self.tau_relax
        else: # input at sharpen port >= 0.5
            exc_rdc_inp = self.mp_inputs[self.rdc_port][self.exc_idx_rdc] # Exc. inputs at the rdc port
            exc_rdc_w = self.get_mp_weights(time)[self.rdc_port][self.exc_idx_rdc] # Exc. weights at rdc port
            r = self.buffer[-1] # current rate
            if not self.autapse: # if unit has no autapses, put a fake one with zero weight
                exc_rdc_inp = np.concatenate((exc_rdc_inp, [r]))
                exc_rdc_w = np.concatenate((exc_rdc_w, [0]))
                self.autapse_idx = len(exc_rdc_inp) - 1
            rate_rank = np.argsort(exc_rdc_inp)  # array index that sorts exc_rdc_inp
            ideal_exc_inp = np.dot(exc_rdc_w[rate_rank], self.ideal_rates)
            my_ideal_rate = self.ideal_rates[np.where(rate_rank == self.autapse_idx)[0][0]]
            if self.autapse:
                u = np.sum(self.scale_facs_rdc[self.exc_idx_rdc] * exc_rdc_inp * exc_rdc_w)
            else:
                u = np.sum(self.scale_facs_rdc[self.exc_idx_rdc]*exc_rdc_inp[:-1]*exc_rdc_w[:-1])
            I = u - self.get_mp_input_sum(time)     
            syn_scale = (my_ideal_rate - I) / (ideal_exc_inp)
            syn_scale = min(syn_scale, 10.) # hard_bound_above
            exp_tau = self.tau_scale
        # same soft weight bounding as upd_exp_scale_mp
        x0 = self.scale_facs_rdc[self.exc_idx_rdc][0] # x0 is scalar cuz all Exc. factors are equal
        x = (x0 * syn_scale) / (x0 + (syn_scale - x0) * np.exp(-exp_tau * syn_scale * self.net.min_delay) )
        self.scale_facs_rdc[self.exc_idx_rdc] = x
 
        
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
    beween zero and one. The sigmoidal unit has the equation:
    f(x) = 1. / (1. + exp(-slope*(x - thresh)))

    Because this unit operates in real time, it updates its value gradualy, with
    a 'tau' time constant. The dynamics of the unit are:
    tau * u' = f(inp) - u,
    where 'inp' is the sum of inputs, and 'u' is the firing rate.
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
        #return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
        return cython_sig(self.thresh, self.slope, arg)
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_input_sum(t)) - y[0] ) * self.rtau
    

class noisy_sigmoidal(unit): 
    """
        An implementation of a sigmoidal unit with intrinsic noise. 
    
        Its output is produced by linearly suming the inputs (times the synaptic weights), 
        and feeding the sum to a sigmoidal function, which constraints the output to values 
        beween zero and one. The sigmoidal unit has the equation:
        f(x) = 1. / (1. + exp(-slope*(x - thresh)))
    
        Because this unit operates in real time, it updates its value gradualy, with
        a 'tau' time constant. When there is no noise the dynamics of the unit are:
        tau * u' = f(inp) - lambda * u,
        where 'inp' is the sum of inputs, 'u' is the firing rate, and lambda is a decay rate.

        The update function of this unit uses a stochastic solver that adds Gaussian
        white noise to the unit's derivative. When lambda>0, the solver used is
        exponential Euler. When lambda=0, the solver is Euler-Maruyama.
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
                'lambda': decay factor
                'mu': mean of the white noise
                'sigma': standard deviation of the white noise process
        """
        unit.__init__(self, ID, params, network)
        self.slope = params['slope']    # slope of the sigmoidal function
        self.thresh = params['thresh']  # horizontal displacement of the sigmoidal
        self.tau = params['tau']  # the time constant of the dynamics
        self.lambd = params['lambda'] # decay rate (can't be called 'lambda')
        self.sigma = params['sigma'] # std. dev. of white noise
        self.mu = params['mu'] # mean of the white noise
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        # if lambd > 0 we'll use the exponential Euler integration method
        if self.lambd > 0.:
            self.diff = lambda y, t: self.get_input_sum(t) * self.rtau
            dt = self.times[1] - self.times[0]
            A = -self.lambd*self.rtau
            self.eAt = np.exp(A*dt)
            self.c2 = (self.eAt-1.)/A
            self.c3 = np.sqrt( (self.eAt**2. - 1.) / (2.*A) )

    def f(self, arg):
        """ This is the sigmoidal function. Could roughly think of it as an f-I curve. """
        return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
        #return cython_sig(self.thresh, self.slope, arg)
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_input_sum(t)) - self.lambd*y[0] ) * self.rtau

    def update(self,time):
        """
        Advance the dynamics from time to time+min_delay.

        This update function overrides the one in unit.update in order to use 
        a stochastic solver. 
        """
        assert abs(self.times[-1]-time) < 1e-6, "Time out of phase in unit's update function"
        new_times = self.times[-1] + self.times_grid
        self.times = np.roll(self.times, -self.min_buff_size) 
        self.times[self.offset:] = new_times[1:] 
        # This is the Euler-Maruyama implementation. Basically the same as forward Euler.
        # The atol and rtol values are meaningless in this case.
        dt = new_times[1] - new_times[0]
        # The integration functions are defined in cython_utils.pyx
        if self.lambd == 0.:
            # needs no buffer roll or buffer assignment
            euler_maruyama(self.derivatives, self.buffer, time, 
                                    self.buff_size-self.min_buff_size, dt, self.mu, self.sigma)
        else:
            new_buff = exp_euler(self.diff, self.buffer[-1], time, len(new_times),
                                     dt, self.mu, self.sigma, self.eAt, self.c2, self.c3)
            new_buff = np.maximum(new_buff, 0.)
            self.buffer = np.roll(self.buffer, -self.min_buff_size)
            self.buffer[self.offset:] = new_buff[1:] 
        self.pre_syn_update(time) # Update any variables needed for the synapse to update.
                                  # It is important this is done after the buffer has been updated.
        # For each synapse on the unit, update its state
        for pre in self.net.syns[self.ID]:
            pre.update(time)
        self.last_time = time # last_time is used to update some pre_syn_update values


class linear(unit): 
    """ An implementation of a linear unit.

        The output approaches the sum of the inputs multiplied by their synaptic weights,
        evolving with time constant 'tau'.
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
        #super(linear, self).__init__(ID, params, network)
        unit.__init__(self, ID, params, network)
        self.tau = params['tau']  # the time constant of the dynamics
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        #assert self.type is unit_types.linear, ['Unit ' + str(self.ID) + 
        #                                                    ' instantiated with the wrong type']
        
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. 
        
            Args: 
                y : a 1-element array or list with the current firing rate.
                t: time when the derivative is evaluated.
        """
        return( self.get_input_sum(t) - y[0] ) * self.rtau
   

class noisy_linear(unit):
    """ A linear unit with passive decay and a noisy update function.

        This unit is meant to emulate the 'lin_rate' units from NEST without multiplicative
        coupling. The noiseless dynamics are:
        tau * u' = -lambda * u + inp ,
        where tau is the dynamics' time constant, lambda a decay parameter, and inp
        is the sum of inputs times their weights.

        The update function adds additive white noise with mean 'mu' and standard
        deviation 'sigma'. The update function also rectifies the output of the
        unit, so it never reaches negative values. When lambda>0, the solver used is
        exponential Euler. When lambda=0, the solver is Euler-Maruyama.
    """
    def __init__(self, ID, params, network):
        """ The constructor for the noisy_linear class.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS
                'tau' : Time constant of the update dynamics.
                'lambda': passive decay rate 
                'mu': mean of the white noise
                'sigma': standard deviation for the white noise process

        """
        unit.__init__(self, ID, params, network)
        self.tau = params['tau']  # the time constant of the dynamics
        self.lambd = params['lambda']  # controls the rate of decay
        self.mu = params['mu']
        self.sigma = params['sigma']
        self.rtau = 1./self.tau   # because you always use 1/tau instead of tau
        # if lambd > 0 we'll use the exponential Euler integration method
        if self.lambd > 0.:
            self.diff = lambda y, t: self.get_input_sum(t) * self.rtau
            dt = self.times[1] - self.times[0]
            A = -self.lambd*self.rtau
            self.eAt = np.exp(A*dt)
            self.c2 = (self.eAt-1.)/A
            self.c3 = np.sqrt( (self.eAt**2. - 1.) / (2.*A) )

    def derivatives(self, y, t):
        """ This function returns the derivative of the activity at a given point in time. 
        
            Args: 
                y : a 1-element array or list with the current firing rate.
                t: time when the derivative is evaluated.
        """
        return (self.get_input_sum(t) - self.lambd * y[0]) * self.rtau
 
    def update(self,time):
        """
        Advance the dynamics from time to time+min_delay.

        This update function overrides the one in unit.update in order to use 
        a stochastic solver. 
        """
        assert abs(self.times[-1]-time) < 1e-6, "Time out of phase in unit's update function"
        new_times = self.times[-1] + self.times_grid
        self.times = np.roll(self.times, -self.min_buff_size) 
        self.times[self.offset:] = new_times[1:] 
        # This is the Euler-Maruyama implementation. Basically the same as forward Euler.
        # The atol and rtol values are meaningless in this case.
        dt = new_times[1] - new_times[0]
        # The integration functions are defined in cython_utils.pyx
        if True: #self.lambd == 0.:
            # needs no buffer roll or buffer assignment
            euler_maruyama(self.derivatives, self.buffer, time, 
                                    self.buff_size-self.min_buff_size, dt, self.mu, self.sigma)
        else:
            new_buff = exp_euler(self.diff, self.buffer[-1], time, len(new_times),
                                     dt, self.mu, self.sigma, self.eAt, self.c2, self.c3)
            new_buff = np.maximum(new_buff, 0.)
            self.buffer = np.roll(self.buffer, -self.min_buff_size)
            self.buffer[self.offset:] = new_buff[1:] 

        self.pre_syn_update(time) # Update any variables needed for the synapse to update.
                                  # It is important this is done after the buffer has been updated.
        # For each synapse on the unit, update its state
        for pre in self.net.syns[self.ID]:
            pre.update(time)
        self.last_time = time # last_time is used to update some pre_syn_update values




