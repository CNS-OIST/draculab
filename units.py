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
        self.multi_port = False # indicates whether the unit has multi-port support, which 
                                # means it has customized get_input* functions

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


    def get_mp_input_sum(self, time):
        """
        Returns the sum of inputs scaled by their weights, assuming there are multiple input ports.

        The sum accounts for transmission delays. All ports are treated identically. 
        """
        return sum([ syn.w * fun(time-dely, syn.plant_out) for syn,fun,dely in zip(self.net.syns[self.ID], 
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

    def get_weights(self, time):
        """ Returns a list with the weights corresponding to the input list obtained with get_inputs.
        """
        # if you include axo-axonic connections you have to modify the list below
        return [ synapse.get_w(time) for synapse in self.net.syns[self.ID] ]


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
        assert (self.times[-1]-time) < 2e-6, 'unit' + str(self.ID) + ': update time is desynchronized'

        new_times = self.times[-1] + self.times_grid
        self.times = np.roll(self.times, -self.min_buff_size)
        self.times[self.offset:] = new_times[1:] 
        
        # odeint also returns the initial condition, so to produce min_buff_size new values
        # we need to provide min_buff_size+1 desired times, starting with the one for the initial condition
        new_buff = odeint(self.derivatives, [self.buffer[-1]], new_times, rtol=self.rtol, atol=self.atol)
        self.buffer = np.roll(self.buffer, -self.min_buff_size)
        self.buffer[self.offset:] = new_buff[1:,0] 

        self.pre_syn_update(time) # update any variables needed for the synapse to update
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

        init_pre_syn_update is called for a unit everytime network.connect() connects it, 
        which may be more than once.

        Raises:
            NameError, NotImplementedError, ValueError.
        """ 
        assert self.net.sim_time == 0, ['Tried to run init_pre_syn_update for unit ' + 
                                         str(self.ID) + ' when simulation time is not zero']

        # Each synapse should know the delay of its connection
        for syn, delay in zip(self.net.syns[self.ID], self.net.delays[self.ID]):
            syn.delay_steps = int(round(delay/self.net.min_delay))

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

            else:  # <----------------------------------------------------------------------
                raise NotImplementedError('Asking for a requirement that is not implemented')

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
        # update the buffer
        self.lpf_fast_buff = np.roll(self.lpf_fast_buff, -1)
        self.lpf_fast_buff[-1] = self.lpf_fast


    def get_lpf_fast(self, steps):
        """ Get the fast low-pass filtered activity, as it was 'step' simulation steps before. """
        return self.lpf_fast_buff[-1-steps]


    def upd_lpf_mid(self,time):
        """ Update the lpf_mid variable. """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_mid updated backwards in time']
        cur_act = self.get_act(time) # current activity
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
        """ Get the mid-speed low-pass filtered activity, as it was 'step' simulation steps before. """
        return self.lpf_mid_buff[-1-steps]


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
        # update the buffer
        self.lpf_slow_buff = np.roll(self.lpf_slow_buff, -1)
        self.lpf_slow_buff[-1] = self.lpf_slow


    def get_lpf_slow(self, steps):
        """ Get the slow low-pass filtered activity, as it was 'step' simulation steps before. """
        return self.lpf_slow_buff[-1-steps]


    def upd_sq_lpf_slow(self,time):
        """ Update the sq_lpf_slow variable. """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' sq_lpf_slow updated backwards in time']
        cur_sq_act = self.get_act(time)**2 # square of current activity
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


class source(unit):
    """ The class of units whose activity comes from some Python function.
    
        source units provide inputs to the network. They can be conceived as units
        whose activity at time 't' comes from a function f(t). This function is passed to
        the constructor as a parameter, or later specified with the set_function method.
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
    """ Same as the linear unit, but with several input ports; useful for tests."""

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
        self.multi_port = True
        assert self.type is unit_types.mp_linear, ['Unit ' + str(self.ID) + 
                                                            ' instantiated with the wrong type']
        
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return (self.get_mp_input_sum(t) - y[0]) * self.rtau
 

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



