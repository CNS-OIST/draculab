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
from numba import jit
#import ray


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
                         This should only be set by network.connect
                'coordinates' : a numpy array specifying the spatial location of the unit.
                'tau_fast' : time constant for the fast low-pass filter.
                'tau_mid' : time constant for the medium-speed low-pass filter.
                'tau_slow' : time constant for the slow low-pass filter.
                'n_ports' : number of inputs ports. Defaults to 1.
                'integ_meth' : a string specifying an integration method for the unit.
            network: the network where the unit lives.

        Raises:
            AssertionError, ValueError
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
        # The final value of 'delay' should be set by network.connect(), after the unit is created;
        # it will have an extra net.min_delay added to the actual delay.
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
        if 'integ_meth' in params: # a particular integration method is specified for the unit
            if self.type is unit_types.source:
                raise AssertionError('Specifying an integration method for source units can result in errors')
            if params['integ_meth'] == "euler":
                self.update = self.euler_update
            elif params['integ_meth'] == "euler_maru":
                if not 'mu' in params or not 'sigma' in params:
                    raise AssertionError('Euler-Maruyama integration requires mu and sigma parameters')
                self.update = self.euler_maru_update
            elif params['integ_meth'] == "exp_euler":
                if not 'lambda' in params:
                    raise AssertionError('The exponential Euler method requires a lambda parameter')
                self.update = self.exp_euler_update
            elif params['integ_meth'] == "odeint":
                self.update = self.odeint_update
            elif params['integ_meth'] == "solve_ivp":
                self.update = self.solve_ivp
                from warnings import warn
                warn(['Additional code modifications are required for solve_ivp. \
                      See unit.solve_ivp_update documentation'], UserWarning)
            elif params['integ_meth'] == 'custom':
                pass
            else:
                raise ValueError('integ_method was incorrectly specified.')
            self.integ_meth = params['integ_meth']
        else:
            if not self.type is unit_types.source:
                self.update = self.odeint_update # default solver
                self.integ_meth = "odeint"
        self.syn_needs = set() # the set of all variables required by synaptic dynamics
                               # It is initialized by the init_pre_syn_update function
        #self.reqs = {} # This dictionary will contain all requirement objects c
        #               # corresponding to the entries in syn_needs. DEPRECATED.
        self.last_time = 0.  # time of last call to the update function
                            # Used by the upd_lpf_X functions
        self.init_buffers() # This will create the buffers that store states and times
        self.functions = set() # will contain all the functions that update requirements

     
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
        self.bf_type = np.float64 # np.dtype('d') # data type of the buffer when it is a numpy array

        # The following buffers are for synaptic requirements that can come from presynaptic units.
        # They store one value per update, in the requirement's update method.
        if syn_reqs.lpf_fast in self.syn_needs:
            self.lpf_fast_buff = np.array( [self.init_val]*self.steps, dtype=self.bf_type)
        if syn_reqs.lpf_mid in self.syn_needs:
            self.lpf_mid_buff= np.array( [self.init_val]*self.steps, dtype=self.bf_type)
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
        # The 'buffer' contains previous activation values
        # currently testing 3 ways of initializing the buffer (if not flattened):
                    # 1) a numpy contiguous array 
        self.buffer = np.ascontiguousarray( [self.init_val]*self.buff_size, dtype=self.bf_type) 
                    # 2) a numpy array. Usually the same as above 
        #self.buffer = np.array( [self.init_val]*self.buff_size, dtype=self.bf_type) 
                    # 3) a python array 
        #self.buffer = array('d', [self.init_val]*self.buff_size)
        self.time_bit = min_del / min_buff # time interval used by get_act
        self.times = np.linspace(-self.delay+self.time_bit, 0., self.buff_size, dtype=self.bf_type) # the 
                                                               #corresponding times for the buffer values
        self.times_grid = np.linspace(0, min_del, min_buff+1, dtype=self.bf_type) # used to create
                                     # values for 'times'. Initially its interval differs with 'times'
        
        
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
        return [ fun(time - dely) for dely,fun in 
                 zip(self.net.delays[self.ID], self.net.act[self.ID]) ]


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
        return [ np.array([self.net.act[self.ID][idx](time - self.net.delays[self.ID][idx]) 
                 for idx in idx_list]) for idx_list in self.port_idx ]


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
        return [ np.array([self.net.syns[self.ID][idx].w for idx in idx_list])
                           for idx_list in self.port_idx ]


    def odeint_update(self, time):
        """ Advance the dynamics from time to time+min_delay.

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
        # odeint also returns the initial condition, so to produce min_buff_size new values
        # we need to provide min_buff_size+1 desired times, starting with 
        # the one for the initial condition
        new_buff = odeint(self.derivatives, [self.buffer[-1]], new_times,
                          rtol=self.rtol, atol=self.atol)
        self.buffer = np.roll(self.buffer, -self.min_buff_size)
        self.buffer[self.offset:] = new_buff[1:,0]
        self.upd_reqs_n_syns(time)


    def solve_ivp_update(self, time):
        """
        Advance the dynamics from time to time+min_delay.

        This update function will replace the values in the activation buffer
        corresponding to the latest "min_delay" time units, introducing "min_buff_size" new 
        values. In addition, all the synapses of the unit are updated.
        source and kwta units override this with shorter update functions.

        NOTICE: solve_ivp requires that the derivatives function has its parameters in the
                opposite of the order used by odeint. Thus, to make it work you need to change
                the order of the arguments in all the derivatives functions, from 
                derivatives(self, y, t) to derivatives(self, t, y).
                In vi it takes one command: :%s/derivatives(self, y, t)/derivatives(self, t, y)/
                Also, make sure that the import command at the top is uncommented for solve_ivp.
                One more thing: to use the stiff solvers the derivatives must return a list or 
                array, sothe returned value must be enclosed in square brackets in 
                <unit_type>.derivatives.
        """
        #assert (self.times[-1]-time) < 2e-6, 'unit' + str(self.ID) + ': update time is desynchronized'
        new_times = self.times[-1] + self.times_grid
        self.times = np.roll(self.times, -self.min_buff_size)
        self.times[self.offset:] = new_times[1:]
        solution = solve_ivp(self.derivatives, (new_times[0], new_times[-1]), [self.buffer[-1]],
                             method='LSODA', t_eval=new_times, rtol=self.rtol, atol=self.atol)
        self.buffer = np.roll(self.buffer, -self.min_buff_size)
        self.buffer[self.offset:] = solution.y[0,1:]
        self.upd_reqs_n_syns(time)


    def euler_update(self, time):
        """ Advance the dynamics from time to time+min_delay using the forward Euler method. 

            This implementation uses forward Euler integration. Although this may be
            imprecise and unstable in some cases, it is a first step to implement the
            Euler-Maruyama method. And it's also faster than odeint.
            Notice the atol and rtol network parameters are not used in this case.
            Precision is controlled by the step size, which is min_delay/min_buff_size.
        """
        new_times = self.times[-1] + self.times_grid
        self.times = np.roll(self.times, -self.min_buff_size)
        self.times[self.offset:] = new_times[1:]
        dt = new_times[1] - new_times[0]
        # euler_int is defined in cython_utils.pyx
        new_buff = euler_int(self.derivatives, self.buffer[-1], time, len(new_times), dt)
        self.buffer = np.roll(self.buffer, -self.min_buff_size)
        self.buffer[self.offset:] = new_buff[1:]
        self.upd_reqs_n_syns(time)


    def euler_maru_update(self, time):
        """ Advance the dynamics from time to time+min_delay with the Euler-Maruyama method.

            The Euler-Maruyama implementation is basically the same as forward Euler.
            The atol and rtol values are meaningless in this case.

            This solver does the buuffer's "rolling" by itself.
            The unit needs to have 'mu' and 'sigma' attributes.
            self.mu = 0. # Mean of the white noise
            self.sigma = 0.0 # standard deviation of Wiener process.
        """
        # euler_maruyama_ is defined in cython_utils.pyx
        euler_maruyama(self.derivatives, self.buffer, time,
                        self.buff_size-self.min_buff_size, self.time_bit, self.mu, self.sigma)
        self.upd_reqs_n_syns(time)

    def exp_euler_update(self, time):
        """ Advance the dynamics from time to time+min_delay with the exponential Euler method.

            This method can only be used when the unit model has noisy dynamics where the 
            activity follows an instaneous input function with a particular decay rate. 
            This includes the noisy_linear and noisy_sigmoidal models.

            The current version is rectifying ouputs by default.
        """
        new_times = self.times[-1] + self.times_grid
        self.times = np.roll(self.times, -self.min_buff_size) 
        self.times[self.offset:] = new_times[1:] 

        new_buff = exp_euler(self.diff, self.buffer[-1], time, len(new_times),
                             self.time_bit, self.mu, self.sigma, self.eAt, self.c2, self.c3)
        new_buff = np.maximum(new_buff, 0.) # THIS IS RECTIFYING BY DEFAULT!!!
        self.buffer = np.roll(self.buffer, -self.min_buff_size)
        self.buffer[self.offset:] = new_buff[1:] 
        self.upd_reqs_n_syns(time)


    #************ I WILL ERASE update AFTER TESTING *************
    #def update(self,time):
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
        #new_times = self.times[-1] + self.times_grid
        #self.times = np.roll(self.times, -self.min_buff_size)
        #self.times[self.offset:] = new_times[1:]
        
        # TO USE A PARTICULAR SOLVER, UNCOMMENT ITS CODE, COMMENT OTHER SOLVERS
        #---------------------------------------------------------------------
        # odeint
        """
        # odeint also returns the initial condition, so to produce min_buff_size new values
        # we need to provide min_buff_size+1 desired times, starting with the one for the initial condition
        new_buff = odeint(self.derivatives, [self.buffer[-1]], new_times, rtol=self.rtol, atol=self.atol)
        self.buffer = np.roll(self.buffer, -self.min_buff_size)
        self.buffer[self.offset:] = new_buff[1:,0]
        """
        #---------------------------------------------------------------------
        # solve_ivp --- REQUIRES ADDITIONAL STEPS (see below)
        """
        # This is a reimplementation with solve_ivp. To make it work you need to change the order of the
        # arguments in all the derivatives functions, from derivatives(self, y, t) to derivatives(self, t, y).
        # In vi it takes one command: :%s/derivatives(self, y, t)/derivatives(self, t, y)/
        # Also, make sure that the import command at the top is uncommented for solve_ivp.
        # One more thing: to use the stiff solvers the derivatives must return a list or array, so
        # the returned value must be enclosed in square brackets in <unit_type>.derivatives.
        solution = solve_ivp(self.derivatives, (new_times[0], new_times[-1]), [self.buffer[-1]],
                             method='LSODA', t_eval=new_times, rtol=self.rtol, atol=self.atol)
        self.buffer = np.roll(self.buffer, -self.min_buff_size)
        self.buffer[self.offset:] = solution.y[0,1:]
        """
        #---------------------------------------------------------------------
        # Euler
        #"""
        # This is an implementation with forward Euler integration. Although this may be
        # imprecise and unstable in some cases, it is a first step to implement the
        # Euler-Maruyama method. And it's also faster than odeint.
        # Notice the atol and rtol values are not used in this case.
        #dt = new_times[1] - new_times[0]
        # euler_int is defined in cython_utils.pyx
        #new_buff = euler_int(self.derivatives, self.buffer[-1], time, len(new_times), dt)
        #self.buffer = np.roll(self.buffer, -self.min_buff_size)
        #self.buffer[self.offset:] = new_buff[1:]
        #"""
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
        
    def upd_reqs_n_syns(self, time):
        """ Update the unit's requirements and its synapses.

            This should be called at every min_delay integration step, after the unit's
            buffers have been updated.
        """
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
        # Make sure to import interp1d at the top of the file before using this.
        # Sometimes the ode solver asks about values slightly out of bounds, 
        # so I set this to extrapolate
        """
        return interp1d(self.times, self.buffer, kind='linear', bounds_error=False, copy=False,
                        fill_value="extrapolate", assume_sorted=True)(time)
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Below the code for the second implementation.
        # This linear interpolation takes advantage of the ordered, regularly-spaced buffer.
        # Time values outside the buffer range receive the buffer endpoints.
        """
        time = min( max(time,self.times[0]), self.times[-1] ) # clipping 'time'
        frac = (time-self.times[0])/(self.times[-1]-self.times[0])
        base = int(np.floor(frac*(self.buff_size-1))) # biggest index s.t. times[index] <= time
        frac2 = ( time-self.times[base] ) / 
                ( self.times[min(base+1,self.buff_size-1)] - self.times[base] + 1e-8 )
        return self.buffer[base] + frac2 * ( self.buffer[min(base+1,self.buff_size-1)] -
                                             self.buffer[base] )
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
        #return cython_get_act2(time, self.times[0], self.times[-1], self.times,
        #                        self.buff_size, self.buffer)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This is the third implementation, written in Cython
        #return cython_get_act3(time, self.times[0], self.time_bit, self.buff_size, self.buffer)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This is for flat networks
        #TODO: remove this if after third type of flattening is standard
        if self.net.flat: # for the third type of flattening this is not necessary
            return self.net.get_act(self.ID, time)
        else:
            return cython_get_act3(time, self.times[0], self.time_bit, self.buff_size, self.buffer)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def pre_syn_update(self, time):
        """ Call the update functions for the requirements added in init_pre_syn_update. """
        #for r in self.reqs:
        #    self.reqs[r].update(time)
        for f in self.functions:
            f(time)

  
    def init_pre_syn_update(self):
        """
        Configure the pre_syn_update function according to current synaptic requirements.

        Correlational learning rules require the pre- and post-synaptic activity, in this
        case low-pass filtered in order to implement a running average. Moreover, for
        heterosynaptic plasticity individual synapses need information about all the
        other synapses on the unit. It is inefficient for each synapse to maintain
        low-pass filtered versions of pre- and post-synaptic activity, as well as to
        obtain by itself all the values required for its update.
        The function pre_syn_update(), initialized by init_pre_syn_update(), 
        is tasked with updating all the 'slow' unit variables, which can be either used by
        the synapses, or by the unit itself.
        pre_syn_update() is called once per simulation step (i.e. every min_delay period).

        In addition, for each one of the unit's synapses, init_pre_syn_update will initialize 
        its delay value.

        An extra task done here is to prepare the 'port_idx' list used by units with multiple 
        input ports.

        init_pre_syn_update is called for a unit everytime network.connect() connects the unit, 
        which may be more than once.

        Raises:
            NameError, NotImplementedError, ValueError.
        """
        """
        DEVELOPER'S NOTES1:
        The names of the possible requirements are in the syn_reqs Enum in draculab.py .
        Any new requirement needs to register its name in there.
        Implementation of a requirement has 2 main parts:
        1) Initialization: where the data structures used by the requirement get their
           initial values. This is done here in init_pre_syn_update using an if-elif
           statement that identifies the requirement. See NOTES2.
           Ideally the variable with the value of the requirement has the name
           of the requirement. For example, the LPF'd activity with a fast time
           constant has the name 'lpf_fast'.
        2) Update: a method with the name upd_<req name>  that is added to the
           'functions' list of the unit. This method belongs to the 'unit' class and
           is written somewhere after init_pre_syn_update.

        Additionally, some requirements have buffers so they can provide their value
        as it was 'n' simulation steps before. The prototypical case is the 'lpf'
        variables, which are retrieved by synapses belonging to a target cell, so 
        the value they get should have a propagation delay. This requires two other
        things in the implementation:
        1) Initialization of the requirement's buffer is added in unit.init_buffers.
           The buffers should have the name <req name>_buff.
        2) There are 'getter' methods to retrieve past values of the requirement.
           The getter methods are named get_<req_name> .
        """
        """
        DEVELOPER'S NOTES2: 
        Previously this method had a for loop with a giant wall of elif statements,
        each one initializing data structures for a particular requirement. This wall
        of elif statements had poor readability, and the update methods of the
        requirements were away from the initialization (they still are), which can 
        make working with requirements a scrolling-intensive deal.

        There were 2 strategies under consideration to deal with the elif wall:
        1) Moving the requirements' variables, initialization, and update routines to
           classes in the requirements.py files. What init_pre_syn update does in this
           case is to create instances of the requirement classes, placing them in the 
           'reqs' dictionary. This way, pre_syn_update operates by calling the 'update' 
           method of all requirements in 'reqs'.
           This organizes things very nicely, but unfortunately it also slows down
           the code considerably, presumably because of the extra indirection.
           The slowing down of the code happens even if the instances of the requirement
           classes are placed outside of the 'reqs' dictionary.
        2) Creating add_<requirement_name> functions that encapsulate the initialization
           or the data structures used by the requirement. The point is to get rid of the
           elif wall withough slowing down the code, so this option attempts to 
           create the same data structures and update methods as with the elif wall,
           but do it all in the add_ function using 'setattr'. As before, the update
           functions are placed in the 'functions' list.
           This method can get rid of the wall of elifs, but unfortunately, setting the
           update methods with 'setattr' makes them slow, so they still need to be defined
           somewhere inside of the 'unit' class. Thus, although the wall of elifs
           can be brought down, there is still separation between the initialization of
           the requierements' data structures and its update method.

        I currently go with option 2. Moreover, to reduce the number of upd_ methods
        in 'unit', for those requirements that are specific to particular unit types, I
        put the upd_ methods in those units. 

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
                    if syn_reqs.pre_lpf_fast in syn.upd_requirements:
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

        # Prepare all the requirements to be updated by the pre_syn_update function. 
        """
        # under option 1 the rest of the method would be reduced to:
        for req in self.syn_needs:
            if issubclass(type(req), syn_reqs):
                self.reqs[req.name] = eval(req.name+'(self)')
            else:  
                raise NotImplementedError('Asking for a requirement that is not implemented')
        """
        # Option 2:
        for req in self.syn_needs:
            if issubclass(type(req), syn_reqs):
                eval('add_'+req.name+'(self)')
                eval('self.functions.add(self.upd_'+req.name+')')
            else:  
                raise NotImplementedError('Asking for a requirement that is not implemented')


    ###################################
    # UPDATE METHODS FOR REQUIREMENTS #
    ###################################

    def upd_lpf_fast(self, time):
        """ Update the lpf_fast variable. """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_fast updated backwards in time']
        cur_act = self.get_act(time)
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
        #return self.reqs['lpf_fast'].get(steps)


    def upd_lpf_mid(self, time):
        """ Update the lpf_mid variable. """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_mid updated backwards in time']
        cur_act = self.get_act(time)
        # Same update method as in upd_lpf_fast
        self.lpf_mid = cur_act + ( (self.lpf_mid - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_mid) )
        # update the buffer
        self.lpf_mid_buff = np.roll(self.lpf_mid_buff, -1)
        self.lpf_mid_buff[-1] = self.lpf_mid
 

    def get_lpf_mid(self, steps):
        """ Get the mid-speed low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_mid_buff[-1-steps]


    def upd_lpf_slow(self, time):
        """ Update the lpf_slow variable. """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_slow updated backwards in time']
        cur_act = self.get_act(time)
        # Same update method as in upd_lpf_fast
        self.lpf_slow = cur_act + ( (self.lpf_slow - cur_act) * 
                                   np.exp( (self.last_time-time)/self.tau_slow) )
        # update the buffer
        self.lpf_slow_buff = np.roll(self.lpf_slow_buff, -1)
        self.lpf_slow_buff[-1] = self.lpf_slow
 

    def get_lpf_slow(self, steps):
        """ Get the slow low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_slow_buff[-1-steps]


    def upd_sq_lpf_slow(self,time):
        """ Update the sq_lpf_slow variable. """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' sq_lpf_slow updated backwards in time']
        cur_sq_act = self.get_act(time)**2.  
        # Same update method as in upd_lpf_fast
        self.sq_lpf_slow = cur_sq_act + ( (self.sq_lpf_slow - cur_sq_act) * 
                                  np.exp( (self.last_time-time)/self.tau_slow ) )

    
    def upd_inp_vector(self, time):
        """ Update the inp_vector variable. """
        self.inp_vector = np.array([ fun(time - dely) for dely,fun in 
                   zip(self.net.delays[self.ID], self.net.act[self.ID]) ])


    def upd_mp_inputs(self, time):
        """ Update the mp_inputs variable. """
        self.mp_inputs = self.get_mp_inputs(time)


    def upd_inp_avg_hsn(self, time):
        """ Update the inp_avg_hsn variable. """
        self.inp_avg_hsn = ( sum([u.get_lpf_fast(s) for u,s in self.snorm_list_dels]) 
                             / self.n_hebbsnorm )


    def upd_pos_inp_avg_hsn(self, time):
        """ Update the pos_inp_avg_hsn variable. """
        # first, update the n vector from Eq. 8.14, pg. 290 in Dayan & Abbott
        self.n_vec = [ 1. if syn.w>0. else 0. for syn in self.snorm_syns ]
        self.pos_inp_avg_hsn = sum([n*(u.get_lpf_fast(s)) for n,u,s in 
                        zip(self.n_vec, self.snorm_units, self.snorm_delys)]) / sum(self.n_vec)


    def upd_err_diff(self, time):
        """ Update an approximate derivative of the error inputs used for input correlation learning. 

            A very simple approach is taken, where the derivative is approximated as the difference
            between the fast and medium low-pass filtered inputs. Each input arrives with its
            corresponding transmission delay.
        """
        self.err_diff = ( sum([ self.net.units[i].get_lpf_fast(s) for i,s in self.err_idx_dels ]) -
                          sum([ self.net.units[i].get_lpf_mid(s) for i,s in self.err_idx_dels ]) )
       

    def upd_sc_inp_sum_sqhsn(self, time):
        """ Update the sum of the inputs multiplied by their synaptic weights.
        
            The actual value being summed is lpf_fast of the presynaptic units.
        """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' sc_inp_sum_sqhsn updated backwards in time']
        self.sc_inp_sum_sqhsn = sum([u.get_lpf_fast(d) * syn.w for u,d,syn in self.u_d_syn])
        
        
    def upd_diff_avg(self, time):
        """ Update the average of derivatives from inputs with diff_hebbsnorm synapses.

            The values being averaged are not the actual derivatives, but approximations
            which are roughly proportional to them, coming from the difference 
            lpf_fast - lpf_mid . 
        """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' diff_avg updated backwards in time']
        self.diff_avg = ( sum([u.get_lpf_fast(s) - u.get_lpf_mid(s) for u,s in self.dsnorm_list_dels]) 
                          / self.n_dhebbsnorm )


    def upd_lpf_mid_inp_sum(self,time):
        """ Update the lpf_mid_inp_sum variable. """
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_mid_inp_sum updated backwards in time']
        cur_inp_sum = self.inp_vector.sum()
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
        inputs = self.mp_inputs    # updated because mp_inputs is a requirement variable
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
            the population of units that connect to the 'rdc_port'.

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
        #I = u - np.sum( self.inp_vector.get()[self.exc_idx] * weights[self.exc_idx] ) 
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


    def upd_slide_thresh(self, time):
        """ Updates the threshold of exp_dist_sig_thr units and some other 'trdc' units.

            The algorithm is an adapted version of the  one used in exp_rate_dist synapses.
        """
        r = self.get_lpf_fast(0)
        r = max( min( .995, r), 0.005 ) # avoids bad arguments and overflows
        exp_cdf = ( 1. - np.exp(-self.c*r) ) / ( 1. - np.exp(-self.c) )
        error = self.below - self.above - 2.*exp_cdf + 1. 
        self.thresh -= self.tau_thr * error
        self.thresh = max(min(self.thresh, 10.), -10.) # clipping large/small values
        

    def upd_slide_thresh_shrp(self, time):
        """ Updates the threshold of trdc units when input at 'sharpen' port is larger than 0.5 .

            The algorithm is based on upd_slide_thresh.
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


    ##########################################
    # END OF UPDATE METHODS FOR REQUIREMENTS #
    ##########################################
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # METHODS FOR FLAT NETWORKS ~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def upd_flat_inp_sum(self,time):
        """ Updates the vector with input sums for each substep of the current step. """
        # Obtain the input vector 'step_inps'
        # step_inps is a 2D numpy array. step_inps[j,k] provides the activity
        # of the j-th input in the k-th substep of the current timestep.
        #"""
        self.step_inps = self.acts[self.acts_idx]
        if self.step_inps.size == 0:  # when the unit has no inputs
            self.inp_sum = np.zeros(self.min_buff_size)
            return
        # The line below is an experimental version in which self.acts_idx was a 1-D 
        # array of logical indexes. For some reason it was considerably slower.
        #self.step_inps = self.acts[self.acts_idx].reshape(self.n_inps, self.min_buff_size)
        # update the input sum
        w_vec = np.array([syn.w for syn in self.net.syns[self.ID]]) # update weights
        self.inp_sum = np.matmul(w_vec, self.step_inps)
        """ 
        # testing a numba version
        w_vec = np.array([syn.w for syn in self.net.syns[self.ID]]) # update weights
        self.inp_sum = ufis_4_numba(self.acts, self.flat_acts_idx, w_vec,
        self.n_inps, self.min_buff_size)
        """

    def flat_euler_update(self, time):
        """ The forward Euler integration method used with network.flat_update3. """
        # This will fail if you haven't called upd_flat_inp_sum in the current step,
        # because dt_fun3 uses self.inp_sum to obtain the derivative
        # Roll the buffer
        base = self.buffer.size - self.min_buff_size
        # rolling is not being done in network.flat_update3
        #self.buffer[0:base] = self.buffer[self.min_buff_size:] # one way to roll
        #np.copyto(self.buffer, np.roll(self.buffer, -self.min_buff_size)) # a different roll
        # put new values in buffer
        for idx in range(self.min_buff_size):
            self.buffer[base+idx] = self.buffer[base+idx-1] + ( self.time_bit *
                                    self.dt_fun3(self.buffer[base+idx-1], idx) )


    def flat_euler_maru_update(self,time):
        """ The Euler-Maruyama integration used with network.flat_update3. """
        base = self.buffer.size - self.min_buff_size
        noise = self.sigma * np.random.normal(loc=0., scale=self.sqrdt,
                                              size=self.min_buff_size)
        for idx in range(self.min_buff_size):
            self.buffer[base+idx] = self.buffer[base+idx-1] + ( self.time_bit *
                                    self.dt_fun3(self.buffer[base+idx-1], idx) +
                                    self.mudt + noise[idx] )


    def flat_exp_euler_update(self, time):
        """ The exponential Euler integration used with network.flat_update3. """
        base = self.buffer.size - self.min_buff_size
        noise = self.sc3 * np.random.normal(loc=0., scale=1.,
                                            size=self.min_buff_size)
        for idx in range(self.min_buff_size):
            self.buffer[base+idx] = self.eAt * self.buffer[base+idx-1] + ( self.c2 *
                                    self.dt_fun3_eu(self.buffer[base+idx-1], idx) +
                                    self.mudt + noise[idx] )
 

#@jit(nopython=True)
def ufis_4_numba(acts, flat_acts_idx, w_vec, n_inps, mbf):
    """ receives the acts vector, its index, and weights, returns the inp_sum vector. """
    step_inps = acts.flatten()[flat_acts_idx].reshape(n_inps, mbf)
    if step_inps.size == 0:  # when the unit has no inputs
        return np.zeros(mbf)
    for row in range(n_inps):
        step_inps[row,:] = w_vec[row] * step_inps[row,:]
    return np.sum(step_inps, axis=0)


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

    def dt_fun1(self, y, t):
        """ The derivatives function used when the network is flat. """
        return ( self.f(self.net.inp_sums[self.ID][t]) - y ) * self.rtau

    def dt_fun2(self, y, s):
        """ The derivatives function used when the network is flat. """
        return ( self.f(self.net.inp_sums[self.ID][s]) - y ) * self.rtau

    def dt_fun3(self, y, s):
        """ The derivatives function used when the network is flat. """
        return ( self.f(self.inp_sum[s]) - y ) * self.rtau

   

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
        self.mudt = self.mu * self.time_bit # used by flat updater
        # if lambd > 0 we'll use the exponential Euler integration method
        if self.lambd > 0.:
            self.update = self.exp_euler_update
            self.diff = lambda y, t: self.f(self.get_input_sum(t)) * self.rtau
            dt = self.times[1] - self.times[0] # same as self.time_bit
            A = -self.lambd*self.rtau
            self.eAt = np.exp(A*dt)
            self.c2 = (self.eAt-1.)/A
            self.c3 = np.sqrt( (self.eAt**2. - 1.) / (2.*A) )
            self.sc3 = self.sigma * self.c3
        else:
            self.update = self.euler_maru_update
            self.sqrdt = np.sqrt(self.time_bit) # used by flat updater

    def f(self, arg):
        """ This is the sigmoidal function. Could roughly think of it as an f-I curve. """
        return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
        #return cython_sig(self.thresh, self.slope, arg)
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_input_sum(t)) - self.lambd*y[0] ) * self.rtau

    def dt_fun3(self, y, s):
        """ The derivatives function used when the network is flat. """
        return ( self.f(self.inp_sum[s]) - y ) * self.rtau

    def dt_fun3_eu(self, y, s):
        """ The derivatives function for flat_exp_euler_update. """
        return  self.f(self.inp_sum[s]) * self.rtau

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
   

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        return ( self.net.inp_sums[self.ID][s] - y ) * self.rtau

    def dt_fun3(self, y, s):
        """ The derivatives function used when the network is flat. """
        return ( self.inp_sum[s] - y ) * self.rtau


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
        self.mudt = self.mu * self.time_bit # used by flat updaters
        # if lambd > 0 we'll use the exponential Euler integration method
        if self.lambd > 0.:
            self.update = self.exp_euler_update
            self.diff = lambda y, t: self.get_input_sum(t) * self.rtau
            dt = self.times[1] - self.times[0] # same as self.time_bit
            A = -self.lambd*self.rtau
            self.eAt = np.exp(A*dt)
            self.c2 = (self.eAt-1.)/A
            self.c3 = np.sqrt( (self.eAt**2. - 1.) / (2.*A) )
            self.sc3 = self.sigma * self.c3 # used by flat_exp_euler_update
        else:
            self.update = self.euler_maru_update
            self.sqrdt = np.sqrt(self.time_bit) # used by flat_euler_maru_update

    def derivatives(self, y, t):
        """ This function returns the derivative of the activity at a given point in time. 
        
            Args: 
                y : a 1-element array or list with the current firing rate.
                t: time when the derivative is evaluated.
        """
        return (self.get_input_sum(t) - self.lambd * y[0]) * self.rtau
 
    def dt_fun3(self, y, s):
        """ The derivatives function used when the network is flat. """
        return ( self.inp_sum[s] - self.lambd * y ) * self.rtau

    def dt_fun3_eu(self, y, s):
        """ The derivatives function for flat_exp_euler_update. """
        return  self.inp_sum[s] * self.rtau
