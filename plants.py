"""
plants.py
This file contains all the physical plant models used in the draculab simulator.
A plant is very similar to a unit, but it has a vector output.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d 
from draculab import plant_models, synapse_types

class plant():
    """ The parent class of all non-unit models that interact with the network.

    Derived classes can implement any system that can be modeled with an ODE.

    Objects of the derived classes are basically the same as draculab's units: they receive inputs
    and they send outputs to and from draculab's units. Moreover, their inputs and outputs are
    mediated by the same synapse types used by units. But wheras a unit's output can only be a firing 
    rate, models have multidimensional outputs that can correspond to the the state variables of a 
    physical model. 
    
    By putting the appropriate dynamics in the derivatives() function, a physical plant
    controlled by the network of units can interact with the network using draculab's synapses and
    delayed connections.

    The added complexity of connecting the plant's multidimensional inputs and outputs is handled
    by network.set_plant_inputs() and network.set_plant_outputs().
    """

    def __init__(self, ID, params, network):
        """ The class constructor. 

        Args:
            ID: An integer identifying the plant. Set by network.create_plant
            params: parameter dictionary
            REQUIRED PARAMETERS
                'dimension' : dimensionality of the state vector. 
                'type' : an enum identifying the type of plant being instantiated.
                'inp_dim' : input dimensionality; number of qualitatively different input types.
            OPTIONAL PARAMETERS
                'delay' : maximum delay in the outputs sent by the plant. Set by network.set_plant_outputs.

        """
        self.ID = ID # An integer identifying the plant
        self.net = network # the network where the plant lives
        self.inp_dim = params['inp_dim'] # input dimensionality; number of qualitatively different input types.
        self.inputs =  [[] for _ in range(self.inp_dim)]
        # self.inputs is a list of lists, with the get_act functions of the units that 
                         # provide inputs to the plant. inputs[i][j] = j-th input of the i-th type.
        self.inp_dels = [[] for _ in range(self.inp_dim)]
                        # for each entry in self.inputs, self.inp_dels gives the corresponding delay
        self.inp_syns = [[] for _ in range(self.inp_dim)]
                        # for each entry in self.inputs, self.inp_syns has its synapse
        self.type = params['type'] # an enum identifying the type of plant being instantiated
        self.dim = params['dimension'] # dimensionality of the state vector
        # The delay of a plant is the maximum delay among the projections it sends. 
        # Its final value is set by network.set_plant_outputs(), after the plant is created.
        if 'delay' in params: 
            self.delay = params['delay']
            # delay must be a multiple of net.min_delay. Next line checks that.
            assert (self.delay+1e-6)%self.net.min_delay < 2e-6, ['unit' + str(self.ID) + 
                                                                 ': delay is not a multiple of min_delay']       
        else:  # giving a temporary value
            self.delay = 2. * self.net.min_delay 
        self.rtol = self.net.rtol # local copies of ODE solver tolerances
        self.atol = self.net.atol

        self.init_buffers() # This will create the buffers that store states and times


    def init_buffers(self):
        """ This method (re)initializes the buffer variables according to the current parameters.

        It is useful because new connections may increase self.delay, and thus the size of the buffers.
        """
    
        assert self.net.sim_time == 0., 'Plant buffers were reset when the simulation time is not zero'
        
        min_del = self.net.min_delay  # just to have shorter lines below
        min_buff = self.net.min_buff_size
        self.steps = int(round(self.delay/min_del)) # delay, in units of the minimum delay
        self.offset = (self.steps-1)*min_buff # an index used in the update function of derived classes
        self.buff_size = int(round(self.steps*min_buff)) # number of state values to store
        # keeping with scipy.integrate.odeint, each row in the buffer will correspond to a state
        if hasattr(self, 'init_state'): # if we have initial values to put in the buffer
            self.buffer = np.array([self.init_state]*self.buff_size) # initializing buffer
        else:
            self.buffer = np.ndarray(shape=(self.buff_size, self.dim), dtype=float) 
        self.times = np.linspace(-self.delay, 0., self.buff_size) # times for each buffer row
        self.times_grid = np.linspace(0, min_del, min_buff+1) # used to create values for 'times'



    def get_state(self, time):
        """ Returns an array with the state vector. """
        return interp1d(self.times, self.buffer, kind='linear', axis=0, bounds_error=False, 
                        copy=False, fill_value="extrapolate", assume_sorted=True)(time)


    def get_state_var(self, t, idx):
        """ Returns the value of the state variable with index 'idx' at time 't'. """
        # Sometimes the ode solver asks about values slightly out of bounds, so I set this to extrapolate
        return interp1d(self.times, self.buffer[:,idx], kind='linear', bounds_error=False, copy=False,
                        fill_value="extrapolate", assume_sorted=True)(t)

    def get_state_var_fun(self, idx):
        """ Returns a function that returns the state variable with index 'idx' at a given time. 
        
        This is used so units can ignore any input port settings while receiving inputs from a plant.
        """
        return lambda t : interp1d(self.times, self.buffer[:,idx], kind='linear', bounds_error=False, copy=False,
                        fill_value="extrapolate", assume_sorted=True)(t)


    def get_input_sum(self, time, port):
        """ Returns the sum of all inputs of type 'port', as received at the given 'time'.

            For each input of type 'port', you'll get its value at the time ('time' - delay),
            multiply that value by the corresponding synaptic weight, and then sum all the
            scaled values. This sum is the returned value, which constitutes the total
            input provided by all inputs in the given port.
        """
        return sum( [ fun(time - dely)*syn.w for fun,dely,syn in 
                      zip(self.inputs[port], self.inp_dels[port], self.inp_syns[port]) ] )
        
    def update(self,time):
        ''' This function advances the state for net.min_delay time units. '''
        assert (self.times[-1]-time) < 2e-6, 'unit' + str(self.ID) + ': update time is desynchronized'

        new_times = self.times[-1] + self.times_grid
        self.times = np.roll(self.times, -self.net.min_buff_size)
        self.times[self.offset:] = new_times[1:] 
        
        # odeint also returns the initial condition, so to produce min_buff_size new states
        # we need to provide min_buff_size+1 desired times, starting with the one for the initial condition
        new_buff = odeint(self.derivatives, self.buffer[-1,:], new_times, rtol=self.rtol, atol=self.atol)
        self.buffer = np.roll(self.buffer, -self.net.min_buff_size, axis=0)
        self.buffer[self.offset:,:] = new_buff[1:,:] 


    def append_inputs(self, inp_funcs, ports, delays, synaps):
        """ Append the given input functions in the given input ports, with the given delays and synapses.

        An input port is just a list in the 'inputs' list. The 'inputs' list contains 'inp_dim' lists,
        each one with all the functions providing that type of input.
        
        append_inputs is an auxiliary method for network.set_plant_inputs.

        Args:
            inp_funcs: a list with functions that will provide inputs to the plant.
            ports: a list with integers, identifying the type of input that the corresponding entry
                   in inp_funcs will provide. 
            delays: a list with the delays for the inputs in inp_funcs.
            synaps: a list of synapse objects that will scale the inputs in inp_funcs.

        Raises:
            ValueError, NotImplementedError.
        """
        # Test the port specification
        test1 = len(inp_funcs) == len(ports)
        test2 = (np.amax(ports) < self.inp_dim) and (np.amin(ports) >= 0)
        if not(test1 and test2):
            raise ValueError('Invalid input port specification when adding inputs to plant')
            
        # Then append the inputs
        for funz, portz in zip(inp_funcs, ports):  # input functions
            self.inputs[portz].append(funz)

        for delz, portz in zip(delays, ports):  # input delays
            self.inp_dels[portz].append(delz)

        for synz, portz in zip(synaps, ports):   # input synapses
            if synz.type is synapse_types.static:
                self.inp_syns[portz].append(synz)
            else:
                raise NotImplementedError('Inputs to plants only use static synapses')


class pendulum(plant):
    """ 
    Model of a rigid, homogeneous rod with a 1-dimensional rotational joint at one end.

    The simplest model of an arm is a rigid rod with one end attached to a rotational
    joint with 1 degree of freedom. Like a rigid pendulum, but the gravity vector is
    optional. 
    There is no extra mass at the end of the rod; the center of mass is at length/2 .
    
    On the XY plane the joint is at the origin, the positive X direction
    aligns with zero degrees, and gravity points in the negative Y direction.
    Counterclockwise rotation and torques are positive. Angles are in radians, time is
    in seconds.

    Inputs to the model at port 0 are torques applied at the joint. Other ports are ignored.

    The get_state(time) function returns the two state variables of the model in a numpy 
    array. Angle has index 0, and angular velocity has index 1. The \'time\' argument
    should be in the interval [sim_time - del, sim_time], where sim_time is the current
    simulation time (time of last simulation step), and del is the \'delay\' value
    of the plant.

    Alternatively, the state variables can be retrieved with the get_angle(t) and
    get_ang_vel(t) functions.
    """

    def __init__(self, ID, params, network):
        """ The class constructor.

        Args:
            ID: An integer serving as a unique identifier in the network.
            params: A dictionary with parameters to initialize the model.
                REQUIRED PARAMETERS
                'type' : A model from the plant_models enum.
                'length' : length of the rod [m]
                'mass' : mass of the rod [kg]
                'init_angle' : initial angle of the rod. [rad]
                'init_ang_vel' : initial angular velocity of the rod. [rad/s]
                OPTIONAL PARAMETERS
                'g' : gravitational acceleration constant. [m/s^2] (Default: 9.8)
                'inp_gain' : A gain value that multiplies the inputs. (Default: 1)
                'mu' : A viscous friction coefficient. (Default: 0)
            network: the network where the plant instance lives.

        Raises:
            AssertionError.

        """
        # This is the stuff that goes into all model constructors. Adjust accordingly.
        #-------------------------------------------------------------------------------
        assert params['type'] is plant_models.pendulum, ['Plant ' + str(self.ID) + 
                                                    ' instantiated with the wrong type']
        params['dimension'] = 2 # notifying dimensionality of model to parent constructor
        params['inp_dim'] = 1 # notifying there is only one type of inputs
        super(pendulum, self).__init__(ID, params, network) # calling parent constructor

        # Using model-specific initial values, initialize the state vector.
        self.init_state = np.array([params['init_angle'], params['init_ang_vel']])
        self.buffer = np.array([self.init_state]*self.buff_size) # initializing buffer
        #-------------------------------------------------------------------------------
        # Initialize the parameters specific to this model
        
        self.length = params['length']  # length of the rod [m]
        self.mass = params['mass']   # mass of the rod [kg]
        # To get the moment of inertia assume a homogeneous rod, so I = m(l/2)^2
        self.I = self.mass * self.length * self.length / 4.
        # The gravitational force per kilogram can be set as a parameter
        if 'g' in params: self.g = params['g']
        else: self.g = 9.8  # [m/s^2]
        # But we'll usually use 'g' mutliplied by mass*length/2, so
        self.glo2 = self.g * self.mass * self.length / 2.
        # We may also have an input gain
        if 'inp_gain' in params: self.inp_gain = params['inp_gain']
        else: self.inp_gain = 1.
        # And finally, we have a viscous friction coefficient
        if 'mu' in params: self.mu = params['mu']
        else: self.mu = 0.


    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. 

            y[0] = angle
            y[1] = angular velocity
            The 'inputs' list should contain functions providing input torques
        """
        torque = self.inp_gain * self.get_input_sum(t,0)
        # torque may also come from gravity and friction
        torque -= ( np.cos(y[0]) * self.glo2 ) + ( self.mu * y[1] ) 
        # angular acceleration = torque / (inertia moment)
        ang_accel = torque / self.I

        return np.array([y[1], ang_accel])

    
    def get_angle(self,time):
        """ Returns the angle in radians, modulo 2*pi. """
        return self.get_state_var(time,0) % (2*np.pi)
              

    def get_ang_vel(self,time):
        """ Returns the angular velocity in rads per second. """
        return self.get_state_var(time,1) 

        
    
class conn_tester(plant):
    """
    This plant is used to test the network.set_plant_inputs and plant.append_inputs functions.

    Pay no attention to it.
    """
    def __init__(self, ID, params, network):
        """ conn_tester constructor """
        # This is the stuff that goes into all model constructors. 
        #-------------------------------------------------------------------------------
        assert params['type'] is plant_models.conn_tester, ['Plant ' + str(self.ID) + 
                                                    ' instantiated with the wrong type']
        params['dimension'] = 3 # notifying dimensionality of model to parent constructor
        params['inp_dim'] = 3 # notifying number of input ports
        super(conn_tester, self).__init__(ID, params, network) # calling parent constructor

        # Using model-specific initial values, initialize the state vector.
        self.init_state = np.array(params['init_state'])
        self.buffer = np.array([self.init_state]*self.buff_size) # initializing buffer
        #-------------------------------------------------------------------------------

    def derivatives(self, y, t):
        ''' The first two equations, when their input sum is 1, have sin and cos solutions. 
    
        Different ratios of the input amplitudes should bring differernt ratios of the solution 
        amplitudes, and the amplitude of the inputs should change the frequency of the solutions.
        The third equation has an exponential solution, with time constant equal to the input.
        '''
        return np.array([-y[1]*self.get_input_sum(t,0),
                          y[0]*self.get_input_sum(t,1),
                         -y[2]*self.get_input_sum(t,2)])


