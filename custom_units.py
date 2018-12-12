"""
custom_units.py
The non-standard unit models in the draculab simulator.
Add your experimental models in here, and don't forget to also add their
names and class names in the unit_types Enum of draculab.py .
"""
from draculab import unit_types, synapse_types, syn_reqs  # names of models and requirements
from synapses import * # synapse models
from units import *   # the unit class and its utilities
import numpy as np
from cython_utils import euler_int # the cythonized forward Euler integration
from cython_utils import euler_maruyama # the cythonized Euler-Maruyama approximation
from cython_utils import exp_euler # the cythonized exponential Euler approximation


"""
source, sigmoidal, noisy sigmoidal, linear, noisy linear
"""

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
        unit.__init__(self, ID, params, network)
        self.tau = params['tau']  # the time constant of the dynamics
        self.f = params['function'] # the f-I curve
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        assert self.type is unit_types.custom_fi, ['Unit ' + str(self.ID) + 
                                                            ' instantiated with the wrong type']
        
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_input_sum(t)) - y[0] ) * self.rtau
 
    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        return ( self.f(self.inp_sum[s]) - y ) * self.rtau

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
        unit.__init__(self, ID, params, network)
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
        unit.__init__(self, ID, params, network)
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
        unit.__init__(self, ID, params, network)
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
        unit.init_buffers() # the init_buffers of the unit parent class

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
        #return 1. / (1. + np.exp(-slope*(arg - thresh)))
        return cython_sig(thresh, slope, arg)

    def derivatives(self, y, t):
        """ This function returns the derivative of the activity given its current values. """
        #return ( self.f(self.thresh, self.slope, self.get_mp_input_sum(t)) - y[0] ) * self.rtau
        return ( cython_sig(self.thresh, self.slope, self.get_mp_input_sum(t)) - y[0] ) * self.rtau


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
        self.bias = 0. # initial value of the bias input (updated in upd_error)

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



