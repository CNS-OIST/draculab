"""
synapses.py
This file contains all the synapse models used in the sirasi simulator
"""

from sirasi import unit_types, synapse_types, syn_reqs  # names of models and requirements
import numpy as np

class synapse():
    """ The parent class of all synapse models. 
    
        The models in this class update their weights using their update() functions.
        It is assumed that this function will be called for each synapse by their postsynaptic
        units, and that this will be done whenever the simulation advances network.min_delay
        time units.
    """

    def __init__(self, params, network):
        """ The class constructor. 
          
        Args:
            params: A parameter dictionary to initialize the synapse.
            REQUIRED PARAMETERS
                'preID' : the ID of the presynaptic unit.
                'postID' : the ID of the postsynaptic unit.
                'init_w' : initial synaptic weight.
                'type' : A type from the synapse_types enum.
            OPTIONAL PARAMETERS
                'inp_port' : the input port for the synapse.
            network: the network where the synapse lives.

        """
        self.preID = params['preID']   # the ID of the presynaptic unit or plant
        self.postID = params['postID'] # the ID of the postsynaptic unit or plant
        self.w = params['init_w'] # initializing the synaptic weight
        self.type = params['type'] # assigning the synapse type
        self.net = network # the network where the synapse lives
        # To allow for qualitatively different types of inputs (e.g. receptors), each 
        # input arrives at a particular 'input port', characterized by some integer.
        if 'inp_port' in params: self.port = params['inp_port']
        else: self.port = 0
        # The input may come from a plant, in which case we want to know from which output
        if 'plant_out' in params: self.plant_out = params['plant_out']
        # TODO: these tests assume unit-to-unit connections, and if there are more plants than units, 
        # the tests may fail. There should be something to indicate whether the synapse is on a plant
        assert self.net.n_units >= self.preID, 'Synapse connected from non existing unit ' + str(self.preID)  
        assert self.net.n_units >= self.postID, 'Synapse connected to non existing unit ' + str(self.postID)  
        assert self.net.sim_time == 0., 'Synapse being created when sim_time is not zero'

        # Each synapse type is tasked with indicating what information it will ask
        # from its corresponding unit when the synapse updates. That information is contained 
        # in the 'upd_requirements' set. Each entry in upd_requirements corresponds to
        # a different variable that the unit must maintain each time the unit's
        # update() function is called. The possible values are in the syn_reqs
        # enumerator of the sirasi module.
        self.upd_requirements = set() # start with an empty set

    def get_w(self, time): 
        """ Returns the current synaptic weight. 

        The 'time' argument is currently not used.
        """
        # If I'm going to use time, I need to use buffers and interpolation.
        # Not necessary so far.
        return self.w

    def update(self,time):
        # The default update rule does nothing.
        return

    
class static_synapse(synapse):
    """ A class for the synapses that don't change their weight value. """

    def __init__(self, params, network):
        """ Unit constructor.

        Args: same as the parent class.

        Raises: AssertionError.
        """
        super(static_synapse, self).__init__(params, network)
        assert self.type is synapse_types.static, ['Synapse from ' + str(self.preID) + ' to ' +
                                                            str(self.postID) + ' instantiated with the wrong type']

    def update(self,time):
        # Static synapses don't do anything when updated.
        return

          
class oja_synapse(synapse):
    """ This class implements a continuous version of the Oja learning rule. 

        Since there are no discrete inputs, the presynaptic activity is tracked with a low-pass filter,
        and used to adapt the weight everytime the "update" function is called.
    """
    def __init__(self, params, network):
        """ The  class constructor.

        Args: 
            params: same as the parent class, with some additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
        Raises:
            AssertionError.

        """
        super(oja_synapse, self).__init__(params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        # The Oja rule requires the current pre- and post-synaptic activity
        self.upd_requirements = set([syn_reqs.lpf_fast, syn_reqs.pre_lpf_fast])
        assert self.type is synapse_types.oja, ['Synapse from ' + str(self.preID) + ' to ' +
                                                          str(self.postID) + ' instantiated with the wrong type']

    
    def update(self, time):
        """ Update the weight according to the Oja learning rule."""
        # If the network is correctly initialized, the pre- and post-synaptic units
        # are updating their lpf_fast variables at each update() call
        lpf_post = self.net.units[self.postID].lpf_fast
        lpf_pre = self.net.units[self.preID].lpf_fast
        
        # A forward Euler step with the Oja learning rule 
        self.w = self.w + self.alpha * lpf_post * ( lpf_pre - lpf_post*self.w )


class anti_hebbian_synapse(synapse):
    """ This class implements a simple version of the anti-Hebbian rule.
    
        Units that fire together learn to inhibit each other.
    """

    def __init__(self, params, network):
        """ The  class constructor.

        Args: 
            params: same as the parent class, with some additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
        Raises:
            AssertionError.

        """
        super(anti_hebbian_synapse, self).__init__(params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        # The anti-Hebbian rule requires the current pre- and post-synaptic activity
        self.upd_requirements = set([syn_reqs.lpf_fast, syn_reqs.pre_lpf_fast])
        assert self.type is synapse_types.antihebb, ['Synapse from ' + str(self.preID) + ' to ' +
                                                          str(self.postID) + ' instantiated with the wrong type']
    
    def update(self, time):
        """ Update the weight according to the anti-Hebbian learning rule."""
        # If the network is correctly initialized, the pre- and post-synaptic units
        # are updating their lpf_fast variables at each update() call
        lpf_post = self.net.units[self.postID].lpf_fast
        lpf_pre = self.net.units[self.preID].lpf_fast
        
        # A forward Euler step with the anti-Hebbian learning rule 
        self.w = self.w - self.alpha * lpf_post * lpf_pre 


class covariance_synapse(synapse):
    """ This class implements a version of the covariance rule.

        No tests are made to see if the synapse becomes negative.
    """
    def __init__(self, params, network):
        """ The  class constructor.

        Args: 
            params: same as the parent class, with some additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
        Raises:
            AssertionError.

        """
        super(covariance_synapse, self).__init__(params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        # The covaraince rule requires the current pre- and post-synaptic activity
        # For the postsynaptic activity, both fast and slow averages are used
        self.upd_requirements = set([syn_reqs.lpf_fast, syn_reqs.pre_lpf_fast, syn_reqs.lpf_slow])
        assert self.type is synapse_types.cov, ['Synapse from ' + str(self.preID) + ' to ' +
                                                          str(self.postID) + ' instantiated with the wrong type']
    
    def update(self, time):
        """ Update the weight according to the covariance learning rule."""
        # If the network is correctly initialized, the pre-synaptic unit is updatig lpf_fast, and the 
        # post-synaptic unit is updating lpf_fast and lpf_slow at each update() call
        avg_post = self.net.units[self.postID].lpf_slow
        post = self.net.units[self.postID].lpf_fast
        pre = self.net.units[self.preID].lpf_fast
        
        # A forward Euler step with the covariance learning rule 
        self.w = self.w + self.alpha * (post - avg_post) * pre 


class anti_covariance_synapse(synapse):
    """ This class implements a version of the covariance rule, with the sign of plasticity reversed."""

    def __init__(self, params, network):
        """ The  class constructor.

        Args: 
            params: same as the parent class, with some additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
        Raises:
            AssertionError.

        """
        super(anti_covariance_synapse, self).__init__(params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        # The anti-covaraince rule requires the current pre- and post-synaptic activity
        # For the postsynaptic activity, both fast and slow averages are used
        self.upd_requirements = set([syn_reqs.lpf_fast, syn_reqs.pre_lpf_fast, syn_reqs.lpf_slow])
        assert self.type is synapse_types.anticov, ['Synapse from ' + str(self.preID) + ' to ' +
                                                          str(self.postID) + ' instantiated with the wrong type']
    
    def update(self, time):
        """ Update the weight according to the anti-covariance learning rule."""
        # If the network is correctly initialized, the pre-synaptic unit is updatig lpf_fast, and the 
        # post-synaptic unit is updating lpf_fast and lpf_slow at each update() call
        avg_post = self.net.units[self.postID].lpf_slow
        post = self.net.units[self.postID].lpf_fast
        pre = self.net.units[self.preID].lpf_fast
        
        # A forward Euler step with the anti-covariance learning rule 
        self.w = self.w - self.alpha * (post - avg_post) * pre 

class hebb_subsnorm_synapse(synapse):
    """ This class implements a version of the Hebbian rule with substractive normalization.

        This is the rule: dw = (pre * post) - (post) * (average of inputs).
        This rule keeps the sum of the synaptic weights constant.
        Notice that this rule will ignore any inputs with other type of synapses 
        when it obtains the average input value (see unit.upd_inp_avg and unit.init_pre_syn_update).
    """
    def __init__(self, params, network):
        """ The  class constructor.

        Args: 
            params: same as the parent class, with some additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
        Raises:
            AssertionError.

        """
        super(hebb_subsnorm_synapse, self).__init__(params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        # The Hebbian rule requires the current pre- and post-synaptic activity.
        # Substractive normalization requires the average input value, obtained from lpf_fast values.
        # The average input value obtained is only considers units with positive synaptic weights.
        self.upd_requirements = set([syn_reqs.lpf_fast, syn_reqs.pre_lpf_fast, syn_reqs.pos_inp_avg])
        assert self.type is synapse_types.hebbsnorm, ['Synapse from ' + str(self.preID) + ' to ' +
                                                          str(self.postID) + ' instantiated with the wrong type']
    
    def update(self, time):
        """ Update the weight with the Hebbian rule with substractive normalization. 
            If the network is correctly initialized, the pre-synaptic unit will update lpf_fast, 
            and the post-synaptic unit is updating lpf_fast and its input average.
        """
        inp_avg = self.net.units[self.postID].pos_inp_avg
        post = self.net.units[self.postID].lpf_fast
        pre = self.net.units[self.preID].lpf_fast
        
        # A forward Euler step with the normalized Hebbian learning rule 
        self.w = self.w + self.alpha *  post * (pre - inp_avg)
        if self.w < 0: self.w = 0


class input_correlation_synapse(synapse):
    """ This class implements a version of the input correlation learning rule.

    The rule is based on Porr & Worgotter 2006, Neural Computation 18, 1380-1412. 
    But we have no constrain on the number of different types of predictive inputs, or on the
    number of error signals, whose sum acts as the \'error\' signal.
    """

    def __init__(self, params, network):
        """ The class constructor.

        Args:
            params: same as the parent class, with some additions.
            'lrate' : A scalar value that will multiply the derivative of the weight.
            'input_type' : each input can be either predictive or error. The predictive inputs are
                           the ones where the input correlation learning rule is applied, based on
                           an approximation of the derivatie of the error inputs.
                           'pred' : predictive input.
                           'error' : error input.

        Raises:
            ValueError, AssertionError.
        """

        super(input_correlation_synapse, self).__init__(params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        self.input_type = params['input_type'] # either 'pred' or 'error'

        # to approximate the input derivatives, we use LPFs at two different time scales
        if self.input_type == 'error':
            self.upd_requirements = set([syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_mid, syn_reqs.err_deriv])
        elif self.input_type = 'pred':
            self.upd_requirements = set([syn_reqs.err_deriv])
        else:
            raise ValueError('The input_type parameter for input correlation synapses should be either pred or error')

        assert self.type is synapse_types.inp_corr, ['Synapse from ' + str(self.preID) + ' to ' +
                                                          str(self.postID) + ' instantiated with the wrong type']
    
    def update(self, time):
        """ Update the weight using input correlation learning. """
        if self.input_type == 'pred':
            pre = self.net.units[self.preID].get_act(time) # If needed, could use lpf_fast instead
            err_diff = self.net.units[self.preID].
            self.w = self.w + self.alpha * pre * ...




