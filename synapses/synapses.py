"""
synapses.py
This file contains all the synapse models used in the draculab simulator
"""

from draculab import unit_types, synapse_types, syn_reqs  # names of models and requirements
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
                'gain' : Extra gain factor. A factor that scales the weight in
                         the method unit.get_sc_input_sum . 
                'syns_loc' : index of the synapse in network.syns[postID]
            network: the network where the synapse lives.
        Raises:
            AssertionError .

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
        if 'plant_out' in params: 
            self.plant_out = params['plant_out']
            # There should also be 'plant_id', indicating the plant of origin
            self.plant_id = params['plant_id']
        # Some unit models call the method get_sc_input_sum, which requires the gain parameter
        if 'gain' in params: self.gain = params['gain'] 
        # syns_loc is the location (the index) in syns[postID]
        if 'syns_loc' in params: self.syns_loc = params['syns_loc']
        self.delay_steps = None # Delay, in simulation steps units.
                                # Initialized in unit.init_pre_syn_update.
        # TODO: these tests assume unit-to-unit connections, and if there are more 
        # plants than units the tests may fail. There should be something to 
        # indicate whether the synapse is on a plant
        assert self.net.n_units >= self.preID, 'Synapse connected from non ' + \
                                               'existing unit ' + str(self.preID)  
        assert self.net.n_units >= self.postID, 'Synapse connected to non ' + \
                                                'existing unit ' + str(self.postID)  
        assert self.net.sim_time == 0., 'Synapse being created when sim_time is not zero'

        # Each synapse type is tasked with indicating what information it will ask
        # from its corresponding unit when the synapse updates. That information is contained 
        # in the 'upd_requirements' set. Each entry in upd_requirements corresponds to
        # a different variable that the unit must maintain each time the unit's
        # update() function is called. The possible values are in the syn_reqs
        # enumerator of the draculab module.
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
        synapse.__init__(self, params, network)
        assert self.type is synapse_types.static, ['Synapse from ' + str(self.preID) + 
                      ' to ' + str(self.postID) + ' instantiated with the wrong type']

    def update(self,time):
        # Static synapses don't do anything when updated.
        return

          
class oja_synapse(synapse):
    """ This class implements a continuous version of the Oja learning rule. 

        The equation is: 
        w' = lrate * post * (pre - w * post)
        Taken from:
        Foldiak (1989) "Adaptive network for optimal linear feature extraction"
        Proc IEEE/INNS IJCNN 1:401-405 .
        A possibly significant difference is that the presynaptic activity (pre) is delayed,
        according to the transmission delay corresponding to this synapse.

        Since there are no discrete inputs, the presynaptic activity is tracked
        with a low-pass filter, and used to adapt the weight everytime the 
        "update" function is called.
        Pre and postsynaptic units need to include a 'tau_fast' parameter.
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
        synapse.__init__(self, params, network)
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
        lpf_post = self.net.units[self.postID].get_lpf_fast(0)  # no delays to get the postsynaptic rate
        lpf_pre = self.net.units[self.preID].get_lpf_fast(self.delay_steps)
        
        # A forward Euler step with the Oja learning rule 
        self.w = self.w + self.alpha * lpf_post * ( lpf_pre - lpf_post*self.w )


class anti_hebbian_synapse(synapse):
    """ This class implements a simple version of the anti-Hebbian rule.
    
        Units that fire together learn to inhibit each other:
        w' = - lrate * pre * post
        The presynaptic activity includes its corresponding transmission delay.

        Presynaptic and postsynaptic activity is tracked with a fast low-pass filter, so
        pre and postsynaptic units need to include a 'tau_fast' parameter.
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
        synapse.__init__(self, params, network)
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
        lpf_post = self.net.units[self.postID].get_lpf_fast(0)
        lpf_pre = self.net.units[self.preID].get_lpf_fast(self.delay_steps)
        
        # A forward Euler step with the anti-Hebbian learning rule 
        self.w = self.w - self.alpha * lpf_post * lpf_pre 


class covariance_synapse(synapse):
    """ This class implements a version of the covariance rule.

        The equation is: w' = lrate * (post - theta) * pre, 
        where theta is a low-pass filtered version of post with a slow time constant.
        This correspond to equation 8.8 in Dayan and Abbott's "Theoretical Neuroscience"
        (MIT Press 2001). No tests are made to see if the synapse becomes negative.
        The presynaptic activity includes its corresponding transmission delay.

        Presynaptic units require the 'tau_fast' parameter.
        Postsynaptic units require 'tau_fast' and 'tau_slow'.
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
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        # The covaraince rule requires the current pre- and post-synaptic activity
        # For the postsynaptic activity, both fast and slow averages are used
        self.upd_requirements = set([syn_reqs.lpf_fast, syn_reqs.pre_lpf_fast, syn_reqs.lpf_slow])
        assert self.type is synapse_types.cov, ['Synapse from ' + str(self.preID) + ' to ' +
                                                str(self.postID)+' instantiated with the wrong type']
    
    def update(self, time):
        """ Update the weight according to the covariance learning rule."""
        # If the network is correctly initialized, the pre-synaptic unit is updatig lpf_fast, and the 
        # post-synaptic unit is updating lpf_fast and lpf_slow at each update() call
        avg_post = self.net.units[self.postID].get_lpf_slow(0)
        post = self.net.units[self.postID].get_lpf_fast(0)
        pre = self.net.units[self.preID].get_lpf_fast(self.delay_steps)
        # A forward Euler step with the covariance learning rule 
        self.w = self.w + self.alpha * (post - avg_post) * pre 


class anti_covariance_synapse(synapse):
    """ This class implements a version of the covariance rule, with the sign of plasticity reversed.
    
        w' = - lrate * (post - theta) * pre,
        where theta is a low-pass filtered version of post with a slow time constant.
        This correspond to equation 8.8 in Dayan and Abbott's "Theoretical Neuroscience"
        (MIT Press 2001), with the sign reversed.
        The presynaptic activity includes its corresponding transmission delay.
        In this implementation, the synaptic weight may go from positive to negative.
    
        Presynaptic units require the 'tau_fast' parameter.
        Postsynaptic units require 'tau_fast' and 'tau_slow'.
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
        synapse.__init__(self, params, network)
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
        avg_post = self.net.units[self.postID].get_lpf_slow(0)
        post = self.net.units[self.postID].get_lpf_fast(0)
        pre = self.net.units[self.preID].get_lpf_fast(self.delay_steps)
        
        # A forward Euler step with the anti-covariance learning rule 
        self.w = self.w - self.alpha * (post - avg_post) * pre 


class anti_cov_pre_synapse(synapse):
    """ This class implements an anticovariance rule.
    
        w' = - lrate * post * (pre - <pre>),
        where <pre> is a low-pass filtered version of pre with a slow time constant.

        The presynaptic activity includes its corresponding transmission delay.
        In this implementation, the synaptic weight can't become negative.
    
        Presynaptic units require the 'tau_fast' and 'tau_slow' parameters.
        Postsynaptic units require 'tau_fast'.
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
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        # The anti-covaraince rule requires the current pre- and post-synaptic activity
        # For the presynaptic activity, both fast and slow averages are used
        self.upd_requirements = set([syn_reqs.lpf_fast, syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_slow])
        assert self.type is synapse_types.anticov_pre, ['Synapse from ' + str(self.preID) + ' to ' +
                                                     str(self.postID) + ' instantiated with the wrong type']
    
    def update(self, time):
        """ Update the weight according to the anti-covariance learning rule."""
        post = self.net.units[self.postID].get_lpf_fast(0)
        pre = self.net.units[self.preID].get_lpf_fast(self.delay_steps)
        avg_pre = self.net.units[self.preID].get_lpf_slow(self.delay_steps)
        
        # A forward Euler step with the anti-covariance learning rule 
        self.w = self.w - self.alpha * post * (pre - avg_pre)
        if self.w < 0.:
            self.w = 0.


class hebb_subsnorm_synapse(synapse):
    """ This class implements a version of the Hebbian rule with substractive normalization.

        This is the rule: w' = lrate * [ (pre * post) - (post) * (average of inputs) ] .
        This rule keeps the sum of the synaptic weights constant.

        If a weight becomes negative, this implementation sets it to zero, and ignores it
        when calculating the input average.

        Notice that this rule will ignore any inputs with other type of synapses 
        when it obtains the average input value (see pos_inp_avg_hsn in requirements.py).

        The equation used is 8.14 from Dayan and Abbott "Theoretical Neuroscience" (MIT Press 2001)
        Presynaptic inputs include their delays; this includes all the inputs used to
        obtain the input average.

        Presynaptic and postsynaptic units require the 'tau_fast' parameter.
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
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        # The Hebbian rule requires the current pre- and post-synaptic activity.
        # Substractive normalization requires the average input value, obtained from lpf_fast values.
        # The average input value obtained only considers units with positive synaptic weights.
        self.upd_requirements = set([syn_reqs.lpf_fast, syn_reqs.pre_lpf_fast, syn_reqs.pos_inp_avg_hsn])
        assert self.type is synapse_types.hebbsnorm, ['Synapse from ' + str(self.preID) + ' to ' +
                                                       str(self.postID) + ' instantiated with the wrong type']
    
    def update(self, time):
        """ Update the weight using the Hebbian rule with substractive normalization. 
        
            If the network is correctly initialized, the pre-synaptic unit updates lpf_fast, 
            and the post-synaptic unit updates lpf_fast and its input average.

            Notice the average input currently comes from pos_inp_avg_hsn, which only considers
            the inputs whose synapses have positive values.
        """
        inp_avg = self.net.units[self.postID].pos_inp_avg_hsn
        #inp_avg = self.net.units[self.postID].reqs['pos_inp_avg_hsn'].val
        post = self.net.units[self.postID].get_lpf_fast(0)
        pre = self.net.units[self.preID].get_lpf_fast(self.delay_steps)
        
        # A forward Euler step with the normalized Hebbian learning rule 
        self.w = self.w + self.alpha *  post * (pre - inp_avg)
        if self.w < 0: self.w = 0


class sq_hebb_subsnorm_synapse(synapse):
    """ This class implements a version of the Hebbian rule with substractive normalization.

        This is the rule: w' = lrate * [ omega * (pre*post) - w * (post) * (scaled sum of inputs) ] .
        This rule keeps the sum of the squared synaptic weights asymptotically approaching omega,
        and w will not go from positive to negative.
        This implementation will ignore any inputs with other type of synapses when it obtains 
        the average input value (see unit.upd_sc_inp_sum_sqhsn in units.py,  and add_sc_inp_sum 
        in requirements.py).

        The equation used is taken from:
        Moldakarimov, MacClelland and Ermentrout 2006, "A homeostatic rule for inhibitory synapses 
        promotes temporal sharpening and cortical reorganization" PNAS 103(44):16526-16531.
        Presynaptic inputs include their delays; this includes all the inputs used to
        obtain the scaled input sum.

        Presynaptic and postsynaptic units require the 'tau_fast' parameter.
    """
    def __init__(self, params, network):
        """ The  class constructor.

        Args: 
            params: same as the parent class, with some additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            'omega' : Desired value of the sum of squared weights.

        Raises:
            AssertionError.

        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.omega = params['omega'] # the sum of squared weights will approach this
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        # The Hebbian rule requires the current pre- and post-synaptic activity.
        # Substractive normalization requires the scaled input sum, obtained from lpf_fast values.
        self.upd_requirements = set([syn_reqs.lpf_fast, syn_reqs.pre_lpf_fast, syn_reqs.sc_inp_sum_sqhsn])
        assert self.type is synapse_types.sq_hebbsnorm, ['Synapse from ' + str(self.preID) + ' to ' +
                                                          str(self.postID) + ' instantiated with the wrong type']
    
    def update(self, time):
        """ Update the weight using the Hebbian rule with substractive normalization. 
        
            If the network is correctly initialized, the pre-synaptic unit updates lpf_fast, 
            and the post-synaptic unit updates lpf_fast and its scaled sum of lpf'd inputs.
        """
        sc_inp_sum = self.net.units[self.postID].sc_inp_sum_sqhsn
        post = self.net.units[self.postID].get_lpf_fast(0)
        pre = self.net.units[self.preID].get_lpf_fast(self.delay_steps)
        
        # A forward Euler step with the normalized Hebbian learning rule 
        self.w = self.w + self.alpha *  post * ( self.omega * pre - self.w * sc_inp_sum )



class input_correlation_synapse(synapse):
    """ This class implements a version of the input correlation learning rule.

        The rule is based on Porr & Worgotter 2006, Neural Computation 18, 1380-1412;
        but we have no constrain on the number of different types of predictive inputs,
        or on the number of error signals, whose sum acts as the 'error' signal.

        Learning happens only on predictive inputs. The equation is: 
        w' = lrate * pre * err_diff,
        where err_diff is an approximation to the derivative of the sum of all 'error' inputs.
        To obtain err_diff, the fast-lpf'd version of the error sum is substracted the
        medium-lpf'd version (see unit.upd_err_diff) .
        Both predictive and error inputs arrive with their respective transmission delays.

        The learning rule for this synapse may fail when combined with a learning rule for
        selecting the error signals, because the combined error signal used here is
        not scaled by the synaptic weights.

        'error' presynaptic units require the 'tau_fast' and 'tau_mid' parameters.
        'pred' presynaptic units require the 'tau_fast' parameter.
    """

    def __init__(self, params, network):
        """ The class constructor.

        Args:
            params: same as the parent class, with some additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            'input_type' : A string with two possible values
                         'pred' : predictive input.
                         'error' : error input.
                         Each input can be either predictive or error. The predictive 
                         inputs are the ones where the input correlation learning 
                         rule is applied, based on an approximation of the 
                         derivative of the error inputs.

        Raises:
            ValueError, AssertionError.
        """

        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        self.input_type = params['input_type'] # either 'pred' or 'error'

        # to approximate the input derivatives, we use LPFs at two different time scales
        if self.input_type == 'error':
            self.upd_requirements = set([syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_mid, 
                                         syn_reqs.err_diff])
        elif self.input_type == 'pred':
            self.upd_requirements = set([syn_reqs.err_diff, syn_reqs.pre_lpf_fast])
        else:
            raise ValueError('The input_type parameter for input correlation ' +
                             'synapses should be either pred or error')

        assert self.type is synapse_types.inp_corr, ['Synapse from ' + str(self.preID) + 
                        ' to ' + str(self.postID) + ' instantiated with the wrong type']
    
    def update(self, time):
        """ Update the weight using input correlation learning. """
        if self.input_type == 'pred':
            pre = self.net.units[self.preID].get_lpf_fast(self.delay_steps)
            err_diff = self.net.units[self.postID].err_diff
            #self.w = self.w + self.alpha * pre * err_diff
            # clipping
            self.w = min(2., max(-2., self.w + self.alpha * pre * err_diff))
        # In this version of the rule, the error signals don't alter their weights,
        # and we don't test for weights becoming negative


class bcm_synapse(synapse):
    """ This class implements a version of the BCM learning rule.

        In particular, the Law and Cooper 1994 version, where the rule is divided by the
        average squared postsynaptic activity in order to enhance stability.

        The equation is: w' = lrate * pre * post * (post - theta) / theta, 
        where theta is the average of the squared postsynaptic activity, obtained by
        low-pass filtering this squared activity using the tau_slow time constant.
        The pre value used considers the transmission delay of its connection.

        Presynaptic units require the 'tau_fast' parameter.
        Postsynaptic units require 'tau_fast' and 'tau_slow'.
    """

    def __init__(self, params, network):
        """ The class constructor.

        Args:
            params: same as the parent class, with one addition.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.

        Raises:
            AssertionError.
        """

        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        self.upd_requirements = set([syn_reqs.pre_lpf_fast, syn_reqs.lpf_fast, syn_reqs.sq_lpf_slow])

        assert self.type is synapse_types.bcm, ['Synapse from ' + str(self.preID) + ' to ' +
                                                 str(self.postID) + ' instantiated with the wrong type']

    
    def update(self, time):
        """ Update the weight using the BCM rule. """
        post = self.net.units[self.postID].get_lpf_fast(0)
        avg_sq = self.net.units[self.postID].sq_lpf_slow
        pre = self.net.units[self.preID].get_lpf_fast(self.delay_steps)
        # A forward Euler step 
        self.w = self.w + self.alpha * post * (post - avg_sq) * pre / avg_sq



class homeo_inhib_synapse(synapse):
    """ An inhibitory synapse that adjusts according to the activity level. 

        The equation comes from: Moldakarimov, MacClelland and Ermentrout 2006, 
        "A homeostatic rule for inhibitory synapses promotes temporal sharpening 
        and cortical reorganization" PNAS 103(44):16526-16531.

        The present implementation has a fixed desired activity. In a second version
        of the rule, the desired activity is adjusted according to a very slow
        average of the activity.

        This synapse "knows" that it is inhibitory; if w becomes positive it is clamped to 0.
        The equation in Moldakarimov et al. is for the magnitude of the weight (e.g.
        very high activity increases the weight), so we use the negative of that:
        w' = lrate * (des_act - post) = - lrate * (post - des_act)

        Postsynaptic units require the 'tau_fast' parameter.
    """
    def __init__(self, params, network):
        """ The class constructor.

        Args:
            params: same as the parent class, with two additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            'des_act' : A value for the desired average activity level.

        Raises:
            AssertionError.
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.des_act = params['des_act'] # desired average level of activity
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        self.upd_requirements = set([syn_reqs.lpf_fast])

        assert self.type is synapse_types.homeo_inh, ['Synapse from ' + str(self.preID) + ' to ' +
                                                       str(self.postID) + ' instantiated with the wrong type']

    
    def update(self, time):
        """ Update the weight using the homeostatic rule. """
        act = self.net.units[self.postID].get_lpf_fast(0)
        # A forward Euler step 
        self.w = self.w + self.alpha * (self.des_act - act)
        if self.w > 0.: self.w = 0.  


class diff_hebb_subsnorm_synapse(synapse):
    """ A differential Hebbian rule with substractive normalization.

        "Delegate" synapses are meant to strengthen synapses from inputs which become stronger 
        shortly after the unit is active. Moreover, there is competition among inputs.

        To achieve this, the learning rule for the i-th synapse is of the form:
        w_i' = lrate *  post' * ( pre_i' - avg(pre_k') )
        where avg(pre_k') is the mean of the derivatives for all inputs with this synapse 
        type. This is the same formula as Hebbian learning with substractive normalization 
        (see hebb_subsnorm_synapse), where the unit activities have been replaced by their 
        derivatives, and there is delay in the post activity used. As with the Hebbian rule,
        the differential version keeps the sum of synaptic weights invariant. 

        The post activity needs to have a delay for the rule to make sense. There is a natural
        correlation between pre_i' and post' arising from the fact that pre_i is exciting
        the cell, so a positive pre_i'(t) will correlate with a positive post'(t). Thus, we
        need to choose a delay D long enough so that post'(t-D) is independent of the
        stimulation from pre_i(t). This delay might be larger than the one required by the
        unit's connections, so in the unit constructor a long-enough 'delay' parameter must
        be specified.
        
        Weights that become negative are clamped to zero and their corresponding inputs are
        subsequently ignored when computing the average derivative, until the weight becomes
        positive again. This can result in small variations in the sum of synaptic weights.
        A slight modification of the update method can remove this feature and use 
        a straightforward implementation of the rule.

        Notice that the input delays can be relevant for this type of synapse, because the
        correlation between the pre and post derivatives may change according to this delay.

        NOTE: A second version with the same idea but using the Boltzmann distro is:
        e_i = post' * ( 2*pre_i' - sum(pre_k') )
        w_i' = lrate * [ { exp(e_i) / sum_k(exp(e_k)) } - w_i ]
        I think this may approach the same rule with multiplicative scaling instead of 
        substractive normalization. Temperature can be introduced too.

        Presynaptic units have the lpf_fast and lpf_mid requirements.
        Postsynaptic units have the lpf_fast, lpf_mid, and pos_diff_avg (diff_avg) requirements.
    """
    def __init__(self, params, network):
        """ The class constructor.

        Args:
            params: same as the parent class, with two additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            'post_delay': delay steps in the post-synaptic activity.

        Raises:
            AssertionError.
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.po_de = params['post_delay'] # delay in postsynaptic activity
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        self.upd_requirements = set([syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_mid, syn_reqs.lpf_fast,
                                     syn_reqs.lpf_mid, syn_reqs.diff_avg]) # pos_diff_avg

        assert self.type is synapse_types.diff_hebbsnorm, ['Synapse from ' + str(self.preID) + ' to ' +
                                                       str(self.postID) + ' instantiated with the wrong type']


    def update(self, time):
        """ Update the weight using the differential Hebbian rule with substractive normalization. 
        
            If the network is correctly initialized, the pre-synaptic unit updates lpf_fast, 
            and lpf_mid, whereas the post-synaptic unit updates lpf_fast, lpf_mid and the 
            average of approximate input derivatives.

            Notice the average of input derivatives can come form upd_pos_diff_avg, which 
            considers only the inputs whose synapses have positive values. To allow synapses
            to potentially become negative, you need to change the synapse requirement in __init__
            from pos_diff_avg to diff_avg, and the pos_diff_avg value used below to diff_avg.
            Also, remove the line "if self.w < 0: self.w = 0"
        """
        diff_avg = self.net.units[self.postID].diff_avg
        # pos_diff_avg = self.net.units[self.postID].pos_diff_avg
        Dpost = (self.net.units[self.postID].get_lpf_fast(self.po_de) - 
                 self.net.units[self.postID].get_lpf_mid(self.po_de) )
        Dpre = ( self.net.units[self.preID].get_lpf_fast(self.delay_steps) -
                 self.net.units[self.preID].get_lpf_mid(self.delay_steps) )
        
        # A forward Euler step with the normalized differential Hebbian learning rule 
        self.w = self.w + self.alpha *  Dpost * (Dpre - diff_avg)
        #if self.w < 0: self.w = 0


class corr_homeo_inhib_synapse(synapse):
    """ An inhibitory synapse with activity-dependent potentiation and depression.

        The equation comes from: Vogels et al. (2011)
        "Inhibitory Plasticity Balances Excitation and Inhibition in Sensory Pathways
        and Memory Networks", Science 334(6062):1569-1573.

        This rule is meant to maintain a state of balanced exchitation and inhibition using
        inhibitory units whose activity correlates with that of this unit. For the rule to
        be effective it is recommended that inputs to the network activate both excitatory and
        inhibitory units, and that inhibitory units project to the co-excited excitatory units.

        This synapse "knows" that it is inhibitory; if w becomes positive it is clamped to 0.
        The equation in Vogels et al. is for the magnitude of the weight (e.g.
        very high pre-post correation increases the weight), so we use the negative of that:
        w' = -lrate*(pre*post - rho_0*pre) = lrate * pre * (rho_0 - post) 

        Notice this rule is similar to the one in Moldakarimov et al 2006 (homeo_inhib_synapse),
        with an extra modulation by the presynaptic activity. rho_0 is the desired activity,
        and you have the extra constraint of not modifying the synapse unless the presynaptic
        unit is active (otherwise the change wouldn't have any effect anyway).  Thus the notation:
        w' = lrate * pre * (des_act - post),
        and thus the name "Correlational Homeostatic Inhibition Synapse".

        Presynaptic and postsynaptic units require the 'tau_fast' parameter.
    """
    def __init__(self, params, network):
        """ The class constructor.

        Args:
            params: same as the parent class, with two additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            'des_act' : A value for the desired average activity level (rho_0 parameter).

        Raises:
            AssertionError.
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.des_act = params['des_act'] # desired average level of activity
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        self.upd_requirements = set([syn_reqs.lpf_fast, syn_reqs.pre_lpf_fast])

        assert self.type is synapse_types.corr_inh, ['Synapse from ' + str(self.preID) + ' to ' +
                                                       str(self.postID) + ' instantiated with the wrong type']

    
    def update(self, time):
        """ Update the weight using the homeostatic rule. """
        post = self.net.units[self.postID].get_lpf_fast(0)
        pre = self.net.units[self.preID].get_lpf_fast(self.delay_steps)
        # A forward Euler step 
        self.w = self.w + self.alpha * pre * (self.des_act - post)
        if self.w > 0.: self.w = 0.  


class exp_rate_dist_synapse(synapse):
    """ A synapse that adapts in order to produce an exponential distribution of firing rates.

        In other words, a fully connected network where all the connections are of this type
        will in theory converge towards a fixed point for both the firing rates and the
        synaptic weights. At this fixed point the firing rates --which are assumed to lie 
        in the interval (0,1) -- will have a probability density function of the form:
        rho(f) = (c/(1-exp(-c))) * exp(-c*f),  where c is a constant parameter, f is the
        firing rate, and rho(f) is the PDF.

        This rule is only meant to be used with sigmodial units.
    """
    def __init__(self, params, network):
        """ The class constructor.

        Args:
            params: same as the parent class, with two additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            'c' : Changes the homogeneity of the firing rate distribution.
                  Values very close to 0 make all firing rates equally probable, whereas
                  larger values make small firing rates more probable. 
                  Shouldn't be set to zero (causes zero division in the cdf function).
            'wshift' : If the rule wants to shift the firing rate of the neuron, it will
                       change the weight proportionally to this amount. 

        Raises:
            AssertionError.
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.c = params['c'] # level of heterogeneity for firing rates
        self.wshift = params['wshift'] # how much to shift weight if unit should change bin
        #self.k = ( 1. - np.exp(-self.c) ) / self.c   # reciprocal of normalizing factor for the exp distribution
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        #self.upd_requirements = set([syn_reqs.pre_lpf_fast, syn_reqs.lpf_fast, syn_reqs.lpf_mid_inp_sum, syn_reqs.n_erd])
        self.upd_requirements = set([syn_reqs.lpf_fast, syn_reqs.inp_vector, syn_reqs.lpf_mid_inp_sum, syn_reqs.balance])
        assert self.type is synapse_types.exp_rate_dist, ['Synapse from ' + str(self.preID) + ' to ' +
                                                       str(self.postID) + ' instantiated with the wrong type']

    def update(self, time):
        """ Update the weight using the firing rate exponential distribution rule. """
        # The version below is a binless version of w_ss_send_balance in histogram_map.ipynb
        r = self.net.units[self.postID].get_lpf_fast(0)
        r = max( min( .9999, r), 0.0001 ) # avoids bad arguments in the log below
        u = (np.log(r/(1.-r))/self.net.units[self.postID].slope) + self.net.units[self.postID].thresh
        mu = self.net.units[self.postID].get_lpf_mid_inp_sum() 
        left_extra = self.net.units[self.postID].below - self.cdf(r)
        right_extra = self.net.units[self.postID].above - (1. - self.cdf(r))
        delta = self.wshift * (left_extra - right_extra)
        self. w = self.w + self.alpha * ( (u + delta) / mu - self.w )


    def legacy_update(self, time):
        """ A bunch of things I was testing before I had the rule I ended up using.  """
        #f = self.net.units[self.postID].get_lpf_fast(0)
        #f = self.net.units[self.postID].buffer[-1] # using instantaneous value...
        #f = max( min( .999, f), 0.001 ) # avoids bad arguments in the log below
        #u = (np.log(f/(1.-f))/self.net.units[self.postID].slope) + self.net.units[self.postID].thresh
        #mu = self.net.units[self.postID].get_lpf_mid_inp_sum() 
        #h = self.net.units[self.postID].n_erd
        #pre = self.net.units[self.preID].get_lpf_fast(self.delay_steps)
        # A forward Euler step 
        #self.w = self.w + self.alpha * ( (self.k * u * np.exp(self.c * pre) / (mu*h*pre*(pre-1.))) - self.w )
        #self.w = self.w + self.alpha * ( (self.k * u * np.exp(self.c * pre) / (mu*h)) - self.w )
        #self.w = self.w + self.alpha * ( ( self.k * f * np.exp(self.c * pre) / (h*pre*(1.-pre))) - self.w )

        """
        # The version below is an adaptation of the successful weight kernel in histogram_map.ipynb
        w2 = self.net.units[self.postID].bin_width / 2.
        extra = self.net.units[self.postID].around - ( self.cdf(f + w2) - self.cdf(f - w2) )
        #if extra > w2: # too many units around; let's move them
        if extra > 0.: # too many units around; let's move them
            left_extra = self.net.units[self.postID].below - self.cdf(f - w2)
            right_extra = - extra - left_extra
            #right_extra = self.net.units[self.postID].above - (1. - self.cdf(f + w2))
            #assert abs(extra+left_extra+right_extra) < 1e-6, ['extras add to ' 
            #                                               + str(extra+left_extra+right_extra) ]
            delta = self.wshift * self.sgnm(left_extra - right_extra)
        else:
            delta = 0.
        self. w = self.w + self.alpha * ( (u + delta) / mu - self.w )
        """
        #self.w = min( max( -3., self.w ), 3.)
        pass

    def cdf(self, x):
        """ The cumulative density function of the distribution we're approaching. """
        return ( 1. - np.exp(-self.c*x) ) / ( 1. - np.exp(-self.c) )

    def sgnm(self,x):
        """ The sign function, but  returns -1 for x==0 """
        return -1. if x == 0 else np.sign(x)


class delta_synapse(synapse):
    """ A synapse that implements a continuous version of the delta rule.

        The discrete delta rule produces a weight update given by:
        Delta_w = alpha * (des - post) * pre, 
        where alpha is a learning rate, des is a desired output for the postsynaptic unit,
        post is the output of the postsynaptic unit, and pre is the input from the
        presynaptic unit.

        The version implemented here assumes that the error (des-post) is provided by the 
        postsynaptic unit, and has weight dynamics given by:
        w' = alpha * error * pre.

        An alternative version (commented-out code) is:
        w' = alpha * error * (pre -pre_lpf_slow) 

        Weight clipping is used to ensure that the weights of excitatory synapses don't
        become negative, and the weights of inhibitory synapses don't become positive. To
        decide whether a synapse is excitatory or inhibitory, the value of 'init_w' is used.
        When init_w = 0, the synapse is assumed to be excitatory.
        
        This synapse is meant to be used with one of the delta_* unit models.
    """
    def __init__(self, params, network):
        """ The class constructor.

        Args:
            params: same as the parent class, with an addition
            REQUIRED PARAMETERS
            lrate: Learning rate. A scalar value that will multiply the derivative of the weight.

        Warnings:
            UserWarning when connecting to a non-delta unit.
        """
        synapse.__init__(self, params, network) # The parent class' constructor
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        self.post_unit = self.net.units[self.postID] # postsynaptic unit
        self.pre_unit = self.net.units[self.preID] # presynaptic unit
        # update requiements:
        self.upd_requirements = set([syn_reqs.pre_lpf_fast, syn_reqs.lpf_fast, 
                                     syn_reqs.error]) #, syn_reqs.pre_lpf_slow]) 
                                     # pre_lpf_slow only required if you're using the 2nd option
        # maximum and minimum values
        if self.w >= 0:   # if initially excitatory
            self.max_value = 10.
            self.min_value = 0.
        else:
            self.max_value = 0.
            self.min_value = -10.
        if self.net.units[self.postID].type != unit_types.delta_linear:
            from warnings import warn
            warn('A delta synapse was connected to a non-delta unit', UserWarning)

    def update(self, time):
        """ Update the weight using the delta rule. """
        #pre_lpf_slow = self.pre_unit.get_lpf_slow(self.delay_steps)
        pre = self.pre_unit.get_lpf_fast(self.delay_steps)
        err = self.post_unit.error
        self.w = self.w + self.alpha * err * pre
        #self.w = self.w + self.alpha * err * (pre - pre_lpf_slow)
        # clipping values that are too large or too small
        self.w = max(min(self.w, self.max_value), self.min_value)
        
        

class switcher(synapse):
    """ The synapse created in part 2 of tutorial 5. """
    def __init__(self, params, network):
        """ The class constructor. 
        
        Args:
            params: same as the 'synapse' parent class, with these additions
            'lrate': learning rate; a scalar that multiplies the weight's derivative
        """
        synapse.__init__(self, params, network) # calling the parent's constructor
        self.lrate = params['lrate'] # learning rate
        self.upd_requirements = set([syn_reqs.lpf_mid, syn_reqs.pre_lpf_mid])
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule 
        
    def update(self, time):
        """ Updates the synaptic weight at each simulation step. """
        pre_avg = self.net.units[self.preID].get_lpf_mid(self.delay_steps)
        post_avg = self.net.units[self.postID].get_lpf_mid(0)
        if pre_avg > post_avg:
            self.w += self.alpha * (1. - self.w)
        else:
            self.w += self.alpha * (-1. - self.w)




        
