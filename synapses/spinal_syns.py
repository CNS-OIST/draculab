"""
spinal_syns.py
This file contains the synapse models used for the spinal cord model.
"""

from draculab import unit_types, synapse_types, syn_reqs  # names of models and requirements
from synapses.synapses import synapse
import numpy as np


class diff_hebb_subsnorm_synapse2(synapse):
    """ A variation on the diff_hebb_subsnorm_synapse.

        This synapse is very similar to the model used by diff_hebb_subsnorm_synapse,
        but the postsynaptic activity is used, rather than its derivative. 
        The rule is:
        w_i' = lrate *  post * ( pre_i' - avg(pre_k') )
        where avg(pre_k') is the mean of the derivatives for all inputs with this synapse 
        type. This is the same formula as Hebbian learning with substractive normalization 
        (see hebb_subsnorm_synapse), where the presynatic activities have been replaced by 
        their derivatives. 
        
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
        e_i = post * ( 2*pre_i' - sum(pre_k') )
        w_i' = lrate * [ { exp(e_i) / sum_k(exp(e_k)) } - w_i ]
        I think this may approach the same rule with multiplicative scaling instead of 
        substractive normalization. Temperature can be introduced too.

        Presynaptic units have the lpf_fast and lpf_mid requirements.
        Postsynaptic units have the lpf_fast, and pos_diff_avg (diff_avg) requirements.
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
        self.upd_requirements = set([syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_mid, 
                           syn_reqs.lpf_fast, syn_reqs.diff_avg]) # pos_diff_avg

        assert self.type is synapse_types.diff_hebbsnorm2, ['Synapse from ' + 
                                str(self.preID) + ' to ' + str(self.postID) + 
                                ' instantiated with the wrong type']


    def update(self, time):
        """ Update using the differential Hebbian rule with substractive normalization. 
        
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
        post = self.net.units[self.postID].get_lpf_fast(self.po_de) 
        Dpre = ( self.net.units[self.preID].get_lpf_fast(self.delay_steps) -
                 self.net.units[self.preID].get_lpf_mid(self.delay_steps) )
        
        # A forward Euler step with the normalized differential Hebbian learning rule 
        self.w = self.w + self.alpha *  post * (Dpre - diff_avg)
        #if self.w < 0: self.w = 0


class rga_synapse(synapse):
    """ A rule used to approach the relatve gain array criterion for choosing inputs.
    
        The particulars are in the 4/11/19 scrap sheet.

        Presynaptic units have the lpf_fast and lpf_mid requirements.
        Postsynaptic units have the lpf_fast, lpf_mid, inp_deriv_mp, 
        avg_inp_deriv_mp, l0_norm_factor_mp requirements. The am_pm_oscillator
        includes the implementation of these last three.

        In addition, units using this type of synapse need to have a
        'custom_inp_del' attribute to indicate the extra delay steps in the 
        'lateral' inputs. Normally it should match the 'post_delay' value below.

        The current implementation normalizes the sum of the absolute values for
        the weights at the 'error' port, making them add to 1.
    """
    def __init__(self, params, network):
        """ The class constructor.

        In its current implementation, the rga synapse assumes that the lateral
        connections are in port 1 of the unit, wheras the error inputs are in port 0.
        This is set in the lat_port and err_port variables.
        
        Args:
            params: same as the parent class, with two additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            'post_delay': NOT USING. delay steps in the post-synaptic activity.
            OPTIONAL PARAMETERS
            'err_port' : port for "error" inputs. Default is 0.
            'lat_port' : port for "lateral" inputs. Default is 1.
            'max_w' : maximum synaptic weight. Default is 3. DEPRECATED
            'min_w' : minimum synaptic weight. Default is -3. DEPRECATED

        Raises:
            AssertionError.
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor to scales the update rule
        #self.po_de = params['post_delay'] # delay in postsynaptic activity
        self.po_de = self.net.units[self.postID].custom_inp_del
        # most of the heavy lifting is done by requirements
        self.upd_requirements = set([syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_mid, 
                             syn_reqs.lpf_fast, syn_reqs.lpf_mid, 
                             syn_reqs.inp_deriv_mp, syn_reqs.avg_inp_deriv_mp,
                             syn_reqs.del_inp_deriv_mp,
                             syn_reqs.del_avg_inp_deriv_mp,
                             syn_reqs.l0_norm_factor_mp,
                             syn_reqs.pre_out_norm_factor])
        assert self.type is synapse_types.rga, ['Synapse from ' + str(self.preID) + 
                   ' to ' + str(self.postID) + ' instantiated with the wrong type']
        if not hasattr(self.net.units[self.postID], 'custom_inp_del'):
            raise AssertionError('An rga synapse has a postsynaptic unit without ' +
                                 'the custom_inp_del attribute')
        if 'lat_port' in params: self.lat_port = params['lat_port']
        else: self.lat_port = 1 
        if 'err_port' in params: self.err_port = params['err_port']
        else: self.err_port = 0 
        # TODO: erase this
        """
        if 'max_w' in params: self.max_w = params['max_w']
        else: self.max_w = 1.
        if 'min_w' in params: self.min_w = params['min_w']
        else: self.min_w = -1.
        """

    def update(self, time):
        """ Update the weight using the RGA-inpsired learning rule.
        
            If the network is correctly initialized, the pre-synaptic unit updates lpf_fast, 
            and lpf_mid, whereas the post-synaptic unit updates lpf_fast, lpf_mid and the 
            average of approximate input derivatives for each port.

            Notice the average of input derivatives can come form upd_pos_diff_avg, which 
            considers only the inputs whose synapses have positive values. To allow synapses
            to potentially become negative, you need to change the synapse requirement in __init__
            from pos_diff_avg to diff_avg, and the pos_diff_avg value used below to diff_avg.
            Also, remove the line "if self.w < 0: self.w = 0"
        """
        u = self.net.units[self.postID]
        xp = u.del_avg_inp_deriv_mp[self.lat_port]
        up = u.get_lpf_fast(self.po_de) - u.get_lpf_mid(self.po_de)
        sp = u.avg_inp_deriv_mp[self.err_port]
        #spj = u.inp_deriv_mp[self.port][?]
        pre = self.net.units[self.preID]
        spj = (pre.get_lpf_fast(self.delay_steps) -
               pre.get_lpf_mid(self.delay_steps) )
        self.w *= 0.5*(u.l0_norm_factor_mp[self.err_port] + pre.out_norm_factor)
        #self.w *= pre.out_norm_factor
        #self.w *= u.l0_norm_factor_mp[self.err_port]
        #self.w += self.alpha * max(up - xp, 0.) * (sp - spj)
        self.w += self.alpha * (up - xp) * (sp - spj)
        #self.w += self.alpha * ((up - xp) * (sp - spj) * 
        #          (self.max_w - self.w) * (self.w - self.min_w))
        #self.w += self.alpha * up * (sp - spj)


class anti_covariance_inh_synapse(synapse):
    """ Anticovariance rule for inhibitory synapses
    
        w' = - lrate * (post - theta) * pre,
        where theta is a low-pass filtered version of post with a slow time constant.
        This correspond to equation 8.8 in Dayan and Abbott's "Theoretical Neuroscience"
        (MIT Press 2001), with the sign reversed.
        The presynaptic activity includes its corresponding transmission delay.

        This synapse type expects to be initialized with a negative weight, and will
        implement soft weight bounding in order to ensure the sign of the weight does
        not become positive, so the rule is actually:
        w' = - lrate * w * (post - theta) * pre,
    
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
            AssertionError, ValueError.

        """
        if params['init_w'] > 0:
            raise ValueError('the anticov_inh synapse expects a negative ' +
                             'initial weight')
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        # The anti-covaraince rule requires the current pre- and post-synaptic activity
        # For the postsynaptic activity, both fast and slow averages are used
        self.upd_requirements = set([syn_reqs.lpf_fast, syn_reqs.pre_lpf_fast, syn_reqs.lpf_slow])
        assert self.type is synapse_types.anticov_inh, ['Synapse from ' + str(self.preID) + ' to ' +
                                                     str(self.postID) + ' instantiated with the wrong type']
    
    def update(self, time):
        """ Update the weight according to the anti-covariance learning rule."""
        # If the network is correctly initialized, the pre-synaptic unit is updatig lpf_fast, and the 
        # post-synaptic unit is updating lpf_fast and lpf_slow at each update() call
        avg_post = self.net.units[self.postID].get_lpf_slow(0)
        post = self.net.units[self.postID].get_lpf_fast(0)
        pre = self.net.units[self.preID].get_lpf_fast(self.delay_steps)
        
        # A forward Euler step with the anti-covariance learning rule 
        self.w = self.w * (1. - self.alpha * (post - avg_post) * pre)


