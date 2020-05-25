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

        Presynaptic units are given the lpf_fast and lpf_mid requirements.

        Postsynaptic units are given lpf_fast, lpf_mid, inp_deriv_mp, 
        avg_inp_deriv_mp, del_inp_deriv_mp, del_avg_inp_deriv_mp, 
        l0_norm_factor_mp, and pre_out_norm_factor requirements. 
        
        The update methods for most of these requirements are currently in the
        rga_reqs class of the spinal_units.py file.        

        In addition, units using this type of synapse need to have a
        'custom_inp_del' attribute to indicate the extra delay steps in the 
        'lateral' inputs. The synapse will use this delay for the activities of
        its postsynaptic unit and for the 'lateral' inputs.

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

        Raises:
            AssertionError.
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor to scales the update rule
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
        # po_de is the delay in postsynaptic activity for the learning rule
        # It is set to match the delay in the 'lateral' input ports of the post unit
        self.po_de = self.net.units[self.postID].custom_inp_del
        if 'lat_port' in params: self.lat_port = params['lat_port']
        else: self.lat_port = 1 
        if 'err_port' in params: self.err_port = params['err_port']
        else: self.err_port = 0 
        
    def update(self, time):
        """ Update the weight using the RGA-inspired learning rule.
        
            If the network is correctly initialized, the pre-synaptic unit 
            updates lpf_fast, and lpf_mid, whereas the post-synaptic unit
            updates lpf_fast, lpf_mid, and the average of approximate input
            derivatives for each port.

            Notice the average of input derivatives can come form 
            upd_pos_diff_avg, which considers only the inputs whose synapses
            have positive values. To allow synapses to potentially become negative,
            you need to change the synapse requirement in __init__ from 
            pos_diff_avg to diff_avg, and the pos_diff_avg value used below to 
            diff_avg.
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
        # weight normalization
        norm_fac = .5*(u.l0_norm_factor_mp[self.err_port] + pre.out_norm_factor)
        self.w += self.alpha * (norm_fac - 1.)*self.w
        # weight update
        #self.w += self.alpha * max(up - xp, 0.) * (sp - spj)
        self.w += self.alpha * (up - xp) * (sp - spj)
        #self.w += self.alpha * ((up - xp) * (sp - spj) * 
        #          (self.max_w - self.w) * (self.w - self.min_w))
        #self.w += self.alpha * up * (sp - spj)


class gated_rga_synapse(synapse):
    """ A variant of the rga synapse with modulated learning rate.

        The RGA rule is described in the 4/11/19 scrap sheet.

        Presynaptic units are given the lpf_fast and lpf_mid requirements.

        Postsynaptic units are given lpf_fast, lpf_mid, inp_deriv_mp, 
        avg_inp_deriv_mp, del_inp_deriv_mp, del_avg_inp_deriv_mp, 
        l0_norm_factor_mp, and pre_out_norm_factor requirements. 
        Postsynaptic units are also expected to include the acc_slow
        requirement, which is used to modulate the learning rate.
        
        The update methods for most of these requirements are currently in the
        rga_reqs class of the spinal_units.py file.        

        In addition, units using this type of synapse need to have a
        'custom_inp_del' attribute to indicate the extra delay steps in the 
        'lateral' inputs. The synapse will use this delay for the activities of
        its postsynaptic unit and for the 'lateral' inputs.

        The current implementation normalizes the sum of the absolute values for
        the weights at the 'error' port, making them add to a parameter 'w_sum'
        times the sum of l0_norm_factor and the out_norm_factor of the 
        presynaptic unit.
        
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
            'w_sum' : multiplies the sum of weight values at the error
                      port. Default is 1.

        Raises:
            AssertionError.
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor to scales the update rule
        # most of the heavy lifting is done by requirements
        self.upd_requirements = set([syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_mid, 
                             syn_reqs.lpf_fast, syn_reqs.lpf_mid, 
                             syn_reqs.inp_deriv_mp, syn_reqs.avg_inp_deriv_mp,
                             syn_reqs.del_inp_deriv_mp,
                             syn_reqs.del_avg_inp_deriv_mp,
                             syn_reqs.l0_norm_factor_mp,
                             syn_reqs.pre_out_norm_factor])
        assert self.type is synapse_types.gated_rga, ['Synapse from ' + str(self.preID) + 
                   ' to ' + str(self.postID) + ' instantiated with the wrong type']
        if not hasattr(self.net.units[self.postID], 'custom_inp_del'):
            raise AssertionError('A gated_rga synapse has a postsynaptic unit without ' +
                                 'the custom_inp_del attribute')
        # po_de is the delay in postsynaptic activity for the learning rule
        # It is set to match the delay in the 'lateral' input ports of the post unit
        self.po_de = self.net.units[self.postID].custom_inp_del
        if 'lat_port' in params: self.lat_port = params['lat_port']
        else: self.lat_port = 1 
        if 'err_port' in params: self.err_port = params['err_port']
        else: self.err_port = 0 
        if 'w_sum' in params: self.w_sum = params['w_sum']
        else: self.w_sum = 1.
        
    def update(self, time):
        """ Update the weight using the RGA-inspired learning rule.
        
            If the network is correctly initialized, the pre-synaptic unit 
            updates lpf_fast, and lpf_mid, whereas the pbly ost-synaptic unit
            updates lpf_fast, lpf_mid, acc_mid, and the average of 
            approximate input derivatives for each port.

            Notice the average of input derivatives can come form 
            upd_pos_diff_avg, which considers only the inputs whose synapses
            have positive values. To allow synapses to potentially become negative,
            you need to change the synapse requirement in __init__ from 
            pos_diff_avg to diff_avg, and the pos_diff_avg value used below to 
            diff_avg.
            Also, remove the line "if self.w < 0: self.w = 0"
        """
        u = self.net.units[self.postID]
        xp = u.del_avg_inp_deriv_mp[self.lat_port]
        up = u.get_lpf_fast(self.po_de) - u.get_lpf_mid(self.po_de)
        sp = u.avg_inp_deriv_mp[self.err_port]
        pre = self.net.units[self.preID]
        spj = (pre.get_lpf_fast(self.delay_steps) -
               pre.get_lpf_mid(self.delay_steps) )
        self.w *= self.w_sum*(u.l0_norm_factor_mp[self.err_port] + 
                              pre.out_norm_factor)
        #self.w += u.acc_mid * self.alpha * (up - xp) * (sp - spj)
        self.w += u.acc_slow * self.alpha * (up - xp) * (sp - spj)


class rga_ge(synapse):
    """ The RGA rule modulated by the derivative of the global error. 
    
        The particulars of RGA are in the 4/11/19 scrap sheet.
        The global error first came up in the "A treatment of RGA" notes.

        Presynaptic units are given the lpf_fast and lpf_mid requirements.

        Postsynaptic units are given lpf_fast, lpf_mid, inp_deriv_mp, 
        avg_inp_deriv_mp, del_inp_deriv_mp, del_avg_inp_deriv_mp, 
        l0_norm_factor_mp, and pre_out_norm_factor requirements. 
        
        The update methods for most of these requirements are currently in the
        rga_reqs class of the spinal_units.py file.        

        In addition, units using this type of synapse need to have a
        'custom_inp_del' attribute to indicate the extra delay steps in the 
        'lateral' inputs. The synapse will use this delay for the activities of
        its postsynaptic unit and for the 'lateral' inputs.

        The current implementation normalizes the sum of the absolute values for
        the weights at the 'error' port, making them add to 1.
    """
    def __init__(self, params, network):
        """ The class constructor.

        In its current implementation, the rga synapse assumes that the lateral
        connections are in port 1 of the unit, wheras the error inputs are in
        port 0, and the global error is in port 2.

        This is set in the lat_port, err_port, and ge_port variables.
        
        Args:
            params: same as the parent class, with two additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            'post_delay': NOT USING. delay steps in the post-synaptic activity.
            OPTIONAL PARAMETERS
            'err_port' : port for "error" inputs. Default is 0.
            'lat_port' : port for "lateral" inputs. Default is 1.
            'ge_port' : port for "global error" inputs. Default is 2.

        Raises:
            AssertionError.
        """
        #TODO: check if requirements still needed
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor to scales the update rule
        # most of the heavy lifting is done by requirements
        self.upd_requirements = set([syn_reqs.pre_lpf_fast,
                             syn_reqs.pre_lpf_mid, 
                             syn_reqs.pre_lpf_slow, # experimental
                             syn_reqs.del_inp_mp, # testing
                             syn_reqs.del_inp_avg_mp, # testing
                             syn_reqs.lpf_slow, # testing
                             syn_reqs.mp_inputs, # testing
                             syn_reqs.mp_weights, # testing
                             syn_reqs.inp_avg_mp, # testing
                             syn_reqs.lpf_fast,
                             syn_reqs.lpf_mid, 
                             syn_reqs.inp_deriv_mp, 
                             syn_reqs.avg_inp_deriv_mp,
                             syn_reqs.del_inp_deriv_mp,
                             syn_reqs.del_avg_inp_deriv_mp,
                             syn_reqs.sc_inp_sum_deriv_mp,
                             syn_reqs.l0_norm_factor_mp,
                             syn_reqs.pre_out_norm_factor])
        assert self.type is synapse_types.rga_ge, ['Synapse from ' + str(self.preID) + 
                   ' to ' + str(self.postID) + ' instantiated with the wrong type']
        if not hasattr(self.net.units[self.postID], 'custom_inp_del'):
            raise AssertionError('An rga_ge synapse has a postsynaptic unit ' +
                                 'without the custom_inp_del attribute')
        # po_de is the delay in postsynaptic activity for the learning rule
        # It is set to match the delay in the 'lateral' input ports of the post unit
        self.po_de = self.net.units[self.postID].custom_inp_del
        if 'lat_port' in params: self.lat_port = params['lat_port']
        else: self.lat_port = 1 
        if 'err_port' in params: self.err_port = params['err_port']
        else: self.err_port = 0 
        if 'ge_port' in params: self.err_port = params['ge_port']
        else: self.ge_port = 2 
        self.gep_slow = 0. # used to obtain the error's second derivative
        self.xp_slow = 0. # used to obtain the error's second derivative
        
    def update(self, time):
        """ Update the weight using the RGA-inspired learning rule.
        
            If the network is correctly initialized, the pre-synaptic unit 
            updates lpf_fast, and lpf_mid, whereas the post-synaptic unit
            updates lpf_fast, lpf_mid, and the average of approximate input
            derivatives for each port.

        """
        #TODO: move calculations into unit requirements
        u = self.net.units[self.postID]
        xp = u.del_avg_inp_deriv_mp[self.lat_port]
        up = u.get_lpf_fast(self.po_de) - u.get_lpf_mid(self.po_de)
        upp = up - u.get_lpf_mid(self.po_de) + u.get_lpf_slow(self.po_de)
        #sp = u.avg_inp_deriv_mp[self.err_port]
        #sp = u.del_avg_inp_deriv_mp[self.err_port]
        pre = self.net.units[self.preID]
        #spj = (pre.get_lpf_fast(self.delay_steps) -
        #       pre.get_lpf_mid(self.delay_steps) )

        norm_fac = .5*(u.l0_norm_factor_mp[self.err_port] + pre.out_norm_factor)
        self.w += self.alpha * (norm_fac - 1.)*self.w # multiplicative?

        gep = u.inp_deriv_mp[self.ge_port][0] # only one GE input
        self.gep_slow += 10.*self.net.min_delay*(gep - self.gep_slow)
        gepp = gep - self.gep_slow
        
        #self.xp_slow += 0.05*(xp - self.xp_slow)
        #xpp = xp - self.xp_slow

        #self.w += self.alpha * (up - xp) * (sp - spj) # normal rga
        #self.w += -self.alpha * gep * (up - xp) * (sp - spj) # modulated rga
        #self.w += self.alpha * ((up - xp) * (sp - spj) - gep*self.w*spj)
        #self.w += self.alpha * ((up - xp) * (sp - spj) - gep*spj)
        u_act = u.act_buff[-1-self.po_de]
        #pre_act = pre.act_buff[-1-self.po_de]
        #self.w -= gep * u_act * pre_act # Hebbian-descent simile
        #self.w += self.alpha * ((up - xp) * (sp - spj) - gep*u_act*spj)
        #up_del = (u.get_lpf_fast(self.po_de) -
        #          u.get_lpf_mid(self.po_de) )
        #spj = (pre.get_lpf_fast(self.po_de+self.delay_steps) -
        #       pre.get_lpf_slow(self.po_de+self.delay_steps) )
        #sj = pre.act_buff[-1-self.po_de-self.delay_steps]
        sj = pre.act_buff[-1-self.delay_steps]
        #s = sum(u.del_inp_mp[self.err_port])/len(u.del_inp_mp[self.err_port])
        #s = u.del_inp_avg_mp[self.err_port]
        s = u.inp_avg_mp[self.err_port]
        #self.w -= self.alpha*gep*u_act*(spj - sp)
        #self.w -= self.alpha*gep*u_act*(sj - s)
        #self.w -= self.alpha*(gep*up + gepp*upp)*(sj - s)
        # see 05/07/20 scrap notes
        #self.w -= self.alpha * (gep*(up-xp) + gepp*upp)*(sj - s)
        #self.w -= self.alpha * (gep*(up-xp) + gepp*(upp-xpp))*(sj - s)
        #self.w -= self.alpha * ((gep*(up-xp) + gepp*(upp-xpp))*
        #                        (sj - s + spj - sp))
        #xp_now = u.avg_inp_deriv_mp[self.lat_port]
        xp_now = u.sc_inp_sum_deriv_mp[self.lat_port]
        self.w -= self.alpha * ((gep*(up-xp) + gepp*upp)*(sj - s) -
                  up * xp_now)

class normal_rga(synapse):
    """ A version of the rga rule with divisive normalization.

        This class is a copy of the gated_normal_rga class, with the 'gating'
        removed from the update rule; e.g. the postsynaptic units don't need 
        the acc_slow requirement, which is no longer in the udpate equation.

        The RGA rule is described in the 4/11/19 scrap sheet.
        The idea of normalizing the correlations is described in the "Thoughts
        on RGA" note. 

        The normalization used is of the type:
        x* = x / (sigma + <x>),
        where x* is the normalized version of x, sigma is a constant, and <x> is
        the low-pass filtered version of x.

        It is important to notice that the "average derivative" value used for
        normalization will be obtained by obtaining the derivative of the
        presynaptic input using lpf_mid and lpf_slow rather than lpf_fast and
        lpf_mid. The tau_mid and tau_slow parameters of the presynatptic unit
        should be adjusted with this in mind.

        Presynaptic units are given the lpf_fast, lpf_mid, and lpf_slow
        requirements.

        Postsynaptic units are given lpf_fast, lpf_mid, inp_deriv_mp, 
        avg_inp_deriv_mp, del_inp_deriv_mp, del_avg_inp_deriv_mp, 
        l0_norm_factor_mp, pre_out_norm_factor, slow_inp_deriv_mp, and
        slow_avg_inp_deriv_mp requirements. 
        
        The update methods for most of these requirements are currently in the
        rga_reqs class of the spinal_units.py file.        

        In addition, units using this type of synapse need to have a
        'custom_inp_del' attribute to indicate the extra delay steps in the 
        'lateral' inputs. The synapse will use this delay for the activities of
        its postsynaptic unit and for the 'lateral' inputs.

        The current implementation normalizes the sum of the absolute values for
        the weights at the 'error' port, making them add to a parameter 'w_sum'
        times the sum of l0_norm_factor and the out_norm_factor of the 
        presynaptic unit.
        
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
            'w_sum' : multiplies the sum of weight values at the error
                      port. Default is 1.
            'sig1' : sigma value for postsynaptic normalization. Default is 1.
            'sig2' : sigma value for presynaptic normalization. Default is 1.

        Raises:
            AssertionError.
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor to scales the update rule
        # most of the heavy lifting is done by requirements
        self.upd_requirements = set([
                             syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_mid, 
                             syn_reqs.pre_lpf_slow,
                             syn_reqs.lpf_fast, syn_reqs.lpf_mid,
                             syn_reqs.lpf_slow,
                             syn_reqs.inp_deriv_mp, syn_reqs.avg_inp_deriv_mp,
                             syn_reqs.del_inp_deriv_mp,
                             syn_reqs.del_avg_inp_deriv_mp,
                             syn_reqs.slow_inp_deriv_mp, 
                             syn_reqs.avg_slow_inp_deriv_mp,
                             syn_reqs.l0_norm_factor_mp,
                             syn_reqs.pre_out_norm_factor])
        assert self.type is synapse_types.normal_rga, ['Synapse from ' + 
                            str(self.preID) + ' to ' + str(self.postID) + 
                            ' instantiated with the wrong type']
        if not hasattr(self.net.units[self.postID], 'custom_inp_del'):
            raise AssertionError('A normal_rga synapse has a postsynaptic' +
                                 ' unit without the custom_inp_del attribute')
        # po_de is the delay in postsynaptic activity for the learning rule
        # It is set to match the delay in the 'lateral' input ports of the post unit
        self.po_de = self.net.units[self.postID].custom_inp_del
        if 'lat_port' in params: self.lat_port = params['lat_port']
        else: self.lat_port = 1 
        if 'err_port' in params: self.err_port = params['err_port']
        else: self.err_port = 0 
        if 'w_sum' in params: self.w_sum = params['w_sum']
        else: self.w_sum = 1.
        if 'sig1' in params: self.sig1 = params['sig1']
        else: self.sig1 = 1.
        if 'sig2' in params: self.sig2 = params['sig2']
        else: self.sig2 = 1.
        # add_slow_inp_deriv_mp will add the sid_idx attribute, which
        # is the index of this synapse in the (avg_)slow_inp_deriv_mp lists.
        
    def update(self, time):
        """ Update the weight using the gated_normal_rga learning rule.
        
            If the network is correctly initialized, the pre-synaptic unit 
            updates lpf_fast, lpf_mid, and lpf_slow, whereas the post-synaptic
            unit updates lpf_fast, lpf_mid, lpf_slow, acc_mid, and the average 
            of approximate input derivatives for each port.

            Notice the average of input derivatives may come form 
            upd_pos_diff_avg, which considers only the inputs whose synapses
            have positive values. To allow synapses to potentially become negative,
            you need to change the synapse requirement in __init__ from 
            pos_diff_avg to diff_avg, and the pos_diff_avg value used below to 
            diff_avg.
            Also, remove the line "if self.w < 0: self.w = 0"
        """
        u = self.net.units[self.postID]
        pre = self.net.units[self.preID]
        # normalizing factors
        up_slow = u.get_lpf_mid(self.po_de) - u.get_lpf_slow(self.po_de)
        normfac1 = 1. / (self.sig1 + abs(up_slow))
        avg_normfac1 = 1. / (self.sig1 + abs(u.avg_slow_inp_deriv_mp[self.lat_port]))
        normfac2 = 1. / (self.sig2 +
                   abs(u.slow_inp_deriv_mp[self.err_port][self.sid_idx]))
        avg_normfac2 = 1. / (self.sig2 + abs(u.avg_slow_inp_deriv_mp[self.err_port]))
        # correlation factors
        up = normfac1 * (u.get_lpf_fast(self.po_de) -
                              u.get_lpf_mid(self.po_de))
        xp = avg_normfac1 * u.del_avg_inp_deriv_mp[self.lat_port]
        sp = avg_normfac2 * u.avg_inp_deriv_mp[self.err_port]
        spj = normfac2 * (pre.get_lpf_fast(self.delay_steps) -
                          pre.get_lpf_mid(self.delay_steps) )
        # weight normalization
        norm_fac = .5*(u.l0_norm_factor_mp[self.err_port] + pre.out_norm_factor)
        self.w += self.alpha * (norm_fac - 1.)*self.w
        # weight update
        self.w += self.alpha * (up - xp) * (sp - spj)


class gated_normal_rga(synapse):
    """ A version of gated_rga using divisive normalization.

        The RGA rule is described in the 4/11/19 scrap sheet.
        The idea of normalizing the correlations is described in the "Thoughts
        on RGA" note. 

        The normalization used is of the type:
        x* = x / (sigma + <x>),
        where x* is the normalized version of x, sigma is a constant, and <x> is
        the low-pass filtered version of x.

        It is important to notice that the "average derivative" value used for
        normalization will be obtained by obtaining the derivative of the
        presynaptic input using lpf_mid and lpf_slow rather than lpf_fast and
        lpf_mid. The tau_mid and tau_slow parameters of the presynatptic unit
        should be adjusted with this in mind.

        Presynaptic units are given the lpf_fast, lpf_mid, and lpf_slow
        requirements.

        Postsynaptic units are given lpf_fast, lpf_mid, inp_deriv_mp, 
        avg_inp_deriv_mp, del_inp_deriv_mp, del_avg_inp_deriv_mp, 
        l0_norm_factor_mp, pre_out_norm_factor, slow_inp_deriv_mp, and
        slow_avg_inp_deriv_mp requirements. 
        Postsynaptic units are also expected to include the acc_slow
        requirement, which is used to modulate the learning rate.
        
        The update methods for most of these requirements are currently in the
        rga_reqs class of the spinal_units.py file.        

        In addition, units using this type of synapse need to have a
        'custom_inp_del' attribute to indicate the extra delay steps in the 
        'lateral' inputs. The synapse will use this delay for the activities of
        its postsynaptic unit and for the 'lateral' inputs.

        The current implementation normalizes the sum of the absolute values for
        the weights at the 'error' port, making them add to a parameter 'w_sum'
        times the sum of l0_norm_factor and the out_norm_factor of the 
        presynaptic unit.
        
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
            'w_sum' : multiplies the sum of weight values at the error
                      port. Default is 1.
            'sig1' : sigma value for postsynaptic normalization. Default is 1.
            'sig2' : sigma value for presynaptic normalization. Default is 1.

        Raises:
            AssertionError.
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor to scales the update rule
        # most of the heavy lifting is done by requirements
        self.upd_requirements = set([
                             syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_mid, 
                             syn_reqs.pre_lpf_slow,
                             syn_reqs.lpf_fast, syn_reqs.lpf_mid,
                             syn_reqs.lpf_slow,
                             syn_reqs.inp_deriv_mp, syn_reqs.avg_inp_deriv_mp,
                             syn_reqs.del_inp_deriv_mp,
                             syn_reqs.del_avg_inp_deriv_mp,
                             syn_reqs.slow_inp_deriv_mp, 
                             syn_reqs.avg_slow_inp_deriv_mp,
                             syn_reqs.l0_norm_factor_mp,
                             syn_reqs.pre_out_norm_factor])
        assert self.type is synapse_types.gated_normal_rga, ['Synapse from ' + 
                            str(self.preID) + ' to ' + str(self.postID) + 
                            ' instantiated with the wrong type']
        if not hasattr(self.net.units[self.postID], 'custom_inp_del'):
            raise AssertionError('A gated_normal_rga synapse has a postsynaptic' +
                                 ' unit without the custom_inp_del attribute')
        # po_de is the delay in postsynaptic activity for the learning rule
        # It is set to match the delay in the 'lateral' input ports of the post unit
        self.po_de = self.net.units[self.postID].custom_inp_del
        if 'lat_port' in params: self.lat_port = params['lat_port']
        else: self.lat_port = 1 
        if 'err_port' in params: self.err_port = params['err_port']
        else: self.err_port = 0 
        if 'w_sum' in params: self.w_sum = params['w_sum']
        else: self.w_sum = 1.
        if 'sig1' in params: self.sig1 = params['sig1']
        else: self.sig1 = 1.
        if 'sig2' in params: self.sig2 = params['sig2']
        else: self.sig2 = 1.
        # add_slow_inp_deriv_mp will add the sid_idx attribute, which
        # is the index of this synapse in the (avg_)slow_inp_deriv_mp lists.
        
    def update(self, time):
        """ Update the weight using the gated_normal_rga learning rule.
        
            If the network is correctly initialized, the pre-synaptic unit 
            updates lpf_fast, lpf_mid, and lpf_slow, whereas the post-synaptic
            unit updates lpf_fast, lpf_mid, lpf_slow, acc_mid, and the average 
            of approximate input derivatives for each port.

            Notice the average of input derivatives may come form 
            upd_pos_diff_avg, which considers only the inputs whose synapses
            have positive values. To allow synapses to potentially become negative,
            you need to change the synapse requirement in __init__ from 
            pos_diff_avg to diff_avg, and the pos_diff_avg value used below to 
            diff_avg.
            Also, remove the line "if self.w < 0: self.w = 0"
        """
        u = self.net.units[self.postID]
        pre = self.net.units[self.preID]
        # normalizing factors
        up_slow = u.get_lpf_mid(self.po_de) - u.get_lpf_slow(self.po_de)
        normfac1 = 1. / (self.sig1 + abs(up_slow))
        avg_normfac1 = 1. / (self.sig1 + abs(u.avg_slow_inp_deriv_mp[self.lat_port]))
        normfac2 = 1. / (self.sig2 +
                   abs(u.slow_inp_deriv_mp[self.err_port][self.sid_idx]))
        avg_normfac2 = 1. / (self.sig2 + abs(u.avg_slow_inp_deriv_mp[self.err_port]))
        # correlation factors
        up = normfac1 * (u.get_lpf_fast(self.po_de) -
                              u.get_lpf_mid(self.po_de))
        xp = avg_normfac1 * u.del_avg_inp_deriv_mp[self.lat_port]
        sp = avg_normfac2 * u.avg_inp_deriv_mp[self.err_port]
        spj = normfac2 * (pre.get_lpf_fast(self.delay_steps) -
                          pre.get_lpf_mid(self.delay_steps) )
        # weight normalization
        self.w *= self.w_sum*(u.l0_norm_factor_mp[self.err_port] + 
                              pre.out_norm_factor)
        # weight update
        self.w += u.acc_slow * self.alpha * (up - xp) * (sp - spj)


class gated_bp_rga_synapse(synapse):
    """ The gated_rga synapse with extra 'betrayal punishing' dynamics. 

        The logic behind this rule is in the 2019/12/03 scrap note.
        The name of this synapse's enum is gated_bp_rga.

        Presynaptic units are given the lpf_fast and lpf_mid requirements.

        Postsynaptic units are given lpf_fast, lpf_mid, inp_deriv_mp, 
        avg_inp_deriv_mp, del_inp_deriv_mp, del_avg_inp_deriv_mp, 
        l0_norm_factor_mp, and pre_out_norm_factor requirements. 
        Postsynaptic units are also expected to include the acc_slow
        requirement, which is used to modulate the learning rate.
        
        The update methods for most of these requirements are currently in the
        rga_reqs class of the spinal_units.py file.        

        In addition, units using this type of synapse need to have a
        'custom_inp_del' attribute to indicate the extra delay steps in the 
        'lateral' inputs. The synapse will use this delay for the activities of
        its postsynaptic unit and for the 'lateral' inputs.

        The current implementation normalizes the sum of the absolute values for
        the weights at the 'error' port, making them add to a parameter 'w_sum'
        times the sum of l0_norm_factor and the out_norm_factor of the 
        presynaptic unit.
        
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
            'w_tau' : time constant to accumulate the weight change.
            OPTIONAL PARAMETERS
            'err_port' : port for "error" inputs. Default is 0.
            'lat_port' : port for "lateral" inputs. Default is 1.
            'w_sum' : multiplies the sum of weight values at the error
                      port. Default is 1.
            'w_thresh' : Amount of change that the synapse is not supposed to 
                         accumulate in both directions. Default is 1.
            'w_decay' : time constant for decay of the change integration.
                        Default is 0.01 .
        Raises:
            AssertionError.
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor to scales the update rule
        self.w_tau = params['w_tau'] # time constant for LPF'ing weight change
        # most of the heavy lifting is done by requirements
        self.upd_requirements = set([syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_mid, 
                             syn_reqs.lpf_fast, syn_reqs.lpf_mid, 
                             syn_reqs.inp_deriv_mp, syn_reqs.avg_inp_deriv_mp,
                             syn_reqs.del_inp_deriv_mp,
                             syn_reqs.del_avg_inp_deriv_mp,
                             syn_reqs.l0_norm_factor_mp,
                             syn_reqs.pre_out_norm_factor])
        assert self.type is synapse_types.gated_bp_rga, ['Synapse from ' + str(self.preID) + 
                   ' to ' + str(self.postID) + ' instantiated with the wrong type']
        if not hasattr(self.net.units[self.postID], 'custom_inp_del'):
            raise AssertionError('A gated_bp_rga synapse has a postsynaptic unit without ' +
                                 'the custom_inp_del attribute')
        # po_de is the delay in postsynaptic activity for the learning rule
        # It is set to match the delay in the 'lateral' input ports of the post unit
        self.po_de = self.net.units[self.postID].custom_inp_del
        if 'lat_port' in params: self.lat_port = params['lat_port']
        else: self.lat_port = 1 
        if 'err_port' in params: self.err_port = params['err_port']
        else: self.err_port = 0 
        if 'w_sum' in params: self.w_sum = params['w_sum']
        else: self.w_sum = 1.
        if 'w_thresh' in params: self.w_thresh = params['w_thresh']
        else: self.w_thresh = 1.
        if 'w_decay' in params: self.w_decay = params['w_decay']
        else: self.w_decay = 0.01
        self.delW = 0. # LPF'd change of weight
        self.corr_type = None # -1=decrease, 1=increase, 0=unreliable
        #self.w_prop = np.exp(-network.min_delay / self.w_tau) # delW propagator

        
    def update(self, time):
        """ Update the weight using the gated_bp_rga learning rule.
        
            If the network is correctly initialized, the pre-synaptic unit 
            updates lpf_fast, and lpf_mid, whereas the post-synaptic unit
            updates lpf_fast, lpf_mid, acc_mid, and the average of 
            approximate input derivatives for each port.

            Notice the average of input derivatives can come form 
            upd_pos_diff_avg, which considers only the inputs whose synapses
            have positive values. To allow synapses to potentially become negative,
            you need to change the synapse requirement in __init__ from 
            pos_diff_avg to diff_avg, and the pos_diff_avg value used below to 
            diff_avg.
            Also, remove the line "if self.w < 0: self.w = 0"
        """
        u = self.net.units[self.postID]
        xp = u.del_avg_inp_deriv_mp[self.lat_port]
        up = u.get_lpf_fast(self.po_de) - u.get_lpf_mid(self.po_de)
        sp = u.avg_inp_deriv_mp[self.err_port]
        pre = self.net.units[self.preID]
        spj = (pre.get_lpf_fast(self.delay_steps) -
            pre.get_lpf_mid(self.delay_steps) )
        delta_w = u.acc_slow * (up - xp) * (sp - spj)
        #delta_w = u.acc_slow * up * (sp - spj)  # FOR TESTING PURPOSES
        #self.delW = delta_w + (self.delW  - delta_w) * self.w_prop
        self.delW += np.abs(self.alpha) * (delta_w - self.w_decay*self.delW)
        self.w *= 0.5 * self.w_sum*(u.l0_norm_factor_mp[self.err_port] + 
                            pre.out_norm_factor)

        if self.corr_type != 0:
            self.w +=  self.alpha * delta_w
            if self.corr_type == -1:  # synapse was decreasing
                if self.delW > self.w_thresh: # synapse is now increasing
                    self.corr_type = 0  # betrayal!
            elif self.corr_type == 1:  # synapse was increasing
                if self.delW < -self.w_thresh: # synapse is now decreasing 
                    self.corr_type = 0  # betrayal!
            else:  # self.corr_type is None
                if self.delW > self.w_thresh: 
                    self.corr_type = 1  
                elif self.delW < -self.w_thresh: 
                    self.corr_type = -1  
        else:
            self.w -= 0.1 * np.abs(self.alpha) * self.w   # punishment
            if self.delW > 1.5*self.w_thresh:
                self.corr_type = 1  # redemption!    
            elif self.delW < -2.*self.w_thresh:
                self.corr_type = -1  # redemption!    


class gated_rga_diff_synapse(synapse):
    """ A variation of the gated_rga using two separate time delays.

        The RGA rule is described in the 4/11/19 scrap sheet.
        The variation using the difference of two RGA-like rules is described in
        the "Further RGA changes" scrap sheet dated 12/18/1019.

        Presynaptic units are given the lpf_fast and lpf_mid requirements.

        Postsynaptic units are given lpf_fast, lpf_mid, inp_deriv_mp, 
        avg_inp_deriv_mp, double_del_inp_deriv_mp, double_del_avg_inp_deriv_mp,
        l0_norm_factor_mp, and pre_out_norm_factor requirements. 
        Postsynaptic units are also expected to include the acc_slow
        requirement, which is used to modulate (gate) the learning rate of this
        synapse. 

        The update methods for most of these requirements are currently in the
        rga_reqs class of the spinal_units.py file.        

        In addition, units using this type of synapse need to have 
        'custom_inp_del' and 'custom_inp_del2' attributes to indicate the two
        delays in the 'lateral' inputs. These delays are expressed as number of
        time steps, and the synapse will use them for the activities of its 
        postsynaptic unit and for the 'lateral' inputs. It is expected that
        custom_inp_del2 > custom_inp_del.

        The current implementation normalizes the sum of the absolute values for
        the weights at the 'error' port, making them add to a parameter 'w_sum'
        times the sum of l0_norm_factor and the out_norm_factor of the 
        presynaptic unit.
        
    """
    def __init__(self, params, network):
        """ The class constructor.

        In its default implementation, the rga synapse assumes that the lateral
        connections are in port 1 of the unit, wheras the error inputs are in port 0.
        This is set in the lat_port and err_port variables.
        
        Args:
            params: same as the parent class, with two additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            OPTIONAL PARAMETERS
            'err_port' : port for "error" inputs. Default is 0.
            'lat_port' : port for "lateral" inputs. Default is 1.
            'w_sum' : multiplies the sum of weight values at the error
                      port. Default is 1.

        Raises:
            AssertionError.
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor to scale the update rule
        # most of the heavy lifting is done by requirements
        self.upd_requirements = set([syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_mid, 
                             syn_reqs.lpf_fast, syn_reqs.lpf_mid, 
                             syn_reqs.inp_deriv_mp, syn_reqs.avg_inp_deriv_mp,
                             syn_reqs.double_del_inp_deriv_mp,
                             syn_reqs.double_del_avg_inp_deriv_mp,
                             syn_reqs.l0_norm_factor_mp,
                             syn_reqs.pre_out_norm_factor])
        assert self.type is synapse_types.gated_rga_diff, ['Synapse from ' +
                            str(self.preID) + ' to ' + str(self.postID) + 
                            ' instantiated with the wrong type']
        u = self.net.units[self.postID]
        if not (hasattr(u, 'custom_inp_del') and hasattr(u, 'custom_inp_del2')):
            raise AssertionErrer('A gated_rga_diff synapse has a postsynaptic' +
                                 'unit without the custom_inp_del(2) attribute')
        # po_de is the delay in postsynaptic activity for the learning rule.
        # It is set to match the delay in the 'lateral' input ports of the post unit
        # pre_de is the delay used for the inputs at the 'error' ports.
        self.po_de = u.custom_inp_del2
        self.pre_de = u.custom_inp_del2 - u.custom_inp_del
        if 'lat_port' in params: self.lat_port = params['lat_port']
        else: self.lat_port = 1 
        if 'err_port' in params: self.err_port = params['err_port']
        else: self.err_port = 0 
        if self.port != self.err_port:
            raise AssertionError('The gated_rga_diff synapses are meant to ' +
                                 'be connected at the error ports')
        if 'w_sum' in params: self.w_sum = params['w_sum']
        else: self.w_sum = 1.
        # The add_double_del_inp_deriv_mp method of requirements.py will add the
        # ddidm_idx attribute, which is the index of this synapse in the
        # double_del_(avg)_inp_deriv_mp lists.
        """
        # we now need to find the index of the input with this synapse in the
        # double_del_inp_deriv_mp lists.
        for idx, iid in enumerate(self.net.units[self.postID].port_idx[self.port]):
            if network.syns[self.postID][iid].preID == self.preID:
                self.ddidm_idx = idx
                break
        else:
            raise AssertionError('The gated_rga_diff constructor could not ' +
                  'find the index of its synapse in double_del_inp_deriv_mp')
        """
        
    def update(self, time):
        """ Update the weight using the gated_rga_diff learning rule.
        
            If the network is correctly initialized, the pre-synaptic unit 
            updates lpf_fast, and lpf_mid, whereas the post-synaptic unit
            updates lpf_fast, lpf_mid, acc_mid, and the average of 
            approximate input derivatives for each port.

            The average of the delayed input derivatives comes from the
            double_del_avg_inp_deriv_mp requirement.

            The current rule allows synapses to become negative.
        """
        u = self.net.units[self.postID]
        pre = self.net.units[self.preID]
        xp = u.double_del_avg_inp_deriv_mp[1][self.lat_port]
        up = u.get_lpf_fast(self.po_de) - u.get_lpf_mid(self.po_de)
        sp_now = u.avg_inp_deriv_mp[self.err_port]
        sp_del = u.double_del_avg_inp_deriv_mp[0][self.err_port]
        spj_now = (pre.get_lpf_fast(self.delay_steps) -
                   pre.get_lpf_mid(self.delay_steps) )
        spj_del = u.double_del_inp_deriv_mp[0][self.port][self.ddidm_idx]
        self.w *= self.w_sum*(u.l0_norm_factor_mp[self.err_port] + 
                              pre.out_norm_factor)
        self.w += u.acc_slow * self.alpha * (up - xp) * (
                              (sp_now - spj_now) - (sp_del - spj_del))
        # next line adds some random drift
                  #+ 0.0002*(np.random.random()-0.5))


class gated_slide_rga_diff(synapse):
    """ A variation of gated_rga_diff with sliding time delays.

        The RGA rule is described in the 4/11/19 scrap sheet.
        The variation using the difference of two RGA-like rules is described in
        the "Further RGA changes" scrap sheet dated 12/18/1019.
        The variation with sliding delays is described in the "Sliding the
        delay in the RGA rule" scrap sheet of 3/11/20.

        Presynaptic units are given the lpf_fast and lpf_mid requirements.

        Postsynaptic units are given lpf_fast, lpf_mid, inp_deriv_mp, 
        avg_inp_deriv_mp, double_del_inp_deriv_mp, double_del_avg_inp_deriv_mp,
        l0_norm_factor_mp, and pre_out_norm_factor requirements. 
        Postsynaptic units are also expected to include the acc_slow
        requirement, which is used to modulate (gate) the learning rate of this
        synapse. 

        The update methods for most of these requirements are currently in the
        rga_reqs class of the spinal_units.py file.        

        In addition, units using this type of synapse need to have 
        'custom_inp_del' and 'custom_inp_del2' attributes to indicate the
        "central" values of the two delays in the 'lateral' inputs. 
        These delays are expressed as number of time steps, and the synapse 
        will use them for the activities of its postsynaptic unit and for the
        'lateral' inputs. It is expected that custom_inp_del2 > custom_inp_del.

        Another requirement for units using this type of synapse are
        "del_mod_max" and "del_mod_min" constants, indicating respectively the
        maximumn and minimum change to the delays for all synapses. These are
        expressed as number of simulation time steps.

        The current implementation normalizes the sum of the absolute values for
        the weights at the 'error' port, making them add to a parameter 'w_sum'
        times the sum of l0_norm_factor and the out_norm_factor of the 
        presynaptic unit.
        
    """
    def __init__(self, params, network):
        """ The class constructor.

        In its default implementation, the rga synapse assumes that the lateral
        connections are in port 1 of the unit, wheras the error inputs are in port 0.
        This is set in the lat_port and err_port variables.
        
        Args:
            params: same as the parent class, with two additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            'del_mod_tau' : time constant for the delay modifier
            OPTIONAL PARAMETERS
            'err_port' : port for "error" inputs. Default is 0.
            'lat_port' : port for "lateral" inputs. Default is 1.
            'w_sum' : multiplies the sum of weight values at the error
                      port. Default is 1.

        Raises:
            AssertionError.
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor to scale the update rule
        self.del_mod_tau = params['del_mod_tau'] # time constant for delay modifier
        self.dm_alpha = self.net.min_delay / self.del_mod_tau
        u = self.net.units[self.postID]
        self.corr_alpha = self.net.min_delay / u.tau_slow
        # most of the heavy lifting is done by requirements
        self.upd_requirements = set([syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_mid, 
                             syn_reqs.lpf_fast, syn_reqs.lpf_mid, 
                             syn_reqs.inp_deriv_mp, syn_reqs.avg_inp_deriv_mp,
                             syn_reqs.double_del_inp_deriv_mp,
                             syn_reqs.double_del_avg_inp_deriv_mp,
                             syn_reqs.l0_norm_factor_mp,
                             syn_reqs.pre_out_norm_factor])
        assert self.type is synapse_types.gated_slide_rga_diff, ['Synapse from ' +
                            str(self.preID) + ' to ' + str(self.postID) + 
                            ' instantiated with the wrong type']
        if not (hasattr(u, 'custom_inp_del') and hasattr(u, 'custom_inp_del2')):
            raise AssertionErrer('A gated_slide_rga_diff synapse has a postsynaptic' +
                                 'unit without the custom_inp_del(2) attribute')
        if not (hasattr(u, 'del_mod_max') and hasattr(u, 'del_mod_min')):
            raise AssertionErrer('A gated_slide_rga_diff synapse has a postsynaptic' +
                                 'unit without one or both del_mod attributes')
        # po_de is the delay in postsynaptic activity for the learning rule.
        # It is set to match the delay in the 'lateral' input ports of the post unit
        self.po_de = u.custom_inp_del2
        # we will have versions of the maximum and minimum delay modifiers
        # expressed in the simulation's time units, rather than as number of
        # simulation time steps
        self.mod_max = u.del_mod_max*self.net.min_delay
        self.mod_min = u.del_mod_min*self.net.min_delay
        self.del_mod = 0. # initializing the delay modifier
        self.corr1 = 0.  # LPF'd correlation with custom_inp_del2 - custom_inp_del
        self.corr2 = 0.  # LPF'd correlation with custom_inp_del2
        if 'lat_port' in params: self.lat_port = params['lat_port']
        else: self.lat_port = 1 
        if 'err_port' in params: self.err_port = params['err_port']
        else: self.err_port = 0 
        if self.port != self.err_port:
            raise AssertionError('The gated_rga_diff synapses are meant to ' +
                                 'be connected at the error ports')
        if 'w_sum' in params: self.w_sum = params['w_sum']
        else: self.w_sum = 1.
        # The add_double_del_inp_deriv_mp method of requirements.py will add the
        # ddidm_idx attribute, which is the index of this synapse in the
        # double_del_(avg)_inp_deriv_mp lists.
        """
        # we now need to find the index of the input with this synapse in the
        # double_del_inp_deriv_mp lists.
        for idx, iid in enumerate(self.net.units[self.postID].port_idx[self.port]):
            if network.syns[self.postID][iid].preID == self.preID:
                self.ddidm_idx = idx
                break
        else:
            raise AssertionError('The gated_rga_diff constructor could not ' +
                  'find the index of its synapse in double_del_inp_deriv_mp')
        """
        
    def update(self, time):
        """ Update the weight using the gated_slide_rga_diff learning rule.
        
            If the network is correctly initialized, the pre-synaptic unit 
            updates lpf_fast, and lpf_mid, whereas the post-synaptic unit
            updates lpf_fast, lpf_mid, acc_mid, and the average of 
            approximate input derivatives for each port.

            The average of the delayed input derivatives comes from the
            double_del_avg_inp_deriv_mp requirement.

            The current rule allows synapses to become negative.
        """
        dm_steps = int(round(self.del_mod/self.net.min_delay))
        po_de = self.po_de + dm_steps # effective postsynaptic delay
        post = self.net.units[self.postID]
        pre = self.net.units[self.preID]
        xp = post.double_del_avg_inp_deriv_mp[1][self.lat_port]
        up = post.get_lpf_fast(po_de) - post.get_lpf_mid(po_de)
        sp_now = post.avg_inp_deriv_mp[self.err_port]
        sp_del = post.double_del_avg_inp_deriv_mp[0][self.err_port]
        spj_now = (pre.get_lpf_fast(self.delay_steps) -
                   pre.get_lpf_mid(self.delay_steps) )
        spj_del = post.double_del_inp_deriv_mp[0][self.port][self.ddidm_idx]
        # weight normalization
        self.w *= self.w_sum*(post.l0_norm_factor_mp[self.err_port] + 
                              pre.out_norm_factor)
        # delay update
        corr1 = (up - xp) * (sp_del - spj_del)
        corr2 = (up - xp) * (sp_now - spj_now)
        self.corr1 += self.corr_alpha * (corr1 - self.corr1)
        self.corr2 += self.corr_alpha * (corr2 - self.corr2)
        self.del_mod += ( (self.mod_max - self.del_mod) *
                          (self.del_mod - self.mod_min) *
                          (abs(self.corr2) - abs(self.corr1)) ) * self.dm_alpha
        # weight update
        self.w += post.acc_slow * self.alpha * (corr2 - corr1)
        # next line adds some random drift
                  #+ 0.0002*(np.random.random()-0.5))


class gated_normal_rga_diff(synapse):
    """ A variation of gated_rga_diff using normalized correlations.

        The RGA rule is described in the 4/11/19 scrap sheet.
        The variation using the difference of two RGA-like rules is described in
        the "Further RGA changes" scrap sheet dated 12/18/1019.
        The idea of normalizing the correlations is described in the "Thoughts
        on RGA" note. 

        The normalization used is of the type:
        x* = x / (sigma + <x>),
        where x* is the normalized version of x, sigma is a constant, and <x> is
        the low-pass filtered version of x.

        It is important to notice that the "average derivative" value used for
        normalization will be obtained by obtaining the derivative of the
        presynaptic input using lpf_mid and lpf_slow rather than lpf_fast and
        lpf_mid. The tau_mid and tau_slow parameters of the presynatptic unit
        should be adjusted with this in mind.

        Presynaptic units are given the lpf_fast, lpf_mid, and lpf_slow
        requirements.

        Postsynaptic units are given lpf_fast, lpf_mid, inp_deriv_mp, 
        avg_inp_deriv_mp, double_del_inp_deriv_mp, double_del_avg_inp_deriv_mp,
        l0_norm_factor_mp, pre_out_norm_factor, slow_inp_deriv_mp, and
        slow_avg_inp_deriv_mp requirements. 
        Postsynaptic units are also expected to include the acc_slow
        requirement, which is used to modulate (gate) the learning rate of this
        synapse. 

        The update methods for most of these requirements are currently in the
        rga_reqs class of the spinal_units.py file.        

        In addition, units using this type of synapse need to have 
        'custom_inp_del' and 'custom_inp_del2' attributes to indicate the two
        delays in the 'lateral' inputs. These delays are expressed as number of
        time steps, and the synapse will use them for the activities of its 
        postsynaptic unit and for the 'lateral' inputs. It is expected that
        custom_inp_del2 > custom_inp_del.

        The current implementation normalizes the sum of the absolute values for
        the weights at the 'error' port, making them add to a parameter 'w_sum'
        times the sum of l0_norm_factor and the out_norm_factor of the 
        presynaptic unit.
        
    """
    def __init__(self, params, network):
        """ The class constructor.

        In its default implementation, the rga synapse assumes that the lateral
        connections are in port 1 of the unit, wheras the error inputs are in port 0.
        This is set in the lat_port and err_port variables.
        
        Args:
            params: same as the parent class, with two additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            OPTIONAL PARAMETERS
            'err_port' : port for "error" inputs. Default is 0.
            'lat_port' : port for "lateral" inputs. Default is 1.
            'w_sum' : multiplies the sum of weight values at the error
                      port. Default is 1.
            'sig1' : sigma value for postsynaptic normalization. Default is 1.
            'sig2' : sigma value for presynaptic normalization. Default is 1.

        Raises:
            AssertionError.
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor to scale the update rule
        # most of the heavy lifting is done by requirements
        self.upd_requirements = set(
                            [syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_mid,
                             syn_reqs.pre_lpf_slow,
                             syn_reqs.lpf_fast, syn_reqs.lpf_mid, 
                             syn_reqs.inp_deriv_mp, syn_reqs.avg_inp_deriv_mp,
                             syn_reqs.double_del_inp_deriv_mp,
                             syn_reqs.double_del_avg_inp_deriv_mp,
                             syn_reqs.slow_inp_deriv_mp, 
                             syn_reqs.avg_slow_inp_deriv_mp,
                             syn_reqs.l0_norm_factor_mp,
                             syn_reqs.pre_out_norm_factor])
        assert self.type is synapse_types.gated_normal_rga_diff, ['Synapse from ' +
                            str(self.preID) + ' to ' + str(self.postID) + 
                            ' instantiated with the wrong type']
        u = self.net.units[self.postID]
        if not (hasattr(u, 'custom_inp_del') and hasattr(u, 'custom_inp_del2')):
            raise AssertionError('A gated_normal_rga_diff synapse has a postsynaptic' +
                                 'unit without the custom_inp_del(2) attribute')
        # po_de is the delay in postsynaptic activity for the learning rule.
        # It is set to match the delay in the 'lateral' input ports of the post unit
        # pre_de is the delay used for the inputs at the 'error' ports.
        self.po_de = u.custom_inp_del2
        self.pre_de = u.custom_inp_del2 - u.custom_inp_del
        if 'lat_port' in params: self.lat_port = params['lat_port']
        else: self.lat_port = 1 
        if 'err_port' in params: self.err_port = params['err_port']
        else: self.err_port = 0 
        if self.port != self.err_port:
            raise AssertionError('The gated_rga_diff synapses are meant to ' +
                                 'be connected at the error ports')
        if 'w_sum' in params: self.w_sum = params['w_sum']
        else: self.w_sum = 1.
        if 'sig1' in params: self.sig1 = params['sig1']
        else: self.sig1 = 1.
        if 'sig2' in params: self.sig2 = params['sig2']
        else: self.sig2 = 1.
        # The add_double_del_inp_deriv_mp method of requirements.py will add the
        # ddidm_idx attribute, which is the index of this synapse in the
        # double_del_(avg)_inp_deriv_mp lists.
        # Similarly, add_slow_inp_deriv_mp will add the sid_idx attribute, which
        # is the index of this synapse in the (avg_)slow_inp_deriv_mp lists.
        # sid_idx and ddidm_idx should be equal.
                
    def update(self, time):
        """ Update the weight using the gated_normal_rga_diff learning rule.
        
            The average of the delayed input derivatives comes from the
            double_del_avg_inp_deriv_mp requirement. The time-average of the
            presynaptic derivatives comes from approximating the rate of change
            using (lpf_mid - lpf_slow) rather than (lpf_fast - lpf_mid).

            The current rule allows synapses to become negative.
        """
        u = self.net.units[self.postID]
        pre = self.net.units[self.preID]

        up_slow = u.get_lpf_mid(self.po_de) - u.get_lpf_slow(self.po_de)
        normfac1 = 1. / (self.sig1 + abs(up_slow))
        up_norm = normfac1 * (u.get_lpf_fast(self.po_de) -
                              u.get_lpf_mid(self.po_de))
        avg_normfac1 = 1. / (self.sig1 + abs(u.avg_slow_inp_deriv_mp[self.lat_port]))
        xp_norm = avg_normfac1 * u.double_del_avg_inp_deriv_mp[1][self.lat_port]

        normfac2 = 1. / (self.sig2 +
                   abs(u.slow_inp_deriv_mp[self.err_port][self.sid_idx]))
        avg_normfac2 = 1. / (self.sig2 + abs(u.avg_slow_inp_deriv_mp[self.err_port]))
        sp_now =  avg_normfac2 * u.avg_inp_deriv_mp[self.err_port]
        sp_del = (avg_normfac2 * 
                  u.double_del_avg_inp_deriv_mp[0][self.err_port])
        spj_now = normfac2 * (pre.get_lpf_fast(self.delay_steps) -
                              pre.get_lpf_mid(self.delay_steps) )
        spj_del = (normfac2 * 
                   u.double_del_inp_deriv_mp[0][self.port][self.ddidm_idx])
        self.w *= self.w_sum*(u.l0_norm_factor_mp[self.err_port] + 
                              pre.out_norm_factor)
        
        self.w += u.acc_slow * self.alpha * (up_norm - xp_norm) * (
                              (sp_now - spj_now) - (sp_del - spj_del))
        # next line adds some random drift
                  #+ 0.0002*(np.random.random()-0.5))


class gated_normal_slide_rga_diff(synapse):
    """ A variation of gated_slide_rga_diff with normalized correlations.

        The RGA rule is described in the 4/11/19 scrap sheet.
        The variation using the difference of two RGA-like rules is described in
        the "Further RGA changes" scrap sheet dated 12/18/1019.
        The variation with sliding delays is described in the "Sliding the
        delay in the RGA rule" scrap sheet of 3/11/20.
        The idea of normalizing the correlations is described in the "Thoughts
        on RGA" note. 

        The normalization used is of the type:
        x* = x / (sigma + <x>),
        where x* is the normalized version of x, sigma is a constant, and <x> is
        the low-pass filtered version of x.

        It is important to notice that the "average derivative" value used for
        normalization will be obtained by obtaining the derivative of the
        presynaptic input using lpf_mid and lpf_slow rather than lpf_fast and
        lpf_mid. The tau_mid and tau_slow parameters of the presynatptic unit
        should be adjusted with this in mind.

        Presynaptic units are given the lpf_fast, lpf_mid, and lpf_slow
        requirements.

        Postsynaptic units are given lpf_fast, lpf_mid, inp_deriv_mp, 
        avg_inp_deriv_mp, double_del_inp_deriv_mp, double_del_avg_inp_deriv_mp,
        l0_norm_factor_mp, pre_out_norm_factor, slow_inp_deriv_mp, and
        slow_avg_inp_deriv_mp requirements. 
        Postsynaptic units are also expected to include the acc_slow
        requirement, which is used to modulate (gate) the learning rate of this
        synapse. 

        The update methods for most of these requirements are currently in the
        rga_reqs class of the spinal_units.py file.        

        In addition, units using this type of synapse need to have 
        'custom_inp_del' and 'custom_inp_del2' attributes to indicate the
        "central" values of the two delays in the 'lateral' inputs. 
        These delays are expressed as number of time steps, and the synapse 
        will use them for the activities of its postsynaptic unit and for the
        'lateral' inputs. It is expected that custom_inp_del2 > custom_inp_del.

        Another requirement for units using this type of synapse are
        "del_mod_max" and "del_mod_min" constants, indicating respectively the
        maximumn and minimum change to the delays for all synapses. These are
        expressed as number of simulation time steps.

        The current implementation normalizes the sum of the absolute values for
        the weights at the 'error' port, making them add to a parameter 'w_sum'
        times the sum of l0_norm_factor and the out_norm_factor of the 
        presynaptic unit.
        
    """
    def __init__(self, params, network):
        """ The class constructor.

        In its default implementation, the rga synapse assumes that the lateral
        connections are in port 1 of the unit, wheras the error inputs are in port 0.
        This is set in the lat_port and err_port variables.
        
        Args:
            params: same as the parent class, with two additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            'del_mod_tau' : time constant for the delay modifier
            OPTIONAL PARAMETERS
            'err_port' : port for "error" inputs. Default is 0.
            'lat_port' : port for "lateral" inputs. Default is 1.
            'w_sum' : multiplies the sum of weight values at the error
                      port. Default is 1.
            'sig1' : sigma value for postsynaptic normalization. Default is 1.
            'sig2' : sigma value for presynaptic normalization. Default is 1.

        Raises:
            AssertionError.
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor to scale the update rule
        self.del_mod_tau = params['del_mod_tau'] # time constant for delay modifier
        self.dm_alpha = self.net.min_delay / self.del_mod_tau
        u = self.net.units[self.postID]
        self.corr_alpha = self.net.min_delay / u.tau_slow
        # most of the heavy lifting is done by requirements
        self.upd_requirements = set(
                            [syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_mid, 
                             syn_reqs.pre_lpf_slow,
                             syn_reqs.lpf_fast, syn_reqs.lpf_mid, 
                             syn_reqs.inp_deriv_mp, syn_reqs.avg_inp_deriv_mp,
                             syn_reqs.double_del_inp_deriv_mp,
                             syn_reqs.double_del_avg_inp_deriv_mp,
                             syn_reqs.slow_inp_deriv_mp, 
                             syn_reqs.avg_slow_inp_deriv_mp,
                             syn_reqs.l0_norm_factor_mp,
                             syn_reqs.pre_out_norm_factor])
        assert self.type is synapse_types.gated_normal_slide_rga_diff, ['Synapse from ' +
                            str(self.preID) + ' to ' + str(self.postID) + 
                            ' instantiated with the wrong type']
        if not (hasattr(u, 'custom_inp_del') and hasattr(u, 'custom_inp_del2')):
            raise AssertionErrer('A gated_normal_slide_rga_diff synapse has a postsynaptic' +
                                 'unit without the custom_inp_del(2) attribute')
        if not (hasattr(u, 'del_mod_max') and hasattr(u, 'del_mod_min')):
            raise AssertionErrer('A gated_normal_slide_rga_diff synapse has a postsynaptic' +
                                 'unit without one or both del_mod attributes')
        # po_de is the delay in postsynaptic activity for the learning rule.
        # It is set to match the delay in the 'lateral' input ports of the post unit
        self.po_de = u.custom_inp_del2
        # we will have versions of the maximum and minimum delay modifiers
        # expressed in the simulation's time units, rather than as number of
        # simulation time steps
        self.mod_max = u.del_mod_max*self.net.min_delay
        self.mod_min = u.del_mod_min*self.net.min_delay
        self.del_mod = 0. # initializing the delay modifier
        self.corr1 = 0.  # LPF'd correlation with custom_inp_del2 - custom_inp_del
        self.corr2 = 0.  # LPF'd correlation with custom_inp_del2
        if 'lat_port' in params: self.lat_port = params['lat_port']
        else: self.lat_port = 1 
        if 'err_port' in params: self.err_port = params['err_port']
        else: self.err_port = 0 
        if self.port != self.err_port:
            raise AssertionError('The gated_rga_diff synapses are meant to ' +
                                 'be connected at the error ports')
        if 'w_sum' in params: self.w_sum = params['w_sum']
        else: self.w_sum = 1.
        if 'sig1' in params: self.sig1 = params['sig1']
        else: self.sig1 = 1.
        if 'sig2' in params: self.sig2 = params['sig2']
        else: self.sig2 = 1.
        # The add_double_del_inp_deriv_mp method of requirements.py will add the
        # ddidm_idx attribute, which is the index of this synapse in the
        # double_del_(avg)_inp_deriv_mp lists.
        # Similarly, add_slow_inp_deriv_mp will add the sid_idx attribute, which
        # is the index of this synapse in the (avg_)slow_inp_deriv_mp lists.
        # sid_idx and ddidm_idx should be equal.
        """
        # we now need to find the index of the input with this synapse in the
        # double_del_inp_deriv_mp lists.
        for idx, iid in enumerate(self.net.units[self.postID].port_idx[self.port]):
            if network.syns[self.postID][iid].preID == self.preID:
                self.ddidm_idx = idx
                break
        else:
            raise AssertionError('The gated_rga_diff constructor could not ' +
                  'find the index of its synapse in double_del_inp_deriv_mp')
        """
        
    def update(self, time):
        """ Update the weight using the gated_slide_rga_diff learning rule.
        
            If the network is correctly initialized, the pre-synaptic unit 
            updates lpf_fast, and lpf_mid, whereas the post-synaptic unit
            updates lpf_fast, lpf_mid, acc_mid, and the average of 
            approximate input derivatives for each port.

            The average of the delayed input derivatives comes from the
            double_del_avg_inp_deriv_mp requirement.

            The current rule allows synapses to become negative.
        """
        dm_steps = int(round(self.del_mod/self.net.min_delay))
        po_de = self.po_de + dm_steps # effective postsynaptic delay
        post = self.net.units[self.postID]
        pre = self.net.units[self.preID]
        # normalizing factors
        up_slow = post.get_lpf_mid(self.po_de) - post.get_lpf_slow(self.po_de)
        normfac1 = 1. / (self.sig1 + abs(up_slow))
        avg_normfac1 = 1. / (self.sig1 + abs(post.avg_slow_inp_deriv_mp[self.lat_port]))
        normfac2 = 1. / (self.sig2 +
                   abs(post.slow_inp_deriv_mp[self.err_port][self.sid_idx]))
        avg_normfac2 = 1. / (self.sig2 + abs(post.avg_slow_inp_deriv_mp[self.err_port]))
        # terms in the correlations
        xp_norm = avg_normfac1 * post.double_del_avg_inp_deriv_mp[1][self.lat_port]
        up_norm = normfac1 * (post.get_lpf_fast(self.po_de) -
                              post.get_lpf_mid(self.po_de))
        sp_now =  avg_normfac2 * post.avg_inp_deriv_mp[self.err_port]
        sp_del = (avg_normfac2 * 
                  post.double_del_avg_inp_deriv_mp[0][self.err_port])
        spj_now = normfac2 * (pre.get_lpf_fast(self.delay_steps) -
                              pre.get_lpf_mid(self.delay_steps) )
        spj_del = (normfac2 * 
                   post.double_del_inp_deriv_mp[0][self.port][self.ddidm_idx])
        # weight normalization
        self.w *= self.w_sum*(post.l0_norm_factor_mp[self.err_port] + 
                              pre.out_norm_factor)
        # delay update
        corr1 = (up_norm - xp_norm ) * (sp_del - spj_del)
        corr2 = (up_norm - xp_norm ) * (sp_now - spj_now)
        self.corr1 += self.corr_alpha * (corr1 - self.corr1)
        self.corr2 += self.corr_alpha * (corr2 - self.corr2)
        self.del_mod += ( (self.mod_max - self.del_mod) *
                          (self.del_mod - self.mod_min) *
                          (abs(self.corr2) - abs(self.corr1)) ) * self.dm_alpha
        # weight update
        self.w += post.acc_slow * self.alpha * (corr2 - corr1)
        # next line adds some random drift
                  #+ 0.0002*(np.random.random()-0.5))


class input_selection_synapse(synapse):
    """ A variant of the input correlation synapse.

        Like the input correlation synapse, it is assumed that a unit receiving
        an input_selection_synapse will have two types of inputs. One will be
        called "error", and the other "aff", for afferent. Unlike the
        inp_corr model, this synapse model assumes that its corresponding input
        is of 'aff' type. Thus the synapses for inputs at the 'error' port in
        the postsynaptic unit can use any other model.

        A second difference with the input correlation model is that weights
        from synapses at the "aff" postsynaptic port will be normalized so their
        absolute values add to a parameter w_sum. This is achieved with the 
        l0_norm_factor_mp requirement.

        The learning equation is: 
        w' = lrate * pre * err_diff,
        where err_diff is an approximation to the derivative of the scaled sum 
        of 'error' inputs, which are in the default case the inputs at port 1 of
        the postsynaptic unit. This derivative is obtained through the
        upd_sc_inp_sum_diff_mp requirement of the postsynaptic unit.
 
        'error' presynaptic units require the 'tau_fast' and 'tau_mid' parameters.
        'aff' presynaptic units require the 'tau_fast' parameter.

        The Enum name for this synapse type is 'inp_sel'.

    """
    def __init__(self, params, network):
        """ The class constructor.

        Args:
            params: same as the parent class, with some additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            OPTIONAL PARAMETERS
            'error_port' : port where the 'error' signals are received in the
                           postsynaptic unit. Default 1.
            'aff_port' : port where the 'afferent' signals are received in the
                         postsynaptic unit. Default is the port of the synapse
                         (e.g. self.port).
                         This port may be deprecated, and the afferent port
                         will be considered to be the synapse's port.
            'w_sum' : value of the sum of synaptic weights at the 'aff' synapse.
                      Default is 1.

        Raises:
            ValueError, AssertionError.
        """

        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        if 'error_port' in params:
            self.error_port = params['error_port']
        else:
            self.error_port = 1
        if 'aff_port' in params:
            self.aff_port = params['aff_port']
        else:
            self.aff_port = self.port
        if self.aff_port != self.port:
            from warnings import warn
            warn("aff_port and port don't coincide in input selection synapse",
                 UserWarning)
        if 'w_sum' in params:
            self.w_sum = params['w_sum']
        else:
            self.w_sum = 1.
        # this may be inefficient because the lpf_mid_sc_inp_sum is being
        # calculated also for the afferent inputs. It is, however, convenient
        # for the derived class gated_diff_inp_sel
        self.upd_requirements = set([syn_reqs.lpf_fast_sc_inp_sum_mp,
                                     syn_reqs.lpf_mid_sc_inp_sum_mp,
                                     syn_reqs.l0_norm_factor_mp,
                                     syn_reqs.pre_lpf_fast])

    def update(self, time):
        """ Update the weight using the input selection rule. """
        u = self.net.units[self.postID]
        err_diff = (u.lpf_fast_sc_inp_sum_mp[self.error_port] -
                    u.lpf_mid_sc_inp_sum_mp[self.error_port]) 
        pre = self.net.units[self.preID].get_lpf_fast(self.delay_steps)
            
        self.w *= self.w_sum * u.l0_norm_factor_mp[self.aff_port]
        self.w = self.w + self.alpha * pre * err_diff


class gated_input_selection_synapse(input_selection_synapse):
    """ The input selection synapse with modulated learning rate.
        
        The acc_slow attribute of the postsynaptic unit is used to modulate the
        learning rate. Thus, this synapse should only connect to units that
        contain that requirement. In the case of the gated_out_norm_am_sig unit,
        inputs at port 2 are used to reset acc_slow. For the
        gated_rga_inpsel_adapt unit inputs at port 3 are used instead.

        There is also an extra delay for the presynaptic input, in order to
        synchronize the error and the afferent signal. This delay is given as
        the number of 'min_delay' steps, in the 'extra_steps' parameter of the
        constructor. The 'delay' parameter of the presynaptic units must be
        large enough to accomodate this extra delay. This means:
        delay > min_delay * (delay_steps + extra_steps).

        The name of this synapse type is 'gated_inp_sel'.
    """
    def __init__(self, params, network):
        """ The class constructor.

        Args:
            params: same as the parent class, with some additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            'extra_steps' : extra delay steps for presynaptic input.
            OPTIONAL PARAMETERS
            'error_port' : port where the 'error' signals are received in the
                           postsynaptic unit. Default 1.
            'aff_port' : port where the 'afferent' signals are received in the
                         postsynaptic unit. Default is the port of the synapse
                         (e.g. self.port).
            'w_sum' : value of the sum of synaptic weights at the 'aff' synapse.
                      Default is 1.
            'normalize' : Binary value indicating whether the sum of synaptic
                          weights at the 'aff' synapse should be 'w_sum'.
                          Default is True.
        Raises:
            ValueError, AssertionError.
        """
        input_selection_synapse.__init__(self, params, network)
        self.extra_steps = params['extra_steps']
        if 'normalize' in params:
            self.normalize = params['normalize']
        else:
            self.normalize = True

    def update(self, time):
        """ Update the weight using the input selection rule. """
        u = self.net.units[self.postID]
        err_diff = (u.lpf_fast_sc_inp_sum_mp[self.error_port] -
                    u.lpf_mid_sc_inp_sum_mp[self.error_port]) 
        pre = self.net.units[self.preID].get_lpf_fast(self.delay_steps +
                                                      self.extra_steps)
        if self.normalize:
            self.w *= self.w_sum * u.l0_norm_factor_mp[self.aff_port]
        self.w = self.w + u.acc_slow * self.alpha * pre * err_diff


class anti_covariance_inh_synapse(synapse):
    """ Anticovariance rule for inhibitory synapses
    
        w' = - lrate * (post - theta) * pre,
        where theta is a low-pass filtered version of post with a slow time constant.
        This correspond to equation 8.8 in Dayan and Abbott's "Theoretical Neuroscience"
        (MIT Press 2001), with the sign reversed.
        The presynaptic activity includes its corresponding transmission delay.

        This synapse type expects to be initialized with a negative weight, and will
        implement soft weight bounding in order to ensure the sign of the weigh  does
        not become positive, so the rule is actually:
        w' = lrate * w * (post - theta) * pre,
        Notice that the negative sign was removed due to the assumption that w
        is negative.
    
        Presynaptic units require the 'tau_fast' parameter.
        Postsynaptic units require 'tau_fast' and 'tau_slow'.

        The name of this rule's Enum is 'anticov_inh'.
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
        self.upd_requirements = set([syn_reqs.lpf_fast, syn_reqs.pre_lpf_fast, 
                                     syn_reqs.lpf_slow])
        assert self.type is synapse_types.anticov_inh, ['Synapse from ' + str(self.preID) + 
                           ' to ' + str(self.postID) + ' instantiated with the wrong type']
    
    def update(self, time):
        """ Update the weight according to the anti-covariance learning rule."""
        # If the network is correctly initialized, the pre-synaptic unit 
        # is updatig lpf_fast, and the  post-synaptic unit is updating 
        # lpf_fast and lpf_slow at each update() call
        avg_post = self.net.units[self.postID].get_lpf_slow(0)
        post = self.net.units[self.postID].get_lpf_fast(0)
        pre = self.net.units[self.preID].get_lpf_fast(self.delay_steps)
        
        # A forward Euler step with the anti-covariance learning rule 
        self.w = self.w * (1. + self.alpha * (post - avg_post) * pre)


class chg_synapse(synapse):
    """ The change detection synapse (CHG).

        This synapse type was first described in the "flow control units"
        tiddler. It is meant to turn a normal unit into a "change detection
        unit". The equations make clear why this would be the case:
        w' = alpha * (|<x>_f - <x>_m| - w)
        where  <x>_f and <x>_m are the fast and medium LPF'd versions of the
        presynaptic input. As long as not every input goes down near zero, after
        a change the weights will momentarily increase.
    """
    def __init__(self, params, network):
        """ The  class constructor.

        Args: 
            params: same as the parent class, with some additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
        Raises:
            AssertionError
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        self.upd_requirements = set([syn_reqs.pre_lpf_fast, syn_reqs.pre_lpf_mid])
        assert self.type is synapse_types.chg, ['Synapse from ' + str(self.preID) + 
                           ' to ' + str(self.postID) + ' instantiated with the wrong type']

    def update(self, time):
        """ Update the weight according to the chg learning rule."""
        pre_fast = self.net.units[self.preID].get_lpf_fast(self.delay_steps)
        pre_mid = self.net.units[self.preID].get_lpf_mid(self.delay_steps)
        self.w += self.alpha * (abs(pre_fast - pre_mid) - self.w)


class gated_diff_input_selection_synapse(gated_input_selection_synapse):
    """ Differential version of the gated_input_selection synapse.

        The presynaptic activity is replaced by its approximate derivative.
        
        The acc_slow attribute of the postsynaptic unit is used to modulate the
        learning rate. Thus, this synapse should only connect to units that
        contain that requirement. In the case of the gated_out_norm_am_sig unit,
        inputs at port 2 are used to reset acc_slow. For the
        gated_rga_inpsel_adapt unit inputs at port 3 are used instead.

        There is also an extra delay for the presynaptic input, in order to
        synchronize the error and the afferent signal. This delay is given as
        the number of 'min_delay' steps, in the 'extra_steps' parameter of the
        constructor. The 'delay' parameter of the presynaptic units must be
        large enough to accomodate this extra delay. This means:
        delay > min_delay * (delay_steps + extra_steps).

        The name of this synapse type is 'gated_diff_inp_sel'.
    """
    def __init__(self, params, network):
        """ The class constructor.

        Args:
            params: same as the synapse class, with some additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            'extra_steps' : extra delay steps for presynaptic input.
            OPTIONAL PARAMETERS
            'error_port' : port where the 'error' signals are received in the
                           postsynaptic unit. Default 1.
            'aff_port' : port where the 'afferent' signals are received in the
                         postsynaptic unit. Default is the port of the synapse
                         (e.g. self.port).
            'w_sum' : value of the sum of synaptic weights at the 'aff' synapse.
                      Default is 1.
            'normalize' : Binary value indicating whether the sum of synaptic
                          weights at the 'aff' synapse should be 'w_sum'.
                          Default is True.
        Raises:
            ValueError, AssertionError.
        """
        gated_input_selection_synapse.__init__(self, params, network)
        self.upd_requirements.update([syn_reqs.pre_lpf_mid])

    def update(self, time):
        """ Update the weight using the input selection rule. """
        u = self.net.units[self.postID]
        err_diff = (u.lpf_fast_sc_inp_sum_mp[self.error_port] -
                    u.lpf_mid_sc_inp_sum_mp[self.error_port]) 
        preU = self.net.units[self.preID]
        Dpre = (preU.get_lpf_fast(self.delay_steps + self.extra_steps) -
                preU.get_lpf_mid(self.delay_steps + self.extra_steps))
        if self.normalize:
            self.w *= self.w_sum * u.l0_norm_factor_mp[self.aff_port]
        self.w = self.w + u.acc_slow * self.alpha * Dpre * err_diff


class gated_diff_inp_corr(synapse):
    """ Differential version of the input correlation synapse.

        This synapse assumes that its presynaptic input is some type of "error"
        signal. The synapse's weight will be modified according to the
        correlation between the derivative of this presynaptic input, and the
        derivative of the input at different port of the postsynaptic unit.

        Such a rule is meant to select the "error" input based on how it
        correlates with another lateral/ascending input to the postsynaptic unit.
        See the later part of "A generalization of RGA, and Fourier components."
        
        The acc_slow attribute of the postsynaptic unit is used to modulate the
        learning rate. Thus, this synapse should only connect to units that
        contain that requirement. In the case of the am_pulse unit, inputs at 
        port 3 are used to reset acc_slow.

        There is an extra delay for the ascending presynaptic input in order to
        synchronize it with the error. This delay is comes from using the
        xtra_del_inp_deriv_mp requirement for the ascending inputs, whereas the
        inp_deriv_mp requirement is used for the error.

        The weights are normalized so their sum is always moving to a given
        'w_sum' value.

        The name of this synapse type is 'gated_diff_inp_corr'.
    """
    def __init__(self, params, network):
        """ The class constructor.

        Args:
            params: same as the synapse class, with some additions.
            REQUIRED PARAMETERS
            'lrate' : A scalar value that will multiply the derivative of the weight.
            OPTIONAL PARAMETERS
            'error_port' : port where the 'error' signals are received in the
                           postsynaptic unit. Default is the port of the synapse
                           (e.g. self.port).
            'asc_port' : port where the 'ascending' signals are received in the
                         postsynaptic unit. Default is 2.
            'w_sum' : value of the sum of synaptic weights at the 'asc' synapse.
                      Default is 1.
        Raises:
            ValueError
        """
        synapse.__init__(self, params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        if 'asc_port' in params:
            self.asc_port = params['asc_port']
        else:
            self.asc_port = 2
        if 'error_port' in params:
            self.error_port = params['error_port']
        else:
            self.error_port = self.port
        if self.error_port != self.port:
            from warnings import warn
            warn("error_port and port don't coincide in diff_inp_corr synapse",
                 UserWarning)
        if 'w_sum' in params:
            self.w_sum = params['w_sum']
        else:
            self.w_sum = 1.
        # this may be inefficient because the lpf_mid_sc_inp_sum is being
        # calculated also for the afferent inputs. It is, however, convenient
        # for the derived class gated_diff_inp_sel
        self.upd_requirements = set([syn_reqs.pre_lpf_fast,
                                     syn_reqs.pre_lpf_mid,
                                     syn_reqs.mp_weights, # for xtra_del_inp_deriv_mp_sc_sum
                                     syn_reqs.inp_deriv_mp,
                                     syn_reqs.xtra_del_inp_deriv_mp,
                                     syn_reqs.xtra_del_inp_deriv_mp_sc_sum,
                                     syn_reqs.l0_norm_factor_mp])
        # The plasticity in this synapse is driven by two inputs in the
        # postsynaptic unit. One is the presynaptic input of this synapse, and
        # the other is the ascending input to the unit. We need an index to
        # extract the derivative of the presynaptic input from inp_deriv_mp,
        # called idm_id.
        # Since synapses are created serially, and this constructor is called
        # when the synapse is being created, we can assume that idm_id will be
        # the number of entries in post.port_idx[self.error_port]
        self.idm_id = len(network.units[self.postID].port_idx[self.error_port])

        #found = False
        #post = network.units[self.postID] # postsynaptic unit
        #for idx, uid in enumerate(post.port_idx[self.error_port]):
            #if uid == self.syns_loc:
                #self.idm_id = idx
                #found = True
                #break
        #if not found:
            #raise AssertionError('diff_inp_corr synapse could not find its ' +
                                 #'id in inp_deriv_mp')
        # For the derivative of the ascending input, usually there will be a 
        # single ascending input at this port, but just to be safe we will use
        # the derivative of the scaled input sum.
           
    def update(self, time):
        """ Update the weight using the diff_inp_corr rule. """
        post = self.net.units[self.postID]
        #pre = self.net.units[self.preID]
        
        Derror = post.inp_deriv_mp[self.error_port][self.idm_id]
        Dasc = post.xtra_del_inp_deriv_mp_sc_sum[self.asc_port]

        self.w += self.alpha * (post.acc_slow * Derror * Dasc +
                  (self.w_sum * post.l0_norm_factor_mp[self.error_port] - 1.) 
                   * self.w )


class static_l0_normal(synapse):
    """ A static synapse with multiplicative normalization.

        NOT RECOMMENDED FOR GENERAL USE
        You can instead normalize the weights on initialization.

        Weigths for all synapses of this type in a unit are scaled so that the
        sum of their absolute values is 'w_sum'.
        This is done by adding the l0_norm_factor to the postsynaptic unit.

        Weights are not normalized instantly, but instead change with a given
        time constant.
    """
    def __init__(self, params, network):
        """
            Constructor of the static_l0_normal synapse.
    
            Args:
                params: same as the synapse class. 
                REQUIRED PARAMETERS
                'w_sum' : the sum of absolute weight values will add to this.
                        Must be the same for all normalized synapses.
                'tau_norml' : time constant for normalization.
            Raises:
                ValueError
        """
        synapse.__init__(self, params, network)
        self.tau_norml = params['tau_norml'] 
        self.w_sum = params['w_sum']
        if self.tau_norml <= 0.:
            raise ValueError('tau_norml should be a positive value')
        if self.w_sum <= 0.:
            raise ValueError('w_sum should be a positive value')
        self.alpha = self.net.min_delay / self.tau_norml
        self.upd_requirements.update([syn_reqs.l0_norm_factor])

    def update(self, time):
        """ Update the weight normalization. """
        nf = self.net.units[self.postID].l0_norm_factor
        self.w = self.w + self.alpha * self.w * (nf*self.w_sum - 1.)


class comp_pot(synapse):
    """ Competitive potentiation synapse.

        The weight increases according to how large the presynaptic input is,
        compared to the average of port 0 inputs. This requires the inp_avg_mp
        requirement, which in turn depends on the mp_inputs requirement.

        Port 0 weights are normalized so they add to a value 'w_sum'. This uses
        the l0_norm_factor_mp requirement.

        The current version has soft weight bounding so weights remain positive.
        Initial weight values should be positive!
    """
    def __init__(self, params, network):
        """ Constructor of the comp_pot synapse.

            Args:
                params: same as the synapse class with two additions.
                REQUIRED PARAMETERS
                'lrate' : learning rate for the potentiation.
                'w_sum' : the sum of absolute weight values will add to this.
                        Must be the same for all normalized synapses.
                'tau_norml' : time constant for normalization.
            Raises:
                ValueError
        """
        synapse.__init__(self, params, network)
        self.tau_norml = params['tau_norml'] 
        self.w_sum = params['w_sum']
        if self.tau_norml <= 0.:
            raise ValueError('tau_norml should be a positive value')
        if self.w_sum <= 0.:
            raise ValueError('w_sum should be a positive value')
        self.alpha1 = params['lrate'] * self.net.min_delay # scale update rule
        self.alpha2 = self.net.min_delay / self.tau_norml # scale normalization
        self.upd_requirements.update([syn_reqs.l0_norm_factor_mp,
                                      syn_reqs.mp_inputs,
                                      syn_reqs.inp_avg_mp])

    def update(self, time):
        """ Update the comp_pot synapse. """
        post = self.net.units[self.postID]
        pre = self.net.units[self.preID]
        self.w += (self.w * self.alpha1 * (pre.act_buff[-1] - post.inp_avg_mp[0]) +
               self.alpha2 * self.w * (post.l0_norm_factor_mp[0]*self.w_sum - 1.))

        




