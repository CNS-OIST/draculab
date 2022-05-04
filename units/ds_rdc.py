"""
ds_rdc.py
This module contains the family of unit models characterized by the double
sigma model of the dendritic tree, and/or rate distribution control.
"""
from draculab import unit_types, synapse_types, syn_reqs  # names of models and requirements
from units.units import *
import numpy as np


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

        unit.__init__(self, ID, params, network)
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
        #return cython_sig(self.thresh, self.slope, arg)
    
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

    The method to use for rate distribution control can be specified through the sort_rdc
    parameter given to the constructor.
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
                OPTIONAL PARAMETERS
                'sort_rdc' : A boolean value specifying whether to use the 'sorting' type of
                             rdc control, which uses the exp_scale_sort_sharp requirement.
                             If sort_rdc is absent or false, the default is non-sorting rdc.

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
        elif params['sharpen_port'] >= self.n_ports:
            raise ValueError('sharpen_port exceeds the number of ports in an ssrdc_sharp unit')
        else:
            self.multiport = True # This causes the port_idx list to be created in init_pre_syn_update
        self.rdc_port = params['rdc_port'] # port for rate distribution control
        self.tau_scale = params['tau_scale']  # the scaling time constant
        self.tau_relax = params['tau_relax']  # the second scaling time constant
        self.Kp = params['Kp']  # gain for synaptic scaling
        self.c = params['c']  # The coefficient in the exponential distribution
        self.sharpen_port = params['sharpen_port']
        
        if 'sort_rdc' in params and params['sort_rdc']:
            self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.exp_scale_sort_shrp, syn_reqs.lpf_fast])
        else:
            self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.balance_mp, 
                                   syn_reqs.exp_scale_shrp, syn_reqs.lpf_fast]) 

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
 

class sig_ssrdc_sharp(ssrdc_sharp_base): 
    """
    The synaptic weights are scaled to produce an exponential distribution when input at port 1 > 0.5 .
    
    In other words, this is the same as the exp_dist_sigmoidal unit, but there is an extra port
    that controls whether or not the synaptic scaling is applied. Normal inputs must arrive at
    port 0, and the input that signals whether to apply rate distribution control should be 
    at port 1.

    Whether an input is excitatory or inhibitory is decided by the sign of its initial value.
    Synaptic weights initialized to zero will be considered excitatory.

    The method to use for rate distribution control can be specified through the sort_rdc
    parameter given to the constructor.
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
                OPTIONAL PARAMETERS
                'sort_rdc' : A boolean value specifying whether to use the 'sorting' type of
                             rdc control, which uses the exp_scale_sort_sharp requirement.
                             If sort_rdc is absent or false, the default is non-sorting rdc.

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
        #return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
        return cython_sig(self.thresh, self.slope, arg)
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        #return ( self.f(self.get_mp_input_sum(t)) - y[0] ) * self.rtau
        return ( cython_sig(self.thresh, self.slope, self.get_mp_input_sum(t)) - y[0] ) * self.rtau

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

    The method to usd for rate distribution control can be specified through the sort_rdc
    parameter given to the constructor.
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
                'tau_scale' : sets the speed of change for the scaling factor
                'c' : Changes the homogeneity of the firing rate distribution.
                    Values very close to 0 make all firing rates equally probable, whereas
                    larger values make small firing rates more probable. 
                    Shouldn't be set to zero (causes zero division in the cdf function).
                'Kp' : Gain factor for the scaling of weights (makes changes bigger/smaller).
                'rdc_port' : port ID of inputs used for rate distribution control.
                OPTIONAL PARAMETERS
                'sort_rdc' : A boolean value specifying whether to use the 'sorting' type of
                             rdc control, which uses the exp_scale_sort_mp requirement.
                             If sort_rdc is absent or false, the default is non-sorting rdc.

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
        if 'sort_rdc' in params and params['sort_rdc']:
            self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.exp_scale_sort_mp, syn_reqs.lpf_fast])
        else:
            self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.balance_mp, 
                                   syn_reqs.exp_scale_mp, syn_reqs.lpf_fast]) 
        
    def f(self, arg):
        """ This is the sigmoidal function. Could roughly think of it as an f-I curve. """
        #return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
        return cython_sig(self.thresh, self.slope, arg)
    
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

        unit.__init__(self, ID, params, network)
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
        #return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
        return cython_sig(self.thresh, self.slope, arg)
    
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
        self.needs_mp_inp_sum = False # use upd_flat_inp_sum when network is flat
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.balance_mp,
                               syn_reqs.slide_thresh, syn_reqs.lpf_fast]) 
        
    def f(self, arg):
        """ This is the sigmoidal function. Could roughly think of it as an f-I curve. """
        #return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
        return cython_sig(self.thresh, self.slope, arg)
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        return ( self.f(self.get_mp_input_sum(t)) - y[0] ) * self.rtau

    def dt_fun(self, y, s):
        """ The derivative function used by flat networks. """
        return ( self.f(self.inp_sum[s]) - y ) * self.rtau
 
    def get_mp_input_sum(self, time):
        """ The input function of sig_trdc units. """
        weights = self.get_mp_weights(time)
        inps = self.get_mp_inputs(time)
        return sum([np.dot(weights[prt], inps[prt]) for prt in range(self.n_ports)])


class trdc_sharp_base(unit):
    """ 
    Parent class for units with switchable threshold-based rate distro control. 
    
    One of the parameters is a port number, called "sharpen_port". When the 
    scaled sum of inputs to the sharpen port are larger than 0.5, based on 
    the distribution of inputs at port 'rdc_port' this unit will use threshold-
    based rate distribution control to produce an exponential distribution of 
    firing rates. When the inputs to the sharpen port are smaller than 0.5 
    the threshold will decay exponentially to a default value called "thr_fix",
    with time constant reciprocal to "tau_fix".
    """
    # The only difference with multiport_trdc_base is the use of the slide_thresh_shrp 
    # instead of slide_thresh

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
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.balance_mp, 
                               syn_reqs.slide_thresh_shrp, syn_reqs.lpf_fast])
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
        #return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
        return cython_sig(self.thresh, self.slope, arg)
    
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
                            of branches. Each entry is a positive number.
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
        self.needs_mp_inp_sum = True # use upd_flat_mp_inp_sum when flat
     
        # testing and initializing the branch parameters
        if n_branches != len(self.br_w):
            raise ValueError('Number of ports'+txt+'should equal the length of the branch_w parameter')
        elif min(self.br_w) <= 0:
            raise ValueError('Elements of branch_w should be positive')
        # Getting rid of this test...
        #elif sum(self.br_w) - 1. > 1e-4:
        #    raise ValueError('Elements of branch_w should add to 1')
        
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
                            of branches. Each entry is a positive number.
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

    def dt_fun(self, y, s):
        """ Returns the derivative when state is y, at time substep s. """
        inp_sum = sum( [ o * (self.f(th,sl,self.mp_inp_sum[p][s]) - self.phi)
                     for o, th, sl, p in zip(self.br_w, self.threshs, self.slopes,
                     range(self.n_ports)) ] )
        return ( cython_sig(self.thresh, self.slope, inp_sum) - y ) * self.rtau


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
                            of branches. Each entry is a positive number.
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
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.mp_weights,
                               syn_reqs.lpf_slow_mp_inp_sum]) 
     
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
                            of branches. Each entry is a positive number.
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
                    [ o*(self.f(th,sl,np.dot(w,i)) - self.phi) for o,th,sl,w,i in 
                      zip(self.br_w, self.threshs, self.slopes, w[1:], inp[1:]) ] )
        # version 2. The accumulator
        #ret = np.dot(w[0], inp[0])
        #for i in range(1,self.n_ports):
        #    ret += self.br_w[i-1] * (self.f(self.threshs[i-1], self.slopes[i-1], 
        #                             np.dot(w[i],inp[i])) - self.phi)
        #return ret
    
    def dt_fun(self, y, s):
        """ Returns the derivative when state is y, at time substep s. """
        inp_sum = self.mp_inp_sum[0][s] # input sum at port 0
        inp_sum += sum( [ o * (self.f(th,sl,self.mp_inp_sum[p][s]) - self.phi)
                     for o, th, sl, p in zip(self.br_w, self.threshs, self.slopes,
                     range(1,self.n_ports)) ] )
        return ( cython_sig(self.thresh, self.slope, inp_sum) - y ) * self.rtau



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
                            of branches. Each entry is a positive number.
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
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.mp_weights,
                               syn_reqs.lpf_slow_mp_inp_sum]) 
     
    
    def get_mp_input_sum(self, time):
        """ The input function of the normalized double sigma unit. """
        # Removing zero or near-zero values from lpf_slow_mp_inp_sum
        lpf_inp_sum = [np.sign(arry)*(np.maximum(np.abs(arry), 1e-3)) for arry in self.lpf_slow_mp_inp_sum]
        w = self.get_mp_weights(time)
        inp = self.get_mp_inputs(time)
         
        return np.dot(w[0], inp[0]) + sum( [ o*( self.f(th, sl, (np.dot(w,i)-y) / y) - self.phi ) 
               for o,th,sl,y,w,i in zip(self.br_w, self.threshs, self.slopes, lpf_inp_sum[1:], w[1:], inp[1:]) ] )
 

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

    The method to usd for rate distribution control can be specified through the sort_rdc
    parameter given to the constructor.

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
                            of branches. Each entry is a positive number.
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
                OPTIONAL PARAMETERS
                'sort_rdc' : A boolean value specifying whether to use the 'sorting' type of
                             rdc control, which uses the exp_scale_sort_shrp requirement.
                             If sort_rdc is absent or false, the default is non-sorting rdc.

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
                            of branches. Each entry is a positive number.
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
                            of branches. Each entry is a positive number.
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
                    [ o*(self.f(th,sl,np.dot(w,i))-self.phi) for o,th,sl,w,i in 
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
                            of branches. Each entry is a positive number.
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
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.mp_weights,
                               syn_reqs.lpf_slow_mp_inp_sum]) 
     
    
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
                            of branches. Each entry is a positive number.
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
                            of branches. Each entry is a positive number.
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
                    [ o*(self.f(th,sl,np.dot(w,i))-self.phi) for o,th,sl,w,i in 
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
                            of branches. Each entry is a positive number.
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
        self.syn_needs.update([syn_reqs.lpf_slow_mp_inp_sum, syn_reqs.mp_weights]) 
     
    
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
                            of branches. Each entry is a positive number.
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
        self.syn_needs.update([syn_reqs.lpf_slow_mp_inp_sum, syn_reqs.mp_weights]) 
     
    
    def get_mp_input_sum(self, time):
        """ The input function of the normalized double sigma unit. """
        # Removing zero or near-zero values from lpf_slow_mp_inp_sum
        lpf_inp_sum = [np.sign(arry)*(np.maximum(np.abs(arry), 1e-3)) 
                       for arry in self.lpf_slow_mp_inp_sum]
        w = self.get_mp_weights(time)
        inp = self.get_mp_inputs(time)
        del w[self.sharpen_port]
        del inp[self.sharpen_port]
         
        return np.dot(w[0], inp[0]) + sum( [ o*( self.f(th, sl, (np.dot(w,i)-y) / y) - self.phi ) 
               for o,th,sl,y,w,i in zip(self.br_w, self.threshs, self.slopes, lpf_inp_sum[1:], w[1:], inp[1:]) ] )
 

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
    ~~~~ Because of this, these units make sense when the rdc port is 0 (e.g. the soma) ~~~~
    Nevertheless, this is not enforced.
    The inputs at the sharpen port will not contribute to the activation of the unit. 
    The method to use for rate distribution control can be specified through the sort_rdc
    parameter given to the constructor.
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
                            of branches. Each entry is a positive number.
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
                OPTIONAL PARAMETERS
                'sort_rdc' : A boolean value specifying whether to use the 'sorting' type of
                             rdc control, which uses the exp_scale_sort_mp requirement.
                             If sort_rdc is absent or false, the default is non-sorting rdc.
        """
        self.extra_ports = 2 # The soma and the sharpen port 
        double_sigma_base.__init__(self, ID, params, network)
        self.unit_initialized = True  # to avoid calling the unit constructor twice
        ssrdc_sharp_base.__init__(self, ID, params, network)
        self.syn_needs.update([syn_reqs.lpf_slow_mp_inp_sum, syn_reqs.mp_weights])

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
        # If I ever use this unit seriously, this should be done at upd_lpf_slow_mp_inp_sum
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
    whose inputs are used to harmonize the rate are in the hr_port parameter.
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
        #return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
        return cython_sig(self.thresh, self.slope, arg)
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        #return ( self.f(self.get_mp_input_sum(t)) - y[0] ) * self.rtau
        return ( cython_sig(self.thresh, self.slope, self.get_mp_input_sum(t)) - y[0] ) * self.rtau

    def get_mp_input_sum(self, time):
        """ The input function of st_hr_sig unit. """
        weights = self.get_mp_weights(time)
        inps = self.get_mp_inputs(time)
        return sum([np.dot(weights[prt], inps[prt]) for prt in range(self.n_ports)])
     
    def upd_slide_thr_hr(self, time):
        """ Updates the threshold of units with rate harmonization.
        
            This is the update method of the slide_thr_hr requirement. 
        """
        inputs = self.mp_inputs[self.hr_port]
        mean_inp = np.mean(inputs) 
        r = self.get_lpf_fast(0)
        self.thresh -= self.tau_thr * ( mean_inp - r )
        self.thresh = max(min(self.thresh, 10.), -10.) # clipping large/small values



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
    whose inputs are used to harmonize the rate are in the hr_port parameter.
 
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

        The actual values used for scaling are calculated in upd_syn_scale_hr (below), 
        which updates the array self.scale_facs .
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
        #return 1. / (1. + np.exp(-self.slope*(arg - self.thresh)))
        return cython_sig(self.thresh, self.slope, arg)
    
    def derivatives(self, y, t):
        """ This function returns the derivatives of the state variables at a given point in time. """
        # there is only one state variable (the activity)
        #return ( self.f(self.get_mp_input_sum(t)) - y[0] ) * self.rtau
        return ( cython_sig(self.thresh, self.slope, self.get_mp_input_sum(t)) - y[0] ) * self.rtau

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

    def upd_syn_scale_hr(self, time):
        """ Update the synaptic scale for units with 'rate harmonization'. 
            
            The scaling factors in self.scale_facs_hr correspond to the excitatory synapses
            at the port hr_port. 

            exc_idx_hr, and scale_facs_hr are initialized in add_syn_scale_hr, in
            requirements.py .
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


