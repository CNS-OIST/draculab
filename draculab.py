"""@package draculab

draculab.py  (Delayed-Rate Adaptively-Connected Units Laboratory)
A simulator of rate units with delayed connections.

\author     Sergio Verduzco Flores
\date       2017
"""

# Enumeration classes provide human-readable and globally consistent constants
from enum import Enum  # enumerators for the synapse and unit types

class unit_types(Enum):
    """ An enum class with all the implemented unit models. """
    source = 1
    sigmoidal = 2
    linear = 3
    mp_linear  = 4
    custom_fi = 5
    custom_sc_fi = 6
    kwta = 7
    exp_dist_sig = 8
    exp_dist_sig_thr = 9
    double_sigma = 10
    sigma_double_sigma = 11
    double_sigma_n = 12
    double_sigma_trdc = 13
    sds_trdc = 14
    ds_n_trdc = 15
    ds_sharp = 16
    sds_sharp = 17
    ds_n_sharp = 18
    sds_n_sharp = 19
    sig_ssrdc_sharp = 20
    st_hr_sig = 21
    ss_hr_sig = 22
    sig_trdc = 23
    sig_ssrdc = 24
    sig_trdc_sharp = 25
    sds_n = 26

    def get_class(self):
        """ Return the class object corresponding to a given object type enum. 

        Because this method is in a subclass of Enum, it does NOT show in the
        unit_types class; it only shows in its members. For example,
        unit_types.source.get_class()
        returns the source class object.
    
        Raises:
            NotImplementedError.
        """
        if self == unit_types.source:
            unit_class = source
        elif self == unit_types.sigmoidal:
            unit_class = sigmoidal
        elif self == unit_types.linear:
            unit_class = linear
        elif self == unit_types.mp_linear:
            unit_class = mp_linear
        elif self == unit_types.custom_fi:
            unit_class = custom_fi
        elif self == unit_types.custom_sc_fi:
            unit_class = custom_scaled_fi
        elif self == unit_types.kwta:
            unit_class = kWTA
        elif self == unit_types.exp_dist_sig:
            unit_class = exp_dist_sigmoidal
        elif self == unit_types.exp_dist_sig_thr:
            unit_class = exp_dist_sig_thr
        elif self == unit_types.double_sigma:
            unit_class = double_sigma
        elif self == unit_types.sigma_double_sigma:
            unit_class = sigma_double_sigma
        elif self == unit_types.double_sigma_n:
            unit_class = double_sigma_normal
        elif self == unit_types.double_sigma_trdc:
            unit_class = double_sigma_trdc
        elif self == unit_types.sds_trdc:
            unit_class = sigma_double_sigma_trdc
        elif self == unit_types.ds_n_trdc:
            unit_class = double_sigma_normal_trdc
        elif self == unit_types.ds_sharp:
            unit_class = double_sigma_sharp
        elif self == unit_types.sds_sharp:
            unit_class = sigma_double_sigma_sharp
        elif self == unit_types.ds_n_sharp:
            unit_class = double_sigma_normal_sharp
        elif self == unit_types.sds_n_sharp:
            unit_class = sigma_double_sigma_normal_sharp
        elif self == unit_types.sig_ssrdc_sharp:
            unit_class = sig_ssrdc_sharp
        elif self == unit_types.st_hr_sig:
            unit_class = sliding_threshold_harmonic_rate_sigmoidal
        elif self == unit_types.ss_hr_sig:
            unit_class = synaptic_scaling_harmonic_rate_sigmoidal
        elif self == unit_types.sig_trdc:
            unit_class = sig_trdc
        elif self == unit_types.sig_ssrdc:
            unit_class = sig_ssrdc
        elif self == unit_types.sig_trdc_sharp:
            unit_class = sig_trdc_sharp
        elif self == unit_types.sds_n:
            unit_class = sigma_double_sigma_normal
        else:
            raise NotImplementedError('Attempting to retrieve the class of an unknown unit model')
            # NameError instead?
        return unit_class

    
class synapse_types(Enum):
    """ An enum class with all the implemented synapse models. """
    static = 1
    oja = 2   # implements the Oja learning rule 
    antihebb = 3  # implements the anti-Hebbian learning rule
    cov = 4 # implements the covariance learning rule
    anticov = 5  # implements the anti-covariance rule 
    hebbsnorm = 6 # Hebbian rule with substractive normalization
    inp_corr = 7 # Input correlation
    bcm = 8 # Bienenstock, Cooper, and Munro learning rule
    sq_hebbsnorm = 9 # Hebbian rule with substractive normalization, second version
    homeo_inh = 10 # Homeostatic inhibition from Moldakarimov et al. 2006
    corr_inh = 11 # Homeostatic inhibition from Vogels et al. 2011
    diff_hebbsnorm = 12 # Differential Hebbian with substractive normalization
    exp_rate_dist = 13 # To create an exponential firing rate distribution
    anticov_pre = 14  # anticovariance rule using the presynaptic mean activity
    #axoaxo = auto()  # axo-axonic synapses will provide presynaptic inhibition

    def get_class(self):
        """ Return the class object corresponding to a given synapse type enum. 

        Because this method is in a subclass of Enum, it does NOT show in the
        synapse_types class; it only shows in its members. For example,
        synapse_types.static.get_class()
        returns the static_synapse class object.
    
        Raises:
            ValueError.
        """
        if self == synapse_types.static:
            syn_class = static_synapse
        elif self == synapse_types.oja:
            syn_class = oja_synapse
        elif self == synapse_types.antihebb:
            syn_class = anti_hebbian_synapse
        elif self == synapse_types.cov:
            syn_class = covariance_synapse
        elif self == synapse_types.anticov:
            syn_class = anti_covariance_synapse
        elif self == synapse_types.hebbsnorm:
            syn_class = hebb_subsnorm_synapse
        elif self == synapse_types.sq_hebbsnorm:
            syn_class = sq_hebb_subsnorm_synapse
        elif self == synapse_types.inp_corr:
            syn_class = input_correlation_synapse
        elif self == synapse_types.bcm:
            syn_class = bcm_synapse
        elif self == synapse_types.homeo_inh:
            syn_class = homeo_inhib_synapse
        elif self == synapse_types.corr_inh:
            syn_class = corr_homeo_inhib_synapse
        elif self == synapse_types.diff_hebbsnorm:
            syn_class = diff_hebb_subsnorm_synapse
        elif self == synapse_types.exp_rate_dist:
            syn_class = exp_rate_dist_synapse
        elif self == synapse_types.anticov_pre:
            syn_class = anti_cov_pre_synapse 
        else:
            raise NotImplementedError('Attempting retrieve the class of an unknown synapse model')
        
        return syn_class


class plant_models(Enum):
    """ An enum class with all the implemented plant models. """
    pendulum = 1
    conn_tester = 2

    def get_class(self):
        """ Return the class object corresponding to a given plant enum. 

        Because this method is in a subclass of Enum, it does NOT show in the
        plant_models class; it only shows in its members. For example,
        plant_models.pendulum.get_class()
        returns the pendulum class object.
    
        Raises:
            NotImplementedError.
        """
        if self == plant_models.pendulum:
            plant_class = pendulum 
        elif self == plant_models.conn_tester:
            plant_class = conn_tester
        else:
            raise NotImplementedError('Attempting to retrieve the class for an unknown plant model')

        return plant_class


class syn_reqs(Enum):
    """ This enum contains all the variables that a synapse model may ask a unit
        to maintain in order to support its learning function.

        Details are in the function which updates each requirement. The updating function
        of each requirement is specified in unit.init_pre_syn_update.
    """
    lpf_fast = 1  # postsynaptic activity low-pass filtered with a fast time constant
    lpf_mid = 2   # postsynaptic activity low-pass filtered with a medium time constant
    lpf_slow = 3  # postsynaptic activity low-pass filtered with a slow time constant
    pre_lpf_fast = 4  # presynaptic activity low-pass filtered with a fast time constant
    pre_lpf_mid = 5   # presynaptic activity low-pass filtered with a medium time constant
    pre_lpf_slow = 6  # presynaptic activity low-pass filtered with a slow time constant
    sq_lpf_slow = 7 # squared postsynaptic activity lpf'd with a slow time constant
    inp_avg = 8   # Sum of fast-LPF'd hebbsnorm inputs, divided by number of hebbsnorm inputs 
    pos_inp_avg = 9 # as inp_avg, but only considers inputs with positive synaptic weights
    err_diff = 10 # The approximate derivative of the error signal used in input correlation
    sc_inp_sum = 11 # Scaled input sum. This is the sum of presynaptic inputs, each multiplied 
                    # by its synaptic weight.
    diff_avg = 12 # Average of derivatives for inputs with diff_hebb_subsnorm synapses.
    pos_diff_avg = 13 # As diff_avg, but only considers inputs with positive synaptic weights
    lpf_mid_inp_sum = 14 # LPF'd sum of presynaptic inputs with a medium time constant.
    n_erd = 15 # number of exp_rate_dist synapses on the postsynaptic unit. ONLY USED IN LEGACY CODE
    balance = 16 # fraction of inputs above, below, and around the current rate
    exp_scale = 17 # synaptic scaling to produce an exponential firing rate distribution
    inp_vector = 18 # the current "raw" input vector in a numpy array
    slide_thresh = 19 # An activation threshold that adjusts to produce an exp firing rate distro
    lpf_slow_mp_inp_sum = 20 # slow LPF'd scaled sum of inputs from individual ports
    mp_inputs = 21 # the multiport inputs, as returned by the get_mp_inputs method
    balance_mp = 22 # the multiport version of the 'balance' requirement above
    slide_thr_shrp = 23 # move threshold to produce exp rate distro if 'sharpen' input > 0.5
    exp_scale_mp = 24 # scale factors to produce exp distro in multiport units
    slide_thr_hr = 25 # Sliding threshold that produces "harmonic" rate patterns (multiport units)
    syn_scale_hr = 26 # Synaptic scaling that produces "harmonic" rate patterns (multiport units)
    exp_scale_shrp = 27 # scale synapses to produce exp rate distro if 'sharpen' input > 0.5



# Importing the classes used by the simulator
from network import *
from units import *
from synapses import *
from plants import *
from topology import *

