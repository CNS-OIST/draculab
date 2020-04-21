"""
draculab.py

draculab.py  (Delayed-Rate Adaptively-Connected Units Laboratory)
A simulator of rate units with delayed connections.

    Copyright (C) 2018 Okinawa Institute of Science and Technology

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

Author:     Sergio Verduzco Flores
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
    ds_ssrdc_sharp = 27
    delta_linear = 28
    sds_n_ssrdc_sharp = 29
    noisy_linear = 30
    noisy_sigmoidal = 31
    presyn_inh_sig = 32
    test_oscillator = 33 

    binary = 101 # from tutorial 5

    # Units from spinal_units.py
    am_pm_oscillator = 102
    out_norm_sig = 103
    out_norm_am_sig = 104
    logarithmic = 105
    rga_sig = 106
    gated_rga_sig = 107
    gated_rga_adapt_sig = 108
    act = 109
    gated_out_norm_am_sig = 110
    gated_rga_inpsel_adapt_sig = 111
    inpsel_linear = 112
    chwr_linear = 113
    am_oscillator = 114

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
            from units.units import source
            unit_class = source
        elif self == unit_types.sigmoidal:
            from units.units import sigmoidal
            unit_class = sigmoidal
        elif self == unit_types.linear:
            from units.units import linear
            unit_class = linear
        elif self == unit_types.noisy_linear:
            from units.units import noisy_linear 
            unit_class = noisy_linear 
        elif self == unit_types.noisy_sigmoidal:
            from units.units import noisy_sigmoidal
            unit_class = noisy_sigmoidal
        elif self == unit_types.exp_dist_sig:
            from units.ds_rdc import exp_dist_sigmoidal
            unit_class = exp_dist_sigmoidal
        elif self == unit_types.exp_dist_sig_thr:
            from units.ds_rdc import exp_dist_sig_thr
            unit_class = exp_dist_sig_thr
        elif self == unit_types.double_sigma:
            from units.ds_rdc import double_sigma
            unit_class = double_sigma
        elif self == unit_types.sigma_double_sigma:
            from units.ds_rdc import sigma_double_sigma
            unit_class = sigma_double_sigma
        elif self == unit_types.double_sigma_n:
            from units.ds_rdc import double_sigma_normal
            unit_class = double_sigma_normal
        elif self == unit_types.double_sigma_trdc:
            from units.ds_rdc import double_sigma_trdc
            unit_class = double_sigma_trdc
        elif self == unit_types.sds_trdc:
            from units.ds_rdc import sigma_double_sigma_trdc
            unit_class = sigma_double_sigma_trdc
        elif self == unit_types.ds_n_trdc:
            from units.ds_rdc import double_sigma_normal_trdc
            unit_class = double_sigma_normal_trdc
        elif self == unit_types.ds_sharp:
            from units.ds_rdc import double_sigma_sharp
            unit_class = double_sigma_sharp
        elif self == unit_types.sds_sharp:
            from units.ds_rdc import sigma_double_sigma_sharp
            unit_class = sigma_double_sigma_sharp
        elif self == unit_types.ds_n_sharp:
            from units.ds_rdc import double_sigma_normal_sharp
            unit_class = double_sigma_normal_sharp
        elif self == unit_types.sds_n_sharp:
            from units.ds_rdc import sigma_double_sigma_normal_sharp
            unit_class = sigma_double_sigma_normal_sharp
        elif self == unit_types.sig_ssrdc_sharp:
            from units.ds_rdc import sig_ssrdc_sharp
            unit_class = sig_ssrdc_sharp
        elif self == unit_types.st_hr_sig:
            from units.ds_rdc import sliding_threshold_harmonic_rate_sigmoidal
            unit_class = sliding_threshold_harmonic_rate_sigmoidal
        elif self == unit_types.ss_hr_sig:
            from units.ds_rdc import synaptic_scaling_harmonic_rate_sigmoidal
            unit_class = synaptic_scaling_harmonic_rate_sigmoidal
        elif self == unit_types.sig_trdc:
            from units.ds_rdc import sig_trdc
            unit_class = sig_trdc
        elif self == unit_types.sig_ssrdc:
            from units.ds_rdc import sig_ssrdc
            unit_class = sig_ssrdc
        elif self == unit_types.sig_trdc_sharp:
            from units.ds_rdc import sig_trdc_sharp
            unit_class = sig_trdc_sharp
        elif self == unit_types.sds_n:
            from units.ds_rdc import sigma_double_sigma_normal
            unit_class = sigma_double_sigma_normal
        elif self == unit_types.ds_ssrdc_sharp:
            from units.ds_rdc import ds_ssrdc_sharp
            unit_class = ds_ssrdc_sharp
        elif self == unit_types.sds_n_ssrdc_sharp:
            from units.ds_rdc import sds_n_ssrdc_sharp
            unit_class = sds_n_ssrdc_sharp
        elif self == unit_types.mp_linear:
            from units.custom_units import mp_linear
            unit_class = mp_linear
        elif self == unit_types.custom_fi:
            from units.custom_units import custom_fi
            unit_class = custom_fi
        elif self == unit_types.custom_sc_fi:
            from units.custom_units import custom_scaled_fi
            unit_class = custom_scaled_fi
        elif self == unit_types.kwta:
            from units.custom_units import kWTA
            unit_class = kWTA
        elif self == unit_types.delta_linear:
            from units.custom_units import delta_linear
            unit_class = delta_linear
        elif self == unit_types.presyn_inh_sig:
            from units.custom_units import presyn_inh_sig
            unit_class = presyn_inh_sig
        # Temporary code from tutorial 5
        elif self == unit_types.binary:
            from units.custom_units import binary_unit
            unit_class = binary_unit
        # Models from spinal
        elif self == unit_types.test_oscillator:
            from units.custom_units import test_oscillator
            unit_class = test_oscillator
        elif self == unit_types.am_pm_oscillator:
            from units.spinal_units import am_pm_oscillator
            unit_class = am_pm_oscillator
        elif self == unit_types.out_norm_sig:
            from units.spinal_units import out_norm_sig 
            unit_class = out_norm_sig
        elif self == unit_types.out_norm_am_sig:
            from units.spinal_units import out_norm_am_sig 
            unit_class = out_norm_am_sig
        elif self == unit_types.logarithmic:
            from units.spinal_units import logarithmic
            unit_class = logarithmic
        elif self == unit_types.rga_sig:
            from units.spinal_units import rga_sig 
            unit_class = rga_sig 
        elif self == unit_types.gated_rga_sig:
            from units.spinal_units import gated_rga_sig 
            unit_class = gated_rga_sig 
        elif self == unit_types.gated_rga_adapt_sig:
            from units.spinal_units import gated_rga_adapt_sig 
            unit_class = gated_rga_adapt_sig 
        elif self == unit_types.act:
            from units.spinal_units import act 
            unit_class = act 
        elif self == unit_types.gated_out_norm_am_sig:
            from units.spinal_units import gated_out_norm_am_sig 
            unit_class = gated_out_norm_am_sig
        elif self == unit_types.gated_rga_inpsel_adapt_sig:
            from units.spinal_units import gated_rga_inpsel_adapt_sig 
            unit_class = gated_rga_inpsel_adapt_sig 
        elif self == unit_types.inpsel_linear:
            from units.spinal_units import inpsel_linear
            unit_class = inpsel_linear
        elif self == unit_types.chwr_linear:
            from units.spinal_units import chwr_linear
            unit_class = chwr_linear
        elif self == unit_types.am_oscillator:
            from units.spinal_units import am_oscillator
            unit_class = am_oscillator
        else:
            raise NotImplementedError('Attempting to retrieve the class of an ' + 
                                      'unknown unit model')
            # NameError instead?
        return unit_class

    def list_names():
        """ Return a list with the name of all defined units. """
        return [name for name, member in unit_types.__members__.items()]

    
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
    delta = 15 # continuous version of the delta rule

    switcher = 101 # from tutorial 5


    diff_hebbsnorm2 = 201 # a variation on diff_hebbsnorm
    anticov_inh = 202 # anticovariance for inhibitory synapses
    rga = 203 # relative gain array-inspired version of diff_hebbsnorm
    gated_rga = 204 # rga with modulated learning rate
    inp_sel = 205 # variant of input correlation
    chg = 206 # the change detection synapse (CHG)
    gated_inp_sel = 207 # inp sel with gated learning rate
    gated_diff_inp_sel = 208 # differential version of gated_inp_sel
    gated_bp_rga = 209 # gated_rga with betrayal punishment
    gated_rga_diff = 210 # gated_rga with double time delays
    gated_slide_rga_diff = 211 # gated_rga_diff with sliding time delay
    gated_normal_rga_diff = 212 # gated_rga_diff with normalized correlations
    gated_normal_slide_rga_diff = 213 # time delay slide, normalized correlations
    gated_normal_rga = 214 # gated_rga with normalized correlation

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
            from synapses.synapses import static_synapse
            syn_class = static_synapse
        elif self == synapse_types.oja:
            from synapses.synapses import oja_synapse
            syn_class = oja_synapse
        elif self == synapse_types.antihebb:
            from synapses.synapses import anti_hebbian_synapse
            syn_class = anti_hebbian_synapse
        elif self == synapse_types.cov:
            from synapses.synapses import covariance_synapse
            syn_class = covariance_synapse
        elif self == synapse_types.anticov:
            from synapses.synapses import anti_covariance_synapse
            syn_class = anti_covariance_synapse
        elif self == synapse_types.hebbsnorm:
            from synapses.synapses import hebb_subsnorm_synapse
            syn_class = hebb_subsnorm_synapse
        elif self == synapse_types.sq_hebbsnorm:
            from synapses.synapses import sq_hebb_subsnorm_synapse
            syn_class = sq_hebb_subsnorm_synapse
        elif self == synapse_types.inp_corr:
            from synapses.synapses import input_correlation_synapse
            syn_class = input_correlation_synapse
        elif self == synapse_types.bcm:
            from synapses.synapses import bcm_synapse
            syn_class = bcm_synapse
        elif self == synapse_types.homeo_inh:
            from synapses.synapses import homeo_inhib_synapse
            syn_class = homeo_inhib_synapse
        elif self == synapse_types.corr_inh:
            from synapses.synapses import corr_homeo_inhib_synapse
            syn_class = corr_homeo_inhib_synapse
        elif self == synapse_types.diff_hebbsnorm:
            from synapses.synapses import diff_hebb_subsnorm_synapse
            syn_class = diff_hebb_subsnorm_synapse
        elif self == synapse_types.exp_rate_dist:
            from synapses.synapses import exp_rate_dist_synapse
            syn_class = exp_rate_dist_synapse
        elif self == synapse_types.anticov_pre:
            from synapses.synapses import anti_cov_pre_synapse 
            syn_class = anti_cov_pre_synapse 
        elif self == synapse_types.delta:
            from synapses.synapses import delta_synapse 
            syn_class = delta_synapse 
        # Temporary code from tutorial 5
        elif self == synapse_types.switcher:
            from synapses.synapses import switcher
            syn_class = switcher
        # Synapses for the spinal model
        elif self == synapse_types.diff_hebbsnorm2:
            from synapses.spinal_syns import diff_hebb_subsnorm_synapse2
            syn_class = diff_hebb_subsnorm_synapse2
        elif self == synapse_types.anticov_inh:
            from synapses.spinal_syns import anti_covariance_inh_synapse
            syn_class = anti_covariance_inh_synapse
        elif self == synapse_types.rga:
            from synapses.spinal_syns import rga_synapse 
            syn_class = rga_synapse
        elif self == synapse_types.gated_rga:
            from synapses.spinal_syns import gated_rga_synapse 
            syn_class = gated_rga_synapse
        elif self == synapse_types.inp_sel:
            from synapses.spinal_syns import input_selection_synapse
            syn_class = input_selection_synapse
        elif self == synapse_types.chg:
            from synapses.spinal_syns import chg_synapse
            syn_class = chg_synapse 
        elif self == synapse_types.gated_inp_sel:
            from synapses.spinal_syns import gated_input_selection_synapse
            syn_class = gated_input_selection_synapse
        elif self == synapse_types.gated_diff_inp_sel:
            from synapses.spinal_syns import gated_diff_input_selection_synapse
            syn_class = gated_diff_input_selection_synapse
        elif self == synapse_types.gated_bp_rga:
            from synapses.spinal_syns import gated_bp_rga_synapse 
            syn_class = gated_bp_rga_synapse
        elif self == synapse_types.gated_rga_diff:
            from synapses.spinal_syns import gated_rga_diff_synapse 
            syn_class = gated_rga_diff_synapse
        elif self == synapse_types.gated_slide_rga_diff:
            from synapses.spinal_syns import gated_slide_rga_diff
            syn_class = gated_slide_rga_diff
        elif self == synapse_types.gated_normal_rga_diff:
            from synapses.spinal_syns import gated_normal_rga_diff
            syn_class = gated_normal_rga_diff
        elif self == synapse_types.gated_normal_slide_rga_diff:
            from synapses.spinal_syns import gated_normal_slide_rga_diff
            syn_class = gated_normal_slide_rga_diff
        elif self == synapse_types.gated_normal_rga:
            from synapses.spinal_syns import gated_normal_rga
            syn_class = gated_normal_rga
        else:
            raise NotImplementedError('Attempting retrieve the class of an unknown synapse model')
        
        return syn_class

    def list_names():
        """ Return a list with the name of all defined synapses. """
        return [name for name, member in synapse_types.__members__.items()]


class plant_models(Enum):
    """ An enum class with all the implemented plant models. """
    pendulum = 1
    point_mass_2D = 2
    conn_tester = 3
    simple_double_pendulum = 4
    compound_double_pendulum = 5
    planar_arm = 6
    bouncy_pendulum = 7
    bouncy_planar_arm = 8
    planar_arm_v2 = 9
    bouncy_planar_arm_v2 = 10
    planar_arm_v3 = 11
    bouncy_planar_arm_v3 = 12

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
            from plants.plants import pendulum 
            plant_class = pendulum 
        elif self == plant_models.point_mass_2D:
            from plants.plants import point_mass_2D
            plant_class = point_mass_2D
        elif self == plant_models.conn_tester:
            from plants.plants import conn_tester
            plant_class = conn_tester
        elif self == plant_models.simple_double_pendulum:
            from plants.plants import simple_double_pendulum
            plant_class = simple_double_pendulum
        elif self == plant_models.compound_double_pendulum:
            from plants.plants import compound_double_pendulum
            plant_class = compound_double_pendulum
        elif self == plant_models.planar_arm:
            from plants.plants import planar_arm
            plant_class = planar_arm
        elif self == plant_models.bouncy_pendulum:
            from plants.spinal_plants import bouncy_pendulum 
            plant_class = bouncy_pendulum
        elif self == plant_models.bouncy_planar_arm:
            from plants.spinal_plants import bouncy_planar_arm
            plant_class = bouncy_planar_arm
        elif self == plant_models.planar_arm_v2:
            from plants.plants import planar_arm_v2
            plant_class = planar_arm_v2
        elif self == plant_models.bouncy_planar_arm_v2:
            from plants.spinal_plants import bouncy_planar_arm_v2
            plant_class = bouncy_planar_arm_v2
        elif self == plant_models.planar_arm_v3:
            from plants.plants import planar_arm_v3
            plant_class = planar_arm_v3
        elif self == plant_models.bouncy_planar_arm_v3:
            from plants.spinal_plants import bouncy_planar_arm_v3
            plant_class = bouncy_planar_arm_v3
        else:
            raise NotImplementedError('Attempting to retrieve the class for an unknown plant model')
        return plant_class

    def list_names():
        """ Return a list with the name of all defined plants. """
        return [name for name, member in plant_models.__members__.items()]


class syn_reqs(Enum):
    """ This enum contains all the variables that a synapse model may ask a unit
        to maintain in order to support its learning function.

        For each requirement here there is a corresponding function or class
        in requirements.py, where more information can be found.
    """
    lpf_fast = 1  # postsynaptic activity low-pass filtered with a fast time constant
    lpf_mid = 2   # postsynaptic activity low-pass filtered with a medium time constant
    lpf_slow = 3  # postsynaptic activity low-pass filtered with a slow time constant
    pre_lpf_fast = 4  # presynaptic activity low-pass filtered with a fast time constant
    pre_lpf_mid = 5   # presynaptic activity low-pass filtered with a medium time constant
    pre_lpf_slow = 6  # presynaptic activity low-pass filtered with a slow time constant
    sq_lpf_slow = 7 # squared postsynaptic activity lpf'd with a slow time constant
    inp_avg_hsn = 8   # Sum of fast-LPF'd hebbsnorm inputs divided by number of hebbsnorm
                      # inputs
    pos_inp_avg_hsn = 9 # as inp_avg_hsn, but only considers inputs with positive weights
    err_diff = 10 # The approximate derivative of the error signal used in 
                   #input correlation
    sc_inp_sum_sqhsn = 11 # Scaled input sum from sq_hebssnorm synapses. 
    diff_avg = 12 # Average of derivatives for inputs with diff_hebb_subsnorm synapses.
    pos_diff_avg = 13 # As diff_avg, but only considers inputs with positive synaptic weights
    lpf_mid_inp_sum = 14 # LPF'd sum of presynaptic inputs with a medium time constant.
    n_erd = 15 # number of exp_rate_dist synapses on the postsynaptic unit. LEGACY CODE
    balance = 16 # fraction of inputs above, below, and around the current rate
    exp_scale = 17 # synaptic scaling to produce an exponential firing rate distribution
    inp_vector = 18 # the current "raw" input vector in a numpy array
    slide_thresh = 19 # An activation threshold adjusting to produce an exp rate distro
    lpf_slow_mp_inp_sum = 20 # slow LPF'd scaled sum of inputs from individual ports
    mp_inputs = 21 # the multiport inputs, as returned by the get_mp_inputs method
    balance_mp = 22 # the multiport version of the 'balance' requirement above
    slide_thresh_shrp = 23 # move threshold to produce exp rate distro if 
                           # 'sharpen' input > 0.5
    exp_scale_mp = 24 # scale factors to produce exp distro in multiport units
    slide_thr_hr = 25 # Sliding threshold to produce "harmonic" rate patterns (mp units)
    syn_scale_hr = 26 # Synaptic scaling to produce "harmonic" rate patterns (mp units)
    exp_scale_shrp = 27 # scale synapses to produce exp rate distro if 'sharpen' input > 0.5
    error = 28 # desired output minus true output, used for delta synapses
    inp_l2 = 29 # L2 norm of the inputs at port 0
    exp_scale_sort_mp = 30 # A single scale factor to produce exp rate distro in some 
                           # mp ssrdc units
    exp_scale_sort_shrp = 31 # A scale factor to produce exp rate distro in some 
                             # ssrdc_sharp units
    norm_factor = 32 # A scale factor to normalize the influence of each input on 
                     # presyn_inh_sig units
    exp_euler_vars = 33 # several variables used for the exponential Euler integrator
    
    # requirements used for the spinal model
    lpf_mid_mp_raw_inp_sum = 101 # mid LPF'd multiport list of raw input sums
    inp_deriv_mp =102 # derivatives of inputs listed by port (as in mp_inputs)
    avg_inp_deriv_mp = 103 # average of input derivatives for each port
    del_inp_deriv_mp = 104 # version of inp_deriv_mp with custom input delays
    del_avg_inp_deriv_mp = 105 # avg_inp_deriv_mp with custom input delays
    l0_norm_factor_mp = 106 # Factors to normalize the absolute sum of weight values
    out_norm_factor = 107 # factor to normalize the absolute sum of outgoing weights
    pre_out_norm_factor = 108 # shows that the presynaptic unit needs out_norm_factor 
    sc_inp_sum_diff_mp = 109 # derivative of the scaled input sum for each port
    lpf_fast_sc_inp_sum_mp = 110 # fast LPF'd scaled input sum per port
    lpf_mid_sc_inp_sum_mp = 111 # mid LPF'd scaled input sum per port
    lpf_slow_sc_inp_sum_mp = 112 # slow LPF'd scaled input sum per port
    lpf_slow_sc_inp_sum = 113 # slow LPF'd scaled input sum, single port
    acc_mid = 114 # accumulator with dynamics acc_mid' = (1 - acc_mid)/tau_mid
    acc_slow = 115 # accumuluator with dynamics acc_slow' = (1 - acc_slow)/tau_slow
    slow_decay_adapt = 116 # adapation factor, decays with tau_slow time constant
    mp_weights = 117 # weights by port, as returned by get_mp_weights()
    sc_inp_sum_mp = 118 # scaled sum of inputs by port
    integ_decay_act = 119 # integral of the activity with slow decay
    double_del_inp_deriv_mp = 120 # del_inp_deriv_mp with two values for two delays
    double_del_avg_inp_deriv_mp = 121 # del_avg_inp_deriv_mp with two delays
    slow_inp_deriv_mp = 122 # derivative of the inputs using lpf_mid and lpf_slow
    avg_slow_inp_deriv_mp = 123 # avarage of all slow_inp_deriv_mp values per port


    def list_names():
        """ Return a list with the name of all defined synaptic requirements. """
        return [name for name, member in syn_reqs.__members__.items()]

    def get_priority(self):
        """ Returns the priority of a given requirement.

            Requirements with higher priority (e.g. a lower priority number)
            are placed first in the unit's `functions` list, and are thus
            executed first.

        """
        high_priority = {'lpf_fast', 'lpf_mid', 'lpf_slow', 
                         'mp_inputs', 'mp_weights'}
        mid_priority = {'sc_inp_sum', 'inp_deriv_mp'}
        if self.name in high_priority:
            return 1
        elif self.name in mid_priority:
            return 2
        else:
            return 3
        

# Importing the classes used by the simulator
from network import *
import plants
import synapses
import units
import requirements
from tools.topology import *

