"""
requirements.py
This file contains functions to initialize all synaptic requirements used
in draculab units. For some requirements there are also classes that contain
initialization and updating methods.
"""

from draculab import unit_types, synapse_types, syn_reqs
from units import *
import numpy as np


def add_lpf_fast(unit):
    """ Adds 'lpf_fast', a low-pass filtered version of the unit's activity.

        The name lpf_fast indicates that the time constant of the low-pass filter,
        whose name is 'tau_fast', should be relatively fast. In practice this is
        arbitrary.
        The user needs to set the value of 'tau_fast' in the parameter dictionary 
        that initializes the unit.

        This method is meant to be called from init_pre_syn_update whenever the
        'lpf_fast' requirement is found. This method only takes care of the 
        initialization of the variables used for the lpf_fast requirement. Updating
        the value is done by unit.upd_lpf_fast, added to unit.functions by
        init_pre_syn_update. The buffer is re-initialized after each connection
        in unit.init_bufers .
    """
    if not hasattr(unit,'tau_fast'): 
        raise NameError( 'Synaptic plasticity requires unit parameter tau_fast, not yet set' )
    setattr(unit, 'lpf_fast', unit.init_val)
    setattr(unit, 'lpf_fast_buff', np.array( [unit.init_val]*unit.steps, dtype=unit.bf_type) )


def add_lpf_mid(unit):
    """ See 'add_lpf_fast' above. """
    if not hasattr(unit,'tau_mid'): 
        raise NameError( 'The tau_mid requirement needs the parameter tau_mid, not yet set' )
    setattr(unit, 'lpf_mid', unit.init_val)
    setattr(unit, 'lpf_mid_buff', np.array( [unit.init_val]*unit.steps, dtype=unit.bf_type) )


def add_lpf_slow(unit):
    """ See 'add_lpf_fast' above. """
    if not hasattr(unit,'tau_slow'): 
        raise NameError( 'The tau_slow requirement needs the parameter tau_slow, not yet set' )
    setattr(unit, 'lpf_slow', unit.init_val)
    setattr(unit, 'lpf_slow_buff', np.array( [unit.init_val]*unit.steps, dtype=unit.bf_type) )


def add_sq_lpf_slow(unit):
    """ Adds a low pass filtered version of the squared activity.
        
        As the name implies, the filter uses the "slow" time constant 'tau_slow'.
        The purpose of this is to have a mean value of the square of the activity,
        as used in some versions of the BCM learning rule. Accordingly, this 
        requirement is used by the bcm_synapse class.
    """
    if not hasattr(unit,'tau_slow'): 
        raise NameError( 'sq_lpf_slow requires unit parameter tau_slow, not yet set' )
    setattr(unit, 'sq_lpf_slow', unit.init_val)


def add_inp_vector(unit):
    """ Add a numpy array with the unit's inputs at the start of the simulation step.

        The inputs are not multiplied by synaptic weights, and come with their 
        appropriate delays.

        This input vector is used by other synaptic requirements, namely lpf_mid_inp_sum,
        balance, and exp_scale. In this last requirement, it is used to obtain the 'mu'
        factor used by the exp_scale rule.
    """
    setattr(unit, 'inp_vector', np.tile(unit.init_val, len(unit.net.syns[unit.ID])))


def add_mp_inputs(unit):
    """ Add a list with all the inputs, in the format of get_mp_inputs method. 

        In fact, the list is updated using the get_mp_inputs method. Some requirements
        like mp_balance and lpf_slow_mp_inp_sum use the output of get_mp_inputs, and
        the mp_inputs requirement saves computations by ensuring that get_mp_inputs
        only gets called once per simulation step.

        Quoting the docstring of unit.get_mp_inputs:
        "This method is for units where multiport = True, and that have a port_idx attribute.
        The i-th element of the returned list is a numpy array containing the raw (not multiplied
        by the synaptic weight) inputs at port i. The inputs include transmision delays."
    """
    if not hasattr(unit,'port_idx'): 
        raise NameError( 'the mp_inputs requirement is for multiport units with a port_idx list' )
    val = [] 
    for prt_lst in unit.port_idx:
        val.append(np.array([unit.init_val for _ in range(len(prt_lst))]))
    setattr(unit, 'mp_inputs', val)


def add_inp_avg_hsn(unit):
    """ Add an average of the inputs arriving at hebbsnorm synapses.

        In other words, the sum of fast-LPF'd hebbsnorm inputs divided by the the number
        of hebbsnorm inputs.

        Since we are using the lpf_fast signals, all the presynaptic units with hebbsnorm
        synapses need to have the lpf_fast requirement. 

        This initialization code calculates the 'snorm_list_dels' list used in
        upd_in_avg_hsn, and also the number of hebbsnorm synapses.
    """
    snorm_list = []  # a list with all the presynaptic units
                     # providing hebbsnorm synapses
    snorm_dels = []  # a list with the delay steps for each connection from snorm_list
    for syn in unit.net.syns[unit.ID]:
        if syn.type is synapse_types.hebbsnorm:
            if not syn_reqs.lpf_fast in unit.net.units[syn.preID].syn_needs:
                raise AssertionError('inp_avg_hsn needs lpf_fast on presynaptic units')
            snorm_list.append(unit.net.units[syn.preID])
            snorm_dels.append(syn.delay_steps)
    n_hebbsnorm = len(snorm_list) # number of hebbsnorm synapses received
    snorm_list_dels = list(zip(snorm_list, snorm_dels)) # both lists zipped
    setattr(unit, 'inp_avg_hsn', 0.2)  # arbitrary initialization of the average input
    setattr(unit, 'n_hebbsnorm', n_hebbsnorm)
    setattr(unit, 'snorm_list_dels', snorm_list_dels)


def add_pos_inp_avg_hsn(unit):
    """ Add an average of the inputs arriving at hebbsnorm synapses with positive weights.

        More precisely, the sum of fast-LPF'd hebbsnorm inputs divided by the the number
        of hebbsnorm inputs, but only considering inputs with positive synapses.

        Since we are using the lpf_fast signals, all the presynaptic units with hebbsnorm
        synapses need to have the lpf_fast requirement. 

        In addition to creating the 'pos_inp_avg_hsn' variable, this initialization code
        obtains four lists: n_vec, snorm_syns, snorm_units, snorm_delys.
        snorm_syns : a list with the all the hebbsnorm synapses
        snorm_units : a list with the presynaptic units for each entry in snorm_syns
        snorm_delys : a list with the delays for each connection of snorm_syns
        n_vec : n_vec[i]=1 if the i-th connection of snorm_syns has positive weight,
                and n_vec[i]=0 otherwise. The 'n' vector from Pg.290 of Dayan&Abbott.
    """
    snorm_units = []  # a list with all the presynaptic units
                      # providing hebbsnorm synapses
    snorm_syns = []  # a list with the synapses for the list above
    snorm_delys = []  # a list with the delay steps for these synapses
    for syn in unit.net.syns[unit.ID]:
        if syn.type is synapse_types.hebbsnorm:
            if not syn_reqs.lpf_fast in unit.net.units[syn.preID].syn_needs:
                raise AssertionError('pos_inp_avg_hsn needs lpf_fast on presynaptic units')
            snorm_syns.append(syn)
            snorm_units.append(unit.net.units[syn.preID])
            snorm_delys.append(syn.delay_steps)
    setattr(unit, 'pos_inp_avg_hsn', .2) # arbitrary initialization
    setattr(unit, 'snorm_syns', snorm_syns)
    setattr(unit, 'snorm_units', snorm_units)
    setattr(unit, 'snorm_delys', snorm_delys)
    setattr(unit, 'n_vec', np.ones(len(snorm_units)))


def add_err_diff(unit):
    """ Adds the approximate derivative of the error signal.

        The err_diff requirement is used by the input_correlation_synapse (inp_corr).

        The err_idx_dels list (see below) is also initialized.
    """
    err_idx = [] # a list with the indexes of the units that provide errors
    pred_idx = [] # a list with the indexes of the units that provide predictors 
    err_dels = [] # a list with the delays for each unit in err_idx
    err_diff = 0. # approximation of error derivative, updated by upd_err_diff
    for syn in unit.net.syns[unit.ID]:
        if syn.type is synapse_types.inp_corr:
            if syn.input_type == 'error':
                err_idx.append(syn.preID)
                err_dels.append(syn.delay_steps)
            elif syn.input_type == 'pred': 
                pred_idx.append(syn.preID) # not currently using this list
            else:
                raise ValueError('Incorrect input_type ' + str(syn.input_type) + ' found in synapse')
    err_idx_dels = list(zip(err_idx, err_dels)) # both lists zipped
    setattr(unit, 'err_diff', err_diff)
    setattr(unit, 'err_idx_dels', err_idx_dels)


def add_sc_inp_sum_sqhsn(unit):
    """ Adds the scaled input sum of fast LPF'd inputs from sq_hebbsnorm synapses. """
    sq_snorm_units = [] # a list with all the presynaptic neurons providing
                        # sq_hebbsnorm synapses
    sq_snorm_syns = []  # a list with all the sq_hebbsnorm synapses
    sq_snorm_dels = []  # a list with the delay steps for the sq_hebbsnorm synapses
    for syn in unit.net.syns[unit.ID]:
        if syn.type is synapse_types.sq_hebbsnorm:
            sq_snorm_syns.append(syn)
            sq_snorm_units.append(unit.net.units[syn.preID])
            sq_snorm_dels.append(syn.delay_steps)
    u_d_syn = list(zip(sq_snorm_units, sq_snorm_dels, sq_snorm_syns)) # all lists zipped
    setattr(unit, 'sc_inp_sum_sqhsn', 0.2)  # an arbitrary initialization 
    setattr(unit, 'u_d_syn', u_d_syn) 


def add_diff_avg(unit):
    """ Adds the average of derivatives for inputs with diff_hebb_subsnorm synapses."""
    dsnorm_list = [] # list with all presynaptic units providing
                     # diff_hebb_subsnorm synapses
    dsnorm_dels = [] # list with delay steps for each connection in dsnorm_list
    for syn in unit.net.syns[unit.ID]:
        if syn.type is synapse_types.diff_hebbsnorm:
            dsnorm_list.append(unit.net.units[syn.preID])
            dsnorm_dels.append(syn.delay_steps)
    n_dhebbsnorm = len(dsnorm_list) # number of diff_hebbsnorm synapses received
    dsnorm_list_dels = list(zip(dsnorm_list, dsnorm_dels)) # both lists zipped
    setattr(unit, 'diff_avg', 0.2)  # arbitrary initialization
    setattr(unit, 'n_dhebbsnorm', n_dhebbsnorm)
    setattr(unit, 'dsnorm_list_dels', dsnorm_list_dels)


def add_lpf_mid_inp_sum(unit):
    """ Adds the low-pass filtered sum of inputs.

        This requirement is used by the exp_rate_dist_synapse model.

        As the name suggests, the sum of inputs is filtered with the tau_mid time
        constant. The inputs include transmission delays, but are not multiplied
        by the synaptic weights. The sum of inputs comes from the inp_vector
        requirement.

        The lpf_mid_inp_sum keeps a buffer with past values. It is re-initialized
        by unit.init_buffers.
    """
    if not hasattr(unit,'tau_mid'): 
        raise NameError( 'Synaptic plasticity requires unit parameter tau_mid, not yet set' )
    if not syn_reqs.inp_vector in unit.syn_needs:
        raise AssertionError('lpf_mid_inp_sum requires the inp_vector requirement to be set')
    setattr(unit, 'lpf_mid_inp_sum', unit.init_val) # arbitrary initialization
    setattr(unit, 'lpf_mid_inp_sum_buff', 
            np.array( [unit.init_val]*unit.steps, dtype=unit.bf_type))


def add_lpf_slow_mp_inp_sum(unit):
    """ Adds a slow LPF'd scaled sum of inputs for each input port.

        lpf_slow_mp_inp_sum will be a list whose i-th element will be the sum of all
        the inputs at the i-th port, low-pass filtered with the 'tau_slow' time
        constant. The mp_inputs requirement is used for this.

        The lpf_slow_mp_inp_sum requirement is used by units of the 
        "double_sigma_normal" type in order to normalize their inputs according to
        the average of their sum.
    """
    if not hasattr(unit,'tau_slow'): 
        raise NameError( 'Requirement lpf_slow_mp_inp_sum requires parameter tau_slow, not yet set' )
    if not syn_reqs.mp_inputs in unit.syn_needs:
        raise AssertionError('lpf_slow_mp_inp_sum requires the mp_inputs requirement')
    setattr(unit, 'lpf_slow_mp_inp_sum', [ 0.4 for _ in range(unit.n_ports) ])


def add_error(unit):
    """ Adds the error value used by delta_linear units. """
    if not syn_reqs.mp_inputs in unit.syn_needs:
        raise AssertionError('The error requirement needs the mp_inputs requirement.')
    setattr(unit, 'error', 0.)
    setattr(unit, 'learning', 0.)


def add_inp_l2(unit):
    """ Adds the L2-norm of the vector with all inputs at port 0. 
    
        This requirement is used by delta_linear units to normalize their inputs.
    """
    if not syn_reqs.mp_inputs in unit.syn_needs:
        raise AssertionError('The inp_l2 requirement needs the mp_inputs requirement.')
    setattr(unit, 'inp_l2', 1.)



