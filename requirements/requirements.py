"""
requirements.py
This file contains functions to initialize all synaptic requirements used
in draculab units. For some requirements there are also classes that contain
initialization and updating methods.
"""

from draculab import unit_types, synapse_types, syn_reqs
import numpy as np

def add_propagator(unit, speed):
    """ Adds a propagator to update one of the LPF requirements.

        The propagator is a factor that can be used to update the value
        of an exponential function, which is the result of the equation
        for the first-order low-pass filter.

        This requirement can be added by any requirements that uses a LPF.

        Args:
            unit : the unit object where the propagator will be added.
            speed : A string. Either "slow", "mid", or "fast".
    """
    tau_name = "tau_" + speed
    prop_name = speed + "_prop"
    if not hasattr(unit, tau_name):
        raise NameError("A requirement needs the " + tau_name +
                        " value, which was not specified. ")
    min_del = unit.net.min_delay
    tau = eval('unit.'+tau_name)
    prop = np.exp(-min_del / tau)
    setattr(unit, prop_name, prop)
         

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
        raise NameError( 'lpf_fast requires unit parameter tau_fast, not yet set' )
    add_propagator(unit, 'fast')
    setattr(unit, 'lpf_fast', unit.init_act)
    setattr(unit, 'lpf_fast_buff', np.array( [unit.init_act]*unit.steps, dtype=unit.bf_type) )


def add_lpf_mid(unit):
    """ See 'add_lpf_fast' above. """
    if not hasattr(unit,'tau_mid'): 
        raise NameError( 'The lpf_mid requirement needs the parameter tau_mid, not yet set' )
    add_propagator(unit, 'mid')
    setattr(unit, 'lpf_mid', unit.init_act)
    setattr(unit, 'lpf_mid_buff', np.array( [unit.init_act]*unit.steps, dtype=unit.bf_type) )


def add_lpf_slow(unit):
    """ See 'add_lpf_fast' above. """
    if not hasattr(unit,'tau_slow'): 
        raise NameError( 'The lpf_slow requirement needs the parameter tau_slow, not yet set' )
    add_propagator(unit, 'slow')
    setattr(unit, 'lpf_slow', unit.init_act)
    setattr(unit, 'lpf_slow_buff', np.array( [unit.init_act]*unit.steps, dtype=unit.bf_type) )


def add_sq_lpf_slow(unit):
    """ Adds a low pass filtered version of the squared activity.
        
        As the name implies, the filter uses the "slow" time constant 'tau_slow'.
        The purpose of this is to have a mean value of the square of the activity,
        as used in some versions of the BCM learning rule. Accordingly, this 
        requirement is used by the bcm_synapse class.
    """
    if not hasattr(unit,'tau_slow'): 
        raise NameError( 'sq_lpf_slow requires unit parameter tau_slow, not yet set' )
    add_propagator(unit, 'slow')
    setattr(unit, 'sq_lpf_slow', unit.init_act)


def add_inp_vector(unit):
    """ Add a numpy array with the unit's inputs at the start of the simulation step.

        The inputs are not multiplied by synaptic weights, and come with their 
        appropriate delays.

        This input vector is used by other synaptic requirements, namely lpf_mid_inp_sum,
        balance, and exp_scale. In this last requirement, it is used to obtain the 'mu'
        factor used by the exp_scale rule.
    """
    setattr(unit, 'inp_vector', np.tile(unit.init_act, len(unit.net.syns[unit.ID])))


def add_mp_inputs(unit):
    """ Add a list with all the inputs, in the format of get_mp_inputs method. 

        In fact, the list is updated using the get_mp_inputs method. Some requirements
        like mp_balance and lpf_slow_mp_inp_sum use the output of get_mp_inputs, and
        the mp_inputs requirement saves computations by ensuring that get_mp_inputs
        only gets called once per simulation step.

        Quoting the docstring of unit.get_mp_inputs:
        "This method is for units where multiport = True, and that have a
        port_idx attribute.
        The i-th element of the returned list is a numpy array containing the raw 
        (not multiplied by the synaptic weight) inputs at port i. The inputs include 
        transmision delays."

        Since mp_inputs is used by other requirements, it has a priority of 1 so
        it gets updated before them.

        The update function is part of the `unit` class, and consists of the line:
        self.mp_inputs = self.get_mp_inputs(time)
    """
    if not hasattr(unit,'port_idx'): 
        raise NameError( 'the mp_inputs requirement is for multiport units ' +
                         'with a port_idx list' )
    val = [] 
    for prt_lst in unit.port_idx:
        val.append(np.array([unit.init_act for _ in range(len(prt_lst))]))
    setattr(unit, 'mp_inputs', val)


def add_inp_avg_mp(unit):
    """ Add the averages of the inputs at each port.

        This requirement uses the mp_inputs requirement in order to create the
        inp_avg_mp list.
        
        inp_avg_mp[i] is the average of the inputs at port i, not multiplied by
        their weights.
    """
    if not syn_reqs.mp_inputs in unit.syn_needs:
        raise AssertionError('The inp_avg_mp requirement depends on the ' +
                             'mp_inputs requirement. ')
    inp_avg_mp = []
    n_inps_mp = []
    uid = unit.ID
    for plist in unit.port_idx:
        inp_avg_mp.append(sum([unit.net.act[uid][idx](0.) for idx in plist]))
        n_inps_mp.append(len(plist))
    setattr(unit, 'inp_avg_mp', inp_avg_mp)
    # the port i average will come from the sum of inputs times inp_recip_mp[i]
    inp_recip_mp = [1./n if n>0 else 0. for n in n_inps_mp]
    setattr(unit, 'inp_recip_mp', inp_recip_mp)


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


def add_del_inp_mp(unit):
    """ Add an array of the raw inputs at each port with a custom delay.

        del_inp_mp is a list whose elements are lists of scalars.
        del_inp_mp[i] contains all inputs at port 'i', not multiplied by their 
        synaptic weights, with a delay given by the 'custom_inp_del' attribute
        of the unit plus the delay of the connection.

        The ports for which the inputs are obtained can be specified through
        the del_inp_ports attribute, which is added to the unit by passing it as
        a parameter to the rga_reqs constructor. del_inp_ports is a list of
        integers corresponding to the ports where the inputs will be obtained.
        Its default value includes all ports: list(range(unit.n_ports)).

        This requirement was created for the rga_ge synapse, and its
        implementation is in the rga_reqs class.

        Because this requirement is a prerequisite to del_inp_avg_mp, it has
        medium priorirty.
    """
    if not hasattr(unit,'port_idx'): 
        raise NameError( 'the del_inp_mp requirement is for multiport units ' +
                         'with a port_idx list' )
    # The delay is an attribute of the unit. Checking if it's there.
    if not hasattr(unit, 'custom_inp_del'):
        raise AssertionError('The del_inp_mp requirement needs units to have ' + 
                              'the attribute custom_inp_del.')
    # finding ports where the derivative will be calculated
    if not hasattr(unit, 'del_inp_ports'):
        setattr(unit, 'del_inp_ports', list(range(unit.n_ports)))
    syns = unit.net.syns[unit.ID]
    # dim_act[i,j] is the act function of the j-th input at the i-th port.
    # dim_del[i,j] is the delay of the j-th input at the i-th port.
    dim_act = []
    dim_del = []
    acts = unit.net.act[unit.ID]
    delys = unit.net.delays[unit.ID]
    cid = unit.custom_inp_del*unit.net.min_delay
    for p, lst in enumerate(unit.port_idx):
        if p in unit.del_inp_ports:
            dim_act.append([acts[uid] for uid in lst])
            dim_del.append([cid+delys[uid] for uid in lst])
        else:
            dim_act.append([])
            dim_del.append([])

    dim_act_del = []
    for da, dd in zip(dim_act, dim_del):
        dim_act_del.append(list(zip(da,dd)))

    setattr(unit, 'dim_act_del', dim_act_del)
    # initializing inputs
    del_inp_mp = [[a(0.) for a in l] for l in dim_act if len(l)>0]
    setattr(unit, 'del_inp_mp', del_inp_mp)




def add_del_inp_avg_mp(unit):
    """ Add the average of the raw inputs at each port with a custom delay.

        del_inp_avg_mp is a list of scalars.
        del_inp_avg_mp[i] contains the sum of all inputs at port 'i', not
        multiplied by their synaptic weights, with a delay given by the
        'custom_inp_del' attribute of the unit plus the delay of the connection.
        
        This requirement depends on the del_inp_mp requirement, from where it
        obtains the inputs.
        The ports for which the average is calculated can be specified through
        the del_inp_ports attricute, which is added to the unit by passing it as
        a parameter to the rga_reqs constructor. inp_avg_ports is a list of
        integers corresponding to the ports where the averages will be
        calculated. Its default value is: list(range(unit.n_ports)).
        del_inp_ports works by making the del_inp_mp requirement to consider
        the inputs only for the listed ports.

        This requirement was created for the rga_ge synapse, and its
        implementation is in the rga_reqs class.
    """
    if not hasattr(unit,'port_idx'): 
        raise NameError( 'the del_inp_avg_mp requirement is for multiport units ' +
                         'with a port_idx list' )
    # The delay is an attribute of the unit. Checking if it's there.
    if not hasattr(unit, 'custom_inp_del'):
        raise AssertionError('The del_inp_avg_mp requirement needs units to have ' + 
                              'the attribute custom_inp_del.')
    # The unit needs to have the del_inp_mp requirement
    if not syn_reqs.del_inp_mp in unit.syn_needs:
        raise AssertionError('The del_inp_avg_mp requirement depends on the ' +
                             'del_inp_mp requirement, which was not found.')
    del_inp_avg_mp = []
    n_inps_mp = []  # number of inputs at each port
    # initializing
    for plist in unit.port_idx:
        del_inp_avg_mp.append(sum([unit.net.units[idx].get_act(0.) 
                                   for idx in plist]))
        n_inps_mp.append(len(plist))
    setattr(unit, 'del_inp_avg_mp', del_inp_avg_mp)
    # the port i average will come from the sum of inputs times avg_fact_mp[i]
    avg_fact_mp = [1./n if n>0 else 0. for n in n_inps_mp]
    setattr(unit, 'avg_fact_mp', avg_fact_mp)
    

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
                raise ValueError('Incorrect input_type ' + str(syn.input_type) +
                                 ' found in synapse')
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
    """ Adds the average of derivatives for inputs with diff_hebb_subsnorm(2) synapses."""
    dsnorm_list = [] # list with all presynaptic units providing
                     # diff_hebb_subsnorm synapses
    dsnorm_dels = [] # list with delay steps for each connection in dsnorm_list
    for syn in unit.net.syns[unit.ID]:
        if (syn.type is synapse_types.diff_hebbsnorm or
            syn.type is synapse_types.diff_hebbsnorm2):
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
    if not hasattr(unit, 'mid_prop'):
        add_propagator(unit, 'mid')
    setattr(unit, 'lpf_mid_inp_sum', unit.init_act) # arbitrary initialization
    setattr(unit, 'lpf_mid_inp_sum_buff', 
            np.array( [unit.init_act]*unit.steps, dtype=unit.bf_type))


def add_lpf_slow_sc_inp_sum(unit):
    """ Adds the slow low-pass filtered scaled sum of inputs.

        This requirement was first used by the rga_sig unit model, but its
        implemenation is now in the unit class.

        As the name suggests, the sum of inputs is filtered with the tau_slow time
        constant. The inputs include transmission delays, and each one is multiplied
        by its synaptic weight. The sum of inputs comes from the get_input_sum
        function.

        Unlike lpf_slow_inp_sum, this requirement has no associated buffer at
        this point.
    """
    if not hasattr(unit,'tau_slow'): 
        raise NameError( 'The lpf_slow_sc_inp_sum requirement needs the ' +
                         'parameter tau_slow, not yet set' )
    if not hasattr(unit, 'slow_prop'):
        add_propagator(unit, 'slow')
    setattr(unit, 'lpf_slow_sc_inp_sum', unit.init_act) # arbitrary initialization


def add_lpf_slow_mp_inp_sum(unit):
    """ Adds a slow LPF'd scaled sum of inputs for each input port.

        lpf_slow_mp_inp_sum will be a list whose i-th element will be the scaled 
        sum of all the inputs at the i-th port, low-pass filtered with the 
        'tau_slow' time constant. The mp_inputs requirement is used for this.

        The lpf_slow_mp_inp_sum requirement is used by units of the 
        "double_sigma_normal" type in order to normalize their inputs according to
        the average of their sum.

        Units of the am_pm_oscillator class have their own implementation.
    """
    if not hasattr(unit,'tau_slow'): 
        raise NameError('Requirement lpf_slow_mp_inp_sum ' +
                        'requires parameter tau_slow, not yet set')
    if not syn_reqs.mp_inputs in unit.syn_needs:
        raise AssertionError('lpf_slow_mp_inp_sum requires the mp_inputs requirement')
    if not syn_reqs.mp_weights in unit.syn_needs:
        raise AssertionError('lpf_slow_mp_inp_sum requires the mp_weights requirement')
    if not hasattr(unit, 'slow_prop'):
        add_propagator(unit, 'slow')
    setattr(unit, 'lpf_slow_mp_inp_sum', [ 0.4 for _ in range(unit.n_ports) ])


def add_lpf_mid_mp_raw_inp_sum(unit):
    """ Adds a medium LPF'd raw sum of inputs for each input port.

        lpf_mid_mp_inp_sum will be a list whose i-th element is the sum of all
        the inputs at the i-th port, low-pass filtered with the 'tau_mid' time
        constant. The mp_inputs requirement is used for this.

        The lpf_mid_mp_raw_inp_sum requirement is used by units of the 
        am_pm_oscillator class.
    """
    if not hasattr(unit,'tau_mid'): 
        raise NameError( 'Requirement lpf_mid_mp_raw inp_sum requires ' +
                         'parameter tau_mid, not yet set' )
    if not syn_reqs.mp_inputs in unit.syn_needs:
        raise AssertionError('lpf_mid_mp_raw_inp_sum requires the mp_inputs requirement')
    if not hasattr(unit, 'mid_prop'):
        add_propagator(unit, 'mid')
    setattr(unit, 'lpf_mid_mp_raw_inp_sum', [ 0.4 for _ in range(unit.n_ports) ])


def add_balance(unit):
    """ Adds two numbers called  below, and above.

        below = fraction of inputs with rate lower than the unit.
        above = fraction of inputs with rate higher than the unit.

        Those numbers are useful to produce a given firing rate distribtuion by the
        various trdc and ssrdc models.
    """
    if not syn_reqs.inp_vector in unit.syn_needs:
        raise AssertionError('Balance requirement has the inp_vector ' +
                             'requirement as a prerequisite')
    setattr(unit, 'below', 0.5)
    setattr(unit, 'above', 0.5)


def add_balance_mp(unit):
    """ Updates two numbers called  below, and above. Used in multiport units.

        below = fraction of inputs with rate lower than the unit.
        above = fraction of inputs with rate higher than the unit.

        Those numbers are useful to produce a given firing rate distribtuion among
        the population of units that connect to the 'rdc_port'.

        This is the same as upd_balance, but ports other than the rdc_port are ignored.
    """
    if not syn_reqs.mp_inputs in unit.syn_needs:
        raise AssertionError('Balance_mp requirement has the mp_inputs ' +
                             'requirement as a prerequisite')
    setattr(unit, 'below', 0.5)
    setattr(unit, 'above', 0.5)


def add_exp_scale(unit):
    """ Adds the 'scale_facs' list, which specifies a scale factor for each weight.

        The scale factors are calculated so that the network acquires an exponential
        distribution of firing rates.
    """
    if not syn_reqs.balance in unit.syn_needs:
        raise AssertionError('exp_scale requires the balance requirement to be set')
    if not syn_reqs.lpf_fast in unit.syn_needs:
        raise AssertionError('exp_scale requires the lpf_fast requirement to be set')
    scale_facs= np.tile(1., len(unit.net.syns[unit.ID])) # array with scale factors
    # exc_idx = numpy array with index of all excitatory units in the input vector 
    exc_idx = [ idx for idx,syn in enumerate(unit.net.syns[unit.ID]) if syn.w >= 0]
    # ensure the integer data type; otherwise you can't index numpy arrays
    exc_idx = np.array(exc_idx, dtype='uint32')
    # inh_idx = numpy array with index of all inhibitory units 
    inh_idx = [idx for idx,syn in enumerate(unit.net.syns[unit.ID]) if syn.w < 0]
    inh_idx = np.array(inh_idx, dtype='uint32')
    setattr(unit, 'exc_idx', exc_idx)
    setattr(unit, 'inh_idx', inh_idx)
    setattr(unit, 'scale_facs', scale_facs)


def add_exp_scale_mp(unit):
    """ Adds the synaptic scaling factors used in multiport ssrdc units.

        The algorithm is the same as upd_exp_scale, but only the inputs at the rdc_port
        are considered.
    """
    if not syn_reqs.balance_mp in unit.syn_needs:
        raise AssertionError('exp_scale_mp requires the balance_mp requirement to be set')
    if not syn_reqs.lpf_fast in unit.syn_needs:
        raise AssertionError('exp_scale_mp requires the lpf_fast requirement to be set')
    scale_facs_rdc = np.tile(1., len(unit.port_idx[unit.rdc_port])) # array with scale factors
    # exc_idx_rdc = numpy array with index of all excitatory units at rdc port
    exc_idx_rdc = [ idx for idx,syn in enumerate([unit.net.syns[unit.ID][i] 
                         for i in unit.port_idx[unit.rdc_port]]) if syn.w >= 0 ]
    # ensure the integer data type; otherwise you can't index numpy arrays
    exc_idx_rdc = np.array(exc_idx_rdc, dtype='uint32')
    setattr(unit, 'exc_idx_rdc', exc_idx_rdc)
    setattr(unit, 'scale_facs_rdc', scale_facs_rdc)


def add_slide_thresh(unit):
    """ Adds a 'sliding threshold' for single-port trdc sigmoidal units.
    
        The sliding threshold is meant to produce an exponential distribution of firing
        rates in a population of units. The first unit type to use trdc (Threshold-based
        Rate Distribution Control) was exp_dist_sig_thr, which uses this requirement.

        Since the requirement adjusts the 'thresh' attribute already present in
        sigmoidal models, this initialization code adds no variables.
    """
    if (not syn_reqs.balance in unit.syn_needs) and (not syn_reqs.balance_mp in unit.syn_needs):
        raise AssertionError('slide_thresh requires the balance(_mp) requirement to be set')


def add_slide_thresh_shrp(unit):
    """ Adds a sliding threshold that falls back to a default value when signalled.

        The sliding threshold is meant to produce an exponential distribution of firing
        rates in a population of units. Units using this requirement will have a
        'sharpen_port', and when its signal is smaller than 0.5 the trdc will cease,
        and the threshold will revert to a fixed value.

        The units that use this requirement are those derived from the trdc_sharp_base
        class.
    """
    if not syn_reqs.balance_mp in unit.syn_needs:
        raise AssertionError('slide_thresh_shrp requires the balance_mp requirement to be set')
    if not unit.multiport:
        raise AssertionError('The slide_thresh_shrp is for multiport units only')


def add_slide_thr_hr(unit):
    """ Adds a sliding threshold that aims to produce a harmonic pattern of firing rates.

        See the sliding_threshold_harmonic_rate_sigmoidal unit type for the
        meaning of this.
    """
    if not syn_reqs.mp_inputs in unit.syn_needs:
        raise AssertionError('slide_thr_hr requires the mp_inputs requirement to be set')
    if not unit.multiport:
        raise AssertionError('The slide_thr_hr requirment is for multiport units only')


def add_syn_scale_hr(unit):
    """ Adds factors to scale the synaptic weights, producing 'harmonic rates'.

        Too see what this means see the synaptic_scaling_harmonic_rate_sigmoidal
        unit type.
    """
    if not syn_reqs.mp_inputs in unit.syn_needs:
        raise AssertionError('syn_scale_hr requires the mp_inputs requirement to be set')
    if not unit.multiport:
        raise AssertionError('The syn_scale_hr requirment is for multiport units only')
    if not syn_reqs.lpf_fast in unit.syn_needs:
        raise AssertionError('syn_scale_hr requires the lpf_fast requirement to be set')
    # exc_idx_hr = numpy array with index of all excitatory units at hr port
    exc_idx_hr = [ idx for idx,syn in enumerate([unit.net.syns[unit.ID][i] 
                         for i in unit.port_idx[unit.hr_port]]) if syn.w >= 0 ]
    # ensure the integer data type; otherwise you can't index numpy arrays
    exc_idx_hr = np.array(exc_idx_hr, dtype='uint32')
    #scale_facs = np.tile(1., len(unit.net.syns[unit.ID])) # array with scale factors
    scale_facs_hr = np.tile(1., len(unit.port_idx[unit.hr_port])) # array with scale factors
    setattr(unit, 'exc_idx_hr', exc_idx_hr)
    setattr(unit, 'scale_facs_hr', scale_facs_hr)


def add_exp_scale_shrp(unit):
    """ Adds the scaling factors used in ssrdc_sharp units not using sort_rdc.

        This requirement is found in units that inherit from the ssrdc_sharp_base class.
    """
    if not syn_reqs.balance_mp in unit.syn_needs:
        raise AssertionError('exp_scale_shrp requires the balance_mp requirement to be set')
    if not unit.multiport:
        raise AssertionError('The exp_scale_shrp requirement is for multiport units only')
    if not syn_reqs.lpf_fast in unit.syn_needs:
        raise AssertionError('exp_scale_shrp requires the lpf_fast requirement to be set')
    scale_facs_rdc = np.tile(1., len(unit.port_idx[unit.rdc_port])) # array with scale factors
    # exc_idx_rdc = numpy array with index of all excitatory units at rdc port
    exc_idx_rdc = [ idx for idx,syn in enumerate([unit.net.syns[unit.ID][i] 
                         for i in unit.port_idx[unit.rdc_port]]) if syn.w >= 0 ]
    # ensure the integer data type; otherwise you can't index numpy arrays
    exc_idx_rdc = np.array(exc_idx_rdc, dtype='uint32')
    rdc_exc_ones = np.tile(1., len(exc_idx_rdc))
    setattr(unit, 'exc_idx_rdc', exc_idx_rdc)
    setattr(unit, 'rdc_exc_ones', rdc_exc_ones)
    setattr(unit, 'scale_facs_rdc', scale_facs_rdc)


def add_exp_scale_sort_mp(unit):
    """ Adds the scaling factors used in the sig_ssrdc unit when sort_rdc==True.  """
    if not unit.multiport:
        raise AssertionError('The exp_scale_sort_mp requirement is for multiport units only')
    scale_facs_rdc = np.tile(1., len(unit.port_idx[unit.rdc_port])) # array with scale factors
    # exc_idx_rdc = numpy array with index of all excitatory units at rdc port
    exc_idx_rdc = [ idx for idx,syn in enumerate([unit.net.syns[unit.ID][i] 
                         for i in unit.port_idx[unit.rdc_port]]) if syn.w >= 0 ]
    autapse = False
    autapse_idx = len(exc_idx_rdc) - 1
    for idx, syn in enumerate([unit.net.syns[unit.ID][eir] for eir in exc_idx_rdc]):
        if syn.preID == unit.ID: # if there is an excitatory autapse at the rdc port
            autapse_idx = idx
            autapse = True
            break
    # Gotta produce the array of ideal rates given the truncated exponential distribution
    if autapse:
        N = len(exc_idx_rdc)
    else:
        N = len(exc_idx_rdc) + 1 # an extra point for the fake autapse
    points = [ (k + 0.5) / (N + 1.) for k in range(N) ]
    ideal_rates = np.array([-(1./unit.c) * np.log(1.-(1.-np.exp(-unit.c))*pt) for pt in points])
    setattr(unit, 'scale_facs_rdc', scale_facs_rdc) 
    setattr(unit, 'exc_idx_rdc', exc_idx_rdc) 
    setattr(unit, 'autapse', autapse) 
    setattr(unit, 'autapse_idx', autapse_idx) 
    setattr(unit, 'ideal_rates', ideal_rates) 


def add_exp_scale_sort_shrp(unit):
    """ Adds the scaling factors used in ssrdc_sharp units where sort_rdc==False.  
    
        This requirement is found in units that inherit from the ssrdc_sharp_base class.
        This initialization function is identical to add_exp_scale_sort_mp
    """
    if not unit.multiport:
        raise AssertionError('The exp_scale_sort_mp requirement is for multiport units only')
    scale_facs_rdc = np.tile(1., len(unit.port_idx[unit.rdc_port])) # array with scale factors
    # exc_idx_rdc = numpy array with index of all excitatory units at rdc port
    exc_idx_rdc = [ idx for idx,syn in enumerate([unit.net.syns[unit.ID][i] 
                         for i in unit.port_idx[unit.rdc_port]]) if syn.w >= 0 ]
    autapse = False
    autapse_idx = len(exc_idx_rdc) - 1
    for idx, syn in enumerate([unit.net.syns[unit.ID][eir] for eir in exc_idx_rdc]):
        if syn.preID == unit.ID: # if there is an excitatory autapse at the rdc port
            autapse_idx = idx
            autapse = True
            break
    # Gotta produce the array of ideal rates given the truncated exponential distribution
    if autapse:
        N = len(exc_idx_rdc)
    else:
        N = len(exc_idx_rdc) + 1 # an extra point for the fake autapse
    points = [ (k + 0.5) / (N + 1.) for k in range(N) ]
    ideal_rates = np.array([-(1./unit.c) * np.log(1.-(1.-np.exp(-unit.c))*pt) for pt in points])
    setattr(unit, 'scale_facs_rdc', scale_facs_rdc) 
    setattr(unit, 'exc_idx_rdc', exc_idx_rdc) 
    setattr(unit, 'autapse', autapse) 
    setattr(unit, 'autapse_idx', autapse_idx) 
    setattr(unit, 'ideal_rates', ideal_rates) 


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


def add_norm_factor(unit):
    """ Adds the normalization factor for inhibitory and excitatory inputs on the model cell.
    
        This requirement is used by the presyn_inh_sig units.
    """
    n_inh = 0 # count of inhibitory synapses 
    n_exc = 0 # count of excitatory synapses 
    for syn in unit.net.syns[unit.ID]:
        if syn.w < 0:
            n_inh += 1
        else:
            n_exc += 1
    if n_inh == 0:
        n_inh = 1
    if n_exc == 0:
        n_exc = 1
    setattr(unit, 's_inh', -unit.HYP/n_inh)
    setattr(unit, 's_exc', (1.+unit.OD)/n_exc)


def add_l1_norm_factor(unit):
    """ Factor to normalize the L0 norm of the weight vector.

        This requirement was originially created for the 'normalized' synapse
        class, and is implemented in the unit class.
        The requirement  is general enough to be used in any model where the 
        sum of absolute values for the weights should be 1.

        l1_norm_factor is 1 divided by the sum of absolute values for the
        weights in all the unit's synapses.
        By multiplying all the weights by this factors, the sum of their 
        absolute values will be 1 if the weight vector is not zero, and zero
        otherwise.
    """
    setattr(unit, 'l1_norm_factor', 1.)


def add_l1_norm_factor_mp(unit):
    """ Factors to normalize the L0 norm of weight vectors for each port.

        This requirement is used by the rga_synapse, and the implementation is
        in the unit class. The requirement  is general enough to be used in 
        any multiport model where the sum of absolute values for the
        weights should be 1.

        l1_norm_factor_mp is a list whose length is the number of ports.
        By multiplying all the weights from incoming connections at port i
        by l1_norm_fact0r_mp[i], the sum of their absolute values will be 1 if 
        the weight vector is not zero, and zero otherwise.
    """
    if not unit.multiport:
        raise AssertionError('The l1_norm_factor_mp requirement is for multiport '+
                             'units only.')
    if not syn_reqs.mp_weights in unit.syn_needs:
        raise AssertionError('The l1_norm_factor_mp requirement needs the ' +
                             'mp_weights requirement.')
    l1_norm_factor_mp = list(np.ones(unit.n_ports))
    setattr(unit, 'l1_norm_factor_mp', l1_norm_factor_mp)


def add_w_sum_mp(unit):
    """ Sum of synaptic weights for each port.

        The sum of signed weights can be used to created balanced excitation and
        inhibition, if during updates a fraction of this sum is substracted from
        all the weights. This must be done really slowly, or synapses will get
        stuck in their initial positions.

        Initially this was tested in the meca_hebb synapse, and the
        implementation is in the unit class.
    """
    if not unit.multiport:
        raise AssertionError('The w_sum_mp requirement is for multiport '+
                             'units only.')
    if not syn_reqs.mp_weights in unit.syn_needs:
        raise AssertionError('The w_sum_mp requirement needs the ' +
                             'mp_weights requirement.')
    w_sum_mp = list(np.ones(unit.n_ports))
    setattr(unit, 'w_sum_mp', w_sum_mp)


def add_inp_deriv_mp(unit):
    """ Adds the input derivatives listed by port.

        This is tantamount to a differential version of mp_inputs, so it works only
        for multiport units.

        inp_deriv_mp[i,j] will contain the derivative of the j-th input at the i-th
        port, following the order in the port_idx list.

        This requirement was created for the rga synapse, was originally in the
        am_pm_oscillator class, and then moved to the rga_reqs class in order to
        consolidate all versions. The optional parameter 'inp_deriv_ports' can
        be passed to the 'rga_reqs' constructor so that the input derivative is
        only calculated for the ports whose number is in the inp_deriv_ports
        list.

        Any synapse using this requirement should have the pre_lpf_fast and
        pre_lpf_mid requirements.
    """
    if not unit.multiport:
        raise AssertionError('The inp_deriv_mp requirement is for multiport units.')
    if not hasattr(unit, 'inp_deriv_ports'):
        setattr(unit, 'inp_deriv_ports', list(range(unit.n_ports)))
    # So that the unit can access the LPF'd activities of its presynaptic units
    # it needs a list with the ID's of the presynaptic units, arranged by port.
    # pre_list_mp[i,j] will contain the ID of the j-th input at the i-th port.
    # In addition, we need to know how many delay steps there are for each input.
    # pre_del_mp[i,j] will contain the delay steps of the j-th input at the i-th
    # port. Both lists will show no inputs in the ports not present in the
    # optional list inp_deriv_ports.
    syns = unit.net.syns[unit.ID]
    pre_list_mp = []
    pre_del_mp = []
    for p, lst in enumerate(unit.port_idx):
        if p in unit.inp_deriv_ports:  
            pre_list_mp.append([syns[uid].preID for uid in lst])
            pre_del_mp.append([syns[uid].delay_steps for uid in lst])
        else:
            pre_list_mp.append([])
            pre_del_mp.append([])
    pre_list_del_mp = list(zip(pre_list_mp, pre_del_mp)) # prezipping for speed
    # initializing all derivatives with zeros
    inp_deriv_mp = [[0. for uid in prt_lst] for prt_lst in pre_list_mp]
    setattr(unit, 'pre_list_del_mp', pre_list_del_mp)
    setattr(unit, 'inp_deriv_mp', inp_deriv_mp)


def add_avg_inp_deriv_mp(unit):
    """ Adds the average of the derivatives of all inputs at each port. 
        
        This requirement was created for the rga synapse, was originally in the
        am_pm_oscillator class, and then moved to the rga_reqs class in order to
        consolidate all versions. Depending on the optional parameter 
        'inp_deriv_ports' of the unit (used by upd_inp_deriv_mp) the
        derivatives may come only for some ports.

        The current implementation is in the rga_reqs class.

    """
    if not syn_reqs.inp_deriv_mp in unit.syn_needs:
        raise AssertionError('The avg_inp_deriv_mp requirement needs inp_deriv_mp')
    avg_inp_deriv_mp = [ 0. for _ in unit.port_idx]
    setattr(unit, 'avg_inp_deriv_mp', avg_inp_deriv_mp)


def add_del_inp_deriv_mp(unit):
    """ Adds input derivatives listed by port with custom delays.

        This is a version of inp_deriv_mp where the inputs have a custom delay.

        del_inp_deriv_mp[i,j] will contain the delayed derivative of the j-th 
        input at the i-th port, following the order in the port_idx list.

        This requirement was created for the rga synapse, was originally in the
        am_pm_oscillator class, and then moved to the rga_reqs class in order to
        consolidate all versions. The optional parameter 'inp_deriv_ports' can
        be passed to the 'rga_reqs' constructor so that the input derivative is
        only calculated for the ports whose number is in the inp_deriv_ports
        list.
    """
    if not unit.multiport:
        raise AssertionError('The del_inp_deriv_mp requirement is ' + 
                             'for multiport units.')
    #if (not syn_reqs.pre_lpf_fast in unit.syn_needs or
    #    not syn_reqs.pre_lpf_mid in unit.syn_needs):
    #    raise AssertionError('A unit with the del_inp_deriv_mp requierement needs its ' +
    #                         'presynaptic units to include lpf_fast and lpf_mid.')
    # The delay is an attribute of the unit. Checking if it's there.
    if not hasattr(unit, 'custom_inp_del'):
        raise AssertionError('The del_inp_deriv_mp requirement needs units to have ' + 
                              'the attribute custom_inp_del.')
    # finding ports where the derivative will be calculated
    if not hasattr(unit, 'inp_deriv_ports'):
        setattr(unit, 'inp_deriv_ports', list(range(unit.n_ports)))
    syns = unit.net.syns[unit.ID]
    # pre_list_mp[i,j] will contain the ID of the j-th input at the i-th port.
    pre_list_mp = []
    for lst in unit.port_idx:
        pre_list_mp.append([syns[uid].preID for uid in lst])
    # ports not in inp_deriv_ports will be considered empty
    pre_list_mp = [pre_list_mp[p] if p in unit.inp_deriv_ports else []
                   for p in range(unit.n_ports)]
    setattr(unit, 'pre_list_mp', pre_list_mp)
    # initializing all derivatives with zeros
    del_inp_deriv_mp = [[0. for uid in prt_lst] for prt_lst in pre_list_mp]
    setattr(unit, 'del_inp_deriv_mp', del_inp_deriv_mp)


def add_del_avg_inp_deriv_mp(unit):
    """ Adds the delayed average of the derivatives of all inputs at each port. 

        This requirement was created for the rga synapse, was originally in the
        am_pm_oscillator class, and then moved to the rga_reqs class in order to
        consolidate all versions. Depending on the optional parameter 
        'inp_deriv_ports' of the unit (used by upd_del_avg_inp_deriv_mp) the
        derivatives may come only for some ports.

        The current implementation is in the rga_reqs class.
    """
    if not syn_reqs.del_inp_deriv_mp in unit.syn_needs:
        raise AssertionError('The del_avg_inp_deriv_mp requirement ' + 
                             'needs del_inp_deriv_mp')
    del_avg_inp_deriv_mp = [ 0. for _ in unit.port_idx]
    setattr(unit, 'del_avg_inp_deriv_mp', del_avg_inp_deriv_mp)


def add_double_del_inp_deriv_mp(unit):
    """ Adds input derivatives listed by port for two custom delays.

        This is a version of del_inp_deriv_mp where two values are stored for
        each input, corresponding to two separate delays.

        double_del_inp_deriv_mp will be a list with 2 elements. Each element
        will be like the del_inp_deriv_mp attribute. In other words, both
        double_del_inp_deriv_mp[0] and double_del_inp_deriv_mp[1] will contain
        nested lists "del_inp_deriv_mp".
        del_inp_deriv_mp[i,j] contains the delayed derivative of the j-th
        input at the i-th port following the order in the port_idx list. 

        The first element in double_del_inp_deriv_mp corresponds to the delay
        custom_del_diff, whereas the second is for the delay custom_inp_del2.
        These delays are attributes of the postsynaptic unit. The postsynaptic
        unit receives the paramters custom_inp_del2 and custom_inp_del. From
        these we get: custom_del_diff = custom_inp_del2 - custom_inp_del.

        This requirement was created for the gated_bp_rga_diff synapse, and its
        implementation lives in the rga_reqs class of spinal_units.py.
        If it is used with a synapse type other than gated_rga_diff or
        gated_slide_rga_diff the code below must be modified so ddidm_idx is set
        on the synapse.
        
        The optional parameter 'inp_deriv_ports' can be passed to the 'rga_reqs'
        constructor so that the input derivative is only calculated for the
        ports whose number is in the inp_deriv_ports list.
    """
    if not unit.multiport:
        raise AssertionError('The double_del_inp_deriv_mp requirement is ' + 
                             'for multiport units.')
    # The delays are an attribute of the unit. Checking if they are there.
    if not (hasattr(unit, 'custom_inp_del') and hasattr(unit, 'custom_inp_del2')):
        raise AssertionError('The double del_inp_deriv_mp requirement needs units to' +
                             ' have  custom_inp_del and custom_inp_del2 attributes.')
    if not hasattr(unit, 'custom_del_diff'):
        raise AssertionError('The postsynaptic unit should obtain the ' +
                             'custom_del_diff attribute in its constructor')
    # finding ports where the derivative will be calculated
    if not hasattr(unit, 'inp_deriv_ports'):
        setattr(unit, 'inp_deriv_ports', list(range(unit.n_ports)))
    if syn_reqs.del_inp_deriv_mp in unit.syn_needs:
        raise AssertionError('The double_del_inp_deriv_mp and del_inp_deriv_mp'+
                             ' attributes should not be together in a unit.')
    syns = unit.net.syns[unit.ID]
    # pre_list_mp[i,j] will contain the ID of the j-th input at the i-th port.
    pre_list_mp = []
    for lst in unit.port_idx:
        pre_list_mp.append([syns[uid].preID for uid in lst])
    # ports not in inp_deriv_ports will be considered empty
    pre_list_mp = [pre_list_mp[p] if p in unit.inp_deriv_ports else []
                   for p in range(unit.n_ports)]
    setattr(unit, 'pre_list_mp', pre_list_mp)
    # initializing all derivatives with zeros
    didm1 = [[0. for uid in prt_lst] for prt_lst in pre_list_mp]
    didm2 = [[0. for uid in prt_lst] for prt_lst in pre_list_mp]
    setattr(unit, 'double_del_inp_deriv_mp', [didm1, didm2])
    # the gated_rga_diff synapses need their indexes in the didm1 and didm2
    # lists. We set it for them here.
    syns = unit.net.syns
    for p in unit.inp_deriv_ports:
        for loc_idx, syn_idx in enumerate(unit.port_idx[p]):
            if (syns[unit.ID][syn_idx].type is synapse_types.gated_rga_diff or
                syns[unit.ID][syn_idx].type is synapse_types.gated_slide_rga_diff or
                (syns[unit.ID][syn_idx].type is 
                    synapse_types.gated_normal_rga_diff) or
                (syns[unit.ID][syn_idx].type is 
                    synapse_types.noisy_gated_normal_rga_diff) or
                (syns[unit.ID][syn_idx].type is 
                    synapse_types.gated_normal_slide_rga_diff)):
                setattr(syns[unit.ID][syn_idx], 'ddidm_idx', loc_idx)
                assert pre_list_mp[p][loc_idx] == syns[unit.ID][syn_idx].preID, [
                       'Failed sanity check at add_double_del_inp_deriv_mp']
 

def add_double_del_avg_inp_deriv_mp(unit):
    """ Adds the average of double_del_avg_inp_deriv_mp for each port.

        double_del_avg_inp_deriv_mp is a list with two entries. Each one is like
        a copy of del_avg_inp_deriv_mp, but they correspond to two different
        delays. 
        del_avg_inp_deriv_mp[i] provides the average of the derivatives for all
        inputs arriving at port i, with a given delay.

        This requirement was created for the gated_bp_rga_diff synapse, and its
        implementation lives in the rga_reqs class of spinal_units.py.
        
        The optional parameter 'inp_deriv_ports' can be passed to the 'rga_reqs'
        constructor so that the input derivative is only calculated for the
        ports whose number is in the inp_deriv_ports list.
    """
    if not syn_reqs.double_del_inp_deriv_mp in unit.syn_needs:
        raise AssertionError('The double_del_avg_inp_deriv_mp requirement ' + 
                             'needs double_del_inp_deriv_mp')
    daidm1 = [0. for _ in unit.port_idx]
    daidm2 = [0. for _ in unit.port_idx]
    setattr(unit, 'double_del_avg_inp_deriv_mp', [daidm1, daidm2])


def add_slow_inp_deriv_mp(unit):
    """ Adds the slow input derivatives listed by port.

        The slow input derivatives come from (lpf_mid - lpf_slow).
        As with inp_deriv_mp, this works only for multiport units.

        slow inp_deriv_mp[i,j] will contain the slow derivative of the j-th
        input at the i-th port, following the order in the port_idx list.

        This requirement was created for normal rga synapses, with its
        implementation in the rga_reqs class.

        The optional parameter 'inp_deriv_ports' can
        be passed to the 'rga_reqs' constructor so that the input derivative is
        only calculated for the ports whose number is in the inp_deriv_ports
        list.

        Any synapse using this requirement should have the pre_lpf_mid and
        pre_lpf_slow requirements.
    """
    if not unit.multiport:
        raise AssertionError('The slow inp_deriv_mp requirement is for ' + 
                             'multiport units.')
    if not hasattr(unit, 'inp_deriv_ports'):
        setattr(unit, 'inp_deriv_ports', list(range(unit.n_ports)))
    # So that the unit can access the LPF'd activities of its presynaptic units
    # it needs a list with the ID's of the presynaptic units, arranged by port.
    # pre_list_mp[i,j] will contain the ID of the j-th input at the i-th port.
    # In addition, we need to know how many delay steps there are for each input.
    # pre_del_mp[i,j] will contain the delay steps of the j-th input at the i-th
    # port. Both lists will show no inputs in the ports not present in the
    # optional list inp_deriv_ports.
    # inp_deriv_mp also adds pre_list_del_mp. You should add it in both
    # requirements so the lists stay up to date with any new synapses.
    syns = unit.net.syns[unit.ID]
    pre_list_mp = []
    pre_del_mp = []
    for p, lst in enumerate(unit.port_idx):
        if p in unit.inp_deriv_ports:  
            pre_list_mp.append([syns[uid].preID for uid in lst])
            pre_del_mp.append([syns[uid].delay_steps for uid in lst])
        else:
            pre_list_mp.append([])
            pre_del_mp.append([])
    pre_list_del_mp = list(zip(pre_list_mp, pre_del_mp)) # prezipping
    setattr(unit, 'pre_list_del_mp', pre_list_del_mp)
    # initializing all derivatives with zeros
    slow_inp_deriv_mp = [[0. for uid in prt_lst] for prt_lst in pre_list_mp]
    setattr(unit, 'slow_inp_deriv_mp', slow_inp_deriv_mp)
    # the normal_rga synapses need a list with their index in the
    # slow_inp_deriv_mp list
    syns = unit.net.syns
    for p in unit.inp_deriv_ports:
        for loc_idx, syn_idx in enumerate(unit.port_idx[p]):
            setattr(syns[unit.ID][syn_idx], 'sid_idx', loc_idx)
            assert pre_list_mp[p][loc_idx] == syns[unit.ID][syn_idx].preID, [
                   'Failed sanity check at add_slow_inp_deriv_mp']

def add_avg_slow_inp_deriv_mp(unit):
    """ Adds the average of the slow derivatives of all inputs at each port. 
        
        This requirement was created for normal rga synapses, with its
        implementation in the rga_reqs class.
        Depending on the optional parameter 'inp_deriv_ports' of the unit 
        (used by upd_slow_inp_deriv_mp) the derivatives may come only for 
        some ports.
    """
    if not syn_reqs.slow_inp_deriv_mp in unit.syn_needs:
        raise AssertionError('The avg_slow_inp_deriv_mp requirement needs ' +
                             'the slow_inp_deriv_mp requirement')
    avg_slow_inp_deriv_mp = [ 0. for _ in unit.port_idx]
    setattr(unit, 'avg_slow_inp_deriv_mp', avg_slow_inp_deriv_mp)


def add_sc_inp_sum_deriv_mp(unit):
    """ Add a list with the derivatives of scaled input sums per port.

        inp_sum_deriv_mp[i] will be the derivative of the scaled sum of inputs at
        port i. This derivative comes from the scaled sum of individual 
        derivatives calculated using the inp_deriv_mp requirement.

        This requirement was initially used with the rga_ge synapse,
        and its update implementation is currently in the rga_reqs
        class of spinal_units.py .

        The sc_inp_sum_diff_mp requirement does the same thing, but relies on
        the lpf_fast/mid_sc_inp_sum_mp requirements, rather than inp_deriv_mp.
    """
    if not syn_reqs.inp_deriv_mp in unit.syn_needs:
        raise AssertionError('The sc_inp_sum_deriv_mp requirement needs the ' + 
                             'inp_deriv_mp requirement.')
    if not syn_reqs.mp_weights in unit.syn_needs:
        raise AssertionError('The sc_inp_sum_deriv_mp requirement needs the ' + 
                             'mp_weights requirement.')
    if not unit.multiport:
        raise AssertionError('The sc_inp_sum_deriv_mp requirement is for ' +
                             'multiport units')
    sc_inp_sum_deriv_mp = [0.]*unit.n_ports
    setattr(unit, 'sc_inp_sum_deriv_mp', sc_inp_sum_deriv_mp)


def add_lpf_fast_sc_inp_sum_mp(unit):
    """ Add the fast LPF'd scaled sum of inputs for each port separately. 

        This requirement was initially used with the input_selection synapse,
        and its update implementation is currently in the out_norm_am_sig unit.
    """
    if not unit.multiport:
        raise AssertionError('The lpf_fast_sc_inp_sum_mp requirement is for ' +
                             'multiport units')
    if not hasattr(unit,'tau_fast'): 
        raise NameError( 'Requirement lpf_fast_sc_inp_sum requires ' +
                         'parameter tau_fast, not yet set' )
    if not hasattr(unit, 'fast_prop'):
        add_propagator(unit, 'fast')
    if (not syn_reqs.mp_weights in unit.syn_needs or
        not syn_reqs.mp_inputs in unit.syn_needs):
        raise AssertionError('Requirement lpf_fast_sc_inp_sum_mp depends on ' +
                             'the mp_weights and mp_inputs requirements.')
    setattr(unit, 'lpf_fast_sc_inp_sum_mp', [0.2]*unit.n_ports)


def add_xtra_del_inp_deriv_mp(unit):
    """ Adds input derivatives listed by port with normal plus extra delays.

        This is a version of inp_deriv_mp where the inputs have an extra delay
        that gets added to the normal delay. The difference with del_inp_deriv_mp
        is that, unlike this requirement, it doesn't add the normal delay.

        xtra_del_inp_deriv_mp[i,j] will contain the delayed derivative of the
        j-th input at the i-th port, following the order in the port_idx list.

        This requirement was created for the gated_diff_inp-corr synapse, and
        its implementation is in the rga_reqs class of spinal_syns.py.
        
        The optional parameter 'xd_inp_deriv_p' can be passed to the 
        'rga_reqs' constructor so that the input derivative is
        only calculated for the ports whose number is in the inp_deriv_ports
        list.
    """
    if not unit.multiport:
        raise AssertionError('The xtra_del_inp_deriv_mp requirement is ' + 
                             'for multiport units.')
    # The delay is an attribute of the unit. Checking if it's there.
    if not hasattr(unit, 'xtra_inp_del'):
        raise AssertionError('The del_inp_deriv_mp requirement needs units to have ' + 
                              'the attribute xtra_inp_del.')
    # finding ports where the derivative will be calculated
    if not hasattr(unit, 'xd_inp_deriv_p'):
        setattr(unit, 'xd_inp_deriv_p', list(range(unit.n_ports)))
    syns = unit.net.syns[unit.ID]

    # So that the unit can access the LPF'd activities of its presynaptic units
    # it needs a list with the ID's of the presynaptic units, arranged by port.
    # pre_uid_mp[i,j] will contain the ID of the j-th input at the i-th port.
    # In addition, we need to know how many delay steps there are for each input.
    # pre_del_mp[i,j] will contain the delay steps of the j-th input at the i-th
    # port. Both lists will show no inputs in the ports not present in the
    # optional list xd_inp_deriv_p.
    syns = unit.net.syns[unit.ID]
    pre_uid_mp = []
    pre_del_mp = []
    for p, lst in enumerate(unit.port_idx):
        if p in unit.xd_inp_deriv_p:  
            pre_uid_mp.append([syns[uid].preID for uid in lst])
            pre_del_mp.append([syns[uid].delay_steps for uid in lst])
        else:
            pre_uid_mp.append([])
            pre_del_mp.append([])
    pre_uid_del_mp = list(zip(pre_uid_mp, pre_del_mp)) # prezipping for speed
    setattr(unit, 'pre_uid_del_mp', pre_uid_del_mp)
    # initializing all derivatives with zeros
    xdidm = [[0. for uid in prt_lst] for prt_lst in pre_uid_mp]
    setattr(unit, 'xtra_del_inp_deriv_mp', xdidm)
 

def add_xtra_del_inp_deriv_mp_sc_sum(unit):
    """ Add scaled sum of input derivatives per port with extra delays.

        xtra_del_inp_deriv_mp_sc_sum[i] is the scaled sum of all the inputs in
        xtra_del_inp-deriv_mp[i], e.g. each input multiplied by its
        corresponding entry in mp_weights[i].
    """
    #if not 'xtra_del_inp_deriv_mp' in unit.syn_needs:
    #    raise AssertionError('xtra_del_inp_deriv_mp_sc_sum relies on the ' +
    #           'xtra_del_inp_deriv_mp requirement, not found in the unit.')
    # initializing the scaled sums with zeros
    xdidmss = [0. for _ in range(unit.n_ports)]
    setattr(unit, 'xtra_del_inp_deriv_mp_sc_sum', xdidmss)


def add_exp_euler_vars(unit):
    """ Adds several variables used by the exp_euler integration method. """
    dt = unit.times[1] - unit.times[0] # same as unit.time_bit
    A = -unit.lambd*unit.rtau
    eAt = np.exp(A*dt)
    c2 = (eAt-1.)/A
    c3 = np.sqrt( (eAt**2. - 1.) / (2.*A) )
    sc3 = unit.sigma * c3 # used by flat_exp_euler_update
    upd_exp_euler_vars = lambda t: None
    setattr(unit, 'eAt', eAt)
    setattr(unit, 'c2', c2)
    setattr(unit, 'c3', c3)
    setattr(unit, 'sc3', sc3)
    setattr(unit, 'upd_exp_euler_vars', upd_exp_euler_vars)


def add_out_norm_factor(unit):
    """ Add a factor to normalize the sum of absolute values for outgoing weights. 
    
        A unit having this requirement will, on each update, calculate the sum
        of the absolute value of the weights for all the projections it sends.
        The out_norm_factor value produced comes from a parameter called
        des_out_w_abs_sum divided by the value of this sum.

        If the optional paramter 'out_norm_type' is included in the postsynaptic
        unit, only the synapses whose type has a value equal to this integer
        will be included in the computation of the factor.

        The update function implementaion is part of the unit class. 
    """
    if not hasattr(unit, 'des_out_w_abs_sum'):
        raise AssertionError('The out_norm_factor requirement needs the ' +
                             'des_out_w_abs_sum attribute in its unit.')
    if hasattr(unit, 'out_norm_type'):
        sel_type = True
        syn_type_val = unit.out_norm_type
    else:
        sel_type = False
        syn_type_val = None
    # Obtain indexes to all of the unit's connections in net.syns
    out_syns_idx = []
    out_w_abs_sum = 0.
    for list_idx, syn_list in enumerate(unit.net.syns):
        for syn_idx, syn in enumerate(syn_list):
            if syn.preID == unit.ID:
                if ((sel_type is True and syn_type_val == syn.type.value) or
                    sel_type is False):                    
                    out_syns_idx.append((list_idx, syn_idx))
                    out_w_abs_sum += abs(syn.w)
                
    out_norm_factor = unit.des_out_w_abs_sum / (out_w_abs_sum + 1e-32)
    setattr(unit, 'out_syns_idx', out_syns_idx)
    setattr(unit, 'out_norm_factor', out_norm_factor)


def add_sc_inp_sum_diff_mp(unit):
    """ Add a list with the derivatives of scaled input sums per port.

        inp_sum_diff_mp[i] will be the derivative of the scaled sum of inputs at
        port i. This derivative comes from the fast lpf'd input sum minus the mid
        lpf'd input sum. These two values come from the lpf_fast_sc_inp_sum_mp, 
        and lpf_mid_sc_inp_sum_mp requirements.

        This requirement was initially used with the input_selection synapse,
        and its update implementation is currently in the lpf_sc_inp_sum_mp_reqs
        class of spinal_units.py .

        The sc_inp_sum_deriv_mp requirement does the same thing, but relies on
        the inp_deriv_mp requirement rather than the lpf_fast/mid_sc_inp_sum_mp
        requirements.
    """
    if (not syn_reqs.lpf_fast_sc_inp_sum_mp in unit.syn_needs or
        not syn_reqs.lpf_mid_sc_inp_sum_mp in unit.syn_needs):
        raise AssertionError('The sc_inp_sum_diff_mp requirement  needs both' + 
                             'lpf_fast_inp_sum_mp, and lpf_mid_inp_sum_mp')
    if not unit.multiport:
        raise AssertionError('The sc_inp_sum_diff_mp requirement is for ' +
                             'multiport units')
    sc_inp_sum_diff_mp = [0.]*unit.n_ports
    setattr(unit, 'sc_inp_sum_diff_mp', sc_inp_sum_diff_mp)


def add_lpf_fast_sc_inp_sum_mp(unit):
    """ Add the fast LPF'd scaled sum of inputs for each port separately. 

        This requirement was initially used with the input_selection synapse,
        and its update implementation is currently in the out_norm_am_sig unit.
    """
    if not unit.multiport:
        raise AssertionError('The lpf_fast_sc_inp_sum_mp requirement is for ' +
                             'multiport units')
    if not hasattr(unit,'tau_fast'): 
        raise NameError( 'Requirement lpf_fast_sc_inp_sum requires ' +
                         'parameter tau_fast, not yet set' )
    if not hasattr(unit, 'fast_prop'):
        add_propagator(unit, 'fast')
    if (not syn_reqs.mp_weights in unit.syn_needs or
        not syn_reqs.mp_inputs in unit.syn_needs):
        raise AssertionError('Requirement lpf_fast_sc_inp_sum_mp depends on ' +
                             'the mp_weights and mp_inputs requirements.')
    setattr(unit, 'lpf_fast_sc_inp_sum_mp', [0.2]*unit.n_ports)


def add_lpf_mid_sc_inp_sum_mp(unit):
    """ Add the medium LPF'd scaled sum of inputs for each port separately. 

        This requirement was initially used with the input_selection synapse,
        and its update implementation is currently in the logarithmic unit.
    """
    if not unit.multiport:
        raise AssertionError('The lpf_mid_sc_inp_sum_mp requirement is for ' +
                             'multiport units')
    if not hasattr(unit,'tau_mid'): 
        raise NameError( 'Requirement lpf_mid_sc_inp_sum_mp requires ' +
                         'parameter tau_mid, not yet set' )
    if not hasattr(unit, 'mid_prop'):
        add_propagator(unit, 'mid')
    if (not syn_reqs.mp_weights in unit.syn_needs or
        not syn_reqs.mp_inputs in unit.syn_needs):
        raise AssertionError('Requirement lpf_mid_sc_inp_sum_mp depends on ' +
                             'the mp_weights and mp_inputs requirements.')
    setattr(unit, 'lpf_mid_sc_inp_sum_mp', [0.2]*unit.n_ports)


def add_lpf_slow_sc_inp_sum_mp(unit):
    """ Add the slow LPF'd scaled sum of inputs for each port separately. 

        This requirement was initially used with the act unit, where its
        update implementation currently resides.
    """
    if not unit.multiport:
        raise AssertionError('The lpf_slow_sc_inp_sum_mp requirement is for ' +
                             'multiport units')
    if not hasattr(unit,'tau_slow'): 
        raise NameError( 'Requirement lpf_slow_sc_inp_sum_mp requires ' +
                         'parameter tau_slow, not yet set' )
    if not hasattr(unit, 'slow_prop'):
        add_propagator(unit, 'slow')
    if (not syn_reqs.mp_weights in unit.syn_needs or
        not syn_reqs.mp_inputs in unit.syn_needs):
        raise AssertionError('Requirement lpf_slow_sc_inp_sum_mp depends on ' +
                             'the mp_weights and mp_inputs requirements.')
    setattr(unit, 'lpf_slow_sc_inp_sum_mp', [0.2]*unit.n_ports)


def add_acc_mid(unit):
    """ Add value that approaches 1 asymptotically.

        acc_mid' = (1 - acc_mid) / tau_mid
        This was originally used in the rga_sig units as a part of a mechanism
        that stopped learning for a moment whenever the target was changed.

        Currently upd_acc_mid is in the acc_sda_reqs class. The implementation
        resets the value to 0 if the scaled input sum at port 'i' is larger than
        0.5, where i is the value of the 'acc_port' parameter (0 by default).
    """
    if not hasattr(unit,'tau_mid'): 
        raise NameError( 'Requirement acc_mid requires the ' +
                         'parameter tau_mid, not yet set' )
    if not hasattr(unit, 'mid_prop'):
        add_propagator(unit, 'mid')
    setattr(unit, 'acc_mid', 0.)
    if not hasattr(unit, 'acc_mid_port'):
        from warnings import warn
        warn('setting the 0 value for the acc_mid reset port')
        setattr(unit, 'acc_mid_port', 0)


def add_acc_slow(unit):
    """ Add value that approaches 1 asymptotically.

        acc_slow' = (1 - acc_slow) / tau_slow
        This was originally used in the rga_sig units as a part of a mechanism
        that stopped learning for a moment whenever the target was changed.

        Currently upd_acc_slow is in the acc_sda_reqs class. The implementation
        resets the value to 0 if the scaled input sum at port 'i' is larger than
        0.5, where i is the value of the 'acc_port' parameter (0 by default).
    """
    if not hasattr(unit,'tau_slow'): 
        raise NameError( 'Requirement acc_slow requires the ' +
                         'parameter tau_slow, not yet set' )
    if not hasattr(unit, 'slow_prop'):
        add_propagator(unit, 'slow')
    setattr(unit, 'acc_slow', 0.)
    if not hasattr(unit, 'acc_slow_port'):
        from warnings import warn
        warn('setting the 0 value for the acc_slow reset port')
        setattr(unit, 'acc_slow_port', 0)


def add_slow_decay_adapt(unit):
    """ Add an adaptation factor that decays with tau_slow time constant.

        This requirement was designed to implement gated adaptation in the
        gated_rga_adap_sig model. The update implementation is now in the
        acc_sda_reqs class.

        There is one port called the sda_port that resets the adaptation.
        When the inputs at this port are below 0.8, the adaptation will
        exponentially decay to 0. When the port 3 inputs exceed 0.8 the
        adaptation will be reset to its initial value, which depends on the slow
        LPF'd value of the activity. Afterwards no further reset will be 
        possible until the adaptation falls below 0.2.
    """
    if not hasattr(unit, 'tau_slow'):
        raise NameError( 'Requirement slow_decay_adapt requires the ' +
                         'parameter tau_slow, not yet set' )
    if (not syn_reqs.lpf_slow in unit.syn_needs):
        raise AssertionError('The slow_decay_adapt requirement  needs the ' + 
                             'lpf_slow requirement')
    setattr(unit, 'slow_decay_adapt', 0.)
    if not hasattr(unit, 'sda_port'):
        from warnings import warn
        warn('setting the 0 value for the slow_decay_adapt reset port')
        setattr(unit, 'sda_port', 0)


def add_mp_weights(unit):
    """ Add a list with synaptic weights, as returned by unit.get_mp_weights.

        This implementation lives in the unit model, and has priority 1.
    """
    if not hasattr(unit,'port_idx'): 
        raise NameError( 'the mp_weights requirement is for multiport units ' +
                         'with a port_idx list' )
    mp_weights = [np.array([unit.net.syns[unit.ID][idx].w for idx in idx_list])
                  for idx_list in unit.port_idx]
    setattr(unit, 'mp_weights', mp_weights)


def add_sc_inp_sum_mp(unit):
    """ Add the scaled sum of inputs for each port.

        sc_inp_sum_mp will be a numpy array whose i-th element is the sum of
        inputs at the i-th port, each multiplied by its corresponding weight.

        This requirement depends on mp_inputs and mp_weights, but it is also
        used by other requirements, so it has an intermediate priority: 2.
        The update implementation is in the 'unit' class.
    """
    if not hasattr(unit,'port_idx'): 
        raise NameError( 'the sc_inp_sum_mp requirement is for multiport units ' +
                         'with a port_idx list' )
    if (not syn_reqs.mp_weights in unit.syn_needs or
        not syn_reqs.mp_inputs in unit.syn_needs):
        raise AssertionError('Requirement sc_inp_sum_mp depends on ' +
                             'the mp_weights and mp_inputs requirements.')
    mp_weights = [np.array([unit.net.syns[unit.ID][idx].w for idx in idx_list])
                  for idx_list in unit.port_idx]
    ia = unit.init_act
    mp_inputs = [np.array([ia]*len(prt_lst)) for prt_lst in unit.port_idx] 
    sc_inp_sum_mp = [(i*w).sum() for i,w in zip(mp_inputs, mp_weights)]
    setattr(unit, 'sc_inp_sum_mp', sc_inp_sum_mp)


def add_integ_decay_act(unit):
    """ Add an integral of the unit's activity, with slow decay.

        Let "ida" denote this factor. Then it is obtained as:
        ida' = min_delay*(act - int_decay*ida)
        The implementation is in the rga_reqs class of spinal_units.
    """
    if not hasattr(unit, 'integ_decay'):
        raise NameError('The integ_decay_act requirement requires the ' +
                        'int_decay parameter')
    setattr(unit, 'integ_decay_act', 0.)


def add_idel_ip_ip_mp(unit):
    """ Add inner product of delayed inputs and their derivatives by port.

        Basically, it takes the (non-delayed) ihput derivatives from the
        inp_deriv_mp requirement, and for each port included, it obtains the
        inner product with the corresponding entries of del_inp_mp.

        Because del_inp_mp specifies its ports with the 'del_inp_ports' list,
        and inp_deriv_mp does it through the 'inp_deriv_ports' list, these two
        must be equal if add_idel_ip_ip_mp is to work.

        The implementation for idel_ip_ip_mp lives in the rga_reqs class.
    """
    if (not syn_reqs.inp_deriv_mp in unit.syn_needs or
        not syn_reqs.del_inp_mp in unit.syn_needs):
        raise AssertionError('The add_idel_ip_ip_mp requirement needs the ' + 
                             'inp_deriv_mp and del_inp_mp requirements.')
    """ This test fails when idel_ip_ip_mp is initialized in-between
        del_inp_ports and inp_deriv_ports.
    if hasattr(unit, 'del_inp_ports') or hasattr(unit, 'inp_deriv_ports'):
        if not(hasattr(unit, 'del_inp_ports') and 
               hasattr(unit, 'inp_deriv_ports')):
            raise AssertionError('If one of del_inp_ports or inp_deriv_ports' +
                                 ' is present, the other must be too.')
        else:
            if unit.del_inp_ports != unit.inp_deriv_ports:
                raise AssertionError('The del_inp_ports and inp_deriv_ports ' +
                      'lists must be equal to use idel_ip_ip_mp.')
            else:
                idel_ip_ip_mp = [0.1 for p in unit.del_inp_ports]
    else:
        idel_ip_ip_mp = [0.1] * unit.n_ports
    """
    idel_ip_ip_mp = [0.1] * unit.n_ports
    setattr(unit, 'idel_ip_ip_mp', idel_ip_ip_mp)

def add_dni_ip_ip_mp(unit):
    """ Add dot product of delayed-normalized inputs and their derivs by port.

        Basically, it takes the delayed inputs from del_inp_mp, substracts the
        means from del_inp_avg_mp, and takes the dot product with the
        (non-delayed) ihput derivatives from the inp_deriv_mp requirement for 
        each port.

        Because del_inp_mp specifies its ports with the 'del_inp_ports' list,
        and inp_deriv_mp does it through the 'inp_deriv_ports' list, these two
        must be equal if add_dni_ip_ip_mp is to work.

        The implementation for dni_ip_ip_mp lives in the rga_reqs class.
    """
    if (not syn_reqs.inp_deriv_mp in unit.syn_needs or
        not syn_reqs.del_inp_avg_mp in unit.syn_needs):
        raise AssertionError('The add_dni_ip_ip_mp requirement needs the ' + 
                             'inp_deriv_mp and del_inp_avg_mp requirements.')
    dni_ip_ip_mp = [0.1] * unit.n_ports
    setattr(unit, 'dni_ip_ip_mp', dni_ip_ip_mp)


def add_i_ip_ip_mp(unit):
    """ Add inner product of inputs and their derivs by port.

        For each entry (a list) in  mp_inputs, i_ip_ip_mp will have the inner
        product of that list with the corresponding list in inp_deriv_mp.

        Since inp_deriv_mp uses the 'inp_deriv_ports' list, when this list is
        present i_ip_ip_mp will only contain inner products for the listed
        ports.

        This requirement was implemented for the meca_hebb synapse. Its
        implementation lives in the rga_reqs class.
    """
    if (not syn_reqs.inp_deriv_mp in unit.syn_needs or
        not syn_reqs.mp_inputs in unit.syn_needs):
        raise AssertionError('The add_i_ip_ip_mp requirement needs the ' + 
                             'inp_deriv_mp and mp_inputs requirements.')
    i_ip_ip_mp = [0.1] * unit.n_ports
    setattr(unit, 'i_ip_ip_mp', i_ip_ip_mp)



#-------------------------------------------------------------------------------------
# Use of the following classes has been deprecated because they slow down execution
#-------------------------------------------------------------------------------------

class requirement():
    """ The parent class of requirement classes.  """
    
    def __init__(self, unit):
        """ The class constructor.

            Args:
                unio : a reference to the unit where the requirement is used.

        """
        self.val = None

    def update(self, time):
        pass

    def get(self):
        return self.val


class lpf_fast(requirement):
    """ Maintains a low-pass filtered version of the unit's activity.

        The name lpf_fast indicates that the time constant of the low-pass filter,
        whose name is 'tau_fast', should be relatively fast. In practice this is
        arbitrary.
        The user needs to set the value of 'tau_fast' in the parameter dictionary 
        that initializes the unit.

        An instance of this class is meant to be created by init_pre_syn_update 
        whenever the unit has the 'lpf_fast' requiremnt. In this case, the update
        method of this class will be included in the unit's 'functions' list, and 
        called at each simulation step by pre_syn_update.

        Additionally, when the unit has the 'lpf_fast' requirement, the init_buff
        method will be invoked by the unit's init_buffers method.
    """
    def __init__(self, unit):
        """ The class' constructor.
            Args:
                unit : the unit containing the requirement.
        """
        if not hasattr(unit,'tau_fast'): 
            raise NameError( 'Synaptic plasticity requires unit parameter tau_fast, not yet set' )
        self.val = unit.init_act
        self.unit = unit
        self.init_buff()
 
    def update(self, time):
        """ Update the lpf_fast variable. """
        #assert time >= self.unit.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_fast updated backwards in time']
        cur_act = self.unit.get_act(time)
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.val = cur_act + ( (self.val - cur_act) * 
                                   np.exp( (self.unit.last_time-time)/self.unit.tau_fast ) )
        # update the buffer
        self.lpf_fast_buff = np.roll(self.lpf_fast_buff, -1)
        self.lpf_fast_buff[-1] = self.val

    def init_buff(self):
        """ Initialize the buffer with past values of lpf_fast. """
        self.lpf_fast_buff = np.array( [self.unit.init_act]*self.unit.steps, dtype=self.unit.bf_type)

    def get(self, steps):
        """ Get the fast low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_fast_buff[-1-steps]


class lpf_mid(requirement):
    """ Maintains a low-pass filtered version of the unit's activity.

        The name lpf_mid indicates that the time constant of the low-pass filter,
        whose name is 'tau_mid', should have an intermediate value. In practice 
        this is arbitrary.
        The user needs to set the value of 'tau_mid' in the parameter dictionary 
        that initializes the unit.

        An instance of this class is meant to be created by init_pre_syn_update 
        whenever the unit has the 'lpf_mid' requiremnt. In this case, the update
        method of this class will be included in the unit's 'functions' list, and 
        called at each simulation step by pre_syn_update.

        Additionally, when the unit has the 'lpf_mid' requirement, the init_buff
        method will be invoked by the unit's init_buffers method.
    """
    def __init__(self, unit):
        """ The class' constructor.
            Args:
                unit : the unit containing the requirement.
        """
        if not hasattr(unit,'tau_mid'): 
            raise NameError( 'Synaptic plasticity requires unit parameter tau_mid, not yet set' )
        self.val = unit.init_act
        self.unit = unit
        self.init_buff()
 
    def update(self, time):
        """ Update the lpf_fast variable. """
        #assert time >= self.unit.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_fast updated backwards in time']
        cur_act = self.unit.get_act(time)
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.val = cur_act + ( (self.val - cur_act) * 
                                   np.exp( (self.unit.last_time-time)/self.unit.tau_mid ) )
        # update the buffer
        self.lpf_mid_buff = np.roll(self.lpf_mid_buff, -1)
        self.lpf_mid_buff[-1] = self.val

    def init_buff(self):
        """ Initialize the buffer with past values of lpf_fast. """
        self.lpf_mid_buff = np.array( [self.unit.init_act]*self.unit.steps, dtype=self.unit.bf_type)

    def get(self, steps):
        """ Get the fast low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_mid_buff[-1-steps]


class lpf_slow(requirement):
    """ Maintains a low-pass filtered version of the unit's activity.

        The name lpf_slow indicates that the time constant of the low-pass filter,
        whose name is 'tau_slow', should have a relatively large value. In practice 
        this is arbitrary.
        The user needs to set the value of 'tau_slow' in the parameter dictionary 
        that initializes the unit.

        An instance of this class is meant to be created by init_pre_syn_update 
        whenever the unit has the 'lpf_slow' requiremnt. In this case, the update
        method of this class will be included in the unit's 'functions' list, and 
        called at each simulation step by pre_syn_update.

        Additionally, when the unit has the 'lpf_slow' requirement, the init_buff
        method will be invoked by the unit's init_buffers method.
    """
    def __init__(self, unit):
        """ The class' constructor.
            Args:
                unit : the unit containing the requirement.
        """
        if not hasattr(unit,'tau_slow'): 
            raise NameError( 'Synaptic plasticity requires unit parameter tau_slow, not yet set' )
        self.val = unit.init_act
        self.unit = unit
        self.init_buff()
 
    def update(self, time):
        """ Update the lpf_fast variable. """
        #assert time >= self.unit.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_fast updated backwards in time']
        cur_act = self.unit.get_act(time)
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.val = cur_act + ( (self.val - cur_act) * 
                                   np.exp( (self.unit.last_time-time)/self.unit.tau_slow ) )
        # update the buffer
        self.lpf_slow_buff = np.roll(self.lpf_slow_buff, -1)
        self.lpf_slow_buff[-1] = self.val

    def init_buff(self):
        """ Initialize the buffer with past values of lpf_fast. """
        self.lpf_slow_buff = np.array( [self.unit.init_act]*self.unit.steps, dtype=self.unit.bf_type)

    def get(self, steps):
        """ Get the fast low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_slow_buff[-1-steps]


class lpf(requirement):
    """ A low pass filter with a given time constant. """
    def __init__(self, unit):
        self.tau = unit.lpf_tau
        self.val = unit.init_act
        self.unit = unit
        self.init_buff()
 
    def update(self, time):
        """ Update the lpf_fast variable. """
        #assert time >= self.unit.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_fast updated backwards in time']
        cur_act = self.unit.get_act(time)
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.val = cur_act + ( (self.val - cur_act) * 
                                   np.exp( (self.unit.last_time-time)/self.tau ) )
        # update the buffer
        self.buff = np.roll(self.buff, -1)
        self.buff[-1] = self.val

    def init_buff(self):
        """ Initialize the buffer with past values. """
        self.buff = np.array( [self.unit.init_act]*self.unit.steps, dtype=self.unit.bf_type)

    def get(self, steps):
        """ Get the fast low-pass filtered activity, as it was 'steps' simulation steps before. """
        return self.lpf_fast_buff[-1-steps]


class sq_lpf_slow(requirement):
    """ A low pass filtered version of the squared activity.
        
        As the name implies, the filter uses the "slow" time constant 'tau_slow'.
        The purpose of this is to have a mean value of the square of the activity,
        as used in some versions of the BCM learning rule. Accordingly, this 
        requirement is used byt the bcm_synapse class.
    """
    def __init__(self, unit):
        if not hasattr(unit,'tau_slow'): 
            raise NameError( 'sq_lpf_slow requires unit parameter tau_slow, not yet set' )
        self.tau = unit.tau_slow
        self.val = unit.init_act
        self.unit = unit

    def update(self,time):
        """ Update the sq_lpf_slow variable. """
        #assert time >= self.unit.last_time, ['Unit ' + str(self.unit.ID) + 
        #                                ' sq_lpf_slow updated backwards in time']
        cur_sq_act = self.unit.get_act(time)**2.  
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.val = cur_sq_act + ( (self.val - cur_sq_act) * 
                                  np.exp( (self.unit.last_time-time)/self.tau ) )

class inp_vector(requirement):
    """ A numpy array with the unit's inputs at the start of the current simulation step.

        The inputs are not multiplied by synaptic weights, and come with their 
        appropriate delays.

        This input vector is used by other synaptic requirements, namely lpf_mid_inp_sum,
        balance, and exp_scale. In this last requirement, it is used to obtain the 'mu'
        factor used by the rule.
    """
    def __init__(self, unit):
        self.unit = unit
        self.val = np.tile(unit.init_act, len(unit.net.syns[unit.ID]))
        self.uid = unit.ID

    def update(self, time):
        self.val = np.array([ fun(time - dely) for dely,fun in 
                   zip(self.unit.net.delays[self.uid], self.unit.net.act[self.uid]) ])


class mp_inputs(requirement):
    """ Maintains a list with all the inputs, in the format of get_mp_inputs method. 

        In fact, the list is updated using the get_mp_inputs method. Some requirements
        like mp_balance and lpf_slow_mp_inp_sum use the output of get_mp_inputs, and
        the mp_inputs requirement saves computations by ensuring that get_mp_inputs
        only gets called once per simulation step.

        Repeating the docsting of unit.get_mp_inputs:
        This method is for units where multiport = True, and that have a port_idx attribute.
        The i-th element of the returned list is a numpy array containing the raw (not multiplied
        by the synaptic weight) inputs at port i. The inputs include transmision delays.

        WARNING: Currently this requirement has the disadvantage that it can't
        be ensured that mp_inputs will be udpated before the other requirements
        that use mp_inputs in their computations.
    """
    def __init__(self, unit):
        if not hasattr(unit,'port_idx'): 
            raise NameError( 'the mp_inputs requirement is for multiport units with a port_idx list' )
        self.val = [] 
        for prt_lst in unit.port_idx:
            self.val.append(np.array([unit.init_act for _ in range(len(prt_lst))]))
        self.unit = unit

    def update(self, time):
        self.val = self.unit.get_mp_inputs(time)
        

class inp_avg_hsn(requirement):
    """ Provides an average of the inputs arriving at hebbsnorm synapses.

        More precisely, the sum of fast-LPF'd hebbsnorm inputs divided by the the number
        of hebbsnorm inputs.

        Since we are using the lpf_fast signals, all the presynaptic units with hebbsnorm
        synapses need to have the lpf_fast requirement. 
    """
    def __init__(self, unit):  
        self.snorm_list = []  # a list with all the presynaptic units
                              # providing hebbsnorm synapses
        self.snorm_dels = []  # a list with the delay steps for each connection from snorm_list
        for syn in unit.net.syns[unit.ID]:
            if syn.type is synapse_types.hebbsnorm:
                if not syn_reqs.lpf_fast in unit.net.units[syn.preID].syn_needs:
                    raise AssertionError('inp_avg_hsn needs lpf_fast on presynaptic units')
                self.snorm_list.append(unit.net.units[syn.preID])
                self.snorm_dels.append(syn.delay_steps)
        self.n_hebbsnorm = len(self.snorm_list) # number of hebbsnorm synapses received
        self.val = 0.2  # an arbitrary initialization of the average input value
        self.snorm_list_dels = list(zip(self.snorm_list, self.snorm_dels)) # both lists zipped

    def update(self, time):
        self.val = sum([u.get_lpf_fast(s) for u,s in self.snorm_list_dels]) / self.n_hebbsnorm


class pos_inp_avg_hsn(requirement):
    """ Provides an average of the inputs arriving at hebbsnorm synapses with positive weights.

        More precisely, the sum of fast-LPF'd hebbsnorm inputs divided by the the number
        of hebbsnorm inputs, but only considering inputs with positive synapses.

        Since we are using the lpf_fast signals, all the presynaptic units with hebbsnorm
        synapses need to have the lpf_fast requirement. 
    """
    def __init__(self, unit):  
        self.snorm_units = []  # a list with all the presynaptic units
                               # providing hebbsnorm synapses
        self.snorm_syns = []  # a list with the synapses for the list above
        self.snorm_delys = []  # a list with the delay steps for these synapses
        for syn in unit.net.syns[unit.ID]:
            if syn.type is synapse_types.hebbsnorm:
                if not syn_reqs.lpf_fast in unit.net.units[syn.preID].syn_needs:
                    raise AssertionError('pos_inp_avg_hsn needs lpf_fast on presynaptic units')
                self.snorm_syns.append(syn)
                self.snorm_units.append(unit.net.units[syn.preID])
                self.snorm_delys.append(syn.delay_steps)
        self.val = 0.2  # an arbitrary initialization of the average input value
        self.n_vec = np.ones(len(self.snorm_units)) # the 'n' vector from Pg.290 of Dayan&Abbott

    def update(self, time):
        # first, update the n vector from Eq. 8.14, pg. 290 in Dayan & Abbott
        self.n_vec = [ 1. if syn.w>0. else 0. for syn in self.snorm_syns ]
        
        self.val = sum([n*(u.get_lpf_fast(s)) for n,u,s in 
                                zip(self.n_vec, self.snorm_units, self.snorm_delys)]) / sum(self.n_vec)


