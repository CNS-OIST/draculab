"""
spinal_units.py
The units used in the motor control model.
"""
from draculab import unit_types, synapse_types, syn_reqs  # names of models and requirements
from units.units import unit, sigmoidal
import numpy as np


#00000000000000000000000000000000000000000000000000000000000000000000000
#000000000 CLASSES INCORPORATING REQUIREMENT UPDATE FUNCTIONS 0000000000
#00000000000000000000000000000000000000000000000000000000000000000000000

class rga_reqs():
    """ A class with the update functions for requirements in rga synapses. """
    def __init__(self, params):
        """ This constructor adds required syn_needs values. 

            The constructor receives the parameters dictionary of the unit's
            creator, but only considers one entry: 
            'inp_deriv_ports' : A list with the numbers of the ports where
                                inp_deriv_mp and del_inp_deriv_mp will 
                                calculate their derivatives.
                                Defaults to a list with all ports.
        """
        self.syn_needs.update([syn_reqs.lpf_slow_sc_inp_sum])
        if 'inp_deriv_ports' in params:
            self.inp_deriv_ports = params['inp_deriv_ports']
        # inp_deriv_ports works by indicating the add_ methods to restrict the
        # entries in pre_list_del_mp and pre_list_mp.

    def upd_inp_deriv_mp(self, time):
        """ Update the list with input derivatives for each port.  """
        u = self.net.units
        self.inp_deriv_mp = [[u[uid[idx]].get_lpf_fast(dely[idx]) -
                              u[uid[idx]].get_lpf_mid(dely[idx]) 
                              for idx in range(len(uid))]
                              for uid, dely in self.pre_list_del_mp]

    def upd_avg_inp_deriv_mp(self, time):
        """ Update the list with the average of input derivatives for each port. """
        self.avg_inp_deriv_mp = [np.mean(l) if len(l) > 0 else 0.
                                 for l in self.inp_deriv_mp]

    def upd_del_inp_deriv_mp(self, time):
        """ Update the list with custom delayed input derivatives for each port. """
        u = self.net.units
        self.del_inp_deriv_mp = [[u[uid].get_lpf_fast(self.custom_inp_del) - 
                                  u[uid].get_lpf_mid(self.custom_inp_del) 
                                  for uid in lst] for lst in self.pre_list_mp]
 
    def upd_del_avg_inp_deriv_mp(self, time):
        """ Update the list with delayed averages of input derivatives for each port. """
        self.del_avg_inp_deriv_mp = [np.mean(l) if len(l) > 0 else 0.
                                     for l in self.del_inp_deriv_mp]

    def upd_integ_decay_act(self, time):
        """ Update the slow-decaying integral of the activity. """
        self.integ_decay_act += self.min_delay * (self.act_buff[-1] - 
                                self.integ_decay * self.integ_decay_act)


    def upd_double_del_inp_deriv_mp(self, time):
        """ Update two input derivatives with two delays for each port. """
        u = self.net.units
        self.double_del_inp_deriv_mp[0] = [[u[uid].get_lpf_fast(self.custom_inp_del) - 
                                u[uid].get_lpf_mid(self.custom_inp_del) 
                                for uid in lst] for lst in self.pre_list_mp]
        self.double_del_inp_deriv_mp[1] = [[u[uid].get_lpf_fast(self.custom_inp_del2) - 
                        u[uid].get_lpf_mid(self.custom_inp_del2) 
                        for uid in lst] for lst in self.pre_list_mp]
 
    def upd_double_del_avg_inp_deriv_mp(self, time):
        """ Update averages of input derivatives with two delays for each port. """
        self.del_avg_inp_deriv_mp[0] = [np.mean(l) if len(l) > 0 else 0.
                                    for l in self.double_del_inp_deriv_mp[0]]
        self.del_avg_inp_deriv_mp[1] = [np.mean(l) if len(l) > 0 else 0.
                                    for l in self.double_del_inp_deriv_mp[1]]


class lpf_sc_inp_sum_mp_reqs():
    """ Class with the update functions for the X_lpf_sc_inp_sum_mp_reqs. """
    def __init__(self, params):
        """ Class constructor. 

            Receives the params dictionray of the unit's creator.
        """
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.mp_weights,
                              syn_reqs.sc_inp_sum_mp])

    def upd_lpf_fast_sc_inp_sum_mp(self, time):
        """ Update a list with fast LPF'd scaled sums of inputs at each port. """
        # updated to use mp_inputs, mp_weights
        #sums = [(w*i).sum() for i,w in zip(self.mp_inputs, self.mp_weights)]
        # same update rule from other upd_lpf_X methods, put in a list comprehension
        self.lpf_fast_sc_inp_sum_mp = [self.sc_inp_sum_mp[i] + 
                                      (self.lpf_fast_sc_inp_sum_mp[i] - 
                                       self.sc_inp_sum_mp[i])
                                       * self.fast_prop for i in range(self.n_ports)]

    def upd_lpf_mid_sc_inp_sum_mp(self, time):
        """ Update a list with medium LPF'd scaled sums of inputs at each port. """
        # untouched copy from gated_out_norm_am_sig
        #sums = [(w*i).sum() for i,w in zip(self.mp_inputs, self.mp_weights)]
        # same update rule from other upd_lpf_X methods, put in a list comprehension
        self.lpf_mid_sc_inp_sum_mp = [self.sc_inp_sum_mp[i] + 
                                     (self.lpf_mid_sc_inp_sum_mp[i] - 
                                      self.sc_inp_sum_mp[i])
                                      * self.mid_prop for i in range(self.n_ports)]

    def upd_lpf_slow_sc_inp_sum_mp(self, time):
        """ Update a list with slow LPF'd scaled sums of inputs at each port. """
        #sums = [(w*i).sum() for i,w in zip(self.mp_inputs, self.mp_weights)]
        #sums = [(w*i).sum() for i,w in zip(self.get_mp_inputs(time),
        #                                   self.get_mp_weights(time))]
        # same update rule from other upd_lpf_X methods, put in a list comprehension
        self.lpf_slow_sc_inp_sum_mp = [self.sc_inp_sum_mp[i] + 
                                      (self.lpf_slow_sc_inp_sum_mp[i] - 
                                       self.sc_inp_sum_mp[i])
                                       * self.slow_prop for i in range(self.n_ports)]

class acc_sda_reqs():
    """ The acc_(mid|slow) and slow_decay_adapt update functions. """
    def __init__(self, params):
        """ Class constructor. 

            Receives the params dictionray of the unit's creator.
            It considers these entries:
            'acc_slow_port' : port number of the acc_slow reset input.
            'acc_mid_port' : port number of the acc_mid reset input.
            'sda_port' : reset port for the slow_decay_adaptation.
        """
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.mp_weights,
                              syn_reqs.sc_inp_sum_mp])
        if 'acc_slow_port' in params:
            self.acc_slow_port = params['acc_slow_port']
        if 'acc_mid_port' in params:
            self.acc_mid_port = params['acc_mid_port']
        if 'sda_port' in params:
            self.sda_port = params['sda_port']

    def upd_acc_slow(self, time):
        """ Update the slow speed accumulator. """
        if self.sc_inp_sum_mp[self.acc_slow_port] > 0.5 and self.acc_slow > 0.5:
        #if ( (self.get_mp_inputs(time)[2]*self.get_mp_weights(time)[2]).sum()
        #     > 0.5 and self.acc_slow > 0.5 ):
             self.acc_slow = 0.
        else: 
            self.acc_slow = 1. - (1.-self.acc_slow)*self.slow_prop

    def upd_acc_mid(self, time):
        """ Update the medium speed accumulator. """
        if self.sc_inp_sum_mp[self.acc_mid_port] > 0.5 and self.acc_mid > 0.5:
        #if ( (self.get_mp_inputs(time)[2]*self.get_mp_weights(time)[2]).sum()
        #     > 0.5 and self.acc_mid > 0.5 ):
             self.acc_mid = 0.
        else: 
            self.acc_mid = 1. - (1.-self.acc_mid)*self.mid_prop

    def upd_slow_decay_adapt(self, time):
        """ Update method for the slow-decaying adaptation. """
        if (self.sc_inp_sum_mp[self.sda_port] > 0.8 and 
            self.slow_decay_adapt < 0.2):
        #if ((self.get_mp_inputs(time)[3]*self.get_mp_weights(time)[3]).sum()
        #    > 0.8 and self.slow_decay_adapt < 0.2):
            #self.slow_decay_adapt = self.lpf_slow
            self.slow_decay_adapt = self.lpf_slow * self.lpf_slow
            # to produce increas in low-activity units
            #self.slow_decay_adapt = self.lpf_slow - 0.3
        else:
            self.slow_decay_adapt *= self.slow_prop



#00000000000000000000000000000000000000000000000000000000000000000000000
#0000000000               UNIT IMPLEMENTATIONS                0000000000
#00000000000000000000000000000000000000000000000000000000000000000000000

class am_pm_oscillator(unit, rga_reqs):
    """
    An oscillator with amplitude and phase modulated by inputs.

    The outuput of the unit is the sum of a constant part and a sinusoidal part.
    Both of their amplitudes are modulated by the sum of the inputs at port 0.
    The phase of the sinusoidal part is modulated by the relative phase of the 
    inputs at port 1 through a method chosen by the user. 

    The model uses 4-dimensional dynamics, so it's state at a given time is a 
    4-element array. The first element corresponds to the unit's activity, which
    comes from the sum of the constant part and the oscillating part. The second
    element corresponds to the constant part of the output. The third element
    corresponds to the phase of the oscillating part. The fourth element is a
    variable that follows the scaled sum of the inputs at port 0; this is used in
    order to obtain the derivative of those inputs without the need of two low-pass
    filters and new requirements. 

    The equations of the model look like this:
    tau_u * u'   = c' + Is*th'*cos(th) + <Is>'*sin(th)
    tau_c * c'   = Is*(1-c) + Im*c
    tau_t * th'  = w + F(u_1, ..., u_n, U)
    tau_s * <Is>'= Is - <Is>

    where: 
        u = units's activity,
        c = constant part of the unit's activity,
        th = phase of the oscillation,
        <Is> = low-pass filtered Is,
        Is = scaled sum of inputs at port 0,
        Im = scaled sum of INHIBITORY inputs at port 1,
        w = intrinsic frequency of the oscillation times tau_t,
        F = interaction function,
        n = number of inputs at port 1,
        u_j = j-th input at port 1, 
        U = unit's output normalized by its medium LPF'd value.

    There are several choices for the interaction function:
    1) input_sum:
        F = alpha*sin(Ic(t) - U(t)), where Ic is the scaled sum of inputs (each
        multiplied by its synaptic weight), normalized by its slow LPF'd value.
    2) kuramoto:
        F = (1/max(n,1))*sum_j K_j*[f(u_j, u) - f(u, u_j)], where
            K_j = weight of j-th connection at port 1,
            f(a,b) = a*sqrt(1-b**2)
        This aims to reproduce the original Kuramoto model, despite the phases
        not being communicated, but deduced from the received activities.
        It may require normalization of individual inputs (e.g. removing their 
        temporal average and dividing by that average), so it is not implemented.
    3) zero
        F = 0
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS
                'tau_u' : Time constant for the unit's activity.
                'tau_c' : Time constant for non-oscillatory dynamics.
                'tau_t' : Time constant for oscillatory dynamics.
                'tau_s' : Time constant for the port 0 low-pass filter.
                          A small value is recommended, so the derivative
                          of the LPF'd signal is similar to the derivative
                          of the original signal.
                'omega' : intrinsic oscillation angular frequency times tau_t.
                'multidim' : the Boolean literal 'True'. This is used to indicate
                             net.create that the 'init_val' parameter may be a single
                             initial value even if it is a list.
                'F' : a string specifying which interaction function to use.
                      Options: 'zero', 'kuramoto', 'input_sum'.
                      The 'input_sum' option requires an extra parameter: 
                      'alpha' : strength of phase interactions
                If 'input_sum' is the interaction function we also require:
                'tau_slow' : time constant for the slow low-pass filter.
                Using rga synapses brings an extra required parameter:
                'custom_inp_del' : an integer indicating the delay that the rga
                                  learning rule uses for the lateral port inputs. 
                                  The delay is in units of min_delay steps. 
                OPTIONAL PARAMETERS
                'n_ports' : number of input ports. Must equal 2. Defaults to 2.
                'mu' : mean of white noise when using noisy integration
                'sigma' : standard deviation of noise when using noisy integration
        Raises:
            ValueError

        """
        params['multidim'] = True
        if len(params['init_val']) != 4:
            raise ValueError("Initial values for the am_pm_oscillator must " +
                             "consist of a 4-element array.")
        if 'n_ports' in params:
            if params['n_ports'] != 2:
                raise ValueError("am_pm_oscillator units require 2 input ports.")
        else:
            params['n_ports'] = 2
        unit.__init__(self, ID, params, network) # parent's constructor
        self.tau_u = params['tau_u']
        self.tau_c = params['tau_c']
        self.tau_t = params['tau_t']
        self.tau_s = params['tau_s']
        self.omega = params['omega']
        # you need these by default depending on the Dc update rule
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.mp_weights,
                               syn_reqs.lpf_slow_mp_inp_sum])
        if params['F'] == 'zero':
            self.f = lambda x : 0.
        elif params['F'] == 'input_sum':
            self.f = self.input_sum_f
            self.syn_needs.update([syn_reqs.lpf_slow, syn_reqs.mp_inputs, 
                                   syn_reqs.lpf_slow_mp_inp_sum])
            rga_reqs.__init__(self, params)
        elif params['F'] == 'kuramoto':
            self.f = self.kuramoto_f
        else:
            raise ValueError('Wrong specification of the interaction function F')
        if 'custom_inp_del' in params:
            self.custom_inp_del = params['custom_inp_del']
        if 'mu' in params:
            self.mu = params['mu']
        if 'sigma' in params:
            self.sigma = params['sigma']
        self.mudt = self.mu * self.time_bit # used by flat updaters
        self.mudt_vec = np.zeros(self.dim)
        self.mudt_vec[0] = self.mudt
        self.sqrdt = np.sqrt(self.time_bit) # used by flat updater
        self.needs_mp_inp_sum = True # dt_fun uses mp_inp_sum

    def derivatives(self, y, t):
        """ Implements the equations of the am_mp_oscillator.

        Args:
            y : list or Numpy array with the 4-element state vector:
              y[0] : u  -- unit's activity,
              y[1] : c  -- constant part of the input,
              y[2] : th -- phase of the oscillation,
              y[3] : I0 -- LPF'd sum of port 0 inputs.
            t : time when the derivative is evaluated.
        Returns:
            4-element numpy array with state variable derivatives.
        """
        # get the input sum at each port
        I = [ np.dot(i, w) for i, w in 
              zip(self.get_mp_inputs(t), self.get_mp_weights(t)) ]
        # Obtain the derivatives
        #Dc = y[1]*(1. - y[1])*(I[0] + I[1]) / self.tau_c
        #Dc = max(y[1],1e-3)*max(1. - y[1],1e-3)*(I[0] + I[1]) / self.tau_c
        # The Dc version below is when I[0] is always positive
        Dc = (I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        #slow_I0 = self.lpf_slow_mp_inp_sum[0]
        #Dc = ( I[0]*(1. - y[1]) + (I[1] - slow_I0)*y[1] ) / self.tau_c
        #Dc = ( I[0]*(1. - y[1]) - slow_I0*y[1] ) / self.tau_c
        Dth = (self.omega + self.f(I)) / self.tau_t
        DI0 = np.tanh(I[0] - y[3]) / self.tau_s
        #DI0 = (sum(self.get_mp_inputs(t)[0]) - y[3]) / self.tau_s
        #DIs = self.D_factor * (I[0] - y[3])
        #Du = (Dc + I[0]*Dth*np.cos(y[2]) + DI0*np.sin(y[2])) / self.tau_u
        #ex = np.exp(-y[3]*np.sin(y[2]))
        #prime = 0.2*ex/(1.+ex)
        Du = ((1.-y[0]) * (y[0]-.01) * (Dc + (y[1] - y[0]) + 
               #prime*(y[3]*Dth*np.cos(y[2]) + DI0*np.sin(y[2])))) / self.tau_u
               y[3]*Dth*np.cos(y[2]) + DI0*np.sin(y[2]))) / self.tau_u
        return np.array([Du, Dc, Dth, DI0])

    def H(self, t):
        """ The Heaviside step function. """
        return 1. if t > 0. else 0.

    def dt_fun(self, y, s):
        """ the derivatives function when the network is flat.
:
            y : list or Numpy array with the 4-element state vector:
              y[0] : u  -- unit's activity,
              y[1] : c  -- constant part of the input,
              y[2] : th -- phase of the oscillation,
              y[3] : I0 -- LPF'd sum of port 0 inputs.
            s : index to inp_sum for current time point
        Returns:
            4-element numpy array with state variable derivatives.
        """
        t = self.times[s - self.min_buff_size]
        # get the input sum at each port
        I = [ port_sum[s] for port_sum in self.mp_inp_sum ]
        # Obtain the derivatives
        #Dc = max(y[1],1e-3)*max(1. - y[1],1e-3)*(I[0] + I[1]) / self.tau_c
        # The Dc version below is when I[0] is always positive
        #Dc = (I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        Dc = 0.01*(I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        Dth = (self.omega + self.f(I)) / self.tau_t
        DI0 = np.tanh(I[0] - y[3]) / self.tau_s
        Dc += DI0 #/self.tau_c
        #Dc = self.H(self.net.sim_time-600.)*y[1]*(1. - y[1])*(I[0] + I[1]) / self.tau_c
        Du = ((1.-y[0]) * (y[0]-.01) * (Dc + (y[1] - y[0]) + 
               y[3]*Dth*np.cos(y[2]) + DI0*np.sin(y[2]))) / self.tau_u
        return np.array([Du, Dc, Dth, DI0])

    def input_sum_f(self, I):
        """ Interaction function based on port 1 input sum. """
        # TODO: I changed the sign here, should revert later
        #       Also, the factors here are not normalized.
        return -np.sin(2.*np.pi*(self.lpf_slow_mp_inp_sum[1] - self.lpf_slow))

    def kuramoto_f(self, I):
        """ Interaction function inspired by the Kuramoto model. """
        raise NotImplementedError('kuramoto_f not yet implemented')

    """
    def upd_lpf_slow_mp_inp_sum(self, time):
        #Update the list with slow LPF'd scaled sums of inputs at individual ports.
        # Don't remember why I made a separate implementation here, so I commented
        # this out
        
        #assert time >= self.last_time, ['Unit ' + str(self.ID) + 
        #                                ' lpf_slow_mp_inp_sum updated backwards in time']
        inputs = self.mp_inputs    # updated because mp_inputs is a requirement variable
        weights = self.get_mp_weights(time)
        dots = [ np.dot(i, w) for i, w in zip(inputs, weights) ]
        # same update rule from other upd_lpf_X methods above, put in a list comprehension
        self.lpf_slow_mp_inp_sum = [dots[i] + (self.lpf_slow_mp_inp_sum[i] - dots[i]) * 
                                   np.exp( (self.last_time-time)/self.tau_slow ) 
                                   for i in range(self.n_ports)]
    """
    
    def upd_lpf_mid_mp_raw_inp_sum(self, time):
        """ Update the list with medium LPF'd sums of inputs at individual ports. """
        sums = [sum(inps) for inps in self.mp_inputs]
        self.lpf_mid_mp_raw_inp_sum = [sums[i] + (self.lpf_mid_mp_raw_inp_sum[i] - sums[i])
                                       * self.mid_prop for i in range(self.n_ports)]


class out_norm_sig(sigmoidal):
    """ The sigmoidal unit with one extra attribute.

        The attribute is called des_out_w_abs_sum. This is needed by the 
        out_norm_factor requirement. 
        
        Units from this class are sigmoidals that can normalize the sum
        of their outgoing weights using out_norm_factor. This requirement should 
        be added by synapses that have the pre_out_norm_factor requirement.
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

            Args:
                ID, params, network: same as in the 'unit' class.
                In addition, params should have the following entries.
                    REQUIRED PARAMETERS
                    'slope' : Slope of the sigmoidal function.
                    'thresh' : Threshold of the sigmoidal function.
                    'tau' : Time constant of the update dynamics.
                    'des_out_w_abs_sum' : desired sum of absolute weight values
                                          for the outgoing connections.
            Raises:
                AssertionError.
        """
        sigmoidal.__init__(self, ID, params, network)
        self.des_out_w_abs_sum = params['des_out_w_abs_sum']
        #self.syn_needs.update([syn_reqs.out_norm_factor])


class out_norm_am_sig(sigmoidal, lpf_sc_inp_sum_mp_reqs):
    """ A sigmoidal whose amplitude is modulated by inputs at port 1.

        The output of the unit is the sigmoidal function applied to the
        scaled sum of port 0 inputs, times the scaled sum of port 1 inputs.

        This model also includes the 'des_out_w_abs_sum' attribute, so the
        synapses that have this as their presynaptic unit can have the
        'out_norm_factor' requirement for presynaptic weight normalization.
    """
    def __init__(self, ID, params, network):
        """ The unit constructor.

            Args:
                ID, params, network: same as in the 'unit' class.
                In addition, params should have the following entries.
                    REQUIRED PARAMETERS
                    'slope' : Slope of the sigmoidal function.
                    'thresh' : Threshold of the sigmoidal function.
                    'tau' : Time constant of the update dynamics.
                    'des_out_w_abs_sum' : desired sum of absolute weight values
                                          for the outgoing connections.
            Raises:
                AssertionError.
        """
        if 'n_ports' in params and params['n_ports'] != 2:
            raise AssertionError('out_norm_am_sig units must have n_ports=2')
        else:
            params['n_ports'] = 2
        sigmoidal.__init__(self, ID, params, network)
        self.des_out_w_abs_sum = params['des_out_w_abs_sum']
        self.needs_mp_inp_sum = True # in case we flatten
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.mp_weights,
                               syn_reqs.sc_inp_sum_mp])
        lpf_sc_inp_sum_mp_reqs.__init__(self, params)
      
    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. """
        I = [(w*i).sum() for w, i in zip(self.get_mp_weights(t), 
                                         self.get_mp_inputs(t))]
        return ( I[1]*self.f(I[0]) - y[0] ) * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        return ( self.mp_inp_sum[1][s]*self.f(self.mp_inp_sum[0][s]) - y ) * self.rtau


class gated_out_norm_am_sig(sigmoidal, lpf_sc_inp_sum_mp_reqs, acc_sda_reqs):
    """ A sigmoidal with modulated amplitude and plasticity.

        The output of the unit is the sigmoidal function applied to the
        scaled sum of port 0 inputs, times the scaled sum of port 1 inputs.
        Inputs at port 2 reset the acc_slow variable, which is used by the 
        gated_corr_inh synapses in order to modulate plasticity.

        Optionally, the scaled sum of port 0 inputs can also appear in the
        argument to the sigmoidal function. This is controlled by the p0_inp
        parameter.

        This model also includes the 'des_out_w_abs_sum' attribute, so the
        synapses that have this as their presynaptic unit can have the
        'out_norm_factor' requirement for presynaptic weight normalization.
    """
    def __init__(self, ID, params, network):
        """ The unit constructor.

            Args:
                ID, params, network: same as in the 'unit' class.
                In addition, params should have the following entries.
                    REQUIRED PARAMETERS
                    'slope' : Slope of the sigmoidal function.
                    'thresh' : Threshold of the sigmoidal function.
                    'tau' : Time constant of the update dynamics.
                    'des_out_w_abs_sum' : desired sum of absolute weight values
                                          for the outgoing connections.
                    'tau_slow' : slow LPF time constant.
                    OPTIONAL PARAMETERS
                    'p0_inp' : The scaled sum of port 0 inputs is multiplied by
                               this parameter before becoming being added to the
                               arguments of the sigmoidal. Default 0.
            Raises:
                AssertionError.
        """
        if 'n_ports' in params and params['n_ports'] != 3:
            raise AssertionError('gated_out_norm_am_sig units must have n_ports=3')
        else:
            params['n_ports'] = 3
        sigmoidal.__init__(self, ID, params, network)
        self.des_out_w_abs_sum = params['des_out_w_abs_sum']
        self.syn_needs.update([syn_reqs.acc_slow, syn_reqs.mp_inputs, 
                               syn_reqs.mp_weights])
        self.needs_mp_inp_sum = True # in case we flatten
        lpf_sc_inp_sum_mp_reqs.__init__(self, params)
        params['acc_slow_port'] = 2 # so inputs at port 2 reset acc_slow
        acc_sda_reqs.__init__(self, params)
        if 'p0_inp' in params:
            self.p0_inp = params['p0_inp']
        else:
            self.p0_inp = 0.
      
    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. """
        I = [(w*i).sum() for w, i in zip(self.get_mp_weights(t), 
                                         self.get_mp_inputs(t))]
        #return ( I[1]*self.f(I[0]) - y[0] ) * self.rtau
        return ( I[1]*self.f(I[0] + self.p0_inp*I[1]) - y[0] ) * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        #return ( self.mp_inp_sum[1][s] * self.f(self.mp_inp_sum[0][s]) - y ) * self.rtau
        return ( self.mp_inp_sum[1][s] * self.f(self.mp_inp_sum[0][s] +
                 self.p0_inp*self.mp_inp_sum[1][s]) - y ) * self.rtau


class logarithmic(unit):
    """ A unit with a logarithminc activation function. 
    
        The output is zero if the scaled sum of inputs is smaller than a given
        threshold 'thresh'. Otherwise, the activation approaches
        log(1 + (I - thresh)) with time constant 'tau'.
    """
    def __init__(self, ID, params, network):
        """ The unit constructor.

            Args:
                ID, params, network: same as in the 'unit' class.
                In addition, params should have the following entries.
                    REQUIRED PARAMETERS
                    'thresh' : Threshold of the activation function.
                    'tau' : Time constant of the update dynamics.
            Raises:
                AssertionError.
        """
        unit.__init__(self, ID, params, network)
        self.thresh = params['thresh']
        self.tau = params['tau']

    def derivatives(self, y, t):
        """ Return the derivative of the activity y at time t. """
        return (np.log( 1. + max(0., self.get_input_sum(t)-self.thresh)) 
                - y[0]) / self.tau

    def dt_fun(self, y, s):
        """ The derivatives function for flat networks. """
        return (np.log( 1. + max(0., self.inp_sum[s]-self.thresh))
                - y) / self.tau

    def upd_sc_inp_sum_diff_mp(self, time):
        """ Update the derivatives for the scaled sum of inputs at each port."""
        # TODO: there is replication of computations when the requirements
        # obtain the input sums
        self.sc_inp_sum_diff_mp = [lpf_fast - lpf_mid for lpf_fast, lpf_mid in
                  zip(self.lpf_fast_sc_inp_sum_mp, self.lpf_mid_sc_inp_sum_mp)]


class rga_sig(sigmoidal, rga_reqs):
    """ A sigmoidal unit that can receive rga synapses.

        This means it has the extra custom_inp_del attribute, as well as
        implementations of all required requirement update functions.
        Moreover, it is a multiport unit.

        Another addition is that this unit has an integral component. This means
        that the input is integrated (actually, low-pass filtered with the
        tau_slow time constant), and the output comes from the sigmoidal 
        function applied to the scaled input sum plus the integral component.

        The des_out_w_abs_sum parameter is included in case the
        pre_out_norm_factor requirement is used.
    """
    def __init__(self, ID, params, network):
        """ The unit constructor.

            Args:
                ID, params, network: same as in the 'unit' class.
                In addition, params should have the following entries.
                    REQUIRED PARAMETERS
                    'slope' : Slope of the sigmoidal function.
                    'thresh' : Threshold of the sigmoidal function.
                    'tau' : Time constant of the update dynamics.
                    'tau_slow' : Slow LPF time constant.
                    'integ_amp' : amplitude multiplier of the integral
                                    component.
                    Using rga synapses brings an extra required parameter:
                    'custom_inp_del' : an integer indicating the delay that the rga
                                  learning rule uses for the lateral port inputs. 
                                  The delay is in units of min_delay steps. 

                OPTIONAL PARAMETERS
                    'des_out_w_abs_sum' : desired sum of absolute weight values
                                          for the outgoing connections.
                                          Default is 1.
            Raises:
                AssertionError.
        """
        if 'n_ports' in params and params['n_ports'] != 2:
            raise AssertionError('rga_sig units must have n_ports=2')
        else:
            params['n_ports'] = 2
        if 'custom_inp_del' in params:
            self.custom_inp_del = params['custom_inp_del']
        else:
            raise AssertionError('rga_sig units need a custom_inp_del parameter')
        sigmoidal.__init__(self, ID, params, network)
        if 'des_out_w_abs_sum' in params:
            self.des_out_w_abs_sum = params['des_out_w_abs_sum']
        else:
            self.des_out_w_abs_sum = 1.
        self.integ_amp = params['integ_amp']
        rga_reqs.__init__(self, params) # add requirements and update functions 
        self.needs_mp_inp_sum = False # the sigmoidal uses self.inp_sum

    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. """
        inp = self.get_input_sum(t) + self.integ_amp * self.lpf_slow_sc_inp_sum
        return ( self.f(inp) - y[0] ) * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        inp = self.inp_sum[s] + self.integ_amp * self.lpf_slow_sc_inp_sum
        return ( self.f(inp) - y ) * self.rtau


class act(unit, lpf_sc_inp_sum_mp_reqs):
    """ Action clock unit from the cortex wiki.

        In the context of the spinal model, they are meant to receive input
        from the SPF layer at port 0, increasing their value from 0 to 1, going
        slower when F is close to P, and faster when the SPF signals are larger.
        The equations are in the "detecting layer distances" tiddler.

        When their value gets close enough to 1, their target unit will trigger
        an oscillatory/exploratory phase.

        When the scaled sum of inputs at port 1 is larger than a threshold,  the
        unit will reset its value to 0, and will remain at 0 until the port 1 
        inputs decrease below 0.5 .
        
        This unit has the lpf_slow_sc_inp_sum_mp requirement, which requires the
        tau_slow parameter.
    """
    def __init__(self, ID, params, network):
        """
        The unit constructor.

        Args:
            ID, params, network: same as in the 'unit' class.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS
                tau_u : time constant for the activation growth.
                gamma : multiplier to the "sigmoidal distance" Y.
                g : slope of the sigmoidal that produces Y.
                theta : threshold of the sigmoidal that produces Y.
                tau_slow : slow LPF time constant.
                y_min : value of Y below which increase of activity stops.
                OPTIONAL PARAMETERS
                rst_thr : threshold for port 1 input to cause a reset.
                          Default is 0.5 .
        """
        if 'n_ports' in params and params['n_ports'] != 2:
            raise AssertionError('act units must have n_ports=2')
        else:
            params['n_ports'] = 2
        unit.__init__(self, ID, params, network)
        self.syn_needs.update([syn_reqs.lpf_slow_sc_inp_sum_mp, 
                               syn_reqs.mp_inputs, syn_reqs.mp_weights])
        self.tau_u = params['tau_u']
        self.g = params['g']
        self.theta = params['theta']
        self.gamma = params['gamma']
        self.tau_slow = params['tau_slow']
        self.y_min = params['y_min']
        if "rst_thr" in params: self.rst_thr = params['rst_thr']
        else: self.rst_thr = 0.5
        self.needs_mp_inp_sum = True  # don't want port 1 inputs in the sum
        # the next line provides the lpf_slow_sc_inp_sum_mp requirement
        lpf_sc_inp_sum_mp_reqs.__init__(self, params)

    def derivatives(self, y, t):
        """ Return the derivative of the activation at the given time. """
        I1 = (self.get_mp_inputs(t)[1]*self.get_mp_weights(t)[1]).sum()
        if I1 < self.rst_thr:
            I0 = (self.get_mp_inputs(t)[0]*self.get_mp_weights(t)[0]).sum()
            I0_lpf = self.lpf_slow_sc_inp_sum_mp[0]
            Y = 1. / (1. + np.exp(-self.g*(I0 - self.theta)))
            Y_lpf = 1. / (1. + np.exp(-self.g*(I0_lpf - self.theta)))
            dY = Y - Y_lpf
            Y_dist = Y-self.y_min  #max(Y - self.y_min, 0.)
            if Y_dist < 0.:
                du = y[0]*Y_dist 
            else:
                du = Y_dist*(1. - y + self.gamma*dY) / self.tau_u
        else:
            du = -40.*y[0] # rushing towards 0
        return du

    def dt_fun(self, y, s):
        """ Return the derivative of the activation in a flat network. """
        I1 = self.mp_inp_sum[1][s]
        if I1 < self.rst_thr:
            I0 = self.mp_inp_sum[0][s]
            I0_lpf = self.lpf_slow_sc_inp_sum_mp[0]
            Y = 1. / (1. + np.exp(-self.g*(I0 - self.theta)))
            Y_lpf = 1. / (1. + np.exp(-self.g*(I0_lpf - self.theta)))
            dY = Y - Y_lpf
            Y_dist = Y-self.y_min # max(Y - self.y_min, 0.)
            if Y_dist < 0.:
                du = y*Y_dist 
            else:
                du = Y_dist*(1. - y + self.gamma*dY) / self.tau_u
        else:
            du = -40.*y # rushing towards 0
        return du


class gated_rga_sig(sigmoidal, rga_reqs, acc_sda_reqs):
    """ A sigmoidal unit that can receive gated_rga synapses.

        This means it has the extra custom_inp_del attribute, as well as
        implementations of all required requirement update functions.
        Moreover, it is a multiport unit, with 3 ports.

        Another addition is that this unit has an integral component. This means
        that the input is integrated (actually, low-pass filtered with the
        tau_slow time constant), and the output comes from the sigmoidal 
        function applied to the scaled input sum plus the integral component.

        The difference with the rga_sig model is that this class includes the
        acc_mid attribute, which is required by the gated_rga synapse class.
        Moreover, this model has 3 ports instead of 2. Ports 0 and 1 are for the
        "error" and "lateral" inputs, although this distinction is only done by
        the synapses. Port 2 is for the signal that resets the acc_mid
        accumulator. When the scaled sum of inputs at port 2 is larger than 0.5,
        acc_mid is set to 0.

        The des_out_w_abs_sum parameter is included in case the
        pre_out_norm_factor requirement is used.

        The presynaptic units should have the lpf_fast and lpf_mid requirements,
        since these will be added by the gated_rga synapse.
    """
    def __init__(self, ID, params, network):
        """ The unit constructor.

            Args:
                ID, params, network: same as in the 'unit' class.
                In addition, params should have the following entries.
                    REQUIRED PARAMETERS
                    'slope' : Slope of the sigmoidal function.
                    'thresh' : Threshold of the sigmoidal function.
                    'tau' : Time constant of the update dynamics.
                    'tau_slow' : Slow LPF time constant.
                    'tau_mid' : Medium LPF time constant.
                    'integ_amp' : amplitude multiplier of the integral
                                    component.
                    Using rga synapses brings an extra required parameter:
                    'custom_inp_del' : an integer indicating the delay that the rga
                                  learning rule uses for the lateral port inputs. 
                                  The delay is in units of min_delay steps. 

                OPTIONAL PARAMETERS
                    'des_out_w_abs_sum' : desired sum of absolute weight values
                                          for the outgoing connections.
                                          Default is 1.
            Raises:
                AssertionError.
        """
        if 'n_ports' in params and params['n_ports'] != 3:
            raise AssertionError('gated_rga_sig units must have n_ports=3')
        else:
            params['n_ports'] = 3
        if 'custom_inp_del' in params:
            self.custom_inp_del = params['custom_inp_del']
        else:
            raise AssertionError('gated_rga_sig units need a custom_inp_del parameter')
        sigmoidal.__init__(self, ID, params, network)
        params['inp_deriv_ports'] = [0, 1] # ports for inp_deriv_mp
        rga_reqs.__init__(self, params)
        if 'des_out_w_abs_sum' in params:
            self.des_out_w_abs_sum = params['des_out_w_abs_sum']
        else:
            self.des_out_w_abs_sum = 1.
        self.integ_amp = params['integ_amp']
        self.syn_needs.update([syn_reqs.lpf_slow_sc_inp_sum, syn_reqs.acc_slow])
        params['acc_slow_port'] = 2 # so inputs at port 2 reset acc_slow
        acc_sda_reqs.__init__(self, params)
        self.needs_mp_inp_sum = True # to avoid adding the reset signal

    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. """
        sums = [(w*i).sum() for i,w in zip(self.get_mp_inputs(t),
                                           self.get_mp_weights(t))]
        inp = sums[0] + sums[1] + self.integ_amp * self.lpf_slow_sc_inp_sum
        return ( self.f(inp) - y[0] ) * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        inp = self.mp_inp_sum[0][s] + self.mp_inp_sum[1][s] + \
              self.integ_amp * self.lpf_slow_sc_inp_sum
        return ( self.f(inp) - y ) * self.rtau


class gated_rga_adapt_sig(sigmoidal, rga_reqs, acc_sda_reqs):
    """ A sigmoidal unit that can use gated_rga synapses and gated adaptation.

        The difference with the gated_rga_sig model is that this model has 4
        ports instead of 3. When the inputs at port 3 surpass a threshold the
        unit experiences adaptation, which means that its slow-lpf'd activity
        will be used to decrease its current activation. This is achieved
        through the slow_decay_adapt requirement.

        Like the other rga models it has the extra custom_inp_del attribute, 
        as well as implementations of all required requirement update 
        functions.

        Like gated_rga_sig this unit has an integral component. This means
        that the input is integrated (actually, low-pass filtered with the
        tau_slow time constant), and the output comes from the sigmoidal 
        function applied to the scaled input sum plus the integral component.

        Ports 0 and 1 are for the "error" and "lateral" inputs, although this
        distinction is only done by the synapses. Port 2 is for the signal that
        resets the acc_slow accumulator. When the scaled sum of inputs at port 2
        is larger than 0.5, acc_slow is set to 0.

        The des_out_w_abs_sum parameter is included in case the
        pre_out_norm_factor requirement is used.

        The presynaptic units should have the lpf_fast and lpf_mid requirements,
        since these will be added by the gated_rga synapse.
    """
    def __init__(self, ID, params, network):
        """ The unit constructor.

            Args:
                ID, params, network: same as in the 'unit' class.
                In addition, params should have the following entries.
                    REQUIRED PARAMETERS
                    'slope' : Slope of the sigmoidal function.
                    'thresh' : Threshold of the sigmoidal function.
                    'tau' : Time constant of the update dynamics.
                    'tau_slow' : Slow LPF time constant.
                    'tau_mid' : Medium LPF time constant.
                    'integ_amp' : amplitude multiplier of the integral
                                    component.
                    Using rga synapses brings an extra required parameter:
                    'custom_inp_del' : an integer indicating the delay that the rga
                                  learning rule uses for the lateral port inputs. 
                                  The delay is in units of min_delay steps. 

                OPTIONAL PARAMETERS
                    'des_out_w_abs_sum' : desired sum of absolute weight values
                                          for the outgoing connections.
                                          Default is 1.
                    'adapt_amp' : amplitude of adapation. Default is 1.
            Raises:
                AssertionError.
        """
        if 'n_ports' in params and params['n_ports'] != 4:
            raise AssertionError('gated_rga_adapt_sig units must have n_ports=4')
        else:
            params['n_ports'] = 4
        if 'custom_inp_del' in params:
            self.custom_inp_del = params['custom_inp_del']
        else:
            raise AssertionError('gated_rga_adapt_sig units need a ' +
                                'custom_inp_del parameter')
        sigmoidal.__init__(self, ID, params, network)
        if 'des_out_w_abs_sum' in params:
            self.des_out_w_abs_sum = params['des_out_w_abs_sum']
        else:
            self.des_out_w_abs_sum = 1.
        if 'adapt_amp' in params:
            self.adapt_amp = params['adapt_amp']
        else:
            self.adapt_amp = 1.
        self.integ_amp = params['integ_amp']
        self.syn_needs.update([syn_reqs.lpf_slow_sc_inp_sum_mp, syn_reqs.acc_slow,
                               syn_reqs.mp_weights, syn_reqs.mp_inputs,
                               syn_reqs.lpf_slow, syn_reqs.slow_decay_adapt])
        params['inp_deriv_ports'] = [0, 1] # ports for inp_deriv_mp
        rga_reqs.__init__(self, params)
        params['acc_slow_port'] = 2 # so inputs at port 2 reset acc_slow
        params['sda_port'] = 3
        acc_sda_reqs.__init__(self, params)
        self.needs_mp_inp_sum = True # to avoid adding the reset signal

    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. """
        sums = [(w*i).sum() for i,w in zip(self.get_mp_inputs(t),
                                           self.get_mp_weights(t))]
        inp = (sums[0] + sums[1] 
               + self.integ_amp * self.lpf_slow_sc_inp_sum[0]
               - self.adapt_amp * self.slow_decay_adapt)
        return ( self.f(inp) - y[0] ) * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        inp = (self.mp_inp_sum[0][s] + self.mp_inp_sum[1][s]
               + self.integ_amp * self.lpf_slow_sc_inp_sum_mp[0]
               - self.adapt_amp * self.slow_decay_adapt)
        return ( self.f(inp) - y ) * self.rtau


class gated_rga_inpsel_adapt_sig(sigmoidal, rga_reqs, lpf_sc_inp_sum_mp_reqs,
                                 acc_sda_reqs):
    """ Sigmoidal with gated rga, inp_sel synapses and gated adaptation.

        Unlike gated_rga_adapt_sig this model has 5 ports instead of 4. 
        The scaled sum of inputs at port 0 constitutes the error signal for 
        both the gated_rga and gated_input_selection synapses. Inputs at port 1
        are the "lateral" inputs for the gated_rga synapses. Inputs at port 2
        are the "afferent" inputs for the gated_input_selection synapse. 
        The output of the unit is the sigmoidal function applied to the scaled
        sum of inputs at ports 0, 1, and 2, plus an integral component, minus
        an adaptation component.

        The scaled sum of inputs at port 3 is for the signal that resets the 
        acc_slow accumulator, used to gate both rga and input selection synapses.
        When the scaled sum of inputs at port 3 is larger than 0.5, acc_slow is
        set to 0.
         
        When the input at port 4 surpasses a threshold the unit experiences
        adaptation, which means that its slow-lpf'd activity will be used to 
        decrease its current activation. This is achieved through the 
        slow_decay_adapt requirement.

        Like the other rga models it has the extra custom_inp_del attribute, 
        as well as implementations of all required requirement update 
        functions.

        Like gated_rga_sig this unit has an integral component. This means
        that the input is integrated using the integ_decay_act requirement,
        and the output comes from the sigmoidal function applied to the scaled
        input sum plus a term proportional to the integral component.

        The des_out_w_abs_sum parameter is included in case the
        pre_out_norm_factor requirement is used.

        The presynaptic units should have the lpf_fast and lpf_mid requirements,
        since these will be added by the gated_rga synapse.

        This units has the required parameters to use euler_maru integration if
        this is specified in the parameters.
    """
    def __init__(self, ID, params, network):
        """ The unit constructor.

            Args:
                ID, params, network: same as in the 'unit' class.
                In addition, params should have the following entries.
                    REQUIRED PARAMETERS
                    'slope' : Slope of the sigmoidal function.
                    'thresh' : Threshold of the sigmoidal function.
                    'tau' : Time constant of the update dynamics.
                    'tau_slow' : Slow LPF time constant.
                    'tau_mid' : Medium LPF time constant.
                    'integ_amp' : amplitude multiplier of the integral
                                    component.
                    'integ_decay' : decay rate of the integral component.
                   Using rga synapses brings an extra required parameter:
                    'custom_inp_del' : an integer indicating the delay that the rga
                                  learning rule uses for the lateral port inputs. 
                                  The delay is in units of min_delay steps. 

                OPTIONAL PARAMETERS
                    'des_out_w_abs_sum' : desired sum of absolute weight values
                                          for the outgoing connections.
                                          Default is 1.
                    'adapt_amp' : amplitude of adapation. Default is 1.
                    'mu' : noise bias in for euler_maru integration.
                    'sigma' : standard deviation for euler_maru integration.
            Raises:
                AssertionError.
        """
        if 'n_ports' in params and params['n_ports'] != 5:
            raise AssertionError('gated_rga_inpsel_adapt_sig units must have n_ports=5')
        else:
            params['n_ports'] = 5
        if 'custom_inp_del' in params:
            self.custom_inp_del = params['custom_inp_del']
        else:
            raise AssertionError('gated_rga_inpsel_adapt_sig units need a ' +
                                 'custom_inp_del parameter')
        sigmoidal.__init__(self, ID, params, network)
        if 'des_out_w_abs_sum' in params:
            self.des_out_w_abs_sum = params['des_out_w_abs_sum']
        else:
            self.des_out_w_abs_sum = 1.
        if 'adapt_amp' in params:
            self.adapt_amp = params['adapt_amp']
        else:
            self.adapt_amp = 1.
        self.integ_amp = params['integ_amp']
        self.integ_decay = params['integ_decay']
        self.syn_needs.update([syn_reqs.integ_decay_act, syn_reqs.acc_slow,
                               syn_reqs.lpf_slow, syn_reqs.slow_decay_adapt,
                               syn_reqs.mp_inputs, syn_reqs.mp_weights,
                               syn_reqs.sc_inp_sum_mp])
                               # syn_reqs.lpf_slow_mp_inp_sum
        params['inp_deriv_ports'] = [0, 1] # ports for inp_deriv_mp
        rga_reqs.__init__(self, params)
        lpf_sc_inp_sum_mp_reqs.__init__(self, params)
        self.needs_mp_inp_sum = True # to avoid adding signals from ports 3,4
        params['acc_slow_port'] = 3 # so inputs at port 3 reset acc_slow
        params['sda_port'] = 4
        acc_sda_reqs.__init__(self, params)
        if 'mu' in params and 'sigma' in params: 
            self.mu = params['mu']
            self.sigma = params['sigma']
            self.mudt = self.mu * self.time_bit # used by flat updater
            self.sqrdt = np.sqrt(self.time_bit) # used by flat updater

    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. """
        sums = [(w*i).sum() for i,w in zip(self.get_mp_inputs(t),
                                           self.get_mp_weights(t))]
        inp = (sums[0] + sums[1] + sums[2]
               #+ self.integ_amp * self.lpf_slow_mp_inp_sum[0]
               + self.integ_amp * self.integ_decay_act
               - self.adapt_amp * self.slow_decay_adapt)
        return ( self.f(inp) - y[0] ) * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        inp = (self.mp_inp_sum[0][s] + self.mp_inp_sum[1][s] + self.mp_inp_sum[2][s]
               #+ self.integ_amp * self.lpf_slow_mp_inp_sum[0]
               + self.integ_amp * self.integ_decay_act
               - self.adapt_amp * self.slow_decay_adapt)
        return ( self.f(inp) - y ) * self.rtau



from units.units import linear

class inpsel_linear(linear):
    """ The linear unit with mp_weights and mp_inputs requirements. """
    def __init__(self, ID, params, network):
        linear.__init__(self, ID, params, network)
        self.syn_needs.update([syn_reqs.mp_weights, syn_reqs.mp_inputs,
                               syn_reqs.acc_slow])

    def upd_lpf_fast_sc_inp_sum_mp(self, time):
        """ Update a list with fast LPF'd scaled sums of inputs at each port. """
        # updated to use mp_inputs, mp_weights
        sums = [(w*i).sum() for i,w in zip(self.mp_inputs, self.mp_weights)]
        # same update rule from other upd_lpf_X methods, put in a list comprehension
        self.lpf_fast_sc_inp_sum_mp = [sums[i] + (self.lpf_fast_sc_inp_sum_mp[i] - sums[i])
                                       * self.fast_prop for i in range(self.n_ports)]

    def upd_lpf_mid_sc_inp_sum_mp(self, time):
        """ Update a list with medium LPF'd scaled sums of inputs at each port. """
        # untouched copy from gated_out_norm_am_sig
        sums = [(w*i).sum() for i,w in zip(self.mp_inputs, self.mp_weights)]
        # same update rule from other upd_lpf_X methods, put in a list comprehension
        self.lpf_mid_sc_inp_sum_mp = [sums[i] + (self.lpf_mid_sc_inp_sum_mp[i] - sums[i])
                                       * self.mid_prop for i in range(self.n_ports)]

    def upd_acc_slow(self, time):
        """ Update the slow speed accumulator. """
        # modified to reset from port 3
        if ( (self.get_mp_inputs(time)[2]*self.get_mp_weights(time)[2]).sum()
             > 0.5 and self.acc_slow > 0.5 ):
             self.acc_slow = 0.
        else: 
            self.acc_slow = 1. - (1.-self.acc_slow)*self.slow_prop



class chwr_linear(unit):
    """ A linear unit with centered half-wave rectification.

        This means that the output of the unit is the positive part of the
        input minus its mean value minus a threshold. The mean value comes
        from slow low-pass filtering. In other words, the output approaches:
        max( I - <I> - thr, 0.)

        A negative threshold can be used in order to set an output activity
        when the input is at its mean value.
    """
    def __init__(self, ID, params, network):
        """ The class constructor. 

        Args:
            ID, params, network: same as the unit class.
            
            OPTIONAL PARAMETERS
            thresh : input threshold. Default is 0.
    """
        unit.__init__(self, ID, params, network)
        if 'thresh' in params: self.thresh = params['thresh']
        else: self.thresh = 0.
        self.syn_needs.update([syn_reqs.lpf_slow_sc_inp_sum])

    def derivatives(self, y, t):
        """ Returns the firing rate derivative at time t, given rate y.

        Args:
            y : array-like with one single element.
            t : a float.
        """
        return max(self.get_input_sum(t) -
                   self.lpf_slow_sc_inp_sum -
                   self.thresh, 0.) - y[0]

    def dt_fun(self, y, s):
        """ The derivatives function for flat networks. """
        return max(self.inp_sum[s] - 
                   self.lpf_slow_sc_inp_sum - 
                   self.thresh, 0.) - y
        
        
