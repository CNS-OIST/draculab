"""
spinal_units.py
The units used in the motor control model.
"""
from draculab import unit_types, synapse_types, syn_reqs  # names of models and requirements
from units.units import unit, sigmoidal
import numpy as np

class am_pm_oscillator(unit):
    """
    An oscillator with amplitude and phase modulated by inputs.

    The outuput of the unit is the sum of a constant part and a sinusoidal part.
    Both of their amplitudes are modulated by the sum of the inputs at port 0.
    The phase of the sinusoidal part is modulated by the relative phase of the inputs
    at port 1 through a method chosen by the user. 

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
                                  The delay is in units of min_delay steps. It is 
                                  recommended to use the 'post_delay' value of the 
                                  rga synapses. 
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
        self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.lpf_slow_mp_inp_sum])
        if params['F'] == 'zero':
            self.f = lambda x : 0.
        elif params['F'] == 'input_sum':
            self.f = self.input_sum_f
            self.syn_needs.update([syn_reqs.lpf_slow, syn_reqs.mp_inputs, 
                                   syn_reqs.lpf_slow_mp_inp_sum])
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
        Dc = y[1]*(1. - y[1])*(I[0] + I[1]) / self.tau_c
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
        Dc = (I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        Dth = (self.omega + self.f(I)) / self.tau_t
        DI0 = np.tanh(I[0] - y[3]) / self.tau_s
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

    def upd_lpf_slow_mp_inp_sum(self, time):
        """ Update the list with slow LPF'd scaled sums of inputs at individual ports.
        """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_slow_mp_inp_sum updated backwards in time']
        inputs = self.mp_inputs    # updated because mp_inputs is a requirement variable
        weights = self.get_mp_weights(time)
        dots = [ np.dot(i, w) for i, w in zip(inputs, weights) ]
        # same update rule from other upd_lpf_X methods above, put in a list comprehension
        self.lpf_slow_mp_inp_sum = [dots[i] + (self.lpf_slow_mp_inp_sum[i] - dots[i]) * 
                                   np.exp( (self.last_time-time)/self.tau_slow ) 
                                   for i in range(self.n_ports)]

    def upd_lpf_mid_mp_raw_inp_sum(self, time):
        """ Update the list with medium LPF'd sums of inputs at individual ports. """
        sums = [sum(inps) for inps in self.mp_inputs]
        # same update rule from other upd_lpf_X methods above, put in a list comprehension
        self.lpf_mid_mp_raw_inp_sum = [sums[i] + (self.lpf_mid_mp_raw_inp_sum[i] - sums[i])
                                       * self.mid_prop for i in range(self.n_ports)]

    def upd_inp_deriv_mp(self, time):
        """ Update the list with input derivatives for each port.  """
        u = self.net.units
        self.inp_deriv_mp = [[u[uid[idx]].get_lpf_fast(dely[idx]) -
                              u[uid[idx]].get_lpf_mid(dely[idx]) 
                              for idx in range(len(uid))]
                              for uid, dely in self.pre_list_del_mp]
        #self.inp_deriv_mp = [[u[uid].get_lpf_fast(dely) - u[uid].get_lpf_mid(dely) 
        #                      for uid, dely in uid_dely] 
        #                      for uid_dely in self.pre_list_del_mp]

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

    def upd_l0_norm_factor_mp(self, time):
        """ Update the factors to normalize the L0 norms for weight vectors. """
        self.l0_norm_factor_mp = [ 1. / (np.absolute(ws).sum() + 1e-32) 
                                      for ws in self.get_mp_weights(time)]
 

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


class out_norm_gated_sig(unit):
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
            raise AssertionError('out_norm_gated_sig units must have n_ports=2')
        else:
            params['n_ports'] == 2
        unit.__init__(self, ID, params, network)
        self.des_out_w_abs_sum = params['des_out_w_abs_sum']
        self.needs_mp_inp_sum = True # in case we flatten
      
    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. """
        I = [(w*i).sum() for w, i in zip(get_mp_weights(t), get_mp_weights(t))]
        return ( I[1]*self.f(I[0]) - y[0] ) * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        return ( self.mp_inp_sum[1][s]*self.f(self.mp_inp_sum[0][s]) - y ) * self.rtau


