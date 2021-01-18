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
            creator, but only considers these entries: 
            'inp_deriv_ports' : A list with the numbers of the ports where all
                                inp_deriv_mp methods will calculate their 
                                derivatives. Defaults to a list with all ports.
            'del_inp_ports' : A list with the numbers of the ports where 
                              del_inp_mp, and del_inp_avg_mp will obtain their
                              inputs. Defaults to all ports.
            'xd_inp_deriv_p' : List with the number of the ports where
                               xtra_del_inp_deriv_mp and
                               xtra_del_inp_deriv_mp_sc_sum will work.
                               Defaults to all ports.
        """
        #self.syn_needs.update([syn_reqs.lpf_slow_sc_inp_sum])

        # inp_deriv_ports works by indicating the add_ methods to restrict the
        # entries in pre_list_del_mp and pre_list_mp.
        if 'inp_deriv_ports' in params:
            self.inp_deriv_ports = params['inp_deriv_ports']
        # del_inp_ports works by indicating the add_del_inp_mp method to
        # restrict the entries in dim_act_del.
        if 'del_inp_ports' in params:
            self.del_inp_ports = params['del_inp_ports']
        # xd_inp_deriv works by indicating the add_xtra_del_inp_deriv_mp method
        # to restrict the entries in ??? 
        if 'xd_inp_deriv_p' in params:
            self.xd_inp_deriv_p = params['xd_inp_deriv_p']

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
 
    def upd_xtra_del_inp_deriv_mp(self, time):
        """ Update the list with extra delayed input derivatives for each port. """
        u = self.net.units
        self.xtra_del_inp_deriv_mp = [
            [u[uid[idx]].get_lpf_fast(self.xtra_inp_del + dely[idx]) - 
             u[uid[idx]].get_lpf_mid(self.xtra_inp_del + dely[idx]) 
             for idx in range(len(uid))] for uid, dely in self.pre_uid_del_mp]

    def upd_xtra_del_inp_deriv_mp_sc_sum(self,time):
        """ Update list with scaled sums of extra delayed input derivatives. """
        #self.xtra_del_inp_deriv_mp_sc_sum = [sum([w*ddiff for w,ddiff in
        #    zip(w_list, ddiff_list)]) for w_list, ddiff_list in
        #    zip(self.mp_weights, self.xtra_del_inp_deriv_mp)]
        self.xtra_del_inp_deriv_mp_sc_sum = [sum([w_l[idx]*ddiff_l[idx] 
                    for idx in range(len(ddiff_l))]) for w_l,ddiff_l in 
                    zip(self.mp_weights, self.xtra_del_inp_deriv_mp)]

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
        self.double_del_inp_deriv_mp[0] = [[u[uid].get_lpf_fast(self.custom_del_diff) - 
                                u[uid].get_lpf_mid(self.custom_del_diff) 
                                for uid in lst] for lst in self.pre_list_mp]
        self.double_del_inp_deriv_mp[1] = [[u[uid].get_lpf_fast(self.custom_inp_del2) - 
                        u[uid].get_lpf_mid(self.custom_inp_del2) 
                        for uid in lst] for lst in self.pre_list_mp]
 
    def upd_double_del_avg_inp_deriv_mp(self, time):
        """ Update averages of input derivatives with two delays for each port. """
        self.double_del_avg_inp_deriv_mp[0] = [np.mean(l) if len(l) > 0 else 0.
                                    for l in self.double_del_inp_deriv_mp[0]]
        self.double_del_avg_inp_deriv_mp[1] = [np.mean(l) if len(l) > 0 else 0.
                                    for l in self.double_del_inp_deriv_mp[1]]

    def upd_slow_inp_deriv_mp(self, time):
        """ Update the list with slow input derivatives for each port.  """
        u = self.net.units
        self.slow_inp_deriv_mp = [[u[uid[idx]].get_lpf_mid(dely[idx]) -
                                   u[uid[idx]].get_lpf_slow(dely[idx]) 
                                   for idx in range(len(uid))]
                                   for uid, dely in self.pre_list_del_mp]

    def upd_avg_slow_inp_deriv_mp(self, time):
        """ Update the list with average slow input derivatives per port. """
        self.avg_slow_inp_deriv_mp = [np.mean(l) if len(l) > 0 else 0.
                                 for l in self.slow_inp_deriv_mp]

    def upd_inp_avg_mp(self, time):
        """ Update the averages of the inputs for each port. """
        self.inp_avg_mp = [r*p_inps.sum() for p_inps,r in 
                           zip(self.mp_inputs, self.inp_recip_mp)]

    def upd_del_inp_mp(self, time):
        """ Update the arrays with delayed inputs for each port. """
        self.del_inp_mp = [[a[0](time-a[1]) for a in l] 
                           for l in self.dim_act_del] # if len(l)>0]

    def upd_del_inp_avg_mp(self, time):
        """ Update the average of delayed inputs for each port. """
        self.del_inp_avg_mp = [r*sum(l) for r,l in
                               zip(self.avg_fact_mp, self.del_inp_mp)]

    def upd_sc_inp_sum_deriv_mp(self, time):
        """ Update the derivatives for the scaled sum of inputs at each port."""
        self.sc_inp_sum_deriv_mp = [sum([w*diff for w,diff in 
                zip(w_list, diff_list)]) for w_list, diff_list in 
                zip(self.mp_weights, self.inp_deriv_mp)]

    def upd_idel_ip_ip_mp(self, time):
        """ Update the dot product of delayed and derived inputs per port."""
        self.idel_ip_ip_mp = [sum([ldi[i]*lip[i] for i in range(len(ldi))])
                   for ldi, lip in zip(self.del_inp_mp, self.inp_deriv_mp)]

    def upd_dni_ip_ip_mp(self, time):
        """ Update dot product of delayed-normalized and diff'd inputs per port."""
        self.dni_ip_ip_mp = [
            sum([(ldi[i]-iavg)*lip[i] for i in range(len(ldi))]) 
            for ldi, lip, iavg in 
                zip(self.del_inp_mp, self.inp_deriv_mp, self.del_inp_avg_mp)]

    def upd_i_ip_ip_mp(self, time):
        """ Update the inner product of input with its derivative per port."""
        self.i_ip_ip_mp = [ sum([inp[j]*dinp[j] for j in range(len(inp))]) if dinp
                else 0. for inp, dinp in zip(self.mp_inputs, self.inp_deriv_mp) ]

    def upd_ni_ip_ip_mp(self, time):
        """ Update dot product of normalized input with its derivative per port."""
        self.ni_ip_ip_mp = [ sum([(inp[j]-avg)*dinp[j] for j in range(len(inp))])
                if dinp else 0. for inp, dinp, avg in 
                zip(self.mp_inputs, self.inp_deriv_mp, self.inp_avg_mp) ]


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

    def upd_sc_inp_sum_diff_mp(self, time):
        """ Update the derivatives for the scaled sum of inputs at each port."""
        self.sc_inp_sum_diff_mp = [lpf_fast - lpf_mid for lpf_fast, lpf_mid in
                  zip(self.lpf_fast_sc_inp_sum_mp, self.lpf_mid_sc_inp_sum_mp)]



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
            # to produce increase in low-activity units
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

    The model uses 4-dimensional dynamics, so its state at a given time is a 
    4-element array. The first element corresponds to the unit's activity, which
    comes from the sum of the constant part and the oscillating part. The second
    element corresponds to the constant part of the output. The third element
    corresponds to the phase of the oscillating part. The fourth element is a
    variable that follows the scaled sum of the inputs at port 0; this is used in
    order to obtain the derivative of those inputs without the need of two low-pass
    filters and new requirements. 

    The equations of the model currently look like this:
    tau_u * u'   = u*(1-u)*[c' + (c-u) + <Is>'*sin(th) + <Is>th'cos(th)]
    tau_c * c'   = [Is + Im*c] * (1-c)
    tau_t * th'  = w + F(u_1, ..., u_n, U)
    tau_s * <Is>'= tanh(Is - <Is>)
    ACTUALLY, I'M IN THE MIDDLE OF MODIFICATIONS

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
                'inp_del_steps' : integer with the delay to be used by the
                                del_inp_mp requirement, overriding
                                custom_inp_del.
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
        if 'inp_del_steps' in params:
            self.inp_del_steps = params['inp_del_steps']
        if 'mu' in params:
            self.mu = params['mu']
        if 'sigma' in params:
            self.sigma = params['sigma']
        # latency is the time delay in the phase shift of a first-order LPF
        # with a sinusoidal input of angular frequency omega
        #self.latency = np.arctan(self.tau_c*self.omega)/self.omega
        self.mudt = self.mu * self.time_bit # used by flat updaters
        self.mudt_vec = np.zeros(self.dim)
        self.mudt_vec[0] = self.mudt
        self.sqrdt = np.sqrt(self.time_bit) # used by flat updater
        self.needs_mp_inp_sum = True # dt_fun uses mp_inp_sum
        rga_reqs.__init__(self, params)

    def derivatives(self, y, t):
        """ Implements the equations of the am_pm_oscillator.

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
        #Dc = y[1] * (I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        #slow_I0 = self.lpf_slow_mp_inp_sum[0]
        #Dc = ( I[0]*(1. - y[1]) + (I[1] - slow_I0)*y[1] ) / self.tau_c
        #Dc = ( I[0]*(1. - y[1]) - slow_I0*y[1] ) / self.tau_c
        Dth = (self.omega + self.f(I)) / self.tau_t
        #DI0 = np.tanh(I[0] - y[3]) / self.tau_s # default implementation
        DI0 = np.tanh(I[0]+I[1] - y[3]) / self.tau_s # Add both inputs
        #DI = (np.tanh(I[0]+I[1]) - y[3]) / self.tau_s
        #DI0 = (sum(self.get_mp_inputs(t)[0]) - y[3]) / self.tau_s
        #DIs = self.D_factor * (I[0] - y[3])
        #Du = (Dc + I[0]*Dth*np.cos(y[2]) + DI0*np.sin(y[2])) / self.tau_u
        #ex = np.exp(-y[3]*np.sin(y[2]))
        #prime = 0.2*ex/(1.+ex)
        #Du = ((1.-y[0]) * (y[0]-.01) * (y[1]*Dc + (y[1] - y[0]) + 
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
        Dc = (I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        #Dc = y[1] * (I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        Dth = (self.omega + self.f(I)) / self.tau_t
        #DI0 = np.tanh(I[0] - y[3]) / self.tau_s # default implementation
        DI0 = np.tanh(I[0]+I[1] - y[3]) / self.tau_s # Add both inputs
        #DI = (np.tanh(I[0]+I[1]) - y[3]) / self.tau_s
        #Dc += DI0 #/self.tau_c
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


class am_oscillator(unit, rga_reqs):
    """
    An oscillator with amplitude modulated by inputs.

    The outuput of the unit is the sum of a constant part and a sinusoidal part.
    Both of their amplitudes are modulated by the sum of the inputs at port 0.

    The model uses 3-dimensional dynamics, so its state at a given time is a 
    3-element array. The first element corresponds to the unit's activity, which
    comes from the sum of the constant part and the oscillating part. The second
    element corresponds to the constant part of the output. The third element
    is a bounded LPF'd scaled sum of inputs.

    For the sake of RGA synapses two ports are assumed, with port 0 being
    the "error" port. Using a different number of ports causes a warning.
    
    The equations of the model currently look like this:
    tau_u * u'   = u*(1-u)*[c' + (c-u) + <I>'*sin(wt) + <I>wcos(wt)]
    tau_c * c'   = I  * c * (1-c)
    tau_s * <I>'= tanh(I - <I>)

    where: 
        u = units's activity,
        c = constant part of the unit's activity,
        <I> = low-pass filtered scaled input,
        I = scaled sum of inputs from both ports, 
        w = intrinsic frequency of the oscillation, 
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS
                'tau_u' : Time constant for the unit's activity.
                'tau_c' : Time constant for non-oscillatory dynamics.
                'tau_s' : Time constant for the input low-pass filter.
                          A small value is recommended, so the derivative
                          of the LPF'd signal is similar to the derivative
                          of the original signal.
                'omega' : intrinsic oscillation angular frequency. 
                'multidim' : the Boolean literal 'True'. This is used to indicate
                             net.create that the 'init_val' parameter may be a single
                             initial value even if it is a list.
                Using rga synapses brings an extra required parameter:
                'custom_inp_del' : an integer indicating the delay that the rga
                                  learning rule uses for the lateral port inputs. 
                                  The delay is in units of min_delay steps. 
                OPTIONAL PARAMETERS
                'mu' : mean of white noise when using noisy integration
                'sigma' : standard deviation of noise when using noisy integration
                'inp_del_steps' : integer with the delay to be used by the
                                del_inp_mp requirement, overriding
                                custom_inp_del.
        Raises:
            ValueError

        """
        params['multidim'] = True
        if len(params['init_val']) != 3:
            raise ValueError("Initial values for the am_oscillator must " +
                             "consist of a 3-element array.")
        if 'n_ports' in params:
            if params['n_ports'] != 2:
                from warnings import warn
                warn("am_oscillator uses two input ports with rga synapses.",
                 UserWarning)
                #raise ValueError("am_oscillator units use two input ports.")

        else:
            params['n_ports'] = 2
        unit.__init__(self, ID, params, network) # parent's constructor
        rga_reqs.__init__(self, params)
        self.tau_u = params['tau_u']
        self.tau_c = params['tau_c']
        self.tau_s = params['tau_s']
        self.omega = params['omega']
        #self.syn_needs.update([syn_reqs.mp_inputs, syn_reqs.mp_weights,
        #                       syn_reqs.lpf_slow_mp_inp_sum])
        if 'custom_inp_del' in params:
            self.custom_inp_del = params['custom_inp_del']
        if 'inp_del_steps' in params:
            self.inp_del_steps = params['inp_del_steps']
        if 'mu' in params:
            self.mu = params['mu']
        if 'sigma' in params:
            self.sigma = params['sigma']
        # latency is the time delay in the phase shift of a first-order LPF
        # with a sinusoidal input of angular frequency omega
        #self.latency = np.arctan(self.tau_c*self.omega)/self.omega
        self.mudt = self.mu * self.time_bit # used by flat updaters
        self.mudt_vec = np.zeros(self.dim)
        self.mudt_vec[0] = self.mudt
        self.sqrdt = np.sqrt(self.time_bit) # used by flat updater
        self.needs_mp_inp_sum = True # dt_fun uses mp_inp_sum

    def derivatives(self, y, t):
        """ Implements the equations of the am_oscillator.

        Args:
            y : list or Numpy array with the 3-element state vector:
              y[0] : u  -- unit's activity,
              y[1] : c  -- constant part of the input,
              y[2] : I -- LPF'd sum of inputs.
            t : time when the derivative is evaluated.
        Returns:
            3-element numpy array with state variable derivatives.
        """
        # get the input sum at each port
        I = [ np.dot(i, w) for i, w in 
              zip(self.get_mp_inputs(t), self.get_mp_weights(t)) ]
        # Obtain the derivatives
        #Dc = (I[0]+I[1]) * y[1] * (1. - y[1]) / self.tau_c
        #Dc = (I[0]+I[1]) * (-y[1] - .5) * (1. - y[1]) / self.tau_c
        #Dc = (I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        Dc = y[1]*(I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        #DI = np.tanh(I - y[2]) / self.tau_s
        #DI = np.tanh(I[0]+I[1] - y[2]) / self.tau_s
        #DI = (np.tanh(I[0]+I[1]) - y[2]) / self.tau_s
        #Is = I[0] + I[1]
        #DI = (Is + 1.)*(1. - Is)/self.tau_s
        DI = (np.tanh(I[0]+I[1]) - y[2]) / self.tau_s
        th = self.omega*t
        #Du = ((1.-y[0]) * (y[0]-.01) * (Dc + (y[1] - y[0]) + 
        Du = ((1.-y[0]) * (y[0]-.01) * ( y[1] - y[0] + 
               y[2]*self.omega*np.cos(th) + DI*np.sin(th))) / self.tau_u
        return np.array([Du, Dc, DI])

    def dt_fun(self, y, s):
        """ the derivatives function when the network is flat.
:
            y : list or Numpy array with the 3-element state vector:
              y[0] : u  -- unit's activity,
              y[1] : c  -- constant part of the input,
              y[2] : I -- LPF'd sum of inputs.
            s : index to inp_sum for current time point
        Returns:
            3-element numpy array with state variable derivatives.
        """
        t = self.times[s - self.min_buff_size]
        # get the input sum at each port
        #I = sum(self.mp_inp_sum[:,s])
        I = [ port_sum[s] for port_sum in self.mp_inp_sum ]
        # Obtain the derivatives
        #Dc = (I[0]+I[1]) * (-y[1] -.5) * (1. - y[1]) / self.tau_c
        #Dc = (I[0]+I[1]) * y[1] * (1. - y[1]) / self.tau_c
        #Dc = (I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        Dc = y[1]*(I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        #DI = np.tanh(I - y[2]) / self.tau_s
        #DI = np.tanh(I[0]+I[1] - y[2]) / self.tau_s
        DI = 0. #(np.tanh(I[0]+I[1]) - y[2]) / self.tau_s
        #DI = (1./(1.+np.exp(-0.5*(I[0] + I[1])))-y[2])/self.tau_s
        th = self.omega*t
        #Du = ((1.-y[0]) * (y[0]-.01) * (Dc + (y[1] - y[0]) + 
        #Du = ((1.-y[0]) * (y[0]-.01) * ( y[1] - y[0] + 
        #       y[2]*self.omega*np.cos(th) + DI*np.sin(th))) / self.tau_u
        # experimental 2D dynamics:
        Is = I[0] #+ I[1]
        Du = (y[1] - y[0] + np.tanh(Is) * np.sin(th)) / self.tau_u
        return np.array([Du, Dc, DI])


class am_oscillator2D(unit, rga_reqs):
    """
    Amplitude-modulated oscillator with two state variables.

    The outuput of the unit is the sum of a constant part and a sinusoidal part.
    Both of their amplitudes are modulated by the sum of the inputs at port 0.

    The model uses 2-dimensional dynamics, so its state at a given time is a 
    2-element array. The first element corresponds to the unit's activity, which
    comes from the sum of the constant part and the oscillating part. The second
    element corresponds to the constant part of the output.

    For the sake of RGA synapses at least two ports are used. Port 0 is assumed
    to be the "error" port, whereas port 1 is the "lateral" port. Additionally,
    if n_ports=3, then port 2 is the "global error" port, to be used by rga_ge
    synapses.
    
    The equations of the model currently look like this:
    u'   = (c - u + A*tanh(I)*sin(th) / tau_u
    c'   = c * (I0 + I1*c)*(1-c) / tau_c

    where: 
        u = units's activity,
        c = constant part of the unit's activity,
        I0 = scaled input sum at port 0,
        I1 = scaled input sum at port 1,
        I = I0+I1,
        th = t*omega (phase of oscillation),
        A = amplitude of the oscillations.
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS
                'tau_u' : Time constant for the unit's activity.
                'tau_c' : Time constant for non-oscillatory dynamics.
                'omega' : intrinsic oscillation angular frequency. 
                'multidim' : the Boolean literal 'True'. This is used to indicate
                             net.create that the 'init_val' parameter may be a single
                             initial value even if it is a list.
                Using rga synapses brings an extra required parameter:
                'custom_inp_del' : an integer indicating the delay that the rga
                                  learning rule uses for the lateral port inputs. 
                                  The delay is in units of min_delay steps. 
                OPTIONAL PARAMETERS
                A = amplitude of the oscillations. Default is 1.
                Using rga_ge synapses requires to set n_ports = 3:
                'n_ports' : number of input ports. Default is 2.
                'mu' : mean of white noise when using noisy integration
                'sigma' : standard deviation of noise when using noisy integration
                'inp_del_steps' : integer with the delay to be used by the
                                del_inp_mp requirement, overriding
                                custom_inp_del.
        Raises:
            ValueError

        """
        params['multidim'] = True
        if len(params['init_val']) != 2:
            raise ValueError("Initial values for the am_oscillator2D must " +
                             "consist of a 2-element array.")
        if 'n_ports' in params:
            if params['n_ports'] != 2 and params['n_ports'] != 3:
                raise ValueError("am_oscillator2D uses 2 or 3 input ports.")
        else:
            params['n_ports'] = 2
        unit.__init__(self, ID, params, network) # parent's constructor
        rga_reqs.__init__(self, params)
        self.tau_u = params['tau_u']
        self.tau_c = params['tau_c']
        self.omega = params['omega']
        if 'custom_inp_del' in params:
            self.custom_inp_del = params['custom_inp_del']
        if 'inp_del_steps' in params:
            self.inp_del_steps = params['inp_del_steps']
        if 'A' in params: self.A = params['A']
        else: self.A = 1.
        if 'mu' in params:
            self.mu = params['mu']
        if 'sigma' in params:
            self.sigma = params['sigma']
        # latency is the time delay in the phase shift of a first-order LPF
        # with a sinusoidal input of angular frequency omega
        self.latency = np.arctan(self.tau_c*self.omega)/self.omega
        self.mudt = self.mu * self.time_bit # used by flat updaters
        self.mudt_vec = np.zeros(self.dim)
        self.mudt_vec[0] = self.mudt
        self.sqrdt = np.sqrt(self.time_bit) # used by flat updater
        self.needs_mp_inp_sum = True # dt_fun uses mp_inp_sum

    def derivatives(self, y, t):
        """ Implements the equations of the am_oscillator.

        Args:
            y : list or Numpy array with the 2-element state vector:
              y[0] : u  -- unit's activity,
              y[1] : c  -- constant part of the input,
            t : time when the derivative is evaluated.
        Returns:
            2-element numpy array with state variable derivatives.
        """
        # get the input sum at each port
        I = [ np.dot(i, w) for i, w in 
              zip(self.get_mp_inputs(t), self.get_mp_weights(t)) ]
        # Obtain the derivatives
        Dc = y[1]*(I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        #Dc = (I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        th = self.omega*t
        Du = (y[1] - y[0] + Dc +
              self.A * np.tanh(I[0]+I[1]) * np.sin(th)) / self.tau_u
        #Du = (1. - y[0]) * (y[1] - y[0] + Dc +
        #      self.A * np.tanh(I[0]) * np.sin(th)) / self.tau_u
        return np.array([Du, Dc])

    def dt_fun(self, y, s):
        """ The derivatives function when the network is flat.

            y : list or Numpy array with the 3-element state vector:
              y[0] : u  -- unit's activity,
              y[1] : c  -- constant part of the input,
            s : index to inp_sum for current time point
        Returns:
            3-element numpy array with state variable derivatives.
        """
        t = self.times[s - self.min_buff_size]
        # get the input sum at each port
        #I = sum(self.mp_inp_sum[:,s])
        I = [ port_sum[s] for port_sum in self.mp_inp_sum ]
        # Obtain the derivatives
        Dc = y[1]*(I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        #Dc = (I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        th = self.omega*t
        Du = (y[1] - y[0] + Dc +
              self.A * np.tanh(I[0]+I[1])*np.sin(th)) / self.tau_u
        #Du = (1. - y[0]) * (y[1] - y[0] + Dc +
        #      self.A * np.tanh(I[0])*np.sin(th)) / self.tau_u
        return np.array([Du, Dc])



# CURRENTLY DECIDING WHETHER TO FINISH WRITING THIS CLASS...
#==============================================================================
class am_osc2D_adapt(unit, rga_reqs, acc_sda_reqs):
    """
    Two-dimensional amplitude-modulated oscillator with adaptation.

    The outuput of the unit is the sum of a constant part and a sinusoidal part.
    Both of their amplitudes are modulated by the sum of the inputs at port 0.

    The model uses 2-dimensional dynamics, so its state at a given time is a 
    2-element array. The first element corresponds to the unit's activity, which
    comes from the sum of the constant part and the oscillating part. The second
    element corresponds to the constant part of the output.

    For the sake of RGA synapses and adaptation, four ports are used. Port 0 
    is assumed to be the "error" port, whereas port 1 is the "lateral" port.
    Port 2 is the "global error" port, to be used by rga_ge synapses. 
    The fourth port is used to trigger adaptation, whih is meant to depress the
    units that were the most active.
    
    The equations of the model currently look like this:
    u'   = (c - u + A*tanh(I)*sin(th) / tau_u
    c'   = c * (I0 + I1*c)*(1-c) / tau_c

    where: 
        u = units's activity,
        c = constant part of the unit's activity,
        I0 = scaled input sum at port 0,
        I1 = scaled input sum at port 1,
        I = I0+I1,
        th = t*omega (phase of oscillation),
        A = amplitude of the oscillations.
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS
                'tau_u' : Time constant for the unit's activity.
                'tau_c' : Time constant for non-oscillatory dynamics.
                'omega' : intrinsic oscillation angular frequency. 
                'multidim' : the Boolean literal 'True'. This is used to indicate
                             net.create that the 'init_val' parameter may be a single
                             initial value even if it is a list.
                Using rga synapses brings an extra required parameter:
                'custom_inp_del' : an integer indicating the delay that the rga
                                  learning rule uses for the lateral port inputs. 
                                  The delay is in units of min_delay steps. 
                OPTIONAL PARAMETERS
                A = amplitude of the oscillations. Default is 1.
                Using rga_ge synapses requires to set n_ports = 3:
                'n_ports' : number of input ports. Default is 2.
                'mu' : mean of white noise when using noisy integration
                'sigma' : standard deviation of noise when using noisy integration
                'inp_del_steps' : integer with the delay to be used by the
                                del_inp_mp requirement, overriding
                                custom_inp_del.
        Raises:
            ValueError

        """
        params['multidim'] = True
        if len(params['init_val']) != 2:
            raise ValueError("Initial values for the am_oscillator2D must " +
                             "consist of a 2-element array.")
        if 'n_ports' in params:
            if params['n_ports'] != 4:
                raise ValueError("am_osc2D_adapt uses 4 input ports.")
        else:
            params['n_ports'] = 4
        unit.__init__(self, ID, params, network) # parent's constructor
        rga_reqs.__init__(self, params)
        self.tau_u = params['tau_u']
        self.tau_c = params['tau_c']
        self.omega = params['omega']
        params['sda_port'] = 3
        acc_sda_reqs.__init__(self, params)
        if 'custom_inp_del' in params:
            self.custom_inp_del = params['custom_inp_del']
        if 'inp_del_steps' in params:
            self.inp_del_steps = params['inp_del_steps']
        if 'A' in params: self.A = params['A']
        else: self.A = 1.
        if 'mu' in params:
            self.mu = params['mu']
        if 'sigma' in params:
            self.sigma = params['sigma']
        # latency is the time delay in the phase shift of a first-order LPF
        # with a sinusoidal input of angular frequency omega
        self.latency = np.arctan(self.tau_c*self.omega)/self.omega
        self.mudt = self.mu * self.time_bit # used by flat updaters
        self.mudt_vec = np.zeros(self.dim)
        self.mudt_vec[0] = self.mudt
        self.sqrdt = np.sqrt(self.time_bit) # used by flat updater
        self.needs_mp_inp_sum = True # dt_fun uses mp_inp_sum

    def derivatives(self, y, t):
        """ Implements the equations of the am_oscillator.

        Args:
            y : list or Numpy array with the 2-element state vector:
              y[0] : u  -- unit's activity,
              y[1] : c  -- constant part of the input,
            t : time when the derivative is evaluated.
        Returns:
            2-element numpy array with state variable derivatives.
        """
        # get the input sum at each port
        I = [ np.dot(i, w) for i, w in 
              zip(self.get_mp_inputs(t), self.get_mp_weights(t)) ]
        # Obtain the derivatives
        Dc = y[1]*(I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        #Dc = (I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        th = self.omega*t
        Du = (y[1] - y[0] + Dc +
              self.A * np.tanh(I[0]+I[1]) * np.sin(th)) / self.tau_u
        #Du = (1. - y[0]) * (y[1] - y[0] + Dc +
        #      self.A * np.tanh(I[0]) * np.sin(th)) / self.tau_u
        return np.array([Du, Dc])

    def dt_fun(self, y, s):
        """ The derivatives function when the network is flat.

            y : list or Numpy array with the 3-element state vector:
              y[0] : u  -- unit's activity,
              y[1] : c  -- constant part of the input,
            s : index to inp_sum for current time point
        Returns:
            3-element numpy array with state variable derivatives.
        """
        t = self.times[s - self.min_buff_size]
        # get the input sum at each port
        #I = sum(self.mp_inp_sum[:,s])
        I = [ port_sum[s] for port_sum in self.mp_inp_sum ]
        # Obtain the derivatives
        Dc = y[1]*(I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        #Dc = (I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        th = self.omega*t
        Du = (y[1] - y[0] + Dc +
              self.A * np.tanh(I[0]+I[1])*np.sin(th)) / self.tau_u
        #Du = (1. - y[0]) * (y[1] - y[0] + Dc +
        #      self.A * np.tanh(I[0])*np.sin(th)) / self.tau_u
        return np.array([Du, Dc])

#==============================================================================


class out_norm_sig(sigmoidal, lpf_sc_inp_sum_mp_reqs):
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


class gated_out_norm_sig(sigmoidal, lpf_sc_inp_sum_mp_reqs, acc_sda_reqs):
    """ A sigmoidal with modulated plasticity and output normaliztion.

        The output of the unit is the sigmoidal function applied to the
        scaled sum of port 0 inputs, plus a `p1_inp` factor times the scaled sum
        of port 1 inputs.
        Inputs at port 2 reset the acc_slow variable, which is used by the 
        gated_corr_inh synapses in order to modulate plasticity.

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
                    'p1_inp' : The scaled sum of port 0 inputs is multiplied by
                               this parameter before becoming being added to the
                               arguments of the sigmoidal. Default 0.
                    'out_norm_type' : a synapse type's integer value. If included,
                                 the sum of absolute weights for outgoing
                                 connections will only consider synapses of
                                 that type. For example, you may set it as:
                                 {...,
                                 'out_norm_type' : synapse_types.gated_rga_diff.value}
            Raises:
                AssertionError.
        """
        if 'n_ports' in params and params['n_ports'] != 3:
            raise AssertionError('gated_out_norm_am_sig units must have n_ports=3')
        else:
            params['n_ports'] = 3
        sigmoidal.__init__(self, ID, params, network)
        self.des_out_w_abs_sum = params['des_out_w_abs_sum']
        if 'out_norm_type' in params:
            self.out_norm_type = params['out_norm_type']
        self.syn_needs.update([syn_reqs.acc_slow, syn_reqs.mp_inputs, 
                               syn_reqs.mp_weights])
        self.needs_mp_inp_sum = True # in case we flatten
        lpf_sc_inp_sum_mp_reqs.__init__(self, params)
        params['acc_slow_port'] = 2 # so inputs at port 2 reset acc_slow
        acc_sda_reqs.__init__(self, params)
        if 'p1_inp' in params:
            self.p1_inp = params['p1_inp']
        else:
            self.p1_inp = 0.
      
    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. """
        I = [(w*i).sum() for w, i in zip(self.get_mp_weights(t), 
                                         self.get_mp_inputs(t))]
        return ( self.f(I[0] + self.p1_inp*I[1]) - y[0] ) * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        return ( self.f(self.mp_inp_sum[0][s] + 
                 self.p1_inp*self.mp_inp_sum[1][s]) - y ) * self.rtau


class gated_out_norm_am_sig(sigmoidal, lpf_sc_inp_sum_mp_reqs, acc_sda_reqs):
    """ A sigmoidal with modulated amplitude and plasticity.

        The output of the unit is the sigmoidal function applied to the
        scaled sum of port 0 inputs, times the scaled sum of port 1 inputs.
        Inputs at port 2 reset the acc_slow variable, which is used by the 
        gated_corr_inh synapses in order to modulate plasticity.

        Optionally, the scaled sum of port 1 inputs can also appear in the
        argument to the sigmoidal function. This is controlled by the p1_inp
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
                    'p1_inp' : The scaled sum of port 0 inputs is multiplied by
                               this parameter before becoming being added to the
                               arguments of the sigmoidal. Default 0.
                    'out_norm_type' : a synapse type's integer value. If included,
                                 the sum of absolute weights for outgoing
                                 connections will only consider synapses of
                                 that type. For example, you may set it as:
                                 {...,
                                 'out_norm_type' : synapse_types.gated_rga_diff.value}
            Raises:
                AssertionError.
        """
        if 'n_ports' in params and params['n_ports'] != 3:
            raise AssertionError('gated_out_norm_am_sig units must have n_ports=3')
        else:
            params['n_ports'] = 3
        sigmoidal.__init__(self, ID, params, network)
        self.des_out_w_abs_sum = params['des_out_w_abs_sum']
        if 'out_norm_type' in params:
            self.out_norm_type = params['out_norm_type']
        self.syn_needs.update([syn_reqs.acc_slow, syn_reqs.mp_inputs, 
                               syn_reqs.mp_weights])
        self.needs_mp_inp_sum = True # in case we flatten
        lpf_sc_inp_sum_mp_reqs.__init__(self, params)
        params['acc_slow_port'] = 2 # so inputs at port 2 reset acc_slow
        acc_sda_reqs.__init__(self, params)
        if 'p1_inp' in params:
            self.p1_inp = params['p1_inp']
        else:
            self.p1_inp = 0.
      
    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. """
        I = [(w*i).sum() for w, i in zip(self.get_mp_weights(t), 
                                         self.get_mp_inputs(t))]
        return ( I[1]*self.f(I[0] + self.p1_inp*I[1]) - y[0] ) * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        return ( self.mp_inp_sum[1][s] * self.f(self.mp_inp_sum[0][s] +
                 self.p1_inp*self.mp_inp_sum[1][s]) - y ) * self.rtau


class am_pulse(unit, rga_reqs):
    """
    Integrator with amplitude-modulated periodic pulses.

    The outuput of the unit is the sum of a constant part and a pulsating part.
    The constant part is a soft-bounded integral of the inputs. The pulsating
    emits a pulse whose amplitude is modulated by the sum of the inputs at port
    0.

    The model uses 2-dimensional dynamics, so its state at a given time is a 
    2-element array. The first element corresponds to the unit's activity, which
    comes from the sum of the constant part and the oscillating part. The second
    element corresponds to the constant part of the output.

    For the sake of RGA synapses at least two ports are used. Port 0 is assumed
    to be the "error" port, whereas port 1 is the "lateral" port. Additionally,
    if n_ports=3, then port 2 is the "global error" port, to be used by rga_ge
    synapses. If n_ports=4, then port 3 is the reset signal for the acc_slow
    attribute (used for gated synapses).
    
    The equations of the model currently look like this:
    u'   = (c + A*|tanh(I0+I1)|*sig - u) / tau_u
    c'   = c * (I0 + I1*c)*(1-c) / tau_c

    where: 
        u = units's activity,
        c = constant part of the unit's activity,
        I0 = scaled input sum at port 0,
        I1 = scaled input sum at port 1,
        A = amplitude of the oscillations.
        omega = angular frequency of the pulses
        sig = 1/(1+exp(-b*(cos(omega*t)-thr)))
        b = slope of the sigmoidal
        thr = threshold of the sigmoidal
    """

    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS
                'tau_u' : Time constant for the unit's activity.
                'tau_c' : Time constant for non-oscillatory dynamics.
                'omega' : Angular frequency of the pulses.
                Using rga synapses brings an extra required parameter:
                'custom_inp_del' : an integer indicating the delay that the rga
                                  learning rule uses for the lateral port inputs. 
                                  The delay is in units of min_delay steps. 
                OPTIONAL PARAMETERS
                A = amplitude of the pulses . Default is 1.
                Using rga_ge synapses requires to set n_ports = 3:
                'n_ports' : number of input ports. Default is 2.
                'mu' : mean of white noise when using noisy integration
                'sigma' : standard deviation of noise when using noisy integration
                'multidim' : the Boolean literal 'True'. This is used to indicate
                             net.create that the 'init_val' parameter may be a single
                             initial value even if it is a list.
                'b' : slope of the sigmoidal used for the pulse. Default is 10.
                'thr' : threshold for the sigmoidal. Default is 0.6 .
                'inp_del_steps' : integer with the delay to be used by the
                                del_inp_mp requirement, overriding
                                custom_inp_del.
        Raises:
            ValueError

        """
        params['multidim'] = True
        if len(params['init_val']) != 2:
            raise ValueError("Initial values for the am_pulse model must " +
                             "consist of a 2-element array.")
        if 'n_ports' in params:
            if params['n_ports'] < 2 or params['n_ports'] > 4:
                raise ValueError("am_pulse units use two to four input ports.")
        else:
            params['n_ports'] = 2
        unit.__init__(self, ID, params, network) # parent's constructor
        self.tau_u = params['tau_u']
        self.tau_c = params['tau_c']
        self.omega = params['omega']
        if 'custom_inp_del' in params:
            self.custom_inp_del = params['custom_inp_del']
        if 'inp_del_steps' in params:
            self.inp_del_steps = params['inp_del_steps']
        if 'A' in params: self.A = params['A']
        else: self.A = 1.
        if 'mu' in params:
            self.mu = params['mu']
        if 'sigma' in params:
            self.sigma = params['sigma']
        if 'b' in params: self.b = params['b']
        else: self.b = 10.
        if 'thr' in params: self.thr = params['thr']
        else: self.thr = .6
        # latency is the time delay in the phase shift of a first-order LPF
        # with a sinusoidal input of angular frequency omega
        #self.latency = np.arctan(self.tau_c*self.omega)/self.omega
        self.mudt = self.mu * self.time_bit # used by flat updaters
        self.mudt_vec = np.zeros(self.dim)
        self.mudt_vec[0] = self.mudt
        self.sqrdt = np.sqrt(self.time_bit) # used by flat updater
        self.needs_mp_inp_sum = True # dt_fun uses mp_inp_sum
        # calculate derivatives for all ports?
        # TODO: adjust to final form of rga rule
        #params['inp_deriv_ports'] = [0, 1, 2] 
        #params['del_inp_ports'] = [0, 1, 2] 
        rga_reqs.__init__(self, params)

    def derivatives(self, y, t):
        """ Implements the equations of the am_pulse model.

        Args:
            y : list or Numpy array with the 2-element state vector:
              y[0] : u  -- unit's activity,
              y[1] : c  -- constant part of the input,
            t : time when the derivative is evaluated.
        Returns:
            2-element numpy array with state variable derivatives.
        """
        # get the input sum at each port
        I = [ np.dot(i, w) for i, w in 
              zip(self.get_mp_inputs(t), self.get_mp_weights(t)) ]
        # Obtain the derivatives
        Dc = y[1]*(I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        th = self.omega*t
        sig = 1./(1. + np.exp(-self.b*(np.cos(th) - self.thr)))
        #Du = (y[1] - y[0] + np.tanh(I[0]+I[1]) * (
        #      self.A * self.omega * self.b * sig * (1.-sig) * np.sin(th)) /
        #      self.tau_u )
        #Du = (y[1] + self.A * abs(np.tanh(I[0]+I[1])) * sig - y[0]) / self.tau_u
        Du = (y[1] + self.A * abs(np.tanh(I[0])) * sig - y[0]) / self.tau_u
        return np.array([Du, Dc])

    def dt_fun(self, y, s):
        """ The derivatives function when the network is flat.

            y : list or Numpy array with the 2-element state vector:
              y[0] : u  -- unit's activity,
              y[1] : c  -- constant part of the input,
            s : index to inp_sum for current time point
        Returns:
            2-element numpy array with state variable derivatives.
        """
        t = self.times[s - self.min_buff_size]
        # get the input sum at each port
        I = [ port_sum[s] for port_sum in self.mp_inp_sum ]
        # Obtain the derivatives
        Dc = y[1]*(I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        #Dc = (I[0] + I[1]*y[1]) * (1. - y[1]) / self.tau_c
        th = self.omega*t
        sig = 1./(1. + np.exp(-self.b*(np.cos(th) - self.thr)))
        #Du = (y[1] - y[0] + np.tanh(I[0]+I[1]) * (
        #      self.A * self.omega * self.b * sig * (1.-sig) * np.sin(th)) 
        #      / self.tau_u )
        #Du = (y[1] + self.A * abs(np.tanh(I[0]+I[1])) * sig - y[0]) / self.tau_u
        Du = (y[1] + self.A * abs(np.tanh(I[0])) * sig - y[0]) / self.tau_u
        return np.array([Du, Dc])


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
                    'mu' : drift for the euler_maru solver.
                    'sigma' : standard deviation for the euler_maru solver
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
        self.syn_needs.update([syn_reqs.lpf_slow_sc_inp_sum]) 
        if 'mu' in params: 
            self.mu = params['mu']
            self.mudt = self.mu * self.time_bit # used by flat updater
            self.sqrdt = np.sqrt(self.time_bit) # used by flat updater
        if 'sigma' in params: 
            self.sigma = params['sigma']

    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. """
        inp = self.get_input_sum(t) + self.integ_amp * self.lpf_slow_sc_inp_sum
        return ( self.f(inp) - y[0] ) * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        inp = self.inp_sum[s] + self.integ_amp * self.lpf_slow_sc_inp_sum
        return ( self.f(inp) - y ) * self.rtau


class m_sig(sigmoidal, lpf_sc_inp_sum_mp_reqs, rga_reqs):
    """ A unit meant to be in the motor error layer.

        This means that the unit must be able to handle diff_rm_hebbian, as well
        as input_selection synapses. This turns out to be an extension of the
        rga_syn class.

        There are 2 main uses of this unit, one in the `rl5B` programs, and
        another in the `rl5E`.

        In the case of the rl5B and similar programs, the M units may receive
        4 types of inputs. The first type (port 0 by default)
        consists of the state inputs, connected with reward-modulated
        Hebbian synapses. The next input type (port 1 by default) are the
        "reward" inputs, which are used by the diff_rm_hebbian and inp_sel
        synapses to adapt. The next input type are the "afferent" inputs, by
        default in port 2. These are expected to use the input_selection
        (inp_sel) synapses. The last type of input (port 3) is lateral
        inhibition, usually with static synapses.

        The input_selection class needs an "error" input, whereas the
        diff_rm_hebbian synapse expects a "value" measure. One of the learning
        rates must therefore be negative, since these are opposites.

        In the case of rl5E programs, the unit receives 3 types of input.
        There is an "error" input at port 1, "afferent" inputs at port 2, and
        "lateral" inputs at port 3; this is the same as before, except for the
        "state" inputs, which are no longer present. To keep a homogeneous
        implementation, the number of input ports is still 4, although only 3
        ports are used. 
        
        When calculating the derivatives, the current version sums the inputs
        from all ports to get the scaled input sum. Notice this will also
        include the "value" inputs in the rl5B scenario, but hopefully the
        effect of this is not important.

        This unit has the extra custom_inp_del attribute, and inherits all the
        update functions necessary for rga, inp_sel, and diff_rm_hebbian
        synapses.

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
                    'custom_inp_del' : an integer indicating the delay that the
                                  learning rule uses for the lateral port inputs. 
                                  The delay is in units of min_delay steps. 

                OPTIONAL PARAMETERS
                    'des_out_w_abs_sum' : desired sum of absolute weight values
                                          for the outgoing connections.
                                          Default is 1.
            Raises:
                AssertionError.
        """
        if 'n_ports' in params and params['n_ports'] != 4:
            raise AssertionError('m_sig units must have n_ports=4')
        else:
            params['n_ports'] = 4
        if 'custom_inp_del' in params:
            self.custom_inp_del = params['custom_inp_del']
        else:
            raise AssertionError('m_sig units need a custom_inp_del parameter')
        if 'des_out_w_abs_sum' in params:
            self.des_out_w_abs_sum = params['des_out_w_abs_sum']
        else:
            self.des_out_w_abs_sum = 1.
        sigmoidal.__init__(self, ID, params, network)
        self.integ_amp = params['integ_amp']
        rga_reqs.__init__(self, params) # add requirements and update functions 
        lpf_sc_inp_sum_mp_reqs.__init__(self, params)
        self.needs_mp_inp_sum = False # the sigmoidal uses self.inp_sum
        self.syn_needs.update([syn_reqs.lpf_slow_sc_inp_sum]) 

    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. """
        inp = self.get_input_sum(t) + self.integ_amp * self.lpf_slow_sc_inp_sum
        return ( self.f(inp) - y[0] ) * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        inp = self.inp_sum[s] + self.integ_amp * self.lpf_slow_sc_inp_sum
        return ( self.f(inp) - y ) * self.rtau


class x_sig(sigmoidal, lpf_sc_inp_sum_mp_reqs, rga_reqs):
    """ A unit to configure controllers so they increase a value.

        For the rl5D and rl5E architectures of December 1/2, 2020, these are the
        units that use reward-modulated Hebbian learning to configure the
        afferent input.

        These units must be able to handle diff_rm_hebbian synapses, so they are
        a restricted version of the m_sig class, and an extension of the rga_sig
        class.

        The X units receive at least 2 input types. The first type (port 0 by
        default) consists of the state inputs, connected with reward-modulated
        Hebbian synapses. The next input type (port 1 by default) are the
        "reward" inputs, which are used by the diff_rm_hebbian synapses. 
        Other ports are currently not included in the derivatives/dt_fun
        methods.

        This unit type has the extra custom_inp_del attribute.

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
                    'custom_inp_del' : an integer indicating the delay that the
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
            raise AssertionError('x_sig units must have n_ports=2')
        else:
            params['n_ports'] = 2
        if 'custom_inp_del' in params:
            self.custom_inp_del = params['custom_inp_del']
        else:
            raise AssertionError('x_sig units need a custom_inp_del parameter')
        if 'des_out_w_abs_sum' in params:
            self.des_out_w_abs_sum = params['des_out_w_abs_sum']
        else:
            self.des_out_w_abs_sum = 1.
        sigmoidal.__init__(self, ID, params, network)
        self.integ_amp = params['integ_amp']
        rga_reqs.__init__(self, params) # add requirements and update functions 
        lpf_sc_inp_sum_mp_reqs.__init__(self, params)
        self.needs_mp_inp_sum = True
        self.syn_needs.update([syn_reqs.lpf_slow_sc_inp_sum]) 

    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. """
        mp_i = self.get_mp_inputs(t)
        mp_w = self.get_mp_weights(t)
        inp = (mp_i[0]*mp_w[0]).sum() + self.integ_amp * self.lpf_slow_sc_inp_sum
        return ( self.f(inp) - y[0] ) * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        inp = self.mp_inp_sum[0][s] + self.integ_amp * self.lpf_slow_sc_inp_sum
        return ( self.f(inp) - y ) * self.rtau


class x_switch_sig(sigmoidal, lpf_sc_inp_sum_mp_reqs, rga_reqs):
    """ A configurator (x) unit that switches its output on command.

        This is the brainchild of simulations on Dec. 21st 2020, where it was
        decided that the X unit could have a port that switched its output. This
        means that when port 2 input derivative is larger than a threshold:
            * If output was larger than 0.5, it becomes smaller than 0.5
            * If output was smaller than 0.5, it becomes larger than 0.5

        An overhaul was decided on Jan 5, 2021. When the "switch" binary
        variable is True, then behavior is as above. When "switch" is False,
        then there is no switch. Instead, the output of the unit is held static,
        and is only updated when the port 2 input derivative is larger than a
        threshold.

        The derivative of the port 2 inputs is obtained with the
        lpf_(fast/mid)_sc_inp_sum_mp requirements, since this is also required
        by the diff_rm_hebbian synapses commonly used with this unit.

        The switch in output is achieved by moving the threshold of the unit
        through the addition of an additional "extra threshold", so it becomes
        equal to the current port 0 input sum, plus or minus a 'sw_len' value.
        When switch=False, the extra threshold is the slow LPF'd scaled input
        sum for port 0.

        This unit is a variation of x_sig. For the rl5D and rl5E architectures
        of December 1/2, 2020, x_sig are units that use reward-modulated Hebbian
        learning to configure the afferent input.

        These units must be able to handle diff_rm_hebbian synapses, so they are
        a restricted version of the m_sig class, and an extension of the rga_sig
        class.

        The X units receive at least 3 input types. The first type (port 0 by
        default) consists of the state inputs, connected with reward-modulated
        Hebbian synapses. The next input type (port 1 by default) are the
        "reward" inputs, which are used by the diff_rm_hebbian synapses. 
        The final input type, by default in port 2, is the signal that switches
        the output. Other ports are currently not included in the
        derivatives/dt_fun methods.

        This unit type has the extra custom_inp_del attribute.

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
                    'thresh' : Static threshold of the sigmoidal function.
                    'tau' : Time constant of the update dynamics.
                    'tau_slow' : Slow LPF time constant.
                    'custom_inp_del' : an integer indicating the delay that the
                                  learning rule uses for the lateral port inputs. 
                                  The delay is in units of min_delay steps. 
                    OPTIONAL PARAMETERS
                    'des_out_w_abs_sum' : desired sum of absolute weight values
                                          for the outgoing connections.
                                          Default is 1.
                    'refr_per' : minimum time between switches. Defaul=1 sec.
                    'sw_thresh' : If port 2 derivative larger than this, switch,
                                  or update (if "switch" is False). Default=0.5.
                    'sw_len' : Distance from port 0 input sum after switch.
                               Default is 0.2 .
                    'switch' : Whether port 2 causes a switch. Default is True.
            Raises:
                AssertionError.
        """
        if 'n_ports' in params and params['n_ports'] != 3:
            raise AssertionError('x_switch_sig units must have n_ports=3')
        else:
            params['n_ports'] = 3
        if 'custom_inp_del' in params:
            self.custom_inp_del = params['custom_inp_del']
        else:
            raise AssertionError('x_switch_sig units need a custom_inp_del parameter')
        if 'des_out_w_abs_sum' in params:
            self.des_out_w_abs_sum = params['des_out_w_abs_sum']
        else:
            self.des_out_w_abs_sum = 1.
        if 'refr_per' in params: self.refr_per = params['refr_per']
        else: self.refr_per = 1.
        if 'sw_thresh' in params: self.sw_thresh = params['sw_thresh']
        else: self.sw_thresh = .5
        if 'sw_len' in params: self.sw_len = params['sw_len']
        else: self.sw_len = .2
        if 'switch' in params: self.switch = params['switch']
        else: self.switch = True
        self.inp = 0. # port 0 scaled input minus extra threshold
        self.xtra_thr = 0. # extra threshold 
        self.lst = -1. # last switch/update time
        sigmoidal.__init__(self, ID, params, network)
        rga_reqs.__init__(self, params) # add requirements and update functions 
        lpf_sc_inp_sum_mp_reqs.__init__(self, params)
        self.needs_mp_inp_sum = True
        self.syn_needs.update([syn_reqs.lpf_slow_sc_inp_sum_mp,
                               syn_reqs.lpf_fast_sc_inp_sum_mp,
                               syn_reqs.lpf_mid_sc_inp_sum_mp]) 

    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. 
            The input (self.inp) is only updated when the switch conditions are
            True.
        """
        mp_i = self.get_mp_inputs(t)
        mp_w = self.get_mp_weights(t)
        is0 = (mp_i[0]*mp_w[0]).sum()
        is2p = self.lpf_fast_sc_inp_sum_mp[2] - self.lpf_mid_sc_inp_sum_mp[2]
        if  abs(is2p) > self.sw_thresh and t - self.lst > self.refr_per:
            self.lst = t
            if self.switch:
                self.xtra_thr = (is0 + self.sw_len * np.sign(is0 -
                                 self.thresh - self.xtra_thr) - self.thresh)
            else:
                self.xtra_thr = self.lpf_slow_sc_inp_sum_mp[0]
            self.inp = is0  - self.xtra_thr
        return ( self.f(self.inp) - y[0] ) * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        is2p = self.lpf_fast_sc_inp_sum_mp[2] - self.lpf_mid_sc_inp_sum_mp[2]
        if (abs(is2p) > self.sw_thresh and 
            self.net.sim_time - self.lst > self.refr_per):
            self.lst = self.net.sim_time
            if self.switch:
                self.xtra_thr = (self.mp_inp_sum[0][s] + self.sw_len * 
                                 np.sign(self.mp_inp_sum[0][s] - self.thresh -
                                         self.xtra_thr) -self.thresh )
            else:
                self.xtra_thr = self.lpf_slow_sc_inp_sum_mp[0]
            self.inp = self.mp_inp_sum[0][s] - self.xtra_thr
        return ( self.f(self.inp) - y ) * self.rtau


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
            du = -20.*y[0] # rushing towards 0
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

        This unit has the required parameters to use euler_maru integration if
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
                    'integ_decay' : change rate of the integral component.
                    Using rga synapses brings an extra required parameter:
                    'custom_inp_del' : an integer indicating the delay that the rga
                                  learning rule uses for the lateral port inputs. 
                                  The delay is in units of min_delay steps. 
                    Using rga_diff synapses brings two extra required parameters:
                    'custom_inp_del' : the shortest of the two delays.
                    'custom_inp_del2': the longest of the two delays.
                    When custom_inp_del2 is not in the params dictionary no
                    exception is thrown, but instead it is assumed that
                    rga_diff synapses are not used, and the custom_del_diff
                    attribute will not be generated.
                    Using slide_rga_diff synapses, in addition to the two delay
                    values, requires maximum and minimum delay modifiers:
                    'del_mod_max': maximum delay modifier.
                    'del_mod_min': minimum delay modifier.
                    The two delay modifiers are expressed as number of time
                    steps, as is the case for the custom delays.
                OPTIONAL PARAMETERS
                    'des_out_w_abs_sum' : desired sum of absolute weight values
                                          for the outgoing connections.
                                          Default is 1.
                    'out_norm_type' : a synapse type's integer value. If included,
                                 the sum of absolute weights for outgoing
                                 connections will only consider synapses of
                                 that type. For example, you may set it as:
                                 {...,
                                 'out_norm_type' : synapse_types.gated_rga_diff.value}
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
        if 'custom_inp_del2' in params:
            if params['custom_inp_del2'] > params['custom_inp_del']:
                self.custom_inp_del2 = params['custom_inp_del2']
                self.custom_del_diff = self.custom_inp_del2-self.custom_inp_del
            else:
                raise ValueError('custom_inp_del2 must be larger than ' +
                                 'custom_inp_del')

        sigmoidal.__init__(self, ID, params, network)

        if 'del_mod_max' in params or 'del_mod_min' in params:
            if params['del_mod_max'] < params['del_mod_min']:
                raise ValueError('del_mod_max cannot be smaller than del_mod_min')
            self.del_mod_max = params['del_mod_max']
            self.del_mod_min = params['del_mod_min']
            if self.delay < (self.custom_inp_del2 + 
                             self.del_mod_max)*self.min_delay:
                raise ValueError('The delay in a gated_slide_rga_diff unit is ' +
                                 'smaller than custom_inp_del2 + del_mod_max')
            if 1 > (self.custom_inp_del + self.del_mod_min):
                raise ValueError('custom_inp_del + del_mod_min is too small')
        if 'des_out_w_abs_sum' in params:
            self.des_out_w_abs_sum = params['des_out_w_abs_sum']
        else:
            self.des_out_w_abs_sum = 1.
        if 'adapt_amp' in params:
            self.adapt_amp = params['adapt_amp']
        else:
            self.adapt_amp = 1.
        if 'out_norm_type' in params:
            self.out_norm_type = params['out_norm_type']
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
        # same update rule from other upd_lpf_X methods, 
        # put in a list comprehension
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
        

class inpsel_linear2(unit, acc_sda_reqs, rga_reqs): 
    """ A multiport linear unit that only sums inputs from ports 0 and 1.

        The current implementation only allows positive values (half-wave
        rectification).

        Port 0 is meant to receive error inputs, which should be the main
        drivers of the activity. Port 1 receives lateral inhibition. Port 2 is
        for the ascending motor command inputs. Port 3 is for resetting the slow
        accumulator (acc_slow), used to pause plasticity.
        Inputs from ports 2 and 3 are not used to produce activation in the unit.

        This is meant to be used in the intermediate layer of an input
        selection mechanism, as described in the "A generalization of RGA, and
        Fourier components" note, used in the test2p5.ipynb notebook.

        This model also includes the mp_inputs and mp_weights requirements.
    """
    def __init__(self, ID, params, network):
        """ The class constructor.

        Args:
            ID, params, network: same as the linear class.
            REQUIRED PARAMETERS
            'tau' : time constant of the dynamics.
            'xtra_inp_del' : extra delay in the port 2 (ascending) inputs.
            OPTIONAL PARAMETERS
            use_acc_slow : Whether to include the acc_slow requirement. Default
                           is 'False'.
        """
        self.tau = params['tau']  # the time constant of the dynamics
        if 'n_ports' in params and params['n_ports'] != 4:
            raise AssertionError('inpsel_linear2 units must have n_ports=4')
        else:
            params['n_ports'] = 4
        # TODO: its probably possible to inplement this withouth xtra delays.
        self.xtra_inp_del = params['xtra_inp_del']
        unit.__init__(self, ID, params, network)
        self.syn_needs.update([syn_reqs.acc_slow,
                               syn_reqs.mp_inputs, 
                               syn_reqs.mp_weights ])
                               #syn_reqs.sc_inp_sum_mp]) # so synapses don't call?
        self.needs_mp_inp_sum = True # in case we flatten
        params['acc_slow_port'] = 3 # so inputs at port 3 reset acc_slow
        acc_sda_reqs.__init__(self, params)
        # The following is for the xtra_del_inp_deriv_mp requirement used by the
        # gated_diff_inp_corr synapse.
        params['xd_inp_deriv_p'] = [2] 
        rga_reqs.__init__(self, params)
       
    def derivatives(self, y, t):
        """ Derivatives of the state variables at the given time. 
        
            Args: 
                y : a 1-element array or list with the current firing rate.
                t: time when the derivative is evaluated.
        """
        I = sum([(w*i).sum() for i,w in zip(self.get_mp_inputs(t)[:2],
                                            self.get_mp_weights(t)[:2])])
        return (I - y[0]) / self.tau #if y[0]>0 else 0.01

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        I = self.mp_inp_sum[0][s] + self.mp_inp_sum[1][s] 
        return (I - y) / self.tau #if y > 0 else 0.01


class bell_shaped_1D(unit):
    """ A unit that responds maximally to a given value of the input sum.
    
        The constructor is given a 'center' parameter. The output y of the unit
        will evolve with dynamics:
        
        tau y' = exp(-b(I - center)^2) - y,
        
        where 'b' is a parameter for the sharpness of the tuning, and 'I' is the
        scaled input sum.

        If the parameter "wrap" is True, then the distance between 'I' and
        'center' will be calculated assuming that both are in the [0,1]
        interval, and that the boundary is periodic. For example, the distance
        between 0.1 and 0.9 is 0.2 .
    """
    def __init__(self, ID, params, network):
        """ The class constructor.

            Args:
                ID, params, network: same as the unit class.
                REQUIRED PARAMETERS
                'tau' : time constant of the dynamics.
                'center' : input value causing maximum activation.
                'b' : sharpness parameter.
                OPTIONAL PARAMETER
                'wrap' : Whether distance is periodic. Default=False.
        """
        unit.__init__(self, ID, params, network)
        self.tau = params['tau']  # the time constant of the dynamics
        self.rtau = 1./self.tau
        self.center = params['center']
        self.b = params['b']
        if 'wrap' in params: self.wrap = params['wrap']
        else: self.wrap = False

    def derivatives(self, y, t):
        """ Derivative of y at time t for bell_shaped_1D.

            Args:
                y : a 1-element array or list with the current firing rate.
                t: time when the derivative is evaluated.
        """
        if self.wrap:
            I = self.get_input_sum(t)
            diff = min(abs(I-self.center), 
                       1.-max(I,self.center) + min(I,self.center))
        else:
            diff = self.get_input_sum(t) - self.center
        return self.rtau * (np.exp(-self.b*diff*diff) - y[0])
            
    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        if self.wrap:
            I = self.inp_sum[s]
            diff = min(abs(I-self.center), 
                       1.-max(I,self.center) + min(I,self.center))
        else:
            diff = self.inp_sum[s] - self.center
        return self.rtau * (np.exp(-self.b*diff*diff) - y)


class td_sigmo(sigmoidal, lpf_sc_inp_sum_mp_reqs):
    """ A sigmoidal unit used to implement a value function with the TD rule.
    
        This unit is meant to receive a reward at port 1, and state inputs at
        port 0. The state inputs should use the TD_synapse, and the reward a
        static synapse. 

        To maintain the dynamic range of the unit, the inputs are substracted
        their slow low-pass filtered version. It is recommended to give a large
        'tau_slow' value for this purpose.
    """
    def __init__(self, ID, params, network):
        """ The class constructor.

            Args:
                ID, params, network: same as the unit class.
                REQUIRED PARAMETERS
                'tau' : time constant of the dynamics.
                'slope' : Slope of the sigmoidal function.
                'thresh' : Threshold of the sigmoidal function.
                'delta' : time delay for updates (in seconds)
                'tau_slow' : time constant for slow low-pass filter.

        """
        self.delta = params['delta']
        self.del_steps = int(np.round(self.delta / network.min_delay))
        if 'n_ports' in params and params['n_ports'] != 2:
            raise AssertionError('td_sigmo uses n_ports=2')
        else:
            params['n_ports'] = 2
        if not 'tau_slow' in params:
            raise AssertionError('params for td_sigmo should include tau_slow')
        sigmoidal.__init__(self, ID, params, network)
        self.syn_needs.update([syn_reqs.mp_inputs,
                               syn_reqs.mp_weights,
                               syn_reqs.sc_inp_sum_mp,
                               syn_reqs.lpf_slow_sc_inp_sum_mp])
        self.needs_mp_inp_sum = True

    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. """
        p0_inps = np.array([self.net.act[self.ID][idx](t - 
                  self.net.delays[self.ID][idx]) for idx in self.port_idx[0]])
        p0_ws = np.array([self.net.syns[self.ID][idx].w for idx in
                self.port_idx[0]])
        return ( self.f((p0_inps*p0_ws).sum() - self.lpf_slow_sc_inp_sum_mp[0])
                 - y[0] )  * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        return ( self.f(self.mp_inp_sum[0][s] - self.lpf_slow_sc_inp_sum_mp[0])
                 - y ) * self.rtau


class layer_dist(sigmoidal):
    """ A unit to obtain a distance in the activities of two layers.

        The need for this unit is explained deep in the November 20, 2020 
        notes.

        If you have layers A and B with the same structure (e.g. for each unit
        in A there is a corresponding unit in B), the output of this unit
        reflects how similar their activities are. To this end the units in A
        connect to port 0, and the units in B to port 1 in the same order. The
        layer_dist unit updates using:

        \tau y' = f(\sum_i |A_i - B_i|) - y
        where f is the sigmoidal function.

        Input weights are ignored for efficiency.
    """
    def __init__(self, ID, params, network):
        """ The class constructor.

            Args:
                ID, params, network: same as the unit class.
                REQUIRED PARAMETERS
                'tau' : time constant of the dynamics.
                'slope' : Slope of the sigmoidal function.
                'thresh' : Threshold of the sigmoidal function.
        """
        if 'n_ports' in params and params['n_ports'] != 2:
            raise AssertionError('layer_dist uses n_ports=2')
        else:
            params['n_ports'] = 2
        sigmoidal.__init__(self, ID, params, network)
        #self.syn_needs.update([syn_reqs.mp_inputs])
        self.needs_mp_inp_sum = True

    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. """
        mp_i = self.get_mp_inputs(t)
        return (self.f(np.abs(mp_i[0] - mp_i[1]).sum()) - y[0]) * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        msi0 = self.mp_step_inps[0][:,s]
        msi1 = self.mp_step_inps[1][:,s]
        return (self.f(np.abs(msi0 - msi1).sum()) - y) * self.rtau


class linear_mplex(unit):
    """ Linear unit that multiplexes inputs at port 0 and 1.

        This means that when inputs at port 2 are larger than 0.5 the output
        will be the scaled input sum at port 1, and when inputs at port 2 are
        smaller than 0.5 the output is the scaled input sum at port 0.
    """
    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the 'unit' parent class.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS
                'tau' : Time constant of the update dynamics.

        Raises:
            AssertionError.

        """
        if 'n_ports' in params and params['n_ports'] != 3:
            raise AssertionError('linear_mplex units use n_ports=3')
        else:
            params['n_ports'] = 3
        unit.__init__(self, ID, params, network)
        self.tau = params['tau']  # the time constant of the dynamics
        self.rtau = 1/self.tau   # because you always use 1/tau instead of tau
        self.needs_mp_inp_sum = True
        
    def derivatives(self, y, t):
        """ Derivatives of the state variables at time t. 
        
            Args: 
                y : a 1-element array or list with the current firing rate.
                t: time when the derivative is evaluated.
        """
        mp_i = self.get_mp_inputs(t)
        mp_w = self.get_mp_weights(t)
        if (mp_i[2]*mp_w[2]).sum() < 0.5:
            sel_port = 0
        else:
            sel_port = 1
        return ( (mp_i[sel_port]*mp_w[sel_port]).sum() - y[0] ) * self.rtau

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        if self.mp_step_inps[2][:,s] < 0.5:
            sel_port = 0
        else:
            sel_port = 1
        return ( self.mp_inp_sum[sel_port][s] - y ) * self.rtau


class v_net(sigmoidal, lpf_sc_inp_sum_mp_reqs, rga_reqs):
    """ A unit to replace the L,V units, and L__V synapses in rl5E_lite.ipynb
    
        This unit is meant to receive an angle input at port 0, a desired angle
        input at port 1, and a reward at port 2. The output is what would be
        expected from the V unit (of the td_sigmo class) in rl5E. Weights from
        all ports are ignored. The port 0 input is expected in the (-pi,pi)
        range, and port 1 inputs are expected to be in the (0,2*pi) range.

        The dynamics are 101-dimensional. The last 100 state variables
        correspond to the L__V synaptic weights. The first variable is for the
        activity. The number 100 corresponds to the square of the N parameter
        given to the constructor, which may be changed.
    """
    def __init__(self, ID, params, network):
        """ The class constructor.

            Args:
                ID, params, network: same as the unit class
                REQUIRED PARAMETERS
                'tau' : time constant of the dynamics
                'slope' : Slope of the sigmoidal function.
                'thresh' : Threshold of the sigmoidal function
                'delta' : time delay for updates (in seconds)
                #'tau_slow' : time constant for slow low-pass filter
                'td_lrate' : learning rate for the td rule
                'td_gamma' : discount factor for the td rule
                OPTIONAL PARAMETERS
                'N' : number of units to represent an angle. Default=10
                'w_sum' : sum of synaptic weights. Default=10
                'normalize' : if True, weights add to w_sum. Default=False
                'L_wid' : controls the width of tuning in the L units. Default=N
                #'R_wid' : how rapidly reward decays with error. Default=2

        """
        if 'N' in params: self.N = params['N']
        else: self.N = 10
        params['multidim'] = True
        if len(params['init_val']) != self.N*self.N + 1:
            raise ValueError("Initial values for the v_net must " +
                             "consist of a (N*N + 1)-element array.")
        if 'n_ports' in params:
            if params['n_ports'] != 3:
                raise ValueError("v_net units require 3 input ports.")
        else:
            params['n_ports'] = 3

        self.delta = params['delta']
        self.dl_st = int(self.delta / network.min_delay)
        self.del_steps = int(np.round(self.delta / network.min_delay))
        self.custom_inp_del = self.del_steps
        #if not 'tau_slow' in params:
        #    raise AssertionError('params for v_net should include tau_slow')
        sigmoidal.__init__(self, ID, params, network)
        self.td_lrate = params['td_lrate']
        self.td_gamma = params['td_gamma']
        self.eff_gamma = self.td_gamma**(network.min_delay / self.delta)
        self.alpha = self.td_lrate * self.min_delay
        if 'w_sum' in params: self.w_sum = params['w_sum']
        else: self.w_sum = 10.
        if 'normalize' in params: self.normalize = params['normalize']
        else: self.normalize = False
        if 'L_wid' in params: self.L_wid = params['L_wid']
        else: self.L_wid = self.N
        #if 'R_wid' in params: self.R_wid = params['R_wid']
        #else: self.R_wid = 2.
        # initializing the "centers" array
        bit = 1./self.N
        self.centers = []
        for row in range(self.N):
            for col in range(self.N):
                self.centers.append(np.array([bit*(row+0.5), bit*(col+0.5)]))
        self.centers = 2. * np.pi * np.array(self.centers)
        self.syn_needs.update([syn_reqs.mp_inputs,
                               syn_reqs.lpf_slow,
                               syn_reqs.del_inp_mp])
                               #syn_reqs.mp_weights,
                               #syn_reqs.lpf_slow_sc_inp_sum_mp])
        self.z = np.zeros_like(params['init_val']) # to store derivatives
        #if self.normalize:
        #    self.z[1:] = 2.* self.td_lrate * (self.w_sum - self.z[1:].sum())
        #self.R = 0.1 # reward value

    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. 
        
            Args:
                y : numpy array with state variables.
                    y[0] : unit's activity
                    y[1:] : L__V weights
        """
        #if t - self.last_time >= self.min_delay:
        del_qc = self.del_inp_mp[0][0] # delayed current angle in (-pi,pi)
        del_qd = self.del_inp_mp[1][0] # delayed desired angle in (0,2*pi)
        del_qc = del_qc if del_qc > 0. else 2.*np.pi + del_qc # now in (0,2*pi)
        del_state = np.array([del_qd, del_qc])
        #del_L_inps = np.linalg.norm(self.centers - del_state, axis=1)
        del_L_inps = self.dists(del_state)
        del_L_out = np.exp(-self.L_wid*(del_L_inps*del_L_inps))
        #sf_id = self.net.syns[self.ID][self.port_idx[0][0]].preID
        #sp_id = self.net.syns[self.ID][self.port_idx[1][0]].preID
        #sf_unit = self.net.units[sf_id] # assuming single unit
        #sp_unit = self.net.units[sp_id] # assuming single unit
        #del_state = np.array([sp_unit.act_buff[-1-self.del_steps],
                              #sf_unit.act_buff[-1-self.del_steps]])
        qc = self.mp_inputs[0][0] # current angle in (-pi, pi)
        qc = qc if qc > 0. else 2.*np.pi + qc  # from (-pi,pi) to (0,2*pi)
        state = np.array([self.mp_inputs[1][0], qc])

        #state = np.array([self.mp_inputs[1][0], self.mp_inputs[0][0]])
        #L_inps = np.linalg.norm(self.centers - state, axis=1)
        L_inps = self.dists(state) # periodic boundaries
        L_out = np.exp(-self.L_wid*(L_inps*L_inps))

        self.L_out_copy = L_out # for debugging purposes
        # using a variable threshold
        lpfs = self.get_lpf_slow(0)
        slow_inp = self.thresh + (np.log(lpfs) - np.log(1.-lpfs))/self.slope

        #del_V_out = self.f((del_L_out * y[1:]).sum()) # sanity check
        del_V_out = self.act_buff[-1-self.del_steps]
        V_out = self.f(((L_out-slow_inp) * y[1:]).sum())
        #V_out = self.f((L_out * y[1:]).sum()) - lpfs
        #print(L_out.sum())
        #r = np.exp(-self.R_wid * abs(state[1] - state[0]))
        #self.R += 0.1*(r - self.R) # so reward is not instantaneous
        #R = self.mp_inputs[2] # not entirely correct
        # closer to the integral of R (a single trapezoid)
        r_id = self.net.syns[self.ID][self.port_idx[2][0]].preID
        R = self.delta*(self.net.units[r_id].get_act(t) +
             self.net.units[r_id].get_act(t-self.delta)) / 2. 

        self.z[0] = (V_out - y[0]) * self.rtau
        self.z[1:] = self.alpha * (R + self.eff_gamma*y[0] -
                                   del_V_out) * del_L_out
        if self.normalize:
            # additive
            #self.z[1:] += 0.05*np.sign(self.w_sum - np.abs(y[1:]).sum())
            # multiplicative
            self.z[1:] += (y[1:] * (self.w_sum / max(1e-10, 
                           np.abs(y[1:]).sum())) - y[1:])
        return self.z

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        raise NotImplementedError("v_net not available for flat networks.")
        #return ( self.f(self.mp_inp_sum[0][s] - self.lpf_slow_sc_inp_sum_mp[0])
        #         - y ) * self.rtau

    def dists(self, state):
        """ Given a state, provide periodic distances to centers.

            Args:
                state: 2-element numpy array
            Returns:
                100-element numpy array with distances from state to each one of
                the elements in 'centers', using periodic boundaries.
        """
        mins = np.minimum(self.centers, state)
        maxs = np.maximum(self.centers, state)
        diff = self.centers - state
        return np.linalg.norm(np.minimum(2.*np.pi-maxs + mins, 
                                         maxs - mins), axis=1)


class x_net(sigmoidal, lpf_sc_inp_sum_mp_reqs, rga_reqs):
    """ A unit to replace the L,X units, and L__X synapses in rl5E_lite.ipynb
    
        This unit is meant to receive a SF input at port 0, a SP input at port
        1, and a value input at port 2. The output is what would be expected
        from the X unit (of the x_switch_sig class) in rl5E. Weights from SF
        and SP inputs are ignored.

        The dynamics are 101-dimensional. The first variable is for the
        activity of X.The next 100 state variables correspond to the L__X
        synaptic weights.  The number 100 corresponds to the square of the N
        parameter given to the constructor (Default=10), which may be changed.

        The plasticity rule is modulated so changes are reduced when the value
        of SP has recently changed.
    """
    def __init__(self, ID, params, network):
        """ The class constructor.

            Args:
                ID, params, network: same as the unit class
                REQUIRED PARAMETERS
                'tau' : time constant of the dynamics
                'slope' : Slope of the sigmoidal function.
                'thresh' : Threshold of the sigmoidal function
                'del_steps' : an integer indicating the delay that the
                            learning rule uses for the lateral port inputs. 
                            The delay is in units of min_delay steps. 
                'lrate' : learning rate of the diff_rm_hebbian synapses
                OPTIONAL PARAMETERS
                'N' : number of units to represent an angle. Default=10
                'w_sum' : sum of synaptic weights. Default=10
                'normalize' : if True, weights add to w_sum. Default=False
                'L_wid' : controls the width of tuning in the L units. Default=N
                'refr_per' : minimum time between switches. Defaul=1 sec.
                'sw_thresh' : If port 1 derivative larger than this, switch,
                              or update (if "switch" is False). Default=0.5.
                'sw_len' : Distance from L input sum after switch. Default=0.2 .
                'switch' : Whether port 1 causes a switch. Default is True.
                'beta' : changes how fast plasticity recovers after SP changes


        """
        if 'N' in params: self.N = params['N']
        else: self.N = 10
        params['multidim'] = True
        if len(params['init_val']) != self.N*self.N + 1:
            raise ValueError("Initial values for the v_net must " +
                             "consist of a (N*N + 1)-element array.")
        if 'n_ports' in params:
            if params['n_ports'] != 3:
                raise ValueError("x_net units require 3 input ports.")
        else:
            params['n_ports'] = 3
        if 'del_steps' in params: self.del_steps = params['del_steps']
        else: raise AssertionError('x_net units need a del_steps parameter')
        self.custom_inp_del = self.del_steps
        #if not 'tau_slow' in params:
        #    raise AssertionError('params for x_net should include tau_slow')
        sigmoidal.__init__(self, ID, params, network)
        self.lrate = params['lrate']
        self.alpha = self.lrate * self.min_delay
        if 'w_sum' in params: self.w_sum = params['w_sum']
        else: self.w_sum = 10.
        if 'normalize' in params: self.normalize = params['normalize']
        else: self.normalize = False
        if 'L_wid' in params: self.L_wid = params['L_wid']
        else: self.L_wid = self.N
        if 'refr_per' in params: self.refr_per = params['refr_per']
        else: self.refr_per = 1.
        if 'sw_thresh' in params: self.sw_thresh = params['sw_thresh']
        else: self.sw_thresh = .5
        if 'sw_len' in params: self.sw_len = params['sw_len']
        else: self.sw_len = .2
        if 'switch' in params: self.switch = params['switch']
        else: self.switch = True
        if 'beta' in params: self.beta = params['beta']
        else: self.beta = 1.
        # initializing the "centers" array
        bit = 1./self.N
        self.centers = []
        for row in range(self.N):
            for col in range(self.N):
                self.centers.append(np.array([bit*(row+0.5), bit*(col+0.5)]))
        self.centers = 2. * np.pi * np.array(self.centers)
        self.syn_needs.update([syn_reqs.mp_inputs,
                               syn_reqs.mp_weights,
                               syn_reqs.del_inp_mp,
                               syn_reqs.lpf_slow,
                               syn_reqs.sc_inp_sum_mp,
                               syn_reqs.lpf_fast_sc_inp_sum_mp,
                               syn_reqs.lpf_mid_sc_inp_sum_mp ])
                               #syn_reqs.lpf_slow_sc_inp_sum_mp])
        self.z = np.zeros_like(params['init_val']) # to store derivatives
        #if self.normalize:
        #    self.z[1:] = self.lrate * (self.w_sum - np.abs(self.z[1:]).sum())
        self.xtra_thr = 0. # extra threshold
        self.lst = -1. # last switch/update time
        self.inp = 0. # L input minus extra threshold
        self.wlmod = 1. # modulates the learning rule after desired angle changes

    def derivatives(self, y, t):
        """ Return the derivative of the activity at time t. 
        
            Args:
                y : numpy array with state variables.
                    y[0] : unit's activity
                    y[1:] : L__V weights
        """
        #if t - self.last_time >= self.min_delay:
        # obtain L inputs
        del_qc = self.del_inp_mp[0][0] # delayed current angle in (0,2*pi)
        del_qd = self.del_inp_mp[1][0] # delayed desired angle in (0,2*pi)
        del_qc = del_qc if del_qc > 0. else 2.*np.pi + del_qc # now in (0,2*pi)
        del_state = np.array([del_qd, del_qc])
        #del_L_inps = np.linalg.norm(self.centers - del_state, axis=1)
        del_L_inps = self.dists(del_state) # periodic boundaries
        del_L_out = np.exp(-self.L_wid*(del_L_inps*del_L_inps))
        #sf_id = self.net.syns[self.ID][self.port_idx[0][0]].preID
        #sp_id = self.net.syns[self.ID][self.port_idx[1][0]].preID
        #sf_unit = self.net.units[sf_id] # assuming single unit
        #sp_unit = self.net.units[sp_id] # assuming single unit
            # delayed and current state
        #del_state = np.array([sp_unit.act_buff[-1-self.del_steps],
        #                      sf_unit.act_buff[-1-self.del_steps]])
        #qc = self.del_inp_mp[0][0] # current angle in (2,2*pi)
        #qd = self.del_inp_mp[1][0] # desired angle in (0, 2*pi)
        #state = np.array([qd, qc])
            # delayed and current L outputs
        qc = self.mp_inputs[0][0]
        qc = qc if qc > 0. else 2.*np.pi + qc  # from (-pi,pi) to (0,2*pi)
        state = np.array([self.mp_inputs[1][0], qc])
        #state = np.array([self.mp_inputs[1][0], self.mp_inputs[0][0]])
        #L_inps = np.linalg.norm(self.centers - state, axis=1)
        L_inps = self.dists(state) # periodic boundaries
        L_out = np.exp(-self.L_wid*(L_inps*L_inps)) # output from L
        self.L_out_copy = L_out
        # input sums and input derivatives
        is0 = (L_out * y[1:]).sum() # scaled input sum from L
        is1p = self.lpf_fast_sc_inp_sum_mp[1] - self.lpf_mid_sc_inp_sum_mp[1]
        vp = self.lpf_fast_sc_inp_sum_mp[2] - self.lpf_mid_sc_inp_sum_mp[2]
        # update activity
        if  (abs(is1p) > self.sw_thresh and t - self.lst > self.refr_per and
                abs(t-self.net.sim_time) <= self.net.min_delay + 1e-16):
            #print('t=%.2f' % (t), end=', ')
            #print('lst=' + str(self.lst))
            #print('is1p=' + str(is1p))
            self.lst = t
            if self.switch:
                self.xtra_thr = (is0 + self.sw_len * np.sign(is0 -
                                 self.thresh - self.xtra_thr) - self.thresh)
            else:
                # technically incorrect, but probably good enough:
                #slow_state = np.array([self.lpf_slow_sc_inp_sum_mp[1],
                #                       self.lpf_slow_sc_inp_sum_mp[0]])
                #slow_L_inps = np.linalg.norm(self.centers - slow_state, 
                #                             axis=1)
                #slow_L_out = np.exp(-self.L_wid*(slow_L_inps*slow_L_inps))
                #self.xtra_thr = (slow_L_out * y[1:]).sum() 
                lpfs = self.get_lpf_slow(0)
                self.xtra_thr = self.thresh + (np.log(lpfs) - 
                                np.log(1.-lpfs))/self.slope
            self.inp = is0  - self.xtra_thr
        #self.z[0] = (self.f(self.inp)-self.get_lpf_slow(0)+0.5-y[0]) * self.rtau
        self.z[0] = (self.f(self.inp) - y[0]) * self.rtau
        # update weights
        #self.wlmod = 1. - np.exp(-self.beta * max(t - self.lst, 0.))
        #if t < self.lst:
        #    print('t :' + str(t))
        #    print('lst :' + str(self.lst))
        #    print('is1p :' + str(is1p))
            
        self.wlmod = 1. - np.exp(-self.beta * (t - self.lst))
        self.z[1:] = self.alpha * self.wlmod * (vp * (del_L_out - 
                                  np.mean(del_L_out)) * (y[0] - 0.5))
        if self.normalize:
            #self.z[1:] *= 1. + (0.1 * self.alpha * np.sign(self.w_sum - 
            #              np.abs(y[1:]).sum()) * self.z[1:] * y[1:])
            self.z[1:] += (y[1:] * (self.w_sum / max(1e-10, 
                           np.abs(y[1:]).sum())) - y[1:])
        return self.z 

    def dt_fun(self, y, s):
        """ The derivatives function used when the network is flat. """
        raise NotImplementedError("x_net not available for flat networks.")
        #return ( self.f(self.mp_inp_sum[0][s] - self.lpf_slow_sc_inp_sum_mp[0])
        #         - y ) * self.rtau

    def dists(self, state):
        """ Given a state, provide periodic distances to centers.

            Args:
                state: 2-element numpy array
            Returns:
                100-element numpy array with distances from state to each one of
                the elements in 'centers', using periodic boundaries.
        """
        mins = np.minimum(self.centers, state)
        maxs = np.maximum(self.centers, state)
        diff = self.centers - state
        return np.linalg.norm(np.minimum(2.*np.pi-maxs + mins, 
                                         maxs - mins), axis=1)


