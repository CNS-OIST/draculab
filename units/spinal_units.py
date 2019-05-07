"""
spinal_units.py
The units used in the motor control model.
"""
from draculab import unit_types, synapse_types, syn_reqs  # names of models and requirements
from units.units import unit
import numpy as np

class am_pm_oscillator(unit):
    """
    An oscillator with amplitude and phase modulated by inputs.

    The outuput of the unit is the sum of a constant part and a sinusoidal part.
    Both of their amplitudes are modulated by the sum of the inputs at port 0.
    The phase of the sinusoidal part is modulated by the relative phase of the inputs
    at port 1 through a method chosen by the user. 

    The equations of the model look like this:
    u'(t) = Is*(1 + sin(th)) - <Is>_s,
    th' = w + F(u_1, ..., u_n, U)
    where: 
        u = units's activity,
        Is = scaled sum of inputs at port 0,
        th = phase of the oscillation,
        <Is>_s = slow low-pass filtered Is,
        w = intrinsic frequency of the oscillator,
        F = interaction function,
        n = number of inputs at port 1,
        u_j = j-th input at port 1, 
        U = unit's output normalized by its medium LPF'd value.

    There are several choices for the interaction function:
    1) input_sum:
        F = alpha*sin(Ic(t) - U(t)), where Ic is the scaled sum of inputs (each
        multiplied by its synaptic weight), normalized by its medium LPF'd value.
    2) kuramoto:
        F = (1/max(n,1))*sum_j K_j*[f(u_j, u) - f(u, u_j)], where
            K_j = weight of j-th connection at port 1,
            f(a,b) = a*sqrt(1-b**2)
        This aims to reproduce the original Kuramoto model, despite the phases
        not being communicated, but deduced from the received activities.
    3) zero
        F = 0
    """
    def __init__(self, ID, params, network):
        """ The unit constructor.

        Args:
            ID, params, network: same as in the parent's constructor.
            In addition, params should have the following entries.
                REQUIRED PARAMETERS
                'tau' : Time constant of the update dynamics.
                'F' : a string specifying which interaction function to use.
                      Options: 'zero', 'kuramoto', 'input_sum'.
                      The 'input_sum' option requires an extra parameter: 
                      'alpha' : strength of phase interactions
                'tau_mid' : time constant for the medium low-pass filter.
        Raises:
            ValueError

        This unit requires 
        """
        unit.__init__(self, ID, params, network)
        self.tau = params['tau']
        if params['F'] == 'zero':
            self.f = lambda x : 0.
        elif params['F'] == 'input_sum':
            self.f = self.input_sum_f
        elif params['F'] == 'kuramoto':
            self.f = self.kuramoto_f
        else:
            raise ValueError('Wrong specification of the interaction function F')
        
    def odeint_update(self, time):
        """ Advance the dynamics from time to time+min_delay for a multidimensinal unit.

        This update function will replace the values in the activation buffer
        corresponding to the latest "min_delay" time units, introducing "min_buff_size" new values.
        In addition, all the synapses of the unit are updated.
        """
        # the 'time' argument is currently only used to ensure the 'times' buffer is in sync
        #assert (self.times[-1]-time) < 2e-6, 'unit' + str(self.ID) + ': update time is desynchronized'
        new_times = self.times[-1] + self.times_grid
        self.times = np.roll(self.times, -self.min_buff_size)
        self.times[self.offset:] = new_times[1:]
        # odeint also returns the initial condition, so to produce min_buff_size new values
        # we need to provide min_buff_size+1 desired times, starting with 
        # the one for the initial condition
        new_buff = odeint(self.derivatives, [self.buffer[-1]], new_times,
                          rtol=self.rtol, atol=self.atol)
        self.buffer = np.roll(self.buffer, -self.min_buff_size)
        self.buffer[self.offset:] = new_buff[1:,0]
        self.upd_reqs_n_syns(time)

    def get_act(self,time):
        """ Gives you the activity at a previous time 't' (within buffer range).

        This version works for units that store their previous activity values in a buffer.
        Units without buffers (e.g. source units) have their own get_act function.
        """
        ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ## This is the third implementation, written in Cython
        ## self.using_interp1d should be set to False
        return cython_get_act3(time, self.times[0], self.time_bit, 
                               self.buff_size, self.buffer[-1,:])
        ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





    def upd_lpf_slow_mp_inp_sum(self, time):
        """ Update the slow LPF'd scaled sum of inputs at individual ports, returning them in a list. """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_slow_mp_inp_sum updated backwards in time']
        inputs = self.mp_inputs    # updated because mp_inputs is a requirement variable
        weights = self.get_mp_weights(time)
        dots = [ np.dot(i, w) for i, w in zip(inputs, weights) ]
        # same update rule from other upd_lpf_X methods above, put in a list comprehension
        self.lpf_slow_mp_inp_sum = [dots[i] + (self.lpf_slow_mp_inp_sum[i] - dots[i]) * 
                                   np.exp( (self.last_time-time)/self.tau_slow ) for i in range(self.n_ports)]


