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
        params['multidim'] = True
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
        

    def upd_lpf_slow_mp_inp_sum(self, time):
        """ Update the slow LPF'd scaled sum of inputs at individual ports, returning them in a list. """
        assert time >= self.last_time, ['Unit ' + str(self.ID) + 
                                        ' lpf_slow_mp_inp_sum updated backwards in time']
        inputs = self.mp_inputs    # updated because mp_inputs is a requirement variable
        weights = self.get_mp_weights(time)
        dots = [ np.dot(i, w) for i, w in zip(inputs, weights) ]
        # same update rule from other upd_lpf_X methods above, put in a list comprehension
        self.lpf_slow_mp_inp_sum = [dots[i] + (self.lpf_slow_mp_inp_sum[i] - dots[i]) * 
                                   np.exp( (self.last_time-time)/self.tau_slow ) 
                                   for i in range(self.n_ports)]


class test_oscillator(unit):
    """ 
    A model used to test the frameowork for multidimensional units.

    The model consists of the equations for a sinusoidal:
    y' = w*x
    x' = -w*y
    with y being the unit's output, and w the frequency of oscillation. w is the
    reciprocal of the unit's time constant tau.
    Inputs are currently ignored.
    """

    def __init__(self, ID, params, network):
        """ The unit's constructor.

        Args:
            ID, params, network, same as in unit.__init__, butthe 'init_val' parameter
            needs to be a 1D array with two elements, corresponding to the initial values
            of y and x.
            In addition, params should have the following entries.
            REQUIRED PARAMETERS
            'tau' : Time constant of the update dynamics. Also, reciprocal of the 
                    oscillator's frequency.

        """
        unit.__init__(self, ID, params, network)
        self.tau = params['tau']
        self.w = 1/self.tau

    def derivatives(self, y, t):
        """ Implements the ODE of the oscillator.

        Args:
            y : 1D, 2-element array with the values of the state variables.
            t : time when the derivative is evaluated (not used).
        Returns:
            numpy array with [w*y[1], -w*y[0]]
        """
        return np.array([self.w*y[1], -self.w*y[0]])

    def dt_fun(self, y, s):
        """ The derivatives function used for flat networks. 
        
        Args:
            y : 1D, 2-element array with the values of the state variables.
            s: index to the inp_sum array (not used).
        Returns:
            Numpy array with 2 elements.
            
        """
        return np.array([self.w*y[1], -self.w*y[0]])



