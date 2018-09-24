"""
cython_utils.pyx

This file has cython functions used in draculab.
"""
import numpy as np
cimport numpy as np
cimport cython
#import scipy
from scipy.integrate import solve_ivp 
#from scipy.integrate import odeint

# Since I have explicit bounds check, I can afford to throw away these checks:
@cython.boundscheck(False)
@cython.wraparound(False)
def cython_get_act2(float time, float t0, float t1, np.ndarray times, int b_size, np.ndarray buff):
    """
        The Cython version of the second implementation of unit.get_act()

        In the namespace of the unit.get_act method, the arguments are:
            time : time
            t0 : self.times[0]
            t1 : self.times[-1]
            times : self.times
            b_size : self.buff_size
            buff : self.buffer
    """
    cdef float t = min( max(time, t0), t1 ) # clipping 'time'
    cdef float frac = (t-t0)/(t1-t0)
    cdef int base = <int>(np.floor(frac*(b_size-1))) # biggest index s.t. times[index] <= time
    cdef frac2 = ( t-times[base] ) / ( times[min(base+1,b_size-1)] - times[base] + 1e-8 )
    return buff[base] + frac2 * ( buff[min(base+1,b_size-1)] - buff[base] )


@cython.boundscheck(False)
@cython.wraparound(False)
def cython_get_act3(float time, float times0, float time_bit, int b_size, np.ndarray buff):
    """
        The Cython version of the third implementation of unit.get_act()

        In the namespace of the unit.get_act method, the arguments are:
            time : time
            times0 : self.times[0]
            time_bit : self.time_bit
            b_size : self.buff_size
            buff : self.buffer
    """
    cdef float t = time - times0
    cdef int base = <int>(t // time_bit)
    cdef float rem = t % time_bit
    base = max( 0, min( base, b_size-2 ) )
    cdef float frac2 = rem / time_bit
    return buff[base] + frac2 * ( buff[base+1] - buff[base] )


def cython_sig(float thresh, float slope, float arg):
    """ The sigmoidal function used by most of draculab's units.

        It is called by the sigmoidal units' f function.
    """
    return 1. / (1. + np.exp(-slope*(arg - thresh)))


def cython_update(self, float time):
    """ The Cython implementation of unit.update. """
    # If the unit object (self) is being passed as more than just a reference this could
    # slow down things
    new_times = self.times[-1] + self.times_grid
    self.times = np.roll(self.times, -self.min_buff_size)
    self.times[self.offset:] = new_times[1:]

    """
    new_buff = odeint(self.derivatives, [self.buffer[-1]], new_times, rtol=self.rtol, atol=self.atol) 
    self.buffer = np.roll(self.buffer, -self.min_buff_size)
    self.buffer[self.offset:] = new_buff[1:,0] 
    """
    solution = solve_ivp(self.derivatives, (new_times[0], new_times[-1]), [self.buffer[-1]], method='LSODA',
                                            t_eval=new_times, rtol=self.rtol, atol=self.atol)
    self.buffer = np.roll(self.buffer, -self.min_buff_size)
    self.buffer[self.offset:] = solution.y[0,1:]
    #"""

    self.pre_syn_update(time) # Update any variables needed for the synapse to update.

    for pre in self.net.syns[self.ID]:
        pre.update(time)

    self.last_time = time # last_time is used to update some pre_syn_update values


def euler_int(derivatives, float x0, float t0, int n_steps, float dt):
    """ A simple implementation of the forward Euler integration method.
    
        Args:
            derivatives: The function derivatives(y, t) returns 
                         the derivative of the firing rate at time t given that
                         the current rate is y.
            x0: initial state
            t0: initial time
            n_steps: number of integration steps
            dt: integration step size
    """
    x = np.zeros(n_steps, dtype=float)
    x[0] = x0
    cdef float t = t0
    for step in range(1, n_steps, 1):
        x[step] = x[step-1] + dt * derivatives([x[step-1]], t)
        t = t + dt
    return x


def euler_maruyama_int(derivatives, float x0, float t0, int n_steps, 
                       float dt, float mu, float sigma):
    """ The Euler-Maruyama method for stochastic differential equations.

        This solver turns the neural model into a Langevin equation by adding white noise.

        Args:
            derivatives: The function derivatives(y, t) returns 
                         the derivative of the firing rate at time t given that
                         the current rate is y.
            x0: initial state
            t0: initial time
            n_steps: number of integration steps
            dt: integration step size
            mu: mean of the white noise
            sigma: the standard deviation associated to the Wiener process
    """
    x = np.zeros(n_steps, dtype=float)
    x[0] = x0
    cdef float t = t0
    cdef sqrdt = np.sqrt(dt)
    for step in range(1, n_steps, 1):
        x[step] = x[step-1] + ( dt * (derivatives([x[step-1]], t) + mu) 
                            + sigma * np.random.normal(loc=0, scale=sqrdt) )
                            #+ sigma * np.random.normal(loc=0., scale=dt) )
                            #+ sigma * sqrdt * np.random.normal(loc=0., scale=1.) )
        t = t + dt
    return x


def exp_euler_int(derivatives, float x0, float t0, int n_steps, 
                       float dt, float mu, float sigma, float lambd):
    """ A version of the  exponential Euler method for stochastic differential equations.

        This method is used when the derivatives function decomposes into a linear part
        plus a remainder. In the case of the noisy_leaky_linear unit the linear part
        comes from the parameter 'lambd' (with the sign reversed), and the nonlinear
        remainder is the input.

        Args:
            derivatives: The function derivatives(y, t) returns 
                         the derivative of the firing rate at time t given that
                         the current rate is y.
            x0: initial state
            t0: initial time
            n_steps: number of integration steps
            dt: integration step size
            mu: mean of the white noise
            sigma: the standard deviation associated to the Wiener process
            lambd: the coefficient of the linear part
    """
    x = np.zeros(n_steps, dtype=float)
    x[0] = x0
    cdef float t = t0
    cdef float expAt = np.exp(lambd*dt)
    cdef float coeff2 = (expAt - 1.)/lambd
    cdef float coeff3 = np.sqrt( (expAt*expAt - 1.)/(2.*lambd) )
    for step in range(1, n_steps, 1):
        x[step] = ( expAt*x[step-1] + coeff2*derivatives(x[step-1], t)
                + coeff3*sigma*np.random.normal(loc=0., scale=1.) + mu*dt )
        t = t + dt
    return x

 

