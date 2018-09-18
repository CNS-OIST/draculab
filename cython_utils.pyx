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
def cython_get_act(float time, float times0, float time_bit, int b_size, np.ndarray buff):
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
    for step in range(1, n_steps, 1):
        x[step] = x[step-1] + ( dt * derivatives([x[step-1]], t)
                            #+ sigma * np.random.normal(loc=mu, scale=dt) )
                            #+ np.sqrt(dt) * sigma * (mu + np.random.random()) )
                            + mu + sigma * np.random.normal(loc=0., scale=dt) )
        t = t + dt
    return x

 

