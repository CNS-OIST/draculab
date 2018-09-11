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

