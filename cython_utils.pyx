"""
cython_utils.pyx

This file has cython functions used in draculab.
"""
import numpy as np
cimport numpy as np
cimport cython

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
