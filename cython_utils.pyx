#cython: language_level=3
"""
cython_utils.pyx

This file has cython functions used in draculab.
"""
import numpy as np
cimport numpy as np
cimport cython
cimport cpython.array
#import scipy
#from scipy.integrate import solve_ivp 
from scipy.integrate import odeint

# Since I have explicit bounds check, I can afford to throw away these checks:
@cython.boundscheck(False)
@cython.wraparound(False)
#def cython_get_act2(double time, double t0, double t1, np.array[double, ndim=1] times, 
#                    int b_size, np.ndarray[double, ndim=1] buff):
#def cython_get_act2(double time, double t0, double t1, double[:] times, int b_size, double[:] buff):
def cython_get_act2(float time, float t0, float t1, np.ndarray[np.float64_t, ndim=1] times, int b_size, np.ndarray[np.float64_t, ndim=1] buff):
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
    #cdef double t = min( max(time, t0), t1 ) # clipping 'time'
    cdef float frac = (t-t0)/(t1-t0)
    #cdef double frac = (t-t0)/(t1-t0)
    frac = (t-t0)/(t1-t0)
    cdef int base = <int>(frac*(b_size-1)) # biggest index s.t. times[index] <= time
    #cdef double frac2 = ( t-times[base] ) / ( times[min(base+1,b_size-1)] - times[base] + 1e-8 )
    cdef float frac2 = ( t-times[base] ) / ( times[min(base+1,b_size-1)] - times[base] + 1e-8 )
    return buff[base] + frac2 * ( buff[min(base+1,b_size-1)] - buff[base] )
    """
    cdef int i = 0

    if time > times[-2]:
        i = b_size - 2
    else:
        while time > times[i+1]:
            i += 1
    return buff[i] + (times[i+1] - time) * (buff[i+1] - buff[i])/(times[i+1]-times[i])
    """


@cython.boundscheck(False)
@cython.wraparound(False)
#def cython_get_act3(float time, float times0, float time_bit, int b_size, np.ndarray[float, ndim=1] buff):
def cython_get_act3(float time, float times0, float time_bit, int b_size, np.ndarray[np.float64_t, ndim=1] buff):
#def cython_get_act3(double time, double times0, double time_bit, Py_ssize_t b_size, double[:] buff):
    """
        The Cython version of the third implementation of unit.get_act()

        In the namespace of the unit.get_act method, the arguments are:
            time : time
            times0 : self.times[0]
            time_bit : self.time_bit
            b_size : self.buff_size
            buff : self.buffer
    """
    #cdef double t = time - times0
    cdef float t = time - times0
    cdef Py_ssize_t base = <int>(t // time_bit)
    #cdef double rem = t % time_bit
    cdef float rem = t % time_bit
    base = max( 0, min( base, b_size-2 ) )
    #cdef double frac2 = rem / time_bit
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

    new_buff = odeint(self.derivatives, [self.buffer[-1]], new_times, rtol=self.rtol, atol=self.atol) 
    self.buffer = np.roll(self.buffer, -self.min_buff_size)
    self.buffer[self.offset:] = new_buff[1:,0] 
    """
    solution = solve_ivp(self.derivatives, (new_times[0], new_times[-1]), [self.buffer[-1]], method='LSODA',
                                            t_eval=new_times, rtol=self.rtol, atol=self.atol)
    self.buffer = np.roll(self.buffer, -self.min_buff_size)
    self.buffer[self.offset:] = solution.y[0,1:]
    """

    self.pre_syn_update(time) # Update any variables needed for the synapse to update.

    for pre in self.net.syns[self.ID]:
        pre.update(time)

    self.last_time = time # last_time is used to update some pre_syn_update values


def euler_int(derivatives, double x0, double t0, int n_steps, double dt):
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
    x = np.zeros(n_steps, dtype=np.dtype('d'))
    x[0] = x0
    cdef double t = t0
    for step in range(1, n_steps, 1):
        x[step] = x[step-1] + dt * derivatives([x[step-1]], t)
        t = t + dt
    return x


cdef euler_maruyama_impl(derivatives, double[::1] buff, double t0, Py_ssize_t offset, 
                       double dt, double mudt, double sigma, double[:] noise, double factor):
    cdef Py_ssize_t mbs = buff.shape[0] - offset # minimal buffer size
    cdef Py_ssize_t i, step
    for i in range(offset):
        buff[i] = buff[i+mbs]

    cdef double t = t0
    for step in range(offset, len(buff)):
        buff[step] = buff[step-1] + ( dt * <double>derivatives([buff[step-1]], t) + mudt
                                  + factor * noise[step-offset] )
        buff[step] = max(buff[step], 0.)
        t += dt


def euler_maruyama(derivatives, double[::1] buff, double t0, Py_ssize_t offset, 
                       double dt, double mu, double sigma):
#def euler_maruyama(derivatives,  np.ndarray[np.float64_t, ndim=1] buff, float t0, Py_ssize_t offset, 
#                       float dt, float mu, float sigma):
    """ The Euler-Maruyama method for stochastic differential equations.

        This solver turns the neural model into a Langevin equation by adding white noise.

        Args:
            derivatives: The function derivatives(y, t) returns 
                         the derivative of the firing rate at time t given that
                         the current rate is y.
            x0: initial state
            buff: the unit's activation buffer
            t0: initial time
            offset: index of last value of previous integration
            dt: integration step size
            mu: mean of the white noise
            sigma: the standard deviation associated to the Wiener process
    """
    #"""
    cdef Py_ssize_t mbs = len(buff) - offset # minimal buffer size
    cdef Py_ssize_t i, step
    for i in range(offset):
        buff[i] = buff[i+mbs]

    cdef double t = t0
    cdef double factor = sigma*np.sqrt(dt)
    cdef double mudt = mu*dt
    #cdef float[:] noise = np.random.normal(loc=0., scale=1., size=buff.shape[0]-offset)
    sqrdt = np.sqrt(dt)
    for step in range(offset, len(buff)):
        #buff[step] = buff[step-1] + ( dt * derivatives([buff[step-1]], t) + mudt
        buff[step] = buff[step-1] + ( dt * <double>derivatives([buff[step-1]], t) + mudt
                                  + sigma * np.random.normal(loc=0., scale=sqrdt) )
        buff[step] = max(buff[step], 0.)
        t = t + dt
    """
    cdef double factor = sigma*np.sqrt(dt)
    cdef double mudt = mu*dt
    cdef double[:] noise = np.random.normal(loc=0., scale=1., size=buff.shape[0]-offset)
    euler_maruyama_impl(derivatives, buff, t0, offset, dt, mudt, sigma, noise, factor)
    """

#cpdef float[:] exp_euler(derivatives, float x0, float t0, int n_steps, 
#                       float dt, float mu, float sigma, float eAt, float c2, float c3):
def exp_euler(derivatives, double x0, double t0, int n_steps, 
                       double dt, double mu, double sigma, double eAt, double c2, double c3):
    """ A version of the  exponential Euler method for stochastic differential equations.

        This method is used when the derivatives function decomposes into a linear part
        plus a remainder. In the case of the noisy_leaky_linear unit the linear part
        comes from the term -lambda*y,  and the nonlinear remainder is the input.

        Args:
            derivatives: The function derivatives(y, t) returns 
                         the derivative of the firing rate at time t given that
                         the current rate is y, but it no longer includes the
                         term -lambda*y .
            x0: initial state
            t0: initial time
            n_steps: number of integration steps
            dt: integration step size
            mu: mean of the white noise
            sigma: the standard deviation associated to the Wiener process
            eAt: np.exp(A*dt), where A = -self.lambd*self.rtau
            c2 = (self.eAt-1)/A
            c3 = np.sqrt( (self.eAt**2. - 1) / (2.*A) )
    """
    """
    x = np.zeros(n_steps, dtype=np.dtype('f'))
    x[0] = x0
    cdef double t = t0
    cdef double mudt = mu*dt
    cdef sc3 = c3*sigma
    #cdef float eAt = np.exp(-lambd*dt/tau)
    #cdef float c2 = -(expAt - 1.)/(lambd*tau)
    #cdef float c3 = np.sqrt( (eAt*eAt - 1.)/(-2.*lambd*tau) )
    for step in range(1, n_steps, 1):
        x[step] = ( eAt*x[step-1] + c2*derivatives(x[step-1], t)
                + sc3*np.random.normal(loc=0., scale=1.) + mudt )
        t += dt
    return x
    """
    cdef double sc3 = c3*sigma
    x = np.zeros(n_steps, dtype=np.dtype('d'))
    x[0] = x0
    cdef double[:] noise = np.random.normal(loc=0., scale=1., size=n_steps)
    return exp_euler_impl(derivatives, x, t0, n_steps, dt, mu, eAt, c2, sc3, noise)

 
cdef double[:] exp_euler_impl(derivatives, double[:] x, double t0, int n_steps, 
                       double dt, double mu, double eAt, double c2, double sc3, double[:] noise):
    cdef Py_ssize_t step
    cdef double t = t0
    cdef double mudt = mu*dt
    for step in range(1, n_steps, 1):
        x[step] = ( eAt*x[step-1] + c2*<double>derivatives(<float>x[step-1], <float>t)
                + sc3*noise[step] + mudt )
        t += dt
    return x
