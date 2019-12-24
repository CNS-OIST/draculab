
# coding: utf-8

# # Tutorial 7
# 
# ## Multidimensional units

# In draculab all units have a one-dimensional output, but they may have many-dimensional dynamics, with the output being one of their state variables.
# 
# The `multidim` Boolean attribute indicates whether the unit uses more than one state variable. When `multidim` is not specified it is assumed that the dynamics are one-dimensional, and `multidim` is automatically set to `False`.
# 
# When `multidim` is set to `True` the dimensionality of the state vectors are obtained from the length of the `init_val` argument. Notice that if `multidim` is `False` and `init_val` is a list of length longer than 1, it is assumed that the dynamics are one-dimensional, and the values of `initi_val` are used to initialize several units.
# 
# Obviously, besides setting `multidim` to `True`, in order to create a multidimensional unit we need to create a `derivatives` or `dt_fun` method that updates all the state variables. By convention, the **first** (index 0) state variable is the unit's activity, and gets transmitted when the unit is connected to other units or plants.

# In[ ]:


# Let's have a quick look at the arguments of the 'unit' constructor, where
# 'init_val' and 'multidim' are described
from draculab import *
from units.units import unit
import matplotlib.pyplot as plt
import numpy as np

help(unit.__init__)


# Next there is a simple example on how to create a multidimensional unit. The unit has the dynamics of a sinusuodial, e.g.
# 
# $y' = \omega x$  
# $x' = -\omega y$
# 
# where y is the unit's activity, and $\omega$ is the oscillation frequency.
# 
# The code for this unit is in the `units/custom_units.py` file. We now look at it.

# In[ ]:


from units.custom_units import test_oscillator
help(test_oscillator.__init__)


# In[ ]:


# Here is the source code of the constructor

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
            OPTIONAL PARAMETERS
            'mu' : mean of white noise when using noisy integration
            'sigma' : standard deviation of noise for noisy integration.
        """
        params['multidim'] = True
        unit.__init__(self, ID, params, network)
        self.tau = params['tau']
        self.w = 1/self.tau
        if 'mu' in params:
            self.mu = params['mu']
        if 'sigma' in params:
            self.sigma = params['sigma']
        self.mudt = self.mu * self.time_bit # used by flat updaters
        self.mudt_vec = np.zeros(self.dim)
        self.mudt_vec[0] = self.mudt
        self.sqrdt = np.sqrt(self.time_bit) # used by flat updater


# Notice that the first thing we do ist to set `multidim` to `True`.  
# Next, the parameters of the model are obtained.
# 
# The rest of the constructor fills parameters used by the Euler_Maruyama (`euler_maru`) integrator, and are included for completeness. The final four lines are only required when using Euler_Maruyama together with a flat network.
# 
# Next we can look at the methods that implement the dynamics.

# In[ ]:


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


# The implementation of the `derivatives` or `dt_fun` methods is a straightforward generalization of what was done before: we now return an array of derivatives, rather than a single value. when multidim is `True` draculab automatically uses a multidimensional solver to advance the unit's dynamics. The name of these solvers ends with "`_md`" .
# 
# The default integrator for multidimentional units in non-flat networks is `odeint`. The other options are `euler_maru`, and `solve_ivp`. For flat networks the two options are `euler` and `euler_maru`.
# 
# Let's create some `test_oscillator` units and simulate.

# In[ ]:


net_params = {'min_delay' : 0.05,
              'min_buff_size' : 20 }
unit_params = {'type' : unit_types.test_oscillator,
               'integ_meth' : 'solve_ivp',
               'tau' : .5,
               'multidim' : True, # not really necessary in this case
               'init_val' : [[1., 1.], [0, np.sqrt(2.)]],
               'mu' : 0.,
               'sigma' : 0.}
src_params = {'type' : unit_types.source,
              'init_val' : 0.,
              'function' : lambda t: np.sin(t) }
net = network(net_params)
net.create(2, unit_params)


# Notice that two units are created, and their initial values are specified using a list where each element is another list.

# In[ ]:


#times, data, _  = net.flat_run(10.)
times, data, _  = net.run(10.)
data = np.array(data)


# In[ ]:


# We can compare the results against the analytical solution (a sine function)
fig = plt.figure(figsize=(15,8))
plt.plot(times, data.transpose())
plt.plot(times, np.sqrt(2.)*np.sin(2.*times), 'r')
plt.legend(['0', '1', 'sin'])
plt.show()


# ### EXPLORATION
# This is an opportunity to test the accuracy of different solvers under different `min_delay` and `min_buff_size` settings.
