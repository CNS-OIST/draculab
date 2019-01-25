
# coding: utf-8

# # Tutorial 6
# **PART 1**  
# Creating unique models:
# * Creating units with multiport inputs.
# * Creating new requirements.
# 
# **PART 2**  
# Simulations using the **ei_net** class.

# ## Part 1
# ### Units with multiport inputs and new requirements
# 
# For some unit models, there may be several types of qualitatively different inputs. An example can be found in the `delta_linear` class of `custom_units.py`. This class, together with the `delta_synapse` class in in `synapses.py` implements a continuous-time version of the _delta_ learning rule.
# 
# The _delta_ learning rule is one of the most widespread forms of supervised learning in neural networks. Given a _training set_ consisting of  input vectors 
# $(\bf{x_1}, \dots, \bf{x_n})$ and their corresponding desired outptus $(y_1, \dots, y_n)$, the purpose of the delta rule is to adjust the input weights of a unit so
# the response to $\bf{x_i}$ is $y_i$. Without going into details, this can sometimes be achieved by presenting the inputs one by one, each time adjusting the weights using:
# 
# $\Delta \omega_{j} = \alpha (y - u)x_j$,
# 
# where $y$ is the desired response to the input, $u$ is the actual response, $x_j$ is the $j$-th component of the input vector, $\omega_j$ is the corresponding weight, and $\alpha$ is a learning rate.  
# To implement this, we require to know not only the input to the unit, but also the desired output. Since the error $e \equiv y - u$ is used by every synapse,
# it is computationally efficient to calculate it once in the unit class and make it available to all synapses. Thus, the error is a synaptic _requirement_ that must be updated at every simulation step.
# 
# This tutorial is a guide to the source code in the `delta_linear`, and `delta_synapse` classes (found in `custom_units.py` and 
# `synapses.py` respectively),
# which illustrate the tools used to create units with multiple input ports, and synapses that use custom requirements.

# In[ ]:


# We can begin by looking at the docstring of the two classes
from draculab import *
help(delta_linear)


# In[ ]:


help(delta_synapse)


# Creating a unit with many input ports is the same as creating any other type of unit. As was shown in tutorial 5, you just need to
# register its name in `draculab.py`, and also write `init` and `derivatives`or `dt_fun` methods. The difference comes when you write
# what's inside of the `derivatives` (or `dt_fun`) function.
# 
# In tutorial 5, the binary unit received the input sum from the `get_input_sum` function in `derivatives`, and from `inp_sum` in `dt_fun`.
# Both `get_input_sum` and `inp_sum` ignore the input ports, but depending on the type of unit, this may not be appropriate.
# For example, for the `delta_linear` unit we don't want to add the error (at port 1) or the learning trigger (at port 2) into the input
# sum. Thus, the input sum should only consider inputs at port 0. Draculab offers tools for multiport units to deal with these type
# of situations.
# 
# When the creator method of the `unit` class receives a parameter dictionary with the `n_ports` (number of ports) entry, and that entry has a 
# value larger than 1, then the unit is considered a _multiport_ unit, and the `multiport` attribute is set to `True`. This
# causes the `init_pre_syn_update` method of the unit to add a new attribute, called `port_idx`.

# In[ ]:


# ***** Quoting the source code comments at 'init_pre_syn_update' *****

# If we require support for multiple input ports, create the port_idx list.
# port_idx is a list whose elements are lists of integers.
# port_idx[i] contains the indexes in net.syns[self.ID]
# (or net.delays[self.ID], or net.act[self.ID])
# of the synapses whose input port is 'i'.


# Using the `port_idx` list we can write a version of `get_input_sum` that considers only the inputs from port 0.
# This is what the `delta_linear.get_mp_input_sum` does:

# In[ ]:


def get_mp_input_sum(self,time):                                                                                      
    """ The input sum function of the delta_linear unit. """                                                              
    return  sum( [syn.w * act(time - dely) for syn, act, dely in zip(                                                 
                 [self.net.syns[self.ID][i] for i in self.port_idx[0]],                                               
                 [self.net.act[self.ID][i] for i in self.port_idx[0]],                                                
                 [self.net.delays[self.ID][i] for i in self.port_idx[0]])] ) / self.inp_l2


# It can be seen that the sum is divided by `inp_l2`. This quantity is the L2 norm of the input vector, and is a synaptic
# requirement, which means it will be updated at each iteration step, every `min_delay` time units. The requirements of
# a unit are added in its `__init__` method, at the line where the `syn_needs` set is updated.
# 
# An alternative to using `port_idx` for defining `get_mp_input_sum` is to use the `get_mp_inputs`, and `get_mp_weights` functions.

# In[ ]:


help(unit.get_mp_inputs)


# In[ ]:


help(unit.get_mp_weights)


# ### Exercise 1
# Write `get_mp_input_sum` using `get_mp_inputs`.  
# (solution at the bottom)

# In the case of flat networks things are a bit different. For units with a single input port we obtained the input sum
# from the `inp_sum` array present in _flattened_ units. This array is updated at each simulation step by the `unit.upd_flat_inp_sum`
# method, which is invoked by `network.flat_update`. This later method is the one tasked with updating the activities of all units
# and plants in the network, using them to fill the `network.acts` array.
# 
# When a flattened unit requires the inputs segregated by port, then `upd_flat_inp_sum` should be replaced by `upd_flat_mp_inp_sum`,
# which produces the `mp_inp_sum` array. Whereas `inp_sum` is a 1-dimensional array, `mp_inp_sum` is 2-dimensional. 
# `mp_inp_sum[i]` is a 1D array, the equivalent of `inp_sum` if we only consider inputs at port `i`. The `upd_flat_mp_inp_sum` for linear
# units will be used only if the `need_mp_inp_sum` attribute of the unit is set to `True`, which is done in the `__init__` method.
# 
# From this discussion, it is straightforward to understand the `delta_linear.dt_fun` method:

# In[ ]:


def dt_fun(self, y, s):
    """ Returns the derivative when state is y, at time substep s. """ 
    return ( self.gain * self.mp_inp_sum[0][s] / self.inp_l2 + self.bias - y ) / self.tau


# At this point the source code in `delta_linear.__init__`, and `delta_linear.derivatives` should be easy to understand.
# Please read it now, if you have not done it yet.
# 
# What is still not clear is how to implement the `error` requirement that will be used by the `delta_synapse` class.
# Creating a new _requirement_ takes 3 basic steps:
# 1. Register the requirement's name in `draculab.py`.
# 2. Create an initialization method in `requirements.py`.
# 3. Create a function to update the requirement, either in `units.py` or in `custom_units.py`.
# 
# The first step is similar to how things are done when creating a new unit, plant, or synapse type. There is an _Enum_ subclass called
# `syn_reqs` in `draculab.py`. All that is required is to add the name of your requirement here, assigning it a unique integer.
# 
# For the second step, the following "notes" (from the source code in `init_pre_syn_update`) are illuminating:

# In[ ]:


"""
        DEVELOPER'S NOTES: 
        The names of the possible requirements are in the syn_reqs Enum in draculab.py .
        Any new requirement needs to register its name in there.
        Implementation of a requirement has 2 main parts:
        1) Initialization: where the data structures used by the requirement get their
           initial values. This is done here by calling the 'add_<req_name>' functions.
           These functions, defined in requirements.py, add the necessary attributes
           and initialize them.
        2) Update: a method with the name upd_<req name>  that is added to the
           'functions' list of the unit. This method usually belongs to the 'unit' 
           class and is written somewhere after init_pre_syn_update, but in
           some cases it is defined only for one of the descendants of 'unit'.
           
           The name of the requirement, as it appears in the syn_reqs Enum, must
           also be used in the 'add_' and 'upd_' methods. Failing to use this
           naming convention will result in failure to add the requirement.

           For example, the LPF'd activity with a fast time
           constant has the name 'lpf_fast', which is its name in the syn_reqs
           Enum. The file requirements.py has the 'add_lpf_fast' method, and
           the method that updates 'lpf_fast' is called upd_lpf_fast.

        Additionally, some requirements have buffers so they can provide their value
        as it was 'n' simulation steps before. The prototypical case is the 'lpf'
        variables, which are retrieved by synapses belonging to a target cell, so 
        the value they get should have a propagation delay. This requires two other
        things in the implementation:
        1) Initialization of the requirement's buffer is added in unit.init_buffers.
           The buffers should have the name <req name>_buff.
        2) There are 'getter' methods to retrieve past values of the requirement.
           The getter methods are named get_<req_name> .
"""


# At this point you can look at the `add_error` and `add_inp_l2` methods in `requirements.py`.
# They are providing the unit with the new attributes used by the requirement, and checking whether any
# prerequisite requirements are present. In this case, we want the `mp_inputs` requirement to be present.

# ### Exercise 2
# What does `mp_inputs` do?

# ### Exercise 3
# Does it make sense to write a version of `get_mp_input_sum` that takes advantage of the `mp_inputs` requirement?

# The `add_error` method will create `error` and `learning` attributes in the unit.
# Now we need a method to update those attributes. This method needs to satisfy two requisites: it has to be available to the `delta_linear`
# class, and its name must be `upd_error`. As mentioned in the notes above, there are naming conventions to follow when creating a
# requirement. The requirement can have any arbitrary name, but the attribute that implements it must share that name, and the name
# must be used in the function that initializes it (`add_<req name>`) and in the function that updates it (`upd_<req name>`). There are reasons for these conventions.

# In[ ]:


# EXERCISE 4

# In unit.init_pre_syn_update we can find the code that adds the 
# upd_<req name> method to the list of functions that must be executed
# at every simulation step to update requirements. This list is
# named 'functions'.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prepare all the requirements to be updated by the pre_syn_update function. 
    for req in self.syn_needs:
        if issubclass(type(req), syn_reqs):
            eval('add_'+req.name+'(self)')
            eval('self.functions.add(self.upd_'+req.name+')')
        else:  
            raise NotImplementedError('Asking for a requirement that is not implemented')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Can you see how the naming conventions are used?


# Regarding the code of the `delta_synapse` class, there is nothing new about it. The only thing
# to remark is that it uses the `error` requirement that we created, and that various alternative configurations
# can be used by commenting and uncommenting source code.

# ### Exercise 5
# Have a look at the source code of `delta_synapse`.

# ## Part 2
# ### The ei_net class
# Now we will create a delta unit and test it using random input vectors with unit norm.
# The desired output will be the norm of the projection of the input vector on a given vector $\bf{v}$.
# If the learning algorithm works, the weight vector of the delta unit must approach $\bf{v}$.
# 
# To create this simulation we use the `ei_net` class from `ei_net.py`.  
# This class is used to configure  and run simulations involving three different populations, called **e**, **i**, and **x**.  
# These three populations are meant to contain excitatory, inhibitory, and source units respectively. An instance of the `ei_net` class contains default parameter dictionaries for the three populations, and for their connections. In theory we could just create an instance of `ei_net`, run `ei_net.build`, and then start running simulations with `ei_net.run`. Of course, those simulations would not use the network we want to simulate, so we need to adapt the parameters of `ei_net` for this end.
# 
# We want the **e** population to contain a single `delta_linear` unit, the **i** population to be empty, and the **x** population to contain __inp_dim__ source units, where __inp_dim__ stands for the dimensionality of the input vectors. Moreover, we want to provide our `delta_linear` unit with the desired output for each input, and with the signal that triggers learning.
# 
# This part of the tutorial shows how to configure `ei_net` for this end. Given the multiple input dimensionality of the `delta_linear` unit, the `ei_network` class would be a better tool, but for the purpose of this demonstration `ei_net` is appropriate.

# In[ ]:


from ei_net import *
help(ei_net)
# ei_net contains many methods. 
# It is not necessary to read all this documentation.


# The first step will be to configure the inputs.  
# the `ei_net.run` method is used to run simulations where a different input pattern is presented to the **e** and **i** populations
# every `pres_time` time units. This is reminiscent of how discrete inputs are presented to artificial neural networks, although in this case the inputs are continuous in time.
# 
# Two arguments tell `ei_net.run` how to present inputs: `set_inp_pat`, and `set_inp_fun`. `set_inp_pat` is a function that receives a presentation number, and returns its corresponding input pattern. The input patterns, however, are discrete, and still need to be converted into functions of time. This is the role of `set_inp_fun`, which sets the functions of the source units in the **x** population.
# 
# `ei_net.run` works using a _for_ cycle, where at each loop a new pattern is selected using `set_inp_pat`, and this pattern is set into the **x** population using `set_inp_fun`. Then the simulation is advanced `pres_time` time units.
# 
# `set_inp_fun` can be used to provide continuous transitions between patterns (e.g. the input signals have no "jumps"), and if desired, it can also be used to add noise into the patterns. There is a default version of `set_inp_fun` that is used when this argument is not provided to `ei_net.run`. The default version of `set_inp_fun`, uses linear interpolation to transition between different values of the patterns during the first fifth of the presentation. We will use this.

# In[ ]:


# A bit more detail can be seen in the documentation
help(ei_net.run)


# In[ ]:


# The first thing we may do is to create the inputs
import numpy as np
# Create the input vectorsgt
inp_dim = 10  # dimensionality of the input vectors
n_inp = 300  # number of input vectors
inps = np.zeros((inp_dim, n_inp)) # each column is an input
for col in range(n_inp):
    vec = np.random.uniform(0., 1., inp_dim)
    inps[:, col] = vec / np.linalg.norm(vec)
    
# Create the desired outputs
v = np.zeros(inp_dim)
v[0::2] = 1. # every other entry is non-zero
v = v / np.linalg.norm(v)
des_out = np.zeros(n_inp)
for idx in range(n_inp):
    des_out[idx] = np.dot(v, inps[:, idx])

# Creating the set_inp_pat argument to ei_net.run
def inp_pat(pres, rows, columns):
    """ The set_inp_pat argument to ei_net.run .
    
        The rows and columns arguments are not used for this case.
        We use the 'inps' and 'des_out' arrays created above.
    """
    return np.concatenate((inps[:, pres%n_inp], [des_out[pres%n_inp], 1.])) 


# In the `set_inp_pat` function above it can be seen that in addition to the units with the input pattern, there will be two other input units in **x**, one providing the desired output, and another one the learning trigger signal (always 1).  
# 
# We must be careful when specifying the input ports to our `delta_linear` unit. The `ei_net` class connects populations using the `topology.topo_connect` method, which specifies input ports using the `inp_ports` entry of its `syn_spec` dictionary. In `ei_net`
# the `syn_spec` dictionary for the **x** to **e** connection is in the `xe_syn` dictionary.
# 
# Creation of the **e**, **i**, and **x** populations is done with the `topology.create_group` method, which requires geometry and parameter dictionaries. the `*_geom` and `*_pars` dictionaries (* = e, i, or x) provide the corresponding entries in `ei_net`.
# 
# Configuration of the parameter dictionaries in `ei_net` is done with the `set_param` method. The advantage of using this method over directly modifying the dictionaries is that the changes done with `set_param` automatically get logged into a `history` list, which records all the modifications done to the standard parameters. Moreover, `set_param` ensures that no parameter modifications are done after the `ei_net.build` method has been called.

# In[ ]:


help(ei_net.set_param)


# In[ ]:


help(ei_net.__init__)


# In[ ]:


## Create an instance of ei_net
one_delta = ei_net()

## Specify the number of units in e, i, and x
# One single unit in e
one_delta.set_param('e_geom', 'rows', 1)
one_delta.set_param('e_geom', 'columns', 1)
one_delta.set_param('e_geom', 'center', [1., 0.])
# No units in i
one_delta.set_param('i_geom', 'rows', 0)
one_delta.set_param('i_geom', 'center', [1., 0.])
# inp_dim+2 units in x
one_delta.set_param('x_geom', 'rows', inp_dim+2)
one_delta.set_param('x_geom', 'columns', 1)

## configure the xi connection
one_delta.set_param('xe_syn', 'type', synapse_types.delta)
one_delta.set_param('xe_syn', 'lrate', 0.4)
one_delta.set_param('xe_syn', 'inp_ports', [0]*inp_dim + [1,2])
one_delta.set_param('xe_conn', 'mask', {'circular': {'radius':10}})

## avoid the ee connection
one_delta.set_param('ee_conn', 'allow_autapses', False)

## configure the delta unit
one_delta.set_param('e_pars', 'type', unit_types.delta_linear)
one_delta.set_param('e_pars', 'gain', 1.)
one_delta.set_param('e_pars', 'tau_min', 0.02 )
one_delta.set_param('e_pars', 'tau_wid', 0.)
one_delta.set_param('e_pars', 'tau_e', 1.)
one_delta.set_param('e_pars', 'bias_lrate', 0.01)
one_delta.set_param('e_pars', 'n_ports', 3)


# The `center` entry of the `e_geom` dictionary specifies the center of the rectangular grid where the units will be placed.
# The location of the units affects not only how they are visualized, but also the delay between the connections.
# 
# Another thing to observe is how the input ports of the "xe" connection are specified using a single list with `inp_dim` zeros and a `[1,2]` appended at the end. This is not a good general way to set the input ports, since it is harder to read, and the length of the list must match the number of connections being made, which is tricky when using random connectivity. In general, it is better to use `ei_network` when setting multiport connections, using separate layers for all the unit groups that target a specific port.
# 
# Next, the ee connection is removed. The `ei_net.__init__` method has dictionaries for the connection and synapse specifications of the ee, ei, ie, ii, xe, and xi connections. By default, if the populations exist, these connections will be made. In our case the **i** population is absent, so ei, ie, ii, and xi will not appear. On the other hand, if we don't specify anything the ee connections will be made, in this case consisting of the delta unit connecting to itself. To avoid this we disallow autapses.
# 
# Finally, notice that when configuring the delta unit we did not set the value of the `tau` parameter directly, but instead `tau_min` and `tau_wid`. The `tau` parameter of the units created by `ei_net.build` is set stochastically using a uniform distribution ranging from `tau_min` to `tau_min` + `tau_wid`. This approach is used for setting the `slope`, `thresh`, `tau`, and `init_val` parameters.

# ### Exercise 6
# What is wrong with the configuration of the xi synapses?
.
.
.
.
.
.
# As can be seen, all the synapses in the the xi connection use `delta` synapses. 
# This is not appropriate for the inputs to ports 1 and 2. Unlike the `inp_ports` entry, we can't use a list to specify the synapse types.
# A fix to this is to set `lrate`=0 for these two synapses. Unfortunately, `lrate` also can't be set with multiple values on a list.
# Thus we are pushed to do something "bad". We will build the `ei_net` object, and then we will set the learning rates to zero.
# 
# Changing parameters after building has to be done directly in the synapse object, rather than on the dictionaries of `ei_net`.
# Such changes are not automatically logged, and can hurt reproducibility of the results, but the user can do as she pleases.
# 
# This problem is simple to avoid using the `ei_network` class, but this case is more illustrative.

# In[ ]:


## build the ei_net object
one_delta.build()

## locate the synapses to ports 1 and 2, and freeze them
for syn_list in one_delta.net.syns:
    for syn in syn_list:
        if syn.port == 1 or syn.port == 2:
            syn.w = 1.
            syn.lrate = 0.
            syn.alpha = 0.
            # alpha = min_delay * lrate. 
            # It is used instead of lrate in the update function


# There may be a warning caused by `topo_connect` receiving an empty list. This is because the **i** population is empty.
# 
# After running `build()` the `ei_net` object has a `net` attribute containing the Draculab network.
# It is important to understand that this is just a regular Draculab network, no different from a network where everything is setup by hand
# using the regular methods from the `network` and `topology` classes (which `ei_net` uses). It is not necessary to use `ei_net.run` to
# run simulations with the `net` object that was created, but `ei_net.run` makes it easier to run simulations where particular inputs
# are presented sequentially. This type of *open_loop* simulations complement the *closed_loop* simulations such as the one in tutorial 4.
# 
# 
# After running simulations using `ei_net.run` the results will be available in the `all_times` and `all_activs` arrays of `ei_net`.
# 
# There are several methods that can be used to visualize the results:
# * basic_plot
# * conn_anim
# * act_anim
# * hist_anim
# * double_anim

# In[ ]:


# Before simulating we can have a quick look at how the connection weights
# The input units are on the left, and the delta unit is on the right.
# The radius and color of the circles indicate the initial strength of the connection.
help(one_delta.conn_anim)
one_delta.conn_anim(one_delta.x[0:10], one_delta.e, interv=200, slider=False, weights=True)


# In[ ]:


# We can also visualize the connections "by hand"
for slist in one_delta.net.syns:
    for syn in slist:
        pre_type = one_delta.net.units[syn.preID].type.name
        post_type = one_delta.net.units[syn.postID].type.name
        print("%s (%d) --> %s (%d)" %(pre_type, syn.preID, post_type, syn.postID))


# In[ ]:


# First a short simulation, to see what's happening
n_pres = 5 #n_inp # number of input presentations
pres_time = 1. # time duration for each presentation
one_delta.run(n_pres, pres_time, set_inp_pat=inp_pat)
one_delta.basic_plot()


# As can be observed, `basic_plot` shows not only the activity of the inputs and the delta unit, but also some synaptic weights,
# the 'learning' and 'error' variables of the delta unit. Behind the scenes `ei_net.build` created some source units and set their functions equal to the value of these variables. The number of source units created to track synaptic weights can be set with in the `w_track` entry of the `n` dictionary.

# In[ ]:


# Now a longer simulation
n_pres = 400 # number of input presentations
pres_time = 1. # time duration for each presentation
one_delta.run(n_pres, pres_time, set_inp_pat=inp_pat)
one_delta.basic_plot()


# In[ ]:


# Now let's see if the learning is working

# First, the error should be converging to zero
# The ei_net object recorded the errors for basic_plot().
# They are in the activity of the unit with 'error_track[0]' ID.
err_var = one_delta.all_activs[one_delta.error_track[0]]
# Let's plot the first points of the simulation against the last ones
err_fig = plt.figure(figsize=(12,10))
n_points = 1000  # how many points to plot
ts = one_delta.all_times
ts1 = ts[0:n_points]
err_var1 = err_var[0:n_points]
ts2 = ts[-n_points:] - ts[-n_points] + ts1[0]
err_var2 = err_var[-n_points:]
plt.plot(ts1, err_var1, 'b', label='initial', figure=err_fig)
plt.plot(ts2, err_var2, 'r', label='final', figure=err_fig)
plt.plot(ts1, np.zeros_like(ts1), 'k--', figure=err_fig)
plt.legend()
plt.show()


# Notice that after learning the error goes through brief "jumps" and then goes back near zero.
# The "jumps" in the error happen when there are transitions between the patterns.  
# The `delta_linear` unit does not respond instantaneously; it adjusts its output with a `tau` time constant, and during this adjustment
# its output is not as desired.

# In[ ]:


# Second, the input weights of the delta unit should resemble the 'v' vector
net = one_delta.net
weights = net.units[one_delta.e[0]].get_mp_weights(net.sim_time)
weights = weights[0] # only weights at port 0
plt.figure()
plt.bar(list(range(len(weights))), weights)
plt.title('weights vector')
plt.figure()
plt.bar(list(range(len(v))), v)
plt.title('v vector')
plt.show()
#print(v)
#print(weights)


# If things went well, both vectors should resemble each other.
# 
# As a final note, **ei_net** is useful mainly because it is the basis of the more general **ei_network** set of tools.
# 
# **ei_network** takes many objects similar to **ei_net**, rebranding them as _layers_. With **ei_network** we can
# build networks consisting of many interconnected _layers_. This gives enough flexibility to build almost any network
# you may want.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
# In[ ]:


# SOLUTION TO EXERCISE 1

def get_mp_input_sum(self,time):
    return np.dot(self.get_mp_inputs(time)[0], self.get_mp_weights(time)[0]) / self.inp_l2


# ### SOLUTION TO EXERCISE 2
# The documentation of `mp_inputs` is in the docstring of the `add_mp_inputs` method in `requirements.py`.
# In general, the documentation to a requirement is kept in the `add_<requirement_name>` method.

# ### SOLUTION TO EXERCISE 3
# It would not be recommended to use the `mp_inputs` array to implement the `get_mp_input_sum` function.  
# The reason is that `mp_inputs` like all requirements, gets updated **once** per simulation step. On the other hand, the activity of
# a unit is calculated at least `min_buffer_size` times in one simulation step of `min_delay` length. Using `mp_inputs` would "freeze" the inputs during
# the simulation step, and lead to a decrease in precision.
