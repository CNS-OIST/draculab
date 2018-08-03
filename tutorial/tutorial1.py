# coding: utf-8

# # tutorial1.ipynb
# 
# In this lesson a simple network with 10 sigmoidal units is created. Inputs are added to the units, and their activity is visualized.

# ### Part 1
# Create a network, run a simulation, visualize the activity of the units.

# First, let's import draculab
# Assuming we are currently the ../tutorial folder. The working directory must be where the source code is.
get_ipython().run_line_magic('cd', '..')
from draculab import *


# We want to create a network, so let's have a look at the constructor of the network class
help(network.__init__)

# To create create the network first we need a parameters dictionary
net_params = {
    'min_delay' : 0.005, # minimum delay in all the network's connections (in seconds)
    'min_buff_size' : 10 } # minimum buffer size. How many values are stored per simulation step. 
# Then we call the constructor
net = network(net_params)

# We will create 10 sigmoidal units
help(network.create)
help(network.create_units)

# Here's how to create the sigmoidal units
n_units = 10 # how many units to create

## first the parameters dictionary
sig_params = {
    'type' : unit_types.sigmoidal,  # unit_types is an Enum in draculab.py
    'init_val' : 0.5, # initial value
    'thresh' : .1, # all sigmoidal units will have threshold 1
    'slope' : np.random.uniform(0.5, 2., n_units), # the slopes come from a random distribution
    'tau' : 0.02 } # time constant for the dynamics of all sigmoidal units

## then we call the creator
sig_ids = net.create(n_units, sig_params)
# this puts the ID's of the created units in the sig_ids list. The ID of a unit is an integer that uniquely 
# identifies it in the network. We will later use the sig_ids list to connect the units.

# Now we create an input. 
# It will come from a 'source' unit, whose activity comes from a Python function that
# takes time as its argument. The function we will use is a cosine.
input_params = {
    'type' : unit_types.source,
    'init_val' : 1.,
    'function' : lambda t: np.cos(t) } # numpy is imported as np in the draculab module
inp_ids = net.create(1, input_params)

# Next we should connect our input unit to the sigmoidal units.
# For this we use the network.connect method.
# In preparation, we need to create conn_spec and syn_spec dictionaries, which 
# configure various details about the connection and about its synapse.
conn_spec = {
    'rule' : 'all_to_all',  # all sources connect to all targets
    'delay' : {'distribution': 'uniform', 'low': 0.01, 'high':0.1} }# connection delays will be uniformly distributed
syn_spec = {
    'type': synapse_types.static, # synapse_types is an Enum in draculab.py
    'init_w' : [0.1*n for n in range(n_units)] } # the initial weights range from 0. to 0.9

# There are many options for connections and synapses:
help(network.connect)
# In addition to network.connect, there is a 'topology' module that can create
# spatially-arranged connections. This module is covered in another tutorial.

# Create the connection
net.connect(inp_ids, sig_ids, conn_spec, syn_spec)

# We can now simulate for a few seconds
sim_time = 10. # simulation time
times, unit_acts, _ = net.run(sim_time)

# The method that runs the simulation is straightforward
help(network.run)

# We can plot the activities of the units using Matplotlib
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,10))
# Plot the activity of a single sigmoidal unit
plt.subplot(221)
plt.plot(times, unit_acts[sig_ids[2]])
plt.title('one sigmoidal')
# Plot the activity of the input unit
plt.subplot(222)
plt.plot(times, unit_acts[inp_ids[0]])
plt.title('one input')
# Plot the activity of all units
plt.subplot(223)
plt.plot(times, np.transpose(unit_acts))
plt.title('all units')
# Plot the activity of all sigmoidal units
plt.subplot(224)
sig_acts = [ unit_acts[u] for u in sig_ids ]
plt.plot(times, np.transpose(sig_acts))
plt.title('sigmoidal units')
plt.show()


# ### Part 2
# The main attributes of the `network` object.

# The network created in Part 1 has 10 sigmoidal units, each one receiving one connection. 
# Morevoer, each connection has a particular delay, and its own synapse. The delays are scalar values,
# but synapses are objects with their own attributes.
# 
# All the connection information in draculab is stored in 3 lists, which are attributes of the network object.
# * **delays**. `delays[i]` is a list that contains the delays for all the connections received by the unit whose ID is `i`. `delays[i][j]` is the delay of the j-th connection to unit i. 
# * **syns**. `syns[i]` is a list that contains the synapses of all the connections received by the unit whose ID is `i`. `syns[i][j]` is the synapse object for the j-th connection to unit i.
# * **act**. `act[i]` is a list whose elements are Python functions. `act[i][j]` is the function from which unit i obtains its j-th input.
# 
# All the units created are stored in the `network.units` list. `network.units[i]` is the unit object whose ID is `i`.
# When plants are created, they are stored in the `network.plants` list.

# We can look at the units of the plant create in part 1
net.units

# Similarly, we can look at the delays, synapses, and activity functions.
net.delays

# The attribute containing the synaptic weight in the synapse objects is called w
net.syns[2][0].w  # for unit 'n', w = 0.1*n

# Another useful attribute in the synapses are the IDs of the presynaptic and postsynaptic units,
# contained in the preID and postID attributes respectively
net.syns[2][0].preID


# All sigmoidals obtain their inputs from the same cosine function
net.act[0][0](3.141592)


# ### Part 3
# Create 10 input units, connect them to the 10 sigmoidals.
# 
# This unit is intended to show the proper way to initialize the function of source units.
# 
# ***Plase reset the kernel before continuing***

# Importing...
get_ipython().run_line_magic('cd', '..')
from draculab import *
import matplotlib.pyplot as plt

# We once more create the network as before, this time with 10 source units
net_params = {
    'min_delay' : 0.005, # minimum delay in all the network's connections (in seconds)
    'min_buff_size' : 10 } # minimum buffer size. How many values are stored per simulation step. 
net = network(net_params)

n_sigs = 10 # how many sigmoidal units to create
sig_params = {
    'type' : unit_types.sigmoidal,  # unit_types is an Enum in draculab.py
    'init_val' : 0.5, # initial value
    'thresh' : .1, # all sigmoidal units will have threshold 1
    'slope' : np.random.uniform(0.5, 2., n_sigs), # the slopes come from a random distribution
    'tau' : 0.02 } # time constant for the dynamics of all sigmoidal units
sig_ids = net.create(n_sigs, sig_params)

n_sources = 10 # how many input units to create
input_params = {
    'type' : unit_types.source,
    'init_val' : 0.5,
    'function' : lambda t: None } 
inp_ids = net.create(n_sources, input_params)

# Notice that the created source units have a function that returns 'None'. 
# We need to initialize their functions. To do this we create an auxiliary function.

def create_cosine(ang_freq, phase):
    return lambda t: np.cos(ang_freq*(t - phase))

# With our auxiliary function we now initialize the function of all source units
for idx, uid in enumerate(inp_ids):
    net.units[uid].set_function(create_cosine(2.*np.pi, 0.1*idx))


# ### The whole point of part 3 is to show that the auxiliary function is necessary.
# In other words, initializing the source units using
# ```
# for idx, uid in enumerate(inp_ids):
#     net.units[uid].set_function(lambda t : np.cos(2.*np.pi*t - 0.1*idx))
# ```
# will lead to a subtle error. If interested see: https://eev.ee/blog/2011/04/24/gotcha-python-scoping-closures/

# We connect the units, and run the simulation
# This time, each sigmoidal unit gets its own input, with a unique phase
conn_spec = {
    'rule' : 'one_to_one',  # all sources connect to all targets
    'delay' : {'distribution': 'uniform', 'low': 0.01, 'high':0.1} }# connection delays will be uniformly distributed
syn_spec = {
    'type': synapse_types.static, # synapse_types is an Enum in draculab.py
    'init_w' : [0.1*n for n in range(n_sigs)] } # the initial weights range from 0. to 0.9
net.connect(inp_ids, sig_ids, conn_spec, syn_spec)

sim_time = 10. # simulation time
times, unit_acts, _ = net.run(sim_time)


# Plot the activities of the units using Matplotlib
fig = plt.figure(figsize=(20,10))
# Plot the activity of a single sigmoidal unit
plt.subplot(221)
plt.plot(times, unit_acts[sig_ids[2]])
plt.title('one sigmoidal')
# Plot the activity of one input unit
plt.subplot(222)
plt.plot(times, unit_acts[inp_ids[5]])
plt.title('one input')
# Plot the activity of all inputs
inp_acts = [ unit_acts[u] for u in inp_ids ]
plt.subplot(223)
plt.plot(times, np.transpose(inp_acts))
plt.title('all inputs')
# Plot the activity of all sigmoidal units
plt.subplot(224)
sig_acts = [ unit_acts[u] for u in sig_ids ]
plt.plot(times, np.transpose(sig_acts))
plt.title('all sigmoidal units')
plt.show()


# ### Exercise 1
# Create a numpy 2D array `weights` that contains all the synaptic weights.
# `weights[i,j]` should be the weight of the connection from unit `j` to unit `i`. If the units are not connected it should equal zero.
# 
# ***BONUS:***
# Plot the connection matrix as an image
# 
# Solution is below.


# ### Exercise 2
# What happens if you don't use the auxiliary function to initialize the input in part 3?


# ### Exercise 3
# Repeat part 3, but this time connect the sigmoidal units so that unit 0 projects to unit 1, unit 1 to unit 2, ..., unit 9 to unit 0.  
# Use delays of 0.01, and static synapses with weight 0.5.








































# SOLUTION TO EXERCISE 1
N = len(net.units) # number of units
weights = np.zeros((N,N)) 
for syn_list in net.syns:
    for syn in syn_list:
        weights[syn.postID, syn.preID] = syn.w

# BONUS
fig_ex1 = plt.figure(figsize=(10,10))
ax = fig_ex1.add_axes([0., 0., 1., 1.], aspect=1)
ax.set_xticks(list(range(N)))
ax.set_yticks(list(range(N)))
ax.imshow(weights)
plt.show()




# SOLUTION TO EXERCISE 2

# All the input units are initialized with the same function.





# SOLUTION TO EXERCISE 3
#--------------- copy-paste ---------------
get_ipython().run_line_magic('cd', '..')
from draculab import *
import matplotlib.pyplot as plt

net_params = {
    'min_delay' : 0.005, # minimum delay in all the network's connections (in seconds)
    'min_buff_size' : 10 } # minimum buffer size. How many values are stored per simulation step. 
net = network(net_params)

n_sigs = 10 # how many sigmoidal units to create
sig_params = {
    'type' : unit_types.sigmoidal,  # unit_types is an Enum in draculab.py
    'init_val' : 0.5, # initial value
    'thresh' : .1, # all sigmoidal units will have threshold 1
    'slope' : np.random.uniform(0.5, 2., n_sigs), # the slopes come from a random distribution
    'tau' : 0.02 } # time constant for the dynamics of all sigmoidal units
sig_ids = net.create(n_sigs, sig_params)

n_sources = 10 # how many input units to create
input_params = {
    'type' : unit_types.source,
    'init_val' : 0.5,
    'function' : lambda t: None } 
inp_ids = net.create(n_sources, input_params)

def create_cosine(ang_freq, phase):
    return lambda t: np.cos(ang_freq*(t - phase))

for idx, uid in enumerate(inp_ids):
    net.units[uid].set_function(create_cosine(2.*np.pi, 0.1*idx))
    
conn_spec = {
    'rule' : 'one_to_one',  # all sources connect to all targets
    'delay' : {'distribution': 'uniform', 'low': 0.01, 'high':0.1} }# connection delays will be uniformly distributed
syn_spec = {
    'type': synapse_types.static, # synapse_types is an Enum in draculab.py
    'init_w' : [0.1*n for n in range(n_sigs)] } # the initial weights range from 0. to 0.9
net.connect(inp_ids, sig_ids, conn_spec, syn_spec)
#--------------------------------------

# THE ACTUAL SOLUTION
s2s_conn_spec = {
    'rule' : 'one_to_one',  
    'delay' : 0.01 }
s2s_syn_spec = {
    'type': synapse_types.static, 
    'init_w' : 0.5 }
target_ids = [(i+1)%n_sigs for i in sig_ids] # assuming sig_ids range from 0 to n_sigs...
net.connect(sig_ids, target_ids, s2s_conn_spec, s2s_syn_spec)

#-------- more copy-paste ----------
sim_time = 10. # simulation time
times, unit_acts, _ = net.run(sim_time)

fig = plt.figure(figsize=(20,10))
# Plot the activity of a single sigmoidal unit
plt.subplot(221)
plt.plot(times, unit_acts[sig_ids[2]])
plt.title('one sigmoidal')
# Plot the activity of one input unit
plt.subplot(222)
plt.plot(times, unit_acts[inp_ids[5]])
plt.title('one input')
# Plot the activity of all inputs
inp_acts = [ unit_acts[u] for u in inp_ids ]
plt.subplot(223)
plt.plot(times, np.transpose(inp_acts))
plt.title('all inputs')
# Plot the activity of all sigmoidal units
plt.subplot(224)
sig_acts = [ unit_acts[u] for u in sig_ids ]
plt.plot(times, np.transpose(sig_acts))
plt.title('all sigmoidal units')
plt.show()


# Using the solution to exercise 1 to visualize the connections made in exercise 3
N = len(net.units) # number of units
weights = np.zeros((N,N)) 
for syn_list in net.syns:
    for syn in syn_list:
        weights[syn.postID, syn.preID] = syn.w

fig_ex1 = plt.figure(figsize=(10,10))
ax = fig_ex1.add_axes([0., 0., 1., 1.], aspect=1)
ax.set_xticks(list(range(N)))
ax.set_yticks(list(range(N)))
ax.imshow(weights)
plt.show()

