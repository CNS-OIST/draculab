''' 
network.py
The network class used in the sirasi simulator.
'''

from sirasi import unit_types, synapse_types
from units import *
from synapses import *
import numpy as np
import random # to create random connections

''' This class has the tools to build a network.
    First, you create an instance of 'network'; 
    second, you use the create() method to add units;
    third, you use set_function() for source units, which provide inputs;
    fourth, you use the connect() method to connect the units, 
    finally you use the run() method to run a simulation.
'''
class network():
    def __init__(self, params):
        self.sim_time = 0.0  # current simulation time [ms]
        self.n_units = 0     # current number of units in the network
        self.units = []      # list with all the unit objects
        # The next 3 lists implement the connectivity of the network
        self.delays = [] # delays[i][j] is the delay of the j-th connection to unit i in [ms]
        self.act = []    # act[i][j] is the function from which unit i obtains its j-th input
        self.syns = []   # syns[i][j] is the synapse object for the j-th connection to unit i
        self.min_delay = params['min_delay'] # minimum transmission delay [ms]
        self.min_buff_size = params['min_buff_size']  # number of values stored during a minimum delay period
        
    def create(self, n, params):
        # create 'n' units of type 'params['type']' and parameters from 'params'
        assert (type(n) == int) and (n > 0), 'Number of units must be a positive integer'
        assert self.sim_time == 0., 'Units are being created when the simulation time is not zero'
        
        unit_list = list(range(self.n_units, self.n_units + n))
        if params['type'] == unit_types.source:
            default_fun = lambda x: 0.  # source units start with a null function
            for ID in unit_list:
                self.units.append(source(ID,params,default_fun,self))
        elif params['type'] == unit_types.sigmoidal:
            for ID in unit_list:
                self.units.append(sigmoidal(ID,params,self))
        elif params['type'] == unit_types.linear:
            for ID in unit_list:
                self.units.append(linear(ID,params,self))
        else:
            printf('Attempting to create an unknown model type. Nothing done.')
            return []
        
        self.n_units += n
        # putting n new slots in the delays, act, and syns lists
        self.delays += [[] for i in range(n)]
        self.act += [[] for i in range(n)]
        self.syns += [[] for i in range(n)]
        # note:  [[]]*n causes all empty lists to be the same object 
        return unit_list
    
    def connect(self, from_list, to_list, conn_spec, syn_spec):
        # connect the units in the 'from_list' to the units in the 'to_list' using the
        # connection specifications in the 'conn_spec' dictionary, and the
        # synapse specfications in the 'syn_spec' dictionary
        # The current version only handles one single delay value
        
        # A quick test first
        if (np.amax(from_list + to_list) > self.n_units-1) or (np.amin(from_list + to_list) < 0):
            print('Attempting to connect units with an ID out of range. Nothing done.')
            return
       
        # Let's find out the synapse type
        if syn_spec['type'] == synapse_types.static:
            syn_class = static_synapse
        elif syn_spec['type'] == synapse_types.oja:
            syn_class = oja_synapse
        else:
            print('Attempting connect with an unknown synapse type. Nothing done.')
            return
        
        # To specify connectivity, you need to update 3 lists: delays, act, and syns
        # The units connected depend on the connectivity rule in conn_spec
        if conn_spec['rule'] == 'fixed_outdegree':
            sources = from_list
            targets = random.sample(conn_spec['outdegree'], to_list)
        elif conn_spec['rule'] == 'all_to_all':
            targets = to_list
            sources = from_list
        else:
            print('Attempting connect with an unknown rule. Nothing done.')
            return
            
        for source in sources:
            for target in targets:
                if source == target and conn_spec['allow_autapses'] == False:
                    continue     
                self.delays[target].append(conn_spec['delay'])
                self.act[target].append(self.units[source].get_act)
                syn_params = syn_spec # a copy of syn_spec just for this connection
                syn_params['preID'] = source
                syn_params['postID'] = target
                self.syns[target].append(syn_class(syn_params, self))
                if self.units[source].delay <= conn_spec['delay']: # this is the longest delay for this source
                    self.units[source].delay = conn_spec['delay']+self.min_delay
                    self.units[source].init_buffers()

         # TODO: run init_pre_syn_update for all the units, but make sure that running it
         # each time you add new synapses doesn't screw things up
    
    def run(self, total_time):
        # A basic runner.
        # It takes steps of 'min_delay' length, in which the units and synapses 
        # use their own methods to advance their state variables.
        Nsteps = int(total_time/self.min_delay)
        storage = [np.zeros(Nsteps) for i in range(self.n_units)]
        times = np.zeros(Nsteps)
        
        for step in range(Nsteps):
            self.sim_time = self.min_delay*step
            times[step] = self.sim_time
            
            # store current state
            for unit in range(self.n_units):
                storage[unit][step] = self.units[unit].get_act(self.sim_time)
                
            # update units
            for unit in range(self.n_units):
                self.units[unit].update(self.sim_time)

        return times, storage        

