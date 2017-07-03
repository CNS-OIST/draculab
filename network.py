''' 
network.py
The network class used in the sirasi simulator.
'''

from sirasi import unit_types, synapse_types, plant_models, syn_reqs  # names of models and requirements
from units import *
from synapses import *
from plants import *
import numpy as np
import random # to create random connections

class network():
    ''' 
    This class has the tools to build a network.
    First, you create an instance of network(); 
    second, you use the create() method to add units and plants;
    third, you use source.set_function() for source units, which provide inputs;
    fourth, you use the connect() method to connect the units, 
    finally you use the run() method to run a simulation.
    '''

    def __init__(self, params):
        '''
        The constructor receives a 'params' dictionary, which only requires two entries:
        A minimum transmission delay 'min_delay', and
        a minimum buffer size 'min_buff_size'.
        '''
        self.sim_time = 0.0  # current simulation time [ms]
        self.n_units = 0     # current number of units in the network
        self.units = []      # list with all the unit objects
        self.n_plants = 0    # current number of plants in the network
        self.plants = []     # list with all the plant objects
        # The next 3 lists implement the connectivity of the network
        self.delays = [] # delays[i][j] is the delay of the j-th connection to unit i in [ms]
        self.act = []    # act[i][j] is the function from which unit i obtains its j-th input
        self.syns = []   # syns[i][j] is the synapse object for the j-th connection to unit i
        self.min_delay = params['min_delay'] # minimum transmission delay [ms]
        self.min_buff_size = params['min_buff_size']  # number of values stored during a minimum delay period
        

    def create(self, n, params):
        '''
        This method is just a front to find out whether we're creating units or a plant.

        If we're creating units, it will call create_units().
        If we're creating a plant, it will call create_plant().
        '''
        if hasattr(unit_types, params['type'].name):
            return self.create_units(n,params)
        elif hasattr(plant_models, params['type'].name):
            return self.create_plant(n, params)
        else:
            raise TypeError('Tried to create an object of an unknown type')


    def create_plant(self, n, params):
        '''
        Create a plant with model params['model']. The current implementation only 
        creates one plant per call.
        The method returns the ID of the created plant.
        '''
        assert self.sim_time == 0., 'A plant is being created when the simulation time is not zero'
        if n != 1:
            raise ValueError('Only one plant can be created on each call to create()')
        plantID = self.n_plants
        if params['type'] == plant_models.pendulum:
            self.plants.append(pendulum(plantID, params, self))
        elif params['type'] == plant_models.conn_tester:
            self.plants.append(conn_tester(plantID, params, self))
        else:
            raise NotImplementedError('Attempting to create a plant with an unknown model type.')
        self.n_plants += 1
        return plantID
   

    def create_units(self, n, params):
        '''
        create 'n' units of type 'params['type']' and parameters from 'params'.
        The method returns a list with the ID's of the created units.
        If you want one of the parameters to have different values for each unit, you can have a list
        (or numpy array) of length 'n' in the corresponding 'params' entry
        '''
        assert (type(n) == int) and (n > 0), 'Number of units must be a positive integer'
        assert self.sim_time == 0., 'Units are being created when the simulation time is not zero'

        # Any entry in 'params' other than 'coordinates' and 'type' 
        # should either be a scalar, a list of length 'n', or a numpy array of length 'n'
        # 'coordinates' should be either a list (with 'n' tuples) or a (1|2|3)-tuple
        listed = [] # the entries in 'params' specified with a list
        for par in params:
            if par != 'type':
                if (type(params[par]) is list) or (type(params[par]) is np.ndarray):
                    if len(params[par]) == n:
                        listed.append(par)
                    else:
                        raise ValueError('Found parameter list of incorrect size during unit creation')
                elif (type(params[par]) != float) and (type(params[par]) != int):
                    if not (par == 'coordinates' and type(params[par]) is tuple):
                        raise TypeError('Found a parameter of the wrong type during unit creation')
                    
        params_copy = params.copy() # The 'params' dictionary that a unit receives in its constructor
                                    # should only contain scalar values. params_copy won't have lists
        # Creating the units
        unit_list = list(range(self.n_units, self.n_units + n))
        if params['type'] == unit_types.source:
            default_fun = lambda x: None  # source units start with a null function
            for ID in unit_list:
                for par in listed:
                    params_copy[par] = params[par][ID-self.n_units]
                self.units.append(source(ID, params_copy, default_fun, self))
        elif params['type'] == unit_types.sigmoidal:
            for ID in unit_list:
                for par in listed:
                    params_copy[par] = params[par][ID-self.n_units]
                self.units.append(sigmoidal(ID, params_copy, self))
        elif params['type'] == unit_types.linear:
            for ID in unit_list:
                for par in listed:
                    params_copy[par] = params[par][ID-self.n_units]
                self.units.append(linear(ID, params_copy, self))
        else:
            raise NotImplementedError('Attempting to create a unit with an unknown model type.')
        
        self.n_units += n
        # putting n new slots in the delays, act, and syns lists
        self.delays += [[] for i in range(n)]
        self.act += [[] for i in range(n)]
        self.syns += [[] for i in range(n)]
        # note:  [[]]*n causes all empty lists to be the same object 
        return unit_list
    
    def connect(self, from_list, to_list, conn_spec, syn_spec):
        '''
        connect the units in the 'from_list' to the units in the 'to_list' using the
        connection specifications in the 'conn_spec' dictionary, and the
        synapse specfications in the 'syn_spec' dictionary.
        The current version always allows multapses.

        :param from_list: A list with all the units sending the connections
        :param to_list: A list of all the units receiving the connections
        :conn_spec: A dictionary specifying a connection rule, and delays
        :return: returns nothing
        '''
        
        # A quick test first
        if (np.amax(from_list + to_list) > self.n_units-1) or (np.amin(from_list + to_list) < 0):
            raise ValueError('Attempting to connect units with an ID out of range')
       
        # Let's find out the synapse type
        if syn_spec['type'] == synapse_types.static:
            syn_class = static_synapse
        elif syn_spec['type'] == synapse_types.oja:
            syn_class = oja_synapse
        elif syn_spec['type'] == synapse_types.antihebb:
            syn_class = anti_hebbian_synapse
        elif syn_spec['type'] == synapse_types.cov:
            syn_class = covariance_synapse
        elif syn_spec['type'] == synapse_types.anticov:
            syn_class = anti_covariance_synapse
        elif syn_spec['type'] == synapse_types.hebbsnorm:
            syn_class = hebb_subsnorm_synapse
        else:
            raise ValueError('Attempting connect with an unknown synapse type')
        
        # The units connected depend on the connectivity rule in conn_spec
        # We'll specify  connectivity by creating a list of 2-tuples with all the
        # pairs of units to connect
        connections = []  # the list with all the connection pairs as (source,target)
        if conn_spec['rule'] == 'fixed_outdegree':  #<----------------------
            for u in from_list:
                if conn_spec['allow_autapses']:
                    targets = random.sample(to_list, conn_spec['outdegree'])
                else:
                    to_copy = to_list.copy()
                    while u in to_copy:
                        to_copy.remove(u)
                    targets = random.sample(to_copy, conn_spec['outdegree'])
                connections += [(u,y) for y in targets]
        elif conn_spec['rule'] == 'fixed_indegree':   #<----------------------
            for u in to_list:
                if conn_spec['allow_autapses']:
                    sources = random.sample(from_list, conn_spec['indegree'])
                else:
                    from_copy = from_list.copy()
                    while u in from_copy:
                        from_copy.remove(u)
                    sources = random.sample(from_copy, conn_spec['indegree'])
                connections += [(x,u) for x in sources]
        elif conn_spec['rule'] == 'all_to_all':    #<----------------------
            targets = to_list
            sources = from_list
            if conn_spec['allow_autapses']:
                connections = [(x,y) for x in sources for y in targets]
            else:
                connections = [(x,y) for x in sources for y in targets if x != y]
        elif conn_spec['rule'] == 'one_to_one':   #<----------------------
            if len(to_list) != len(from_list):
                raise ValueError('one_to_one connectivity requires equal number of sources and targets')
            connections = list(zip(from_list, to_list))
            if conn_spec['allow_autapses'] == False:
                connections = [(x,y) for x,y in connections if x != y]
        else:
            raise ValueError('Attempting connect with an unknown rule')
            
        n_conns = len(connections)  # number of connections we'll make

        # Initialize the weights. We'll create a list called 'weights' that
        # has a weight for each entry in 'connections'
        if type(syn_spec['init_w']) is dict: 
            w_dict = syn_spec['init_w']
            if w_dict['distribution'] == 'uniform':  #<----------------------
                weights = np.random.uniform(w_dict['low'], w_dict['high'], n_conns)
            else:
                raise NotImplementedError('Initializing weights with an unknown distribution')
        elif type(syn_spec['init_w']) is float or type(syn_spec['init_w']) is int:
            weights = [float(syn_spec['init_w'])] * n_conns
        else:
            raise TypeError('The value given to the initial weight is of the wrong type')

        # Initialize the delays. We'll create a list 'delayz' that
        # has a delay value for each entry in 'connections'
        if type(conn_spec['delay']) is dict: 
            d_dict = conn_spec['delay']
            if d_dict['distribution'] == 'uniform':  #<----------------------
                # delays must be multiples of the minimum delay
                low_int = max(1, round(d_dict['low']/self.min_delay))
                high_int = max(1, round(d_dict['high']/self.min_delay)) + 1 #+1, so randint can choose it
                delayz = np.random.randint(low_int, high_int,  n_conns)
                delayz = self.min_delay * delayz
            else:
                raise NotImplementedError('Initializing delays with an unknown distribution')
        elif type(conn_spec['delay']) is float or type(conn_spec['delay']) is int:
            delayz = [float(conn_spec['delay'])] * n_conns
        else:
            raise TypeError('The value given to the delay is of the wrong type')

        # To specify connectivity, you need to update 3 lists: delays, act, and syns
        # Using 'connections', 'weights', and 'delayz' this is straightforward
        for idx, (source,target) in enumerate(connections):
            # specify that 'target' neuron has the 'source' input
            self.act[target].append(self.units[source].get_act)
            # add a new synapse object for our connection
            syn_params = syn_spec # a copy of syn_spec just for this connection
            syn_params['preID'] = source
            syn_params['postID'] = target
            syn_params['init_w'] = weights[idx]
            self.syns[target].append(syn_class(syn_params, self))
            # specify the delay of the connection
            self.delays[target].append( delayz[idx] )
            if self.units[source].delay <= delayz[idx]: # this is the longest delay for this source
                self.units[source].delay = delayz[idx]+self.min_delay
                # added self.min_delay because the ODE solver may ask for values a bit out of range
                self.units[source].init_buffers()

        # After connecting, run init_pre_syn_update for all the units connected 
        connected = [x for x,y in connections] + [y for x,y in connections]
        for u in set(connected):
            self.units[u].init_pre_syn_update()


    def set_plant_inputs(self, unitIDs, plantID, conn_spec, syn_spec):
        """ Make the activity of some units provide inputs to a plant.

            Args:
                unitIDs: a list with the IDs of the input units
                plantID: ID of the plant that will receive the inputs
                conn_spec: a dictionary with the connection specifications
                    REQUIRED ENTRIES
                    'inp_ports' : A list. The i-th entry determines the input type of
                                  the i-th element in the unitIDs list.
                    'delays' : Delay value for the inputs. A scalar, or a list of length len(unitIDs)
                syn_spec: a dictionary with the synapses specifications.
                    'type' : one of the synapse_types. Currently only 'static' allowed.
                    'init_w': initial synaptic weight. A scalar, or a list of length len(unitIDs)

            Raises:
                ValueError, NotImplementedError

        """
        # First check that the IDs are inside the right range
        if (np.amax(unitIDs) > self.n_units-1) or (np.amin(unitIDs) < 0):
            raise ValueError('Attempting to connect units with an ID out of range')
        if (plantID > self.n_plants-1) or (plantID < 0):
            raise ValueError('Attempting to connect to a plant with an ID out of range')

        # Then connect them to the plant...
        # Have to create a list with the delays (if one is not given in the conn_spec)
        if type(conn_spec['delays']) is float:
            delys = [conn_spec['delays'] for _ in inp_funcs]
        elif (type(conn_spec['delays']) is list) or (type(conn_spec['delays']) is np.ndarray):
            delys = conn_spec['delays']
        else:
            raise ValueError('Invalid value for delays when connecting units to plant')
        # Have to create a list with all the synaptic weights
        if syn_spec['type'] is synapse_types.static: 
            synaps = []
            syn_spec['postID'] = plantID
            if type(syn_spec['init_w']) is float:
                weights = [syn_spec['init_w']]*len(unitIDs)
            elif (type(syn_spec['init_w']) is list) or (type(syn_spec['init_w']) is np.ndarray):
                weights = syn_spec['init_w']
            else:
                raise ValueError('Invalid value for initial weights when connecting units to plant')

            for pre,w in zip(unitIDs,weights):
                syn_spec['preID'] = pre
                syn_spec['init_w'] = w
                synaps.append( static_synapse(syn_spec, self) )
        else:
            raise NotImplementedError('Inputs to plants only use static synapses')

        inp_funcs = [self.units[uid].get_act for uid in unitIDs]
        ports = conn_spec['inp_ports']
        
        # Now just use this auxiliary function in the plant class
        self.plants[plantID].append_inputs(inp_funcs, ports, delys, synaps)

        # You may need to update the delay of some sending units
        for dely, unit in zip(delys, [self.units[id] for id in unitIDs]):
            if dely >= unit.delay: # This is the longest delay for the sending unit
                unit.delay = dely + self.min_delay
                unit.init_buffers()



    def run(self, total_time):
        '''
        Simulate the network for the given time.

        It takes steps of 'min_delay' length, in which the units, synapses 
        and plants use their own methods to advance their state variables.
        The method returns a 3-tuple with numpy arrays containing the simulation
        times when the update functions were called, and the unit activities and 
        plant states corresponding to those times.
        After run(T) is finished, calling run(T) again continues the simulation
        starting at the last state of the previous simulation.
        '''
        Nsteps = int(total_time/self.min_delay)
        unit_stor = [np.zeros(Nsteps) for i in range(self.n_units)]
        plant_stor = [np.zeros((Nsteps,p.dim)) for p in self.plants]
        times = np.zeros(Nsteps) + self.sim_time
        
        for step in range(Nsteps):
            times[step] = self.sim_time
            
            # store current unit activities
            for uid, unit in enumerate(self.units):
                unit_stor[uid][step] = unit.get_act(self.sim_time)
           
            # store current plant state variables 
            for pid, plant in enumerate(self.plants):
                plant_stor[pid][step,:] = plant.get_state(self.sim_time)
                
            # update units
            for unit in self.units:
                unit.update(self.sim_time)

            # update plants
            for plant in self.plants:
                plant.update(self.sim_time)

            self.sim_time += self.min_delay

        return times, unit_stor, plant_stor

