""" 
network.py
The network class used in the draculab simulator.
"""

from draculab import unit_types, synapse_types, plant_models, syn_reqs  # names of models and requirements
from units import *
from synapses import *
from plants import *
import numpy as np
from cython_utils import * # interpolation and integration methods including cython_get_act*,
from requirements import *
from array import array # optionally used for the unit's buffer
#from numba import jit


class network():
    """ 
    This class has the tools to build and simulate a network.

    Roughly, the steps are:
    First, you create an instance of network(); 
    second, you use the create() method to add units and plants;
    third, you use source.set_function() for source units, which provide inputs;
    fourth, you use the connect() method to connect the units, 
    fifth, you use set_plant_inputs() and set_plant_outputs() to connect the plants;
    finally you use the run() method to run the simulation.

    """

    def __init__(self, params):
        """
        The network class constructor.

        Args:
            params : parameter dictionary
            REQUIRED PARAMETERS
                min_delay : minimum transmission delay, and simulation step size
                min_buff_size : number of network states to store for each simulation step.
            OPTIONAL PARAMETERS 
                rtol = relative tolerance in the ODE integrator.
                atol = absolute tolerance in the ODE integrator.
                See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
        """
        self.sim_time = 0.0  # current simulation time [ms]
        self.n_units = 0     # current number of units in the network
        self.units = []      # list with all the unit objects
        self.n_plants = 0    # current number of plants in the network
        self.plants = []     # list with all the plant objects
        # The next 3 lists implement the connectivity of the network
        self.delays = [] # delays[i][j] is the delay of the j-th connection to unit i 
        self.act = []    # act[i][j] is the function from which unit i obtains its j-th input
        self.syns = []   # syns[i][j] is the synapse object for the j-th connection to unit i
        self.min_delay = params['min_delay'] # minimum transmission delay
        self.min_buff_size = params['min_buff_size']  # number of values stored during
                                                      # a minimum delay period
        if 'rtol' in params: self.rtol = params['rtol']
        else: self.rtol = 1e-6 # relative tolerance of the integrator
        if 'atol' in params: self.atol = params['atol']
        else: self.atol = 1e-6 # absolute tolerance of the integrator
        self.flat = False # This network ha not been "flattened"
        

    def create(self, n, params):
        """
        This method is just a front to find out whether we're creating units or a plant.

        If we're creating units, it will call create_units(n, params).
        If we're creating a plant, it will call create_plant(n, params).

        Raises:
            TypeError.
        """
        if hasattr(unit_types, params['type'].name):
            return self.create_units(n,params)
        elif hasattr(plant_models, params['type'].name):
            return self.create_plant(n, params)
        else:
            raise TypeError('Tried to create an object of an unknown type')
        if self.flat:
            raise AssertionError("Adding elements to a flattened network")


    def create_plant(self, n, params):
        """
        Create a plant with model params['type']. 
        
        The current implementation only creates one plant per call, so n != 1 will 
        raise an exception.  The method returns the ID of the created plant.

        Args:
            n: the integer 1.
            params: a dictionary of parameters used for plant creation.
                REQUIRED PARAMETERS
                'type': a model from the plant_models enum.
                Other required parameters depend on the specific plant model.

        Returns: 
            An integer with the ID of the created plant.

        Raises:
            AssertionError, NotImplementedError, ValueError.
        """
        assert self.sim_time == 0., 'A plant is being created when the simulation time is not zero'
        if n != 1:
            raise ValueError('Only one plant can be created on each call to create()')
        plantID = self.n_plants
        try:
            plant_class = params['type'].get_class()
        except NotImplementedError: # raising the same exception with a different message
            raise NotImplementedError('Attempting to create a plant with an unknown model type')

        self.plants.append(plant_class(plantID, params, self))
        self.n_plants += 1
        return plantID
   

    def create_units(self, n, params):
        """
        create 'n' units of type 'params['type']' and parameters from 'params'.

        The method returns a list with the ID's of the created units.
        If you want one of the parameters to have different values for each unit, you can have a list
        (or numpy array) of length 'n' in the corresponding 'params' entry

        In addition, this function can give a particular spatial arrangement to the 
        created units by appropriately setting their 'coordinates' attribute.

        Args:
            n: an integer indicating how many units to create.
            params: a dictionary with the parameters used to initialize the units.
                REQUIRED PARAMETERS
                type: a unit model form the unit_types enum.
                init_val: initial activation value (also required for source units).
                OPTIONAL PARAMETERS
                coordinates: The spatial location of the units can be specified in 2 ways:
                                * One numpy array (a single point). All units will have this location.
                                * A list of n arrays. Each unit will be assigned one array.
                                          
                For other required and optional parameters, consult the specific unit models.

        Returns: a list with the ID's of the created units.
            
        Raises:
            ValueError, TypeError, NotImplementedError
                
        """
        assert (type(n) == int) and (n > 0), 'Number of units must be a positive integer'
        assert self.sim_time == 0., 'Units are being created when the simulation time is not zero'

        # Any entry in 'params' other than 'coordinates', 'type', 'function', 'branch_params',
        # or 'integ_meth' should either be a scalar, a boolean, a list of length 'n', or a
        # numpy array of length 'n'.  
        # 'coordinates' should be either a list (with 'n' arrays) or a (1|2|3) array.
        listed = [] # the entries in 'params' specified with a list
        accepted_types = [float, int, bool, str] # accepted data types for parameters
        for par in params:
            if par == 'type':
                if not issubclass(type(params[par]), unit_types):
                    raise TypeError('Incorrect unit type')
            elif par == 'coordinates':
                if type(params[par]) is np.ndarray:
                    pass
                elif type(params[par]) is list:
                    if len(params[par]) == n:
                        listed.append(par)
                    else:
                        raise ValueError('coordinates list has incorrect size')
                else:
                    raise TypeError('Incorrect type for the coordinates parameter')
            elif par == 'function':
                if type(params[par]) is list:
                    if len(params[par]) == n:
                        listed.append(par)
                elif not callable(params[par]):
                    raise TypeError('Incorrect function initialization in create_units')
            elif par == 'branch_params': # used by the double_sigma family of units
                if type(params[par]) is dict:
                    pass
                elif type(params[par]) is list:
                    if len(params[par]) == n:
                        listed.append(par)
                    else:
                        raise ValueError('branch_params list has incorrect size')
            elif par == 'integ_meth':
                if type(params[par]) is list:
                    if len(params[par]) == n:
                        listed.append(par)
                elif not type(params[par]) is str:
                    raise TypeError('Invalid type given for the integ_meth parameter')
            elif (type(params[par]) is list) or (type(params[par]) is np.ndarray):
                if len(params[par]) == n:
                    listed.append(par)
                else:
                    raise ValueError('Found parameter list of incorrect size during unit creation')
            elif not (type(params[par]) in accepted_types):
                raise TypeError('Found a parameter of the wrong type during unit creation')
                    
        params_copy = params.copy() # The 'params' dictionary that a unit receives in its constructor
                                    # should only contain scalar values. params_copy won't have lists
        # Creating the units
        unit_list = list(range(self.n_units, self.n_units + n))
        try: 
            unit_class = params['type'].get_class()
        except NotImplementedError:
            raise NotImplementedError('Attempting to create a unit with an unknown type')

        for ID in unit_list:
            for par in listed:
                params_copy[par] = params[par][ID-self.n_units]
            self.units.append(unit_class(ID, params_copy, self))

        self.n_units += n
        # putting n new slots in the delays, act, and syns lists
        self.delays += [[] for i in range(n)]
        self.act += [[] for i in range(n)]
        self.syns += [[] for i in range(n)]
        # note:  [[]]*n causes all empty lists to be the same object 
        return unit_list


    def connect(self, from_list, to_list, conn_spec, syn_spec):
        """
        Connect units using delayed transmission lines and synaptic connections.

        connect the units in the 'from_list' to the units in the 'to_list' using the
        connection specifications in the 'conn_spec' dictionary, and the
        synapse specfications in the 'syn_spec' dictionary.

        Args:
            from_list: A list with the IDs of the units sending the connections

            to_list: A list the IDs of the units receiving the connections
            
            conn_spec: A dictionary specifying a connection rule, and delays.
                REQUIRED PARAMETERS
                'rule' : a string specifying a rule on how to create the connections. 
                        Currently implemented: 
                        'fixed_outdegree' - an 'outdegree' integer entry must also be in conn_spec.
                        'fixed_indegree' - an 'indegree' integer entry must also be in conn_spec.
                        'one_to_one' - from_list and to_list should have the same length.
                        'all_to_all'.
                'delay' : either a dictionary specifying a distribution, a scalar delay value that
                        will be applied to all connections, or a list with a delay value for each one
                        of the connections to be made. Implemented distributions:
                        'uniform' - the delay dictionary must also include 'low' and 'high' values.
                            Example:
                            {...,'delay':{'distribution':'uniform', 'low':0.1, 'high':0.3}, ...}
                        Delays should be multiples of the network minimum delay.
                OPTIONAL PARAMETERS
                'allow_autapses' : True or False. Can units connect to themselves? Default is True.
                'allow_multapses' : Can units send many connections to another unit?  Default is False.

            syn_spec: A dictionary used to initialize the synapses in the connections.
                REQUIRED PARAMETERS
                'type' : a synapse type from the synapse_types enum.
                'init_w' : Initial weight values. Either a dictionary specifying a distribution, a
                        scalar value to be applied for all created synapses, or a list with length equal
                        to the number of connections to be made. Distributions:
                        'uniform' : the delay dictionary must also include 'low' and 'high' values.
                        Example: {..., 'init_w':{'distribution':'uniform', 'low':0.1, 'high':1.} }
                        'equal_norm' : the weight vectors for units in 'to_list' are uniformly sampled
                                      from the space of vectors with a given norm. The delay
                                      dictionary must also include the 'norm' parameter.
                        Example: {..., 'init_w':{'distribution':'equal_norm', 'norm':1.5} }
                        If using a list to specify the initial weights, the first entries in the
                        list correspond to the connections from unit 'from_list[0]', the following
                        to the connections from unit 'from_list[1]', and so on.
                Any other required parameters (e.g. 'lrate') depend on the synapse type.
                OPTIONAL PARAMETERS
                'inp_ports' : input ports of the connections. Either a single integer, or a list.
                            If using a list, its length must match the number of connections being
                            created, which depends on the conection rule. The first entries in the
                            list correspond to the connections from unit 'from_list[0]', the following
                            to the connections from unit 'from_list[1]', and so on. In practice it is
                            not recommended to use many input ports in a single call to 'connect'.
                    
        Raises:
            ValueError, TypeError, NotImplementedError.
        """
        
        # A quick test first (all unit ID's are in the right range)
        if (np.amax(from_list + to_list) > self.n_units-1) or (np.amin(from_list + to_list) < 0):
            raise ValueError('Attempting to connect units with an ID out of range')
        # Ensuring the network hasn't been flattened
        if self.flat:
            raise AssertionError("Adding connections to a flattened network")

        # Retrieve the synapse class from its type object
        syn_class = syn_spec['type'].get_class()

        # If 'allow_autapses' not in dictionary, set default value
        if not ('allow_autapses' in conn_spec): conn_spec['allow_autapses'] = True
        # If 'allow_multapses' not in dictionary, set default value
        if not ('allow_multapses' in conn_spec): conn_spec['allow_multapses'] = False
       
        # The units connected depend on the connectivity rule in conn_spec
        # We'll specify  connectivity by creating a list of 2-tuples with all the
        # pairs of units to connect
        connections = []  # the list with all the connection pairs as (source,target)

        if conn_spec['allow_multapses']:
            rep = True  # sampling will be done with replacement
        else:
            rep = False

        if conn_spec['rule'] == 'fixed_outdegree':  #<----------------------
            assert len(to_list) >= conn_spec['outdegree'] or rep, ['Outdegree larger than number of targets']
            for u in from_list:
                if conn_spec['allow_autapses']:
                    #targets = random.sample(to_list, conn_spec['outdegree'])
                    targets = np.random.choice(to_list, size=conn_spec['outdegree'], replace=rep)
                else:
                    to_copy = to_list.copy()
                    while u in to_copy:
                        to_copy.remove(u)
                    #targets = random.sample(to_copy, conn_spec['outdegree'])
                    targets = np.random.choice(to_copy, size=conn_spec['outdegree'], replace=rep)
                connections += [(u,y) for y in targets]
        elif conn_spec['rule'] == 'fixed_indegree':   #<----------------------
            assert len(from_list) >= conn_spec['indegree'] or rep, ['Indegree larger than number of sources']
            for u in to_list:
                if conn_spec['allow_autapses']:
                    #sources = random.sample(from_list, conn_spec['indegree'])
                    sources = np.random.choice(from_list, size=conn_spec['indegree'], replace=rep)
                else:
                    from_copy = from_list.copy()
                    while u in from_copy:
                        from_copy.remove(u)
                    #sources = random.sample(from_copy, conn_spec['indegree'])
                    sources = np.random.choice(from_copy, size=conn_spec['indegree'], replace=rep)
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
            elif w_dict['distribution'] == 'equal_norm':  #<----------------------
                # For each unit in 'to_list', get the indexes where it appears in 'connections',
                # create a vector with the given norm, and distribute it with those indexes in 
                # the 'weights' vector
                weights = np.zeros(len(connections)) # initializing 
                for unit in to_list:  # For each unit in 'to_list'
                    idx_list = []
                    for idx,conn in enumerate(connections): # get the indexes where unit appears
                        if conn[1] == unit:
                            idx_list.append(idx)
                    if len(idx_list) > 0:  # create the vector, set it in 'weights' using the indexes
                        norm_vec = np.random.uniform(0.,1.,len(idx_list))
                        norm_vec = (w_dict['norm'] / np.linalg.norm(norm_vec)) * norm_vec
                        for id_u, id_w in enumerate(idx_list):
                            weights[id_w] = norm_vec[id_u]
            else:
                raise NotImplementedError('Initializing weights with an unknown distribution')
        elif type(syn_spec['init_w']) is float or type(syn_spec['init_w']) is int:
            weights = [float(syn_spec['init_w'])] * n_conns
        elif type(syn_spec['init_w']) is list or type(syn_spec['init_w']) is np.ndarray:
            if len(syn_spec['init_w']) == n_conns:
                weights = syn_spec['init_w']
            else:
                raise ValueError('Received wrong number of weights to initialize connections')
        else:
            raise TypeError('The value given to the initial weights is of the wrong type')

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
            if (conn_spec['delay']+1e-6)%self.min_delay < 2e-6:
                delayz = [float(conn_spec['delay'])] * n_conns
            else:
                raise ValueError('Delays should be multiples of the network minimum delay')
        elif type(conn_spec['delay']) is list:
            if len(conn_spec['delay']) == n_conns:
                delayz = conn_spec['delay']
                for dely in delayz:
                    if (dely+1e-6)%self.min_delay > 2e-6:
                        raise ValueError('Delays should be multiples of the network minimum delay')
            else:
                raise ValueError('Received wrong number of delays to initialize connections')
        else:
            raise TypeError('The value given to the delay is of the wrong type')

        # Initialize the input ports, if specified in the syn_spec dictionary
        if 'inp_ports' in syn_spec:
            if type(syn_spec['inp_ports']) is int:
                portz = [syn_spec['inp_ports']]*n_conns
            elif type(syn_spec['inp_ports']) is list:
                if len(syn_spec['inp_ports']) == n_conns:
                    portz = syn_spec['inp_ports']
                else:
                    raise ValueError('Number of input ports specified does not \
                                      match number of connections created')
            else:
                raise TypeError('Input ports were specified with the wrong data type')
        else:
            portz = [0]*n_conns

        # To specify connectivity, you need to update 3 lists: delays, act, and syns
        # Using 'connections', 'weights', 'delayz', and 'portz' this is straightforward
        for idx, (source,target) in enumerate(connections):
            # specify that 'target' neuron has the 'source' input
            self.act[target].append(self.units[source].get_act)
            # add a new synapse object for our connection
            syn_params = syn_spec.copy() # a copy of syn_spec just for this connection
            syn_params['preID'] = source
            syn_params['postID'] = target
            syn_params['init_w'] = weights[idx]
            syn_params['inp_port'] = portz[idx]
            self.syns[target].append(syn_class(syn_params, self))
            # specify the delay of the connection
            self.delays[target].append( delayz[idx] )
            if self.units[source].delay <= delayz[idx]: # this is the longest delay for this source
                self.units[source].delay = delayz[idx]+self.min_delay
                # added self.min_delay because the ODE solver may ask for values out of range,
                # depending on the order in which the units are updated (e.g. when a unit asks
                # for activation values to a units that has already updated, those values will
                # be 'min_delay' too old for the updated unit).
                # After changing the delay we need to init_buffers again. Done below.

        # After connecting, run init_pre_syn_update and init_buffers for all the units connected 
        connected = [x for x,y in connections] + [y for x,y in connections]
        for u in set(connected):
            self.units[u].init_pre_syn_update()
            self.units[u].init_buffers() # this should go second, so it uses the new syn_needs


    def set_plant_inputs(self, unitIDs, plantID, conn_spec, syn_spec):
        """ Set the activity of some units as the inputs to a plant.

            Args:
                unitIDs: a list with the IDs of the input units
                plantID: ID of the plant that will receive the inputs
                conn_spec: a dictionary with the connection specifications
                    REQUIRED ENTRIES
                    'inp_ports' : A list. The i-th entry determines the input type of
                                  the i-th element in the unitIDs list.
                    'delays' : Delay value for the inputs. A scalar, or a list of length len(unitIDs)
                               Delays should be multiples of the network minimum delay.
                syn_spec: a dictionary with the synapse specifications.
                    REQUIRED ENTRIES
                    'type' : one of the synapse_types. Currently only 'static' allowed, because the
                             plant does not update the synapse dynamics in its update method.
                    'init_w': initial synaptic weight. A scalar, a list of length len(unitIDs), or
                              a dictionary specifying a distribution. Distributions:
                              'uniform' - the delay dictionary must also include 'low' and 'high' values.
                              Example: {..., 'init_w':{'distribution':'uniform', 'low':0.1, 'high':1.} }

            Raises:
                ValueError, NotImplementedError

        """
        # First check that the IDs are inside the right range
        if (np.amax(unitIDs) >= self.n_units) or (np.amin(unitIDs) < 0):
            raise ValueError('Attempting to connect units with an ID out of range')
        if (plantID >= self.n_plants) or (plantID < 0):
            raise ValueError('Attempting to connect to a plant with an ID out of range')

        # Then connect them to the plant...
        # Have to create a list with the delays (if one is not given in the conn_spec)
        if type(conn_spec['delays']) is float:
            delys = [conn_spec['delays'] for _ in unitIDs]
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
            elif type(syn_spec['init_w']) is dict: 
                w_dict = syn_spec['init_w']
                if w_dict['distribution'] == 'uniform':  #<----------------------
                    weights = np.random.uniform(w_dict['low'], w_dict['high'], len(unitIDs))
                else:
                    raise NotImplementedError('Initializing weights with an unknown distribution')
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
        for dely, unit in zip(delys, [self.units[ID] for ID in unitIDs]):
            if dely >= unit.delay: # This is the longest delay for the sending unit
                unit.delay = dely + self.min_delay
                unit.init_buffers()

        # For good measure, run init_pre_syn_update and init_buffers for all the units connected 
        for u in unitIDs:
            self.units[u].init_pre_syn_update()
            self.units[u].init_buffers() # this should go second, so it uses the new syn_needs

   
    def set_plant_outputs(self, plantID, unitIDs, conn_spec, syn_spec):
        """ Connect the outputs of a plant to the units in a list.

        Args:
            plantID: ID of the plant sending the outpus.

            unitIDs: a list with the IDs of the units receiving inputs from the plant.

            conn_spec: a dictionary with the connection specifications.
                REQUIRED ENTRIES
                'port_map': a list used to specify which output of the plant goes to which input
                            port in each of the units. There are two options for this, one uses the
                            same output-to-port map for all units, and one specifies it separately
                            for each individual unit. More precisely, the two options are:
                            1) port_map is a list of 2-tuples (a,b), indicating that output 'a' of
                               the plant (the a-th element in the state vector) connects to port 'b'.
                               This port connectivity scheme will be applied to all units.
                            2) port_map[i] is a list of 2-tuples (a,b), indicating that output 'a' of the 
                               plant is connected to port 'b' for the i-th neuron in the neuronIDs list.
                            For example if unitIDs has two elements:
                                [(0,0),(1,1)] -> output 0 to port 0 and output 1 to port 1 for the 2 units.
                                [ [(0,0),(1,1)], [(0,1)] ] -> Same as above for the first unit,
                                map 0 to 1 for the second unit. 
                'delays' : either a dictionary specifying a distribution, a scalar delay value that
                           will be applied to all connections, or a list of values. 
                           Implemented dsitributions:
                           'uniform' - the delay dictionary must also include 'low' and 'high' values.
                                Example:  'delays':{'distribution':'uniform', 'low':0.1, 'high':1.} 
                            Delays should be multiples of the network minimum delay.

            syn_spec: a dictionary with the synapse specifications.
                REQUIRED ENTRIES
                'type' : one of the synapse_types. The plant parent class does not currently store
                         and update values used for synaptic plasticity, so synapse models that
                         require presynaptic values (e.g. lpf_fast) will lead to errors.
                'init_w': initial synaptic weight. A scalar, or a list of length len(unitIDs)

        Raises:
            ValueError, TypeError.
        """
        # There is some code duplication with connect, when setting weights and delays, 
        # but it's not quite the same.

        # Some utility functions
        # this function gets a list, returns True if all elements are tuples
        T_if_tup = lambda x : (True if (len(x) == 1 and type(x[0]) is tuple) else 
                               (type(x[-1]) is tuple) and T_if_tup(x[:-1]) )
        # this function gets a list, returns True if all elements are lists
        T_if_lis = lambda x : (True if (len(x) == 1 and type(x[0]) is list) else 
                               (type(x[-1]) is list) and T_if_lis(x[:-1]) )
        # this function returns true if its argument is float or int
        foi = lambda x : True if (type(x) is float) or (type(x) is int) else False
        # this function gets a list, returns True if all elements are float or int
        T_if_scal = lambda x : ( True if (len(x) == 1 and foi(x[0])) else 
                                 foi(x[-1])  and T_if_scal(x[:-1]) )

        # First check that the IDs are inside the right range
        if (np.amax(unitIDs) >= self.n_units) or (np.amin(unitIDs) < 0):
            raise ValueError('Attempting to connect units with an ID out of range')
        if (plantID >= self.n_plants) or (plantID < 0):
            raise ValueError('Attempting to connect to a plant with an ID out of range')
       
        # Retrieve the synapse class from its type object
        syn_class = syn_spec['type'].get_class()

        # Now we create a list with all the connections. In this case, each connection is
        # described by a 3-tuple (a,b,c). a=plant's output port. b=ID of receiving unit.
        # c=input port of receiving unit.
        pm = conn_spec['port_map']
        connections = []
        if T_if_tup(pm): # one single list of tuples
            for uid in unitIDs: 
                for tup in pm:
                    connections.append((tup[0], uid, tup[1]))
        elif T_if_lis(pm): # a list of lists of tuples
            if len(pm) == len(unitIDs):
                for uid, lis in zip(unitIDs, pm):
                    if T_if_tup(lis):
                        for tup in lis:
                            connections.append((tup[0], uid, tup[1]))
                    else:
                        raise ValueError('Incorrect port map format for unit ' + str(uid))
            else:
                raise ValueError('Wrong number of entries in port map list')
        else:
            raise TypeError('port map specification should be a list of tuples, ' + \
                             'or a list of lists of tuples')
            
        n_conns = len(connections)

        # Initialize the weights. We'll create a list called 'weights' that
        # has a weight for each entry in 'connections'
        if type(syn_spec['init_w']) is float or type(syn_spec['init_w']) is int:
            weights = [float(syn_spec['init_w'])] * n_conns
        elif type(syn_spec['init_w']) is list and T_if_scal(syn_spec['init_w']):
            if len(syn_spec['init_w']) == n_conns:
                weights = syn_spec['init_w']
            else:
                raise ValueError('Number of initial weights does not match ' + \
                                 'number of connections being created')
        else:
            raise TypeError('The value given to the initial weights is of the wrong type')

        # Initialize the delays. We'll create a list 'delayz' that
        # has a delay value for each entry in 'connections'
        if type(conn_spec['delays']) is dict: 
            d_dict = conn_spec['delays']
            if d_dict['distribution'] == 'uniform':  #<----------------------
                # delays must be multiples of the minimum delay
                low_int = max(1, round(d_dict['low']/self.min_delay))
                high_int = max(1, round(d_dict['high']/self.min_delay)) + 1 #+1, so randint can choose it
                delayz = np.random.randint(low_int, high_int,  n_conns)
                delayz = self.min_delay * delayz
            else:
                raise NotImplementedError('Initializing delays with an unknown distribution')
        elif type(conn_spec['delays']) is float or type(conn_spec['delays']) is int:
            delayz = [float(conn_spec['delays'])] * n_conns
        elif type(conn_spec['delays']) is list and T_if_scal(conn_spec['delays']):
            if len(conn_spec['delays']) == n_conns:
                delayz = conn_spec['delays']
            else:
                raise ValueError('Number of delays does not match the number of connections being created')
        else:
            raise TypeError('The value given to the delay is of the wrong type')

        # To specify connectivity, you need to update 3 lists: delays, act, and syns
        # Using 'connections', 'weights', and 'delayz', this is straightforward
        for idx, (output, target, port) in enumerate(connections):
            # specify that 'target' neuron has the 'output' input
            self.act[target].append(self.plants[plantID].get_state_var_fun(output))
            """ # Instead of the line above I had this code. Not sure why...
            if self.units[target].multiport: # if the unit has support for multiple input ports
                self.act[target].append(self.plants[plantID].get_state_var)
            else:
                # It is important to run init_buffers before this, because the function returned
                # by get_state_var_fun doesn't seem to update the buffer size. This is a potential
                # bug when the plant outputs are connected twice, with larger delays on the second time.
                self.act[target].append(self.plants[plantID].get_state_var_fun(output))
            """
            # add a new synapse object for our connection
            syn_params = syn_spec.copy() # a copy of syn_spec just for this connection
            syn_params['preID'] = plantID
            syn_params['postID'] = target
            syn_params['init_w'] = weights[idx]
            syn_params['inp_port'] = port
            syn_params['plant_out'] = output
            syn_params['plant_id'] = plantID
            self.syns[target].append(syn_class(syn_params, self))

            # specify the delay of the connection
            if (delayz[idx]+1e-6)%self.min_delay < 2e-6:
                self.delays[target].append( delayz[idx] )
            else:
                raise ValueError('Delays should be multiples of the network minimum delay')
            if self.plants[plantID].delay <= delayz[idx]: # this is the longest delay for this source
                # added self.min_delay because the ODE solver may ask for values a bit out of range
                self.plants[plantID].delay = delayz[idx]+self.min_delay
                self.plants[plantID].init_buffers() # update plant buffers
                
        # After connecting, run init_pre_syn_update and init_buffers for all the units connected 
        connected = [y for x,y,z in connections] 
        for u in set(connected):
            self.units[u].init_pre_syn_update()
            self.units[u].init_buffers() # this should go second, so it uses the new syn_needs


    def flatten(self):
        """ Move unit buffers into the network. 
        
            The unit and plant buffers will be placed in a single 2D numpy array called
            'acts' in the network object, but each unit will retain a buffer that is a
            view of part of a single row of 'acts'.

            The 'times' array of all units is also replaced by a sngle 'ts' array.

            After this method is called, the network can only be run with the
            'flat_run' method.
        """
        if self.sim_time > 0.:  # the network has been run before
            raise AssertionError("The network should not be flattened after simulating") 

        # obtain the maximum delay from all unit projections
        max0 = lambda x: 0 if len(x)==0 else max(x)  # auxiliary function
        if len(self.delays) > 0:
            max_u_del = max([max0(l) for l in self.delays])        
        else:
            max_u_del = self.min_delay
        # obtain the maximum delay from all connections to plants
        max_p_del = self.min_delay # maximum delay of connections to plants
        for p in self.plants:
            max_p_del = max(max_p_del, max0([max0(dl) for dl in p.inp_dels]))
        self.max_del = max(max_u_del, max_p_del) + self.min_delay #  min_delay added (see 'connect')
        # initialize the ts array (called 'times' in units)
        self.bf_type = np.float64  # data type of the buffers when using numpy arrays
        max_steps = int(round(self.max_del/self.min_delay)) # max number of min_del steps in all delays
        self.ts_buff_size = int(round(max_steps*self.min_buff_size)) # number of activation values to store
        self.ts_bit = self.min_delay / self.min_buff_size # time interval between buffer values
        self.ts = np.linspace(-self.max_del+self.ts_bit, 0., self.ts_buff_size, dtype=self.bf_type) # the 
                                                             # corresponding times for the buffer values
        self.ts_grid = np.linspace(0., self.min_delay, self.min_buff_size+1, dtype=self.bf_type) # used to
                                                             # create values for 'times' (optionally)
        # copy info about the unit buffers into the network object
        self.has_buffer = [False] * self.n_units # does the unit have a buffer? (filled below)
        self.init_idx = [0] * self.n_units # first index of ts to consider for each unit 
        self.buff_len = [0] * self.n_units # length of buffer for each unit
        for uid, u in enumerate(self.units):
            if hasattr(u, 'buffer'):
                self.has_buffer[uid] = True
                self.buff_len[uid]  = len(u.buffer)
                self.init_idx[uid] = self.ts_buff_size - len(u.buffer)
            else: # initialize init_idx for source units
                self.init_idx[uid] = self.ts_buff_size - (int(round(u.delay/self.min_delay))
                                                          * self.min_buff_size)
        # get a delays list where the time unit is the number of buffer intervals
        self.step_dels = [ [ self.min_buff_size*int(round(d/self.min_delay)) for d in l] 
                                                                   for l in self.delays]
        # If there are plants, you'll need to store past values for all their state variables.
        # Since the indexes of units and plants are independent, you need to create an
        # index for each state variable, in a way that won't collide with the unit indexes.
        # st_var_idx[i][j] provides the index of the j-th state variable from the i-th plant
        # in the acts array, and in the inp_src list (defined below).
        # n_plant_vars counts the total number of state variables from all plants.
        n_plant_vars = 0
        if self.n_plants > 0:   # if there are plants
            index = self.n_units
            self.st_var_idx = [[] for _ in range(self.n_plants)] 
            for pid, plant in enumerate(self.plants):
                n_plant_vars += plant.dim
                for var in range(plant.dim):
                    self.st_var_idx[pid].append(index)
                    index += 1
        # For each unit obtain a vector with the index of input units or plants in acts
        self.inp_src = [ [syn.preID for syn in l] for l in self.syns ]
        # The initialization of inp_src above will fail if there are plants. Modifying it.
        if self.n_plants > 0:
            for idx1, l in enumerate(self.syns):
                for idx2, syn in enumerate(l):
                    if hasattr(syn, 'plant_out'): # synapse comes from a plant
                        self.inp_src[idx1][idx2] = self.st_var_idx[syn.plant_id][syn.plant_out]
        #======================================================================
        # Creating the acts array
        self.acts = np.zeros((self.n_units+n_plant_vars, len(self.ts)), dtype=self.bf_type)
        #======================================================================
        # The indices to extract the array of step inputs for units
        self.acts_idx = [[] for _ in range(self.n_units)]
        for uid, u in enumerate(self.units):
            if hasattr(u, 'buffer'):
                self.acts[uid,self.init_idx[uid]:] = u.init_val  # initializing acts
                idx1 = [ [src]*self.min_buff_size for src in self.inp_src[uid] ]
                idx2 = [list(range(self.ts_buff_size - self.step_dels[uid][inp] - 1, 
                        self.ts_buff_size - self.step_dels[uid][inp] - 1 + self.min_buff_size))
                        for inp in range(len(self.inp_src[uid]))]
                self.acts_idx[uid] = (idx1, idx2)
            else:  # for source units, also initialize acts
                self.acts[uid,self.init_idx[uid]:] = np.array([u.get_act(t) for
                                            t in self.ts[self.init_idx[uid]:] ])
        # Reinitializing the unit buffers as views of act, and times as views of ts
        for uid, u in enumerate(self.units):
            if self.has_buffer[uid]:
                u.buffer = np.ndarray(shape=(self.buff_len[uid]), 
                                      buffer=self.acts[uid,self.init_idx[uid]:],
                                      dtype=self.bf_type)
                u.times = np.ndarray(shape=(self.buff_len[uid]), 
                                     buffer=self.ts[self.init_idx[uid]:], dtype=self.bf_type)
                """
                # Below is code used to create a version where the step_inps matrix is
                # obtained using logical indexing. Somehow it is slower...
                u.acts = np.frombuffer(self.acts.data)
                # Using frombuffer creates a 'flat' view, which the unit will address
                # using a 1-D 'acts_idx' array that we'll create next
                u.acts_idx = [False] * self.acts.size
                idx = self.acts_idx[uid]
                for i,l in enumerate(idx[0]):
                    for j,e in enumerate(l):
                        u.acts_idx[e*self.ts_buff_size + idx[1][i][j]] = True
                u.n_inps = len(idx[0])
                """
                u.acts = self.acts.view()
                u.acts_idx = self.acts_idx[uid]
                # step_inps is a 2D numpy array. step_inps[j,k] provides the activity
                # of the j-th input to unit i in the k-th substep of the current timestep.
                u.step_inps = self.acts[self.acts_idx[uid]]
                """
                # experimental bit to test with numba
                #-----------------------------------------------------
                u.flat_acts_idx = np.array([False] * self.acts.size)
                idx = self.acts_idx[uid]
                for i,l in enumerate(idx[0]):
                    for j,e in enumerate(l):
                        u.flat_acts_idx[e*self.ts_buff_size + idx[1][i][j]] = True
                u.n_inps = len(idx[0])
                #-----------------------------------------------------
                """
                # specify the integration function for all units
                if u.type in [unit_types.noisy_linear, unit_types.noisy_sigmoidal]:
                    if u.lambd > 0.:
                        u.flat_update = u.flat_exp_euler_update
                        #u.flat_update = u.flat_euler_maru_update
                        #u.sqrdt = np.sqrt(u.time_bit)
                        u.integ_meth = "exp_euler"
                    else:
                        u.flat_update = u.flat_euler_maru_update
                        u.integ_meth = "euler_maru"
                    continue
                if u.integ_meth in ["odeint", "solve_ivp", "euler"]:
                    u.flat_update = u.flat_euler_update
                    if u.integ_meth in ["odeint", "solve_ivp"]:
                        from warnings import warn
                        warn('Integration method ' + u.integ_meth + \
                             ' subsituted by Forward Euler in some units', UserWarning)
                elif u.integ_meth == "euler_maru":
                    u.flat_update = u.flat_euler_maru_update
                elif u.integ_meth == "exp_euler":
                    u.flat_update = u.flat_exp_euler_update
                else:
                    raise NotImplementedError('The specified integration method is not \
                                               implemented for flat networks')
        # Reinitializing the buffers of plants as views of acts, times as views of ts
        for plant in self.plants:
            svi = self.st_var_idx[plant.ID][0]
            plant.buffer = np.ndarray(shape=(plant.dim, self.ts.size),
                           buffer=self.acts[svi:svi+plant.dim, :],
                           dtype=self.bf_type) 
            plant.times = self.ts.view()
            plant.buff_width = self.ts.size
            plant.offset = plant.buff_width - self.min_buff_size
            # initialize buffer
            init_buff = np.transpose(np.array([plant.init_state]*plant.buff_width))
            np.copyto(plant.buffer, init_buff)
        #*****************
        self.flat = True 
        #*****************
        # this last step is not strictly necessary, but right now the functions
        # in self.act for plants are for the non-flat buffers. Thus we replace
        # them with new ones, but this time self.flat = True
        if self.n_plants > 0:
            for idx_l, l in enumerate(self.syns):
                for idx_s, syn in enumerate(l):
                    if hasattr(syn, 'plant_out'): # synapse comes from a plant
                        self.act[idx_l][idx_s] = \
                            self.plants[syn.plant_id].get_state_var_fun(syn.plant_out)


    def get_act(self, uid, t):
        """ Get the activity of unit with ID 'uid' at time 't'.

            t should be within the range of values in the unit's buffer.
            This method is only valid for flat networks.
        """
        return cython_get_act3(t, self.ts[self.init_idx[uid]], self.ts_bit, self.buff_len[uid],
                               self.acts[uid][self.init_idx[uid]:])


    def get_act_by_step(self, uid, s):
        """ Get the activity of unit with ID 'uid' as it was 's' buffer time steps before.

            's' should not be larger than the number of steps in the unit's buffer.
            A buffer time steps corresponds to the time interval between consecutive
            buffer entries, namely min_delay/min_buff_size.

            This method is to be used with the second type of flat networks.
        """
        return self.acts[uid][-1 - s]
    

    def flat_update(self, time):
        """ Updates all state variables by advancing them one min_delay time step. """
        # update the times array
        self.ts += self.min_delay 
        #self.ts = np.roll(self.ts, -self.min_buff_size)
        #self.ts[self.ts_buff_size-self.min_buff_size:] = self.ts_grid[1:]+time
        # update input sums
        for uid, u in enumerate(self.units):
            if self.has_buffer[uid]:
                u.upd_flat_inp_sum(time)
        # roll the full acts array
        base = self.ts.size - self.min_buff_size
        self.acts[:,:base] = self.acts[:,self.min_buff_size:]
        # update buffers
        for uid, u in enumerate(self.units):
            if self.has_buffer[uid]:
                u.flat_update(time)
        for p in self.plants:
            p.flat_update(time)
        # update activities of source units and handle requirements
        for uid, u in enumerate(self.units):
            if not self.has_buffer[uid]:
                self.acts[uid,base:] = np.array([u.get_act(t) for
                                        t in self.ts[base:] ])
            # handle requirements
            u.pre_syn_update(time)
            u.last_time = time # important to have it after pre_syn_update
        # update synapses
        for synli in self.syns:
            for syn in synli:
                syn.update(time)


    def flat_run(self, total_time):
        """ Simulate a flattened network for the given time. 
        
            Flat networks keep a single numpy array in the netwok object with 
            the contents of all buffers. However, all units have buffers which are views of
            a slice of that array. 
        """
        #ray.init(num_cpus=10, ignore_reinit_error=True)
        if not self.flat:
            self.flatten()
        Nsteps = int(total_time/self.min_delay)  # total number of simulation steps
        unit_store = [np.zeros(Nsteps) for i in range(self.n_units)] # arrays to store unit activities
        plant_store = [np.zeros((Nsteps,p.dim)) for p in self.plants] # arrays to store plant steps
        times = np.zeros(Nsteps) + self.sim_time # array to store initial time of simulation steps

        for step in range(Nsteps):
            times[step] = self.sim_time # self.sim_time persists between calls to network.run()
            
            # store current unit activities
            for uid, unit in enumerate(self.units):
                unit_store[uid][step] = self.get_act(uid, self.sim_time)
                #unit_store[uid][step] = self.get_act_by_step(uid, 0)
           
            # store current plant state variables 
            for pid, plant in enumerate(self.plants):
                plant_store[pid][step,:] = plant.get_state(self.sim_time)
            
            # update units and plants
            self.flat_update(self.sim_time)

            self.sim_time += self.min_delay

        return times, unit_store, plant_store

           


    def run(self, total_time):
        """
        Simulate the network for the given time.

        This method takes steps of 'min_delay' length, in which the units, synapses 
        and plants use their own methods to advance their state variables.

        After run(T) is finished, calling run(T) again continues the simulation
        starting at the last state of the previous simulation.

        Args:
            total_time: time that the simulation will last.
        
        Returns:
            The method returns a 3-tuple (times, unit_store, plant_store): 
            times: a numpy array with  the simulation times when the update functions 
                      were called. These times will begin at the initial simulation time, 
                      and advance in 'min_delay' increments until 'total_time' is completed.
            unit_store: a list of numpy arrays. unit_store[i][j] contains the activity
                        of the i-th unit at time j-th timepoint (e.g. at times[j]).
            plant_store: a list of 2-dimensional numpy arrays. plant_store[i][j,k] is the value
                         of the k-th state variable, at the j-th timepoint, for the i-th plant.

        Raises:
            AssertionError
        """
        if self.flat:
            raise AssertionError('The run method is not used with flattened networks')
        Nsteps = int(total_time/self.min_delay)  # total number of simulation steps
        unit_store = [np.zeros(Nsteps) for i in range(self.n_units)] # arrays to store unit activities
        plant_store = [np.zeros((Nsteps,p.dim)) for p in self.plants] # arrays to store plant steps
        times = np.zeros(Nsteps) + self.sim_time # array to store initial time of simulation steps

        for step in range(Nsteps):
            times[step] = self.sim_time # self.sim_time persists between calls to network.run()
            
            # store current unit activities
            for uid, unit in enumerate(self.units):
                unit_store[uid][step] = unit.get_act(self.sim_time)
           
            # store current plant state variables 
            for pid, plant in enumerate(self.plants):
                plant_store[pid][step,:] = plant.get_state(self.sim_time)
            
            # update units
            for unit in self.units:
                unit.update(self.sim_time)

            # update plants
            for plant in self.plants:
                plant.update(self.sim_time)

            self.sim_time += self.min_delay

        return times, unit_store, plant_store


    def upd_unit(self, unit_id):
        """ Update the unit with the given ID. 
        
            Auxiliary function to parallel_run
        """
        self.units[unit_id].update(self.sim_time)
        return self.units[unit_id]


    def parallel_run(self, total_time, n_procs):
        """
        Simulate the network for the given time using pathos for parallelization

        This method takes steps of 'min_delay' length, in which the units, synapses 
        and plants use their own methods to advance their state variables.

        After run(T) is finished, calling run(T) again continues the simulation
        starting at the last state of the previous simulation.

        Args:
            total_time: time that the simulation will last.
            n_procs: number of processes
        
        Returns:
            The method returns a 3-tuple (times, unit_store, plant_store): 
            times: a numpy array with  the simulation times when the update functions 
                      were called. These times will begin at the initial simulation time, and
                      advance in 'min_delay' increments until 'total_time' is completed.
            unit_store: a 2-dimensional numpy array. unit_store[i][j] contains the activity
                           of the i-th unit at time j-th timepoint (e.g. at times[j]).
            plant_store: a 3-dimensional numpy array. plant_store[i][j,k] is the value
                            of the k-th state variable, at the j-th timepoint, for the i-th plant.

        """
        from pathos.multiprocessing import ProcessPool as Pool
        #import time
        #if not hasattr(self, 'pool'):
        #    self.pool = ProcessingPool(nodes=n_procs)
        pool = Pool(nodes=n_procs)
        results = [0]*len(self.units)

        Nsteps = int(total_time/self.min_delay)  # total number of simulatin steps
        unit_store = [np.zeros(Nsteps) for i in range(self.n_units)] # arrays to store unit activities
        plant_store = [np.zeros((Nsteps,p.dim)) for p in self.plants] # arrays to store plant steps
        times = np.zeros(Nsteps) + self.sim_time # array to store initial time of simulation steps

        for step in range(Nsteps):
            times[step] = self.sim_time # self.sim_time persists between calls to network.run()
            
            # store current unit activities
            for uid, unit in enumerate(self.units):
                unit_store[uid][step] = unit.get_act(self.sim_time)
           
            # store current plant state variables 
            for pid, plant in enumerate(self.plants):
                plant_store[pid][step,:] = plant.get_state(self.sim_time)
            
            # update units
            self.units = list(pool.imap(self.upd_unit, range(len(self.units))))
            #for unit in self.units:
            #    unit.update(self.sim_time)

            # update plants
            for plant in self.plants:
                plant.update(self.sim_time)

            self.sim_time += self.min_delay

        return times, unit_store, plant_store

    def parallel_run2(self, total_time, n_procs):
        """
        Simulate the network for the given time using PyMP for parallelization.

        This method takes steps of 'min_delay' length, in which the units, synapses 
        and plants use their own methods to advance their state variables.

        After run(T) is finished, calling run(T) again continues the simulation
        starting at the last state of the previous simulation.

        Args:
            total_time: time that the simulation will last.
            n_procs: number of processes
        
        Returns:
            The method returns a 3-tuple (times, unit_store, plant_store): 
            times: a numpy array with  the simulation times when the update functions 
                      were called. These times will begin at the initial simulation time, and
                      advance in 'min_delay' increments until 'total_time' is completed.
            unit_store: a 2-dimensional numpy array. unit_store[i][j] contains the activity
                           of the i-th unit at time j-th timepoint (e.g. at times[j]).
            plant_store: a 3-dimensional numpy array. plant_store[i][j][k] is the value
                            of the k-th state variable, at the j-th timepoint, for the i-th plant.

        """
        import pymp
        #import time

        Nsteps = int(total_time/self.min_delay)  # total number of simulatin steps
        unit_store = [np.zeros(Nsteps) for i in range(self.n_units)] # arrays to store unit activities
        plant_store = [np.zeros((Nsteps,p.dim)) for p in self.plants] # arrays to store plant steps
        times = np.zeros(Nsteps) + self.sim_time # array to store initial time of simulation steps

        for step in range(Nsteps):
            times[step] = self.sim_time # self.sim_time persists between calls to network.run()
        
            # store current unit activities
            for uid, unit in enumerate(self.units):
                unit_store[uid][step] = unit.get_act(self.sim_time)
       
            # store current plant state variables 
            for pid, plant in enumerate(self.plants):
                plant_store[pid][step,:] = plant.get_state(self.sim_time)
        
            # update units
            with pymp.Parallel(n_procs, if_=True) as p:
                #pymp.shared.list(self.delays) 
                #pymp.shared.list(self.act) 
                for index in p.range(len(self.units)):
                    self.units[index].update(self.sim_time)

            # update plants
            for plant in self.plants:
                plant.update(self.sim_time)

            self.sim_time += self.min_delay
            #time.sleep(.1)

        return times, unit_store, plant_store


    def parallel_run3(self, total_time, n_procs):
        """
        Simulate the network for the given time using the multiprocess module for parallelization.

        This method takes steps of 'min_delay' length, in which the units, synapses 
        and plants use their own methods to advance their state variables.

        After run(T) is finished, calling run(T) again continues the simulation
        starting at the last state of the previous simulation.

        Args:
            total_time: time that the simulation will last.
            n_procs: number of processes
        
        Returns:
            The method returns a 3-tuple (times, unit_store, plant_store): 
            times: a numpy array with  the simulation times when the update functions 
                      were called. These times will begin at the initial simulation time, and
                      advance in 'min_delay' increments until 'total_time' is completed.
            unit_store: a 2-dimensional numpy array. unit_store[i][j] contains the activity
                           of the i-th unit at time j-th timepoint (e.g. at times[j]).
            plant_store: a 3-dimensional numpy array. plant_store[i][j][k] is the value
                            of the k-th state variable, at the j-th timepoint, for the i-th plant.

        """
        #from multiprocessing import Process
        import multiprocess as mp
        #import dill 
        #mp.reduction.pickle = dill

        #p = [None]*len(self.units)
        with mp.Pool(processes=n_procs) as pool:

            Nsteps = int(total_time/self.min_delay)  # total number of simulatin steps
            unit_store = [np.zeros(Nsteps) for i in range(self.n_units)] # arrays to store unit activities
            plant_store = [np.zeros((Nsteps,p.dim)) for p in self.plants] # arrays to store plant steps
            times = np.zeros(Nsteps) + self.sim_time # array to store initial time of simulation steps

            for step in range(Nsteps):
                times[step] = self.sim_time # self.sim_time persists between calls to network.run()
            
                # store current unit activities
                for uid, unit in enumerate(self.units):
                    unit_store[uid][step] = unit.get_act(self.sim_time)
           
                # store current plant state variables 
                for pid, plant in enumerate(self.plants):
                    plant_store[pid][step,:] = plant.get_state(self.sim_time)
            
                # update units
                self.units = pool.map(self.upd_unit, range(len(self.units)))
                """
                for index in range(len(self.units)):
                    p[index] = Process(target=self.upd_unit, args=(index,))
                    p[index].start()
                #p[index].join()
                """
                # update plants
                for plant in self.plants:
                    plant.update(self.sim_time)

            self.sim_time += self.min_delay

        return times, unit_store, plant_store


    def parallel_run4(self, total_time, n_procs):
        """
        Simulate the network for the given time using Ray for parallelization.

        This method takes steps of 'min_delay' length, in which the units, synapses 
        and plants use their own methods to advance their state variables.

        After run(T) is finished, calling run(T) again continues the simulation
        starting at the last state of the previous simulation.

        Args:
            total_time: time that the simulation will last.
            n_procs: number of processes
        
        Returns:
            The method returns a 3-tuple (times, unit_store, plant_store): 
            times: a numpy array with  the simulation times when the update functions 
                      were called. These times will begin at the initial simulation time, and
                      advance in 'min_delay' increments until 'total_time' is completed.
            unit_store: a 2-dimensional numpy array. unit_store[i][j] contains the activity
                           of the i-th unit at time j-th timepoint (e.g. at times[j]).
            plant_store: a 3-dimensional numpy array. plant_store[i][j][k] is the value
                            of the k-th state variable, at the j-th timepoint, for the i-th plant.

        """
        ray.init(num_cpus=10, ignore_reinit_error=True)

        @ray.remote
        def ray_upd_unit(unit_id):
            """ Update the unit with the given ID. """
            self.units[unit_id].update(self.sim_time)
            return self.units[unit_id]


        Nsteps = int(total_time/self.min_delay)  # total number of simulatin steps
        unit_store = [np.zeros(Nsteps) for i in range(self.n_units)] # arrays to store unit activities
        plant_store = [np.zeros((Nsteps,p.dim)) for p in self.plants] # arrays to store plant steps
        times = np.zeros(Nsteps) + self.sim_time # array to store initial time of simulation steps

        ray_units = rem_units.remote(self)

        for step in range(Nsteps):
            times[step] = self.sim_time # self.sim_time persists between calls to network.run()
            
            # store current unit activities
            for uid, unit in enumerate(self.units):
                unit_store[uid][step] = unit.get_act(self.sim_time)
           
            # store current plant state variables 
            for pid, plant in enumerate(self.plants):
                plant_store[pid][step,:] = plant.get_state(self.sim_time)
            
            # update units
            #obj_ids = [ray_upd_unit.remote(u) for u in range(len(self.units))]
            [ray_units.update.remote(i, self.sim_time) for i in range(len(self.units))]

            # update plants
            for plant in self.plants:
                plant.update(self.sim_time)

            self.sim_time += self.min_delay

        return times, unit_store, plant_store

"""
@ray.remote
class rem_units():
    def __init__(self, net):
        self.units = net.units

    def update(self, unit_id, time):
        self.units[unit_id].update(time) 
"""

