
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import numpy as np
import time
import dill
import pprint
import re
from draculab import *


class ei_network():
    """ 
        This class is used to create, simulate, and visualize networks consisting of
        interconnected ei_layer objects, which are generic networks of excitatory and
        inhibitory units with optional inputs.

        A standard way to use ei_network is as follows.
        1) The class constructor receives a list of strings with the names of the layers to be created.
           The constructor will create layers with those names using standard parameters.
        2) The user sets the parameters of each layer, retrieving the layer object by name,
           e.g. ei_network.get_layer(<name>), or ei_network.layers[<name>], and then using the 
           ei_layer.set_param method.
        3) For each pair of layers to be connected, the user calls ei_network.add_connection,
           which creates empty parameter dictionaries for the connection, and signals the
           ei_network.build method that the connection should be created.
           The created connection and synapse dictionaries have standardized names:
           <sending_layer>(e|i|x)_<receiving_layer>(e|i)_(conn|syn) .
        4) The user configures the parameter dictionaries for the inter-layer connections using
           ei_network.set_param. Unconfigured parameters will receive the default values.
        5) The user calls ei_network.build method, which creates a draculab network with all the
           units and connections specified so far.
        6) The user runs simulations with ei_network.run .
        7) The user can use the tools to visualize, save, annotate, log, or continue the simulations.

        METHODS OVERVIEW 
        __init__ : creates some default dictionaries, the ei_layer objects, and the draculab network.
        get_layer : returns the ei_layer object with the given name.
        add_connection: add a connection between the populations of two different layers.
        set_param : changes a value in a parameter dictionary.
        build : builds the layers and creates the interlayer connections.
        run : runs simulations.
        basic_plot, act_anim, hist_anim, double_anim : result visualization.
        conn_anim : connections visualization.
        annotate : append a line with text in the ei_network.notes string.
        log : save the parameters, changes, and execution history of the network in a text file.
        save : pickle the object and save it in a file.
        -------- 
        default_inp_pat, default_inp_fun : auxiliary to 'run'.
        make_transform: auxiliary to 'build'.
        make_inp_fun : auxiliary to default_inp_fun.
        H : the Heaviside step function. Auxiliary to make_inp_fun.
        color_fun : auxiliary to act_anim and double_anim.
        update_(conn|hist|act|double)_anim : auxiliary to (conn|hist|act|double)_anim.
        update_weight_anim : auxiliary to conn_anim.

    """
        
    def __init__(self, layer_names, net_number=None):
        """
            ei_network class constructor.

            Args:
                layer_names : non-empty list of strings containing the names of all layers to be created.
                        The constructor will create variables with the names on the string, so the name
                        of the class' methods and variables should be avoided, as well as the name of
                        Pyhton's __X__ functions.
                net_number:  an integer to identify the object in multiprocess simulations (optional).
        """

        self.layers = {}  # the ei_layer objects will be in this dictionary 
        self.layer_connections = [] # one entry per connection. See ei_network.add_connection
        self.history = ["# ei_network.__init__ at " + time.ctime()] # list with object's history 
        self.notes = '' # comments about network configuration or simulation results.
        # fixing random seed
        seed = 19680801
        np.random.seed(seed)
        self.history.append('np.random.seed(%d)'  % (seed))

        self.net_number = net_number
        if self.net_number != None: 
            self.history.append('# parallel runner assigned this network the number ' + str(net_number))
            self.inp_hist = {} # to produce an optional lists with the ID of all the inputs presented. See run()

        # DEFAULT PARAMETER DICTIONARY FOR THE NETWORK
        self.net_params = {'min_delay' : 0.005, 
            'min_buff_size' : 10,
            'rtol' : 1e-5,
            'atol' : 1e-5,
            'cm_del' : .03 }  # delay in seconds for each centimeter in distance. 
        
        # CREATING ei_layer OBJECTS
        if type(layer_names) is list and type(layer_names[0]) is str:
            for name in layer_names:
                self.layers[name] = ei_layer(name, self.net_params)
        else:
            raise TypeError('First argument to network constructor must be a list with layer names')

        # CREATING THE draculab NETWORK
        self.net = network(self.net_params)

        # DOCUMENT THE NETWORK'S PARAMETERS AND LAYER NAMES
        pp = pprint.PrettyPrinter(indent=4, compact=True)
        self.history.append('net_params = ' + pp.pformat(self.net_params))
        self.history.append('layer_names = ' + pp.pformat(layer_names))
        self.history.append('# ()()()()() Non-default settings: ()()()()()')

        # CREATING DEFAULT INTER-LAYER CONNECTION AND SYNAPSE DICTIONARIES
        # They depend on the type of the sending population ('e','i', or 'x').
        # Default boundary and extent properties are set in ei_network.build .
        self.e_conn = { 'connection_type' : 'divergent',
            'mask' : {'circular': {'radius': .4}},  
            'kernel' : .5,
            'delays' : {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}},
            'weights' :{'uniform' : {'low': 0.01, 'high' : 0.5}},
            'allow_autapses' : True,
            'allow_multapses' : False,
            'dist_dim' : 'all',
            'edge_wrap' : True }

        self.i_conn = {'connection_type' : 'divergent',
            'mask' : {'circular': {'radius': .3}}, 
            'kernel' : .5,
            'delays' : {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}},
            'weights' :{'uniform' : {'low': -0.5, 'high' : -0.01}},
            'allow_autapses' : True,
            'allow_multapses' : False,
            'dist_dim' : 'all',
            'edge_wrap' : True }

        self.x_conn = {'connection_type' : 'divergent',
            'mask' : {'circular': {'radius': .2}}, 
            'kernel' : .9,
            'delays' : {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}},
            'weights' :{'uniform' : {'low': 0.01, 'high' : 0.5}},
            'allow_autapses' : True,
            'allow_multapses' : False,
            'dist_dim' : 'all',
            'edge_wrap' : True }
        
        self.e_syn = {'type' : synapse_types.static,  # synapses from excitatory units
            'lrate' : 0.1,  # for all dynamic synapse types
            'omega' : 1.,  # for sq_hebb_subsnorm synapses 
            'input_type' : 'pred',  # for input_correlation synapses
            'des_act' : 0.4, # for homeo_inhib, and corr_homeo_inhib synapses
            'inp_ports' : 0, # for multiport units
            'c' : 1., # for exp_rate_dist synapses
            'wshift' : 1. } # for exp_rate_dist synapses

        self.i_syn = {'type' : synapse_types.static,  # synapses from inhibitory units
            'lrate' : 0.1,  # for all dynamic synapse types
            'omega' : 1.,  # for sq_hebb_subsnorm synapses 
            'input_type' : 'pred',  # for input_correlation synapses
            'des_act' : 0.4, # for homeo_inhib, and corr_homeo_inhib synapses
            'inp_ports' : 0, # for multiport units
            'c' : 1., # for exp_rate_dist synapses
            'wshift' : 1. } # for exp_rate_dist synapses

        self.x_syn = {'type' : synapse_types.static,  # synapses from source units
            'lrate' : 0.1,  # for all dynamic synapse types
            'omega' : 1.,  # for sq_hebb_subsnorm synapses 
            'input_type' : 'pred',  # for input_correlation synapses
            'des_act' : 0.4, # for homeo_inhib, and corr_homeo_inhib synapses
            'inp_ports' : 0, # for multiport units
            'c' : 1., # for exp_rate_dist synapses
            'wshift' : 1. } # for exp_rate_dist synapses

    # Sometimes the plotting functions need to know if a unit has moving threshold or syn scales
        global trdc_u
        global ssrdc_u
        trdc_u = [unit_types.exp_dist_sig_thr, unit_types.double_sigma_trdc, 
                  unit_types.sds_trdc, unit_types.ds_n_trdc, unit_types.ds_sharp, 
                  unit_types.sds_sharp, unit_types.sig_trdc, 
                  unit_types.ds_n_sharp, unit_types.sds_n_sharp, 
                  unit_types.st_hr_sig, unit_types.sig_trdc_sharp]
        ssrdc_u = [unit_types.exp_dist_sig, unit_types.sig_ssrdc_sharp, 
                   unit_types.ss_hr_sig, unit_types.sig_ssrdc, 
                   unit_types.ds_ssrdc_sharp, unit_types.sds_n_ssrdc_sharp]

    
    def get_layer(self, name):
        """ Get the ei_layer object with the given name (a string). """
        if type(name) is str and name in self.layers:
            return self.layers[name]
        else:
            raise KeyError("The argument to get_layer must be a string with the name of an existing layer")

    
    def add_connection(self, source, target):
        """
            Add a connection between the populations of two different layers.
            
            The connection will be added to the draculab network once ei_network.build is run.

            Args:
                source: a list or a tuple with two entries. 
                        source[0] : name of the source layer (a string).
                        source[1] : population sending the projections. Either 'e', 'i', or 'x'.
                target: a list or a tuple with two entries. 
                        target[0] : name of the target layer (a string).
                        target[1] : population receiving the projections. Either 'e', or 'i'.

            This method will create connection and synapse parameter dictionaries for the connection.
            The name of those dictionaries will come from these strings:
              source[0] +  source[1] + '_' + target[0] + target[1] + '_conn'   --> connection dictionary
              source[0] +  source[1] + '_' + target[0] + target[1] + '_syn'    --> synapse dictionary

            Notice that if the 'boundary' and 'transform' entries of the connection dictionary are not
            specified before running build(), then a default boundary and transform will be used.
            The default boundary is the rectangle of the target population. The default transform puts 
            the center of the rectangle of the sending population on the center of the rectangle of the 
            receiving population.
        """
        # The way this method signals ei_network.build to create the connections is by adding an entry
        # in the layer_connections list. Each list entry consists of this dictionary:
        # { 'src_lyr':source[0], 'src_pop':source[1], 'trg_lyr':target[0], 'trg_pop':target[1] }

        # Make some tests
        if 'build()' in self.history:
            raise AssertionError('Adding connections after network has been built')     
        if not source[0] in self.layers or not target[0] in self.layers:
            raise ValueError('Unknown layer name found in the arguments to add_connection')
        if not source[1] in ['e','i','x'] or not target[1] in ['e','i']:
            raise ValueError('Population must be either e, i, (or x for senders) in arguments to add_connection')

        # Add an entry in layer_connections
        self.layer_connections.append( { 'src_lyr':source[0], 'src_pop':source[1], 
                                         'trg_lyr':target[0], 'trg_pop':target[1] } )
        # Create emtpy connection and synapse specification dictionaries
        conn_dict_name = source[0] + source[1] + '_' + target[0] + target[1] + '_conn'
        syn_dict_name = source[0] + source[1] + '_' + target[0] + target[1] + '_syn'
        setattr(self, conn_dict_name, {})
        setattr(self, syn_dict_name, {})


    def set_param(self, dictionary, entry, value):
        """ Change a value in a parameter dictionary. 
        
            dictionary: a string with the name of one of the object's parameter dictionaries
            entry: a string with the name of the entry we'll modify
            value: the value we'll assign (can be any appropriate Python object) 
        """
        if 'build()' in self.history:
            raise AssertionError('Setting a parameter value after network has been built')     
        self.__getattribute__(dictionary)[entry] = value  # set the new value
        self.history.append(dictionary + '[\'' + entry + '\'] = ' + str(value) ) # record assignment
            

    def make_transform(self, tar_center, src_center):
        """ Auxiliary to build, to avoid scoping problems.
            https://eev.ee/blog/2011/04/24/gotcha-python-scoping-closures/
        """
        return lambda x : x + (tar_center - src_center)

    def build(self):
        """ Build the layers and create the interlayer connections. 
            
            This is done after all parameter dictionaries have been specified.

        """
        # Store record of network being built
        self.history.append('#()()()()()()()()()()()()()()()()()()')
        self.history.append('build()')
        if self.net_number != None: # a string to display for multiprocess simulations
            num_str = ' at network ' + str(self.net_number)
        else:
            num_str = ''

        # Build the layers
        for name in self.layers:
            print('Building layer ' + name + num_str, end="\n")
            self.layers[name].build(self.net)
        print('\n')

        # Populate the parameter dictionaries for the inter-layer connections
        for c in self.layer_connections:
            name = c['src_lyr'] + c['src_pop'] + '_' + c['trg_lyr'] + c['trg_pop'] # base name of the dictionaries
            conn_dict = self.__getattribute__(name+'_conn')  # The add_connection method created these
            syn_dict = self.__getattribute__(name+'_syn')  # The add_connection method created these

            for sndr in [('e',self.e_conn, self.e_syn),('i',self.i_conn, self.i_syn),('x',self.x_conn, self.x_syn)]:
                if c['src_pop'] == sndr[0]:   # if the connection is being sent from a population of type sndr[0] 
                    for entry in sndr[1]:  # (e|i|x)_conn are default dictionaries created in __init__
                        if not entry in conn_dict:  # if the user didn't set a value for the entry
                            conn_dict[entry] = (sndr[1])[entry]  # set the default value
                    for entry in sndr[2]: # for each entry in the appropriate default syn dictionary
                        if not entry in syn_dict:  # if the user didn't set a value for the entry
                            syn_dict[entry] = (sndr[2])[entry] # set default value

            # The default 'boundary' and 'transform' entries of conn_dict must be created here
            src_geom = self.layers[c['src_lyr']].__getattribute__(c['src_pop']+'_geom')  # geometry dict of the source population
            trg_geom = self.layers[c['trg_lyr']].__getattribute__(c['trg_pop']+'_geom')  # geometry dict of the target population
            if not 'boundary' in conn_dict:
                conn_dict['boundary'] = {'center': np.array(trg_geom['center']), 'extent' : trg_geom['extent']}
            if not 'transform' in conn_dict:
                # The default transform puts the center of the rectangle of the sending population on the
                # center of the rectangle for the receiving population
                tar_center = np.array( trg_geom['center'] )
                src_center = np.array( src_geom['center'] )
                #conn_dict['transform'] = lambda x : x + (tar_center - src_center) # EVIL BUG FROM HELL!!!
                conn_dict['transform'] = self.make_transform(tar_center, src_center)
            else: # using custom transform
                name = c['src_lyr'] + c['src_pop'] + '_' + c['trg_lyr'] + c['trg_pop'] # base name of the dictionaries
                self.annotate('Using custom transform in ' + name + ' connection.')

        # Create the connections
        topo = topology()
        if len(self.layer_connections) > 0:
            self.history.append("#---Parameter dictionaries used for inter-layer connections---")
        pp = pprint.PrettyPrinter(indent=4, compact=True)
        for c in self.layer_connections:
            name = c['src_lyr'] + c['src_pop'] + '_' + c['trg_lyr'] + c['trg_pop'] # base name of the dictionaries
            print('Creating ' + name + ' connection' + num_str, end='\n')
            conn_dict = self.__getattribute__(name+'_conn')  # The connection dictionary
            syn_dict = self.__getattribute__(name+'_syn')  # The synapse dictionary
            src_lyr = self.layers[c['src_lyr']]  # source layer
            trg_lyr = self.layers[c['trg_lyr']]  # target layer
            src_pop = src_lyr.__getattribute__(c['src_pop'])  # source population (e|i|x)
            trg_pop = trg_lyr.__getattribute__(c['trg_pop'])  # target population (e|i)
            # Place the connection dictionaries in the object's history
            self.history.append(name+'_conn = ' + pp.pformat(conn_dict) + '\n') 
            self.history.append(name+'_syn = ' + pp.pformat(syn_dict) + '\n') 
            topo.topo_connect(self.net, src_pop, trg_pop, conn_dict, syn_dict) # creating layer!
        
        self.history.append("#()()()()()() Post build() history ()()()()()()")

        
    def default_inp_pat(self, pres, rows, columns):
        """ A default set_inp_pat argument for the run() method. 

            The returned input patterns are random vectors with some unit norm.
        """
        n = rows*columns
        #### random vector with unit norm, positive entries
        #vec = np.random.uniform(0., 1., n)
        #return  vec / np.linalg.norm(vec)
        #### random vector with unit norm, positive and negative entries
        #vec = 1. - np.random.uniform(0., 2., n)
        #return  vec / np.linalg.norm(vec)
        #### randomly choose k units to activate
        vec = np.zeros(n)
        k = min(2, n)
        vec[np.random.choice(n, k, replace=False)] = 1./k
        return vec


    def H(self, x): 
        """ The Heaviside step function. """
        return 0.5 * (np.sign(x) + 1.)


    def make_inp_fun(self, prev, cur, init_time, tran_time):
        """ Creates an input function that from init_time to tran_time goes linearly from prev to cur, 
            and remains constant with value cur afterwards.

            Auxiliary to default_inp_fun to avoid scoping problems:
            https://eev.ee/blog/2011/04/24/gotcha-python-scoping-closures/
        """
        return ( lambda t : self.H(t - tran_time) * cur + self.H(tran_time - t) *
                            (prev +  (t - init_time) * (cur - prev) / (tran_time - init_time)) )
   

    def default_inp_fun(self, pre_inp, cur_inp, init_time, pres_time, inp_units, frac=0.2):
        """ A default set_inp_fun argument for the run() method. 
        
            The input function transitions linearly from the previous to the current input
            pattern during the first 'frac' fraction of the presentation.
        """
        t_tran = init_time + frac * pres_time
        for unit, pre, cur in zip(inp_units, pre_inp, cur_inp):
            unit.set_function( self.make_inp_fun(pre, cur, init_time, t_tran) )

 
    def run(self, n_pres,  pres_time, set_inp_pat=None, set_inp_fun=None):
        """ Run a simulation, presenting n_pres patterns, each lasting pres_time. 

            Args:
                n_pres : number of pattern presentations to simulate.
                pres_time : time that each pattern presentation will last.
                set_inp_pat : a dictionary with the methods that set the patterns to appear at the source
                              units of each layer. At the beginning of each presentation, the method 
                              set_inp_pat['name'] (if present) is called for the layer with the given name. 
                              This, together with set_inp_fun, creates a new pattern to be presented. 
                              When we have a multiprocess simulation (e.g. the network has a number ID), the
                              function that gets the input pattern will have an extra argument (net_number), and
                              will return an extra integer.
                              When there is no net_number, the set_inp_pat method has the following specifications:
                        set_inp_pat(pres, rows, columns)
                            # Given a presentation number and the number of rows and columns in the input layer, 
                            # returns an input pattern, which is a 1-D numpy array with the value that the input 
                            # function should attain during the presentation.
                            # To map a given pair (r,c) to its corresponding entry in the returned vector use:
                            # idx = rows*c + r, 
                            # e.g. input[idx] corresponds to the unit in row r, and column c, with the indexes starting
                            # from 0. This is consistent with the way coordinates are assigned in topology.create_group,
                            # and the 2-D pattern can be recovered with numpy.reshape(input, (rows, columns)).
                              When there is a net_number, set_inp_pat is called as:
                        set_inp_pat(pres, rows, columns, net_number)
                            # Returns a 2-tuple with two elements. The first is an input pattern as above. The
                            # second is an integer that serves as an ID for this input.
                            # The arguments are the same as above, with the addition of net_number, which specifies 
                            # that the input pattern to be returned should correspond to the network with 'net_number' ID.
                set_inp_fun : a dictionary with the methods that instantiate Python functions providing the input
                              patterns selected with set_inp_pat. The functions should follow these specifications:
                        set_inp_fun(prev_inp_pat, cur_inp_pat, init_time, pres_time, inp_units))
                            # Assigns a Python function to each of the input units.
                            # pre_inp_pat : input pattern from the previous presentation (in the format of set_inp_pat).
                            # cur_inp_pat : current input pattern.
                            # init_time : time when the presentation will start.
                            # pres_time : duration of the presentation.
                            # inp_units : a list with the input units (e.g. "x").

                If the set_inp_pat or set_inp_fun arguments are not provided for a layer that has an input 
                population (x), the class defaults are used.
            
            Updates:
                self.all_times: 1-D numpy array with the times for each data point in all_activs.
                self.all_activs: 2-D numpy array with the activity of all units at each point in all_times. 
        """
        # set_inp_pat has the net_number argument for multiprocess simulations because in this case
        # it is hard to keep a record of the input ID's at the mp_net_runner object. Thus, ei_network.run
        # keeps a record of the presented input ID's in inp_hist, using the extra value returned by
        # set_inp_pat. This history of inputs is then used in mp_net_runner scripts to analyze the results
        # of the simulation without a need to use the raw input signals.

        # store a record of this simulation
        self.history.append('run(n_pres=%d, pres_time=%f, ...)' % (n_pres, pres_time)) 
        # initialize input patterns and storage of results 
        run_activs = []  # will contain all the input activities for this call to run()
        start_time = time.time() # to keep track of how long the simulation lasts
        prev_pat = {} # dictionary to store the previous input patterns
        inp_pat = {} # dictionary to store the current input patterns

        if not hasattr(self, 'all_times') or not hasattr(self, 'all_activs'): # if it's the first call to run()
            self.all_times = np.array([])  # a times vector that persists through multiple runs
            self.all_activs = np.tile([], (len(self.net.units),1))
            self.present = 0 # a presentation number that persists through multiple runs
            for name in self.layers:
                lay = self.layers[name]
                if lay.n['x'] > 0:  # If the layer has external input units
                    #inp_pat[name] = np.zeros(lay.n['x'])  # initial "null" pattern
                    # initial conditions come from the input functions
                    if set_inp_pat != None and (name in set_inp_pat): # if we received a function to set the layer's input pattern
                        if self.net_number != None: # if multiprocess simulation
                            (inp_pat[name], inp_id) = set_inp_pat[name](self.present, lay.x_geom['rows'], 
                                                                        lay.x_geom['columns'], self.net_number)
                            self.inp_hist[name] = []
                        else:
                            inp_pat[name] = set_inp_pat[name](self.present, lay.x_geom['rows'], lay.x_geom['columns'])
                    else:
                        inp_pat[name] = self.default_inp_pat(self.present, lay.x_geom['rows'], lay.x_geom['columns'])
        else: # if not the first run
            inp_pat = self.last_pat

        if self.net_number != None:
            num_str = ' at network ' + str(self.net_number)
        else:
            num_str = ''
        # present input patterns
        for pres in [self.present + p for p in range(n_pres)]:
            print('Starting presentation ' + str(pres) + num_str)
            pres_start = time.time() # to keep track of how long the presentation lasts
            t = self.net.sim_time
            # Setting input patterns
            for name in inp_pat:
                lay = self.layers[name]
                prev_pat[name] = inp_pat[name]
                inp_units = [self.net.units[i] for i in lay.x]
                if set_inp_pat != None and (name in set_inp_pat): # if we received a function to set the layer's input pattern
                    if self.net_number != None: # if it's a multiprocess simulation
                        (inp_pat[name], inp_id) = set_inp_pat[name](pres, lay.x_geom['rows'], 
                                                                   lay.x_geom['columns'], self.net_number)
                        self.inp_hist[name].append(inp_id)
                    else:
                        inp_pat[name] = set_inp_pat[name](pres, lay.x_geom['rows'], lay.x_geom['columns'])
                else:
                    inp_pat[name] = self.default_inp_pat(pres, lay.x_geom['rows'], lay.x_geom['columns'])
                # Setting input functions
                if set_inp_fun and (name in set_inp_fun): # we received a method to create the input function
                    set_inp_fun[name](prev_pat[name], inp_pat[name], t, pres_time, inp_units)
                else:
                    self.default_inp_fun(prev_pat[name], inp_pat[name], t, pres_time, inp_units)
            # Simulating
            times, activs, plants = self.net.run(pres_time)
            #self.all_times.append(times) # deprecated...
            #self.all_activs.append(activs)
            self.all_times = np.append(self.all_times, times)
            run_activs.append(activs)
            print('Presentation %s took %s seconds ' % (pres, time.time() - pres_start) + num_str, end='\n')

        self.present += n_pres   # number of input presentations so far
        self.last_pat = inp_pat  # used to initialize inp_pat and prev_pat in the next run
        #self.all_times = np.concatenate(self.all_times) # deprecated ...
        #self.all_activs = np.concatenate(self.all_activs, axis=1)
        self.all_activs = np.append(self.all_activs, np.concatenate(run_activs, axis=1), axis=1)
        print('Total execution time is %s seconds ' % (time.time() - start_time) + num_str) 
        print('----------------------')

        return self # see rant on multiprocessing    


    def basic_plot(self, lyr_name):
        """ Creates input, activities, and other optional plots for some units in the given layer.

            Args:
                lyr_name : A string with the name of one of the layers.

            The plots produced will be for:
                * All inputs from the x population (if any).
                * The activity for the units of layer.tracked .
                * The evolution of some synaptic weights 
                * Synaptic scale factors for the units in layer.tracked
                  (if using ssrdc units and layer.n['w_track'] > 0)
                * Thresholds for the units in layer.tracked
                  (if using trdc units and layer.n['w_track'] > 0)
        """
        #%matplotlib inline
        # Plot the inputs
        pl_wid = 18 # width of the plots
        pl_hgt = 5  # height of the plots
        layer = self.layers[lyr_name]
        if layer.n['x'] > 0:
            inp_fig = plt.figure(figsize=(pl_wid,pl_hgt))
            inputs = np.transpose([self.all_activs[i] for i in layer.x])
            plt.plot(self.all_times, inputs, linewidth=1, figure=inp_fig)
            plt.title('Inputs')

        # Plot some unit activities
        unit_fig = plt.figure(figsize=(pl_wid,pl_hgt))
        e_tracked = [e for e in layer.tracked if e in layer.e]
        i_tracked = [i for i in layer.tracked if i in layer.i]
        if len(e_tracked) > 0:
            e_acts = np.transpose(self.all_activs[e_tracked])
            plt.plot(self.all_times, e_acts, figure=unit_fig, linewidth=4)
        if len(i_tracked) > 0:
            i_acts = np.transpose(self.all_activs[i_tracked])
            plt.plot(self.all_times, i_acts, figure=unit_fig, linewidth=1)
        plt.title('Some unit activities at layer '+lyr_name+'. Thick=Exc, Thin=Inh')
        
        if layer.n['w_track'] > 0:
            # Plot the evolution of the synaptic weights
            w_fig = plt.figure(figsize=(pl_wid,pl_hgt))
            weights = np.transpose([self.all_activs[layer.w_track[i]] for i in range(layer.n['w_track'])])
            plt.plot(self.all_times, weights, linewidth=1)
            plt.title('Some synaptic weights')
            # Plot the evolution of the synaptic scale factors
            if layer.e_pars['type'] in ssrdc_u or layer.i_pars['type'] in ssrdc_u:
                sc_fig = plt.figure(figsize=(pl_wid,pl_hgt))
                factors = np.transpose([self.all_activs[layer.sc_track[i]] for i in range(layer.n['w_track'])])
                plt.plot(self.all_times, factors, linewidth=1)
                plt.title('Some synaptic scale factors')
            # Plot the evolution of the thresholds
            if layer.e_pars['type'] in trdc_u or layer.i_pars['type'] in trdc_u:
                thr_fig = plt.figure(figsize=(pl_wid,pl_hgt))
                thresholds = np.transpose([self.all_activs[layer.thr_track[i]] for i in range(layer.n['w_track'])])
                plt.plot(self.all_times, thresholds, linewidth=1)
                plt.title('Some unit thresholds')
        # Plot the error and  learning variables of delta units
        delta_u = [unit_types.delta_linear]
        if layer.e_pars['type'] in delta_u: 
            lrn_fig = plt.figure(figsize=(pl_wid,pl_hgt))
            #lrn_var= np.transpose([self.all_activs[layer.learn_track[i]] for i in range(layer.n['w_track'])])
            lrn_var= self.all_activs[layer.learn_track[0]] 
            plt.plot(self.all_times, lrn_var, self.all_times, np.tile(.5,len(self.all_times)), 'k--')
            plt.title('learning')
            err_fig = plt.figure(figsize=(pl_wid,pl_hgt))
            #err_var= np.transpose([self.all_activs[layer.error_track[i]] for i in range(layer.n['w_track'])])
            err_var= self.all_activs[layer.error_track[0]] 
            plt.plot(self.all_times, err_var) 
            plt.title('error')

        plt.show()


    def conn_anim(self, source, sink, interv=100, slider=False, weights=True):
        """ An animation to visualize the connectivity of populations. 
    
            source and sink are lists with the IDs of the units whose connections we'll visualize. 
            
            Each frame of the animation shows: for a particular unit in source,
            all the neurons in sink that receive connections from it (left plot), and for 
            a particular unit in sink, all the units in source that send it connections
            (right plot).
        
            interv is the refresh interval (in ms) used by FuncAnimation.
            
            If weights=True, then the dots' size and color will reflect the absolute value
            of the connection weight.

            Returns:
                animation object from FuncAnimation if slider = False
                widget object from ipywidgets.interact if slider = True
        """
        get_ipython().run_line_magic('matplotlib', 'qt5')
        # notebook or qt5 
        
        # update_conn_anim uses these values
        self.len_source = len(source)
        self.len_sink = len(sink)
        self.source_0 = source[0]
        self.sink_0 = sink[0]

        # flattening net.syns, leaving only the source-sink connections 
        self.all_syns = []
        for syn_list in [self.net.syns[i] for i in sink]:
            self.all_syns.extend([s for s in syn_list if s.preID in source])
    
        # getting lists with the coordinates of all source, sink units
        source_coords = [u.coordinates for u in [self.net.units[i] for i in source]]
        sink_coords = [u.coordinates for u in [self.net.units[i] for i in sink]]
        source_x = [c[0] for c in source_coords]
        source_y = [c[1] for c in source_coords]
        sink_x = [c[0] for c in sink_coords]
        sink_y = [c[1] for c in sink_coords]

        # id2src[n] maps the unit with network id 'n' to its index in the 'source' list
        self.id2src = np.array([1e8 for _ in range(len(self.net.units))], dtype=int) # 1e8 if not in source
        for src_idx, net_idx in enumerate(source):
            self.id2src[net_idx] = src_idx
        # id2snk[n] maps the unit with network id 'n' to its index in the 'sink' list
        self.id2snk = np.array([1e8 for _ in range(len(self.net.units))], dtype=int) # 1e8 if not in sink
        for snk_idx, net_idx in enumerate(sink):
            self.id2snk[net_idx] = snk_idx

        # setting colors
        self.std_src = [0., 0.5, 0., 0.5]
        self.std_snk = [0.5, 0., 0., 0.5]
        self.big_src = [0., 0., 1., 1.]
        self.big_snk = [0., 0., 1., 1.]

        # constructing figure, axes, path collections
        self.conn_fig = plt.figure(figsize=(12,7))
        self.ax1 = self.conn_fig.add_axes([0.02, 0.01, .47, 0.95], frameon=True, aspect=1)
        self.ax2 = self.conn_fig.add_axes([0.51, 0.01, .47, 0.95], frameon=True, aspect=1)
        self.src_col1 = self.ax1.scatter(source_x, source_y, s=2, c=self.std_src)
        self.snk_col1 = self.ax1.scatter(sink_x, sink_y, s=2, c=self.std_snk)
        self.src_col2 = self.ax2.scatter(source_x, source_y, s=2, c=self.std_src)
        self.snk_col2 = self.ax2.scatter(sink_x, sink_y, s=2, c=self.std_snk)
        self.ax1.set_title('sent connections')
        self.ax2.set_title('received connections')
        self.ax2.set_yticks([])
        
        if weights:
            update_fun = self.update_weight_anim
            # extract the weight matrix
            self.w_mat = np.zeros((len(sink), len(source)))
            for syn in self.all_syns:
                self.w_mat[self.id2snk[syn.postID], self.id2src[syn.preID]] = abs(syn.w)
            self.w_mat /= np.amax(self.w_mat) # normalizing (maximum is 1)
            self.cmap = plt.get_cmap('Reds') # getting colormap
            #print(self.w_mat)
        else:
            update_fun = self.update_conn_anim
            
        if not slider:
            animation = FuncAnimation(self.conn_fig, update_fun, interval=interv, blit=True)
            return animation
        else:
            from ipywidgets import interact
            widget = interact(update_fun, frame=(1, max(self.len_source, self.len_sink)))
            return widget

    
    def update_conn_anim(self, frame): 
        sou_u = frame%self.len_source # source unit whose receivers we'll visualize
        snk_u = frame%self.len_sink # sink unit whose senders we'll visualize
    
        # PLOTTING THE RECEIVERS OF sou_u ON THE LEFT AXIS
        source_sizes = np.tile(2, self.len_source)
        sink_sizes = np.tile(2, self.len_sink)
        source_colors = np.tile(self.std_src,(self.len_source,1))
        sink_colors = np.tile(self.std_snk, (self.len_sink,1))
        source_sizes[sou_u] = 50
        source_colors[sou_u] = self.big_src
        # getting targets of projections from the unit 'sou_u'
        targets = self.id2snk[ [syn.postID for syn in self.all_syns if self.id2src[syn.preID] == sou_u ] ]
        # setting the colors and sizes
        sink_colors[targets] = self.big_snk
        sink_sizes[targets] = 15
        self.src_col1.set_sizes(source_sizes)
        #ax1.get_children()[0].set_sizes(source_sizes)   # sizes for the source units
        self.snk_col1.set_sizes(sink_sizes)   
        self.src_col1.set_color(source_colors)
        self.snk_col1.set_color(sink_colors)
        
        # PLOTTING THE SENDERS TO snk_u ON THE RIGHT AXIS
        source_sizes = np.tile(2, self.len_source)
        sink_sizes = np.tile(2, self.len_sink)
        source_colors = np.tile(self.std_src, (self.len_source,1))
        sink_colors = np.tile(self.std_snk, (self.len_sink,1))
        sink_sizes[snk_u] = 50
        sink_colors[snk_u] = self.big_snk
        # getting senders of projections to the unit 'snk_u'
        senders = self.id2src[ [syn.preID for syn in self.all_syns if self.id2snk[syn.postID] == snk_u] ]
        # setting the colors and sizes
        source_colors[senders] = self.big_src
        source_sizes[senders] = 15
        self.src_col2.set_sizes(source_sizes)
        self.snk_col2.set_sizes(sink_sizes)   
        self.src_col2.set_color(source_colors)
        self.snk_col2.set_color(sink_colors)
    
        return self.ax1, self.ax2,

 
    def update_weight_anim(self, frame): 
        sou_u = frame%self.len_source # source unit whose receivers we'll visualize
        snk_u = frame%self.len_sink # sink unit whose senders we'll visualize
            
        # PLOTTING THE RECEIVERS OF sou_u ON THE LEFT AXIS
        source_sizes = np.tile(2, self.len_source)
        sink_sizes =  2. + 400.*self.w_mat[:,sou_u]
        source_colors = np.tile(self.std_src,(self.len_source,1))
        sink_colors =  self.cmap.__call__(self.w_mat[:,sou_u])
        source_sizes[sou_u] = 320
        source_colors[sou_u] = self.big_src
        self.src_col1.set_sizes(source_sizes)
        self.snk_col1.set_sizes(sink_sizes)   
        self.snk_col1.set_color(sink_colors)
        self.src_col1.set_color(source_colors)
        
        # PLOTTING THE SENDERS TO snk_u ON THE RIGHT AXIS
        source_sizes = 2. + 400.*self.w_mat[snk_u,:]
        sink_sizes = np.tile(2, self.len_sink)
        source_colors = self.cmap.__call__(self.w_mat[snk_u,:])
        sink_colors = np.tile(self.std_snk, (self.len_sink,1))
        sink_sizes[snk_u] = 30
        sink_colors[snk_u] = self.big_snk
        self.src_col2.set_sizes(source_sizes)
        self.snk_col2.set_sizes(sink_sizes)   
        self.src_col2.set_color(source_colors)
        self.snk_col2.set_color(sink_colors)
    
        return self.ax1, self.ax2,
 

    def act_anim(self, pop, thr, interv=100, slider=False):
        """ An animation to visualize the activity of a set of units. 
            
            pop : an array or list with the IDs of the units to visualize.
            interv : refresh interval of the simulation.
            slider : When set to True the animation is substituted by a slider widget.
            thr : units whose activity surpasses 'thr' will be highligthted.
        """
        get_ipython().run_line_magic('matplotlib', 'qt5')
        # notebook or qt5 

        self.unit_fig = plt.figure(figsize=(10,10))
        self.ax = self.unit_fig.add_axes([0., 0., 1., 1.], frameon=False)
        xcoords = [ u.coordinates[0] for u in [self.net.units[i] for i in pop] ]
        ycoords = [ u.coordinates[1] for u in [self.net.units[i] for i in pop] ]
        min_x = min(xcoords) - 0.1;     max_x = max(xcoords) + 0.1
        min_y = min(ycoords) - 0.1;     max_y = max(ycoords) + 0.1
        self.ax.set_xlim(min_x, max_x), self.ax.set_xticks([])
        self.ax.set_ylim(min_y, max_y), self.ax.set_yticks([])
        self.scat = self.ax.scatter(xcoords, ycoords, s=20.*self.all_activs[pop,0])
        self.n_data = len(self.all_activs[0])
        self.act_thr = thr
        self.act_anim_pop = pop
        
        if not slider:
            animation = FuncAnimation(self.unit_fig, self.update_act_anim, 
                                  interval=interv, save_count=int(round(self.net.sim_time/self.net.min_delay)))    
            return animation
        else:
            from ipywidgets import interact
            widget = interact(self.update_act_anim, frame=(1,self.n_data))
            return widget
        
        
    def color_fun(self, activ):
        # given the units activity, maps into a color. activ may be an array.
        activ =  np.maximum(0.1,  activ*np.maximum(np.sign(activ - self.act_thr), 0.))
        return np.outer(activ, np.array([0., .5, .999, .5]))


    def update_act_anim(self, frame):
        # Each frame advances one simulation step (min_delay time units)
        idx = frame%self.n_data
        cur_time = self.net.min_delay*idx
        self.scat.set_sizes(300.*self.all_activs[self.act_anim_pop,idx])
        self.scat.set_color(self.color_fun(self.all_activs[self.act_anim_pop, idx]))
        self.unit_fig.suptitle('Time: ' + '{:f}'.format(cur_time))
        return self.ax,
    
  
    def hist_anim(self, pop, nbins=20, interv=100, slider=False, pdf=False):
        """ An animation to visualize the firing rate histogram of a given population through time. 
        
            pop : list or array with the IDs of the units whose histogram we'll visualize.
            interv : refresh interval of the simulation.
            slider : When set to True the animation is substituted by a slider widget.
            nbins : number of bins in the histogram
            pdf: include a plot of the exponential PDF?
                 Used when visualizing exp_dist_sig units or exp_rate_dist synapses.
                 CURRENTLY ONLY SUPPORTED WHEN pop IS HOMOGENOUS
        """
        get_ipython().run_line_magic('matplotlib', 'qt5')
        # notebook or qt5 
        self.hist_fig = plt.figure(figsize=(10,10))
        if pdf: # assuming pop consists of excitatory units
            #if self.net.units[pop[0]].type in trdc_u:
            if hasattr(self.net.units[pop[0]], 'c'):
                c = self.net.units[pop[0]].c
            else:
                c = self.e_syn['c']
            # plot the exponential distro on the same figure
            k = len(pop) * c / (1. - np.exp(-c)) / nbins
            plt.plot(np.linspace(0,1,100), k * np.exp(-c*np.linspace(0,1,100)))
            upper_tick = round(max(1., np.exp(-c)) * k)  + 5.
        else:
            upper_tick = len(pop)
        # initial histogram plot
        hist, bins, self.patches = plt.hist(self.all_activs[pop,0], bins=nbins, range=(0.,1.001))
        plt.yticks([0.,upper_tick])
        
        self.n_data = len(self.all_activs[0])
        self.n_bins = nbins
        self.ha_pop = pop
                
        if not slider:
            animation = FuncAnimation(self.hist_fig, self.update_hist_anim, 
                                  interval=interv, save_count=int(round(self.net.sim_time/self.net.min_delay)))    
            return animation
        else:
            from ipywidgets import interact
            widget = interact(self.update_hist_anim, frame=(1,self.n_data))
            return widget


    def update_hist_anim(self, frame):
        # Each frame advances one simulation step (min_delay time units)
        idx = frame%self.n_data
        cur_time = self.net.min_delay*idx
        binz, valz = np.histogram(self.all_activs[self.ha_pop, idx], bins=self.n_bins, range=(0.,1.001))
        for count, patch in zip(binz, self.patches):
            patch.set_height(count)
        self.hist_fig.suptitle('Time: ' + '{:f}'.format(cur_time))
        return 
 

    def double_anim(self, pop, nbins=20, interv=100, slider=False, thr=0.9, pdf=False):
        """ An animation of both firing rate histograms and unit activities for a given population.
        
            pop : list or array with the IDs of the units to visualize.
            bins = number of bins in the firing rate histogram
            inter = refresh interval in ms
            slider = whether to use an interactive slider instead of an animation
            thr = value at which the dots change color in the unit activity diagram
            pdf = whether to overlay a plot of the exponential pdf the histogram should approach.
                 CURRENTLY ONLY SUPPORTED WHEN pop IS HOMOGENEOUS (units are of the same type)
                  
        """
        get_ipython().run_line_magic('matplotlib', 'qt5')
        
        self.double_fig = plt.figure(figsize=(16,8))
        self.n_bins = nbins
        self.act_thr = thr
        self.n_data = len(self.all_activs[0])
        # Histogram figure and axis
        self.hist_ax = self.double_fig.add_axes([0.02, .04, .47, .92])
        if pdf: # assuming pop consists of units of the same type
            #if self.net.units[pop[0]].type in trdc_u:
            if hasattr(self.net.units[pop[0]], 'c'):
                c = self.net.units[pop[0]].c
            else:
                c = self.e_syn['c']
            # plot the exponential distro on the left axis
            k = len(pop) * c / (1. - np.exp(-c)) / nbins
            self.hist_ax.plot(np.linspace(0,1,100), k * np.exp(-c*np.linspace(0,1,100)), 'y')
            y_lim = round(max(1., np.exp(-c)) * k)  + 5.
        else:
            y_lim = len(pop)
        hist, bins, self.patches = self.hist_ax.hist(self.all_activs[pop,0], bins=nbins, range=(0.,1.001))
        self.hist_ax.set_ylim(top=y_lim) 
        # Activity figure and axis

        self.act_ax = self.double_fig.add_axes([.51, .02, .47, .94], frameon=False)

        xcoords = [ u.coordinates[0] for u in [self.net.units[i] for i in pop] ]
        ycoords = [ u.coordinates[1] for u in [self.net.units[i] for i in pop] ]
        min_x = min(xcoords) - 0.1;     max_x = max(xcoords) + 0.1
        min_y = min(ycoords) - 0.1;     max_y = max(ycoords) + 0.1
        self.act_ax.set_xlim(min_x, max_x), self.act_ax.set_xticks([])
        self.act_ax.set_ylim(min_y, max_y), self.act_ax.set_yticks([])

        #self.act_ax.set_xlim(-1, 1), self.act_ax.set_xticks([])
        #self.act_ax.set_ylim(-1, 1), self.act_ax.set_yticks([])
        #xcoords = [ u.coordinates[0] for u in [self.net.units[i] for i in pop] ]
        #ycoords = [ u.coordinates[1] for u in [self.net.units[i] for i in pop] ]
        self.scat = self.act_ax.scatter(xcoords, ycoords, s=20.*self.all_activs[pop,0])
        
        self.da_pop = pop
        # select viewing mode
        if not slider:
            animation = FuncAnimation(self.double_fig, self.update_double_anim, 
                                  interval=interv, save_count=int(round(self.net.sim_time/self.net.min_delay)))    
            return animation
        else:
            from ipywidgets import interact
            widget = interact(self.update_double_anim, frame=(1,self.n_data))
            return widget
    

    def update_double_anim(self, frame):
        # Each frame advances one simulation step (min_delay time units)
        idx = frame%self.n_data
        cur_time = self.net.min_delay*idx
        # updating histogram
        binz, valz = np.histogram(self.all_activs[self.da_pop,idx], bins=self.n_bins, range=(0.,1.001))
        for count, patch in zip(binz, self.patches):
            patch.set_height(count)
        # updating activities
        self.scat.set_sizes(300.*self.all_activs[self.da_pop,idx])
        self.scat.set_color(self.color_fun(self.all_activs[self.da_pop,idx]))
        self.double_fig.suptitle('Time: ' + '{:f}'.format(cur_time))
        return self.hist_ax, self.act_ax,
 
         
    def annotate(self, string, make_history=False):
        """ Append a string to self.notes and optionally to self.history. """
        self.notes += '# NOTE: ' + string + '\n'
        if make_history:
            self.history.append('# '  + string)
            

    def log(self, name=None, params=True):
        """ Write the history and notes of the ei_network object in a text file.
        
            name : A string with the name and path of the file to be written.
            params : A boolean. Whether to include the parameter dictionaries of ei_layer objects.
        """
        if not name:
            name = 'ei_network_log' + time.strftime("_%m-%d-%y.txt")
        with open(name, 'a') as f:
            f.write('\n')
            f.write('#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            f.write('#---------LOG OF ei_network OBJECT---------\n')
            f.write('#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            f.write('# NOTES #\n')
            f.write(self.notes)
            f.write('\n')
            f.write('# HISTORY #\n')
            in_dictionaries = False # A flag indicating the entries are parameter dictionaries
            for entry in self.history:
                if params:
                    f.write(entry + '\n')
                else:
                    match = re.search("#=#=#=", entry)
                    if match:
                        in_dictionaries = not in_dictionaries
                    if not in_dictionaries:
                        f.write(entry + '\n')
            f.write('\n')
            f.write('#()()()()()()()()()()()()()()()()()()\n')
            f.write('# ........... Individual layer logs .............')
            f.close() # TODO: not ideal to open and close the file for each layer...
            for lay_name in self.layers:
                self.layers[lay_name].log(name=name, params=params)
        
    def save(self, name=None):
        """ Save the ei_network object as a pickle.

            A draculab network contains lists with functions, so it is not picklable. 
            But it can be serialized with dill: https://github.com/uqfoundation/dill 
            
            After saving, retrieve object using, for example:
            F = open("ei_network_pickle.pkl", 'rb')
            ei = dill.load(F)
            F.close()
        """
        if not name:
            name = 'ei_network' + time.strftime("_%m-%d-%y.pkl")
        self.history.append('# ei_network object being saved as ' + name)
        F = open(name, 'wb')
        dill.dump(self, F)
        F.close()



class ei_layer():
    """
        This class is used to create, simulate, and visualize a generic network containing 3
        populations: excitatory (e), inhibitory (i), and input (x).  The three populations are 
        arranged in a rectangular grid. This class is used by the ei_network object, which 
        performs simulations of networks consisting of interconnected ei_layer objects.

        After creating an instance of the ei_layer class, the parameters can be configured 
        by changing the entries of the parameter dictionaries using the ei_layer.set_param 
        method. Once the parameter dictionaries are as desired the layer is created by running
        ei_layer.build . All units and connections are created using the topology module.
        In practice, the user does not call ei_layer.build, but ei_network.build, which 
        among other things calls ei_layer.build for all the layers in the network.

        An additional 'w_track' population of source units may also be created in order to
        track the temporal evolution of a random sample of synaptic weights.
        Similarly, 'sc_track' or 'thr_track' populations may be created in order to track
        the the values of some scale factors (when using exp_dist_sig units) or thresholds
        (when using exp_dist_sig_thr units).
        The number of 'w_track', 'sc_track', and 'thr_track' units is controlled by the
        'w_track' entry in the 'n' dictionary.

        METHODS OVERVIEW 
        __init__ : creates parameter dictionaries with the default values.
        set_param : changes a value in a parameter dictionary.
        build : creates units and connects them.
        annotate : append a line with text in the ei_layer.notes string.
        log : save the parameter changes and execution history of the network in a text file.
    """

    def __init__(self, name, net_pars):
        """ Create the parameter dictionaries required to build the excitatory-inhibitory network. 

            Args:
                name: a string with the name assigned to this layer by ei_network.
                net_pars: the parameter dictionary used for the draculab network.

            Time units are seconds for compatibility with the pendulum model.
            Length units for geometry dictionaries are centimeters.
            Axonal conduction speed is ~1 m/s, synaptic delay about 5 ms for the population response.

            The slope, thresh, tau, and init_val parameters of sigmoidal units are by default set 
            randomly using the formula:  par = par_min + par_wid * rand ,
            where par is either slope, thresh, tau, or init_val; par_min is the minimum value; 
            par_wid is the 'width' of the parameter distribution; and rand is a random value coming 
            from the uniform distribution in the [0,1) interval.
            To set a different distribution for these parameters, just set the desired value(s) in
            the 'slope', 'thresh', 'tau', or 'init_val' entry of the (e|i)_pars dictionary.

        """

        self.name = name
        # The history list will store strings containing the parameter dictionaries, the instructions doing 
        # parameter changes, building the network, or running a simulation. It also may contain notes.
        self.history = ["# ei_layer.__init__ at " + time.ctime()]
        self.notes = '' # comments about network configuration or simulation results.
            
        """ All the default parameter dictionaries are below.  """
        # NETWORK PARAMTERS
        self.net_params = net_pars
        # UNIT PARAMETERS
        self.e_geom = {'shape':'sheet', 
            'extent':[1.,1.], # 1 square centimeter grid
            'center':[0.,0.], 
            'arrangement':'grid', 
            'rows':16, 
            'columns':16,  
            'jitter' : 0.02 }
        self.i_geom = {'shape':'sheet', 
            'extent':[1.,1.], # 1 square centimeter grid
            'center':[0.,0.], 
            'arrangement':'grid', 
            'rows':8, 
            'columns':8,  
            'jitter' : 0.02 }
        self.x_geom = {'shape':'sheet', 
            'extent':[1.,1.], # 1 square centimeter grid
            'center':[0.,0.], 
            'arrangement':'grid', 
            'rows':1, 
            'columns':1,  
            'jitter' : 0.0 }
        self.n = {'e' : self.e_geom['rows'] * self.e_geom['columns'], # number of excitatory units
            'i' : self.i_geom['rows'] * self.i_geom['columns'],  # number of inhibitory units
            'x' : self.x_geom['rows'] * self.x_geom['columns'],  # number of input units
            'w_track': 6 } # number of weight tracking units
        self.e_pars = {'init_val_min' : 0.001,
            'init_val_wid' : .99,
            'slope_min' : 0.5, # minimum slope
            'slope_wid' : 3.,  # 'width' of the slope distribution
            'thresh_min' : -0.2,
            'thresh_wid' : 0.6,
            'tau_min' : 0.01,  # 10 ms minimum time constant
            'tau_wid' : 0.04,  # 40 ms variation
            'tau_fast' : 0.04, # 40 ms for fast low-pass filter
            'tau_mid' : .1, # 100 ms for medium low-pass filter
            'tau_slow' : 1, # 1 s for slow low-pass filter
            'tau_scale' : 0.05, # for exp_dist_sigmoidal units
            'tau_relax' : 1., # for ssrdc_sharp units
            'tau_thr' : 0.001, # for exp_dist_sig_thr and other trdc or hr units
            'c' : 2., # for exp_dist_sigmoidal and exp_dist_sig_thr units 
            'Kp' : 0.05, # for exp_dist_sigmoidal units
            'omega' : 1.5, # for sq_hebb_subsnorm synapses
            'n_ports' : 1, # multiport units will alter this
            'branch_params' : { # for the double_sigma family of units
                    'branch_w' : [1.],
                    'slopes' : 1.,
                    'threshs' : 0. },
            'phi' : 0.5, # for some of the double_sigma units
            'rdc_port' : 0, # for multiport units with rate distribution control
            'hr_port' : 0, # for harmonic rate units
            'thr_fix' : 0.1, # for the "sharpening" units
            'tau_fix' : 0.1, # for the "sharpening" units
            'sharpen_port' : 1, # for the "sharpening" units
            'type' : unit_types.sigmoidal }
        self.i_pars = {'init_val_min' : 0.001,
            'init_val_wid' : .99,
            'slope_min' : 0.5, # minimum slope
            'slope_wid' : 3.,  # 'width' of the slope distribution
            'thresh_min' : -0.2,
            'thresh_wid' : 0.6,
            'tau_min' : 0.01,  # 10 ms minimum time constant
            'tau_wid' : 0.04,  # 40 ms variation
            'tau_fast' : 0.04, # 40 ms for fast low-pass filter
            'tau_mid' : .1, # 100 ms for medium low-pass filter
            'tau_slow' : 1, # 1 s for slow low-pass filter
            'tau_scale' : 0.05, # for exp_dist_sigmoidal units
            'tau_relax' : 1., # for ssrdc_sharp units
            'tau_thr' : 0.001, # for exp_dist_sig_thr and other trdc or hr units
            'c' : 2., # for exp_dist_sigmoidal and exp_dist_sig_thr units
            'Kp' : 0.05, # for exp_dist_sigmoidal units
            'omega' : 1.5, # for sq_hebb_subsnorm synapses
            'n_ports' : 1, # multiport units will alter this
            'branch_params' : { # for the double_sigma family of units
                    'branch_w' : [1.],
                    'slopes' : 1.,
                    'threshs' : 0. },
            'phi' : 0.5, # for some of the double_sigma units
            'rdc_port' : 0, # for multiport units with rate distribution control
            'hr_port' : 0, # for harmonic rate units
            'thr_fix' : 0.1, # for the "sharpening" units
            'tau_fix' : 0.1, # for the "sharpening" units
            'sharpen_port' : 1, # for the "sharpening" units
            'type' : unit_types.sigmoidal }
        self.x_pars = {'type' : unit_types.source,
            'init_val' : 0.5,
            'tau_fast' : 0.04,
            'tau_mid' : .1, # 100 ms for medium low-pass filter
            'tau_slow' : 1, # 1 s for slow low-pass filter
            'function' : lambda x: None }
        self.track_pars = {'type' : unit_types.source,  # parameters for "tracking" units
            'init_val' : 0.,
            'tau_fast' : 0.04,
            'function' : lambda x: None }
        # CONNECTION PARAMETERS
        self.ee_conn = {'connection_type' : 'divergent',
            'mask' : {'circular': {'radius': .3}}, 
            'kernel' : .9,
            'delays' : {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}},
            'weights' :{'uniform' : {'low': 0.01, 'high' : 0.3}},
            'allow_autapses' : True,
            'allow_multapses' : False,
            'edge_wrap' : True,
            'boundary' : {'center': np.array(self.e_geom['center']), 'extent' : self.e_geom['extent']} }
        self.ei_conn = {'connection_type' : 'divergent',
            'mask' : {'circular': {'radius': .3}}, 
            'kernel' : .9,
            'delays' : {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}},
            'weights' :{'uniform' : {'low': 0.01, 'high' : 0.3}},
            'allow_autapses' : True,
            'allow_multapses' : False,
            'edge_wrap' : True,
            'boundary' : {'center': np.array(self.i_geom['center']), 'extent' : self.i_geom['extent']} }
        self.ie_conn = {'connection_type' : 'divergent',
            'mask' : {'circular': {'radius': .3}}, 
            'kernel' : .9,
            'delays' : {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}},
            'weights' :{'uniform' : {'low': -1.3, 'high' : -0.01}},
            'allow_autapses' : True,
            'allow_multapses' : False,
            'edge_wrap' : True,
            'boundary' : {'center': np.array(self.e_geom['center']), 'extent' : self.e_geom['extent']} }
        self.ii_conn = {'connection_type' : 'divergent',
            'mask' : {'circular': {'radius': .3}}, 
            'kernel' : .9,
            'delays' : {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}},
            'weights' :{'uniform' : {'low': -1., 'high' : -0.01}},
            'allow_autapses' : True,
            'allow_multapses' : False,
            'edge_wrap' : True,
            'boundary' : {'center': np.array(self.i_geom['center']), 'extent' : self.i_geom['extent']} }
        self.xe_conn = {'connection_type' : 'divergent',
            'mask' : {'circular': {'radius': .3}}, 
            'kernel' : 1.,
            'delays' : {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}},
            'weights' :{'uniform' : {'low': 0.05, 'high' : 4.}},
            'allow_autapses' : True,
            'allow_multapses' : False,
            'edge_wrap' : True,
            'boundary' : {'center': np.array(self.e_geom['center']), 'extent' : self.e_geom['extent']} }
        self.xi_conn = {'connection_type' : 'divergent',
            'mask' : {'circular': {'radius': .3}}, 
            'kernel' : 1.,
            'delays' : {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}},
            'weights' :{'uniform' : {'low': 0.05, 'high' : 4.}},
            'allow_autapses' : True,
            'allow_multapses' : False,
            'edge_wrap' : True,
            'boundary' : {'center': np.array(self.i_geom['center']), 'extent' : self.i_geom['extent']} }
        self.ee_syn = {'type' : synapse_types.static,  # excitatory to excitatory synapses
            'lrate' : 0.1,  # for all dynamic synapse types
            'omega' : 1.,  # for sq_hebb_subsnorm synapses 
            'input_type' : 'pred',  # for input_correlation synapses
            'des_act' : 0.4, # for homeo_inhib, and corr_homeo_inhib synapses
            'inp_ports' : 0, # for multiport units
            'c' : 1., # for exp_rate_dist synapses
            'wshift' : 1. } # for exp_rate_dist synapses
        self.ei_syn = {'type' : synapse_types.static,
            'lrate' : 0.1,  # for all dynamic synapse types
            'omega' : 1.,  # for sq_hebb_subsnorm synapses 
            'input_type' : 'pred',  # for input_correlation synapses
            'des_act' : 0.4, # for homeo_inhib, and corr_homeo_inhib synapses
            'inp_ports' : 0, # for multiport units
            'c' : 1., # for exp_rate_dist synapses
            'wshift' : 1. } # for exp_rate_dist synapses
        self.ie_syn = {'type' : synapse_types.static,
            'lrate' : 0.1,  # for all dynamic synapse types
            'omega' : 1.,  # for sq_hebb_subsnorm synapses 
            'input_type' : 'pred',  # for input_correlation synapses
            'des_act' : 0.4, # for homeo_inhib, and corr_homeo_inhib synapses
            'inp_ports' : 0, # for multiport units
            'c' : 1., # for exp_rate_dist synapses
            'wshift' : 1. } # for exp_rate_dist synapses
        self.ii_syn = {'type' : synapse_types.static,
            'lrate' : 0.1,  # for all dynamic synapse types
            'omega' : 1.,  # for sq_hebb_subsnorm synapses 
            'input_type' : 'pred',  # for input_correlation synapses
            'des_act' : 0.4, # for homeo_inhib, and corr_homeo_inhib synapses
            'inp_ports' : 0, # for multiport units
            'c' : 1., # for exp_rate_dist synapses
            'wshift' : 1. } # for exp_rate_dist synapses
        self.xe_syn = {'type' : synapse_types.static,
            'lrate' : 0.1,  # for all dynamic synapse types
            'omega' : 1.,  # for sq_hebb_subsnorm synapses 
            'input_type' : 'pred',  # for input_correlation synapses
            'des_act' : 0.4, # for homeo_inhib, and corr_homeo_inhib synapses
            'inp_ports' : 0, # for multiport units
            'c' : 1., # for exp_rate_dist synapses
            'wshift' : 1. } # for exp_rate_dist synapses
        self.xi_syn = {'type' : synapse_types.static,
            'lrate' : 0.1,  # for all dynamic synapse types
            'omega' : 1.,  # for sq_hebb_subsnorm synapses 
            'input_type' : 'pred',  # for input_correlation synapses
            'des_act' : 0.4, # for homeo_inhib, and corr_homeo_inhib synapses
            'inp_ports' : 0, # for multiport units
            'c' : 1., # for exp_rate_dist synapses
            'wshift' : 1. } # for exp_rate_dist synapses
        

    def build(self, the_net):
        """ Add this layer to the draculab network. 
        
            The draculab network is recevied as an argument.
        """
        # store record of network being built
        self.history.append('build()')
        # Set parameters derived from other parameters
        self.n['e'] =  self.e_geom['rows'] * self.e_geom['columns'] # number of e units
        self.n['i'] =  self.i_geom['rows'] * self.i_geom['columns']
        self.n['x'] =  self.x_geom['rows'] * self.x_geom['columns'] 

        self.ee_conn['delays'] = {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}}
        self.ei_conn['delays'] = {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}}
        self.ie_conn['delays'] = {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}}
        self.ii_conn['delays'] = {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}}
        self.xe_conn['delays'] = {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}}
        self.xi_conn['delays'] = {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}}

        if (self.ee_conn['edge_wrap'] == True or self.ei_conn['edge_wrap'] == True or
            self.ie_conn['edge_wrap'] == True or self.ii_conn['edge_wrap'] == True or
            self.xe_conn['edge_wrap'] == True or self.xi_conn['edge_wrap'] == True):
            assert self.e_geom['extent']==self.i_geom['extent'] and self.e_geom['center']==self.i_geom['center'], [
               'Excitatory and inhibitory grids can not have different shape when using periodic boundaries']
        self.ee_conn['boundary'] = {'center': np.array(self.e_geom['center']), 'extent' : self.e_geom['extent']}
        self.ei_conn['boundary'] = {'center': np.array(self.i_geom['center']), 'extent' : self.i_geom['extent']}
        self.ie_conn['boundary'] = {'center': np.array(self.e_geom['center']), 'extent' : self.e_geom['extent']}
        self.ii_conn['boundary'] = {'center': np.array(self.i_geom['center']), 'extent' : self.i_geom['extent']}
        self.xe_conn['boundary'] = {'center': np.array(self.e_geom['center']), 'extent' : self.e_geom['extent']}
        self.xi_conn['boundary'] = {'center': np.array(self.i_geom['center']), 'extent' : self.i_geom['extent']}

        # Before we populate the parameter dictionaries with random values, let's print them to history
        self.history.append('#=#=#=#=#=# PARAMETER DICTIONARIES OF ' + self.name + ' LAYER  #=#=#=#=#=#')
        # ei_layer.log uses the '#=#=#=' sequence to recognize where the parameter dictionaries' section
        # begins and ends in ei_layer.history . Thus, this decoration shouldn't be removed.
        pp = pprint.PrettyPrinter(indent=4, compact=True)
        for name in vars(self):
            attr = self.__getattribute__(name)
            if type(attr) is dict:
                self.history.append(name + ' = ' + pp.pformat(attr)+'\n')
        self.history.append('#=#=#=#=#=#=#=#=#=#=#')

        # Now we populatate the parameter dictionaries with the appropriate random values
        for par in ['slope', 'thresh', 'tau', 'init_val']:
            if not par in self.e_pars:
                self.e_pars[par] = self.e_pars[par+'_min'] + self.e_pars[par+'_wid'] * np.random.random(self.n['e'])
            if not par in self.i_pars:
                self.i_pars[par] = self.i_pars[par+'_min'] + self.i_pars[par+'_wid'] * np.random.random(self.n['i'])

        # Create an auxiliary topology object
        topo = topology()
        # Assign the network
        self.net = the_net
        # Create unit groups
        self.e = topo.create_group(self.net, self.e_geom, self.e_pars)
        self.i = topo.create_group(self.net, self.i_geom, self.i_pars)
        if self.n['x'] > 0: # if we have external inputs
            self.x = topo.create_group(self.net, self.x_geom, self.x_pars)
        # Create weight tracking units
        if self.n['w_track'] > 0:
            self.w_track = self.net.create(self.n['w_track'], self.track_pars)
        # Create connections
        topo.topo_connect(self.net, self.e, self.e, self.ee_conn, self.ee_syn)
        topo.topo_connect(self.net, self.e, self.i, self.ei_conn, self.ei_syn)
        topo.topo_connect(self.net, self.i, self.e, self.ie_conn, self.ie_syn)
        topo.topo_connect(self.net, self.i, self.i, self.ii_conn, self.ii_syn)
        if self.n['x'] > 0: # if we have external inputs
            topo.topo_connect(self.net, self.x, self.i, self.xi_conn, self.xi_syn)
            topo.topo_connect(self.net, self.x, self.e, self.xe_conn, self.xe_syn)
        # Set the function of the w_track units to follow randomly chosen weights.
        if self.n['w_track'] > 0:
            n_u = min(len(self.e), int(np.floor(np.sqrt(self.n['w_track'])))) # For how many units we'll track weights
            n_syns, remainder = divmod(self.n['w_track'], n_u) # how many synapses per unit
            which_u = np.random.choice(self.e+self.i, n_u) # ID's of the units we'll track
            self.tracked = which_u # so basic_plot knows which units we tracked
            which_syns = [[] for i in range(n_u)]
            for uid,u in enumerate(which_u[0:-1]):
                which_syns[uid] = np.random.choice(range(len(self.net.syns[u])), n_syns)
            which_syns[n_u-1] = np.random.choice(range(len(self.net.syns[which_u[-1]])), n_syns+remainder)
            for uid,u in enumerate(which_u):
                for sid,s in enumerate(which_syns[uid]):
                    self.net.units[self.w_track[uid*n_syns+sid]].set_function(self.net.syns[u][s].get_w)
            # If there are ssrdc units, create some units to track their scale factors
            if self.e_pars['type'] in ssrdc_u or self.i_pars['type'] in ssrdc_u:
                self.sc_track = self.net.create(self.n['w_track'], self.track_pars)
                def scale_tracker(u,s):
                    if hasattr(self.net.units[u], 'scale_facs'):
                        return lambda x: self.net.units[u].scale_facs[s]
                    elif hasattr(self.net.units[u], 'scale_facs_rdc'):
                        syn = s%(len(self.net.units[u].scale_facs_rdc))
                        return lambda x: self.net.units[u].scale_facs_rdc[syn]
                    elif hasattr(self.net.units[u], 'scale_facs_hr'):
                        syn = s%(len(self.net.units[u].scale_facs_hr))
                        return lambda x: self.net.units[u].scale_facs_hr[syn]
                    else:
                        return lambda x: 1.
                for uid,u in enumerate(which_u):
                    for sid,s in enumerate(which_syns[uid]):
                        self.net.units[self.sc_track[uid*n_syns+sid]].set_function(scale_tracker(u,s))
            # If there are trdc units, create some units to track the thresholds
            if self.e_pars['type'] in trdc_u or self.i_pars['type'] in trdc_u:
                self.thr_track = self.net.create(self.n['w_track'], self.track_pars)
                def thresh_tracker(u):
                    return lambda x: self.net.units[u].thresh
                for uid,u in enumerate(which_u):
                    self.net.units[self.thr_track[uid]].set_function(thresh_tracker(u))
        else:
            self.tracked = self.e[0:min(2,self.n['e'])]
            self.tracked += self.i[0:min(2,self.n['i'])]

        # If there is a delta unit, create tracking units for its learning and error variables
        delta_u = [unit_types.delta_linear]
        if self.e_pars['type'] in delta_u: 
            self.learn_track = self.net.create(1, self.track_pars)
            self.error_track = self.net.create(1, self.track_pars)
            def create_lt(unit_id):
                return lambda t : self.net.units[unit_id].learning
            def create_et(unit_id):
                return lambda t : self.net.units[unit_id].error
            for unit in self.learn_track:
                self.net.units[unit].set_function(create_lt(self.e[0]))
            for unit in self.error_track:
                self.net.units[unit].set_function(create_et(self.e[0]))


    def set_param(self, dictionary, entry, value):
        """ Change a value in a parameter dictionary. 
        
            dictionary: a string with the name of one of the parameter dictionaries in __init__
            entry: a string with the name of the entry we'll modify
            value: the value we'll assign (can be any appropriate Python object) 
        """
        if 'build()' in self.history:
            raise AssertionError('Setting a parameter value after network has been built')     
        self.__getattribute__(dictionary)[entry] = value  # set the new value
        self.history.append(dictionary + '[\'' + entry + '\'] = ' + str(value) ) # record assignment
            
   
    def annotate(self, string, make_history=False):
        """ Append a string to self.notes and optionally to self.history. """
        self.notes += '# NOTE (' + self.name + ') : ' + string + '\n'
        if make_history:
            self.history.append('# (' + self.name + ')'  + string)
            
    def log(self, name=None, params=False):
        """ Write the history and notes of the ei_layer object in a text file.
        
            name : A string with the name and path of the file to be written.
            params : A boolean to indicate including the parameter dictionaries used to build the object. 
        """
        if not name:
            name = self.name + '_log.txt'
        with open(name, 'a') as f:
            f.write('\n')
            #f.write('#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            f.write('#---------Log of ' + self.name + ' layer ---------\n')
            #f.write('#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            if len(self.notes) > 0:
                f.write('# NOTES #\n')
                f.write(self.notes)
                f.write('\n')
            f.write('# HISTORY #\n')
            in_dictionaries = False # A flag indicating the entries are parameter dictionaries
            for entry in self.history:
                if params:
                    f.write(entry + '\n')
                else:
                    match = re.search("#=#=#=", entry)
                    if match:
                        in_dictionaries = not in_dictionaries
                    if not in_dictionaries:
                        f.write(entry + '\n')
            f.write('\n')
            f.close()
        

