

class ei_network():
    """ 
        This class is used to create, simulate, and visualize networks consisting of
        interconnected ei_layer objects, which are generic networks of excitatory and
        inhibitory units with optional inputs.

        A standard way to use ei_network is as follows.
        1) The class creator receives a list of strings with the names of the layers to be created.
           The creator will create layers with those names using standard parameters.
        2) For each pair of layers to be connected, the user calls ei_network.add_layer_conn,
           which creates standard parameter dictionaries for the connection, and signals the
           ei_network.build object that the connection should be created.
        3) The user configures the parameter dictionaries for the inter-layer connections using
           ei_network.set_params
        4) The user sets the parameters of each layer by first retrieving the layer object
           using the ei_network.get_layer method, and then using the ei_layer.set_param method 
           (this could also be done before steps 2 or 3).
        5) The user calls ei_network.build method, which creates a draculab network with all the
           units and connections specified so far.
        6) The user runs simulations with ei_network.run .
        7) The user can use the tools to visualize, save, annotate, log, or continue the simulations.


    """
        
    def __init__(self, layers, net_number=None):
        """

            The argument net_number is an integer to identify the object in multiprocess simulations;
        """

        self.history = ["# __init__ at " + time.ctime()]
        self.notes = '' # comments about network configuration or simulation results.
        # fixing random seed
        seed = 19680801
        np.random.seed(seed)
        self.history.append('np.random.seed(%d)'  % (seed))

        self.net_number = net_number
        if net_number:
            self.history.append('# parallel runner assigned this network the number ' + str(net_number))
        # PARAMETER DICTIONARY FOR THE NETWORK
        self.net_params = {'min_delay' : 0.005, 
            'min_buff_size' : 10,
            'rtol' : 1e-5,
            'atol' : 1e-5,
            'cm_del' : .01 }  # delay in seconds for each centimeter in distance. 



