"""
topology.py
This files contains the class topology, which includes two methods that
create spatially structured connections in Draculab networks.

It implements functionality similar to that in NEST's topology module.
"""

from sirasi import unit_types, synapse_types, plant_models, syn_reqs  # names of models and requirements
from units import *
from synapses import *
from network import *
import numpy as np
import random # to create random connections


class topology():
    """ This class has tools to create spatially structured Draculab networks.

        It is based on NEST's topology module.
    """

    def create_group(self, net, geometry, params):
        """ 
        Create a group of units with a particular spatial arrangement.

        This method uses the specifications given in the 'geometry' argument to decide
        how many units to create, and assign them coordinates. It then calls network.create_units.
        This is not like NEST's topology.CreateLayer method. In here we are just creating
        units with particular coordinates, no other objects are involved.

        The units of measurement (e.g. millimeters) are unspecified; the implementer is
        responsible for their consistency.
        
        The current implementation can only do one thing: create flat 2D layers with a given
        number of rows and columns.

        Args:
            net : The network where the units will be created.

            geometry : A dictionary specifying the spatial location of the units.
                REQUIRED ENTRIES
                'shape': a string with the geometrical shape of the space where the units will be placed. 
                         Currently accepted values: 
                         'sheet': a flat 2D rectangular surface. The geometry dictionary should
                                  also include an 'extent' property, which is a 2D list
                                  with the size of this sheet. If 'center' is 3D, its first 2
                                  coordinates will be used as the center of the sheet; other
                                  coordinates are currently ignored.
                'arrangement':  A string specifying how the units will fill the shape.
                                Valid values:
                                'grid': Grid arrangement. The geometry dictionary must also contain the integer
                                    entries 'rows' and 'columns'. The first coordinates (e.g. x coordinate) of
                                    all units will come from putting 'columns' points inside an interval of 
                                    length equal to the first 'extent' value; the distance between 
                                    contiguous points is twice the distance of the first and last points
                                    to the edge of the interval. The second coordinates of all units will be
                                    created similarly using 'rows'.
                'center': a list or array with the coordinates for the geometrical center of the space where
                          the units will be placed.
                CONTINGENT ENTRIES
                'extent' : see 'shape'>'sheet'.
                'rows','columns': see 'arrangement'>'grid'.

            params: a dictionary with the parameters used to initialize the units.
                REQUIRED PARAMETERS
                'type' : a unit model form the unit_types enum.
                'init_val' : initial activation value (also required for source units).
                Other required entries depend on the specific unit model being created.
                                        
        Returns: a list with the ID's of the created units.

        Raises:
            ValueError, TypeError, NotImplementedError, KeyError
        """
        if geometry['shape'] == 'sheet':
            if geometry['arrangement'] == 'grid':
                # generate a list with the coodinates for each point of the grid
                xbit = geometry['extent'][0]/geometry['columns'] # size of intervals on x-axis
                ybit = geometry['extent'][1]/geometry['rows']    # size of intervals on y-axis
                x_vals = np.linspace(xbit/2., geometry['extent'][0]-(xbit/2), geometry['columns'])
                y_vals = np.linspace(ybit/2., geometry['extent'][1]-(ybit/2), geometry['rows'])
                x_vals = [ x + geometry['center'][0] - geometry['extent'][0]/2. for x in x_vals ]
                y_vals = [ y + geometry['center'][1] - geometry['extent'][1]/2. for y in y_vals ]
                
                coord_list = []
                for x in x_vals:
                    for y in y_vals:
                        coord_list.append(np.array([x,y]))
            else:
                raise NotImplementedError('Unknown geometry arrangement in create_group')
        else:
            raise NotImplementedError('Unknown shape specification in create_group')

        params['coordinates'] = coord_list
        return net.create_units(len(coord_list), params)


    def topo_connect(self, net, from_list, to_list, conn_spec, syn_spec):
        """ 
        Connect two groups of units based on their spatial locations.

        This method is a version of network.connect() with two modifications:
        1) All units involved must have a valid 'coordinates' attribute. One way to set the 
           coordinates is by using the create_group() method instead of network.create().
        2) The conn_spec dictionary uses a set of entries different to network.connect(). These 
           entries specify the spatial connection profile using almost the same options as PyNEST's 
           topology module.

        topo_connect() works by first specifing which units to connect, with which weights,
        and with which delays. Using this it calls the network.connect() method. The same syn_spec
        dictionary is used when calling network.connect(), but the init_w values may be modified. 
        
        Args:
            net : The network where the units will be connected.

            from_list: A list with the IDs of the units sending the connections

            to_list: A list the IDs of the units receiving the connections

            conn_spec: A dictionary specifying a connection profile.
                REQUIRED ENTRIES
                'connection_type' : A string that specifies in which group to apply the mask and kernel.
                        Accepted values: 
                        'convergent': When selecting units to connect, for each unit of the 'to_list'
                            we will select input sources from the 'from_list' among those units allowed
                            by the mask and the kernel. NOT IMPLEMENTED YET.
                        'divergent': When selecting units to connect, for each unit of the 'from_list'
                            we will select connection recipients from the 'to_list' among those units
                            within the mask and the kernel.
                'mask' : A dictionary that specifies a set of units that are eligile for receiving
                         (when using the 'divergent' rule) or sending (when using the 'convergent'
                         rule) connections. The regions specified in this dictionary are centered on the
                         coordinates of each sending unit for divergent connections, and on those of
                         each receving unit for convergent connections.
                         The value should be one of the following dictionaries:
                         >> {'circular': {'radius': r}}. Here r is a scalar value. Units at a distance
                            smaller than r will be inside the mask.
                         >> {'rectangular': {'lower_left': ll, 'upper_right': ur}}. ll and ur are
                            lists with the coordinates of the lower left and upper right coordinates of 
                            the rectangular mask, with the origin at the coordinates of the sending 
                            (for divergent connections) or receiving (for convergent connections) unit.
                         >> {'annular': {'inner_radius': ir, 'outer_radius': or}}. Here ir and or are
                            scalars. Units with distance d to the center of the mask will be inside 
                            when ir <= d <= or.
                'kernel' :  For each pair of units eligible for connection (e.g. one is within the mask), 
                          the kernel rule assigns a probability for connecting them, based on some spatial
                          criterion. The center of the kernel is at the sending unit for divergent connections,
                          and at the receiving unit for convergent connections.
                          Accepted values are either a scalar with the probability of connection for each
                          eligible unit inside the mask, or one of the following dictionaries.
                          >> {'gaussian' : {'p_center': p, "sigma": s}}. In this case each eligible
                              unit that is a distance 'd' from the center of the kernel will connect with
                              probability  p*exp(- (d/s)**2 ).
                          >> {'linear' : {'c' : c, 'a' : a}}. Units at a distance 'd' from the center of the 
                              kernel will connect with probability max( c - a*d, 0 )
                          Notice that the 'number_of_connections' option below can  affect these probabilities.
                'delays' : A dictionary that specifies the delay of each connection as a function of
                          the distance between the sending and the receiving unit. It should contain
                          the following dictionary:
                          >> {'linear' :  'c' : c, 'a' : a}}. The connection between units with distance 'd'
                              will have a delay equal to [c + a*d], where the square brackets indicate that 
                              this value will be transformed to the nearest multiple of the minimum 
                              transmission delay. Notice a=0 implies all delays will have value [c].
                OPTIONAL ENTRIES
                'allow_autapses' : Can units connect to themselves? Default is True.
                'allow_multapses' : Can a unit have more than one connection to the same target?
                                    Default is False.
                'number_of_connections' : Fixes how many connections each neuron will send (divergent
                                          connections) or receive (convergent connections).
                                          Notice that using this option will distort the probabilities
                                          of connection set in the 'kernel' entry.
                'weights' : A dictionary specifying a distribution for the initial weights of the connections.
                            The accepted distributions are:
                           >> {'uniform' : {'low': l, 'high' : h}}. Here l and h are scalar values.
                              Same as setting a uniform distribution in 'init_w' of syn_spec, but
                              with different syntax in the dictionary.
                           >> {'linear' : {'c' : c, 'a' : a}}. Units with distance d will connect with
                              synaptic weight max( c  - a*d, 0 ) .
                           >> {'gaussian' : {'p_center' : w, 'sigma' : s}}. Units with distance d will
                              connect with synaptic weight w * exp( - (d/s)**2 ) . 

                'edge_wrap' : Assume that all units are inside a rectangle with periodic boundaries (e.g. 
                              toroidal topology). By default this is set to False, but if set to True the 
                              size and location of that rectangle must be set with the 'boundary' dictionary.
                              NOT IMPLEMENTED YET
                'boundary' : Specifies a rectangle that contains all units. The value should be a
                            dictionary of the form:
                            >> {'center' : c, 'extent' : [x, y]}.
                               In this case c represents a list or an array with the coordinates of the
                               boundary rectangle. x and y are scalars that specify its width and length.
                
            syn_spec: A dictionary used to initialize the synapses in the connections.
                REQUIRED PARAMETERS
                'type' : a synapse type from the synapse_types enum.
                'init_w' : Initial weight values. Either a dictionary specifying a distribution, or a
                        scalar value to be applied for all created synapses. Distributions:
                        'uniform' - the delay dictionary must also include 'low' and 'high' values.
                        Example: {..., 'init_w':{'distribution':'uniform', 'low':0.1, 'high':1.} }
                        Notice that these values will be overwritten whenever there is a 'weights'
                        entry in the conn_spec dictionary.
                OPTIONAL PARAMETERS
                'inp_ports' : input ports of the connections. Either a single integer, or a list.
                            If using a list, its length must match the number of connections being
                            created, which depends on the conection rule.
                         
        Raises:
            ValueError, NotImplementedError, TypeError

        """
        # TODO: it would be very easy to implement the 'targets' option by reducing the to_list to 
        # those units of a particular type at this point. I don't need that feature now, though.

        # TODO: depending on the 'edge_wrap' property you need to set a distance function that takes
        #       the two pairs of coordinates and returns the distance. The simple case is the one
        #       you're using below: d = np.sqrt( sum( (uc - to_c) * (uc - to_c) ) ). The case with
        #       periodic boundaries needs to find the shortest horizontal and vertical distances
        #       (so both are <= conn_spec[boundary][extent][0|1]/2) and return the sqrt of the sum
        #       of the squared distances. Replace the distance calculations below (also in kernel).
        if 'edge_wrap' in conn_spec:
            if conn_spec['edge_wrap'] == True:
                raise NotImplementedError('Periodic boundaries not implemented yet')

        connections = []  # a list with all the connection pairs as (source,target)
        distances = [] # the distance for the units in each entry of 'connections'
        delays = [] # a list with the delay for each entry in 'connections'

        # Utility function: Euclidean distance
        dist = lambda x,y: np.sqrt( sum( (x-y)*(x-y) ) )

        # setting some default values in conn_spec
        if not 'allow_autapses' in conn_spec: conn_spec['allow_autapses'] = True
        if not 'allow_multapses' in conn_spec: conn_spec['allow_multapses'] = False

        if conn_spec['connection_type'] == 'divergent':
            for idx in from_list:
                uc = net.units[idx].coordinates # center for the sending unit, mask, and kernel

                #### First we make a list with all the units in 'to_list' inside the mask
                masked = self.filter_ids(net, to_list, uc, conn_spec['mask'])
                
                #### Next we modify 'masked' to account for autapses and multapses
                if not conn_spec['allow_autapses']:  # autapses not allowed
                    if idx in masked:
                        masked = [ e for e in masked if e != idx ]
                if not conn_spec['allow_multapses']: # multapses not allowed
                    masked = list(set(masked))  # removing duplicates
                    # This tests if enough connections can be made without duplicates
                    if 'number_of_connections' in conn_spec:
                        if len(masked) < conn_spec['number_of_connections']:
                            raise ValueError('number_of_connections larger than number of targets within mask')
                         
                #### Now we use the kernel rule to select connections
                # first, make sure the number of potential targets is larger than number_of_connections
                if 'number_of_connections' in conn_spec:
                    kerneled = self.filter_ids(net, masked, uc, conn_spec['kernel'])
                    if len(kerneled) < conn_spec['number_of_connections']:
                        raise ValueError('number_of_connections is larger than number of targets with significant connection probability')

                # second, specify the kernel function
                if type(conn_spec['kernel']) is float or type(conn_spec['kernel']) is int:
                    # distance-independent constant probability of connection
                    kerfun = lambda d: conn_spec['kernel']
                elif 'linear' in conn_spec['kernel']:
                    a = conn_spec['kernel']['linear']['a']
                    c = conn_spec['kernel']['linear']['c']
                    kerfun = lambda d: c  - a*d
                elif 'gaussian' in conn_spec['kernel']:
                    p_c = conn_spec['kernel']['gaussian']['p_center']
                    s = conn_spec['kernel']['gaussian']['sigma']
                    kerfun = lambda d: p_c * np.exp( - ((d/s)**2.)) 
                else:
                    raise NotImplementedError('Invalid kernel specification')
                # third, do the sampling
                if 'number_of_connections' in conn_spec:
                    n_of_c = conn_spec['number_of_connections']
                    random.shuffle(masked) # to reduce dependency on order in masked
                else:
                    n_of_c = len(masked) + 1 # this value controls a 'break' below
                connected = 0 # to keep track of how many targets we've selected
                while connected < n_of_c:
                    for to_idx in masked:
                        to_c = net.units[to_idx].coordinates
                        d = dist(uc, to_c)
                        if random.random() < kerfun(d):
                            connections.append( (idx, to_idx) )
                            distances.append(d)
                            connected += 1
                            if not conn_spec['allow_multapses']:
                                masked.remove(to_idx)
                        if connected >= n_of_c:
                            break
                    if  not ('number_of_connections' in conn_spec):
                        break 

        elif conn_spec['connection_type'] == 'convergent':
            raise NotImplementedError('Convergent connections are not implemented')
        else:
            raise TypeError("Invalid connection_type")

        # setting the delays
        if 'linear' in conn_spec['delays']:
            c = conn_spec['delays']['linear']['c']
            a = conn_spec['delays']['linear']['a']
            for d in distances:
                # delays must be multiples of net.min_delay
                dely = c + a*d + 1e-7  # 1e-7 deals with a quirk of the % operator
                mod = dely % net.min_delay
                dely = dely - mod
                if mod > net.min_delay/2.:  # rounding to nearest multiple above
                    dely += net.min_delay
                dely = max(net.min_delay, dely)
                delays.append(dely)
        else:
            raise ValueError('Unknown delay specification')
        print('len delays = %f' % len(delays))
        print('len conns = %f' % len(connections))
        print('len dists = %f' % len(distances))

        # optional setting of weights
        if 'weights' in conn_spec:
            max_w = 100. # maximum weight value we'll tolerate
            weights = [] # a list with the weight for each entry in 'connections'
            if 'uniform' in conn_spec['weights']:
                l = conn_spec['weights']['uniform']['low']
                h = conn_spec['weights']['uniform']['high']
                weights = np.random.uniform[l, h, len(connections)]
            elif 'linear' in conn_spec['weights']:
                c = conn_spec['weights']['linear']['c']
                a = conn_spec['weights']['linear']['a']
                weights = list(map(lambda d: max(c - a*d, 0.), distances))
            elif 'gaussian' in conn_spec['weights']:
                w = conn_spec['weights']['gaussian']['p_center']
                s = conn_spec['weights']['gaussian']['sigma']
                weights = [ w*np.exp(-((d/s)**2.)) for d in distances]
            if max(weights) > max_w:
                raise ValueError('Received weights distribution produces values larger than ' + str(max_w))

            syn_spec['init_w'] = weights

        # creating the connections
        senders = [c[0] for c in connections]
        receivers = [c[1] for c in connections]
        conn_dict = {'rule' : 'one_to_one', 'delay' : delays}
        net.connect(senders, receivers, conn_dict, syn_spec)


 
    def filter_ids(self, net, id_list, center, spec):
        """ Returns the IDs of the units in id_list that satisfy a criterion set in spec.

            This is a utility function for topo_connect(). The criterion in spec is either
            the 'mask' or the 'kernel' value from the conn_spec in topo_connect. When
            using the mask criteria, filter_ids will return a list with all the units in
            id_list inside the mask. When using the kernel criteria, filter_ids will return
            all the units that have a probability of connection larger than 1e-7.

        Args:
            net : the network where the units in id_list live.
            id_list: a list with IDs of units in the 'net' network.
            center: coordinates of the center of the mask or kernel used for filtering.
            spec: a dictionary with the filtering criterion. Its value is either
                  conn_spec['mask'] or conn_spec['kernel'] from topo_connect.

        Returns:
            A list with the IDs from id_list that satisfy the criterion in spec.

        Raises:
            ValueError
        """
        # Utility function: Euclidean distance
        dist = lambda x,y: np.sqrt( sum( (x-y)*(x-y) ) )

        filtered = [] # this list will have the filtered unit IDs
        min_prob = 1e-7 # minimum probability for kernel filtering

        #------------------ kernel criteria ------------------
        if type(spec) is float or type(spec) is int: # distance-independent, constant value
                if spec < 0. or spec > 1.:
                    raise ValueError('kernel probability should be between 0 and 1')
                elif spec >= min_prob: # if spec < min_prob, then filtered = []
                    filtered = id_list
        elif 'linear' in spec:
            a = spec['linear']['a']
            c = spec['linear']['c']
            if a > 0:
                max_dist = (c - min_prob)/a 
                for to_idx in id_list:
                    to_c = net.units[to_idx].coordinates
                    d = dist(center, to_c)
                    if d < max_dist:
                        filtered.append(to_idx)
            else:
                if c > min_prob:
                    filtered = id_list
        elif 'gaussian' in spec:
            p_c = spec['gaussian']['p_center']
            s = spec['gaussian']['sigma']
            if p_c > min_prob:
                max_dist = s * np.sqrt(np.log(p_c/min_prob))
                for to_idx in id_list:
                    to_c = net.units[to_idx].coordinates
                    d = dist(center, to_c)
                    if d < max_dist:
                        filtered.append(to_idx)

        #------------------ mask criteria ------------------
        elif 'circular' in spec:
            for idx in id_list:
                to_c = net.units[idx].coordinates
                d = dist(center, to_c)
                if d <= spec['circular']['radius']:
                    filtered.append(idx)
        elif 'annular' in spec:
            for idx in id_list:
                to_c = net.units[idx].coordinates
                d = dist(center, to_c)
                if ( d >= conn_spec['annular']['inner_radius'] and 
                     d <= conn_spec['annular']['outer_radius'] ):
                    filtered.append[idx]
        elif 'rectangular' in spec: # periodic boundary will be tricky in here
            # center the coordinates of the rectangular mask 
            ll0 = conn_spec['rectangular']['lower_left'][0] - center[0]
            ll1 = conn_spec['rectangular']['lower_left'][1] - center[1]
            ur0 = conn_spec['rectangular']['upper_right'][0] - center[0]
            ur1 = conn_spec['rectangular']['upper_right'][1] - center[1]
            for idx in id_list:
                to_c = net.units[idx].coordinates
                # test if the receiving unit is inside the rectangle
                if ( to_c[0] >= ll0 and to_c[1] >= ll1 ):
                    if ( to_c[0] <= ur0 and to_c[1] <= ur1 ):
                        filtered.append[idx]
        else:
            raise ValueError("No valid dictionary was found for the mask parameter")

        return filtered




