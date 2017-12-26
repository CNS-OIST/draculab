
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import numpy as np
import time
import dill
from draculab import *


class ei_net():
    """
        This class is used to create, simulate, and visualize a generic network containing 3
        populations: excitatory (e), inhibitory (i), and input (x).  The three populations are 
        arranged in a rectangular grid. 

        After creating an instance of the ei_net class, the parameters of the network can be
        configured by changing the entries of the parameter dictionaries using the ei_net.set_param 
        method. Once the parameter dictionaries are as desired the network is created by running
        ei_net.build() . All units and connections are created using the topology module.

        An additional 'w_track' population of source units may also be created in order to
        track the temporal evolution of a random sample of synaptic weights.

        METHODS OVERVIEW 
        __init__ : creates parameter dictionaries with the default values.
        set_param : changes a value in a parameter dictionary.
        build : creates units and connects them.
        run : runs simulations.
        basic_plot, act_anim, hist_anim, double_anim : result visualization.
        conn_anim : connections visualization.
        annotate : append a line with text in the ei_net.notes string.
        log : save the parameter changes and execution history of the network in a text file.
        save : pickle the object and save it in a file.
        -------- 
        make_sin_pulse, input_vector : auxiliary to 'run'.
        color_fun : auxiliary to act_anim and double_anim.
        update_(conn|hist|act|double)_anim : auxiliary to (conn|hist|act|double)_anim.
        update_weight_anim : auxiliary to conn_anim.
    """


    def __init__(self):
        """ Create the parameter dictionaries required to build the excitatory-inhibitory network. 

            Time units are seconds for compatibility with the pendulum model.
            Length units for geometry dictionaries are centimeters.
            Axonal conduction speed is ~1 m/s, synaptic delay about 5 ms for the population response.
        """


        # The history list will store strings containing the instructions doing parameter changes, 
        # building the network, or running a simulation. It also may contain notes.
        self.history = ["__init__ at " + time.ctime()]
        self.notes = '' # comments about network configuration or simulation results.
        # fixing random seed
        seed = 19680801
        np.random.seed(seed)
        self.history.append('np.random.seed = ' + str(seed))
        
        """ All the default parameter dictionaries are below.  """
        # PARAMETER DICTIONARY FOR THE NETWORK
        self.net_params = {'min_delay' : 0.005, 
            'min_buff_size' : 10,
            'rtol' : 1e-5,
            'atol' : 1e-5,
            'cm_del' : .01 }  # delay in seconds for each centimeter in distance. 
        # UNIT PARAMETERS
        self.e_geom = {'shape':'sheet', 
            'extent':[1.,1.], # 1 square centimeter grid
            'center':[0.,0.], 
            'arrangement':'grid', 
            'rows':15, 
            'columns':15,  
            'jitter' : 0.02 }
        self.i_geom = {'shape':'sheet', 
            'extent':[1.,1.], # 1 square centimeter grid
            'center':[0.,0.], 
            'arrangement':'grid', 
            'rows':15, 
            'columns':15,  
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
        self.e_pars = {'init_val' : np.random.rand(self.n['e']), 
            'slope' : 1.,
            'thresh' : 0.,
            'tau' : 0.02,  # 20 ms time constant
            'tau_fast' : 0.04, # 40 ms for fast low-pass filter
            'tau_mid' : .1, # 100 ms for medium low-pass filter
            'tau_scale' : 0.05, # for exp_dist_sigmoidal units
            'c' : 2., # for exp_dist_sigmoidal units
            'Kp' : 0.05, # for exp_dist_sigmoidal units
            'type' : unit_types.sigmoidal }
        self.i_pars = {'init_val' : np.random.rand(self.n['i']), 
            'slope' : 1.,
            'thresh' : 0.,
            'tau' : 0.02,  # 20 ms time constant
            'tau_fast' : 0.04, # 40 ms for fast low-pass filter
            'tau_mid' : .1, # 100 ms for medium low-pass filter
            'tau_scale' : 0.05, # for exp_dist_sigmoidal units
            'c' : 2., # for exp_dist_sigmoidal units
            'Kp' : 0.05, # for exp_dist_sigmoidal units
            'type' : unit_types.sigmoidal }
        self.x_pars = {'type' : unit_types.source,
            'init_val' : 0.,
            'tau_fast' : 0.04,
            'function' : lambda x: None }
        self.wt_pars = {'type' : unit_types.source,  # parameters for "weight tracking" units
            'init_val' : 0.,
            'tau_fast' : 0.04,
            'function' : lambda x: None }
        # CONNECTION PARAMETERS
        self.ee_conn = {'connection_type' : 'divergent',
            'mask' : {'circular': {'radius': .3}}, 
            'kernel' : .9,
            'delays' : {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}},
            'weights' :{'uniform' : {'low': 0.01, 'high' : 0.5}},
            'allow_autapses' : True,
            'allow_multapses' : False,
            'edge_wrap' : True,
            'boundary' : {'center': np.array(self.e_geom['center']), 'extent' : self.e_geom['extent']} }
        self.ei_conn = {'connection_type' : 'divergent',
            'mask' : {'circular': {'radius': .3}}, 
            'kernel' : .9,
            'delays' : {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}},
            'weights' :{'uniform' : {'low': 0.01, 'high' : 0.5}},
            'allow_autapses' : True,
            'allow_multapses' : False,
            'edge_wrap' : True,
            'boundary' : {'center': np.array(self.i_geom['center']), 'extent' : self.i_geom['extent']} }
        self.ie_conn = {'connection_type' : 'divergent',
            'mask' : {'circular': {'radius': .3}}, 
            'kernel' : .9,
            'delays' : {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}},
            'weights' :{'uniform' : {'low': -0.5, 'high' : -0.01}},
            'allow_autapses' : True,
            'allow_multapses' : False,
            'edge_wrap' : True,
            'boundary' : {'center': np.array(self.e_geom['center']), 'extent' : self.e_geom['extent']} }
        self.ii_conn = {'connection_type' : 'divergent',
            'mask' : {'circular': {'radius': .3}}, 
            'kernel' : .9,
            'delays' : {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}},
            'weights' :{'uniform' : {'low': -0.5, 'high' : -0.01}},
            'allow_autapses' : True,
            'allow_multapses' : False,
            'edge_wrap' : True,
            'boundary' : {'center': np.array(self.i_geom['center']), 'extent' : self.i_geom['extent']} }
        self.xe_conn = {'connection_type' : 'divergent',
            'mask' : {'circular': {'radius': .1}}, 
            'kernel' : 1.,
            'delays' : {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}},
            'weights' :{'uniform' : {'low': 0.01, 'high' : 0.5}},
            'allow_autapses' : True,
            'allow_multapses' : False,
            'edge_wrap' : True,
            'boundary' : {'center': np.array(self.e_geom['center']), 'extent' : self.e_geom['extent']} }
        self.xi_conn = {'connection_type' : 'divergent',
            'mask' : {'circular': {'radius': .1}}, 
            'kernel' : 1.,
            'delays' : {'linear' : {'c' : self.net_params['min_delay'], 'a' : self.net_params['cm_del']}},
            'weights' :{'uniform' : {'low': 0.01, 'high' : 0.5}},
            'allow_autapses' : True,
            'allow_multapses' : False,
            'edge_wrap' : True,
            'boundary' : {'center': np.array(self.i_geom['center']), 'extent' : self.i_geom['extent']} }
        self.ee_syn = {'type' : synapse_types.static,  # excitatory to excitatory synapses
            'lrate' : 0.1,  # for all dynamic synapse types
            'omega' : 1.,  # for sq_hebb_subsnorm synapses 
            'input_type' : 'pred',  # for input_correlation synapses
            'des_act' : 0.4, # for homeo_inhib, and corr_homeo_inhib synapses
            'c' : 1., # for exp_rate_dist synapses
            'wshift' : 1. } # for exp_rate_dist synapses
        self.ei_syn = {'type' : synapse_types.static,
            'lrate' : 0.1,  # for all dynamic synapse types
            'omega' : 1.,  # for sq_hebb_subsnorm synapses 
            'input_type' : 'pred',  # for input_correlation synapses
            'des_act' : 0.4, # for homeo_inhib, and corr_homeo_inhib synapses
            'c' : 1., # for exp_rate_dist synapses
            'wshift' : 1. } # for exp_rate_dist synapses
        self.ie_syn = {'type' : synapse_types.static,
            'lrate' : 0.1,  # for all dynamic synapse types
            'omega' : 1.,  # for sq_hebb_subsnorm synapses 
            'input_type' : 'pred',  # for input_correlation synapses
            'des_act' : 0.4, # for homeo_inhib, and corr_homeo_inhib synapses
            'c' : 1., # for exp_rate_dist synapses
            'wshift' : 1. } # for exp_rate_dist synapses
        self.ii_syn = {'type' : synapse_types.static,
            'lrate' : 0.1,  # for all dynamic synapse types
            'omega' : 1.,  # for sq_hebb_subsnorm synapses 
            'input_type' : 'pred',  # for input_correlation synapses
            'des_act' : 0.4, # for homeo_inhib, and corr_homeo_inhib synapses
            'c' : 1., # for exp_rate_dist synapses
            'wshift' : 1. } # for exp_rate_dist synapses
        self.xe_syn = {'type' : synapse_types.static,
            'lrate' : 0.1,  # for all dynamic synapse types
            'omega' : 1.,  # for sq_hebb_subsnorm synapses 
            'input_type' : 'pred',  # for input_correlation synapses
            'des_act' : 0.4, # for homeo_inhib, and corr_homeo_inhib synapses
            'c' : 1., # for exp_rate_dist synapses
            'wshift' : 1. } # for exp_rate_dist synapses
        self.xi_syn = {'type' : synapse_types.static,
            'lrate' : 0.1,  # for all dynamic synapse types
            'omega' : 1.,  # for sq_hebb_subsnorm synapses 
            'input_type' : 'pred',  # for input_correlation synapses
            'des_act' : 0.4, # for homeo_inhib, and corr_homeo_inhib synapses
            'c' : 1., # for exp_rate_dist synapses
            'wshift' : 1. } # for exp_rate_dist synapses
        

    def build(self):
        """ Create the draculab network. """
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
        self.e_pars['init_val'] = np.random.rand(self.n['e'])
        self.i_pars['init_val'] = np.random.rand(self.n['i'])
        # Create an auxiliary topology object
        topo = topology()
        # Create network
        self.net = network(self.net_params)
        # Create unit groups
        self.e = topo.create_group(self.net, self.e_geom, self.e_pars)
        self.i = topo.create_group(self.net, self.i_geom, self.i_pars)
        self.x = topo.create_group(self.net, self.x_geom, self.x_pars)
        # Create weight tracking units
        self.w_track = self.net.create(self.n['w_track'], self.wt_pars)
        # Create connections
        topo.topo_connect(self.net, self.e, self.e, self.ee_conn, self.ee_syn)
        topo.topo_connect(self.net, self.e, self.i, self.ei_conn, self.ei_syn)
        topo.topo_connect(self.net, self.i, self.e, self.ie_conn, self.ie_syn)
        topo.topo_connect(self.net, self.i, self.i, self.ii_conn, self.ii_syn)
        topo.topo_connect(self.net, self.x, self.i, self.xi_conn, self.xi_syn)
        topo.topo_connect(self.net, self.x, self.e, self.xe_conn, self.xe_syn)
        # Set the function of the w_track units to follow randomly chosen weights.
        n_u = int(np.floor(np.sqrt(self.n['w_track']))) # For how many units we'll track weights
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
        """ 
        # If there are exp_dist_sigmoidal units, create some units to track their scale factors
        if self.unit_pars['type'] == unit_types.exp_dist_sig:
            self.sc_track = self.net.create(self.n_units['w_track'], self.inp_pars)
            def scale_tracker(u,s):
                return lambda x: self.net.units[u].scale_facs[s]
            for uid,u in enumerate(which_u):
                for sid,s in enumerate(which_syns[uid]):
                    self.net.units[self.sc_track[uid*n_syns+sid]].set_function(scale_tracker(u,s))
        """ 

        
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
            
    def make_sin_pulse(self, t_init, t_end, per, amp): 
        """ This function returns a function implementing a sinusoidal bump(s). """
        return lambda t : amp * ( np.sqrt( 0.5 * (1. + 
                      np.sin( np.pi*( 2.*(t - t_init)/per - 0.5 ) ) ) ) ) if (t_init < t and t < min(t_init+per,t_end)) else 0.
    
    def input_vector(self):
        """This function returns an input vector. Different distributions can be implemented with this. """
        ## random vector with unit norm, positive entries
        # vec = np.random.uniform(0., 1., self.n['x'])
        # return  vec / np.linalg.norm(vec)
        
        ## random vector with unit norm, positive and negative entries
        vec = np.random.uniform(0., 1., self.n['x']) - np.random.uniform(0., 1., self.n['x'])
        return  vec / np.linalg.norm(vec)
        
    def run(self, n_pres, inp_time, inp_amp, pres_time, alpha=0.):
        """ Run a simulation, presenting n_pres patterns, each lasting pres_time, 
            with inputs of amplitude inp_amp and lasting inp_time at the beginning of each presentation.
            
            The argument alpha determines how fast the input amplitudes drift.
            
            Updates:
                self.all_times: 1-D numpy array with the times for each data point in all_activs.
                self.all_activs: 2-D numpy array with the activity of all units at each point in all_times. 
        """
        # store a record of this simulation
        self.history.append('run(%d, %f, %f, %f, %f)' % (n_pres, inp_time, inp_amp, pres_time, alpha)) 
        # initialize storage of results
        self.all_times = []
        self.all_activs = []
        start_time = time.time()
        inp_vel = np.zeros(self.n['x']) # initial velocity for the input drift
        inp_vec = inp_amp*self.input_vector() # initial amplitudes of the inputs
        # present input patterns
        for pres in range(n_pres):
            print('Simulating presentation ' + str(pres), end='\r')
            pres_start = time.time()
            t = self.net.sim_time
            inp_accel = inp_amp*self.input_vector() # having an acceleration confers "momentum" to the input drift
            inp_vel = (1.-alpha)*inp_vel + alpha*inp_accel 
            inp_vec = (1.-alpha)*inp_vec + alpha*inp_vel
            for i in self.x:
                self.net.units[i].set_function(self.make_sin_pulse(t, t+pres_time, inp_time, inp_vec[i-self.x[0]]))
        
            times, activs, plants = self.net.run(pres_time)
            self.all_times.append(times)
            self.all_activs.append(activs)
            print('Presentation %s lasted %s seconds.' % (pres, time.time() - pres_start), end='\n')

        self.all_times = np.concatenate(self.all_times)
        self.all_activs = np.concatenate(self.all_activs, axis=1)
        print('Execution time is %s seconds' % (time.time() - start_time)) 
        return self # see rant on multiprocessing    
    
    def basic_plot(self):
        #%matplotlib inline
        inp_fig = plt.figure(figsize=(10,5))
        inputs = np.transpose([self.all_activs[i] for i in self.x])
        plt.plot(self.all_times, inputs, linewidth=1, figure=inp_fig)
        plt.title('Inputs')

        unit_fig = plt.figure(figsize=(10,5))
        e_tracked = [e for e in self.tracked if e in self.e]
        i_tracked = [i for i in self.tracked if i in self.i]
        e_acts = np.transpose(self.all_activs[e_tracked])
        i_acts = np.transpose(self.all_activs[i_tracked])
        plt.plot(self.all_times, e_acts, figure=unit_fig, linewidth=4)
        plt.plot(self.all_times, i_acts, figure=unit_fig, linewidth=1)
        plt.title('Some unit activities. Thick=Exc, Thin=Inh')
        
        # Plot the evolution of the synaptic weights
        w_fig = plt.figure(figsize=(10,5))
        weights = np.transpose([self.all_activs[self.w_track[i]] for i in range(self.n['w_track'])])
        plt.plot(self.all_times, weights, linewidth=1)
        plt.title('Some synaptic weights')
        
        """
        # Plot the evolution of the synaptic scale factors
        if self.unit_pars['type'] == unit_types.exp_dist_sig:
            sc_fig = plt.figure(figsize=(10,5))
            factors = np.transpose([self.all_activs[self.sc_track[i]] for i in range(self.n_units['w_track'])])
            plt.plot(self.all_times, factors, linewidth=1)
            plt.title('Some synaptic scale factors')
        """
        plt.show()
        
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
        self.ax = self.unit_fig.add_axes([0, 0, 1, 1], frameon=False)
        self.ax.set_xlim(-1, 1), self.ax.set_xticks([])
        self.ax.set_ylim(-1, 1), self.ax.set_yticks([])
        xcoords = [ u.coordinates[0] for u in [self.net.units[i] for i in pop] ]
        ycoords = [ u.coordinates[1] for u in [self.net.units[i] for i in pop] ]
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
    
    def conn_anim(self, source, sink, interv=100, slider=False, weights=True):
        """ An animation to visualize the connectivity of populations. 
    
            source and sink are lists with the IDs of the units whose connections we'll
            visualize. They should consist of contiguous, increasing integers. 
            
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
            syn_list = [s for s in syn_list if s.preID in source]
            self.all_syns += syn_list
    
        # getting lists with the coordinates of all source, sink units
        source_coords = [u.coordinates for u in [self.net.units[i] for i in source]]
        sink_coords = [u.coordinates for u in [self.net.units[i] for i in sink]]
        source_x = [c[0] for c in source_coords]
        source_y = [c[1] for c in source_coords]
        sink_x = [c[0] for c in sink_coords]
        sink_y = [c[1] for c in sink_coords]

        # setting colors
        self.std_src = [0., 0.5, 0., 0.5]
        self.std_snk = [0.5, 0., 0., 0.5]
        self.big_src = [0., 0., 1., 1.]
        self.big_snk = [0., 0., 1., 1.]

        # constructing figure, axes, path collections
        self.conn_fig = plt.figure(figsize=(12,7))
        self.ax1 = self.conn_fig.add_axes([0.0, 0.01, .49, 0.95], frameon=True, aspect=1)
        self.ax2 = self.conn_fig.add_axes([0.51, 0.01, .49, 0.95], frameon=True, aspect=1)
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
                self.w_mat[syn.postID - sink[0], syn.preID - source[0]] = abs(syn.w)
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
        targets = [syn.postID - self.sink_0 for syn in self.all_syns if syn.preID == sou_u + self.source_0]
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
        senders = [syn.preID - self.source_0 for syn in self.all_syns if syn.postID == snk_u + self.sink_0]
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
    
    def hist_anim(self, pop, nbins=20, interv=100, slider=False, pdf=False):
        """ An animation to visualize the firing rate histogram of a given population through time. 
        
            pop : list or array with the IDs of the units whose histogram we'll visualize.
            interv : refresh interval of the simulation.
            slider : When set to True the animation is substituted by a slider widget.
            nbins : number of bins in the histogram
            pdf: include a plot of the exponential PDF?
                 Used when visualizing exp_dist_sig units or exp_rate_dist synapses.
                 CURRENTLY ONLY SUPPORTED WHEN pop IS THE EXCITATORY POPULATION.
        """
        get_ipython().run_line_magic('matplotlib', 'qt5')
        # notebook or qt5 
        self.hist_fig = plt.figure(figsize=(10,10))
        if pdf: # assuming pop consists of excitatory units
            if self.e_pars['type'] == unit_types.exp_dist_sig:
                c = self.net.units[self.e[0]].c
            else:
                c = self.ee_syn['c']
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
                 CURRENTLY ONLY SUPPORTED WHEN pop IS THE EXCITATORY POPULATION.
                  
        """
        get_ipython().run_line_magic('matplotlib', 'qt5')
        
        self.double_fig = plt.figure(figsize=(16,8))
        self.n_bins = nbins
        self.act_thr = thr
        self.n_data = len(self.all_activs[0])
        # Histogram figure and axis
        self.hist_ax = self.double_fig.add_axes([0.02, .04, .47, .92])
        if pdf: # assuming pop consists of excitatory units
            if self.e_pars['type'] == unit_types.exp_dist_sig:
                c = self.net.units[self.e[0]].c
            else:
                c = self.ee_syn['c']
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
        self.act_ax.set_xlim(-1, 1), self.act_ax.set_xticks([])
        self.act_ax.set_ylim(-1, 1), self.act_ax.set_yticks([])
        xcoords = [ u.coordinates[0] for u in [self.net.units[i] for i in pop] ]
        ycoords = [ u.coordinates[1] for u in [self.net.units[i] for i in pop] ]
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
    
    def annotate(self, string, make_history=True):
        """ Append a string to self.notes and self.history. """
        self.notes += '# NOTE: ' + string + '\n'
        if make_history:
            self.history.append('#' + string)
            
    def log(self, name="ei_net_log.txt"):
        with open(name, 'a') as f:
            f.write('#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            f.write('#---------Logging exp_distro object---------\n')
            f.write('# HISTORY #\n')
            for entry in self.history:
                f.write(entry + '\n')
            f.write('# NOTES #\n')
            f.write(self.notes)
            f.write('\n')
            f.close()
        
    def save(self, name="ei_net_pickle.pkl"):
        """ Saving simulation results. 
            A draculab network contains lists with functions, so it is not picklable. 
            But it can be serialized with dill: https://github.com/uqfoundation/dill 
            
            After saving, retrieve object using, for example:
            F = open("ei_net_pickle.pkl", 'rb')
            ei = dill.load(F)
            F.close()
        """
        self.history.append('# exp_distro object being saved as ' + name)
        F = open(name, 'wb')
        dill.dump(self, F)
        F.close()


