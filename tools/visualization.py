# visualization.py
# Classes with helpful methods for Draculab simualtions

from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

class plotter():
    """ A class to produce plots and animations. """
    def __init__(self, network, times, activs, plant_acts=None, backend='qt5'):
        """ Class constructor.

            Args:
		network: The draculab network that produced the simulation data
                times: The times array from the simulation
                activs: The unit data array from the simulation
                plant_acts: Plant data array (optional)
                backend: Matplotlib backend (default is 'qt5')
        """
        get_ipython().run_line_magic('matplotlib', backend)
        self.net = network
        self.times = times
        self.activs = activs
        if plant_acts: self.plant_acts = plant_acts

    def act_anim(self, pop, thr, interv=100, slider=False):
        """ An animation to visualize the activity of a set of units. 
            
            pop : an array or list with the IDs of the units to visualize.
            interv : refresh interval of the simulation.
            slider : When set to True the animation is substituted by a
                     slider widget.
            thr : units whose activity surpasses 'thr' will be highligthted.
        """
        self.unit_fig = plt.figure(figsize=(10,10))
        self.ax = self.unit_fig.add_axes([0., 0., 1., 1.], frameon=False)
        xcoords = [ u.coordinates[0] for u in [self.net.units[i] for i in pop] ]
        ycoords = [ u.coordinates[1] for u in [self.net.units[i] for i in pop] ]
        min_x = min(xcoords) - 0.1;     max_x = max(xcoords) + 0.1
        min_y = min(ycoords) - 0.1;     max_y = max(ycoords) + 0.1
        self.ax.set_xlim(min_x, max_x), self.ax.set_xticks([])
        self.ax.set_ylim(min_y, max_y), self.ax.set_yticks([])
        self.scat = self.ax.scatter(xcoords, ycoords, s=20.*self.activs[pop,0])
        self.n_data = len(self.activs[0])
        self.act_thr = thr
        self.act_anim_pop = pop

        if not slider:
            animation = FuncAnimation(self.unit_fig, self.update_act_anim,
                            interval=interv, save_count=int(
                            round(self.net.sim_time/self.net.min_delay)))
            return animation
        else:
            from ipywidgets import interact
            widget = interact(self.update_act_anim, frame=(1,self.n_data))
            return widget
        
    def color_fun(self, activ):
        # given the units activity, maps into a color. activ may be an array.
        activ =  np.maximum(0.1,  activ*np.maximum(np.sign(
                                                   activ - self.act_thr), 0.))
        return np.outer(activ, np.array([0., .5, .999, .5]))

    def update_act_anim(self, frame):
        # Each frame advances one simulation step (min_delay time units)
        idx = frame%self.n_data
        cur_time = self.net.min_delay*idx
        self.scat.set_sizes(300.*self.activs[self.act_anim_pop,idx])
        self.scat.set_color(self.color_fun(self.activs[self.act_anim_pop, idx]))
        self.unit_fig.suptitle('Time: ' + '{:f}'.format(cur_time))
        return self.ax,
    
    def conn_anim(self, source, sink, interv=100, slider=False, weights=True):
        """ An animation to visualize the connectivity of populations. 
    
            source and sink are lists with the IDs of the units whose
            connections we'll visualize. 
            
            Each frame of the animation shows: for a particular unit in source,
            all the neurons in sink that receive connections from it (left 
            plot), and for a particular unit in sink, all the units in source
            that send it connections (right plot).
        
            interv is the refresh interval (in ms) used by FuncAnimation.
            
            If weights=True, then the dots' size and color will reflect the
            absolute value of the connection weight.

            Returns:
                animation object from FuncAnimation if slider = False
                widget object from ipywidgets.interact if slider = True
        """
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
        source_coords = [u.coordinates for u in [self.net.units[i] 
                                                 for i in source]]
        sink_coords = [u.coordinates for u in [self.net.units[i] for i in sink]]
        source_x = [c[0] for c in source_coords]
        source_y = [c[1] for c in source_coords]
        sink_x = [c[0] for c in sink_coords]
        sink_y = [c[1] for c in sink_coords]

        # id2src[n] maps the unit with network id 'n' to its index
        # in the 'source' list (1e8 if not in source)
        self.id2src = np.array([1e8 for _ in range(
                                len(self.net.units))], dtype=int) 
        for src_idx, net_idx in enumerate(source):
            self.id2src[net_idx] = src_idx
        # id2snk[n] maps the unit with network id 'n' to its index
        # in the 'sink' list
        self.id2snk = np.array([1e8 for _ in range(
                                len(self.net.units))], dtype=int)
        for snk_idx, net_idx in enumerate(sink):
            self.id2snk[net_idx] = snk_idx

        # setting colors
        self.std_src = [0., 0.5, 0., 0.5]
        self.std_snk = [0.5, 0., 0., 0.5]
        self.big_src = [0., 0., 1., 1.]
        self.big_snk = [0., 0., 1., 1.]

        # constructing figure, axes, path collections
        self.conn_fig = plt.figure(figsize=(12,7))
        self.ax1 = self.conn_fig.add_axes([0.02, 0.01, .47, 0.95],
                                          frameon=True, aspect=1)
        self.ax2 = self.conn_fig.add_axes([0.51, 0.01, .47, 0.95],
                                          frameon=True, aspect=1)
        self.src_col1 = self.ax1.scatter(source_x, source_y, s=2, color=self.std_src)
        self.snk_col1 = self.ax1.scatter(sink_x, sink_y, s=2, color=self.std_snk)
        self.src_col2 = self.ax2.scatter(source_x, source_y, s=2, color=self.std_src)
        self.snk_col2 = self.ax2.scatter(sink_x, sink_y, s=2, color=self.std_snk)
        self.ax1.set_title('sent connections')
        self.ax2.set_title('received connections')
        self.ax2.set_yticks([])

        if weights:
            update_fun = self.update_weight_anim
            # extract the weight matrix
            self.w_mat = np.zeros((len(sink), len(source)))
            for syn in self.all_syns:
                self.w_mat[self.id2snk[syn.postID], 
                           self.id2src[syn.preID]] = abs(syn.w)
            self.w_mat /= np.amax(self.w_mat) # normalizing (maximum is 1)
            self.cmap = plt.get_cmap('Reds') # getting colormap
            #print(self.w_mat)
        else:
            update_fun = self.update_conn_anim

        if not slider:
            animation = FuncAnimation(self.conn_fig, update_fun, 
                                      interval=interv, blit=True)
            return animation
        else:
            from ipywidgets import interact
            widget = interact(update_fun, frame=(1, max(self.len_source, 
                                                        self.len_sink)))
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
        targets = self.id2snk[ [syn.postID for syn in self.all_syns 
                                if self.id2src[syn.preID] == sou_u ] ]
        # setting the colors and sizes
        sink_colors[targets] = self.big_snk
        sink_sizes[targets] = 15
        self.src_col1.set_sizes(source_sizes)
        #ax1.get_children()[0].set_sizes(source_sizes) # sizes for source units
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
        senders = self.id2src[ [syn.preID for syn in self.all_syns
                                if self.id2snk[syn.postID] == snk_u] ]
        # setting the colors and sizes
        source_colors[senders] = self.big_src
        source_sizes[senders] = 15
        self.src_col2.set_sizes(source_sizes)
        self.snk_col2.set_sizes(sink_sizes)
        self.src_col2.set_color(source_colors)
        self.snk_col2.set_color(sink_colors)

        return self.ax1, self.ax2,

    def update_weight_anim(self, frame):
        sou_u = frame%self.len_source # source unit whose receivers we visualize
        snk_u = frame%self.len_sink # sink unit whose senders we visualize

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
