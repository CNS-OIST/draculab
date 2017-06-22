'''
synapses.py
This file contains all the synapse models used in the sirasi simulator
'''

from sirasi import unit_types, synapse_types, syn_reqs  # names of models and requirements
import numpy as np

class synapse():
    def __init__(self, params, network):
        self.preID = params['preID']   # the ID of the presynaptic unit 
        self.postID = params['postID'] # the ID of the postsynaptic unit
        self.w = params['init_w'] # initializing the synaptic weight
        self.type = params['type'] # assigning the synapse type
        self.net = network # the network where the synapse lives
        assert self.net.n_units >= self.preID, 'Synapse connected from non existing unit ' + str(self.preID)  
        assert self.net.n_units >= self.postID, 'Synapse connected to non existing unit ' + str(self.postID)  
        assert self.net.sim_time == 0., 'Synapse being created when sim_time is not zero'

        # Each synapse type is tasked with indicating what information it will ask
        # from its corresponding unit when the synapse updates. That information is contained 
        # in the 'upd_requirements' set. Each entry in upd_requirements corresponds to
        # a different variable that the unit must maintain each time the unit's
        # update() function is called. The possible values are in the syn_reqs
        # enumerator of the sirasi module.
        self.upd_requirements = set() # start with an empty set

    def get_w(self, time): 
        # If I'm going to use time, I need to use buffers and interpolation.
        # Not necessary so far.
        return self.w

    def update(self,time):
        # The default update rule does nothing
        return

    
class static_synapse(synapse):
    def __init__(self, params, network):
        super(static_synapse, self).__init__(params, network)
        assert self.type is synapse_types.static, ['Synapse from ' + str(self.preID) + ' to ' +
                                                            str(self.postID) + ' instantiated with the wrong type']

    # static synapses don't do anything when updated
    def update(self,time):
        return

          
class oja_synapse(synapse):
    ''' This class implements a continuous version of the Oja learning rule. 
        Since there are no discrete inputs, the presynaptic activity is tracked with a low-pass filter,
        and used to adapt the weight everytime the "update" function is called.
    '''
    def __init__(self, params, network):
        super(oja_synapse, self).__init__(params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.last_time = self.net.sim_time # time of the last call to the update function
        self.lpf_x = self.net.units[self.preID].get_act(self.last_time) # low-pass filtered presynaptic activity
        self.lpf_y = self.net.units[self.postID].get_act(self.last_time) # low-pass filtered postsynaptic activity
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        # The Oja rule requires the current pre- and post-synaptic activity
        self.upd_requirements = set([syn_reqs.lpf_fast, syn_reqs.pre_lpf_fast])
        assert self.type is synapse_types.oja, ['Synapse from ' + str(self.preID) + ' to ' +
                                                          str(self.postID) + ' instantiated with the wrong type']

    
    def update(self, time):
        # If the network is correctly initialized, the pre- and post-synaptic units
        # are updating their lpf_fast variables at each update() call
        lpf_post = self.net.units[self.postID].lpf_fast
        lpf_pre = self.net.units[self.preID].lpf_fast
        
        # A forward Euler step with the Oja learning rule 
        self.w = self.w + self.alpha * lpf_post * ( lpf_pre - lpf_post*self.w )


