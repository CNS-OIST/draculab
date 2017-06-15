'''
synapses.py
This file contains all the synapse models used in the sirasi simulator
'''

from sirasi import unit_types, synapse_types
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
          
class oja_synapse(synapse):
    ''' This class implements a continuous version of the Oja learning rule. 
        Since there are no discrete inputs, the presynaptic activity is tracked with a low-pass filter,
        and used to adapt the weight everytime the "update" function is called.
    '''
    def __init__(self, params, network):
        super(oja_synapse, self).__init__(params, network)
        self.lrate = params['lrate'] # learning rate for the synaptic weight
        self.tau_x = params['tau_x'] # time constant used to track the presynaptic activity
        self.tau_y = params['tau_y'] # time constant used to track the postsynaptic activity
        self.last_time = self.net.sim_time # time of the last call to the update function
        self.lpf_x = self.net.units[self.preID].get_act(self.last_time) # low-pass filtered presynaptic activity
        self.lpf_y = self.net.units[self.postID].get_act(self.last_time) # low-pass filtered postsynaptic activity
        self.alpha = self.lrate * self.net.min_delay # factor that scales the update rule
        assert self.type is synapse_types.oja, ['Synapse from ' + str(self.preID) + ' to ' +
                                                          str(self.postID) + ' instantiated with the wrong type']

    
    def update(self, time):
        assert time >= self.last_time, ['Synapse from ' + str(self.preID) + ' to ' +
                                         str(self.postID) + ' updated backwards in time']
        cur_x = self.net.units[self.preID].get_act(time) # current presynaptic activity
        cur_y = self.net.units[self.postID].get_act(time) # current postsynaptic activity
        
        # This updating rule comes from analytically solving 
        # lpf_x' = ( x - lpf_x ) / tau
        # and assuming x didn't change much between self.last_time and time.
        # It seems more accurate than an Euler step lpf_x = lpf_x + (dt/tau)*(x - lpf_x)
        self.lpf_x = cur_x + (self.lpf_x - cur_x)*np.exp((self.last_time-time)/self.tau_x)
        self.lpf_y = cur_y + (self.lpf_y - cur_y)*np.exp((self.last_time-time)/self.tau_y)
        
        # A forward Euler step with the Oja learning rule 
        self.w = self.w + self.alpha * self.lpf_y * ( self.lpf_x - self.lpf_y*self.w )
