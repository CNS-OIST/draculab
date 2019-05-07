#!/usr/bin/env python3
# vim:fileencoding=utf-8
"""
unit_tests.py
A suite of unit tests for draculab.
"""

from draculab import *
import re  # regular expressions module, for the load_data function
import matplotlib.pyplot as plt   # more plotting tools
import numpy as np
import time
import unittest
from scipy.interpolate import interp1d
from tools.ei_net import *

# Path were the .dat and .pkl files are located
global test_path 
test_path = '/home/z/projects/draculab/tests/'

def load_data(filename):
    """
    Receives the name of a datafile saved in XPP's .dat format, 
    and returns numpy arrays with the data from the columns
    The format is simply putting in each line the values of time and state
    variables separated by spaces. Time is the first column.
    """
    # Counting the lines and columns in order to allocate the numpy arrays
    file_obj = open(filename, 'r')
    nlines = sum(1 for line in file_obj)
    file_obj.seek(0) # resetting the file object's position
    n_columns = len(re.split(' .', file_obj.readline()))
    # XPP puts an extra space at the end of the line, so I used ' .' instead of ' '
    # We'll store everything in one tuple of numpy arrays, one per column
    values = tuple(np.zeros(nlines) for i in range(n_columns))
    file_obj.seek(0) # resetting the file object's position
    for idx, line in enumerate(file_obj):
        listed = re.split(' ', re.split(' $', line)[0]) # the first split removes 
                                                        # the trailing space
        for jdx, value in enumerate(listed):  
            values[jdx][idx] = float(value) 
    file_obj.close()
    return values


def align_points(times, data_times, data_points):
    """ Returns the data_points interpolated at the points in 'times'. """
    data_fun = interp1d(data_times, data_points)
    return np.array([data_fun(t) for t in times])


class test_comparison_1(unittest.TestCase):
    """ An automated version of the first comparison in test2.ipynb . """

    @classmethod
    def setUpClass(self):
        self.tolerance = 0.0025 # test will fail if error larger than this
        # The .dat files should be in the same directory
        self.xpp_dat = load_data(test_path+'sim1oderun.dat')
        self.matlab_dat = load_data(test_path+'sim1matrun.txt')

    def setUp(self):
        self.net = self.create_test_network_1()

    def create_test_network_1(self, integ='odeint'):
        """ Returns a network that should produce the test data. 
        
            integ a string specifying the integrator to use.
        """
        ######### 1) Create a network
        net_params = {'min_delay' : 0.2, 'min_buff_size' : 50 } 
        n1 = network(net_params)
        ######### 2) Put some units in the network
        # default parameters for the units
        pars = { 'coordinates' : np.zeros(3),
                'delay' : 1., 'init_val' : 0.5, 'tau_fast' : 1.,
                'slope' : 1., 'thresh' : 0.0, 'tau' : 0.02,
                'mu' : 0., 'sigma' : 0., 'lambda' : 1.,
                'type' : unit_types.source, 'function' : lambda x : None } 
        inputs = n1.create(2,pars) # creating two input sources
        # setting the input functions
        n1.units[inputs[0]].set_function(lambda t: 0.5*t)
        n1.units[inputs[1]].set_function(lambda t: -4.*np.sin(t))
        pars['integ_meth'] = integ
        # setting units for noisy integrators
        if integ in ['euler_maru', 'exp_euler']:
            pars['type'] = unit_types.noisy_sigmoidal
        else:
            pars['type'] = unit_types.sigmoidal
        sig_units = n1.create(2,pars) # creating two sigmoidal units
        if integ == 'euler_maru': # we have to force this if lambda!=0
            for unit in [n1.units[uid] for uid in sig_units]:
                unit.update = unit.euler_maru_update
        ######### 3) Connect the units in the network
        conn_spec = {'rule' : 'all_to_all', 'delay' : 1.,
                    'allow_autapses' : False } 
        syn_pars = {'init_w' : 1., 'lrate' : 0.0, 
                    'type' : synapse_types.oja} 
        n1.connect([inputs[0]], [sig_units[0]], conn_spec, syn_pars)
        conn_spec['delay'] = 2.
        n1.connect([inputs[1]], [sig_units[0]], conn_spec, syn_pars)
        conn_spec['delay'] = 5.
        n1.connect([sig_units[0]], [sig_units[1]], conn_spec, syn_pars)
        conn_spec['delay'] = 6.
        n1.connect([sig_units[1]], [sig_units[0]], conn_spec, syn_pars)
        ######## 4) Return
        return n1

    def max_diff_1(self, dracu, xpp, matlab):
        """ Obtain the maximum difference in the dracu time series. """
        # interpolate the XPP and Matlab data so it is at
        # the same time points as the draculab data
        xpp_points1 = align_points(dracu[0], xpp[0], xpp[1])
        matlab_points1 = align_points(dracu[0], matlab[0], matlab[1])
        xpp_points2 = align_points(dracu[0], xpp[0], xpp[2])
        matlab_points2 = align_points(dracu[0], matlab[0], matlab[2])
        
        max_diff_xpp1 = max(np.abs(xpp_points1 - dracu[1][2]))
        max_diff_xpp2 = max(np.abs(xpp_points2 - dracu[1][3]))
        max_diff_matlab1 = max(np.abs(matlab_points1 - dracu[1][2]))
        max_diff_matlab2 = max(np.abs(matlab_points2 - dracu[1][3]))
        return max(max_diff_xpp1, max_diff_xpp2, max_diff_matlab1, max_diff_matlab2)

    def test_network_1(self):
        """ Test if draculab, XPP and Matlab agree for network 1. """
        sim_dat = self.net.run(20.)
        max_diff = self.max_diff_1(sim_dat, self.xpp_dat, self.matlab_dat)
        self.assertTrue(max_diff < self.tolerance) 

    def test_network_1_flat(self):
        """ Test if flat draculab, XPP and Matlab agree for network 1. """
        sim_dat = self.net.flat_run(20.)
        max_diff = self.max_diff_1(sim_dat, self.xpp_dat, self.matlab_dat)
        self.assertTrue(max_diff < self.tolerance) 

    def test_network_1_all_integrators(self):
        """ Same test, different integrators. """
        integ_list = ['odeint', 'solve_ivp', 'euler', 'euler_maru', 'exp_euler']
        for integ in integ_list:
            net = self.create_test_network_1(integ)
            sim_dat = net.run(20.)
            max_diff = self.max_diff_1(sim_dat, self.xpp_dat, self.matlab_dat)
            self.assertTrue(max_diff < self.tolerance, 
                            msg = integ + ' integrator failed') 

    def test_network_1_all_integrators_flat(self):
        """ Same test, different integrators. """
        # currently the flat network does not support 'odeint' or 'solve_ivp'.
        # moreover, this is not configured to force 'euler_maru' for flat networks
        integ_list = ['euler', 'euler_maru', 'exp_euler']
        for integ in integ_list:
            net = self.create_test_network_1(integ)
            sim_dat = net.flat_run(20.)
            max_diff = self.max_diff_1(sim_dat, self.xpp_dat, self.matlab_dat)
            self.assertTrue(max_diff < self.tolerance, 
                            msg = integ + ' integrator failed') 

class test_comparison_2(unittest.TestCase):
    """ An automatic version of the second comparison in test2.ipynb. """

    @classmethod
    def setUpClass(self):
        self.tolerance = 0.01 # test will fail if difference is larger than this
        # The .dat files should be in the same directory
        self.xpp_data = load_data(test_path+'sim3oderun5.dat')
            # ran in XPP with lrate = 0.5

    def create_test_network_2(self, custom_fi=False):
        """ Creates a draculab network equivalent of the one in sim3.ode . 
        
            When running XPP, set nUmerics -> Dt to 0.01 .

            custom_fi == True causes the network to use custom_fi units,
            as in test 2a of test2.ipynb.
        """
        ######### 1) Create a network
        net_params = {'min_delay' : 0.1, 'min_buff_size' : 5, 'rtol' : 1e-4, 'atol' : 1e-4 }
        n2 = network(net_params)
        ######### 2) Put some units in the network
        # default parameters for the units
        pars = { 'init_val' : 0.5, 'tau_fast' : 1.,
                'slope' : 1., 'thresh' : 0.0, 'tau' : 0.02,
                'type' : unit_types.source, 'function' : lambda x : None } 
        inputs = n2.create(5,pars) # creating five input sources
        # setting the input functions
        n2.units[inputs[0]].set_function(lambda t: 0.5*np.sin(t))
        n2.units[inputs[1]].set_function(lambda t: -0.5*np.sin(2*t))
        n2.units[inputs[2]].set_function(lambda t: 0.5*np.sin(3*t))
        n2.units[inputs[3]].set_function(lambda t: -0.5*np.sin(t))
        n2.units[inputs[4]].set_function(lambda t: 2.0*np.sin(t))
        if custom_fi:
            sig_f = lambda x: 1. / (1. + np.exp( -1.*(x - 0.0) ))
            pars['type'] = unit_types.custom_fi
            pars['tau'] = 0.02
            pars['function'] = sig_f
        else:
            pars['type'] = unit_types.sigmoidal
        sig_units = n2.create(5,pars) # creating sigmoidal units
        ######### 3) Connect the units in the network
        conn_spec = {'rule' : 'all_to_all', 'delay' : 1.,
                    'allow_autapses' : False} # connection specification dictionary
        syn_pars = {'init_w' : 1., 'lrate' : 0.5, 
                    'type' : synapse_types.static} # synapse parameters dictionary
        # In the XPP code, projections to unit X have a delay X
        # and synapses from unit X have a weight 2*0.X,
        for idx_to, unit_to in enumerate(sig_units):
            conn_spec['delay'] = float(idx_to+1)
            n2.connect([inputs[idx_to]], [unit_to], conn_spec, syn_pars)
            if idx_to == 4: # the last unit has oja synapses in connections from sigmoidals
                syn_pars['type'] = synapse_types.oja
            for idx_from, unit_from in enumerate(sig_units):
                if unit_from != unit_to:
                    syn_pars['init_w'] = 0.2*(idx_from+1)
                    n2.connect([unit_from], [unit_to], conn_spec, syn_pars)
        ######### 4) Return
        return n2

    def max_diff_2(self, dracu, xpp):
        """ Obtain the maximum difference in the dracu time series. """
        xpp_points = [ [] for _ in range(5) ]
        for i in range(5):
            xpp_points[i] = align_points(dracu[0], xpp[0], xpp[i+1])
        return max( [max(np.abs( x - d )) for x,d in zip(dracu[1][5:10], xpp_points)] )

    def test_network_2(self):
        """ Compare the output of sim3.ode with the draculab equivalent. """
        ####### Test the regular run
        net = self.create_test_network_2()
        sim_dat = net.run(10.)
        max_diff = self.max_diff_2(sim_dat,self.xpp_data)
        self.assertTrue(max_diff < self.tolerance)

    def test_network_2_flat(self):
        """ Compare the output of sim3.ode with the flat draculab equivalent. """
        ####### Test the flat run
        net = self.create_test_network_2()
        sim_dat = net.flat_run(10.)
        max_diff = self.max_diff_2(sim_dat,self.xpp_data)
        self.assertTrue(max_diff < self.tolerance)

    def test_network_2_custom(self):
        """ Compare the output of sim3.ode with the draculab equivalent. """
        ####### Test the regular run with custom_fi units
        net = self.create_test_network_2(custom_fi=True)
        sim_dat = net.run(10.)
        max_diff = self.max_diff_2(sim_dat,self.xpp_data)
        self.assertTrue(max_diff < self.tolerance)

    def test_network_2_custom_flat(self):
        """ Compare the output of sim3.ode with the flat draculab equivalent. """
        ####### Test the flat run with custom_fi units
        net = self.create_test_network_2(custom_fi=True)
        sim_dat = net.flat_run(10.)
        max_diff = self.max_diff_2(sim_dat,self.xpp_data)
        self.assertTrue(max_diff < self.tolerance)


class test_comparison_3(unittest.TestCase):
    """ The comparison with sim4a.ode in test2.ipynb. """

    @classmethod
    def setUpClass(self):
        self.tolerance = 0.06 # test will fail if difference is larger than this
        self.noisy_tol = 0.5 # when noise is added
        # The .dat files should be in the same directory
        self.xpp_data = load_data(test_path+'sim4aoderun.dat')  

    def create_test_network_3(self, noisy=False, sigma=0.):
        """ Creates a draculab network equivalent of the one in sim4a.ode . """
        ######### 1) Create a network
        net_params = {'min_delay' : .5, 'min_buff_size' : 50 } # parameter dictionary for the network
        n3 = network(net_params)
        ######### 2) Put some units in the network
        # default parameters for the units
        pars = { 'coordinates' : [np.array([0.,1.])]*5, 
                'lambda': 1., 'mu':0., 'sigma':sigma,
                'delay' : 2., 'init_val' : 0.5,
                'slope' : 1., 'thresh' : 0.0, 'tau' : 0.02,
                'type' : unit_types.source, 'function' : lambda x: None} 
        inputs = n3.create(5,pars) # creating five input sources
        # setting the input functions
        n3.units[inputs[0]].set_function(lambda t: np.sin(t))
        n3.units[inputs[1]].set_function(lambda t: np.sin(2*t))
        n3.units[inputs[2]].set_function(lambda t: np.sin(3*t))
        n3.units[inputs[3]].set_function(lambda t: np.sin(4*t))
        n3.units[inputs[4]].set_function(lambda t: np.sin(5*t))
        if noisy:
            pars['type'] = unit_types.noisy_linear
        else:
            pars['type'] = unit_types.linear
        sig_units = n3.create(5,pars) # creating units
        ######### 3) Connect the units in the network
        conn_spec = {'rule' : 'all_to_all', 'delay' : {'distribution' : 'uniform', 'low' : 1., 'high' : 1.},
                    'allow_autapses' : False } # connection specification dictionary
        syn_pars = {'init_w' : 0.5, 'lrate' : 0.0, 
                    'type' : synapse_types.static} # synapse parameters dictionary
        # In the XPP code, projections from inputs have delay 1 and weight 1,
        # whereas projections from sigmoidals have delay 2 and weight 0.3
        for inp_unit, sig_unit in zip(inputs,sig_units):
            n3.connect([inp_unit], [sig_unit], conn_spec, syn_pars)
        conn_spec['delay'] = 2.
        syn_pars['init_w'] = 0.3
        n3.connect(sig_units, sig_units, conn_spec, syn_pars)
        ######### 4) Return
        return n3

    def max_diff_3(self, dracu, xpp):
        """ Obtain the maximum difference in the dracu time series. """
        xpp_points = [ [] for _ in range(5) ]
        for i in range(5):
            xpp_points[i] = align_points(dracu[0], xpp[0], xpp[i+1])
        return max( [max(np.abs( x - d )) for x,d in zip(dracu[1][5:10], xpp_points)] )

    def test_network_3(self):
        """ Compare the output of sim4a.ode with the draculab equivalent. """
        net = self.create_test_network_3()
        sim_dat = net.run(10.)
        max_diff = self.max_diff_3(sim_dat,self.xpp_data)
        self.assertTrue(max_diff < self.tolerance)

    def test_network_3_flat(self):
        """ Compare the output of sim4a.ode with the flat draculab equivalent. """
        net = self.create_test_network_3()
        sim_dat = net.flat_run(10.)
        max_diff = self.max_diff_3(sim_dat,self.xpp_data)
        self.assertTrue(max_diff < self.tolerance)

    # I removed the exp_euler test until I can figure why it fails
    '''
    def test_network_3_exp_euler(self):
        #""" Compare the output of sim4a.ode, now using noisy_linear units . """
        net = self.create_test_network_3(noisy=True)
        sim_dat = net.run(10.)
        max_diff = self.max_diff_3(sim_dat,self.xpp_data)
        self.assertTrue(max_diff < self.tolerance,
                        msg='exp_euler test failed with max_diff='+str(max_diff))
    '''

    def test_network_3_exp_euler_flat(self):
        """ Compare the output of sim4a.ode, now using noisy_linear units and flat. """
        net = self.create_test_network_3(noisy=True)
        sim_dat = net.flat_run(10.)
        max_diff = self.max_diff_3(sim_dat,self.xpp_data)
        self.assertTrue(max_diff < self.tolerance)

    def test_network_3_exp_euler_noisy(self):
        """ Compare the output of sim4a.ode, with actual noise in the unit. """ 
        net = self.create_test_network_3(noisy=True, sigma=0.5)
        sim_dat = net.run(10.)
        max_diff = self.max_diff_3(sim_dat,self.xpp_data)
        self.assertTrue(max_diff > self.tolerance and max_diff < self.noisy_tol,
                        msg = 'noisy exp_euler had max_diff'+str(max_diff))

    def test_network_3_exp_euler_noisy_flat(self):
        """ Compare the output of sim4a.ode, with actual noise in the unit. """ 
        net = self.create_test_network_3(noisy=True, sigma=0.5)
        sim_dat = net.flat_run(10.)
        max_diff = self.max_diff_3(sim_dat,self.xpp_data)
        self.assertTrue(max_diff > self.tolerance and max_diff < self.noisy_tol)


class eigenvalue_test(unittest.TestCase):
    """ Test 4 synapse types that extract the leading eigenvalue. """

    @classmethod
    def setUpClass(self):
        self.min_cos = 0.78 # test will fail if cosine is smaller than this
        self.max_change = 0.01 # fail if weight sum changes more than this (hebbsnorm)
        self.omega = 1. # squared sum of weights should converte to this (sq_hebbsnorm)
        self.omega_tol = 0.015 # Fail if sq sum of w and omega diverge mor than this
        ####### SETTING THE INPUT FUNCTIONS
        ### You are going to present 4 input patterns that randomly switch over time.
        ### Imagine the 9 inputs arranged in a grid, like a tic-tac-toe board, numbered
        ### from left to right and from top to bottom:
        ### 1 2 3
        ### 4 5 6
        ### 7 8 9
        ### You'll have input patterns
        ### 0 X 0   0 0 0   X 0 X   0 X 0
        ### 0 X 0   X X X   0 0 0   X 0 X
        ### 0 X 0   0 0 0   X 0 X   0 X 0
        ### The input is always a normalized linear combination of one or two of these 
        ### patterns. Pattern pat1 is presented alone for t_pat time units, and then
        ### there is a transition period during which pat1 becomes pat2 by presenting
        ### at time t an input 
        ### c*(t_pat+t_trans - t)*pat1 + c*(t - tpat)*pat2
        ### where c = 1/t_trans, and t_trans is the duration of the transition period. 
        ### At time t_pat+t_trans, pat2 is presented by itself for t_pat time units.
        ### 
        # here are the patterns as arrays
        self.patterns = [np.zeros(9) for i in range(4)]
        self.patterns[0] = np.array([0., 1., 0., 0., 1., 0., 0., 1., 0.])/3.
        self.patterns[1] = np.array([0., 0., 0., 1., 1., 1., 0., 0., 0.])/3.
        self.patterns[2] = np.array([1., 0., 1., 0., 0., 0., 1., 0., 1.])/4.
        self.patterns[3] = np.array([0., 1., 0., 1., 0., 1., 0., 1., 0.])/4.
        ####### THE LEADING EIGENVECTOR
        # The code that obtains this is in test3,4,5.ipynb
        self.max_evector = [0.0000, 0.4316, 0.0000, 0.4316, 0.5048, 
                            0.4316, 0.0000, 0.4316, 0.0000]

    def make_fun1(self, idx, cur_pat):
        """ This creates a constant function with value: patterns[cur_pat][idx]
            thus avoiding a scoping problem that is sometimes hard to see:
            https://eev.ee/blog/2011/04/24/gotcha-python-scoping-closures/
        """
        fun = lambda t : self.patterns[cur_pat][idx]
        return fun
    
    def make_fun2(self, idx, last_t, cur_pat, next_pat, t_trans):
        """ Creates a function for the pattern transition. """
        fun = lambda t : self.c * ( (t_trans - (t-last_t))*self.patterns[cur_pat][idx] +
                            (t-last_t)*self.patterns[next_pat][idx] )
        return fun

    def create_network(self, syn_type=synapse_types.oja):
        """ Creates a network for a test. """
        ######### 1) Create a network
        net_params = {'min_delay' : 0.05, 'min_buff_size' : 5 } 
        n1 = network(net_params)
        ######### 2) Put some units in the network
        # default parameters for the units
        pars = { 'function' : lambda x : None,
                'delay' : 1., 'init_val' : 0.5, 'tau_fast' : 1.,
                'slope' : 1., 'thresh' : 0.0, 'tau' : 0.02,
                'type' : unit_types.source } 
        self.inputs = n1.create(9,pars) # creating nine input sources
        pars['type'] = unit_types.linear
        self.unit = n1.create(1,pars) # creating one linear unit
        ######### 3) Connect the units in the network
        conn_spec = {'rule' : 'all_to_all', 'delay' : 1.,
                    'allow_autapses' : False} 
        syn_pars = {'init_w' : {'distribution':'uniform', 'low':0.1, 'high':0.5}, 
                    'omega' : self.omega, 'lrate' : 0.02,
                    'type' : syn_type} 
        n1.connect(self.inputs, self.unit, conn_spec, syn_pars)
        ######## 4) Return
        self.net = n1

    def run_net(self, n_pres=80, t_pat=8., t_trans=2.):
        """ Simulate 'n_pres' pattern presentations.

        Each pattern is presented for t_pat time unts, and transitions
        between patterns last t_trans time units.
        """
        self.c = 1/t_trans # auxiliary variable
        cur_pat = np.random.randint(4)  # pattern currently presented
        next_pat = np.random.randint(4) # next pattern to be presented
        last_t = 0.

        for pres in range(n_pres):
        # For each cycle you'll set the input functions and simulate, once with
        # a single pattern, # once with a mix of patterns, as described above
            # first, we present a single pattern
            for u in range(9):
                self.net.units[self.inputs[u]].set_function( self.make_fun1(u, cur_pat) )
            sim_dat = self.net.flat_run(t_pat)  # simulating
            last_t = self.net.sim_time # simulation time after last pattern presentation
            # now one pattern turns into the next
            for u in range(9):
                self.net.units[self.inputs[u]].set_function(self.make_fun2(u, last_t, cur_pat, next_pat, t_trans))
            sim_dat = self.net.flat_run(t_trans) # simulating
            # choose the pattern you'll present next
            cur_pat = next_pat
            next_pat = np.random.randint(4)
            
    def test_oja_synapse(self):
        self.create_network(syn_type=synapse_types.oja) # initializes self.net
        self.run_net(n_pres=90)
        weights = np.array(self.net.units[self.unit[0]].get_weights(self.net.sim_time))
        cos = sum(self.max_evector*weights) / (np.linalg.norm(self.max_evector)*np.linalg.norm(weights))
        self.assertTrue(self.min_cos < cos, 
                        msg='Oja synapse failed with cosine: '+str(cos))

    def test_hebssnorm_synapse(self):
        self.create_network(syn_type=synapse_types.hebbsnorm) # initializes self.net
        # The hebbsnorm synapse is supposed to preserve the sum of weights
        w_sum1 = sum([syn.w for syn in self.net.syns[self.unit[0]][0:9]])
        self.run_net()
        w_sum2 = sum([syn.w for syn in self.net.syns[self.unit[0]][0:9]])
        weights = np.array(self.net.units[self.unit[0]].get_weights(self.net.sim_time))
        cos = sum(self.max_evector*weights) / (np.linalg.norm(self.max_evector)*np.linalg.norm(weights))
        self.assertTrue(self.min_cos < cos, msg='Oja synapse failed with cosine: '+str(cos))
        self.assertTrue(np.abs(w_sum1-w_sum2) < self.max_change,
                        msg='hebbsnorm synapse did not preserve sum of weights')

    def test_sq_hebssnorm_synapse(self):
        self.create_network(syn_type=synapse_types.sq_hebbsnorm) # initializes self.net
        self.run_net()
        sq_w_sum = sum([syn.w*syn.w for syn in self.net.syns[self.unit[0]][0:9]])
        weights = np.array(self.net.units[self.unit[0]].get_weights(self.net.sim_time))
        cos = sum(self.max_evector*weights) / (np.linalg.norm(self.max_evector)*np.linalg.norm(weights))
        self.assertTrue(self.min_cos < cos, msg='Oja synapse failed with cosine: '+str(cos))
        self.assertTrue(np.abs(self.omega - sq_w_sum) < self.omega_tol,
                        msg='sq_hebbsnorm weights do not converge to omega')
                        

class test_plant(unittest.TestCase):
    """ Test some plants. """
    def test_pendulum(self):
        """ test that the pendulum with no inputs is the same in XPP and draculab . 
        
            Based on test6.ipynb. 
        """
        net_params = {'min_delay' : 0.1, 'min_buff_size' : 5 } # parameter dictionary for the network
        n1 = network(net_params)
        # Notice the XPP model uses a massless rod with a mass at the end, so I
        # need L_c=1.5*L_s, mu_c = 0.75 mu_s.
        # Moreover, the angles are measured wrt different axes, so init_angle = 2 - pi/2
        plant_params = {'type' : plant_models.pendulum,
                'length' : 1.5, 'mass' : 10., 'mu' : 0.75 * 2.,
                'init_angle' : 2.-(np.pi/2.), 'init_ang_vel' : 0.}
        n1.create(1, plant_params) # create a plant
        # Run the simulation
        sim_dat = n1.flat_run(20.)
        # In pend.ode the zero angle aligns with the negative Y axis
        sim_dat[2][0][:,0] = sim_dat[2][0][:,0] + np.pi/2
        
        xpp_dat = load_data(test_path+'pendoderun.dat')
        xpp_points1 = align_points(sim_dat[0], xpp_dat[0], xpp_dat[1])
        xpp_points2 = align_points(sim_dat[0], xpp_dat[0], xpp_dat[2])
        diff1 = max(np.abs(xpp_points1-sim_dat[2][0][:,0]))
        diff2 = max(np.abs(xpp_points2-sim_dat[2][0][:,1]))
        max_diff = max(diff1, diff2)
        self.assertTrue(max_diff < 0.001)

    def test_compare_output(self):
        """ Compare with the output of the first test in in test7.ipynb. 

            Output was stored on 03/13/19.
        """
        # Create parameter dictionaries
        plant_params = {'type' : plant_models.pendulum, 'length' : 1.5, 'inp_gain' : 10.,
                        'mass' : 10., 'mu' : 1.5, 'init_angle' : 2-(np.pi/2.), 'init_ang_vel' : 0.}
        unit_pars = { 'init_val' : 0.5, 'function' : lambda x:None, 'type' : unit_types.source } 
        net_params = {'min_delay' : 0.1, 'min_buff_size' : 5, 'rtol' : 1e-5 }
        # Create network, plant, and input units
        n1 = network(net_params)  # creating a newtwork
        sources = n1.create(2,unit_pars) # source units
        pend = n1.create(1,plant_params) # and a pendulum
        # Create some inputs
        f1 = lambda x : np.sign(max(0.,x-5.))*2. # 0 for x<5, 2 for x>5
        f2 = lambda x : 2.*np.sin(max(0.,x-10.)) # 0 for x<10, 2*sin(x) for x>10
        n1.units[sources[0]].set_function(f1)
        n1.units[sources[1]].set_function(f2)
        # Connect the source units to the pendulum
        conn_spec = {'inp_ports' : [0, 0],  # the pendulum only has one input port
                    'delays' : [1.0, 2.5]} # connection specification dictionary
        syn_pars = {'init_w' : [1.0705, 1.2], 'type' : synapse_types.static} # synapse parameters dictionary
        n1.set_plant_inputs(sources, pend, conn_spec, syn_pars)
        # Create some units, and send the plant's output to them
        unit_pars['type'] = unit_types.sigmoidal
        unit_pars['slope'] = 1.
        unit_pars['thresh'] = 0.
        unit_pars['tau'] = 0.5
        unit_pars['n_ports'] = 2
        lins = n1.create(2,unit_pars)
        conn_spec = {'port_map' : [[(0, 0)],[(1,1)]], # angle to port 0 of unit 1, velocity to port 1 of neuron 2
                    'delays' : [1.,2.]} # connection specification dictionary
        syn_spec = {'init_w' : [1.]*2, 'type' : synapse_types.static} # synapse parameters dictionary
        n1.set_plant_outputs(pend, lins, conn_spec, syn_spec)
        # Run the simulation
        sim_time = 50.
        sim_dat = n1.flat_run(sim_time)
        # In pend.ode the zero angle aligns with the negative Y axis
        sim_dat[2][0][:,0] = sim_dat[2][0][:,0] + np.pi/2
        # load the output of the two sigmoidals
        sig_out = np.load(test_path+'tco_03-13-19.pkl') 
        diff1 = max(np.abs(sig_out[0]-sim_dat[1][2]))
        diff2 = max(np.abs(sig_out[1]-sim_dat[1][3]))
        self.assertAlmostEqual(max(diff1, diff2), 0., places=5)
        # Using 5 significant places works only if you use the same exact configuration for
        # the run, including using get_act instead of get_act_by_step in network.flat_run.
        # When there are differences (e.g. using run instead of flat_run) the max of
        # diff1 and diff2 is around 0.02, meaning you'd have to use places=2 in order to
        # pass the unit test.


class test_topology(unittest.TestCase):
    """ Tests of the toopology module (only one...) """
    def test_compare_annular(self):
        """ Compare connection matrices generated with an annular mask.

        The matrix used was created with the code
        at the bottom of test8.ipynb.
        """
        topo = topology()
        # Create network
        net_params = {'min_delay' : 0.2, 'min_buff_size' : 4 } # parameter dictionary for the network
        net = network(net_params)
        # Create two groups of units
        unit_pars = { 'init_val' : 0.5, 'tau' : 1., 'type' : unit_types.linear } 
        geom1 = {'shape':'sheet', 
                'extent':[10.,10.], 
                'center':[-8.,1.], 
                'arrangement':'grid', 
                'rows':10, 
                'columns':14, 
                'jitter' : 0. }
        geom2 = {'shape':'sheet', 
                'extent':[10.,10.], 
                'center':[8.,-1.], 
                'arrangement':'grid', 
                'rows':10, 
                'columns':14, 
                'jitter' : 0. }
        ids1 = topo.create_group(net, geom1, unit_pars)
        ids2 = topo.create_group(net, geom2, unit_pars)
        # Connect the two groups
        def trans(coords):
            return coords + (np.array(geom2['center']) - np.array(geom1['center']))
        conn_spec = {'connection_type' : 'divergent',
                    'mask' : {"annular" : {"inner_radius": 2., 'outer_radius':5.}},
                    'kernel' : 1., 
                    'delays' : {'linear' : {'c':0.1, 'a':0.1}},
                    'weights' : {'linear' : {'c':5., 'a':1.}},
                    'edge_wrap' : True,
                    'boundary' : {'center' : geom2['center'], 'extent' : geom2['extent']},
                    'transform' : trans,
                    }
        syn_spec = {'type' : synapse_types.static, 'init_w' : 0.2 }
        # connect
        topo.topo_connect(net, ids1, ids2, conn_spec, syn_spec)
        # extract connection matrix
        conns = np.zeros((net.n_units,net.n_units))
        for syn_list in net.syns:
            for syn in syn_list:
                conns[syn.postID,syn.preID] = syn.w
        # retrieve saved connection matrix
        retconn = np.load(test_path+'topo_12-14-18.pkl')
        max_diff = np.amax(np.abs(retconn-conns))
        self.assertAlmostEqual(max_diff, 0.) 


class test_sigma_units(unittest.TestCase):
    """ Some of the sigma unit tests from test10.ipynb. """
    
    def create_network_1(self):
        """ Test 0 from test10.ipynb. """
        # Create two groups of 3 double sigma units 
        net_params = {'min_delay' : 0.02, 'min_buff_size' : 8 } # parameter dictionary for the network
        unit_params = {'type' : unit_types.double_sigma, 
                    'init_val' : 0.5, 
                    'n_ports' : 3,
                    'slope' : 1.,
                    'thresh' : 0.,
                    'tau' : 0.01,
                    'phi' : 0,
                    'branch_params' : {
                        'branch_w' : [0.2, 0.3, 0.5],
                        'slopes' : 1.,
                        'threshs' : 0.  } }
        net = network(net_params)  # creating a newtwork
        self.units1 = net.create(3,unit_params) # first group
        self.units2 = net.create(3,unit_params) # second group
        # connect the units
        conn_spec = { 'rule' : 'all_to_all',
                    'delay' : net_params['min_delay'],
                    'allow_autapses' : True }
        syn_spec = { 'type' : synapse_types.static,
                    'init_w' : 1.0,
                    'inp_ports' : [0, 1, 2]*3 } # the 3 units will project to ports 0, 1, 2
        net.connect(self.units1, self.units2, conn_spec, syn_spec) 
        syn_spec['inp_ports'] = [0, 2, 0]*3 # the 3 units project to ports 0, 2, 0
        net.connect(self.units2, self.units1, conn_spec, syn_spec) # second unit projects to port 1
        syn_spec['inp_ports'] = 0
        net.connect([self.units1[0]], self.units1, conn_spec, syn_spec)
        return net

    def test_port_idx(self):
        """ Test 0 from test10.ipynb. """
        net = self.create_network_1()
        # Predicting these is a mind twister. You should have: 
        # net.units1[0].port_idx = [ [0, 1, 2, 3], [], [] ]   -- remember these are indexes, not unit IDs
        # net.units1[1].port_idx = [ [3], [], [0, 1, 2] ]
        # net.units1[2].port_idx = [ [0, 1, 2, 3], [], []]
        # net.units2[0].port_idx = [ [0, 1, 2], [], [] ]
        # net.units2[1].port_idx = [ [], [0, 1, 2], [] ]
        # net.units2[2].port_idx = [ [], [], [0, 1, 2] ]
        units1_port_idx = ( [ [0, 1, 2, 3], [], [] ], [ [3], [], [0, 1, 2] ], [ [0, 1, 2, 3], [], [] ] )
        units2_port_idx = ( [ [0, 1, 2], [], [] ], [ [], [0, 1, 2], [] ], [ [], [], [0, 1, 2] ] )
        for i in [0,1,2]:
            self.assertTrue(units1_port_idx[i] == net.units[self.units1[i]].port_idx)
            self.assertTrue(units2_port_idx[i] == net.units[self.units2[i]].port_idx)

    def test_get_mp_input(self):
        """ Test 1 from test10.ipynb. """
        net = self.create_network_1()
        # compare get_mp_inpt_sum against explicitly calculated values
        f = lambda x: 1./(1. + np.exp(-x))
        ohs = np.array([0.2,0.3,0.5])
        for unit in range(6):
            inputs = net.units[unit].get_mp_inputs(0.)
            weights = net.units[unit].get_mp_weights(0.)
            ret1 = sum([ o*f(np.dot(w,i)) for o,w,i in zip(ohs, inputs, weights) ])
            ret2 = net.units[unit].get_mp_input_sum(0.)
            self.assertAlmostEqual(abs(ret1 - ret2), 0., places=5)


    def test_constraint(self):
        """ Test 2 from test10.ipynb. """
        # TODO: create separate tests with flat=True and flat=False
        # Creating input pattern function
        ## all the input pattern are in this nested list
        all_inps = [ [ .25]*16,
                    [.5, .5, 0., 0.]+([.25]*12),
                    [.5, .5, 0., 0.]+[0., 0., .5, .5]+([.25]*8),
                    [.5, .5, 0., 0.]+[0., 0., .5, .5]+[0., .5, 0., .5]+([.25]*4),
                    [.5, .5, 0., 0.]+[0., 0., .5, .5]+[0., .5, 0., .5]+[.5, 0., .5, 0.] ]
        def inp_pat(pres, rows, cols):
            return all_inps[pres%5]
        ## The next input pattern function was implemented after the ei_net.mr_run() method was finished.
        ## It is meant to be equivalent to the function above.
        all_inps_mr = [ [ [.25]*4, [.25]*4, [.25]*4, [.25]*4 ],
                        [ [.5, .5, 0., 0.], [.25]*4, [.25]*4, [.25]*4, ],
                        [ [.5, .5, 0., 0.], [0., 0., .5, .5], [.25]*4, [.25]*4 ],
                        [ [.5, .5, 0., 0.], [0., 0., .5, .5], [0., .5, 0., .5], [.25]*4 ],
                        [ [.5, .5, 0., 0.], [0., 0., .5, .5], [0., .5, 0., .5], [.5, 0., .5, 0.] ] ]
        def inp_pat_mr(pres, rows, cols, port):
            return all_inps_mr[pres%5][port]
        # Create the ei_net object
        xnet = ei_net()
        # Set its parameters
        ## geometry
        xnet.set_param('e_geom', 'rows', 1)
        xnet.set_param('e_geom', 'columns', 1)
        xnet.set_param('i_geom', 'rows', 0)
        xnet.set_param('i_geom', 'colums', 0)
        xnet.set_param('x_geom', 'rows', 4)
        xnet.set_param('x_geom', 'columns', 4)
        xnet.set_param('n', 'w_track', 1)
        ## double sigma unit parameters
        xnet.set_param('e_pars', 'type', unit_types.double_sigma)
        xnet.set_param('e_pars', 'slope_min', 5.)
        xnet.set_param('e_pars', 'slope_wid', 0.)
        xnet.set_param('e_pars', 'thresh_min', .8)
        xnet.set_param('e_pars', 'thresh_wid', 0.)
        xnet.set_param('e_pars', 'phi', 0.)
        xnet.set_param('e_pars', 'n_ports', 4)
        xnet.set_param('e_pars', 'branch_params', {'branch_w' : [0.25]*4, 'slopes' : 4, 'threshs' : 0.4})
        ## connection parameters
        xnet.set_param('ee_conn', 'allow_autapses', False)
        xnet.set_param('xe_conn', 'connection_type', 'divergent')
        xnet.set_param('xe_conn', 'mask', {'circular' : {'radius' : 10}})
        xnet.set_param('xe_conn', 'kernel', 1.)
        del xnet.xe_conn['weights'] # so I can set weights with 'init_w'
        xnet.set_param('xe_syn', 'init_w',    [.5, .5, 0., 0.]
                       +[0., 0., .5, .5]+[0., .5, 0., .5]+[.5, 0., .5, 0.])
        xnet.set_param('xe_syn', 'inp_ports', [ 0,  0,  0, 0,]
                       +[ 1,  1,  1,  1]+[ 2,  2,  2,  2]+[ 3,  3,  3,  3])
        ## build the network
        xnet.build()
        # Simulate
        n_pres = 5
        pres_time = 1.
        # These two run calls should give the same results
        #xnet.run(n_pres, pres_time, inp_pat)   
        xnet.mr_run(n_pres, pres_time, inp_pat_mr, flat=False)
        # Compare the output of the cell at each presentation with explicitly calculated values
        ## calculate the value at each presentation
        f = lambda x, sl, th : 1./(1. + np.exp(-sl*(x - th)))
        calc_vals = []
        slope = xnet.e_pars['slope_min']
        thresh = xnet.e_pars['thresh_min']
        sl = xnet.e_pars['branch_params']['slopes']
        th = xnet.e_pars['branch_params']['threshs']
        for i in range(5):
            inp = 0.25*( (0.5*0.25*2)*(4-i) + (0.5*0.5*2)*i )
            calc_vals.append(f( f(inp, sl, th), slope, thresh))
        ## extract the values from the simulation and compare with explicit values
        for i in range(5):
            sim_val = xnet.all_activs[xnet.e[0]][200*(i+1) - 1] # min_delay=.005 
                                                                # => 200 data points per second
            self.assertAlmostEqual( calc_vals[i] - sim_val, 0., places=2 )


if __name__=='__main__':
    unittest.main()
