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


def load_data(filename):
    """
    Receives the name of a datafile saved in XPP's .dat format, 
    and returns numpy arrays with the data from the columns
    The format is simply putting in each line the values of time and state variables
    separated by spaces. Time is the first column.
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


class test_comparing_simulators(unittest.TestCase):
    """ This is an automated version of test2.ipynb . """

    def create_test_network_1(self):
        """ Returns a network that should produce the test data. """
        ######### 1) Create a network
        net_params = {'min_delay' : 0.2, 'min_buff_size' : 50 } 
        n1 = network(net_params)
        ######### 2) Put some units in the network
        # default parameters for the units
        pars = { 'coordinates' : np.zeros(3),
                'delay' : 1., 'init_val' : 0.5, 'tau_fast' : 1.,
                'slope' : 1., 'thresh' : 0.0, 'tau' : 0.02,
                'type' : unit_types.source, 'function' : lambda x : None } 
        inputs = n1.create(2,pars) # creating two input sources
        # setting the input functions
        n1.units[inputs[0]].set_function(lambda t: 0.5*t)
        n1.units[inputs[1]].set_function(lambda t: -4.*np.sin(t))
        pars['type'] = unit_types.sigmoidal
        sig_units = n1.create(2,pars) # creating two sigmoidal units
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

    def test_network_1(self):
        """ Test if draculab, XPP and Matlab give the same results for network 1. """
        tolerance = 0.002 # test will fail if error larger than this
        # The .dat files should be in the same directory
        xpp_dat = load_data('./sim1oderun.dat')
        matlab_dat = load_data('./sim1matrun.txt')
        ####### Test the regular run
        net = self.create_test_network_1()
        sim_dat = net.run(20.)
        # interpolate the XPP and Matlab data so it is at
        # the same time points as the draculab data
        xpp_points1 = align_points(sim_dat[0], xpp_dat[0], xpp_dat[1])
        matlab_points1 = align_points(sim_dat[0], matlab_dat[0], matlab_dat[1])
        xpp_points2 = align_points(sim_dat[0], xpp_dat[0], xpp_dat[2])
        matlab_points2 = align_points(sim_dat[0], matlab_dat[0], matlab_dat[2])
        
        max_diff_xpp1 = max(np.abs(xpp_points1 - sim_dat[1][2]))
        max_diff_xpp2 = max(np.abs(xpp_points2 - sim_dat[1][3]))
        max_diff_matlab1 = max(np.abs(matlab_points1 - sim_dat[1][2]))
        max_diff_matlab2 = max(np.abs(matlab_points2 - sim_dat[1][3]))
        # The maximum difference depends on min_buff_size
        max_diff = max(max_diff_xpp1, max_diff_xpp2, max_diff_matlab1, max_diff_matlab2)
        self.assertTrue(max_diff < tolerance) 

    def test_network_1_flat(self):
        """ Test if flat draculab, XPP and Matlab give the same results for network 1. """
        tolerance = 0.002 # test will fail if error larger than this
        # The .dat files should be in the same directory
        xpp_dat = load_data('./sim1oderun.dat')
        matlab_dat = load_data('./sim1matrun.txt')
        ####### Test the flat run
        net = self.create_test_network_1()
        sim_dat = net.flat_run(20.)
        # interpolate the XPP and Matlab data so it is at
        # the same time points as the draculab data
        xpp_points1 = align_points(sim_dat[0], xpp_dat[0], xpp_dat[1])
        matlab_points1 = align_points(sim_dat[0], matlab_dat[0], matlab_dat[1])
        xpp_points2 = align_points(sim_dat[0], xpp_dat[0], xpp_dat[2])
        matlab_points2 = align_points(sim_dat[0], matlab_dat[0], matlab_dat[2])
                
        max_diff_xpp1 = max(np.abs(xpp_points1 - sim_dat[1][2]))
        max_diff_xpp2 = max(np.abs(xpp_points2 - sim_dat[1][3]))
        max_diff_matlab1 = max(np.abs(matlab_points1 - sim_dat[1][2]))
        max_diff_matlab2 = max(np.abs(matlab_points2 - sim_dat[1][3]))
        # The maximum difference depends on min_buff_size
        max_diff = max(max_diff_xpp1, max_diff_xpp2, max_diff_matlab1, max_diff_matlab2)
        self.assertTrue(max_diff < tolerance) 



if __name__=='__main__':
    unittest.main()
