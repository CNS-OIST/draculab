# coding: utf-8

# rl5E_from_cfg.py
# A function to be used with the v3afxB_test notebooks

from draculab import *

def rl5E_net(cfg,
             pres_interv=10.,
             rand_w=False,
             par_heter=0.1):
    """ Create a draculab network as in rl5E.ipynb from a given configuration.


        Args:
            cfg : a paramter dictionary.

            Optional keyword arguments:
            pres_interv : time itnerval to present each desired value
            rand_w : whether to use random A-->M, M-->C weights initially
            par_heter : parameter heterogeneity as a fraction of original value
        Returns:
            A tuple with the following entries:
            net : A draculab network as in rl5E, with the cfg configuration
            pops_dict : dictionary with the ID list for each population in net.
    """
    #===================================================================
    #================ CREATE THE NETWORK ===============================
    #===================================================================
    #np.random.seed(123456) # always the same random values
    np.random.seed()   # different random values ever run
    
    # some parameters 
    C_type = "rga_sig" # unit type for the C population
    des_angs = np.pi*(-1. + 2.*np.random.random(10000)) # desired angles
    randz = lambda : (1. + par_heter*(np.random.rand()-0.5))
    randz2 = lambda : (1. + par_heter*(np.random.rand(2)-0.5))
    randz4 = lambda : (1. + par_heter*(np.random.rand(4)-0.5))
    SN = 20 # number of units to represent one angle
    No2 = int(np.ceil(SN/2)) # L wiill have (No2)**2 units
    
    # PARAMETER DICTIONARIES
    net_params = {'min_delay' : 0.005,
                'min_buff_size' : 8 }
    #--------------------------------------------------------------------
    # Unit parameters
    #--------------------------------------------------------------------
    A_params = {'type' : unit_types.logarithmic,
                'init_val' : 0.,
                'tau' : 0.01, # 0.02
                'tau_fast': 0.005,
                'thresh' : 0. } #[0.1, 0.1, 0.1, 0.1] } 
    if C_type == "am_pm":
        init_base = np.array([0.5, 0.5, 2.*np.pi, 0.5])
        C_unit_type = unit_types.am_pm_oscillator
    elif C_type == "am2D":
        init_base = np.array([0.5, 0.5])
        C_unit_type = unit_types.am_oscillator2D
    elif C_type == "am_pulse":
        init_base = np.array([0.5, 0.5])
        C_unit_type = unit_types.am_pulse
    elif C_type == "am":
        init_base = np.array([0.5, 0.5, 0.5])
        C_unit_type = unit_types.am_oscillator
    elif C_type == "rga_sig":
        init_base = np.array([0.5])
        C_unit_type = unit_types.rga_sig
    C_params = {'type' : C_unit_type, #unit_types.am_pm_oscillator,
                'integ_meth' : 'euler_maru', #'euler_maru', #'odeint',
                'tau_u' : 0.5 * randz2(), # 0.4*randz2()
                'tau_c' : 0.1 * randz2(), # 0.2
                'tau_t' : 1.,
                'tau_s' : 0.02,
                #'delay' : 0.35,
                'init_val' : [r*init_base for r in np.random.random(2)],
                'multidim' : False if C_type=="rga_sig" else True,
                'omega' : 2.*np.pi,
                'F' : 'zero', #'input_sum',
                'A' : 0.5,
                'slope' : cfg['C_slope'],
                'thresh' : cfg['C_thresh'],
                'integ_amp' : cfg['C_integ_amp'],
                'tau' : 0.02, # for rga_sig
                'tau_fast': 0.01, #0.02,
                'tau_mid' : 0.05, #0.1,
                'tau_slow' : 3.,
                'custom_inp_del' : int(cfg['C_custom_inp_del']),
                'delay' : max(0.31, (2 + int(cfg['C_custom_inp_del'])) * 
                            net_params['min_delay'] ),
                'mu' : 0.,
                'sigma' : cfg['C_sigma'] }
    # L is the "RBF" layer to represent S_P, S_F
    L_params = {'type' : unit_types.sigmoidal,
                'delay' : .5,
                'thresh' : 2.,
                'slope' : 5.,
                'tau' : 0.02,
                'init_val' : 0.5 }
    L_geom = {'shape' : 'sheet',
            'arrangement' : 'grid',
            'rows' : No2,
            'columns' : No2,
            'center' : [0., 0.],
            'extent' : [1., 1.] }
    
    M_params = {'type' : unit_types.m_sig,
                'thresh' : 0.5 * randz2(),
                'slope' : 2.5 * randz2(),
                'init_val' : 0.2 * randz2(),
                'delay' : max(0.31, (2 + int(cfg['C_custom_inp_del'])) * 
                            net_params['min_delay'] ),
                'n_ports' : 4,
                'tau_fast': 0.01, #0.2, #0.01,
                'tau_mid': 0.05, #1., #0.05,
                'tau_slow' : 8.,
                'tau' : 0.01 * randz2(),
                'coordinates' : [np.array([0.1, 0.8]), np.array([0.4, 0.8])],
                'integ_amp' : 0.,
                'custom_inp_del' : int(np.round(0.3/net_params['min_delay'])) ,
                'des_out_w_abs_sum' : cfg['M_des_out_w_abs_sum'] }
    MPLEX_params = {'type' : unit_types.linear_mplex,
                    'init_val' : 0.,
                    'tau' : 0.01 }
    # plant parameters
    P_params = {'type' : plant_models.pendulum, #bouncy_pendulum,
                'length' : 0.5,
                'mass' : 1.,
                'init_angle' : 0.,
                'init_ang_vel' : 0.,
                'g' : 0.,
                'inp_gain' : cfg['P_inp_gain'],
                'mu' : cfg['P_mu'],
                'bound_angle' : True,
                'pi_visco' : 0.05,
                'tau' : 0.05 } # a made-up time constant for the plant 
                            # see create_freqs_steps below
    # Reward unit parameters
    R_params = {'type' : unit_types.layer_dist,
                'thresh' : cfg['R_thresh'], 
                'slope' : cfg['R_slope'],
                'tau' : 0.01,
                'coordinates' : np.array([.3, .8]),
                'init_val' : 0.3 }
    # S1 and S2 provide the distributed angle representations
    S_params = {'type' : unit_types.bell_shaped_1D,
                'init_val' : 0.1,
                'tau' : 0.01,
                'center' : list(np.linspace(0., 1., SN)),
                'b' : 50. }
    S1_params = S_params.copy() # the input column on the left
    S2_params = S_params.copy() # the input row below
    S1_params['coordinates'] = [np.array([-.6, -.5 + i/SN]) for i in range(SN)]
    S2_params['coordinates'] = [np.array([-.5 + i/SN, -.6]) for i in range(SN)]
    # SF, SP
    SF_params = {'type' : unit_types.sigmoidal,
                'thresh' : 0.,
                'slope' : 1.,
                'init_val' : 0.2,
                'tau' : 0.02 }  # 0.05
    ## The desired values for SF
    des_sf = 1./(1. + np.exp(-SF_params['slope']*(des_angs - SF_params['thresh'])))
    
    SP_params = {'type' : unit_types.source,
                'init_val' : des_sf[0],
                'tau_fast' : 0.01,
                'tau_mid' : 0.2,
                'function' : lambda t: des_sf[int(round(t/pres_interv))] }
    # 1-D error units
    SPF_params = {'type' : unit_types.sigmoidal,
                'thresh' : 0.5 * randz2(),
                'slope' : 5. * randz2(),
                'init_val' : 0.3 * randz2(),
                'tau_fast': 0.005,
                'tau_mid': 0.05,
                'tau_slow' : 5.,
                'tau' : 0.02 * randz2() }
    # Angle transformation unit
    T_params = {'type' : unit_types.source,
                'init_val' : des_sf[0],
                'function' : lambda t: None } # to be set after unit creation
    # value unit
    V_params = {'type' : unit_types.td_sigmo,
                'delay' : .5,
                'init_val' : 0.1,
                'thresh' : cfg['V_thresh'],
                'slope' : cfg['V_slope'],
                'tau' : 0.02,
                'tau_slow' : 300,
                'delta' : cfg['V_delta'],
                'coordinates' : np.array([.8, .8])}
    # The configurator unit
    X_params = {'type' : unit_types.x_switch_sig, #x_sig,
                'init_val' : 0.1,
                'slope' : cfg['X_slope'],
                'thresh' : cfg['X_thresh'],
                'tau' : 0.02,
                'delay' : 0.35,
                'tau_fast' : 0.01,
                'tau_mid' : 0.1,
                'tau_slow' : 10.,
                'integ_amp' : 0.,
                'sw_thresh' : 0.1,
                'sw_len' : 0.2,
                'coordinates' : np.array([0.7, 0.3]),
                'custom_inp_del' :  int(np.round(0.3/net_params['min_delay'])),
                'des_out_w_abs_sum' : 1.} # usually not used
    
    # units to track synaptic weights or other values
    track_params = {'type' : unit_types.source,
                    'init_val' : 0.02,
                    'function' : lambda t: None }
    
    #--------------------------------------------------------------------
    # Connection dictionaries
    #--------------------------------------------------------------------
    # Afferent to motor error selection
    A__M_conn = {'rule' : 'all_to_all',
                'delay' : 0.02 }
    A__M_syn = {'type' : synapse_types.inp_sel, 
                'inp_ports' : 2, # the deault for m_sig targets
                'error_port' : 1, # the default for m_sig targets
                'aff_port' : 2,
                'lrate' : cfg['A__M_lrate'],
                'w_sum' : cfg['A__M_w_sum'],
                'w_max' : cfg['A__M_w_max'],
                'init_w' : .1 }
    # lateral connections in C
    C__C_conn = {'rule': 'one_to_one',
                'allow_autapses' : False,
                'delay' : 0.015 }
    C__C_syn = {'type' : synapse_types.static,
                'lrate' : 0.1,
                'inp_ports': 1,
                'init_w' : cfg['C__C_init_w'] }
    # spinal units to plant
    C0__P_conn = {'inp_ports' : 0,
                'delays': 0.01 }
    C0__P_syn = {'type': synapse_types.static,
                'init_w' : 2. }
    C1__P_conn = C0__P_conn
    C1__P_syn = {'type': synapse_types.static,
                'init_w' : -2. }
    # expanded state to value unit
    L__V_conn = {'rule': 'all_to_all',
                'delay': 0.02 }
    L__V_syn = {'type' : synapse_types.td_synapse,
                'lrate': cfg['L__V_lrate'],
                'gamma' : cfg['L__V_gamma'],
                'inp_ports': 0,
                'init_w' : 0.2 * np.random.random(No2*No2), # 0.05
                'max_w' : cfg['L__V_max_w'],
                'w_sum' : cfg['L__V_w_sum'] }
    # state to configurator
    L__X_conn = {'rule': 'all_to_all',
                'delay': 0.02 }
    L__X_syn = {'type' : synapse_types.diff_rm_hebbian,
                'lrate': cfg['L__X_lrate'],
                'inp_ports': 0, # default for x_sig, m_sig targets
                #'l_port' : 3,
                #'s_port' : 0,
                #'v_port' : 1,
                'init_w' : 0.2 * np.random.random(No2*No2),
                'w_sum' : cfg['L__X_w_sum'] }
    # motor error to spinal
    M__C_conn = {'rule': 'all_to_all',
                'delay': 0.02 }
    M__C_syn = {'type' : synapse_types.rga_21,
                'lrate': cfg['M__C_lrate'],
                'inp_ports': 0,
                'w_sum' : 2.,
                'init_w' : {'distribution':'uniform', 'low':0.05, 'high':.1}}
    # motor error lateral connections
    M__M_conn = {'rule': 'one_to_one',
                'allow_autapses' : False,
                'delay' : 0.01 }
    M__M_syn = {'type' : synapse_types.static,
                'lrate' : 0.1,
                'inp_ports': 3, # default for m_sig targets
                'init_w' : -1. }
    # motor error to plant
    # M0__P_conn = {'inp_ports' : 0,
    #              'delays': 0.01 }
    # M0__P_syn = {'type': synapse_types.static,
    #             'init_w' : 2. }
    # M1__P_conn = M0__P_conn
    # M1__P_syn = {'type': synapse_types.static,
    #              'init_w' : -2. }
    # multiplexer SP selection to X
    MPLEX1__X_conn = {'rule' : 'one_to_one',
                    'delay' : 0.01}
    MPLEX1__X_syn = {'type' : synapse_types.static,
                    'init_w' : 1.,
                    'inp_ports' : 2 }
    # plant to sensory/motor
    P__A_conn = {'port_map' : [[(0,0)], [(1,0)], [(0,0)], [(1,0)]],
                'delays' : 0.02 }
    P__A_syn = {'type' : synapse_types.static,
                'init_w' : [1., 1., -1., -1.] }
    P__SF_conn = {'port_map' : [(0,0)],
                'delays' : 0.01 }
    P__SF_syn = {'type' : synapse_types.static,
                'init_w' : 1. }
    # reward to value unit
    R__V_conn = {'rule' : 'one_to_one',
                'delay' : 0.01 }
    R__V_syn = {'type' : synapse_types.static,
                'inp_ports' : 1,
                'init_w' : 1.}
    # distributed angle to RBF layer
    S1__L_conn_spec = {'connection_type' : 'divergent',
                    'mask' : {'circular' : {'radius' : 2. }},
                    'kernel' : 1.,
                    'delays' : {'linear' : {'c' : 0.01, 'a': 0.01}},
                    'weights' : {'gaussian' : {'w_center' : 1.5, 'sigma' : 0.01}},
                    'dist_dim' : 'y',
                    'edge_wrap' : True,
                    'boundary' : {'center' : [-0.05, 0.], 'extent':[1.1, 1.]} }
    S2__L_conn_spec = S1__L_conn_spec.copy()
    S2__L_conn_spec['dist_dim'] = 'x'
    S2__L_conn_spec['boundary'] = {'center' : [0., -0.05], 'extent':[1., 1.1]}
    S1__L_syn_spec = {'type' : synapse_types.static }
    S2__L_syn_spec = {'type' : synapse_types.static }
    # distributed angle to layer distance (reward)
    S1__R_conn = {'rule': 'all_to_all',
                'delay': 0.01 }
    S1__R_syn = {'type': synapse_types.static,
                'inp_ports' : 0,
                'init_w' : 1. }
    S2__R_conn = {'rule': 'all_to_all',
                'delay': 0.01 }
    S2__R_syn = {'type': synapse_types.static,
                'inp_ports' : 1,
                'init_w' : 1. }
    # SF rate representation to distributed representation
    SF__S_conn = {'rule' : 'all_to_all',
                'delay' : 0.01 }
    SF__S_syn = {'type' : synapse_types.static,
                'init_w' : 1. }
    # SF/SP to multiplexer
    SF__MPLEX_conn = {'rule' : 'all_to_all',
                    'delay' : 0.01 }
    SF__MPLEX_syn = {'type' : synapse_types.static,
                    'inp_ports' : [0, 1],
                    'init_w' : 1. }
    SP__MPLEX_conn = {'rule' : 'all_to_all',
                    'delay' : 0.01 }
    SP__MPLEX_syn = {'type' : synapse_types.static,
                    'inp_ports' : [0, 1],
                    'init_w' : 1. }
    # SF/SP to SPF
    SFe__SPF_conn = {'rule' : "all_to_all",
                    'delay' : 0.01 }
    SFi__SPF_conn = {'rule' : "all_to_all",
                    'delay' : 0.015 }
    SFe__SPF_syn = {'type' : synapse_types.static,
                    'init_w' : 1. }
    SFi__SPF_syn = {'type' : synapse_types.static,
                    'init_w' : -1. }
    SPe__SPF_conn = {'rule' : "all_to_all",
                    'delay' : 0.01 }
    SPi__SPF_conn = {'rule' : "all_to_all",
                    'delay' : 0.015 }
    SPe__SPF_syn = {'type' : synapse_types.static,
                    'init_w' : 1. }
    SPi__SPF_syn = {'type' : synapse_types.static,
                'init_w' : -1. }
    # SP rate representation to distributed representation
    SP__S_conn = SF__S_conn
    SP__S_syn = SF__S_syn
    # SPF to M
    SPF__M_conn = {'rule': 'one_to_one',
                'delay': 0.01 }
    SPF__M_syn = {'type' : synapse_types.static,
                'inp_ports' : 1,
                'lrate' : 20.,
                'input_type' : 'error',
                'init_w' : 1. }
    # rotated angle to SF[1]
    T__SF_conn = {'rule': 'one_to_one',
                'delay': 0.01 }
    T__SF_syn = {'type' : synapse_types.static,
                'inp_ports' : 0,
                'init_w' : P__SF_syn['init_w'] }
    # Value to configurator
    V__X_conn = {'rule': 'all_to_all',
                'delay': 0.01 }
    V__X_syn = {'type' : synapse_types.static,
                'inp_ports' : 1,
                'init_w' : 1. }
    # configurator to multiplexer
    X__MPLEX_conn = {'rule': 'all_to_all',
                    'delay': 0.01 }
    X__MPLEX_syn = {'type' : synapse_types.static,
                    'inp_ports' : 2,
                    'init_w' : 1. }
    
    def create_freqs_steps(n, w, r):
        """ Returns a 2-tuple with the lists required for heterogeneous frequencies.
        
            We assume that the important loop for the learning rule in the C units
            is the one going through C-P-M-C
            Args:
                n : number of units
                w : base omega value (rad)
                r : amplitude of noise as fraction of original value
            Returns
                2-tuple : (freqs, steps)
                freqs : a list with n angular frequencies.
                steps : a list with the corresponding delays. 
        """
        ws = w * (1. + r*((np.random.random(n) - 0.5)))
        ws = ws / C_params['tau_t'] # angular frequencies
        cp_del = np.arctan(np.mean(P_params['tau'])*ws)/ws
        pm_del = np.arctan(np.mean(M_params['tau'])*ws)/ws
        am_del = np.arctan(np.mean(A_params['tau'])*ws)/ws
        D = [C0__P_conn['delays'], np.mean(P__A_conn['delays']), 
            A__M_conn['delay'], M__C_conn['delay'] ]
        d1 = cp_del + pm_del + am_del + sum(D)
        del_steps = [int(d) for d in np.ceil(d1/net_params['min_delay'])]
        return (list(ws), del_steps)
    
    #--------------------------------------------------------------------
    # CREATING UNITS
    #--------------------------------------------------------------------
    net = network(net_params)
    topo = topology()
    
    A = net.create(4, A_params)
    # creating C with heterogeneous frequencies
    omegas, del_steps = create_freqs_steps(2, C_params['omega'], par_heter)
    C_params['omega'] = omegas
    C_params['custom_inp_del'] = del_steps
    C = net.create(2, C_params)
    L = topo.create_group(net, L_geom, L_params)
    M = net.create(2, M_params)
    MPLEX = net.create(2, MPLEX_params)
    P = net.create(1, P_params)
    R = net.create(1, R_params)
    S1 = net.create(SN, S1_params)
    S2 = net.create(SN, S2_params)
    SF = net.create(2, SF_params)
    SP = net.create(2, SP_params)
    SPF = net.create(2, SPF_params)
    T = net.create(1, T_params)
    V = net.create(1, V_params)
    X = net.create(1, X_params)
    
    # tracking units
    #M_C0_track = net.create(2, track_params) # to track weights from M to C0
    #A_M0_track = net.create(4, track_params) # to track weights from A to M0
    #xp_track = net.create(1, track_params) # del_avg_inp_deriv of C0 at port 1
    #up_track = net.create(1, track_params) # to track the derivative of C0
    #sp_track = net.create(1, track_params) # avg_inp_deriv_mp for C0 at port 0
    #spj_track = net.create(1, track_params) # input derivative for SPF0--C0
    #if C_type != "rga_sig":
        #dc_track = net.create(2, track_params) # DC component of C units
        
    # Configuring transformation unit
    def t_fun(time):
        """ Pendulum angle after a \pi rads rotation of the coordinate system.
        
            Normally the bound angle for the pendulum is between -pi and pi. This
            function retrieves the pendulum angle in the 0, 2pi interval.
        
            Args:
                time : time at which the angle is retrieved
            Returns:
                1-D scalar with rotated angle.
        """
        x = net.plants[P].get_state_bound(time)[0]
        return x - np.pi if x < np.pi and x > 0 else np.pi + x
    net.units[T[0]].set_function(t_fun)
    
    # Configuring alternate desired angle unit
    def alt_des_fun(time):
        """ The alternate desired SF value. """
        sd = des_sf[int(round(time/pres_interv))]
        sd_inv = np.log(sd / (1.-sd))/SF_params['slope'] + SF_params['thresh']
        t = sd_inv - np.pi if sd_inv < np.pi and sd_inv > 0 else np.pi + sd_inv
        return 1. / ( 1. + np.exp(-SF_params['slope']*(t - SF_params['thresh'])) )
    net.units[SP[1]].set_function(alt_des_fun)
        
    #--------------------------------------------------------------------
    # CONNECTING
    #--------------------------------------------------------------------
    net.connect(A, M, A__M_conn, A__M_syn) # this connection should be made before
    net.connect([C[0]], [C[1]], C__C_conn, C__C_syn)
    net.connect([C[1]], [C[0]], C__C_conn, C__C_syn)
    net.set_plant_inputs([C[0]], P, C0__P_conn, C0__P_syn)
    net.set_plant_inputs([C[1]], P, C1__P_conn, C1__P_syn)
    net.connect(L, V, L__V_conn, L__V_syn)
    net.connect(L, X, L__X_conn, L__X_syn)
    net.connect(M, C, M__C_conn, M__C_syn)
    net.connect([M[0]], [M[1]], M__M_conn, M__M_syn)
    net.connect([M[1]], [M[0]], M__M_conn, M__M_syn)
    
    net.connect([MPLEX[0]], [SPF[0]], SFe__SPF_conn, SFe__SPF_syn)
    net.connect([MPLEX[1]], [SPF[0]], SPi__SPF_conn, SPi__SPF_syn)
    net.connect([MPLEX[0]], [SPF[1]], SFi__SPF_conn, SFi__SPF_syn)
    net.connect([MPLEX[1]], [SPF[1]], SPe__SPF_conn, SPe__SPF_syn)
    
    net.connect([MPLEX[1]], X, MPLEX1__X_conn, MPLEX1__X_syn)
    
    net.set_plant_outputs(P, [SF[0]], P__SF_conn, P__SF_syn)
    net.set_plant_outputs(P, A, P__A_conn, P__A_syn) 
    topo.topo_connect(net, S1, L, S1__L_conn_spec, S1__L_syn_spec)
    topo.topo_connect(net, S2, L, S2__L_conn_spec, S2__L_syn_spec)
    net.connect(S1, R, S1__R_conn, S1__R_syn)
    net.connect(S2, R, S2__R_conn, S2__R_syn)
    net.connect([SF[0]], S1, SF__S_conn, SF__S_syn)
    
    net.connect(SF, [MPLEX[0]], SF__MPLEX_conn, SF__MPLEX_syn)
    net.connect(SP, [MPLEX[1]], SP__MPLEX_conn, SP__MPLEX_syn)
    
    net.connect([SP[0]], S2, SP__S_conn, SP__S_syn)
    net.connect(SPF, M, SPF__M_conn, SPF__M_syn)
    net.connect(R, V, R__V_conn, R__V_syn)
    net.connect(T, [SF[1]], T__SF_conn, T__SF_syn)
    net.connect(V, X, V__X_conn, V__X_syn)
    net.connect(X, MPLEX, X__MPLEX_conn, X__MPLEX_syn)
    
    # SETTING UP WEIGHT TRACKING -- depends on the order of statements above!!!!!!
    # This is dependent on the order in which net.connect is called above;
    #net.units[M_C0_track[0]].set_function(lambda t: net.syns[C[0]][1].w) # M0--C0
    #net.units[M_C0_track[1]].set_function(lambda t: net.syns[C[0]][2].w) # M1--C0
    # Nothing should connect to M0 before A
    #net.units[A_M0_track[0]].set_function(lambda t: net.syns[M[0]][0].w) # P0--M0
    #net.units[A_M0_track[1]].set_function(lambda t: net.syns[M[0]][1].w) # P1--M0
    #net.units[A_M0_track[2]].set_function(lambda t: net.syns[M[0]][2].w) # -P0--M0
    #net.units[A_M0_track[3]].set_function(lambda t: net.syns[M[0]][3].w) # -P1--M0
    
    # SETTING TRACKING OF PLASTICITY FACTORS FOR M0-->C0
    #net.units[xp_track[0]].set_function(lambda t: net.units[C[0]].del_avg_inp_deriv_mp[1])
    #po_de = net.units[C[0]].custom_inp_del
    #net.units[up_track[0]].set_function(lambda t: net.units[C[0]].get_lpf_fast(po_de) - 
                                        #net.units[C[0]].get_lpf_mid(po_de))
    #net.units[sp_track[0]].set_function(lambda t: net.units[C[0]].avg_inp_deriv_mp[0])
    #ds = net.syns[C[0]][0].delay_steps
    #net.units[spj_track[0]].set_function(lambda t: net.units[M[0]].get_lpf_fast(ds) - 
                                        #net.units[M[0]].get_lpf_mid(ds))
    
    # SETTING UP TRACKING OF C STATE VARIABLES
    #if C_type != "rga_sig":
        #net.units[dc_track[0]].set_function(lambda t: net.units[C[0]].buffer[1, -1])
        #net.units[dc_track[1]].set_function(lambda t: net.units[C[1]].buffer[1, -1])
    

    #--------------------------------------------------------------------
    # SETTING INITIAL CONTROLLER WEIGHTS, SLOWING PLASTICITY
    #--------------------------------------------------------------------
    # standard values:
    # A__M_mat = np.rray([[0.27141863, 0.29511766, 0.23279135, 0.20067437],
    #                     [0.23307357, 0.1957229 , 0.28739239, 0.28380952]])
    
    # M__C_mat = np.array([[0.0295056,  2.05108517],
    #                      [2.05111951, 0.02950648]])
    # limit values:
    A__M_mat = np.array([[0.4, .5, 0.08, 0.018],
                        [0.05, 0. , 0.4, 0.5]])
    
    M__C_mat = np.array([[0.,  2.15],
                        [2.15, 0.]])
    
    if rand_w:
        for m_idx, m_id in enumerate(M):
            for c_idx, c_id in enumerate(C):
                syn_list = net.syns[c_id]
                for syn in syn_list:
                    if syn.preID == m_id:
                        syn.w = M__C_mat[c_idx, m_idx]
                        syn.alpha = 1e-5 # slowing down learning
                        break
        
        for a_idx, a_id in enumerate(A):
            for m_idx, m_id in enumerate(M):
                syn_list = net.syns[m_id]
                for syn in syn_list:
                    if syn.preID == a_id:
                        syn.w = A__M_mat[m_idx, a_idx]
                    syn.alpha = 1e-5 # slowing down learning
                    break
                
    #--------------------------------------------------------------------
    # RETURN NETWORK, POPULATION IDs
    #--------------------------------------------------------------------
    pops_list = [A, C, L, M, MPLEX, P, R, S1, S2, SF, SP, SPF, T, V, X]
    pops_names = ['A', 'C', 'L', 'M', 'MPLEX', 'P', 'R', 'S1', 'S2',
                 'SF', 'SP', 'SPF', 'T', 'V', 'X']
    pops_dict = {pops_names[idx] : pops_list[idx] for idx in range(len(pops_names))}

    return net, pops_dict
