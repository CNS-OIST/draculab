# coding: utf-8

# rl5E_lite_from_cfg.py
# A function to be used with rl5E_lite_hyper.ipynb

from draculab import *

def rl5E_net(cfg,
             pres_interv=5.,
             rand_w=False,
             par_heter=0.1,
             x_switch=True,
             V_normalize=True,
             X_normalize=True):
    """ Create a network as in rl5E_lite.ipynb from a given configuration.


        Args:
            cfg : a paramter dictionary.

            Optional keyword arguments:
            pres_interv : time itnerval to present each desired value
            rand_w : whether to use random A-->M, M-->C weights initially
            par_heter : parameter heterogeneity as a fraction of original value
            X_switch : whether the X unit switches value each presentation
            V_normalize : whether the L__V weights have a constant sum
            X_normalize : whether the L__X weights have a constant sum
        Returns:
            A tuple with the following entries:
            net : A draculab network as in rl5E_lite from the cfg configuration
            pops_dict : dictionary with the ID list for each population in net.
    """
    #===================================================================
    #================ CREATE THE NETWORK ===============================
    #===================================================================

    #np.random.seed(123456) # always the same random values
    np.random.seed()   # different random values ever run

    cfg['X_switch'] = x_switch
    cfg['V_normalize'] = V_normalize
    cfg['X_normalize'] = X_normalize
    if not 'X_type' in cfg: cfg['X_type'] = unit_types.x_net
    if not 'r_thr' in cfg: cfg['r_thr'] = np.pi/6.
    if not 'X_tau_slow' in cfg: cfg['X_tau_slow'] = 50.
    if not 'X_refr_per' in cfg: cfg['X_refr_per'] = 2.

    # some parameters 
    C_type = "rga_sig" # unit type for the C population
    des_angs = np.pi*(-1. + 2.*np.random.random(10000))
    #des_sf = 0.05 + 0.8*np.random.random(10000) # list of "predicted" SF values| set below now
    #pres_interv = 5. # interval that desired values last
    #par_heter = 0.1 # range of heterogeneity as a fraction of the original value
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
                'slope' : cfg['C_slope'], # for rga_sig
                'thresh' : cfg['C_thresh'], # for rga_sig
                'integ_amp' : cfg['C_integ_amp'], # for rga_sig
                'tau' : 0.02, # for rga_sig
                'tau_fast': 0.01, #0.02,
                'tau_mid' : 0.05, #0.1,
                'tau_slow' : 3.,
                'custom_inp_del' : cfg['C_custom_inp_del'],
                'delay' : 0.31,
                'mu' : 0.,
                'sigma' : cfg['C_sigma'] }
    # desired angle
    da = np.array([a if a > 0. else 2.*np.pi + a for a in des_angs])
    DA_params = {'type' : unit_types.source,
                 'init_val' : da[0],
                 'function' : lambda t: da[int(round(t/pres_interv))] }
    M_params = {'type' : unit_types.m_sig,
                'thresh' : 0.5 * randz2(),
                'slope' : 2.5 * randz2(),
                'init_val' : 0.2 * randz2(),
                'delay' : 0.35,
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
                    'delay' : 0.5,
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
                'delay' : None, # set after V_params
                'bound_angle' : True,
                'pi_visco' : 0.05,
                'tau' : 0.05 } # a made-up time constant for the plant 
                              # see create_freqs_steps below
    # Reward unit parameters
    R_params = {'type' : unit_types.source,
                'function' : lambda t: None, # to be set below
                'coordinates' : np.array([.3, .8]),
                'init_val' : 0.3 }
   # SF, SP
    SF_params = {'type' : unit_types.sigmoidal,
                 'thresh' : 0.,
                 'slope' : cfg['SF_slope'],
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
    V_params = {'type' : unit_types.v_net,
                'multidim' : True,
                'delay' : .5,
                'init_val' : [0.1]*101,
                'thresh' : cfg['V_thresh'],
                'slope' : cfg['V_slope'],
                'L_wid' : .5 * No2 / np.pi,
                #'R_wid' : 2.,
                'tau' : 0.02,
                'tau_slow': 50.,
                'delta' : cfg['V_delta']-cfg['V_delta']%net_params['min_delay'],
                'td_lrate' : cfg['V_td_lrate'],
                'td_gamma' : cfg['V_td_gamma'],
                'normalize' : cfg['V_normalize'],
                'w_sum' : cfg['V_w_sum'],
                'coordinates' : np.array([.8, .8])}
    P_params['delay'] = V_params['delta'] + net_params['min_delay']
    # The configurator unit
    X_del = cfg['X_del'] - cfg['X_del']%net_params['min_delay']
    X_params = {'type' : cfg['X_type'],
                'multidim' : True,
                'init_val' : np.concatenate((np.array([0.5]), 0.1*np.ones(100))),
                'tau' : 0.02,
                'slope' : cfg['X_slope'],
                'thresh' : cfg['X_thresh'],
                'del_steps' : int(np.round(cfg['X_del']/net_params['min_delay'])),
                'lrate' : cfg['X_lrate'],
                'L_wid' : .5 * No2 / np.pi,
                'delay' : X_del + 2.*net_params['min_delay'],
                'tau_fast' : 0.02,
                'tau_mid' : 0.2,
                'tau_slow' : cfg['X_tau_slow'],
                'sw_thresh' : 0.03,
                'sw_len' : 0.4,
                'switch' : cfg['X_switch'],
                'normalize' : cfg['X_normalize'],
                'w_sum' : cfg['X_w_sum'],
                'refr_per' : cfg['X_refr_per'],
                'beta' : 2.,
                'r_thr' : cfg['r_thr'],
                'coordinates' : np.array([0.7, 0.3]) }

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
                'lrate' : cfg['A__M_lrate'], # negative rate for m_sig targets with value inputs
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
                'init_w' : -1.5 }  # Changed from usual -1
    # spinal units to plant
    C0__P_conn = {'inp_ports' : 0,
                 'delays': 0.01 }
    C0__P_syn = {'type': synapse_types.static,
                'init_w' : 2. }
    C1__P_conn = C0__P_conn
    C1__P_syn = {'type': synapse_types.static,
                 'init_w' : -2. }
    DA__VX_conn = {'rule' : 'one_to_one',
                   'delay' : 0.01 }
    DA__VX_syn = {'type' : synapse_types.static,
                  'init_w' : 1.,
                 'inp_ports' : 1 }
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
   # plant to sensory/motor
    P__A_conn = {'port_map' : [[(0,0)], [(1,0)], [(0,0)], [(1,0)]],
                 'delays' : 0.02 }
    P__A_syn = {'type' : synapse_types.static,
                'init_w' : [1., 1., -1., -1.] }
    P__SF_conn = {'port_map' : [(0,0)],
                  'delays' : 0.01 }
    P__SF_syn = {'type' : synapse_types.static,
                 'init_w' : 1. }
    # plant to V,X
    P__VX_conn = {'port_map' : [(0,0)],
                  'delays' : 0.02 }
    P__VX_syn = {'type' : synapse_types.static,
                 'init_w' : 1. }
    # reward to value unit
    R__V_conn = {'rule' : 'one_to_one',
                 'delay' : 0.01 }
    R__V_syn = {'type' : synapse_types.static,
                'inp_ports' : 2,
                'init_w' : 1.}
    # SF rate representation to distributed representation
    #SF__S_conn = {'rule' : 'all_to_all',
    #             'delay' : 0.01 }
    #SF__S_syn = {'type' : synapse_types.static,
    #            'init_w' : 1. }
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
    #SP__S_conn = SF__S_conn
    #SP__S_syn = SF__S_syn
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
                'inp_ports' : 2,
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
    DA = net.create(1, DA_params)
    #L = topo.create_group(net, L_geom, L_params)
    M = net.create(2, M_params)
    MPLEX = net.create(2, MPLEX_params)
    P = net.create(1, P_params)
    R = net.create(1, R_params)
    #S1 = net.create(SN, S1_params)
    #S2 = net.create(SN, S2_params)
    SF = net.create(2, SF_params)
    SP = net.create(2, SP_params)
    SPF = net.create(2, SPF_params)
    T = net.create(1, T_params)
    V = net.create(1, V_params)
    X = net.create(1, X_params)

    # tracking units
    #M_C0_track = net.create(2, track_params) # to track weights from M to C0
    #A_M0_track = net.create(2, track_params) # to track weights from A to M0
    #xp_track = net.create(1, track_params) # del_avg_inp_deriv of C0 at port 1
    #up_track = net.create(1, track_params) # to track the derivative of C0
    #sp_track = net.create(1, track_params) # avg_inp_deriv_mp for C0 at port 0
    #spj_track = net.create(1, track_params) # input derivative for SPF0--C0
    #v_track = net.create(No2*No2, track_params) # L__V weights
    #x_track = net.create(No2*No2, track_params) # L__X weights

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
        sd = des_sf[int(np.floor(time/pres_interv))]
        sd_inv = np.log(sd / (1.-sd))/SF_params['slope'] + SF_params['thresh']
        t = sd_inv - np.pi if sd_inv < np.pi and sd_inv > 0 else np.pi + sd_inv
        return 1. / ( 1. + np.exp(-SF_params['slope']*(t - SF_params['thresh'])) )
    net.units[SP[1]].set_function(alt_des_fun)

    # configuring reward unit
    def r_fun(time):
        """ The reward value at a given time. """
        dang = da[int(np.floor(time/pres_interv))] # desired angle in [0, 2pi]
        cang = net.plants[P].get_angle(time) # angle in [0,2*pi] interval
        dist = min(abs(dang-cang), 2.*np.pi - max(dang,cang) + min(dang,cang))
        return np.exp(-dist*dist)
    net.units[R[0]].set_function(r_fun)

    #--------------------------------------------------------------------
    # CONNECTING
    #--------------------------------------------------------------------
    net.connect([A[1],A[3]], M, A__M_conn, A__M_syn) # only velocity afferents
    net.connect([C[0]], [C[1]], C__C_conn, C__C_syn)
    net.connect([C[1]], [C[0]], C__C_conn, C__C_syn)
    net.set_plant_inputs([C[0]], P, C0__P_conn, C0__P_syn)
    net.set_plant_inputs([C[1]], P, C1__P_conn, C1__P_syn)
    net.connect(DA, V, DA__VX_conn, DA__VX_syn)
    net.connect(DA, X, DA__VX_conn, DA__VX_syn)
    net.connect(M, C, M__C_conn, M__C_syn)
    net.connect([M[0]], [M[1]], M__M_conn, M__M_syn)
    net.connect([M[1]], [M[0]], M__M_conn, M__M_syn)

    net.connect([MPLEX[0]], [SPF[0]], SFe__SPF_conn, SFe__SPF_syn)
    net.connect([MPLEX[1]], [SPF[0]], SPi__SPF_conn, SPi__SPF_syn)
    net.connect([MPLEX[0]], [SPF[1]], SFi__SPF_conn, SFi__SPF_syn)
    net.connect([MPLEX[1]], [SPF[1]], SPe__SPF_conn, SPe__SPF_syn)

    net.set_plant_outputs(P, [SF[0]], P__SF_conn, P__SF_syn)
    net.set_plant_outputs(P, A, P__A_conn, P__A_syn) 
    net.set_plant_outputs(P, V, P__VX_conn, P__VX_syn)
    net.set_plant_outputs(P, X, P__VX_conn, P__VX_syn)

    net.connect(SF, [MPLEX[0]], SF__MPLEX_conn, SF__MPLEX_syn)
    net.connect(SP, [MPLEX[1]], SP__MPLEX_conn, SP__MPLEX_syn)

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
    #net.units[A_M0_track[0]].set_function(lambda t: net.syns[M[0]][0].w) # A0--M0
    #net.units[A_M0_track[1]].set_function(lambda t: net.syns[M[0]][1].w) # A1--M0
    #net.units[A_M0_track[2]].set_function(lambda t: net.syns[M[0]][2].w) # -A0--M0
    #net.units[A_M0_track[3]].set_function(lambda t: net.syns[M[0]][3].w) # -A1--M0

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

    # SETTING UP TRACKING OF V STATE VARIABLES
    #def v_track_fun(idx):
        #return lambda t: net.units[V[0]].buffer[1+idx,-1]
    #for idx, uid in enumerate(v_track):
        #net.units[uid].set_function(v_track_fun(idx))

    # SETTING UP TRACKING OF X STATE VARIABLES
    #def x_track_fun(idx):
        #return lambda t: net.units[X[0]].buffer[1+idx,-1]
    #for idx, uid in enumerate(x_track):
        #net.units[uid].set_function(x_track_fun(idx))


    A__M_mat = np.array([[0.4, 0.1],
                        [0.1, 0.4]])

    M__C_mat = np.array([[0.,  2.15],
                        [2.15, 0.]])

    if not rand_w:
        for m_idx, m_id in enumerate(M):
            for c_idx, c_id in enumerate(C):
                syn_list = net.syns[c_id]
                for syn in syn_list:
                    if syn.preID == m_id:
                        syn.w = M__C_mat[c_idx, m_idx]
                        syn.alpha = 1e-4 # slowing down learning
                        break

        for a_idx, a_id in enumerate([A[1], A[3]]):
            for m_idx, m_id in enumerate(M):
                syn_list = net.syns[m_id]
                for syn in syn_list:
                    if syn.preID == a_id:
                        syn.w = A__M_mat[m_idx, a_idx]
                        syn.alpha = 1e-4 # slowing down learning
                        break

    #--------------------------------------------------------------------
    # RETURN NETWORK, POPULATION IDs
    #--------------------------------------------------------------------
    pops_list = [A, C, DA, M, MPLEX, P, R, SF, SP, SPF, T, V, X,
                 ','.join([str(a) for a in des_angs])]
    pops_names = ['A', 'C', 'DA', 'M', 'MPLEX', 'P', 'R', 
                  'SF', 'SP', 'SPF', 'T', 'V', 'X', 'des_angs']
    pops_dict = {pops_names[idx] : pops_list[idx] for idx in range(len(pops_names))}

    return net, pops_dict
