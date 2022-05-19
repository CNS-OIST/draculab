# coding: utf-8

# net_builders.py
# A function to create the nets in the reach_analysis_C
# simulations.
# This is a copy of v3ft3p2ph2_net (used in reach_analysis_A,
# reach_analysis_B) with some modifications:
# * inp_sel connections from A to CE, CI are added


from draculab import *

def net_from_cfg(cfg,
                 t_pres = 40.,
                 par_heter = 0.01,
                 set_C_delay = False,
                 rand_targets = True,
                 track_weights = False,
                 track_ips = False,
                 C_noise = False,
                 M__C_rand_w = True,
                 M_noise = False,
                 rga_on_M = False,
                 rdc_on_M = False,
                 rot_SPF = False,
                 M__M_conns = False,
                 old_M__C = True):
    """ Create a draculab network with the given configuration. 

        Args:
            cfg : a parameter dictionary
            
            Optional keyword arguments:
            t_pres: number of seconds to hold each set of target lengths
            par_heter: range of heterogeneity as a fraction of the original value
            set_C_delay: whether set C_cid using analytical approach
            rand_targets: whether to train using a large number of random targets
            C_noise: whether C units are noisy (use euler_maru integrator)
            M__C_rand_w: randomly initialize weights for the M__C connections
            M_noise: whether M units are noisy (use euler_maru integration)
            rga_on_M: whether to have rga_21 synapses on SPF__M connections
            rdc_on_M: use rate distribution control on M?
            rot_SPF: whether to rotate the output of SPF
            M__M_conns : whether to have M__M_connections
            old_M__C: M not permuted, and static connections from M can inhibit.

        Returns:
            A tuple with the following entries:
            net : A draculab network as in v3_nst_afx, with the given configuration.
            pops_dict : a dictionary with the list of ID's for each population in net.
            hand_coords : list with the coordinates for all possible targets
            m_idxs : which target coordinats will be used for the i-th presentation
            pds : some parameter dictionaries used to change targets
    """
    M_size = 12 # number of units in the M population
    SPF_size = 12 # number of units in the SPF population

    if not C_noise and not 'C_sigma' in cfg:
        cfg['C_sigma'] = 0.
    if not 'C_tau' in cfg:
        cfg['C_tau'] = 0.05
    if not 'C_tau_slow' in cfg:
        cfg['C_tau_slow'] = 10.
    if not 'k_pe_e' in cfg:
        cfg['k_pe_e'] = 20.
    if not 'k_se_e' in cfg:
        cfg['k_se_e'] = 20.

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameter dictionaries for the network and the plant
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    net_params = {'min_delay' : 0.005,
                  'min_buff_size' : 8 }
    # plant parameters
    P_params = {'type' : plant_models.bouncy_planar_arm_v3,
              'mass1': 1.,
              'mass2': 1.,
              's_min' : -0.8,
              'p1' : (-0.01, 0.04),
              'p2' : (0.29, 0.03),
              'p3' : (0., 0.05),
              'p4' : (0.1, 0.04),
              'p5' : (0.01, -0.05),
              'p6' : (0.11, -0.04),
              'p10': (0.29, 0.03),
              'init_q1': 0.,
              'init_q2': np.pi/2.,
              'init_q1p': 0.,
              'init_q2p': 0.,
              'g': 0.0,
              'mu1': 3.,
              'mu2': 3.,
              'l_torque' : 0.01,
              'l_visco' : 0.01,
              'g_e' : cfg['g_e_factor']*np.array([cfg['g_e_03'], 25., 25., # 20, 20,
                                                  cfg['g_e_03'], 25., 25.]), #30., 30.]), # 22., 23.]),
              'l0_e' : [1.]*6,
              'Ia_gain' : 2.5*np.array([3.,10.,10., 3.,11.,11.]), # 3,10,10,3,10,10
              'II_gain' : 2.*np.array([cfg['II_g_03'], 8., 8.,
                                       cfg['II_g_03'], 8., 8.]), # 8., 8.]),
              'Ib_gain' : 1.,
              'T_0' : 10.,
              'k_pe_e' : cfg['k_pe_e'],  #8
              'k_se_e' : cfg['k_se_e'], #13
              'b_e' : cfg['b_e'],
              'g_s' : 0.02,
              'k_pe_s' : [2., 2., 2., 2., 2., 2.], #2., 
              'k_se_s' : [2., 2., 2., 2., 2., 2.], # 2.,
              'g_d' : 0.01,
              'k_pe_d' : .2, #.1,
              'k_se_d' : 1., #2.,
              'b_s' : .5,
              'b_d' : 2.,#3.,
              'l0_s': .7,
              'l0_d': .8,
              'fs' : 0.1,
              'se_II' : 0.5,
              'cd' : 0.5,
              'cs' : 0.5,
              'tau' : 0.1   # ficticious time constant used in create_freqs_steps
               }
    #--------------------------------------------------------------------
    # Unit parameters
    #--------------------------------------------------------------------
    randz = lambda n: (1. + par_heter*(np.random.rand(n)-0.5))

    A_params = {'type' : unit_types.logarithmic,
                'init_val' : 0.,
                'tau' : 0.01, # 0.02
                'tau_fast': 0.005,
                'tau_mid' : 0.05,
                'thresh' : np.array([.35]*6 + 
                                    [-.15]*6 + 
                                    [1.5, .12, .22, 1.2, .2, .2])} #[1.5,.12,.22,1.2,.14,.26] } # [Ib, Ia, II]
    ACT_params = {'type' : unit_types.act,
                  'tau_u' : 10., #6., #8
                  'gamma' : 8., #6., #2
                  'g' : 2.,
                  'theta' : 1.,
                  'tau_slow' : 5.,
                  'y_min' : 0.1, #0.2
                  'rst_thr' : 0.1,
                  'init_val' : 0. }
    spf_sum_min = .6 # value where no corrections are needed anymore
    y_min = 1./(1. + np.exp(-ACT_params['g']*(spf_sum_min - ACT_params['theta'])))
    ACT_params['y_min'] = y_min

    AL_params = {'type' : unit_types.rga_sig,
                 'thresh' : cfg['AL_thresh'] * randz(6),
                 'slope' : 2. * randz(6),
                 'init_val' : 0.1 * randz(6),
                 'delay' : 0.31, # to be set below
                 'multidim' : False,
                 'n_ports' : 3,
                 'tau_fast': 0.01,
                 'tau_mid' : 0.05,
                 'tau_slow' : cfg['C_tau_slow'],
                 'integ_amp' : 0.,
                 'custom_inp_del' : int(round(
                                    (cfg['C_cid']-0.01)/net_params['min_delay'])),
                 'tau' : 0.02 * randz(6) }
    CE_params = {'type' : unit_types.rga_inpsel_adapt_sig,
                'integ_meth' : 'euler_maru' if C_noise else 'odeint',
                'init_val' : [r*np.array([0.5]) for r in np.random.random(6)],
                'multidim' : False,
                'slope' : cfg['C_slope'],
                'thresh' : cfg['CE_thresh'],
                'tau' : cfg['C_tau'],
                'tau_fast': 0.01,
                'tau_mid' : 0.05,
                'tau_slow' : cfg['C_tau_slow'],
                'custom_inp_del' : int(round(cfg['C_cid']/net_params['min_delay'])),
                'delay' : 0.31, # to be set below
                'adapt_amp' : cfg['C_adapt_amp'],
                'mu' : 0.,
                'sigma' : cfg['C_sigma'] }
    CI_params = {'type' : unit_types.rga_inpsel_adapt_sig,
                'integ_meth' : 'euler_maru' if C_noise else 'odeint',
                'init_val' : [r*np.array([0.5]) for r in np.random.random(6)],
                'multidim' : False,
                'slope' : cfg['CI_slope'] if 'CI_slope' in cfg else cfg['C_slope'],
                'thresh' : cfg['CI_thresh'],
                'tau' :  cfg['CI_tau'] if 'CI_tau' in cfg else cfg['C_tau'],
                'tau_fast': 0.01,
                'tau_mid' : 0.05,
                'tau_slow' : cfg['C_tau_slow'],
                'custom_inp_del' : int(round(cfg['C_cid']/net_params['min_delay'])),
                'delay' : 0.31, # to be set below
                'adapt_amp' : cfg['C_adapt_amp'],
                'mu' : 0.,
                'sigma' : cfg['C_sigma'] }
    M_params = {'type' : unit_types.m_sig,
                'integ_meth' : 'euler_maru' if M_noise else 'odeint',
                'thresh' : cfg['M_thresh'] * randz(M_size) + cfg['A__M_w_sum'] / 2.,
                'slope' : cfg['M_slope'] * randz(M_size),
                'init_val' : 0.2 * randz(M_size),
                'delay' : 0.35, # to be set below
                'n_ports' : 4,
                'tau_fast': 0.01,
                'tau_mid': 0.05,
                'tau_slow' : 8.,
                'tau' : cfg['M_tau'] * randz(M_size),
                'integ_amp' : 0.,
                'inp_deriv_ports' : [[1,3] for _ in range(M_size)],
                'custom_inp_del' : int(np.round(cfg['M_cid']/net_params['min_delay'])),
                'des_out_w_abs_sum' : cfg['M_des_out_w_abs_sum'],
                'use_rdc' : rdc_on_M,
                'mu' : 0.,
                'sigma' : 0 if not 'M_sigma' in cfg else cfg['M_sigma'] }
    if rdc_on_M:
        M_params['c'] = cfg['M_c']
        M_params['thr_fix'] = sum(M_params['thresh'])/len(M_params['thresh'])
        M_params['tau_fix'] = 2.
        M_params['tau_thr'] = cfg['M_tau_thr']
        M_params['rdc_port'] = 3
        M_params['sharpen_port'] = 0
    # SF, SP
    SFth = cfg['SF_thresh_03']
    SF_params = {'type' : unit_types.sigmoidal,
                 'thresh' : np.array([SFth, 0.4, 0.4, SFth, 0.3, 0.4]), # 0.4, 0.4]),
                 'slope' : np.array([cfg['SF_slope']]*6),
                 'init_val' : 0.2,
                 'tau' : 0.02 }  # 0.05
    SP_params = {'type' : unit_types.source,
                 'init_val' : 0.5,
                 'tau_fast' : 0.01,
                 'tau_mid' : 0.2,
                 'function' : lambda t: None }
    SP_CHG_params = {'type' : unit_types.sigmoidal,
                  'thresh' : 0.25,
                  'slope' : 9.,
                  'init_val' : 0.1,
                  'tau' : 0.01 }
    # 1-D error units
    dowas = 1. if not 'SPF_des_out_w_abs_sum' in cfg else cfg['SPF_des_out_w_abs_sum']
    SPF_params = {'type' : unit_types.out_norm_sig,
                  'thresh' : 0.1 * randz(SPF_size),
                  'slope' : 9. * randz(SPF_size),
                  'init_val' : 0.3 * randz(SPF_size),
                  'tau_fast': 0.005,
                  'tau_mid': 0.05,
                  'tau_slow' : 5.,
                  'tau' : 0.02 * randz(SPF_size),
                  'delay' : 0.35, # to be set below
                  'des_out_w_abs_sum' : dowas }
    # units to track synaptic weights or other values
    track_params = {'type' : unit_types.source,
                    'init_val' : 0.02,
                    'function' : lambda t: None }

    #--------------------------------------------------------------------
    # Connection dictionaries
    #--------------------------------------------------------------------
    # We organize the spinal connections through 4 types of symmetric relations
    # these lists are used to set intraspinal connections and test connection matrices
    antagonists = [(0,3), (1,2), (4,5)]
    part_antag = [(0,2),(0,5), (3,4), (1,3)]
    synergists = [(0,1), (0,4), (2,3), (3,5)]
    part_syne = [(1,4), (2,5)]
    self_conn = [(x,x) for x in range(6)]

    antagonists += [(p[1],p[0]) for p in antagonists]
    part_antag += [(p[1],p[0]) for p in part_antag]
    synergists += [(p[1],p[0]) for p in synergists]
    part_syne += [(p[1],p[0]) for p in part_syne]
    all_pairs = [(i,j) for i in range(6) for j in range(6)]
    
    # Afferent to interneurons --------------------------------
    A__C_conn = {'rule' : 'all_to_all',
                 'delay' : 0.01}
    A__C_syn = {'type' : synapse_types.inp_sel,
                'inp_ports' : 2,
                'error_port' : 0,
                'lrate' : cfg['A__C_lrate'],
                'w_sum' : cfg['A__C_w_sum'],
                'w_max' : cfg['A__C_w_max_frac']*cfg['A__C_w_sum'],
                'init_w' : 0.1 }
    # Afferent to motor error selection -----------------------
    A__M_conn = {'rule' : 'all_to_all',
                 'delay' : 0.02 }
    A__M_syn = {'type' : synapse_types.inp_sel,
                'inp_ports' : 2, # the default for m_sig targets
                'error_port' : 1, # the default for m_sig targets
                'aff_port' : 2,
                'lrate' : cfg['A__M_lrate'], # negative rate for m_sig targets with value inputs
                'w_sum' : cfg['A__M_w_sum'],
                'w_max' : cfg['A__M_w_max_frac']*cfg['A__M_w_sum'],
                'init_w' : .1 }
    A__SF_conn = {'rule' : 'one_to_one',
                  'delay' : 0.02 }
    A__SF_syn = {'type' : synapse_types.static,
                 'init_w' : [1.]*6 }
    # ACT to C ------------------------------------------------
    ACT__C_conn = {'rule' : "all_to_all",
                   'delay' : 0.02 } 
    ACT__C_syn = {'type' : synapse_types.static,
                  'inp_ports' : 3,
                  'init_w' : 1. }
    # ACT to M ------------------------------------------------
    ACT__M_conn = {'rule' : "all_to_all",
                   'delay' : 0.01 } 
    ACT__M_syn = {'type' : synapse_types.static,
                  'inp_ports' : 0,
                  'init_w' : 1. }
    # CE, CI to AL ---------------------------------------------
    CE__AL_conn = {'rule' : 'one_to_one',
                   'delay' : 0.01 }
    CE__AL_syn = {'type' : synapse_types.static,
                  'inp_ports' : 2,
                  'init_w' : .8 }
    CI__AL_conn = {'rule' : 'one_to_one',
                   'delay' : 0.01 }
    CI__AL_syn = {'type' : synapse_types.static,
                  'inp_ports' : 2,
                  'init_w' : -.8 }
    # CE,CI to CE,CI within the same population ----------------
    CE__CI_conn = {'rule' : 'one_to_one',
                   'delay' : 0.01 }
    CI__CE_conn = {'rule' : 'one_to_one',
                   'delay' : 0.01 }
    CE__CI_syn = {'type' : synapse_types.static,
                  'inp_ports' : 1,
                  'init_w' : cfg['CE__CI_w'] }
    CI__CE_syn = {'type' : synapse_types.static, #static, #corr_inh,
                  'inp_ports' : 1,
                  'lrate' : .0,
                  'des_act' : 0.5,
                  'init_w' : cfg['CI__CE_w'] }
    # synergist and antagonist connections in C ----------------
    C__C_conn = {'rule': 'one_to_one',
                 'allow_autapses' : False,
                 'delay' : 0.01 }
    C__C_syn_antag = {'type' : synapse_types.static,
                      'inp_ports': 1, # "lateral" port of rga_21 synapses
                      'init_w' : cfg['C__C_antag'] }
    C__C_syn_p_antag = {'type' : synapse_types.static,
                        'inp_ports': 1,
                        'init_w' : cfg['C__C_p_antag'] }
    C__C_syn_syne = {'type' : synapse_types.static,
                      'inp_ports': 1,
                      'init_w' : cfg['C__C_syne'] }
    C__C_syn_p_syne = {'type' : synapse_types.static,
                      'inp_ports': 1,
                      'init_w' : cfg['C__C_p_syne']}
    C__C_syn_null_lat = {'type' : synapse_types.static, # connection with static weight zero
                       'inp_ports': 1,
                       'init_w' : 0. }
    # spinal units to plant ------------------------------------
    AL__P_conn = {'inp_ports' : list(range(6)), #list([0, 1, 2, 3, 5, 4]),
                 'delays': 0.02 }
    AL__P_syn = {'type': synapse_types.static,
                'init_w' : 1. }
    # direct motor cortex to motoneurons ------------------------
    M__AL_conn = {'rule': 'all_to_all',
                  'delay': 0.02 }
    if not M__C_rand_w:
        if old_M__C:
            M_AL = np.array([[1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                             [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                             [0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                             [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
                             [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
                             [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.]])
        else:
            M_AL = np.array([[1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                             [0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                             [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
                             [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.],
                             [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]])
        M_AL_iw = (cfg['M__AL_w_sum']*0.5) * M_AL.flatten('F')
    else:
        M_AL_iw = {'distribution':'uniform', 'low':0.05, 'high':.1}
    M__AL_syn = {'type' : synapse_types.rga_21,
                 'lrate': cfg['M__AL_lrate'],
                 'inp_ports': 0,
                 'err_port' : 0,
                 'lat_port' : 1,
                 'w_sum' : cfg['M__AL_w_sum'],
                 'init_w' : M_AL_iw }
    # motor to spinal ------------------------------------------
    M__C_conn = {'rule': 'all_to_all',
                 'delay': 0.02 }
    if not M__C_rand_w:
        # initializing M__C weights manually
        if old_M__C:
            M_CE = np.array(
                [[ 1., .3, 0.,-.5, .3, 0.],
                 [ .3, 1.,-.5, 0., 0., 0.],
                 [ 0.,-.5, 1., .3, 0., 0.],
                 [-.5, 0., .3, 1., 0., .3],
                 [ .3, 0., 0., 0., 1.,-.5],
                 [ 0., 0., 0., .3,-.5, 1.]])
            M_CE = M_CE.flatten('C')
            M_CE = np.concatenate((M_CE, -M_CE))
            M_CI = -M_CE
        else:
            M_CE = np.array(
                [[1., 0., .5, 0., 0., .5, 0., 1., 0., .5, .5, 0.],
                 [.5, 0., 1., 0., 0., 0., 0., .5, 0., 1., 0., 0.],
                 [0., .5, 0., 1., 0., 0., .5, 0., 1., 0., 0., 0.],
                 [0., 1., 0., .5, .5, 0., 1., 0., .5, 0., 0., 5.],
                 [0., .5, 0., 0., 1., 0., .5, 0., 0., 0., 0., 1.],
                 [.5, 0., 0., 0., 0., 1., 0., .5, 0., 0., 1., 0.]])
            M_CI = np.array(
                [[0., 1., 0., .5, .5, 0., 1., 0., .5, 0., 0., .5],
                 [0., .5, 0., 1., 0., 0., .5, 0., 1., 0., 0., 0.],
                 [.5, 0., 1., 0., 0., 0., 0., .5, 0., 1., 0., 0.],
                 [1., 0., .5, 0., 0., .5, 0., 1., 0., .5, .5, 0.] ,
                 [.5, 0., 0., 0., 0., 1., 0., .5, 0., 0., 1., 0.],
                 [0., .5, 0., 0., 1., 0., .5, 0., 0., 0., 0., 1.]])
            for row_idx in range(M_CE.shape[0]):
                M_CE[row_idx] = M_CE[row_idx] / sum(M_CE[row_idx])
                M_CI[row_idx] = M_CI[row_idx] / sum(M_CI[row_idx])
            M_CE = M_CE.flatten('F')
            M_CI = M_CI.flatten('F')
        M__CE_iw = cfg['M__C_w_sum'] * M_CE
        M__CI_iw = cfg['M__C_w_sum'] * M_CI
    else:
        M__CE_iw = {'distribution':'uniform', 'low':0.05, 'high':.1}
        M__CI_iw = {'distribution':'uniform', 'low':0.05, 'high':.1}
    M__CE_syn = {'type' : synapse_types.rga_21,
                 'lrate': cfg['M__C_lrate'],
                 'inp_ports': 0,
                 'w_sum' : cfg['M__C_w_sum'],
                 'init_w' : M__CE_iw}
    M__CI_syn = {'type' : synapse_types.rga_21,
                 'lrate': cfg['M__C_lrate'],
                 'inp_ports': 0,
                 'w_sum' : cfg['M__C_w_sum'],
                 'init_w' : M__CI_iw}
    # motor error lateral connections -----------------------------
    if M__M_conns:
        M__M_rule = 'all_to_all'
        M_autap = True
        if old_M__C: # M units are not rotated
            M__M_iw = cfg['M__M_w'] * np.concatenate(
                                 (np.concatenate((np.zeros((12,12)), -np.eye(12)), axis=1),
                                  np.concatenate((-np.eye(12), np.zeros((12,12))), axis=1)),
                                 axis=0)
            M__M_iw = cfg['M__M_w'] * M__M_iw.flatten('F')
        else:
            twist = np.array([[0., 1., 0., 0., 0., 0.],
                              [1., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 1., 0., 0.],
                              [0., 0., 1., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 1.],
                              [0., 0., 0., 0., 1., 0.]])
            M__M_iw = np.concatenate((
                      np.concatenate((twist, np.eye(6)), axis=1),
                      np.concatenate((np.eye(6), twist), axis=1)), axis=0)
            M__M_iw = cfg['M__M_w'] * M__M_iw.flatten('F')
    else:
        M__M_rule = 'one_to_one' # no autapses means no connections will be added
        M_autap = False
        M__M_iw = 0.
    
    M__M_conn = {'rule': M__M_rule,
                 'allow_autapses' : M_autap,
                 'delay' : 0.02 } # the delay assumes an intermediate interneuron
    M__M_syn = {'type' : synapse_types.static,
                'inp_ports': 3, # default for m_sig targets
                'init_w' : M__M_iw }
    # plant to afferent -------------------------------------------
    idx_aff = np.arange(22,40) # indexes for afferent output in the arm
    P__A_conn = {'port_map' : [[(p,0)] for p in idx_aff],
                 'delays' : 0.02 }
    P__A_syn = {'type' : synapse_types.static,
                'init_w' : [2.]*6 + [2.]*6 + [4.]*6 } # weights for [Ib, Ia, II]
    # SF/SP to SPF ------------------------------------------------
    SFe__SPF_conn = {'rule' : "one_to_one",
                     'delay' : 0.01 }
    SFi__SPF_conn = {'rule' : "one_to_one",
                     'delay' : 0.01 }
    SFe__SPF_syn = {'type' : synapse_types.static,
                    'init_w' : 1. }
    SFi__SPF_syn = {'type' : synapse_types.static,
                    'init_w' : -1. }
    SPe__SPF_conn = {'rule' : "one_to_one",
                     'delay' : 0.01 }
    SPi__SPF_conn = {'rule' : "one_to_one",
                     'delay' : 0.01 }
    SPe__SPF_syn = {'type' : synapse_types.static,
                    'init_w' : 1. }
    SPi__SPF_syn = {'type' : synapse_types.static,
                   'init_w' : -1. }
    # SP to SP_CHG ------------------------------------------------
    SP__SP_CHG_conn = {'rule' : 'all_to_all',
                        'delay' : 0.01}
    SP__SP_CHG_syn = {'type' : synapse_types.chg,
                      'init_w' : 0.,
                      'lrate' : 20. }
    # SP_CHG to ACT ------------------------------------------------
    SP_CHG__ACT_conn = {'rule' : "all_to_all",
                       'delay' : 0.02 }
    SP_CHG__ACT_syn = {'type' : synapse_types.static,
                      'inp_ports' : 1,
                      'init_w' : 1. }
    # SPF to ACT ------------------------------------------------
    SPF__ACT_conn = {'rule' : "all_to_all",
                     'delay' : 0.02 }
    SPF__ACT_syn = {'type' : synapse_types.static,
                    'inp_ports' : 0,
                    'init_w' : 1. }
    # SPF to M --------------------------------------------------
    SPF__M_iw = 1. # default initial weight
    SPF__M_rule = 'one_to_one' # default connection rule
    SPF__M_syn_type = synapse_types.static # default synapse type
    if rot_SPF:
        SPF__M_conn = {'rule': 'all_to_all',
                       'delay': 0.02 }
        SPF__M_mat= np.array([[ 1,  1,  1,  1,  1,  1],
                              [ 1,  1,  1, -1, -1, -1],
                              [ 1, -2,  1, -1,  2, -1],
                              [-1, -1,  2,  2, -1, -1],
                              [-1,  1,  0,  0,  1, -1],
                              [-1,  0,  1, -1,  0,  1]], dtype=float).transpose()
        for idx, norm in enumerate(np.sqrt([6., 6., 12., 12., 4., 4.])):
            SPF__M_mat[:,idx] = (1./norm) * SPF__M_mat[:,idx]
        SPF__M_mat = np.concatenate((SPF__M_mat.flatten('C'),
                                     -SPF__M_mat.flatten('C'),
                                     -SPF__M_mat.flatten('C'),
                                     SPF__M_mat.flatten('C')))
        SPF__M_iw = SPF__M_mat
        SPF__M_rule = 'all_to_all'
    if rga_on_M:
        SPF__M_rule = 'all_to_all'
        SPF__M_syn_type = synapse_types.rga_21
        SPF__M_iw = {'distribution':'uniform', 'low':0.05, 'high':.5}
    if not (M__C_rand_w or old_M__C): # permuted entries in M
        SPF__M_rule = 'all_to_all'
        SPF__M_mat = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]])
        SPF__M_iw = SPF__M_mat.flatten('F')
    SPF__M_conn = {'rule': SPF__M_rule,
                   'delay': 0.02 }
    SPF__M_syn = {'type': SPF__M_syn_type,
                  'lrate': 0. if not rga_on_M else cfg['SPF__M_lrate'],
                  'w_sum': 1. if not 'SPF__M_w_sum' in cfg else cfg['SPF__M_w_sum'],
                  'err_port': 1,
                  'lat_port': 3,
                  'inp_ports': 1,
                  'init_w': SPF__M_iw }
    
    # sensory error lateral connections --------------------------
    SPF__SPF_conn = {'rule': 'one_to_one',
                     'allow_autapses' : False,
                     'delay' : 0.02 } # the delay assumes an intermediate interneuron
    SPF__SPF_syn = {'type' : synapse_types.static,
                    'inp_ports': 0,
                    'init_w' : cfg['SPF__SPF_w'], }
    # utility function to set C_params['custom_inp_del']
    def approx_del(f):
        """ Returns an estimate fo the optimal delay for rga learning.

            We assume that the important loop for the learning rule in the C units
            is the one going through C-P-A-M-C.

            Args:
                f : main oscillation frequency of in C, in Hertz
            Returns:
                2-tuple : (time_del, del_steps)
                time_del : A float with the time delay.
                del_steps : time delay as integer number of min_del steps.
        """
        w = 2.*np.pi*f
        al_del = np.arctan(np.mean(AL_params['tau'])*w)/w
        p_del = np.arctan(np.mean(P_params['tau'])*w)/w
        a_del = np.arctan(np.mean(A_params['tau'])*w)/w
        m_del = np.arctan(np.mean(M_params['tau'])*w)/w
        D = [CE__AL_conn['delay'], AL__P_conn['delays'],
             np.mean(P__A_conn['delays']), A__M_conn['delay'], M__C_conn['delay']]
        time_del = al_del + p_del + a_del + m_del + sum(D)
        del_steps = int(np.ceil(time_del/net_params['min_delay']))
        time_del = del_steps*net_params['min_delay']
        del_steps -= 1 # because this is an index, and indexes start at 0
        return time_del, del_steps

    if set_C_delay is True:
        C_time_del, C_del_steps = approx_del(1.)
        CE_params['custom_inp_del'] = C_del_steps
        CI_params['custom_inp_del'] = C_del_steps

    AL_params['delay'] = (AL_params['custom_inp_del']+2)*net_params['min_delay']
    CE_params['delay'] = (CE_params['custom_inp_del']+2)*net_params['min_delay']
    CI_params['delay'] = (CI_params['custom_inp_del']+2)*net_params['min_delay']
    M_params['delay'] = max(CE_params['delay'], CI_params['delay'],
                       (M_params['custom_inp_del']+2) * net_params['min_delay'])
    SPF_params['delay'] = (M_params['custom_inp_del']+2) * net_params['min_delay']
    
    #--------------------------------------------------------------------
    # CREATING NETWORK AND UNITS
    #--------------------------------------------------------------------
    net = network(net_params)

    A = net.create(18, A_params)
    AL = net.create(6, AL_params)
    ACT = net.create(1, ACT_params)
    CE = net.create(6, CE_params)
    CI = net.create(6, CI_params)
    M = net.create(12, M_params)
    P = net.create(1, P_params)
    SF = net.create(6, SF_params)
    SP = net.create(6, SP_params)
    SP_CHG = net.create(1, SP_CHG_params)
    SPF = net.create(12, SPF_params)

    # tracking units
    if track_weights:
        A_CE0_track = net.create(18, track_params) # to track weights from A to C0
        A_M0_track = net.create(12, track_params) # to track weights from A to M0
        M_CE0_track = net.create(M_size, track_params) # to track weights from M to C[0]
        M_AL0_track = net.create(M_size, track_params) # to track weights from M to AL[0]
        SPF_M0_track = net.create(len(SPF), track_params) # weights from SPF to M[0]
    else:
        A_CE0_track = []
        A_M0_track = []
        M_CE0_track = []
        M_AL0_track = []
        SPF_M0_track = []
    if track_ips:
        ipx_track = net.create(12, track_params) # x coordinates of insertion points
        ipy_track = net.create(12, track_params) # y coordinates of insertion points
    else:
        ipx_track = []
        ipy_track = []

    #--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--
    # SET THE PATTERNS IN SP -----------------------------------------------------
    # list with hand coordinates [x,y] (meters)
    n_coords = 1000 # number of coordinates to generate
    if rand_targets is False:
        center = np.array([0.3, 0.3]) # initial hand location
        n_trgs = 8 # number of targets around center
        r = .1  # distance of targets from center
        angs = np.linspace(0., 2.*np.pi, n_trgs+1)[:-1]
        circle = np.array([np.array([np.cos(ang),np.sin(ang)]) for ang in angs])
        targets = center + r*circle # coordinates of the targets
        hand_coords = [center, targets[0],
                       center, targets[1],
                       center, targets[2],
                       center, targets[3],
                       center, targets[4],
                       center, targets[5],
                       center, targets[6],
                       center, targets[7]]
        n_rep = int(round(n_coords / len(hand_coords)))
        hand_coords = n_rep * hand_coords
    else:
        # creating a list of random coordinates to use as targets
        min_s_ang = -0.1 # minimum shoulder angle
        max_s_ang = 0.8  # maximum shoulder angle
        min_e_ang = 0.2 # minimum elbow angle
        max_e_ang = 2.3 # maximum elbow angle
        n_coords = 1000 # number of coordinates to generate
        l_arm = net.plants[P].l_arm # upper arm length
        l_farm = net.plants[P].l_farm # forearm length
        hand_coords = [[0.,0.] for _ in range(n_coords)]
        s_angs = (np.random.random(n_coords)+min_s_ang)*(max_s_ang-min_s_ang)
        e_angs = (np.random.random(n_coords)+min_e_ang)*(max_e_ang-min_e_ang)
        for i in range(n_coords):
            hand_coords[i][0] = l_arm*np.cos(s_angs[i]) + l_farm*np.cos(s_angs[i]+e_angs[i]) # x-coordinate
            hand_coords[i][1] = l_arm*np.sin(s_angs[i]) + l_farm*np.sin(s_angs[i]+e_angs[i]) # y-coordinate

    # list with muscle lengths corresponding to the hand coordinates
    m_lengths = []
    for coord in hand_coords:
        m_lengths.append(net.plants[P].coords_to_lengths(coord))
    m_lengths = np.array(m_lengths)
    #(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)
    # We need to translate these lengths to corresponding SF activity levels.
    # For that it is necessary to recreate all their transformations
    # The first transformation is from length to II afferent activity.
    ### OUT OF THE 36 AFFERENT SIGNALS, WE TAKE II ###
    par = net.plants[P].m_params
    # steady state tensions in the static and dynamic bag fibers (no gamma inputs)
    Ts_ss = (par['k_se_s']/(par['k_se_s']+par['k_pe_s'])) * (
             par['k_pe_s']*(m_lengths - par['l0_s']))
    Td_ss = (par['k_se_d']/(par['k_se_d']+par['k_pe_d'])) * (
             par['k_pe_d']*(m_lengths - par['l0_d']))
    # steady state afferent outputs (no gamma inputs)
    #Ia_ss = par['fs']*(Ts_ss/par['k_se_s']) + (1.-par['fs'])*(Td_ss/par['k_se_d'])
    II_ss = par['se_II']*(Ts_ss/par['k_se_s']) + ((1.-par['se_II'])/par['k_pe_s'])*Ts_ss
    #Ia_ss *= par['Ia_gain']
    II_ss *= par['II_gain']
    #Ia_II_ss = np.concatenate((Ia_ss, II_ss), axis=1)
    # Next transformation is through the afferent units
    P__A_ws = np.array(P__A_syn['init_w'][12:18])
    #Ia_II_avgs = np.mean(Ia_II_ss, axis=0)  # when using hundreds of random targets
    # target averages
    A_thr = np.array([net.units[u].thresh for u in A[12:18]])
    A_II = np.log(1. + np.maximum((II_ss)*P__A_ws - A_thr, 0.))
    #(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)
    # Next is from A to SF
    SF_arg = A__SF_syn['init_w']*A_II
    SF_out = 1./ (1. + np.exp(-SF_params['slope']*(SF_arg - SF_params['thresh'])))
    SF_params['init_val'] = SF_out # this might cause a smooth start
    # now we set the values in SP
    if rand_targets:
        m_idxs = np.random.randint(len(hand_coords), size=1000) # index of all targets
    else:
        m_idxs = np.arange(len(hand_coords))
    A_us = [net.units[u] for u in A]

    def SF_sigmo(idx, arg):
        """ The sigmoidal function for SF unit with index SF[idx]. """
        return 1./ (1. + np.exp(-SF_params['slope'][idx]*(arg - SF_params['thresh'][idx])))

    def cur_target(t):
        """ Returns the index of the target at time t. """
        return m_idxs[int(np.floor(t/t_pres))]

    def make_fun(idx):
        """ create a function for the SP unit with index 'idx'. """
        return lambda t: SF_sigmo(idx, 
                            A__SF_syn['init_w'][idx] * (
                            np.log(1. + max(II_ss[cur_target(t)][idx] * P__A_ws[idx] - 
                            net.units[A[12+idx]].thresh, 0.))))
        #return lambda t: SF_out[m_idxs[int(np.floor(t/t_pres))]][idx]

    for idx, u in enumerate(SP):
        net.units[u].set_function(make_fun(idx))
    #--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--

    #--------------------------------------------------------------------
    # CONNECTING
    #--------------------------------------------------------------------
    # From afferent units
    net.connect(A, CE, A__C_conn, A__C_syn)
    net.connect(A, CI, A__C_conn, A__C_syn)
    net.connect(A[:12], M, A__M_conn, A__M_syn)
    net.connect(A[12:18], SF, A__SF_conn, A__SF_syn)
    # From ACT
    net.connect(ACT, CE, ACT__C_conn, ACT__C_syn)
    net.connect(ACT, CI, ACT__C_conn, ACT__C_syn)
    if rdc_on_M:
        net.connect(ACT, M, ACT__M_conn, ACT__M_syn)
    # from AL to P
    net.set_plant_inputs(AL, P, AL__P_conn, AL__P_syn)
    # from CE to AL
    net.connect(CE, AL, CE__AL_conn, CE__AL_syn)
    # from CI to AL
    net.connect(CI, AL, CI__AL_conn, CI__AL_syn)
    # intraspinal connections
    for pair in all_pairs:
        net.connect([AL[pair[0]]], [AL[pair[1]]], C__C_conn, C__C_syn_null_lat)
        if pair in synergists:
            net.connect([CE[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_syne)
            #net.connect([CE[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_lat)
            #net.connect([CI[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_lat)
            #net.connect([CI[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_lat)
        elif pair in part_syne:
            net.connect([CE[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_p_syne)
            #net.connect([CE[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_lat)
            #net.connect([CI[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_lat)
            #net.connect([CI[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_lat)
        elif pair in antagonists:
            net.connect([CE[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_antag)
            #net.connect([CE[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_lat)
            #net.connect([CI[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_lat)
            #net.connect([CI[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_lat)
        elif pair in part_antag:
            net.connect([CE[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_p_antag)
            #net.connect([CE[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_lat)
            #net.connect([CI[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_lat)
            #net.connect([CI[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_lat)
        elif pair in self_conn:
            net.connect([CE[pair[0]]], [CI[pair[1]]], CE__CI_conn, CE__CI_syn)
            net.connect([CI[pair[0]]], [CE[pair[1]]], CI__CE_conn, CI__CE_syn)
        else:
            net.connect([CE[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_lat)
            net.connect([CE[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_lat)
            net.connect([CI[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_lat)
            net.connect([CI[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_lat)
    # From M 
    net.connect(M, AL, M__AL_conn, M__AL_syn)
    net.connect(M, CE, M__C_conn, M__CE_syn)
    net.connect(M, CI, M__C_conn, M__CI_syn)
    net.connect(M, M, M__M_conn, M__M_syn)
    # From plant to afferents
    net.set_plant_outputs(P, A, P__A_conn, P__A_syn) 
    # From SF, SP to SPF
    net.connect(SF, SPF[:6], SFe__SPF_conn, SFe__SPF_syn)
    net.connect(SF, SPF[6:12], SFi__SPF_conn, SFi__SPF_syn)
    net.connect(SP, SPF[:6], SPi__SPF_conn, SPi__SPF_syn)
    net.connect(SP, SPF[6:12], SPe__SPF_conn, SPe__SPF_syn)
    # from SP to SP_CHG
    net.connect(SP, SP_CHG, SP__SP_CHG_conn, SP__SP_CHG_syn)
    # from SP_CHG to ACT
    net.connect(SP_CHG, ACT, SP_CHG__ACT_conn, SP_CHG__ACT_syn)
    # From SPF
    net.connect(SPF, ACT, SPF__ACT_conn, SPF__ACT_syn)
    net.connect(SPF, M, SPF__M_conn, SPF__M_syn)
    net.connect(SPF, SPF, SPF__SPF_conn, SPF__SPF_syn)

    # SETTING UP WEIGHT TRACKING
    if track_weights:
        def M_CE0_fun(idx):
            """ Creates a function to track a weight from M to CE0. """
            return lambda t: net.syns[CE[0]][idx].w
        base = 0
        for idx in range(len(M)):
            rga_syn = False
            while not rga_syn:
                if net.syns[CE[0]][base+idx].type in [synapse_types.rga_21]:
                    rga_syn = True
                else:
                    base += 1
                    if base > 100:
                        raise AssertionError('Could not create M_C tracker unit')
            net.units[M_CE0_track[idx]].set_function(M_CE0_fun(base+idx))
            
        def M_AL0_fun(idx):
            """ Creates a function to track a weight from M to AL0. """
            return lambda t: net.syns[AL[0]][idx].w
        base = 0
        for idx in range(len(M)):
            rga_syn = False
            while not rga_syn:
                if net.syns[AL[0]][base+idx].type in [synapse_types.rga_21]:
                    rga_syn = True
                else:
                    base += 1
                    if base > 100:
                        raise AssertionError('Could not create M_AL tracker unit')
            net.units[M_AL0_track[idx]].set_function(M_AL0_fun(base+idx))
    
        def A_M0_fun(idx):
            """ Creates a function to track a weight from AF to M0. """
            return lambda t: net.syns[M[0]][idx].w
        base = 0
        for idx, uid in enumerate(A_M0_track):
            rga_syn = False
            while not rga_syn:
                if net.syns[M[0]][base+idx].type in [synapse_types.inp_sel]:
                    rga_syn = True
                else:
                    base += 1
                    if base > 100:
                        raise AssertionError('Could not create A_M tracker unit')
            net.units[uid].set_function(A_M0_fun(base+idx))
            
        def A_CE0_fun(idx):
            """ Creates a function to track a weight from AF to CE0. """
            return lambda t: net.syns[CE[0]][idx].w
        base = 0
        for idx, uid in enumerate(A_CE0_track):
            inpsel_syn = False
            while not inpsel_syn:
                if net.syns[CE[0]][base+idx].type in [synapse_types.inp_sel]:
                    inpsel_syn = True
                else:
                    base += 1
                    if base > 100:
                        raise AssertionError('Could not create A_C tracker unit')
            net.units[A_CE0_track[idx]].set_function(A_CE0_fun(base+idx))
            
        def SPF_M0_fun(idx):
            """ Creates a function to track a weight from SPF to M0. """
            return lambda t: net.syns[M[0]][idx].w
        if rga_on_M:
            base = 0
            for idx in range(len(SPF)):
                rga_syn = False
                while not rga_syn:
                    if net.syns[M[0]][base+idx].type in [synapse_types.rga_21]:
                        rga_syn = True
                    else:
                        base += 1
                        if base > 100:
                            raise AssertionError('Could not create SPF_M tracker unit')
                net.units[SPF_M0_track[idx]].set_function(SPF_M0_fun(base+idx))
        
    # TRACKING OF INSERTION POINTS (for the arm animation)
    if track_ips:
        # make the source units track the tensions
        def create_xtracker(arm_id, idx):
            return lambda t: net.plants[arm_id].ip[idx][0]
        def create_ytracker(arm_id, idx):
            return lambda t: net.plants[arm_id].ip[idx][1]
        for idx, uid in enumerate(ipx_track):
            net.units[uid].set_function(create_xtracker(P, idx))
        for idx, uid in enumerate(ipy_track):
            net.units[uid].set_function(create_ytracker(P, idx))

    pops_list = [A, ACT, AL, CE, CI, M, P, SF, SP, SP_CHG,
                 SPF, A_CE0_track, A_M0_track, M_AL0_track,
                 M_CE0_track, SPF_M0_track, ipx_track, ipy_track]
    pops_names = ['A', 'ACT', 'AL', 'CE', 'CI', 'M', 'P', 'SF', 'SP', 'SP_CHG',
                  'SPF', 'A_CE0_track', 'A_M0_track', 'M_AL0_track',
                  'M_CE0_track', 'SPF_M0_track','ipx_track', 'ipy_track']
    
    pops_dict = {pops_names[idx] : pops_list[idx] for 
                 idx in range(len(pops_names))}
    pds = {'P__A_syn' : P__A_syn, 'SF_params' : SF_params, 
           'A__SF_syn' : A__SF_syn }
    
    return net, pops_dict, hand_coords, m_idxs, pds


def syne_net(cfg,
             t_pres = 40.,
             par_heter = 0.01,
             set_C_delay = False,
             rand_targets = True,
             track_weights = False,
             track_ips = False,
             C_noise = False,
             M__C_rand_w = True,
             rot_SPF = False):
    """ Create a draculab network with the given configuration. 

        Args:
            cfg : a parameter dictionary
            
            Optional keyword arguments:
            t_pres: number of seconds to hold each set of target lengths
            par_heter: range of heterogeneity as a fraction of the original value
            set_C_delay: whether set C_cid using analytical approach
            rand_targets: whether to train using a large number of random targets
            C_noise: whether SYNE units are noisy (use euler_maru integrator)
            M__C_rand_w: randomly initialize weights for the M__C connections
            rot_SPF: whether to rotate the output of SPF

        Returns:
            A tuple with the following entries:
            net : A draculab network as in v3_nst_afx, with the given configuration.
            pops_dict : a dictionary with the list of ID's for each population in net.
            hand_coords : list with the coordinates for all possible targets
            m_idxs : which target coordinats will be used for the i-th presentation
            pds : some parameter dictionaries used to change targets
    """
    M_size = 12 # number of units in the M population
    SPF_size = 12 # number of units in the SPF population
    N_syne = 12 # number of excitatory and inhibitory "synergy units"

    if not C_noise and not 'C_sigma' in cfg:
        cfg['C_sigma'] = 0.
    if not 'C_tau' in cfg:
        cfg['C_tau'] = 0.05
    if not 'C_tau_slow' in cfg:
        cfg['C_tau_slow'] = 10.
    if not 'k_pe_e' in cfg:
        cfg['k_pe_e'] = 20.
    if not 'k_se_e' in cfg:
        cfg['k_se_e'] = 20.

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameter dictionaries for the network and the plant
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    net_params = {'min_delay' : 0.005,
                  'min_buff_size' : 8 }
    # plant parameters
    P_params = {'type' : plant_models.bouncy_planar_arm_v3,
              'mass1': 1.,
              'mass2': 1.,
              's_min' : -0.8,
              'p1' : (-0.01, 0.04),
              'p2' : (0.29, 0.03),
              'p3' : (0., 0.05),
              'p4' : (0.1, 0.04),
              'p5' : (0.01, -0.05),
              'p6' : (0.11, -0.04),
              'p10': (0.29, 0.03),
              'init_q1': 0.,
              'init_q2': np.pi/2.,
              'init_q1p': 0.,
              'init_q2p': 0.,
              'g': 0.0,
              'mu1': 3.,
              'mu2': 3.,
              'l_torque' : 0.01,
              'l_visco' : 0.01,
              'g_e' : cfg['g_e_factor']*np.array([cfg['g_e_03'], 20., 20.,
                                                  cfg['g_e_03'], 22., 23.]),
              'l0_e' : [1.]*6,
              'Ia_gain' : 2.5*np.array([3.,10.,10., 3.,10.,10.]),
              'II_gain' : 2.*np.array([cfg['II_g_03'], 8., 8.,
                                       cfg['II_g_03'], 8., 8.]),
              'Ib_gain' : 1.,
              'T_0' : 10.,
              'k_pe_e' : cfg['k_pe_e'], #8
              'k_se_e' : cfg['k_se_e'], #13
              'b_e' : cfg['b_e'],
              'g_s' : 0.02,
              'k_pe_s' : 2., 
              'k_se_s' : 2.,
              'g_d' : 0.01,
              'k_pe_d' : .2, #.1,
              'k_se_d' : 1., #2.,
              'b_s' : .5,
              'b_d' : 2.,#3.,
              'l0_s': .7,
              'l0_d': .8,
              'fs' : 0.1,
              'se_II' : 0.5,
              'cd' : 0.5,
              'cs' : 0.5,
              'tau' : 0.1   # ficticious time constant used in create_freqs_steps
               }
    #--------------------------------------------------------------------
    # Unit parameters
    #--------------------------------------------------------------------
    randz = lambda n: (1. + par_heter*(np.random.rand(n)-0.5))

    A_params = {'type' : unit_types.logarithmic,
                'init_val' : 0.,
                'tau' : 0.01, # 0.02
                'tau_fast': 0.005,
                'thresh' : np.array([.2]*6 + [0.]*6 + [.2]*6) } # [Ib, Ia, II]
    ACT_params = {'type' : unit_types.act,
                  'tau_u' : 10., #6., #8
                  'gamma' : 8., #6., #2
                  'g' : 2.,
                  'theta' : 1., #1.,
                  'tau_slow' : 5.,
                  'y_min' : 0.1, #0.2
                  'rst_thr' : 0.1,
                  'init_val' : 0. }
    spf_sum_min = .6 # .6 # value where no corrections are needed anymore
    y_min = 1./(1. + np.exp(-ACT_params['g']*(spf_sum_min - ACT_params['theta'])))
    ACT_params['y_min'] = y_min

    AL_params = {'type' : unit_types.sigmoidal,
                 'thresh' : cfg['AL_thresh'] * randz(6),
                 'slope' : 2. * randz(6),
                 'init_val' : 0.1 * randz(6),
                 'tau' : 0.02 * randz(6) }
    CE_params = {'type' : unit_types.rga_adapt_sig,
                'integ_meth' : 'euler_maru' if C_noise else 'odeint',
                'init_val' : [r*np.array([0.5]) for r in np.random.random(6)],
                'multidim' : False,
                'slope' : cfg['C_slope'],
                'thresh' : cfg['C_thresh'],
                'tau' : cfg['C_tau'],
                'tau_fast': 0.01,
                'tau_mid' : 0.05,
                'tau_slow' : cfg['C_tau_slow'],
                'custom_inp_del' : int(round(cfg['C_cid']/net_params['min_delay'])),
                'delay' : 0.31, # to be set below
                'adapt_amp' : cfg['C_adapt_amp'],
                'mu' : 0.,
                'sigma' : cfg['C_sigma'] }
    CI_params = {'type' : unit_types.rga_adapt_sig,
                'integ_meth' : 'euler_maru' if C_noise else 'odeint',
                'init_val' : [r*np.array([0.5]) for r in np.random.random(6)],
                'multidim' : False,
                'slope' : cfg['CI_slope'] if 'CI_slope' in cfg else cfg['C_slope'],
                'thresh' : cfg['CI_thresh'] if 'CI_thresh' in cfg else cfg['C_thresh'],
                'tau' :  cfg['CI_tau'] if 'CI_tau' in cfg else cfg['C_tau'],
                'tau_fast': 0.01,
                'tau_mid' : 0.05,
                'tau_slow' : cfg['C_tau_slow'],
                'custom_inp_del' : int(round(cfg['C_cid']/net_params['min_delay'])),
                'delay' : 0.31, # to be set below
                'adapt_amp' : cfg['C_adapt_amp'],
                'mu' : 0.,
                'sigma' : cfg['C_sigma'] }
    M_params = {'type' : unit_types.m_sig,
                'thresh' : 0.5 * randz(M_size),
                'slope' : 2.5 * randz(M_size),
                'init_val' : 0.2 * randz(M_size),
                'delay' : 0.35, # to be set below
                'n_ports' : 4,
                'tau_fast': 0.01,
                'tau_mid': 0.05,
                'tau_slow' : 8.,
                'tau' : cfg['M_tau'] * randz(M_size),
                'integ_amp' : 0.,
                'custom_inp_del' : int(np.round(cfg['M_cid']/net_params['min_delay'])) ,
                'des_out_w_abs_sum' : cfg['M_des_out_w_abs_sum'] }
    # SF, SP
    SFth = cfg['SF_thresh_03'] 
    SF_params = {'type' : unit_types.sigmoidal,
                 'thresh' : np.array([SFth, 0.4, 0.4, SFth, 0.4, 0.4]),
                 'slope' : np.array([4.]*6),
                 'init_val' : 0.2,
                 'tau' : 0.02 }  # 0.05
    SP_params = {'type' : unit_types.source,
                 'init_val' : 0.5,
                 'tau_fast' : 0.01,
                 'tau_mid' : 0.2,
                 'function' : lambda t: None }
    SP_CHG_params = {'type' : unit_types.sigmoidal,
                  'thresh' : 0.25,
                  'slope' : 9.,
                  'init_val' : 0.1,
                  'tau' : 0.01 }
    # 1-D error units
    SPF_params = {'type' : unit_types.sigmoidal,
                  'thresh' : 0.1 * randz(SPF_size),
                  'slope' : 9. * randz(SPF_size),
                  'init_val' : 0.3 * randz(SPF_size),
                  'tau_fast': 0.005,
                  'tau_mid': 0.05,
                  'tau_slow' : 5.,
                  'tau' : 0.02 * randz(SPF_size) }
    SYNE_params = {'type' : unit_types.rga_adapt_sig,
                'integ_meth' : 'euler_maru' if C_noise else 'odeint',
                'init_val' : [r*np.array([0.5]) for r in np.random.random(N_syne)],
                'multidim' : False,
                'slope' : cfg['C_slope'],
                'thresh' : cfg['C_thresh'],
                'tau' : cfg['C_tau'],
                'tau_fast': 0.01,
                'tau_mid' : 0.05,
                'tau_slow' : cfg['C_tau_slow'],
                'custom_inp_del' : int(round(cfg['C_cid']/net_params['min_delay'])),
                'delay' : 0.31, # to be set below
                'adapt_amp' : cfg['C_adapt_amp'],
                'mu' : 0.,
                'sigma' : cfg['C_sigma']}
    SYNI_params = {'type' : unit_types.rga_adapt_sig,
                'integ_meth' : 'euler_maru' if C_noise else 'odeint',
                'init_val' : [r*np.array([0.5]) for r in np.random.random(N_syne)],
                'multidim' : False,
                'slope' : cfg['CI_slope'] if 'CI_slope' in cfg else cfg['C_slope'],
                'thresh' : cfg['CI_thresh'] if 'CI_thresh' in cfg else cfg['C_thresh'],
                'tau' :  cfg['CI_tau'] if 'CI_tau' in cfg else cfg['C_tau'],
                'tau_fast': 0.01,
                'tau_mid' : 0.05,
                'tau_slow' : cfg['C_tau_slow'],
                'custom_inp_del' : int(round(cfg['C_cid']/net_params['min_delay'])),
                'delay' : 0.31, # to be set below
                'adapt_amp' : cfg['C_adapt_amp'],
                'mu' : 0.,
                'sigma' : cfg['C_sigma'] }
    # units to track synaptic weights or other values
    track_params = {'type' : unit_types.source,
                    'init_val' : 0.02,
                    'function' : lambda t: None }

    #--------------------------------------------------------------------
    # Connection dictionaries
    #--------------------------------------------------------------------
    # We organize the spinal connections through 4 types of symmetric relations
    # these lists are used to set intraspinal connections and test connection matrices
    antagonists = [(0,3), (1,2), (4,5)]
    part_antag = [(0,2),(0,5), (3,4), (1,3)]
    synergists = [(0,1), (0,4), (2,3), (3,5)]
    part_syne = [(1,4), (2,5)]
    self_conn = [(x,x) for x in range(6)]

    antagonists += [(p[1],p[0]) for p in antagonists]
    part_antag += [(p[1],p[0]) for p in part_antag]
    synergists += [(p[1],p[0]) for p in synergists]
    part_syne += [(p[1],p[0]) for p in part_syne]
    all_pairs = [(i,j) for i in range(6) for j in range(6)]

    # Organization of synergy units should be straightforward.
    # Unit 1 is the antagonist of 2, 3 of 4, ..., 11 of 12.
    # The following list provides the AL targets for each unit
    syn_targets = ([0,1], [2,3], [0,2], [1,3], [0,4], [3,5],
                   [0,5], [3,4], [1,4], [2,5], [1,5], [2,4])

    # Afferent to motor error selection
    A__M_conn = {'rule' : 'all_to_all',
                 'delay' : 0.02 }
    A__M_syn = {'type' : synapse_types.inp_sel, 
                'inp_ports' : 2, # the default for m_sig targets
                'error_port' : 1, # the default for m_sig targets
                'aff_port' : 2,
                'lrate' : cfg['A__M_lrate'], # negative rate for m_sig targets with value inputs
                'w_sum' : cfg['A__M_w_sum'],
                'w_max' : cfg['A__M_w_max_frac']*cfg['A__M_w_sum'],
                'init_w' : .1 }
    A__SF_conn = {'rule' : 'one_to_one',
                  'delay' : 0.02 }
    A__SF_syn = {'type' : synapse_types.static,
                 'init_w' : [1.]*6 }
    # ACT to C ------------------------------------------------
    ACT__C_conn = {'rule' : "all_to_all",
                   'delay' : 0.02 } 
    ACT__C_syn = {'type' : synapse_types.static,
                  'inp_ports' : 2,
                  'init_w' : 1. }
    # ACT to SYN ------------------------------------------------
    ACT__SYN_conn = {'rule' : "all_to_all",
                     'delay' : 0.02 } 
    ACT__SYN_syn = {'type' : synapse_types.static,
                    'inp_ports' : 2,
                    'init_w' : 1. }
    # CE, CI to AL ----------------------------------------------
    CE__AL_conn = {'rule' : 'one_to_one',
                   'delay' : 0.01 }
    CE__AL_syn = {'type' : synapse_types.static,
                  'init_w' : 1. }
    CI__AL_conn = {'rule' : 'one_to_one',
                   'delay' : 0.01 }
    CI__AL_syn = {'type' : synapse_types.static,
                  'init_w' : -1. }
    # CE,CI to CE,CI within the same population -----------------
    CE__CI_conn = {'rule' : 'one_to_one',
                   'delay' : 0.01 }
    CI__CE_conn = {'rule' : 'one_to_one',
                   'delay' : 0.01 }
    CE__CI_syn = {'type' : synapse_types.static,
                  'inp_ports' : 1,
                  'init_w' : cfg['CE__CI_w'] }
    CI__CE_syn = {'type' : synapse_types.static, #static, #corr_inh,
                  'inp_ports' : 1,
                  'lrate' : .0,
                  'des_act' : 0.5,
                  'init_w' : cfg['CI__CE_w'] }
    # synergist and antagonist connections in C
    C__C_conn = {'rule': 'one_to_one',
                 'allow_autapses' : False,
                 'delay' : 0.01 }
    C__C_syn_antag = {'type' : synapse_types.static,
                      'inp_ports': 1, # "lateral" port of rga_21 synapses
                      'init_w' : cfg['C__C_antag'] }
    C__C_syn_p_antag = {'type' : synapse_types.static,
                        'inp_ports': 1,
                        'init_w' : cfg['C__C_p_antag'] }
    C__C_syn_syne = {'type' : synapse_types.static,
                      'inp_ports': 1,
                      'init_w' : cfg['C__C_syne'] }
    C__C_syn_p_syne = {'type' : synapse_types.static,
                      'inp_ports': 1,
                      'init_w' : cfg['C__C_p_syne']}
    C__C_syn_null_lat = {'type' : synapse_types.static, # connection with static weight zero
                       'inp_ports': 1,
                       'init_w' : 0. }
    # spinal units to plant
    AL__P_conn = {'inp_ports' : list(range(6)),
                 'delays': 0.02 }
    AL__P_syn = {'type': synapse_types.static,
                'init_w' : 1. }
    # motor to spinal
    if not M__C_rand_w:
        # initializing M__C weights manually
        M_CE = np.array(
            [[ 1., .3, 0.,-.5, .3, 0.],
             [ .3, 1.,-.5, 0., 0., 0.],
             [ 0.,-.5, 1., .3, 0., 0.],
             [-.5, 0., .3, 1., 0., .3],
             [ .3, 0., 0., 0., 1.,-.5],
             [ 0., 0., 0., .3,-.5, 1.]])
        M_CE = M_CE.flatten('C')
        M_CE = np.concatenate((M_CE, -M_CE))
        M_CI = -M_CE
        M__CE_iw = M_CE
        M__CI_iw = M_CI
    else:
        M__CE_iw = {'distribution':'uniform', 'low':0.05, 'high':.1}
        M__CI_iw = {'distribution':'uniform', 'low':0.05, 'high':.1}

    M__C_conn = {'rule': 'all_to_all',
                 'delay': 0.02 }
    M__CE_syn = {'type' : synapse_types.rga_21,
                 'lrate': cfg['M__C_lrate'],
                 'inp_ports': 0,
                 'w_sum' : cfg['M__C_w_sum'],
                 'init_w' : M__CE_iw}
    M__CI_syn = {'type' : synapse_types.rga_21,
                 'lrate': cfg['M__C_lrate'],
                 'inp_ports': 0,
                 'w_sum' : cfg['M__C_w_sum'],
                 'init_w' : M__CI_iw}

    M__SYN_conn = {'rule': 'all_to_all',
                   'delay': 0.02 }
    M__SYNE_syn = {'type' : synapse_types.rga_21,
                   'lrate': cfg['M__C_lrate'],
                   'inp_ports': 0,
                   'w_sum' : cfg['M__C_w_sum'],
                   'init_w' : {'distribution':'uniform', 'low':0.05, 'high':.1}}
    M__SYNI_syn = {'type' : synapse_types.rga_21,
                   'lrate': cfg['M__C_lrate'],
                   'inp_ports': 0,
                   'w_sum' : cfg['M__C_w_sum'],
                   'init_w' : {'distribution':'uniform', 'low':0.05, 'high':.1}}
    # motor error lateral connections
    M__M_conn = {'rule': 'one_to_one',
                 'allow_autapses' : False,
                 'delay' : 0.02 } # the delay assumes an intermediate interneuron
    M__M_syn = {'type' : synapse_types.static,
                'inp_ports': 3, # default for m_sig targets
                'init_w' : cfg['M__M_w'] }
    # plant to afferent
    idx_aff = np.arange(22,40) # indexes for afferent output in the arm
    P__A_conn = {'port_map' : [[(p,0)] for p in idx_aff],
                 'delays' : 0.02 }
    P__A_syn = {'type' : synapse_types.static,
                'init_w' : [2.]*6 + [2.]*6 + [4.]*6 } # weights for [Ib, Ia, II]
    # SF/SP to SPF
    SFe__SPF_conn = {'rule' : "one_to_one",
                     'delay' : 0.01 }
    SFi__SPF_conn = {'rule' : "one_to_one",
                     'delay' : 0.01 }
    SFe__SPF_syn = {'type' : synapse_types.static,
                    'init_w' : 1. }
    SFi__SPF_syn = {'type' : synapse_types.static,
                    'init_w' : -1. }
    SPe__SPF_conn = {'rule' : "one_to_one",
                     'delay' : 0.01 }
    SPi__SPF_conn = {'rule' : "one_to_one",
                     'delay' : 0.01 }
    SPe__SPF_syn = {'type' : synapse_types.static,
                    'init_w' : 1. }
    SPi__SPF_syn = {'type' : synapse_types.static,
                   'init_w' : -1. }
    # SP to SP_CHG ------------------------------------------------
    SP__SP_CHG_conn = {'rule' : 'all_to_all',
                        'delay' : 0.01}
    SP__SP_CHG_syn = {'type' : synapse_types.chg,
                      'init_w' : 0.,
                      'lrate' : 20. }
    # SP_CHG to ACT ------------------------------------------------
    SP_CHG__ACT_conn = {'rule' : "all_to_all",
                       'delay' : 0.02 }
    SP_CHG__ACT_syn = {'type' : synapse_types.static,
                      'inp_ports' : 1,
                      'init_w' : 1. }
    # SPF to ACT ------------------------------------------------
    SPF__ACT_conn = {'rule' : "all_to_all",
                     'delay' : 0.02 }
    SPF__ACT_syn = {'type' : synapse_types.static,
                    'inp_ports' : 0,
                    'init_w' : 1. }
    # SPF to M
    if rot_SPF:
        SPF__M_conn = {'rule': 'all_to_all',
                       'delay': 0.02 }
        SPF__M_mat= np.array([[ 1,  1,  1,  1,  1,  1],
                              [ 1,  1,  1, -1, -1, -1],
                              [ 1, -2,  1, -1,  2, -1],
                              [-1, -1,  2,  2, -1, -1],
                              [-1,  1,  0,  0,  1, -1],
                              [-1,  0,  1, -1,  0,  1]], dtype=float).transpose()
        for idx, norm in enumerate(np.sqrt([6., 6., 12., 12., 4., 4.])):
            SPF__M_mat[:,idx] = (1./norm) * SPF__M_mat[:,idx]
        SPF__M_mat = np.concatenate((SPF__M_mat.flatten('C'), -SPF__M_mat.flatten('C'), -SPF__M_mat.flatten('C'), SPF__M_mat.flatten('C')))
#         SPF__M_mat = np.concatenate((SPF__M_mat, -SPF__M_mat), axis=0)
#         SPF__M_mat = np.concatenate((SPF__M_mat, -SPF__M_mat), axis=1)
#         SPF__M_mat = SPF__M_mat.flatten('C')
        SPF__M_syn = {'type' : synapse_types.static,
                      'inp_ports' : 1,
                      'init_w' : SPF__M_mat }
    else:
        SPF__M_conn = {'rule': 'one_to_one',
                       'delay': 0.02 }
        SPF__M_syn = {'type' : synapse_types.static,
                      'inp_ports' : 1,
                      'init_w' : 1. }
    # sensory error lateral connections
    SPF__SPF_conn = {'rule': 'one_to_one',
                     'allow_autapses' : False,
                     'delay' : 0.02 } # the delay assumes an intermediate interneuron
    SPF__SPF_syn = {'type' : synapse_types.static,
                    'inp_ports': 0,
                    'init_w' : cfg['SPF__SPF_w'] }
    # synergy units to alpha units
    SYNE__AL_conn = {'rule' : 'all_to_all',
                     'delay' : 0.01 }
    SYNE__AL_syn = {'type' : synapse_types.static,
                    'init_w' : 1. }
    SYNI__AL_conn = {'rule' : 'all_to_all',
                     'delay' : 0.01 }
    SYNI__AL_syn = {'type' : synapse_types.static,
                    'init_w' : -1. }
    # Connections between SYNE and SYNI
    SYNE__SYNI_conn = {'rule' : 'one_to_one',
                       'delay' : 0.01 }
    SYNI__SYNE_conn = {'rule' : 'one_to_one',
                       'delay' : 0.01 }
    SYNE__SYNI_syn = {'type' : synapse_types.static,
                      'inp_ports' : 1,
                      'init_w' : cfg['CE__CI_w'] }
    SYNI__SYNE_syn = {'type' : synapse_types.static, #static, #corr_inh,
                      'inp_ports' : 1,
                      'init_w' : cfg['CI__CE_w'] }
    SYN_antag_conn = {'rule': 'one_to_one',
                      'allow_autapses' : False,
                      'delay' : 0.01 }
    SYN_antag_syn = {'type' : synapse_types.static,
                      'inp_ports': 1, # "lateral" port of rga_21 synapses
                      'init_w' : cfg['C__C_antag'] }

    # utility function to set C_params['custom_inp_del']
    def approx_del(f):
        """ Returns an estimate fo the optimal delay for rga learning.

            We assume that the important loop for the learning rule in the C units
            is the one going through C-P-A-M-C.

            Args:
                f : main oscillation frequency of in C, in Hertz
            Returns:
                2-tuple : (time_del, del_steps)
                time_del : A float with the time delay.
                del_steps : time delay as integer number of min_del steps.
        """
        w = 2.*np.pi*f
        al_del = np.arctan(np.mean(AL_params['tau'])*w)/w
        p_del = np.arctan(np.mean(P_params['tau'])*w)/w
        a_del = np.arctan(np.mean(A_params['tau'])*w)/w
        m_del = np.arctan(np.mean(M_params['tau'])*w)/w
        D = [CE__AL_conn['delay'], AL__P_conn['delays'], np.mean(P__A_conn['delays']),
             A__M_conn['delay'], M__C_conn['delay'] ]
        time_del = al_del + p_del + a_del + m_del + sum(D)
        del_steps = int(np.ceil(time_del/net_params['min_delay']))
        time_del = del_steps*net_params['min_delay']
        del_steps -= 1 # because this is an index, and indexes start at 0
        return time_del, del_steps

    if set_C_delay is True:
        C_time_del, C_del_steps = approx_del(2.)
        CE_params['custom_inp_del'] = C_del_steps
        CI_params['custom_inp_del'] = C_del_steps
        SYNE_params['custom_inp_del'] = C_del_steps
        SYNI_params['custom_inp_del'] = C_del_steps

    CE_params['delay'] = (CE_params['custom_inp_del'] + 2) * net_params['min_delay']
    CI_params['delay'] = (CI_params['custom_inp_del'] + 2) * net_params['min_delay']
    SYNE_params['delay'] = (SYNE_params['custom_inp_del'] + 2) * net_params['min_delay']
    SYNI_params['delay'] = (SYNI_params['custom_inp_del'] + 2) * net_params['min_delay']
    M_params['delay'] = max(CE_params['delay'], CI_params['delay'],
                            SYNE_params['delay'], SYNI_params['delay'],
                        (M_params['custom_inp_del'] + 2) * net_params['min_delay'])

    #--------------------------------------------------------------------
    # CREATING NETWORK AND UNITS
    #--------------------------------------------------------------------
    net = network(net_params)

    A = net.create(18, A_params)
    AL = net.create(6, AL_params)
    ACT = net.create(1, ACT_params)
    # CE = net.create(6, CE_params)
    # CI = net.create(6, CI_params)
    M = net.create(12, M_params)
    P = net.create(1, P_params)
    SF = net.create(6, SF_params)
    SP = net.create(6, SP_params)
    SP_CHG = net.create(1, SP_CHG_params)
    SPF = net.create(12, SPF_params)
    SYNE = net.create(N_syne, SYNE_params)
    SYNI = net.create(N_syne, SYNI_params)

    # tracking units
    #M_CE0_track = net.create(M_size, track_params) # to track weights from M to C0
    M_SYNE0_track = net.create(M_size, track_params) # to track weights from M to SYNE0
    A_M0_track = net.create(12, track_params) # to track weights from A to M0
    ipx_track = net.create(12, track_params) # x coordinates of insertion points
    ipy_track = net.create(12, track_params) # y coordinates of insertion points

    #--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--
    # SET THE PATTERNS IN SP -----------------------------------------------------
    # list with hand coordinates [x,y] (meters)
    if rand_targets is False:
        hand_coords = [[0.3, 0.45], 
                       [0.35, 0.4],
                       [0.4, 0.35],
                       [0.35, 0.3],
                       [0.3, 0.25],
                       [0.25, 0.3],
                       [0.2, 0.35],
                       [0.25, 0.4]]
    else:
        # creating a list of random coordinates to use as targets
        min_s_ang = -0.1 # minimum shoulder angle
        max_s_ang = 0.8  # maximum shoulder angle
        min_e_ang = 0.2 # minimum elbow angle
        max_e_ang = 2.3 # maximum elbow angle
        n_coords = 1000 # number of coordinates to generate
        l_arm = net.plants[P].l_arm # upper arm length
        l_farm = net.plants[P].l_farm # forearm length
        hand_coords = [[0.,0.] for _ in range(n_coords)]
        s_angs = (np.random.random(n_coords)+min_s_ang)*(max_s_ang-min_s_ang)
        e_angs = (np.random.random(n_coords)+min_e_ang)*(max_e_ang-min_e_ang)
        for i in range(n_coords):
            hand_coords[i][0] = l_arm*np.cos(s_angs[i]) + l_farm*np.cos(s_angs[i]+e_angs[i]) # x-coordinate
            hand_coords[i][1] = l_arm*np.sin(s_angs[i]) + l_farm*np.sin(s_angs[i]+e_angs[i]) # y-coordinate

    # list with muscle lengths corresponding to the hand coordinates
    m_lengths = []
    for coord in hand_coords:
        m_lengths.append(net.plants[P].coords_to_lengths(coord))
    m_lengths = np.array(m_lengths)
    #(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)
    # We need to translate these lengths to corresponding SF activity levels.
    # For that it is necessary to recreate all their transformations
    # The first transformation is from length to II afferent activity.
    ### OUT OF THE 18 AFFERENT SIGNALS, WE TAKE II ###
    par = net.plants[P].m_params
    # steady state tensions in the static and dynamic bag fibers (no gamma inputs)
    Ts_ss = (par['k_se_s']/(par['k_se_s']+par['k_pe_s'])) * (
             par['k_pe_s']*(m_lengths - par['l0_s']))
    Td_ss = (par['k_se_d']/(par['k_se_d']+par['k_pe_d'])) * (
             par['k_pe_d']*(m_lengths - par['l0_d']))
    # steady state afferent outputs (no gamma inputs)
    #Ia_ss = par['fs']*(Ts_ss/par['k_se_s']) + (1.-par['fs'])*(Td_ss/par['k_se_d'])
    II_ss = par['se_II']*(Ts_ss/par['k_se_s']) + ((1.-par['se_II'])/par['k_pe_s'])*Ts_ss
    #Ia_ss *= par['Ia_gain']
    II_ss *= par['II_gain']
    #Ia_II_ss = np.concatenate((Ia_ss, II_ss), axis=1)
    # Next transformation is through the afferent units
    P__A_ws = np.array(P__A_syn['init_w'][12:18])
    #Ia_II_avgs = np.mean(Ia_II_ss, axis=0)  # when using hundreds of random targets
    # target averages
    A_thr = np.array([net.units[u].thresh for u in A[12:18]])
    A_II = np.log(1. + np.maximum((II_ss)*P__A_ws - A_thr, 0.))
    #(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)
    # Next is from A to SF
    SF_arg = A__SF_syn['init_w']*A_II
    SF_out = 1./ (1. + np.exp(-SF_params['slope']*(SF_arg - SF_params['thresh'])))
    SF_params['init_val'] = SF_out # this might cause a smooth start
    # now we set the values in SP
    m_idxs = np.random.randint(len(hand_coords), size=1000) # index of all targets
        #m_idxs[0] = 0 # for testing
    A_us = [net.units[u] for u in A]

    def SF_sigmo(idx, arg):
        """ The sigmoidal function for SF unit with index SF[idx]. """
        return 1./ (1. + np.exp(-SF_params['slope'][idx]*(arg - SF_params['thresh'][idx])))

    def cur_target(t):
        """ Returns the index of the target at time t. """
        return m_idxs[int(np.floor(t/t_pres))]

    def make_fun(idx):
        """ create a function for the SP unit with index 'idx'. """
        return lambda t: SF_sigmo(idx, 
                            A__SF_syn['init_w'][idx] * (
                            np.log(1. + max(II_ss[cur_target(t)][idx] * P__A_ws[idx] - 
                            net.units[A[12+idx]].thresh, 0.))))
        #return lambda t: SF_out[m_idxs[int(np.floor(t/t_pres))]][idx]

    for idx, u in enumerate(SP):
        net.units[u].set_function(make_fun(idx))
    #--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--

    #--------------------------------------------------------------------
    # CONNECTING
    #--------------------------------------------------------------------
    # From afferent units
    net.connect(A[:12], M, A__M_conn, A__M_syn)
    net.connect(A[12:18], SF, A__SF_conn, A__SF_syn)
    # From ACT
    # net.connect(ACT, CE, ACT__C_conn, ACT__C_syn)
    # net.connect(ACT, CI, ACT__C_conn, ACT__C_syn)
    net.connect(ACT, SYNE, ACT__SYN_conn, ACT__SYN_syn)
    net.connect(ACT, SYNI, ACT__SYN_conn, ACT__SYN_syn)
    # from AL to P
    net.set_plant_inputs(AL, P, AL__P_conn, AL__P_syn)
    # # from CE to AL
    # net.connect(CE, AL, CE__AL_conn, CE__AL_syn)
    # # from CI to AL
    # net.connect(CI, AL, CI__AL_conn, CI__AL_syn)
    # # intraspinal connections
    # for pair in all_pairs:
    #     if pair in synergists:
    #         net.connect([CE[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_syne)
    #     elif pair in part_syne:
    #         net.connect([CE[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_p_syne)
    #     elif pair in antagonists:
    #         net.connect([CE[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_antag)
    #     elif pair in part_antag:
    #         net.connect([CE[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_p_antag)
    #     elif pair in self_conn:
    #         net.connect([CE[pair[0]]], [CI[pair[1]]], CE__CI_conn, CE__CI_syn)
    #         net.connect([CI[pair[0]]], [CE[pair[1]]], CI__CE_conn, CI__CE_syn)
    # From M 
    # net.connect(M, CE, M__C_conn, M__CE_syn)
    # net.connect(M, CI, M__C_conn, M__CI_syn)
    net.connect(M, M, M__M_conn, M__M_syn)
    net.connect(M, SYNE, M__SYN_conn, M__SYNE_syn)
    net.connect(M, SYNI, M__SYN_conn, M__SYNI_syn)
    # From plant to afferents
    net.set_plant_outputs(P, A, P__A_conn, P__A_syn) 
    # From SF, SP to SPF
    net.connect(SF, SPF[:6], SFe__SPF_conn, SFe__SPF_syn)
    net.connect(SF, SPF[6:12], SFi__SPF_conn, SFi__SPF_syn)
    net.connect(SP, SPF[:6], SPi__SPF_conn, SPi__SPF_syn)
    net.connect(SP, SPF[6:12], SPe__SPF_conn, SPe__SPF_syn)
    # from SP to SP_CHG
    net.connect(SP, SP_CHG, SP__SP_CHG_conn, SP__SP_CHG_syn)
    # from SP_CHG to ACT
    net.connect(SP_CHG, ACT, SP_CHG__ACT_conn, SP_CHG__ACT_syn)
    # From SPF
    net.connect(SPF, ACT, SPF__ACT_conn, SPF__ACT_syn)
    net.connect(SPF, M, SPF__M_conn, SPF__M_syn)
    net.connect(SPF, SPF, SPF__SPF_conn, SPF__SPF_syn)
    # from SYNE, SYNI
    AL_trgs = []
    for trgs in syn_targets:
        AL_trgs.append([AL[trgs[0]], AL[trgs[1]]])
    for idx in range(N_syne):
        net.connect([SYNE[idx]], AL_trgs[idx], SYNE__AL_conn, SYNE__AL_syn)
        net.connect([SYNI[idx]], AL_trgs[idx], SYNI__AL_conn, SYNI__AL_syn)
        net.connect([SYNE[idx]], [SYNI[idx]], SYNE__SYNI_conn, SYNE__SYNI_syn)
        net.connect([SYNI[idx]], [SYNE[idx]], SYNI__SYNE_conn, SYNI__SYNE_syn)
        if idx%2 == 0:
            net.connect([SYNE[idx]], [SYNI[idx+1]], SYN_antag_conn, SYN_antag_syn)
            net.connect([SYNE[idx+1]], [SYNI[idx]], SYN_antag_conn, SYN_antag_syn)

    # SETTING UP WEIGHT TRACKING
    if track_weights:
        def M_SYNE0_fun(idx):
            """ Creates a function to track a weight from M to SYNE0. """
            return lambda t: net.syns[SYNE[0]][idx].w
        base = 0
        for idx in range(len(M)):
            rga_syn = False
            while not rga_syn:
                if net.syns[SYNE[0]][base+idx].type in [synapse_types.rga_21]:
                    rga_syn = True
                else:
                    base += 1
                    if base > 100:
                        raise AssertionError('Could not create M_CE tracker unit')
            net.units[M_SYNE0_track[idx]].set_function(M_SYNE0_fun(base+idx))

        def A_M0_fun(idx):
            """ Creates a function to track a weight from AF to M0. """
            return lambda t: net.syns[M[0]][idx].w
        base = 0
        for idx, uid in enumerate(A_M0_track):
            rga_syn = False
            while not rga_syn:
                if net.syns[M[0]][base+idx].type in [synapse_types.inp_sel]:
                    rga_syn = True
                else:
                    base += 1
                    if base > 100:
                        raise AssertionError('Could not create A_M tracker unit')
            net.units[uid].set_function(A_M0_fun(base+idx))


    # TRACKING OF INSERTION POINTS (for the arm animation)
    if track_ips:
        # make the source units track the tensions
        def create_xtracker(arm_id, idx):
            return lambda t: net.plants[arm_id].ip[idx][0]
        def create_ytracker(arm_id, idx):
            return lambda t: net.plants[arm_id].ip[idx][1]
        for idx, uid in enumerate(ipx_track):
            net.units[uid].set_function(create_xtracker(P, idx))
        for idx, uid in enumerate(ipy_track):
            net.units[uid].set_function(create_ytracker(P, idx))

    pops_list = [A, ACT, AL, SYNE, SYNI, M, P, SF, SP, SP_CHG, SPF,
                 A_M0_track, M_SYNE0_track, ipx_track, ipy_track]
    pops_names = ['A', 'ACT', 'AL', 'SYNE', 'SYNI', 'M', 'P', 'SF', 'SP', 'SP_CHG',
                  'SPF', 'A_M0_track', 'M_SYNE0_track', 'ipx_track', 'ipy_track']
    pops_dict = {pops_names[idx] : pops_list[idx] for 
                 idx in range(len(pops_names))}
    pds = {'P__A_syn' : P__A_syn, 'SF_params' : SF_params, 
           'A__SF_syn' : A__SF_syn }

    return net, pops_dict, hand_coords, m_idxs, pds


