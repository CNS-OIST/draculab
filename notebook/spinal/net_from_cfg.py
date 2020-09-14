# coding: utf-8

# net_from_cfg.py
# A function to be used with the v3afxB_test notebooks


from draculab import *

def net_from_cfg(cfg,
                 t_pres=30.,
                 rand_w=True,
                 rga_diff=True,
                 rand_targets=True,
                 M_mod=True,
                 lowpass_SP=True,
                 par_heter=0.001,
                 noisy_syns = False,
                 M__C_decay = False,
                 AF__M_decay = False):
    """ Create a draculab network with the given configuration. 

        Args:
            cfg : a parameter dictionary
            
            Optional keyword arguments:
            t_pres: number of seconds to hold each set of target lengths
            rand_w: whether to use random weights in M->C, AF->M
            rga_diff: if True use gated_normal_rga_diff, if False gated_normal_rga
            rand_targets: whether to train using a large number of random targets
            M_mod = True # whether M units are amplitude-modulated by the SPF input
            par_heter: range of heterogeneity as a fraction of the original value
            lowpass_SP: whether to filter SP's output with slow-responding linear units
            noisy_syns: whether to use noisy versions of the M__C, AF__M synapses
            M__C_decay: if using noisy_syns, replace normalization and drift with decay
            AF__M_decay: if using noisy_syns, replace normalization and drift with decay

        Returns:
            A tuple with the following entries:
            net : A draculab network as in v3_nst_afx, with the given configuration.
            pops_dict : a dictionary with the list of ID's for each population in net.
            hand_coords : list with the coordinates for all possible targets
            m_idxs : which target coordinats will be used for the i-th presentation
            t_pres : number of seconds each target is presented
            pds : some parameter dictionaries used to set new targets
    """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameter dictionaries for the network and the plant
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    net_params = {'min_delay' : 0.005,
              'min_buff_size' : 10 }

    P_params = {  'type' : plant_models.bouncy_planar_arm_v3,
              'mass1': 1.,
              'mass2': 1.,
              's_min' : -0.8,
              'p1' : (-0.01, 0.04),
              'p2' : (0.29, 0.03),
              'p3' : (0., 0.05),
              'p5' : (0.01, -0.05),
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
              'g_e' : cfg['g_e_factor']*np.array([18., 20., 20., 18., 22., 23.]),
              'l0_e' : [1.]*6,
              'Ia_gain' : 2.5*np.array([3.,10.,10., 3.,10.,10.]),
              'II_gain' : 2.*np.array([3., 8., 8., 3., 8., 8.]),
              'Ib_gain' : 1.,
              'T_0' : 10.,
              'k_pe_e' : 20.,  #8
              'k_se_e' : 20., #13
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
    net = network(net_params) # network creation

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
    #unrelated = set(all_pairs) - set(antagonists) - set(part_antag) - set(synergists) - set(part_syne) - set(self_conn)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # UNIT PARAMETER DICTIONARIES
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    randz6 = lambda : (1. + par_heter*(np.random.rand(6)-0.5))
    randz12 = lambda : (1. + par_heter*(np.random.rand(12)-0.5))
    randz18 = lambda : (1. + par_heter*(np.random.rand(18)-0.5))
    randz36 = lambda : (1. + par_heter*(np.random.rand(36)-0.5))

    ACT_params = {'type' : unit_types.act,
                  'tau_u' : 6., #8
                  'gamma' : 6., #2
                  'g' : 2.,
                  'theta' : 1.,
                  'tau_slow' : 5.,
                  'y_min' : 0.1, #.2
                  'rst_thr' : 0.1,
                  'init_val' : 0. }
    spf_sum_min = .4 # value where no corrections are needed anymore
    y_min = 1./(1. + np.exp(-ACT_params['g']*(spf_sum_min - ACT_params['theta'])))
    ACT_params['y_min'] = y_min

    AF_params = {'type' : unit_types.logarithmic,
                 'init_val' : [0.1, 0.05, 0.15, 0.1, 0.1, 0.1, # avg afferent values
                               0.2, 0.15, 0.3, 0.3, 0.2, 0.25,
                               0.2, 0.4, 0.4, 0.2, 0.4, 0.4]*2,
                               #0.3, 0.4, 0.5, 0.3, 0.3, 0.5]*2,
                 'tau' : 0.02 * randz36(),
                 'tau_fast': 0.1,
                 'tau_mid' : 1.,
                 'tau_slow' : 40.,
                 'delay' : 0.1,
                 'thresh' : [0.05]*18 + [-0.4]*18 } 
    AL_params = {'type' : unit_types.sigmoidal,
                 'thresh' : cfg['AL_thresh'] * randz6(),
                 'slope' : 2. * randz6(),
                 'init_val' : 0.1 * randz6(),
                 'tau' : 0.02 * randz6() }
    CE_params = {'type' : unit_types.gated_rga_inpsel_adapt_sig,
                 'thresh' : 0. * randz6(),
                 'slope' : 1.5 * randz6(),
                 'init_val' : 0.2 * randz6(),
                 'tau' : 0.02,
                 'tau_fast': 0.1,
                 'tau_mid' : 1.,
                 'tau_slow' : cfg['C_tau_slow'],
                 'custom_inp_del' : 15, # placeholder values
                 'custom_inp_del2': 30,
                 'integ_amp' : cfg['integ_amp'],
                 'integ_decay' : cfg['integ_decay'],
                 'adapt_amp' : cfg['adapt_amp'],
                 'delay' : 0.2,
                 'des_out_w_abs_sum' : 1. }
    CI_params = {'type' : unit_types.gated_rga_inpsel_adapt_sig,
                 'thresh' : 0.5 * randz6(),
                 'slope' : 2. * randz6(),
                 'init_val' : 0.2 * randz6(),
                 'tau' : 0.1,
                 'tau_fast': 0.1,
                 'tau_mid' : 1.,
                 'tau_slow' : cfg['C_tau_slow'],
                 'custom_inp_del' : 15, # placeholder values
                 'custom_inp_del2': 30,
                 'integ_amp' : cfg['integ_amp'], #.5,
                 'integ_decay' : cfg['integ_decay'],
                 'adapt_amp' : cfg['adapt_amp'],
                 'delay' : 0.2,
                 'des_out_w_abs_sum' : 1. }
    LPF_SP_params = {'type' : unit_types.linear,
                 'init_val' : 0.3 * randz12(),
                 'tau_fast': 0.005,
                 'tau_mid': 0.05,
                 'tau_slow' : 5.,
                 'tau' : .5 }
    M_type = unit_types.gated_out_norm_am_sig if M_mod else unit_types.gated_out_norm_sig
    M_params = {'type' : M_type,
                'thresh' : (cfg['M_thresh'] if 'M_thresh' in cfg else 0.1) * randz12(),
                'slope' : (cfg['M_slope'] if 'M_slope' in cfg else 3.) * randz12(),
                'init_val' : 0.2 * randz12(),
                'delay' : 0.2,
                'tau_fast': 0.15,
                'tau_mid': 1.5,
                'tau_slow' : 10.,
                'tau' : 0.01 * randz12(),
                'p1_inp' : cfg['M_p1_inp'] if 'M_p1_inp' in cfg else 0.,
                'des_out_w_abs_sum' : 2. }
    SF_params = {'type' : unit_types.sigmoidal,
                 #'thresh' : np.array([-0.02]*12)
                 'thresh' : np.array([-0.12, -0.13, -0.05, -0.11, -0.03, -0.05,
                                      -0.05, -0.07, 0.05, 0.06, -0.03, -0.02]),
                 #'slope' : np.array([np.log(5.)]*12), #np.array([np.log(9.)]*12),
                 'slope' : cfg['SF_slope_factor']*np.array([2.75, 1.7, 1.37, 2.75] + [1.37]*2 + [2., 1.37, 1.37]*2),
                 'init_val' : 0.2 * randz12(),
                 'tau' : 0.03 * randz12() } 
    SP_params = {'type' : unit_types.source,
                 'init_val' : 0.5,
                 'tau_fast' : 0.02,
                 'tau_mid' : 0.1,
                 'function' : lambda t: None }
    SP_CHG_params = {'type' : unit_types.sigmoidal,
                  'thresh' : 0.25,
                  'slope' : 9.,
                  'init_val' : 0.1,
                  'tau' : 0.01 }
    SPF_params = {'type' : unit_types.logarithmic, #sigmoidal,
                  'thresh' : -0.1, #0.4 * randz12(),
                  'slope' : 6. * randz12(),
                  'init_val' : 0.3 * randz12(),
                  'tau_fast': 0.005,
                  'tau_mid': 0.05,
                  'tau_slow' : 5.,
                  'tau' : 0.02 * randz12() }
    track_params = {'type' : unit_types.source,
                    'init_val' : 0.02,
                    'function' : lambda t: None }

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CONNECTION DICTIONARIES
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ACT to CE,CI ------------------------------------------------
    ACT__CE_conn = {'rule' : "all_to_all",
                    'delay' : 0.02 } 
    ACT__CE_syn = {'type' : synapse_types.static,
                   'inp_ports' : 4,
                   'init_w' : 1. }
    ACT__CI_conn = {'rule' : "all_to_all",
                    'delay' : 0.02 } 
    ACT__CI_syn = {'type' : synapse_types.static,
                   'inp_ports' : 4,
                   'init_w' : 1. }
    # AF to CE, CI --------------------------------------------------
    AF__CE_conn = {'rule' : 'all_to_all',
                   'delay' : 0.02 }
    AF__CE_syn = {'type' : synapse_types.gated_inp_sel,
                  'aff_port' : 2,
                  'inp_ports' : 2,
                  'error_port' : 0,
                  'normalize' : True,
                  'w_sum' : 2.,
                  'lrate' : 0., #10.,
                  'extra_steps' : 1,
                  'init_w' : 0.005 }
    AF__CI_conn = {'rule' : 'all_to_all',
                   'delay' : 0.02 }
    AF__CI_syn = {'type' : synapse_types.gated_inp_sel,
                  'aff_port' : 2,
                  'inp_ports' : 2,
                  'error_port' : 0,
                  'normalize' : True,
                  'w_sum' : 2.,
                  'lrate' : 0., #10.,
                  'extra_steps' : 1,
                  'init_w' : 0.005 }
    # AF to M ------------------------------------------------
    ## Creating a test matrix
    if not rand_w:
        # Initializing manually
        AF_M = np.zeros((36, 12)) # rows are source, columns target
        for src in range(36):
            for trg in range(12):
                src_pop = src%6 #src's population
                trg_pop = trg%6 #trg's population
                if src%18 < 6: # if the afferent is tension, don't reverse signs
                    sig = 1
                else:
                    sig = -1
                if src > 17: sig = -sig # if 'negative' afferent reverse sign
                for pair in antagonists:
                    if pair == (src_pop, trg_pop):
                        AF_M[src, trg] = sig*0.2
                        break
                else: 
                    for pair in part_antag:
                        if pair == (src_pop, trg_pop):
                            AF_M[src, trg] = sig*0.1
                            break
                    else: 
                        for pair in synergists:
                            if pair == (src_pop, trg_pop):
                                AF_M[src, trg] = sig*-0.2
                                break
                        else: 
                            for pair in synergists:
                                if pair == (src_pop, trg_pop):
                                    AF_M[src, trg] = sig*-0.2
                                    break
                            else: 
                                for pair in part_syne:
                                    if pair == (src_pop, trg_pop):
                                        AF_M[src, trg] = sig*-0.1
                                        break
                                else:
                                    if src_pop == trg_pop:
                                        AF_M[src, trg] = sig*-0.3
    else:
        #AF_M = 0.2*(np.random.random((12,12)) - 0.5) # random initial connections!!!!!
        AF_M = 0.2*(np.random.random((12,36)) - 0.5) # random initial connections!!!!!
    if noisy_syns:
        AF__M_syn_type = synapse_types.noisy_gated_diff_inp_sel
    else:
        AF__M_syn_type = synapse_types.gated_diff_inp_sel
    AF__M_conn = {'rule' : 'all_to_all',
                 'delay' : 0.02 }
    AF__M_syn = {'type' : AF__M_syn_type,
                 'aff_port' : 0,
                 'error_port' : 1,
                 'normalize' : not AF__M_decay if noisy_syns else True,
                 'w_sum' : cfg['AF__M_w_sum'] if 'AF__M_w_sum' in cfg else 10.,
                 'inp_ports' : 0, # afferent for out_norm_am_sig
                 'input_type' : 'pred', # if using inp_corr
                 'lrate' : cfg['AF__M_lrate'] if 'AF__M_lrate' in cfg else 15., 
                 'decay' : AF__M_decay, # for noisy_gated_normal_rga_diff
                 'de_rate' : cfg['AF__M_de_rate'] if 'AF__M_de_rate' in cfg else 0.01, 
                 'dr_amp' : 0.01, # drift amplitude (noisy_gated_normal_rga_diff)
                 'extra_steps' : None, # placeholder value; filled below,
                 'init_w' : AF_M.flatten() }
    # AF to SF ------------------------------------------------
    AF__SF_dubya = np.array([0.57, 0.56, 0.56, 0.57, 0.55, 0.55, 0.57, 0.56, 0.56, 0.57, 0.55, 0.55])
    AFe__SF_conn = {'rule' : 'one_to_one',
                  'delay' : 0.02 }
    AFe__SF_syn = {'type' : synapse_types.static,
                 #'init_w' : .5*np.array([ 39.73, 15.03,  9.66, 116.47, 8.63,  63.68, 20.88, 9.69, 5.86, 67.98, 5.44, 57.38]) }
                   #'init_w' : np.array([15.07516819, 20.60577773,  6.32821777, 14.08991768,  5.69679834,  8.89424814,
                   #                     7.68013805, 13.46076843,  3.87962611,  7.1693345,   3.5817668,   5.8075114 ])}
                   'init_w' : AF__SF_dubya}
    AFi__SF_conn = {'rule' : 'one_to_one',
                  'delay' : 0.02 }
    AFi__SF_syn = {'type' : synapse_types.static,
                 #'init_w' : .5*np.array([-162.42, -9.56, -15.19, -50.0, -88.47, -9.73, -67.67, -5.85, -9.96, -23.55, -50.63, -5.81])}
                  #'init_w' : np.array([-15.36496074,  -5.96985989, -21.78133701, -18.88200345, -11.38248201,
                  #            -6.93261446,  -7.53984025,  -3.68499008, -14.59079011,  -9.22649886,  -7.03063956,  -4.19297924])}
                  'init_w' : -AF__SF_dubya }
    # AL to P ------------------------------------------------
    AL__P_conn = {'inp_ports' : list(range(6)),
                 'delays': 0.01 }
    AL__P_syn = {'type': synapse_types.static,
                'init_w' : 1. }
    # CE, CI to AL ----------------------------------------------
    CE__AL_conn = {'rule' : 'one_to_one',
                   'delay' : 0.01 }
    CE__AL_syn = {'type' : synapse_types.static,
                  'init_w' : [1., 1., 1., 1., 1., 1.] }
    CI__AL_conn = {'rule' : 'one_to_one',
                   'delay' : 0.01 }
    CI__AL_syn = {'type' : synapse_types.static,
                  'init_w' : -1. }
    # CE,CI to CE,CI  ------------------------------------------------
    CE__CI_conn = {'rule' : 'one_to_one',
                   'delay' : 0.01 }
    CI__CE_conn = {'rule' : 'one_to_one',
                   'delay' : 0.01 }
    CE__CI_syn = {'type' : synapse_types.static,
                  'inp_ports' : 2, #1, # IN AFFERENT PORT!!!!!!!!!!!!!!!!!!!!!! May affect normalization of afferent inputs
                  'init_w' : cfg['CE__CI_w'] if 'CE__CI_w' in cfg else 1. }
    CI__CE_syn = {'type' : synapse_types.static, #static, #corr_inh,
                  'inp_ports' : 2, #1, # IN AFFERENT PORT!!!!!!!!!!!!!!!!!!!!!! May affect normalization of afferent inputs
                  'lrate' : .0,
                  'des_act' : 0.5,
                  'init_w' : cfg['CI__CE_w'] if 'CI__CE_w' in cfg else -2. }
    C__C_conn = {'rule': 'one_to_one',
                 'allow_autapses' : False,
                 'delay' : 0.015 }
    C__C_syn_antag = {'type' : synapse_types.static, #bcm,
                      'inp_ports': 2, #1, # IN AFFERENT PORT!!!!!!!!!!!!!!!!!!!!!! May affect normalization of afferent inputs
                      'init_w' : cfg['C__C_antag'] if 'C__C_antag' in cfg else 2., #16.,
                      'lrate' : 1.,
                      'des_act' : .5 }
    C__C_syn_p_antag = {'type' : synapse_types.static, #bcm,
                      'inp_ports': 2, #1, # IN AFFERENT PORT!!!!!!!!!!!!!!!!!!!!!! May affect normalization of afferent inputs
                      'init_w' : cfg['C__C_p_antag'] if 'C__C_p_antag' in cfg else .5,
                      'lrate' : 1.,
                      'des_act' : 0.2 }
    C__C_syn_syne = {'type' : synapse_types.static,
                     'inp_ports': 1,
                     'lrate' : 1.,
                     'init_w' : .5 }
    C__C_syn_p_syne = {'type' : synapse_types.static,
                       'inp_ports': 1,
                       'lrate' : 1.,
                       'init_w' : 0.2 }
    C__C_syn_null_lat = {'type' : synapse_types.static, # connection with static weight zero
                       'inp_ports': 1,
                       'lrate' : 1.,
                       'init_w' : 0. }
    C__C_syn_null_aff = {'type' : synapse_types.static, # connection with static weight zero
                       'inp_ports': 2, #1, # IN AFFERENT PORT!!!!!!!!!!!!!!!!!!!!!! May affect normalization of afferent inputs
                       'lrate' : 1.,
                       'init_w' : 0. }
    
    # LPF_SP to SPF (optional) ---------------------------------
    LPF_SPe__SPF_conn = {'rule': 'one_to_one',
                         'delay': 0.01 }
    LPF_SPe__SPF_syn = {'type' : synapse_types.static,
                        'inp_ports' : 1,
                        'lrate' : 0.,
                        'input_type' : 'error', # if using inp_corr
                        'init_w' : cfg['SPF_w'] }
    LPF_SPi__SPF_conn = {'rule': 'one_to_one',
                         'delay': 0.02 }
    LPF_SPi__SPF_syn = {'type' : synapse_types.static,
                        'inp_ports' : 1,
                        'lrate' : 0.,
                        'input_type' : 'error', # if using inp_corr
                        'init_w' : -cfg['SPF_w'] }

    # M to CE,CI ----------------------------------------------
    # creating a test matrix
    if not rand_w:
        # initializing manually
        M_CE = np.array(
            [[ 0.2,  0.1, -0.1, -0.2,  0.1, -0.1],
             [ 0.1,  0.2, -0.2, -0.1,  0.1,  0.0],
             [-0.1, -0.2,  0.2,  0.1,  0.0,  0.0],
             [-0.2, -0.1,  0.1, -0.2, -0.1,  0.1],
             [ 0.1,  0.0,  0.0, -0.1,  0.3, -0.2],
             [-0.1,  0.0,  0.0,  0.1, -0.2,  0.3],
             [ 0.2,  0.1, -0.1, -0.2,  0.1, -0.1],
             [ 0.1,  0.2, -0.2, -0.1,  0.1,  0.0],
             [-0.1, -0.2,  0.2,  0.1,  0.0,  0.0],
             [-0.2, -0.1,  0.1, -0.2, -0.1,  0.1],
             [ 0.1,  0.0,  0.0, -0.1,  0.3, -0.2],
             [-0.1,  0.0,  0.0,  0.1, -0.2,  0.3]])
        M_CI = -M_CE 
    else:
        M_CE = 0.4*(np.random.random((12,6)) - 0.5) # random initial connections!!!!!
        M_CI = 0.4*(np.random.random((12,6)) - 0.5) # random initial connections!!!!!
    if rga_diff:
        if noisy_syns:
            M__C_type = synapse_types.noisy_gated_normal_rga_diff
        else:
            M__C_type = synapse_types.gated_normal_rga_diff
    else:
        if noisy_syns:
            raise NotImplementedError('noisy_gated_normal_rga not implemented')
        else:
            M__C_type = synapse_types.gated_normal_rga
    M__CE_conn = {'rule': 'all_to_all',
                 'delay': 0.02 }
    M__CE_syn = {'type' : M__C_type,
                 'inp_ports' : 0,
                 'lrate' : cfg['M__C_lrate'] if 'M__C_lrate' in cfg else 20.,
                 'w_sum' : cfg['M__C_w_sum'],
                 'sig1' : cfg['sig1'],
                 'sig2' : cfg['sig2'],
                 'w_thresh' : 0.05,
                 'w_decay': 0.005,
                 'decay' : M__C_decay, # for noisy_gated_normal_rga_diff
                 'normalize' : not M__C_decay if noisy_syns else True,
                 'de_rate' : cfg['M__C_de_rate'] if 'M__C_de_rate' in cfg else 0.01,
                 'dr_amp' : cfg['M__C_dr_amp'] if 'M__C_dr_amp' in cfg else 0.01,
                 'w_tau' : 60.,
                 'init_w' : M_CE.flatten() }
    M__CI_conn = {'rule': 'all_to_all',
                 'delay': 0.02 }
    M__CI_syn = {'type' : M__C_type,
                 'inp_ports' : 0,
                 'lrate' : cfg['M__C_lrate'] if 'M__C_lrate' in cfg else 20.,
                 'w_sum' : cfg['M__C_w_sum'],
                 'sig1' : cfg['sig1'],
                 'sig2' : cfg['sig2'],
                 'w_thresh' : 0.05,
                 'w_tau' : 60.,
                 'w_decay': 0.005,
                 'decay' : M__C_decay, # for noisy_gated_normal_rga_diff
                 'normalize' : not M__C_decay if noisy_syns else True,
                 'de_rate' : cfg['M__C_de_rate'] if 'M__C_de_rate' in cfg else 0.01,
                 'dr_amp' : cfg['M__C_dr_amp'] if 'M__C_dr_amp' in cfg else 0.01,
                 'init_w' : M_CI.flatten() }
    # P to AF  ---------------------------------------------------
    idx_aff = np.arange(22,40) # indexes for afferent output in the arm
    P__AF_conn = {'port_map' : [[(p,0)] for p in idx_aff],
                 'delays' : 0.02 }
    Pe__AF_syn = {'type' : synapse_types.static,
                  'init_w' : [1.]*18 } 
    Pi__AF_syn = {'type' : synapse_types.static,
                'init_w' :  [-1.]*18 }
    # SP to LPF_SP ----------------------------------------------
    SP__LPF_SP_conn = {'rule': 'one_to_one',
                       'delay': 0.01 }
    SP__LPF_SP_syn = {'type' : synapse_types.static,
                      'init_w' : 1. }
    # SF, SP to SPF ------------------------------------------------
    SFe__SPF_conn = {'rule' : "one_to_one",
                     'delay' : 0.01 }
    SFi__SPF_conn = {'rule' : "one_to_one",
                     'delay' : 0.02 }
    SFe__SPF_syn = {'type' : synapse_types.static,
                    'init_w' : cfg['SPF_w'] }
    SFi__SPF_syn = {'type' : synapse_types.static,
                    'init_w' : -cfg['SPF_w'] }
    SPe__SPF_conn = {'rule' : "one_to_one",
                     'delay' : 0.01 }
    SPi__SPF_conn = {'rule' : "one_to_one",
                     'delay' : 0.02 }
    SPe__SPF_syn = {'type' : synapse_types.static,
                    'init_w' : cfg['SPF_w'] }
    SPi__SPF_syn = {'type' : synapse_types.static,
                   'init_w' : -cfg['SPF_w'] }
    # SP to SP_CHG ------------------------------------------------
    SP__SP_CHG_conn = {'rule' : 'all_to_all',
                        'delay' : 0.01}
    SP__SP_CHG_syn = {'type' : synapse_types.chg,
                      'init_w' : 0.,
                      'lrate' : 20. }
    # SP_CHG to CE, CI ------------------------------------------------
    SP_CHG__CE_conn = {'rule' : "all_to_all",
                      'delay' : 0.02 }
    SP_CHG__CE_syn = {'type' : synapse_types.static,
                      'inp_ports' : 3,
                      'init_w' : 1. }
    SP_CHG__CI_conn = {'rule' : "all_to_all",
                       'delay' : 0.02 }
    SP_CHG__CI_syn = {'type' : synapse_types.static,
                      'inp_ports' : 3,
                      'init_w' : 1. }
    # SP_CHG to ACT ------------------------------------------------
    SP_CHG__ACT_conn = {'rule' : "all_to_all",
                       'delay' : 0.02 }
    SP_CHG__ACT_syn = {'type' : synapse_types.static,
                      'inp_ports' : 1,
                      'init_w' : 1. }
    # SP_CHG to M ------------------------------------------------
    SP_CHG__M_conn = {'rule' : "all_to_all",
                      'delay' : 0.02 }
    SP_CHG__M_syn = {'type' : synapse_types.static,
                      'inp_ports' : 2,
                      'init_w' : 1. }
    # SPF to ACT ------------------------------------------------
    SPF__ACT_conn = {'rule' : "all_to_all",
                     'delay' : 0.02 }
    SPF__ACT_syn = {'type' : synapse_types.static,
                    'inp_ports' : 0,
                    'init_w' : 1. }
    # SPF to M  ------------------------------------------------
    SPF__M_conn = {'rule': 'one_to_one',
                   'delay': 0.01 }
    SPF__M_syn = {'type' : synapse_types.static, #synapse_types.inp_corr,
                  'inp_ports' : 1,
                  'lrate' : 0.,
                  'input_type' : 'error', # if using inp_corr
                  'init_w' : 1. }

    #*************************************************************
    # Setting the right delay for AF-->M
    f = 1. # going to estimate the extra delay of error inputs wrt afferent inputs at M
    w = 2.*np.pi*f
    sf_del = np.arctan(np.mean(SF_params['tau'])*w)/w
    spf_del = np.arctan(np.mean(SPF_params['tau'])*w)/w
    delay = spf_del + sf_del + AFe__SF_conn['delay'] + SFe__SPF_conn['delay']
    steps = int(round(delay/net.min_delay))
    AF_params['delay'] = AF_params['delay'] + (
                         net_params['min_delay'] * (np.ceil(delay/net_params['min_delay']) + 1))
    AF__M_syn['extra_steps'] = steps
    #*************************************************************
    # utilitiy function for the M-->C delays used in the rga rule
    def approx_del(f):
        """ Returns an estimate fo the optimal delay for rga learning.

            We assume that the important loop for the learning rule in the C units
            is the one going through C-AL-P-AF-M-C.
            We also assume the delays to/from CI are the same as the ones for CE.

            Args:
                f : oscillation frequency of E-I pair in C, in Hertz
            Returns:
                2-tuple : (time_del, del_steps)
                time_del : A float with the time delay.
                del_steps : time delay as integer number of min_del steps.
        """
        w = 2.*np.pi*f
        al_del = np.arctan(np.mean(AL_params['tau'])*w)/w
        p_del = np.arctan(np.mean(P_params['tau'])*w)/w
        af_del = np.arctan(np.mean(AF_params['tau'])*w)/w
        m_del = np.arctan(np.mean(M_params['tau'])*w)/w
        D = [CE__AL_conn['delay'], AL__P_conn['delays'], np.mean(P__AF_conn['delays']),
             AF__M_conn['delay'], M__CE_conn['delay'] ]
        time_del = al_del + p_del + af_del + m_del + sum(D)
        del_steps = int(np.ceil(time_del/net_params['min_delay']))
        time_del = del_steps*net_params['min_delay']
        del_steps -= 1 # because this is an index, and indexes start at 0
        return time_del, del_steps
    ############## Approximating the delays for the rga rule #############
    ######## Using the utility function (for rga synapses)
    # time_del, del_steps = approx_del(0.01) #0.65 was approximate CE/CI frequency observed in simulations
    # #time_del, del_steps = (1., 200-1)
    # CE_params['delay'] = time_del
    # CI_params['delay'] = time_del
    # M_params['delay'] = time_del
    # CE_params['custom_inp_del'] = del_steps
    # CI_params['custom_inp_del'] = del_steps
    ######## Using the two custom delays (for rga_diff synapses)
    dely1 = round(cfg['dely_low']/net.min_delay)*net.min_delay
    dely2 = dely1 + round(cfg['dely_diff']/net.min_delay)*net.min_delay
    del_steps1 = int(np.ceil(dely1/net_params['min_delay'])) - 1
    del_steps2 = int(np.ceil(dely2/net_params['min_delay'])) - 1
    CE_params['delay'] = dely2 + 0.01
    CI_params['delay'] = dely2 + 0.01
    M_params['delay'] = dely2 + 0.01
    CE_params['custom_inp_del'] = del_steps1
    CI_params['custom_inp_del'] = del_steps1
    CE_params['custom_inp_del2'] = del_steps2
    CI_params['custom_inp_del2'] = del_steps2
    #*************************************************************

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CREATING UNITS
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    P = net.create(1, P_params)
    arm = net.plants[P]

    ACT = net.create(1, ACT_params)
    AF = net.create(36, AF_params)
    AL = net.create(6, AL_params)
    CE = net.create(6, CE_params)
    CI = net.create(6, CI_params)
    M = net.create(12, M_params)
    SF = net.create(12, SF_params)
    SP = net.create(12, SP_params)
    SP_CHG = net.create(1, SP_CHG_params)
    SPF = net.create(12, SPF_params)
    if lowpass_SP:
        LPF_SP = net.create(12, LPF_SP_params)

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
                       #[-0.1, 0.3],
                       #[-0.1, 0.35]] # experimental extra coordinates
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
        m_lengths.append(arm.coords_to_lengths(coord))
    m_lengths = np.array(m_lengths)
    #(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)
    # We need to translate these lengths to corresponding SF activity levels.
    # For that it is necessary to recreate all their transformations
    # The first transformation is from length to Ia, II afferent activity.
    ### OUT OF THE 36 AFFERENT SIGNALS, WE TAKE THE Ia AND II ###
    par = net.plants[P].m_params
    # steady state tensions in the static and dynamic bag fibers (no gamma inputs)
    Ts_ss = (par['k_se_s']/(par['k_se_s']+par['k_pe_s'])) * (
             par['k_pe_s']*(m_lengths - par['l0_s']))
    Td_ss = (par['k_se_d']/(par['k_se_d']+par['k_pe_d'])) * (
             par['k_pe_d']*(m_lengths - par['l0_d']))
    # steady state afferent outputs (no gamma inputs)
    Ia_ss = par['fs']*(Ts_ss/par['k_se_s']) + (1.-par['fs'])*(Td_ss/par['k_se_d'])
    II_ss = par['se_II']*(Ts_ss/par['k_se_s']) + ((1.-par['se_II'])/par['k_pe_s'])*Ts_ss
    Ia_ss *= par['Ia_gain']
    II_ss *= par['II_gain']
    Ia_II_ss = np.concatenate((Ia_ss, II_ss), axis=1)
    # Next transformation is through the afferent units
    Pe__AF_ws = np.array(Pe__AF_syn['init_w'][6:18])
    Pi__AF_ws = np.array(Pi__AF_syn['init_w'][6:18])
    #Ia_II_avgs = np.mean(Ia_II_ss, axis=0)  # when using hundreds of random targets
    # target averages
    AFe_thr = np.array([net.units[u].thresh for u in AF[6:18]])
    AFi_thr = np.array([net.units[u].thresh for u in AF[24:36]])
    #AF_Ia = np.maximum((Ia_ss - AF_avgs[0:6])*Pe__AF_Ia_ws - AF_thr[0:6], 0.)
    #AF_II = np.maximum((II_ss - AF_avgs[6:12])*Pe__AF_II_ws - AF_thr[6:12], 0.)
    AFe_Ia_II = np.log(1. + np.maximum((Ia_II_ss)*Pe__AF_ws - AFe_thr, 0.))
    AFi_Ia_II = np.log(1. + np.maximum((Ia_II_ss)*Pi__AF_ws - AFi_thr, 0.))
    #(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)(.)
    # Next is from AF to SF
    SF_arg = AFe__SF_syn['init_w']*AFe_Ia_II + AFi__SF_syn['init_w']*AFi_Ia_II
    SF_out = 1./ (1. + np.exp(-SF_params['slope']*(SF_arg - SF_params['thresh'])))
    SF_params['init_val'] = SF_out # this might cause a smooth start
    # now we set the values in SP
    m_idxs = np.random.randint(len(hand_coords), size=1000) # index of all targets
        #m_idxs[0] = 0 # for testing
    AF_us = [net.units[u] for u in AF]

    def SF_sigmo(idx, arg):
        """ The sigmoidal function for SF unit with index SF[idx]. """
        return 1./ (1. + np.exp(-SF_params['slope'][idx]*(arg - SF_params['thresh'][idx])))

    def cur_target(t):
        """ Returns the index of the target at time t. """
        return m_idxs[int(np.floor(t/t_pres))]

    def make_fun(idx):
        """ create a function for the SP unit with index 'idx'. """
        return lambda t: SF_sigmo(idx, 
                            AFe__SF_syn['init_w'][idx] * (
                            np.log(1. + max(Ia_II_ss[cur_target(t)][idx] * Pe__AF_ws[idx] - 
                            net.units[AF[6+idx]].thresh, 0.))) +
                            AFi__SF_syn['init_w'][idx] * (
                            np.log(1. + max(Ia_II_ss[cur_target(t)][idx] * Pi__AF_ws[idx] - 
                            net.units[AF[24+idx]].thresh, 0.))))
        #return lambda t: SF_out[m_idxs[int(np.floor(t/t_pres))]][idx]

    for idx, u in enumerate(SP):
        net.units[u].set_function(make_fun(idx))

    # tracking units
    #M_CE_track = net.create(len(M), track_params) # to track weights from M to CE
    #M_CI_track = net.create(len(M), track_params) # to track weights from M to CI
    #AF_M0_track = net.create(18, track_params) # to track the weights from AF to M0

    # xp_track = net.create(1, track_params) # del_avg_inp_deriv of C0 at port 1
    # up_track = net.create(1, track_params) # to track the derivative of CE0
    # if rga_diff is True:
    #     sp_now_track = net.create(1, track_params)
    #     sp_del_track = net.create(1, track_params)
    #     spj_now_track = net.create(1, track_params)
    #     spj_del_track = net.create(1, track_params)
    # else:
    #     sp_track = net.create(1, track_params) # avg_inp_deriv_mp for CE0 at port 0
    #     spj_track = net.create(1, track_params) # input derivative for MX--CE0

    ipx_track = net.create(12, track_params) # x coordinates of insertion points
    ipy_track = net.create(12, track_params) # y coordinates of insertion points

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CONNECTING
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # from M to CE
    net.connect(M, CE, M__CE_conn, M__CE_syn)
    # from M to CI
    net.connect(M, CI, M__CI_conn, M__CI_syn)
    # from CE to AL
    net.connect(CE, AL, CE__AL_conn, CE__AL_syn)
    # from CI to AL
    net.connect(CI, AL, CI__AL_conn, CI__AL_syn)
    # from AL to P
    net.set_plant_inputs(AL, P, AL__P_conn, AL__P_syn)
    # from P to AF
    net.set_plant_outputs(P, AF[0:18], P__AF_conn, Pe__AF_syn)
    net.set_plant_outputs(P, AF[18:36], P__AF_conn, Pi__AF_syn)
    # from AF to SF. Only Ia and II are selected
    net.connect(AF[6:18], SF, AFe__SF_conn, AFe__SF_syn)
    net.connect(AF[24:36], SF, AFi__SF_conn, AFi__SF_syn)
    # from AF to M
    ## When connecting from all afferents:
    net.connect(AF, M, AF__M_conn, AF__M_syn) # should be made before SPF-->M
    ## When connecting only from tension afferents
    #net.connect(AF[0:6]+AF[18:24], M, AF__M_conn, AF__M_syn) # should be made before SPF-->M
    # from AF to CE,CI
    #net.connect(AF, CE, AF__CE_conn, AF__CE_syn)
    #net.connect(AF, CI, AF__CI_conn, AF__CI_syn)
    # from SP to SPF (or LP->LPF_SP->SPF)
    if lowpass_SP:
        net.connect(SP, LPF_SP, SP__LPF_SP_conn, SP__LPF_SP_syn)
        #net.connect(LPF_SP, SPF, LPF_SPe__SPF_conn, LPF_SPe__SPF_syn) # P-F
        net.connect(LPF_SP, SPF, LPF_SPi__SPF_conn, LPF_SPi__SPF_syn) # F-P
    else:
        net.connect(SP, SPF, SPi__SPF_conn, SPi__SPF_syn) # F-P
        #net.connect(SP, SPF, SPe__SPF_conn, SPe__SPF_syn)  # P-F
    # from SF to SPF
    net.connect(SF, SPF, SFe__SPF_conn, SFe__SPF_syn) # F-P
    #net.connect(SF, SPF, SFi__SPF_conn, SFi__SPF_syn)  # P-F
    # from SPF to M
    net.connect(SPF, M, SPF__M_conn, SPF__M_syn) # should be after AF-->M
    # from SPF to ACT
    net.connect(SPF, ACT, SPF__ACT_conn, SPF__ACT_syn)
    # from SP to SP_CHG
    net.connect(SP, SP_CHG, SP__SP_CHG_conn, SP__SP_CHG_syn)
    # from SP_CHG to CE,CI
    net.connect(SP_CHG, CE, SP_CHG__CE_conn, SP_CHG__CE_syn)
    net.connect(SP_CHG, CI, SP_CHG__CI_conn, SP_CHG__CI_syn)
    # from SP_CHG to M
    net.connect(SP_CHG, M, SP_CHG__M_conn, SP_CHG__M_syn)
    # from SP_CHG to ACT
    net.connect(SP_CHG, ACT, SP_CHG__ACT_conn, SP_CHG__ACT_syn)
    # from ACT to CE, CI
    net.connect(ACT, CE, ACT__CE_conn, ACT__CE_syn)
    net.connect(ACT, CI, ACT__CI_conn, ACT__CI_syn)
    # intraspinal connections 
    # from CE to CI, and CI to CE
    #net.connect(CE, CI, CE__CI_conn, CE__CI_syn)
    #net.connect(CI, CE, CI__CE_conn, CI__CE_syn)
    # agonists and antagonists
    for pair in all_pairs:
        if pair in synergists:
            net.connect([CE[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_syne)
            net.connect([CE[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_aff)
            net.connect([CI[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_aff)
            net.connect([CI[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_lat)
        elif pair in part_syne:
            net.connect([CE[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_p_syne)
            net.connect([CE[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_aff)
            net.connect([CI[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_aff)
            net.connect([CI[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_lat)
        elif pair in antagonists:
            net.connect([CE[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_antag)
            net.connect([CE[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_lat)
            net.connect([CI[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_aff)
            net.connect([CI[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_lat)
        elif pair in part_antag:
            net.connect([CE[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_p_antag)
            net.connect([CE[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_lat)
            net.connect([CI[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_aff)
            net.connect([CI[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_lat)
        elif pair in self_conn:
            net.connect([CE[pair[0]]], [CI[pair[1]]], CE__CI_conn, CE__CI_syn)
            net.connect([CI[pair[0]]], [CE[pair[1]]], CI__CE_conn, CI__CE_syn)
        else:
            net.connect([CE[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_lat)
            net.connect([CE[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_aff)
            net.connect([CI[pair[0]]], [CE[pair[1]]], C__C_conn, C__C_syn_null_aff)
            net.connect([CI[pair[0]]], [CI[pair[1]]], C__C_conn, C__C_syn_null_lat)

    # SETTING UP WEIGHT TRACKING -- depends on the order of statements above!!!!!!
    # This assumes the first connections to C are M-->C
#     def M_CE0_fun(idx):
#         """ Creates a function to track a weight from M to CE0. """
#         return lambda t: net.syns[CE[0]][idx].w
#     for idx in range(len(M)):
#         net.units[M_CE_track[idx]].set_function(M_CE0_fun(idx))
#     # This assumes the first connections to M are AF-->M    
#     def AF_M0_fun(idx):
#         """ Creates a function to track a weight from AF to M0. """
#         return lambda t: net.syns[M[0]][idx].w
#     for idx, uid in enumerate(AF_M0_track):
#         net.units[uid].set_function(AF_M0_fun(idx))

    # TRACKING OF INSERTION POINTS (for the arm animation)
    # make the source units track the tensions
    def create_xtracker(arm_id, idx):
        return lambda t: net.plants[arm_id].ip[idx][0]
    def create_ytracker(arm_id, idx):
        return lambda t: net.plants[arm_id].ip[idx][1]
    for idx, uid in enumerate(ipx_track):
        net.units[uid].set_function(create_xtracker(P, idx))
    for idx, uid in enumerate(ipy_track):
        net.units[uid].set_function(create_ytracker(P, idx))
        
    pops_list = [SF, SP, SPF, AL, AF, SP_CHG, CE, CI, M, ACT, P, ipx_track, ipy_track]
    pops_names = ['SF', 'SP', 'SPF', 'AL', 'AF', 'SP_CHG', 'CE', 'CI', 
                  'M', 'ACT', 'P', 'ipx_track', 'ipy_track']
    if lowpass_SP:
        pops_list.append(LPF_SP)
        pops_names.append('LPF_SP')
    pops_dict = {pops_names[idx] : pops_list[idx] for idx in range(len(pops_names))}
    pds = {'Pe__AF_syn' : Pe__AF_syn, 'Pi__AF_syn' :Pi__AF_syn, 'SF_params' : SF_params, 
           'AFe__SF_syn' : AFe__SF_syn, 'AFi__SF_syn' : AFi__SF_syn}
    return net, pops_dict, hand_coords, m_idxs, t_pres, pds

