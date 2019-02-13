# rate_neuron_dm_dracu.py
'''
rate_neuron decision making
------------------
A replication of rate_neuron_dm.py using draculab

'''
from draculab import *
import pylab
import numpy as np

'''
First, the Function build_network is defined to build the network and
return the handles of two decision units and the mutimeter
'''

def build_network(sigma, dt):
    net_params = {'min_delay': 1.,    # 1 ms
                  'min_buff_size': int(np.ceil(1./dt)) } 
    net = network(net_params)

    unit_params = {'type': unit_types.noisy_linear,
                   'init_val': 0.,
                   'tau': 1.,       # time constant in ms
                   'lambda': 0.1,    # passive decay rate
                   'mu': 0.,        # mean of Gaussian noise
                   'sigma': sigma}  # Std. Dev. of Gaussian noise
    D1 = net.create(1, unit_params)
    D2 = net.create(1, unit_params)

    conn_spec = {'rule': 'all_to_all',
                 'delay': 4.}
    syn_spec = {'type': synapse_types.static,
                'init_w': -.2}
    net.connect(D1, D2, conn_spec, syn_spec)
    net.connect(D2, D1, conn_spec, syn_spec)

    return net, D1[0], D2[0]


'''
The function build_network takes the standard deviation of Gaussian
white noise and the time resolution as arguments.
'''

'''
The decision making process is simulated for three different levels of noise
and three differences in evidence for a given decision. The activity of both
decision units is plotted for each scenario.
'''

fig_size = [14, 8]
fig_rows = 3
fig_cols = 3
fig_plots = fig_rows * fig_cols
face = 'white'
edge = 'white'

ax = [None] * fig_plots
fig = pylab.figure(facecolor=face, edgecolor=edge, figsize=fig_size)

dt = 1e-3
print('dt = ' + str(dt))
sigma = [1.*x for x in [0.0, 0.1, 0.2]]
dE = [1.*x for x in [0.0, 0.004, 0.008]]
noiseless_activs = [[], [], []]
for i in range(9):

    c = i % 3
    r = int(i / 3)
    net, D1, D2 = build_network(sigma[r], dt)

    times, activs, _ = net.run(100.)
    net.units[D1].mu =  1. + dE[c]
    net.units[D2].mu =  1. - dE[c]
    times2, activs2, _ = net.run(100.)
    times = np.append(times, times2)
    activs = np.concatenate((activs, activs2), axis=1)

    if sigma[r] == 0.:
        noiseless_activs[c] = activs

    ax[i] = fig.add_subplot(fig_rows, fig_cols, i + 1)
    ax[i].plot(times, activs[D1], 'b', linewidth=2, label="D1")
    ax[i].plot(times, activs[D2], 'r', linewidth=2, label="D2")
    ax[i].set_ylim([-.5, 12.])
    ax[i].get_xaxis().set_ticks([])
    ax[i].get_yaxis().set_ticks([])
    if c == 0:
        ax[i].set_ylabel("activity ($\sigma=%.1f$) " % (sigma[r]))
        ax[i].get_yaxis().set_ticks([0, 3, 6, 9, 12])

    if r == 0:
        ax[i].set_title("$\Delta E=%.3f$ " % (dE[c]))
        if c == 2:
            pylab.legend(loc=0)
    if r == 2:
        ax[i].get_xaxis().set_ticks([0, 50, 100, 150, 200])
        ax[i].set_xlabel('time (ms)')

    '''
    The activity of the two units is plottedin each scenario.

    In the absence of noise, the network will not make a decision if evidence
    for both choices is equal. With noise, this symmetry can be broken and a
    decision wil be taken despite identical evidence.

    As evidence for D1 relative to D2 increases, it becomes more likely that
    the corresponding decision will be taken. For small differences in the
    evidence for the two decisions, noise can lead to the 'wrong' decision.

    '''

pylab.show()
