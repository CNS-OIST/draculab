Draculab.py is a Python module containing classes to simulate
networks of rate units with delayed connections.

The original purpose behind this simulator was to create neural networks
that autonomously learn to control a simulated physical plant.
Because of this, rate units and their synapses are dynamical
systems operating in continuous time. Moreover, since 
instantaneous connections are unrealistic, there is a minimum
connection delay, and all connection delays in the network are
multiples of it.

The simulator is has a relatively simple architecture, so that
experienced Python users can understand its source code and make any 
required adjustments.

See the INSTALLATION file for installation instructions.

A way to get started is to follow the tutorials in the
'tutorial' folder.
