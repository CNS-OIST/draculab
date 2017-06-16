sirasy.py is a Python module containing classes to simulate
networks of rate units with delayed connections.

The purpose behind this simulator is to create neural networks
that autonomously learn to control a simulated physical plant.
Because of this, rate units and their synapses are dynamical
systems operating in continuous time. Moreover, since 
instantaneous connections are unrealistic, there is a minimum
connection delay, and all connection delays in the network 
multiples of it.

The simulator is expected to be simple and compact enough that 
any experienced Python user can understand its source code
and make any required adjustments.

A basic usage example is in the Jupyter notebook test2.ipynb .
