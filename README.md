![alt text](https://i.ibb.co/qJs7NZ7/drcul.png)

`draculab.py` is a Python module containing classes to simulate
networks of rate units with delayed connections.
Draculab is an acronym of Delayed-Rate Adaptively-Connected Units Laboratory. 

The original purpose behind this simulator was to create neural networks
that autonomously learn to control a simulated physical plant.
Because of this, rate units and their synapses are dynamical
systems operating in continuous time. Moreover, since 
instantaneous connections are unrealistic, there is a minimum
connection delay, and all connection delays in the network are
multiples of it.

The simulator has a relatively simple architecture, so that
experienced Python users can understand its source code and make any 
required adjustments.

See the INSTALL.md file for installation instructions.

A fast way to get started is to follow the tutorials in the `tutorial` folder.
Users who want a more systematic introduction can begin by reading the
accompanying paper, and then following the tutorials.

The paper should be in the `tutorial` folder, in the `draculab_manuscript.pdf` file.
Its reference is:
Verduzco-Flores SO, DeSchutter E (2019) "Draculab: A Python simulator for firing rate
neural networks with delayed adaptive connections " (under review)

The lessons of this tutorial are given as [Jupyter](https://jupyter.org/) notebooks with filenames `tutorialX.ipynb`, where X stands for the tutorial number. It is recommended to follow the lessons in order using the notebooks, but for those users who prefer not to use Jupyter, the Python source code and instructions are included in the `tutorialX.py` files.
