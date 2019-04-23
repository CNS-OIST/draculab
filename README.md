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
neural networks with delayed adaptive connections " Front. Neuroinform. 13,
https://doi.org/10.3389/fninf.2019.00018

The lessons of the tutorial are given as [Jupyter](https://jupyter.org/) notebooks with filenames `tutorialX.ipynb`, where X stands for the tutorial number. It is recommended to follow the lessons in order using the notebooks, but for those users who prefer not to use Jupyter, the Python source code and instructions are included in the `tutorialX.py` files.

The `tests` folder contains miscellaneous code to test various aspects of Draculab.
Some basic unit tests are in `unit_tests.py`. Most unit tests have an associated Jupyter
notebook named `testX.ipynb`, where `X` is some number. The `testX` notebooks run similar
tests, but these may include plots and animations in order to facilitate debugging.

Draculab contains a research-grade implementation of a planar arm, based on the
dynamics of the double pendulum. The `planar_arm.ipynb` file in `tests` contains code to
visualize the arm and its muscles for a given set of insertion points. There is
also code to run a simulation with the `planar_arm` class, and visualize the
results using an animation where the arm and the muscle activations can be observed.

The `double_pendulum_validation.ipynb` notebook in `tests`
shows how the double pendulum equations are derived using Sympy, and how
those equations can be validated by comparing them with other derivations. 
