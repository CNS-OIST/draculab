"""
spinal_plants.py
The plant models specially adapted for the spinal model.
"""
from draculab import unit_types, synapse_types, syn_reqs 
#from units.units import unit, sigmoidal
from plants.plants import plant, pendulum
import numpy as np

class bouncy_pendulum(pendulum):
    """ 
    A pendulum that bounces away when its angle approaches pi.

    This is a variation of the pendulum class with a modified derivatives
    function.

    PENDULUM DOCSTRING:
    Model of a rigid, homogeneous rod with a 1-dimensional rotational joint at one end.

    The simplest model of an arm is a rigid rod with one end attached to a rotational
    joint with 1 degree of freedom. Like a rigid pendulum, but the gravity vector is
    optional. 
    There is no extra mass at the end of the rod; the center of mass is at length/2 .
    
    On the XY plane the joint is at the origin, the positive X direction
    aligns with zero degrees, and gravity points in the negative Y direction.
    Counterclockwise rotation and torques are positive. Angles are in radians, time is
    in seconds.

    The equation of motion used is: I a = (mgL/2)cos(theta) - mu v + T
    where I=inertia moment about axis of rotation, a=angular acceleration, g=gravity constant,
    L=length of rod, theta=pendulum angle, mu=viscous friction coefficeint, m=mass of rod, 
    v=angular velocity, T=external torque.

    Inputs to the model at port 0 are torques applied at the joint. Other ports are ignored.

    The get_state(time) function returns the two state variables of the model in a numpy 
    array. Angle has index 0, and angular velocity has index 1. The 'time' argument
    should be in the interval [sim_time - del, sim_time], where sim_time is the current
    simulation time (time of last simulation step), and del is the 'delay' value
    of the plant. This method can be configured to provide angles in the 
    [-pi, pi) interval by using the 'bound_angle' parameter.

    Alternatively, the state variables can be retrieved with the get_angle(t) and
    get_ang_vel(t) functions.
    """

    def derivatives(self, y, t):
        """ Returns the derivatives of the state variables at a given point in time. 

	Args:
            y[0] = angle [radians]
            y[1] = angular velocity [radians / s]
            t = time at which the derivative is evaluated [s]
        Returns:
            2-element 1D Numpy array with angular velocity and acceleration.
        """
        torque = self.inp_gain * self.get_input_sum(t,0)
        # torque may also come from gravity and friction
        torque -= self.c * np.cos(y[0]) +  self.mu * y[1]
        # torque will also appear when angle is close to pi
        torque -= (np.tan((y[0]%(2.*np.pi))/2.)/4.)**3
        # angular acceleration = torque / (inertia moment)
        ang_accel = torque / self.I
        return np.array([y[1], ang_accel])

