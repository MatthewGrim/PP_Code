"""
Author: Rohan Ramasamy
Date: 04/06/2017

This script contains the circuit modules made use of in the coupled coaxial liner simulation
"""

import numpy as np


class BaseCircuit(object):
    def __init__(self, times):
        """
        Base constructor for circuit module type
        :return:
        """
        self.I = np.zeros(times.shape)
        self.dt = times[1] - times[0]
        self.number_of_ts = times.shape[0] - 1
        self.current_ts = 0

    def evolve_timestep(self, ts, t, R, L, L_dot, **kwargs):
        """
        Function to evolve circuit by a single time step
        :return:
        """
        raise NotImplementedError()

    def results(self):
        """
        Return the output results of the simulation
        """
        raise NotImplementedError()

