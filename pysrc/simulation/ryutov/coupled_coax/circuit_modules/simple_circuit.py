"""
Author: Rohan Ramasamy
Date: 04/06/2017

This script contains the circuit modules made use of in the coupled coaxial liner simulation
"""

import numpy as np

from plasma_physics.pysrc.simulation.ryutov.coupled_coax.circuit_modules.base_circuit import BaseCircuit


class SimpleCircuit(BaseCircuit):
    """
    Simple sinusoidal current pulse with no back emf incorporated
    """
    I_0 = 2.5e7
    tau = 1e-4

    def __init__(self, times):
        """
        Constructor for simple circuit model
        """
        super(SimpleCircuit, self).__init__(times)

    def evolve_timestep(self, ts, t, R, L, L_dot, **kwargs):
        """
        Evolves simulation by a single timestep
        :param ts: time step
        :param t: time
        :param R: Load resistance
        :param L: Load inductance
        :param L_dot: Load inductance variation
        :return:
        """
        assert self.current_ts == ts
        self.current_ts = ts + 1

        self.I[ts] = self.I_0 * (np.sin(np.pi * t / self.tau)) ** 2

        return self.I[ts]

    def results(self):
        """
        Return the output results of the simulation
        """
        return self.I
