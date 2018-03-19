"""
Author: Rohan Ramasamy
Date: 04/06/2017

This script contains the circuit modules made use of in the coupled coaxial liner simulation
"""

import numpy as np

from Physical_Models.ryutov_model.coax.coupled_coax.circuit_modules.base_circuit import BaseCircuit


class RLCircuit(BaseCircuit):
    """
    RL circuit in which a capacitor discharges into the circuit
    """
    def __init__(self, times, **kwargs):
        """
        Constructor for RL circuit
        :return:
        """
        super(RLCircuit, self).__init__(times)

        # Get input parameters
        V = kwargs.pop("V")
        self.C = kwargs.pop("C")
        self.L_circ = kwargs.pop("L_circ")

        # Set up result arrays
        self.v_gen = np.zeros(times.shape)
        self.q_gen = np.zeros(times.shape)
        self.I_dot = np.zeros(times.shape)

        # Set up initial conditions
        self.v_gen[0] = V
        self.q_gen[0] = self.C * V

    def evolve_timestep(self, ts, t, R_load, L_load, L_dot_load, **kwargs):
        """
        Function to evolve simulation by a single timestep
        :return:
        :param ts: time step
        :param t: time
        :param R: Load resistance
        :param L: Load inductance
        :param L_dot: Load inductance variation
        :return:
        """
        assert self.current_ts == ts
        self.current_ts = ts + 1

        if ts == self.number_of_ts:
            return self.I[ts]

        self.I[ts + 1] = self.I[ts] + self.I_dot[ts] * self.dt
        self.q_gen[ts + 1] = self.q_gen[ts] - self.I[ts + 1] * self.dt
        self.v_gen[ts + 1] = self.q_gen[ts + 1] / self.C
        self.I_dot[ts + 1] = (self.v_gen[ts + 1] - self.I[ts + 1] * (R_load + L_dot_load)) / (self.L_circ + L_load)

        return self.I[ts]

    def results(self):
        """
        Return the output results of the simulation
        """
        return self.v_gen, self.q_gen, self.I, self.I_dot