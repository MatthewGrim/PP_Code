"""
Author: Rohan Ramasamy
Date: 04/06/2017

This script contains the circuit modules made use of in the coupled coaxial liner simulation
"""

import numpy as np

from Physical_Models.ryutov_model.coax.coupled_coax.circuit_modules.base_circuit import BaseCircuit


class FCCircuit(BaseCircuit):
    """
    FC circuit in which a fast switch is used to compress a capacitor discharge pulse
    """
    def __init__(self, times, **kwargs):
        """
        Constructor for FC circuit
        :return:
        """
        super(FCCircuit, self).__init__(times)

        V = kwargs.pop("V")
        self.C = kwargs.pop("C")
        self.R_circ = kwargs.pop("R_circ")
        self.L_circ = kwargs.pop("L_circ")

        self.v_gen = np.zeros(times.shape)
        self.q_gen = np.zeros(times.shape)
        self.I_1 = np.zeros(times.shape)
        self.I_1_dot = np.zeros(times.shape)
        self.R_variable = np.zeros(times.shape)

        self.v_gen[0] = V
        self.q_gen[0] = self.C * V

        # Switch variables
        self.switched = False
        self.resistivity_increase = kwargs.get("resistivity_increase", 1e6)
        self.I_2 = np.zeros(times.shape)
        self.I_2_dot = np.zeros(times.shape)
        self.v_load = np.zeros(times.shape)

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
            return self.I_2[ts]

        if self.switched:
            # Calculate new currents
            self.I_1[ts + 1] = self.I_1[ts] + self.I_1_dot[ts] * self.dt
            self.I_2[ts + 1] = self.I_2[ts] + self.I_2_dot[ts] * self.dt

            # Calculate variable resistance
            self.R_variable[ts + 1] = self.R_variable[ts] + self.resistivity_increase * self.dt

            # Calculate change in load current and load voltage
            self.I_2_dot[ts + 1] = (self.I_1[ts + 1] * self.R_variable[ts + 1] - (self.R_variable[ts + 1] + R_load) * self.I_2[ts + 1]) / L_load
            self.v_load[ts + 1] = self.I_2[ts + 1] * R_load + self.I_2_dot[ts + 1] * L_load

            #  Calculate generator charge and voltage
            self.q_gen[ts + 1] = self.q_gen[ts] - self.I_1[ts + 1] * self.dt
            self.v_gen[ts + 1] = self.q_gen[ts + 1] / self.C

            # Calculate change in total current
            self.I_1_dot[ts + 1] = (self.v_gen[ts + 1] - self.v_load[ts + 1]) / self.L_circ
        else:
            # Calculate change in total current
            self.I_1[ts + 1] = self.I_1[ts] + self.I_1_dot[ts] * self.dt

            #  Calculate generator charge and voltage
            self.q_gen[ts + 1] = self.q_gen[ts] - self.I_1[ts + 1] * self.dt
            self.v_gen[ts + 1] = self.q_gen[ts + 1] / self.C

            # Calculate change in total current
            self.I_1_dot[ts + 1] = (self.v_gen[ts + 1] - self.I_1[ts + 1] * (self.R_circ + self.R_variable[ts + 1])) / self.L_circ

            # Check if peak current is reached and switch is flipped
            if self.I_1_dot[ts + 1] < 0.0 <= self.I_1_dot[ts] and not self.switched:
                self.switched = True

        return self.I_2[ts]

    def results(self):
        """
        Return the output results of the simulation
        """
        return self.v_gen, self.v_load, self.q_gen, self.I_1, self.I_1_dot, self.I_2, self.I_2_dot


