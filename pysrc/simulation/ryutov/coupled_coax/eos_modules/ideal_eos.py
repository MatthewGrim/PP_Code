"""
Author: Rohan Ramasamy
Date: 22/03/2018

This file contains the ideal equation of state (EOS) for the coupled coaxial
liner simulation
"""

import numpy as np

from plasma_physics.pysrc.simulation.ryutov.coupled_coax.eos_modules.base_eos import BaseEOS


class IdealEOS1D(BaseEOS):
    """
    Ideal gas equation of state, which models the fuel as an ideal gas using
    the shock and isentropic compression relationships to define the feedback
    pressure
    """

    def __init__(self, times, **kwargs):
        """
        Constructor for Ideal EOS. This model only works in 1D

        :param times:
        :param kwargs:
        :return:
        """
        super(IdealEOS1D, self).__init__(times)

        self.r_0 = kwargs.pop("r_inner")
        self.p_0 = kwargs.pop("p_0")
        self.rho_0 = kwargs.pop("rho_0")
        self.gamma = kwargs.get("gamma", 5.0 / 3.0)
        self.M = kwargs.pop("molecular_mass")

        self.p = np.zeros(times.shape)
        self.rho = np.zeros(times.shape)

        self.p[0] = self.p_0
        self.rho[0] = self.rho_0

    def evolve_timestep(self, ts, r, v):
        """
        Function to evolve model by single timestep

        :param ts:
        :param r:
        :param v:
        :return:
        """
        rho_prev = self.rho_0 if ts == 0 else self.rho[ts - 1]
        p_shock = (self.gamma + 1) / 2.0 * rho_prev * v ** 2
        p_isentropic = self.p_0 * (self.r_0 / r) ** (2 * self.gamma)

        # if p_shock > p_isentropic:
        #     self.rho[ts] = (self.gamma + 1) / (self.gamma - 1) * rho_prev
        #     self.p[ts] = p_shock
        # else:
        self.rho[ts] = self.rho_0 * (self.r_0 / r) ** 2
        self.p[ts] = p_isentropic

        return self.get_timestep_results(ts)

    def get_timestep_results(self, ts):
        """
        Return the results of the timestep
        :return:
        """
        return self.p[ts]

    def results(self):
        """
        Return the result arrays from the simulation
        :return:
        """
        return self.rho, self.p
