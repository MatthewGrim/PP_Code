"""
Author: Rohan Ramasamy
Date: 04/06/2017

This script contains the liner modules made use of in the coupled coaxial liner simulation
"""

import numpy as np
import scipy.integrate as integrate

from plasma_physics.pysrc.simulation.ryutov.coupled_coax.liner_modules.base_liner import BaseLiner


class CoaxialLiner2D(BaseLiner):
    """
    2D Liner module that considers a specified length of liner. The radius of the liner is allowed to vary along the
    length
    """

    def __init__(self, times, **kwargs):
        """
        Constructor for 2D coaxial liner module

        :return:
        """
        super(CoaxialLiner2D, self).__init__(times)

        # Get liner geometry
        h = kwargs.pop("h")
        self.h = h
        r_inner = kwargs.pop("r_inner")
        R_Inner = kwargs.pop("R_Inner")
        r_outer = kwargs.get("r_outer", None)
        R_Outer = kwargs.pop("R_Outer", None)
        r_outer = 1.1 * r_inner if r_outer is None else r_outer
        R_Outer = 1.1 * R_Inner if R_Outer is None else R_Outer
        assert np.all(r_inner < r_outer)
        assert np.all(r_outer < R_Inner)
        assert np.all(R_Inner < R_Outer)

        # Get minimum radius of simulation
        convergence_ratio = kwargs.get("minimum_radius", 0.1)
        self.minimum_radius = convergence_ratio * r_inner

        # Get shell densities - kgm-1
        self.liner_density = kwargs.get("liner_density", 2700.0)
        self.rho_inner = self.liner_density * np.pi * (r_outer ** 2 - r_inner ** 2)
        self.Rho_Outer = self.liner_density * np.pi * (R_Outer ** 2 - R_Inner ** 2)

        print("Inner Liner mass: {}".format(integrate.simps(self.rho_inner, self.h)))
        print("Outer Liner mass: {}".format(integrate.simps(self.Rho_Outer, self.h)))

        # Define constants for magnetic pressure and inductance
        self.p_const_inner = self.mu_0 / (4.0 * np.pi * self.rho_inner)
        self.p_const_outer = self.mu_0 / (4.0 * np.pi * self.Rho_Outer)
        self.l_const = self.mu_0 / (2.0 * np.pi)

        # Set up result arrays
        self.r_i = np.zeros((times.shape[0], r_inner.shape[0]))
        self.r_o = np.zeros((times.shape[0], r_inner.shape[0]))
        self.v = np.zeros((times.shape[0], r_inner.shape[0]))
        self.e_kin = np.zeros((times.shape[0], r_inner.shape[0]))
        self.R_I = np.zeros((times.shape[0], r_inner.shape[0]))
        self.V = np.zeros((times.shape[0], r_inner.shape[0]))
        self.E_kin = np.zeros((times.shape[0], r_inner.shape[0]))
        self.l = np.zeros((times.shape[0], r_inner.shape[0]))
        self.r_i[0, :] = r_inner
        self.r_i[1, :] = r_inner
        self.r_o[0, :] = r_outer
        self.r_o[1, :] = r_outer
        self.R_I[0, :] = R_Inner
        self.R_I[1, :] = R_Inner

    def evolve_timestep(self, ts, I, p_feedback):
        """
            Function to evolve simulation by a single timestep
            """
        assert self.current_ts == ts
        self.current_ts = ts + 1

        # Calculate outer radius of inner liner and inductance
        self.l[ts, :] = self.l_const * np.log(self.R_I[ts, :] / self.r_o[ts, :])
        self.L[ts] = integrate.simps(self.l[ts, :], self.h)
        self.L_dot[ts] = (self.L[ts] - self.L[ts - 1]) / self.dt if ts != 0 else 0.0

        # Skip first time step F = ma calculation
        if ts == 0:
            return self.get_timestep_result(ts)

        # On last time step, use backward difference to set results
        if ts == self.number_of_ts:
            self.v[ts, :] = (self.r_o[ts, :] - self.r_o[ts - 1, :]) / self.dt
            self.V[ts, :] = (self.R_I[ts, :] - self.R_I[ts - 1, :]) / self.dt
            self.e_kin[ts, :] = 0.5 * self.rho_inner * self.v[ts, :] ** 2
            self.E_kin[ts, :] = 0.5 * self.Rho_Outer * self.V[ts, :] ** 2
            return self.get_timestep_result(ts)

        # Inner Liner motion
        self.r_o[ts + 1, :] = 2 * self.r_o[ts, :] - self.r_o[ts - 1, :] - (self.dt ** 2 * (self.p_const_inner * I ** 2 - p_feedback)) / self.r_o[ts, :]
        self.v[ts, :] = (self.r_o[ts + 1, :] - self.r_o[ts - 1, :]) / (2 * self.dt)
        self.e_kin[ts, :] = 0.5 * self.rho_inner * self.v[ts, :] ** 2

        # Outer Liner motion
        self.R_I[ts + 1, :] = 2 * self.R_I[ts, :] - self.R_I[ts - 1, :] + (self.dt ** 2 * self.p_const_outer * I ** 2) / self.R_I[ts, :]
        self.V[ts, :] = (self.R_I[ts + 1, :] - self.R_I[ts - 1, :]) / (2 * self.dt)
        self.E_kin[ts, :] = 0.5 * self.Rho_Outer * self.V[ts, :] ** 2

        # If any point surpasses the minimum radius, end the simulation
        self.r_i[ts + 1, :] = np.sqrt(self.r_o[ts + 1, :] ** 2 - self.rho_inner / (self.liner_density * np.pi))
        if np.any(self.r_i[ts + 1, :] <= self.minimum_radius):
            self.r_i[ts + 1, :] = self.minimum_radius
            self.r_o[ts + 1, :] = self.r_o[ts, :]
            self.v[ts + 1, :] = self.v[ts - 1, :]
            self.e_kin[ts + 1, :] = self.e_kin[ts - 1, :]
            self.final_time = ts + 1
            return self.get_timestep_result(ts)
        else:
            return self.get_timestep_result(ts)

    def get_timestep_result(self, i):
        """
        Function to return module parameters needed in main coaxial liner loop
        :param i:
        :return:
        """
        if i == self.number_of_ts or np.all(self.r_i[i + 1] > self.minimum_radius):
            self.final_time = i
            return self.R[i], self.L[i], self.L_dot[i], self.r_i[i, :], self.v[i, :], False
        else:
            self.final_time = i
            return self.R[i], self.L[i], self.L_dot[i], self.r_i[i, :], self.v[i, :], True

    def results(self):
        """
        Return the result arrays from the simulation
        :return:
        """
        return self.r_i, self.r_o, self.v, self.e_kin, self.R_I, self.V, self.E_kin, self.l, self.L, self.L_dot, self.final_time
