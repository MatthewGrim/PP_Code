"""
Author: Rohan Ramasamy
Date: 04/06/2017

This script contains the liner modules made use of in the coupled coaxial liner simulation
"""

import numpy as np

from plasma_physics.pysrc.simulation.ryutov.coupled_coax.liner_modules.base_liner import BaseLiner


class CoaxialLiner1D(BaseLiner):
    """
    Simple liner module, which assumes that the length of the liner is 1m. As a result the inductance per metre, and
    density per metre are valid values for the inductance and mass
    """
    def __init__(self, times, **kwargs):
        """
        Constructor for 1D coaxial liner module

        :param times: array of time points for the simulation
        """
        super(CoaxialLiner1D, self).__init__(times)

        # Set up liner geometry
        self.h = kwargs.get("h", 1.0)
        r_inner = kwargs.pop("r_inner")
        R_Inner = kwargs.pop("R_Inner")
        r_outer = kwargs.get("r_outer", None)
        R_Outer = kwargs.get("R_Outer", None)
        assert isinstance(r_inner, float)
        assert isinstance(R_Inner, float)

        # Set default values of outer radii, and check for consistency
        r_outer = 1.1 * r_inner if r_outer is None else r_outer
        R_Outer = 1.1 * R_Inner if R_Outer is None else R_Outer
        assert r_inner < r_outer < R_Inner < R_Outer

        # Get minimum radius of simulation
        convergence_ratio = kwargs.get("minimum_radius", 0.1)
        self.minimum_radius = convergence_ratio * r_inner

        # Get shell densities - kgm-1
        self.liner_resistivity = kwargs.get("liner_resistivity", 25e-9)
        self.liner_density = kwargs.get("liner_density", 2700.0)
        self.m_inner = self.liner_density * np.pi * (r_outer ** 2 - r_inner ** 2) * self.h
        self.m_outer = self.liner_density * np.pi * (R_Outer ** 2 - R_Inner ** 2) * self.h

        # Print liner geometries and densities
        if kwargs.pop("print_geometry", False):
            print("Inner Liner thickness: {}".format(r_outer - r_inner))
            print("Inner Liner mass: {}".format(self.m_inner))
            print("Outer Liner thickness: {}".format(R_Outer - R_Inner))
            print("Outer Liner mass: {}".format(self.m_outer))

        # Define constants for magnetic pressure and inductance
        self.p_const_inner = self.mu_0 / (4.0 * np.pi * self.m_inner)
        self.p_const_outer = self.mu_0 / (4.0 * np.pi * self.m_outer)
        self.l_const = self.mu_0 / (2.0 * np.pi)

        # Set up result arrays
        self.r_i = np.zeros(times.shape)
        self.r_o = np.zeros(times.shape)
        self.v = np.zeros(times.shape)
        self.e_kin = np.zeros(times.shape)
        self.R_I = np.zeros(times.shape)
        self.R_O = np.zeros(times.shape)
        self.A = np.zeros(times.shape)
        self.V = np.zeros(times.shape)
        self.E_kin = np.zeros(times.shape)
        self.r_i[0] = r_inner
        self.r_i[1] = r_inner
        self.r_o[0] = r_outer
        self.r_o[1] = r_outer
        self.R_I[0] = R_Inner
        self.R_I[1] = R_Inner
        self.R_O[0] = R_Outer
        self.R_O[1] = R_Outer

    def evolve_timestep(self, i, I):
        """
        Function to evolve simulation by a single time step
        :return:
        """
        assert self.current_ts == i
        self.current_ts = i + 1

        # Calculate current inductance Lm-1
        self.A[i] = np.pi * (self.R_O[i] ** 2 + self.r_o[i] ** 2 - self.r_i[i] ** 2 - self.R_I[i] ** 2)
        self.R[i] = self.liner_resistivity * self.h / self.A[i]
        self.L[i] = self.l_const * np.log(self.R_I[i] / self.r_o[i]) * self.h
        self.L_dot[i] = (self.L[i] - self.L[i - 1]) / self.dt if i != 0 else 0.0

        # Skip first time step F = ma calculation
        if i == 0:
            return self.get_timestep_result(i)

        # On last time step, use backward difference to set results
        if i == self.number_of_ts:
            self.v[i] = (self.r_o[i] - self.r_o[i - 1]) / self.dt
            self.V[i] = (self.R_I[i] - self.R_I[i - 1]) / self.dt
            self.e_kin[i] = 0.5 * self.m_inner * self.v[i] ** 2
            self.E_kin[i] = 0.5 * self.m_outer * self.V[i] ** 2
            return self.get_timestep_result(i)

        # Inner liner motion
        self.r_o[i + 1] = 2 * self.r_o[i] - self.r_o[i - 1] - (self.dt ** 2 * self.p_const_inner * I ** 2) / self.r_o[i]
        self.v[i] = (self.r_o[i + 1] - self.r_o[i - 1]) / (2 * self.dt)
        self.e_kin[i] = 0.5 * self.m_inner * self.v[i] ** 2

        # Outer liner motion
        self.R_I[i + 1] = 2 * self.R_I[i] - self.R_I[i - 1] + (self.dt ** 2 * self.p_const_outer * I ** 2) / self.R_I[i]
        self.V[i] = (self.R_I[i + 1] - self.R_I[i - 1]) / (2 * self.dt)
        self.E_kin[i] = 0.5 * self.m_outer * self.V[i] ** 2

        # Integrate up parameters from acceleration to radii. This does not seem to improve accuracy so I've stuck to
        # using the second differential
        # self.v[i] = self.v[i - 1] - (self.dt * self.p_const_inner * I ** 2) / self.r_o[i - 1]
        # self.r_o[i] = self.r_o[i - 1] + self.dt * self.v[i]
        # self.e_kin[i] = 0.5 * self.m_inner * self.v[i] ** 2
        #
        # self.V[i] = self.V[i - 1] + (self.dt * self.p_const_outer * I ** 2) / self.R_I[i - 1]
        # self.R_I[i] = self.R_I[i - 1] + (self.V[i] * self.dt)
        # self.E_kin[i] = 0.5 * self.m_outer * self.V[i] ** 2

        # Get other radii and check they are still geometrically consistent
        self.r_i[i + 1] = np.sqrt(self.r_o[i + 1] ** 2 - self.m_inner / (self.liner_density * np.pi * self.h))
        self.R_O[i + 1] = np.sqrt(self.m_outer / (self.liner_density * np.pi * self.h) + self.R_I[i + 1] ** 2)
        assert self.r_i[i + 1] < self.r_o[i + 1] < self.R_I[i + 1] < self.R_O[i + 1]

        # If the convergence ratio is surpassed, end the simulation
        if self.r_i[i] <= self.minimum_radius:
            self.r_i[i] = self.minimum_radius
            self.r_o[i] = self.r_o[i]
            self.v[i] = self.v[i - 1]
            self.e_kin[i] = self.e_kin[i - 1]
            return self.get_timestep_result(i)
        else:
            return self.get_timestep_result(i)

    def get_timestep_result(self, i):
        """
        Function to return module parameters needed in main coaxial liner loop
        :param i:
        :return:
        """
        if i == self.number_of_ts or self.r_i[i] > self.minimum_radius:
            return self.R[i], self.L[i], self.L_dot[i], False
        else:
            return self.R[i], self.L[i], self.L_dot[i], True

    def results(self):
        """
        Return the result arrays from the simulation
        :return:
        """
        return self.r_i, self.r_o, self.v, self.e_kin, self.R_I, self.V, self.E_kin, self.L, self.L_dot, self.R