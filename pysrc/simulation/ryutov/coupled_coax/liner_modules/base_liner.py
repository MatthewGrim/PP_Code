"""
Author: Rohan Ramasamy
Date: 04/06/2017

This script contains the liner modules made use of in the coupled coaxial liner simulation
"""


import numpy as np
import scipy.integrate as integrate


class BaseLiner(object):
    # Class variable for the permeability of free space - Inner space is assumed to be a vacuum
    mu_0 = 4.0 * np.pi * 1e-7

    def __init__(self, times):
        """
        Constructor for base liner class
        :return:
        """
        self.R = np.zeros(times.shape)
        self.R_dot = np.zeros(times.shape)
        self.L = np.zeros(times.shape)
        self.L_dot = np.zeros(times.shape)

        self.dt = times[1] - times[0]
        self.number_of_ts = times.shape[0] - 1
        self.final_time = None

        self.r_i = None
        self.r_o = None
        self.v = None
        self.e_kin = None
        self.R_I = None
        self.V = None
        self.E_kin = None

        # Keep a record of current time step
        self.current_ts = 0

    def get_timestep_result(self, i):
        """
        Function to return module parameters needed in main coaxial liner loop
        :param i:
        :return:
        """
        raise NotImplementedError()

    def results(self):
        """
        Return the result arrays from the simulation
        :return:
        """
        raise NotImplementedError()
