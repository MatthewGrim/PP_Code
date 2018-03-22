"""
Author: Rohan Ramasamy
Date: 22/03/2018

This script contains the equation of state (EOS) modules for the coupled coaxial
liner simulation
"""

import numpy as np


class BaseEOS(object):

    def __init__(self, times):
        """
        Constructor for base EOS class
        """
        self.pressure = np.zeros(times.shape)
        self.densities = np.zeros(times.shape)
        self.temperatures = np.zeros(times.shape)

    def get_timestep_result(self, i):
        """
        Function to return module parameters needed in main coaxial liner loop
        """
        raise NotImplementedError()

    def results(self):
        """
        Return the result arrays from the simulation
        """
        raise NotImplementedError()
