"""
Author: Rohan Ramasamy
Date: 22/03/2018

This script contains the vacuum equation of state (EOS) for the coupled coaxial
liner simulation
"""

import numpy as np

from plasma_physics.pysrc.simulation.ryutov.coupled_coax.eos_modules.base_eos import BaseEOS


class VacuumEOS(BaseEOS):
    """
    Simple equation of state module that assumes no internal fuel. As a result, the
    feedback pressure is always 0
    """

    def __init__(self, times, **kwargs):
        """
        Constructor for vacuum EOS. This model should work in 1D and 2D

        :param times: array of time points for the simulation
        """
        super(VacuumEOS, self).__init__(times)

        liner_shape = kwargs.pop("liner_shape")
        if isinstance(liner_shape, tuple):
            self.p_feedback = np.zeros(liner_shape)
        elif liner_shape == 1:
            self.p_feedback = 0.0
        else:
            raise ValueError("liner shape must correspond to liner array size")

    def evolve_timestep(self, ts, r, v):
        """
        Function to evolve simulation by a single timestep
        """
        self.get_timestep_results(ts)

    def get_timestep_results(self, i):
        """
        Return the results of the timestep
        """
        return self.p_feedback
