"""
Author: Rohan Ramasamy
Date: 02/10/17

This file contains a simple class for modelling a charged particle for modelling single particles in an arbitrary field
"""

import numpy as np


class PICParticle(object):
    """
    A charge particle that acts as a struct to store the properties of the charged particle
    """
    def __init__(self, mass, charge, position, velocity):
        assert isinstance(mass, float), "mass must be a float"
        assert isinstance(charge, float), "charge must be a float"
        assert isinstance(velocity, np.ndarray) and velocity.shape[0] == 3 and len(velocity.shape) == 1, \
            "velocity must be a 3D vector"
        assert isinstance(position, np.ndarray) and position.shape[0] == 3 and len(position.shape) == 1, \
            "position must be a 3D vector"

        self._mass = mass
        self._charge = charge
        # Position and velocity need to have this shape so that they can be used in the boris solver without
        # reshaping on every iteration
        self.position = np.reshape(position, (1, 3))
        self.velocity = np.reshape(velocity, (1, 3))

    @property
    def mass(self):
        return self._mass

    @property
    def charge(self):
        return self._charge