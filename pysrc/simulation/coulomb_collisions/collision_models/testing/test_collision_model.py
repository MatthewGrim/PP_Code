"""
Author: Rohan Ramasamy
Date: 13/03/2018

This code tests the functionality of different coulomb collision models.
"""

import unittest
import numpy as np

from plasma_physics.pysrc.simulation.coulomb_collisions.collision_models.abe_collison_model import AbeCoulombCollisionModel


class AbeCollisionModelTest(unittest.TestCase):
    def test_randomise_velocities(self):
        shape = (4, 3)
        vel = np.arange(12).reshape(shape)
        vel_copy = np.copy(vel)

        AbeCoulombCollisionModel.randomise_velocities(vel)
        assert vel.shape == shape
        assert not np.array_equal(vel_copy, vel)

        for i in range(vel.shape[0]):
            for j in range(vel.shape[1]):
                assert vel[i, j] == vel[i, 0] + j


if __name__ == '__main__':
    unittest.main()
