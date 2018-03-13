"""
Author: Rohan Ramasamy
Date: 28/10/2017

This file contains testing for analytic B field generators
"""

import os
import unittest
import numpy as np

from plasma_physics.pysrc.simulation.pic.algo.fields.magnetic_fields.generic_b_fields import CurrentLoop, InterpolatedBField
from plasma_physics.pysrc.simulation.pic.algo.geometry.vector_ops import magnitude


class CurrentLoopTest(unittest.TestCase):
    def test_b_field(self):
        """
        Function to test B field along axis is as expected from analytic solution
        :return:
        """
        for offset in [1.0, 0.0, -1.0]:
            I = 1e6
            radius = 0.15
            loop_pts = 500
            loop = CurrentLoop(I, radius, np.asarray([0.0, 0.0, offset]), np.asarray([0.0, 0.0, 1.0]), loop_pts)

            Z = np.linspace(-5.0, 5.0, 50)

            for i, z in enumerate(Z):
                analytic_z_field = CurrentLoop.mu_0 / 2 * I * radius ** 2 / ((radius ** 2 + (z + offset) ** 2) ** (3.0 / 2.0))

                b = loop.b_field(np.asarray([0.0, 0.0, z]))
                B = magnitude(b)

                self.assertAlmostEqual(analytic_z_field, B, 5)

    def test_interpolated_b_field(self):
        """
        Function to test interpolated B field behaviour is as expected
        :return:
        """
        # Generate Interpolated field
        I = 1e6
        loop_pts = 20
        domain_pts = 50
        dom_size = 0.2
        radius = 0.15
        file_name = "current_loop_{}_{}_{}_{}".format(I * 1e-6, loop_pts, domain_pts, dom_size)
        file_path = os.path.join(file_name)
        interp_field = InterpolatedBField(file_path)

        Z = np.linspace(-5.0, 5.0, 50)
        for i, z in enumerate(Z):
            analytic_z_field = CurrentLoop.mu_0 / 2 * I * radius ** 2 / ((radius ** 2 + z ** 2) ** (3.0 / 2.0))

            b = interp_field.b_field(np.asarray([0.0, 0.0, z]))
            B = magnitude(b)

            self.assertAlmostEqual(analytic_z_field, B, 4)


if __name__ == '__main__':
    unittest.main()

