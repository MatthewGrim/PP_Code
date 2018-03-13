"""
Author: Rohan
Date: 27/03/17

Tests for geometric vector operations
"""

import unittest

from plasma_physics.pysrc.simulation.pic.algo.geometry.vector_ops import *


class VectorOpsTest(unittest.TestCase):
    def test_magnitude(self):
        self.assertEqual(0.0, magnitude(np.zeros(3)))
        self.assertEqual(np.sqrt(3.0), magnitude(np.ones(3)))
        self.assertEqual(3.0, magnitude(np.asarray([1.0, 2.0, -2.0])))

    def test_dot(self):
        # Zero vectors
        self.assertEqual(0.0, dot(np.zeros(3), np.ones(3)))
        self.assertEqual(0.0, dot(np.ones(3), np.zeros(3)))

        # Parallel vectors
        self.assertEqual(3.0, dot(np.asarray([0.0, 0.0, 3.0]), np.asarray([0.0, 0.0, 1.0])))
        self.assertEqual(-3.0, dot(np.asarray([0.0, 0.0, -3.0]), np.asarray([0.0, 0.0, 1.0])))
        self.assertEqual(-3.0, dot(np.asarray([0.0, 0.0, 3.0]), np.asarray([0.0, 0.0, -1.0])))
        self.assertEqual(3.0, dot(np.asarray([0.0, 0.0, -3.0]), np.asarray([0.0, 0.0, -1.0])))

        # Orthogonal vectors
        self.assertEqual(0.0, dot(np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 0.1, 0.0])))
        self.assertEqual(0.0, dot(np.asarray([-1.0, 0.0, 0.0]), np.asarray([0.0, 0.1, 0.0])))
        self.assertEqual(0.0, dot(np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, -0.1, 0.0])))
        self.assertEqual(0.0, dot(np.asarray([-1.0, 0.0, 0.0]), np.asarray([0.0, -0.1, 0.0])))
        self.assertEqual(0.0, dot(np.asarray([0.0, 0.1, 0.0]), np.asarray([0.0, 0.0, 1.0])))
        self.assertEqual(0.0, dot(np.asarray([0.0, -0.1, 0.0]), np.asarray([0.0, 0.0, 1.0])))
        self.assertEqual(0.0, dot(np.asarray([0.0, 0.1, 0.0]), np.asarray([0.0, 0.0, -1.0])))
        self.assertEqual(0.0, dot(np.asarray([0.0, -0.1, 0.0]), np.asarray([0.0, 0.0, -1.0])))
        self.assertEqual(0.0, dot(np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 0.3])))
        self.assertEqual(0.0, dot(np.asarray([-1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 0.3])))
        self.assertEqual(0.0, dot(np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, -0.3])))
        self.assertEqual(0.0, dot(np.asarray([-1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, -0.3])))

    def test_vector_projection(self):
        # Orthogonal vectors
        res_1 = vector_projection(np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0]))
        for res in res_1:
            self.assertEqual(0.0, res)

        # Parallel vectors
        res_2 = vector_projection(np.asarray([0.0, 0.0, 0.5]), np.asarray([0.0, 0.0, 9.0]))
        self.assertEqual(0.0, res_2[0])
        self.assertEqual(0.0, res_2[1])
        self.assertEqual(0.5, res_2[2])

        # Opposite vectors
        res_3 = vector_projection(np.asarray([-0.7, 0.0, 0.5]), np.asarray([0.1, 0.0, 0.0]))
        self.assertEqual(-0.7, res_3[0])
        self.assertEqual(0.0, res_3[1])
        self.assertEqual(0.0, res_3[2])

        # Right angle triangle
        vec = np.asarray([2.0, 1.0, 0.0])
        res_4 = vector_projection(vec, np.asarray([0.33, 0.0, 0.0]))
        self.assertEqual(2.0, res_4[0])
        self.assertEqual(0.0, res_4[1])
        self.assertEqual(0.0, res_4[2])

        res_5 = vector_projection(vec, np.asarray([0.0, 0.8, 0.0]))
        self.assertEqual(0.0, res_5[0])
        self.assertEqual(1.0, res_5[1])
        self.assertEqual(0.0, res_5[2])

    def test_cross(self):
        # Orthogonal axis aligned vectors
        res_1 = cross(np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0]))
        self.assertEqual(1.0, res_1[0])
        self.assertEqual(0.0, res_1[1])
        self.assertEqual(0.0, res_1[2])

        res_2 = cross(np.asarray([0.0, 0.0, 1.0]), np.asarray([0.0, 1.0, 0.0]))
        self.assertEqual(-1.0, res_2[0])
        self.assertEqual(0.0, res_2[1])
        self.assertEqual(0.0, res_2[2])

        # Parallel vectors
        for vec in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [3.54, 1.45, 0.69]]:
            res = cross(np.asarray(vec), np.asarray(vec))
            self.assertEqual(0.0, res[0])
            self.assertEqual(0.0, res[1])
            self.assertEqual(0.0, res[2])

    def test_rotate_2d(self):
        vector = np.asarray([1.0, 0.0])

        res = rotate_2d(vector, np.pi / 6.0)
        self.assertAlmostEqual(np.sqrt(3) / 2.0, res[0])
        self.assertAlmostEqual(0.5, res[1])

        res = rotate_2d(vector, np.pi / 3.0)
        self.assertAlmostEqual(0.5, res[0])
        self.assertAlmostEqual(np.sqrt(3) / 2.0, res[1])

        res_1 = rotate_2d(vector, np.pi / 2.0)
        self.assertAlmostEqual(0.0, res_1[0])
        self.assertAlmostEqual(1.0, res_1[1])

        res = rotate_2d(vector, 2 * np.pi / 3.0)
        self.assertAlmostEqual(-0.5, res[0])
        self.assertAlmostEqual(np.sqrt(3) / 2.0, res[1])

        res = rotate_2d(vector, 5 * np.pi / 6.0)
        self.assertAlmostEqual(-np.sqrt(3) / 2.0, res[0])
        self.assertAlmostEqual(0.5, res[1])

        res_2 = rotate_2d(vector, np.pi)
        self.assertAlmostEqual(-1.0, res_2[0])
        self.assertAlmostEqual(0.0, res_2[1])

        res_3 = rotate_2d(vector, -np.pi / 2.0)
        self.assertAlmostEqual(0.0, res_3[0])
        self.assertAlmostEqual(-1.0, res_3[1])

    def test_rotate_3d(self):
        # X Axis
        vector = np.asarray([1.0, 0.0, 0.0])
        res_1 = rotate_3d(vector, np.asarray([np.pi / 2.0, 0.0, 0.0]))
        self.assertAlmostEqual(1.0, res_1[0])
        self.assertAlmostEqual(0.0, res_1[1])
        self.assertAlmostEqual(0.0, res_1[2])

        res_2 = rotate_3d(vector, np.asarray([0.0, np.pi / 2.0, 0.0]))
        self.assertAlmostEqual(0.0, res_2[0])
        self.assertAlmostEqual(0.0, res_2[1])
        self.assertAlmostEqual(-1.0, res_2[2])

        res_3 = rotate_3d(vector, np.asarray([0.0, 0.0, np.pi / 2.0]))
        self.assertAlmostEqual(0.0, res_3[0])
        self.assertAlmostEqual(1.0, res_3[1])
        self.assertAlmostEqual(0.0, res_3[2])

        # Y Axis
        vector = np.asarray([0.0, 1.0, 0.0])
        res_1 = rotate_3d(vector, np.asarray([np.pi / 2.0, 0.0, 0.0]))
        self.assertAlmostEqual(0.0, res_1[0])
        self.assertAlmostEqual(0.0, res_1[1])
        self.assertAlmostEqual(1.0, res_1[2])

        res_2 = rotate_3d(vector, np.asarray([0.0, np.pi / 2.0, 0.0]))
        self.assertAlmostEqual(0.0, res_2[0])
        self.assertAlmostEqual(1.0, res_2[1])
        self.assertAlmostEqual(0.0, res_2[2])

        res_3 = rotate_3d(vector, np.asarray([0.0, 0.0, np.pi / 2.0]))
        self.assertAlmostEqual(-1.0, res_3[0])
        self.assertAlmostEqual(0.0, res_3[1])
        self.assertAlmostEqual(0.0, res_3[2])

        # Z Axis
        vector = np.asarray([0.0, 0.0, 1.0])
        res_1 = rotate_3d(vector, np.asarray([np.pi / 2.0, 0.0, 0.0]))
        self.assertAlmostEqual(0.0, res_1[0])
        self.assertAlmostEqual(-1.0, res_1[1])
        self.assertAlmostEqual(0.0, res_1[2])

        res_2 = rotate_3d(vector, np.asarray([0.0, np.pi / 2.0, 0.0]))
        self.assertAlmostEqual(1.0, res_2[0])
        self.assertAlmostEqual(0.0, res_2[1])
        self.assertAlmostEqual(0.0, res_2[2])

        res_3 = rotate_3d(vector, np.asarray([0.0, 0.0, np.pi / 2.0]))
        self.assertAlmostEqual(0.0, res_3[0])
        self.assertAlmostEqual(0.0, res_3[1])
        self.assertAlmostEqual(1.0, res_3[2])

    def test_rotate_arbitrary_axis(self):
        point = np.array([1.0, 0.0, 0.0])
        vector = np.array([0.0, 1.0, 0.0])
        rotated_point = arbitrary_axis_rotation_3d(point, vector, np.pi / 2)
        self.assertAlmostEqual(0.0, rotated_point[0])
        self.assertAlmostEqual(0.0, rotated_point[1])
        self.assertAlmostEqual(-1.0, rotated_point[2])

        point = np.array([0.0, 0.0, 1.0])
        vector = np.array([0.0, 1.0, 0.0])
        rotated_point = arbitrary_axis_rotation_3d(point, vector, np.pi / 2)
        self.assertAlmostEqual(1.0, rotated_point[0])
        self.assertAlmostEqual(0.0, rotated_point[1])
        self.assertAlmostEqual(0.0, rotated_point[2])

        point = np.array([0.0, 1.0, 0.0])
        vector = np.array([1.0, 0.0, 0.0])
        rotated_point = arbitrary_axis_rotation_3d(point, vector, np.pi / 2)
        self.assertAlmostEqual(0.0, rotated_point[0])
        self.assertAlmostEqual(0.0, rotated_point[1])
        self.assertAlmostEqual(1.0, rotated_point[2])

        point = np.array([0.0, 0.0, 1.0])
        vector = np.array([1.0, 0.0, 0.0])
        rotated_point = arbitrary_axis_rotation_3d(point, vector, np.pi / 2)
        self.assertAlmostEqual(0.0, rotated_point[0])
        self.assertAlmostEqual(-1.0, rotated_point[1])
        self.assertAlmostEqual(0.0, rotated_point[2])

        point = np.array([0.0, 1.0, 0.0])
        vector = np.array([0.0, 0.0, 1.0])
        rotated_point = arbitrary_axis_rotation_3d(point, vector, np.pi / 2)
        self.assertAlmostEqual(-1.0, rotated_point[0])
        self.assertAlmostEqual(0.0, rotated_point[1])
        self.assertAlmostEqual(0.0, rotated_point[2])

        point = np.array([1.0, 0.0, 0.0])
        vector = np.array([0.0, 0.0, 1.0])
        rotated_point = arbitrary_axis_rotation_3d(point, vector, np.pi / 2)
        self.assertAlmostEqual(0.0, rotated_point[0])
        self.assertAlmostEqual(1.0, rotated_point[1])
        self.assertAlmostEqual(0.0, rotated_point[2])

        point = np.array([0.0, 1.0, 0.0])
        vector = np.array([0.0, 1.0, 0.0])
        rotated_point = arbitrary_axis_rotation_3d(point, vector, np.pi / 2)
        self.assertAlmostEqual(0.0, rotated_point[0])
        self.assertAlmostEqual(1.0, rotated_point[1])
        self.assertAlmostEqual(0.0, rotated_point[2])


if __name__ == '__main__':
    unittest.main()