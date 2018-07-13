"""
Author: Rohan Ramasamy
Date: 28/10/2017

This file contains tests for the Boris solver to ensure that results at least match some basic analytic solutions for
combinations of electric and magnetic fields
"""

import unittest
import numpy as np

from plasma_physics.pysrc.simulation.pic.algo.particle_pusher.boris_solver import boris_solver
from plasma_physics.pysrc.simulation.pic.simulations.analytic_single_particle_motion import solve_B_field, solve_E_field, solve_aligned_fields
from plasma_physics.pysrc.simulation.pic.data.particles.charged_particle import PICParticle


class BorisSolverTest(unittest.TestCase):
    def test_uniform_B_field(self):
        """
        This test considers a particle in a uniform B field, and aims to compare analytic and numerical solution. The
        particle can have an arbitrary velocity, and the B field is relatively in an arbitrary direction
        :return:
        """
        seed = 1
        num_tests = 100
        np.random.seed(seed)

        Bs = np.random.uniform(low=0.0, high=10.0, size=(3, num_tests))
        X_0s = np.random.uniform(low=-1.0, high=1.0, size=(3, num_tests))
        V_0s = np.random.uniform(low=-1.0, high=1.0, size=(3, num_tests))

        for idx in range(num_tests):
            b_field = Bs[:, idx]
            X_0 = X_0s[:, idx]
            V_0 = V_0s[:, idx]

            def B_field(x):
                B = np.zeros(x.shape)
                for i, b in enumerate(B):
                    B[i, :] = b_field
                return B

            def E_field(x):
                E = np.zeros(x.shape)
                return E

            X = X_0.reshape((1, 3))
            V = V_0.reshape((1, 3))
            Q = np.asarray([1.0])
            M = np.asarray([1.0])

            final_time = 4.0
            num_pts = 500
            times = np.linspace(0.0, final_time, num_pts)
            positions = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
            velocities = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
            for i, t in enumerate(times):
                if i == 0:
                    positions[i, :, :] = X
                    velocities[i, :, :] = V
                    continue

                dt = times[i] - times[i - 1]

                x, v = boris_solver(E_field, B_field, X, V, Q, M, dt)

                positions[i, :, :] = x
                velocities[i, :, :] = v
                X = x
                V = v

            particle = PICParticle(1.0, Q[0], X_0, V_0)
            B = b_field
            analytic_times, analytic_positions = solve_B_field(particle, B, final_time, num_pts=num_pts)

            x = positions[:, :, 0].flatten()
            y = positions[:, :, 1].flatten()
            z = positions[:, :, 2].flatten()

            self.assertLess(np.absolute(np.average(x - analytic_positions[:, 0])), 0.01)
            self.assertLess(np.absolute(np.average(y - analytic_positions[:, 1])), 0.01)
            self.assertLess(np.absolute(np.average(z - analytic_positions[:, 2])), 0.01)

    def test_uniform_e_field(self):
        """
        This test considers a particle moving in a uniform electric field
        :return:
        """
        seed = 1
        num_tests = 100
        np.random.seed(seed)

        Es = np.random.uniform(low=0.0, high=1.0, size=(3, num_tests))
        X_0s = np.random.uniform(low=-1.0, high=1.0, size=(3, num_tests))
        V_0s = np.random.uniform(low=-1.0, high=1.0, size=(3, num_tests))

        for idx in range(num_tests):
            e_field = Es[:, idx]
            X_0 = X_0s[:, idx]
            V_0 = V_0s[:, idx]

            def B_field(x):
                B = np.zeros(x.shape)
                return B

            def E_field(x):
                E = np.zeros(x.shape)
                for i, b in enumerate(E):
                    E[i, :] = e_field
                return E

            X = X_0.reshape((1, 3))
            V = V_0.reshape((1, 3))
            Q = np.asarray([1.0])
            M = np.asarray([1.0])

            final_time = 4.0
            num_pts = 1000
            times = np.linspace(0.0, final_time, num_pts)
            positions = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
            velocities = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
            for i, t in enumerate(times):
                if i == 0:
                    positions[i, :, :] = X
                    velocities[i, :, :] = V
                    continue

                dt = times[i] - times[i - 1]

                x, v = boris_solver(E_field, B_field, X, V, Q, M, dt)

                positions[i, :, :] = x
                velocities[i, :, :] = v
                X = x
                V = v

            particle = PICParticle(1.0, Q[0], X_0, V_0)
            E = e_field
            analytic_times, analytic_positions = solve_E_field(particle, E, final_time, num_pts=num_pts)

            x = positions[:, :, 0].flatten()
            y = positions[:, :, 1].flatten()
            z = positions[:, :, 2].flatten()

            self.assertLess(np.absolute(np.average(x - analytic_positions[:, 0])), 0.01, msg="{}, {}, {}".format(X_0, V_0, e_field))
            self.assertLess(np.absolute(np.average(y - analytic_positions[:, 1])), 0.01, msg="{}, {}, {}".format(X_0, V_0, e_field))
            self.assertLess(np.absolute(np.average(z - analytic_positions[:, 2])), 0.01, msg="{}, {}, {}".format(X_0, V_0, e_field))

    def test_aligned_fields(self):
        """
        This test considers a particle moving in a uniform orthogonal electric and magnetic field
        :return:
        """
        seed = 1
        np.random.seed(seed)
        for direction in [np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0])]:
            for sign in [-1, 1]:
                # randomise initial conditions
                B_mag = np.random.uniform(low=0.0, high=1.0)
                E_mag = np.random.uniform(low=0.0, high=1.0)
                X_0 = np.random.uniform(low=-1.0, high=1.0, size=(1, 3))
                V_0 = np.random.uniform(low=-1.0, high=1.0, size=(1, 3))

                def B_field(x):
                    B = np.zeros(x.shape)
                    for i, b in enumerate(B):
                        B[i, :] = B_mag * direction * sign
                    return B

                def E_field(x):
                    E = np.zeros(x.shape)
                    for i, b in enumerate(E):
                        E[i, :] = E_mag * direction * sign
                    return E

                X = X_0.reshape((1, 3))
                V = V_0.reshape((1, 3))
                Q = np.asarray([1.0])
                M = np.asarray([1.0])

                final_time = 4.0
                num_pts = 1000
                times = np.linspace(0.0, final_time, num_pts)
                positions = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
                velocities = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
                for i, t in enumerate(times):
                    if i == 0:
                        positions[i, :, :] = X
                        velocities[i, :, :] = V
                        continue

                    dt = times[i] - times[i - 1]

                    x, v = boris_solver(E_field, B_field, X, V, Q, M, dt)

                    positions[i, :, :] = x
                    velocities[i, :, :] = v
                    X = x
                    V = v

                particle = PICParticle(1.0, Q[0], X_0[0], V_0[0])
                E = E_mag * direction * sign
                B = B_mag * direction * sign
                analytic_times, analytic_positions = solve_aligned_fields(particle, E, B, final_time, num_pts=num_pts)

                x = positions[:, :, 0].flatten()
                y = positions[:, :, 1].flatten()
                z = positions[:, :, 2].flatten()

                self.assertLess(np.absolute(np.average(x - analytic_positions[:, 0])), 0.01, msg="{}, {}, {}".format(X_0, V_0, E, B))
                self.assertLess(np.absolute(np.average(y - analytic_positions[:, 1])), 0.01, msg="{}, {}, {}".format(X_0, V_0, E, B))
                self.assertLess(np.absolute(np.average(z - analytic_positions[:, 2])), 0.01, msg="{}, {}, {}".format(X_0, V_0, E, B))

if __name__ == '__main__':
    unittest.main()

