"""
Author: Rohan Ramasamy
Date: 18/07/2018

This file contains code for a simple particle pusher based on the outline in Gummersall's paper:

Scaling law of electron confinement in a zero beta polywell device
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from plasma_physics.pysrc.simulation.pic.algo.particle_pusher.boris_solver import boris_solver
from plasma_physics.pysrc.simulation.pic.simulations.analytic_single_particle_motion import solve_B_field, solve_E_field, solve_aligned_fields
from plasma_physics.pysrc.simulation.pic.data.particles.charged_particle import PICParticle
from plasma_physics.pysrc.simulation.pic.algo.geometry.vector_ops import *


def gummersall_solver(E_field, B_field, X, V, Q, M, dt):
    """
    Function to update the positon of a set of particles in an electromagnetic field over the time dt

    :param E_field: function to evaluate the 3D E field at time t
    :param B_field: function to evaluate the 3D B field at time t
    :param X: position of the particles in the simulation domain
    :param V: velocities of the particles in the simulation domain
    :param Q: charges of the particles in the simulation domain
    ;param M: masses of the particles in the simulation domain
    :return:
    """
    assert isinstance(X, np.ndarray) and X.shape[1] == 3
    assert isinstance(V, np.ndarray) and V.shape[1] == 3
    assert X.shape[0] == V.shape[0] == Q.shape[0] == M.shape[0]
    assert isinstance(dt, float)

    B = B_field(X)
    E = E_field(X)
    assert magnitude(B) != 0.0
    assert magnitude(E) != 0.0
    B = B[0]
    E = E[0]
    V = V[0]
    e_x = normalise(E)
    e_y = cross(e_x, e_z)
    e_z = normalise(B)

    v_parallel = dot(V, e_z)
    v_perp = V - v_parallel
    B_mag = magnitude(B)
    v_perp_mag = magnitude(v_perp)
    F = cross(v_perp, B) * Q[0]
    r_g = (M[0] * dot(v_perp, v_perp)) / magnitude(F)
    omega = B_mag * Q / M

    E_mag = magnitude(E)
    X_new = X + r_g * np.sin(omega * dt) * e_x - r_g * np.cos(omega * dt) * e_y + v_parallel * e_z * dt
    V_new = v_perp_mag * np.cos(omega * dt) * e_x + v_perp_mag * np.sin(omega * dt) * e_y + v_parallel * e_z
    X_new -= E_mag / B_mag * dt * e_y
    V_new -= E_mag / B_mag * e_y
    V_new = np.asarray([V_new])

    return X_new, V_new


def test_aligned_fields():
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

            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot('111', projection='3d')
            ax.plot(x, y, z, label='numerical')
            ax.plot(analytic_positions[:, 0], analytic_positions[:, 1], analytic_positions[:, 2], label='analytic')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend(loc='best')
            ax.set_title("Analytic and Numerical Particle Motion")
            plt.show()


if __name__ == "__main__":
    test_aligned_fields()

