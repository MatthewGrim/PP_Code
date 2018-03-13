"""
Author: Rohan Ramasamy
Date: 28/10/2017

This file contains simple simulations using a boris solver in a known field to test the algorithm
"""

import matplotlib.pyplot as plt
import numpy as np

from plasma_physics.pysrc.simulation.pic.algo.particle_pusher.boris_solver import boris_solver
from plasma_physics.pysrc.simulation.pic.data.particles.charged_particle import ChargedParticle
from plasma_physics.pysrc.simulation.pic.simulations.analytic_single_particle_motion import solve_B_field


def single_particle_example(b_field=np.asarray([0.0, 0.0, 1.0]),
                            X_0=np.asarray([[0.0, 1.0, 0.0]]),
                            V_0=np.asarray([[-2.0, 0.0, 1.0]])):
    """
    Example solution to simple magnetic field case
    :return:
    """
    def B_field(x):
        B = np.zeros(x.shape)
        for i, b in enumerate(B):
            B[i, :] = b_field
        return B

    def E_field(x):
        E = np.zeros(x.shape)
        return E

    X = X_0
    V = V_0
    Q = np.asarray([1.0])
    M = np.asarray([1.0])

    times = np.linspace(0.0, 4.0, 1000)
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

    particle = ChargedParticle(1.0, Q[0], X_0[0], V_0[0])
    B = b_field
    analytic_times, analytic_positions = solve_B_field(particle, B, 4.0)

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

    fig, axes = plt.subplots(3, figsize=(20, 10))
    axes[0].plot(x)
    axes[1].plot(y)
    axes[2].plot(z)
    fig.suptitle("Numerical Solution")
    plt.show()

    fig, axes = plt.subplots(3, figsize=(20, 10))
    axes[0].plot(x - analytic_positions[:, 0])
    axes[1].plot(y - analytic_positions[:, 1])
    axes[2].plot(z - analytic_positions[:, 2])
    fig.suptitle("Deviation of Numerical Solution from the Analytic")
    plt.show()

if __name__ == '__main__':
    single_particle_example()