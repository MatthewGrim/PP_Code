"""
Author: Rohan Ramasamy
Date: 02/10/17

This file contains a solver to obtain the motion of a particle in a spherical charge E field
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plasma_physics.pysrc.simulation.pic.algo.fields.electric_fields.generic_e_fields import PointField
from plasma_physics.pysrc.simulation.pic.algo.particle_pusher.boris_solver import boris_solver
from plasma_physics.pysrc.simulation.pic.data.particles.charged_particle import ChargedParticle


def E_field_example():
    """
    Example E field solution

    :return:
    """
    particle = ChargedParticle(1.6e-27, 1.6e-19, np.asarray([0.1, 0.0, 0.0]), np.asarray([0.0, 0.1, 0.0]))
    field = PointField(-1.6e-19, 0.1, np.zeros(3))

    def B_field(x):
        B = np.zeros(x.shape)
        return B

    X = particle.position
    V = particle.velocity
    Q = np.asarray([particle.charge])
    M = np.asarray([particle.mass])

    times = np.linspace(0.0, 1, 10000)
    positions = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
    velocities = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
    for i, t in enumerate(times):
        if i == 0:
            positions[i, :, :] = X
            velocities[i, :, :] = V
            continue

        dt = times[i] - times[i - 1]

        x, v = boris_solver(field.e_field, B_field, X, V, Q, M, dt)

        positions[i, :, :] = x
        velocities[i, :, :] = v
        X = x
        V = v

    x = positions[:, :, 0].flatten()
    y = positions[:, :, 1].flatten()
    z = positions[:, :, 2].flatten()
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot('111', projection='3d')
    ax.plot(x, y, z, label='numerical')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='best')
    ax.set_title("Analytic and Numerical Particle Motion")
    plt.show()


if __name__ == '__main__':
    E_field_example()

