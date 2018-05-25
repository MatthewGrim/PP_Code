"""
Author: Rohan Ramasamy
Date: 16/10/17

This file contains a solver to obtain the motion of a particle in a polywell B field
"""

import os

from plasma_physics.pysrc.simulation.pic.fields.magnetic_fields.generic_b_fields import *
from plasma_physics.pysrc.simulation.pic.algo.particle_pusher.boris_solver import *


def B_field_example_single(particle, vel):
    """
    Example of B field solution for a polywell

    :return:
    """
    loop_pts = 100
    domain_pts = 100
    dom_size = 0.175
    current_offset = 0.0
    # file_name = "b_field_{}_{}_{}".format(loop_pts, domain_pts, dom_size)
    file_name = "b_field_{}_{}_{}_{}".format(loop_pts, domain_pts, dom_size, current_offset)
    file_path = os.path.join("data", file_name)
    b_field = InterpolatedBField(file_path)

    def e_field(x):
        return np.zeros(x.shape)

    X = particle.position
    V = particle.velocity
    Q = np.asarray([particle.charge])
    M = np.asarray([particle.mass])

    vel_magnitude = np.sqrt(np.sum(V * V))
    dt = 0.3 * dom_size * 2 / (domain_pts * vel_magnitude)

    num_traversals = 20
    final_time = dom_size * 2 * num_traversals / vel_magnitude

    num_steps = int(final_time / dt)
    print(num_steps)
    times = np.linspace(0.0, dt * num_steps, num_steps)
    positions = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
    velocities = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
    for i, t in enumerate(times):
        print(i / num_steps)
        if i == 0:
            positions[i, :, :] = X
            velocities[i, :, :] = V
            continue

        dt = times[i] - times[i - 1]

        try:
            x, v = boris_solver(e_field, b_field.b_field, X, V, Q, M, dt)
        except ValueError:
            break

        positions[i, :, :] = x
        velocities[i, :, :] = v
        X = x
        V = v

    # Convert points to x, y and z locations
    x = positions[:, :, 0].flatten()
    y = positions[:, :, 1].flatten()
    z = positions[:, :, 2].flatten()

    # Plot x, y, z locations independently
    fig, axes = plt.subplots(1, 3)
    axes[0].plot(times, x)
    axes[1].plot(times, y)
    axes[2].plot(times, z)
    plt.show()

    # Plot 3D motion
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot('111', projection='3d')
    ax.plot(x, y, z, label='numerical')
    ax.set_xlim([-dom_size, dom_size])
    ax.set_ylim([-dom_size, dom_size])
    ax.set_zlim([-dom_size, dom_size])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='best')
    ax.set_title("Analytic and Numerical Particle Motion")
    plt.show()


if __name__ == '__main__':
    for i in range(10):
        # Define fields and charge particle
        seed = 12
        max_vel = 1e6
        vel = np.random.uniform(low=-1.0, high=1.0, size=(3, )) * max_vel

        # particle = ChargedParticle(6.64e-27, 3.2e-19, np.asarray([0.0, 0.0, 0.0]), vel)
        # particle = ChargedParticle(3.32e-27, 1.6e-19, np.asarray([0.0, 0.0, 0.0]), vel)
        particle = PICParticle(9.1e-31, 1.6e-19, np.asarray([0.0, 0.0, 0.0]), vel)

        B_field_example_single(particle, vel)

