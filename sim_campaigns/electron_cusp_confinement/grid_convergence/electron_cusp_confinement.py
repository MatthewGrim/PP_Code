"""
Author: Rohan Ramasamy
Date: 10/07/2018

This file contains a simulation to model a single electron in an analytic polywell field. This can be 
used to estimate the loss in electrons over time.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plasma_physics.pysrc.simulation.pic.algo.fields.magnetic_fields.generic_b_fields import *
from plasma_physics.pysrc.simulation.pic.algo.particle_pusher.boris_solver import *
from plasma_physics.pysrc.simulation.pic.data.particles.charged_particle import PICParticle

def run_sim(particle):
    # Generate Polywell field
    I = 1e4
    radius = 0.15
    loop_offset = 0.175
    loop_pts = 100
    comp_loops = list()
    comp_loops.append(CurrentLoop(I, radius, np.asarray([-loop_offset, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), loop_pts))
    comp_loops.append(CurrentLoop(I, radius, np.asarray([loop_offset, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0]), loop_pts))
    comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, -loop_offset, 0.0]), np.asarray([0.0, 1.0, 0.0]), loop_pts))
    comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, loop_offset, 0.0]), np.asarray([0.0, -1.0, 0.0]), loop_pts))
    comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, 0.0, -loop_offset]), np.asarray([0.0, 0.0, 1.0]), loop_pts))
    comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, 0.0, loop_offset]), np.asarray([0.0, 0.0, -1.0]), loop_pts))
    b_field = CombinedField(comp_loops)

    # There is no E field in the simulations
    def e_field(x):
        return np.zeros(x.shape) 

    X = particle.position
    V = particle.velocity
    Q = np.asarray([particle.charge])
    M = np.asarray([particle.mass])

    vel_magnitude = np.sqrt(np.sum(V * V))
    dt = 1e-2 * radius / vel_magnitude

    num_traversals = 2
    final_time = radius * 4 * num_traversals / vel_magnitude

    num_steps = int(final_time / dt)
    print(dt, final_time, num_steps)
    times = np.linspace(0.0, dt * num_steps, num_steps)
    positions = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
    velocities = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
    for i, t in enumerate(times):
        print(i / float(num_steps))
        if i == 0:
            positions[i, :, :] = X
            velocities[i, :, :] = V
            continue

        dt = times[i] - times[i - 1]

        try:
            x, v = boris_solver(e_field, b_field.b_field, X, V, Q, M, dt)
        except ValueError:
            print("PARTICLE ESCAPED!")
            break

        positions[i, :, :] = x
        velocities[i, :, :] = v
        X = x
        V = v

    # Convert points to x, y and z locations
    x = positions[:, :, 0].flatten()
    y = positions[:, :, 1].flatten()
    z = positions[:, :, 2].flatten()

    # Convert velocities to x, y and z locations
    v_x = velocities[:, :, 0].flatten()
    v_y = velocities[:, :, 1].flatten()
    v_z = velocities[:, :, 2].flatten()
    v_total = np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)

    # Plot x, y, z locations independently
    fig, axes = plt.subplots(1, 3)
    axes[0].plot(times, x)
    axes[1].plot(times, y)
    axes[2].plot(times, z)
    plt.show()

    plt.figure()
    plt.plot(times, v_total)
    plt.show()

    # Plot 3D motion
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot('111', projection='3d')
    ax.plot(x, y, z, label='numerical')
    ax.set_xlim([-loop_offset, loop_offset])
    ax.set_ylim([-loop_offset, loop_offset])
    ax.set_zlim([-loop_offset, loop_offset])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='best')
    ax.set_title("Analytic and Numerical Particle Motion")
    plt.show()


if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    for i in range(10):
        # Define charge particle
        max_vel = 1e4
        vel = np.random.uniform(low=-1.0, high=1.0, size=(3, )) * max_vel
        particle = PICParticle(9.1e-31, 1.6e-19, np.asarray([0.0, 0.0, 0.0]), vel)

        run_sim(particle)