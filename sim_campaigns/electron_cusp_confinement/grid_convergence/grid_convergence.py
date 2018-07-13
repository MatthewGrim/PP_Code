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
import scipy

from plasma_physics.pysrc.simulation.pic.algo.fields.magnetic_fields.generic_b_fields import *
from plasma_physics.pysrc.simulation.pic.algo.particle_pusher.boris_solver import *
from plasma_physics.pysrc.simulation.pic.data.particles.charged_particle import PICParticle

def run_sim(field, particle, dt_factor):
    # There is no E field in the simulations
    def e_field(x):
        return np.zeros(x.shape) 

    X = particle.position
    V = particle.velocity
    Q = np.asarray([particle.charge])
    M = np.asarray([particle.mass])

    vel_magnitude = np.sqrt(np.sum(V * V))
    dt = dt_factor * radius / vel_magnitude

    dt = 1e-9 * radius
    final_time = 1e5 * dt
    dt *= dt_factor

    num_steps = int(final_time / dt)
    print(dt, final_time, num_steps)
    times = np.linspace(0.0, final_time, num_steps)
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
    plt.savefig("grid_convergence_{}.png".format(dt_factor))

    return times, x, y, z


if __name__ == '__main__':
    # Generate Polywell field
    use_interpolator = True
    if use_interpolator:
        loop_pts = 100
        domain_pts = 100
        I = 1e4
        radius = 0.1
        loop_offset = 1.25
        dom_size = 1.1 * loop_offset * radius
        file_name = "b_field_{}_{}_{}_{}_{}_{}".format(I * 1e-3, radius, loop_offset, domain_pts, loop_pts, dom_size)
        file_path = os.path.join("..", "mesh_generation", "{}".format(I), file_name)
        b_field = InterpolatedBField(file_path, dom_pts_idx=6, dom_size_idx=8)
    else:
        I = 1e4
        radius = 0.15
        loop_offset = 0.175
        loop_pts = 200
        comp_loops = list()
        comp_loops.append(CurrentLoop(I, radius, np.asarray([-loop_offset, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), loop_pts))
        comp_loops.append(CurrentLoop(I, radius, np.asarray([loop_offset, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0]), loop_pts))
        comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, -loop_offset, 0.0]), np.asarray([0.0, 1.0, 0.0]), loop_pts))
        comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, loop_offset, 0.0]), np.asarray([0.0, -1.0, 0.0]), loop_pts))
        comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, 0.0, -loop_offset]), np.asarray([0.0, 0.0, 1.0]), loop_pts))
        comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, 0.0, loop_offset]), np.asarray([0.0, 0.0, -1.0]), loop_pts))
        b_field = CombinedField(comp_loops)

    t_results = []
    x_results = []
    y_results = []
    z_results = []
    dt_factors = [1000.0, 100.0, 10.0, 1.0]
    for dt in dt_factors:
        seed = 1
        np.random.seed(seed)
        # Define charge particle
        max_vel = 1e4
        vel = np.random.uniform(low=-1.0, high=1.0, size=(3, )) * max_vel
        particle = PICParticle(9.1e-31, 1.6e-19, np.asarray([0.0, 0.0, 0.0]), vel)

        t, x, y, z = run_sim(b_field, particle, dt)
        t_results.append(t)
        x_results.append(x)
        y_results.append(y)
        z_results.append(z)

    fig, ax = plt.subplots(4)

    x_interp = scipy.interpolate.interp1d(t_results[-1], x_results[-1])
    y_interp = scipy.interpolate.interp1d(t_results[-1], y_results[-1])
    z_interp = scipy.interpolate.interp1d(t_results[-1], z_results[-1])
    for i in range(len(dt_factors) - 1):
        dx = np.abs(x_interp(t_results[i]) - x_results[i])
        dy = np.abs(y_interp(t_results[i]) - y_results[i])
        dz = np.abs(z_interp(t_results[i]) - z_results[i])
        d_tot = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        ax[0].plot(t_results[i], dx, label="{}".format(dt_factors[i]))
        ax[1].plot(t_results[i], dy, label="{}".format(dt_factors[i]))
        ax[2].plot(t_results[i], dx, label="{}".format(dt_factors[i]))
        ax[3].plot(t_results[i], d_tot, label="{}".format(dt_factors[i]))

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()

    plt.savefig("Grid_Convergence_Overall_{}_{}_{}.png".format(max_vel, I, radius))
    plt.show()

    results = np.concatenate((t_results[-1], x_results[-1], y_results[-1], z_results[-1]))
    np.savetxt("High_Res_{}.txt".format(dt_factors[-1]), results)

