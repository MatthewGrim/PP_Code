"""
Author: Rohan Ramasamy
Date: 26/10/17

This file contains a solver to obtain the motion of a particle in a polywell B field in parallel
"""

import multiprocessing as mp
import os

from enum import Enum
from plasma_physics.pysrc.simulation.pic.fields.magnetic_fields.generic_b_fields import *

from plasma_physics.pysrc.simulation.pic.algo.particle_pusher.boris_solver import *


class ParticleType(Enum):
    ELECTRON = 1
    PROTON = 2
    ALPHA = 3


def get_output_file():
    # Find output file
    cwd = os.getcwd()
    out_dir = os.path.join(cwd, "results")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, "{}_param_scan_{}_{}".format(particle_type.name, num_sims, seed))

    return out_file


def B_field_example_multi(particle_type, vel):
    """
    Example of B field solution for a polywell

    :return:
    """
    if particle_type is ParticleType.ALPHA:
        particle = PICParticle(6.64e-27, 3.2e-19, np.asarray([0.0, 0.0, 0.0]), vel)
    elif particle_type is ParticleType.PROTON:
        particle = PICParticle(3.32e-27, 1.6e-19, np.asarray([0.0, 0.0, 0.0]), vel)
    elif particle_type is ParticleType.ELECTRON:
        particle = PICParticle(9.1e-31, 1.6e-19, np.asarray([0.0, 0.0, 0.0]), vel)
    else:
        raise ValueError("Invalid value")

    loop_pts = 100
    domain_pts = 100
    dom_size = 0.175
    current_offset = 0.0
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

    num_traversals = 15
    final_time = dom_size * 2 * num_traversals / vel_magnitude

    num_steps = int(final_time / dt)
    times = np.linspace(0.0, dt * num_steps, num_steps)
    positions = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
    velocities = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
    for i, t in enumerate(times):
        if i == 0:
            positions[i, :, :] = X
            velocities[i, :, :] = V
            continue

        dt = times[i] - times[i - 1]

        try:
            x, v = boris_solver(e_field, b_field.b_field, X, V, Q, M, dt)
        except ValueError:
            return X, V, False

        positions[i, :, :] = x
        velocities[i, :, :] = v
        X = x
        V = v

    return X, V, True


def run_param_scan():
    np.random.seed(seed)

    max_vel = 1e6
    vel = np.random.uniform(low=-1.0, high=1.0, size=(3, num_sims)) * max_vel

    pool = mp.Pool(processes=2)
    results = [pool.apply(B_field_example_multi, args=(particle_type, vel[:, i], )) for i in range(num_sims)]

    X = np.zeros((num_sims, 3))
    V = np.zeros((num_sims, 3))
    confined = np.zeros((num_sims, ))
    for i, result in enumerate(results):
        X[i, :] = result[0]
        V[i, :] = result[1]
        confined[i] = result[2]

    out_file = get_output_file()

    np.savetxt(out_file, (X[:, 0], X[:, 1], X[:, 2], V[:, 0], V[:, 1], V[:, 2], confined[:]))


def process_sims():
    out_file = get_output_file()

    # Get data
    data = np.loadtxt(out_file)
    x = data[0, :]
    y = data[1, :]
    z = data[2, :]

    confined = data[6, :]
    success = np.where(confined > 0.5)[0]
    failed = np.where(confined < 0.5)[0]
    percentage_confined = len(np.where(confined > 0.5)[0]) / num_sims * 100
    print("Confinement: {}%".format(percentage_confined))

    # Plot 3D motion
    dom_size = 0.175
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot('111', projection='3d')
    ax.scatter(x[success], y[success], z[success], c='b', label='confined')
    ax.scatter(x[failed], y[failed], z[failed], c='r', label='escaped')
    ax.set_xlim([-dom_size, dom_size])
    ax.set_ylim([-dom_size, dom_size])
    ax.set_zlim([-dom_size, dom_size])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='best')
    ax.set_title("Final particle positions in random velocity scan")
    plt.show()


if __name__ == '__main__':
    particle_type = ParticleType.ELECTRON
    num_sims = 5000
    seed = 12

    # run_param_scan()
    process_sims()

