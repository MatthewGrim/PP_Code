"""
Author: Rohan Ramasamy
Date: 23/07/2018

This script is used to generate radial probability distributions for electrons within a polywell device at different
coil spacings
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import multiprocessing as mp

from plasma_physics.pysrc.simulation.pic.algo.fields.magnetic_fields.generic_b_fields import *
from plasma_physics.pysrc.simulation.pic.algo.particle_pusher.boris_solver import *
from plasma_physics.pysrc.simulation.pic.data.particles.charged_particle import PICParticle
from plasma_physics.pysrc.simulation.pic.algo.geometry.vector_ops import *
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


def run_sim(params):
    b_field, particle, radius, domain_size, I, dI_dt = params
    print_output = False

    # There is no E field in the simulations
    def e_field(x):
        B = b_field.b_field(x)
        dB_dt = B / I * dI_dt
        return -dB_dt

    X = particle.position
    V = particle.velocity
    Q = np.asarray([particle.charge])
    M = np.asarray([particle.mass])

    # Set timestep according to Gummersall approximation
    dt = 1e-9 * radius
    final_time = 1e5 * dt

    num_steps = int(final_time / dt)
    times = np.linspace(0.0, final_time, num_steps)
    positions = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
    velocities = np.zeros((times.shape[0], X.shape[0], X.shape[1]))
    for i, t in enumerate(times):
        if print_output:
            print(t / final_time)
        if i == 0:
            positions[i, :, :] = X
            velocities[i, :, :] = V
            continue

        dt = times[i] - times[i - 1]

        x, v = boris_solver(e_field, b_field.b_field, X, V, Q, M, dt)

        if np.any(x[0, :] < -domain_size) or np.any(x[0, :] > domain_size):
            if print_output:
                print("PARTICLE ESCAPED! - {}, {}, {}".format(i, times[i], X[0]))

            x = positions[:, :, 0].flatten()
            y = positions[:, :, 1].flatten()
            z = positions[:, :, 2].flatten()

            return times, x, y, z, i - 1

        positions[i, :, :] = x
        velocities[i, :, :] = v
        X = x
        V = v

    # Convert points to x, y and z locations
    x = positions[:, :, 0].flatten()
    y = positions[:, :, 1].flatten()
    z = positions[:, :, 2].flatten()

    return times, x, y, z, None


def run_parallel_sims(params):
    electron_energy, I, batch_num = params
    dI_dt = 0.0
    to_kA = 1e-3

    process_name = "electron_energy-{}eV-current-{}kA-batch-{}".format(electron_energy, I * to_kA, batch_num)
    print("Starting process: {}".format(process_name))
    
    # Generate Polywell field
    radius = 1.0
    loop_pts = 200
    domain_pts = 130
    loop_offset = 1.25
    dom_size = 1.1 * loop_offset * radius
    file_name = "b_field_{}_{}_{}_{}_{}_{}".format(I * to_kA, radius, loop_offset, domain_pts, loop_pts, dom_size)
    file_path = os.path.join("..", "mesh_generation", "data", "radius-{}m".format(radius), "current-{}kA".format(I * to_kA), "domres-{}".format(domain_pts), file_name)
    b_field = InterpolatedBField(file_path, dom_pts_idx=6, dom_size_idx=8)

    seed = batch_num
    np.random.seed(seed)

    # Run simulations
    num_bins = 100
    particle_distribution_count = np.zeros((num_bins, ))
    radial_bins = np.linspace(0.0, np.sqrt(3) * loop_offset * radius, num_bins)
    num_sims = 200
    final_positions = []
    vel = np.sqrt(2.0 * electron_energy * PhysicalConstants.electron_charge / PhysicalConstants.electron_mass)
    for i in range(num_sims):
        # Define particle velocity and 100eV charge particle
        z_unit = np.random.uniform(-1.0, 1.0)
        xy_plane = np.sqrt(1 - z_unit ** 2)
        phi = np.random.uniform(0.0, 2 * np.pi)
        velocity = np.asarray([xy_plane * np.cos(phi), xy_plane * np.sin(phi), z_unit]) * vel
        particle = PICParticle(9.1e-31, 1.6e-19, np.random.uniform(-3.0 * radius / 16.0, 3.0 * radius / 16.0, size=(3, )), velocity)

        t, x, y, z, final_idx = run_sim((b_field, particle, radius, loop_offset * radius, I, dI_dt))

        # Add results to list
        escaped = False if final_idx is None else True
        final_idx = final_idx if escaped else -1

        # Get probability of electron in 50 radial spacings in sim
        radial_position = np.sqrt(x[:final_idx] ** 2 + y[:final_idx] ** 2 + z[:final_idx] ** 2)
        for i, r in enumerate(radial_bins):
            if i == 0.0:
                continue

            points_in_range = np.where(np.logical_and(radial_position >= radial_bins[i - 1], radial_position <= r)) 
            particle_distribution_count[i - 1] += points_in_range[0].shape[0]
        
    # Save to file
    if not os.path.exists("results"):
        os.makedirs("results")
    output_dir = os.path.join("results", "radius-{}m".format(radius))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir, "current-{}kA".format(I * to_kA))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "radial_distribution-current-{}-radius-{}-energy-{}-batch-{}.txt".format(I, radius, electron_energy, batch_num))
    np.savetxt(output_path, np.stack((radial_bins, particle_distribution_count)))

    print("Finished process: {}".format(process_name))


def get_radial_distributions():
    electron_energies = [0.1, 1.0, 10.0, 100.0, 1000.0, 1e4]
    I = [100.0, 200.0, 500.0, 1e3, 5e3, 2e4, 1e5]
    pool = mp.Pool(processes=4)
    args = []
    for current in I:
        for electron_energy in electron_energies:
            if electron_energy <= 10.0:
                batch_numbers = 2
            elif electron_energy <= 100.0:
                batch_numbers = 4
            else:
                batch_numbers = 8

            for batch_num in range(batch_numbers): 
                args.append((electron_energy, current, batch_num + 1))
    pool.map(run_parallel_sims, args)
    pool.close()
    pool.join()


if __name__ == '__main__':
    get_radial_distributions()

