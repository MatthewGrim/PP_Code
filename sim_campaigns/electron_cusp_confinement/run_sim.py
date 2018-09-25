"""
Author: Rohan Ramasamy
Date: 08/08/2018

This script contains the main function to run single particle simulations using the non-relativistic boris solver in
this simulation campaign
"""

import os
import numpy as np
from matplotlib import pyplot as plt


from plasma_physics.pysrc.simulation.pic.algo.fields.magnetic_fields.generic_b_fields import InterpolatedBField
from plasma_physics.pysrc.simulation.pic.algo.particle_pusher.boris_solver import boris_solver_internal
from plasma_physics.pysrc.simulation.pic.algo.geometry.vector_ops import magnitude
from plasma_physics.pysrc.simulation.pic.data.particles.charged_particle import PICParticle
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


def run_simulation(params):
    b_field, particle, radius, domain_size, I, dI_dt = params
    print_output = False
    plot_sim = False

    # There is no E field in the simulations
    def e_field(x):
        B = b_field.b_field(x)
        dB_dt = B / I * dI_dt
        return -dB_dt

    def b_field_func(x):
        B = b_field.b_field(x / radius)
        B *= I / radius
        return B

    X = particle.position
    V = particle.velocity
    Q = np.asarray([particle.charge])
    M = np.asarray([particle.mass])

    # Set timestep according to Gummersall approximation
    max_dt = 1e-9 * radius
    min_dt = 1e-3 * max_dt
    final_time = 1e5 * max_dt
    max_steps = int(1e7)

    times = []
    positions = []
    velocities = []
    t = 0.0
    ts = 0
    times.append(t)
    positions.append(X)
    velocities.append(V)
    # Calculate fields
    while t < final_time and ts < max_steps:
        if print_output:
            print(t / final_time)

        # Get fields
        E = e_field(X)
        B = b_field_func(X)

        # Calculate time step
        dt = 0.2 * particle.mass / (magnitude(B[0]) * particle.charge)
        dt = min(max_dt, dt)
        dt = max(min_dt, dt)

        # Update time step
        ts += 1
        t += dt

        # Move particles
        x, v = boris_solver_internal(E, B, X, V, Q, M, dt)

        if np.any(x[0, :] < -domain_size) or np.any(x[0, :] > domain_size):
            if print_output:
                print("PARTICLE ESCAPED! - {}, {}, {}".format(ts, t, X[0]))

            x = np.asarray(positions)[:, :, 0].flatten()
            y = np.asarray(positions)[:, :, 1].flatten()
            z = np.asarray(positions)[:, :, 2].flatten()
            v_x = np.asarray(velocities)[:, :, 0].flatten()
            v_y = np.asarray(velocities)[:, :, 1].flatten()
            v_z = np.asarray(velocities)[:, :, 2].flatten()

            if plot_sim:
                # Plot 3D motion
                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot('111', projection='3d')
                ax.plot(x, y, z, label='numerical')
                ax.set_xlim([-1.25 * radius, 1.25 * radius])
                ax.set_ylim([-1.25 * radius, 1.25 * radius])
                ax.set_zlim([-1.25 * radius, 1.25 * radius])
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.legend(loc='best')
                ax.set_title("Analytic and Numerical Particle Motion")
                plt.show()

            return times, x, y, z, v_x, v_y, v_z, True

        times.append(t)
        positions.append(x)
        velocities.append(v)
        X = x
        V = v

    # Convert points to x, y and z locations
    x = np.asarray(positions)[:, :, 0].flatten()
    y = np.asarray(positions)[:, :, 1].flatten()
    z = np.asarray(positions)[:, :, 2].flatten()
    v_x = np.asarray(velocities)[:, :, 0].flatten()
    v_y = np.asarray(velocities)[:, :, 1].flatten()
    v_z = np.asarray(velocities)[:, :, 2].flatten()

    if plot_sim:
        # Plot 3D motion
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot('111', projection='3d')
        ax.plot(x, y, z, label='numerical')
        ax.set_xlim([-1.25 * radius, 1.25 * radius])
        ax.set_ylim([-1.25 * radius, 1.25 * radius])
        ax.set_zlim([-1.25 * radius, 1.25 * radius])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='best')
        ax.set_title("Analytic and Numerical Particle Motion")
        plt.show()

    return times, x, y, z, v_x, v_y, v_z, False


def run_parallel_sims(params):
    radius, electron_energy, I, batch_num, get_final_state, get_histograms = params
    assert get_final_state or get_histograms
    dI_dt = 0.0
    to_kA = 1e-3
    use_cartesian_reference_frame = False

    # Get output directory
    res_dir = "results_low_loop_res_25"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    output_dir = os.path.join(res_dir, "radius-{}m".format(radius))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir, "current-{}kA".format(I * to_kA))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get process name
    process_name = "radius-{}m-energy-{}eV-current-{}kA-batch-{}".format(radius, electron_energy, I * to_kA, batch_num)
    print("Starting process: {}".format(process_name))

    # Generate Polywell field
    loop_pts = 200
    domain_pts = 130
    loop_offset = 1.25
    dom_size = 1.1 * loop_offset * 1.0
    file_name = "b_field_{}_{}_{}_{}_{}_{}".format(1.0 * to_kA, 1.0, loop_offset, domain_pts, loop_pts, dom_size)
    file_path = os.path.join("..", "mesh_generation", "data", "radius-1.0m", "current-0.001kA", "domres-{}".format(domain_pts), file_name)
    b_field = InterpolatedBField(file_path, dom_pts_idx=6, dom_size_idx=8)

    seed = batch_num
    np.random.seed(seed)

    # Run simulations
    num_radial_bins = 200
    num_velocity_bins = 250
    total_particle_position_count = np.zeros((num_radial_bins,))
    total_particle_velocity_count_x = np.zeros((num_radial_bins, num_velocity_bins - 1))
    total_particle_velocity_count_y = np.zeros((num_radial_bins, num_velocity_bins - 1))
    total_particle_velocity_count_z = np.zeros((num_radial_bins, num_velocity_bins - 1))
    radial_bins = np.linspace(0.0, np.sqrt(3) * loop_offset * radius, num_radial_bins)
    vel = np.sqrt(2.0 * electron_energy * PhysicalConstants.electron_charge / PhysicalConstants.electron_mass)
    velocity_bins = np.linspace(-vel, vel, num_velocity_bins)
    num_sims = 420
    final_positions = []
    for i in range(num_sims):
        # Define particle velocity
        z_unit = np.random.uniform(-1.0, 1.0)
        xy_plane = np.sqrt(1 - z_unit ** 2)
        phi = np.random.uniform(0.0, 2 * np.pi)
        velocity = np.asarray([xy_plane * np.cos(phi), xy_plane * np.sin(phi), z_unit]) * vel

        # Generate particle position
        z_unit = np.random.uniform(-1.0, 1.0)
        xy_plane = np.sqrt(1 - z_unit ** 2)
        phi = np.random.uniform(0.0, 2 * np.pi)
        position = np.asarray([xy_plane * np.cos(phi), xy_plane * np.sin(phi), z_unit]) * np.random.uniform(0.0, 3.0 * radius / 16.0)

        # Generate particle
        particle = PICParticle(9.1e-31, 1.6e-19, position, velocity)

        # Run simulation
        t, x, y, z, v_x, v_y, v_z, escaped = run_simulation((b_field, particle, radius, loop_offset * radius, I, dI_dt))

        # Save final position output
        if get_final_state:
            final_positions.append([t[-1], x[-1], y[-1], z[-1], escaped])

        # Change coordinate system
        if get_histograms:
            radial_position = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            if use_cartesian_reference_frame:
                particle_position_count, particle_velocity_count = get_particle_count(radial_bins, velocity_bins, radial_position, v_x, v_y, v_z)
            else:
                r_unit = np.zeros((3, x.shape[0]))
                r_unit[0, :] = x
                r_unit[1, :] = y
                r_unit[2, :] = z
                r_unit /= np.sqrt(x ** 2 + y ** 2 + z ** 2)

                xy_unit = np.zeros((3, x.shape[0]))
                xy_unit[0, :] = x
                xy_unit[1, :] = y
                xy_unit /= np.sqrt(np.sum(xy_unit ** 2, axis=0))

                latitude_unit = np.zeros(xy_unit.shape)
                latitude_unit[0] = xy_unit[1, :]
                latitude_unit[1] = -xy_unit[0, :]
                latitude_unit[2] = 0.0

                longitude_unit = np.zeros((3, x.shape[0]))
                longitude_unit[0, :] = r_unit[1, :] * latitude_unit[2, :] - r_unit[2] * latitude_unit[1]
                longitude_unit[1, :] = r_unit[2, :] * latitude_unit[0, :] - r_unit[0] * latitude_unit[2]
                longitude_unit[2, :] = r_unit[0, :] * latitude_unit[1, :] - r_unit[1] * latitude_unit[0]

                v_r = v_x * r_unit[0, :] + v_y * r_unit[1, :] + v_z * r_unit[2, :]
                v_lat = v_x * latitude_unit[0, :] + v_y * latitude_unit[1, :] + v_z * latitude_unit[2, :]
                v_long = v_x * longitude_unit[0, :] + v_y * longitude_unit[1, :] + v_z * longitude_unit[2, :]

                particle_position_count, particle_velocity_count = get_particle_count(radial_bins, velocity_bins, radial_position, v_r, v_lat, v_long)

            # Get probability of electron in radial spacings in sim
            total_particle_position_count += particle_position_count
            total_particle_velocity_count_x += particle_velocity_count[0, :, :]
            total_particle_velocity_count_y += particle_velocity_count[1, :, :]
            total_particle_velocity_count_z += particle_velocity_count[2, :, :]

    # Save results to file
    if get_histograms:
        position_output_path = os.path.join(output_dir, "radial_distribution-current-{}-radius-{}-energy-{}-batch-{}.txt".format(I, radius, electron_energy, batch_num))
        velocity_output_path = os.path.join(output_dir, "velocity_distribution-current-{}-radius-{}-energy-{}-batch-{}".format(I, radius, electron_energy, batch_num))
        np.savetxt(position_output_path, np.stack((radial_bins, total_particle_position_count)))
        np.savetxt("{}_x.txt".format(velocity_output_path), total_particle_velocity_count_x)
        np.savetxt("{}_y.txt".format(velocity_output_path), total_particle_velocity_count_y)
        np.savetxt("{}_z.txt".format(velocity_output_path), total_particle_velocity_count_z)
    if get_final_state:
        final_state_output_path = os.path.join(output_dir, "final_state-current-{}-radius-{}-energy-{}-batch-{}.txt".format(I, radius, electron_energy, batch_num))
        np.savetxt(final_state_output_path,  np.asarray(final_positions))

    print("Finished process: {}".format(process_name))


def get_particle_count(radial_bins, velocity_bins, radial_positions, v_x, v_y, v_z):
    """
    This function counts the particles in each radial bin for location and velocity.
    """
    position_count = np.zeros(radial_bins.shape)
    velocity_count = np.zeros((3, radial_bins.shape[0], velocity_bins.shape[0] - 1))
    for i, bin_max in enumerate(radial_bins):
        if i == 0.0:
            continue

        bin_min = radial_bins[i - 1]
        points_in_range = np.where(np.logical_and(radial_positions >= bin_min, radial_positions < bin_max))
        position_count[i - 1] = points_in_range[0].shape[0]

        x_values = v_x[points_in_range]
        y_values = v_y[points_in_range]
        z_values = v_z[points_in_range]
        for j, v_bin_max in enumerate(velocity_bins):
            if j == 0.0:
                continue

            v_bin_min = velocity_bins[j - 1]
            x_points_in_range = np.where(np.logical_and(x_values >= v_bin_min, x_values < v_bin_max))
            y_points_in_range = np.where(np.logical_and(y_values >= v_bin_min, y_values < v_bin_max))
            z_points_in_range = np.where(np.logical_and(z_values >= v_bin_min, z_values < v_bin_max))

            velocity_count[0, i, j - 1] = x_points_in_range[0].shape[0]
            velocity_count[1, i, j - 1] = y_points_in_range[0].shape[0]
            velocity_count[2, i, j - 1] = z_points_in_range[0].shape[0]

    return position_count, velocity_count