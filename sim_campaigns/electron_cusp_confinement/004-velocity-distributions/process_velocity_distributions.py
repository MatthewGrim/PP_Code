"""
Author: Rohan Ramasamy
Date: 09/08/2018

This script is used to process results for velocity probability distributions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


def process_radial_locations(energies, radii, currents,
                             plot_velocity_histograms=True,
                             plot_normalised_radii=True,
                             plot_mean_confinement=True):
    # Loop through simulations
    output_dirs = ["results"]
    for output_dir in output_dirs:
        mean_confinement_times = np.zeros((len(radii), len(currents), len(energies)))
        for i, radius in enumerate(radii):
            # Output arrays for normalised average radii, its standard deviation and mean confinement time
            normalised_average_radii = np.zeros((len(currents), len(energies)))
            normalised_std_radii = np.zeros((len(currents), len(energies)))

            for j, I in enumerate(currents):
                for k, energy in enumerate(energies):
                    position_name = "radial_distribution-current-{}-radius-{}-energy-{}".format(I, radius, energy)
                    velocity_name = "velocity_distribution-current-{}-radius-{}-energy-{}".format(I, radius, energy)
                    state_name = "final_state-current-{}-radius-{}-energy-{}".format(I, radius, energy)
                    data_dir = os.path.join(output_dir, "radius-{}m".format(radius), "current-{}kA".format(I * 1e-3))
                    dir_files = os.listdir(data_dir)

                    # Read results from batches
                    position_files = []
                    velocity_files_x = []
                    velocity_files_y = []
                    velocity_files_z = []
                    state_files = []
                    for file in dir_files:
                        if os.path.isfile(os.path.join(data_dir, file)):
                            output_path = os.path.join(data_dir, file)
                            # Load radial positions
                            if position_name in file:
                                position_files.append(output_path)
                            elif velocity_name in file:
                                if "_x" in file:
                                    velocity_files_x.append(output_path)
                                elif "_y" in file:
                                    velocity_files_y.append(output_path)
                                elif "_z" in file:
                                    velocity_files_z.append(output_path)
                                else:
                                    raise RuntimeError("Should not be possible to get here!")
                            elif state_name in file:
                                state_files.append(output_path)
                            else:
                                pass

                    assert len(position_files) == len(velocity_files_x) == len(velocity_files_y) == len(velocity_files_z) == len(state_files)
                    num_batches = len(position_files)
                    # Assumes there are the same number of simulations per batch
                    half_results = num_batches // 2

                    # --- Load distributions ---
                    radial_bins = None
                    radial_numbers = None
                    velocity_bins = None
                    v_x_numbers = None
                    v_y_numbers = None
                    v_z_numbers = None
                    radial_numbers_half_set = None
                    v_x_numbers_half_set = None
                    v_y_numbers_half_set = None
                    v_z_numbers_half_set = None
                    final_state_results = np.zeros((2,))
                    confinement_time_sum = 0.0

                    # Load radial positions
                    for count, file in enumerate(position_files):
                        new_results = np.loadtxt(file)
                        radial_bins = new_results[0, :] if radial_bins is None else radial_bins
                        radial_numbers = new_results[1, :] if radial_numbers is None else radial_numbers + new_results[1, :]
                        radial_numbers_half_set = radial_numbers if count < half_results else radial_numbers_half_set

                    # Load velocity distributions
                    for count, file in enumerate(velocity_files_x):
                        v_x = np.loadtxt(file)
                        v_x_numbers = v_x if v_x_numbers is None else v_x + v_x_numbers
                        v_x_numbers_half_set = v_x_numbers if count < half_results else v_x_numbers_half_set

                        if velocity_bins is None:
                            vel = np.sqrt(2.0 * energy * PhysicalConstants.electron_charge / PhysicalConstants.electron_mass)
                            velocity_bins = np.linspace(-vel, vel, v_x.shape[1])

                    for count, file in enumerate(velocity_files_y):
                        v_y = np.loadtxt(file)
                        v_y_numbers = v_y if v_y_numbers is None else v_y + v_y_numbers
                        v_y_numbers_half_set = v_y_numbers if count < half_results else v_y_numbers_half_set

                    for count, file in enumerate(velocity_files_z):
                        v_z = np.loadtxt(file)
                        v_z_numbers = v_z if v_z_numbers is None else v_z + v_z_numbers
                        v_z_numbers_half_set = v_z if count < half_results else v_z_numbers_half_set

                    # Load final state
                    fig = plt.figure(figsize=(10, 10))
                    for count, file in enumerate(state_files):
                        # Load file and get statistics
                        new_results = np.loadtxt(file)
                        final_state_results[0] += np.sum(new_results[:, 4])
                        final_state_results[1] += new_results.shape[0]
                        confinement_time_sum += np.sum(new_results[:, 0])

                        # Plot escaped locations
                        final_x_position = new_results[:, 1]
                        final_y_position = new_results[:, 2]
                        final_z_position = new_results[:, 3]
                        escaped = np.where(new_results[:, 4] > 0.5)[0]
                        confined = np.where(new_results[:, 4] < 0.5)[0]
                        ax = fig.add_subplot('111', projection='3d')
                        ax.scatter(final_x_position[confined], final_y_position[confined], final_z_position[confined], c='b', label='confined')
                        ax.scatter(final_x_position[escaped], final_y_position[escaped], final_z_position[escaped], c='r', label='escaped')
                    # Get number of samples and escape ratio
                    num_samples = final_state_results[1]
                    escaped_ratio = final_state_results[0] / final_state_results[1]

                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    plt.title("Location of escaped electrons for {}eV electron in a {}m device at {}kA - {}% Escaped from {} particles".format(energy, radius, I * 1e-3, round(escaped_ratio * 100.0, 2), num_samples))
                    plot_name = "escape_locations-{}-{}-{}.png".format(radius, energy, I * 1e-3)
                    plt.savefig(os.path.join(data_dir, plot_name))
                    plt.close()

                    # --- Print number of samples ---
                    print("Number of samples: {}".format(np.sum(v_x_numbers) * 1e-6))
                    print("Escaped ratio: {}".format(escaped_ratio))

                    # --- Get mean confinement time ---
                    mean_confinement_time = confinement_time_sum / num_samples
                    mean_confinement_times[i, j, k] = mean_confinement_time

                    # --- Plot Histograms ---
                    if plot_velocity_histograms:
                        # Get number of samples per radial bin
                        num_per_v_x = np.sum(v_x_numbers, axis=1)
                        max_in_x = np.amax(v_x_numbers, axis=1)
                        max_in_y = np.amax(v_y_numbers, axis=1)
                        max_in_z = np.amax(v_z_numbers, axis=1)
                        num_per_v_x_half = np.sum(v_x_numbers_half_set, axis=1)
                        max_in_x_half = np.amax(v_x_numbers_half_set, axis=1)
                        max_in_y_half = np.amax(v_y_numbers_half_set, axis=1)
                        max_in_z_half = np.amax(v_z_numbers_half_set, axis=1)

                        # Plot histograms for each radial point in the distribution
                        fig, ax = plt.subplots(5, 2, figsize=(10, 10), sharex='col')

                        v_x_normalised = v_x_numbers.transpose() / max_in_x
                        v_x_half_normalised = v_x_numbers_half_set.transpose() / max_in_x_half
                        v_x_error = np.abs(v_x_normalised - v_x_half_normalised) / v_x_normalised
                        ax[0, 0].contourf(radial_bins, velocity_bins, v_x_normalised, 100)
                        im = ax[0, 1].contourf(radial_bins, velocity_bins, v_x_error, 100)
                        fig.colorbar(im, ax=ax[0, 1])
                        ax[0, 0].set_ylabel("v_r")

                        v_y_normalised = v_y_numbers.transpose() / max_in_y
                        v_y_half_normalised = v_y_numbers_half_set.transpose() / max_in_y_half
                        v_y_error = np.abs(v_y_normalised - v_y_half_normalised) / v_y_normalised
                        ax[1, 0].contourf(radial_bins, velocity_bins, v_y_normalised, 100)
                        im = ax[1, 1].contourf(radial_bins, velocity_bins, v_y_error, 100)
                        fig.colorbar(im, ax=ax[1, 1])
                        ax[1, 0].set_ylabel("v_ort_1")

                        v_z_normalised = v_z_numbers.transpose() / max_in_z
                        v_z_half_normalised = v_z_numbers_half_set.transpose() / max_in_z_half
                        v_z_error = np.abs(v_z_normalised - v_z_half_normalised) / v_z_normalised
                        ax[2, 0].contourf(radial_bins, velocity_bins, v_z_normalised, 100)
                        im = ax[2, 1].contourf(radial_bins, velocity_bins, v_z_error, 100)
                        fig.colorbar(im, ax=ax[2, 1])
                        ax[2, 0].set_ylabel("v_ort_2")

                        radial_probabilities = radial_numbers / np.max(radial_numbers)
                        radial_half_probabilities = radial_numbers_half_set / np.max(radial_numbers_half_set)
                        radial_error = np.abs(radial_probabilities - radial_half_probabilities) / radial_probabilities
                        ax[3, 0].plot(radial_bins, radial_probabilities)
                        ax[3, 0].set_xlim([radial_bins[0], radial_bins[-1]])
                        ax[3, 1].plot(radial_bins, radial_error)
                        ax[3, 1].set_xlim([radial_bins[0], radial_bins[-1]])
                        ax[3, 0].set_ylabel("Radial Distribution")

                        ax[4, 0].semilogy(radial_bins, num_per_v_x)
                        ax[4, 0].set_xlim([radial_bins[0], radial_bins[-1]])
                        ax[4, 0].set_xlabel("Radial Location [m]")
                        ax[4, 1].semilogy(radial_bins, num_per_v_x / num_per_v_x_half)
                        ax[4, 1].set_xlim([radial_bins[0], radial_bins[-1]])
                        ax[4, 1].set_xlabel("Radial Location [m]")
                        ax[4, 0].set_ylabel("Sample size")

                        fig.suptitle("Histograms for {}eV electron in a {}m device at {}kA - {}% Escaped from {} particles".format(energy, radius, I * 1e-3,
                                                                                                                                   round(escaped_ratio * 100.0, 2),
                                                                                                                                   num_samples))
                        result_name = "histogram_results-{}-{}-{}.png".format(radius, energy, I * 1e-3)
                        plt.savefig(os.path.join(data_dir, result_name))
                        plt.close()

                    # --- Get average radial location ---
                    radial_probabilities = radial_numbers / np.sum(radial_numbers)
                    normalised_radii = radial_bins / radius
                    normalised_average_radius = np.sum(normalised_radii * radial_probabilities)
                    normalised_std_radius = np.sum(normalised_radii ** 2 * radial_probabilities) - normalised_average_radius ** 2
                    normalised_average_radii[j, k] = normalised_average_radius
                    normalised_std_radii[j, k] = normalised_std_radius

            # --- Plot average confinement times ---
            if plot_normalised_radii:
                plt.figure()
                for k, energy in enumerate(energies):
                    plt.errorbar(currents, normalised_average_radii[:, k], yerr=normalised_std_radii[:, k],
                                 label="energy-{}-radius-{}".format(energy, radius))
                plt.xscale('log')
                plt.xlabel("Currents [kA]")
                plt.ylabel("Normalised average radius")
                plt.title("Average radial locations for {}m device".format(radius))
                plt.legend()
                plt.savefig(os.path.join(output_dir, "normalised_average_radii_{}.png".format(radius)))

        if plot_mean_confinement:
            for j, current in enumerate(currents):
                # --- Plot mean confinement times ---
                plt.figure()
                for k, energy in enumerate(energies):
                    plt.plot(radii, mean_confinement_times[:, j, k], label="energy-{}eV".format(energy))
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('Radius [m]')
                plt.ylabel('Normalised mean confinement time')
                plt.title('Mean confinement times for {}kA device'.format(current))
                plt.legend()
                plt.savefig(os.path.join(output_dir, 'mean_confinement_time_{}.png'.format(current)))


if __name__ == "__main__":
    radius = [0.1, 1.0, 5.0, 10.0]
    current = [1e3, 1e4, 1e5]
    energies = [1.0, 10.0, 100.0, 1000.0]
    process_radial_locations(energies, radius, current,
                             plot_velocity_histograms=False, plot_normalised_radii=False, plot_mean_confinement=False)

