"""
Author: Rohan Ramasamy
Date: 09/08/2018

This script is used to process results for velocity probability distributions
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


def process_radial_locations(energies, radii, currents, number_densities,
                             plot_velocity_histograms=True):
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
                    for m, n in enumerate(number_densities):
                        post_name = "current-{}-radius-{}-energy-{}-n-{:.2E}".format(I, radius, energy, n)
                        position_name = "radial_distribution-{}".format(post_name)
                        velocity_name = "velocity_distribution-{}".format(post_name)
                        state_name = "final_state-{}".format(post_name)
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

                        # assert len(position_files) == len(velocity_files_x) == len(velocity_files_y) == len(velocity_files_z) == len(state_files)
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
                        for count, file in enumerate(state_files):
                            new_results = np.loadtxt(file)
                            final_state_results[0] += np.sum(new_results[:, 4])
                            final_state_results[1] += new_results.shape[0]
                            confinement_time_sum += np.sum(new_results[:, 0])

                        # --- Print number of samples ---
                        num_samples = final_state_results[1]
                        escaped_ratio = final_state_results[0] / final_state_results[1]
                        print("Number of samples for {}m {}A {}eV {:.2E}: {}".format(radius, I, energy, n, np.sum(v_x_numbers) * 1e-6))

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

                            # Plot histograms for each radial point in the distribution
                            fig, ax = plt.subplots(5, figsize=(10, 10), sharex='col')

                            v_x_normalised = v_x_numbers.transpose() / max_in_x
                            ax[0].contourf(radial_bins, velocity_bins, v_x_normalised, 100)
                            ax[0].set_ylabel("v_r")

                            v_y_normalised = v_y_numbers.transpose() / max_in_y
                            ax[1].contourf(radial_bins, velocity_bins, v_y_normalised, 100)
                            ax[1].set_ylabel("v_ort_1")

                            v_z_normalised = v_z_numbers.transpose() / max_in_z
                            ax[2].contourf(radial_bins, velocity_bins, v_z_normalised, 100)
                            ax[2].set_ylabel("v_ort_2")

                            radial_probabilities = radial_numbers / np.max(radial_numbers)
                            ax[3].plot(radial_bins, radial_probabilities)
                            ax[3].set_xlim([radial_bins[0], radial_bins[-1]])
                            ax[3].set_ylabel("Radial Distribution")

                            ax[4].semilogy(radial_bins, num_per_v_x)
                            ax[4].set_xlim([radial_bins[0], radial_bins[-1]])
                            ax[4].set_xlabel("Radial Location [m]")
                            ax[4].set_ylabel("Sample size")

                            fig.suptitle("Histograms for {}eV electron in a {}m device at {}kA - {}% Escaped from {} particles".format(energy, radius, I * 1e-3, round(escaped_ratio * 100.0, 2), num_samples))
                            result_name = "histogram_results-{}.png".format(post_name)
                            plt.savefig(os.path.join(data_dir, result_name))

                        # --- Get average radial location ---
                        radial_probabilities = radial_numbers / np.sum(radial_numbers)
                        normalised_radii = radial_bins / radius
                        normalised_average_radius = np.sum(normalised_radii * radial_probabilities)
                        normalised_std_radius = np.sum(normalised_radii ** 2 * radial_probabilities) - normalised_average_radius ** 2
                        normalised_average_radii[j, k] = normalised_average_radius
                        normalised_std_radii[j, k] = normalised_std_radius


if __name__ == "__main__":
    radius = [1.0]
    current = [1e4, 1e5]
    energies = [100.0]
    number_densities = [0.0, 1e3, 1e6, 1e9, 1e12]
    process_radial_locations(energies, radius, current, number_densities,
                             plot_velocity_histograms=True)

