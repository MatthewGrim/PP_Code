"""
Author: Rohan Ramasamy
Date: 09/08/2018

This script is used to process results for velocity probability distributions
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


def process_radial_locations(energies, radii, currents, limit_radius=False, plot_velocity_histograms=True):
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
                    radial_bins = None
                    radial_numbers = None
                    velocity_bins = None
                    v_x_numbers = None
                    v_y_numbers = None
                    v_z_numbers = None
                    final_state_results = np.zeros((2,))
                    confinement_time_sum = 0.0
                    for file in dir_files:
                        if os.path.isfile(os.path.join(data_dir, file)):
                            output_path = os.path.join(data_dir, file)
                            # Load radial positions
                            if position_name in file:
                                new_results = np.loadtxt(output_path)
                                radial_bins = new_results[0, :] if radial_bins is None else radial_bins
                                radial_numbers = new_results[1, :] if radial_numbers is None else radial_numbers + new_results[1, :]

                            # Load velocity distributions
                            if velocity_name in file:
                                if "_x" in file:
                                    v_x = np.loadtxt(output_path)
                                    v_x_numbers = v_x if v_x_numbers is None else v_x + v_x_numbers

                                    vel = np.sqrt(2.0 * energy * PhysicalConstants.electron_charge / PhysicalConstants.electron_mass)
                                    velocity_bins = np.linspace(-vel, vel, v_x.shape[1])
                                elif "_y" in file:
                                    v_y = np.loadtxt(output_path)
                                    v_y_numbers = v_y if v_y_numbers is None else v_y + v_y_numbers
                                elif "_z" in file:
                                    v_z = np.loadtxt(output_path)
                                    v_z_numbers = v_z if v_z_numbers is None else v_z + v_z_numbers
                                else:
                                    v_tot = np.loadtxt(output_path)
                                    v_tot_numbers = v_tot if v_tot_numbers is None else v_tot + v_tot_numbers

                            # Load final state
                            if state_name in file:
                                new_results = np.loadtxt(output_path)
                                final_state_results[0] += np.sum(new_results[:, 4])
                                final_state_results[1] += new_results.shape[0]
                                confinement_time_sum += np.sum(new_results[:, 0])

                    # Limit radial distance to 1.5 times coil radius
                    if limit_radius:
                        indices = np.logical_and(0.02 * radius < radial_bins, radial_bins < 1.0 * radius)
                        radial_bins = radial_bins[indices]
                        v_x_numbers = v_x_numbers[indices, :]
                        v_y_numbers = v_y_numbers[indices, :]
                        v_z_numbers = v_z_numbers[indices, :]

                    # --- Print number of samples ---
                    num_samples = final_state_results[1]
                    escaped_ratio = final_state_results[0] / final_state_results[1]
                    # print("Number of samples: {}".format(np.sum(v_x_numbers) * 1e-6))

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
                        fig, ax = plt.subplots(5, figsize=(10, 10), sharex=True)
                        ax[0].contourf(radial_bins, velocity_bins, v_x_numbers.transpose() / max_in_x, 100)
                        ax[0].set_ylabel("v_r")
                        ax[1].contourf(radial_bins, velocity_bins, v_y_numbers.transpose() / max_in_y, 100)
                        ax[1].set_ylabel("v_ort_1")
                        ax[2].contourf(radial_bins, velocity_bins, v_z_numbers.transpose() / max_in_z, 100)
                        ax[2].set_ylabel("v_ort_2")
                        ax[3].plot(radial_bins, radial_numbers / np.max(radial_numbers))
                        ax[3].set_xlim([radial_bins[0], radial_bins[-1]])
                        ax[3].set_ylabel("Radial Distribution")
                        ax[4].semilogy(radial_bins, num_per_v_x)
                        ax[4].set_xlim([radial_bins[0], radial_bins[-1]])
                        ax[4].set_ylabel("Sample size")

                        fig.suptitle("Histograms for {}eV electron in a {}m device at {}kA - {}% Escaped from {} particles".format(energy, radius, I * 1e-3,
                                                                                                                                   round(escaped_ratio * 100.0, 2),
                                                                                                                                   num_samples))
                        result_name = "histogram_results-{}-{}-{}.png".format(radius, energy, I * 1e-3)
                        plt.savefig(os.path.join(data_dir, result_name))
                        plt.show()

                    # --- Get average radial location ---
                    radial_probabilities = radial_numbers / np.sum(radial_numbers)
                    normalised_radii = radial_bins / radius
                    normalised_average_radius = np.sum(normalised_radii * radial_probabilities)
                    normalised_std_radius = np.sum(normalised_radii ** 2 * radial_probabilities) - normalised_average_radius ** 2
                    normalised_average_radii[j, k] = normalised_average_radius
                    normalised_std_radii[j, k] = normalised_std_radius

            # --- Plot average radial distributions ---
            plt.figure()
            for k, energy in enumerate(energies):
                plt.errorbar(currents, normalised_average_radii[:, k], yerr=normalised_std_radii[:, k],
                             label="energy-{}-radius-{}".format(energy, radius))
            plt.xscale('log')
            plt.xlabel("Currents [kA]")
            plt.ylabel("Normalised average radius")
            plt.title("Average radial locations for {}m device".format(radius))
            plt.legend()
            plt.savefig("normalised_average_radii_{}.png".format(radius))
            plt.show()

    for j, current in enumerate(currents):
        # --- Plot mean confinement times ---
        plt.figure()
        for k, energy in enumerate(energies):
            plt.plot(radii, mean_confinement_times[:, j, k] / np.max(mean_confinement_times[:, j, k]), label="energy-{}eV".format(energy))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Radius [m]')
        plt.ylabel('Normalised mean confinement time')
        plt.title('Mean confinement times for {}kA device'.format(current))
        plt.legend()
        # plt.savefig('mean_confinement_time_{}.png'.format(radius))
        plt.show()


if __name__ == "__main__":
    radius = [1.0, 10.0]
    current = [1e3, 1e4, 1e5]
    energies = [1.0, 10.0, 100.0, 1000.0]
    process_radial_locations(energies, radius, current, plot_velocity_histograms=False)

