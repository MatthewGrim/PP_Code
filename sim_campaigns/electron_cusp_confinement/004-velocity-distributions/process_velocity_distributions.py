"""
Author: Rohan Ramasamy
Date: 09/08/2018

This script is used to process results for velocity probability distributions
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def process_radial_locations(energies, radii, currents, plot_histograms=False):
    # Loop through simulations
    output_dirs = ["results"]
    for output_dir in output_dirs:
        for radius in radii:
            normalised_average_radii = np.zeros((len(currents), len(energies)))
            normalised_std_radii = np.zeros((len(currents), len(energies)))

            for k, I in enumerate(currents):
                escaped_ratios = list()
                if plot_histograms:
                    fig, ax = plt.subplots(4, figsize=(20, 10))

                for j, energy in enumerate(energies):
                    position_name = "radial_distribution-current-{}-radius-{}-energy-{}".format(I, radius, energy)
                    velocity_name = "velocity_distribution-current-{}-radius-{}-energy-{}".format(I, radius, energy)
                    state_name = "final_state-current-{}-radius-{}-energy-{}".format(I, radius, energy)
                    data_dir = os.path.join(output_dir, "radius-{}m".format(radius), "current-{}kA".format(I * 1e-3))
                    dir_files = os.listdir(data_dir)

                    # Read results from batches
                    radial_bins = None
                    radial_numbers = None
                    velocity_bins = None
                    velocity_numbers = None
                    final_state_results = np.zeros((2,))
                    for file in dir_files:
                        if os.path.isfile(os.path.join(data_dir, file)):
                            output_path = os.path.join(data_dir, file)
                            new_results = np.loadtxt(output_path)
                            # Load radial positions
                            if position_name in file:
                                radial_bins = new_results[0, :] if radial_bins is None else radial_bins
                                radial_numbers = new_results[1, :] if radial_numbers is None else radial_numbers + new_results[1, :]

                            # Load velocity distributions
                            if velocity_name in file:
                                velocity_bins = new_results if velocity_bins is None else velocity_bins
                                velocity_numbers += new_results

                            # Load final state
                            if state_name in file:
                                final_state_results[0] += np.sum(new_results[:, 4])
                                final_state_results[1] += new_results.shape[0]

                    # Get radial probabilities and plot position histograms
                    samples = np.sum(radial_numbers)
                    radial_probabilities = radial_numbers / samples
                    velocity_probabilities = velocity_numbers / np.sum(velocity_numbers)
                    if plot_histograms:
                        ax[0].plot(radial_bins, radial_probabilities, label="energy-{}-samples-{}".format(energy, samples * 1e-6))
                        ax[1].plot(velocity_bins / velocity_bins[-1], velocity_probabilities[0, :], label="energy-{}-samples-{}".format(energy, samples * 1e-6))
                        ax[2].plot(velocity_bins / velocity_bins[-1], velocity_probabilities[1, :], label="energy-{}-samples-{}".format(energy, samples * 1e-6))
                        ax[3].plot(velocity_bins / velocity_bins[-1], velocity_probabilities[2, :], label="energy-{}-samples-{}".format(energy, samples * 1e-6))

                    # Get average radial location
                    normalised_radii = radial_bins / radius
                    normalised_average_radius = np.sum(normalised_radii * radial_probabilities)
                    normalised_std_radius = np.sum(normalised_radii ** 2 * radial_probabilities) - normalised_average_radius ** 2
                    normalised_average_radii[k, j] = normalised_average_radius
                    normalised_std_radii[k, j] = normalised_std_radius

                    # Collect escaped particle ratios
                    escaped_ratios.append(final_state_results[0] / final_state_results[1])

                if plot_histograms:
                    ax[0].set_title("Radial position of different energy levels for {}m device operating at {}kA".format(radius, I * 1e-3))
                    ax[0].set_ylabel("Probability of Position")
                    ax[0].set_xlabel("Radial Position (m)")
                    ax[0].legend()

                    ax[1].set_title("Velocity distribution of different energy levels for {}m device operating at {}kA".format(radius, I * 1e-3))
                    ax[1].set_ylabel("Probability of Velocity X")
                    ax[1].set_xlabel("Velocity X")
                    ax[1].legend()

                    ax[2].set_title("Velocity distribution of different energy levels for {}m device operating at {}kA".format(radius, I * 1e-3))
                    ax[2].set_ylabel("Probability of Velocity Y")
                    ax[2].set_xlabel("Velocity Y")
                    ax[2].legend()

                    ax[3].set_title("Velocity distribution of different energy levels for {}m device operating at {}kA".format(radius, I * 1e-3))
                    ax[3].set_ylabel("Probability of Velocity Z")
                    ax[3].set_xlabel("Velocity Z")
                    ax[3].legend()

                    plt.savefig("radial_positions")
                    plt.show()

                    plt.figure()

                    plt.semilogx(energies, escaped_ratios)

                    plt.show()

            # Plot average radial distributions
            plt.figure()
            for j, energy in enumerate(energies):
                plt.errorbar(currents, normalised_average_radii[:, j], yerr=normalised_std_radii[:, j],
                             label="energy-{}-radius-{}".format(energy, radius))
            plt.xscale('log')
            plt.xlabel("Currents [kA]")
            plt.ylabel("Normalised average radius")
            plt.title("Average radial locations for {}m device".format(radius))
            plt.legend()
            plt.savefig("normalised_average_radii_{}.png".format(radius))
            plt.show()


if __name__ == "__main__":
    radius = [1.0]
    current = [1e4]
    energies = [1.0, 10.0, 100.0, 1000.0, 1e4]
    process_radial_locations(energies, radius, current, plot_histograms=True)

