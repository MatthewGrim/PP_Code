"""
Author: Rohan Ramasamy
Date: 16/07/2018

This script is used to process results for radial probability distributions
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
            for k, I in enumerate(currents):
                if plot_histograms:
                    plt.figure(figsize=(20, 10))

                for j, energy in enumerate(energies):
                    sim_type = "radial_distribution-current-{}-radius-{}-energy-{}".format(I, radius, energy)
                    data_dir = os.path.join(output_dir, "radius-{}m".format(radius), "current-{}kA".format(I * 1e-3))
                    dir_files = os.listdir(data_dir)

                    # Accumulate results from batches
                    total_results = []
                    for file in dir_files:
                        if os.path.isfile(os.path.join(data_dir, file)) and sim_type in file:
                            output_path = os.path.join(data_dir, file)
                            total_results.append(np.loadtxt(output_path))
                    radial_numbers = total_results[0][1, :]
                    for i, result in enumerate(total_results):
                        if i == 0:
                            continue
                        radial_numbers += result[1, :]

                    # Get radial probabilities and plot histograms
                    samples = np.sum(radial_numbers)
                    radial_probabilities = radial_numbers / np.sum(radial_numbers)
                    if plot_histograms:
                        plt.plot(total_results[0][0, :], radial_probabilities, label="energy-{}-samples-{}".format(energy, samples * 1e-6))

                    # Get average radial location
                    normalised_average_radius = np.sum(total_results[0][0, :] * radial_numbers) / (samples * radius)
                    normalised_average_radii[k, j] = normalised_average_radius

                if plot_histograms:
                    plt.title("Radial position of different energy levels for {}m device operating at {}kA".format(radius, I * 1e-3))
                    plt.ylabel("Probability of position")
                    plt.xlabel("Radial Position (m)")
                    plt.legend()
                    plt.savefig("radial_positions")
                    plt.show()

            # Plot average radial distributions
            plt.figure()
            for j, energy in enumerate(energies):
                plt.semilogx(currents, normalised_average_radii[:, j], label="energy-{}-radius-{}".format(energy, radius))
            plt.xlabel("Currents [kA]")
            plt.ylabel("Normalised average radius")
            plt.title("Average radial locations for {}m device".format(radius))
            plt.legend()
            plt.savefig("normalised_average_radii_{}.png".format(radius))
            plt.show()


if __name__ == "__main__":
    radius = [0.5, 1.0, 10.0]
    current = [100.0, 200.0, 500.0, 1e3, 5e3, 1e5]
    energies = [1.0, 10.0, 100.0, 1000.0, 1e4]
    process_radial_locations(energies, radius, current, plot_histograms=False)

