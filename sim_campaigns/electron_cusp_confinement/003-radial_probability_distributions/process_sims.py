"""
Author: Rohan Ramasamy
Date: 16/07/2018

This script is used to process results for radial probability distributions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy


def process_radial_locations(energies, radii, currents):
    plt.figure(figsize=(20, 10))

    output_dirs = ["results"]
    for output_dir in output_dirs:
        for radius in radii:
            for I in currents: 
                for energy in energies:
                    print(I, energy)
                    sim_type = "radial_distribution-current-{}-radius-{}-energy-{}".format(I, radius, energy)
                    data_dir = os.path.join(output_dir, "radius-{}m".format(radius), "current-{}kA".format(I * 1e-3))
                    dir_files = os.listdir(data_dir)

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


                    samples = np.sum(radial_numbers)
                    radial_probabilities = radial_numbers / np.sum(radial_numbers)
                    plt.plot(total_results[0][0, :], radial_probabilities, label="energy-{}-samples-{}".format(energy, samples * 1e-6))
                    
                    dx = total_results[0][0, 1] - total_results[0][0, 0]

                plt.title("Radial position of different energy levels for {}m device operating at {}kA".format(radius, I * 1e-3))
                plt.ylabel("Probability of position")
                plt.xlabel("Radial Position (m)")
                plt.legend()
                plt.savefig("radial_positions")
                plt.show()


if __name__ == "__main__":
    radius = [1.0]
    current = [100.0, 200.0, 500.0, 1e3, 5e3, 2e4, 1e5]
    energies = [0.1, 1.0, 10.0, 100.0, 1000.0, 1e4]
    process_radial_locations(energies, radius, current)

