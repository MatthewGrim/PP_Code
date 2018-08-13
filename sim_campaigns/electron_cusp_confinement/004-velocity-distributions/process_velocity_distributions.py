"""
Author: Rohan Ramasamy
Date: 09/08/2018

This script is used to process results for velocity probability distributions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from vtk.util.numpy_support import numpy_to_vtk

from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


def process_radial_locations(energies, radii, currents):
    # Loop through simulations
    output_dirs = ["results"]
    for output_dir in output_dirs:
        for radius in radii:
            for k, I in enumerate(currents):
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
                    v_x_numbers = None
                    v_y_numbers = None
                    v_z_numbers = None
                    final_state_results = np.zeros((2,))
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
                                if "_y" in file:
                                    v_y = np.loadtxt(output_path)
                                    v_y_numbers = v_y if v_y_numbers is None else v_y + v_y_numbers
                                if "_z" in file:
                                    v_z = np.loadtxt(output_path)
                                    v_z_numbers = v_z if v_z_numbers is None else v_z + v_z_numbers

                            # Load final state
                            if state_name in file:
                                new_results = np.loadtxt(output_path)
                                final_state_results[0] += np.sum(new_results[:, 4])
                                final_state_results[1] += new_results.shape[0]

                    print(np.sum(v_x_numbers))
                    print(np.sum(v_y_numbers))
                    print(np.sum(v_z_numbers))

                    fig, ax = plt.subplots(3, figsize=(20, 10))
                    im = ax[0].contourf(velocity_bins, radial_bins, v_x_numbers / np.sum(v_x_numbers, axis=0), 100)
                    fig.colorbar(im, ax=ax[0])
                    im = ax[1].contourf(velocity_bins, radial_bins, v_y_numbers / np.sum(v_y_numbers, axis=0), 100)
                    fig.colorbar(im, ax=ax[1])
                    im = ax[2].contourf(velocity_bins, radial_bins, v_z_numbers / np.sum(v_z_numbers, axis=0), 100)
                    fig.colorbar(im, ax=ax[2])

                    plt.savefig("contours")
                    plt.show()


if __name__ == "__main__":
    radius = [1.0]
    current = [1e4]
    energies = [1000.0]
    process_radial_locations(energies, radius, current)

