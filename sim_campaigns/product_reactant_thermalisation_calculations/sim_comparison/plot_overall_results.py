"""
Author: Rohan Ramasamy
Date: 20/06/2018

This script contains code to generate simulation results from comparison against theoretical approximations
"""


import numpy as np
from matplotlib import pyplot as plt
import os


def plot_sim_results(number_densities, T):
    N = int(1e4)
    dt_factor = 0.01

    # Plot results
    plt.figure()
    sim_names = ["stationary", "maxwellian"]
    species_names = ["product"]
    for i, sim_name in enumerate(sim_names):
        for j, species_name in enumerate(species_names):
            file_name = "{}_{}_{}_{}_half_times".format(sim_name, species_name, N, dt_factor)
            file_directory = sim_name

            path = os.path.join(file_directory, file_name) 
            if os.path.exists(path):
                t_halves = np.loadtxt(path)

            plt.semilogx(number_densities, t_halves, label="{}-{}".format(sim_name, species_name))


    plt.title("Comparison of thermalisation rates of products and reactants in different plasmas")
    plt.legend()
    plt.savefig("energy_half_times")
    plt.show()


if __name__ == '__main__':
    number_densities = np.logspace(15, 24, 9)
    temperature = 10000.0
    plot_sim_results(number_densities, temperature)

