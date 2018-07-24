"""
Author: Rohan Ramasamy
Date: 20/07/2018

This script is used to get compare results from Gummersall and Khachan Physics of Plasmas 2013 figure 6 against simulations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def process_fig5_results(electron_energies):
    plt.figure(figsize=(20, 10))

    # Load Gummersall data
    gummersall_fit = np.loadtxt(os.path.join("data", "fig6-sim-fit.csv"), delimiter=",")
    plt.loglog(gummersall_fit[:, 0], gummersall_fit[:, 1])

    # Load sim results
    radius = 1.0
    output_dirs = ["results"]
    for output_dir in output_dirs:
        t_means = []
        for electron_energy in electron_energies:
            output_path = os.path.join(output_dir, "mean_confinement-10kA-1.0m-{}.txt".format(electron_energy))
            results = np.loadtxt(output_path)

            t_means.append(np.average(results[:, 0] * 1e6))

        plt.scatter(electron_energies, t_means, label="sim_results")

    plt.xlabel("Energy (eV)")
    plt.ylabel("Mean confinement time (microseconds)")
    plt.title("Replication of Gummersall Figure 6")
    plt.legend()
    plt.savefig("gummersall_fig6")
    plt.show()


if __name__ == '__main__':
	process_fig5_results([50.0, 100.0, 200.0, 500.0])

