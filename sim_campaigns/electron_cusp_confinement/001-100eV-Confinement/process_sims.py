"""
Author: Rohan Ramasamy
Date: 16/07/2018

This script is used to process results from the replications of Gummersall's plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy


def process_fig2_results(I, seed):
    plt.figure(figsize=(20, 10))

    radius = 0.1
    output_dirs = ["results"]
    for output_dir in output_dirs:
        for current in I:
            output_path = os.path.join(output_dir, "final_positions-current-{}-radius-{}-seed-{}.txt".format(current, radius, seed))
            results = np.loadtxt(output_path)

            results = results[results[:, 0].argsort()]
            fraction_in_polywell = np.linspace(1.0, 0.0, results.shape[0])
            
            plt.semilogx(results[:, 0], fraction_in_polywell, label="current-{}-{}".format(current, output_dir))

    # Get data from Gummersall plots - digitised from thesis
    for current in [100.0, 1000.0, 1e4, 1e5]:
        gummersall_results = np.loadtxt(os.path.join("data", "gummersall-current-{}-energy-100.txt".format(int(current))), delimiter=",")
        plt.scatter(gummersall_results[:, 0], gummersall_results[:, 1], label="gummersall-{}".format(current))

    plt.xlim([1e-8, 1e-5])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Time (s)")
    plt.ylabel("Fraction of electrons remaining in Polywell")
    plt.title("Replication of Gummersall Figure 2")
    plt.legend()
    plt.savefig("gummersall_fig2")
    plt.show()


def process_fig5_results(I, seed):
    plt.figure(figsize=(20, 10))

    # Load Gummersall data
    gummersall_fit = np.loadtxt(os.path.join("data", "gummersall-figure-5-sim-fit.txt"), delimiter=",")
    plt.semilogx(gummersall_fit[:, 0], gummersall_fit[:, 1])

    # Load sim results
    radius = 1.0
    output_dirs = ["results"]
    for output_dir in output_dirs:
        t_means = []
        for current in I:
            output_path = os.path.join(output_dir, "final_positions-current-{}-radius-{}-seed-{}.txt".format(current, radius, seed))
            results = np.loadtxt(output_path)

            t_means.append(np.average(results[:, 0]))

        plt.scatter(I, t_means, label="sim_results")

    # plt.xlim([100.0, 2e4])
    # plt.ylim([0.0, 5.0])
    plt.xlabel("Current (A)")
    plt.ylabel("Mean confinement time (microseconds)")
    plt.title("Replication of Gummersall Figure 5")
    plt.legend()
    plt.savefig("gummersall_fig5")
    plt.show()


if __name__ == '__main__':
    I = [100.0, 1e3, 1e4]
    seed = 1

    # process_fig2_results(I, seed)
    process_fig5_results(I, seed)