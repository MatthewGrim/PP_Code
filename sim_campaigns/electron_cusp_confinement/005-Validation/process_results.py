"""
Author: Rohan Ramasamy
Date: 18/09/2018

This script is used to process results from the replications of Gummersall's plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy


def process_fig2_results():

    plt.figure(figsize=(20, 10))

    I = [0.01, 0.1, 1.0, 10.0, 100.0]
    seed = 1
    radius = 0.1
    energy = 100.0
    output_dirs = ["results"]
    for output_dir in output_dirs:
        for current in I:
            res_dir = os.path.join(output_dir, "radius-{}m".format(radius), "current-{}kA".format(current))
            output_path = os.path.join(res_dir, "final_state-current-{}-radius-{}-energy-{}-batch-{}.txt".format(current * 1e3, radius, energy, seed))
            results = np.loadtxt(output_path)

            results = results[results[:, 0].argsort()]
            fraction_in_polywell = np.linspace(1.0, 0.0, results.shape[0])

            plt.semilogx(results[:, 0], fraction_in_polywell, label="current-{}".format(current))

    # Get data from Gummersall plots - digitised from thesis
    for current in I:
        gummersall_results = np.loadtxt(
            os.path.join("data", "gummersall-current-{}-energy-100.txt".format(int(current * 1e3))), delimiter=",")
        plt.scatter(gummersall_results[:, 0], gummersall_results[:, 1], label="gummersall-{}".format(current))

    plt.xlim([1e-8, 1e-5])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Time (s)")
    plt.ylabel("Fraction of electrons remaining in Polywell")
    plt.title("Replication of Gummersall Figure 2")
    plt.legend()
    plt.savefig("gummersall_fig2")
    plt.show()


def process_fig5_results():
    seed = 1
    radius = 1.0
    I = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    energy = 100.0

    # Load Gummersall data
    plt.figure(figsize=(20, 10))
    gummersall_fit = np.loadtxt(os.path.join("data", "gummersall-figure-5-sim-fit.txt"), delimiter=",")
    plt.semilogx(gummersall_fit[:, 0], gummersall_fit[:, 1])

    # Load sim results
    output_dirs = ["results"]
    for output_dir in output_dirs:
        t_means = []
        for current in I:
            res_dir = os.path.join(output_dir, "radius-{}m".format(radius), "current-{}kA".format(current))
            output_path = os.path.join(res_dir, "final_state-current-{}-radius-{}-energy-{}-batch-{}.txt".format(current * 1e3, radius, energy, seed))
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


def process_fig6_results(electron_energies):
    seed = 1
    radius = 1.0
    I = [10.0]
    electron_energies = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]

    plt.figure(figsize=(20, 10))

    # Load Gummersall data
    gummersall_fit = np.loadtxt(os.path.join("data", "fig6-sim-fit.csv"), delimiter=",")
    plt.loglog(gummersall_fit[:, 0], gummersall_fit[:, 1])

    # Load sim results
    output_dirs = ["results"]
    for output_dir in output_dirs:
        t_means = []
        for energy in electron_energies:
            res_dir = os.path.join(output_dir, "radius-{}m".format(radius), "current-{}kA".format(I))
            output_path = os.path.join(res_dir, "final_state-current-{}-radius-{}-energy-{}-batch-{}.txt".format(current * 1e3, radius, energy, seed))
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
    process_fig2_results()
    # process_fig5_results()
    # process_fig6_results()

