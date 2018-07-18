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


def process_results(I, seed):
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
    for current in I:
	    gummersall_results = np.loadtxt(os.path.join("data", "gummersall-current-{}-energy-100-redone.txt".format(int(current))), delimiter=",")
	    plt.scatter(gummersall_results[:, 0], gummersall_results[:, 1], label="gummersall-{}".format(current))

    plt.xlim([1e-8, 1e-5])
    plt.xlabel("Fraction of electrons remaining in Polywell")
    plt.xlabel("Time (s)")
    plt.title("Replication of Gummersall thesis plot")
    plt.legend()
    plt.savefig("gummersall_replication")
    plt.show()

if __name__ == '__main__':
	process_results([100.0, 1e3, 1e4], 1)