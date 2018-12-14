"""
Author: Rohan Ramasamy
Date: 17/10/2018

This script contains a multivariate regression model for the mean confinement time of particles. This is to be compared
to the work by Gummersall et al.
"""

import numpy as np
import scipy.optimize as optimize
import os


def fit_data():
    # Parameter space of full study
    radius = np.asarray([0.1, 1.0, 5.0, 10.0])
    current = np.asarray([1e3, 5e3, 1e4, 5e4, 1e5])
    energies = np.asarray([5.0, 10.0, 50.0, 1e2, 5e2])

    # Load mean confinement times
    res_dir = "results_long"
    mean_confinement_times = np.zeros((radius.shape[0], current.shape[0], energies.shape[0]))
    for i, r in enumerate(radius):
        file_name = "mean_confinement_times_{}m".format(r)
        file = os.path.join(res_dir, file_name)
        mean_confinement_times[i, :, :] = np.loadtxt(file)

    # Define approximate function according to theory
    def func(X, a, b, c, d):
        I, R, K = X
        return a * I ** b * R ** c * K ** d

    # Set data up as a 1D array for use with scipy optimize
    radii_unraveled = list()
    currents_unraveled = list()
    energies_unraveled = list()
    confinement_times = list()
    for i in range(mean_confinement_times.shape[0]):
        for j in range(mean_confinement_times.shape[1]):
            for k in range(mean_confinement_times.shape[2]):
                confinement_times.append(mean_confinement_times[i, j, k])
                radii_unraveled.append(radius[i])
                currents_unraveled.append(current[j])
                energies_unraveled.append(energies[k])
    radii_unraveled = np.asarray(radii_unraveled)
    currents_unraveled = np.asarray(currents_unraveled)
    energies_unraveled = np.asarray(energies_unraveled)
    confinement_times = np.asarray(confinement_times)

    # Get curve fit and error
    X = [currents_unraveled, radii_unraveled, energies_unraveled]
    p0 = 3.7e-7, 0.5, 1.5, -0.75
    curve = optimize.curve_fit(func, X, confinement_times, p0)
    print("Parameters: {}".format(curve[0]))
    print("Covariance: {}".format(curve[1]))


if __name__ == '__main__':
    fit_data()

