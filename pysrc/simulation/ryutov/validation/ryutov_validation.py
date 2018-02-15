"""
Author: Rohan Ramasamy
Date: 31/05/17

This code is used to validate the implementation of the 1D ideal magnetic liner, by comparing it with a simple case from
the paper with similar parameters. The parameters will be slightly different so an exact match is not expected.
"""


import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate as interpolate

from Physical_Models.ryutov_model.ideal_thin_liner import integrate_liner_implosion


def get_experimental_data(num_interpolation_pts=400, plot_data=False):
    """
    Get tabulated data recording current pulse and radius profile through time from experiment
    :return:
    """
    radius = np.loadtxt("ryutov_radius.txt")
    current = np.loadtxt("ryutov_current_2.txt")

    r_interpolator = interpolate.interp1d(radius[:, 0], radius[:, 1])
    i_interpolator = interpolate.interp1d(current[:, 0], current[:, 1])

    times = np.linspace(0.01e-7, 1.49e-7, num_interpolation_pts)
    r_int = r_interpolator(times)
    i_int = i_interpolator(times)

    if plot_data:
        fig, ax = plt.subplots(2)
        ax[0].scatter(radius[:, 0], radius[:, 1])
        ax[0].plot(times, r_int)
        ax[0].set_xlim([radius[0, 0], radius[-1, 0]])
        ax[1].scatter(current[:, 0], current[:, 1])
        ax[1].plot(times, i_int)
        ax[1].set_xlim([current[0, 0], current[-1, 0]])
        plt.show()

    return times, r_int, i_int, i_interpolator


def compare_to_1d_model():
    """
    Function to run 1D model and compare output to the experimental data
    :return:
    """
    times_int, r_int, i_int, i_interpolator = get_experimental_data()

    R_0 = 2e-2
    times = np.linspace(0.0e-7, 1.5e-7, 1000)
    r, v, e_kin = integrate_liner_implosion(times, R_0, i_interpolator, R_Outer=1.00005*R_0)

    # Plot implosion variables
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle("Comparison with Experimental Data from Ryutov")
    ax[0, 0].plot(times_int, i_int)
    ax[0, 0].set_title("Current (A)")
    ax[0, 1].plot(times, r)
    ax[0, 1].plot(times_int, r_int)
    ax[0, 1].set_title("Radius (m)")
    ax[0, 1].set_ylim([0.0, 1.01 * R_0])
    ax[1, 0].plot(times, v)
    ax[1, 0].set_title("Velocity (m/s)")
    ax[1, 1].plot(times, e_kin)
    ax[1, 1].set_title("Kinetic Energy (J)")
    plt.show()

if __name__ == '__main__':
    # get_experimental_data(plot_data=True)
    compare_to_1d_model()