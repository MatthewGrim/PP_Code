"""
Author: Rohan Ramasamy
Date: 03/06/2017

This script builds on the simple ideal liner model to add a second outer liner, making the system act as a coaxial liner
"""

import numpy as np
from matplotlib import pyplot as plt

from Physical_Models.ryutov_model.coax.coaxial_ideal_liner import coaxial_ideal_implosion
from Physical_Models.ryutov_model.coax.coaxial_ideal_liner import coaxial_ideal_implosion_2d


def simple_current(t):
    """
    Function that returns a simple current profile from a pulse power machine

    :param I_0: Peak current
    :param t: time relative to closed switch
    :param tau: time of complete discharge
    """
    assert isinstance(I_0, float)
    assert isinstance(tau, float)

    return I_0 * (np.sin(np.pi * t / tau)) ** 2


def oned_example():
    r_inner = 2e-2
    R_Inner = 2.5e-2
    num_pts = 10000
    times = np.linspace(0.0, tau, num_pts)

    # Get current
    I = simple_current(times)

    # Integrate through implosion
    results = coaxial_ideal_implosion(times, r_inner, R_Inner, simple_current,
                                      r_outer=1.1*r_inner, R_Outer=2*R_Inner)
    r_i, r_o, v, e_kin, R, V, E_kin, L = results

    # Plot implosion variables
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Characteristic curves for an ideal liner implosion")
    ax[0, 0].plot(times, I)
    ax[0, 0].set_title("Current (A)")
    ax[0, 1].plot(times, r_i, c='r')
    ax[0, 1].plot(times, r_o, c='r', linestyle='--')
    ax[0, 1].plot(times, R, c='g')
    ax[0, 1].set_title("Radius (m)")
    ax[0, 2].plot(times, L)
    ax[0, 2].set_title("Inductance (L/m)")
    ax[1, 0].plot(times, v, c='r')
    ax[1, 0].plot(times, V, c='g')
    ax[1, 0].set_title("Velocity (m/s)")
    ax[1, 1].plot(times, e_kin, c='r')
    ax[1, 1].plot(times, E_kin, c='g')
    ax[1, 1].set_title("Kinetic Energy (J)")
    ax[1, 2].plot(times, r_o - r_i)
    ax[1, 2].set_title("Inner liner difference in radius (m)")
    plt.show()


def twod_example():
    """
    Simple 2D example that mimics the parameters of the 1D example above. Because h=1, the inductance should be
    comparable
    :return:
    """
    num_h_pts = 1000
    h = np.linspace(0, 1.0, num_h_pts)
    r_inner = np.ones(num_h_pts) * 2e-2
    R_Inner = np.ones(num_h_pts) * 2.5e-2
    num_pts = 10000
    times = np.linspace(0.0, tau, num_pts)

    # Get current
    I = simple_current(times)

    # Integrate through implosion
    results = coaxial_ideal_implosion_2d(times, r_inner, R_Inner, h, simple_current,
                                         r_outer=1.1*r_inner, R_Outer=2*R_Inner)
    r_i, r_o, v, e_kin, R, V, E_kin, L, L_tot, L_dot, t_final = results        # e_kin, E_kin and L are per metre

    for ts in [t_final]:
        fig, ax = plt.subplots(2, 3, figsize=(18, 9))
        fig.suptitle("Characteristic curves for an ideal liner implosion")
        ax[0, 0].plot(times, I)
        ax[0, 0].set_title("Current (A)")
        ax[0, 1].plot(h, r_i[ts, :], c='r')
        ax[0, 1].plot(h, r_o[ts, :], c='r', linestyle='--')
        ax[0, 1].plot(h, R[ts, :], c='g')
        ax[0, 1].set_title("Radius (m)")
        ax[0, 2].plot(h, v[ts, :], c='r')
        ax[0, 2].plot(h, V[ts, :], c='g')
        ax[0, 2].set_title("Velocity (m/s)")
        ax[1, 0].plot(times, L_tot)
        ax[1, 0].set_title("Total Inductance (L)")
        ax[1, 1].plot(times, L_dot)
        ax[1, 1].set_title("Total Inductance Change (L_dot)")
        ax[1, 2].plot(h, L[ts, :])
        ax[1, 2].set_title("Inductance (L/m)")
        plt.show()


def twod_advanced_example():
    """
    2D example with a varying initial radius
    :return:
    """
    num_h_pts = 1000
    h_max = 20.0
    h = np.linspace(0, h_max, num_h_pts)
    r_inner = np.ones(num_h_pts)
    R_Inner = np.ones(num_h_pts)
    r_outer = np.ones(num_h_pts)
    R_Outer = np.ones(num_h_pts)
    r_inner_max = 1.0
    r_inner_min = 2e-2
    R_Inner_max = 1.005
    R_Inner_min = 3e-2
    r_outer_max = 1.0025
    r_outer_min = 2.5e-2
    R_Outer_max = 1.2
    R_Outer_min = 4e-2
    h_left = 0.1 * h_max
    h_right = 0.9 * h_max
    h_mid_left = 0.45 * h_max
    h_mid_right = 0.55 * h_max
    for i, h_i in enumerate(h):
        if h_i < h_left or h_i > h_right:
            r_inner[i] = r_inner_max
            r_outer[i] = r_outer_max
            R_Inner[i] = R_Inner_max
            R_Outer[i] = R_Outer_max
        elif h_mid_left < h_i < h_mid_right:
            r_inner[i] = r_inner_min
            r_outer[i] = r_outer_min
            R_Inner[i] = R_Inner_min
            R_Outer[i] = R_Outer_min
        elif h_i < 0.45 * h_max:
            r_inner[i] = r_inner_max + (h_i - h_left) / (h_mid_left - h_left) * (r_inner_min - r_inner_max)
            r_outer[i] = r_outer_max + (h_i - h_left) / (h_mid_left - h_left) * (r_outer_min - r_outer_max)
            R_Inner[i] = R_Inner_max + (h_i - h_left) / (h_mid_left - h_left) * (R_Inner_min - R_Inner_max)
            R_Outer[i] = R_Outer_max + (h_i - h_left) / (h_mid_left - h_left) * (R_Outer_min - R_Outer_max)
        else:
            r_inner[i] = r_inner_max - (h_i - h_right) / (h_right - h_mid_right) * (r_inner_min - r_inner_max)
            r_outer[i] = r_outer_max - (h_i - h_right) / (h_right - h_mid_right) * (r_outer_min - r_outer_max)
            R_Inner[i] = R_Inner_max - (h_i - h_right) / (h_right - h_mid_right) * (R_Inner_min - R_Inner_max)
            R_Outer[i] = R_Outer_max - (h_i - h_right) / (h_right - h_mid_right) * (R_Outer_min - R_Outer_max)

    num_pts = 10000
    times = np.linspace(0.0, tau, num_pts)

    # Get current
    I = simple_current(times)

    # Integrate through implosion
    results = coaxial_ideal_implosion_2d(times, r_inner, R_Inner, h, simple_current,
                                         r_outer=r_outer, R_Outer=R_Outer)
    r_i, r_o, v, e_kin, R, V, E_kin, L, L_tot, L_dot, t_final = results        # e_kin, E_kin and L are per metre

    for ts in [0, t_final]:
        fig, ax = plt.subplots(2, 3, figsize=(18, 9))
        fig.suptitle("Characteristic curves for an ideal liner implosion t={}".format(times[ts]))
        ax[0, 0].plot(times, I)
        ax[0, 0].set_title("Current (A)")
        ax[0, 1].plot(h, r_i[ts, :], c='r')
        ax[0, 1].plot(h, r_o[ts, :], c='r', linestyle='--')
        ax[0, 1].plot(h, R[ts, :], c='g')
        ax[0, 1].set_title("Radius (m)")
        ax[0, 2].plot(h, v[ts, :], c='r')
        ax[0, 2].plot(h, V[ts, :], c='g')
        ax[0, 2].set_title("Velocity (m/s)")
        ax[1, 0].plot(times, L_tot)
        ax[1, 0].set_title("Total Inductance (L)")
        ax[1, 1].plot(times, L_dot)
        ax[1, 1].set_title("Total Inductance Change (L_dot)")
        ax[1, 2].plot(h, L[ts, :])
        ax[1, 2].plot(h, L[ts, :] - L[0, :])
        ax[1, 2].set_title("Inductance (L/m)")
        plt.show()


if __name__ == '__main__':
    I_0 = 2.5e7
    tau = 1e-4
    oned_example()
    twod_example()
    twod_advanced_example()