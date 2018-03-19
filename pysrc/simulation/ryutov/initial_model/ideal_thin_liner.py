"""
Author: Rohan Ramasamy
Date: 27/05/2017

This script implements a simple model by Ryutov to simulate ideal magnetically
 driven liner implosions. This implementation is a first pass attempt and the
simplest possible, ignoring feedback inductance, and pressure from the imploding
plasma.
"""

import numpy as np
from matplotlib import pyplot as plt


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


def integrate_liner_implosion(times, R_Inner, current_func, R_Outer=None, convergence_ratio=0.1):
    """
    Function to integrate the liner implosion equation

    :param times: array of times
    :param R_Inner: Initial radius
    :return:
    """
    assert isinstance(R_Inner, float)
    mu_0 = 4.0 * np.pi * 1e-7 # Permeability of free space ~Aluminium
    if R_Outer is None:
        R_Outer = 1.1 * R_Inner
    liner_mass = 2700.0 * np.pi * (R_Outer ** 2 - R_Inner ** 2)   # Thin aluminium shell
    print("Liner thickness: {}".format(R_Outer - R_Inner))
    print("Liner mass: {}".format(liner_mass))
    magnetic_pressure_constant = mu_0 / (4.0 * np.pi * liner_mass)

    r = np.zeros(times.shape)
    v = np.zeros(times.shape)
    e_kin = np.zeros(times.shape)
    r[0] = R_Inner
    r[1] = R_Inner
    dt = times[1] - times[0]

    e_kin_f = 0.0
    for i, t in enumerate(times):
        if i == 0:
            continue

        if i == times.shape[0] - 1:
            v[i] = (r[i] - r[i - 1]) / dt
            continue

        I = current_func(t)
        r[i + 1] = 2 * r[i] - r[i - 1] - (dt ** 2 * magnetic_pressure_constant * I ** 2) / r[i]
        v[i] = (r[i + 1] - r[i - 1]) / (2 * dt)
        e_kin[i] = 0.5 * liner_mass * v[i] ** 2
        e_kin_f = e_kin[i]

        if r[i + 1] <= convergence_ratio * R_Inner:
            r[i + 1] = convergence_ratio * R_Inner
            v[i] = v[i - 1]
            e_kin[i] = e_kin[i - 1]
            e_kin_f = e_kin[i]
            break

    print("Impact Kinetic Energy: {}".format(e_kin_f))
    print("Impact Pressure: {}".format(2.0 / 3.0 * e_kin_f))        # Assumes a monatomic gas
    return r, v, e_kin


if __name__ == '__main__':
    I_0 = 2.5e7
    R_0 = 2e-2
    tau = 1e-4
    num_pts = 10000
    times = np.linspace(0.0, tau, num_pts)

    # Get current
    I = simple_current(times)

    # Integrate through implosion
    radii, velocities, kinetic_energies = integrate_liner_implosion(times, R_0, simple_current)

    # Plot implosion variables
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle("Characteristic curves for an ideal liner implosion")
    ax[0, 0].plot(times, I)
    ax[0, 0].set_title("Current (A)")
    ax[0, 1].plot(times, radii)
    ax[0, 1].set_title("Radius (m)")
    ax[0, 1].set_ylim([0.0, 1.01 * R_0])
    ax[1, 0].plot(times, velocities)
    ax[1, 0].set_title("Velocity (m/s)")
    ax[1, 1].plot(times, kinetic_energies)
    ax[1, 1].set_title("Kinetic Energy (J)")
    plt.show()

