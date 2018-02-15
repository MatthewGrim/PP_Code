"""
Author: Rohan Ramasamy
Date: 31/05/2017

This script builds on the simple ideal liner model to add a second outer liner, making the system act as a coaxial liner
"""

import numpy as np
import scipy.integrate as integrate


def coaxial_ideal_implosion(times, r_inner, R_Inner, current_func,
                            r_outer=None, R_Outer=None, convergence_ratio=0.1):
    """
    Function to solve the differential equation for the coaxial implosion/explosion

    :param times:
    :param r_inner: inner radius of the inner liner
    :param R_Inner: inner radius of the outer liner
    :param current_func: function determining the current
    :param r_outer: outer radius of the inner liner
    :param R_Outer: outer radius of the outer liner
    :param convergence_ratio: implosion radial convergence at which simulation is stopped
    :return:
    """
    assert isinstance(r_inner, float)
    assert isinstance(R_Inner, float)

    # Permeability of free space ~Aluminium
    mu_0 = 4.0 * np.pi * 1e-7

    # Set default values of outer radii, and check for consistency
    r_outer = 1.1 * r_inner if r_outer is None else r_outer
    R_Outer = 1.1 * R_Inner if R_Outer is None else R_Outer
    assert r_inner < r_outer < R_Inner < R_Outer

    # Get shell densities - kgm-1
    liner_density = 2700.0
    m_inner = liner_density * np.pi * (r_outer ** 2 - r_inner ** 2)
    m_outer = liner_density * np.pi * (R_Outer ** 2 - R_Inner ** 2)

    print("Inner Liner thickness: {}".format(r_outer - r_inner))
    print("Inner Liner mass: {}".format(m_inner))
    print("Outer Liner thickness: {}".format(R_Outer - R_Inner))
    print("Outer Liner mass: {}".format(m_outer))

    # Define constants for magnetic pressure and inductance
    p_const_inner = mu_0 / (4.0 * np.pi * m_inner)
    p_const_outer = mu_0 / (4.0 * np.pi * m_outer)
    l_const = mu_0 / (2.0 * np.pi)

    # Set up result arrays
    r_i = np.zeros(times.shape)
    r_o = np.zeros(times.shape)
    v = np.zeros(times.shape)
    e_kin = np.zeros(times.shape)
    R_I = np.zeros(times.shape)
    V = np.zeros(times.shape)
    E_kin = np.zeros(times.shape)
    L = np.zeros(times.shape)
    r_i[0] = r_inner
    r_i[1] = r_inner
    r_o[0] = r_outer
    r_o[1] = r_outer
    R_I[0] = R_Inner
    R_I[1] = R_Inner
    dt = times[1] - times[0]

    e_kin_f = 0.0
    for i, t in enumerate(times):
        # Calculate current inductance Lm-1
        L[i] = l_const * np.log(R_I[i] / r_o[i])

        # Skip first time step F = ma calculation
        if i == 0:
            continue

        # On last time step, use backward difference to set results
        if i == times.shape[0] - 1:
            v[i] = (r_o[i] - r_o[i - 1]) / dt
            V[i] = (R_I[i] - R_I[i - 1]) / dt
            e_kin[i] = 0.5 * m_inner * v[i] ** 2
            E_kin[i] = 0.5 * m_outer * V[i] ** 2
            e_kin_f = e_kin[i]
            continue

        I = current_func(t)
        # Inner liner
        r_o[i + 1] = 2 * r_o[i] - r_o[i - 1] - (dt ** 2 * p_const_inner * I ** 2) / r_i[i]
        v[i] = (r_o[i + 1] - r_o[i - 1]) / (2 * dt)
        e_kin[i] = 0.5 * m_inner * v[i] ** 2
        e_kin_f = e_kin[i]

        # Outer liner
        R_I[i + 1] = 2 * R_I[i] - R_I[i - 1] + (dt ** 2 * p_const_outer * I ** 2) / R_I[i]
        V[i] = (R_I[i + 1] - R_I[i - 1]) / (2 * dt)
        E_kin[i] = 0.5 * m_outer * V[i] ** 2

        # If the convergence ratio is surpassed, end the simulation
        r_i[i + 1] = np.sqrt(r_o[i + 1] ** 2 - m_inner / (liner_density * np.pi))
        if r_i[i + 1] <= convergence_ratio * r_inner:
            r_i[i + 1] = convergence_ratio * r_inner
            r_o[i + 1] = r_o[i]
            v[i] = v[i - 1]
            e_kin[i] = e_kin[i - 1]
            e_kin_f = e_kin[i]
            break

    print("Impact Kinetic Energy: {}".format(e_kin_f))
    print("Impact Pressure: {}".format(2.0 / 3.0 * e_kin_f))        # Assumes a monatomic gas
    return r_i, r_o, v, e_kin, R_I, V, E_kin, L


def coaxial_ideal_implosion_2d(times, r_inner, R_Inner, h, current_func,
                            r_outer=None, R_Outer=None, convergence_ratio=0.1):
    """
    Function to solve the differential equation for the coaxial implosion/explosion

    :param times:
    :param r_inner: inner radius of the inner liner
    :param R_Inner: inner radius of the outer liner
    :param current_func: function determining the current
    :param r_outer: outer radius of the inner liner
    :param R_Outer: outer radius of the outer liner
    :param convergence_ratio: implosion radial convergence at which simulation is stopped
    :return:
    """
    # Inner liner radius at which the simulation is stopped
    minimum_radius = convergence_ratio * np.min(r_inner)
    # Permeability of free space ~Aluminium
    mu_0 = 4.0 * np.pi * 1e-7

    # Set default values of outer radii, and check for consistency
    r_outer = 1.1 * r_inner if r_outer is None else r_outer
    R_Outer = 1.1 * R_Inner if R_Outer is None else R_Outer
    assert np.all(r_inner < r_outer)
    assert np.all(r_outer < R_Inner)
    assert np.all(R_Inner < R_Outer)

    # Get shell densities - kgm-1
    liner_density = 2700.0
    rho_inner = liner_density * np.pi * (r_outer ** 2 - r_inner ** 2)   # Thin aluminium shell - kgm-1
    Rho_Outer = liner_density * np.pi * (R_Outer ** 2 - R_Inner ** 2)

    print("Inner Liner mass: {}".format(np.sum(rho_inner)))
    print("Outer Liner mass: {}".format(np.sum(Rho_Outer)))

    # Define constants for magnetic pressure and inductance
    p_const_inner = mu_0 / (4.0 * np.pi * rho_inner)
    p_const_outer = mu_0 / (4.0 * np.pi * Rho_Outer)
    l_const = mu_0 / (2.0 * np.pi)

    # Set up result arrays
    r_i = np.zeros((times.shape[0], r_inner.shape[0]))
    r_o = np.zeros((times.shape[0], r_inner.shape[0]))
    v = np.zeros((times.shape[0], r_inner.shape[0]))
    e_kin = np.zeros((times.shape[0], r_inner.shape[0]))
    R_I = np.zeros((times.shape[0], r_inner.shape[0]))
    V = np.zeros((times.shape[0], r_inner.shape[0]))
    E_kin = np.zeros((times.shape[0], r_inner.shape[0]))
    L = np.zeros((times.shape[0], r_inner.shape[0]))
    L_tot = np.zeros(times.shape)
    L_dot = np.zeros(times.shape)
    r_i[0, :] = r_inner
    r_i[1, :] = r_inner
    r_o[0, :] = r_outer
    r_o[1, :] = r_outer
    R_I[0, :] = R_Inner
    R_I[1, :] = R_Inner
    dt = times[1] - times[0]

    for ts, t in enumerate(times):
        # Calculate outer radius of inner liner and inductance
        L[ts, :] = l_const * np.log(R_I[ts, :] / r_o[ts, :])
        L_tot[ts] = integrate.simps(L[ts, :], h)
        L_dot[ts] = (L_tot[ts] - L_tot[ts - 1]) / dt if ts != 0 else 0.0

        # Skip first time step F = ma calculation
        if ts == 0:
            continue

        # On last time step, use backward difference to set results
        if ts == times.shape[0] - 1:
            v[ts, :] = (r_o[ts, :] - r_o[ts - 1, :]) / dt
            V[ts, :] = (R_I[ts, :] - R_I[ts - 1, :]) / dt
            e_kin[ts, :] = 0.5 * rho_inner * v[ts, :] ** 2
            E_kin[ts, :] = 0.5 * Rho_Outer * V[ts, :] ** 2
            continue

        I = current_func(t)
        # Inner Liner
        r_o[ts + 1, :] = 2 * r_o[ts, :] - r_o[ts - 1, :] - (dt ** 2 * p_const_inner * I ** 2) / r_o[ts, :]
        v[ts, :] = (r_o[ts + 1, :] - r_o[ts - 1, :]) / (2 * dt)
        e_kin[ts, :] = 0.5 * rho_inner * v[ts, :] ** 2

        # Outer Liner
        R_I[ts + 1, :] = 2 * R_I[ts, :] - R_I[ts - 1, :] + (dt ** 2 * p_const_outer * I ** 2) / R_I[ts, :]
        V[ts, :] = (R_I[ts + 1, :] - R_I[ts - 1, :]) / (2 * dt)
        E_kin[ts, :] = 0.5 * Rho_Outer * V[ts, :] ** 2

        # If any point surpasses the minimum radius, end the simulation
        r_i[ts + 1, :] = np.sqrt(r_o[ts + 1, :] ** 2 - rho_inner / (liner_density * np.pi))
        if np.any(r_i[ts + 1, :] <= minimum_radius):
            r_i[ts + 1, :] = minimum_radius
            r_o[ts + 1, :] = r_o[ts, :]
            break

    return r_i, r_o, v, e_kin, R_I, V, E_kin, L, L_tot, L_dot, ts




