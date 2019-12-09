"""
Author: Rohan Ramasamy
Date: 25/08/2019

This file contains code to assess the stability of tearing modes in a circular plasma, where the current profile is relaxed 
by some factor from the original distribution.
"""

import numpy as np
import sys
import os
from scipy import integrate, interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm

from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants
from plasma_physics.pysrc.theory.tearing_modes.cylindrical_tearing_modes import TearingModeSolver, MatsuokaTearingModeSolver


def compare_matsuoka(plot_results=True):
    current_factor = 1.0

    # Get profiles
    dat = np.loadtxt(os.path.join('input_profiles.dat'))
    
    # Modify and plot input profiles
    r = dat[:, 0]
    R = dat[:, 1]
    j_phi = dat[:, 2]
    B_z = dat[:, 3]
    B_theta = dat[:, 4]
    psi_n = dat[:, 5]
    q = r * B_z / (R * B_theta)
    q[0] = q[1]
    r_to_psin = interpolate.interp1d(r, psi_n)
    psin_to_r = interpolate.interp1d(r, psi_n)

    num_skip = 20
    num_spline_pts = 1000
    r_spline = np.linspace(0.0, r[-1], num_spline_pts)
    R_spline = interpolate.CubicSpline(r[::num_skip], R[::num_skip])
    j_phi_spline = interpolate.CubicSpline(r[::num_skip], j_phi[::num_skip])
    Btheta_spline = interpolate.CubicSpline(r[::num_skip], B_theta[::num_skip])
    Bz_spline = interpolate.CubicSpline(r[::num_skip], B_z[::num_skip])
    q_spline = interpolate.CubicSpline(r[::num_skip], q[::num_skip])
    R = R_spline(r_spline)
    j_phi = j_phi_spline(r_spline)
    B_theta = Btheta_spline(r_spline)
    B_z = Bz_spline(r_spline)
    q = q_spline(r_spline)
    r = r_spline

    r_max = r[-1]
    j_phi_plasma = j_phi * current_factor
    j_phi_external = j_phi * (1 - current_factor)

    # Integrate B_pol for j_plasma and j_virtual
    poloidal_integrand = PhysicalConstants.mu_0 * j_phi_plasma * r
    poloidal_field = integrate.cumtrapz(poloidal_integrand, r, initial=0.0)
    poloidal_field[1:] /= r[1:]
    q_plasma = r * B_z / (R * poloidal_field)
    q_plasma[0] = q_plasma[1]
    q_plasma = q

    poloidal_integrand_ext = PhysicalConstants.mu_0 * j_phi_external * r
    poloidal_field_ext = integrate.cumtrapz(poloidal_integrand_ext, r, initial=0.0)
    poloidal_field_ext[1:] /= r[1:]
    q_ext = r * B_z / (R * poloidal_field_ext)
    q_ext[0] = q_ext[1]

    # Plot input profiles and compare q profile for j_plasma with JOREK input
    if plot_results:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
        ax[0, 0].plot(r, B_z, c='k')
        ax[0, 0].set_ylabel('$B_{\phi}$ [T]', fontsize=fontsize)
        ax[1, 0].plot(r, B_theta, c='k')
        ax[1, 0].set_ylabel('$B_{\\theta}$ [T]', fontsize=fontsize)
        ax[1, 0].plot(r, poloidal_field)
        ax[1, 0].plot(r, poloidal_field_ext)
        ax[0, 1].plot(r, current_factor * j_phi * 1e-3)
        ax[0, 1].set_ylabel('$j_{\phi}$ [kA]', fontsize=fontsize)
        ax[1, 1].plot(r, q, c='k')
        ax[1, 1].plot(r, q_plasma)
        ax[1, 1].plot(r, q_ext)
        ax[1, 1].set_ylabel('q', fontsize=fontsize)
        ax[1, 1].set_xlabel('r [m]', fontsize=fontsize)
        ax[1, 0].set_xlabel('r [m]', fontsize=fontsize)
        ax[0, 0].set_xlim([0.0, 1.0])
        fig.tight_layout()
        plt.savefig('input_profiles')
        plt.show()

    wesson_solver = TearingModeSolver(m, r, r_max, B_theta, B_z, j_phi_plasma, q, num_pts, delta=1e-12)
    matsuoka_solver = MatsuokaTearingModeSolver(m, r, r_max, B_theta, B_z, j_phi_plasma, 1 / q_plasma, 1 / q_ext, 1 / q, num_pts, delta=1e-12)
    wesson_solver.find_delta(plot_output=plot_results)
    matsuoka_solver.find_delta(plot_output=plot_results)

    if plot_results:
        fig, ax = plt.subplots(2, sharex=True)
        for solver in [wesson_solver, matsuoka_solver]:
            psin_lower = r_to_psin(solver.r_lower)
            psin_upper = r_to_psin(solver.r_upper)
        
            ax[0].plot(psin_lower, solver.psi_sol_lower[:, 0], label='solution from axis')
            ax[0].plot(psin_upper, solver.psi_sol_upper[:, 0], label='solution from boundary')
            ax[1].plot(psin_lower, solver.psi_sol_lower[:, 1])
            ax[1].plot(psin_upper, solver.psi_sol_upper[:, 1])
        
        ax[0].set_ylabel('$\Psi_1$', fontsize=fontsize)
        ax[0].legend()
        ax[0].set_xlim([0.0, 1.0])
        ax[1].set_ylabel('$\\frac{\partial \Psi_1}{\partial r}$', fontsize=fontsize)
        ax[1].set_xlabel('$\hat \Psi$', fontsize=fontsize)
        plt.savefig('matsuoka_comparison')
        plt.show()


if __name__ == '__main__':
    num_pts = 200000
    m = 2
    fontsize=22

    compare_matsuoka(plot_results=True)

