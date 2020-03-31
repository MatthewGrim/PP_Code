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
    current_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Get profiles
    dat = np.loadtxt(os.path.join('input_profiles.dat'))
    
    # Modify and plot input profiles
    r = dat[:, 0]
    R = dat[:, 1]
    j_phi = dat[:, 2]
    B_z = dat[:, 3]
    B_theta = dat[:, 4]
    psi_n = dat[:, 5]
    r_to_psin = interpolate.interp1d(r, psi_n)
    psin_to_r = interpolate.interp1d(r, psi_n)
    r_max = r[-1]

    alphas = list()
    deltas_wesson = list()
    deltas_matsuoka = list()
    for current_factor in current_factors:
        j_phi_plasma = j_phi * current_factor
        j_phi_external = j_phi * (1 - current_factor)

        # Integrate B_pol for j_plasma and j_virtual
        poloidal_integrand = PhysicalConstants.mu_0 * j_phi_plasma * r
        poloidal_field = integrate.cumtrapz(poloidal_integrand, r, initial=0.0)
        poloidal_field[1:] /= r[1:]
        q_plasma = r * B_z / (R * poloidal_field)
        q_plasma[0] = q_plasma[1]

        poloidal_integrand_ext = PhysicalConstants.mu_0 * j_phi_external * r
        poloidal_field_ext = integrate.cumtrapz(poloidal_integrand_ext, r, initial=0.0)
        poloidal_field_ext[1:] /= r[1:]
        q_ext = r * B_z / (R * poloidal_field_ext)
        q_ext[0] = q_ext[1]
        q_ext[np.isinf(q_ext)] = 1e12

        B_theta = poloidal_field + poloidal_field_ext

        # Smooth input profiles using splines    
        num_skip = 20
        R_spline = interpolate.CubicSpline(r[::num_skip], R[::num_skip])
        j_phi_spline = interpolate.CubicSpline(r[::num_skip], j_phi[::num_skip])
        j_phi_plasma_spline = interpolate.CubicSpline(r[::num_skip], j_phi_plasma[::num_skip])
        j_phi_ext_spline = interpolate.CubicSpline(r[::num_skip], j_phi_external[::num_skip])
        Btheta_spline = interpolate.CubicSpline(r[::num_skip], B_theta[::num_skip])
        Bz_spline = interpolate.CubicSpline(r[::num_skip], B_z[::num_skip])
        q_plasma_spline = interpolate.CubicSpline(r[::num_skip], q_plasma[::num_skip])
        q_ext_spline = interpolate.CubicSpline(r[::num_skip], q_ext[::num_skip])
        R = R_spline(r)
        j_phi = j_phi_spline(r)
        j_phi_plasma = j_phi_plasma_spline(r)
        j_phi_ext = j_phi_ext_spline(r)
        B_theta = Btheta_spline(r)
        B_z = Bz_spline(r)
        q_plasma = q_plasma_spline(r)
        q_ext = q_ext_spline(r)
        q = 1 / (1 / q_plasma + 1 / q_ext)

        # Plot input profiles and compare q profile for j_plasma with JOREK input
        if plot_results:
            fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
            ax[0, 0].plot(r, B_z, c='k')
            ax[0, 0].set_ylabel('$B_{\phi}$ [T]', fontsize=fontsize)
            ax[1, 0].plot(r, B_theta, c='k')
            ax[1, 0].set_ylabel('$B_{\\theta}$ [T]', fontsize=fontsize)
            ax[1, 0].plot(r, poloidal_field)
            ax[1, 0].plot(r, poloidal_field_ext)
            ax[0, 1].plot(r, j_phi * 1e-3, c='k')
            ax[0, 1].plot(r, j_phi_plasma * 1e-3)
            ax[0, 1].plot(r, j_phi_ext * 1e-3)
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
        matsuoka_solver = MatsuokaTearingModeSolver(m, r, r_max, R, B_theta, B_z, 1 / q_plasma, 1 / q_ext, num_pts, delta=1e-12)
        wesson_solver.find_delta(plot_output=False)
        matsuoka_solver.find_delta(plot_output=False)

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
        
        deltas_wesson.append(wesson_solver.delta)
        deltas_matsuoka.append(matsuoka_solver.delta)
        alphas.append(1 - current_factor)
    
    deltas = np.stack((np.asarray(alphas[::-1]), np.asarray(deltas_wesson[::-1]), np.asarray(deltas_matsuoka[::-1]))).transpose()
    header_string = "alpha    wesson    matsuoka"
    # np.savetxt('deltas.txt', deltas, header=header_string)

    fig, ax = plt.subplots(1)
    ax.plot(deltas[:, 0], deltas[:, 1], label='$\Delta\'_{Wesson}$')
    ax.scatter(deltas[:, 0], deltas[:, 1])
    ax.plot(deltas[:, 0], deltas[:, 2], label='$\Delta\'_{Matsuoka}$')
    ax.scatter(deltas[:, 0], deltas[:, 2])
    ax.set_ylabel('$\Delta\'$', fontsize=fontsize)
    ax.set_xlabel('$\\alpha$', fontsize=fontsize)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([-10.0, 22.0])
    ax.grid(linestyle='--')
    plt.legend()
    plt.savefig('matsuoka_comparison')
    plt.show()


if __name__ == '__main__':
    num_pts = 200000
    m = 2
    fontsize=22

    compare_matsuoka(plot_results=False)

