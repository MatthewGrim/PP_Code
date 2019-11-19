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
from plasma_physics.pysrc.theory.tearing_modes.cylindrical_tearing_modes import TearingModeSolver


def get_stability_condition(plot_results=True):
    current_factors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    colors = cm.copper(np.linspace(0.0, 1.0, len(current_factors)))

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
    
    q = r * B_z / (R * B_theta)
    q[0] = q[1]
    if plot_results:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True)

        ax[0, 0].plot(r, B_z, c='k')
        ax[0, 0].set_ylabel('$B_{\phi}$ [T]', fontsize=fontsize)
        ax[1, 0].plot(r, B_theta, c='k')
        ax[1, 0].set_ylabel('$B_{\\theta}$ [T]', fontsize=fontsize)
        for i, factor in enumerate(current_factors[::-1]):
            linestyle = '--' if i > 0 else '-'
            ax[0, 1].plot(r, factor * j_phi * 1e-3, linestyle=linestyle, c=colors[i])
        ax[0, 1].set_ylabel('$j_{\phi}$ [kA]', fontsize=fontsize)
        ax[1, 1].plot(r, q, c='k')
        ax[1, 1].set_ylabel('q', fontsize=fontsize)
        ax[1, 1].set_xlabel('r [m]', fontsize=fontsize)
        ax[1, 0].set_xlabel('r [m]', fontsize=fontsize)
        ax[0, 0].set_xlim([0.0, 1.0])
        fig.tight_layout()
        plt.savefig('input_profiles')
        plt.show()

    r_max = r[-1]
    deltas = list()
    alphas = list()
    for factor in current_factors:
        j_phi_sim = j_phi * factor

        solver = TearingModeSolver(m, r, r_max, B_theta, B_z, j_phi_sim, q, num_pts, delta=1e-12)
        solver.find_delta(plot_output=False)

        psin_lower = r_to_psin(solver.r_lower)
        psin_upper = r_to_psin(solver.r_upper)
        
        if plot_results:
            fig, ax = plt.subplots(2, sharex=True)
            ax[0].plot(psin_lower, solver.psi_sol_lower[:, 0], label='solution from axis')
            ax[0].plot(psin_upper, solver.psi_sol_upper[:, 0], label='solution from boundary')

            # Plot comparison solution if it exists        
            validation_file = 'flux_validation_vc_{}.csv'.format(factor)
            if os.path.exists(validation_file):
                flux_validation = np.loadtxt(validation_file, delimiter=',')
                flux_validation[:, 1] /= np.max(flux_validation[:, 1]) 
                flux_validation[:, 1] *= np.max(solver.psi_sol_lower[:, 0])
                ax[0].plot(flux_validation[:, 0], flux_validation[:, 1], linestyle='--', label='JOREK')

            ax[0].set_ylabel('$\Psi_1$', fontsize=fontsize)
            ax[0].legend()
            ax[0].set_xlim([0.0, 1.0])
            
            ax[1].plot(psin_lower, solver.psi_sol_lower[:, 1])
            ax[1].plot(psin_upper, solver.psi_sol_upper[:, 1])
            ax[1].set_ylabel('$\\frac{\partial \Psi_1}{\partial r}$', fontsize=fontsize)
            ax[1].set_xlabel('$\hat \Psi$', fontsize=fontsize)
            plt.show()

        deltas.append(solver.delta)
        alphas.append(1 - factor)

    np.savetxt('deltas', np.asarray(deltas))

    plt.figure()
    plt.plot(alphas, deltas)
    plt.axvline(0.346, linestyle='--', color='grey', label='Linear stability boundary')
    plt.axvline(0.25, linestyle='--', color='g', label='JOREK stability boundary')
    plt.grid(linestyle='--')
    plt.ylabel('$\Delta\'$', fontsize=fontsize)
    plt.xlabel('$\\alpha$', fontsize=fontsize)
    plt.legend()
    plt.xlim([0.0, 1.0])
    plt.savefig('current_stability_{}'.format(m))
    plt.show()


if __name__ == '__main__':
    num_pts = 200000
    m = 2
    fontsize=16

    get_stability_condition(plot_results=False)

