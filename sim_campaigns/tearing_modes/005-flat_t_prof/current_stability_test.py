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
    current_factors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.675, 0.6875, 0.7, 0.8, 0.9, 1.0]
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
    gammas = list()
    alphas = list()
    if os.path.exists(data_output):
        alphas = 1 - np.asarray(current_factors)
        deltas = np.loadtxt(data_output)
        gammas = np.loadtxt(gamma_output)
    else:
        for factor in current_factors:
            j_phi_sim = j_phi * factor

            solver = TearingModeSolver(m, r, r_max, B_theta, B_z, j_phi_sim, q, num_pts, delta=1e-12)
            solver.find_delta(plot_output=False)
            solver.get_gamma()

            # Transform to psi_norm for comparison with JOREK
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
            gammas.append(solver.gamma)
            alphas.append(1 - factor)

        np.savetxt(data_output, np.asarray(deltas))
        np.savetxt(gamma_output, np.asarray(gammas))    

    alphas = np.asarray(alphas)
    deltas = np.asarray(deltas)
    gammas = np.asarray(gammas)

    # Plot gammas
    gammas_jor = np.asarray([0.000112257, 7.25606e-5, 3.87257e-5, 2.29265e-5, 1.50064e-5, 6.86054e-6, 2.586e-6])
    gammas_jor /= 6.4836e-7
    alphas_jor = [0.0, 0.1, 0.2, 0.25, 0.275, 0.3, 0.3125]
    eta = 1e-7 * 1.9382
    rho = 1e20 * 2 * 1.673 * 1e-27
    gammas *= eta ** 0.6 / rho ** 0.2
    fig, ax = plt.subplots(1)
    ax.plot(alphas, gammas, label='Linear $\Delta\'$')
    ax.scatter(alphas, gammas)
    ax.plot(alphas_jor, gammas_jor, label='JOREK $\Delta\'$')
    ax.scatter(alphas_jor, gammas_jor)
    plt.savefig('current_stability_gammas_{}'.format(m))
    ax.set_ylabel('$\gamma\ [s^{-1}]$', fontsize=fontsize)
    ax.set_xlabel('$\\alpha$', fontsize=fontsize)
    plt.show()

    alphas_jor_8 = [0.0, 0.1, 0.2, 0.3]
    gammas_8 = np.asarray([41.701, 25.616, 12.98966, 2.5974])
    gammas_9 = np.asarray([6.809, 4.0393, 1.99651, 0.46058])

    # Plot deltas
    solver = TearingModeSolver(m, r, r_max, B_theta, B_z, j_phi, q, num_pts, delta=1e-12)
    deltas_jor = gammas_jor / 0.55 / (eta / PhysicalConstants.mu_0) ** 0.6
    deltas_jor /= (m * solver.r_to_B_theta(solver.r_instability) / np.sqrt(PhysicalConstants.mu_0 * rho)) ** 0.4
    deltas_jor /= (solver.r_to_q_deriv(solver.r_instability) / (solver.r_instability * solver.r_to_q(solver.r_instability))) ** 0.4 
    deltas_jor = deltas_jor ** (5.0 / 4.0)

    deltas_jor_8 = gammas_8 / 0.55 / (eta * 0.1 / PhysicalConstants.mu_0) ** 0.6
    deltas_jor_8 /= (m * solver.r_to_B_theta(solver.r_instability) / np.sqrt(PhysicalConstants.mu_0 * rho)) ** 0.4
    deltas_jor_8 /= (solver.r_to_q_deriv(solver.r_instability) / (solver.r_instability * solver.r_to_q(solver.r_instability))) ** 0.4 
    deltas_jor_8 = deltas_jor_8 ** (5.0 / 4.0)

    deltas_jor_9 = gammas_9 / 0.55 / (eta * 0.01 / PhysicalConstants.mu_0) ** 0.6
    deltas_jor_9 /= (m * solver.r_to_B_theta(solver.r_instability) / np.sqrt(PhysicalConstants.mu_0 * rho)) ** 0.4
    deltas_jor_9 /= (solver.r_to_q_deriv(solver.r_instability) / (solver.r_instability * solver.r_to_q(solver.r_instability))) ** 0.4 
    deltas_jor_9 = deltas_jor_9 ** (5.0 / 4.0)

    scaling = 1.05
    fig, ax = plt.subplots(1)
    ax.plot(alphas, deltas, label='Linear $\Delta\'$')
    ax.scatter(alphas, deltas)
    ax.plot(alphas_jor, deltas_jor, label='JOREK $\Delta\'$')
    ax.scatter(alphas_jor, deltas_jor)
    ax.plot(alphas_jor_8, deltas_jor_8, label='JOREK $\Delta\' for $\eta=10^{-8}$')
    ax.scatter(alphas_jor_8, deltas_jor_8)
    ax.plot(alphas_jor_8, deltas_jor_9, label='JOREK $\Delta\'$ for $\eta=10^{-9}$')
    ax.scatter(alphas_jor_8, deltas_jor_9)
    ax.axhline(0.0, c='k', linestyle='--')
    ax.axvline(0.346, linestyle='--', color='grey', label='Linear stability boundary')
    ax.axvline(0.3125, linestyle='--', color='g', label='JOREK stability boundary')
    ax.axvline(0.325, linestyle='--', color='g')
    ax.fill_betweenx(scaling * deltas, 0.3125, 0.325, facecolor='green', alpha=0.2)
    ax.set_ylabel('$\Delta\'$', fontsize=fontsize)
    ax.set_xlabel('$\\alpha$', fontsize=fontsize)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([scaling * min(deltas), scaling * max(deltas)])
    ax.grid(linestyle='--')
    ax.legend()
    plt.savefig('current_stability_deltas_{}'.format(m))
    plt.show()


if __name__ == '__main__':
    data_output = 'deltas'
    gamma_output = 'gammas'
    num_pts = 200000
    m = 2
    fontsize=16

    get_stability_condition(plot_results=True)

