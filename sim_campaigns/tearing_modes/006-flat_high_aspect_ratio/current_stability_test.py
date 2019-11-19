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
    current = np.zeros((len(current_factors), 3))
    if os.path.exists(data_output):
        alphas = 1 - np.asarray(current_factors)
        deltas = np.loadtxt(data_output)
        gammas = np.loadtxt(gamma_output)

        for i, factor in enumerate(current_factors):
            j_phi_sim = j_phi * (1 - factor)
            solver = TearingModeSolver(m, r, r_max, B_theta, B_z, j_phi_sim, q, num_pts, delta=1e-12)
            current[i, 0] = factor
            current[i, 1] = solver.r_to_j_phi(solver.r_instability)
            current[i, 2] = solver.r_to_j_phi_deriv(solver.r_instability)
        np.savetxt('instability_currents.dat', current, header='Alpha\t\tJ [A]\t\tdJ_dr [A/m]')
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
                    flux_validation = np.loadtxt(validation_file)
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

        alpha_data = np.stack((np.asarray(alphas[::-1]), np.asarray(deltas[::-1]))).transpose()
        gamma_data = np.stack((np.asarray(alphas[::-1]), np.asarray(gammas[::-1]))).transpose()
        np.savetxt(data_output, alpha_data)
        np.savetxt(gamma_output, gamma_data)    

    alphas = np.asarray(alphas)
    deltas = np.asarray(deltas)
    gammas = np.asarray(gammas)

    # Load JOREK gammas
    alphas_jor = [0.0, 0.1, 0.2, 0.25, 0.275, 0.3, 0.3125, 0.325, 0.3375]
    gammas_jor = np.asarray([0.000110055, 7.14455e-5, 3.94265e-5, 2.48401e-5, 1.77428e-5, 1.0663e-5, 7.15803e-6, 3.77522e-6, 9.28732e-7])
    gammas_jor /= 6.4836e-7
    np.savetxt('gammas_jor_vs_alpha.dat', np.stack((np.asarray(alphas_jor), np.asarray(gammas_jor))).transpose())

    alphas_jor_8 = [0.0, 0.1, 0.2, 0.3]
    gammas_8 = np.asarray([41.701, 25.616, 12.98966, 2.5974])
    gammas_9 = np.asarray([6.809, 4.0393, 1.99651, 0.46058])

    # Load tm1 data
    tm1_data = np.loadtxt('gammas_tm1_vs_alpha.dat')

    # Multiply normalised solver gamma by density and resistivity
    eta = 1e-7 * 1.9382
    rho = 1e20 * 2 * 1.673 * 1e-27
    gammas[:, 1] *= eta ** 0.6 / rho ** 0.2
    np.savetxt('gammas_vs_alpha_si.dat', gammas)

    # Plot gammas
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(gammas[:, 0], gammas[:, 1], label='Linear')
    ax[0].scatter(gammas[:, 0], gammas[:, 1])
    ax[0].plot(alphas_jor, gammas_jor, label='JOREK')
    ax[0].scatter(alphas_jor, gammas_jor)
    ax[0].plot(tm1_data[:, 0], tm1_data[:, 1], label='TM1')
    ax[0].scatter(tm1_data[:, 0], tm1_data[:, 1])
    ax[0].set_ylabel('$\gamma\ [s^{-1}]$', fontsize=fontsize)
    ax[0].legend()

    gammas_interp = interpolate.interp1d(gammas[:, 0], gammas[:, 1])
    gammas_tm1_interp = interpolate.interp1d(tm1_data[:, 0], tm1_data[:, 1])
    ax[1].plot(alphas_jor, 100 * np.abs(gammas_jor - gammas_interp(alphas_jor)) / gammas_interp(alphas_jor), label='JOREK vs. Linear', c='orange')
    ax[1].scatter(alphas_jor, 100 * np.abs(gammas_jor - gammas_interp(alphas_jor)) / gammas_interp(alphas_jor), c='orange')
    ax[1].plot(tm1_data[:, 0], 100 * np.abs(tm1_data[:, 1] - gammas_interp(tm1_data[:, 0])) / gammas_interp(tm1_data[:, 0]), label='TM1 vs. Linear', c='g')
    ax[1].scatter(tm1_data[:, 0], 100 * np.abs(tm1_data[:, 1] - gammas_interp(tm1_data[:, 0])) / gammas_interp(tm1_data[:, 0]), c='g')
    skipped_index = -3
    ax[1].plot(alphas_jor[:skipped_index], 100 * np.abs(gammas_jor[:skipped_index] - gammas_tm1_interp(alphas_jor[:skipped_index])) / gammas_tm1_interp(alphas_jor[:skipped_index]), label='TM1 vs. JOREK', c='r')
    ax[1].scatter(alphas_jor[:skipped_index], 100 * np.abs(gammas_jor[:skipped_index] - gammas_tm1_interp(alphas_jor[:skipped_index])) / gammas_tm1_interp(alphas_jor[:skipped_index]), c='r')
    ax[1].set_ylabel('Relative difference $[\%]$', fontsize=fontsize)
    ax[1].set_xlabel('$\\alpha$', fontsize=fontsize)
    plt.savefig('current_stability_gammas_{}'.format(m))
    plt.show()

    # Get equivalent JOREK and TM1 delta
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

    deltas_tm1 = tm1_data[:,1] / 0.55 / (eta / PhysicalConstants.mu_0) ** 0.6
    deltas_tm1 /= (m * solver.r_to_B_theta(solver.r_instability) / np.sqrt(PhysicalConstants.mu_0 * rho)) ** 0.4
    deltas_tm1 /= (solver.r_to_q_deriv(solver.r_instability) / (solver.r_instability * solver.r_to_q(solver.r_instability))) ** 0.4 
    deltas_tm1 = deltas_tm1 ** (5.0 / 4.0)

    # Plot equivalent gammas
    scaling = 1.05               # Factor to improve plot
    fig, ax = plt.subplots(1)
    ax.plot(deltas[:, 0], deltas[:, 1], label='Linear $\Delta\'$')
    ax.scatter(deltas[:, 0], deltas[:, 1])
    ax.plot(alphas_jor, deltas_jor, label='JOREK $\Delta\'$ for $\eta=10^{-7}$')
    ax.scatter(alphas_jor, deltas_jor)
    ax.plot(alphas_jor_8, deltas_jor_8, label='JOREK $\Delta\'$ for $\eta=10^{-8}$')
    ax.scatter(alphas_jor_8, deltas_jor_8)
    ax.plot(alphas_jor_8, deltas_jor_9, label='JOREK $\Delta\'$ for $\eta=10^{-9}$')
    ax.scatter(alphas_jor_8, deltas_jor_9)
    ax.plot(tm1_data[:, 0], deltas_tm1, label='TM1')
    ax.scatter(tm1_data[:, 0], deltas_tm1)
    ax.axhline(0.0, c='k', linestyle='--')
    ax.axvline(0.358, linestyle='--', color='grey', label='Linear stability boundary')
    ax.axvline(0.3375, linestyle='--', color='g', label='JOREK stability boundary')
    ax.axvline(0.35, linestyle='--', color='g')
    ax.fill_betweenx(scaling * deltas[:, 1], 0.3375, 0.35, facecolor='green', alpha=0.2)
    ax.set_ylabel('$\Delta\'$', fontsize=fontsize)
    ax.set_xlabel('$\\alpha$', fontsize=fontsize)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([scaling * min(deltas[:, 1]), scaling * max(deltas[:, 1])])
    ax.grid(linestyle='--')
    ax.legend()
    plt.savefig('current_stability_deltas_{}'.format(m))
    plt.show()


if __name__ == '__main__':
    data_output = 'deltas_vs_alpha.dat'
    gamma_output = 'gammas_vs_alpha.dat'
    num_pts = 200000
    m = 2
    fontsize=16

    get_stability_condition(plot_results=True)
