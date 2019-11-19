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
    current_factors = [0.6, 0.7, 0.8, 0.9, 1.0]
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

    plt.figure()
    for epsilon in [1e-12, 1e-8, 1e-6, 1e-4]:
        r_max = r[-1]
        deltas = list()
        alphas = list()

        for factor in current_factors:
            j_phi_sim = j_phi * factor

            solver = TearingModeSolver(m, r, r_max, B_theta, B_z, j_phi_sim, q, num_pts, delta=epsilon)
            solver.find_delta(plot_output=False)

            deltas.append(solver.delta)
            alphas.append(1 - factor)

        plt.plot(alphas, deltas, label="$\epsilon$={}".format(epsilon))
        np.savetxt('deltas_{}.dat'.format(epsilon), np.stack((alphas, deltas), axis=-1))
    
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
    fontsize=22

    get_stability_condition(plot_results=True)

