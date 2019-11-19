"""
Author: Rohan Ramasamy
Date: 16/07/2019

This file contains code to compare results from a home brew tearing mode solver against results from

Tearing mode in the cylindrical tokamak - H. P. Furth, P. H. Rutherford, and H. Selberg
"""

import numpy as np
import matplotlib.pyplot as plt

from plasma_physics.pysrc.theory.tearing_modes.cylindrical_tearing_modes import TearingModeSolverNormalised

def compare_figures():
    x_s_points = np.linspace(0.2, 1.6, 8)
    m = 2
    num_grid_pts = 100000
    
    fig, ax = plt.subplots(2, sharex=True)
    r_Deltas = list()
    psi_s = list()
    for x_s in x_s_points:
        solver = TearingModeSolverNormalised(m, x_s, num_grid_pts, delta=1e-10)
        solver.find_delta(plot_output=False)

        r_Delta = (solver.A_upper - solver.A_lower)
        r_Deltas.append(r_Delta)
        psi_s.append(solver.psi_rs)

        ax[0].plot(solver.r_lower, solver.psi_sol_lower[:, 0], c='k')
        ax[0].plot(solver.r_upper, solver.psi_sol_upper[:, 0], c='k')
    ax[0].plot(x_s_points, psi_s, linestyle='--', c='k')
    ax[1].plot(x_s_points, r_Deltas)
    ax[0].set_ylabel("$\hat \Psi$")
    ax[1].set_ylabel("$r \Delta$")
    ax[1].set_xlabel("x")
    for a in ax:
        a.grid(linestyle='--')

    psi_s = np.loadtxt('psi_s_validation.csv', delimiter=',')
    psi_profiles = np.loadtxt('psi_1_validation.csv', delimiter=',')
    ax[0].plot(psi_s[:, 0], psi_s[:, 1], linestyle='--', c='grey')
    ax[0].scatter(psi_profiles[:, 0], psi_profiles[:, 1], c='grey', marker='+')
    delta_s = np.loadtxt('delta_validation.csv', delimiter=',')
    ax[1].plot(delta_s[:, 0], delta_s[:, 1], linestyle='--', c='grey')
    
    ax[0].set_xlim([0.0, 2.0])
    ax[0].set_ylim([0.0, 1.0])
    plt.savefig('furth_validation')
    plt.show()

    r_Deltas = np.asarray(r_Deltas)
    psi_s = np.asarray(psi_s)

    
if __name__ == '__main__':
    compare_figures()

