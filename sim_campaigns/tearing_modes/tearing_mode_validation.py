"""
Author: Rohan Ramasamy
Date: 16/07/2019

This file contains code to compare results from a home brew tearing mode solver against results from

Tearing mode in the cylindrical tokamak - H. P. Furth, P. H. Rutherford, and H. Selberg
"""

import numpy as np
import matplotlib.pyplot as plt

from plasma_physics.pysrc.theory.tearing_modes.cylindrical_tearing_modes import TearingModeSolver

def compare_figures():
    x_s_points = np.linspace(0.2, 1.6, 8)
    m = 2
    n = 1
    r_s = 0.05
    R = 1.0
    num_grid_pts = 100000
    B_z0 = 1.0
    integrate_from_bnds = True
    
    fig, ax = plt.subplots(2, sharex=True)
    r_Deltas = list()
    psi_s = list()
    for x_s in x_s_points:
        solver = TearingModeSolver(n, m, B_z0, x_s, r_s, R, num_grid_pts, integrate_from_bnds=integrate_from_bnds)
        solver.find_delta(plot_output=False)

        r_Delta = solver.r_s * (solver.A_upper - solver.A_lower)
        r_Deltas.append(r_Delta)
        psi_s.append(solver.psi_rs)

        ax[0].plot(solver.x_lower, solver.psi_sol_lower[:, 0], c='k')
        ax[0].plot(solver.x_upper, solver.psi_sol_upper[:, 0], c='k')
    ax[0].plot(x_s_points, psi_s, linestyle='--', c='k')
    ax[1].plot(x_s_points, r_Deltas)
    ax[0].set_ylabel("$\hat \Psi$")
    ax[1].set_ylabel("$r \Delta$")
    ax[1].set_xlabel("x")
    plt.show()

    r_Deltas = np.asarray(r_Deltas)
    psi_s = np.asarray(psi_s)
    if integrate_from_bnds:
        np.savetxt("rDelta_bnd", r_Deltas)
        np.savetxt("psi_s_bnd", psi_s)
    else:
        np.savetxt("rDelta_tearing", r_Deltas)
        np.savetxt("psi_s_tearing", psi_s)


if __name__ == '__main__':
    compare_figures()

