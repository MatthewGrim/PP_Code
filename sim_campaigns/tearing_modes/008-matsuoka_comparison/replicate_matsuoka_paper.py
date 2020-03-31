"""
Author: Rohan Ramasamy
Date: 19/12/2019

This file contains code to replicate stability diagram results from the paper:

Magnetohydrodynamic instabilities in a current- carrying stellarator - K. Matsuoka et al.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from plasma_physics.pysrc.theory.newcomb_stability.newcomb_solver import NewcombSolver
from plasma_physics.sim_campaigns.tearing_modes.iota_profiles import *


def get_stability_diagrams():
    for m in m_values:
        for i, iota_0v in enumerate(iota_0v_samples):
            for j, iota_0p in enumerate(iota_0p_samples):
                print(m, iota_0v, iota_0p)

                if stability_results[i, j] == 1:
                    continue

                def g(r):
                    iota_ext, _ = params[0](r, a, iota_0v) 
                    iota_plasma, iota_plasma_deriv, iota_plasma_deriv2 = params[1](r, a, iota_0p)
                    nu = -n / m + iota_ext + iota_plasma
                    return (m ** 2 + 1 / (r * nu) * (3 * r ** 2 * iota_plasma_deriv + r ** 3 * iota_plasma_deriv2)) / r

                def singular_func(r):
                    iota_ext, _ = params[0](r, a, iota_0v) 
                    iota_plasma, iota_plasma_deriv, iota_plasma_deriv2 = params[1](r, a, iota_0p)
                    nu = -n / m + iota_ext + iota_plasma
                    return nu

                if np.isclose(singular_func(0.0), 0.0, rtol=1e-3, atol=1e-3):
                    continue

                r_minor = np.linspace(0.0, b, num_pts)
                
                iota_v, _ = params[0](r_minor, a, iota_0v)
                iota_p, iota_p_deriv, iota_p_deriv2 = params[1](r_minor, a, iota_0p)

                solver = NewcombSolver(m, r_minor, a, f, g, a, b, n=n, singularity_func=singular_func, f_deriv=f_deriv)
                solver.find_zero_crossings_in_f()
                stability_results[i, j] = solver.determine_stability(plot_results=False)

    data = np.loadtxt(os.path.join('datasets', '{}.csv'.format(params[2])), delimiter=',')

    fig, ax = plt.subplots(1)
    ax.contourf(iota_0p_samples, iota_0v_samples, stability_results, 100)
    num_data_pts = params[3]
    for i in range(data.shape[0] // num_data_pts):
        ax.plot(data[i*num_data_pts:(i+1)*num_data_pts, 0], data[i*num_data_pts:(i+1)*num_data_pts, 1], linestyle='--', c='white')
    plt.savefig('{}_stability_diagram'.format(params[2]))
    plt.show()


if __name__ == '__main__':
    R = 20
    k = 0.05
    a = 1
    b = 3.0
    B_z = 1.0
    num_samples = 51
    stability_results = np.ones((num_samples, num_samples)) * -1
    iota_0v_samples = np.linspace(0.0, 1.0, num_samples)
    iota_0p_samples = np.linspace(0.0, 1.0, num_samples)
    # params = [iota_vac_l2, iota_plasma_flat, 'l2_flat', 3]
    params = [iota_vac_l2, iota_plasma_parabola, 'l2_parabolic', 3]
    # params = [iota_vac_l2, iota_plasma_peaked, 'l2_peaked', 3]
    # params = [iota_vac_l23, iota_plasma_flat, 'l23_flat', 6]
    # params = [iota_vac_l23, iota_plasma_parabola, 'l23_parabolic', 6]
    # params = [iota_vac_l23, iota_plasma_peaked, 'l23_peaked', 4]

    m_values = [1, 2, 3]
    n = 1
    num_pts = 1001
    get_stability_diagrams()

