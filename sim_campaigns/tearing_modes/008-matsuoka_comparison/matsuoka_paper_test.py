"""
Author: Rohan Ramasamy
Date: 19/12/2019

This file contains code to replicate stability diagram results from the paper:

Magnetohydrodynamic instabilities in a current- carrying stellarator - K. Matsuoka et al.
"""

import numpy as np
import matplotlib.pyplot as plt

from plasma_physics.pysrc.theory.newcomb_stability.newcomb_solver import NewcombSolver
from plasma_physics.sim_campaigns.tearing_modes.iota_profiles import *


def get_stability_diagrams():
    def g(r):
        iota_ext, _ = iota_vac_func(r, a, iota_0v) 
        iota_plasma, iota_plasma_deriv, iota_plasma_deriv2 = iota_plasma_func(r, a, iota_0p)
        nu = -n / m + iota_ext + iota_plasma
        return (m ** 2 + 1 / (r * nu) * (3 * r ** 2 * iota_plasma_deriv + r ** 3 * iota_plasma_deriv2)) / r


    def singular_func(r):
        iota_ext, _ = iota_vac_func(r, a, iota_0v) 
        iota_plasma, iota_plasma_deriv, iota_plasma_deriv2 = iota_plasma_func(r, a, iota_0p)
        nu = -n / m + iota_ext + iota_plasma
        return nu

    r_minor = np.linspace(0.0, b, num_pts)
    
    iota_v, _ = iota_vac_func(r_minor, a, iota_0v)
    iota_p, iota_p_deriv, iota_p_deriv2 = iota_plasma_func(r_minor, a, iota_0p)
    
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(r_minor, iota_v, c='k', linestyle='--', label='Vacuum')
    ax[0].plot(r_minor, iota_p, c='grey', linestyle='--', label='Plasma')
    ax[0].plot(r_minor, iota_v + iota_p, c='k', label='Total')
    ax[0].set_ylabel('$\iota$')
    ax[0].set_xlabel('r')
    ax[0].set_xlim([0.0, b])
    
    ax[1].plot(r_minor, iota_p_deriv)
    ax[1].plot(r_minor, iota_p_deriv2)
    
    plt.show()
    
    fig, ax = plt.subplots(4, sharex=True)
    ax[0].plot(r_minor, f(r_minor))
    ax[0].set_ylabel('f(r)')
    ax[1].plot(r_minor, f_deriv(r_minor))
    ax[1].set_ylabel('f_deriv(r)')
    ax[2].plot(r_minor, singular_func(r_minor))
    ax[2].set_ylabel('singular(r)')
    ax[3].plot(r_minor, g(r_minor))
    ax[3].set_ylabel('g(r)')
    ax[3].set_xlabel('r')
    ax[3].set_xlim([0.0, b])
    plt.show()

    solver = NewcombSolver(m, r_minor, a, f, g, a, b, n=n, singularity_func=singular_func, f_deriv=f_deriv, verbose=True)
    solver.find_zero_crossings_in_f()
    solver.determine_stability(plot_results=True)


if __name__ == '__main__':
    R = 20
    k = 0.05
    a = 1
    b = 3.0
    B_z = 1.0
    iota_0p = 0.18
    iota_0v = 0.8
    m = 1
    n = 1
    num_pts = 1001
    iota_vac_func = iota_vac_l2
    iota_plasma_func = iota_plasma_flat
    get_stability_diagrams()

