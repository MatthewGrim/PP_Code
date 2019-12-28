"""
Author: Rohan Ramasamy
Date: 19/12/2019

This file contains code to replicate stability diagram results from the paper:

Magnetohydrodynamic instabilities in a current- carrying stellarator - K. Matsuoka et al.
"""

import numpy as np
import matplotlib.pyplot as plt

from plasma_physics.pysrc.theory.newcomb_stability.newcomb_solver import NewcombSolver


def get_stability_diagrams():
    def iota_vac_l2(r, a, iota_0v):
        iota_v = iota_0v * np.ones(r.shape) if isinstance(r, np.ndarray) else iota_0v
        return iota_v, 0

    def iota_vac_l23(r, a, iota_0v):
        iota = iota_0v * (0.286 - 0.714 * r ** 2 / a ** 2)
        iota_deriv = -1.428 * iota_0v * r / a ** 2
        return iota, iota_deriv

    def iota_vac_l3(r, a, iota_0v):
        iota = iota_0v * r ** 2 / a ** 2
        iota_deriv = 2 * iota_0v * r / a ** 2
        return iota, iota_deriv

    def iota_plasma_flat(r, a, iota_0p):
        if isinstance(r, np.ndarray):
            iota = np.zeros(r.shape)
            iota_deriv = np.zeros(r.shape)
            iota_deriv2 = np.zeros(r.shape)
            iota[r < a] = iota_0p * (1 - r[r < a] ** 6 / (4 * a ** 6))
            iota_deriv[r < a] = -3 * iota_0p * r[r < a] ** 5 / (2 * a ** 6)
            iota_deriv2[r < a] = -15 * iota_0p * r[r < a] ** 4 / (2 * a ** 6)
            iota[r > a] = iota_0p * 0.75 * a ** 2 / r[r > a] ** 2
            iota_deriv[r > a] = iota_0p * -1.5 * a ** 2 / r[r > a] ** 3
            iota_deriv2[r > a] = iota_0p * 4.5 * a ** 2 / r[r > a] ** 4        
        else:
            if r < a:
                iota = iota_0p * (1 - r ** 6 / (4 * a ** 6))
                iota_deriv = -3 * iota_0p * r ** 5 / (2 * a ** 6)
                iota_deriv2 = -15 * iota_0p * r ** 4 / (2 * a ** 6)
            else:
                iota = iota_0p * 0.75 * a ** 2 / r ** 2
                iota_deriv = iota_0p * -1.5 * a ** 2 / r ** 3
                iota_deriv2 = iota_0p * 4.5 * a ** 2 / r ** 4      
        
        return iota, iota_deriv, iota_deriv2

    def iota_parabola(r, a, j_0):
        j = j_0 * (1 - r ** 2 / a ** 2)
        j_deriv = j_0 * (1 - 2 * r / a ** 2)
        return j, j_deriv

    def iota_peaked(r, a, j_0):
        j = j_0 * (1 - r ** 2 / a ** 2) ** 4
        j_deriv = - 8 * r / a ** 2 * (1 - r ** 2 / a ** 2) ** 3
        return j, j_deriv

    def f(r):
        return r

    def f_deriv(r):
        if isinstance(r, np.ndarray):
            return np.ones(r.shape)
        else:
            return 1
    
    def g(r):
        iota_ext, _ = iota_vac_l2(r, a, iota_0v) 
        iota_plasma, iota_plasma_deriv, iota_plasma_deriv2 = iota_plasma_flat(r, a, iota_0p)
        nu = -n / m + iota_ext + iota_plasma
        return m ** 2 + 1 / (r * nu) * (3 * r ** 2 * iota_plasma_deriv + r ** 3 * iota_plasma_deriv2)


    def singular_func(r):
        iota_ext, _ = iota_vac_l2(r, a, iota_0v) 
        iota_plasma, iota_plasma_deriv, iota_plasma_deriv2 = iota_plasma_flat(r, a, iota_0p)
        nu = -n / m + iota_ext + iota_plasma
        return nu

    r_minor = np.linspace(0.0, b, num_pts)
    
    iota_v, _ = iota_vac_l2(r_minor, a, iota_0v)
    iota_p, iota_p_deriv, iota_p_deriv2 = iota_plasma_flat(r_minor, a, iota_0p)
    
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

    solver = NewcombSolver(m, r_minor, a, f, g, a, b, n=n, singularity_func=singular_func, f_deriv=f_deriv)
    solver.find_zero_crossings_in_f()
    solver.determine_stability(plot_results=True)


if __name__ == '__main__':
    R = 20
    k = 0.05
    a = 1
    b = 1.44
    B_z = 1.0
    iota_0v = 0.2
    iota_0p = 0.5

    m = 1
    n = 1
    num_pts = 1001
    get_stability_diagrams()

