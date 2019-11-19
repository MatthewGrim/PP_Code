"""
Author: Rohan Ramasamy
Date: 25/08/2019

This file contains code to assess the stability of tearing modes in a circular plasma, where the FF' profile is flattened 
from the original distribution.
"""

import numpy as np
import sys
import os
from scipy import integrate, interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants
from plasma_physics.pysrc.theory.tearing_modes.cylindrical_tearing_modes import TearingModeSolver


def get_ffp_stability_condition(plot_results=True):
    ffp_factors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    deltas = list()
    alphas = list()
    for factor in ffp_factors:
        # Modify and plot input profiles
        dat = np.loadtxt(os.path.join('FFp_profiles', 'input_profiles_{}.dat'.format(factor)))
        r = dat[:, 0]
        R = dat[:, 1]
        j_phi = dat[:, 2]
        B_z = dat[:, 3]
        B_theta = dat[:, 4]
        psi_n = dat[:, 5]
        r_to_psin = interpolate.interp1d(r, psi_n)
        
        q = r * B_z / (R * B_theta)
        q[0] = q[1]
        if plot_results:
            fig, ax = plt.subplots(2, 2, sharex=True)

            ax[0, 0].plot(r, B_z)
            ax[0, 0].set_ylabel('B_tor')
            ax[1, 0].plot(r, B_theta)
            ax[1, 0].set_ylabel('B_theta')
            ax[0, 1].plot(r, j_phi)
            ax[0, 1].set_ylabel('j_phi')
            ax[1, 1].plot(r, q)
            ax[1, 1].set_ylabel('q')
            ax[0, 0].set_xlim([0.0, 1.0])
            plt.show()

        r_max = r[-1]
        solver = TearingModeSolver(m, r, r_max, B_theta, B_z, j_phi, q, num_pts, delta=1e-12)
        solver.find_delta(plot_output=False)

        psin_lower = r_to_psin(solver.r_lower)
        psin_upper = r_to_psin(solver.r_upper)
        
        if plot_results:
            fig, ax = plt.subplots(2, sharex=True)
            ax[0].plot(psin_lower, solver.psi_sol_lower[:, 0])
            ax[0].plot(psin_upper, solver.psi_sol_upper[:, 0])
            ax[0].set_ylabel('$\Psi$')
            ax[0].set_xlabel('$\hat \Psi$')
            ax[0].set_xlim([0.0, 1.0])
            
            ax[1].plot(psin_lower, np.abs(solver.psi_sol_lower[:, 1]))
            ax[1].plot(psin_upper, np.abs(solver.psi_sol_upper[:, 1]))
            ax[1].set_ylabel('$\\frac{\partial \Psi}{\partial r}$')
            ax[1].set_xlabel('$\hat \Psi$')
            plt.show()

        deltas.append(solver.delta)
        alphas.append(1 - factor)

    plt.figure()
    plt.plot(alphas, deltas)
    plt.axhline(0.0, linestyle='--', color='grey')
    plt.ylabel('$\Delta$')
    plt.xlabel('$\\alpha$')
    plt.title('Stability profile for cylindrical plasma with locally varying FFprime')
    plt.savefig('ffp_stability_local')
    plt.show()

    np.savetxt('deltas.dat', deltas)

def get_ffp_stability_condition_with_vc(plot_results=True):
    ffp_factors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # Modify and plot input profiles
    dat = np.loadtxt(os.path.join('FFp_profiles', 'input_profiles_0.0.dat'))
    r = dat[:, 0]
    R = dat[:, 1]
    j_phi = dat[:, 2]
    B_z = dat[:, 3]
    B_theta = dat[:, 4]
    psi_n = dat[:, 5]
    r_to_psin = interpolate.interp1d(r, psi_n)
    
    q = r * B_z / (R * B_theta)
    q[0] = q[1]
    if plot_results:
        fig, ax = plt.subplots(2, 2, sharex=True)

        ax[0, 0].plot(r, B_z)
        ax[0, 0].set_ylabel('B_tor')
        ax[1, 0].plot(r, B_theta)
        ax[1, 0].set_ylabel('B_theta')
        ax[0, 1].plot(r, j_phi)
        ax[0, 1].set_ylabel('j_phi')
        ax[1, 1].plot(r, q)
        ax[1, 1].set_ylabel('q')
        ax[0, 0].set_xlim([0.0, 1.0])
        plt.show()

    r_max = r[-1]    
    deltas = list()
    alphas = list()
    for factor in ffp_factors:
        # Update only j_phi to represent virtual current
        dat = np.loadtxt(os.path.join('FFp_profiles', 'input_profiles_{}.dat'.format(factor)))
        j_phi = dat[:, 2]

        solver = TearingModeSolver(m, r, r_max, B_theta, B_z, j_phi, q, num_pts, delta=1e-12)
        solver.find_delta(plot_output=False)

        psin_lower = r_to_psin(solver.r_lower)
        psin_upper = r_to_psin(solver.r_upper)
        
        if plot_results:
            fig, ax = plt.subplots(2, sharex=True)
            ax[0].plot(psin_lower, solver.psi_sol_lower[:, 0])
            ax[0].plot(psin_upper, solver.psi_sol_upper[:, 0])
            ax[0].set_ylabel('$\Psi$')
            ax[0].set_xlabel('$\hat \Psi$')
            ax[0].set_xlim([0.0, 1.0])
            
            ax[1].plot(psin_lower, np.abs(solver.psi_sol_lower[:, 1]))
            ax[1].plot(psin_upper, np.abs(solver.psi_sol_upper[:, 1]))
            ax[1].set_ylabel('$\\frac{\partial \Psi}{\partial r}$')
            ax[1].set_xlabel('$\hat \Psi$')
            plt.show()

        deltas.append(solver.delta)
        alphas.append(1 - factor)

    plt.figure()
    plt.plot(alphas, deltas)
    plt.axhline(0.0, linestyle='--', color='grey')
    plt.ylabel('$\Delta$')
    plt.xlabel('$\\alpha$')
    plt.title('Stability profile for cylindrical plasma with locally varying FFprime')
    plt.savefig('ffp_stability_local_vc')
    plt.show()

    np.savetxt('deltas_vc.dat', deltas)


if __name__ == '__main__':
    num_pts = 100000
    m = 2

    # get_ffp_stability_condition(plot_results=False)
    # get_ffp_stability_condition_with_vc(plot_results=False)

    ffp_factors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    deltas = np.loadtxt('deltas.dat')
    deltas_vc = np.loadtxt('deltas_vc.dat')
    plt.figure()
    plt.plot(ffp_factors, deltas)
    plt.plot(ffp_factors, deltas_vc)
    plt.xlabel('$\\alpha$')
    plt.ylabel('$\Delta$')
    plt.show()

