"""
Author: Rohan Ramasamy
Date: 09/03/2018

This file contains code to assess the relaxation rates for different relaxation
processes
"""

import numpy as np
from matplotlib import pyplot as plt

from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import CoulombCollision, ChargedParticle
from plasma_physics.pysrc.theory.coulomb_collisions.relaxation_processes import RelaxationProcess
from plasma_physics.pysrc.utils.unit_conversions import UnitConversions
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


def plot_collisional_frequencies():
    """
    Plot the variation of collisional frequency with density, and temperature for different
    collisions
    """
    deuterium = ChargedParticle(2.01410178 * 1.66054e-27, PhysicalConstants.electron_charge)
    electron = ChargedParticle(9.014e-31, -PhysicalConstants.electron_charge)

    impact_parameter_ratio = 1.0  # Is not necessary for this analysis
    velocities = np.logspace(4, 6, 100)
    number_density = np.logspace(20, 30, 100)
    VEL, N = np.meshgrid(velocities, number_density, indexing='ij')

    collision_pairs = [["Deuterium-Deuterium", deuterium, deuterium],
                       ["Electron-Deuterium", electron, deuterium],
                       ["Electron-Electron", electron, electron]]
    num_temp = 3
    temperatures = np.logspace(3, 5, num_temp) * UnitConversions.eV_to_K
    for j, pair in enumerate(collision_pairs):
        name = pair[0]
        p_1 = pair[1]
        p_2 = pair[2]

        fig, ax = plt.subplots(2, num_temp, figsize=(15, 20))
        fig.suptitle("Collisional Frequencies for {} Relaxation".format(name))
        for i, temp in enumerate(temperatures):
            collision = CoulombCollision(p_1, p_2, impact_parameter_ratio, VEL)
            relaxation_process = RelaxationProcess(collision)

            kinetic_frequency = relaxation_process.kinetic_loss_stationary_frequency(N, temp, VEL)
            momentum_frequency = relaxation_process.momentum_loss_stationary_frequency(N, temp, VEL, 
                                                                                       first_background=True)

            im = ax[0, i].contourf(np.log10(VEL), np.log10(N), np.log10(kinetic_frequency), 100)
            ax[0, i].set_title("Kinetic: T = {}".format(temp * UnitConversions.K_to_eV))
            ax[0, i].set_xlabel("Velocity (ms-1)")
            ax[0, i].set_ylabel("Number density (m-3)")
            fig.colorbar(im, ax=ax[0, i])

            im = ax[1, i].contourf(np.log10(VEL), np.log10(N), np.log10(momentum_frequency), 100)
            ax[1, i].set_title("Momentum: T = {}".format(temp * UnitConversions.K_to_eV))
            ax[1, i].set_xlabel("Velocity (ms-1)")
            ax[1, i].set_ylabel("Number density (m-3)")
            fig.colorbar(im, ax=ax[1, i])

        plt.show()


def compare_product_and_reactant_energy_loss_rates():
    # Generate charged particles
    alpha = ChargedParticle(6.64424e-27, PhysicalConstants.electron_charge * 2)
    deuterium = ChargedParticle(2.01410178 * 1.66054e-27, PhysicalConstants.electron_charge)
    electron = ChargedParticle(9.014e-31, -PhysicalConstants.electron_charge)

    # Get deuterium and alpha velocities from product and reactant beam
    # energies
    e_alpha = 3.5e6 * PhysicalConstants.electron_charge
    v_alpha = np.sqrt(2 * e_alpha / alpha.m)
    e_deuterium = 50e3 * PhysicalConstants.electron_charge
    v_deuterium = np.sqrt(2 * e_deuterium / deuterium.m)

    impact_parameter_ratio = 1.0  # Is not necessary for this analysis
    number_density = np.logspace(18, 30, 100)
    temperature = 20e3 * UnitConversions.eV_to_K
    alpha_frequency = np.zeros(number_density.shape)
    deuterium_frequency = np.zeros(number_density.shape)
    alpha_loss_rate = np.zeros(number_density.shape)
    deuterium_loss_rate = np.zeros(number_density.shape)
    for i, n in enumerate(number_density):
        # Get reactant collision frequency and energy loss rate
        reactant_collision = CoulombCollision(deuterium, electron,
                                              impact_parameter_ratio,
                                              v_deuterium)
        reactant_relaxation = RelaxationProcess(reactant_collision)
        reactant_v_K = reactant_relaxation.kinetic_loss_stationary_frequency(n, temperature, v_deuterium)
        reactant_loss_rate = reactant_relaxation.energy_loss_with_distance(n, temperature, v_deuterium)

        # Get product collision frequency and energy loss rate
        product_collision = CoulombCollision(alpha, electron,
                                             impact_parameter_ratio,
                                             v_alpha)
        product_relaxation = RelaxationProcess(product_collision)
        product_v_K = product_relaxation.kinetic_loss_stationary_frequency(n, temperature, v_alpha)
        product_loss_rate = product_relaxation.energy_loss_with_distance(n, temperature, v_alpha)

        # Store results
        deuterium_loss_rate[i] = reactant_loss_rate
        deuterium_frequency[i] = reactant_v_K
        alpha_loss_rate[i] = product_loss_rate
        alpha_frequency[i] = product_v_K

    v_electron = np.sqrt(2 * PhysicalConstants.boltzmann_constant * temperature / electron.m)

    print("For stationary collisions to be a reasonable approximation, the beams need to be travelling faster than the electron thermal velocity... ")
    print("Alpha-Electron Velocity Ratio: {}".format(v_alpha / v_electron))
    print("Deuterium-Electron Velocity Ratio: {}".format(v_deuterium / v_electron))

    fig, ax = plt.subplots(3, figsize=(10, 10))

    ax[0].loglog(number_density, deuterium_frequency, label="Reactant Beam")
    ax[0].loglog(number_density, alpha_frequency, label="Product Beam")
    ax[0].set_xlabel("Number density (m-3)")
    ax[0].set_ylabel("Collision Frequency (s-1)")
    ax[0].set_title("Beam Collision Frequencies")
    ax[0].legend()

    ax[1].loglog(number_density, deuterium_loss_rate, label="Reactant Beam")
    ax[1].loglog(number_density, alpha_loss_rate, label="Product Beam")
    ax[1].set_xlabel("Number density (m-3)")
    ax[1].set_ylabel("Energy Loss Rate per metre (Jm-1)")
    ax[1].set_title("Energy Loss Rates per metre")
    ax[1].legend()

    ax[2].loglog(number_density, deuterium_loss_rate / e_deuterium, label="Reactant Beam")
    ax[2].loglog(number_density, alpha_loss_rate / e_alpha, label="Product Beam")
    ax[2].axhline(0.1, linestyle="--")
    ax[2].set_xlabel("Number density (m-3)")
    ax[2].set_ylabel("Normalised Kinetic Loss Rate (m-1)")
    ax[2].set_title("Normalised Kinetic Loss Rate")
    ax[2].legend()

    plt.title("Comparison of product and reactant beam loss rates due to electron collisions")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_collisional_frequencies()
    compare_product_and_reactant_energy_loss_rates()
