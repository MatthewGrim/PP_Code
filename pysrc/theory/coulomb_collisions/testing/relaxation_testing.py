"""
Author: Rohan Ramasamy
Date: 09/03/2018

This file contains code to assess the relaxation rates for different relaxation
processes
"""

import numpy as np
from matplotlib import pyplot as plt

from pysrc.theory.coulomb_collisions.coulomb_collision import CoulombCollision, ChargedParticle
from pysrc.theory.coulomb_collisions.relaxation_processes import RelaxationProcess
from pysrc.utils.unit_conversions import UnitConversions
from pysrc.utils.physical_constants import PhysicalConstants


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


def get_maxwellian_collisional_frequencies():
    # Generate charged particles
    deuterium = ChargedParticle(2.01410178 * 1.66054e-27, PhysicalConstants.electron_charge)
    electron = ChargedParticle(9.014e-31, -PhysicalConstants.electron_charge)

    impact_parameter_ratio = 1.0
    beam_velocity = 5e6
    n = 1e25
    temperature = 20e3 * UnitConversions.eV_to_K

    # Get reactant collision frequency and energy loss rate
    collision = CoulombCollision(electron, deuterium,
                                 impact_parameter_ratio,
                                 beam_velocity)
    relaxation = RelaxationProcess(collision)
    numerical_v_P = relaxation.momentum_loss_maxwellian_frequency(n, temperature, beam_velocity)

    theoretical_v_P = 2 / (3 * np.sqrt(2 * np.pi)) * relaxation.momentum_loss_stationary_frequency(n, temperature, beam_velocity)

    print("Numerical Momentum Loss Frequency: {}".format(numerical_v_P[0]))
    print("Theoretical Momentum Loss Frequency: {}".format(theoretical_v_P))
    print("Ratio: {}\n".format(numerical_v_P[0] / theoretical_v_P))

    numerical_v_K = relaxation.kinetic_loss_maxwellian_frequency(n, temperature, beam_velocity)
    print("Numerical Kinetic Loss Frequency: {}".format(numerical_v_K[0]))


def compare_product_and_reactant_energy_loss_rates(number_density, temperature, plot_energy_frequencies=True):
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

    # Get reactant collision frequency and energy loss rate
    impact_parameter_ratio = 1.0  # Is not necessary for this analysis
    reactant_collision = CoulombCollision(deuterium, electron,
                                          impact_parameter_ratio,
                                          v_deuterium)
    reactant_relaxation = RelaxationProcess(reactant_collision)
    deuterium_kinetic_frequency = reactant_relaxation.kinetic_loss_stationary_frequency(number_density, temperature, v_deuterium)
    deuterium_energy_loss_rates = reactant_relaxation.energy_loss_with_distance(number_density, temperature, v_deuterium)
    deuterium_momentum_frequency = reactant_relaxation.momentum_loss_stationary_frequency(number_density, temperature, v_deuterium)
    deuterium_momentum_loss_rates = deuterium_momentum_frequency * deuterium.m

    # Get product collision frequency and energy loss rate
    product_collision = CoulombCollision(alpha, electron,
                                         impact_parameter_ratio,
                                         v_alpha)
    product_relaxation = RelaxationProcess(product_collision)
    alpha_kinetic_frequency = product_relaxation.kinetic_loss_stationary_frequency(number_density, temperature, v_alpha)
    alpha_energy_loss_rates = product_relaxation.energy_loss_with_distance(number_density, temperature, v_alpha)
    alpha_momentum_frequency = product_relaxation.momentum_loss_stationary_frequency(number_density, temperature, v_alpha)
    alpha_momentum_loss_rates = alpha_momentum_frequency * alpha.m

    # Compare velocities of particles
    v_electron = np.sqrt(2 * PhysicalConstants.boltzmann_constant * temperature / electron.m)
    print("For stationary collisions to be a reasonable approximation, the beams need "
          "to be travelling faster than the electron thermal velocity... ")
    print("Electron Normalised Velocity: {}".format(v_electron / 3e8))
    print("Alpha-Electron Velocity Ratio: {}".format(v_alpha / v_electron))
    print("Deuterium-Electron Velocity Ratio: {}".format(v_deuterium / v_electron))

    print(deuterium_energy_loss_rates * e_alpha / alpha_energy_loss_rates / e_deuterium)

    fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    if plot_energy_frequencies:
        # energy results
        ax[0, 0].loglog(number_density, deuterium_kinetic_frequency, label="Reactant Beam")
        ax[0, 0].loglog(number_density, alpha_kinetic_frequency, label="Product Beam")
        ax[0, 0].set_xlabel("Number density (m-3)")
        ax[0, 0].set_ylabel("Energy Collision Frequency (s-1)")
        ax[0, 0].set_title("Energy Collision Frequencies")
        ax[0, 0].legend()

        ax[1, 0].loglog(number_density, deuterium_energy_loss_rates, label="Reactant Beam")
        ax[1, 0].loglog(number_density, alpha_energy_loss_rates, label="Product Beam")
        ax[1, 0].set_xlabel("Number density (m-3)")
        ax[1, 0].set_ylabel("Energy Loss Rate per metre (Jm-1)")
        ax[1, 0].set_title("Energy Loss Rates per metre")
        ax[1, 0].legend()

        ax[0, 1].loglog(number_density, deuterium_energy_loss_rates / e_deuterium, label="Reactant Beam")
        ax[0, 1].loglog(number_density, alpha_energy_loss_rates / e_alpha, label="Product Beam")
        ax[0, 1].axhline(0.1, linestyle="--")
        ax[0, 1].set_xlabel("Number density (m-3)")
        ax[0, 1].set_ylabel("Normalised Kinetic Loss Rate (m-1)")
        ax[0, 1].set_title("Normalised Kinetic Loss Rate")
        ax[0, 1].legend()

        ax[1, 1].loglog(number_density, e_deuterium / deuterium_energy_loss_rates, label="Reactant Beam")
        ax[1, 1].loglog(number_density, e_alpha / alpha_energy_loss_rates, label="Product Beam")
        ax[1, 1].set_xlabel("Number density (m-3)")
        ax[1, 1].set_ylabel("Distance traveller (m)")
        ax[1, 1].set_title("Distance travelled")
        ax[1, 1].legend()
    else:
        # momentum results
        ax[0, 0].loglog(number_density, deuterium_momentum_frequency, label="Reactant Beam")
        ax[0, 0].loglog(number_density, alpha_momentum_frequency, label="Product Beam")
        ax[0, 0].set_xlabel("Number density (m-3)")
        ax[0, 0].set_ylabel("Momentum Collision Frequency (s-1)")
        ax[0, 0].set_title("Momentum Collision Frequencies")
        ax[0, 0].legend()

        ax[1, 0].loglog(number_density, deuterium_momentum_loss_rates, label="Reactant Beam")
        ax[1, 0].loglog(number_density, alpha_momentum_loss_rates, label="Product Beam")
        ax[1, 0].set_xlabel("Number density (m-3)")
        ax[1, 0].set_ylabel("Momentum Loss Rate per metre (Jm-1)")
        ax[1, 0].set_title("Momentum Loss Rates per metre")
        ax[1, 0].legend()

        deuterium_momentum = deuterium.m * v_deuterium
        alpha_momentum = alpha.m * v_alpha
        ax[0, 1].loglog(number_density, deuterium_momentum_loss_rates / deuterium_momentum, label="Reactant Beam")
        ax[0, 1].loglog(number_density, alpha_momentum_loss_rates / alpha_momentum, label="Product Beam")
        ax[0, 1].axhline(0.1, linestyle="--")
        ax[0, 1].set_xlabel("Number density (m-3)")
        ax[0, 1].set_ylabel("Normalised Momentum Loss Rate (m-1)")
        ax[0, 1].set_title("Normalised Momentum Loss Rate")
        ax[0, 1].legend()

        ax[1, 1].loglog(number_density, deuterium_momentum / deuterium_momentum_loss_rates, label="Reactant Beam")
        ax[1, 1].loglog(number_density, alpha_momentum / alpha_momentum_loss_rates, label="Product Beam")
        ax[1, 1].set_xlabel("Number density (m-3)")
        ax[1, 1].set_ylabel("Distance traveller (m)")
        ax[1, 1].set_title("Distance travelled")
        ax[1, 1].legend()

    # fig.suptitle("Comparison of product and reactant beam loss rates due to electron collisions")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_collisional_frequencies()
    number_density = np.logspace(18, 30, 100)
    temperature = 20e3 * UnitConversions.eV_to_K
    compare_product_and_reactant_energy_loss_rates(number_density, temperature)
    # get_maxwellian_collisional_frequencies()
