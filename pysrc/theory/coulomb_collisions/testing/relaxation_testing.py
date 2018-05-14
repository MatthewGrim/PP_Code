"""
Author: Rohan Ramasamy
Date: 09/03/2018

This file contains code to assess the relaxation rates for different relaxation
processes
"""

import numpy as np
from matplotlib import pyplot as plt

from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import CoulombCollision, ChargedParticle
from plasma_physics.pysrc.theory.coulomb_collisions.relaxation_processes import RelaxationProcess, MaxwellianRelaxationProcess
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


def get_maxwellian_collisional_frequencies():
    # Generate charged particles
    alpha = ChargedParticle(6.64424e-27, PhysicalConstants.electron_charge * 2)
    electron = ChargedParticle(9.014e-31, -PhysicalConstants.electron_charge)

    impact_parameter_ratio = 1.0
    beam_velocity = np.sqrt(2 * 3.5e6 * PhysicalConstants.electron_charge / alpha.m)
    n = 1e20
    temperature = 20e3 * UnitConversions.eV_to_K
    print(n, temperature, beam_velocity)

    # Get reactant collision frequency and energy loss rate
    collision = CoulombCollision(electron, alpha,
                                 impact_parameter_ratio,
                                 beam_velocity)
    relaxation = MaxwellianRelaxationProcess(collision)
    numerical_v_P = relaxation.momentum_loss_maxwellian_frequency(n, temperature, beam_velocity)

    theoretical_v_P = 2 / (3 * np.sqrt(2 * np.pi)) * relaxation.momentum_loss_stationary_frequency(n, temperature, beam_velocity)

    print("Numerical Momentum Loss Frequency: {}".format(numerical_v_P))
    print("Theoretical Momentum Loss Frequency: {}".format(theoretical_v_P))
    print("Numerical to theoretical momentum loss ratio: {}\n".format(numerical_v_P / theoretical_v_P))

    numerical_v_K = relaxation.kinetic_loss_maxwellian_frequency(n, temperature, beam_velocity)
    print("Numerical Kinetic Loss Frequency: {}".format(numerical_v_K))
    print("Numerical kinetic to momentum loss ratio: {}".format(numerical_v_K / numerical_v_P))


def compare_maxwellian_and_stationary_frequencies_against_density(number_density, temperature):
    # Generate charged particles
    alpha = ChargedParticle(6.64424e-27, PhysicalConstants.electron_charge * 2)
    electron = ChargedParticle(9.014e-31, -PhysicalConstants.electron_charge)

    # Get deuterium and alpha velocities from product and reactant beam
    # energies
    e_alpha = 3.5e3 * PhysicalConstants.electron_charge
    v_alpha = np.sqrt(2 * e_alpha / alpha.m)

    # Get product collision frequency and energy loss rate
    impact_parameter_ratio = 1.0  # Is not necessary for this analysis
    product_collision = CoulombCollision(alpha, electron,
                                         impact_parameter_ratio,
                                         v_alpha)
    product_relaxation = MaxwellianRelaxationProcess(product_collision)
    alpha_stationary_momentum_frequency = product_relaxation.momentum_loss_stationary_frequency(number_density, temperature, v_alpha)

    maxwellian_frequency = np.zeros(number_density.shape)
    alpha_maxwellian_momentum_frequency = np.zeros(number_density.shape)
    for i, n in enumerate(number_density):
        # Get maxwellian frequencies
        v = product_relaxation.maxwellian_collisional_frequency(n, temperature, v_alpha)
        maxwellian_frequency[i] = v
        v_P = product_relaxation.numerical_momentum_loss_maxwellian_frequency(n, temperature, v_alpha, epsrel=1e-8)
        alpha_maxwellian_momentum_frequency[i] = v_P

        print(n, v_P, v, np.abs(v_P - v) / v)

    fig, ax = plt.subplots()

    ax.loglog(number_density, alpha_stationary_momentum_frequency, label="Stationary")
    ax.loglog(number_density, maxwellian_frequency, label="Maxwellian")
    ax.loglog(number_density, alpha_maxwellian_momentum_frequency, label="Numerical Maxwellian")
    ax.legend()

    plt.show()


def compare_maxwellian_and_stationary_frequencies_against_temperature(number_density, temperature):
    # Generate charged particles
    alpha = ChargedParticle(6.64424e-27, PhysicalConstants.electron_charge * 2)
    electron = ChargedParticle(9.014e-31, -PhysicalConstants.electron_charge)

    # Get deuterium and alpha velocities from product and reactant beam
    # energies
    e_alpha = 1e3 * PhysicalConstants.electron_charge
    v_alpha = np.sqrt(2 * e_alpha / alpha.m)

    # Get product collision frequency and energy loss rate
    impact_parameter_ratio = 1.0  # Is not necessary for this analysis
    product_collision = CoulombCollision(alpha, electron,
                                         impact_parameter_ratio,
                                         v_alpha)
    product_relaxation = MaxwellianRelaxationProcess(product_collision)
    alpha_stationary_momentum_frequency = product_relaxation.momentum_loss_stationary_frequency(number_density, temperature, v_alpha)

    maxwellian_frequency = np.zeros(temperature.shape)
    alpha_maxwellian_momentum_frequency = np.zeros(temperature.shape)
    for i, T in enumerate(temperature):
        # Get maxwellian frequencies
        v = product_relaxation.maxwellian_collisional_frequency(number_density, T, v_alpha)
        maxwellian_frequency[i] = v
        v_P = product_relaxation.numerical_momentum_loss_maxwellian_frequency(number_density, T, v_alpha, epsrel=1e-8)
        alpha_maxwellian_momentum_frequency[i] = v_P

        print(T, v_P, v_P - alpha_stationary_momentum_frequency[i], np.abs(v_P - v) / v)

    fig, ax = plt.subplots()

    temperature *= UnitConversions.K_to_eV
    ax.loglog(temperature, alpha_stationary_momentum_frequency, label="Stationary")
    ax.loglog(temperature, maxwellian_frequency, label="Maxwellian")
    ax.loglog(temperature, alpha_maxwellian_momentum_frequency, label="Numerical Maxwellian")
    ax.legend()

    plt.show()


if __name__ == '__main__':
    # plot_collisional_frequencies()
    # get_maxwellian_collisional_frequencies()

    # number_density = np.logspace(20, 30, 5)
    # temperature = 200e3 * UnitConversions.eV_to_K
    # compare_maxwellian_and_stationary_frequencies_against_density(number_density, temperature)

    number_density = 1e20
    temperature = np.logspace(1, 6, 6) * UnitConversions.eV_to_K
    compare_maxwellian_and_stationary_frequencies_against_temperature(number_density, temperature)
