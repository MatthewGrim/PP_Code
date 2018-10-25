"""
Author: Rohan Ramasamy
Date: 14/05/2018

This script contains code to analyse using simple theoretical models the thermalisation rates of different IEC devices.
"""

import numpy as np
from matplotlib import pyplot as plt

from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import CoulombCollision, ChargedParticle
from plasma_physics.pysrc.theory.coulomb_collisions.relaxation_processes import RelaxationProcess
from plasma_physics.pysrc.utils.unit_conversions import UnitConversions
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


def compare_product_and_reactant_energy_loss_rates(number_density, reaction_name,
                                                   plot_results=True, plot_energy_frequencies=True):
    # Generate charged particles
    alpha = ChargedParticle(6.64424e-27, PhysicalConstants.electron_charge * 2)
    electron = ChargedParticle(9.014e-31, -PhysicalConstants.electron_charge)
    if reaction_name == "DT":
        reactant = ChargedParticle(5.0064125184e-27, 2 * PhysicalConstants.electron_charge)
        e_alpha = 3.5e6 * PhysicalConstants.electron_charge
        e_reactant = 30e3 * PhysicalConstants.electron_charge
        Z = 2
    elif reaction_name == "pB":
        reactant = ChargedParticle((10.81 + 1) * UnitConversions.amu_to_kg, PhysicalConstants.electron_charge * 12)
        e_alpha = 8.6e6 / 3 * PhysicalConstants.electron_charge
        e_reactant = 500e3 * PhysicalConstants.electron_charge
        Z = 6
    else:
        raise ValueError("Name is invalid!")
    temperature = e_reactant

    # Get deuterium and alpha velocities from product and reactant beam
    # energies
    v_alpha = np.sqrt(2 * e_alpha / alpha.m)
    v_reactant = np.sqrt(2 * e_reactant / reactant.m)

    # Get reactant collision frequency and energy loss rate
    impact_parameter_ratio = 1.0  # Is not necessary for this analysis
    reactant_collision = CoulombCollision(reactant, reactant,
                                          impact_parameter_ratio,
                                          v_reactant)
    reactant_relaxation = RelaxationProcess(reactant_collision)
    deuterium_kinetic_frequency = reactant_relaxation.kinetic_loss_stationary_frequency(Z * number_density, temperature, v_reactant)
    deuterium_energy_loss_rates = reactant_relaxation.energy_loss_with_distance(Z * number_density, temperature, v_reactant)
    deuterium_momentum_frequency = reactant_relaxation.momentum_loss_stationary_frequency(Z * number_density, temperature, v_reactant)
    deuterium_momentum_loss_rates = deuterium_momentum_frequency * reactant.m

    # Get product collision frequency and energy loss rate
    product_collision = CoulombCollision(alpha, reactant,
                                         impact_parameter_ratio,
                                         v_alpha)
    product_relaxation = RelaxationProcess(product_collision)
    alpha_kinetic_frequency = product_relaxation.kinetic_loss_stationary_frequency(Z * number_density, temperature, v_alpha)
    alpha_energy_loss_rates = product_relaxation.energy_loss_with_distance(Z * number_density, temperature, v_alpha)
    alpha_momentum_frequency = product_relaxation.momentum_loss_stationary_frequency(Z * number_density, temperature, v_alpha)
    alpha_momentum_loss_rates = alpha_momentum_frequency * alpha.m

    # Compare velocities of particles
    v_electron = np.sqrt(2 * PhysicalConstants.boltzmann_constant * temperature / electron.m)
    print("For stationary collisions to be a reasonable approximation, the beams need "
          "to be travelling faster than the electron thermal velocity... ")
    print("Electron Normalised Velocity: {}".format(v_electron / 3e8))
    print("Alpha-Electron Velocity Ratio: {}".format(v_alpha / v_electron))
    print("Deuterium-Electron Velocity Ratio: {}".format(v_reactant / v_electron))

    print(deuterium_energy_loss_rates * e_alpha / alpha_energy_loss_rates / e_reactant)

    deuterium_energy_distance_travelled = e_reactant / deuterium_energy_loss_rates
    alpha_energy_distance_travelled = e_alpha / alpha_energy_loss_rates
    if plot_results:
        fig, ax = plt.subplots(2, 2, figsize=(7, 7))
        if plot_energy_frequencies:
            # energy results
            ax[0, 0].loglog(number_density, deuterium_kinetic_frequency, label="Reactant Beam")
            ax[0, 0].loglog(number_density, alpha_kinetic_frequency, label="Product Beam")
            ax[0, 0].set_xlabel("Number density (m-3)")
            ax[0, 0].set_ylabel("Energy Collision Frequency (s-1)")
            ax[0, 0].set_title("Energy Collision Frequencies - {}".format(reactant_name))
            ax[0, 0].legend()

            ax[1, 0].loglog(number_density, deuterium_energy_loss_rates, label="Reactant Beam")
            ax[1, 0].loglog(number_density, alpha_energy_loss_rates, label="Product Beam")
            ax[1, 0].set_xlabel("Number density (m-3)")
            ax[1, 0].set_ylabel("Energy Loss Rate per metre (Jm-1)")
            ax[1, 0].set_title("Energy Loss Rates per metre - {}".format(reactant_name))
            ax[1, 0].legend()

            ax[0, 1].loglog(number_density, deuterium_energy_loss_rates / e_reactant, label="Reactant Beam")
            ax[0, 1].loglog(number_density, alpha_energy_loss_rates / e_alpha, label="Product Beam")
            ax[0, 1].axhline(0.5, linestyle="--")
            ax[0, 1].set_xlabel("Number density (m-3)")
            ax[0, 1].set_ylabel("Normalised Kinetic Loss Rate (m-1)")
            ax[0, 1].set_title("Normalised Kinetic Loss Rate - {}".format(reactant_name))
            ax[0, 1].legend()

            ax[1, 1].loglog(number_density, deuterium_energy_distance_travelled, label="Reactant Beam")
            ax[1, 1].loglog(number_density, alpha_energy_distance_travelled, label="Product Beam")
            ax[1, 1].set_xlabel("Number density (m-3)")
            ax[1, 1].set_ylabel("Distance traveller (m)")
            ax[1, 1].set_title("Distance travelled - {}".format(reactant_name))
            ax[1, 1].legend()
        else:
            # momentum results
            ax[0, 0].loglog(number_density, deuterium_momentum_frequency, label="Reactant Beam")
            ax[0, 0].loglog(number_density, alpha_momentum_frequency, label="Product Beam")
            ax[0, 0].set_xlabel("Number density (m-3)")
            ax[0, 0].set_ylabel("Momentum Collision Frequency (s-1)")
            ax[0, 0].set_title("Momentum Collision Frequencies - {}".format(reactant_name))
            ax[0, 0].legend()

            ax[1, 0].loglog(number_density, deuterium_momentum_loss_rates, label="Reactant Beam")
            ax[1, 0].loglog(number_density, alpha_momentum_loss_rates, label="Product Beam")
            ax[1, 0].set_xlabel("Number density (m-3)")
            ax[1, 0].set_ylabel("Momentum Loss Rate per metre (Jm-1)")
            ax[1, 0].set_title("Momentum Loss Rates per metre - {}".format(reactant_name))
            ax[1, 0].legend()

            deuterium_momentum = reactant.m * v_reactant
            alpha_momentum = alpha.m * v_alpha
            ax[0, 1].loglog(number_density, deuterium_momentum_loss_rates / deuterium_momentum, label="Reactant Beam")
            ax[0, 1].loglog(number_density, alpha_momentum_loss_rates / alpha_momentum, label="Product Beam")
            ax[0, 1].axhline(0.1, linestyle="--")
            ax[0, 1].set_xlabel("Number density (m-3)")
            ax[0, 1].set_ylabel("Normalised Momentum Loss Rate (m-1)")
            ax[0, 1].set_title("Normalised Momentum Loss Rate - {}".format(reactant_name))
            ax[0, 1].legend()

            ax[1, 1].loglog(number_density, deuterium_momentum / deuterium_momentum_loss_rates, label="Reactant Beam")
            ax[1, 1].loglog(number_density, alpha_momentum / alpha_momentum_loss_rates, label="Product Beam")
            ax[1, 1].set_xlabel("Number density (m-3)")
            ax[1, 1].set_ylabel("Distance traveller (m)")
            ax[1, 1].set_title("Distance travelled - {}".format(reactant_name))
            ax[1, 1].legend()

        # fig.suptitle("Comparison of product and reactant beam loss rates due to electron collisions")
        fig.tight_layout()
        plt.savefig("relaxation_rates_{}".format(reactant_name))
        plt.show()

    return deuterium_energy_distance_travelled, alpha_energy_distance_travelled

if __name__ == '__main__':
    reactant_name = "DT"
    number_density = np.logspace(20, 30, 25)
    compare_product_and_reactant_energy_loss_rates(number_density, reactant_name)

