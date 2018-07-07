"""
Author: Rohan Ramasamy
Date: 20/06/2018

This script contains code to generate simulation results from comparison against theoretical approximations
"""


import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import sys
import os

from pysrc.theory.coulomb_collisions.coulomb_collision import CoulombCollision, ChargedParticle
from pysrc.theory.coulomb_collisions.relaxation_processes import RelaxationProcess, MaxwellianRelaxationProcess
from pysrc.utils.unit_conversions import UnitConversions
from pysrc.utils.physical_constants import PhysicalConstants
from pysrc.simulation.coulomb_collisions.collision_models.nanbu_collision_model import NanbuCollisionModel


def generate_sim_results(number_densities, T, plot_individual_sims=False):
    # Set simulation independent parameters
    N = int(1e3)
    dt_factor = 0.01
    w_1 = int(1)
    
    # Generate result lists
    energy_results = dict()
    velocity_results = dict()
    t_halves = dict()
    t_theory = dict()
    
    # Make results directory if it does not exist
    res_dir = "results"
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
        
    # Run simulations
    names = ["reactant", "product"]
    for j, name in enumerate(names):
        t_halves[name] = np.zeros(number_densities.shape)
        t_theory[name] = np.zeros(number_densities.shape)
        energy_results[name] = []
        velocity_results[name] = []
        for i, n in enumerate(number_densities):
            # Set up beam sim
            if "product" == name:
                p_1 = ChargedParticle(6.64424e-27, 2 * PhysicalConstants.electron_charge)
                energy = 3.5e6 * PhysicalConstants.electron_charge
            elif "reactant" == name:
                p_1 = ChargedParticle(2.014102 * UnitConversions.amu_to_kg, PhysicalConstants.electron_charge)
                energy = 50e3 * PhysicalConstants.electron_charge
            else:
                raise ValueError()
            beam_velocity = np.sqrt(2 * energy / p_1.m)
            deuterium = ChargedParticle(2.014102 * UnitConversions.amu_to_kg, PhysicalConstants.electron_charge)
            electron = ChargedParticle(9.014e-31, -PhysicalConstants.electron_charge)

            # Instantiate simulation
            w_b = int(n / N)
            particle_numbers = np.asarray([N, N, N])
            sim = NanbuCollisionModel(particle_numbers, np.asarray([p_1, deuterium, electron]), np.asarray([w_1, w_b, w_b]), 
                                      coulomb_logarithm=10.0, frozen_species=np.asarray([False, False, False]))

            # Set up velocities
            velocities = np.zeros((np.sum(particle_numbers), 3))
            velocities[:N, :] = np.asarray([0.0, 0.0, beam_velocity])

            # Small maxwellian distribution used for background species
            k_T = T * PhysicalConstants.boltzmann_constant
            sigma = np.sqrt(2 * k_T / deuterium.m)
            deuterium_velocities = np.random.normal(loc=0.0, scale=sigma, size=velocities[N:2*N, :].shape) / np.sqrt(3)
            velocities[N:2*N, :] = deuterium_velocities
            sigma = np.sqrt(2 * k_T / electron.m)
            electron_velocities = np.random.normal(loc=0.0, scale=sigma, size=velocities[2*N:, :].shape) / np.sqrt(3)
            velocities[2*N:, :] = electron_velocities

            # Get approximate time scale
            impact_parameter_ratio = 1.0    # Is not necessary for this analysis
            tau = sys.float_info.max
            for background_particle in [deuterium]:
                reactant_collision = CoulombCollision(p_1, background_particle,
                                                      impact_parameter_ratio,
                                                      beam_velocity)
                relaxation = MaxwellianRelaxationProcess(reactant_collision)
                kinetic_frequency = relaxation.numerical_kinetic_loss_maxwellian_frequency(N * w_b, T, beam_velocity)
                tau = min(tau, 1.0 / kinetic_frequency)
            dt = dt_factor * tau
            final_time = 4.0 * tau

            t, v_results = sim.run_sim(velocities, dt, final_time)

            # Get salient results
            velocities = np.mean(np.sqrt(v_results[:N, 0, :] ** 2 + v_results[:N, 1, :] ** 2 + v_results[:N, 2, :] ** 2), axis=0)
            energies = 0.5 * p_1.m * np.mean(v_results[:N, 0, :] ** 2 + v_results[:N, 1, :] ** 2 + v_results[:N, 2, :] ** 2, axis=0)
            assert np.isclose(energies[0], w_1 * N * energy), "{} != {}".format(energies[0], w_1 * N * energy)
            velocity_results[name].append(velocities)
            energy_results[name].append(energies)

            velocities_deuterium = np.mean(np.sqrt(v_results[N:2*N, 0, :] ** 2 + v_results[N:2*N, 1, :] ** 2 + v_results[N:2*N, 2, :] ** 2), axis=0)
            energies_deuterium = 0.5 * deuterium.m * np.mean(v_results[N:2*N, 0, :] ** 2 + v_results[N:2*N, 1, :] ** 2 + v_results[N:2*N, 2, :] ** 2, axis=0)
            velocities_electron = np.mean(np.sqrt(v_results[2*N:, 0, :] ** 2 + v_results[2*N:, 1, :] ** 2 + v_results[2*N:, 2, :] ** 2), axis=0)
            energies_electron = 0.5 * electron.m * np.mean(v_results[2*N:, 0, :] ** 2 + v_results[2*N:, 1, :] ** 2 + v_results[2*N:, 2, :] ** 2, axis=0) 

            # Plot results
            fig, ax = plt.subplots(2)

            ax[0].plot(t, velocities_deuterium, label="ions")
            ax[0].plot(t, velocities_electron, label="electron")
            ax[0].plot(t, velocities, label="beam")
            ax[1].plot(t, energies_deuterium, label="ions")
            ax[1].plot(t, energies_electron, label="electron")
            ax[1].plot(t, energies, label="beam")
            ax[1].plot(t, energies + energies_electron + energies_deuterium, label="total")

            plt.legend()
            plt.savefig(os.path.join(res_dir, "stationary_{}_{}_{}_{}.png".format(name, N, dt_factor, n)))
            if plot_individual_sims:
                plt.show()
            plt.close()

            # Get energy half times
            energy_time_interpolator = interp1d(energies / energies[0], t)
            t_half = energy_time_interpolator(0.5)
            t_halves[name][i] = t_half
            t_theory[name][i] = tau

        # Save results
        np.savetxt(os.path.join(res_dir, "stationary_{}_{}_{}_half_times".format(name, N, dt_factor)), t_halves[name], fmt='%s')
        np.savetxt(os.path.join(res_dir, "stationary_{}_{}_{}_velocities".format(name, N, dt_factor)), velocity_results[name], fmt='%s')
        np.savetxt(os.path.join(res_dir, "stationary_{}_{}_{}_energies".format(name, N, dt_factor)), energy_results[name], fmt='%s')

    # Plot results
    plt.figure()

    for name in names:
        plt.loglog(number_densities, t_halves[name], label="{}_Simulation".format(name))
        plt.loglog(number_densities, t_theory[name], label="{}_Theoretical values".format(name))
    
    plt.title("Comparison of thermalisation rates of products and reactants")
    plt.legend()
    plt.savefig(os.path.join(res_dir, "energy_half_times"))
    plt.show()


if __name__ == '__main__':
    number_densities = np.logspace(20, 25, 3)
    print("Simulated number densities: {}".format(number_densities))
    temperature = 1000.0
    generate_sim_results(number_densities, temperature, plot_individual_sims=False)

