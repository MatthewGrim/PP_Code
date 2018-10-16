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

from plasma_physics.pysrc.theory.reactivities.fusion_reactivities import BoschHaleReactivityFit, FusionReaction, DDReaction, DTReaction, pBReaction
from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import CoulombCollision, ChargedParticle
from plasma_physics.pysrc.theory.coulomb_collisions.relaxation_processes import RelaxationProcess, MaxwellianRelaxationProcess
from plasma_physics.pysrc.utils.unit_conversions import UnitConversions
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants
from plasma_physics.pysrc.simulation.coulomb_collisions.collision_models.nanbu_collision_model import NanbuCollisionModel


def generate_sim_results(number_densities, reactant_name, plot_individual_sims=False):
    # Set simulation independent parameters
    N = int(1e4)
    dt_factor = 0.01
    
    # Generate result lists
    energy_results = dict()
    velocity_results = dict()
    t_halves = dict()
    t_theory = dict()
    
    # Make results directory if it does not exist
    res_dir = os.path.join("results", reactant_name)
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
            # Set temperature of background - this is assumed to be lower than the energy of the beam
            T_b = 1e3 if reactant_name is "DT" else 10e3
            T_b *= UnitConversions.eV_to_K

            # Set up beam sim
            if "product" == name:
                p_1 = ChargedParticle(6.64424e-27, 2 * PhysicalConstants.electron_charge)
                if reactant_name == "DT":
                    p_2 = ChargedParticle(5.0064125184e-27, 2 * PhysicalConstants.electron_charge)
                    energy = 3.5e6 * PhysicalConstants.electron_charge
                    Z = 2

                    # Get number density of products from reactivity
                    T = 50
                    reaction = DTReaction()
                    reactivity_fit = BoschHaleReactivityFit(reaction)
                    reactivity = reactivity_fit.get_reactivity(T)
                    n_1 = n * reactivity
                elif reactant_name == "pB":
                    p_2 = ChargedParticle((10.81 + 1) * UnitConversions.amu_to_kg, PhysicalConstants.electron_charge * 12)
                    energy = 8.6e6 / 3.0 * PhysicalConstants.electron_charge
                    Z = 6

                    # Get number density of products from reactivity
                    T = 500
                    reaction = pBReaction()
                    reactivity_fit = BoschHaleReactivityFit(reaction)
                    reactivity = reactivity_fit.get_reactivity(T)
                    n_1 = n * reactivity
                else:
                    raise ValueError("Invalid reactant")
            elif "reactant" == name:
                n_1 = n
            	if reactant_name == "DT":
                	p_1 = ChargedParticle(5.0064125184e-27, 2 * PhysicalConstants.electron_charge)
            		p_2 = ChargedParticle(5.0064125184e-27, 2 * PhysicalConstants.electron_charge)
	                energy = 50e3 * PhysicalConstants.electron_charge
            		Z = 2
                elif reactant_name == "pB":
	                p_1 = ChargedParticle((10.81 + 1) * UnitConversions.amu_to_kg, PhysicalConstants.electron_charge * 12)
	                p_2 = ChargedParticle((10.81 + 1) * UnitConversions.amu_to_kg, PhysicalConstants.electron_charge * 12)
	                energy = 500e3 * PhysicalConstants.electron_charge
	                Z = 6
                else:
                	raise ValueError("Invalid reactant")
            else:
                raise ValueError()
            beam_velocity = np.sqrt(2 * energy / p_1.m)
            electron = ChargedParticle(9.014e-31, -PhysicalConstants.electron_charge)

            # Instantiate simulation
            w_b = long(n / N)
            w_1 = long(n_1 / N)
            print(w_1, w_b)
            particle_numbers = np.asarray([N, N, N])
            sim = NanbuCollisionModel(particle_numbers, np.asarray([p_1, p_2, electron]), np.asarray([w_1, w_b, Z * w_b]),
                                      coulomb_logarithm=10.0, frozen_species=np.asarray([False, False, False]), include_self_collisions=True)

            # Set up velocities
            velocities = np.zeros((np.sum(particle_numbers), 3))
            velocities[:N, :] = np.asarray([0.0, 0.0, beam_velocity])

            # Small maxwellian distribution used for background species
            k_T = T_b * PhysicalConstants.boltzmann_constant
            sigma = np.sqrt(2 * k_T / p_2.m)
            p_2_velocities = np.random.normal(loc=0.0, scale=sigma, size=velocities[N:2*N, :].shape) / np.sqrt(3)
            velocities[N:2*N, :] = p_2_velocities
            sigma = np.sqrt(2 * k_T / electron.m)
            electron_velocities = np.random.normal(loc=0.0, scale=sigma, size=velocities[2*N:, :].shape) / np.sqrt(3)
            velocities[2*N:, :] = electron_velocities

            # Get approximate time scale
            impact_parameter_ratio = 1.0    # Is not necessary for this analysis
            tau = sys.float_info.max
            for background_particle in [p_2]:
                reactant_collision = CoulombCollision(p_1, background_particle,
                                                      impact_parameter_ratio,
                                                      beam_velocity)
                relaxation = MaxwellianRelaxationProcess(reactant_collision)
                kinetic_frequency = relaxation.numerical_kinetic_loss_maxwellian_frequency(N * w_b, T_b, beam_velocity)
                tau = min(tau, 1.0 / kinetic_frequency)
            dt = dt_factor * tau
            final_time = 4.0 * tau

            t, v_results = sim.run_sim(velocities, dt, final_time)

            # Get salient results
            velocities = np.mean(np.sqrt(v_results[:N, 0, :] ** 2 + v_results[:N, 1, :] ** 2 + v_results[:N, 2, :] ** 2), axis=0)
            energies = 0.5 * p_1.m * np.mean(v_results[:N, 0, :] ** 2 + v_results[:N, 1, :] ** 2 + v_results[:N, 2, :] ** 2, axis=0)
            total_energies = 0.5 * p_1.m * np.sum(v_results[:N, 0, :] ** 2 + v_results[:N, 1, :] ** 2 + v_results[:N, 2, :] ** 2, axis=0)
            total_energy = N * w_1 * energy
            assert np.isclose(energies[0], energy, atol=0.01*energy), "{} != {}".format(energies[0], w_1 * N * energy)
            assert np.isclose(total_energies[0] * w_1, total_energy, atol=0.01*total_energy), "{} != {}".format(total_energies[0], total_energy)
            velocity_results[name].append(velocities)
            energy_results[name].append(energies)

            velocities_p_2 = np.mean(np.sqrt(v_results[N:2*N, 0, :] ** 2 + v_results[N:2*N, 1, :] ** 2 + v_results[N:2*N, 2, :] ** 2), axis=0)
            energies_p_2 = 0.5 * p_2.m * np.mean(v_results[N:2*N, 0, :] ** 2 + v_results[N:2*N, 1, :] ** 2 + v_results[N:2*N, 2, :] ** 2, axis=0)
            velocities_electron = np.mean(np.sqrt(v_results[2*N:, 0, :] ** 2 + v_results[2*N:, 1, :] ** 2 + v_results[2*N:, 2, :] ** 2), axis=0)
            energies_electron = 0.5 * electron.m * np.mean(v_results[2*N:, 0, :] ** 2 + v_results[2*N:, 1, :] ** 2 + v_results[2*N:, 2, :] ** 2, axis=0) 

            # Plot total velocity and energy results
            fig, ax = plt.subplots(2, sharex=True)

            ax[0].plot(t, velocities_p_2, label="ions")
            ax[0].plot(t, velocities_electron, label="electron")
            ax[0].plot(t, velocities, label="beam")
            ax[0].set_ylabel("Velocities [$ms^{-1}$]")
            ax[1].plot(t, energies_p_2, label="ions")
            ax[1].plot(t, energies_electron, label="electron")
            ax[1].plot(t, energies, label="beam")
            ax[1].plot(t, energies + energies_electron + energies_p_2, label="total")
            ax[1].set_ylabel("Energies [J]")
            ax[1].set_xlabel("Time [s]")

            plt.legend()
            plt.savefig(os.path.join(res_dir, "{}_{}_{}_{}_{}.png".format(name, N, dt_factor, n, reactant_name)))
            if plot_individual_sims:
                plt.show()
            plt.close()

            # Plot velocities
            fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex='col')

            v_x_1 = np.mean(v_results[:N, 0, :], axis=0)
            v_y_1 = np.mean(v_results[:N, 1, :], axis=0)
            v_z_1 = np.mean(v_results[:N, 2, :], axis=0)
            
            v_x_2 = np.mean(v_results[N:2*N, 0, :], axis=0)
            v_y_2 = np.mean(v_results[N:2*N, 1, :], axis=0)
            v_z_2 = np.mean(v_results[N:2*N, 2, :], axis=0)
            
            v_x_3 = np.mean(v_results[2*N:, 0, :], axis=0)
            v_y_3 = np.mean(v_results[2*N:, 1, :], axis=0)
            v_z_3 = np.mean(v_results[2*N:, 2, :], axis=0)

            ax[0, 0].plot(t, v_x_1, label="v_x_1")
            ax[0, 0].plot(t, v_x_2, label="v_x_2")
            ax[0, 0].plot(t, v_x_3, label="v_x_3")
            ax[0, 0].legend()

            ax[0, 1].plot(t, v_y_1, label="v_y_1")
            ax[0, 1].plot(t, v_y_2, label="v_y_2")
            ax[0, 1].plot(t, v_y_3, label="v_y_3")
            ax[0, 1].legend()

            ax[1, 0].plot(t, v_z_1, label="v_z_1")
            ax[1, 0].plot(t, v_z_2, label="v_z_2")
            ax[1, 0].plot(t, v_z_3, label="v_z_3")
            ax[1, 0].legend()

            ax[1, 1].plot(t, velocities, label="v_t_1")
            ax[1, 1].plot(t, velocities_p_2, label="v_t_2")
            ax[1, 1].plot(t, velocities_electron, label="v_t_3")
            ax[1, 1].legend()

            plt.savefig(os.path.join(res_dir, "species_velocities_{}_{}_{}_{}_{}.png".format(name, N, dt_factor, n, reactant_name)))
            if plot_individual_sims:
                plt.show()
            plt.close()

            # Plot temperatures
            fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex='col', sharey='row')

            velocities_total = np.sqrt(v_results[:, 0, :] ** 2 + v_results[:, 1, :] ** 2 + v_results[:, 2, :] ** 2)

            T_x_1 = np.std(v_results[:N, 0, :], axis=0) ** 2 * p_1.m / (PhysicalConstants.boltzmann_constant)
            T_y_1 = np.std(v_results[:N, 1, :], axis=0) ** 2 * p_1.m / (PhysicalConstants.boltzmann_constant)
            T_z_1 = np.std(v_results[:N, 2, :], axis=0) ** 2 * p_1.m / (PhysicalConstants.boltzmann_constant)
            T_t_1 = np.std(velocities_total[:N, :], axis=0) ** 2 * p_1.m / (PhysicalConstants.boltzmann_constant)
            
            T_x_2 = np.std(v_results[N:2*N, 0, :], axis=0) ** 2 * p_2.m / (PhysicalConstants.boltzmann_constant)
            T_y_2 = np.std(v_results[N:2*N, 1, :], axis=0) ** 2 * p_2.m / (PhysicalConstants.boltzmann_constant)
            T_z_2 = np.std(v_results[N:2*N, 2, :], axis=0) ** 2 * p_2.m / (PhysicalConstants.boltzmann_constant)
            T_t_2 = np.std(velocities_total[N:2*N, :], axis=0) ** 2 * p_2.m / (PhysicalConstants.boltzmann_constant)
            
            T_x_3 = np.std(v_results[2*N:, 0, :], axis=0) ** 2 * electron.m / (PhysicalConstants.boltzmann_constant)
            T_y_3 = np.std(v_results[2*N:, 1, :], axis=0) ** 2 * electron.m / (PhysicalConstants.boltzmann_constant)
            T_z_3 = np.std(v_results[2*N:, 2, :], axis=0) ** 2 * electron.m / (PhysicalConstants.boltzmann_constant)
            T_t_3 = np.std(velocities_total[2*N:, :], axis=0) ** 2 * electron.m / (PhysicalConstants.boltzmann_constant)

            ax[0, 0].plot(t, T_x_1, label="T_x_1")
            ax[0, 0].plot(t, T_x_2, label="T_x_2")
            ax[0, 0].plot(t, T_x_3, label="T_x_3")
            ax[0, 0].legend()

            ax[0, 1].plot(t, T_y_1, label="T_y_1")
            ax[0, 1].plot(t, T_y_2, label="T_y_2")
            ax[0, 1].plot(t, T_y_3, label="T_y_3")
            ax[0, 1].legend()

            ax[1, 0].plot(t, T_z_1, label="T_z_1")
            ax[1, 0].plot(t, T_z_2, label="T_z_2")
            ax[1, 0].plot(t, T_z_3, label="T_z_3")
            ax[1, 0].legend()

            ax[1, 1].plot(t, T_t_1, label="T_t_1")
            ax[1, 1].plot(t, T_t_2, label="T_t_2")
            ax[1, 1].plot(t, T_t_3, label="T_t_3")
            ax[1, 1].legend()

            plt.savefig(os.path.join(res_dir, "species_temperatures_{}_{}_{}_{}_{}.png".format(name, N, dt_factor, n, reactant_name)))
            if plot_individual_sims:
                plt.show()
            plt.close()

            # Get energy half times - assuming beam has equilibrated at the end of the simulation
            energies = energies - energies[-1]
            energy_time_interpolator = interp1d(energies / energies[0], t)
            t_half = energy_time_interpolator(0.5)
            t_halves[name][i] = t_half
            t_theory[name][i] = tau

    # Plot results
    plt.figure()

    for name in names:
        plt.loglog(number_densities, t_halves[name], label="{}_Simulation".format(name))
        plt.loglog(number_densities, t_theory[name], label="{}_Theoretical values".format(name))
    
    plt.title("Comparison of thermalisation rates of products and reactants")
    plt.legend()
    plt.savefig(os.path.join(res_dir, "energy_half_times_{}".format(reactant_name)))
    plt.show()


if __name__ == '__main__':
    number_densities = np.logspace(15, 25, 4)
    reactant_name = "pB"
    generate_sim_results(number_densities, reactant_name, plot_individual_sims=False)

