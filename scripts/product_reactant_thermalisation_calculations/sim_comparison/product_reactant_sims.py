"""
Author: Rohan Ramasamy
Date: 20/06/2018

This script contains code to generate simulation results from comparison against theoretical approximations
"""


import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import CoulombCollision, ChargedParticle
from plasma_physics.pysrc.theory.coulomb_collisions.relaxation_processes import RelaxationProcess
from plasma_physics.pysrc.utils.unit_conversions import UnitConversions
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants
from plasma_physics.pysrc.simulation.coulomb_collisions.collision_models.abe_collison_model import AbeCoulombCollisionModel
from plasma_physics.pysrc.simulation.coulomb_collisions.collision_models.nanbu_collision_model import NanbuCollisionModel


def generate_sim_results(number_densities, T, alpha = False):
    # Set up beam sim
    if alpha:
        name = "product"
        p_1 = ChargedParticle(6.64424e-27, 2 * PhysicalConstants.electron_charge)
        energy = 3.5e6 * PhysicalConstants.electron_charge
    else:
        neam = "reactant"
        p_1 = ChargedParticle(2.014102 * UnitConversions.amu_to_kg, PhysicalConstants.electron_charge)
        energy = 50e3 * PhysicalConstants.electron_charge
    beam_velocity = np.sqrt(2 * energy / p_1.m)
    p_2 = ChargedParticle(2.014102 * UnitConversions.amu_to_kg, PhysicalConstants.electron_charge)
    
    # Set simulation independent parameters
    N = int(1e3)
    w_1 = int(1)
    
    # Generate result lists
    energy_results = []
    velocity_list = []
    t_halves = np.zeros(number_densities.shape)
    t_theory = np.zeros(number_densities.shape)
    
    # Run simulations
    for i, n in enumerate(number_densities): 
        # Instantiate simulation
        w_2 = int(n / N)
        sim = AbeCoulombCollisionModel(N, p_1, w_1=w_1, N_2=N, particle_2=p_2, w_2=w_2, freeze_species_2=False)
        
        # Set up velocities
        velocities = np.zeros((2 * N, 3))
        velocities[:N, :] = np.asarray([0.0, 0.0, beam_velocity])
        velocities[N:] = np.asarray([0.0, 0.0, 0.0])

        # Get approximate time scale
        impact_parameter_ratio = 1.0    # Is not necessary for this analysis
        reactant_collision = CoulombCollision(p_1, p_2,
                                              impact_parameter_ratio,
                                              beam_velocity)
        reactant_relaxation = RelaxationProcess(reactant_collision)
        deuterium_kinetic_frequency = reactant_relaxation.kinetic_loss_stationary_frequency(N * w_2, T, beam_velocity)
        tau = 1.0 / deuterium_kinetic_frequency
        dt = 0.01 * tau
        final_time = 4.0 * tau

        t, v_results = sim.run_sim(velocities, dt, final_time)

        # Get salient results
        velocities = np.mean(np.sqrt(v_results[:N, 0, :] ** 2 + v_results[:N, 1, :] ** 2 + v_results[:N, 2, :] ** 2), axis=0)
        energies = 0.5 * p_1.m * velocities ** 2

        energy_time_interpolator = interp1d(energies / energy, t)
        t_half = energy_time_interpolator(0.5)
        t_halves[i] = t_half
        t_theory[i] = tau

    # Save results
    np.savetxt("{}_energies".format(name), np.asarray(energies))
    np.savetxt("{}_velocities".format(name), np.asarray(velocities))
    np.savetxt("{}_half_times".format(name), t_halves)

    # Plot results
    plt.figure()

    plt.loglog(number_densities, t_halves, label="Simulation")
    plt.loglog(number_densities, t_theory, label="Theoretical values")
    
    plt.legend()
    plt.savefig("energy_half_times")
    plt.show()


if __name__ == '__main__':
    number_densities = np.logspace(22, 26, 4)
    temperature = 10000.0
    generate_sim_results(number_densities, temperature)

