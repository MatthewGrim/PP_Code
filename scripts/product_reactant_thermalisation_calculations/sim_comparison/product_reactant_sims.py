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


def get_relaxation_time(p_1, n_background, velocity):
    """
    Get the momentum loss rate for a particular particle collision,
    where one species is assumed stationary. This equation is specialised
    to the case where the beam and background species are the same

    p_1: particle forming beam and background
    n_background: number density pof background species
    temperature: temperature characterising beam
    """

    # Approximate time scale of reaction
    coulomb_logarithm = 10.0
    particle_energy = 0.5 * p_1.m * velocity ** 2

    tau = 8.0 * np.pi * np.sqrt(2 * p_1.m * particle_energy ** 3)
    tau *= PhysicalConstants.epsilon_0 ** 2
    tau /= n_background * p_1.q ** 4 * coulomb_logarithm

    return tau

def generate_sim_results(n, T):
    # # Generate charged particles
    # deuterium = ChargedParticle(PhysicalConstants.electron_mass, -PhysicalConstants.electron_charge)
    # electron = ChargedParticle(1e31, -PhysicalConstants.electron_charge)

    # # Get deuterium velocity
    # v_deuterium = 1.0

    # N = int(1e3)
    # n = N

    # # Generate simulation
    # w_2 = 1
    # sim = AbeCoulombCollisionModel(N, electron, w_1=1, N_2=N, particle_2=electron, w_2=w_2, freeze_species_2=True)

    # # Generate velocities
    # velocities = np.zeros((2 * N, 3))
    # velocities[:N, :] = np.asarray([0.0, 0.0, v_deuterium])
    # velocities[N:] = np.asarray([0.0, 0.0, 0.0])

    # # Approximate time scale of reaction
    # impact_parameter_ratio = 1.0    # Is not necessary for this analysis
    # reactant_collision = CoulombCollision(electron, electron,
    #                                       impact_parameter_ratio,
    #                                       v_deuterium)
    # reactant_relaxation = RelaxationProcess(reactant_collision)
    # deuterium_kinetic_frequency = reactant_relaxation.kinetic_loss_stationary_frequency(n, T, v_deuterium)
    # tau = 1.0 / deuterium_kinetic_frequency
    # dt = 0.01 * tau
    # final_time = 2.0 * tau

    # # Run sim
    # t, v_results = sim.run_sim(velocities, dt, final_time)

    # # Get and plot energies
    # v_mag = np.mean(np.sqrt(v_results[:N, 0, :] ** 2 + v_results[:N, 1, :] ** 2 + v_results[:N, 2, :] ** 2), axis=0)
    # energies = 0.5 * deuterium.m * v_mag 
    # fig, ax = plt.subplots(2)
    # ax[0].plot(t, energies)
    # ax[1].plot(t, np.mean(v_results[:N, 0, :], axis=0))
    # ax[1].plot(t, np.mean(v_results[:N, 1, :], axis=0))
    # ax[1].plot(t, np.mean(v_results[:N, 2, :], axis=0))
    # ax[1].plot(t, v_mag)
    # plt.show()

    # Electron beam sim
    p_1 = ChargedParticle(2.014102 * UnitConversions.amu_to_kg, PhysicalConstants.electron_charge)
    p_2 = ChargedParticle(2.014102 * UnitConversions.amu_to_kg, PhysicalConstants.electron_charge)
    w_1 = 1
    w_2 = 10
    n = int(1e3)
    beam_velocity = 1.0

    sim = AbeCoulombCollisionModel(n, p_1, w_1=w_1, N_2=n, particle_2=p_2, w_2=w_2, freeze_species_2=False)
    
    velocities = np.zeros((2 * n, 3))
    velocities[:n, :] = np.asarray([0.0, 0.0, beam_velocity])
    velocities[n:] = np.asarray([0.0, 0.0, 0.0])

    tau = get_relaxation_time(p_1, n * w_2, beam_velocity)
    impact_parameter_ratio = 1.0    # Is not necessary for this analysis
    reactant_collision = CoulombCollision(p_1, p_2,
                                          impact_parameter_ratio,
                                          beam_velocity)
    reactant_relaxation = RelaxationProcess(reactant_collision)
    deuterium_kinetic_frequency = reactant_relaxation.kinetic_loss_stationary_frequency(n * w_2, T, beam_velocity)
    tau = 1.0 / deuterium_kinetic_frequency
    dt = 0.01 * tau
    final_time = 4.0 * tau

    t, v_results = sim.run_sim(velocities, dt, final_time)

    t /= tau

    fig, ax = plt.subplots(2, figsize=(10, 10))

    ax[0].plot(t, np.mean(v_results[:n, 0, :] ** 2 + v_results[:n, 1, :] ** 2, axis=0) / beam_velocity ** 2, label="<v_ort^2>")
    ax[0].plot(t, np.mean(v_results[:n, 2, :], axis=0), label="<v_z_beam>")
    ax[0].plot(t, np.mean(v_results[n:, 2, :], axis=0), label="<v_z_background>")
    ax[0].set_ylim([0.0, 1.0])
    ax[0].set_xlim([0.0, t[-1]])
    ax[0].legend()
    ax[0].set_xlabel("Timestep")
    ax[0].set_ylabel("Velocities ms-1")
    ax[0].set_title("Beam Velocities")

    ax[1].plot(t, np.mean(v_results[n:, 0, :], axis=0), label="<v_x>")
    ax[1].plot(t, np.mean(v_results[n:, 1, :], axis=0), label="<v_y>")
    ax[1].plot(t, np.mean(v_results[n:, 2, :], axis=0), label="<v_z>")
    ax[1].legend()
    ax[1].set_xlabel("Timestep")
    ax[1].set_ylabel("Velocities ms-1")
    ax[1].set_title("Background Velocities")

    plt.show()

    print(np.mean(v_results[:n, 2, :], axis=0)[-1])


if __name__ == '__main__':
    number_densities = int(1e4)
    temperature = 10000.0
    generate_sim_results(number_densities, temperature)

