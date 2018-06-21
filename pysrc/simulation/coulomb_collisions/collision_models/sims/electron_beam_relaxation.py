"""
Author: Rohan Ramasamy
Date: 19/03/2018

This file contains simulations of different electron beam relaxation processes in electron and ion
background gases. The original test problems is outlined
in:

Theory of cumulative small angle collisions - Nanbu
"""

import numpy as np
from matplotlib import pyplot as plt

from plasma_physics.pysrc.simulation.coulomb_collisions.collision_models.abe_collison_model import AbeCoulombCollisionModel
from plasma_physics.pysrc.simulation.coulomb_collisions.collision_models.nanbu_collision_model import NanbuCollisionModel
from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import ChargedParticle
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


def get_relaxation_time(p_1, n_background, velocity):
    """
    Get the momentum loss rate for a particular particle collision,
    where one species is assumed stationary. This equation is specialised
    to the case where the beam and background species are the same

    p_1: particle forming beam and background
    n_background: number density pof background species
    temperature: temperature characterising beam
    """
    coulomb_logarithm = 10.0
    particle_energy = 0.5 * p_1.m * velocity ** 2

    tau = 8.0 * np.pi * np.sqrt(2 * p_1.m * particle_energy ** 3)
    tau *= PhysicalConstants.epsilon_0 ** 2
    tau /= n_background * p_1.q ** 4 * coulomb_logarithm

    return tau


def run_electron_beam_into_stationary_target_sim(sim_type):
    """
    Run simulation of velocity relaxation of electron beam to stationary
    background species
    """
    p_1 = ChargedParticle(PhysicalConstants.electron_mass, -PhysicalConstants.electron_charge)
    p_2 = ChargedParticle(1e31, -PhysicalConstants.electron_charge)
    w_1 = 1
    w_2 = 1
    n = int(1e3)
    beam_velocity = 1.0

    if sim_type == "Abe":
        sim = AbeCoulombCollisionModel(n, p_1, w_1=w_1, N_2=n, particle_2=p_2, w_2=w_2, freeze_species_2=True)
    elif sim_type == "Nanbu":
        sim = NanbuCollisionModel(np.asarray([n, n]), np.asarray([p_1, p_2]), np.asarray([1, 1]), coulomb_logarithm=10.0, freeze_species_2=True)
    else:
        raise ValueError("Invalid sim type")

    velocities = np.zeros((2 * n, 3))
    velocities[:n, :] = np.asarray([0.0, 0.0, beam_velocity])
    velocities[n:] = np.asarray([0.0, 0.0, 0.0])

    tau = get_relaxation_time(p_1, n * w_2, beam_velocity)
    dt = 0.05 * tau
    final_time = 3.0 * tau

    t, v_results = sim.run_sim(velocities, dt, final_time)

    t /= tau
    v_z_estimate = 1.0 - t
    v_ort_estimate = 2 * t

    fig, ax = plt.subplots(2, figsize=(10, 10))

    ax[0].plot(t, np.mean(v_results[:n, 0, :] ** 2 + v_results[:n, 1, :] ** 2, axis=0) / beam_velocity ** 2, label="<v_ort^2>")
    ax[0].plot(t, np.mean(v_results[:n, 2, :], axis=0), label="<v_z>")
    ax[0].plot(t, v_z_estimate, linestyle="--", label="<v_z_estimate>")
    ax[0].plot(t, v_ort_estimate, linestyle="--", label="<v_ort_estimate>")
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

    return t, v_results


def run_electon_beam_into_electron_gas_sim(sim_type):
    """
    Run a simulation of an electron beam into a background gas at a
    given temperature
    """
    p_1 = ChargedParticle(PhysicalConstants.electron_mass, -PhysicalConstants.electron_charge)
    p_2 = ChargedParticle(PhysicalConstants.electron_mass, -PhysicalConstants.electron_charge)
    w_1 = 1
    w_2 = 1
    n = int(1e3)

    if sim_type == "Abe":
        sim = AbeCoulombCollisionModel(n, p_1, w_1=w_1, N_2=n, particle_2=p_2, w_2=w_2, freeze_species_2=True)
    elif sim_type == "Nanbu":
        sim = NanbuCollisionModel(np.asarray([n, n]), np.asarray([p_1, p_2]), np.asarray([1, 1]), coulomb_logarithm=10.0, freeze_species_2=True)
    else:
        raise ValueError("Invalid sim type")

    # Set initial velocity conditions of beam
    beam_velocity = np.sqrt(2 * 100 * PhysicalConstants.electron_charge / p_1.m)
    velocities = np.zeros((2 * n, 3))
    velocities[:n, :] = np.asarray([0.0, 0.0, beam_velocity])

    # Set initial velocity conditions of background
    k_T = 2.0
    sigma = np.sqrt(2 * k_T * PhysicalConstants.electron_charge / p_1.m)
    # Not sure if this is correct - taking the 3D velocity magnitude, and dividing by sqrt(3)
    maxwell_velocities = np.random.normal(loc=0.0, scale=sigma, size=velocities[n:, :].shape) / np.sqrt(3)
    velocities[n:] = maxwell_velocities

    tau = get_relaxation_time(p_1, n * w_2, beam_velocity)
    dt = 0.01 * tau
    final_time = 1.0 * tau

    t, v_results = sim.run_sim(velocities, dt, final_time)

    t /= tau
    v_z_estimate = 1.0 - 2 * t
    v_ort_estimate = 1.98 * t

    fig, ax = plt.subplots(2, figsize=(10, 10))

    l_v_z = ax[0].plot(t, np.mean(v_results[:n, 2, :], axis=0) / beam_velocity, color="b", label="<v_z>")
    l_v_z_estimate = ax[0].plot(t, v_z_estimate, linestyle="--", color="c", label="<v_z_estimate>")
    ax[0].set_xlim([0.0, t[-1]])
    ax[0].set_xlabel("Timestep")
    ax[0].set_ylabel("Velocities ms-1")
    ax[0].set_ylim([0.0, 1.0])
    ax[0].set_title("Beam Velocities")

    # Twin axes for second plot to replicate paper set up
    ax_twin = ax[0].twinx()
    l_ort = ax_twin.plot(t, np.mean(v_results[:n, 0, :] ** 2 + v_results[:n, 1, :] ** 2, axis=0) / beam_velocity ** 2, color="r", label="<v_ort^2>")
    l_ort_estimate = ax_twin.plot(t, v_ort_estimate, linestyle="--", color="y", label="<v_ort_estimate>")
    ax_twin.set_ylabel("Square of Velocities (ms-1)^2")
    ax_twin.set_ylim([0.0, 0.22])
    
    lines = l_v_z + l_v_z_estimate + l_ort + l_ort_estimate
    labs = [l.get_label() for l in lines]
    ax[0].legend(lines, labs)

    ax[1].plot(t, np.mean(v_results[n:, 0, :], axis=0), label="<v_x>")
    ax[1].plot(t, np.mean(v_results[n:, 1, :], axis=0), label="<v_y>")
    ax[1].plot(t, np.mean(v_results[n:, 2, :], axis=0), label="<v_z>")
    ax[1].set_xlim([0.0, t[-1]])
    ax[1].legend()
    ax[1].set_xlabel("Timestep")
    ax[1].set_ylabel("Velocities ms-1")
    ax[1].set_title("Background Velocities")

    plt.show()

    energy = p_1.m * np.sum(v_results[:n, 0, :] ** 2 + v_results[:n, 1, :] ** 2 + v_results[:n, 2, :] ** 2, axis=0)
    energy += p_2.m * np.sum(v_results[n:, 0, :] ** 2 + v_results[n:, 1, :] ** 2 + v_results[n:, 2, :] ** 2, axis=0)
    x_mom = p_1.m * np.sum(v_results[:n, 0, :], axis=0) + p_2.m * np.sum(v_results[n:, 0, :], axis=0)
    y_mom = p_1.m * np.sum(v_results[:n, 1, :], axis=0) + p_1.m * np.sum(v_results[n:, 1, :], axis=0)
    z_mom = p_1.m * np.sum(v_results[:n, 2, :], axis=0) + p_1.m * np.sum(v_results[n:, 2, :], axis=0)

    fig, ax = plt.subplots(2)

    ax[0].plot(t, energy)
    ax[0].set_title("Conservation of Energy")
    ax[1].plot(t, x_mom, label="x momentum")
    ax[1].plot(t, y_mom, label="y momentum")
    ax[1].plot(t, z_mom, label="z momentum")
    ax[1].legend()
    ax[1].set_title("Conservation of Momentum")

    fig.suptitle("Conservation Plots")
    plt.show()

    return t, v_results


if __name__ == '__main__':
    # sim_type = "Abe"
    sim_type = "Nanbu"

    _, _ = run_electron_beam_into_stationary_target_sim(sim_type)
    _, _ = run_electon_beam_into_electron_gas_sim(sim_type)
