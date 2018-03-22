"""
Author: Rohan Ramasamy
Date: 20/03/2018

This file contains a simulations of electron thermal relaxation. The original test problem is outlined
in:

Theory of cumulative small angle collisions - Nanbu
"""

import numpy as np
from matplotlib import pyplot as plt

from plasma_physics.pysrc.simulation.coulomb_collisions.collision_models.abe_collison_model import AbeCoulombCollisionModel
from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import CoulombCollision, ChargedParticle
from plasma_physics.pysrc.theory.coulomb_collisions.relaxation_processes import RelaxationProcess
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


def get_relaxation_time(p_1, n_background, temperature):
    """
    Get the momentum loss rate for a particular particle collision,
    where one species is assumed stationary. This equation is specialised
    to the case where the beam and background species are the same

    p_1: particle forming beam and background
    n_background: number density pof background species
    temperature: temperature characterising beam
    """
    m_eff = p_1.m / 2.0
    b_90 = p_1.q ** 2 / (4 * np.pi * PhysicalConstants.epsilon_0)
    b_90 /= m_eff * np.sqrt(PhysicalConstants.boltzmann_constant * temperature / p_1.m)
    debye_length = np.sqrt(PhysicalConstants.epsilon_0 * temperature / (n_background * p_1.q ** 2))
    coulomb_logarithm = np.log(debye_length / b_90)

    tau = 8.0 * np.pi * np.sqrt(2 * p_1.m * (PhysicalConstants.boltzmann_constant * temperature) ** 3)
    tau *= PhysicalConstants.epsilon_0 ** 2
    tau /= n_background * p_1.q ** 4 * coulomb_logarithm

    return tau


def run_electron_thermal_relaxation_sim():
    """
    Run a simulation of single species electron thermal relaxation
    """
    p_1 = ChargedParticle(PhysicalConstants.electron_mass, -PhysicalConstants.electron_charge)
    n = int(1e4)

    sim = AbeCoulombCollisionModel(n, p_1, particle_weighting=1)

    T_y = 11500.0
    T_factor = 1.3
    T_e = (T_factor * T_y + 2 * T_y) / 3.0
    velocities = np.zeros((n, 3))
    sigma_ort = np.sqrt(PhysicalConstants.boltzmann_constant * T_y / p_1.m)
    sigma_x = np.sqrt(PhysicalConstants.boltzmann_constant * T_factor * T_y / p_1.m)
    velocities[:, 0] = np.random.normal(loc=0.0, scale=sigma_x, size=velocities[:, 0].shape)
    velocities[:, 1] = np.random.normal(loc=0.0, scale=sigma_ort, size=velocities[:, 1].shape)
    velocities[:, 2] = np.random.normal(loc=0.0, scale=sigma_ort, size=velocities[:, 2].shape)

    # Get simulation time
    tau = get_relaxation_time(p_1, n, T_e)
    dt = 0.01 * tau
    final_time = 12.0 * tau

    T_x = np.std(velocities[:, 0], axis=0) ** 2 * p_1.m / (PhysicalConstants.boltzmann_constant)
    T_y = np.std(velocities[:, 1], axis=0) ** 2 * p_1.m / (PhysicalConstants.boltzmann_constant)
    T_z = np.std(velocities[:, 2], axis=0) ** 2 * p_1.m / (PhysicalConstants.boltzmann_constant)

    t, v_results = sim.run_sim(velocities, dt, final_time)

    t /= tau
    dT_0 = (T_factor - 1.0) * T_y
    dT = dT_0 * np.exp(-8.0 * t / (5.0 * np.sqrt(2.0 * np.pi)))
    T_x_theory = 1.0 + dT / T_e * 2.0 / 3.0
    T_ort_theory = 1.0 - dT / T_e / 3.0

    fig, ax = plt.subplots(1, figsize=(10, 10))

    T_x = np.std(v_results[:, 0, :], axis=0) ** 2 * p_1.m / (PhysicalConstants.boltzmann_constant)
    T_y = np.std(v_results[:, 1, :], axis=0) ** 2 * p_1.m / (PhysicalConstants.boltzmann_constant)
    T_z = np.std(v_results[:, 2, :], axis=0) ** 2 * p_1.m / (PhysicalConstants.boltzmann_constant)

    ax.plot(t, T_x / T_e, label="T_x")
    ax.plot(t, T_y / T_e, label="T_y")
    ax.plot(t, T_z / T_e, label="T_z")
    ax.plot(t, T_x_theory, label="T_x_num", linestyle="--")
    ax.plot(t, T_ort_theory, label="T_ort_num", linestyle="--")
    ax.set_xlim([0.0, t[-1]])
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Temperature")
    ax.set_title("Electron thermal relaxation")

    plt.show()


if __name__ == '__main__':
    run_electron_thermal_relaxation_sim()