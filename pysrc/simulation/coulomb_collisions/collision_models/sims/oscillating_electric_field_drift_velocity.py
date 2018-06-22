"""
Author: Rohan Ramasamy
Date: 22/06/2018

This file contains simulations of electrons and ions in an oscillating electrostatic field. The original test problems is outlined
in:

Theory of cumulative small angle collisions - Nanbu
"""

import numpy as np
from matplotlib import pyplot as plt

from plasma_physics.pysrc.simulation.coulomb_collisions.collision_models.abe_collison_model import AbeCoulombCollisionModel
from plasma_physics.pysrc.simulation.coulomb_collisions.collision_models.nanbu_collision_model import NanbuCollisionModel
from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import ChargedParticle
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants
from plasma_physics.pysrc.utils.unit_conversions import UnitConversions


def run_sim():
    p_1 = ChargedParticle(PhysicalConstants.electron_mass, -PhysicalConstants.electron_charge)
    p_2 = ChargedParticle(39.948 * UnitConversions.amu_to_kg, PhysicalConstants.electron_charge)
    n = int(1e4)

    sim = NanbuCollisionModel(np.asarray([n, n]), np.asarray([p_1, p_2]), np.asarray([1, 1]), 
                              coulomb_logarithm=15.9, frozen_species=np.asarray([False, True]))

    # Set initial velocity conditions of background
    velocities = np.zeros((2 * n, 3))
    k_T = 1e3
    sigma = np.sqrt(2 * k_T * PhysicalConstants.electron_charge / p_1.m)
    electron_velocities = np.random.normal(loc=0.0, scale=sigma, size=velocities[:n, :].shape) / np.sqrt(3)
    velocities[:n, :] = electron_velocities
    sigma = np.sqrt(2 * k_T * PhysicalConstants.electron_charge / p_2.m)
    ion_velocities = np.random.normal(loc=0.0, scale=sigma, size=velocities[:n, :].shape) / np.sqrt(3)
    velocities[n:, :] = ion_velocities

    V = np.sqrt(8.0 * k_T / (np.pi * p_1.m))
    f = 100e3
    t_p = 1 / f
    omega = 2 * np.pi * t_p
    dt = 2.5e-7
    J = t_p / dt
    t = 0.0
    n = 1
    times = [t]
    v_squared_results = []
    I = np.zeros((n, 3))
    I[:, :] = np.asarray([1.0, 0.0, 0.0])
    new_vel = np.copy(velocities)
    while t < 50.0 * t_p:
        I_results = []
        while t < n * t_p:
            print(t / t_p)
            # Accelerate particles due to field
            new_vel[:n, :] -= I * V * np.sin(omega * t)

            # Carry out coulomb collisions
            new_vel = sim.single_time_step(new_vel, dt)
            
            # Calculate I
            current = np.mean(np.sqrt(new_vel[:n, 0] ** 2 + new_vel[:n, 1] ** 2 + new_vel[:n, 2] ** 2)) ** 2
            I_results.append(current)

            # Correct velocities
            new_vel[:n, :] -= I * V * np.sin(omega * t)

            t += dt
        
        v_squared_results.append(np.sum(np.asarray(I_results)) / t_p)

    plt.figure()
    plt.plot(v_squared_results / V ** 2)
    plt.show()

if __name__ == '__main__':
    run_sim()