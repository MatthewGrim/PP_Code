"""
Author: Rohan Ramasamy
Date: 15/03/2018

This simulation models a single particle species relaxing from a uniform 3D velocity distribution
 to a maxwellian
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import CoulombCollision, ChargedParticle
from plasma_physics.pysrc.theory.coulomb_collisions.relaxation_processes import RelaxationProcess
from plasma_physics.pysrc.simulation.coulomb_collisions.collision_models.abe_collison_model import AbeCoulombCollisionModel
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


def run_sim():
    """
    Runs simulation of a single species in a uniform velocity distribution
    """
    # Set seed
    np.random.seed(1)

    # Set up simulation
    particle = ChargedParticle(2.01410178 * 1.66054e-27, PhysicalConstants.electron_charge)
    n = 10000
    weight = 1
    weighted_particle = ChargedParticle(2.01410178 * 1.66054e-27 * weight,
                                        PhysicalConstants.electron_charge * weight)
    sim = AbeCoulombCollisionModel(n, particle, weight)

    # Get initial uniform velocities
    velocities = np.random.uniform(-1.0, 1.0, size=(n, 3))

    # Get time step from collisional frequencies
    c = CoulombCollision(weighted_particle, weighted_particle, 1.0, 2.0)
    process = RelaxationProcess(c)
    v_K = process.kinetic_loss_stationary_frequency(n * weight, 1.0, 2.0)
    v_P = process.momentum_loss_stationary_frequency(n * weight, 1.0, 2.0)
    collision_time = 1.0 / max([v_K, v_P])
    dt = 0.1 * collision_time
    final_time = 2.5 * collision_time

    t, v_results = sim.run_sim(velocities, dt, final_time)

    # Get average momentum in each direction - this should be where
    # the normal distribution is centred
    v_x_ave_init = np.average(velocities[:, 0])
    v_y_ave_init = np.average(velocities[:, 1])
    v_z_ave_init = np.average(velocities[:, 2])
    print("Initial average x velocity: {}".format(v_x_ave_init))
    print("Initial average y velocity: {}".format(v_y_ave_init))
    print("Initial average z velocity: {}\n".format(v_z_ave_init))

    v_x_ave_fin = np.average(v_results[:, 0, -1])
    v_y_ave_fin = np.average(v_results[:, 1, -1])
    v_z_ave_fin = np.average(v_results[:, 2, -1])
    print("Final average x velocity: {}".format(v_x_ave_fin))
    print("Final average y velocity: {}".format(v_y_ave_fin))
    print("Final average z velocity: {}\n".format(v_z_ave_fin))

    print("Change in average x velocity: {}".format(v_x_ave_fin - v_x_ave_init))
    print("Change in average y velocity: {}".format(v_y_ave_fin - v_y_ave_init))
    print("Change in average z velocity: {}".format(v_z_ave_fin - v_z_ave_init))

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))

    ax[0, 0].hist(velocities[:, 0], 100)
    ax[0, 1].hist(velocities[:, 1], 100)
    ax[0, 2].hist(velocities[:, 2], 100)

    ax[1, 0].hist(v_results[:, 0, -1], 100)
    ax[1, 1].hist(v_results[:, 1, -1], 100)
    ax[1, 2].hist(v_results[:, 2, -1], 100)

    ax[2, 0].hist(v_results[:, 0, -1] - velocities[:, 0], 1000)
    ax[2, 1].hist(v_results[:, 1, -1] - velocities[:, 1], 1000)
    ax[2, 2].hist(v_results[:, 2, -1] - velocities[:, 2], 1000)

    plt.show()

    energy = np.sum(v_results[:, 0, :] ** 2 + v_results[:, 1, :] ** 2 + v_results[:, 2, :] ** 2, axis=0)
    x_mom = np.sum(v_results[:, 0, :], axis=0)
    y_mom = np.sum(v_results[:, 1, :], axis=0)
    z_mom = np.sum(v_results[:, 2, :], axis=0)

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

if __name__ == '__main__':
    run_sim()
