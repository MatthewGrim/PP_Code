"""
Author: Rohan Ramasamy
Date: 12/03/2018

This file contains a numerical model of binary collisions, as outlined in:

A Binary Collision Model for Plasma Simulation with a Particle Code
 - T. Takizuka and H. Abe
"""

import numpy as np
import math

from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import ChargedParticle
from plasma_physics.pysrc.simulation.pic.algo.geometry import vector_ops
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


class AbeCoulombCollisionModel(object):
    def __init__(self, n_1, n_2, particle_1, particle_weighting,
                 particle_2=None):
        """
        Used to simulate collisions between two particle species

        n_1: total number of unweighted particles of species 1
        n_2: total number of unweighted particles of species 2
        particle_1: first particle species involved in collisions
        particle_2: second particle species involved in collisions
        particle_weighting: number of particles per macro-particle
                            in simulation
        """
        assert isinstance(n_1, int)
        assert isinstance(n_2, int)
        assert isinstance(particle_weighting, int)
        assert isinstance(particle_1, ChargedParticle)
        assert particle_2 is None or isinstance(particle_2, ChargedParticle)

        # Define first particle species
        self.__m_1 = particle_1.p_1.m * particle_weighting
        self.__q_1 = particle_1.p_1.q * particle_weighting
        self.__n_1 = n_1

        # Define second particle species, if it exists
        if particle_2 is not None:
            self.__single_species = False
            self.__m_2 = particle_2.p_2.m * particle_weighting
            self.__q_2 = particle_2.p_2.q * particle_weighting
            self.__n_2 = n_2
            self.__m_eff = self.__m_1 * self.__m_2 / (self.__m_1 + self.__m_2)
        else:
            self.__single_species = False
            self.__m_eff = self.__m_1 ** 2 / (2 * self.__m_1)

        # Coulomb logarithm is currently fixed in method
        self.__coulomb_logarithm = 10.0

    @staticmethod
    def randomise_velocities(vel):
        """
        Randomise the addresses of particles from each species. In such a way
        pairs of particles in adjacent indices are formed for collisions

        vel: Nx3 array of velocity components for particles, the velocities
             contain the particles of each species sequentially, N = n_1 + n_2
        """
        np.random.shuffle(vel)

    @staticmethod
    def calculate_post_collision_velocities(v_1, v_2, dt):
        """
        Calculate the post collisional velocities of  a given pair of particles
        """
        # Step 1 - Get relative velocity u
        u_rel = v_1 - v_2
        u_bar = vector_ops.normalise(u_lab)
        u = magnitude(u_rel)

        # Step 2 - Calculate angles theta and phi between the lab and reference
        #          frame
        u_xy = u_bar
        u_xy[2] = 0.0
        u_xy_bar = vector_ops.normalise(u_xy)
        u_xy = magnitude(u_xy)

        theta = np.arccos(u_bar[2])
        phi = np.arccos(u_xy_bar[0])

        # Step 3 - Get scattering angles THETA and PHI
        PHI = np.random.uniform(0.0, 2.0 * np.pi)
        delta_squared = self.__q_1 ** 2 * self.__q_2 ** 2 * np.min([self.__n_1, self.__n_2])
        delta_squared *= dt * self.__coulomb_logarithm
        delta_squared /= 8.0 * np.pi * u_bar ** 3 * self.__m_eff ** 2 * PhysicalConstants.epsilon_0 ** 2

        delta = np.random.normal(0.0, delta_squared)
        THETA = np.arcsin(2 * delta / (1 + delta ** 2))

        # Step 4 - Calculate du
        du = np.zeros((3,))
        du[0] = u_rel[0] / u_xy * u_rel[2] * np.sin(THETA) * np.cos(PHI) - \
                u_rel[1] / u_xy * u * np.sin(THETA) * np.sin(PHI) - \
                u_rel[0] * (1 - np.cos(THETA))
        du[1] = u_rel[1] / u_xy * u_rel[2] * np.sin(THETA) * np.cos(PHI) + \
                u_rel[0] / u_xy * u * np.sin(PHI) * np.sin(THETA) - \
                u_rel[1] * (1 - np.cos(THETA))
        du[2] = -u_xy * np.sin(THETA) * np,cos(PHI) - \
                u_rel[2] * (1 - np.cos(THETA))

        # Step 5 - Update velocities
        v_1 = v_1 + self.__m_eff / self.__m_1 * du
        v_2 = v_2 - self.__m_eff / self.__m_2 * du

    def single_time_step(self, vel, dt):
        """
        Carry out a single time step of simulation

        vel: Nx3 array of velocity components for particles, the velocities
             contain the particles of each species sequentially, N = n_1 + n_2
        dt: timestep size
        """
        # Function is not implemented for multiple species, or odd particle
        # number
        if self.__single_species is not True or self.__n_1 % 2 != 0:
            raise RuntimeError("Multiple species is not yet implemented!")

        # Step 1 - Randomly change addresses of velocities, for each species
        AbeCoulombCollisionModel.randomise_velocities(vel)

        # Step 2 - Calculate post-collisional velocities
        for i in range(vel.shape[0] / 2):
            idx = 2 * i
            v_1 = vel[idx, :]
            v_2 = vel[idx + 1, :]
            AbeCoulombCollisionModel.calculate_post_collision_velocities(v_1, v_2)

        return vel

    def run_sim(self, vel, dt, final_time):
        """
        Run simulation

        vel: Nx3 array of velocities for particles, the velocities
             contain the particles of each species sequentially, N = n_1 + n_2
        dt: time step to be used in simulation
        final_time: time of simulation
        """
        if self.__single_species:
            assert vel.shape[0] == self.__n_1
        else:
            assert vel.shape[0] == self.__n_1 + self.__n_2
        assert vel.shape[1] == 3

        num_steps = ceil(final_time / dt)
        vel_results = np.zeros((vel.shape[0], vel.shape[1], num_steps))
        vel_results[:, :, 0] = vel
        idx = 1
        t = 0.0
        for t < final_time:
            vel = self.single_time_step(vel, dt)

            vel_results[:, :, idx] = vel

            idx += 1
            t += dt

        return vel_results


if __name__ == '__main__':
    pass
