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
    def __init__(self, N_1, particle_1, w_1=1,
                 N_2=None, particle_2=None, w_2=None, freeze_species_2=False,
                 coulomb_logarithm=10.0):
        """
        Used to simulate collisions between two particle species

        N_1: total number of unweighted particles of species 1
        N_2: total number of unweighted particles of species 2
        particle_1: first particle species involved in collisions
        particle_2: second particle species involved in collisions
        w_1: number of particles per simulated particle of species 1
        w_2: number of particles per simulated particle of species 2
        freeze_species_2: boolean to determine if second species is
                          frozen so that its velocities are not updated
        """
        assert isinstance(N_1, int)
        assert isinstance(w_1, int)
        assert isinstance(particle_1, ChargedParticle)
        assert N_2 is None or isinstance(N_2, int)
        assert particle_2 is None or isinstance(particle_2, ChargedParticle)
        assert w_2 is None or isinstance(w_2, int)

        # Define first particle species
        self.__m_1 = particle_1.m
        self.__q_1 = particle_1.q
        self.__N_1 = N_1
        self.__w_1 = w_1
        self.__n_1 = N_1 * w_1

        # Define second particle species, if it exists
        if particle_2 is not None:
            self.__single_species = False
            self.__m_2 = particle_2.m
            self.__q_2 = particle_2.q
            self.__N_2 = N_2
            self.__w_2 = w_2
            self.__n_2 = w_2 * N_2
            self.__m_eff = self.__m_1 * self.__m_2 / (self.__m_1 + self.__m_2)
            self.__freeze_species_2 = freeze_species_2

            # Set collision thresholds
            self.__max_w = float(max(w_1, w_2))
            self.__collision_threshold_1 = w_2 / self.__max_w
            self.__collision_threshold_2 = w_1 / self.__max_w
        else:
            self.__single_species = True
            self.__m_2 = self.__m_1
            self.__q_2 = self.__q_1
            self.__n_2 = self.__n_1
            self.__m_eff = self.__m_1 ** 2 / (2 * self.__m_1)

            # Set collision thresholds
            self.__collision_threshold_1 = 1.0
            self.__collision_threshold_2 = 1.0

        # Coulomb logarithm is currently fixed in method
        self.__coulomb_logarithm = coulomb_logarithm

    def __randomise_velocities(self, vel):
        """
        Randomise the addresses of particles from each species. In such a way,
        pairs of particles in adjacent indices are formed for collisions. The 
        randomised velocities need to be unshuffled. 

        vel: Nx3 array of velocity components for particles, the velocities
             contain the particles of each species sequentially, N = n_1 + n_2
        """
        current_state = np.random.get_state()
        indices = np.asarray(range(vel.shape[0]))
        if self.__single_species:
            np.random.shuffle(vel)
            np.random.set_state(current_state)

            np.random.shuffle(indices)
        else:
            np.random.shuffle(vel[:self.__N_1, :])
            np.random.shuffle(vel[self.__N_1:, :])
            np.random.set_state(current_state)

            np.random.shuffle(indices[:self.__N_1])
            np.random.shuffle(indices[self.__N_1:])

        return indices

    def calculate_post_collision_velocities(self, v_1, v_2, dt):
        """
        Calculate the post collisional velocities of  a given pair of particles
        """
        # Step 1 - Get relative velocity u and perpendicular velocity u_xy
        u_rel = v_1 - v_2
        u = vector_ops.magnitude(u_rel)
        u_xy = np.sqrt(u_rel[0] ** 2 + u_rel[1] ** 2)

        # Step 2 - Get scattering angles THETA and PHI
        PHI = np.random.uniform(0.0, 2.0 * np.pi)
        c_phi = np.cos(PHI)
        s_phi = np.sin(PHI)

        delta_squared = self.__q_1 ** 2 * self.__q_2 ** 2 * self.__n_1
        delta_squared *= dt * self.__coulomb_logarithm
        delta_squared /= 8.0 * np.pi * u ** 3 * self.__m_eff ** 2 * PhysicalConstants.epsilon_0 ** 2

        delta = np.random.normal(0.0, np.sqrt(delta_squared))
        s_theta = 2 * delta / (1 + delta ** 2)
        one_minus_c_theta = 2 * delta ** 2 / (1 + delta ** 2)

        # Step 3 - Calculate du
        du = np.zeros((3,))
        if u_xy != 0.0:
            du[0] = u_rel[0] / u_xy * u_rel[2] * s_theta * c_phi - \
                u_rel[1] / u_xy * u * s_theta * s_phi - \
                u_rel[0] * one_minus_c_theta
            du[1] = u_rel[1] / u_xy * u_rel[2] * s_theta * c_phi + \
                u_rel[0] / u_xy * u * s_theta * s_phi - \
                u_rel[1] * one_minus_c_theta
            du[2] = -u_xy * s_theta * c_phi - u_rel[2] * one_minus_c_theta
        else:
            du[0] = u * s_theta * c_phi
            du[1] = u * s_theta * s_phi
            du[2] = -u * one_minus_c_theta

        # Step 4 - Update velocities
        P_1 = 1.0 if np.random.uniform(0, 1) <= self.__collision_threshold_1 else 0.0
        P_2 = 1.0 if np.random.uniform(0, 1) <= self.__collision_threshold_2 else 0.0
        new_v_1 = v_1 + P_1 * self.__m_eff / self.__m_1 * du
        new_v_2 = v_2 - P_2 * self.__m_eff / self.__m_2 * du

        return new_v_1, new_v_2

    def single_time_step(self, vel, dt):
        """
        Carry out a single time step of simulation

        vel: Nx3 array of velocity components for particles, the velocities
             contain the particles of each species sequentially, N = n_1 + n_2
        dt: timestep size
        """
        # Function is not implemented for multiple species, or odd particle
        # number
        if self.__single_species:
            assert(self.__N_1 % 2 == 0), "Uneven number of particles is not implemented"

            # Step 1 - Randomly change addresses of velocities, for each species
            indices = self.__randomise_velocities(vel)

            # Step 2 - Calculate post-collisional velocities
            new_vel = np.zeros(vel.shape)
            for i in range(vel.shape[0] // 2):
                idx = 2 * i
                v_1 = vel[idx, :]
                v_2 = vel[idx + 1, :]
                new_v_1, new_v_2 = self.calculate_post_collision_velocities(v_1, v_2, dt)
                new_vel[idx, :] = new_v_1
                new_vel[idx + 1, :] = new_v_2
        else:
            assert(self.__N_1 == self.__N_2), "Different number of particles is not implemented"

            # Step 1 - Randomly change addresses of velocities, for each species
            indices = self.__randomise_velocities(vel)

            # Step 2 - Calculate post-collisional velocities
            new_vel = np.zeros(vel.shape)
            for i in range(self.__N_1):
                idx_1 = i
                idx_2 = self.__N_1 + i
                v_1 = vel[idx_1, :]
                v_2 = vel[idx_2, :]
                new_v_1, new_v_2 = self.calculate_post_collision_velocities(v_1, v_2, dt)
                new_vel[idx_1, :] = new_v_1
                new_vel[idx_2, :] = v_2 if self.__freeze_species_2 else new_v_2

        # Step 3 - Sort the indices of velocity to maintain ordering
        new_vel = new_vel[indices.argsort(), :]

        return new_vel


    def run_sim(self, vel, dt, final_time, seed=1):
        """
        Run simulation

        vel: Nx3 array of velocities for particles, the velocities
             contain the particles of each species sequentially, N = n_1 + n_2
        dt: time step to be used in simulation
        final_time: time of simulation
        """
        if self.__single_species:
            assert vel.shape[0] == self.__N_1
        else:
            assert vel.shape[0] == self.__N_1 + self.__N_2
        assert vel.shape[1] == 3

        # Set seed
        np.random.seed(seed)

        num_steps = int(math.ceil(final_time / dt) + 1)
        vel_results = np.zeros((vel.shape[0], vel.shape[1], num_steps))
        vel_results[:, :, 0] = vel
        times = np.zeros((num_steps,))
        idx = 1
        t = 0.0
        times[0] = t
        print("Starting simulation...")
        while idx < num_steps:
            t += dt
            print("Timestep {}: t = {}".format(idx, t))

            vel = self.single_time_step(vel, dt)

            vel_results[:, :, idx] = vel
            times[idx] = t

            idx += 1
        print("Simulation Complete!")

        return times, vel_results


if __name__ == '__main__':
    pass
