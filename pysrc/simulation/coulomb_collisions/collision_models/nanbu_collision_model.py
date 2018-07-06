"""
Author: Rohan Ramasamy
Date: 06/06/2018

This file contains the model for coulomb collisions outlined in:

"Theory of cumulative small-anle collisions in plasmas" - K. Nanbu
"""

import numpy as np
import math
from scipy.interpolate import interp1d as interp1d
import os
import sys
import matplotlib.pyplot as plt

from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import ChargedParticle
from plasma_physics.pysrc.simulation.pic.algo.geometry import vector_ops
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants
from plasma_physics.pysrc.utils.unit_conversions import UnitConversions

class NanbuCollisionModel(object):
    def __init__(self, number_densities, particles, particle_weightings, 
                 coulomb_logarithm=None, frozen_species=None, include_self_collisions=False):
        """
        Initialiser for Nanbu simulation class

        number_densities: array or integer of number densities of different species
        particles: array or ChargedParticle of different species
        particle_weightings: array or integer of particle weights
        """
        # Carry out defensive checks
        if isinstance(number_densities, np.ndarray):
            assert isinstance(particles, np.ndarray)
            assert isinstance(particle_weightings, np.ndarray)
            assert number_densities.shape == particles.shape == particle_weightings.shape
            assert len(number_densities.shape) == 1
            for i, n in enumerate(number_densities):
                assert n == number_densities[0]

            # Set particle variables
            self.__num_species = number_densities.shape[0]
            self.__particles = particles
            self.__particle_weights = particle_weightings
            self.__number_densities = number_densities
            self.__frozen_species = frozen_species if frozen_species is not None else np.zeros(number_densities.shape).astype(bool)
            
            assert np.all(self.__number_densities == self.__number_densities[0]), "Variable simulated particles currently not handled"

            # Set boolean to determine if self collisions are enabled
            self.__include_self_collisions = include_self_collisions
        elif isinstance(number_densities, int):
            assert isinstance(particles, ChargedParticle)
            assert isinstance(particle_weightings, int)
            assert frozen_species is None or isinstance(frozen_species, bool)
            
            # Set particle variables
            self.__num_species = 1
            self.__particles = [particles]
            self.__particle_weights = [particle_weightings]
            self.__number_densities = [number_densities]
            self.__frozen_species = [frozen_species] if frozen_species is not None else [False]

            # For a single species simulations, self collisions must be true
            self.__include_self_collisions = True
        else:
            raise RuntimeError("number_densities must be either a float or numpy array")

        # Get start index in velocities array for each species
        species_start_idx = np.zeros((self.__num_species))
        idx = 0
        for i, n in enumerate(self.__number_densities):
            species_start_idx[i] = idx
            idx += n
        self.__species_start_idx = species_start_idx.astype(int)
        
        # Define temperatures of plasma - this will be set at the beginning of the simulation
        self.temperature = None

        # Generator interpolator for A
        data_file = os.path.join("/home/rohan/Code/plasma_physics/pysrc/simulation/coulomb_collisions/collision_models", "data", "A_interpolation_values.txt")
        self.__A_data = np.loadtxt(data_file)
        s_data = self.__A_data[0, :]
        A_data = self.__A_data[1, :]
        A_interpolator = interp1d(s_data, A_data)
        self._A_interpolator = A_interpolator

        # Set max s value to 6.0 as in Nanbu
        self.__max_s = 6.0

        # Set coulomb logarithm to a fixed value if it is specified
        self.__coulomb_logarithm = coulomb_logarithm

    def __calculate_s(self, idx_A, idx_B, g_mag, dt):
        """
        Calculate s parameter for collisions

        idx_A: index of particle A in arrays
        idx_B: index of particle B in arrays
        g_mag: relative velocity magnitude of collision
        dt: timestep
        """
        if self.__num_species == 1:
            n = self.__number_densities[0] / 2
        else:
            # Assume number of simulated particles is equal for all species and that species B is the 
            # background species for the interaction
            n = long(self.__number_densities[0]) * self.__particle_weights[idx_B]

        # Get charges, and calculate m_eff for collisions
        q_A = self.__particles[idx_A].q
        m_A = self.__particles[idx_A].m
        q_B = self.__particles[idx_B].q
        m_B = self.__particles[idx_B].m
        m_eff = m_A * m_B / (m_A + m_B)

        # Calculate coulomb logarithm
        if self.__coulomb_logarithm is None:
            T_background = self.temperature
            debye_length = PhysicalConstants.epsilon_0 * T_background
            debye_length /= n * PhysicalConstants.electron_charge ** 2
            debye_length = np.sqrt(debye_length)
            g_bar = np.mean(g_mag ** 2)
            b_90 = q_A * q_B / (2 * np.pi * PhysicalConstants.epsilon_0 * m_eff * g_bar)

            coulomb_logarithm = np.log(debye_length / b_90)
        else:
            coulomb_logarithm = self.__coulomb_logarithm

        # Calculate s
        # b_90 = q_A * q_B / (2 * np.pi * PhysicalConstants.epsilon_0 * m_eff * g_mag ** 2)
        # s = n * g_mag * np.pi * b_90 ** 2 * coulomb_logarithm * dt
        s = coulomb_logarithm / (4 * np.pi) * (q_A * q_B / (PhysicalConstants.epsilon_0 * m_eff)) ** 2
        s *= n / g_mag ** 3 * dt

        return s

    def __calculate_A(self, s):
        # Interpolate from pre-calculated values
        A = np.zeros(s.shape)
        for i, s_val in enumerate(s):
            try:
                A_val = self._A_interpolator(s_val)
            except ValueError:
                if s_val < self.__A_data[0, 0]:
                    A_val = 1.0 / s_val
                elif self.__A_data[0, -1] < s_val:
                    A_val = 3.0 * np.exp(-s_val)
                else:
                    raise ValueError("Unexpected behaviour!")

            A[i] = A_val

        return A

    def __calculate_cos_chi(self, A, s, debug=False):
        cos_chi = np.zeros(s.shape)
        U = np.random.uniform(0, 1, s.shape)
        for i, s_val in enumerate(s):
            U_val = U[i]
            A_val = A[i]
            if s_val > self.__max_s or A_val == 0.0:
                # Assume isotropic
                cos_chi_val = 2 * U_val - 1
            else:
                cos_chi_val = 1 / A_val * np.log(np.exp(-A_val) + 2 * U_val * np.sinh(A_val))

            # Correct overflow in simulations
            if np.isinf(cos_chi_val):
                cos_chi_val = 1 + s_val * np.log(U_val)

            cos_chi[i] = cos_chi_val
    
        for i, cos_chi_val in enumerate(cos_chi):
            assert -1.0 <= cos_chi_val <= 1.0, "{}, {}, {}, {}".format(cos_chi_val, s[i], A[i], U[i])

        if debug:
            plt.figure()
            plt.hist(np.arccos(cos_chi), 100, normed=True)
            plt.show()

        return cos_chi

    def __calculate_post_collision_velocities(self, idx_A, idx_B, vel_A, vel_B, g_comp, g_mag, cos_chi, epsilon):
        """
        Calculate new velocities after collision

        idx_A: index of particle A in arrays
        idx_B: index of particle B in arrays
        vel_A: velocities of species A in collision
        vel_B: velocities of species B in collision
        g_comp: relative velocity components
        g_mag: relative velocity magnitudes
        cos_chi: cosine of scattering angles
        epsilon: angle of rotation about x axis
        """
        # Get particle weightings and collision probabilities
        w_A = self.__particle_weights[idx_A]
        w_B = self.__particle_weights[idx_B]
        w_max = float(max(w_A, w_B))
        collision_threshold_A = w_B / w_max
        collision_threshold_B = w_A / w_max
        Z_A = np.random.uniform(0, 1, size=(g_comp.shape[0], 1)) < collision_threshold_A 
        Z_B = np.random.uniform(0, 1, size=(g_comp.shape[0], 1)) < collision_threshold_B 

        # Calculate mass factors
        m_A = self.__particles[idx_A].m
        m_B = self.__particles[idx_B].m
        A_factor = (m_B / (m_A + m_B))
        B_factor = (m_A / (m_A + m_B))

        # Calculate h vectors
        g_perp = np.sqrt(g_comp[:, 1] ** 2 + g_comp[:, 2] ** 2)
        cos_e = np.cos(epsilon)
        sin_e = np.sin(epsilon)
        h_vec = np.zeros(g_comp.shape)
        h_vec[:, 0] = g_perp * cos_e
        h_vec[:, 1] = -(g_comp[:, 1] * g_comp[:, 0] * cos_e + g_mag * g_comp[:, 2] * sin_e) / g_perp
        h_vec[:, 2] = -(g_comp[:, 2] * g_comp[:, 0] * cos_e - g_mag * g_comp[:, 1] * sin_e) / g_perp

        # Give chi a new axis to allow matrix multiplication
        cos_chi = cos_chi[:, np.newaxis]
        sin_chi = np.sqrt(1.0 - cos_chi ** 2)
        deflection_vec = (g_comp * (1.0 - cos_chi) + h_vec * sin_chi) 
        if not self.__frozen_species[idx_A]:
            vel_A -= Z_A * A_factor * deflection_vec
        if not self.__frozen_species[idx_B]:
            vel_B += Z_B * B_factor * deflection_vec

    def __randomise_velocities(self, start_A, start_B, N_A, N_B, velocities):
        """
        Randomise velocities for coulomb collisions

        start_A: start index in velocities of species A
        start_B: start index in velocities of species B
        N_A: number of simulated particles of species A
        N_B: number of simulated particles of species B
        velocities: N_T X 3 array of all particle velocities in simulation
        """
        current_state = np.random.get_state()
        indices_A = np.asarray(range(N_A))
        indices_B = np.asarray(range(N_B))
        velocities_A = velocities[start_A:start_A + N_A, :]
        velocities_B = velocities[start_B:start_B + N_B, :]
        np.random.shuffle(velocities_A)
        np.random.shuffle(velocities_B)
        np.random.set_state(current_state)
        np.random.shuffle(indices_A)
        np.random.shuffle(indices_B)

        return velocities_A, velocities_B, indices_A, indices_B

    def __unshuffle_velocities(self, start_A, start_B, N_A, N_B, velocities_A, velocities_B, indices_A, indices_B, new_vel):
        """
        Unshuffle velocities for coulomb collisions

        start_A: start index in velocities of species A
        start_B: start index in velocities of species B
        N_A: number of simulated particles of species A
        N_B: number of simulated particles of species B
        velocities_A: N_A X 3 array of all particle velocities from species_A
        velocities_B: N_B X 3 array of all particle velocities from species_B
        indicies_A: randomised particle indices of species A
        indicies_B: randomised particle indices of species B
        new_vel: N_T X 3 array of all new particle velocities in simulation
        """
        velocities_A = velocities_A[indices_A.argsort(), :]
        velocities_B = velocities_B[indices_B.argsort(), :]
        new_vel[start_A:start_A+N_A, :] = velocities_A
        new_vel[start_B:start_B+N_B, :] = velocities_B

    def __simulate_coulomb_collisions(self, idx_A, idx_B, start_A, start_B, N_A, N_B, new_vel, dt):
        """
        idx_A:start index of species properties for species A
        idx_B: start index of species properties for species B
        start_A: start index of velocities for species A
        start_B: start index of velocities for species B
        N_A: Number of simulated particles of species A
        N_B: Number of simulated particles of species B
        new_vel: N_T X 3 array of particle velocities of all species within the simulation
        dt: time step of simulation
        """
        velocities_A, velocities_B, indices_A, indices_B = self.__randomise_velocities(start_A, start_B, N_A, N_B, new_vel)

        # Calculate relative velocities of species pairs and their magnitudes
        g_components = velocities_A - velocities_B
        g_mag = np.sqrt(g_components[:, 0] ** 2 + g_components[:, 1] ** 2 + g_components[:, 2] ** 2)

        # Calculate scattering angles chi and epsilon
        s = self.__calculate_s(idx_A, idx_B, g_mag, dt)
        A = self.__calculate_A(s)
        cos_chi = self.__calculate_cos_chi(A, s)
        epsilon = np.random.uniform(0, 2 * np.pi, g_mag.shape)

        self.__calculate_post_collision_velocities(idx_A, idx_B, velocities_A, velocities_B, g_components, g_mag, cos_chi, epsilon)
        self.__unshuffle_velocities(start_A, start_B, N_A, N_B, velocities_A, velocities_B, indices_A, indices_B, new_vel)    

    def single_time_step(self, velocities, dt):
        # Get array for new velocities, and number of simulated particles - assumed constant for all species
        new_vel = np.copy(velocities)
        n = self.__number_densities[0]
        
        # Carry out binary collisions between plasmas of different species
        for i in range(self.__num_species):
            for j in range(i+1, self.__num_species):
                start_A = self.__species_start_idx[i]
                start_B = self.__species_start_idx[j]

                # Simulate collisions
                self.__simulate_coulomb_collisions(i, j, start_A, start_B, n, n, new_vel, dt)

        # Carry out self-collisions of each plasma species
        if self.__include_self_collisions:
            n /= 2
            for i in range(self.__num_species):
                start_A = self.__species_start_idx[i]
                start_B = start_A + n

                # Simulate collisions
                self.__simulate_coulomb_collisions(i, i, start_A, start_B, n, n, new_vel, dt)

        # Get plasma temperature
        if self.__num_species == 1:
            vel_mag = np.sqrt(new_vel[:, 0] ** 2 + new_vel[:, 1] ** 2 + new_vel[:, 2] ** 2)
            self.temperature = np.std(vel_mag) ** 2 * self.__particles[0].m / (3.0 * PhysicalConstants.boltzmann_constant)
        else:
            # Setting temperature to None so that the code will break if coulomb logarithm is not set
            self.temperature = None

        return new_vel
        
    def run_sim(self, velocities, dt, final_time, seed=1):
        """
        Run simulation

        velocities: Nx3 array of velocities for particles, the velocities
             contain the particles of each species sequentially, N = n_1 + n_2
        dt: time step to be used in simulation
        final_time: time of simulation
        """
        assert velocities.shape[0] == np.sum(self.__number_densities), "{} != {}".format(velocities.shape[0], np.sum(self.__number_densities)) 
        assert velocities.shape[1] == 3, velocities.shape[1]

        # # Set seed before simulation
        np.random.seed(seed)

        # Set temperature
        vel_mag = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2 + velocities[:, 2] ** 2)
        self.temperature = np.std(vel_mag) ** 2 * self.__particles[0].m / (3.0 * PhysicalConstants.boltzmann_constant)

        num_steps = int(math.ceil(final_time / dt) + 1)
        vel_results = np.zeros((velocities.shape[0], velocities.shape[1], num_steps))
        vel_results[:, :, 0] = velocities
        times = np.zeros((num_steps,))
        idx = 1
        t = 0.0
        times[0] = t
        print("Starting simulation...")
        while idx < num_steps:
            t += dt
            print("Timestep {}: t = {}".format(idx, t))

            velocities = self.single_time_step(velocities, dt)

            vel_results[:, :, idx] = velocities
            times[idx] = t

            idx += 1
        print("Simulation Complete!")

        return times, vel_results


if __name__ == '__main__':
    pass

