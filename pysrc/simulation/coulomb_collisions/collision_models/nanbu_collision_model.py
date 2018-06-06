"""
Author: Rohan Ramasamy
Date: 06/06/2018

This file contains the model for coulomb collisions outlined in:

"Theory of cumulative small-anle collisions in plasmas" - K. Nanbu
"""

import numpy as np
import scipy.interpolate.interp1d as interp1d

from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import ChargedParticle
from plasma_physics.pysrc.simulation.pic.algo.geometry import vector_ops
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants

class NanbuCollisionModel(object):
	def __init__(self, number_densities, particles, particle_weightings):
		"""
		Initialiser for Nanbu simulation class

		number_densities: array or integer of number densities of different species
		particles: array or ChargedParticle of different species
		particle_weightings: array or integer of particle weights
		"""
		if isinstance(number_densities, np.ndarray):
			assert isinstance(particles, np.ndarray)
			assert isinstance(particle_weightings, np.ndarray)
			assert number_densities.shape == particles.shape == particle_weightings.shape
			assert len(number_densities.shape) == 1
			for i, n in enumerate(number_densities):
				assert n == number_densities[0]

			self.__num_species = number_densities.shape[0]
		elif isinstance(number_densities, int)
			assert isinstance(particles, ChargedParticle)
			assert isinstance(particle_weightings, int)
			
			self.__num_species = 1
		else:
			raise RuntimeError("number_densities must be either a float or numpy array")

		self.__particles = particles
		self.__particle_weights = particle_weightings
		self.__number_densities = number_densities

		if self.__num_species > 2:
			raise RuntimeError("Multicomponent species are currently not handled")

	def __calculate_s(self, g_mag, dt):
		# Assume number density is equal for all species
		n = self.number_densities[0]

		# Calculate b_90 for collisions
		q_A = self.__particles[0].q
		q_B = self.__particles[1].q
		m_A = self.__particles[0].m
		m_B = self.__particles[1].m
		m_eff = m_A * m_B / (m_A + m_B)
		b_90 = q_A * q_B / (2 * np.pi * PhysicalConstants.epsilon_0 * m_eff * g_mag ** 2)

		# Calculate coulomb logarithm
		T_background = 1000.0
		debye_length = PhysicalConstants.epsilon_0 * T_background
        debye_length /= n * PhysicalConstants.electron_charge ** 2
        debye_length = np.sqrt(debye_length)

        coulomb_logarithm = np.log(debye_length / b_90)

		# Calculate s
		s = n * g_mag * np.pi * b_90 ** 2 * coulomb_logarithm * dt

		return s

	def __calculate_A(self, s):
		data_file = os.path.join("data", "A_interpolation_values.txt")
		data = np.loadtxt(data_file)
		s_data = data[0, :]
		A_data = data[1, :]
		A_interpolator = interp1d(s_data, A_data)

		A = A_interpolator(s)
		return A

	def __calculate_chi(self):
		pass

	def __calculate_post_collision_velocities(self):
		pass

	def __run_single_timestep(self, velocities, dt):
		# Assume number density is equal for all species
		n = self.number_densities[0]

		# Calculate relative velocities of species pair
		velocities_A = velocities[:n, :]
		velocities_B = velocities[n:, :]
		g_components = velocities_A - velocities_B
		g_mag = np.sqrt(g_components[:, 0] ** 2 + g_components[:, 1] ** 2 + g_components[:, 2] ** 2)

		# Calculate parameter s
		s = self.__calculate_s(g_mag, dt)

		# Calculate parameter A
		A = self.__calculate_A(s)

		# Calculate scattering angle chi
		chi = self.__calculate_chi(A)

		# Calculate post collisional velocities
		self.__calculate_post_collision_velocities()


	def run_sim(velocities, final_time, dt):
        """
        Run simulation

        vel: Nx3 array of velocities for particles, the velocities
             contain the particles of each species sequentially, N = n_1 + n_2
        dt: time step to be used in simulation
        final_time: time of simulation
        """
		assert velocities.shape[0] == np.sum(self.__number_densities)
		assert velocities.shape[1] == 3

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

