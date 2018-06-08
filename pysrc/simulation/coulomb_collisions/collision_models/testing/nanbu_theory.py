"""
Author: Rohan Ramasamy
Date: 07/06/2018

This file contains code to validate figure 2 in the paper:

Theory of cumulative angle scattering in plasmas - K. Nanbu
"""

import os
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from plasma_physics.pysrc.simulation.pic.algo.geometry.vector_ops import *


def get_theta_and_phi_for_sims(theta_min, n):
	"""
	Equation for theta and phi of scattered particles

	n: number of samples
	"""
	U = np.random.uniform(0, 1, n)
	theta = 2 * np.arctan(theta_min / (2 * np.sqrt(U)))
	phi = 2 * np.pi * np.random.uniform(0, 1, n)

	return theta, phi


def get_chi(theta, phi, vec, original_vec):
	new_vec = np.zeros((3,))
	c_theta = np.cos(theta)
	s_theta = np.sin(theta)
	c_phi = np.cos(phi)
	s_phi = np.sin(phi)

	new_vec[0] = (c_theta * c_phi ** 2 + s_phi ** 2) * vec[0]
	new_vec[0] += (c_theta - 1.0) * s_phi * c_phi * vec[1]
	new_vec[0] += -s_theta * c_phi * vec[2]

	new_vec[1] = (c_theta - 1.0) * s_phi * c_phi * vec[0]
	new_vec[1] += (c_theta * s_phi ** 2 + c_phi ** 2) * vec[1]
	new_vec[1] += -s_theta * s_phi * vec[2]

	new_vec[2] = s_theta * c_phi * vec[0]
	new_vec[2] += s_theta * s_phi * vec[1]
	new_vec[2] += c_theta * vec[2]

	chi = np.arccos(dot(original_vec, new_vec) / (magnitude(original_vec) * magnitude(new_vec)))
	return chi, new_vec


def get_expectation_of_theta(theta_min):
	def func(eta):
		return 8.0 * (np.arctan(theta_min / (2 * eta))) ** 2 * eta

	return integrate.quad(func, 0.0, 1.0)[0]

def simulate_expectation(theta_min, num_particles=2000, n=500, plot_result=False):
	results = np.zeros((num_particles, n))
	file_name = "sin_chi_squared_{}_{}_{}".format(num_particles, n, theta_min)
	if os.path.exists(file_name):
		results = np.loadtxt(file_name)
	else:
		for i in range(num_particles):
			print("Particle: {}".format(i))
			vec = np.asarray([0.0, 0.0, 1.0])
			original_vec = np.asarray([0.0, 0.0, 1.0])
			thetas, phis = get_theta_and_phi_for_sims(theta_min, n)
			for j, theta in enumerate(thetas):
				phi = phis[j]

				chi, new_vec = get_chi(theta, phi, vec, original_vec)

				results[i, j] = np.sin(chi / 2.0)

				vec = new_vec

		np.savetxt("sin_chi_squared_{}_{}_{}".format(num_particles, n, theta_min), results)
	
	results = np.mean(results ** 2, axis=0)

	if plot_result:
		plt.figure()
		plt.plot(results)
		plt.show()



if __name__ == '__main__':
	for theta in np.logspace(-2, 1, 4):
		theta_min = theta / 180.0 * np.pi
		theta_2_expectation = get_expectation_of_theta(theta_min)
		print(theta_min, theta_2_expectation)

		n = 1000
		num_particles=20000
		simulate_expectation(theta_min, num_particles=num_particles, n=n)




