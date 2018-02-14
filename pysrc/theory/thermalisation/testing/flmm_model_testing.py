"""
Author: Rohan Ramasamy
Date: 13/02/2018

This script contains a set of functions to explore the FLMM model to see if it is consistent with the version from the paper:

"Thermonuclear burn characteristics of compressed deuterium-tritium microspheres" - G. S. Fraley,
	E. J. Linnebur, R. J. Mason, and R. L. Morse
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


from plasma_physics.pysrc.theory.thermalisation.flmm_model import FLMMmodel


def plot_stopping_distances():
	normalise = True
	num_pts = 100
	
	temperatures = np.logspace(0, 2, num_pts)
	solid_density = 0.213
			
	plt.figure()
	for i, density in enumerate([0.213, 10.0, 100.0, 1000.0, 10000.0]):
	# for i, density in enumerate([0.213]):

		# Get approximate electron and ion contributions 
		lamda_e = FLMMmodel.electron_stopping_distance(temperatures, density, solid_density)
		lamda_i = FLMMmodel.ion_stopping_distance(temperatures, density, solid_density)

		# Numerical integration for more exact value
		lamda_numerical = np.zeros(temperatures.shape)
		for i, T in enumerate(temperatures):
				rho = density

				x, U = FLMMmodel.get_U_x(T, rho, solid_density)
				lamda_numerical[i] = x[-1]

		# Multiply by densities for plotting
		if normalise:
			lamda_e *= density
			lamda_i *= density
			lamda_numerical *= density

		# plt.loglog(temperatures, lamda_e, label='electron_stopping_distance_{}'.format(density))
		# plt.loglog(temperatures, lamda_i, label='ion_stopping_distance_{}'.format(density))
		plt.loglog(temperatures, lamda_numerical, label='numerical_stopping_distance_{}'.format(density))

	plt.axhline(1, linestyle='--')
	plt.title("Stopping Distances of Alpha particles in different bulk plasma conditions")
	plt.ylim([1e-2, 3])
	plt.xlim([1, 100])
	plt.legend()
	plt.show()


def plot_U_x():
	normalise = True
	temperature = 100.0
	rho = 100.0
	solid_density = 0.213

	x, U = FLMMmodel.get_U_x(temperature, rho, solid_density)

	if normalise:
		x *= rho

	plt.figure()

	plt.plot(x, U)

	plt.title("Energy Profile for T = {}, density = {}, solid density = ".format(temperature, rho, solid_density))
	plt.show()


if __name__ == '__main__':
	plot_stopping_distances()
	
	# plot_U_x()

