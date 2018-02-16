"""
Author: Rohan Ramasamy
Date: 13/02/2018

This script contains a set of functions to explore the FLMM model to see if it is consistent with the version from the paper:

"Thermonuclear burn characteristics of compressed deuterium-tritium microspheres" - G. S. Fraley,
	E. J. Linnebur, R. J. Mason, and R. L. Morse
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

from plasma_physics.pysrc.theory.thermalisation.flmm_model import FLMMmodel


def plot_stopping_distances():
	normalise = True
	num_pts = 100
	
	temperatures = np.logspace(0, 2, num_pts)
	solid_density = 0.213
	densities = [0.213, 10.0, 100.0, 1000.0, 10000.0]
	# densities = [8.3e-6, 0.000830, 0.001, 0.01, 0.1, 0.213]
	# densities = [0.213]
	t_halves = np.zeros((len(densities), ))

	fig, ax = plt.subplots(2, figsize=(10, 10))
	for i, density in enumerate(densities):
		# Get approximate electron and ion contributions 
		lamda_e = FLMMmodel.electron_stopping_distance(temperatures, density, solid_density)
		lamda_i = FLMMmodel.ion_stopping_distance(temperatures, density, solid_density)

		# Numerical integration for more exact value
		lamda_numerical = np.zeros(temperatures.shape)
		final_electron_energy = np.zeros(temperatures.shape)
		final_ion_energy = np.zeros(temperatures.shape)
		for j, T in enumerate(temperatures):
			rho = density

			x, U, U_ele, U_ion = FLMMmodel.get_U_x(T, rho, solid_density)
			lamda_numerical[j] = x[-1]
			final_electron_energy[j] = U_ele[-1]
			final_ion_energy[j] = U_ion[-1]
			
		T_f = interpolate.interp1d(final_ion_energy, temperatures)			
		t_halves[i] = T_f(0.5)

		# Multiply by densities for plotting
		if normalise:
			lamda_e *= density
			lamda_i *= density
			lamda_numerical *= density

		# ax[0].loglog(temperatures, lamda_e, label='electron_stopping_distance_{}'.format(density))
		# ax[0].loglog(temperatures, lamda_i, label='ion_stopping_distance_{}'.format(density))
		ax[0].loglog(temperatures, lamda_numerical, label='{}'.format(density))
		ax[1].semilogy(final_ion_energy, temperatures, label='{}'.format(density))
	
	# Set up plot to be similar to FLMM paper 
	fig.suptitle("Alpha particle thermalisation in different bulk plasma conditions")
	if normalise:
		ax[0].set_ylim([1e-2, 3])
		ax[0].set_xlim([1, 100])
	
	ax[1].axhline(32.0, linestyle ='--')
	ax[1].axvline(0.5, linestyle ='--')
	ax[1].set_ylim([1, 100])
	ax[1].set_xlim([0.0, 0.9])

	ax[0].set_title("Temperature vs. Stopping Distance")
	ax[1].set_title("Ion fractional energy vs. Temperature")
	ax[0].legend()
	ax[1].legend()
	
	plt.show()

	print("Half Energy Partition Temperatures")
	for i, density in enumerate(densities):
		print("Density: {}, Temperature: {}".format(density, t_halves[i]))


def plot_U_x():
	normalise = True
	temperature = 100.0
	rho = 100.0
	solid_density = 0.213

	x, U, U_ele, U_ion = FLMMmodel.get_U_x(temperature, rho, solid_density)

	if normalise:
		x *= rho

	plt.figure()

	plt.plot(x, U, label="Total")
	plt.plot(x, U_ele, label="Electron")
	plt.plot(x, U_ion, label="Ion")

	plt.legend()
	plt.title("Energy Profile for T = {}, density = {}, solid density = ".format(temperature, rho, solid_density))
	plt.show()


if __name__ == '__main__':
	plot_stopping_distances()
	# plot_U_x()

