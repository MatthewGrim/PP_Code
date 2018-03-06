"""
Author: Rohan Ramasamy
Date: 06/03/2018

This file contains code to test the different scattering angles for charged particle interactions
between electrons and ions
"""

import numpy as np
from matplotlib import pyplot as plt

from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import ChargedParticle, CoulombCollision
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants
from plasma_physics.pysrc.utils.unit_conversions import UnitConversions


def plot_collisional_angles():
	"""
	Simple function to plot the scattering angles for different binary collisions at they vary with
	impact parameter ratio  
	"""
	impact_parameter_ratios = np.linspace(0.0, 10.0, 100)

	deuterium = ChargedParticle(2.01410178 * 1.66054e-27, PhysicalConstants.electron_charge)
	electron = ChargedParticle(9.014e-31, -PhysicalConstants.electron_charge)
	
	temperature = 11600.0 # 1eV taken as base case
	electron_velocity = np.sqrt(PhysicalConstants.boltzmann_constant * temperature / electron.m)
	deuterium_velocity = np.sqrt(PhysicalConstants.boltzmann_constant * temperature / deuterium.m)
	print("Electron Thermal Velocity at {}K: {}".format(temperature, electron_velocity))
	print("Ion Thermal Velocity at {}K: {}".format(temperature, deuterium_velocity))

	collision_pairs = [["Deuterium-Deuterium", deuterium, deuterium, deuterium_velocity],
	                   ["Electron-Deuterium", electron, deuterium, electron_velocity],
	                   ["Electron-Electron", electron, electron, electron_velocity]]
	for pair in collision_pairs:
		name = pair[0]
		p_1 = pair[1]
		p_2 = pair[2]
		velocity = pair[3]

		chi = np.zeros(impact_parameter_ratios.shape)
		for i, impact_parameter_ratio in enumerate(impact_parameter_ratios):
			collision = CoulombCollision(p_1, p_2, impact_parameter_ratio, velocity)

			chi[i] = collision.chi

		collision = CoulombCollision(p_1, p_2, 1.0, velocity)
		b_90 = collision.b_90

		print("Impact Parameter for {} Collision: {}".format(name, b_90))

		# Convert chi to degrees
		chi /= np.pi
		chi *= 180.0

		# Plot chi vs.impact parameter 
		plt.figure()

		plt.plot(impact_parameter_ratios, chi)

		plt.title("Scattering angle vs Impact parameter for {} Collision")
		plt.xlabel("Impact parameter ratio (b / b_90)")
		plt.ylabel("Scattering angle (degrees)")
		plt.show()


def plot_impact_parameter_variation_with_temperature():
	"""
	Simple function to see how the impact parameter varies with temperature
	"""
	temperatures = UnitConversions.eV_to_K * np.logspace(1, 4, 100)
	deuterium = ChargedParticle(2.01410178 * 1.66054e-27, PhysicalConstants.electron_charge)
	electron = ChargedParticle(9.014e-31, -PhysicalConstants.electron_charge)
	
	electron_b_90 = np.zeros(temperatures.shape)
	deuterium_b_90 = np.zeros(temperatures.shape)
	for i, temperature in enumerate(temperatures):
		electron_velocity = np.sqrt(PhysicalConstants.boltzmann_constant * temperature / electron.m)
		deuterium_velocity = np.sqrt(PhysicalConstants.boltzmann_constant * temperature / deuterium.m)
		
		collision = CoulombCollision(deuterium, electron, 1.0, electron_velocity)
		b_90 = collision.b_90
		electron_b_90[i] = b_90

		collision = CoulombCollision(deuterium, electron, 1.0, deuterium_velocity)
		b_90 = collision.b_90
		deuterium_b_90[i] = b_90

	plt.figure()

	plt.loglog(temperatures, electron_b_90, label="Electron Velocity")
	plt.loglog(temperatures, deuterium_b_90, label="Deuterium Velocity")

	plt.title("Impact Parameter vs Temperature for Deuterium-Electron binary collision")
	plt.ylabel("Impact Parameter (m)")
	plt.xlabel("Temperature (K)")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	plot_collisional_angles()
	plot_impact_parameter_variation_with_temperature()

