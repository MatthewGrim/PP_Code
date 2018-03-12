"""
Author: Rohan Ramasamy
Date: 09/03/2018

This file contains code to assess the relaxation rates for different relaxation processes
"""

import numpy as np
from matplotlib import pyplot as plt 

from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import CoulombCollision, ChargedParticle
from plasma_physics.pysrc.theory.coulomb_collisions.relaxation_processes import RelaxationProcess 
from plasma_physics.pysrc.utils.unit_conversions import UnitConversions 
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants

def plot_collisional_frequencies():
	"""
	Plot the variation of collisional frequency with density, and temperature for different 
	collisions
	"""
	deuterium = ChargedParticle(2.01410178 * 1.66054e-27, PhysicalConstants.electron_charge)
	electron = ChargedParticle(9.014e-31, -PhysicalConstants.electron_charge)

	impact_parameter_ratio = 1.0 # Is not necessary for this analysis
	velocities = np.logspace(4, 6, 100)
	number_density = np.logspace(20, 30, 100)
	VEL, N = np.meshgrid(velocities, number_density, indexing='ij')

	collision_pairs = [["Deuterium-Deuterium", deuterium, deuterium],
	                   ["Electron-Deuterium", electron, deuterium],
	                   ["Electron-Electron", electron, electron]]
	num_temp = 3
	temperatures = np.logspace(3, 5, num_temp) * UnitConversions.eV_to_K 
	for j, pair in enumerate(collision_pairs):
		name = pair[0]
		p_1 = pair[1]
		p_2 = pair[2]

		fig, ax = plt.subplots(2, num_temp, figsize=(15, 20))
		fig.suptitle("Collisional Frequencies for {} Relaxation".format(name))
		for i, temp in enumerate(temperatures):
			collision = CoulombCollision(p_1, p_2, impact_parameter_ratio, VEL)
			relaxation_process = RelaxationProcess(collision)

			kinetic_frequency = relaxation_process.kinetic_loss_stationary_frequency(N, temp, VEL)
			momentum_frequency = relaxation_process.momentum_loss_stationary_frequency(N, temp, VEL, 
				                                                                       first_background=True)

			im = ax[0, i].contourf(np.log10(VEL), np.log10(N), np.log10(kinetic_frequency), 100)
			ax[0, i].set_title("Kinetic: T = {}".format(temp * UnitConversions.K_to_eV))
			ax[0, i].set_xlabel("Velocity (ms-1)")
			ax[0, i].set_ylabel("Number density (m-3)")
			fig.colorbar(im, ax=ax[0, i])

			im = ax[1, i].contourf(np.log10(VEL), np.log10(N), np.log10(momentum_frequency), 100)
			ax[1, i].set_title("Momentum: T = {}".format(temp * UnitConversions.K_to_eV))
			ax[1, i].set_xlabel("Velocity (ms-1)")
			ax[1, i].set_ylabel("Number density (m-3)")
			fig.colorbar(im, ax=ax[1, i])

		plt.show()


if __name__ == '__main__':
	plot_collisional_frequencies()

