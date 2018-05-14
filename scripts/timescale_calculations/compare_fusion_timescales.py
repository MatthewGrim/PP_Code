"""
Author: Rohan Ramasamy
Date: 11/05/2018

This script contains code to compare the fusion reaction rate with the thermalisation rate of a plasma. This can be 
used to define whether it is possible to sustain a non-Maxwellian plasma within a fusion device.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from plasma_physics.pysrc.theory.reactivities.fusion_reactivities import BoschHaleReactivityFit, FusionReaction, DDReaction, DTReaction, pBReaction
from plasma_physics.pysrc.utils.unit_conversions import UnitConversions
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants
from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import ChargedParticle, CoulombCollision
from plasma_physics.pysrc.theory.coulomb_collisions.relaxation_processes import RelaxationProcess, MaxwellianRelaxationProcess


def get_fusion_timescale(n, T, reaction):
	"""
	Function to get the reaction time scale of a given fusion reaction

	n: number density of the plasma (in particles per m3)
	T: temperature of the plasma (in keV)
	reaction: Fusion Reaction to get reactivity of
	"""
	assert isinstance(n, float) or isinstance(n, np.ndarray)
	assert isinstance(T, float)
	assert isinstance(reaction, FusionReaction)

	# Use fit to get reactivity - Bosch Hale fit assumes a Maxwellian plasma
	reactivity_fit = BoschHaleReactivityFit(reaction)
	reactivity = reactivity_fit.get_reactivity(T)

	# Get the reaction rate and invert to get the timescale
	reaction_rate = n * reactivity 
	return 1 / reaction_rate


def get_thermalisation_rate(n, T, relaxation_process, beam_velocity, reaction_name, force_calculation=False):
	"""
	Get the thermalisation rate of a given fusion reaction

	n: number density of the plasma (in particles per m3)
	T: temperature of the plasma (in keV)
	relaxation_process: relaxation process corresponding to the chosen fusion reaction
	reaction_name: name of reaction being simulated
	"""
	assert isinstance(n, float) or isinstance(n, np.ndarray)
	assert isinstance(T, float)
	assert isinstance(relaxation_process, RelaxationProcess)

	# Get stationary frequency
	stationary_frequency =  1.0 / relaxation_process.kinetic_loss_stationary_frequency(n, T, beam_velocity)
	
	# Get maxwellian frequency
	print(reaction_name)
	file_name = "maxwellian_frequencies_{}".format(reaction_name)
	file_path = os.path.join("data", file_name)
	if not os.path.exists(file_path) or force_calculation:
		if isinstance(n, np.ndarray):
			maxwellian_frequency = np.zeros(stationary_frequency.shape)
			for i, number_density in enumerate(n):
				print(i)
				# maxwellian_frequency[i] = relaxation_process.numerical_kinetic_loss_maxwellian_frequency(number_density, T, beam_velocity)
				maxwellian_frequency[i] = relaxation_process.maxwellian_collisional_frequency(number_density, T, beam_velocity)

			# Save result to file
			np.savetxt(file_name, maxwellian_frequency)
		else:
			maxwellian_frequency = relaxation_process.numerical_kinetic_loss_maxwellian_frequency(n, T, beam_velocity)
	else:
		maxwellian_frequency = np.loadtxt(file_path)

	return stationary_frequency, 1.0 / maxwellian_frequency


if __name__ == '__main__':
	# Define parameter space of scan
	n = np.logspace(20, 30, 10)
	T_keV = 10.0
	T_K = T_keV * 1e3 * UnitConversions.eV_to_K
	e = 1.5 * PhysicalConstants.boltzmann_constant * T_K

	# Set up reaction
	reaction_name = "pB"
	if reaction_name == "DD":
		reaction = DDReaction()
		deuterium = ChargedParticle(2.01410178 * UnitConversions.amu_to_kg, PhysicalConstants.electron_charge)

		beam_velocity = np.sqrt(2 * e / deuterium.m)	
		beam_species = deuterium
		background_species = deuterium
	elif reaction_name == "DT":
		reaction = DTReaction()
		deuterium_tritium = ChargedParticle(5.0064125184e-27, 2 * PhysicalConstants.electron_charge)

		beam_velocity = np.sqrt(2 * e / deuterium_tritium.m)
		beam_species = deuterium_tritium
		background_species = deuterium_tritium
	elif reaction_name == "pB":
		reaction = pBReaction()
		proton_boron = ChargedParticle((10.81 + 1) * UnitConversions.amu_to_kg, PhysicalConstants.electron_charge * 12)

		beam_velocity = np.sqrt(2 * e / proton_boron.m)
		beam_species = proton_boron
		background_species = proton_boron
	else:
		raise ValueError("Name is invalid!")
	collision = CoulombCollision(beam_species, background_species, 1.0, beam_velocity)
	relaxation_process = MaxwellianRelaxationProcess(collision)

	# Get thermalisation rate
	t_thermalisation_stationary, t_thermalisation_maxwellian = get_thermalisation_rate(n, T_K, relaxation_process, beam_velocity, reaction_name=reaction_name,
		                                                                               force_calculation=False)

	# Get fusion reaction rate
	t_fus = get_fusion_timescale(n, T_keV, reaction)

	fig, ax = plt.subplots(2, figsize=(10, 10))
	
	ax[0].loglog(n, t_fus, label="Fusion Reaction")
	ax[0].loglog(n, t_thermalisation_stationary, label="Stationary Thermalisation Rate")
	ax[0].loglog(n, t_thermalisation_maxwellian, label="Maxwellian Thermalisation Rate")
	
	ax[1].loglog(n, t_thermalisation_stationary / t_fus, label="Stationary Thermalisation Rate")
	ax[1].loglog(n, t_thermalisation_maxwellian / t_fus, label="Maxwellian Thermalisation Rate")

	ax[0].set_ylabel("Seconds")
	ax[0].set_xlabel("Number density")
	ax[0].set_title("Fusion Timescales")
	ax[0].legend()
	ax[1].set_ylabel("Ratio")
	ax[1].set_xlabel("Number density")
	ax[1].legend()
	ax[1].set_title("Timescales normalised by fusion reaction rate")
	ax[1].legend()
	
	plt.show()
