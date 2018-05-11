"""
Author: Rohan Ramasamy
Date: 11/05/2018

This script contains code to compare the fusion reaction rate with the thermalisation rate of a plasma. This can be 
used to define whether it is possible to sustain a non-Maxwellian plasma within a fusion device.
"""

import numpy as np
import matplotlib.pyplot as plt

from plasma_physics.pysrc.theory.reactivities.fusion_reactivities import BoschHaleReactivityFit, FusionReaction, DDReaction, DTReaction
from plasma_physics.pysrc.utils.unit_conversions import UnitConversions
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants
from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import ChargedParticle, CoulombCollision
from plasma_physics.pysrc.theory.coulomb_collisions.relaxation_processes import RelaxationProcess


def get_fusion_timescale(n, T, reaction):
	"""
	Function to get the reaction time scale of a given fusion reaction

	n: number density of the plasma (in particles per m3)
	T: temperature of the plasma (in keV)
	reaction: Fusion Reaction to get reactivity of
	"""
	assert isinstance(n, float) or isinstance(n, np.ndarray)
	assert isinstance(T, float) or isinstance(T, np.ndarray)
	assert isinstance(reaction, FusionReaction)

	# Use fit to get reactivity - Bosch Hale fit assumes a Maxwellian plasma
	reactivity_fit = BoschHaleReactivityFit(reaction)
	reactivity = reactivity_fit.get_reactivity(T)

	# Get the reaction rate and invert to get the timescale
	reaction_rate = n * reactivity 
	return 1 / reaction_rate


def get_thermalisation_rate(n, T, relaxation_process, beam_velocity):
	"""
	Get the thermalisation rate of a given fusion reaction

	n: number density of the plasma (in particles per m3)
	T: temperature of the plasma (in keV)
	relaxation_process: realxation process corresponding to the chosen fusion reaction
	"""
	assert isinstance(n, float) or isinstance(n, np.ndarray)
	assert isinstance(T, float) or isinstance(T, np.ndarray)
	assert isinstance(relaxation_process, RelaxationProcess)

	stationary_frequency =  1.0 / relaxation_process.kinetic_loss_stationary_frequency(n, T, beam_velocity)
	
	return stationary_frequency


if __name__ == '__main__':
	# Define parameter space of scan
	n = np.logspace(20, 30, 10)
	T = 10.0
	e = 1.5 * PhysicalConstants.boltzmann_constant * T * 1e3 * UnitConversions.eV_to_K
	print(e)

	# Set up reaction
	use_DD = True
	if use_DD:
		reaction = DDReaction()
		deuterium = ChargedParticle(2.01410178 * UnitConversions.amu_to_kg, PhysicalConstants.electron_charge)

		beam_velocity = np.sqrt(2 * e / deuterium.m)	
		beam_species = deuterium
		background_species = deuterium
	else:
		reaction = DTReaction()
		deuterium_tritium = ChargedParticle(5.0064125184e-27 * UnitConversions.amu_to_kg, 2 * PhysicalConstants.electron_charge)

		beam_velocity = np.sqrt(2 * e / deuterium_tritium.m)
		beam_species = deuterium_tritium
		background_species = deuterium_tritium
	collision = CoulombCollision(beam_species, background_species, 1.0, beam_velocity)
	relaxation_process = RelaxationProcess(collision)


	# Get thermalisation rate
	t_thermalisation_stationary = get_thermalisation_rate(n, T, relaxation_process, beam_velocity)

	# Get fusion reaction rate
	t_fus = get_fusion_timescale(n, T, reaction)

	fig, ax = plt.subplots(2, figsize=(10, 10))
	
	ax[0].loglog(n, t_fus, label="Fusion Reaction")
	ax[0].loglog(n, t_thermalisation_stationary, label="Stationary Thermalisation Rate")
	
	ax[1].plot(n, t_thermalisation_stationary / t_fus, label="Stationary Thermalisation Rate")

	ax[0].set_ylabel("Seconds")
	ax[0].set_xlabel("Number density")
	ax[0].set_title("Fusion Timescales")
	ax[0].legend()
	ax[1].set_ylabel("Ratio")
	ax[1].set_xlabel("Number density")
	ax[1].set_title("Timescales normalised by fusion reaction rate")
	ax[1].legend()
	
	plt.show()

