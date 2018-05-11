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


def get_fusion_timescale(n, T, reaction):
	"""
	Function to get the reaction time scale of a given fusion reaction

	n: number density of the plasma (in particles per m3)
	T: temperature of the plasma (in keV)
	reaction: 
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


if __name__ == '__main__':
	reaction = DDReaction()
	n = np.logspace(20, 30, 10)
	T = 10.0

	t_fus = get_fusion_timescale(n, T, reaction)

	plt.figure()
	plt.loglog(n, t_fus, label="Fusion Reaction")
	
	plt.ylabel("Seconds")
	plt.xlabel("Number density")
	plt.legend()
	plt.title("Fusion Timescales")
	plt.show()

