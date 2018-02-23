"""
Author: Rohan Ramasamy
Date: 23/02/2018

This script is used to recreate the Fraley curve used in Leo Basov's thesis, Figure 6.14:

"Kinetic investigation of the thermalization of highly energetic fusion products" - L. Basov, 2016
"""

import numpy as np
import matplotlib.pyplot as plt

from plasma_physics.pysrc.theory.thermalisation.flmm_model import FLMMmodel


def plot_basov_thermalisation_test():
	"""
	Take parameters from section 6.4 of Basov's thesis and regenerate the same curve using Fraley
	"""
	solid_density = 0.213
	m3_to_cm3 = 1e-6
	M_dt = 5.0301510601
	avogadro_constant = 6.023e23
	plasma_density = 1e30 / avogadro_constant * M_dt * m3_to_cm3
	temperatures = np.linspace(0.0, 200.0, 50)

	# Numerical integration for more exact value
	final_electron_energy = np.zeros(temperatures.shape)
	final_ion_energy = np.zeros(temperatures.shape)
	for j, T in enumerate(temperatures):
		x, U, U_ele, U_ion = FLMMmodel.get_U_x(T, plasma_density, solid_density)
		final_electron_energy[j] = U_ele[-1]
		final_ion_energy[j] = U_ion[-1]

	basov_data = np.loadtxt("data/basov_curve.txt")

	print("Plasma Density: {}gcm-3".format(plasma_density))

	plt.figure()

	plt.plot(temperatures, final_ion_energy)
	plt.plot(basov_data[:, 0], basov_data[:, 1])
	plt.axhline(0.8, linestyle='--')
	plt.axhline(0.9, linestyle='--')

	plt.xlabel("Temperature (keV)")
	plt.xlabel("U_ion / U")
	plt.title("Comparison of Current Fraley implementation with Basov thesis")
	plt.show()


if __name__ == '__main__':
	plot_basov_thermalisation_test()