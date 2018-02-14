"""
Author: Rohan Ramasamy
Date: 13/02/2018

The model for the absorption of energy by high energy particles in a plasma from:

"Thermonuclear burn characteristics of compressed deuterium-tritium microspheres" - G. S. Fraley,
	E. J. Linnebur, R. J. Mason, and R. L. Morse

is implemented, and explored in this script.
"""

import numpy as np


class FLMMmodel(object):
	"""
	Class to implement the equations in:

	"Thermonuclear burn characteristics of compressed deuterium-tritium microspheres" - G. S. Fraley,
		E. J. Linnebur, R. J. Mason, and R. L. Morse
	
	for the energy deposition of a high particle in an ambient plasma.
	"""
	@staticmethod
	def electron_stopping_distance(temperature, density, solid_density):
		"""
		temperature: the temperature of the ambient plasma
		density: density of the ambient plasma
		solid_density: reference solid density of the plasma
		"""
		A = 0.086 * solid_density * (1.0 + 0.17 * np.log(temperature * np.sqrt(solid_density / density)) ** (1.0 / 2.0))
		electron_stopping_distance = A * temperature ** (3.0 / 2.0) / density

		return electron_stopping_distance

	@staticmethod
	def ion_stopping_distance(temperature, density, solid_density):
		"""
		temperature: the temperature of the ambient plasma
		density: density of the ambient plasma
		solid_density: reference solid density of the plasma
		"""
		B = 10.65 * solid_density * (1.0 + 0.075 * np.log((temperature * (solid_density / density)) ** 0.5)) ** (-1.0)
		ion_stopping_distance = B / density

		return ion_stopping_distance

	@staticmethod
	def energy_range_relationship(temperature, density, solid_density, particle_energy):
		"""
		temperature: the temperature of the ambient plasma
		density: density of the ambient plasma
		solid_density: reference solid density of the plasma
		particle_energy: energy of the particle being thermalised
		"""
		electron_absorption = 1.0 + 0.17 * np.log(temperature * (solid_density / density) ** 0.5)
		electron_absorption *= particle_energy ** 0.5 / temperature ** 1.5 * (density / solid_density)
		electron_absorption *= -23.2

		ion_absorption = 1.0 + 0.075 * np.log(temperature ** 0.5 * (solid_density / density) ** 0.5 * particle_energy ** 0.5)
		ion_absorption *= (density / solid_density) / particle_energy
		ion_absorption *= -0.047

		return electron_absorption + ion_absorption

	@staticmethod
	def get_U_x(temperature, density, solid_density):
		"""
		Particle energy is normalised by the alpha particle energy

		temperature: the temperature of the ambient plasma
		density: density of the ambient plasma
		solid_density: reference solid density of the plasma
		"""
		U = 1.0
		x = 0.0
		U_val = [U]
		x_val = [x]

		dx = 0.0
		dU = -0.001

		while U > 0.0:
			du_dx = FLMMmodel.energy_range_relationship(temperature, density, solid_density, U)

			dx = dU / du_dx
			U += dU
			x += dx

			U_val.append(U)
			x_val.append(x)

		x_val = np.asarray(x_val)
		U_val = np.asarray(U_val)

		return x_val, U_val


if __name__ == '__main__':
	pass
