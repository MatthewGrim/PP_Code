"""
Author: Rohan Ramasamy
Date: 08/03/2019

This file contains functions to solve Bennett equilibria. Given a current profile, the magnetic field and pressure 
profiles are calculated. The poloidal flux could equally be used as an input, but because this code was written 
to consider current driven processes, like in a fast Z pinch, the current was chosen as the input.
"""


import numpy as np
import matplotlib.pyplot as plt

from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


class BennetEquilibriumCalculator(object):
	def __init__(self):
		raise RuntimeError('This is a static class')

	@staticmethod
	def calculate_poloidal_field_and_pressure(radial_points, current_density):
		"""
		radial_points [m]: radial positions of current samples
		current_density [Am-2]: radial profile of current density
		"""
		# Integrate magnetic field from Ampere's law
		poloidal_integrand = PhysicalConstants.mu_0 * current_density * radial_points
		poloidal_field = np.zeros(radial_points.shape[0] - 1)
		for i in range(1, radial_points.shape[0]):
			poloidal_field[i-1] = np.trapz(poloidal_integrand[:i], radial_points[:i])
		poloidal_radial_points = (radial_points[:-1] + radial_points[1:]) / 2
		new_current_points = (current_density[:-1] + current_density[1:]) / 2
		poloidal_field /= poloidal_radial_points

		# Integrate pressure from momentum equation
		pressure_integrand = -new_current_points * poloidal_field
		pressure = np.zeros(poloidal_radial_points.shape[0] - 1)
		for i in range(1, poloidal_radial_points.shape[0]):
			pressure[i-1] = np.trapz(pressure_integrand[:i], poloidal_radial_points[:i])
		# Apply boundary condition that at plasma boundary, pressure is zero
		pressure -= pressure[-1]

		# Correct point positions to be consistent
		final_radial_points = (poloidal_radial_points[:-1] + poloidal_radial_points[1:]) / 2
		final_current_points = (new_current_points[:-1] + new_current_points[1:]) / 2
		final_poloidal_field_points = (poloidal_field[:-1] + poloidal_field[1:]) / 2

		return final_radial_points, final_current_points, final_poloidal_field_points, pressure

if __name__ == '__main__':
	r_min = 0.0
	r_max = 1.0
	num_pts = 100
	J_analytic = 1.0

	radial_points = np.linspace(r_min, r_max, num_pts)
	current_density = np.ones(radial_points.shape) * J_analytic

	B_analytic = PhysicalConstants.mu_0 * J_analytic * radial_points / 2.0
	p_analytic = PhysicalConstants.mu_0 * J_analytic ** 2 * (1 - radial_points ** 2 / r_max ** 2) / 4.0

	r, J_z, B, p = BennetEquilibriumCalculator.calculate_poloidal_field_and_pressure(radial_points, current_density)

	fig, ax = plt.subplots(3)

	ax[0].plot(r, J_z)
	ax[0].axhline(J_analytic, linestyle='--')
	
	ax[1].plot(r, B)
	ax[1].plot(radial_points, B_analytic, linestyle='--')
	
	ax[2].plot(r, p)
	ax[2].plot(radial_points, p_analytic, linestyle='--')

	plt.show()