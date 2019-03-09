"""
Author: Rohan Ramasamy
Date: 08/03/2019

This file contains functions to solve Bennett equilibria. Given a current profile, the magnetic field and pressure 
profiles are calculated. The poloidal flux could equally be used as an input, but because this code was written 
to consider current driven processes, like in a fast Z pinch, the current was chosen as the input.
"""


import numpy as np
import scipy.integrate as integrate
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
		poloidal_field = integrate.cumtrapz(poloidal_integrand, radial_points)
		radial_points = radial_points[1:]
		current_density = current_density[1:]
		poloidal_field /= radial_points

		# Integrate pressure from momentum equation
		pressure_integrand = -current_density * poloidal_field
		pressure = integrate.cumtrapz(pressure_integrand, radial_points)
		# Apply boundary condition that at plasma boundary, pressure is zero
		pressure -= pressure[-1]
		radial_points = radial_points[1:]
		current_density = current_density[1:]
		poloidal_field = poloidal_field[1:]

		beta_integrand = pressure
		beta = integrate.trapz(beta_integrand, radial_points) / (np.pi * radial_points[-1] ** 2)
		beta /= poloidal_field[-1] ** 2 / (2 * PhysicalConstants.mu_0)
		print('Beta: {}'.format(beta))

		return radial_points, current_density, poloidal_field, pressure

if __name__ == '__main__':
	r_min = 0.0
	r_max = 1.0
	num_pts = 1000
	J_analytic = 1.0
	beta_analytic = 1.0
	radial_points = np.linspace(r_min, r_max, num_pts)
	current_density = np.ones(radial_points.shape) * J_analytic

	# Get analytic solutions for constant current density
	B_analytic = PhysicalConstants.mu_0 * J_analytic * radial_points / 2.0
	p_analytic = PhysicalConstants.mu_0 * J_analytic ** 2 * (1 - radial_points ** 2 / r_max ** 2) / 4.0

	r, J_z, B, p = BennetEquilibriumCalculator.calculate_poloidal_field_and_pressure(radial_points, current_density)

	# Plot results
	fig, ax = plt.subplots(3, figsize=(10, 10), sharex=True)

	ax[0].plot(r, J_z)
	ax[0].axhline(J_analytic, linestyle='--')
	ax[0].set_ylabel('$J_z$ [$Am^{-2}$]')

	ax[1].plot(r, B)
	ax[1].plot(radial_points, B_analytic, linestyle='--')
	ax[1].set_ylabel('$B_T$ [$T$]')
	
	ax[2].plot(r, p)
	ax[2].plot(radial_points, p_analytic, linestyle='--')
	ax[2].set_ylabel('$P$ [$Pa$]')
	ax[2].set_xlim([r_min, r_max])

	plt.show()