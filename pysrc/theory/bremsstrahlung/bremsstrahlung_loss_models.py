"""
Author: Rohan Ramasamy
Date: 05/07/2018

This file contains models of Bremsstrahlung radiation losses in different plasmas
"""

import numpy as np
from matplotlib import pyplot as plt

from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


class MaxonBremsstrahlungRadiation(object):
	def __init__(self):
		raise NotImplementedError("This class acts as a static library!")

	@staticmethod
	def power_loss_density(n_e, T_e, Z_eff=1.0):
		"""
		Calculate the power loss density in Wm-3

		n_e: number density (m-3)
		T_e: electron temperature (in eV)
		Z_eff: Effective ionisation
		"""
		c = 3e8
		m_c_2 = PhysicalConstants.electron_mass * c ** 2 / PhysicalConstants.electron_charge
		T_m_c_2 = T_e / m_c_2
		Z_eff = 1.0

		p = 1.0 + 0.7936 * T_m_c_2 + 1.874 * T_m_c_2 ** 2
		p *= Z_eff
		p += 3.0 / np.sqrt(2) * T_m_c_2
		p *= 1.69e-32 * n_e ** 2 * np.sqrt(T_e)

		return p


if __name__ == '__main__':
	n_e = np.logspace(18, 25, 100)
	T_e = np.logspace(1, 5, 100)

	T_E, N_E = np.meshgrid(T_e, n_e, indexing='ij')
	P_BREM = MaxonBremsstrahlungRadiation.power_loss_density(N_E, T_E)

	# Convert bremsstrahlung radiation loss to megawatts
	P_BREM *= 1e-6

	fig, ax = plt.subplots()

	im = ax.contourf(np.log10(T_E), np.log10(N_E), np.log10(P_BREM), 100)
	fig.colorbar(im, ax=ax)
	ax.set_ylabel("Number density (m-3)")
	ax.set_xlabel("Temperature (eV)")

	fig.suptitle("Bremsstrahlung radiation loss (MWm-3)")
	plt.show()
