"""
Author: Rohan Ramasamy
Date: 08/03/2018

This file contains code to model relaxation processes in plasmas due to binary coulomb collisions.
"""

import numpy as np
from scipy import integrate

from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import ChargedParticle, CoulombCollision
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants
from plasma_physics.pysrc.utils.unit_conversions import UnitConversions


class RelaxationProcess(object):
    def __init__(self, collision):
        """
        Initialise the class with the collision being modelled. This will
        allow the kinetic and momentum loss frequencies to be calculated
         for stationary particles

        collision: the binary collision being modelled
        """
        assert isinstance(collision, CoulombCollision)

        self.__c = collision

    def kinetic_loss_stationary_frequency(self, n_background, T_background, beam_velocity):
        """
        Calculate the collision frequency for kinetic losses in stationary
        background

        n_background: the density of the background species
        T_background: the temperature of the background species
        beam_velocity: speed of collision
        """
        q_1 = self.__c.p_1.q
        m_1 = self.__c.p_1.m
        q_2 = self.__c.p_2.q
        m_2 = self.__c.p_2.m
        b_90 = self.__c.b_90

        debye_length = PhysicalConstants.epsilon_0 * T_background
        debye_length /= n_background * PhysicalConstants.electron_charge ** 2
        debye_length = np.sqrt(debye_length)

        coulomb_logarithm = np.log(debye_length / b_90)

        v_K = n_background * q_1 ** 2 * q_2 ** 2 / (4.0 * np.pi * PhysicalConstants.epsilon_0) ** 2
        v_K *= 8.0 * np.pi / (m_1 * m_2 * beam_velocity ** 3)
        v_K *= coulomb_logarithm

        return v_K

    def momentum_loss_stationary_frequency(self, n_background, T_background, beam_velocity,
                                           first_background=False):
        """
        Calculate the collision frequency for momentum losses in stationary background

        n_background: the density of the background species
        T_background: the temperature of the background species
        beam_velocity: speed of collision
        first_background: boolean to determine which species is the background
        """
        v_K = self.kinetic_loss_stationary_frequency(n_background, T_background, beam_velocity)

        m_1 = self.__c.p_1.m
        m_2 = self.__c.p_2.m

        conversion_factor = (m_1 + m_2) / (2 * m_2) if first_background else (m_1 + m_2) / (2 * m_1)

        v_P = v_K * conversion_factor

        return v_P

    def maxwellian_collisional_frequency(self, n_background, T_background, beam_velocity, first_background=False):
        """
        Calculate an approximate maxwellian collisional frequency - this approximation is valid when
        the electron thermal velocity is much greater than the beam velocity

        n_background: the density of the background species
        T_background: the temperature of the background species
        first_background: boolean to determine which species is the background
        """
        m_background = self.__c.p_1.m if first_background else self.__c.p_2.m
        m_beam = self.__c.p_2.m if first_background else self.__c.p_1.m
        v_thermal = np.sqrt(PhysicalConstants.boltzmann_constant * T_background / m_background)

        v_ratio = v_thermal / beam_velocity
        if v_ratio < 100.0:
            print("Warning! Assumption that thermal velocty is greater than ion velocity may be invalid")

        # Get relevant particle parameters
        q_1 = self.__c.p_1.q
        q_2 = self.__c.p_2.q
        b_90 = self.__c.b_90

        # Get coulomb logarithm
        debye_length = PhysicalConstants.epsilon_0 * T_background
        debye_length /= n_background * PhysicalConstants.electron_charge ** 2
        debye_length = np.sqrt(debye_length)
        coulomb_logarithm = np.log(debye_length / b_90)

        v = 2 / (3 * np.sqrt(2 * np.pi)) * n_background * q_1 ** 2 * q_2 ** 2 / (4.0 * np.pi * PhysicalConstants.epsilon_0) ** 2
        v *= 4.0 * np.pi * (m_beam + m_background) / (m_background * m_beam ** 2 * v_thermal ** 3)
        v *= coulomb_logarithm

        return v

    def numerical_kinetic_loss_maxwellian_frequency(self, n_background, T_background, beam_velocity,
                                                    first_background=False, epsrel=1e-3):
        """
        Calculate the collision frequency for momentum losses in a Maxwellian background. This
        value is calculated numerically

        n_background: the density of the background species
        T_background: the temperature of the background species
        beam_velocity: speed of collision
        first_background: boolean to determine which species is the background
        epsabs: absolute error in integration
        :return:
        """
        m_background = self.__c.p_1.m if first_background else self.__c.p_2.m
        oned_variance = np.sqrt(PhysicalConstants.boltzmann_constant * T_background / m_background)
        v_max = 3 * oned_variance

        max_freq = 1e12

        def get_distribution_component(u, v, w):
            v_total = np.sqrt((beam_velocity - u) ** 2 + v ** 2 + w ** 2)

            # Get stationary collision frequencies
            stationary_frequency = self.kinetic_loss_stationary_frequency(n_background, T_background, v_total)
            stationary_frequency = max_freq if stationary_frequency > max_freq else stationary_frequency

            # Get Maxwell distribution of plasma
            f_background = (m_background / (2 * np.pi * PhysicalConstants.boltzmann_constant * T_background)) ** 1.5
            f_background *= np.exp(-m_background * (u ** 2 + v ** 2 + w ** 2) / (2 * PhysicalConstants.boltzmann_constant * T_background))

            return f_background * stationary_frequency

        # Integrate 3D distribution
        v_K = integrate.tplquad(get_distribution_component, -v_max, v_max, lambda x: -v_max, lambda x: v_max, lambda x, y: -v_max,
                                lambda x, y: v_max, epsrel=epsrel)

        return v_K

    def numerical_momentum_loss_maxwellian_frequency(self, n_background, T_background, beam_velocity,
                                                     first_background=False, epsrel=1e-3):
        """
        Calculate the collision frequency for momentum losses in a Maxwellian background. This
        value is calculated numerically

        n_background: the density of the background species
        T_background: the temperature of the background species
        beam_velocity: speed of collision
        first_background: boolean to determine which species is the background
        epsabs: absolute error in integration
        :return:
        """
        m_background = self.__c.p_1.m if first_background else self.__c.p_2.m
        oned_variance = np.sqrt(PhysicalConstants.boltzmann_constant * T_background / m_background)
        v_max = 3 * oned_variance

        max_freq = 1e12

        def get_distribution_component(u, v, w):
            v_total = np.sqrt((beam_velocity - u) ** 2 + v ** 2 + w ** 2)

            # Get stationary collision frequencies
            stationary_frequency = self.momentum_loss_stationary_frequency(n_background, T_background, v_total,
                                                                           first_background)
            stationary_frequency = max_freq if stationary_frequency > max_freq else stationary_frequency

            # Get Maxwell distribution of plasma
            f_background = (m_background / (2 * np.pi * PhysicalConstants.boltzmann_constant * T_background)) ** 1.5
            f_background *= np.exp(-m_background * (u ** 2 + v ** 2 + w ** 2) / (2 * PhysicalConstants.boltzmann_constant * T_background))

            return f_background * stationary_frequency

        # Integrate 3D distribution
        v_P = integrate.tplquad(get_distribution_component, -v_max, v_max, lambda x: -v_max, lambda x: v_max, lambda x, y: -v_max,
                                lambda x, y: v_max, epsrel=epsrel)

        return v_P

    def energy_loss_with_distance(self, n_background, T_background, beam_velocity,
                                  first_background=False):
        """
        Calculate energy loss rate with distance

        n_background: the density of the background species
        T_background: the temperature of the background species
        beam_velocity: speed of collision
        first_background: boolean to determine which species is the background
        """
        v_K = self.kinetic_loss_stationary_frequency(n_background, T_background, beam_velocity)

        m_1 = self.__c.p_1.m
        m_2 = self.__c.p_2.m

        conversion_factor = m_2 if first_background else m_1
        conversion_factor *= 0.5 * beam_velocity

        dK_dL = v_K * conversion_factor

        return dK_dL


def get_iec_frequencies(use_alpha, temperature, use_maxwellian=False):
    electron_mass = PhysicalConstants.electron_mass
    deuterium_mass = 2.01410178 * UnitConversions.amu_to_kg
    deuterium_tritium_mass = 5.0064125184e-27
    alpha_mass = 3.7273 * UnitConversions.amu_to_kg

    alpha_species = ChargedParticle(alpha_mass, 2 * PhysicalConstants.electron_charge)
    dt_species = ChargedParticle(deuterium_tritium_mass, 5 * PhysicalConstants.electron_charge)
    background_species = ChargedParticle(electron_mass, -PhysicalConstants.electron_charge)
    n_background = 1e20
    dt_velocity = np.sqrt(2 * 50e3 * PhysicalConstants.electron_charge / dt_species.m)
    alpha_velocity = np.sqrt(2 * 3.5e6 * PhysicalConstants.electron_charge / alpha_species.m)

    beam_species = alpha_species if use_alpha else dt_species
    beam_velocity = alpha_velocity if use_alpha else dt_velocity

    collision = CoulombCollision(beam_species, background_species, 1.0, beam_velocity)
    relaxation_process = RelaxationProcess(collision)

    if use_maxwellian:
        v_K = relaxation_process.numerical_kinetic_loss_maxwellian_frequency(n_background, temperature, beam_velocity)
        v_P = relaxation_process.numerical_kinetic_loss_maxwellian_frequency(n_background, temperature, beam_velocity)
        return v_K[0], v_P[0]

        # maxwellian_frequency = relaxation_process.maxwellian_collisional_frequency(n_background, temperature, beam_velocity)
        # return maxwellian_frequency, maxwellian_frequency
    else:
        stationary_kinetic_frequency = relaxation_process.kinetic_loss_stationary_frequency(n_background, temperature, beam_velocity)
        stationary_momentum_frequency = relaxation_process.momentum_loss_stationary_frequency(n_background, temperature, beam_velocity)
        return stationary_kinetic_frequency, stationary_momentum_frequency


if __name__ == '__main__':
    use_alpha = True
    T = 10
    kinetic_frequency, momentum_frequency = get_iec_frequencies(use_alpha, T)

    print("Kinetic Relaxation Time: {}".format(1 / kinetic_frequency))
    print("Momentum Relaxation Time: {}".format(1 / momentum_frequency))
