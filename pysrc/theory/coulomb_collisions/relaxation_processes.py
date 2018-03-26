"""
Author: Rohan Ramasamy
Date: 08/03/2018

This file contains code to model relaxation processes in plasmas due to binary coulomb collisions.
"""

import numpy as np
from scipy.integrate import tplquad

from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import CoulombCollision
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


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

    def kinetic_loss_maxwellian_frequency(self, n_background, T_background, beam_velocity,
                                           first_background=False):
        """
        Calculate the collision frequency for momentum losses in a Maxwellian background. This
        value is calculated numerically

        n_background: the density of the background species
        T_background: the temperature of the background species
        beam_velocity: speed of collision
        first_background: boolean to determine which species is the background
        :return:
        """
        m_background = self.__c.p_1.m if first_background else self.__c.p_2.m
        oned_variance = np.sqrt(PhysicalConstants.boltzmann_constant * T_background / m_background)
        v_max = 3 * oned_variance
        assert v_max < beam_velocity, "There is an asymptote at v_total=0.0. Therefore v_max / beam_velocity < 1.0. " \
                                      "The ratio is currently: {}".format(v_max / beam_velocity)

        def get_distribution_component(u, v, w):
            v_total = np.sqrt((beam_velocity - u) ** 2 + v ** 2 + w ** 2)

            # Get stationary collision frequencies
            stationary_frequency = self.kinetic_loss_stationary_frequency(n_background, T_background, v_total)

            # Get Maxwell distribution of plasma
            f_background = (m_background / (2 * np.pi * PhysicalConstants.boltzmann_constant * T_background)) ** 1.5
            f_background *= np.exp(-m_background * (u ** 2 + v ** 2 + w ** 2) / (2 * PhysicalConstants.boltzmann_constant * T_background))

            return f_background * stationary_frequency

        # Integrate 3D distribution
        v_K = tplquad(get_distribution_component, -v_max, v_max, lambda x: -v_max, lambda x: v_max, lambda x, y: -v_max,
                      lambda x, y: v_max)

        return v_K

    def momentum_loss_maxwellian_frequency(self, n_background, T_background, beam_velocity,
                                           first_background=False):
        """
        Calculate the collision frequency for momentum losses in a Maxwellian background. This
        value is calculated numerically

        n_background: the density of the background species
        T_background: the temperature of the background species
        beam_velocity: speed of collision
        first_background: boolean to determine which species is the background
        :return:
        """
        m_background = self.__c.p_1.m if first_background else self.__c.p_2.m
        oned_variance = np.sqrt(PhysicalConstants.boltzmann_constant * T_background / m_background)
        v_max = 3 * oned_variance
        assert v_max < beam_velocity, "There is an asymptote at v_total=0.0. Therefore v_max / beam_velocity < 1.0. " \
                                      "The ratio is currently: {}".format(v_max / beam_velocity)

        def get_distribution_component(u, v, w):
            v_total = np.sqrt((beam_velocity - u) ** 2 + v ** 2 + w ** 2)

            # Get stationary collision frequencies
            stationary_frequency = self.momentum_loss_stationary_frequency(n_background, T_background, v_total,
                                                                         first_background)

            # Get Maxwell distribution of plasma
            f_background = (m_background / (2 * np.pi * PhysicalConstants.boltzmann_constant * T_background)) ** 1.5
            f_background *= np.exp(-m_background * (u ** 2 + v ** 2 + w ** 2) / (2 * PhysicalConstants.boltzmann_constant * T_background))

            return f_background * stationary_frequency

        # Integrate 3D distribution
        v_P = tplquad(get_distribution_component, -v_max, v_max, lambda x: -v_max, lambda x: v_max, lambda x, y: -v_max,
                      lambda x, y: v_max)

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


if __name__ == '__main__':
    pass
