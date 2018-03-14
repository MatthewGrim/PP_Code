"""
Author: Rohan Ramasamy
Date: 08/03/2018

This file contains code to model relaxation processes in plasmas due to binary coulomb collisions. 
"""

import numpy as np

from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import CoulombCollision, ChargedParticle
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants
from plasma_physics.pysrc.utils.unit_conversions import UnitConversions


class RelaxationProcess(object):
    def __init__(self, collision):
        """
        Initialise the class with the collision being modelled. This will allow the
        kinetic and momentum loss frequencies to be calculated for stationary particles

        collision: the binary collision being modelled
        """
        assert isinstance(collision, CoulombCollision)

        self.__c = collision

    def kinetic_loss_stationary_frequency(self, n_background, T_background, beam_velocity):
        """
        Calculate the collision frequency for kinetic losses in stationary background

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

        coulomb_logarithm = debye_length / b_90 

        v_K = n_background * q_1 ** 2 * q_2 ** 2 / (4.0 * np.pi * PhysicalConstants.epsilon_0) ** 2
        v_K *= 8.0 * np.pi / (m_1 * m_2 * beam_velocity ** 3)
        v_K *= debye_length

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

        conversion_factor = m_1 if first_background else m_2
        conversion_factor *= 0.5 * beam_velocity ** 3

        v_P = v_K * conversion_factor

        return v_P


if __name__ == '__main__':
    pass
