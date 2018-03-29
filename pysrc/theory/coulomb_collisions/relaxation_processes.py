"""
Author: Rohan Ramasamy
Date: 08/03/2018

This file contains code to model relaxation processes in plasmas due to binary coulomb collisions.
"""

import numpy as np

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
    electron_mass = PhysicalConstants.electron_mass
    deuterium_mass = 2.01410178 * UnitConversions.amu_to_kg
    deuterium_tritium_mass = 5.0064125184e-27
    alpha_mass = 3.7273 * UnitConversions.amu_to_kg

    beam_species = ChargedParticle(deuterium_tritium_mass, 5 * PhysicalConstants.electron_charge)
    background_species = ChargedParticle(electron_mass, -PhysicalConstants.electron_charge)
    n_background = 1e30
    e_beam = 0.5e6 * PhysicalConstants.electron_charge
    beam_velocity = np.sqrt(2 * e_beam / beam_species.m)
    temp = 20e3 * UnitConversions.eV_to_K

    collision = CoulombCollision(background_species, beam_species, 1.0, beam_velocity)
    relaxation_process = RelaxationProcess(collision)

    kinetic_frequency = relaxation_process.kinetic_loss_stationary_frequency(n_background, temp, beam_velocity)
    momentum_frequency = relaxation_process.momentum_loss_stationary_frequency(n_background, temp, beam_velocity,
                                                                               first_background=True)

    print("Kinetic Relaxation Time: {}".format(1 / kinetic_frequency))
    print("Momentum Relaxation Time: {}".format(1 / momentum_frequency))
