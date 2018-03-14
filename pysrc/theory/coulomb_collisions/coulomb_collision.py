"""
Author: Rohan Ramasamy
Date: 06/03/2018

This file contains code to model binary coulomb interactions, calculating
the scattering angle and terminal velocity for a given impact parameter
and charge particle pair
"""

import numpy as np

from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


class ChargedParticle(object):
    def __init__(self, mass, charge):
        """
        Initialise a charged particle with its given mass and charge
        """
        self.__m = mass
        self.__q = charge

    @property
    def m(self):
        return self.__m

    @property
    def q(self):
        return self.__q


class CoulombCollision(object):
    def __init__(self, particle_1, particle_2, impact_parameter_ratio, velocity):
        """
        Initialise class with impacting particles and the initial conditions of the collision

        particle_1: First charged particle
        particle_2: Second charged particle (modelled as stationary)
        impact_parameter_ratio: distance between two particles (as a ratio from b_90)
        velocity: velocity of particle_1 as observed from particle_2
        """
        assert isinstance(particle_1, ChargedParticle)
        assert isinstance(particle_2, ChargedParticle)
        assert isinstance(impact_parameter_ratio, float) or isinstance(impact_parameter_ratio, np.ndarray)
        assert isinstance(velocity, float) or isinstance(velocity, np.ndarray)

        self.__p_1 = particle_1
        self.__p_2 = particle_2

        # Calculate the effective mass of the particle in the reference frame of particle 2
        m_1 = particle_1.m
        m_2 = particle_2.m
        effective_mass = m_1 * m_2 / (m_1 + m_2)

        theta = np.arctan(-impact_parameter_ratio)

        q_1 = np.abs(particle_1.q)
        q_2 = np.abs(particle_2.q)
        self.__b_90 = q_1 * q_2 / (4.0 * np.pi * PhysicalConstants.epsilon_0)
        self.__b_90 /= effective_mass * velocity ** 2

        self.__theta = theta
        self.__chi = np.pi + 2 * theta

        self.__velocity = velocity
        v_x = velocity * np.cos(self.__chi)
        v_y = velocity * np.sin(self.__chi)
        self.__v_final = np.asarray([v_x, v_y])

        self.__differential_cross_section = self.__b_90 ** 2 / (4 * np.sin(self.__chi) ** 4.0)

    @property
    def p_1(self):
        return self.__p_1

    @property
    def p_2(self):
        return self.__p_2

    @property
    def b_90(self):
        return self.__b_90

    @property
    def chi(self):
        return self.__chi

    @property
    def v_final(self):
        return self.__v_final

    @property
    def differential_cross_section(self):
        return self.__differential_cross_section


if __name__ == '__main__':
    p_1 = ChargedParticle(1.0, 2.0)
    p_2 = ChargedParticle(1.0, 2.0)

    c = CoulombCollision(p_1, p_2, 1.0, 1.0)
