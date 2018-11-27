"""
Author: Rohan Ramasamy
Date: 27/11/2018

This file contains code to validate whether the power scaling being predicted is consistent with values in the literature.
"""

import numpy as np
import matplotlib.pyplot as plt

from plasma_physics.pysrc.theory.reactivities.fusion_reactivities import BoschHaleReactivityFit, DDReaction, DTReaction, FusionReaction
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants
from plasma_physics.pysrc.theory.reactivities.fusion_reaction_rates import ReactionRatesCalculator


def validate_against_park_2014():
    # energy - In keV
    energy = 30.0
    # number density per cm3
    n = 2e15

    # Calculate reaction rates - we are using Bosch Hale reactivities which Nevins original work stated were not far from beam
    # reactivities
    reactivity_calculator = BoschHaleReactivityFit(DTReaction())
    reactivity = reactivity_calculator.get_reactivity(energy)
    n_1_n_2 = (n / 2) ** 2
    # Number of reactions per second
    number_of_reactions = n_1_n_2 * reactivity

    # Calculate power produced per metre cubed
    m3_conversion = 1e6
    GW_conversion = 1e-9
    dt_energy_released = 17.59e6 * PhysicalConstants.electron_charge * number_of_reactions * m3_conversion * GW_conversion
    print("Fusion Power: {} GWm^-3".format(dt_energy_released))


if __name__ == '__main__':
    validate_against_park_2014()

