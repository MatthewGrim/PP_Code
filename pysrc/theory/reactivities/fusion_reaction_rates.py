"""
Author: Rohan Ramasamy
Date: 26/12/2017

This file contains code to generate the predicted nuclear reaction rates for a given density and temperature of fusion
reactants - this is used to gain an understanding of where in the reaction space we need to get to for a given fusion
fuel
"""

import numpy as np
import matplotlib.pyplot as plt

from plasma_physics.pysrc.theory.reactivities.fusion_reactivities import BoschHaleReactivityFit, DDReaction, DTReaction
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


class ReactionRatesCalculator(object):
    def __init__(self, fusion_reaction):
        """
        Initialise class with fusion reaction
        """
        self.fusion_reaction = fusion_reaction()

    def get_reaction_rates(self, T, rho):
        assert isinstance(T, np.ndarray)
        assert len(T.shape) == 1
        assert len(rho.shape) == 1

        reactivity_fit = BoschHaleReactivityFit(self.fusion_reaction)
        TEMP, RHO = np.meshgrid(T, rho, indexing='ij')
        reactivities = reactivity_fit.get_reactivity(TEMP)
        n_1_n_2 = self.fusion_reaction.number_density(RHO)

        reaction_rate = n_1_n_2 * reactivities

        return TEMP, RHO, reaction_rate


if __name__ == '__main__':
    T = np.logspace(0, 3, 200)
    rho = np.logspace(-5, 0, 200)

    # Generate calculators
    dd_reaction_rate_calculator = ReactionRatesCalculator(DDReaction)
    dt_reaction_rate_calculator = ReactionRatesCalculator(DTReaction)

    # Calculate reaction rates
    TEMP, RHO, dd_reaction_rate = dd_reaction_rate_calculator.get_reaction_rates(T, rho)
    TEMP, RHO, dt_reaction_rate = dt_reaction_rate_calculator.get_reaction_rates(T, rho)

    # Calculate power produced per metre cubed
    MW_conversion = 1e-6
    dd_energy_released = 7.3e6 * PhysicalConstants.electron_charge
    dt_energy_released = 17.59e6 * PhysicalConstants.electron_charge
    P_dd = dd_reaction_rate * dd_energy_released * MW_conversion
    P_dt = dd_reaction_rate * dt_energy_released * MW_conversion

    # Get number densities
    N_dd = np.sqrt(2 * DDReaction().number_density(RHO))
    N_dt = np.sqrt(DTReaction().number_density(RHO))

    fig, ax = plt.subplots(2, 2)

    # Plot Reaction Rate
    im = ax[0, 0].contourf(np.log10(TEMP), np.log10(N_dd), np.log10(dd_reaction_rate), 100)
    fig.colorbar(im, ax=ax[0, 0])
    ax[0, 0].set_title("DD Reaction Rate (reactions per second)")
    im = ax[0, 1].contourf(np.log10(TEMP), np.log10(N_dt), np.log10(dt_reaction_rate), 100)
    fig.colorbar(im, ax=ax[0, 1])
    ax[0, 1].set_title("DT Reaction Rate (reactions per second)")

    # Plot Power Production
    im = ax[1, 0].contourf(np.log10(TEMP), np.log10(N_dd), np.log10(P_dd), 100)
    fig.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_title("DD Power Produced (MWm-3)")
    im = ax[1, 1].contourf(np.log10(TEMP), np.log10(N_dt), np.log10(P_dt), 100)
    fig.colorbar(im, ax=ax[1, 1])
    ax[1, 1].set_title("DT Power Produced (MWm-3)")

    fig.suptitle("Plots of fusion reactivity and power for different fusion reactions")
    plt.show()