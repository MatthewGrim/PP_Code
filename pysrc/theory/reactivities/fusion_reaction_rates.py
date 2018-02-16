"""
Author: Rohan Ramasamy
Date: 26/12/2017

This file contains code to generate the predicted nuclear reaction rates for a given density and temperature of fusion
reactants - this is used to gain an understanding of where in the reaction space we need to get to for a given fusion
fuel
"""

import numpy as np
import matplotlib.pyplot as plt

from fusion_reactivities import BoschHaleReactivityFit, DDReaction, DTReaction


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
        reactivities = reactivity_fit.get_reactivity(T)
        n_1_n_2 = self.fusion_reaction.number_density(RHO)

        reaction_rate = n_1_n_2 * reactivities

        return TEMP, RHO, reaction_rate


if __name__ == '__main__':
    T = np.linspace(1, 20, 200)
    rho = np.logspace(0, 6, 200)

    dd_reaction_rate_calculator = ReactionRatesCalculator(DDReaction)
    dt_reaction_rate_calculator = ReactionRatesCalculator(DTReaction)

    TEMP, RHO, dd_reaction_rate = dd_reaction_rate_calculator.get_reaction_rates(T, rho)
    TEMP, RHO, dt_reaction_rate = dt_reaction_rate_calculator.get_reaction_rates(T, rho)

    dt = 1e-9
    radius = 1e-5
    vol = np.pi * 4.0 / 3.0 * radius ** 3
    MJ_conversion = 1e-6
    E_dd = dd_reaction_rate * 7.3e6 * 1.602e-19 * dt * vol * MJ_conversion
    E_dt = dd_reaction_rate * 17.59e6 * 1.602e-19 * dt * vol * MJ_conversion

    fig, ax = plt.subplots(2, 2)

    # Plot Reaction Rate
    im = ax[0, 0].contourf(TEMP, RHO, dd_reaction_rate, 100)
    fig.colorbar(im, ax=ax[0, 0])
    im = ax[1, 0].contourf(TEMP, RHO, dt_reaction_rate, 100)
    fig.colorbar(im, ax=ax[1, 0])

    # Plot Power Production
    im = ax[0, 1].contourf(TEMP, RHO, E_dd, 100)
    fig.colorbar(im, ax=ax[0, 1])
    im = ax[1, 1].contourf(TEMP, RHO, E_dt, 100)
    fig.colorbar(im, ax=ax[1, 1])

    plt.show()