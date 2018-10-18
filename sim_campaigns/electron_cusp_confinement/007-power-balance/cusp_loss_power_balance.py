"""
Author: Rohan Ramasamy
Date: 18/10/2018

This script contains code to compare the IEC electron cusp losses to the fusion power produced
"""

import numpy as np
import matplotlib.pyplot as plt

from plasma_physics.pysrc.theory.reactivities.fusion_reactivities import BoschHaleReactivityFit, DDReaction, DTReaction
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants
from plasma_physics.pysrc.theory.reactivities.fusion_reaction_rates import ReactionRatesCalculator


def get_power_balance():
    """
    Function to get a parameter scan of the power balance for different fusion
    reactions
    """
    K = 100.0
    current = 1e5
    radius = 1.0
    well_depth = np.logspace(0, 2, 200)
    rho = np.logspace(-10, 0, 200)

    # Generate calculators
    dd_reaction_rate_calculator = ReactionRatesCalculator(DDReaction)
    dt_reaction_rate_calculator = ReactionRatesCalculator(DTReaction)

    # Calculate reaction rates
    WELL_DEPTH, RHO, dd_reaction_rate = dd_reaction_rate_calculator.get_reaction_rates(well_depth, rho)
    WELL_DEPTH, RHO, dt_reaction_rate = dt_reaction_rate_calculator.get_reaction_rates(well_depth, rho)

    # Calculate power produced per metre cubed
    MW_conversion = 1e-6
    dd_energy_released = 7.3e6 * PhysicalConstants.electron_charge
    dt_energy_released = 17.59e6 * PhysicalConstants.electron_charge
    P_dd = dd_reaction_rate * dd_energy_released * MW_conversion
    P_dt = dd_reaction_rate * dt_energy_released * MW_conversion

    # Get number densities
    N_dd = np.sqrt(2 * DDReaction().number_density(RHO))
    N_dt = np.sqrt(DTReaction().number_density(RHO))

    # Get necessary well depth - assuming uniform charge
    n = 2 * PhysicalConstants.epsilon_0 * WELL_DEPTH / (radius ** 2 * PhysicalConstants.electron_charge)
    n_dd = N_dd + n
    n_dt = N_dt + n

    # Calculate power losses
    P_cusp_dd = n_dd * 4.3e-13 * K ** 1.75 / (np.sqrt(current) * radius ** 1.5)
    P_cusp_dt = n_dt * 4.3e-13 * K ** 1.75 / (np.sqrt(current) * radius ** 1.5)

    # Get power balances
    power_threshold = 1.0
    dd_balance = P_dd - P_cusp_dd
    dd_balance[dd_balance < power_threshold] = power_threshold
    dt_balance = P_dt - P_cusp_dt
    dt_balance[dt_balance < power_threshold] = power_threshold

    # Plot power balance results
    fig, ax = plt.subplots(2, sharex=True)

    # Plot Power Production
    im = ax[0].contourf(np.log10(WELL_DEPTH), np.log10(N_dd), np.log10(dd_balance), 100)
    fig.colorbar(im, ax=ax[0])
    ax[0].set_ylabel("Number density [$m^3$]")
    ax[0].set_title("DD Power Balance [$MWm-3$]")
    im = ax[1].contourf(np.log10(WELL_DEPTH), np.log10(N_dt), np.log10(dt_balance), 100)
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("DT Power Balance [$MWm-3$]")
    ax[1].set_ylabel("Number density [$m^3$]")
    ax[1].set_xlabel("Well Depth [$keV$]")

    fig.suptitle("Power balance for different fusion reactions")
    plt.show()


if __name__ == "__main__":
    get_power_balance()

