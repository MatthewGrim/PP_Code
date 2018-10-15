"""
Author: Rohan Ramasamy
Date: 05/07/2018

This file contains scripts to get the power balance within IEC devices assuming maxwellian plasmas
"""

import numpy as np
import matplotlib.pyplot as plt

from plasma_physics.pysrc.theory.reactivities.fusion_reactivities import BoschHaleReactivityFit, DDReaction, DTReaction
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants
from plasma_physics.pysrc.theory.reactivities.fusion_reaction_rates import ReactionRatesCalculator
from plasma_physics.pysrc.theory.bremsstrahlung.bremsstrahlung_loss_models import MaxonBremsstrahlungRadiation


def get_maxwellian_power_balance():
    """
    Function to get a parameter scan of the power balance for different fusion 
    reactions
    """
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

    # Calculate bremsstrahlung loss rates - don't need to worry about Z_eff 
    # as these are assumed fully ionised hydrogen plasmas
    P_brem_dd = MaxonBremsstrahlungRadiation.power_loss_density(N_dd, TEMP)
    P_brem_dd *= MW_conversion
    P_brem_dt = MaxonBremsstrahlungRadiation.power_loss_density(N_dt, TEMP)
    P_brem_dt *= MW_conversion

    # Get power balances
    power_threshold = 1.0
    dd_balance = P_dd - P_brem_dd
    dd_balance[dd_balance < power_threshold] = power_threshold
    dt_balance = P_dt - P_brem_dt 
    dt_balance[dt_balance < power_threshold] = power_threshold

    # Plot power balance results
    fig, ax = plt.subplots(2)

    # Plot Power Production
    im = ax[0].contourf(np.log10(TEMP), np.log10(N_dd), np.log10(dd_balance), 100)
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("DD Power Balance (MWm-3)")
    im = ax[1].contourf(np.log10(TEMP), np.log10(N_dt), np.log10(dt_balance), 100)
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("DT Power Balance (MWm-3)")

    fig.suptitle("Power balance for different fusion reactions")
    plt.show()


if __name__ == "__main__":
    get_maxwellian_power_balance()

