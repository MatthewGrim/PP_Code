"""
Author: Rohan Ramasamy
Date: 10/06/2017
"""


import numpy as np
from matplotlib import pyplot as plt

from plasma_physics.pysrc.simulation.ryutov.coupled_coax.coupled_coaxial_liner_implosion import CoupledCoaxialLinerImplosion
from plasma_physics.pysrc.simulation.ryutov.coupled_coax.circuit_modules.rl_circuit import RLCircuit
from plasma_physics.pysrc.simulation.ryutov.coupled_coax.liner_modules.coaxial_liner_1d import CoaxialLiner1D


def run_sims(plot_implosion=True, plot_circuit=True):
    """
    Function to run simulations
    :return:
    """
    voltages = np.linspace(100, 200, 11) * 1e3

    for V in voltages:
        sim = CoupledCoaxialLinerImplosion(RLCircuit, CoaxialLiner1D,
                                           final_time=5e-6,
                                           liner_density=6095.0, liner_resistivity=270e-9,
                                           r_inner=2e-2, R_Inner=2.5e-2, R_Outer=4.0e-2, h=0.3,
                                           C=1313e-6, V=V, L_circ=3.75e-9)

        circuit, liner = sim.run_simulation(decoupled=False)

        times = sim.times
        v_gen, q_gen, I, I_dot = circuit.results()
        r_i, r_o, v, e_kin, R, V, E_kin, L, L_dot, R_load = liner.results()

        # Plot implosion variables
        if plot_implosion:
            fig, ax = plt.subplots(2, 3, figsize=(18, 9))
            fig.suptitle("Characteristic curves for an ideal liner implosion")
            ax[0, 0].plot(times, I)
            ax[0, 0].set_title("Current (A)")
            ax[0, 1].plot(times, r_i, c='r')
            ax[0, 1].plot(times, r_o, c='r', linestyle='--')
            ax[0, 1].plot(times, R, c='g')
            ax[0, 1].set_title("Radius (m)")
            ax[0, 2].plot(times, L)
            ax[0, 2].set_title("Inductance (L/m)")
            ax[1, 0].plot(times, v, c='r')
            ax[1, 0].plot(times, V, c='g')
            ax[1, 0].set_title("Velocity (m/s)")
            ax[1, 1].plot(times, e_kin, c='r')
            ax[1, 1].plot(times, E_kin, c='g')
            ax[1, 1].set_title("Kinetic Energy (J)")
            ax[1, 2].plot(times, r_o - r_i)
            ax[1, 2].set_title("Inner liner difference in radius (m)")
            plt.show()

        # Plot circuit variables
        if plot_circuit:
            fig, ax = plt.subplots(2, 2, figsize=(18, 9))
            fig.suptitle("Characterisitc curves of liner circuit")
            ax[0, 0].plot(times, v_gen)
            ax[0, 0].set_title("Generator Voltage (V)")
            ax[0, 1].plot(times, R_load)
            ax[0, 1].set_title("Load Resistance (Ohms)")
            ax[1, 0].plot(times, I)
            ax[1, 0].set_title("Circuit current (A)")
            ax[1, 1].plot(times, L_dot)
            ax[1, 1].set_title("Load Inductance Derivative (L_dot)")
            plt.show()


if __name__ == '__main__':
    run_sims(plot_implosion=True, plot_circuit=True)