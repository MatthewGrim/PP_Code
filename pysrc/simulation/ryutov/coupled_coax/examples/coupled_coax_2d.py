"""
Author: Rohan Ramasamy
Date: 04/06/2017
"""

import numpy as np
from matplotlib import pyplot as plt

from plasma_physics.pysrc.simulation.ryutov.coupled_coax.coupled_coaxial_liner_implosion import CoupledCoaxialLinerImplosion
from plasma_physics.pysrc.simulation.ryutov.coupled_coax.circuit_modules.rl_circuit import RLCircuit
from plasma_physics.pysrc.simulation.ryutov.coupled_coax.liner_modules.coaxial_liner_2d import CoaxialLiner2D
from plasma_physics.pysrc.simulation.ryutov.coupled_coax.eos_modules.vacuum_eos import VacuumEOS


def example(plot_implosion=True, plot_circuit=True):
    """
    Simple example of a 2D liner sim
    :return:
    """
    num_h_pts = 1000
    h = np.linspace(0, 1.0, num_h_pts)
    r_inner = np.ones(num_h_pts) * 2e-2
    R_Inner = np.ones(num_h_pts) * 2.5e-2

    sim = CoupledCoaxialLinerImplosion(RLCircuit, CoaxialLiner2D, VacuumEOS,
                                       final_time=5e-6,
                                       h=h, r_inner=r_inner, R_Inner=R_Inner,
                                       C=1313e-6, V=120e3, L_circ=3.75e-9,
                                       liner_shape=r_inner.shape)

    circuit, liner = sim.run_simulation(decoupled=False)
    times = sim.times
    v_gen, q_gen, I, I_dot = circuit.results()
    r_i, r_o, v, e_kin, R, V, E_kin, L, L_tot, L_dot, t_final = liner.results()

    if plot_implosion:
        for ts in [0, t_final]:
            fig, ax = plt.subplots(2, 3, figsize=(18, 9))
            fig.suptitle("Characteristic curves for an ideal liner implosion")
            ax[0, 0].plot(times, I)
            ax[0, 0].set_title("Current (A)")
            ax[0, 1].plot(h, r_i[ts, :], c='r')
            ax[0, 1].plot(h, r_o[ts, :], c='r', linestyle='--')
            ax[0, 1].plot(h, R[ts, :], c='g')
            ax[0, 1].set_title("Radius (m)")
            ax[0, 2].plot(h, v[ts, :], c='r')
            ax[0, 2].plot(h, V[ts, :], c='g')
            ax[0, 2].set_title("Velocity (m/s)")
            ax[1, 0].plot(times, L_tot)
            ax[1, 0].set_title("Total Inductance (L)")
            ax[1, 1].plot(times, L_dot)
            ax[1, 1].set_title("Total Inductance Change (L_dot)")
            ax[1, 2].plot(h, L[ts, :])
            ax[1, 2].set_title("Inductance (L/m)")
            plt.show()

    # Plot circuit variables
    if plot_circuit:
        fig, ax = plt.subplots(2, figsize=(18, 9))
        ax[0].plot(times, v_gen)
        ax[0].set_title("Generator Voltage (V)")
        ax[1].plot(times, I)
        ax[1].set_title("Circuit current (A)")
        plt.show()


if __name__ == '__main__':
    example(plot_circuit=False)