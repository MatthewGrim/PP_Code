"""
Author: Rohan Ramasamy
Date: 04/06/2017
"""

from matplotlib import pyplot as plt

from plasma_physics.pysrc.simulation.ryutov.coupled_coax.coupled_coaxial_liner_implosion import CoupledCoaxialLinerImplosion
from plasma_physics.pysrc.simulation.ryutov.coupled_coax.circuit_modules.rl_circuit import RLCircuit
from plasma_physics.pysrc.simulation.ryutov.coupled_coax.liner_modules.coaxial_liner_1d import CoaxialLiner1D


def example(plot_implosion=True, plot_circuit=True):
    """
    Run a simple 1D example simulation
    :return:
    """
    sim = CoupledCoaxialLinerImplosion(RLCircuit, CoaxialLiner1D,
                                       final_time=5e-6,
                                       liner_density=2700.0, liner_resistivity=0.0,
                                       r_inner=1.0e-2, r_outer=1.1e-2, R_Inner=1.4e-2, R_Outer=3.0e-2, h=0.3,
                                       C=1313e-6, V=120e3, R_circ=0.0, L_circ=3.75e-9)

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
        fig, ax = plt.subplots(2, figsize=(18, 9))
        ax[0].plot(times, v_gen)
        ax[0].set_title("Generator Voltage (V)")
        ax[1].plot(times, I)
        ax[1].set_title("Circuit current (A)")
        plt.show()


if __name__ == '__main__':
    example(plot_circuit=False)