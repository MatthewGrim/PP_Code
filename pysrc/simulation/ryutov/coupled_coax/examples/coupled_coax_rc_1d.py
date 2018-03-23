"""
Author: Rohan Ramasamy
Date: 04/06/2017
"""

from matplotlib import pyplot as plt

from plasma_physics.pysrc.simulation.ryutov.coupled_coax.coupled_coaxial_liner_implosion import CoupledCoaxialLinerImplosion
from plasma_physics.pysrc.simulation.ryutov.coupled_coax.circuit_modules.rl_circuit import RLCircuit
from plasma_physics.pysrc.simulation.ryutov.coupled_coax.liner_modules.coaxial_liner_1d import CoaxialLiner1D
from plasma_physics.pysrc.simulation.ryutov.coupled_coax.eos_modules.vacuum_eos import VacuumEOS
from plasma_physics.pysrc.simulation.ryutov.coupled_coax.eos_modules.ideal_eos import IdealEOS1D


def example(use_vacuum, plot_implosion=True, plot_circuit=True):
    """
    Run a simple 1D example simulation
    :return:
    """
    final_time = 4e-6
    time_res = 100000
    if use_vacuum:
        sim = CoupledCoaxialLinerImplosion(RLCircuit, CoaxialLiner1D, VacuumEOS,
                                           final_time=final_time, time_resolution=time_res,
                                           liner_density=2700.0, liner_resistivity=0.0, convergence_ratio=0.1,
                                           r_inner=1.0e-2, r_outer=1.1e-2, R_Inner=1.4e-2, R_Outer=3.0e-2, h=0.3,
                                           C=1313e-6, V=120e3, R_circ=0.0, L_circ=3.75e-9,
                                           liner_shape=1)
    else:
        sim = CoupledCoaxialLinerImplosion(RLCircuit, CoaxialLiner1D, IdealEOS1D,
                                           final_time=final_time, time_resolution=time_res,
                                           p_0=1e5, rho_0=0.213, molecular_mass=3,
                                           liner_density=2700.0, liner_resistivity=0.0, convergence_ratio=None,
                                           r_inner=1.0e-2, r_outer=1.1e-2, R_Inner=1.4e-2, R_Outer=3.0e-2, h=0.3,
                                           C=1313e-6, V=120e3, R_circ=0.0, L_circ=3.75e-9,
                                           liner_shape=1)

    circuit, liner, eos = sim.run_simulation()

    times = sim.times
    v_gen, q_gen, I, I_dot = circuit.results()
    r_i, r_o, v, e_kin, R, V, E_kin, L, L_dot, R_load = liner.results()
    p, rho = eos.results()

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
        ax[0, 2].plot(times, p)
        ax[0, 2].set_title("Pressure (Pa)")
        ax[1, 0].plot(times, v, c='r')
        ax[1, 0].plot(times, V, c='g')
        ax[1, 0].set_title("Velocity (m/s)")
        ax[1, 1].plot(times, e_kin, c='r')
        ax[1, 1].plot(times, E_kin, c='g')
        ax[1, 1].set_title("Kinetic Energy (J)")
        ax[1, 2].plot(times, r_o - r_i)
        ax[1, 2].set_title("Inner liner difference in radius (m)")
        plt.tight_layout()
        plt.show()

    # Plot circuit variables
    if plot_circuit:
        fig, ax = plt.subplots(2, 3, figsize=(18, 9))
        ax[0, 0].plot(times, v_gen)
        ax[0, 0].set_title("Generator Voltage (V)")
        ax[1, 0].plot(times, I)
        ax[1, 0].set_title("Circuit current (A)")
        ax[0, 1].plot(times, q_gen)
        ax[0, 1].set_title("Charge in Capacitor Bank (Q)")
        ax[1, 1].plot(times, I_dot)
        ax[1, 1].set_title("Change in current (As-1)")
        ax[0, 2].plot(times, L)
        ax[0, 2].set_title("Inductance (H)")
        ax[1, 2].plot(times, L_dot)
        ax[1, 2].set_title("Change in Inductance (Hs-1)")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    example(False)
