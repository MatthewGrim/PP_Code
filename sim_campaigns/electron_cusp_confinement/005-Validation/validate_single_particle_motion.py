"""
Author: Rohan Ramasamy
Date: 08/08/2018

This script contains code that builds from study 003, getting the velocity distributions from simulations
"""

import multiprocessing as mp

from plasma_physics.sim_campaigns.electron_cusp_confinement.run_sim import run_parallel_sims


def validate_simulations():
    radii = [1.0]
    electron_energies = [1.0, 10.0, 100.0, 1000.0]
    I = [1e5]
    pool = mp.Pool(processes=4)
    args = []
    for radius in radii:
        for current in I:
            for electron_energy in electron_energies:
                args.append((radius, electron_energy, current, 1))
    pool.map(run_parallel_sims, args)
    pool.close()
    pool.join()


if __name__ == '__main__':
    validate_simulations()

