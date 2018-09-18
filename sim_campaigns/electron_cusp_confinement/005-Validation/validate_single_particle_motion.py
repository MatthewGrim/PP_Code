"""
Author: Rohan Ramasamy
Date: 08/08/2018

This script contains code that builds from study 003, getting the velocity distributions from simulations
"""

import multiprocessing as mp

from plasma_physics.sim_campaigns.electron_cusp_confinement.run_sim import run_parallel_sims


def replicate_fig2():
    radii = [0.1]
    I = [10.0, 100.0, 1000.0, 10000.0, 100000.0]
    pool = mp.Pool(processes=2)
    args = []
    for current in I:
        for radius in radii:
            args.append((radius, 100.0, current, 1, True, False))
    pool.map(run_parallel_sims, args)
    pool.close()
    pool.join()


def replicate_fig5():
    radii = [1.0]
    I = [100.0, 200.0, 500.0, 1e3, 2e3, 5e3, 1e4, 2e4]
    pool = mp.Pool(processes=2)
    args = []
    for current in I:
        for radius in radii:
            args.append((radius, 100.0, current, 1, True, False))
    pool.map(run_parallel_sims, args)
    pool.close()
    pool.join()

    # run_parallel_sims((1.0, 100.0, 100.0, 1, True, False))


def replicate_fig6():
    radii = [1.0]
    I = [1e4]
    electron_energies = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    pool = mp.Pool(processes=2)
    args = []
    for radius in radii:
        for current in I:
            for e_eV in electron_energies:
                args.append((radius, e_eV, current, 1, True, False))
    pool.map(run_parallel_sims, args)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # replicate_fig2()
    # replicate_fig5()
    replicate_fig6()

