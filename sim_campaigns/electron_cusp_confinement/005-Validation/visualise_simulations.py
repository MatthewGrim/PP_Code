"""
Author: Rohan Ramasamy
Date: 21/09/2018

This script is used to visualise simulations at different kinetic energies, currents and radial locations.
"""

import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

from plasma_physics.pysrc.simulation.pic.algo.fields.magnetic_fields.generic_b_fields import InterpolatedBField
from plasma_physics.pysrc.simulation.pic.data.particles.charged_particle import PICParticle
from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants
from plasma_physics.sim_campaigns.electron_cusp_confinement.run_sim import run_simulation


def run_sims():
    radius = 0.1
    electron_energy = 1000.0
    I = 1e4
    batch_num = 1
    dI_dt = 0.0
    to_kA = 1e-3

    # Generate Polywell field
    loop_pts = 200
    domain_pts = 130
    loop_offset = 1.25
    dom_size = 1.1 * loop_offset * radius
    file_name = "b_field_{}_{}_{}_{}_{}_{}".format(I * to_kA, radius, loop_offset, domain_pts, loop_pts, dom_size)
    file_path = os.path.join("..", "mesh_generation", "data", "radius-{}m".format(radius),
                             "current-{}kA".format(I * to_kA), "domres-{}".format(domain_pts), file_name)
    b_field = InterpolatedBField(file_path, dom_pts_idx=6, dom_size_idx=8)

    seed = batch_num
    np.random.seed(seed)

    # Run simulations
    vel = np.sqrt(2.0 * electron_energy * PhysicalConstants.electron_charge / PhysicalConstants.electron_mass)
    num_sims = 1
    for i in range(num_sims):
        # Define particle velocity
        z_unit = np.random.uniform(-1.0, 1.0)
        xy_plane = np.sqrt(1 - z_unit ** 2)
        phi = np.random.uniform(0.0, 2 * np.pi)
        velocity = np.asarray([xy_plane * np.cos(phi), xy_plane * np.sin(phi), z_unit]) * vel

        # Generate particle position
        z_unit = np.random.uniform(-1.0, 1.0)
        xy_plane = np.sqrt(1 - z_unit ** 2)
        phi = np.random.uniform(0.0, 2 * np.pi)
        position = np.asarray([xy_plane * np.cos(phi), xy_plane * np.sin(phi), z_unit]) * np.random.uniform(0.0, 1.0 * radius / 16.0)

        # Generate particle
        particle = PICParticle(9.1e-31, 1.6e-19, position, velocity)

        t, x, y, z, v_x, v_y, v_z, final_idx = run_simulation((b_field, particle, radius, loop_offset * radius, I, dI_dt))


if __name__ == '__main__':
    run_sims()