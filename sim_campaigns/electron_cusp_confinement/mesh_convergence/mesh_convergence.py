"""
Author: Rohan Ramasamy
Date: 13/07/2018

This file contains code to determine what loop resolution is necessary for accurate simulations in this campaign
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy

from plasma_physics.pysrc.simulation.pic.algo.fields.magnetic_fields.generic_b_fields import *
from plasma_physics.pysrc.simulation.pic.algo.particle_pusher.boris_solver import *
from plasma_physics.pysrc.simulation.pic.data.particles.charged_particle import PICParticle


def loop_pt_convergence():
    # Sim parameters
    I = 1e4
    radius = 0.15
    loop_offset = 0.175

    # Generate sample points
    num_tests = 1000
    np.random.seed(1)
    sample_points = np.random.uniform(-radius, radius, (3, num_tests))

    # Generate result points
    results = []
    loop_points = [250, 200]
    for i, loop_pts in enumerate(loop_points):
        comp_loops = list()
        comp_loops.append(
            CurrentLoop(I, radius, np.asarray([-loop_offset, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), loop_pts))
        comp_loops.append(
            CurrentLoop(I, radius, np.asarray([loop_offset, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0]), loop_pts))
        comp_loops.append(
            CurrentLoop(I, radius, np.asarray([0.0, -loop_offset, 0.0]), np.asarray([0.0, 1.0, 0.0]), loop_pts))
        comp_loops.append(
            CurrentLoop(I, radius, np.asarray([0.0, loop_offset, 0.0]), np.asarray([0.0, -1.0, 0.0]), loop_pts))
        comp_loops.append(
            CurrentLoop(I, radius, np.asarray([0.0, 0.0, -loop_offset]), np.asarray([0.0, 0.0, 1.0]), loop_pts))
        comp_loops.append(
            CurrentLoop(I, radius, np.asarray([0.0, 0.0, loop_offset]), np.asarray([0.0, 0.0, -1.0]), loop_pts))
        b_field = CombinedField(comp_loops)

        results.append([])
        for j in range(num_tests):
            b = b_field.b_field(np.asarray([sample_points[:, j]]))
            results[i].append(magnitude(b[0]))

    plt.figure()
    for i, loop_pts in enumerate(loop_points):
        if i == 0:
            continue

        plt.plot(np.asarray(results[i]) / np.asarray(results[0]), label="{}".format(loop_pts))
    plt.legend()
    plt.savefig("field_resolution_convergence")
    plt.show()


def dom_pt_convergence():
    # Sim parameters
    I = 1e4
    radius = 0.15
    loop_offset = 1.25

    # Generate sample points
    num_tests = 100000
    np.random.seed(1)
    sample_points = np.random.uniform(-radius, radius, (3, num_tests))

    # Generate result points
    results = []
    dom_points = [130, 56]
    for i, dom_pts in enumerate(dom_points):
        file_name = "b_field_{}_{}_{}_{}_{}_{}".format(1.0 * 1e-3, 1.0, loop_offset, dom_pts, 200, 1.375)
        file_path = os.path.join("..", "mesh_generation", "data", "radius-1.0m", "current-0.001kA", "domres-{}".format(dom_pts), file_name)
        b_field = InterpolatedBField(file_path, dom_pts_idx=6, dom_size_idx=8)

        results.append([])
        for j in range(num_tests):
            b = b_field.b_field(np.asarray([sample_points[:, j] / radius])) * I / radius
            results[i].append(magnitude(b[0]))

    plt.figure()
    for i, dom_pts in enumerate(dom_points):
        if i == 0:
            continue

        plt.plot(np.asarray(results[i]) / np.asarray(results[0]), label="{}".format(dom_pts))
    plt.legend()
    plt.savefig("field_resolution_convergence")
    plt.show()


if __name__ == '__main__':
    dom_pt_convergence()


