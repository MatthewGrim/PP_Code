"""
Author: Rohan Ramasamy
Date: 11/07/2018

This file contains code to generate meshes for simulations
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing as mp

from plasma_physics.pysrc.simulation.pic.algo.fields.magnetic_fields.generic_b_fields import CurrentLoop, CombinedField, InterpolatedBField
from plasma_physics.pysrc.simulation.pic.algo.geometry.vector_ops import magnitude
from plasma_physics.pysrc.simulation.pic.io.vtk_writers import write_vti_file


def generate_polywell_fields(params):
    """
    Generic function to plot a polywell field given geometry and current
    
    I: Current in coils
    radius: radius of coil
    loop_offset: spacing of coils as a ratio of the radius
    loop_pts: Number of loop segments used to solve Biot Savart law
    """
    I, radius, loop_offset, domain_pts, loop_pts = params
    assert loop_offset >= 1.0

    dom_size = 1.1 * loop_offset * radius
    file_dir = os.path.join("{}".format(radius), "{}".format(I))
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = "b_field_{}_{}_{}_{}_{}_{}".format(I * 1e-3, radius, loop_offset, domain_pts, loop_pts, dom_size)
    print("Starting mesh {}".format(file_name))

    # Generate Polywell field
    comp_loops = list()
    comp_loops.append(CurrentLoop(I, radius, np.asarray([-loop_offset * radius, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), loop_pts))
    comp_loops.append(CurrentLoop(I, radius, np.asarray([loop_offset * radius, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0]), loop_pts))
    comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, -loop_offset * radius, 0.0]), np.asarray([0.0, 1.0, 0.0]), loop_pts))
    comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, loop_offset * radius, 0.0]), np.asarray([0.0, -1.0, 0.0]), loop_pts))
    comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, 0.0, -loop_offset * radius]), np.asarray([0.0, 0.0, 1.0]), loop_pts))
    comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, 0.0, loop_offset * radius]), np.asarray([0.0, 0.0, -1.0]), loop_pts))
    combined_field = CombinedField(comp_loops)

    # Calculate polywell field at all points
    min_dom = -dom_size
    max_dom = dom_size
    X = np.linspace(min_dom, max_dom, domain_pts)
    Y = np.linspace(min_dom, max_dom, domain_pts)
    Z = np.linspace(min_dom, max_dom, domain_pts)
    B_x = np.zeros((domain_pts, domain_pts, domain_pts))
    B_y = np.zeros((domain_pts, domain_pts, domain_pts))
    B_z = np.zeros((domain_pts, domain_pts, domain_pts))
    B = np.zeros((domain_pts, domain_pts, domain_pts))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            for k, z in enumerate(Z):
                b = combined_field.b_field(np.asarray([[x, y, z]]))
                B_x[i, j, k] = b[0, 0]
                B_y[i, j, k] = b[0, 1]
                B_z[i, j, k] = b[0, 2]
                B[i, j, k] = magnitude(b[0])

    # Write output files
    np.savetxt(os.path.join(file_dir, "{}_x".format(file_name)), B_x.reshape((domain_pts, domain_pts ** 2)))
    np.savetxt(os.path.join(file_dir, "{}_y".format(file_name)), B_y.reshape((domain_pts, domain_pts ** 2)))
    np.savetxt(os.path.join(file_dir, "{}_z".format(file_name)), B_z.reshape((domain_pts, domain_pts ** 2)))
    np.savetxt(os.path.join(file_dir, file_name), B.reshape((domain_pts, domain_pts ** 2)))
    write_vti_file(B, file_name)


def generate_10cm_meshes():
    """
    Generate 10cm radius meshes to replicate figure 2 from Gummersall et al. from 2013
    """
    radius = 0.1
    # generate_polywell_fields((100.0, radius, 1.25, 50, 20))
    # generate_polywell_fields((1e3, radius, 1.25, 50, 20))
    # generate_polywell_fields((1e4, radius, 1.25, 50, 20))

    I = [100.0, 1e3, 1e4]
    pool = mp.Pool(processes=3)
    args = [(I[i], radius, 1.25, 100, 100, ) for i in range(3)]
    pool.map(generate_polywell_fields, args)
    pool.close()
    pool.join()


if __name__ == "__main__":
    generate_10cm_meshes()