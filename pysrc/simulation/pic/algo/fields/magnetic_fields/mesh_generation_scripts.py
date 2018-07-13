"""
Author: Rohan Ramasamy
Date: 26/10/17

This file contains code to generate and test simple B fields to be used in a simplified PIC code with a frozen B field
"""

import os
import numpy as np
from matplotlib import pyplot as plt

from plasma_physics.pysrc.simulation.pic.algo.fields.magnetic_fields.generic_b_fields import CurrentLoop, CombinedField, InterpolatedBField
from plasma_physics.pysrc.simulation.pic.algo.geometry.vector_ops import magnitude
from plasma_physics.pysrc.simulation.pic.io.vtk_writers import write_vti_file


def generate_current_loop_field():
    I = 1e6
    radius = 0.15
    loop_offset = 0.0
    loop_pts = 40
    loop = CurrentLoop(I, radius, np.asarray([-loop_offset, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0]), loop_pts)

    # Calculate current loop field at all points
    domain_pts = 50
    dom_size = 0.2
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
                print(i, j, k)
                b = loop.b_field(np.asarray([x, y, z]))
                B_x[i, j, k] = b[0]
                B_y[i, j, k] = b[1]
                B_z[i, j, k] = b[2]
                B[i, j, k] = magnitude(b)

    # Write output files
    file_name = "../../../testing/algo/fields/magnetic_fields/current_loop_{}_{}_{}_{}".format(I * 1e-6, loop_pts, domain_pts, dom_size)
    np.savetxt("{}_x".format(file_name), B_x.reshape((domain_pts, domain_pts ** 2)))
    np.savetxt("{}_y".format(file_name), B_y.reshape((domain_pts, domain_pts ** 2)))
    np.savetxt("{}_z".format(file_name), B_z.reshape((domain_pts, domain_pts ** 2)))
    np.savetxt(file_name, B.reshape((domain_pts, domain_pts ** 2)))
    write_vti_file(B, file_name)


def generate_polywell_fields(current_offset_factor=1.0, plot_fields=False):
    assert 0.0 <= current_offset_factor <= 1.0

    # Generate Polywell field
    I = 1e6
    radius = 0.15
    loop_offset = 0.175
    loop_pts = 20
    comp_loops = list()
    comp_loops.append(CurrentLoop(I, radius, np.asarray([-loop_offset, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), loop_pts))
    comp_loops.append(CurrentLoop(I, radius, np.asarray([loop_offset, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0]), loop_pts))
    comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, -loop_offset, 0.0]), np.asarray([0.0, 1.0, 0.0]), loop_pts))
    comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, loop_offset, 0.0]), np.asarray([0.0, -1.0, 0.0]), loop_pts))
    comp_loops.append(CurrentLoop(current_offset_factor * I, radius, np.asarray([0.0, 0.0, -loop_offset]), np.asarray([0.0, 0.0, 1.0]), loop_pts))
    comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, 0.0, loop_offset]), np.asarray([0.0, 0.0, -1.0]), loop_pts))
    combined_field = CombinedField(comp_loops)

    # Calculate polywell field at all points
    domain_pts = 20
    dom_size = 0.2
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
                print(i, j, k)
                b = combined_field.b_field(np.asarray([x, y, z]))
                B_x[i, j, k] = b[0]
                B_y[i, j, k] = b[1]
                B_z[i, j, k] = b[2]
                B[i, j, k] = magnitude(b)

    # Write output files
    file_name = "b_field_{}_{}_{}_{}_{}".format(I * 1e-6, loop_pts, domain_pts, dom_size, current_offset_factor)
    np.savetxt("{}_x".format(file_name), B_x.reshape((domain_pts, domain_pts ** 2)))
    np.savetxt("{}_y".format(file_name), B_y.reshape((domain_pts, domain_pts ** 2)))
    np.savetxt("{}_z".format(file_name), B_z.reshape((domain_pts, domain_pts ** 2)))
    np.savetxt(file_name, B.reshape((domain_pts, domain_pts ** 2)))
    write_vti_file(B, file_name)

    if plot_fields:
        # Plots overall field
        B = np.log10(B)
        fig, ax = plt.subplots(3, figsize=(15, 15))

        X_1, Y_1 = np.meshgrid(X, Y, indexing='ij')
        im = ax[0].contourf(X_1, Y_1, B[:, :, domain_pts // 2], 100)
        fig.colorbar(im, ax=ax[0])
        ax[0].quiver(X_1, Y_1, B_x[:, :, domain_pts // 2], B_y[:, :, domain_pts // 2], 100)
        ax[0].set_title("XY Plane")

        X_2, Z_2 = np.meshgrid(X, Z, indexing='ij')
        im = ax[1].contourf(X_2, Z_2, B[:, domain_pts // 2, :], 100)
        fig.colorbar(im, ax=ax[1])
        ax[1].quiver(X_2, Z_2, B_x[:, :, domain_pts // 2], B_z[:, :, domain_pts // 2], 100)
        ax[1].set_title("XZ Plane")

        Y_3, Z_3 = np.meshgrid(X, Y, indexing='ij')
        im = ax[2].contourf(Y_3, Z_3, B[domain_pts // 2, :, :], 100)
        fig.colorbar(im, ax=ax[2])
        ax[2].quiver(Y_3, Z_3, B_y[:, :, domain_pts // 2], B_z[:, :, domain_pts // 2], 100)
        ax[2].set_title("YZ Plane")
        plt.show()


def generate_interpolated_fields(current_offset_factor, compare_fields=False):
    # Generate Interpolated field
    I = 1e6
    loop_pts = 20
    domain_pts = 20
    dom_size = 0.2
    file_name = "b_field_{}_{}_{}_{}_{}".format(I * 1e-6, loop_pts, domain_pts, dom_size, current_offset_factor)
    # file_path = os.path.join("mesh_data", file_name)
    interp_field = InterpolatedBField(file_name)

    if compare_fields:
        # Generate Polywell field
        radius = 0.15
        loop_offset = 0.175
        comp_loops = list()
        comp_loops.append(CurrentLoop(I, radius, np.asarray([-loop_offset, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), loop_pts))
        comp_loops.append(CurrentLoop(I, radius, np.asarray([loop_offset, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0]), loop_pts))
        comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, -loop_offset, 0.0]), np.asarray([0.0, 1.0, 0.0]), loop_pts))
        comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, loop_offset, 0.0]), np.asarray([0.0, -1.0, 0.0]), loop_pts))
        comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, 0.0, -loop_offset]), np.asarray([0.0, 0.0, 1.0]), loop_pts))
        comp_loops.append(CurrentLoop(I, radius, np.asarray([0.0, 0.0, loop_offset]), np.asarray([0.0, 0.0, -1.0]), loop_pts))
        combined_field = CombinedField(comp_loops)

        B_poly = np.zeros((domain_pts, domain_pts, domain_pts))

    X = np.linspace(-dom_size, dom_size, domain_pts)
    Y = np.linspace(-dom_size, dom_size, domain_pts)
    Z = np.linspace(-dom_size, dom_size, domain_pts)
    B_interp = np.zeros((domain_pts, domain_pts, domain_pts))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            for k, z in enumerate(Z):
                print(i, j, k)
                b_interp = interp_field.b_field(np.asarray([[x, y, z]]))
                B_interp[i, j, k] = magnitude(b_interp[0])

                if compare_fields:
                    b_poly = combined_field.b_field(np.asarray([x, y, z]))
                    B_poly[i, j, k] = magnitude(b_poly)

    if compare_fields:
        B_comp = B_poly - B_interp
        plot_fields = [B_poly, B_interp, B_comp]
    else:
        plot_fields = [B_interp]

    # Plots overall field
    for B in plot_fields:
        fig, ax = plt.subplots(3, figsize=(5, 5))
        X_1, Y_1 = np.meshgrid(X, Y)
        im = ax[0].contourf(X_1, Y_1, B[:, :, domain_pts // 2], 100)
        fig.colorbar(im, ax=ax[0])
        ax[0].set_title("XY Plane")

        X_2, Z_2 = np.meshgrid(X, Z)
        im = ax[1].contourf(X_2, Z_2, B[:, domain_pts // 2, :], 100)
        fig.colorbar(im, ax=ax[1])
        ax[1].set_title("XZ Plane")

        Y_3, Z_3 = np.meshgrid(X, Y)
        im = ax[2].contourf(Y_3, Z_3, B[domain_pts // 2, :, :], 100)
        fig.colorbar(im, ax=ax[2])
        ax[2].set_title("YZ Plane")
        plt.show()


if __name__ == '__main__':
    generate_current_loop_field()
    # generate_polywell_fields()
    # generate_polywell_fields(0.75)
    # generate_polywell_fields(0.5)
    # generate_polywell_fields(0.25)
    # generate_polywell_fields(0.0)
    # generate_interpolated_fields(1.0, True)

