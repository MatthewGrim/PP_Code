"""
Author: Rohan Ramasamy
Date: 20/09/2018

This script contains code to load and test a particular magnetic field by eye. Thie current loop method is unit tested
for the axial magnetic field, and passes. But this will test the 3D interpolator to a greater degree.
"""

import os
import numpy as np
from matplotlib import pyplot as plt

from plasma_physics.pysrc.simulation.pic.algo.fields.magnetic_fields.generic_b_fields import InterpolatedBField
from plasma_physics.pysrc.simulation.pic.algo.geometry.vector_ops import magnitude


def load_field(I, radius):
    assert isinstance(I, float)
    assert isinstance(radius, float)

    # Generate Polywell field
    loop_pts = 200
    domain_pts = 130
    loop_offset = 1.25
    dom_size = 1.1 * loop_offset * radius
    to_kA = 1e-3
    file_name = "b_field_{}_{}_{}_{}_{}_{}".format(I * to_kA, radius, loop_offset, domain_pts, loop_pts, dom_size)
    file_path = os.path.join("..", "mesh_generation", "data", "radius-{}m".format(radius),
                             "current-{}kA".format(I * to_kA), "domres-{}".format(domain_pts), file_name)
    b_field = InterpolatedBField(file_path, dom_pts_idx=6, dom_size_idx=8)

    return b_field


def compare_fields(dom_size, numerical_pts, b_field, radius, current):
    b_factor = current / radius
    unit_field = load_field(1.0, 1.0)

    min_dom = -dom_size
    max_dom = dom_size
    X = np.linspace(min_dom, max_dom, numerical_pts)
    Y = np.linspace(min_dom, max_dom, numerical_pts)
    Z = np.linspace(min_dom, max_dom, numerical_pts)
    B = np.zeros((numerical_pts, numerical_pts, numerical_pts, 4))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            for k, z in enumerate(Z):
                print(i, j, k)
                field_point = np.zeros((1, 3))
                field_point[0, 0] = x
                field_point[0, 1] = y
                field_point[0, 2] = z
                b = np.abs(b_field.b_field(field_point))
                gummersall_field = np.abs(unit_field.b_field(field_point / radius) * b_factor)
                b = gummersall_field - b

                B[i, j, k, 0] = b[0, 0]
                B[i, j, k, 1] = b[0, 1]
                B[i, j, k, 2] = b[0, 2]
                B[i, j, k, 3] = magnitude(b[0])

    fig, ax = plt.subplots(1)
    X_1, Y_1 = np.meshgrid(X, Y, indexing='ij')
    im = ax.contourf(X_1, Y_1, B[:, :, numerical_pts // 2, 3], 100)
    fig.colorbar(im, ax=ax)
    ax.quiver(X_1, Y_1, B[:, :, numerical_pts // 2, 0], B[:, :, numerical_pts // 2, 1])
    plt.savefig("magnetic_field_sample_xy")
    plt.show()

    fig, ax = plt.subplots(1)
    X_2, Z_2 = np.meshgrid(X, Z, indexing='ij')
    im = ax.contourf(X_2, Z_2, B[:, numerical_pts // 2, :, 3], 100)
    fig.colorbar(im, ax=ax)
    ax.quiver(X_2, Z_2, B[:, numerical_pts // 2, :, 0], B[:, numerical_pts // 2, :, 2])
    plt.savefig("magnetic_field_sample_xz")
    plt.show()

    fig, ax = plt.subplots(1)
    Y_3, Z_3 = np.meshgrid(X, Y, indexing='ij')
    im = ax.contourf(Y_3, Z_3, B[numerical_pts // 2, :, :, 3], 100)
    fig.colorbar(im, ax=ax)
    ax.quiver(Y_3, Z_3, B[numerical_pts // 2, :, :, 1], B[numerical_pts // 2, :, :, 2])
    plt.savefig("magnetic_field_sample_yz")
    plt.show()


def generate_domain(dom_size, numerical_pts, b_field):
    min_dom = -dom_size
    max_dom = dom_size
    X = np.linspace(min_dom, max_dom, numerical_pts)
    Y = np.linspace(min_dom, max_dom, numerical_pts)
    Z = np.linspace(min_dom, max_dom, numerical_pts)
    B = np.zeros((numerical_pts, numerical_pts, numerical_pts, 4))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            for k, z in enumerate(Z):
                print(i, j, k)
                field_point = np.zeros((1, 3))
                field_point[0, 0] = x
                field_point[0, 1] = y
                field_point[0, 2] = z
                b = b_field.b_field(field_point)
                B[i, j, k, 0] = b[0, 0]
                B[i, j, k, 1] = b[0, 1]
                B[i, j, k, 2] = b[0, 2]
                B[i, j, k, 3] = magnitude(b[0])

    fig, ax = plt.subplots(1)
    X_1, Y_1 = np.meshgrid(X, Y, indexing='ij')
    im = ax.contourf(X_1, Y_1, np.log10(B[:, :, numerical_pts // 2, 3]), 100)
    fig.colorbar(im, ax=ax)
    ax.quiver(X_1, Y_1, B[:, :, numerical_pts // 2, 0], B[:, :, numerical_pts // 2, 1])
    plt.savefig("magnetic_field_sample_xy")
    plt.show()

    fig, ax = plt.subplots(1)
    X_2, Z_2 = np.meshgrid(X, Z, indexing='ij')
    im = ax.contourf(X_2, Z_2, np.log10(B[:, numerical_pts // 2, :, 3]), 100)
    fig.colorbar(im, ax=ax)
    ax.quiver(X_2, Z_2, B[:, numerical_pts // 2, :, 0], B[:, numerical_pts // 2, :,  2])
    plt.savefig("magnetic_field_sample_xz")
    plt.show()

    fig, ax = plt.subplots(1)
    Y_3, Z_3 = np.meshgrid(X, Y, indexing='ij')
    im = ax.contourf(Y_3, Z_3, np.log10(B[numerical_pts // 2, :, :, 3]), 100)
    fig.colorbar(im, ax=ax)
    ax.quiver(Y_3, Z_3, B[numerical_pts // 2, :, :, 1], B[numerical_pts // 2, :, :, 2])
    plt.savefig("magnetic_field_sample_yz")
    plt.show()


def visualise_field():
    I = 1e4
    radius = 1.0
    b_field = load_field(I, radius)
    num_samples = 100
    generate_domain(1.25 * radius, num_samples, b_field)


def compare_field():
    I = 1e2
    radius = 0.1
    b_field = load_field(I, radius)
    num_samples = 20
    compare_fields(1.25 * radius, num_samples, b_field, radius, I)


if __name__ == '__main__':
    # visualise_field()
    compare_field()

