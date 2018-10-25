"""
Author: Rohan Ramasamy
Date: 04/10/17

This file contains simple B fields to be used in a simplified PIC code with a frozen B field
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from plasma_physics.pysrc.simulation.pic.algo.geometry.vector_ops import cross, magnitude, arbitrary_axis_rotation_3d


class CurrentLoop(object):
    mu_0 = 1.25663706e-6

    def __init__(self, I, radius, centre, normal, num_pts):
        """"
        Initialise primary and secondary variables of the class

        I: Current in loop
        radius: Radius of the loop
        normal: Normal to the loop. This also defines the direction of the current, as the normal is in the direction
                of the B field.
        num_pts: Number of points that are used to integrate the biot savart law across the loop
        """
        assert isinstance(I, float)
        assert isinstance(radius, float)
        assert isinstance(centre, np.ndarray)
        assert isinstance(normal, np.ndarray)
        assert isinstance(num_pts, int)

        self.__I = I
        self.__radius = radius
        self.__centre = centre
        self.__normal = normal
        self.__num_pts = num_pts

        # Discretise the loop by angle
        self.d_theta = 2.0 * np.pi / num_pts
        self.theta = np.linspace(0.0, 2 * np.pi - self.d_theta, num_pts)

        # Find an arbitrary parallel unit vector
        unit_vector = np.asarray([1.0, 1.0, 1.0])
        self.r = cross(unit_vector, normal)
        self.r_unit = self.r / magnitude(self.r)
        self.R = self.r_unit * self.__radius

        self.radial_locations = np.zeros((num_pts, 3))
        self.current_direction = np.zeros((num_pts, 3))
        for i, rotation_angle in enumerate(self.theta):
            self.radial_locations[i, :] = arbitrary_axis_rotation_3d(self.R, self.__normal, rotation_angle) - self.__centre
            self.current_direction[i, :] = cross(self.__normal, self.radial_locations[i, :] - self.__centre)

    @property
    def I(self):
        return self.__I

    @property
    def radius(self):
        return self.__radius

    @property
    def centre(self):
        return self.__centre

    @property
    def normal(self):
        return self.__normal

    @property
    def num_pts(self):
        return self.__num_pts

    def b_field(self, field_point):
        """
        Calculate the B field at an arbitrary point from the loop
        """
        assert isinstance(field_point, np.ndarray)

        permeability_constant = CurrentLoop.mu_0 / (4 * np.pi)
        d_arc_length = self.__radius * self.d_theta
        integral_constant = permeability_constant * d_arc_length * self.__I

        # Integrate vector contributions
        b_field_contributions = np.zeros((self.theta.shape[0], 3))
        for i, rotation_angle in enumerate(self.theta):
            # Get b field direction
            loop_to_point = self.radial_locations[i, :] - field_point
            b_field_direction = cross(loop_to_point, self.current_direction[i, :])
            b_unit = b_field_direction / magnitude(b_field_direction)

            loop_distance = magnitude(loop_to_point)
            b_field_contributions[i, :] = b_unit / loop_distance ** 2

        b_field_contributions *= integral_constant
        b_field = np.sum(b_field_contributions, axis=0)

        return b_field


class CombinedField(object):
    """
    This class is used to combine the fields from multiple smaller component fields. The fields are simply superposed
    to get the overall field.
    """
    def __init__(self, component_fields, domain_size=None):
        """
        Component fields

        :param component_fields: list of the component fields in the system
        :param domain_size: The domain size assumed to be square with each dimension between (-domain_size, domain_size)
        """
        assert isinstance(component_fields, list)
        assert domain_size is None or isinstance(domain_size, float)

        self.component_fields = component_fields
        self.domain_size = domain_size 

    def b_field(self, field_point):
        """
        Calculate the overall field by combining the fields of components

        :param field_point: point at which the field is evaluated
        :return:
        """
        if self.domain_size is not None: 
            if np.any(field_point < -self.domain_size) or np.any(field_point > self.domain_size):
                raise ValueError("Field point is outside simulations domain")

        b_tot = np.zeros(field_point.shape)
        for comp in self.component_fields:
            b_comp = comp.b_field(field_point[0])
            b_tot += b_comp

        return b_tot


class InterpolatedBField(object):
    """
    This class reads in a pre-calculated B field from file, and linearly interpolated the points to get the overall
    field
    """
    def __init__(self, data_file, dom_pts_idx=4, dom_size_idx=5):
        """"
        Read in fields

        :param data_file: file containing 3D data of the field to be generated
        :dom_pts_idx: The name of the file must be split in such a way that the domain points can be determined by 
                      getting the value from this index
        :dom_size_idx: The name of the file must be split in such a way that the domain size can be determined by 
                      getting the value from this index
        """
        split_name = data_file.split("_")

        dom_pts = int(split_name[dom_pts_idx])
        dom_size = float(split_name[dom_size_idx])

        b_points_x = np.loadtxt("{}_x".format(data_file)).reshape((dom_pts, dom_pts, dom_pts))
        b_points_y = np.loadtxt("{}_y".format(data_file)).reshape((dom_pts, dom_pts, dom_pts))
        b_points_z = np.loadtxt("{}_z".format(data_file)).reshape((dom_pts, dom_pts, dom_pts))
        x = np.linspace(-dom_size, dom_size, dom_pts)
        y = np.linspace(-dom_size, dom_size, dom_pts)
        z = np.linspace(-dom_size, dom_size, dom_pts)

        self.b_interpolator_x = RegularGridInterpolator((x, y, z), b_points_x)
        self.b_interpolator_y = RegularGridInterpolator((x, y, z), b_points_y)
        self.b_interpolator_z = RegularGridInterpolator((x, y, z), b_points_z)

    def b_field(self, field_point):
        """
        Return the field at location
        """
        B = np.zeros(field_point.shape)
        B[0, 0] = self.b_interpolator_x(field_point[0])
        B[0, 1] = self.b_interpolator_y(field_point[0])
        B[0, 2] = self.b_interpolator_z(field_point[0])
        return B


""""
TESTING
"""
def radial_locations():
    I = 1e6
    radius = 0.15
    loop_pts = 20
    loop = CurrentLoop(I, radius, np.asarray([0.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), loop_pts)

    # Plot radial locations
    fig, axes = plt.subplots(3, figsize=(5, 5))
    axes[0].plot(loop.radial_locations[:, 0])
    axes[1].plot(loop.radial_locations[:, 1])
    axes[2].plot(loop.radial_locations[:, 2])
    plt.show()


def field_along_axis(offset=0.0):
    I = 1e4
    radius = 0.15
    loop_pts = 20
    loop = CurrentLoop(I, radius, np.asarray([0.0, 0.0, offset]), np.asarray([0.0, 0.0, 1.0]), loop_pts)

    # Calculate b_field along axis
    numerical_pts = 100
    Z = np.linspace(-5.0, 5.0, numerical_pts)
    B = np.zeros((numerical_pts, 3))
    for i, z in enumerate(Z):
        b = loop.b_field(np.asarray([0.0, 0.0, z]))
        B[i, :] = b
    analytic_x_field = np.zeros(numerical_pts)
    analytic_y_field = np.zeros(numerical_pts)
    analytic_z_field = CurrentLoop.mu_0 / 2 * I * radius ** 2 / ((radius ** 2 + (Z + offset) ** 2) ** (3.0 / 2.0))

    # Plot B_field on axis compare results with analytical
    fig, axes = plt.subplots(3, 2, figsize=(5, 5))
    fig.suptitle("On axis field comparison numerical vs analytic")

    axes[0, 0].plot(Z, B[:, 0])
    axes[0, 0].plot(Z, analytic_x_field, label="Analytic")
    axes[0, 0].legend()

    axes[0, 1].plot(Z, B[:, 0] - analytic_x_field, label="Diff")
    axes[0, 1].legend()

    axes[1, 0].plot(Z, B[:, 1])
    axes[1, 0].plot(Z, analytic_y_field, label="Analytic")
    axes[1, 0].legend()

    axes[1, 1].plot(Z, B[:, 1] - analytic_y_field, label="Diff")
    axes[1, 1].legend()

    axes[2, 0].plot(Z, B[:, 2], label="Numerical")
    axes[2, 0].plot(Z, analytic_z_field, label="Analytic")
    axes[2, 0].legend()

    axes[2, 1].plot(Z, B[:, 2] - analytic_z_field, label="Diff")
    axes[2, 1].legend()

    plt.show()


def full_b_field():
    I = 1e6
    radius = 2.0
    loop_pts = 20
    loop = CurrentLoop(I, radius, np.asarray([0.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0]), loop_pts)

    numerical_pts = 40
    min_dom = -2.5
    max_dom = 2.5
    X = np.linspace(min_dom, max_dom, numerical_pts)
    Y = np.linspace(min_dom, max_dom, numerical_pts)
    Z = np.linspace(min_dom, max_dom, numerical_pts)
    B = np.zeros((numerical_pts, numerical_pts, numerical_pts, 4))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            for k, z in enumerate(Z):
                b = loop.b_field(np.asarray([x, y, z]))
                B[i, j, k, 0] = b[0]
                B[i, j, k, 1] = b[1]
                B[i, j, k, 2] = b[2]
                B[i, j, k, 3] = magnitude(b)

    fig, ax = plt.subplots(3, figsize=(5, 5))
    X_1, Y_1 = np.meshgrid(X, Y, indexing='ij')
    im = ax[0].contourf(X_1, Y_1, B[:, :, numerical_pts // 2, 3], 100)
    fig.colorbar(im, ax=ax[0])
    ax[0].quiver(X_1, Y_1, B[:, :, numerical_pts // 2, 0], B[:, :, numerical_pts // 2, 1], headlength=7)

    X_2, Z_2 = np.meshgrid(X, Z, indexing='ij')
    im = ax[1].contourf(X_2, Z_2, B[:, numerical_pts // 2, :, 3], 100)
    fig.colorbar(im, ax=ax[1])
    ax[1].quiver(X_2, Z_2, B[:, numerical_pts // 2, :, 0], B[:, numerical_pts // 2, :,  2], headlength=7)

    Y_3, Z_3 = np.meshgrid(X, Y, indexing='ij')
    im = ax[2].contourf(Y_3, Z_3, B[numerical_pts // 2, :, :, 3], 100)
    fig.colorbar(im, ax=ax[2])
    ax[2].quiver(Y_3, Z_3, B[numerical_pts // 2, :, :, 1], B[numerical_pts // 2, :, :, 2], headlength=7)

    plt.show()


if __name__ == '__main__':
    radial_locations()
    field_along_axis()
    field_along_axis(0.5)
    field_along_axis(-0.5)
    # full_b_field()

