""""
Author: Rohan Ramasamy
Date: 02/10/2017

This file contains generic E fields to be applied with the particle pusher. This is used to simulate simple systems
where the E field does not need to be solved.
"""

import numpy as np
from matplotlib import pyplot as plt

from plasma_physics.pysrc.simulation.pic.algo.geometry.vector_ops import magnitude


class PointField(object):
    epsilon = 8.85e-12

    def __init__(self, charge_density, radius, centre):
        """"
        Sets up field variables
        """
        assert isinstance(charge_density, float)
        assert isinstance(radius, float)
        assert isinstance(centre, np.ndarray)

        self.__rho = charge_density
        self.__radius = radius
        self.__centre = centre

        # Set up secondary variables
        self.volume = 4.0 / 3.0 * np.pi * radius ** 3
        self.surface_area = 4.0 * np.pi * radius ** 2
        self.total_charge = self.__rho * self.volume

    @property
    def rho(self):
        return self.__rho

    @property
    def radius(self):
        return self.__radius

    @property
    def centre(self):
        return self.__centre

    def e_field(self, field_point):
        """"
        Calculated the electric field at a given point

        field_point: point at which the field is being calculated
        """
        radial_distance = field_point - self.__centre
        radial_magnitude = magnitude(radial_distance.flatten())
        radial_direction = radial_distance / radial_magnitude

        if radial_magnitude < self.__radius:
            E = self.total_charge * radial_magnitude / (4.0 * np.pi * self.__radius ** 3 * PointField.epsilon)
            return E * radial_distance
        else:
            E = self.total_charge / (4.0 * np.pi * radial_magnitude ** 2 * PointField.epsilon)
            return E * radial_direction

    def v_field(self, field_point):
        """"
        Calculated the potential difference at a given point

        field_point: point at which the field is being calculated
        """
        radial_distance = field_point - self.__centre
        radial_magnitude = magnitude(radial_distance)

        if radial_magnitude < self.__radius:
            V = self.total_charge / (8.0 * np.pi * self.__radius * PointField.epsilon) * (3.0 - radial_magnitude ** 2 / self.__radius ** 2)
            return V
        else:
            V = self.total_charge / (4.0 * np.pi * radial_magnitude * PointField.epsilon)
            return V


if __name__ == '__main__':
    # Construct point field
    field = PointField(1.0, 1.0, np.zeros(2))

    # Generate fields
    num_pts = 200
    X = np.linspace(-5.0, 5.0, num_pts)
    Y = np.linspace(-5.0, 5.0, num_pts)
    E = np.zeros((num_pts, num_pts))
    E_vector = np.zeros((num_pts, num_pts, 2))
    V = np.zeros((num_pts, num_pts))

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            point = np.asarray([x, y])
            e = field.e_field(point)
            v = field.v_field(point)

            E_vector[i, j, 0] = e[0]
            E_vector[i, j, 1] = e[1]
            E[i, j] = magnitude(e)
            V[i, j] = v

    # Plot results
    X, Y = np.meshgrid(X, Y)
    X = X.transpose()
    Y = Y.transpose()
    num_contours = 100
    fig, axes = plt.subplots(2, figsize=(10, 10))
    im = axes[0].contourf(X, Y, E, num_contours)
    quiv = axes[0].quiver(X, Y, E_vector[:, :, 0], E_vector[:, :, 1])
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("Electric Field (V/m)")
    im = axes[1].contourf(X, Y, V, num_contours)
    fig.colorbar(im, ax=axes[1])
    axes[1].set_title("Potential Field (V)")
    plt.show()

