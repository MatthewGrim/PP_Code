"""
Author: Rohan Ramasamy
Date: 15/07/17

This file contains the implementation of the boris method for updating the position and velocities of charged particles
in an electromagnetic field.
"""


from plasma_physics.pysrc.simulation.pic.algo.geometry.vector_ops import *


def boris_solver(E_field, B_field, X, V, Q, M, dt):
    """
    Function to update the positon of a set of particles in an electromagnetic field over the time dt

    :param E_field: function to evaluate the 3D E field at time t
    :param B_field: function to evaluate the 3D B field at time t
    :param X: position of the particles in the simulation domain
    :param V: velocities of the particles in the simulation domain
    :param Q: charges of the particles in the simulation domain
    ;param M: masses of the particles in the simulation domain
    :return:
    """
    assert isinstance(X, np.ndarray) and X.shape[1] == 3
    assert isinstance(V, np.ndarray) and V.shape[1] == 3
    assert X.shape[0] == V.shape[0] == Q.shape[0] == M.shape[0]
    assert isinstance(dt, float)

    # Calculate v minus
    E_field_offset = Q * E_field(X) / M * dt / 2
    v_minus = V + E_field_offset

    # Calculate v prime
    t = Q * B_field(X) / M * 0.5 * dt
    v_prime = np.zeros(v_minus.shape)
    for i, v in enumerate(v_prime[:, 0]):
        v_prime[i, :] = v_minus[i, :] + cross(v_minus[i, :], t[i, :])

    # Calculate s
    s = np.zeros(t.shape)
    for i, _ in enumerate(t[:, 0]):
        s[i, :] = 2 * t[i, :]
        s[i, :] /= 1 + magnitude(t[i, :]) ** 2

    # Calculate v_plus
    v_plus = np.zeros(v_minus.shape)
    for i, v in enumerate(v_prime[:, 0]):
        v_plus[i, :] = v_minus[i, :] + cross(v_prime[i, :], s[i, :])

    # Calculate new velocity
    V_plus = v_plus + E_field_offset

    # Integrate to get new positions
    X_plus = X + V_plus * dt

    return X_plus, V_plus

 