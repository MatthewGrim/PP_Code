"""
Author: Rohan Ramasamy
Date: 14/03/2018

Simulation of collisional frequencies, replicating results from Abe, and
Takizuka paper
"""

import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

from plasma_physics.pysrc.theory.coulomb_collisions.coulomb_collision import ChargedParticle


def mu(x):
    """
    Function defined to solve for integral across maxwell distribution
    """
    def func(eta):
        return np.exp(-eta) * np.sqrt(eta)

    mu, err = integrate.quad(func, 0, x)
    mu *= 2 / np.sqrt(np.pi)

    return mu


def mu_dash(x):
    """
    Derivative of function defined to solve for integral across maxwell
    distribution
    """
    mu_dash = 2 * np.sqrt(x / np.pi) * np.exp(-x)

    return mu_dash


def plot_mu():
    x = np.linspace(0.0, 10.0, 100)
    m = np.zeros(x.shape)
    m_dash = np.zeros(x.shape)

    for i, x_val in enumerate(x):
        m[i] = mu(x_val)
        m_dash[i] = mu_dash(x_val)

    fig, ax = plt.subplots(2, figsize=(10, 10))

    ax[0].plot(x, m)
    ax[1].plot(x, m_dash)

    plt.show()


if __name__ == '__main__':
    plot_mu()