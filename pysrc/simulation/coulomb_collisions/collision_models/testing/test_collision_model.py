"""
Author: Rohan Ramasamy
Date: 13/03/2018

This code tests the functionality of different aspects of coulomb collision
 models.
"""

import numpy as np
from matplotlib import pyplot as plt


def plot_phi_distribution():
    """
    This is just to check that the phi distribution I am using is
    reasonable
    """
    PHI = np.random.uniform(0.0, 2 * np.pi, 100000)

    plt.figure()
    plt.hist(PHI, 100)
    plt.show()


if __name__ == '__main__':
    plot_phi_distribution()
