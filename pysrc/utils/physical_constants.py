"""
Author: Rohan Ramasamy
Date: 06/03/2018

This file contains constants used in this repository
"""

import numpy as np


class PhysicalConstants(object):
    epsilon_0 = 8.854187817e-12
    mu_0 = 4 * np.pi * 1e-7
    electron_charge = 1.60218e-19
    boltzmann_constant = 1.3806490351e-23
    electron_mass = 9.10938356e-31
    
    avogadro_constant = 6.023e23

    def __init__(self):
        raise RuntimeError("This class is static!")


if __name__ == '__main__':
    pass
