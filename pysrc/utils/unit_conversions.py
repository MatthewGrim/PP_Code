"""
Author: Rohan Ramasamy
Date: 06/03/2018

This file contains unit conversions used in this repository
"""


class UnitConversions(object):
    eV_to_K = 1.1604522167e4
    K_to_eV = 1 / eV_to_K
    amu_to_kg = 1.66054e-27

    def __init__(self):
        raise RuntimeError("This class is static!")


if __name__ == '__main__':
    pass
