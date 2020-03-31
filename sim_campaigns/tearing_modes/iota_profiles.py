"""
Author: Rohan Ramasamy
Date: 30/12/2019

Iota profiles for comparison with Matsuoka paper.
"""

import numpy as np


def iota_vac_l2(r, a, iota_0v):
        iota_v = iota_0v * np.ones(r.shape) if isinstance(r, np.ndarray) else iota_0v
        return iota_v, 0

def iota_vac_l23(r, a, iota_0v):
    iota = iota_0v * (0.286 + 0.714 * r ** 2 / a ** 2)
    iota_deriv = 1.428 * iota_0v * r / a ** 2
    return iota, iota_deriv

def iota_vac_l3(r, a, iota_0v):
    iota = iota_0v * r ** 2 / a ** 2
    iota_deriv = 2 * iota_0v * r / a ** 2
    return iota, iota_deriv

def iota_plasma_flat(r, a, iota_0p):
    iota_0p = iota_0p * 4.0 / 3.0
    if isinstance(r, np.ndarray):
        iota = np.zeros(r.shape)
        iota_deriv = np.zeros(r.shape)
        iota_deriv2 = np.zeros(r.shape)
        iota[r < a] = iota_0p * (1 - r[r < a] ** 6 / (4 * a ** 6))
        iota_deriv[r < a] = -3 * iota_0p * r[r < a] ** 5 / (2 * a ** 6)
        iota_deriv2[r < a] = -15 * iota_0p * r[r < a] ** 4 / (2 * a ** 6)
        iota[r > a] = iota_0p * 0.75 * a ** 2 / r[r > a] ** 2
        iota_deriv[r > a] = iota_0p * -1.5 * a ** 2 / r[r > a] ** 3
        iota_deriv2[r > a] = iota_0p * 4.5 * a ** 2 / r[r > a] ** 4        
    else:
        if r < a:
            iota = iota_0p * (1 - r ** 6 / (4 * a ** 6))
            iota_deriv = -3 * iota_0p * r ** 5 / (2 * a ** 6)
            iota_deriv2 = -15 * iota_0p * r ** 4 / (2 * a ** 6)
        else:
            iota = iota_0p * 0.75 * a ** 2 / r ** 2
            iota_deriv = iota_0p * -1.5 * a ** 2 / r ** 3
            iota_deriv2 = iota_0p * 4.5 * a ** 2 / r ** 4      
    
    return iota, iota_deriv, iota_deriv2

def iota_plasma_parabola(r, a, iota_0p):
    iota_0p = 2 * iota_0p
    if isinstance(r, np.ndarray):
        iota = np.zeros(r.shape)
        iota_deriv = np.zeros(r.shape)
        iota_deriv2 = np.zeros(r.shape)
        iota[r < a] = iota_0p * (1 - r[r < a] ** 2 / (2 * a ** 2))
        iota_deriv[r < a] = -1.0 * iota_0p * r[r < a] / a ** 2
        iota_deriv2[r < a] = -iota_0p * np.ones(r[r < a].shape) / a ** 2
        iota[r > a] = iota_0p * 0.5 * a ** 2 / r[r > a] ** 2
        iota_deriv[r > a] = -iota_0p * a ** 2 / r[r > a] ** 3
        iota_deriv2[r > a] = iota_0p * 3.0 * a ** 2 / r[r > a] ** 4        
    else:
        if r < a:
            iota = iota_0p * (1 - r ** 2 / (2 * a ** 2))
            iota_deriv = -iota_0p * r / a ** 2
            iota_deriv2 = -iota_0p / a ** 2
        else:
            iota = iota_0p * 0.5 * a ** 2 / r ** 2
            iota_deriv = -iota_0p * a ** 2 / r ** 3
            iota_deriv2 = iota_0p * 3.0 * a ** 2 / r ** 4
    return iota, iota_deriv, iota_deriv2

def iota_peaked(r, a, j_0):
    j = j_0 * (1 - r ** 2 / a ** 2) ** 4
    j_deriv = - 8 * r / a ** 2 * (1 - r ** 2 / a ** 2) ** 3
    return j, j_deriv


def f(r):
    return r

def f_deriv(r):
    if isinstance(r, np.ndarray):
        return np.ones(r.shape)
    else:
        return 1

if __name__ == '__main__':
    pass

