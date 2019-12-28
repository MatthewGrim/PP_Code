"""
Author: Rohan Ramasamy
Date: 15/12/2019

Implementation of the Newcomb stability criterion for a screw pinch:

Hydromagnetic stability of a diffuse linear pinch - W. A. Newcomb

Such stability analysis can be used to find current driven instabilities in large aspect ratio
MCF devices
"""

import numpy as np
import sys
import os
from scipy import integrate, interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


class NewcombSolver(object):
    def __init__(self, m, r, r_max, f, g, a, b, n=1, f_deriv=None, singularity_func=None, num_crit_points=1001, cross_tol=1e-14):
        """
        Initialisation routine for Newcomb solver
        
        Arguments:
            m -- Poloidal mode number
            r -- Array or radial locations beginning at r=0 and ending at plasma boundary
            r_max -- maximum radial location
            f -- Array or f values
            g -- Array or g values
            num_crit_points -- number of points ode is evaluated at in integration of eta
            cross_tol -- tolerance for finding zero point crossings in f
        """
        assert isinstance(r, np.ndarray)
        assert isinstance(f, np.ndarray) or callable(f)
        assert isinstance(g, np.ndarray) or callable(g)
        assert f_deriv is None or callable(f_deriv)
        assert r[0] == 0.0
        self.r = r
        self.f = interpolate.interp1d(r, f) if isinstance(f, np.ndarray) else f
        self.g = interpolate.interp1d(r, g) if isinstance(g, np.ndarray) else g
        if f_deriv is None:
            f_num = f(r)
            f_deriv = np.gradient(f_num, r)
            self.f_deriv = interpolate.interp1d(r, f_deriv)
        else:
            self.f_deriv = f_deriv
        self.singularity_func = self.f if singularity_func is None else singularity_func

        self.a = a
        self.b = b
        self.r_crossings = list()
        self.num_crit_points = num_crit_points
        self.cross_tol = cross_tol

    def find_zero_crossings_in_f(self):
        """
        Find zero crossings in function f, which determing the bounded regions over which to integrate
        """
        sign = None
        r_prev = None
        for i, r_local in enumerate(self.r):
            f_val = self.singularity_func(r_local)
            # if i == 0:
            #     continue
            if i == 0:
                sign = np.sign(f_val)

            if not (sign == np.sign(f_val)):
                r_crossing = 0.5 * (r_prev + r_local)
                r_min = r_prev
                r_max = r_local
                iterations = 0
                while max((np.abs(r_crossing - r_min), np.abs(r_crossing - r_max))) > self.cross_tol:
                    if iterations > 100:
                        raise RuntimeError("Number of iterations for r_crossing exceeded")
                    
                    if not (np.sign(self.singularity_func(r_crossing)) == sign):
                        r_max = r_crossing
                        r_crossing = 0.5 * (r_min + r_max)
                    else:
                        r_min = r_crossing
                        r_crossing = 0.5 * (r_max + r_min)
                    
                    iterations += 1
                
                if r_crossing < self.a:
                    self.r_crossings.append(r_crossing)
                print('Zero crossing found at r = {}'.format(r_crossing))

            sign = np.sign(f_val)
            r_prev = r_local

        self.r_crossings.append(self.r[-1])
    
    def integrate_eta(self, r_crit, r_start):
        """
        Integrate eta over a particular subsection of the domain
        
        Arguments:
            r_crit -- points at which to evaluate eta
        """
        def integration_model(c, t):
            if t < r_start + 1e-8:
                dpsi_dt = c[1]
                d2psi_dt2 = 0.0
            else:
                dpsi_dt = c[1]
                d2psi_dt2 = (self.f_deriv(t) * c[1] + self.g(t) * c[0]) / self.f(t)
            
            return [dpsi_dt, d2psi_dt2]

        return integrate.odeint(integration_model, [0, 1.0], r_crit, hmax=1e-2)

    def determine_stability(self, plot_results=True):
        """
        Main function to determine the stability of the screw pinch
        
        Keyword Arguments:
            plot_results {bool} -- determine whether to plot results
        """
        r_min = self.r[0]
        unstable = False
        print(self.r_crossings)
        for r_crossing in self.r_crossings:
            r_crit = np.linspace(r_min, r_crossing - self.cross_tol, self.num_crit_points)
            eta = self.integrate_eta(r_crit, r_min)
            unstable = np.any(eta[:, 0] < 0.0)
            if unstable:
                print('Screw pinch is unstable between {} and {}'.format(r_min, r_crossing))

            if plot_results:
                plt.figure()
                plt.plot(r_crit, eta[:, 0])
                plt.ylabel('$\eta$')
                plt.xlabel('$r$')
                plt.tight_layout()
                plt.show()

            r_min = r_crossing + self.cross_tol

        return unstable

if __name__ == '__main__':
    pass

