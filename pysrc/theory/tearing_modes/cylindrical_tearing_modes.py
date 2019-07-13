"""
Author: Rohan Ramasamy
Date: 12/07/2019

A solution for the growth rate of resistive tearing modes in circular plasmas. The implementation is 
based on:

Tearing mode analysis in tokamaks, revisited - Y. Nishimura, J. D. Callen, and C. C. Hegna
"""

import numpy as np
from scipy import integrate, interpolate
import matplotlib.pyplot as plt

from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


class TearingModeSolver(object):
    def __init__(self, k, m, B_z0, epsilon, R, num_pts):
        r_max = epsilon * R
        r = np.linspace(0.001, r_max, num_pts)
        x = np.linspace(0.002, 2.0, num_pts)
        self.x_s = 1.0
        self.x_lower = np.linspace(0.01, 0.99 * self.x_s, num_pts) 
        self.x_upper = np.linspace(1.01 * self.x_s, 1.99, num_pts) 
        r_s = epsilon * R / 2.0
        q_0 = m / k / 2
        b = x / (1 + x ** 2)
        q = q_0 * (1 + x ** 2)

        factor = 1 / (r ** 2 * k ** 2 + m ** 2)
        self.H = r ** 3 * factor
        self.Hprime = 3 * r ** 2 * factor - 2 * r * r ** 3 * factor ** 2
        self.Hprime *= r_s

        self.F = B_z0 / R * (1 - m/ q)
        self.Fprime = -B_z0 / R * m * 2 * x * (1 + x ** 2) ** -2
        self.Fprime2 = -B_z0 / R * m * (2 * (1 + x ** 2) ** -2 - 8 * x ** 2 * (1 + x ** 2) ** -3)

        self.g = k ** 2 * r ** 2 * factor 
        self.g *= r * self.F ** 2 + self.F * 2 * B_z0 * (k * r - m * b) * factor
        self.g += (m ** 2 - 1) * r * self.F ** 2 * factor

        g1 = 1 / self.H * (self.g / self.F ** 2 + (self.H * self.Fprime2 + self.Hprime * self.Fprime) / self.F)
        g2 = self.Hprime / self.H

        self.g1_interp = interpolate.interp1d(x, g1)
        self.g2_interp = interpolate.interp1d(x, g2)

        self.axis_max = None
        self.axis_min = None
        self.bnd_max = None
        self.bnd_min = None
        self.A_lower = None
        self.A_upper = None

    def solve_to_bnd(self, psi, psi_deriv, x):
        def integration_model(c, t):
            dpsi_dt = c[1]
            d2psi_dt2 = -self.g2_interp(t) * c[1] + self.g1_interp(t) * c[0]

            return [dpsi_dt, d2psi_dt2]

        return integrate.odeint(integration_model, [psi, psi_deriv], x, hmax=1e-2)

    def local_psi(self, s, A):
        kappa = self.g1_interp(self.x_s)
        psi = 1.0 + (kappa * s + 0.5 * kappa ** 2 * s ** 2) * np.log(np.abs(s)) * + A * (s + 0.5 * kappa * s ** 2)

        return psi
    
    def local_psi_derivative(self, s, A):
        kappa = self.g1_interp(self.x_s)
        psi_deriv = kappa * np.log(np.abs(s)) + kappa + A + (kappa ** 2 + A * kappa) * s

        return psi_deriv

    def find_delta(self, plot_output=False):
        # Estimate initial conditions at tearing mode
        self.A_lower = -2.0
        self.A_upper = -1.0

        # --- Solve upper solution --- 
        # Hunt for closed bounds for bisection
        psi = self.local_psi(self.x_upper[0] - self.x_s, self.A_upper)
        psi_deriv_upper = self.local_psi_derivative(self.x_upper[0] - self.x_s, self.A_upper)
        self.psi_sol_upper = self.solve_to_bnd(psi, psi_deriv_upper, self.x_upper)
        if self.psi_sol_upper[-1, 0] > 0.0:
            self.bnd_max = self.A_upper
            while (self.psi_sol_upper[-1, 0] > 0.0):
                self.A_upper = -2 * np.abs(self.A_upper)
                psi = self.local_psi(self.x_upper[0] - self.x_s, self.A_upper)
                psi_deriv_upper = self.local_psi_derivative(self.x_upper[0] - self.x_s, self.A_upper)
                self.psi_sol_upper = self.solve_to_bnd(psi, psi_deriv_upper, self.x_upper)
            self.bnd_min = self.A_upper
        else:
            self.bnd_min = self.psi_sol_upper[-1, 0]
            while (self.psi_sol_upper[-1, 0] < 0.0):
                self.A_upper = 2 * np.abs(self.A_upper)
                psi = self.local_psi(self.x_upper[0] - self.x_s, self.A_upper)
                psi_deriv_upper = self.local_psi_derivative(self.x_upper[0] - self.x_s, self.A_upper)
                self.psi_sol_upper = self.solve_to_bnd(psi, psi_deriv_upper, self.x_upper)
            self.bnd_max = self.A_upper
        
        # Bisect to find A
        iterations = 0
        tol = 1e-8
        while (iterations < 1000):
            self.A_upper = 0.5 * (self.bnd_max + self.bnd_min)
            
            # Test for convergence
            if (np.abs(self.bnd_max - self.A_upper) < tol):
                break

            psi = self.local_psi(self.x_upper[0] - self.x_s, self.A_upper)
            psi_deriv_upper = self.local_psi_derivative(self.x_upper[0] - self.x_s, self.A_upper)
            self.psi_sol_upper = self.solve_to_bnd(psi, psi_deriv_upper, self.x_upper)
            
            if (self.psi_sol_upper[-1, 0] > 0.0):
                self.bnd_max = self.A_upper
            else:
                self.bnd_min = self.A_upper

            # Increase iteration count
            iterations += 1

        # --- Solve lower solution ---
        # Hunt for closed bounds for bisection
        psi = self.local_psi(self.x_lower[-1] - self.x_s, self.A_lower)
        psi_deriv_lower = self.local_psi_derivative(self.x_lower[-1] - self.x_s, self.A_lower)
        self.psi_sol_lower = self.solve_to_bnd(1.0, psi_deriv_lower, self.x_lower[::-1])
        if self.psi_sol_lower[-1, 0] > 0.0:
            self.axis_max = self.A_lower
            while (self.psi_sol_lower[-1, 0] > 0.0):
                self.A_lower = -2 * np.abs(self.A_lower)
                psi = self.local_psi(self.x_lower[-1] - self.x_s, self.A_lower)
                psi_deriv_lower = self.local_psi_derivative(self.x_lower[-1] - self.x_s, self.A_lower)
                self.psi_sol_lower = self.solve_to_bnd(1.0, psi_deriv_lower, self.x_lower[::-1])
            self.axis_min = self.A_lower
        else:
            self.axis_min = self.psi_sol_lower[-1, 0]
            while (self.psi_sol_lower[-1, 0] < 0.0):
                self.A_lower = 2 * np.abs(self.A_lower)
                psi = self.local_psi(self.x_lower[-1] - self.x_s, self.A_lower)
                psi_deriv_lower = self.local_psi_derivative(self.x_lower[-1] - self.x_s, self.A_lower)
                self.psi_sol_lower = self.solve_to_bnd(1.0, psi_deriv_lower, self.x_lower[::-1])
            self.axis_max = self.A_lower
        
        # Bisect to find A
        iterations = 0
        while (iterations < 1000):
            self.A_lower = 0.5 * (self.axis_max + self.axis_min)
            
            # Test for convergence
            if (np.abs(self.axis_max - self.A_lower) < tol):
                break

            psi = self.local_psi(self.x_lower[-1] - self.x_s, self.A_lower)
            psi_deriv_lower = self.local_psi_derivative(self.x_lower[-1] - self.x_s, self.A_lower)
            self.psi_sol_lower = self.solve_to_bnd(1.0, psi_deriv_lower, self.x_lower[::-1])
            
            if (self.psi_sol_lower[-1, 0] > 0.0):
                self.axis_max = self.A_lower
            else:
                self.axis_min = self.A_lower

            # Increase iteration count
            iterations += 1

        print("A_I, A_III: {}, {}".format(self.A_lower, self.A_upper))
        print('Delta: {}'.format(self.A_upper - self.A_lower))
        if plot_output:
            plt.figure()
            plt.plot(self.x_lower[::-1], self.psi_sol_lower[:, 0])
            plt.plot(self.x_upper, self.psi_sol_upper[:, 0])
            plt.ylabel('$\Psi$')
            plt.xlabel('x')
            plt.show()

            plt.figure()
            plt.plot(self.x_lower[::-1], self.psi_sol_lower[:, 1])
            plt.plot(self.x_upper, self.psi_sol_upper[:, 1])
            plt.ylabel('$\\frac{\partial \Psi}{\partial r}$')
            plt.xlabel('x')
            plt.show()


if __name__ == '__main__':
    solver = TearingModeSolver(1, 2, 1.0, 4.0, 1.0, 10000)
    solver.find_delta()
