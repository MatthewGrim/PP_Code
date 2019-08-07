"""
Author: Rohan Ramasamy
Date: 12/07/2019

A solution for the growth rate of resistive tearing modes in circular plasmas. The implementation is 
based on:

Tearing mode analysis in tokamaks, revisited - Y. Nishimura, J. D. Callen, and C. C. Hegna

and

Tearing mode in the cylindrical tokamak - H. P. Furth, P. H. Rutherford, and H. Selberg
"""

import numpy as np
import sys
from scipy import integrate, interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


class TearingModeSolver(object):
    def __init__(self, m, x_s, num_pts, integrate_from_bnds=False):
        self.integrate_from_bnds = integrate_from_bnds
        self.r_s = 1.0
        self.x_s = x_s
        r_max = 2 * self.r_s
        k = 1.0 / (20 * r_max)
        R = r_max / k
        # k = 1 / 20.0
        # R = 20.0
        print("kr_0: {}".format(k * r_max))
        print("R: {}".format(R))

        # Generate simulation domain and integration points
        x_max = 2
        if self.integrate_from_bnds:
            x = np.concatenate((np.linspace(0.0, 0.01 * x_max, num_pts),
                                np.linspace(0.01 * x_max, 0.99 * self.x_s, num_pts),
                                np.linspace(0.99 * self.x_s, 1.01 * self.x_s, num_pts),
                                np.linspace(1.01 * self.x_s, x_max, num_pts)))
            self.x_lower = np.linspace(0.0001, (1 - 1e-12) * self.x_s, num_pts) 
            self.x_upper = np.linspace((1 + 1e-12) * self.x_s, 2, num_pts) 
        else:
            x = np.linspace(0.0, x_max, num_pts)
            self.x_lower = np.linspace(0.01, (1 - 1e-12) * self.x_s, num_pts) 
            self.x_upper = np.linspace((1 + 1e-12) * self.x_s, 1.99, num_pts)
        x_to_r = self.r_s
        r = x * x_to_r

        # Get input profiles
        q_0 = m / (1 + self.x_s ** 2)
        b = x / (1 + x ** 2)
        q = q_0 * (1 + x ** 2)
        # B_z0 = R * b / (r * q)
        B_z0 = 1.0

        # plt.figure()
        # plt.plot(r, B_z0)
        # plt.show()

        factor = 1 / (x ** 2 * k ** 2 + m ** 2 / x_to_r ** 2)
        self.H = x ** 3 * factor * x_to_r
        self.Hprime = (3 * x ** 2 - 2 * k ** 2 * x ** 4 * factor) * x_to_r * factor

        # plt.figure()
        # plt.plot(x, self.H)
        # plt.plot(x, self.Hprime)
        # plt.show()

        self.F = B_z0 / R * (1 - m/ q)
        self.Fprime = B_z0 * m / (R * q_0) * 2 * x * (1 + x ** 2) ** -2
        self.Fprime2 = B_z0 * m / (R * q_0) * (2 * (1 + x ** 2) ** -2 - 8 * x ** 2 * (1 + x ** 2) ** -3)

        # plt.figure()
        # plt.plot(x, self.F)
        # plt.plot(x, self.Fprime)
        # plt.plot(x, self.Fprime2)
        # plt.show()

        factor = 1 / (r ** 2 * k ** 2 + m ** 2)
        self.g = k ** 2 * r ** 2 * factor 
        self.g *= r * self.F ** 2 + self.F * 2 * B_z0 * (k * r - m * b) * factor
        self.g += (m ** 2 - 1) * r * self.F ** 2 * factor

        # plt.figure()
        # plt.plot(x, self.g)
        # plt.show()

        g1 = 1 / self.H * (self.g / self.F ** 2 + (self.H * self.Fprime2 + self.Hprime * self.Fprime) / self.F)
        g2 = self.Hprime / self.H

        # plt.figure()
        # plt.plot(x, g1)
        # plt.plot(x, g2)
        # plt.show()

        self.g1_interp = interpolate.interp1d(x, g1)
        self.g2_interp = interpolate.interp1d(x, g2)

        # Set up output variables
        self.axis_max = None
        self.axis_min = None
        self.bnd_max = None
        self.bnd_min = None
        self.A_lower = None
        self.A_upper = None
        self.psi_sol_lower = None
        self.psi_sol_upper = None

    def solve_to_bnd(self, psi, psi_deriv, x):
        def integration_model(c, t):
            if t == 0:
                dpsi_dt = c[1]
                d2psi_dt2 = 0.0
            else:
                dpsi_dt = c[1]
                d2psi_dt2 = -self.g2_interp(t) * c[1] + self.g1_interp(t) * c[0]

            return [dpsi_dt, d2psi_dt2]

        return integrate.odeint(integration_model, [psi, psi_deriv], x, hmax=1e-2)

    def local_psi(self, s, A):
        kappa = self.g1_interp(self.x_s)
        psi = 1.0 + (kappa * s + 0.5 * kappa ** 2 * s ** 2 - 0.75 * kappa ** 2 * s ** 2) * np.log(np.abs(s)) + \
              A * (s + 0.5 * kappa * s ** 2 + 1.0 / 12.0 * kappa ** 2 * s ** 3)

        return psi
    
    def local_psi_derivative(self, s, A):
        kappa = self.g1_interp(self.x_s)
        psi_deriv = kappa * np.log(np.abs(s)) + kappa + A + (kappa ** 2 + A * kappa) * s

        return psi_deriv

    def fit_A_from_psi(self, s, psi_s, psi_sol):
        kappa = self.g1_interp(self.x_s)
        def psi(s, A):
            return psi_s * (1.0 + (kappa * s + 0.5 * kappa ** 2 * s ** 2 - 0.75 * kappa ** 2 * s ** 2) * np.log(np.abs(s))) + \
                   A * (s + 0.5 * kappa * s ** 2 + 1.0 / 12.0 * kappa ** 2 * s ** 3)

        popt, _ = curve_fit(psi, s, psi_sol)

        return popt[0]

    def find_delta_from_boundaries(self, plot_output):
        # Get upper solution
        self.psi_sol_upper = self.solve_to_bnd(0, -1, self.x_upper[::-1])
        target_sol = self.psi_sol_upper[-1, 0]

        # Find matching lower solution
        grad_0 = 10
        x_start = 0.0
        psi_start = grad_0 * x_start
        self.psi_sol_lower = self.solve_to_bnd(psi_start, grad_0, self.x_lower)
        if self.psi_sol_lower[-1, 0] > target_sol:
            factor = 0.5
            while (self.psi_sol_lower[-1, 0] > target_sol):
                self.bnd_max = grad_0
                grad_0 *= factor
                psi_start = grad_0 * x_start
                self.psi_sol_lower = self.solve_to_bnd(psi_start, grad_0, self.x_lower)
            self.bnd_min = grad_0
        else:
            factor = 2.0
            while (self.psi_sol_lower[-1, 0] < target_sol):
                self.bnd_min = grad_0
                grad_0 *= factor
                psi_start = grad_0 * x_start
                self.psi_sol_lower = self.solve_to_bnd(psi_start, grad_0, self.x_lower)
            self.bnd_max = grad_0
        print("Upper and lower bound for bisection: {}, {}".format(self.bnd_max, self.bnd_min))
        
        iterations = 0
        num_iterations = 1000
        tol = 1e-12
        while (iterations < num_iterations):
            grad_0 = 0.5 * (self.bnd_max + self.bnd_min)
            
            # Test for convergence
            if (np.abs(self.bnd_max - self.bnd_min) < tol):
                break
    
            psi_start = grad_0 * x_start
            self.psi_sol_lower = self.solve_to_bnd(psi_start, grad_0, self.x_lower)
            if (self.psi_sol_lower[-1, 0] < target_sol):
                self.bnd_min = grad_0  
            else:
                self.bnd_max = grad_0
            iterations += 1
        psi_start = grad_0 * x_start
        self.psi_sol_lower = self.solve_to_bnd(psi_start, grad_0, self.x_lower)
        
        if iterations == num_iterations: print("WARNING: Bisection failed to converge!")
        
        # Estimate A
        num_samples = 1000
        psi_max = max(np.max(self.psi_sol_lower[:, 0]), np.max(self.psi_sol_upper[:, 0]))
        self.psi_sol_lower[:, 0] /= psi_max
        self.psi_sol_upper[:, 0] /= psi_max
        psi_s = 0.5 * (self.psi_sol_lower[-1:, 0] + self.psi_sol_upper[-1:, 0])
        self.A_lower = self.fit_A_from_psi(self.x_lower[-num_samples:] - self.x_s, psi_s, self.psi_sol_lower[-num_samples:, 0])
        self.A_upper = self.fit_A_from_psi(self.x_upper[:num_samples] - self.x_s, psi_s, self.psi_sol_upper[-num_samples:, 0])

    def find_delta_from_tearing_mode(self, plot_output):
        # Estimate initial conditions at tearing mode
        self.A_lower = -1e7
        self.A_upper = -1e7

        # --- Solve upper solution --- 
        print("Solving upper solution...")
        # Hunt for closed bounds for bisection
        psi = self.local_psi(self.x_upper[0] - self.x_s, self.A_upper)
        psi_deriv_upper = self.local_psi_derivative(self.x_upper[0] - self.x_s, self.A_upper)
        self.psi_sol_upper = self.solve_to_bnd(psi, psi_deriv_upper, self.x_upper)
        if self.psi_sol_upper[-1, 0] > 0.0:
            while (self.psi_sol_upper[-1, 0] > 0.0):
                self.bnd_max = self.A_upper
                self.A_upper = -2 * np.abs(self.A_upper)
                psi = self.local_psi(self.x_upper[0] - self.x_s, self.A_upper)
                psi_deriv_upper = self.local_psi_derivative(self.x_upper[0] - self.x_s, self.A_upper)
                self.psi_sol_upper = self.solve_to_bnd(psi, psi_deriv_upper, self.x_upper)
            self.bnd_min = self.A_upper
        else:
            while (self.psi_sol_upper[-1, 0] < 0.0):
                self.bnd_min = self.A_upper
                self.A_upper = 2 * np.abs(self.A_upper)
                psi = self.local_psi(self.x_upper[0] - self.x_s, self.A_upper)
                psi_deriv_upper = self.local_psi_derivative(self.x_upper[0] - self.x_s, self.A_upper)
                self.psi_sol_upper = self.solve_to_bnd(psi, psi_deriv_upper, self.x_upper)
            self.bnd_max = self.A_upper
        
        # Bisect to find A
        print("\tBisecting upper solution ({}, {})...".format(self.bnd_min, self.bnd_max))
        iterations = 0
        num_iterations = 100
        tol = 1e-12
        while (iterations < num_iterations):
            self.A_upper = 0.5 * (self.bnd_max + self.bnd_min)
            
            # Test for convergence
            if (np.abs(self.A_upper - self.bnd_max) < tol):
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
        if iterations >= num_iterations:
            print("WARNING: Bisection failed to converge")

        # --- Solve lower solution ---
        print("Solving lower solution...")
        # Hunt for closed bounds for bisection
        psi = self.local_psi(self.x_lower[-1] - self.x_s, self.A_lower)
        psi_deriv_lower = self.local_psi_derivative(self.x_lower[-1] - self.x_s, self.A_lower)
        self.psi_sol_lower = self.solve_to_bnd(psi, psi_deriv_lower, self.x_lower[::-1])
        if self.psi_sol_lower[-1, 0] < 0.0:
            self.axis_max = self.A_lower
            while (self.psi_sol_lower[-1, 0] > 0.0):
                self.A_lower = 2 * np.abs(self.A_lower)
                psi = self.local_psi(self.x_lower[-1] - self.x_s, self.A_lower)
                psi_deriv_lower = self.local_psi_derivative(self.x_lower[-1] - self.x_s, self.A_lower)
                self.psi_sol_lower = self.solve_to_bnd(psi, psi_deriv_lower, self.x_lower[::-1])
            self.axis_min = self.A_lower
        else:
            self.axis_min = self.psi_sol_lower[-1, 0]
            while (self.psi_sol_lower[-1, 0] < 0.0):
                self.A_lower = -2 * np.abs(self.A_lower)
                psi = self.local_psi(self.x_lower[-1] - self.x_s, self.A_lower)
                psi_deriv_lower = self.local_psi_derivative(self.x_lower[-1] - self.x_s, self.A_lower)
                self.psi_sol_lower = self.solve_to_bnd(psi, psi_deriv_lower, self.x_lower[::-1])
            self.axis_max = self.A_lower
        
        # Bisect to find A
        print("\tBisecting lower solution ({}, {})...".format(self.axis_min, self.axis_max))
        iterations = 0
        while (iterations < num_iterations):
            self.A_lower = 0.5 * (self.axis_max + self.axis_min)
            
            # Test for convergence
            if (np.abs(self.A_lower - self.axis_max) < tol):
                break

            psi = self.local_psi(self.x_lower[-1] - self.x_s, self.A_lower)
            psi_deriv_lower = self.local_psi_derivative(self.x_lower[-1] - self.x_s, self.A_lower)
            self.psi_sol_lower = self.solve_to_bnd(psi, psi_deriv_lower, self.x_lower[::-1])
            
            if (self.psi_sol_lower[-1, 0] > 0.0):
                self.axis_max = self.A_lower
            else:
                self.axis_min = self.A_lower

            # Increase iteration count
            iterations += 1
        if iterations >= num_iterations:
            print("WARNING: Bisection failed to converge")

    def find_delta(self, plot_output=False):
        if self.integrate_from_bnds:
            self.find_delta_from_boundaries(plot_output)
        else:
            self.find_delta_from_tearing_mode(plot_output)

        if self.integrate_from_bnds:
            self.x_upper = self.x_upper[::-1]
        else:
            self.x_lower = self.x_lower[::-1]

        psi_max = max(np.max(self.psi_sol_lower[:, 0]), np.max(self.psi_sol_upper[:, 0]))
        rs_idx = -1 if self.integrate_from_bnds else 0
        self.psi_rs = 0.5 * (self.psi_sol_lower[rs_idx, 0] + self.psi_sol_upper[rs_idx, 0]) / psi_max
        self.psi_sol_lower[:, 0] /= psi_max
        self.psi_sol_upper[:, 0] /= psi_max
        print("A_I, A_III: {}, {}".format(self.A_lower, self.A_upper))
        print('Delta: {}'.format(self.A_upper - self.A_lower))
        print('r_0 Delta: {}'.format(self.r_s * (self.A_upper - self.A_lower)))
        print("Psi_rs: {}".format(self.psi_rs))
        if plot_output:
            fig, ax = plt.subplots(2, sharex=True)
            ax[0].plot(self.x_lower, self.psi_sol_lower[:, 0])
            ax[0].plot(self.x_upper, self.psi_sol_upper[:, 0])
            ax[0].set_ylabel('$\Psi$')
            ax[0].set_xlabel('x')
            ax[0].set_xlim([0.0, 2.0])
            
            ax[1].plot(self.x_lower, self.psi_sol_lower[:, 1])
            ax[1].plot(self.x_upper, self.psi_sol_upper[:, 1])
            ax[1].set_ylabel('$\\frac{\partial \Psi}{\partial r}$')
            ax[1].set_xlabel('x')
            plt.show()

if __name__ == '__main__':
    solver = TearingModeSolver(2, 0.2, 10000, integrate_from_bnds=True)
    solver.find_delta(plot_output=True)

