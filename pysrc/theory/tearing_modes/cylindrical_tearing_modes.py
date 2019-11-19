"""
Author: Rohan Ramasamy
Date: 12/07/2019

A solution for the growth rate of resistive tearing modes in circular plasmas. The implementation is 
based on:

Tearing mode analysis in tokamaks, revisited - Y. Nishimura, J. D. Callen, and C. C. Hegna
Tearing mode in the cylindrical tokamak - H. P. Furth, P. H. Rutherford, and H. Selberg
Tokomaks - J. Wesson (section 6.8 to 6.9)
"""

import numpy as np
import sys
import os
from scipy import integrate, interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


class TearingModeSolverBase(object):
    def __init__(self):
        assert isinstance(self.m, int)
        assert isinstance(self.r_lower, np.ndarray)
        assert isinstance(self.r_upper, np.ndarray)


    def get_g1_and_g2(self, r):
        """
        Function to get g1 and g2 for radial location used in ode integration
        
        Arguments:
            r {np.ndarray} -- radial locations to calculate functions g1 and g2
        """
        raise NotImplementedError("Calling function from base class")

    def fit_A_from_psi(self, s, psi_s, psi_sol):
        """[summary]
        
        Arguments:
            s {np.ndarray} -- radial coordinate centred at singular surface
            psi_s {float} -- value of psi at singular surface
            psi_sol {np.ndarray} -- solution of psi in s coordinates
        """
        raise NotImplementedError("Calling function from base class")

    def get_psi(self, s, A, psi_s):
        """[summary]
        
        Arguments:
            s {np.ndarray} -- radial coordinate centred at singular surface
            A {float} -- coefficient of fit to psi at singular surface
            psi_s {float} -- value of psi at singular surface
        """
        raise NotImplementedError("Calling function from base class")

    def solve_to_bnd(self, psi, psi_deriv, x):
        """
        Function to solve tearing mode equation by integration from boundaries to singular surface

        Arguments:
            psi {float} -- initial condition for psi
            psi_deriv {float} -- initial condition for radial derivative of psi
            x {np.ndarray} -- points to solve ode
        """
        def integration_model(c, t):
            if t < 1e-6:
                dpsi_dt = c[1]
                d2psi_dt2 = 0.0
            else:
                g1, g2 = self.get_g1_and_g2(t)
                dpsi_dt = c[1]
                d2psi_dt2 = -g2 * c[1] + g1 * c[0]

            return [dpsi_dt, d2psi_dt2]

        return integrate.odeint(integration_model, [psi, psi_deriv], x, hmax=1e-2)

    def find_delta_from_boundaries(self, plot_output):
        """
        Solve delta from boundaries using a shooting algorithm
        
        Arguments:
            plot_output {bool} -- determines whether to plot output
        """
        # Get upper solution
        self.psi_sol_upper = self.solve_to_bnd(0, -1, self.r_upper[::-1])
        target_sol = self.psi_sol_upper[-1, 0] - self.delta * self.psi_sol_upper[-1, 1]

        # Find matching lower solution
        grad_0 = 1e4
        r_start = 0.0
        psi_start = 0.0
        self.psi_sol_lower = self.solve_to_bnd(psi_start, grad_0, self.r_lower)
        if self.psi_sol_lower[-1, 0] > target_sol:
            factor = 0.5
            while (self.psi_sol_lower[-1, 0] > target_sol):
                self.bnd_max = grad_0
                grad_0 *= factor
                self.psi_sol_lower = self.solve_to_bnd(psi_start, grad_0, self.r_lower)
            self.bnd_min = grad_0
        else:
            factor = 2.0
            while (self.psi_sol_lower[-1, 0] < target_sol):
                self.bnd_min = grad_0
                grad_0 *= factor
                self.psi_sol_lower = self.solve_to_bnd(psi_start, grad_0, self.r_lower)
            self.bnd_max = grad_0
        print("Upper and lower bound for bisection: {}, {}".format(self.bnd_max, self.bnd_min))
        
        iterations = 0
        num_iterations = 1000
        tol = 1e-12
        while (iterations < num_iterations):
            grad_0 = 0.5 * (self.bnd_max + self.bnd_min)
            
            # Test for convergence
            diff = np.abs(self.bnd_max - self.bnd_min) / np.abs(self.bnd_max)
            print("Iteration: {}, Difference: {}:".format(iterations, diff))
            if (diff < tol):
                break
    
            self.psi_sol_lower = self.solve_to_bnd(psi_start, grad_0, self.r_lower)
            if (self.psi_sol_lower[-1, 0] + self.delta * self.psi_sol_lower[-1, 1] < target_sol):
                self.bnd_min = grad_0  
            else:
                self.bnd_max = grad_0
            iterations += 1
        self.psi_sol_lower = self.solve_to_bnd(psi_start, grad_0, self.r_lower)
        
        if iterations == num_iterations: 
            print("WARNING: Bisection failed to converge!")
        else:
            print("Bisection converged in {} iterations: {}".format(iterations, grad_0))
        
        # Estimate A
        num_samples = 100
        psi_s = 0.5 * (self.psi_sol_lower[-1, 0] + self.delta * self.psi_sol_lower[-1, 1] + self.psi_sol_upper[-1, 0] - self.delta * self.psi_sol_upper[-1, 1])
        self.psi_sol_lower[:, 0] /= psi_s
        self.psi_sol_upper[:, 0] /= psi_s
        self.A_lower = self.fit_A_from_psi(self.r_lower[-num_samples:] - self.r_instability, 1.0, self.psi_sol_lower[-num_samples:, 0])
        self.A_upper = self.fit_A_from_psi(self.r_upper[:num_samples] - self.r_instability, 1.0, self.psi_sol_upper[-num_samples:, 0][::-1]) 

    def find_delta(self, plot_output=False):
        self.find_delta_from_boundaries(plot_output)
        self.r_upper = self.r_upper[::-1]

        psi_max = max(np.max(self.psi_sol_lower[:, 0]), np.max(self.psi_sol_upper[:, 0]))
        rs_idx = -1
        self.psi_rs = 0.5 * (self.psi_sol_lower[rs_idx, 0] + self.psi_sol_upper[rs_idx, 0]) / psi_max
        self.psi_sol_lower[:, 0] /= psi_max
        self.psi_sol_upper[:, 0] /= psi_max
        self.delta = self.A_upper - self.A_lower
        print("A_I, A_III: {}, {}".format(self.A_lower, self.A_upper))
        print('Delta: {}'.format(self.delta))
        print('r_0 Delta: {}'.format(self.r_instability * (self.A_upper - self.A_lower)))
        print("Psi_rs: {}".format(self.psi_rs))
        if plot_output:
            num_samples = 100
            psi_lower_local = self.get_psi(self.r_lower[-num_samples:] - self.r_instability, self.A_lower, self.psi_rs)
            psi_upper_local = self.get_psi(self.r_upper[-num_samples:] - self.r_instability, self.A_upper, self.psi_rs)
            fig, ax = plt.subplots(2, sharex=True)
            ax[0].plot(self.r_lower, self.psi_sol_lower[:, 0])
            ax[0].plot(self.r_upper, self.psi_sol_upper[:, 0])
            ax[0].plot(self.r_lower[-num_samples:], psi_lower_local, linestyle='--', c='r')
            ax[0].plot(self.r_upper[-num_samples:], psi_upper_local, linestyle='--', c='r')
            ax[0].set_ylabel('$\Psi$')
            ax[0].set_xlabel('r')
            ax[0].set_xlim([0.0, self.r_max])
            
            ax[1].plot(self.r_lower, self.psi_sol_lower[:, 1])
            ax[1].plot(self.r_upper, self.psi_sol_upper[:, 1])
            ax[1].set_ylabel('$\\frac{\partial \Psi}{\partial r}$')
            ax[1].set_xlabel('r')
            plt.show()


class TearingModeSolverNormalised(object):
    def __init__(self, m, x_s, num_pts, delta=1e-12):
        self.delta = delta
        self.m = m
        self.r_s = 1.0
        self.x_s = x_s
        r_max = 2 * self.r_s
        self.k = 1.0 / (20 * r_max)
        self.R = r_max / self.k
        # self.k = 1 / 20.0
        # self.R = 20.0
        print("kr_0: {}".format(self.k * r_max))
        print("R: {}".format(self.R))

        # Generate simulation domain and integration points
        x_max = 2
        self.x_lower = np.concatenate((np.linspace(0.0, 0.99 * self.x_s, num_pts),
                        np.linspace(0.99 * self.x_s, (1 - delta) * self.x_s, num_pts)))
        self.x_upper = np.concatenate((np.linspace((1 + delta) * self.x_s, 1.01 * self.x_s, num_pts),
                                        np.linspace(1.01 * self.x_s, 2, num_pts)))
        self.x_to_r = self.r_s
        self.q_0 = self.m / (1 + self.x_s ** 2)

        # Set up output variables
        self.bnd_max = None
        self.bnd_min = None
        self.A_lower = None
        self.A_upper = None
        self.psi_sol_lower = None
        self.psi_sol_upper = None

    def get_g1_and_g2(self, x):
        # Get input profiles
        b = x / (1 + x ** 2)
        q = self.q_0 * (1 + x ** 2)
        B_z0 = 1.0

        factor = 1 / (x ** 2 * self.k ** 2 + self.m ** 2 / self.x_to_r ** 2)
        self.H = x ** 3 * factor * self.x_to_r
        self.Hprime = (3 * x ** 2 - 2 * self.k ** 2 * x ** 4 * factor) * self.x_to_r * factor

        self.F = B_z0 / self.R * (1 - self.m / q)
        self.Fprime = B_z0 * self.m / (self.R * self.q_0) * 2 * x * (1 + x ** 2) ** -2
        self.Fprime2 = B_z0 * self.m / (self.R * self.q_0) * (2 * (1 + x ** 2) ** -2 - 8 * x ** 2 * (1 + x ** 2) ** -3)

        r = x * self.x_to_r
        factor = 1 / (r ** 2 * self.k ** 2 + self.m ** 2)
        self.g = self.k ** 2 * r ** 2 * factor 
        self.g *= r * self.F ** 2 + self.F * 2 * B_z0 * (self.k * r - self.m * b) * factor
        self.g += (self.m ** 2 - 1) * r * self.F ** 2 * factor

        g1 = 1 / self.H * (self.g / self.F ** 2 + (self.H * self.Fprime2 + self.Hprime * self.Fprime) / self.F)
        g2 = self.Hprime / self.H

        return g1, g2

    def solve_to_bnd(self, psi, psi_deriv, x):
        def integration_model(c, t):
            if t == 0:
                dpsi_dt = c[1]
                d2psi_dt2 = 0.0
            else:
                g1, g2 = self.get_g1_and_g2(t)
                dpsi_dt = c[1]
                d2psi_dt2 = -g2 * c[1] + g1 * c[0]

            return [dpsi_dt, d2psi_dt2]

        return integrate.odeint(integration_model, [psi, psi_deriv], x, hmax=1e-2)

    def get_psi(self, s, A, psi_s):
        kappa = 0.0
        return psi_s * (1.0 + (kappa * s + 0.5 * kappa ** 2 * s ** 2 - 0.75 * kappa ** 2 * s ** 2) * np.log(np.abs(s)) + \
            A * (s + 0.5 * kappa * s ** 2 + 1.0 / 12.0 * kappa ** 2 * s ** 3))

    def fit_A_from_psi(self, s, psi_s, psi_sol):
        kappa = 0.0
        def psi(s, A):
            return psi_s * (1.0 + (kappa * s + 0.5 * kappa ** 2 * s ** 2 - 0.75 * kappa ** 2 * s ** 2) * np.log(np.abs(s)) + \
                   A * (s + 0.5 * kappa * s ** 2 + 1.0 / 12.0 * kappa ** 2 * s ** 3))
            # return psi_s + A * s

        popt, pcov = curve_fit(psi, s, psi_sol)
        
        print("Error in fit for A: {}".format(pcov[0]))
        return popt[0]

    def find_delta_from_boundaries(self, plot_output):
        # Get upper solution
        self.psi_sol_upper = self.solve_to_bnd(0, -1, self.x_upper[::-1])
        target_sol = self.psi_sol_upper[-1, 0] - self.delta * self.psi_sol_upper[-1, 1]

        # Find matching lower solution
        grad_0 = 10
        x_start = 0.0
        psi_start = 0.0
        self.psi_sol_lower = self.solve_to_bnd(psi_start, grad_0, self.x_lower)
        if self.psi_sol_lower[-1, 0] > target_sol:
            factor = 0.5
            while (self.psi_sol_lower[-1, 0] > target_sol):
                self.bnd_max = grad_0
                grad_0 *= factor
                self.psi_sol_lower = self.solve_to_bnd(psi_start, grad_0, self.x_lower)
            self.bnd_min = grad_0
        else:
            factor = 2.0
            while (self.psi_sol_lower[-1, 0] < target_sol):
                self.bnd_min = grad_0
                grad_0 *= factor
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
    
            self.psi_sol_lower = self.solve_to_bnd(psi_start, grad_0, self.x_lower)
            if (self.psi_sol_lower[-1, 0] + self.delta * self.psi_sol_lower[-1, 1] < target_sol):
                self.bnd_min = grad_0  
            else:
                self.bnd_max = grad_0
            iterations += 1
        self.psi_sol_lower = self.solve_to_bnd(psi_start, grad_0, self.x_lower)
        
        if iterations == num_iterations: print("WARNING: Bisection failed to converge!")
        
        # Estimate A
        num_samples = 100
        psi_max = max(np.max(self.psi_sol_lower[:, 0]), np.max(self.psi_sol_upper[:, 0]))
        self.psi_sol_lower /= psi_max
        self.psi_sol_upper /= psi_max
        self.psi_rs = 0.25 * (self.psi_sol_lower[-1, 0] + self.delta * self.psi_sol_lower[-1, 1]) + 0.75 * (self.psi_sol_upper[-1, 0] - self.delta * self.psi_sol_upper[-1, 1])
        # assert self.psi_sol_lower[-1, 0] > self.psi_sol_upper[-1, 0], print(self.psi_sol_lower[-1, 0], self.psi_sol_upper[-1, 0])
        self.A_lower = self.fit_A_from_psi(self.x_lower[-num_samples:] - self.x_s, self.psi_rs, self.psi_sol_lower[-num_samples:, 0])
        self.A_upper = self.fit_A_from_psi(self.x_upper[:num_samples] - self.x_s, self.psi_rs, self.psi_sol_upper[-num_samples:, 0][::-1])

    def find_delta(self, plot_output=False):
        self.find_delta_from_boundaries(plot_output)
        self.x_upper = self.x_upper[::-1]

        print("A_I, A_III: {}, {}".format(self.A_lower, self.A_upper))
        print('Delta: {}'.format(self.A_upper - self.A_lower))
        print('r_0 Delta: {}'.format(self.r_s * (self.A_upper - self.A_lower)))
        print("Psi_rs: {}".format(self.psi_rs))
        if plot_output:
            num_samples = 100
            psi_lower_local = self.get_psi(self.x_lower[-num_samples:] - self.x_s, self.A_lower, self.psi_rs)
            psi_upper_local = self.get_psi(self.x_upper[-num_samples:] - self.x_s, self.A_upper, self.psi_rs)
            fig, ax = plt.subplots(2, sharex=True)
            ax[0].plot(self.x_lower, self.psi_sol_lower[:, 0])
            ax[0].plot(self.x_upper, self.psi_sol_upper[:, 0])
            ax[0].plot(self.x_lower[-num_samples:], psi_lower_local, linestyle='--', c='r')
            ax[0].plot(self.x_upper[-num_samples:], psi_upper_local, linestyle='--', c='r')
            ax[0].set_ylabel('$\Psi$')
            ax[0].set_xlabel('x')
            ax[0].set_xlim([0.0, 2.0])
            
            ax[1].plot(self.x_lower, self.psi_sol_lower[:, 1])
            ax[1].plot(self.x_upper, self.psi_sol_upper[:, 1])
            ax[1].set_ylabel('$\\frac{\partial \Psi}{\partial r}$')
            ax[1].set_xlabel('x')
            plt.show()


class TearingModeSolver(TearingModeSolverBase):
    def __init__(self, m, r, r_max, B_theta, B_z, j_phi, q, num_pts, delta=1e-12):
        self.delta = delta
        self.m = m
        self.r_max = r_max
        
        # Find instability location
        self.r_to_q = interpolate.interp1d(r, q)
        self.q_to_r = interpolate.interp1d(q, r)
        self.r_instability = self.q_to_r(m)

        # Generate simulation domain and integration points
        self.r_lower = np.concatenate((np.linspace(0.0, 0.99 * self.r_instability, num_pts),
                        np.linspace(0.99 * self.r_instability, (1 - delta) * self.r_instability, num_pts)))
        self.r_upper = np.concatenate((np.linspace((1 + delta) * self.r_instability, 1.01 * self.r_instability, num_pts),
                                        np.linspace(1.01 * self.r_instability, r_max, num_pts)))
        
        # Generate other interpolators
        self.r_to_B_theta = interpolate.interp1d(r, B_theta)
        self.r_to_B_z = interpolate.interp1d(r, B_z)
        self.r_to_j_phi = interpolate.interp1d(r, j_phi)
        j_phi_deriv = np.gradient(j_phi, r)
        self.r_to_j_phi_deriv = interpolate.interp1d(r, j_phi_deriv)
        q_deriv = np.gradient(q, r)
        self.r_to_q_deriv = interpolate.interp1d(r, q_deriv)

        g2 = 1.0 / r
        g1 = self.r_to_j_phi_deriv(r)
        g1 /= self.r_to_B_theta(r) / PhysicalConstants.mu_0 * (1 - self.r_to_q(r) / self.m)
        g1 += self.m ** 2 / r ** 2

        # Set up output variables
        self.bnd_max = None
        self.bnd_min = None
        self.A_lower = None
        self.A_upper = None
        self.psi_sol_lower = None
        self.psi_sol_upper = None

        super(TearingModeSolver, self).__init__()

    def get_g1_and_g2(self, r):
        g2 = 1.0 / r
        g1 = self.r_to_j_phi_deriv(r)
        g1 /= self.r_to_B_theta(r) / PhysicalConstants.mu_0 * (1 - self.r_to_q(r) / self.m)
        g1 += self.m ** 2 / r ** 2

        return g1, g2
    
    def get_psi(self, s, A, psi_s):
        kappa = -PhysicalConstants.mu_0 * self.r_to_j_phi_deriv(self.r_instability) * self.r_to_q(self.r_instability)
        kappa /= self.r_to_B_theta(self.r_instability) * self.r_to_q_deriv(self.r_instability)
        return psi_s * (1.0 + kappa * s * np.log(np.abs(s))) + A * (s + 0.5 * kappa * s ** 2)

    def fit_A_from_psi(self, s, psi_s, psi_sol):
        kappa = -PhysicalConstants.mu_0 * self.r_to_j_phi_deriv(self.r_instability) * self.r_to_q(self.r_instability)
        kappa /= self.r_to_B_theta(self.r_instability) * self.r_to_q_deriv(self.r_instability)
        def psi(s, A):
            return psi_s * (1.0 + kappa * s * np.log(np.abs(s))) + A * (s + 0.5 * kappa * s ** 2)

        popt, pcov = curve_fit(psi, s, psi_sol)
        
        print("Error in fit for A: {}".format(pcov[0]))
        return popt[0]

    def get_gamma(self):
        """
        Get gamma normalised by density and resistivity.
        """
        B_theta = self.r_to_B_theta(self.r_instability)
        q_deriv = self.r_to_q_deriv(self.r_instability)
        q = self.r_to_q(self.r_instability)

        gamma = self.delta ** 0.8
        gamma *= (q_deriv / (self.r_instability * q)) ** 0.4
        gamma *= (self.m * B_theta / np.sqrt(PhysicalConstants.mu_0)) ** 0.4
        gamma /= PhysicalConstants.mu_0 ** 0.6
        gamma *= 0.55

        self.gamma = gamma
            

if __name__ == '__main__':
    num_pts = 100000
    m = 2
    solver = TearingModeSolverNormalised(m, 1.0, num_pts)
    solver.find_delta(plot_output=True)

