"""
Author: Rohan Ramasamy
Date: 8th June 2016

Description: A python script containing fits to fusion reactions for DT, DD and D3He. These functions have been taken
from Atzeni - The Physics of Inertial Fusion, p19
"""

import numpy as np
from matplotlib import pyplot as plt

from plasma_physics.pysrc.utils.physical_constants import PhysicalConstants


class FusionReaction(object):
    """
    Class to store coefficients of fits
    """
    avogadro_constant = PhysicalConstants.avogadro_constant

    def __init__(self, C0, C1, C2, C3, C4, C5, C6, C7):
        self.C0 = C0
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.C6 = C6
        self.C7 = C7

        assert self.M1 is not None
        assert self.M2 is not None
        assert self.same_species_reaction is not None

    def __number_of_species_1(self, rho):
        return rho / (self.M1 + self.M2) * self.M1 * FusionReaction.avogadro_constant

    def __number_of_species_2(self, rho):
        return rho / (self.M1 + self.M2) * self.M2 * FusionReaction.avogadro_constant

    def number_density(self, rho):
        n_1 = self.__number_of_species_1(rho)
        n_2 = self.__number_of_species_2(rho)

        if self.same_species_reaction:
            assert np.all(n_1 == n_2), "{}, {}".format(n_1, n_2)
            return n_1 * n_2 / 2.0
        else:
            return n_1 * n_2


class DTReaction(FusionReaction):
    def __init__(self):
        self.same_species_reaction = False
        self.M1 = 2
        self.M2 = 3
        C0 = 6.6610
        C1 = 643.41e-16
        C2 = 15.136e-3
        C3 = 75.189e-3
        C4 = 4.6064e-3
        C5 = 13.5e-3
        C6 = -0.10675e-3
        C7 = 0.01366e-3
        super(DTReaction, self).__init__(C0, C1, C2, C3, C4, C5, C6, C7)


class DDReaction(FusionReaction):
    def __init__(self):
        self.same_species_reaction = True
        self.M1 = 2
        self.M2 = 2
        C0 = 6.2696
        C1 = 3.7212e-16
        C2 = 3.4127e-3
        C3 = 1.9917e-3
        C4 = 0
        C5 = 0.010506e-3
        C6 = 0
        C7 = 0
        super(DDReaction, self).__init__(C0, C1, C2, C3, C4, C5, C6, C7)


class D3HeReaction(FusionReaction):
    def __init__(self):
        self.same_species_reaction = False
        self.M1 = 2
        self.M2 = 3
        C0 = 10.572
        C1 = 151.16e-16
        C2 = 6.4192e-3
        C3 = -2.0290e-3
        C4 = -0.019108e-3
        C5 = 0.13578e-3
        C6 = 0
        C7 = 0
        super(D3HeReaction, self).__init__(C0, C1, C2, C3, C4, C5, C6, C7)


class pBReaction(FusionReaction):
    def __init__(self):
        self.same_species_reaction = False
        self.M1 = 11
        self.M2 = 1
        C0 = 17.708
        C1 = 6382e-16
        C2 = -59.357e-3
        C3 = 201.65e-3
        C4 = 1.0404e-3
        C5 = 2.7621e-3
        C6 = -0.0091653e-3
        C7 = 0.00098305e-3
        super(pBReaction, self).__init__(C0, C1, C2, C3, C4, C5, C6, C7)


class ReactivityFit(object):
    def __init__(self, fusion_reaction):
        assert isinstance(fusion_reaction, FusionReaction)
        self.reactants = fusion_reaction

    def zeta(self, T):
        return 1 - (self.reactants.C2 * T + self.reactants.C4 * T ** 2 + self.reactants.C6 * T ** 3) / \
                   (1 + self.reactants.C3 * T + self.reactants.C5 * T ** 2 + self.reactants.C7 * T ** 3)

    def epsilon(self, T):
        return self.reactants.C0 / (T ** (1.0 / 3.0))


class BoschHaleReactivityFit(ReactivityFit):
    """
    Class to generate reactivity approximation by Bosch and Hale 1992
    """
    def __init__(self, fusion_reaction):
        assert isinstance(fusion_reaction, FusionReaction)
        self.reactants = fusion_reaction

    def get_reactivity(self, T):
        zeta = self.zeta(T)
        epsilon = self.epsilon(T)
        return self.reactants.C1 * zeta ** (-5.0/6.0) * epsilon ** 2 * np.exp(-3 * zeta ** (1 / 3.0) * epsilon)


class NevinsSwainReactivityFit(ReactivityFit):
    """
    Class to generate reactivity approximation by Bosch and Hale 1992
    """
    def __init__(self, fusion_reaction):
        self.reactants = fusion_reaction

    def get_reactivity(self, T):
        zeta = self.zeta(T)
        epsilon = self.epsilon(T)
        
        reactivity = self.reactants.C1 * zeta ** (-5.0 / 6.0) * epsilon ** 2
        reactivity *= np.exp(-3.0 * zeta ** (1.0 / 3.0) * epsilon)
        reactivity += 5.41e-15 * T ** (-1.5) * np.exp(-148.0 / T)

        return reactivity


if __name__ == '__main__':
    dt_reaction = DTReaction()
    dd_reaction = DDReaction()
    d3he_reaction = D3HeReaction()
    pBReaction = pBReaction()
    dt_reactivity = BoschHaleReactivityFit(dt_reaction)
    dd_reactivity = BoschHaleReactivityFit(dd_reaction)
    d3he_reactivity = BoschHaleReactivityFit(d3he_reaction)
    pB_reactivity = NevinsSwainReactivityFit(pBReaction)

    T = np.logspace(0, 3, 200)
    dt_reactivities =  dt_reactivity.get_reactivity(T)
    dd_reactivities = dd_reactivity.get_reactivity(T)
    d3he_reactivities = d3he_reactivity.get_reactivity(T)
    pB_reactivities = pB_reactivity.get_reactivity(T)

    plt.figure(figsize=(12, 7))
    plt.loglog(T, dt_reactivities, label="DT")
    plt.loglog(T, dd_reactivities, label="DD")
    plt.loglog(T, d3he_reactivities, label="D3He")
    plt.loglog(T, pB_reactivities, label="pB")
    plt.title("Reactivity against Temperature for Different Fusion Reactions")
    plt.ylim([1e-20, 1e-14])
    plt.xlim([1, 100])
    plt.xlabel("T (keV)")
    plt.ylabel("<sigma> (cm^3/s)")
    plt.legend()
    plt.show()