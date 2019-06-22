/**
 * Author: Rohan Ramasamy
 * Date: 24/05/2019
 **/ 

#include <mcf/src/algo/physicsSolvers/GradShafranovSolver.h>
#include <mcf/src/utils/PhysicalConstants.h>

#include <vector>

int main() {
    std::vector<double> psi = {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0}; 
    std::vector<double> pressure(psi.size(), 0.0);
    std::vector<double> ffprime(psi.size(), 1.1 * 1.1);
    for (unsigned int i = 0; i < pressure.size(); ++i) {
        pressure[i] = 1.0 / mcf::MU_0 * psi[i];
    }
    mcf::GridType gridType = mcf::GridType::solovievBoundary;

    mcf::GradShafranovSolver gradShafranovSolver(1.1, 0.0, 1.0, 5, gridType);
    gradShafranovSolver.solveGradShafranov(psi, pressure, ffprime);

    return 0;
}