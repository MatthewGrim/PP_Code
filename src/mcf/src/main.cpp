/**
 * Author: Rohan Ramasamy
 * Date: 24/05/2019
 **/ 

#include <mcf/src/algo/physicsSolvers/GradShafranovSolver.h>

#include <vector>

int main() {
    std::vector<double> a(8, 0.0); 
    std::vector<double> b(8, 0.0);
    std::vector<double> c(8, 0.0);

    mcf::GradShafranovSolver gradShafranovSolver(10.0, 0.0, 1.0, 2);
    gradShafranovSolver.solveGradShafranov(a, b, c);

    return 0;
}