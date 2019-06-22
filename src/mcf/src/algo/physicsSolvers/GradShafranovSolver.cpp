/**
Author: Rohan Ramasamy
Date: 24/05/2019
**/

#include <mcf/src/algo/physicsSolvers/GradShafranovSolver.h>
#include <mcf/src/macros.h>
#include <mcf/src/utils/PhysicalConstants.h>

#include <deal.II/base/point.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

// #include <deal.II/fe/component_mask.h>

#include <stdexcept>
#include <cassert>


namespace mcf {
    /**
     * Transform function used to generate plasma boundary of Soloviev problem
     **/
    struct SolovevTransformFunc {
        dealii::Point<GradShafranovSolver::DIM> operator() (const dealii::Point<GradShafranovSolver::DIM> &in) const
        {
            double R0 = 1.1;
            double a = 0.5;

            double x = in[0] - R0;
            double y = in[1];
            double r = sqrt(x * x + y * y);

            double alpha;
            if ((x == 0.0) && (y == 0.0)) {
                return dealii::Point<GradShafranovSolver::DIM> (R0, 0.0);
            }
            else {
                alpha = atan(y / x);
                if (x < 0.0) {
                    if (y < 0.0) {
                        alpha = -3.14159265 + alpha;
                    }
                    else {
                        alpha = 3.14159265 + alpha;
                    }
                }

                double rBound = R0 * sqrt(1 + 2 * a * cos(alpha) / R0);
                double zBound = a * R0 * sin(alpha);

                return dealii::Point<GradShafranovSolver::DIM> (R0 + r * (rBound - R0), r * zBound);
            }
        }
    };

    GradShafranovSolver::
    GradShafranovSolver(
        const double& centreX,
        const double& centreY,
        const double& radius,
        const int& resolution,
        const GridType& gridType
        ) :
        mCentreX(centreX),
        mCentreY(centreY),
        mRadius(radius),
        mResolution(resolution),
        fe(dealii::FE_Q<DIM>(ORDER)),
        dof_handler (mTriangulation),
        quadrature_formula(QUADRULE),
        mGridType(gridType) 
    {
        f0 = 1.0;
        p0 = f0 / MU_0;
        R0 = 1.1;
        a = 0.5;
        // For output
        //Nodal Solution names - this is for writing the output file
        nodal_solution_names.push_back("psi");
        nodal_data_component_interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);

        // Set grid name for outputs
        switch (mGridType) {
            case GridType::rectangular    : mGridName = "rectangular"; break;
            case GridType::circular       : mGridName = "circular"; break;
            case GridType::solovievBoundary : mGridName = "solovievBoundary"; break;
        }
    }

    void
    GradShafranovSolver:: 
    makeGrid() 
    {
        if (mGridType == GridType::rectangular) {
            std::cout << "\tMaking rectangular grid" << std::endl;
            dealii::Point<DIM> lowerLeft(0.0, -0.7);
            dealii::Point<DIM> upperRight(1.5, 0.7);
            dealii::GridGenerator::hyper_rectangle (mTriangulation, lowerLeft, upperRight);
            mTriangulation.refine_global(mResolution);
        }
        else if (mGridType == GridType::circular) {
            std::cout << "\tMaking circular grid" << std::endl;
            const dealii::Point<DIM> centre (mCentreX, mCentreY);
            dealii::GridGenerator::hyper_ball(mTriangulation, centre, mRadius);
            
            mTriangulation.refine_global(mResolution);
        }
        else if (mGridType == GridType::solovievBoundary) {
            std::cout << "\tMaking plasma boundary grid" << std::endl;
            //radius of initial grid needs to be one for transform to correctly map points to new boundary
            assert(mRadius == 1.0);
            
            const dealii::Point<DIM> centre(mCentreX, mCentreY);
            dealii::GridGenerator::hyper_ball(mTriangulation, centre, mRadius);
            mTriangulation.refine_global(mResolution);
            // Transform grid to plasma shape
            dealii::GridTools::transform(SolovevTransformFunc(), mTriangulation);
        }
        else {
            throw std::runtime_error("Chosen grid type is not implemented yet!");
        }
#if DEBUG
        std::ofstream out ("grid-1.eps");
        dealii::GridOut grid_out;
        grid_out.write_eps (mTriangulation, out);
        std::cout << "\tGrid written to grid-1.eps" << std::endl;
#endif
    }

    void
    GradShafranovSolver::
    setUpSystem()
    {
        //Let deal.II organize degrees of freedom
        dof_handler.distribute_dofs (fe);

        //Get a vector of global degree-of-freedom x-coordinates
        dealii::MappingQ1<DIM,DIM> mapping;
        std::vector< dealii::Point<DIM,double> > dof_coords(dof_handler.n_dofs());
        dofLocation.reinit(dof_handler.n_dofs(),DIM);
        dealii::DoFTools::map_dofs_to_support_points<DIM,DIM>(mapping,dof_handler,dof_coords);
        for(unsigned int i=0; i<dof_coords.size(); i++){
            for(unsigned int j=0; j<DIM; j++){
            dofLocation[i][j] = dof_coords[i][j];
            }
        }

        //Define the size of the global matrices and vectors
        sparsity_pattern.reinit (dof_handler.n_dofs(), 
                                 dof_handler.n_dofs(),
                                 dof_handler.max_couplings_between_dofs());
        dealii::DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern);
        sparsity_pattern.compress();
        K.reinit (sparsity_pattern);
        D.reinit(dof_handler.n_dofs());
        F.reinit(dof_handler.n_dofs());
        Derror.reinit(dof_handler.n_dofs());

        // Initial condition
        for(unsigned int i=0; i < dof_handler.n_dofs(); i++){
            double x = dofLocation[i][0];
            double y = dofLocation[i][1];
            D[i] = f0 * R0 * R0 * a * a / 2.0 * (1.0 - y * y / (a * a) - std::pow((x - R0) / a + (x - R0) * (x - R0) / (2 * a * R0), 2.0));
        }

#if DEBUG
        std::cout << "\tNumber of active elems:       " << mTriangulation.n_active_cells() << std::endl;
        std::cout << "\tNumber of degrees of freedom: " << dof_handler.n_dofs() << std::endl;  
#endif
    }

    void 
    GradShafranovSolver::
    applyBoundaryConditions()
    {
        const unsigned int totalDOFs = dof_handler.n_dofs(); //Total number of degrees of freedom
        double tol = mRadius * 1e-6;

        std::vector<bool> is_boundary_dofs (totalDOFs);
        dealii::DoFTools::extract_boundary_dofs (dof_handler,
                                        dealii::ComponentMask(),
                                        is_boundary_dofs);
        for (unsigned int globalDOF = 0; globalDOF < totalDOFs; globalDOF++) {
            if (is_boundary_dofs[globalDOF]) {
                double x = dofLocation[globalDOF][0];
                double y = dofLocation[globalDOF][1];
                boundary_values[globalDOF] = f0 * R0 * R0 * a * a / 2.0 * (1.0 - y * y / (a * a) - std::pow((x - R0) / a + (x - R0) * (x - R0) / (2 * a * R0), 2.0));
            }
        }

        dealii::MatrixTools::apply_boundary_values (boundary_values, K, D, F);
    }

    void
    GradShafranovSolver::
    findAxis()
    {
        throw std::runtime_error("Not implemented yet!");
    }

    void 
    GradShafranovSolver::
    assembleMatrix(
        const Interpolator1D& pInterp,
        const Interpolator1D& ffPrimeInterp
        )
    {
        //For volume integration/quadrature points
        dealii::FEValues<DIM> fe_values (fe,
                    quadrature_formula, 
                    dealii::update_values | 
                    dealii::update_gradients | 
                    dealii::update_JxW_values);
        const dealii::Quadrature<DIM> quadrature = fe_values.get_quadrature();
        const dealii::Mapping<DIM> *mapping = &fe_values.get_mapping();

        // Zero matrices
        K=0; F=0;

        // Get element size and number of quadrature points
        const unsigned int dofs_per_elem = fe.dofs_per_cell;                               
        const unsigned int num_quad_pts = quadrature_formula.size();                        
        #if DEBUG
                std::cout << "\tNumber of Quad points: " << std::to_string(num_quad_pts) << std::endl;
                std::cout << "\tDegrees of Freedom per Element: " << std::to_string(dofs_per_elem) << std::endl;
        #endif
        // Define local matrices and mapping between them
        dealii::FullMatrix<double> Klocal (dofs_per_elem, dofs_per_elem);
        dealii::Vector<double>     Flocal (dofs_per_elem);
        std::vector<unsigned int> local_dof_indices (dofs_per_elem);              //This relates local dof numbering to global dof numbering

        //loop over elements  
        typename dealii::DoFHandler<DIM>::active_cell_iterator elem = dof_handler.begin_active(), endc = dof_handler.end();
        for (; elem!=endc; ++elem){
            //Update fe_values for the current element
            fe_values.reinit(elem);

            //Retrieve the effective "connectivity matrix" for this element
            elem->get_dof_indices (local_dof_indices);
                
            // Zero Klocal and Flocal
            Klocal = 0.0, Flocal = 0.0;
            //Loop over local DOFs and quadrature points to populate Klocal
            //Note that all quadrature points are included in this single loop
            for (unsigned int q=0; q < num_quad_pts; ++q){
                const dealii::Point<DIM> quadPoint = quadrature.point(q);
                dealii::Point<DIM> realQuadPoint = mapping->transform_unit_to_real_cell(elem, quadPoint);
                double R = realQuadPoint[0];

                // Get local matrix
                for (unsigned int i=0; i < dofs_per_elem; ++i) {
                    for (unsigned int j=0; j < dofs_per_elem; ++j) {
                        if (R == 0.0) R = 1e-10;
                        double contribution = -(fe_values.shape_grad (i, q) *
                                               fe_values.shape_grad (j, q) * 
                                               fe_values.JxW(q)) / R;
                        
                        Klocal(i, j) +=  contribution;
                    }
                }
                
                // Get local forcing function
                for (unsigned int i=0; i < dofs_per_elem; ++i) {
                    double psi = D[i];
                    double pressure = p0;
                    double ffPrime = f0 * R0 * R0;

                    double contribution = -(MU_0 * R * R * pressure + ffPrime);
                    contribution *= fe_values.shape_value(i, q) / R;
                    contribution *= fe_values.JxW(q);
                    Flocal(i) += contribution;
                }
            }

            //Assemble local K and F into global K and F
            unsigned int f_global_index, k_global_index;
            for(unsigned int i=0; i<dofs_per_elem; ++i){
                f_global_index = local_dof_indices[i];
                F[f_global_index] += Flocal(i);
                for(unsigned int j=0; j<dofs_per_elem; ++j){
                    k_global_index = local_dof_indices[j];
                    K.add(f_global_index, k_global_index, Klocal(i, j));
                }
            }
        }
    }

    void
    GradShafranovSolver::
     solveIteration()
    {
        dealii::SparseDirectUMFPACK  A;
        A.initialize(K);
        A.vmult (D, F);
    }

    void 
    GradShafranovSolver::
    outputResults(
        const int& i,
        const bool& isError
    )
    {
        //Write results to VTK file
        std::ofstream output1 ("solution_" + mGridName + std::to_string(i) + ".vtk");
        dealii::DataOut<DIM> data_out; data_out.attach_dof_handler (dof_handler);

        //Add nodal DOF data
        if (isError) {
            data_out.add_data_vector (Derror,
                            nodal_solution_names,
                            dealii::DataOut<DIM>::type_dof_data,
                            nodal_data_component_interpretation);
        }
        else {
            data_out.add_data_vector (D,
                            nodal_solution_names,
                            dealii::DataOut<DIM>::type_dof_data,
                            nodal_data_component_interpretation);
        }
        data_out.build_patches();
        data_out.write_vtk(output1);
        output1.close();
    }

    void
    GradShafranovSolver::
    computeError()
    {
        double l2Error = 0.0;
        for(unsigned int i=0; i < dof_handler.n_dofs(); i++){
            double x = dofLocation[i][0];
            double y = dofLocation[i][1];
            double sol = f0 * R0 * R0 * a * a / 2.0 * (1.0 - y * y / (a * a) - std::pow((x - R0) / a + (x - R0) * (x - R0) / (2 * a * R0), 2.0));
            Derror[i] = fabs(D[i] - sol); 
            l2Error += Derror[i];
        }   
        l2Error = l2Error / dof_handler.n_dofs();
        std::cout << "\tMean relative error: " << std::to_string(l2Error) << std::endl;

        outputResults(-1, true);
    }

    void 
    GradShafranovSolver::
    solveGradShafranov(
        const std::vector<double> psi,
        const std::vector<double> pressure,
        const std::vector<double> ffPrime
        ) 
    {
        const Interpolator1D pInterp = Interpolator1D(psi, pressure);
        const Interpolator1D ffPrimeInterp = Interpolator1D(psi, ffPrime);

        std::cout << "Building grid..." << std::endl;
        makeGrid();

        std::cout << "Setting up FE system..." << std::endl;
        setUpSystem();

        std::cout << "Output results..." << std::endl;
        outputResults(99);

        for (int i = 0; i < 1; ++i) {
            // std::cout << "Finding Axis..." << std::endl;
            // findAxis();

            std::cout << "Assembling matrix..." << std::endl;
            assembleMatrix(pInterp, ffPrimeInterp);

            std::cout << "Initialise boundary conditions..." << std::endl;
            applyBoundaryConditions();
            
            std::cout << "Solving..." << std::endl;
            solveIteration();
        
            std::cout << "Output results..." << std::endl;
            outputResults(i);
        }
        computeError();
    }
}