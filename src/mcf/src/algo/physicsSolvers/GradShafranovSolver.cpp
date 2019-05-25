/**
Author: Rohan Ramasamy
Date: 24/05/2019
**/

#include <mcf/src/algo/physicsSolvers/GradShafranovSolver.h>
#include <mcf/src/macros.h>

#include <deal.II/grid/tria.h>
#include <deal.II/base/point.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <stdexcept>


namespace mcf {
    GradShafranovSolver::
    GradShafranovSolver(
        const double& centreX,
        const double& centreY,
        const double& radius,
        const int& resolution
        ) :
        mCentreX(centreX),
        mCentreY(centreY),
        mRadius(radius),
        mResolution(resolution),
        fe(dealii::FE_Q<DIM>(ORDER), DIM),
        dof_handler (mTriangulation),
        quadrature_formula(QUADRULE) 
    {
        // For output
        //Nodal Solution names - this is for writing the output file
        nodal_solution_names.push_back("psi");
        nodal_data_component_interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    }

    void
    GradShafranovSolver:: 
    makeGrid() 
    {
        const dealii::Point<DIM> centre (mCentreX, mCentreY, 0);
        dealii::GridGenerator::hyper_ball(mTriangulation, centre, mRadius);
        
        mTriangulation.refine_global(mResolution);

#if DEBUG
        std::ofstream out ("grid-1.eps");
        dealii::GridOut grid_out;
        grid_out.write_eps (mTriangulation, out);
        std::cout << "Grid written to grid-1.eps" << std::endl;
#endif
    }

    void 
    GradShafranovSolver::
    initialiseBoundaryConditions()
    {
        const unsigned int totalDOFs = dof_handler.n_dofs(); //Total number of degrees of freedom

        for(unsigned int globalDOF = 0; globalDOF < totalDOFs; globalDOF++){
            double x = dofLocation[globalDOF][0];
            double y = dofLocation[globalDOF][1];
            double r = x * x + y * y;
            // Apply Dirichlet boundary condition on outer boundary = 0.0
            if (r == mRadius * mRadius) {
                boundary_values[globalDOF] = 0.0;
            }
        }
    }

    void 
    GradShafranovSolver::
    solveIteration(
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

        // Zero matrices
        K=0; F=0;

        // Get element size and number of quadrature points
        const unsigned int dofs_per_elem = fe.dofs_per_cell;                               
        const unsigned int num_quad_pts = quadrature_formula.size();                        
        
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
                // Get local matrix
                for (unsigned int i=0; i < dofs_per_elem; ++i) {
                    for (unsigned int j=0; j < dofs_per_elem; ++j) {
                        auto globalDOF = local_dof_indices[i];
                        double R = dofLocation[globalDOF][0];
                        double contribution = -(fe_values.shape_grad (i, q) *
                                               fe_values.shape_grad (j, q) * 
                                               fe_values.JxW (q)) / R;
                        
                        Klocal(i, j) +=  contribution;
                    }
                }
                
                // Get local forcing function
                for (unsigned int i=0; i < dofs_per_elem; ++i) {
                    auto globalDOF = local_dof_indices[i];
                    double R = dofLocation[globalDOF][0];
                    double psi = D[globalDOF];
                    double pressure = pInterp.interpY(psi);
                    double ffPrime = ffPrimeInterp.interpY(psi);

                    double contribution = -(MU_0 * R * R * pressure + ffPrime);
                    contribution *= fe_values.shape_value(i, q) / R;
                    contribution *= fe_values.JxW(q);
                    Flocal(i) += contribution;
                }
            }

            //Assemble local K and F into global K and F
            unsigned int f_global_index, k_global_index;
            for(unsigned int i=0; i<dofs_per_elem; i++){
                f_global_index = local_dof_indices[i];
                F[f_global_index] += Flocal[i];
                for(unsigned int j=0; j<dofs_per_elem; j++){
                    k_global_index = local_dof_indices[j];
                    K.add(f_global_index, k_global_index, Klocal[i][j]);
                }
            }
        }

        //Let deal.II apply Dirichlet conditions WITHOUT modifying the size of K and F global
        dealii::MatrixTools::apply_boundary_values (boundary_values, K, D, F, false);
    
        dealii::SolverControl solver_control (1000, 1e-12);
        dealii::SolverCG<> solver(solver_control);
        solver.solve (K, D, F, dealii::PreconditionIdentity());
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

        initialiseBoundaryConditions();
        solveIteration(pInterp, ffPrimeInterp);
        outputResults();
    }

    void 
    GradShafranovSolver::
    outputResults()
    {
        //Write results to VTK file
        std::ofstream output1 ("solution.vtk");
        DataOut<dim> data_out; data_out.attach_dof_handler (dof_handler);

        //Add nodal DOF data
        data_out.add_data_vector (D,
                        nodal_solution_names,
                        DataOut<dim>::type_dof_data,
                        nodal_data_component_interpretation);
        data_out.build_patches();
        data_out.write_vtk(output1);
        output1.close();
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

        // set up boundary conditions
        initialiseBoundaryConditions();

        //Define the size of the global matrices and vectors
        sparsity_pattern.reinit (dof_handler.n_dofs(), 
                                 dof_handler.n_dofs(),
                                 dof_handler.max_couplings_between_dofs());
        dealii::DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern);
        sparsity_pattern.compress();
        K.reinit (sparsity_pattern);
        D.reinit(dof_handler.n_dofs());
        F.reinit(dof_handler.n_dofs());

#if DEBUG
        std::cout << "   Number of active elems:       " << mTriangulation.n_active_cells() << std::endl;
        std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;  
#endif
    }
}