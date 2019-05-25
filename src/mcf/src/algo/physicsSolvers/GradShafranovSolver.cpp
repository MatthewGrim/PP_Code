/**
Author: Rohan Ramasamy
Date: 24/05/2019
**/

#include <mcf/src/algo/physicsSolvers/GradShafranovSolver.h>
#include <mcf/src/algo/interpolators/Interpolator1D.h>
#include <mcf/src/macros.h>

#include <deal.II/grid/tria.h>
#include <deal.II/base/point.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

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
        quadrature_formula(QUADRULE),
        face_quadrature_formula(QUADRULE) {}

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
    solveIteration()
    {
        //For volume integration/quadrature points
        dealii::FEValues<DIM> fe_values (fe,
                    quadrature_formula, 
                    dealii::update_values | 
                    dealii::update_gradients | 
                    dealii::update_JxW_values);

        //For surface integration/quadrature points
        dealii::FEFaceValues<DIM> fe_face_values (fe,
                            face_quadrature_formula, 
                            dealii::update_values | 
                            dealii::update_quadrature_points | 
                            dealii::update_JxW_values);

        // Zero matrices
        K=0; F=0;
        const unsigned int dofs_per_elem = fe.dofs_per_cell;                      //This gives you dofs per element
        const unsigned int nodes_per_elem = dealii::GeometryInfo<DIM>::vertices_per_cell;
        const unsigned int num_quad_pts = quadrature_formula.size();              //Total number of quad points in the element
        const unsigned int num_face_quad_pts = face_quadrature_formula.size();    //Total number of quad points in the face
        const unsigned int faces_per_elem = dealii::GeometryInfo<DIM>::faces_per_cell;
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
                
            // Define Klocal and Flocal
            Klocal = 0.0;
            //Loop over local DOFs and quadrature points to populate Klocal
            //Note that all quadrature points are included in this single loop
            for (unsigned int q=0; q < num_quad_pts; ++q){
                //evaluate elemental stiffness matrix, K^{AB}_{ik} = \integral N^A_{,j}*C_{ijkl}*N^B_{,l} dV 
                for (unsigned int A=0; A < nodes_per_elem; A++) { //Loop over nodes
                    for(unsigned int i=0; i < DIM; i++){ //Loop over nodal dofs
                        for (unsigned int B=0; B < nodes_per_elem; B++) {
                            for(unsigned int k=0; k < DIM; k++){
                                for (unsigned int j = 0; j < DIM; j++){
                                    for (unsigned int l = 0; l < DIM; l++){
                                        /*//EDIT - You need to define Klocal here. Note that the indices of Klocal are the element dof numbers (0 through 23),
                                        which you can calculate from the element node numbers (0 through 8) and the nodal dofs (0 through 2).
                                        You'll need the following information:
                                        basis gradient vector: fe_values.shape_grad(elementDOF,q), where elementDOF is DIM*A+i or DIM*B+k
                                        NOTE: this is the gradient with respect to the real domain (not the bi-unit domain)
                                        elasticity tensor: use the function C(i,j,k,l)
                                        det(J) times the total quadrature weight: fe_values.JxW(q)*/
                                        unsigned int DOF_A_idx = A * DIM + i;
                                        unsigned int DOF_B_idx = B * DIM + k;
                                        double basis_gradient_A = fe_values.shape_grad(DOF_A_idx, q)[j];
                                        double basis_gradient_B = fe_values.shape_grad(DOF_B_idx, q)[l];
                                        double det_JxW = fe_values.JxW(q);
                                        Klocal[DOF_A_idx][DOF_B_idx] += basis_gradient_B * basis_gradient_A * det_JxW;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            //Loop over faces (for Neumann BCs), local DOFs and quadrature points to populate Flocal.
            Flocal = 0.0;

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

        throw std::runtime_error("Method has not been implemented yet!");
    }

    void 
    GradShafranovSolver::
    outputResults()
    {
        throw std::runtime_error("Method has not been implemented yet!");
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