/**
Author: Rohan Ramasamy
Date: 24/05/2019
**/

#include <mcf/src/algo/interpolators/Interpolator1D.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
//Mesh related classes
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
//Finite element implementation classes
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <vector>


namespace mcf {
   class GradShafranovSolver
   {
      public:
         // Grad Shafranov is fundamentally a 2D problem
         static const int DIM = 2;
         static const int ORDER = 2;
         static const int QUADRULE = 2;
         static constexpr double MU_0 = 1.25663706e-6;

         /**
          * Initialise Grad Shafranov solver with grid parameters
          **/
         GradShafranovSolver(
            const double& centreX,
            const double& centreY,
            const double& radius,
            const int& resolution
            );

         /**
          * Solve Grad shafranov equation on grid given input profiles
          **/
         void 
         solveGradShafranov(
            const std::vector<double> psi,
            const std::vector<double> pressure,
            const std::vector<double> ffPrime
            );

         /**
          * Output results from Grad Shafranov solver
          **/
         void 
         outputResults();

      private:
         /**
          * Internal function to set up Deal II to solve system
          **/
         void 
         setUpSystem();
         
         /**
          * Use grid parameters to construct grid
          **/
         void 
         makeGrid();

         /**
          * Set up boundary conditions for grid
          **/
         void 
         initialiseBoundaryConditions();

         /**
          * Carry out single Picard iteration
          **/
         void 
         solveIteration(
            const Interpolator1D& pInterp,
            const Interpolator1D& ffPrimeInterp
            );

         double mCentreX, mCentreY, mRadius;
         int mResolution;

         // Finite element data structures
         dealii::Triangulation<DIM> mTriangulation;            // Grid triangulation
         dealii::FESystem<DIM>      fe;                 
         dealii::DoFHandler<DIM>    dof_handler;

         dealii::QGauss<DIM>   quadrature_formula;             //Quadrature
         dealii::Table<2,double>	        dofLocation;	         //Table of the coordinates of dofs by global dof number
         std::map<unsigned int,double> boundary_values;        //Map of dirichlet boundary conditions
                  
         // Matrix Data structures
         dealii::SparsityPattern      sparsity_pattern;        //Sparse matrix pattern
         dealii::SparseMatrix<double> K;                       //Global stiffness matrix - Sparse matrix - used in the solver
         dealii::Vector<double>       D, F;                    //Global vectors - Solution vector (D) and Global force vector (F)
   
         // Output variables
         std::vector<std::string> nodal_solution_names;
         std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> nodal_data_component_interpretation;
   };
}