/**
Author: Rohan Ramasamy
Date: 24/05/2019
**/

#include <mcf/src/algo/interpolators/Interpolator1D.h>

// DEAL II
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

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <vector>


namespace mcf {
   enum class GridType {rectangular, circular, solovievBoundary};

   class GradShafranovSolver
   {
      public:
         // Grad Shafranov is fundamentally a 2D problem
         static const int DIM = 2;
         static const int ORDER = 1;
         static const int QUADRULE = 2;

         /**
          * Initialise Grad Shafranov solver with grid parameters
          **/
         GradShafranovSolver(
            const double& centreX,
            const double& centreY,
            const double& radius,
            const int& resolution,
            const GridType& gridType
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
         outputResults(
            const int& i,
            const bool& isError=false
         );

         /**
          * Compute error between solution and known analytic solution
          **/
         void
         computeError();
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
         applyBoundaryConditions();

         /**
          * Find psi value on axis 
          */
         void
         findAxis();

         /**
          * Construct matrices for solution
          **/
         void 
         assembleMatrix(
            const Interpolator1D& pInterp,
            const Interpolator1D& ffPrimeInterp
            );

         /**
          * Carry out single Picard iteration
          **/
         void
         solveIteration();

         double f0, p0, R0, a;
         double mCentreX, mCentreY, mRadius;
         int mResolution;
         double psiAxis;

         // Finite element data structures
         dealii::Triangulation<DIM> mTriangulation;            // Grid triangulation
         dealii::FE_Q<DIM>      fe;                 
         dealii::DoFHandler<DIM>    dof_handler;

         dealii::QGauss<DIM>   quadrature_formula;             //Quadrature
         dealii::Table<2,double>	        dofLocation;	         //Table of the coordinates of dofs by global dof number
         std::map<dealii::types::global_dof_index,double> boundary_values;        //Map of dirichlet boundary conditions
                  
         // Matrix Data structures
         dealii::SparsityPattern      sparsity_pattern;        //Sparse matrix pattern
         dealii::SparseMatrix<double> K;                       //Global stiffness matrix - Sparse matrix - used in the solver
         dealii::Vector<double>       D, F, Derror, Jphi, P, FFprime;                    //Global vectors - Solution vector (D) and Global force vector (F)
   
         // Output variables
         std::vector<std::string> psi_solution, jphi_solution, p_solution, ffp_solution;
         std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> nodal_data_component_interpretation;
         
         // Define which grid type to use in equilibrium calculation
         GridType mGridType;
         std::string mGridName;
   };
}