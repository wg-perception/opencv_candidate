#ifndef MODEL_CAPTURE_CREATE_OPTIMIZER
#define MODEL_CAPTURE_CREATE_OPTIMIZER

#include "g2o/core/block_solver.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
//#include "g2o/core/factory.h"
//#include "g2o/core/optimization_algorithm_factory.h"

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/solver.h"
//#include "g2o/types/icp/types_icp.h"

typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> >  G2OBlockSolver;
typedef g2o::LinearSolver< G2OBlockSolver::PoseMatrixType> G2OLinearSolver;
typedef g2o::LinearSolverCholmod<G2OBlockSolver::PoseMatrixType> G2OLinearCholmodSolver;

const std::string DEFAULT_LINEAR_SOLVER_TYPE = "cholmod";
const std::string DEFAULT_NON_LINEAR_SOLVER_TYPE  = "GN";

inline
G2OLinearSolver* createLinearSolver(const std::string& type)
{
    G2OLinearSolver* solver = 0;
    if(type == "cholmod")
        solver = new G2OLinearCholmodSolver();
    else
    {
        CV_Assert(0);
    }

    return solver;
}

inline
G2OBlockSolver* createBlockSolver(G2OLinearSolver* linearSolver)
{
    return new G2OBlockSolver(linearSolver);
}

inline
g2o::OptimizationAlgorithm* createNonLinearSolver(const std::string& type, G2OBlockSolver* blockSolver)
{
    g2o::OptimizationAlgorithm* solver = 0;
    if(type == "GN")
        solver = new g2o::OptimizationAlgorithmGaussNewton(blockSolver);
    else if(type == "LM")
        solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);
    else
        CV_Assert(0);

    return solver;
}

inline
g2o::SparseOptimizer* createOptimizer(g2o::OptimizationAlgorithm* solver)
{
    g2o::SparseOptimizer* optimizer = new g2o::SparseOptimizer();
    optimizer->setAlgorithm(solver);
    optimizer->setVerbose(true);
    return optimizer;
}

#endif
