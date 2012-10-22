#include "model_capture.hpp"
#include "create_optimizer.hpp"
#include "ocv_pcl_eigen_convert.hpp"

#include "g2o/types/slam3d/se3quat.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/core/block_solver.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_factory.h"


#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/solver.h"
#include "g2o/types/icp/types_icp.h"

#include "opencv2/core/eigen.hpp"

using namespace std;
using namespace cv;

static
g2o::SE3Quat eigen2G2O(const Eigen::Matrix4d& eigen_mat)
{
  Eigen::Affine3d eigen_transform(eigen_mat);
  Eigen::Quaterniond eigen_quat(eigen_transform.rotation());
  Eigen::Vector3d translation(eigen_mat(0, 3), eigen_mat(1, 3), eigen_mat(2, 3));
  g2o::SE3Quat result(eigen_quat, translation);

  return result;
}

static
g2o::SE3Quat cv2G2O(const cv::Mat& cv_mat)
{
    Eigen::Matrix4d eigen_mat;
    cv::cv2eigen(cv_mat, eigen_mat);

    return eigen2G2O(eigen_mat);
}

static inline
Eigen::Matrix<double,6,6> informationMatrixSE3()
{
    const float w = 10000;
    Eigen::Matrix<double,6,6> informationMatrix = Eigen::Matrix<double,6,6>::Identity();
    informationMatrix(3,3) = w;
    informationMatrix(4,4) = w;
    informationMatrix(5,5) = w;

    return informationMatrix;
}

void fillGraphSE3(g2o::SparseOptimizer* optimizer, const vector<Mat>& poses, int startVertexIndex, bool addVertices)
{
    // poses[0] .. poses[poses.size() - 2] are sequential frame-to-frame poses
    // poses[poses.size() - 1] is a pose of last frame found by Odometry with the first frame directly (for the loop closure)

    const size_t frameToFramePosesCount = poses.size() - 1;
    for(size_t i = 0; i < frameToFramePosesCount; i++)
    {
        int vertexIndex = startVertexIndex + i;

        if(addVertices)
        {
            // add vertex
            g2o::VertexSE3* pose = new g2o::VertexSE3;
            pose->setId(vertexIndex);
            g2o::SE3Quat g2o_se3 = cv2G2O(poses[i]);
            pose->setEstimate(g2o_se3);

            optimizer->addVertex(pose);
        }

        if(i == 0)
            optimizer->vertex(startVertexIndex)->setFixed(true); //fix at origin


        if(i > 0)
        {
            // Add edge with previous frame
            g2o::EdgeSE3* g2o_edge = new g2o::EdgeSE3;

            g2o_edge->vertices()[0] = optimizer->vertex(vertexIndex-1);
            g2o_edge->vertices()[1] = optimizer->vertex(vertexIndex);

            Mat odometryConstraint = poses[i-1].inv(DECOMP_SVD) * poses[i];
            g2o_edge->setMeasurement(cv2G2O(odometryConstraint));
            g2o_edge->setInformation(informationMatrixSE3());

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            g2o_edge->setRobustKernel(rk);

            optimizer->addEdge(g2o_edge);
        }
    }

    // Add edge of loop closure
    {
        g2o::EdgeSE3* g2o_edge = new g2o::EdgeSE3;
        g2o_edge->vertices()[0] = optimizer->vertex(startVertexIndex);
        g2o_edge->vertices()[1] = optimizer->vertex(startVertexIndex + frameToFramePosesCount - 1);

        Mat odometryConstraint = poses[0].inv(DECOMP_SVD) * *poses.rbegin();
        g2o_edge->setMeasurement(cv2G2O(odometryConstraint));
        g2o_edge->setInformation(informationMatrixSE3());

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        g2o_edge->setRobustKernel(rk);

        optimizer->addEdge(g2o_edge);
    }
}

void getSE3Poses(g2o::SparseOptimizer* optimizer, const Range& verticesRange, vector<Mat>& poses)
{
    poses.resize(verticesRange.end - verticesRange.start);
    for(int i = verticesRange.start; i < verticesRange.end; i++)
    {
        g2o::VertexSE3* v = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(i));
        Eigen::Isometry3d pose = v->estimate();
        poses[i] = cvtIsometry_egn2ocv(pose);
    }
}

void refineSE3Poses(const vector<Mat>& poses, vector<Mat>& refinedPoses)
{
    // Refine poses by pose graph oprimization
    G2OLinearSolver* linearSolver =  createLinearSolver(DEFAULT_LINEAR_SOLVER_TYPE);
    G2OBlockSolver* blockSolver = createBlockSolver(linearSolver);
    g2o::OptimizationAlgorithm* nonLinerSolver = createNonLinearSolver(DEFAULT_NON_LINEAR_SOLVER_TYPE, blockSolver);
    g2o::SparseOptimizer* optimizer = createOptimizer(nonLinerSolver);

    fillGraphSE3(optimizer, poses, 0, true);

    optimizer->initializeOptimization();
    optimizer->optimize(5);

    getSE3Poses(optimizer, Range(0, optimizer->vertices().size()), refinedPoses);

    optimizer->clear();
}
