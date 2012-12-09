#include <g2o/types/icp/types_icp.h>

#include "reconst3d.hpp"
#include "ocv_pcl_convert.hpp"
#include "graph_optimizations.hpp"

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
g2o::SE3Quat cv2G2O(const Mat& cv_mat)
{
    Eigen::Matrix4d eigen_mat;
    cv2eigen(cv_mat, eigen_mat);

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

static
g2o::EdgeSE3* createEdgeSE3(g2o::HyperGraph::Vertex* v0, g2o::HyperGraph::Vertex* v1, const Mat& Rt)
{
    g2o::EdgeSE3* g2o_edge = new g2o::EdgeSE3;

    g2o_edge->vertices()[0] = v0;
    g2o_edge->vertices()[1] = v1;

    g2o_edge->setMeasurement(cv2G2O(Rt));
    g2o_edge->setInformation(informationMatrixSE3());

    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    g2o_edge->setRobustKernel(rk);

    return g2o_edge;
}

static 
bool addGraphVertex(g2o::SparseOptimizer* optimizer, int vertexID, const Mat& pose)
{
    g2o::VertexSE3* existVertex = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(vertexID));
    if(!existVertex)
    {
        g2o::VertexSE3* newVertex = new g2o::VertexSE3;
        newVertex->setId(vertexID);
        newVertex->setEstimate(cv2G2O(pose));
        optimizer->addVertex(newVertex);
        return true;
    }
    return false;
}

void fillGraphSE3(g2o::SparseOptimizer* optimizer,
                  const vector<cv::Mat>& poses, const std::vector<PosesLink>& posesLinks, std::vector<int>& frameIndices)
{
    CV_Assert(!poses.empty());
    CV_Assert(!posesLinks.empty());

    frameIndices.clear();
    for(size_t edgeIndex = 0; edgeIndex < posesLinks.size(); edgeIndex++)
    {
        int srcVertexIndex = posesLinks[edgeIndex].srcIndex;
        int dstVertexIndex = posesLinks[edgeIndex].dstIndex;

        // add vetices
        if(addGraphVertex(optimizer, srcVertexIndex, poses[srcVertexIndex]))
            frameIndices.push_back(srcVertexIndex);
        if(addGraphVertex(optimizer, dstVertexIndex, poses[dstVertexIndex]))
            frameIndices.push_back(dstVertexIndex);

        // add edge between the vertices
        Mat Rt = posesLinks[edgeIndex].Rt.empty() ? poses[srcVertexIndex].inv(DECOMP_SVD) * poses[dstVertexIndex] : posesLinks[edgeIndex].Rt;
        optimizer->addEdge(createEdgeSE3(optimizer->vertex(srcVertexIndex), optimizer->vertex(dstVertexIndex), Rt));
    }

    int fixedVertexIndex = posesLinks[0].srcIndex;
    optimizer->vertex(fixedVertexIndex)->setFixed(true); //fix at origin
}

void refineGraphSE3(const vector<Mat>& poses, const std::vector<PosesLink>& posesLinks,
                    vector<Mat>& refinedPoses, vector<int>& frameIndices)
{
    refinedPoses.resize(poses.size());
    for(size_t i = 0; i < poses.size(); i++)
        refinedPoses[i] = poses[i].clone();

    // Refine poses by pose graph oprimization
    G2OLinearSolver* linearSolver =  createLinearSolver(DEFAULT_LINEAR_SOLVER_TYPE);
    G2OBlockSolver* blockSolver = createBlockSolver(linearSolver);
    g2o::OptimizationAlgorithm* nonLinerSolver = createNonLinearSolver(DEFAULT_NON_LINEAR_SOLVER_TYPE, blockSolver);
    g2o::SparseOptimizer* optimizer = createOptimizer(nonLinerSolver);

    fillGraphSE3(optimizer, poses, posesLinks, frameIndices);

    optimizer->initializeOptimization();
    const int optIterCount = 5;
    cout << "Vertices count: " << optimizer->vertices().size() << endl;
    cout << "Edges count: " << optimizer->edges().size() << endl;
    if(optimizer->optimize(optIterCount) != optIterCount)
        CV_Error(CV_StsError, "Cann't do given count of iterations\n");

    getSE3Poses(optimizer, frameIndices, refinedPoses);

    optimizer->clear();
    delete optimizer;
}

void getSE3Poses(g2o::SparseOptimizer* optimizer, const std::vector<int>& frameIndices, vector<Mat>& poses)
{
    for(size_t i = 0; i < frameIndices.size(); i++)
    {
        int frameIndex = frameIndices[i];

        g2o::VertexSE3* v = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(frameIndex));
        Eigen::Isometry3d pose = v->estimate();

        CV_Assert(frameIndex >= 0 && frameIndex < static_cast<int>(poses.size()));

        poses[frameIndex] = cvtIsometry_egn2ocv(pose);
    }
}
