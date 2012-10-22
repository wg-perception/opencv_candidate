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

namespace g2o {

    class Edge_V_V_GICPLandmark : public  BaseBinaryEdge<3, EdgeGICP, VertexPointXYZ, VertexSE3>
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Edge_V_V_GICPLandmark() {}
        Edge_V_V_GICPLandmark(const Edge_V_V_GICPLandmark* e);

      // I/O functions
      virtual bool read(std::istream& /*is*/) {return false;}
      virtual bool write(std::ostream& /*os*/) const {return false;}

      // return the error estimate as a 3-vector
      void computeError()
      {
        // from <ViewPoint> to <Point>
        const VertexPointXYZ *vp0 = static_cast<const VertexPointXYZ*>(_vertices[0]);
        const VertexSE3 *vp1 = static_cast<const VertexSE3*>(_vertices[1]);

        Vector3d p1 = vp1->estimate() * measurement().pos1;
        _error = p1 - vp0->estimate();
      }

      virtual void linearizeOplus();

      static Matrix3d dRidx;
      static Matrix3d dRidy;
      static Matrix3d dRidz;

      static void initializeStaticMatrices()
      {
          //if(dRidx.data() == 0)
          {
              dRidx << 0.0,0.0,0.0,
                0.0,0.0,2.0,
                0.0,-2.0,0.0;
              dRidy  << 0.0,0.0,-2.0,
                0.0,0.0,0.0,
                2.0,0.0,0.0;
              dRidz  << 0.0,2.0,0.0,
                -2.0,0.0,0.0,
                0.0,0.0,0.0;
          }
      }
    };

    G2O_REGISTER_TYPE(Edge_V_V_GICPLandmark, Edge_V_V_GICPLandmark);

    Matrix3d Edge_V_V_GICPLandmark::dRidx; // differential quat matrices
    Matrix3d Edge_V_V_GICPLandmark::dRidy; // differential quat matrices
    Matrix3d Edge_V_V_GICPLandmark::dRidz; // differential quat matrices


    // Copy constructor
    Edge_V_V_GICPLandmark::Edge_V_V_GICPLandmark(const Edge_V_V_GICPLandmark* e)
      : BaseBinaryEdge<3, EdgeGICP, VertexPointXYZ, VertexSE3>()
    {

      // Temporary hack - TODO, sort out const-ness properly
      _vertices[0] = const_cast<HyperGraph::Vertex*> (e->vertex(0));
      _vertices[1] = const_cast<HyperGraph::Vertex*> (e->vertex(1));

      _measurement.pos0 = e->measurement().pos0;
      _measurement.pos1 = e->measurement().pos1;
      _measurement.normal0 = e->measurement().normal0;
      _measurement.normal1 = e->measurement().normal1;
      _measurement.R0 = e->measurement().R0;
      _measurement.R1 = e->measurement().R1;
    }

    void Edge_V_V_GICPLandmark::linearizeOplus()
    {
      //  std::cout << "START Edge_V_V_GICPLandmark::linearizeOplus() " << std::endl;
      VertexPointXYZ* vp0 = static_cast<VertexPointXYZ*>(_vertices[0]);
      VertexSE3* vp1 = static_cast<VertexSE3*>(_vertices[1]);
      Vector3d p1 = measurement().pos1;

      if (!vp0->fixed())
        {
          _jacobianOplusXi.block<3,3>(0,0) = -Matrix3d::Identity();
        }

      if (!vp1->fixed())
        {
          Matrix3d R1 = vp1->estimate().matrix().topLeftCorner<3,3>();
          _jacobianOplusXj.block<3,3>(0,0) = R1;
          _jacobianOplusXj.block<3,1>(0,3) = R1*dRidx.transpose()*p1;
          _jacobianOplusXj.block<3,1>(0,4) = R1*dRidy.transpose()*p1;
          _jacobianOplusXj.block<3,1>(0,5) = R1*dRidz.transpose()*p1;
        }
    }
}

static
void computeCorrespsFiltered(const Mat& K, const Mat& K_inv, const Mat& Rt,
                            const Mat& depth0, const Mat& validMask0,
                            const Mat& depth1, const Mat& selectMask1, float maxDepthDiff,
                            Mat& corresps,
                            const Mat& normals0, const Mat& transformedNormals1,
                            const Mat& image0, const Mat& image1)
{
    const double maxNormalsDiff = 30; // in degrees
    const double maxColorDiff = 50;
    const double maxNormalAngleDev = 75; // in degrees

    computeCorresps(K, K_inv, Rt, depth0, validMask0, depth1, selectMask1,
                    maxDepthDiff, corresps);

    const Point3f Oz_inv(0,0,-1);
    for(int v0 = 0; v0 < corresps.rows; v0++)
    {
        for(int u0 = 0; u0 < corresps.cols; u0++)
        {
            int c = corresps.at<int>(v0, u0);
            if(c != -1)
            {
                Point3f curNormal = normals0.at<Point3f>(v0,u0);
                if(std::abs(curNormal.ddot(Oz_inv)) < std::cos(maxNormalAngleDev / 180 * CV_PI))
                {
                    corresps.at<int>(v0, u0) = -1;
                    continue;
                }

                int u1, v1;
                get2shorts(c, u1, v1);

                Point3f transfPrevNormal = transformedNormals1.at<Point3f>(v1,u1);
                if(std::abs(curNormal.ddot(transfPrevNormal)) < std::cos(maxNormalsDiff / 180 * CV_PI))
                {
                    corresps.at<int>(v0, u0) = -1;
                    continue;
                }

                if(std::abs(image0.at<uchar>(v0,u0) - image1.at<uchar>(v1,u1)) > maxColorDiff)
                {
                    corresps.at<int>(v0, u0) = -1;
                    continue;
                }
            }
        }
    }
}

void refineICPSE3Landmarks(std::vector<cv::Ptr<cv::OdometryFrameCache> >& frames, const std::vector<cv::Mat>& poses, const cv::Mat& cameraMatrix,
                           std::vector<cv::Mat>& refinedPoses)
{
    g2o::Edge_V_V_GICPLandmark::initializeStaticMatrices(); // TODO: make this more correctly

    Mat cameraMatrix_64F, cameraMatrix_inv_64F;
    cameraMatrix.convertTo(cameraMatrix_64F, CV_64FC1);
    cameraMatrix_inv_64F = cameraMatrix_64F.inv();

    refinedPoses.resize(poses.size());
    std::copy(poses.begin(), poses.end(), refinedPoses.begin());

    RgbdICPOdometry odom;
    odom.set("cameraMatrix", cameraMatrix);
    odom.set("pointsPart", 1.);

    const double maxDepthDiff = 0.002;
    for(size_t i = 0; i < frames.size(); i++)
        odom.prepareFrameCache(*frames[i], OdometryFrameCache::CACHE_ALL);

    const int iterCount = 3;//7
    const int minCorrespCount = 3;

    const double maxTranslation = 0.20;
    const double maxRotation = 30;
    for(int iter = 0; iter < iterCount; iter++)
    {
        G2OLinearSolver* linearSolver =  createLinearSolver(DEFAULT_LINEAR_SOLVER_TYPE);
        G2OBlockSolver* blockSolver = createBlockSolver(linearSolver);
        g2o::OptimizationAlgorithm* nonLinerSolver = createNonLinearSolver(DEFAULT_NON_LINEAR_SOLVER_TYPE, blockSolver);
        g2o::SparseOptimizer* optimizer = createOptimizer(nonLinerSolver);

        fillGraphRgbdICPSE3(optimizer, frames, refinedPoses, cameraMatrix_64F, 0);

        vector<Mat> vertexIndices(frames.size());
        int vertexIdx = optimizer->vertices().size();
        int posesVertexCount = optimizer->vertices().size();
        for(size_t curFrameIdx = 0; curFrameIdx < frames.size(); curFrameIdx++)
        {
            vertexIndices[curFrameIdx] = Mat(frames[curFrameIdx]->image.size(), CV_32SC1, Scalar(-1));

            Mat& curVertexIndices = vertexIndices[curFrameIdx];
            const Mat& curCloud = frames[curFrameIdx]->pyramidCloud[0];
            const Mat& curNormals = frames[curFrameIdx]->normals;

            // compute count of correspondences
            Mat correspsCounts = Mat(frames[curFrameIdx]->image.size(), CV_32SC1, Scalar(0));
            for(size_t prevFrameIdx = 0; prevFrameIdx < frames.size(); prevFrameIdx++)
            {
                if(curFrameIdx == prevFrameIdx)
                    continue;

                Mat curToPrevRt = refinedPoses[prevFrameIdx].inv(DECOMP_SVD) * refinedPoses[curFrameIdx];
                if(tvecNorm(curToPrevRt) > maxTranslation || rvecNormDegrees(curToPrevRt) > maxRotation)
                    continue;

                Mat transformedPrevNormals;
                perspectiveTransform(frames[prevFrameIdx]->pyramidNormals[0], transformedPrevNormals,
                                     curToPrevRt.inv(DECOMP_SVD));
                Mat corresps;
                computeCorrespsFiltered(cameraMatrix_64F, cameraMatrix_inv_64F, curToPrevRt.inv(DECOMP_SVD),
                                        frames[curFrameIdx]->depth,
                                        frames[curFrameIdx]->mask,
                                        frames[prevFrameIdx]->depth,
                                        frames[prevFrameIdx]->pyramidNormalsMask[0],
                                        maxDepthDiff, corresps,
                                        frames[curFrameIdx]->pyramidNormals[0],
                                        transformedPrevNormals,
                                        frames[curFrameIdx]->image,
                                        frames[prevFrameIdx]->image);

                for(int v0 = 0; v0 < corresps.rows; v0++)
                {
                    for(int u0 = 0; u0 < corresps.cols; u0++)
                    {
                        int c = corresps.at<int>(v0, u0);
                        if(c != -1)
                            correspsCounts.at<int>(v0,u0)++;
                    }
                }
            }

            // set up edges
            for(size_t prevFrameIdx = 0; prevFrameIdx < frames.size(); prevFrameIdx++)
            {
                if(curFrameIdx == prevFrameIdx)
                    continue;

                const Mat& prevCloud = frames[prevFrameIdx]->pyramidCloud[0];
                const Mat& prevNormals = frames[prevFrameIdx]->normals;
                Mat curToPrevRt = refinedPoses[prevFrameIdx].inv(DECOMP_SVD) * refinedPoses[curFrameIdx];
                if(tvecNorm(curToPrevRt) > maxTranslation || rvecNormDegrees(curToPrevRt) > maxRotation)
                    continue;

                Mat transformedPrevNormals;
                perspectiveTransform(frames[prevFrameIdx]->pyramidNormals[0], transformedPrevNormals,
                                     curToPrevRt.inv(DECOMP_SVD));
                Mat corresps;
                computeCorrespsFiltered(cameraMatrix_64F, cameraMatrix_inv_64F, curToPrevRt.inv(DECOMP_SVD),
                                        frames[curFrameIdx]->depth,
                                        frames[curFrameIdx]->mask,
                                        frames[prevFrameIdx]->depth,
                                        frames[prevFrameIdx]->pyramidNormalsMask[0],
                                        maxDepthDiff, corresps,
                                        frames[curFrameIdx]->pyramidNormals[0],
                                        transformedPrevNormals,
                                        frames[curFrameIdx]->image,
                                        frames[prevFrameIdx]->image);

                std::cout << "refineICPSE3Landmarks iter " << iter << "; cur " << curFrameIdx << "; prev " << prevFrameIdx << "; corresps " << countNonZero(corresps != -1) << std::endl;

                // poses and edges for points3d
                for(int v0 = 0; v0 < corresps.rows; v0++)
                {
                    for(int u0 = 0; u0 < corresps.cols; u0++)
                    {
                        int c = corresps.at<int>(v0, u0);
                        if(c == -1)
                            continue;

                        if(correspsCounts.at<int>(v0,u0) < minCorrespCount)
                            continue;

                        int u1_, v1_;
                        get2shorts(c, u1_, v1_);

                        const Rect rect(0,0,curCloud.cols, curCloud.rows);
                        const int Rad = 0;
                        for(int v1 = v1_ - Rad; v1 <= v1_ + Rad; v1++)
                        {
                            for(int u1 = u1_ - Rad; u1 <= u1_ + Rad; u1++)
                            {
                                if(rect.contains(Point(u1, v1)) && !cvIsNaN(prevCloud.at<Point3f>(v1,u1).x) &&
                                   std::abs(prevCloud.at<Point3f>(v1,u1).z - prevCloud.at<Point3f>(v1_,u1_).z) < maxDepthDiff)
                                {
                                    Eigen::Vector3d pt_prev, pt_cur, norm_prev, norm_cur, global_norm_prev;
                                    {
                                        pt_prev = cvtPoint_ocv2egn(prevCloud.at<Point3f>(v1,u1));
                                        norm_prev = cvtPoint_ocv2egn(prevNormals.at<Point3f>(v1,u1));

                                        vector<Point3f> tp;
                                        perspectiveTransform(vector<Point3f>(1, prevNormals.at<Point3f>(v1,u1)),
                                                             tp, refinedPoses[prevFrameIdx]);
                                        global_norm_prev = cvtPoint_ocv2egn(tp[0]);

                                        perspectiveTransform(vector<Point3f>(1, curCloud.at<Point3f>(v0,u0)),
                                                             tp, refinedPoses[curFrameIdx]);
                                        pt_cur = cvtPoint_ocv2egn(tp[0]);
                                        perspectiveTransform(vector<Point3f>(1, curNormals.at<Point3f>(v0,u0)),
                                                             tp, refinedPoses[curFrameIdx]);
                                        norm_cur = cvtPoint_ocv2egn(tp[0]);
                                    }

                                    // add new pose
                                    if(curVertexIndices.at<int>(v0,u0) == -1)
                                    {
                                        g2o::VertexPointXYZ* modelPoint = new g2o::VertexPointXYZ;
                                        modelPoint->setId(vertexIdx);
                                        modelPoint->setEstimate(pt_cur);
                                        modelPoint->setMarginalized(true);
                                        optimizer->addVertex(modelPoint);

                                        curVertexIndices.at<int>(v0,u0) = vertexIdx;
                                        vertexIdx++;
                                    }

                                    int vidx = curVertexIndices.at<int>(v0,u0);

                                    g2o::Edge_V_V_GICPLandmark * e = new g2o::Edge_V_V_GICPLandmark();
                                    e->setVertex(0, optimizer->vertex(vidx));
                                    e->setVertex(1, optimizer->vertex(prevFrameIdx));

                                    g2o::EdgeGICP meas;
                                    meas.pos0 = pt_cur;
                                    meas.pos1 = pt_prev;
                                    meas.normal0 = norm_cur;
                                    meas.normal1 = norm_prev;

                                    e->setMeasurement(meas);
                                    meas = e->measurement();

//                                    e->information() = meas.prec0(0.01);
                                    meas.normal1 = global_norm_prev; // to get global covariation
                                    e->information() = 0.001 * (meas.cov0(1.).inverse() + meas.cov1(1.).inverse());
                                    meas.normal1 = norm_prev; // set local normal

                                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                                    e->setRobustKernel(rk);

                                    optimizer->addEdge(e);
                                }
                            }
                        }
                    }
                }
            }
        }


        cout << "All vertices count " << optimizer->vertices().size() << endl;
        cout << "Vertices of points " << vertexIdx - posesVertexCount << endl;
        cout << "Edges count " << optimizer->edges().size() << endl;

        optimizer->initializeOptimization();
        const int optIterCount = 1;
        if(optimizer->optimize(optIterCount) != optIterCount)
        {
            optimizer->clear();
            break;
        }

        getSE3Poses(optimizer, Range(0, posesVertexCount), refinedPoses);

        // update points poses
        cout << "Updating model points..." << endl;
        for(size_t i = 0; i < vertexIndices.size(); i++)
        {
            const Mat& curVertexIndices = vertexIndices[i];
            Mat& depth = frames[i]->depth;
            for(int y = 0; y < curVertexIndices.rows; y++)
            {
                for(int x = 0; x < curVertexIndices.cols; x++)
                {
                    int vidx = curVertexIndices.at<int>(y,x);
                    if(vidx < 0)
                        continue;

                    Point3f p;
                    {
                        g2o::VertexPointXYZ* v = dynamic_cast<g2o::VertexPointXYZ*>(optimizer->vertex(vidx));
                        Eigen::Vector3d ep = v->estimate();
                        p.x = ep[0]; p.y = ep[1]; p.z = ep[2];
                    }
                    vector<Point3f> tp;
                    perspectiveTransform(vector<Point3f>(1, p), tp, refinedPoses[i].inv(DECOMP_SVD));
                    depth.at<float>(y,x) = tp[0].z;
                }
            }

            frames[i]->pyramidDepth.clear();
            frames[i]->normals.release();
            frames[i]->pyramidNormals.clear();
            frames[i]->pyramidCloud.clear();
            frames[i]->pyramidNormalsMask.clear();
            odom.prepareFrameCache(*frames[i], OdometryFrameCache::CACHE_ALL);
        }

        optimizer->clear();
    }

    // remove points without correspondences
    for(size_t curFrameIdx = 0; curFrameIdx < frames.size(); curFrameIdx++)
    {
        // compute count of correspondences
        Mat& curCloud = frames[curFrameIdx]->pyramidCloud[0];
        Mat correspsCounts = Mat(frames[curFrameIdx]->image.size(), CV_32SC1, Scalar(0));
        for(size_t prevFrameIdx = 0; prevFrameIdx < frames.size(); prevFrameIdx++)
        {
            if(curFrameIdx == prevFrameIdx)
                continue;

            Mat curToPrevRt = refinedPoses[prevFrameIdx].inv(DECOMP_SVD) * refinedPoses[curFrameIdx];
            Mat transformedPrevNormals;
            perspectiveTransform(frames[prevFrameIdx]->pyramidNormals[0], transformedPrevNormals,
                                 curToPrevRt.inv(DECOMP_SVD));
            Mat corresps;
            computeCorrespsFiltered(cameraMatrix_64F, cameraMatrix_inv_64F, curToPrevRt.inv(DECOMP_SVD),
                                    frames[curFrameIdx]->depth,
                                    frames[curFrameIdx]->mask,
                                    frames[prevFrameIdx]->depth,
                                    frames[prevFrameIdx]->pyramidNormalsMask[0],
                                    maxDepthDiff, corresps,
                                    frames[curFrameIdx]->pyramidNormals[0],
                                    transformedPrevNormals,
                                    frames[curFrameIdx]->image,
                                    frames[prevFrameIdx]->image);

            for(int v0 = 0; v0 < corresps.rows; v0++)
            {
                for(int u0 = 0; u0 < corresps.cols; u0++)
                {
                    int c = corresps.at<int>(v0, u0);
                    if(c != -1)
                        correspsCounts.at<int>(v0,u0)++;
                }
            }
        }

        for(int v0 = 0; v0 < correspsCounts.rows; v0++)
        {
            for(int u0 = 0; u0 < correspsCounts.cols; u0++)
            {
                if(correspsCounts.at<int>(v0,u0) < minCorrespCount)
                    curCloud.at<Point3f>(v0,u0) = Point3f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
            }
        }
    }
}
