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

//class EdgeRGBDOdom
//{
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//    public:
//    // point positions
//    Vector3d pos0, pos1;

//    // unit normals
//    Mat *img0, img1;

//    EdgeRGBDOdom()
//    {
//        pos0.setZero();
//        pos1.setZero();
//    }
//}
//---------------------------------------------------------------------------------------------------------
// TODO: add a check of normals
// this function is from *Odometry implementation

int computeCorresps(const Mat& K, const Mat& K_inv, const Mat& Rt,
                    const Mat& depth0, const Mat& validMask0,
                    const Mat& depth1, const Mat& selectMask1, float maxDepthDiff,
                    Mat& corresps)
{
    CV_Assert(K.type() == CV_64FC1);
    CV_Assert(K_inv.type() == CV_64FC1);
    CV_Assert(Rt.type() == CV_64FC1);

    corresps.create(depth1.size(), CV_32SC1);
    corresps.setTo(-1);

    Rect r(0, 0, depth1.cols, depth1.rows);
    Mat Kt = Rt(Rect(3,0,1,3)).clone();
    Kt = K * Kt;
    const double * Kt_ptr = reinterpret_cast<const double *>(Kt.ptr());

    AutoBuffer<float> buf(3 * (depth1.cols + depth1.rows));
    float *KRK_inv0_u1 = buf;
    float *KRK_inv1_v1_plus_KRK_inv2 = KRK_inv0_u1 + depth1.cols;
    float *KRK_inv3_u1 = KRK_inv1_v1_plus_KRK_inv2 + depth1.rows;
    float *KRK_inv4_v1_plus_KRK_inv5 = KRK_inv3_u1 + depth1.cols;
    float *KRK_inv6_u1 = KRK_inv4_v1_plus_KRK_inv5 + depth1.rows;
    float *KRK_inv7_v1_plus_KRK_inv8 = KRK_inv6_u1 + depth1.cols;
    {
        Mat R = Rt(Rect(0,0,3,3)).clone();

        Mat KRK_inv = K * R * K_inv;
        const double * KRK_inv_ptr = reinterpret_cast<const double *>(KRK_inv.ptr());
        for(int u1 = 0; u1 < depth1.cols; u1++)
        {
            KRK_inv0_u1[u1] = KRK_inv_ptr[0] * u1;
            KRK_inv3_u1[u1] = KRK_inv_ptr[3] * u1;
            KRK_inv6_u1[u1] = KRK_inv_ptr[6] * u1;
        }

        for(int v1 = 0; v1 < depth1.rows; v1++)
        {
            KRK_inv1_v1_plus_KRK_inv2[v1] = KRK_inv_ptr[1] * v1 + KRK_inv_ptr[2];
            KRK_inv4_v1_plus_KRK_inv5[v1] = KRK_inv_ptr[4] * v1 + KRK_inv_ptr[5];
            KRK_inv7_v1_plus_KRK_inv8[v1] = KRK_inv_ptr[7] * v1 + KRK_inv_ptr[8];
        }
    }

    int correspCount = 0;
    for(int v1 = 0; v1 < depth1.rows; v1++)
    {
        const float *depth1_row = depth1.ptr<float>(v1);
        const uchar *mask1_row = selectMask1.ptr<uchar>(v1);
        for(int u1 = 0; u1 < depth1.cols; u1++)
        {
            float d1 = depth1_row[u1];
            if(mask1_row[u1])
            {
                CV_DbgAssert(!cvIsNaN(d1));
                float transformed_d1 = static_cast<float>(d1 * (KRK_inv6_u1[u1] + KRK_inv7_v1_plus_KRK_inv8[v1]) + Kt_ptr[2]);
                if(transformed_d1 > 0)
                {
                    float transformed_d1_inv = 1.f / transformed_d1;
                    int u0 = cvRound(transformed_d1_inv * (d1 * (KRK_inv0_u1[u1] + KRK_inv1_v1_plus_KRK_inv2[v1]) + Kt_ptr[0]));
                    int v0 = cvRound(transformed_d1_inv * (d1 * (KRK_inv3_u1[u1] + KRK_inv4_v1_plus_KRK_inv5[v1]) + Kt_ptr[1]));

                    if(r.contains(Point(u0,v0)))
                    {
                        float d0 = depth0.at<float>(v0,u0);
                        if(validMask0.at<uchar>(v0, u0) && std::abs(transformed_d1 - d0) <= maxDepthDiff)
                        {
                            CV_DbgAssert(!cvIsNaN(d0));
                            int c = corresps.at<int>(v0,u0);
                            if(c != -1)
                            {
                                int exist_u1, exist_v1;
                                get2shorts(c, exist_u1, exist_v1);

                                float exist_d1 = (float)(depth1.at<float>(exist_v1,exist_u1) *
                                    (KRK_inv6_u1[exist_u1] + KRK_inv7_v1_plus_KRK_inv8[exist_v1]) + Kt_ptr[2]);

                                if(transformed_d1 > exist_d1)
                                    continue;
                            }
                            else
                                correspCount++;

                            set2shorts(corresps.at<int>(v0,u0), u1, v1);
                        }
                    }
                }
            }
        }
    }
    return correspCount;
}

void fillGraphICPSE3(g2o::SparseOptimizer* optimizer, const std::vector<cv::Ptr<cv::OdometryFrameCache> >& frames,
                     const std::vector<cv::Mat>& poses, const Mat& cameraMatrix,
                     int startVertexIndex, bool addVertices)
{
    const double maxTranslation = 0.20;
    const double maxRotation = 30;
    const double maxDepthDiff = 0.01;

    CV_Assert(frames.size() == poses.size());
    {
        vector<Mat> posesExt(poses.size() + 1);
        copy(poses.begin(), poses.end(), posesExt.begin());
        *posesExt.rbegin() = (*poses.begin()).inv(DECOMP_SVD) * (*poses.rbegin());
        fillGraphSE3(optimizer, posesExt, startVertexIndex, addVertices);
    }

    Mat cameraMatrix_64F, cameraMatrix_inv_64F;
    cameraMatrix.convertTo(cameraMatrix_64F, CV_64FC1);
    cameraMatrix_inv_64F = cameraMatrix_64F.inv();

    // set up ICP edges
    for(size_t currFrameIdx = 0; currFrameIdx < frames.size(); currFrameIdx++)
    {
        const Mat& curCloud = frames[currFrameIdx]->pyramidCloud[0];
        const Mat& curNormals = frames[currFrameIdx]->normals;

        for(size_t prevFrameIdx = 0; prevFrameIdx < frames.size(); prevFrameIdx++)
        {
            if(currFrameIdx == prevFrameIdx)
                continue;

            const Mat& prevCloud = frames[prevFrameIdx]->pyramidCloud[0];
            const Mat& prevNormals = frames[prevFrameIdx]->normals;

            Mat curToPrevRt = poses[prevFrameIdx].inv(DECOMP_SVD) * poses[currFrameIdx];
            if(tvecNorm(curToPrevRt) > maxTranslation || rvecNormDegrees(curToPrevRt) > maxRotation)
                continue;

            Mat corresps;
            int correspsCount = computeCorresps(cameraMatrix_64F, cameraMatrix_inv_64F, curToPrevRt.inv(DECOMP_SVD),
                                                frames[currFrameIdx]->depth,
                                                frames[currFrameIdx]->mask,
                                                frames[prevFrameIdx]->depth,
                                                frames[prevFrameIdx]->pyramidNormalsMask[0],
                                                maxDepthDiff, corresps);

            cout << currFrameIdx << " -> " << prevFrameIdx << ": correspondences count " << correspsCount << endl;
            if(correspsCount <= 0)
                continue;

            // edges for poses
            for(int v0 = 0; v0 < corresps.rows; v0++)
            {
                for(int u0 = 0; u0 < corresps.cols; u0++)
                {
                    int c = corresps.at<int>(v0, u0);
                    if(c == -1)
                        continue;

                    int u1, v1;
                    get2shorts(c, u1, v1);

                    g2o::Edge_V_V_GICP * e = new g2o::Edge_V_V_GICP();
                    e->setVertex(0, optimizer->vertex(startVertexIndex + prevFrameIdx));
                    e->setVertex(1, optimizer->vertex(startVertexIndex + currFrameIdx));

                    g2o::EdgeGICP meas;
                    meas.pos0 = cvtPoint_ocv2egn(prevCloud.at<Point3f>(v1,u1));
                    meas.pos1 = cvtPoint_ocv2egn(curCloud.at<Point3f>(v0,u0));
                    meas.normal0 = cvtPoint_ocv2egn(prevNormals.at<Point3f>(v1,u1));
                    meas.normal1 = cvtPoint_ocv2egn(curNormals.at<Point3f>(v0,u0));

                    e->setMeasurement(meas);
                    meas = e->measurement();
                    e->information() = meas.prec0(0.01);

                    optimizer->addEdge(e);
                }
            }
        }
    }
}


void refineICPSE3Poses(std::vector<cv::Ptr<cv::OdometryFrameCache> >& frames, const std::vector<cv::Mat>& poses,
                       const cv::Mat& cameraMatrix, float pointsPart,
                       std::vector<cv::Mat>& refinedPoses)
{
    const int iterCount = 5;

    // TODO: find corresp to main API?
    // TODO: icp with one level here
    ICPOdometry icp;
    icp.set("cameraMatrix", cameraMatrix);
    icp.set("pointsPart", pointsPart);
    for(size_t i = 0; i < frames.size(); i++)
        icp.prepareFrameCache(*frames[i], OdometryFrameCache::CACHE_ALL);

    refinedPoses.resize(poses.size());
    for(size_t i = 0; i < poses.size(); i++)
        refinedPoses[i] = poses[i].clone();

    for(int iter = 0; iter < iterCount; iter++)
    {
        G2OLinearSolver* linearSolver =  createLinearSolver(DEFAULT_LINEAR_SOLVER_TYPE);
        G2OBlockSolver* blockSolver = createBlockSolver(linearSolver);
        g2o::OptimizationAlgorithm* nonLinerSolver = createNonLinearSolver(DEFAULT_NON_LINEAR_SOLVER_TYPE, blockSolver);
        g2o::SparseOptimizer* optimizer = createOptimizer(nonLinerSolver);

        fillGraphICPSE3(optimizer, frames, refinedPoses, cameraMatrix, 0, true);

        optimizer->initializeOptimization();
        optimizer->optimize(1);

        getSE3Poses(optimizer, Range(0, optimizer->vertices().size()), refinedPoses);

        optimizer->clear();
    }
}
