#if 0
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#endif

#include <opencv2/rgbd/rgbd.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "graph_optimizations.hpp"

using namespace cv;
using namespace std;

static void
preparePosesLinksWithoutRt(const vector<PosesLink>& srcLinks, vector<PosesLink>& dstLinks)
{
    dstLinks.resize(srcLinks.size());
    for(size_t i = 0; i < dstLinks.size(); i++)
        dstLinks[i] = PosesLink(srcLinks[i].srcIndex, srcLinks[i].dstIndex);
}

static void
preparePosesLinksWithoutRt(const vector<int>& poseIndices, vector<PosesLink>& dstLinks)
{
    CV_Assert(poseIndices.size() > 2);
    dstLinks.resize(poseIndices.size() - 1);
    for(size_t i = 1; i < poseIndices.size(); i++)
        dstLinks[i-1] = PosesLink(poseIndices[i-1], poseIndices[i]);
}

#if 0
static
void filterFramesByViewHist(vector<Ptr<RgbdFrame> >& frames,
                            const vector<int>& frameIndices, const vector<Mat>& poses, const Mat& cameraMatrix)
{
    vector<Point3f> transformedPoints3d;
    vector<Point3f> transformedNormals;
    vector<Vec3i> pointFrameIndices;

    // fill trainig samples matrix
    for(size_t frameIndex = 0; frameIndex < frames.size(); frameIndex++)
    {
        Mat cloud;
        Mat normals;
        {
            depthTo3d(frames[frameIndex]->depth, cameraMatrix, cloud);

            Mat tcloud, tnormals;
            perspectiveTransform(cloud.reshape(3,1), tcloud, poses[frameIndex]);
            Mat R = poses[frameIndex](Rect(0,0,3,3));
            transform(frames[frameIndex]->normals.reshape(3,1), tnormals, R);

            cloud = tcloud.reshape(3, frames[frameIndex]->depth.rows);
            normals = tnormals.reshape(3, frames[frameIndex]->depth.rows);
        }
    
        bool isRefinedPose = false;
        if(std::find(frameIndices.begin(), frameIndices.end(), frameIndex) != frameIndices.end())
            isRefinedPose = true;

        const Mat& mask = frames[frameIndex]->mask;
        for(int y = 0; y < cloud.rows; y++)
        {
            for(int x = 0; x < cloud.cols; x++)
            {
                const Point3f& p = cloud.at<Point3f>(y,x);
                CV_Assert(normals.type() == CV_32FC3);
                Point3f n = normals.at<Point3f>(y,x);

                if(isValidDepth(p.z) && mask.at<uchar>(y,x))
                {
                    transformedPoints3d.push_back(p);
                    transformedNormals.push_back(n);
                    pointFrameIndices.push_back(Vec3i(frameIndex, -1, -1));
                    if(isRefinedPose)
                    {
                        (*pointFrameIndices.rbegin())[1] = x;
                        (*pointFrameIndices.rbegin())[2] = y;
                    }
                }
            }
        }
    }

    Ptr<flann::Index> flannIndex = new flann::Index(Mat(transformedPoints3d).reshape(1, transformedPoints3d.size()), flann::KDTreeIndexParams());
    
    const int angleBinCount = 8;
    const int minViewAnglesCount = 3; // TODO

    //const double maxCosDiff = cos(20./180 * CV_PI);
    const int maxResults = cvRound(0.05 * transformedPoints3d.size());
    const double radius = 0.005;//m // TODO
    const double radius2 = radius * radius;

    for(size_t pointIndex = 0; pointIndex < transformedPoints3d.size(); pointIndex++)
    {
        int frameIndex = pointFrameIndices[pointIndex][0];
        int x =          pointFrameIndices[pointIndex][1];
        int y =          pointFrameIndices[pointIndex][2];

        if(x == -1)
            continue;

        const Point3f& p = transformedPoints3d[pointIndex];
        const Point3f& n = transformedNormals[pointIndex];

        Mat indices(1, maxResults, CV_32SC1, Scalar(-1));
        Mat dists(1, maxResults, CV_32FC1, Scalar(-1));
        flannIndex->radiusSearch(Mat(p).reshape(1,1), indices, dists, radius2, maxResults, flann::SearchParams());

        // compute reference direction
        Point3f refDirection = Point3f(1,0,0);
        const double badAngleCos = 0.9;
        if(std::abs(transformedNormals[pointIndex].dot(refDirection)) > badAngleCos)
            refDirection = Point3f(0,1,0);

        Mat viewAngleHist(angleBinCount, angleBinCount, CV_8UC1, Scalar(0));
        CV_Assert(indices.at<int>(0) == static_cast<int>(pointIndex));

        vector<char> framesMask(frames.size(), 0);
        for(int i = 0; i < maxResults; i++)
        {
            int nearPointIndex = indices.at<int>(i);
            if(nearPointIndex < 0)
                break;
            
            int nearPointFrameIndex = pointFrameIndices[nearPointIndex][0];
            if(framesMask[nearPointFrameIndex])
                continue;

//            if(std::abs(transformedNormals[nearPointIndex].ddot(n)) < maxCosDiff)
//                continue;

            // TODO normal angles check
            // if they are different then continue();
            const Mat cameraPose = poses[nearPointFrameIndex];
            Point3f cameraPoint;
            CV_Assert(cameraPose.type() == CV_64FC1);
            cameraPoint.x = cameraPose.at<double>(0,3);
            cameraPoint.y = cameraPose.at<double>(1,3);
            cameraPoint.z = cameraPose.at<double>(2,3);

            // compute polar angle
            Point3f toCameraVec = cameraPoint - p;
            double polarCos = n.dot(toCameraVec);
            if(polarCos > 1.) polarCos /= polarCos;
            float polarAngle = std::acos(polarCos);
            Point3f rn = n;
            if(polarAngle > 0.5 * CV_PI) // if is was incorrect defined
            {
                polarAngle = CV_PI - polarAngle;
                rn = -n;
            }

            // compute azimuth angle
            Point3f projectNormal = toCameraVec - cv::norm(toCameraVec) * polarCos * rn;
            double azimuthCos = refDirection.ddot(projectNormal);
            if(azimuthCos > 1.)
                azimuthCos /= azimuthCos;

            float azimuthAngle = std::acos(azimuthCos);
            if((refDirection.cross(projectNormal)).ddot(rn) < 0)
                azimuthAngle  = 2 * CV_PI - azimuthAngle;

            int polarIndex = static_cast<int>(polarAngle * 2 / CV_PI * angleBinCount) % angleBinCount;
            int azimuthIndex = static_cast<int>(azimuthAngle / (2 * CV_PI) * angleBinCount) % angleBinCount;

            CV_Assert(polarIndex < angleBinCount);
            CV_Assert(azimuthIndex < angleBinCount);
            viewAngleHist.at<uchar>(polarIndex, azimuthIndex) = 1;
            framesMask[nearPointFrameIndex] = 1;
        }
        // TODO maybe merge bins with close to zero polar angle?
        int differentViewAngles = countNonZero(viewAngleHist);
        //cout << " " << differentViewAngles << flush;

        CV_Assert(differentViewAngles >= 1);

        if(differentViewAngles < minViewAnglesCount)
            frames[frameIndex]->mask.at<uchar>(y,x) = 0;
    }
}
#endif

static void
prepareFramesForModelRefinement(const Ptr<TrajectoryFrames>& trajectoryFrames, const vector<int>& frameIndices, vector<Ptr<RgbdFrame> >& dstFrames)
{
    dstFrames.resize(trajectoryFrames->frames.size());
    for(size_t frameIndex = 0; frameIndex < dstFrames.size(); frameIndex++)
    {
        const Ptr<RgbdFrame> srcFrame = trajectoryFrames->frames[frameIndex];
        if(std::find(frameIndices.begin(), frameIndices.end(), frameIndex) != frameIndices.end())
        {
            // clone data for used frames because we can modify them
            dstFrames[frameIndex] = new RgbdFrame(srcFrame->image.clone(), srcFrame->depth.clone(),
                                                  trajectoryFrames->objectMasks[frameIndex].clone(), srcFrame->normals.clone(),
                                                  srcFrame->ID);
        }
        else
        {
            dstFrames[frameIndex] = new RgbdFrame(srcFrame->image, srcFrame->depth,
                                                  trajectoryFrames->objectMasks[frameIndex], srcFrame->normals,
                                                  srcFrame->ID);
        }
    }
}

ModelReconstructor::ModelReconstructor()
    : isShowStepResults(false),
      maxBAPosesCount(DEFAULT_MAX_BA_POSES_COUNT)
{}

void ModelReconstructor::reconstruct(const Ptr<TrajectoryFrames>& trajectoryFrames, const Mat& cameraMatrix,
                                     Ptr<ObjectModel>& model) const
{
    CV_Assert(trajectoryFrames);
    CV_Assert(!trajectoryFrames->frames.empty());
    CV_Assert(trajectoryFrames->poses.size() == trajectoryFrames->frames.size());
    CV_Assert(!trajectoryFrames->keyframePosesLinks.empty());

    const float voxelSize = 0.005f;

    if(isShowStepResults)
    {
        cout << "Frame-to-frame odometry result" << endl;
        ObjectModel(trajectoryFrames->frames, trajectoryFrames->poses, cameraMatrix).show(voxelSize, true);
    }

    vector<Mat> refinedPosesSE3;
    vector<int> frameIndices;
    refineGraphSE3(trajectoryFrames->poses, trajectoryFrames->keyframePosesLinks, refinedPosesSE3, frameIndices);

    if(isShowStepResults)
    {
        cout << "Result of the loop closure" << endl;
        ObjectModel(trajectoryFrames->frames, refinedPosesSE3, cameraMatrix, frameIndices).show(voxelSize, true);
    }

    // fill posesLinks with empty Rt because we want that they will be recomputed
    vector<PosesLink> keyframePosesLinks;

    if(maxBAPosesCount > 0 && maxBAPosesCount < static_cast<int>(frameIndices.size())-1)
    {
        vector<int> subsetIndices;
        selectPosesSubset(refinedPosesSE3, frameIndices, subsetIndices, maxBAPosesCount);
        std::sort(subsetIndices.begin(), subsetIndices.end());
        preparePosesLinksWithoutRt(subsetIndices, keyframePosesLinks);
    }
    else
    {
        preparePosesLinksWithoutRt(trajectoryFrames->keyframePosesLinks, keyframePosesLinks);
    }

    vector<Mat> refinedPosesSE3RgbdICP;
    const float pointsPart = 0.05f;
    refineGraphSE3RgbdICP(trajectoryFrames->frames, refinedPosesSE3,
                          keyframePosesLinks, cameraMatrix, pointsPart, refinedPosesSE3RgbdICP, frameIndices);

    if(isShowStepResults)
    {
        cout << "Result of RgbdICP for camera poses" << endl;
        ObjectModel(trajectoryFrames->frames, refinedPosesSE3RgbdICP, cameraMatrix, frameIndices).show(0.001, true);
    }

    vector<Ptr<RgbdFrame> > objectFrames; // with mask for object points only,
                                          // they will modified while refining the object points
    prepareFramesForModelRefinement(trajectoryFrames, frameIndices, objectFrames);

#if 0
    vector<Mat> refinedAllPoses;
    refineGraphSE3Segment(trajectoryFrames->poses, refinedPosesSE3RgbdICP, frameIndices, refinedAllPoses);
    genModel(objectFrames, refinedAllPoses, cameraMatrix, frameIndices)->show();
    filterFramesByViewHist(objectFrames, frameIndices, refinedAllPoses, cameraMatrix);
    genModel(objectFrames, refinedAllPoses, cameraMatrix, frameIndices)->show();
#endif

    vector<Mat> refinedSE3ICPSE3ModelPoses;
    refineGraphSE3RgbdICPModel(objectFrames, refinedPosesSE3RgbdICP,
                               keyframePosesLinks, cameraMatrix, refinedSE3ICPSE3ModelPoses, frameIndices);

    model = new ObjectModel(objectFrames, refinedSE3ICPSE3ModelPoses, cameraMatrix, frameIndices);

    if(isShowStepResults)
    {
        cout << "Result of RgbdICP  for camera poses and model points refinement" << endl;
        model->show(0.001, true);
    }
}
