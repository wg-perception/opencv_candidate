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
            perspectiveTransform(frames[frameIndex]->normals.reshape(3,1), tnormals, poses[frameIndex]);

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
                n *= 1./cv::norm(n);

                if(isValidDepth(p.z) && mask.at<uchar>(y,x))
                {
                    //CV_Assert(cv::norm(n) > 0.98);

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
        genModel(trajectoryFrames->frames, trajectoryFrames->poses, cameraMatrix)->show(voxelSize);
    }

    vector<Mat> refinedPosesSE3;
    vector<int> frameIndices;
    refineGraphSE3(trajectoryFrames->poses, trajectoryFrames->keyframePosesLinks, refinedPosesSE3, frameIndices);

    if(isShowStepResults)
    {
        cout << "Result of the loop closure" << endl;
        genModel(trajectoryFrames->frames, refinedPosesSE3, cameraMatrix, frameIndices)->show(voxelSize);
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
        genModel(trajectoryFrames->frames, refinedPosesSE3RgbdICP, cameraMatrix, frameIndices)->show(/*voxelSize*/);
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

    model = genModel(objectFrames, refinedSE3ICPSE3ModelPoses, cameraMatrix, frameIndices);

    if(isShowStepResults)
    {
        cout << "Result of RgbdICP  for camera poses and model points refinement" << endl;
        model->show();
    }
}

#if 1
Ptr<ObjectModel> ModelReconstructor::genModel(const std::vector<Ptr<RgbdFrame> >& frames,
                                              const std::vector<cv::Mat>& poses, const Mat& cameraMatrix, const std::vector<int>& frameIndices)
{
    CV_Assert(frames.size() == poses.size());

    Ptr<ObjectModel> model = new ObjectModel();

    cout << "Convert frames to model" << endl;
    size_t usedFrameCount = frameIndices.empty() ? frames.size() : frameIndices.size();
    for(size_t i = 0; i < usedFrameCount; i++)
    {
        int  frameIndex = frameIndices.empty() ? i : frameIndices[i];
        CV_Assert(frameIndex < static_cast<int>(frames.size()) && frameIndex >= 0);

        const Ptr<RgbdFrame>& frame = frames[frameIndex];

        CV_Assert(!frame->image.empty());
        CV_Assert(frame->depth.size() == frame->image.size());
        CV_Assert(frame->mask.size() == frame->image.size());
        CV_Assert(frame->normals.empty() || frame->normals.size() == frame->image.size());

        const int maskElemCount = countNonZero(frame->mask);
        model->colors.reserve(model->colors.size() + maskElemCount);
        model->points3d.reserve(model->points3d.size() + maskElemCount);
        if(!frame->normals.empty())
            model->normals.reserve(model->normals.size() + maskElemCount);

        Mat cloud;
        depthTo3d(frame->depth, cameraMatrix, cloud);
        Mat transfPoints3d;
        perspectiveTransform(cloud.reshape(3,1), transfPoints3d, poses[frameIndex]);
        transfPoints3d = transfPoints3d.reshape(3, cloud.rows);

        Mat transfNormals;
        if(!frame->normals.empty())
        {
            perspectiveTransform(frame->normals.reshape(3,1), transfNormals, poses[frameIndex]);
            transfNormals = transfNormals.reshape(3, frame->normals.rows);
        }

        for(int y = 0, pointIndex = 0; y < frame->mask.rows; y++)
        {
            const uchar* maskRow = frame->mask.ptr<uchar>(y);
            for(int x = 0; x < frame->mask.cols; x++, pointIndex++)
            {
                if(maskRow[x] && isValidDepth(frame->depth.at<float>(y,x)))
                {
                    model->colors.push_back(frame->image.at<Vec3b>(y,x));
                    model->points3d.push_back(transfPoints3d.at<Point3f>(y,x));
                    if(!frame->normals.empty())
                    {
                        Point3f n = transfNormals.at<Point3f>(y,x);
                        n *= 1./cv::norm(n);
                        model->normals.push_back(n);
                    }
                }
            }
        }
        CV_Assert(model->colors.size() == model->points3d.size());
        CV_Assert(model->normals.empty() || model->normals.size() == model->points3d.size());
    }

    return model;
}

#else

template<class T>
void voxelFilter(pcl::PointCloud<T>& cloud, double gridSize)
{
    if(gridSize > 0.f)
    {
        typename pcl::PointCloud<T>::ConstPtr cloudPtr = boost::make_shared<const pcl::PointCloud<T> >(cloud);
        pcl::PointCloud<T> cloudDownsampled;
        pcl::VoxelGrid<T> voxelGridFilter;
        voxelGridFilter.setLeafSize(gridSize, gridSize, gridSize);
        voxelGridFilter.setDownsampleAllData(true);
        voxelGridFilter.setInputCloud(cloudPtr);
        voxelGridFilter.filter(cloudDownsampled);
        cloudDownsampled.swap(cloud);
    }
}

Ptr<ObjectModel> ModelReconstructor::genModel(const std::vector<Ptr<RgbdFrame> >& frames,
                                              const std::vector<cv::Mat>& poses, const Mat& cameraMatrix, const std::vector<int>& frameIndices)
{
    CV_Assert(frames.size() == poses.size());

    Ptr<ObjectModel> model = new ObjectModel();

    size_t usedFrameCount = frameIndices.empty() ? frames.size() : frameIndices.size();

    cout << "Convert frames to model" << endl;

    pcl::PointCloud<pcl::PointXYZINormal> totalCloud;
    for(size_t i = 0; i < usedFrameCount; i++)
    {
        int  frameIndex = frameIndices.empty() ? i : frameIndices[i];
        CV_Assert(frameIndex < static_cast<int>(frames.size()) && frameIndex >= 0);

        const Ptr<RgbdFrame>& frame = frames[frameIndex];

        CV_Assert(!frame->image.empty());
        CV_Assert(frame->depth.size() == frame->image.size());
        CV_Assert(frame->mask.size() == frame->image.size());
        CV_Assert(frame->normals.empty() || frame->normals.size() == frame->image.size());

        Mat cloud;
        depthTo3d(frame->depth, cameraMatrix, cloud);
        Mat transfPoints3d;
        perspectiveTransform(cloud.reshape(3,1), transfPoints3d, poses[frameIndex]);
        transfPoints3d = transfPoints3d.reshape(3, cloud.rows);

        Mat transfNormals;
        CV_Assert(!frame->normals.empty());
        perspectiveTransform(frame->normals.reshape(3,1), transfNormals, poses[frameIndex]);
        transfNormals = transfNormals.reshape(3, frame->normals.rows);

        for(int y = 0; y < frame->mask.rows; y++)
        {
            const uchar* maskRow = frame->mask.ptr<uchar>(y);
            for(int x = 0; x < frame->mask.cols; x++)
            {
                if(maskRow[x] && isValidDepth(frame->depth.at<float>(y,x)))
                {
                    Point3f coord = transfPoints3d.at<Point3f>(y,x);

                    pcl::PointXYZINormal p;
                    p.x = coord.x; p.y = coord.y; p.z = coord.z;

                    if(!frame->normals.empty())
                    {
                        Point3f n = transfNormals.at<Point3f>(y,x);
                        n *= 1./cv::norm(n);

                        p.normal_x = n.x; p.normal_y = n.y; p.normal_z = n.z;
                    }
                    else
                        p.normal_x = p.normal_y = p.normal_z = 0;

                    totalCloud.push_back(p);
                }
            }
        }
    }

    const double gridSize = 0.0005;
    voxelFilter(totalCloud, gridSize);

    model->colors.resize(totalCloud.size());
    model->points3d.resize(totalCloud.size());
    model->normals.resize(totalCloud.size());

    // save points and normals
    for(size_t i = 0; i < totalCloud.points.size(); i++)
    {
        const pcl::PointXYZINormal& p = totalCloud.points[i];
        model->points3d[i] = Point3f(p.x, p.y, p.z);
        model->normals[i] = Point3f(p.normal_x, p.normal_y, p.normal_z);
    }

    // color
    vector<Vec3d> colorSum(model->points3d.size(), Vec3d(0,0,0));
    vector<double> wSum(model->points3d.size(), 0);

    for(size_t i = 0; i < usedFrameCount; i++)
    {
        int  frameIndex = frameIndices.empty() ? i : frameIndices[i];
        CV_Assert(frameIndex < static_cast<int>(frames.size()) && frameIndex >= 0);

        const Ptr<RgbdFrame>& frame = frames[frameIndex];

        const Mat cameraPose = poses[frameIndex];
        Point3f cameraPoint;
        CV_Assert(cameraPose.type() == CV_64FC1);
        cameraPoint.x = cameraPose.at<double>(0,3);
        cameraPoint.y = cameraPose.at<double>(1,3);
        cameraPoint.z = cameraPose.at<double>(2,3);

        Mat zBuffer(frame->image.size(), CV_32FC1, FLT_MAX);
        Mat zIndex(frame->image.size(), CV_32SC1, -1);

        vector<Point3f> points3d;
        perspectiveTransform(model->points3d, points3d, poses[frameIndex].inv(DECOMP_SVD));

        vector<Point2f> points2d;
        projectPoints(points3d, Mat::eye(3, 3, CV_64FC1), Mat::zeros(3, 1, CV_64FC1), cameraMatrix, Mat(), points2d);

        Rect r(0, 0, zBuffer.cols, zBuffer.rows);

        for(size_t pi = 0; pi < points3d.size(); pi++)
        {
            const Point3f& p3 = points3d[pi];
            Point p2(cvRound(points2d[pi].x), cvRound(points2d[pi].y));

            if(p3.z <= 0 || !r.contains(p2) || p3.z >= zBuffer.at<float>(p2.y, p2.x))
                continue;

            zBuffer.at<float>(p2.y, p2.x) = p3.z;
            zIndex.at<int>(p2.y, p2.x) = static_cast<int>(pi);
        }

        for(int y = 0; y < zIndex.rows; y++)
        {
            for(int x = 0; x < zIndex.cols; x++)
            {
                int pointIndex = zIndex.at<int>(y, x);

                if(pointIndex < 0)
                    continue;

                double polarCos = model->normals[pointIndex].ddot(cameraPoint - model->points3d[pointIndex]);
                double w = std::abs(polarCos);

                colorSum[pointIndex] += w * frame->image.at<Vec3b>(y,x);
                wSum[pointIndex] += w;
            }
        }
    }

    for(size_t i = 0; i < colorSum.size(); i++)
    {
        if(wSum[i] > FLT_EPSILON)
        {
            Vec3b color = colorSum[i] / wSum[i];
            model->colors[i] = color;
        }
        // TODO remove points with zero wSum
    }

    CV_Assert(model->points3d.size() == model->colors.size());

    return model;
}
#endif
