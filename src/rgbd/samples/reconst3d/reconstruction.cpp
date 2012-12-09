#include <opencv2/rgbd/rgbd.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "reconst3d.hpp"
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

static Mat
refineObjectMask(const Mat& initObjectMask)
{
    vector<vector<Point> > contours;
    findContours(initObjectMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    if(contours.empty())
        return initObjectMask.clone();

    int maxMaskArea = 0.;
    int objectMaskIndex = -1;
    for(size_t i = 0; i < contours.size(); i++)
    {
        Mat mask(initObjectMask.size(), CV_8UC1, Scalar(0));
        drawContours(mask, contours, i, Scalar(255), CV_FILLED, 8);

        int curMaskArea = countNonZero(mask);
        if(curMaskArea > maxMaskArea)
        {
            maxMaskArea = curMaskArea;
            objectMaskIndex = i;
        }
    }

    Mat objectMask(initObjectMask.size(), CV_8UC1, Scalar(0));
    drawContours(objectMask, contours, objectMaskIndex, Scalar(255), CV_FILLED, 8);

    return objectMask & initObjectMask;
}

static void
prepareFramesForModelRefinement(const Ptr<TrajectoryFrames>& trajectoryFrames, vector<Ptr<RgbdFrame> >& dstFrames)
{
    dstFrames.resize(trajectoryFrames->frames.size());
    for(size_t i = 0; i < dstFrames.size(); i++)
    {
        const Ptr<RgbdFrame> srcFrame = trajectoryFrames->frames[i];
        dstFrames[i] = new RgbdFrame(srcFrame->image.clone(), srcFrame->depth.clone(),
                                     refineObjectMask(trajectoryFrames->objectMasks[i]), srcFrame->normals.clone(),
                                     srcFrame->ID);
    }
}

ModelReconstructor::ModelReconstructor()
    : isShowStepResults(false)
{}

void ModelReconstructor::reconstruct(const Ptr<TrajectoryFrames>& trajectoryFrames, const Mat& cameraMatrix,
                                     Ptr<ObjectModel>& model) const
{
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
    preparePosesLinksWithoutRt(trajectoryFrames->keyframePosesLinks, keyframePosesLinks);

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
    prepareFramesForModelRefinement(trajectoryFrames, objectFrames);

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
                        model->normals.push_back(transfNormals.at<Point3f>(y,x));
                }
            }
        }
        CV_Assert(model->colors.size() == model->points3d.size());
        CV_Assert(model->normals.empty() || model->normals.size() == model->points3d.size());
    }

    return model;
}
