#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "reconst3d.hpp"

using namespace std;
using namespace cv;

const char checkDataMessage[] = "Please check the data! Sequential frames have to be close to each other (location and color). "
                                "The first and one of the last frames also have to be "
                                "really taken from close camera positions and the close lighting conditions.";

// get norms
//
static inline
float tvecNorm(const Mat& Rt)
{
    return norm(Rt(Rect(3,0,1,3)));
}

static inline
float rvecNormDegrees(const Mat& Rt)
{
    Mat rvec;
    Rodrigues(Rt(Rect(0,0,3,3)), rvec);
    return norm(rvec) * 180. / CV_PI;
}

//
static
float computeInliersRatio(const Ptr<OdometryFrame>& srcFrame,
                          const Ptr<OdometryFrame>& dstFrame,
                          const Mat& Rt, const Mat& cameraMatrix,
                          int maxColorDiff=OnlineCaptureServer::DEFAULT_MAX_CORRESP_COLOR_DIFF,
                          float maxDepthDiff=OnlineCaptureServer::DEFAULT_MAX_CORRESP_DEPTH_DIFF())
{
    Mat warpedSrcImage, warpedSrcDepth, warpedSrcMask;
    warpFrame(srcFrame->image, srcFrame->depth, srcFrame->mask,
              Rt, cameraMatrix, Mat(), warpedSrcImage, &warpedSrcDepth, &warpedSrcMask);

    int inliersCount = countNonZero(dstFrame->mask & warpedSrcMask &
                                    (cv::abs(dstFrame->image - warpedSrcImage) < maxColorDiff) &
                                    (cv::abs(dstFrame->depth - warpedSrcDepth) < maxDepthDiff));

    int intersectSize = countNonZero(dstFrame->mask & warpedSrcMask);
    return intersectSize ? static_cast<float>(inliersCount) / intersectSize : 0;
}

//---------------------------------------------------------------------------------------------------------------------------
PosesLink::PosesLink(int srcIndex, int dstIndex, const Mat& Rt)
    : srcIndex(srcIndex), dstIndex(dstIndex), Rt(Rt)
{}

void TrajectoryFrames::push(const Ptr<RgbdFrame>& frame, const Mat& pose, const Mat& objectMask, int state)
{
    CV_Assert(frame);
    CV_Assert(!pose.empty() && pose.size() == Size(4,4) && pose.type() == CV_64FC1);

    Ptr<RgbdFrame> rgbdFrame = new RgbdFrame();
    *rgbdFrame = *frame;
    frames.push_back(rgbdFrame);
    poses.push_back(pose);
    objectMasks.push_back(objectMask);
    states.push_back(state);
}

void TrajectoryFrames::clear()
{
    frames.clear();
    poses.clear();
    states.clear();
    posesLinks.clear();
}

OnlineCaptureServer::FramePushOutput::FramePushOutput()
    : state(0)
{}

OnlineCaptureServer::OnlineCaptureServer() :
    maxCorrespColorDiff(DEFAULT_MAX_CORRESP_COLOR_DIFF),
    maxCorrespDepthDiff(DEFAULT_MAX_CORRESP_DEPTH_DIFF()),
    minInliersRatio(DEFAULT_MIN_INLIERS_RATIO()),
    skippedTranslation(DEFAULT_SKIPPED_TRANSLATION()),
    minTranslationDiff(DEFAULT_MIN_TRANSLATION_DIFF()),
    maxTranslationDiff(DEFAULT_MAX_TRANSLATION_DIFF()),
    minRotationDiff(DEFAULT_MIN_ROTATION_DIFF()),
    maxRotationDiff(DEFAULT_MAX_ROTATION_DIFF()),
    isInitialied(false),
    isFinalized(false)
{}

void OnlineCaptureServer::filterImage(const Mat& src, Mat& dst) const
{
    dst = src; // TODO maybe median
    //medianBlur(src, dst, 3);
}

void OnlineCaptureServer::firterDepth(const Mat& src, Mat& dst) const
{
    dst = src; // TODO maybe bilateral
}

Ptr<OnlineCaptureServer::FramePushOutput> OnlineCaptureServer::push(const Mat& _image, const Mat& _depth, int frameID)
{
    Ptr<FramePushOutput> pushOutput = new FramePushOutput();

    CV_Assert(isInitialied);
    CV_Assert(!isFinalized);

    CV_Assert(!normalsComputer.empty());
    CV_Assert(!tableMasker.empty());
    CV_Assert(!odometry.empty());

    if(isTrajectoryBroken)
    {
        cout << "Frame " << frameID << ". Trajectory was broken starting from keyframe " << (*trajectoryFrames->frames.rbegin())->ID << "." << endl;
        return pushOutput;
    }

    if(isLoopClosed)
    {
        cout << "Frame " << frameID << ". Loop is already closed. Keyframes count " << trajectoryFrames->frames.size() << endl;
        return pushOutput;
    }

    if(_image.empty() || _depth.empty())
    {
        cout << "Warning: Empty frame " << frameID << endl;
        return pushOutput;
    }

    //color information is ingored now but can be used in future
    Mat _grayImage = _image;
    if (_image.channels() == 3)
        cvtColor(_image, _grayImage, CV_BGR2GRAY);

    CV_Assert(_grayImage.type() == CV_8UC1);
    CV_Assert(_depth.type() == CV_32FC1);
    CV_Assert(_grayImage.size() == _depth.size());

    Mat image, depth;
    filterImage(_grayImage, image);
    filterImage(_depth, depth);

    Mat cloud;
    depthTo3d(depth, cameraMatrix, cloud);

    Mat normals = (*normalsComputer)(cloud);

    Mat tableWithObjectMask;
    bool isTableMaskOk = (*tableMasker)(cloud, normals, tableWithObjectMask, &pushOutput->objectMask);
    pushOutput->frame = new OdometryFrame(_grayImage, _depth, tableWithObjectMask, normals, frameID);
    if(!isTableMaskOk)
    {
        cout << "Warning: bad table mask for the frame " << frameID << endl;
        return pushOutput;
    }

    //Ptr<OdometryFrameCache> currFrame = new OdometryFrameCache(image, depth, tableWithObjectMask);
    Ptr<OdometryFrame> currFrame = pushOutput->frame;

    if(lastKeyframe.empty())
    {
        firstKeyframe = currFrame;
        pushOutput->state |= TrajectoryFrames::KEYFRAME;
        pushOutput->pose = Mat::eye(4,4,CV_64FC1);
        cout << "First keyframe ID " << frameID << endl;
    }
    else
    {
        // find frame to frame motion transformations
        {
            Mat Rt;
            cout << "odometry " << frameID << " -> " << prevFrameID << endl;
            if(odometry->compute(currFrame, prevFrame, Rt) && computeInliersRatio(currFrame, prevFrame, Rt, cameraMatrix, maxCorrespColorDiff,
                                                                                  maxCorrespDepthDiff) >= minInliersRatio)
            {
                pushOutput->state |= TrajectoryFrames::VALIDFRAME;
            }

            pushOutput->pose = prevPose * Rt;
            if(!(pushOutput->state & TrajectoryFrames::VALIDFRAME))
            {
                cout << "Warning: Bad odometry (too far motion or low inliers ratio) " << frameID << "->" << prevFrameID << endl;
                return pushOutput;
            }
        }

        // check for the current frame: is it keyframe?
        {
            Mat Rt = (*trajectoryFrames->poses.rbegin()).inv(DECOMP_SVD) * pushOutput->pose;
            float tnorm = tvecNorm(Rt);
            float rnorm = rvecNormDegrees(Rt);

            if(tnorm > maxTranslationDiff || rnorm > maxRotationDiff)
            {
                cout << "Camera trajectory is broken (starting from " << (*trajectoryFrames->frames.rbegin())->ID << " frame)." << endl;
                cout << checkDataMessage << endl;
                isTrajectoryBroken = true;
                return pushOutput;
            }

            if((tnorm >= minTranslationDiff || rnorm >= minRotationDiff)) // we don't check inliers ratio here because it was done by frame-to-frame above
            {
                translationSum += tnorm;
                if(isLoopClosing)
                    cout << "possible ";
                cout << "keyframe ID " << frameID << endl;
                pushOutput->state |= TrajectoryFrames::KEYFRAME;
            }
        }

        // match with the first keyframe
        if(translationSum > skippedTranslation) // ready for closure
        {
            Mat Rt;
            if(odometry->compute(currFrame, firstKeyframe, Rt))
            {
                // we check inliers ratio for the loop closure frames because we didn't do this before
                float inliersRatio = computeInliersRatio(currFrame, firstKeyframe, Rt, cameraMatrix, maxCorrespColorDiff, maxCorrespDepthDiff);
                if(inliersRatio > minInliersRatio)
                {
                    if(inliersRatio >= closureInliersRatio)
                    {
                        isLoopClosing = true;
                        closureInliersRatio = inliersRatio;
                        closureFrame = currFrame;
                        closureFrameID = frameID;
                        closurePoseWithFirst = Rt;
                        closurePose = pushOutput->pose;
                        closureObjectMask = pushOutput->objectMask;
                        closureBgrImage = _image;
                        isClosureFrameAdded = pushOutput->state == TrajectoryFrames::KEYFRAME;
                    }
                    else if(isLoopClosing)
                    {
                        isLoopClosed = true;
                    }
                }
                else if(isLoopClosing)
                {
                    isLoopClosed = true;
                }
            }
            else if(isLoopClosing)
            {
                isLoopClosed = true;
            }
        }
    }

    if((pushOutput->state == TrajectoryFrames::KEYFRAME) && (!isLoopClosed || frameID <= closureFrameID ))
    {
        trajectoryFrames->push(new RgbdFrame(_image, _depth, currFrame->mask, currFrame->normals, frameID),
                               pushOutput->pose, pushOutput->objectMask, TrajectoryFrames::KEYFRAME);
        lastKeyframe = currFrame;
    }

    prevFrame = currFrame;
    prevFrameID = frameID;
    prevPose = pushOutput->pose.clone();

    return pushOutput;
}

void OnlineCaptureServer::reset()
{
    trajectoryFrames = new TrajectoryFrames();

    firstKeyframe.release();
    lastKeyframe.release();
    prevFrame.release();
    closureFrame.release();

    prevPose.release();
    prevFrameID = -1;

    isTrajectoryBroken = false;
    isLoopClosing = false;
    isLoopClosed = false;
    translationSum = 0.;
    closureInliersRatio = 0.;
    closureFrameID = -1;
    isClosureFrameAdded = false;

    closureBgrImage.release();
    closureObjectMask.release();
    closurePose.release();
    closurePoseWithFirst.release();

    isFinalized = false;
}

void OnlineCaptureServer::initialize(const Size& frameResolution, int storeFramesWithState)
{
    if(storeFramesWithState != TrajectoryFrames::KEYFRAME)
        CV_Error(CV_StsError, "TODO: add a support of storing all valid frames. Now only keyframes are supported (have to modified push method)!!!\n");

    reset();

    CV_Assert(!cameraMatrix.empty());
    CV_Assert(cameraMatrix.type() == CV_32FC1);
    CV_Assert(cameraMatrix.size() == Size(3,3));

    normalsComputer = new RgbdNormals(frameResolution.height, frameResolution.width, CV_32FC1, cameraMatrix); // inner
    if(tableMasker.empty())
    {
        tableMasker = new TableMasker();
        Ptr<RgbdPlane> planeComputer = new RgbdPlane();
        planeComputer->set("sensor_error_a", 0.0075f);
        tableMasker->set("planeComputer", planeComputer);
    }
    tableMasker->set("cameraMatrix", cameraMatrix);

    if(odometry.empty())
        odometry = new RgbdOdometry();
    odometry->set("cameraMatrix", cameraMatrix);

    isInitialied = true;
}

Ptr<TrajectoryFrames> OnlineCaptureServer::finalize()
{
    CV_Assert(isInitialied);
    CV_Assert(!isFinalized);

    if(!closureFrame.empty())
    {
    	cout << "Closure frame index " << closureFrame->ID << endl;
        CV_Assert((*trajectoryFrames->frames.rbegin())->ID <= closureFrameID);
        if(!isClosureFrameAdded)
            trajectoryFrames->push(new RgbdFrame(closureBgrImage, closureFrame->depth, closureFrame->mask, closureFrame->normals, closureFrameID),
                                   closurePose, closureObjectMask, TrajectoryFrames::KEYFRAME);

        // fill camera poses links
        trajectoryFrames->posesLinks.clear();
        for(size_t i = 1; i < trajectoryFrames->frames.size(); i++)
            trajectoryFrames->posesLinks.push_back(PosesLink(i-1, i));
        trajectoryFrames->posesLinks.push_back(PosesLink(0, trajectoryFrames->poses.size()-1, closurePoseWithFirst));

        cout << "Last keyframe index " << closureFrameID << endl;
        cout << "keyframes count = " << trajectoryFrames->frames.size() << endl;

        isLoopClosed = true;
    }
    else
    {

        cout << "The algorithm can not make loop closure on given data. " << endl;
        cout << checkDataMessage << endl;
    }

    isFinalized = true;

    return trajectoryFrames;
}
