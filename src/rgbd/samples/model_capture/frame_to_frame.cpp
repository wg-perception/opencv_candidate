#include "model_capture.hpp"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

const char checkDataMessage[] = "Please check the data! Sequential frames have to be close to each other (location and color). "
                                "The first and one of the last frames also have to be "
                                "really taken from close camera positions and the lighting has to be the same.";
static
float computeInliersRatio(const Ptr<OdometryFrameCache>& srcFrame,
                          const Ptr<OdometryFrameCache>& dstFrame,
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

#if 0
    cout << "inliersRatio " << static_cast<float>(inliersCount) / countNonZero(dstFrame->mask) << endl;

    imshow("src_image", srcFrame->image);
    imshow("dst_image", dstFrame->image);
    imshow("warped_image", warpedSrcImage);
    imshow("diff_image", abs(dstFrame->image - warpedSrcImage));
    //imshow("src_mask", srcFrame->mask);
    //imshow("dst_mask", dstFrame->mask);
    //imshow("warped_mask", warpedSrcMask);
    waitKey();
#endif

    int intersectSize = countNonZero(dstFrame->mask & warpedSrcMask);
    return intersectSize ? static_cast<float>(inliersCount) / intersectSize : 0;
}

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

bool OnlineCaptureServer::isLoopClosed() const
{
  return isClosed;
}

Mat OnlineCaptureServer::push(const Mat& _image, const Mat& _depth, int frameID, bool *isKeyframePtr)
{
    CV_Assert(isInitialied);
    CV_Assert(!isFinalized);

    CV_Assert(!normalsComputer.empty());
    CV_Assert(!tableMasker.empty());
    CV_Assert(!odometry.empty());

    if(isClosed)
    {
        cout << "Frame " << frameID << ". Loop is already closed. Keyframes count " << keyframesData->frames.size() << endl;
        return Mat();
    }

    if(_image.empty() || _depth.empty())
    {
        cout << "Warning: Empty frame " << frameID << endl;
        return Mat();
    }

    //color information is ingored now but can be used in future
    Mat _grayImage = _image;
    if (_image.channels() == 3)
    {
      cvtColor(_image, _grayImage, CV_BGR2GRAY);
    }

    CV_Assert(_grayImage.type() == CV_8UC1);
    CV_Assert(_depth.type() == CV_32FC1);
    CV_Assert(_grayImage.size() == _depth.size());

    Mat image, depth;
    filterImage(_grayImage, image);
    filterImage(_depth, depth);

    Mat cloud;
    depthTo3d(depth, cameraMatrix, cloud);

    Mat normals = (*normalsComputer)(cloud);

    Mat tableWithObjectMask, tableMask;
    if(!(*tableMasker)(cloud, normals, tableWithObjectMask, &tableMask))
    {
        cout << "Warning: bad table mask for the frame " << frameID << endl;
        return Mat();
    }

    Ptr<OdometryFrameCache> currFrame = new OdometryFrameCache(image, depth, tableWithObjectMask);

    Mat pose;
    bool isKeyframe = false;
    if(lastKeyframe.empty())
    {
        firstKeyframe = currFrame;
        pose = Mat::eye(4,4,CV_64FC1);
        isKeyframe = true;
        cout << "First keyframe ID " << frameID << endl;
    }
    else
    {
        // find frame to frame motion transformations
        {
            Mat Rt;
            cout << "odometry " << frameID << " -> " << prevFrameID << endl;
            if(!odometry->compute(*currFrame, *prevFrame, Rt) ||
               computeInliersRatio(currFrame, prevFrame, Rt, cameraMatrix, maxCorrespColorDiff, maxCorrespDepthDiff) < minInliersRatio)
            {
                cout << "Warning: Bad odometry (too far motion or low inliers ratio) " << frameID << "->" << prevFrameID << endl;
                return Mat();
            }
            pose = prevPose * Rt;
        }

        // check for the current frame: is it keyframe?
        {
            Mat Rt = (*keyframesData->poses.rbegin()).inv(DECOMP_SVD) * pose;

            float tnorm = tvecNorm(Rt);
            float rnorm = rvecNormDegrees(Rt);

            if(tnorm > maxTranslationDiff || rnorm > maxRotationDiff)
            {
                cout << "Camera trajectory is broken (starting from " << (*keyframesData->frames.rbegin())->ID << " frame)." << endl;
                cout << checkDataMessage << endl;
                return Mat();
            }

            if((tnorm >= minTranslationDiff || rnorm >= minRotationDiff)) // we don't check inliers ratio here because it was done by frame-to-frame above
            {
                translationSum += tnorm;
                if(isClosing)
                    cout << "possible ";
                cout << "keyframe ID " << frameID << endl;
                isKeyframe = true;
            }
        }

        // match with the first keyframe
        if(translationSum > skippedTranslation) // ready for closure
        {
            Mat Rt;
            if(odometry->compute(*currFrame, *firstKeyframe, Rt))
            {
                // we check inliers ratio for the loop closure frames because we didn't do this before
                float inliersRatio = computeInliersRatio(currFrame, firstKeyframe, Rt, cameraMatrix, maxCorrespColorDiff, maxCorrespDepthDiff);
                if(inliersRatio > minInliersRatio)
                {
                    if(inliersRatio >= closureInliersRatio)
                    {
                        isClosing = true;
                        closureInliersRatio = inliersRatio;
                        closureFrame = currFrame;
                        closureFrameID = frameID;
                        closurePoseWithFirst = Rt;
                        closurePose = pose;
                        closureTableMask = tableMask;
                        isClosureFrameAdded = isKeyframe;
                    }
                    else if(isClosing)
                    {
                        isClosed = true;
                    }
                }
                else if(isClosing)
                {
                    isClosed = true;
                }
            }
            else if(isClosing)
            {
                isClosed = true;
            }
        }
    }

    if(isKeyframe && (!isClosed || frameID <= closureFrameID ))
    {
        Ptr<OdometryFrameCache> frame = new OdometryFrameCache(currFrame->image, currFrame->depth, tableWithObjectMask, frameID);
        keyframesData->frames.push_back(frame);
        keyframesData->tableMasks.push_back(tableMask);
        keyframesData->poses.push_back(pose);

        lastKeyframe = currFrame;
    }

    if (isKeyframePtr != 0)
    {
      *isKeyframePtr = isKeyframe;
    }

    prevFrame = currFrame;
    prevFrameID = frameID;
    prevPose = pose.clone();

    return pose.clone();
}

void OnlineCaptureServer::reset()
{
    keyframesData = new KeyframesData();

    firstKeyframe.release();
    lastKeyframe.release();
    prevFrame.release();
    closureFrame.release();

    prevPose.release();
    prevFrameID = -1;

    isClosing = false;
    isClosed = false;
    translationSum = 0.;
    closureInliersRatio = 0.;
    closureFrameID = -1;
    closureTableMask.release();
    isClosureFrameAdded = false;
    closurePose.release();
    closurePoseWithFirst.release();

    isFinalized = false;
}

void OnlineCaptureServer::initialize(const Size& frameResolution)
{
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

cv::Ptr<KeyframesData> OnlineCaptureServer::finalize()
{
    CV_Assert(isInitialied);
    CV_Assert(!isFinalized);

    if(!closureFrame.empty())
    {
        CV_Assert((*keyframesData->frames.rbegin())->ID <= closureFrameID);
        if(!isClosureFrameAdded)
        {
            Ptr<OdometryFrameCache> frame = new OdometryFrameCache(closureFrame->image, closureFrame->depth, closureFrame->mask, closureFrameID);
            keyframesData->frames.push_back(frame);
            keyframesData->tableMasks.push_back(closureTableMask);
            keyframesData->poses.push_back(closurePose);
        }
        keyframesData->poses.push_back(closurePoseWithFirst);

        cout << "Last keyframe index " << closureFrameID << endl;
        cout << "keyframes count = " << keyframesData->frames.size() << endl;

        isClosed = true;
    }
    else
    {
        cout << "The algorithm can not make loop closure on given data. " << endl;
        cout << checkDataMessage << endl;
    }

    isFinalized = true;

    return keyframesData;
}
