#include "model_capture.hpp"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

static
float computeInliersRatio(const Ptr<OdometryFrameCache>& srcFrame,
                          const Ptr<OdometryFrameCache>& dstFrame,
                          const Mat& Rt, const Mat& cameraMatrix)
{
    const int maxColorDiff = 30; // it's rough now, because first and last frame may have large changes of light conditions
                                 // TODO: do something with light changes
    const float maxDepthDiff = 0.01; // meters

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

bool frameToFrameProcess(vector<Ptr<OdometryFrameCache> >& frames,
                         const Mat& cameraMatrix, const Ptr<Odometry>& odometry,
                         vector<Ptr<OdometryFrameCache> >& keyframes, vector<Mat>& keyframePoses,
                         vector<int>* keyframeIndices)
{
    CV_Assert(!frames.empty());

    keyframes.clear();
    keyframePoses.clear();
    if(keyframeIndices) keyframeIndices->clear();

    // first nonempty frame is the first keyframe
    Ptr<OdometryFrameCache> firstFrame;
    vector<Mat> frameToFramePoses;
    size_t firstKeyframeIndex = 0;
    for(; firstKeyframeIndex < frames.size(); firstKeyframeIndex++)
    {
        if(!frames[firstKeyframeIndex].empty())
        {
            firstFrame = frames[firstKeyframeIndex];
            keyframes.push_back(firstFrame);

            keyframePoses.push_back(Mat::eye(4,4,CV_64FC1));
            frameToFramePoses.push_back(*keyframePoses.rbegin());

            if(keyframeIndices) keyframeIndices->push_back(firstKeyframeIndex);
            break;
        }

        frameToFramePoses.push_back(Mat());
    }

    if(firstKeyframeIndex == frames.size())
    {
        cout << "Frames are empty." << endl;
        return false;
    }

    // find frame to frame motion transformations
    {
        Ptr<OdometryFrameCache> prevFrame = firstFrame, currFrame;
        int prevFrameIndex = firstKeyframeIndex;
        Mat prevPose = frameToFramePoses.rbegin()->clone();

        for(size_t i = firstKeyframeIndex + 1; i < frames.size(); i++)
        {
            currFrame = frames[i];
            if(currFrame.empty())
            {
                frameToFramePoses.push_back(Mat());
                cout << "Warning: Emty frame " << i << endl;
                continue;
            }

            Mat Rt;
            cout << "odometry " << i << " -> " << prevFrameIndex << endl;
            if(!odometry->compute(*currFrame, *prevFrame, Rt))
            {
                frameToFramePoses.push_back(Mat());
                cout << "Warning: Bad odometry " << i << "->" << prevFrameIndex << endl;
#if 0
                cout << "tnorm " << tvecNorm(Rt) << endl;
                cout << "rnorm " << rvecNormDegrees(Rt) << endl;

                Mat warpedImage;
                warpFrame(currFrame->image, currFrame->depth, currFrame->mask, Rt, cameraMatrix, Mat(),
                          warpedImage);

                imshow("diff", prevFrame->image - warpedImage);
                waitKey();
#endif
                continue;
            }

            prevFrame = currFrame;
            prevFrameIndex = i;
            prevPose = prevPose * Rt;
            frameToFramePoses.push_back(prevPose.clone());
        }
    }

    CV_Assert(frames.size() == frameToFramePoses.size());

    cout << "First keyframe index " << firstKeyframeIndex << endl;

    float translationSum = 0;
    int lastKeyframeIndex = firstKeyframeIndex;

    float maxInliersRatioWithFirst = -1;
    int closureFrameIndex = -1;
    Mat closureRt;
    size_t closedKeyframesCount = 0;
    bool isClosing = false;
    bool isClosureFrameAdded = false;

    stringstream checkDataMessage;
    checkDataMessage << "Please check the data! Sequential frames have to be close to each other (location and color). "
                        "The first and one of the last frames also have to be "
                        "really taken from close camera positions and the lighting has to be the same.";
    for(size_t i = firstKeyframeIndex + 1; i < frames.size(); i++)
    {
        if(frameToFramePoses[i].empty())
            continue;

        Ptr<OdometryFrameCache> lastKeyframe = *keyframes.rbegin();
        Ptr<OdometryFrameCache> currFrame = frames[i];

        bool isKeyframeAdded = false;

        const float minInliersRatio = 0.7f;
        const float skippedTranslation = 0.4f; // meters
        const float minTranslationDiff = 0.05f; // meters
        const float maxTranslationDiff = 0.30f; // meters
        const float minRotationDiff = 5.f; // degrees
        const float maxRotationDiff = 30.f;// degrees

        // check for the current frame: is it keyframe?
        {
            Mat Rt = (*keyframePoses.rbegin()).inv(DECOMP_SVD) * frameToFramePoses[i];

            float tnorm = tvecNorm(Rt);
            float rnorm = rvecNormDegrees(Rt);

            if(tnorm > maxTranslationDiff || rnorm > maxRotationDiff)
            {
                cout << "Camera trajectory is broken (between frames " << lastKeyframeIndex << " - " << i << ")." << endl;
                cout << checkDataMessage << endl;
                return false;
            }

            if((tnorm > minTranslationDiff || rnorm > minRotationDiff) &&
               computeInliersRatio(currFrame, lastKeyframe, Rt, cameraMatrix) > minInliersRatio)
            {
                keyframes.push_back(currFrame);
                keyframePoses.push_back(frameToFramePoses[i]);
                if(keyframeIndices) keyframeIndices->push_back(i);
                lastKeyframeIndex = i;
                isKeyframeAdded = true;
                translationSum += tnorm;
                if(isClosing)
                    cout << "possible ";
                cout << "keyframe index " << i << endl;
            }
        }

        // match with the first keyframe
        if(translationSum > skippedTranslation) // ready for closure
        {
            Mat Rt;
            if(odometry->compute(*currFrame, *firstFrame, Rt))
            {
                float inliersRatio = computeInliersRatio(currFrame, firstFrame, Rt, cameraMatrix);
                if(inliersRatio > minInliersRatio)
                {
                    if(inliersRatio > maxInliersRatioWithFirst)
                    {
                        isClosing = true;
                        maxInliersRatioWithFirst = inliersRatio;
                        closureFrameIndex = i;
                        closureRt = Rt;
                        isClosureFrameAdded = isKeyframeAdded;
                        closedKeyframesCount = keyframes.size();
                    }
                }
                else if(isClosing)
                    break;
            }
            else if(isClosing)
                    break;
        }
    }
    if(closureFrameIndex > 0)
    {
        keyframes.resize(closedKeyframesCount);
        keyframePoses.resize(closedKeyframesCount);
        if(keyframeIndices) keyframeIndices->resize(closedKeyframesCount);

        if(!isClosureFrameAdded)
        {
            keyframes.push_back(frames[closureFrameIndex]);
            keyframePoses.push_back(frameToFramePoses[closureFrameIndex]);
            if(keyframeIndices) keyframeIndices->push_back(closureFrameIndex);
        }

        keyframePoses.push_back((*keyframePoses.begin()) * closureRt);

        cout << "Last keyframe index " << closureFrameIndex << endl;
        cout << "keyframes count = " << keyframes.size() << endl;
    }
    else
    {
        cout << "The algorithm can not make loop closure on given data. " << endl;
        cout << checkDataMessage << endl;
    }

    return closureFrameIndex > 0;
}
