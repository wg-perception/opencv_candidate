#include "model_capture/model_capture.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        cout << "Format: " << argv[0] << " dirname " << endl;
        cout << "   dirname - a path to the directory with TOD-like base." << endl;
        return -1;
    }

    // Load the data
    const string dirname = argv[1];
    vector<Mat> bgrImages, depthes;
    loadTODLikeBase(dirname, bgrImages, depthes);
    if(bgrImages.empty())
    {
        cout << "Can not load the data from given directory of the base: " << dirname << endl;
        return -1;
    }

    float vals[] = {525., 0., 3.1950000000000000e+02,
                    0., 525., 2.3950000000000000e+02,
                    0., 0., 1.};
    Mat cameraMatrix = Mat(3,3,CV_32FC1,vals).clone();

    OnlineCaptureServer onlineCaptureServer;
    onlineCaptureServer.set("cameraMatrix", cameraMatrix);

    CV_Assert(!bgrImages[0].empty());

    onlineCaptureServer.initialize(bgrImages[0].size());
    for(size_t i = 0; i < bgrImages.size(); i++)
    {
        Mat gray;
        cvtColor(bgrImages[i], gray, CV_BGR2GRAY);
        onlineCaptureServer.push(gray, depthes[i], i);
    }
    Ptr<KeyframesData> keyframesData = onlineCaptureServer.finalize();

#if 0
    for(size_t i = 0; i < frames.size(); i++)
    {
        if(find(indicesToBgrImages.begin(), indicesToBgrImages.end(), i) == indicesToBgrImages.end())
        {
            frames[i].release();
            bgrImages[i].release();
            depthes[i].release();
        }
    }
#endif

    cout << "Frame-to-frame odometry result" << endl;
    showModel(bgrImages, keyframesData->frames, keyframesData->poses, cameraMatrix, 0.005);

    vector<Mat> refinedPosesSE3;
    refineSE3Poses(keyframesData->poses, refinedPosesSE3);

    cout << "Result of the loop closure" << endl;
    showModel(bgrImages, keyframesData->frames, refinedPosesSE3, cameraMatrix, 0.003);

    vector<Mat> refinedPosesICPSE3;
    float pointsPart = 0.05f;
    refineRgbdICPSE3Poses(keyframesData->frames, refinedPosesSE3, cameraMatrix, pointsPart, refinedPosesICPSE3);

    cout << "Result of RgbdICP for camera poses" << endl;
    float modelVoxelSize = 0.003f;
    showModel(bgrImages, keyframesData->frames, refinedPosesICPSE3, cameraMatrix, modelVoxelSize);

#if 1
    // remove table from the further refinement
    for(size_t i = 0; i < keyframesData->frames.size(); i++)
    {
        keyframesData->frames[i]->mask &= ~keyframesData->tableMasks[i];
        keyframesData->frames[i]->releasePyramids();
    }
    pointsPart = 1.f;
    modelVoxelSize = 0.001;
#endif

    vector<Mat> refinedPosesICPSE3Landmarks;
    refineICPSE3Landmarks(keyframesData->frames, refinedPosesICPSE3, cameraMatrix, refinedPosesICPSE3Landmarks);

    cout << "Result of RgbdICP for camera poses and moving the model points" << endl;
    modelVoxelSize = 0.000001;
    showModel(bgrImages, keyframesData->frames, refinedPosesICPSE3Landmarks, cameraMatrix, modelVoxelSize);
//    showModelWithNormals(bgrImages, keyframesData->frame, refinedPosesICPSE3Landmarks, cameraMatrix);

    return 0;
}
