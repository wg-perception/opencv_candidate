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

    // Create the odometry object
    Ptr<Odometry> odometry = Algorithm::create<Odometry>("RGBD.RgbdOdometry");
    if(odometry.empty())
    {
        cout << "Can not create Odometry algorithm. Check the passed odometry name." << endl;
        return -1;
    }
    float vals[] = {525., 0., 3.1950000000000000e+02,
                    0., 525., 2.3950000000000000e+02,
                    0., 0., 1.};
    Mat cameraMatrix = Mat(3,3,CV_32FC1,vals).clone();
    odometry->set("cameraMatrix", cameraMatrix);

    // Create normals computer
    Ptr<RgbdNormals> normalsComputer = new cv::RgbdNormals(depthes[0].rows, depthes[0].cols, depthes[0].depth(), cameraMatrix);

    // Fill vector of initial frames
    vector<Ptr<OdometryFrameCache> > frames;
    vector<Mat> tableMasks, tableWithObjectMasks;
    for(size_t i = 0; i < bgrImages.size(); i++)
    {
        Ptr<OdometryFrameCache> frame = new OdometryFrameCache();

        Mat gray;
        cvtColor(bgrImages[i], gray, CV_BGR2GRAY);
        {
            Mat tmp;
            medianBlur(gray, tmp, 3);
            gray = tmp;
        }

        Mat cloud;
        depthTo3d(depthes[i], cameraMatrix, cloud);

        Mat normals = (*normalsComputer)(cloud);

        Mat tableWithObjectMask, tableMask;
        cout << "Masking the frame " << i << endl;
        if(!computeTableWithObjectMask(cloud, normals, cameraMatrix, tableWithObjectMask, 0.1, &tableMask))
        {
            cout << "Skip the frame because calcTableWithObjectMask was failed" << endl;
        }
        else
        {
            frame->image = gray;
            frame->depth = depthes[i];
            CV_Assert(!tableWithObjectMask.empty());
            frame->mask = tableWithObjectMask;
            frame->normals = normals;
        }

        frames.push_back(frame);
        tableMasks.push_back(tableMask.clone());
        tableWithObjectMasks.push_back(tableWithObjectMask.clone());
    }

    vector<Ptr<OdometryFrameCache> > keyframes;
    vector<Mat> keyframePoses;
    vector<int> indicesToBgrImages; // to frames vector
    if(!frameToFrameProcess(frames, cameraMatrix, odometry, keyframes, keyframePoses, &indicesToBgrImages))
        return -1;

    cout << "Frame-to-frame odometry result" << endl;
    showModel(bgrImages, indicesToBgrImages, keyframes, keyframePoses, cameraMatrix, 0.005);

    vector<Mat> refinedPosesSE3;
    refineSE3Poses(keyframePoses, refinedPosesSE3);

    cout << "Result of the loop closure" << endl;
    showModel(bgrImages, indicesToBgrImages, keyframes, refinedPosesSE3, cameraMatrix, 0.003);

    vector<Mat> refinedPosesICPSE3;
    float pointsPart = 0.05f;
    refineRgbdICPSE3Poses(keyframes, refinedPosesSE3, cameraMatrix, pointsPart, refinedPosesICPSE3);

    cout << "Result of RgbdICP for camera poses" << endl;
    float modelVoxelSize = 0.003f;
    showModel(bgrImages, indicesToBgrImages, keyframes, refinedPosesICPSE3, cameraMatrix, modelVoxelSize);

#if 1
    // remove table from the further refinement
    for(size_t i = 0; i < keyframes.size(); i++)
    {
        keyframes[i]->mask = tableWithObjectMasks[indicesToBgrImages[i]] & ~tableMasks[indicesToBgrImages[i]];
        keyframes[i]->pyramidMask.clear();
        keyframes[i]->pyramidTexturedMask.clear();
        keyframes[i]->pyramidNormalsMask.clear();
        keyframes[i]->pyramidMask.clear();
    }
    pointsPart = 1.f;
    modelVoxelSize = 0.001;
#endif

    vector<Mat> refinedPosesICPSE3Landmarks;
    refineICPSE3Landmarks(keyframes, refinedPosesICPSE3, cameraMatrix, refinedPosesICPSE3Landmarks);

    cout << "Result of RgbdICP for camera poses and moving the model points" << endl;
    modelVoxelSize = 0.000001;
    showModel(bgrImages, indicesToBgrImages, keyframes, refinedPosesICPSE3Landmarks, cameraMatrix, modelVoxelSize);
//    showModelWithNormals(bgrImages, indicesToBgrImages, keyframes, refinedPosesICPSE3Landmarks, cameraMatrix);

    return 0;
}
