#include <iostream>

#include <opencv_candidate_reconst3d/reconst3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv_candidate/feature2d.hpp>

using namespace std;
using namespace cv;

static Mat defaultCameraMatrix()
{
    float vals[] = {525., 0., 3.1950000000000000e+02,
                    0., 525., 2.3950000000000000e+02,
                    0., 0., 1.};
    return Mat(3,3,CV_32FC1,vals).clone();
}

int main(int argc, char** argv)
{
    if(argc != 2  && argc != 3)
    {
        cout << "Format: " << argv[0] << " train_dirname [model_filename]" << endl;
        cout << "   train_dirname - a path to the directory with TOD-like training base." << endl;
        cout << "   model_filename - an optional parameter, it's a filename that will be used to save trained model." << endl;
        return -1;
    }

    // Load the data
    const string dirname = argv[1];

    vector<string> frameIndices;
    readFrameIndices(dirname, frameIndices);
    if(frameIndices.empty())
    {
        cout << "Can not load the data from given directory of the base: " << dirname << endl;
        return -1;
    }

    cout << "Frame indices count " << frameIndices.size() << endl;

    Mat cameraMatrix = defaultCameraMatrix();

    Ptr<ArbitraryCaptureServer> captureServer = new ArbitraryCaptureServer();
    captureServer->set("cameraMatrix", cameraMatrix);
    captureServer->initialize(Size(640,480));
    for(size_t srcIndex = 0; srcIndex < frameIndices.size(); srcIndex++)
    {
//        if(srcIndex % 4 != 0)
//            continue;

        Mat srcBgrImage, srcDepth;
        loadFrameData(dirname, frameIndices[srcIndex], srcBgrImage, srcDepth);

        captureServer->push(srcBgrImage, srcDepth, srcIndex);
    }

    Ptr<TrajectoryFrames> trajectoryFrames = captureServer->finalize();
    captureServer.release();

    if(trajectoryFrames.empty())
        return -1;

    ModelReconstructor reconstructor;
    reconstructor.set("isShowStepResults", true);
    reconstructor.set("maxBAPosesCount", 50);

    Ptr<ObjectModel> model;
    reconstructor.reconstruct(trajectoryFrames, cameraMatrix, model);

    if(argc == 3)
    {
        model->write_ply(argv[2]);
    }

    model->show();

    return 0;
}
