#include "model_capture.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dirent.h>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace cv;

void readDirectory(const string& directoryName, vector<string>& filenames, bool addDirectoryName)
{
    filenames.clear();

    DIR* dir = opendir(directoryName.c_str());
    if(dir != NULL)
    {
        struct dirent* dent;
        while((dent = readdir(dir)) != NULL)
        {
            if(addDirectoryName)
                filenames.push_back(directoryName + "/" + string(dent->d_name));
            else
                filenames.push_back(string(dent->d_name));
        }
    }
    sort(filenames.begin(), filenames.end());
}

void loadTODLikeBase(const string& dirname, vector<Mat>& bgrImages, vector<Mat>& depthes32F, vector<string>* imageFilenames)
{
    CV_Assert(!dirname.empty());

    bgrImages.clear();
    depthes32F.clear();
    if(imageFilenames)
        imageFilenames->clear();

    vector<string> allFilenames;
    readDirectory(dirname, allFilenames, false);

    for(size_t i = 0; i < allFilenames.size(); i++)
    {
        const string& imageFilename = allFilenames[i];
        if(imageFilename.size() != 15)
            continue;

        const string imageIndex = imageFilename.substr(6, 5);

        if(imageFilename.substr(0, 6) == "image_" &&
           imageIndex.find_first_not_of("0123456789") == std::string::npos &&
           imageFilename.substr(imageFilename.length() - 4, 4) == ".png")
        {
            cout << "Load " << imageFilename << endl;

            if(imageFilenames)
                imageFilenames->push_back(imageFilename);

            // read image
            {
                string imagePath = dirname + imageFilename;
                Mat image = imread(imagePath);
                CV_Assert(!image.empty());
                bgrImages.push_back(image);
            }

            // read depth
            {
                const string depthPath = "depth_image_" + imageIndex + ".xml.gz";
                Mat depth;
                FileStorage fs(dirname + depthPath, FileStorage::READ);
                CV_Assert(fs.isOpened());
#if 1
                fs["depth_image"] >> depth;
#else
                cout << "Bilateral iltering" << endl;
                fs["depth_image"] >> depth;

                const double depth_sigma = 0.003;
                const double space_sigma = 3.5;  // in pixels
                Mat invalidDepthMask = (depth != depth) | (depth == 0.);
                depth.setTo(-5*depth_sigma, invalidDepthMask);
                Mat filteredDepth;
                bilateralFilter(depth, filteredDepth, -1, depth_sigma, space_sigma);
                filteredDepth.setTo(std::numeric_limits<float>::quiet_NaN(), invalidDepthMask);
                depth = filteredDepth;
#endif
                CV_Assert(!depth.empty());
                CV_Assert(depth.type() == CV_32FC1);
                depthes32F.push_back(depth);
            }
        }
    }
}
