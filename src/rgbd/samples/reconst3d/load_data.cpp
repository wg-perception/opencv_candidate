#include <dirent.h>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "reconst3d.hpp"

using namespace std;
using namespace cv;

static void readDirectory(const string& directoryName, vector<string>& filenames, bool addDirectoryName)
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

    vector<string> allFilenames;
    readDirectory(dirname, allFilenames, false);

    vector<string> imageIndices;
    imageIndices.reserve(allFilenames.size());
    for(size_t i = 0; i < allFilenames.size(); i++)
    {
        const string& imageFilename = allFilenames[i];
        // image_* and .png is at least 11 character
        if (imageFilename.size() < 11)
          continue;

        const string imageIndex = imageFilename.substr(6, imageFilename.length() - 6 - 4);

        if(imageFilename.substr(0, 6) == "image_" &&
           imageIndex.find_first_not_of("0123456789") == std::string::npos &&
           imageFilename.substr(imageFilename.length() - 4, 4) == ".png")
        {
            imageIndices.push_back(imageIndex);
        }
    }

    bgrImages.resize(imageIndices.size());
    depthes32F.resize(imageIndices.size());
    if(imageFilenames)
        imageFilenames->resize(imageIndices.size());

#pragma omp parallel for
    for(size_t i = 0; i < imageIndices.size(); i++)
    {
        string imageFilename = "image_" + imageIndices[i] + ".png";
        cout << "Load " << imageFilename << endl;

        if(imageFilenames)
            (*imageFilenames)[i] = imageFilename;

        // read image
        {
            string imagePath = dirname + "/" + imageFilename;
            Mat image = imread(imagePath);
            CV_Assert(!image.empty());
            bgrImages[i] = image;
        }

        // read depth
        {
            Mat depth;
            string depthPath = "depth_image_" + imageIndices[i] + ".xml.gz";
            FileStorage fs(dirname + "/" + depthPath, FileStorage::READ);
            if(fs.isOpened())
            {
                fs["depth_image"] >> depth;
            }
            else
            {
                depthPath = "depth_" + imageIndices[i] + ".png";
                depth = imread(dirname + "/" + depthPath, -1);
                CV_Assert(!depth.empty());
                Mat depth_flt;
                depth.convertTo(depth_flt, CV_32FC1, 0.001);
                depth_flt.setTo(std::numeric_limits<float>::quiet_NaN(), depth == 0);
                depth = depth_flt;
            }
#if 0
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
            depthes32F[i] = depth;
        }
    }
}
