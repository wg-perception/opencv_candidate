#include "model_capture.hpp"
#include "ocv_pcl_eigen_convert.hpp"

#include "pcl/point_types.h"
#include <pcl/point_cloud.h>
#include "pcl/ModelCoefficients.h"
#include <pcl/surface/convex_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>

//#include <boost/thread/thread.hpp>
//#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/rgbd/rgbd.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

bool computeTableWithObjectMask(const Mat& cloud, const Mat& normals, const Mat& cameraMatrix,
                                Mat& tableWithObjectMask, float _minTableArea, Mat* _tableMask)
{
    const float minTableArea = _minTableArea * cloud.total();
    const float kinect_error_a = 0.0075f;
    const double table_z_filter_min = 0.005;
    const double table_z_filter_max = 0.5;

    CV_Assert(!cloud.empty() && cloud.type() == CV_32FC3);
    CV_Assert(!normals.empty() && normals.type() == CV_32FC3);
    CV_Assert(cloud.size() == normals.size());

    // Find all planes in the frame.
    Ptr<RgbdPlane> planeComputer = cv::Algorithm::create<cv::RgbdPlane>("RGBD.RgbdPlane");
    planeComputer->set("sensor_error_a", kinect_error_a);

    Mat_<uchar> planesMask;
    vector<Vec4f> planesCoeffs;
    (*planeComputer)(cloud, normals, planesMask, planesCoeffs);

    // Find table plane:
    // filter planes with small count of points,
    // the table plane is a plane with the smallest average distance among the remaining planes.
    int tableIndex = -1;
    float tableAvgDist = FLT_MAX;
    for(size_t i = 0; i < planesCoeffs.size(); i++)
    {
        Mat curMask = planesMask == i;
        int pointsNumber = countNonZero(curMask);
        if(pointsNumber < minTableArea)
            continue;

        float curDist = 0.f;
        for(int y = 0; y < curMask.rows; y++)
        {
            const uchar* curMaskRow = curMask.ptr<uchar>(y);
            const Point3f* cloudRow = cloud.ptr<Point3f>(y);
            for(int x = 0; x < curMask.cols; x++)
            {
                if(curMaskRow[x])
                    curDist += cloudRow[x].z;
            }
        }
        curDist /= pointsNumber;

        if(tableAvgDist > curDist)
        {
            tableIndex = i;
            tableAvgDist = curDist;
        }
    }

    if(tableIndex < 0)
        return false;

    // Find a mask of the object. For this find convex hull for the table and
    // get the points that are in the prism corresponding to the hull lying above the table.

    // Convert the data to pcl types
    pcl::PointCloud<pcl::PointXYZ> pclTableCloud;
    Mat tableMask = planesMask == tableIndex;
    cvtCloud_cv2pcl(cloud, tableMask, pclTableCloud);

    pcl::ModelCoefficients pclTableCoeffiteints;
    pclTableCoeffiteints.values.resize(4);
    Vec4f tableCoeffitients = planesCoeffs[tableIndex];
    pclTableCoeffiteints.values[0] = tableCoeffitients[0];
    pclTableCoeffiteints.values[1] = tableCoeffitients[1];
    pclTableCoeffiteints.values[2] = tableCoeffitients[2];
    pclTableCoeffiteints.values[3] = tableCoeffitients[3];

    // Find convex hull
    pcl::ConvexHull<pcl::PointXYZ> pclHullRreconstruntor;
    pcl::PointCloud<pcl::PointXYZ> pclTableHull;
    pclHullRreconstruntor.setInputCloud(boost::make_shared<const pcl::PointCloud<pcl::PointXYZ> >(pclTableCloud));
    pclHullRreconstruntor.setDimension(2);
    pclHullRreconstruntor.reconstruct(pclTableHull);

    // Get indices of points in the prism
    pcl::PointIndices pclPrismPointsIndices;
    pcl::ExtractPolygonalPrismData<pcl::PointXYZ> pclPrismSegmentator;
    pclPrismSegmentator.setHeightLimits(table_z_filter_min, table_z_filter_max);
    pcl::PointCloud<pcl::PointXYZ> pclCloud;
    cvtCloud_cv2pcl(cloud, Mat(), pclCloud);
    pclPrismSegmentator.setInputCloud(boost::make_shared<const pcl::PointCloud<pcl::PointXYZ> >(pclCloud));
    pclPrismSegmentator.setInputPlanarHull(boost::make_shared<const pcl::PointCloud<pcl::PointXYZ> >(pclTableHull));
    pclPrismSegmentator.segment(pclPrismPointsIndices);

    // Get points from the prism
    pcl::PointCloud<pcl::PointXYZ> pclPrismPoints;
    pcl::ExtractIndices<pcl::PointXYZ> extract_object_indices;
    extract_object_indices.setInputCloud(boost::make_shared<const pcl::PointCloud<pcl::PointXYZ> >(pclCloud));
    extract_object_indices.setIndices(boost::make_shared<const pcl::PointIndices>(pclPrismPointsIndices));
    extract_object_indices.filter(pclPrismPoints);

    // Draw the object points to the mask
    vector<Point3f> objectCloud;
    cvtCloud_pcl2cv(pclPrismPoints, objectCloud);
    tableWithObjectMask = tableMask;
    if(!objectCloud.empty())
    {
        vector<Point2f> objectPoints2d;
        projectPoints(objectCloud, Mat::zeros(3,1,CV_32FC1), Mat::zeros(3,1,CV_32FC1), cameraMatrix, Mat(), objectPoints2d);
        Mat_<uchar> objectMask = tableWithObjectMask;
        Rect r(0, 0, cloud.cols, cloud.rows);
        for(size_t i = 0; i < objectPoints2d.size(); i++)
        {
            if(r.contains(objectPoints2d[i]))
                objectMask(objectPoints2d[i]) = 255;
        }
    }

    if(_tableMask)
        (*_tableMask) = (planesMask == tableIndex);

    return true;
}
