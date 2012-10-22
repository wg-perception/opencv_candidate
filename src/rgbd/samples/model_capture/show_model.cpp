#include "model_capture.hpp"
#include "ocv_pcl_eigen_convert.hpp"

#include <boost/thread/thread.hpp>
#include "pcl/point_types.h"
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

template<class T>
void addToCloud(pcl::PointCloud<pcl::PointXYZRGB>& dstCloud, const T& srcPoint, const cv::Scalar& color)
{
    pcl::PointXYZRGB dstPoint;
    dstPoint.x = srcPoint.x;
    dstPoint.y = srcPoint.y;
    dstPoint.z = srcPoint.z;
    dstPoint.r = color[2];
    dstPoint.g = color[1];
    dstPoint.b = color[0];
    dstCloud.points.push_back( dstPoint );
}

static
void addToCloud(pcl::PointCloud<pcl::PointXYZRGB>& dst, const pcl::PointCloud<pcl::PointXYZ>& src, const Mat& mask,
                const Mat& bgrImage)
{
    CV_Assert(static_cast<int>(src.width) == bgrImage.cols);
    CV_Assert(static_cast<int>(src.height) == bgrImage.rows);

    for(int y = 0; y < static_cast<int>(src.height); y++)
    {
        for(int x = 0; x < static_cast<int>(src.width); x++)
        {
            const pcl::PointXYZ& p = src(x,y);
            if((mask.empty() || mask.at<uchar>(y,x)) && isValidDepth(p.z))
            {
                Vec3b c = bgrImage.at<cv::Vec3b>(y,x);
                addToCloud(dst, p , Scalar(c[0], c[1], c[2]));
            }
        }
    }
}

static
void voxelFilter(pcl::PointCloud<pcl::PointXYZRGB>& cloud, double grid_size)
{
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_ptr = boost::make_shared<const pcl::PointCloud<pcl::PointXYZRGB> >(cloud);
    pcl::PointCloud<pcl::PointXYZRGB> cloud_downsampled;

    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid_filter;
    voxel_grid_filter.setFilterFieldName("z");
    voxel_grid_filter.setLeafSize(grid_size, grid_size, grid_size);
    voxel_grid_filter.setDownsampleAllData(true);
    voxel_grid_filter.setInputCloud( cloud_ptr );
    voxel_grid_filter.filter( cloud_downsampled );

    cloud_downsampled.swap(cloud);
}

void showModel(const vector<Mat>& bgrImages, const vector<int>& indicesInBgrImages,
               const vector<Ptr<OdometryFrameCache> >& frames, const vector<Mat>& poses,
               const Mat& cameraMatrix, float voxelFilterSize)
{
    pcl::PointCloud<pcl::PointXYZRGB> globalCloud;

    CV_Assert(indicesInBgrImages.empty() || indicesInBgrImages.size() == frames.size());
    CV_Assert(frames.size() == poses.size() || (frames.size() + 1) == poses.size());

    for(size_t i = 0; i < frames.size(); i++)
    {
        int bgrIdx = indicesInBgrImages.empty() ? i : indicesInBgrImages[i];
        Mat cloud;
        if(!frames[i]->pyramidCloud.empty())
            cloud = frames[i]->pyramidCloud[0];
        else
            depthTo3d(frames[i]->depth, cameraMatrix, cloud);

        vector<Point3f> transfPointCloud;
        cv::perspectiveTransform(cloud.reshape(3,1), transfPointCloud, poses[i]);

        pcl::PointCloud<pcl::PointXYZ> addCloud;
        cvtCloud_cv2pcl(Mat(transfPointCloud), Mat(), addCloud);

        CV_Assert(addCloud.size() == frames[i]->image.total());
        addCloud.width = frames[i]->image.cols;
        addCloud.height = frames[i]->image.rows;
        addToCloud(globalCloud, addCloud, frames[i]->mask, bgrImages[bgrIdx]);

        voxelFilter(globalCloud, voxelFilterSize);
        cout << "Show: cloud size " << globalCloud.size() << endl;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr drawCloudPtr = boost::make_shared<const pcl::PointCloud<pcl::PointXYZRGB> >(globalCloud);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer =
            boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(drawCloudPtr);
    viewer->addPointCloud<pcl::PointXYZRGB>(drawCloudPtr, rgb, "result");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 0, "result");
    viewer->initCameraParameters ();
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
}


void showModelWithNormals(const vector<Mat>& bgrImages, const vector<int>& indicesInBgrImages,
               const vector<Ptr<OdometryFrameCache> >& frames, const vector<Mat>& poses,
               const Mat& cameraMatrix)
{
    pcl::PointCloud<pcl::PointXYZRGB> globalCloud;
    pcl::PointCloud<pcl::Normal> globalNormals;

    CV_Assert(indicesInBgrImages.empty() || indicesInBgrImages.size() == frames.size());
    CV_Assert(frames.size() == poses.size() || (frames.size() + 1) == poses.size());

    for(size_t i = 0; i < frames.size(); i++)
    {
        int bgrIdx = indicesInBgrImages.empty() ? i : indicesInBgrImages[i];
        Mat cloud;
        if(!frames[i]->pyramidCloud.empty())
            cloud = frames[i]->pyramidCloud[0];
        else
            depthTo3d(frames[i]->depth, cameraMatrix, cloud);

        vector<Point3f> transfPointCloud;
        cv::perspectiveTransform(cloud.reshape(3,1), transfPointCloud, poses[i]);

        vector<Point3f> transfNormals;
        cv::perspectiveTransform(frames[i]->pyramidNormals[0].reshape(3,1), transfNormals, poses[i]);

        Mat nrmls(transfNormals);
        nrmls = nrmls.reshape(3, frames[i]->image.rows);
        Mat mask = frames[i]->mask;
        CV_Assert(mask.size() == nrmls.size());
        for(int y = 0; y < mask.rows; y++)
        {
            for(int x = 0; x < mask.cols; x++)
            {
                const Point3f& p = nrmls.at<Point3f>(y,x);
                CV_Assert(frames[i]->mask.type() == CV_8UC1);
                if(mask.at<uchar>(y,x) && isValidDepth(frames[i]->depth.at<float>(y,x)))
                {
                    pcl::Normal normal;
                    normal.normal_x = p.x;
                    normal.normal_y = p.y;
                    normal.normal_z = p.z;
                    globalNormals.push_back(normal);
                }
            }
        }

        pcl::PointCloud<pcl::PointXYZ> addCloud;
        cvtCloud_cv2pcl(Mat(transfPointCloud), Mat(), addCloud);

        CV_Assert(addCloud.size() == frames[i]->image.total());
        addCloud.width = frames[i]->image.cols;
        addCloud.height = frames[i]->image.rows;
        addToCloud(globalCloud, addCloud, frames[i]->mask, bgrImages[bgrIdx]);

        cout << "Global cloud size " << globalCloud.size() << endl;
        cout << "Global normal size " << globalNormals.size() << endl;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr drawCloudPtr = boost::make_shared<const pcl::PointCloud<pcl::PointXYZRGB> >(globalCloud);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer =
            boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(drawCloudPtr);
    viewer->addPointCloud<pcl::PointXYZRGB>(drawCloudPtr, rgb, "result");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 0, "result"); //
    pcl::PointCloud<pcl::Normal>::ConstPtr normalsPtr  = boost::make_shared<const pcl::PointCloud<pcl::Normal> >(globalNormals);
    viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(drawCloudPtr, normalsPtr, 20, 0.003, "normals");
    viewer->initCameraParameters ();

    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
}
