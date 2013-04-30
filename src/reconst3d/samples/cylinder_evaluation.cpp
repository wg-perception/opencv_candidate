#include <iostream>

#include <opencv_candidate_reconst3d/reconst3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

static Mat defaultCameraMatrix()
{
    double vals[] = {525., 0., 3.1950000000000000e+02,
                    0., 525., 2.3950000000000000e+02,
                    0., 0., 1.};
    return Mat(3,3,CV_64FC1,vals).clone();
}

static
Mat calcTableRotation(const Mat& tablePlane)
{
    Mat tableNormal = tablePlane.rowRange(0,3).clone();
    tableNormal *= 1./norm(tableNormal);

    Mat rvec = tableNormal.cross(Mat(Vec3f(0,0,1)));
    rvec *= 1./norm(rvec);

    rvec *= acos(tableNormal.dot(Mat(Vec3f(0,0,1))));

    Mat R;
    Rodrigues(rvec, R);

    return R;
}

static
void prepareCylinderModel(const vector<Point3f>& points,
                          const Vec4f& tablePlane, float minZ, float maxZ,
                          vector<Point3f>& cylinderPoints)
{
    // Rotate the table plane to make it parallel to the plane z=0
    Mat R = calcTableRotation(Mat(tablePlane));
    CV_Assert(R.type() == CV_32FC1);

    Mat Rt = Mat::eye(4,4,CV_32FC1);
    R.copyTo(Rt(Rect(0,0,3,3)));

    vector<Point3f> rotatedPoints;
    perspectiveTransform(points, rotatedPoints, Rt);

    // Translate the table plane to make it z=0
    // take a point of the table before the rotation
    Point3f tablePoint;
    if(std::abs(tablePlane[2]) > std::numeric_limits<float>::epsilon())
        tablePoint = Point3d(0, 0, -tablePlane[3]/tablePlane[2]);
    else if(std::abs(tablePlane[1]) > std::numeric_limits<float>::epsilon())
        tablePoint = Point3d(0, -tablePlane[3]/tablePlane[1], 0);
    else if(std::abs(tablePlane[0]) > std::numeric_limits<float>::epsilon())
        tablePoint = Point3d(-tablePlane[3]/tablePlane[0], 0, 0);
    else
        CV_Assert(0);
    // the point Z coordinate after the rotation
    float zShift = R.row(2).dot(Mat(tablePoint));

    // select only points within (minZ, maxZ)
    cylinderPoints.clear();
    for(size_t i = 0; i < rotatedPoints.size(); i++)
    {
        float shiftedZ = rotatedPoints[i].z - zShift;
        if(shiftedZ > minZ && shiftedZ < maxZ)
            cylinderPoints.push_back(Point3f(rotatedPoints[i].y, rotatedPoints[i].y, shiftedZ));
    }
}

static void
calcCorresps(const vector<Point3f>& src, vector<Point3f>& dst, float Rad)
{
    dst.resize(src.size());

    for(size_t i = 0; i < src.size(); i++)
    {
        const Point3f& srcP = src[i];
        if(abs(srcP.x) < std::numeric_limits<float>::epsilon())
        {
            float d0 = (srcP.y - Rad) * (srcP.y - Rad);
            float d1 = (srcP.y + Rad) * (srcP.y + Rad);

            if(d0 < d1)
                dst[i] = Point3f(0.f, Rad, srcP.z);
            else
                dst[i] = Point3f(0.f, -Rad, srcP.z);
        }
        else
        {
            const float r = srcP.y / srcP.x;
            const float s = 1.f / sqrt(1.f + r*r);
            const float x = Rad * s;
            const float y = r * s * Rad;


            float d0 = (srcP.x - x) * (srcP.x - x) + (srcP.y - y) * (srcP.y - y);
            float d1 = (srcP.x + x) * (srcP.x + x) + (srcP.y + y) * (srcP.y + y);

            if(d0 < d1)
                dst[i] = Point3f(x, y, srcP.z);
            else
                dst[i] = Point3f(-x, -y, srcP.z);
        }
    }
}

static
void alignCylinderModelWithGroundTruth(vector<Point3f>& points, float Rad)
{
    vector<Point3f> correspPoints;
    calcCorresps(points, correspPoints, Rad);

    const int itersCount = 10;

    for(int iter = 0; iter < itersCount; iter++)
    {
        // compute points centers
        Mat srcPoints3d(points),
            dstPoints3d(correspPoints);

        srcPoints3d.convertTo(srcPoints3d, CV_64FC3);
        dstPoints3d.convertTo(dstPoints3d, CV_64FC3);
        srcPoints3d = srcPoints3d.reshape(1,srcPoints3d.rows);
        dstPoints3d = dstPoints3d.reshape(1,dstPoints3d.rows);

        Mat meanSrcPoint, meanDstPoint;
        reduce(srcPoints3d, meanSrcPoint, 0, CV_REDUCE_AVG);
        reduce(dstPoints3d, meanDstPoint, 0, CV_REDUCE_AVG);

        // Comupte H
        Mat H = Mat::zeros(3,3,CV_64FC1);
        for(size_t i = 0; i < points.size(); i++)
            H += (srcPoints3d.row(i) - meanSrcPoint).t() * (dstPoints3d.row(i) - meanDstPoint);

        SVD svd(H);
        Mat v = svd.vt.t();
        Mat R = v * svd.u.t();
        if(determinant(R) < 0.)
        {
            v.col(2) = -1 * v.col(2);
            R = v * svd.u.t();
        }
        Mat t = meanDstPoint.t() - R * meanSrcPoint.t();

        Mat Rt = Mat::eye(4,4,CV_64FC1);
        R.copyTo(Rt(Rect(0,0,3,3)));
        t.copyTo(Rt(Rect(3,0,1,3)));

        vector<Point3f> transformedPoints;
        transform(points, transformedPoints, Rt);

        transformedPoints.swap(points);
    }
}

int main(int argc, char** argv)
{
    if(argc != 6)
    {
        cout << "Format: " << argv[0] << " model_ply table_coeffs_xml minZ maxZ Rad" << endl;
        return -1;
    }

    // Load the data
    const string model_filename = argv[1];

    ObjectModel model;
    model.read_ply(model_filename);

    Mat tablePlane;
    FileStorage fs(argv[2], FileStorage::READ);
    CV_Assert(fs.isOpened());
    fs["tablePlane"] >> tablePlane;
    CV_Assert(!tablePlane.empty());

    const float minZ = atof(argv[3]);
    const float maxZ = atof(argv[4]);
    const float Rad = atof(argv[5]);

    vector<Point3f> cylinderModel;
    prepareCylinderModel(model.points3d, tablePlane, minZ, maxZ, cylinderModel);

    alignCylinderModelWithGroundTruth(cylinderModel, Rad);

    vector<Point3f> correspPoints;
    calcCorresps(cylinderModel, correspPoints, Rad);

    vector<float> dists;
    dists.resize(cylinderModel.size());
    for(size_t i = 0; i < cylinderModel.size(); i++)
        dists[i] = cv::norm(cylinderModel[i] - correspPoints[i]);

    int halfIndex = dists.size()/2;
    nth_element(dists.begin(), dists.begin() + halfIndex, dists.end());

    cout << "median dist" << dists[halfIndex] << endl;

    double minVal, maxVal;
    minMaxLoc(Mat(dists), &minVal, &maxVal);

    cout << "min dist" << minVal << endl;
    cout << "max dist" << maxVal << endl;

    return 0;
}
