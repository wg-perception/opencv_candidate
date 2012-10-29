#ifndef MODEL_CAPTURE_HPP
#define MODEL_CAPTURE_HPP

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/rgbd/rgbd.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "g2o/core/sparse_optimizer.h"

class TableMasker: public cv::Algorithm
{
public:
    static double DEFAULT_Z_FILTER_MIN() {return 0.005;}
    static double DEFAULT_Z_FILTER_MAX() {return 0.5;}
    static double DEFAULT_MIN_TABLE_PART() {return 0.1;}

    TableMasker();
    bool operator()(const cv::Mat& cloud, const cv::Mat& normals,
                    cv::Mat& tableWithObjectMask, cv::Mat* tableMask=0) const;

    cv::AlgorithmInfo*
    info() const;

protected:
    mutable cv::Ptr<cv::RgbdPlane> planeComputer;

    double z_filter_min;
    double z_filter_max;
    double min_table_part;

    cv::Mat cameraMatrix;
};

struct KeyframesData
{
    std::vector<cv::Ptr<cv::OdometryFrameCache> > frames;
    std::vector<cv::Mat> tableMasks;
    std::vector<cv::Mat> poses; // +1 for the loop closure (if the trajectory is closed)
};

class OnlineCaptureServer: public cv::Algorithm
{
public:
    static const int DEFAULT_MAX_CORRESP_COLOR_DIFF = 50; // it's rough now, because first and last frame may have large changes of light conditions
                                                          // TODO: do something with light changes
    static double DEFAULT_MAX_CORRESP_DEPTH_DIFF() {return 0.01;} // meters
    static double DEFAULT_MIN_INLIERS_RATIO() {return 0.6;}
    static double DEFAULT_SKIPPED_TRANSLATION() {return 0.4;} //meters
    static double DEFAULT_MIN_TRANSLATION_DIFF() {return 0.08;} //meters
    static double DEFAULT_MAX_TRANSLATION_DIFF() {return 0.3;} //meters
    static double DEFAULT_MIN_ROTATION_DIFF() {return 10;} //degrees
    static double DEFAULT_MAX_ROTATION_DIFF() {return 30;} //degrees

    OnlineCaptureServer();

    cv::Mat push(const cv::Mat& image, const cv::Mat& depth, int frameID, bool *isKeyframe = 0);
    bool isLoopClosed() const;

    void initialize(const cv::Size& frameResolution);
    void reset();
    cv::Ptr<KeyframesData> finalize();

    cv::AlgorithmInfo*
    info() const;

protected:
    void filterImage(const cv::Mat& src, cv::Mat& dst) const;
    void firterDepth(const cv::Mat& src, cv::Mat& dst) const;

    // used algorithms
    cv::Ptr<cv::RgbdNormals> normalsComputer; // inner only
    cv::Ptr<TableMasker> tableMasker;
    cv::Ptr<cv::Odometry> odometry;

    // output keyframes data
    cv::Ptr<KeyframesData> keyframesData;

    // params
    cv::Mat cameraMatrix;

    int maxCorrespColorDiff;
    double maxCorrespDepthDiff;

    double minInliersRatio;
    double skippedTranslation;
    double minTranslationDiff;
    double maxTranslationDiff;
    double minRotationDiff;
    double maxRotationDiff;

    // state variables
    cv::Ptr<cv::OdometryFrameCache> firstKeyframe, lastKeyframe, prevFrame, closureFrame;
    cv::Mat prevPose;
    int prevFrameID;

    bool isClosing, isClosed;
    double translationSum;
    float closureInliersRatio;
    int closureFrameID;
    cv::Mat closureTableMask;
    bool isClosureFrameAdded;
    cv::Mat closurePose, closurePoseWithFirst;

    bool isInitialied, isFinalized;
};

static inline
float tvecNorm(const cv::Mat& Rt)
{
    return cv::norm(Rt(cv::Rect(3,0,1,3)));
}

static inline
float rvecNormDegrees(const cv::Mat& Rt)
{
    cv::Mat rvec;
    cv::Rodrigues(Rt(cv::Rect(0,0,3,3)), rvec);
    return cv::norm(rvec) * 180. / CV_PI;
}

/*
 * Loading the data
 */

// Read all filenames containing in the directory and sort them in lexicographic order
void readDirectory(const std::string& directoryName, std::vector<std::string>& filenames, bool addDirectoryName=false);

// Read frames from TOD-like base
void loadTODLikeBase(const std::string& dirname, std::vector<cv::Mat>& bgrImages,
                     std::vector<cv::Mat>& depthes32F, std::vector<std::string>* imageFilenames=0);


/*
 * Graph optimization
 */

inline
void set2shorts(int& dst, int short_v1, int short_v2)
{
    unsigned short* ptr = reinterpret_cast<unsigned short*>(&dst);
    ptr[0] = static_cast<unsigned short>(short_v1);
    ptr[1] = static_cast<unsigned short>(short_v2);
}

inline
void get2shorts(int src, int& short_v1, int& short_v2)
{
    typedef union { int vint32; unsigned short vuint16[2]; } s32tou16;
    const unsigned short* ptr = (reinterpret_cast<s32tou16*>(&src))->vuint16;
    short_v1 = ptr[0];
    short_v2 = ptr[1];
}

// TODO: add a check of normals
// this function is from *Odometry implementation
int computeCorresps(const cv::Mat& K, const cv::Mat& K_inv, const cv::Mat& Rt1to0,
                    const cv::Mat& depth0, const cv::Mat& validMask0,
                    const cv::Mat& depth1, const cv::Mat& selectMask1, float maxDepthDiff,
                    cv::Mat& corresps);

// Fill the given graph by vertices (if it's required) and edges for the camera pose refinement.
// Each vertex is a camera pose. Each edge constraint is the odometry between linked vertices.
// poses[0] .. poses[poses.size() - 2] are sequential frame-to-frame poses
// poses[poses.size() - 1] is a pose of the last frame found by Odometry with the first frame directly (for the loop closure)
void fillGraphSE3(g2o::SparseOptimizer* optimizer, const std::vector<cv::Mat>& poses, int startVertexIndex, bool addVertices=true);

// Restore refined camera poses from the graph.
// verticesRange is a range of camera poses that have to be restore: [verticesRange.start, verticesRange.end)
void getSE3Poses(g2o::SparseOptimizer* optimizer, const cv::Range& verticesRange, std::vector<cv::Mat>& poses);

// Refine camera poses by graph with odometry edges
// poses.size() == frames.size() + 1 (the last pose is for the loop closure)
// refinedPoses.size() == frames.size()
void refineSE3Poses(const std::vector<cv::Mat>& poses, std::vector<cv::Mat>& refinedPoses);



// Grap with 2 types of edges: frame-to-frame odometry and Rgbd+ICP for correspondences.
// poses.size() == frames.size()
void fillGraphRgbdICPSE3(g2o::SparseOptimizer* optimizer, const std::vector<cv::Ptr<cv::OdometryFrameCache> >& frames,
                     const std::vector<cv::Mat>& poses, const cv::Mat& cameraMatrix,
                     int startVertexIndex, bool addVertices=true);

// Refine camera poses by graph with odometry edges + icp edges
// poses.size() == frames.size()
// refinedPoses.size() == poses.size()
void refineRgbdICPSE3Poses(std::vector<cv::Ptr<cv::OdometryFrameCache> >& frames, const std::vector<cv::Mat>& poses, const cv::Mat& cameraMatrix, float pointsPart,
                       std::vector<cv::Mat>& refinedPoses);


void refineICPSE3Landmarks(std::vector<cv::Ptr<cv::OdometryFrameCache> >& frames, const std::vector<cv::Mat>& poses, const cv::Mat& cameraMatrix,
                           std::vector<cv::Mat>& refinedPoses);


/*
 * Show model
 */

// Show merged point clouds in PCL Visualizer window
void showModel(const std::vector<cv::Mat>& bgrImages,
               const std::vector<cv::Ptr<cv::OdometryFrameCache> >& frames, const std::vector<cv::Mat>& poses,
               const cv::Mat& cameraMatrix, float voxelFilterSize);

void showModelWithNormals(const std::vector<cv::Mat>& bgrImages,
               const std::vector<cv::Ptr<cv::OdometryFrameCache> >& frames, const std::vector<cv::Mat>& poses,
               const cv::Mat& cameraMatrix);


#endif // MODEL_CAPTURE_HPP
