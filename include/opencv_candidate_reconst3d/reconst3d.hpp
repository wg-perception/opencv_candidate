#ifndef RECONST3D_HPP
#define RECONST3D_HPP

#include <opencv2/rgbd/rgbd.hpp>

// Read frames from TOD-like base
void loadTODLikeBase(const std::string& dirname, std::vector<cv::Mat>& bgrImages,
                     std::vector<cv::Mat>& depthes32F, std::vector<std::string>* imageFilenames=0);

// Find a table mask
class TableMasker: public cv::Algorithm
{
public:
    static double DEFAULT_Z_FILTER_MIN() {return 0.005;}
    static double DEFAULT_Z_FILTER_MAX() {return 0.5;}
    static double DEFAULT_MIN_TABLE_PART() {return 0.1;}

    TableMasker();
    bool operator()(const cv::Mat& cloud, const cv::Mat& normals,
                    cv::Mat& tableWithObjectMask, cv::Mat* objectMask=0, cv::Vec4f* planeCoeffs=0) const;

    cv::AlgorithmInfo*
    info() const;

protected:
    mutable cv::Ptr<cv::RgbdPlane> planeComputer;

    double zFilterMin;
    double zFilterMax;
    double minTablePart;

    cv::Mat cameraMatrix;
};

struct PosesLink
{
    PosesLink(int srcIndex=-1, int dstIndex=-1, const cv::Mat& Rt=cv::Mat());
    int srcIndex;
    int dstIndex;
    cv::Mat Rt; // optional (for loop closure)
};

struct TrajectoryFrames
{
    enum { VALIDFRAME = 1,
           KEYFRAME   = VALIDFRAME | 2,
           DEFAULT    = VALIDFRAME
         };

    void push(const cv::Ptr<cv::RgbdFrame>& frame, const cv::Mat& pose,
              const cv::Mat& objectMask, int state);
    void clear();

    int resumeFrameState;
    std::vector<cv::Ptr<cv::RgbdFrame> > frames;
    std::vector<int> frameStates;
    std::vector<cv::Mat> objectMasks;
    std::vector<cv::Mat> poses;
    std::vector<PosesLink> keyframePosesLinks;
};

class OnlineCaptureServer: public cv::Algorithm
{
public:
    struct FramePushOutput
    {
        FramePushOutput();

        int frameState;
        cv::Ptr<cv::OdometryFrame> frame;
        cv::Mat pose;
        cv::Mat objectMask;
    };

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

    cv::Ptr<FramePushOutput> push(const cv::Mat& image, const cv::Mat& depth, int frameID);

    void initialize(const cv::Size& frameResolution, int storeFramesWithState=TrajectoryFrames::KEYFRAME);

    void reset();

    cv::Ptr<TrajectoryFrames> finalize();

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
    cv::Ptr<TrajectoryFrames> trajectoryFrames;

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
    cv::Ptr<cv::OdometryFrame> firstKeyframe, lastKeyframe, prevFrame, closureFrame;
    cv::Mat prevPose;
    int prevFrameID;

    bool isTrajectoryBroken;
    bool isLoopClosing, isLoopClosed;
    double translationSum;
    float closureInliersRatio;
    int closureFrameID;
    bool isClosureFrameKey;
    cv::Mat closureBgrImage;
    cv::Mat closureObjectMask;
    cv::Mat closurePose, closurePoseWithFirst;

    bool isInitialied, isFinalized;
};

class ObjectModel
{
public:
    ObjectModel();
    void clear();

    void read_ply(const std::string& filename);
    void write_ply(const std::string& filename) const;

    void show(float gridSize=0.001f, int normalLevel=0) const;

    std::vector<cv::Vec3b> colors;
    std::vector<cv::Point3f> points3d;
    std::vector<cv::Point3f> normals;
};

class ModelReconstructor : public cv::Algorithm
{
public:

    ModelReconstructor();

    void reconstruct(const cv::Ptr<TrajectoryFrames>& trajectoryFrames, const cv::Mat& cameraMatrix, cv::Ptr<ObjectModel>& model) const;

    static cv::Ptr<ObjectModel> genModel(const std::vector<cv::Ptr<cv::RgbdFrame> >& frames, const std::vector<cv::Mat>& poses,
                                         const cv::Mat& cameraMatrix, const std::vector<int>& frameIndices=std::vector<int>());
    
    cv::AlgorithmInfo*
    info() const;

private:
    // TODO make more algorithm params available outside

    bool isShowStepResults;
};

#endif // RECONST3D_HPP
