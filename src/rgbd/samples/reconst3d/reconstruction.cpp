#include <opencv2/rgbd/rgbd.hpp>

#include "reconst3d.hpp"
#include "graph_optimizations.hpp"

using namespace cv;
using namespace std;

static void
preparePosesLinksWithoutRt(const vector<PosesLink>& srcLinks, vector<PosesLink>& dstLinks)
{
    dstLinks.resize(srcLinks.size());
    for(size_t i = 0; i < dstLinks.size(); i++)
        dstLinks[i] = PosesLink(srcLinks[i].srcIndex, srcLinks[i].dstIndex);
}

static void
prepareFramesForModelRefinement(const Ptr<TrajectoryFrames>& trajectoryFrames, vector<Ptr<RgbdFrame> >& dstFrames)
{
    dstFrames.resize(trajectoryFrames->frames.size());
    for(size_t i = 0; i < dstFrames.size(); i++)
    {
        const Ptr<RgbdFrame> srcFrame = trajectoryFrames->frames[i];
        dstFrames[i] = new RgbdFrame(srcFrame->image.clone(), srcFrame->depth.clone(),
                                     trajectoryFrames->objectMasks[i].clone(), srcFrame->normals.clone(),
                                     srcFrame->ID);
    }
}

void ModelReconstructor::reconstruct(const Ptr<TrajectoryFrames>& trajectoryFrames, const Mat& cameraMatrix,
                                     Ptr<ObjectModel>& model) const
{
    cout << "Frame-to-frame odometry result" << endl;
    vector<Mat> bgrImages;
    for(size_t i = 0; i < trajectoryFrames->frames.size(); i++)
        bgrImages.push_back(trajectoryFrames->frames[i]->image);

    const float voxelSize = 0.005f;
    genModel(trajectoryFrames->frames, trajectoryFrames->poses, cameraMatrix)->show(voxelSize);

    vector<Mat> refinedPosesSE3;
    refineSE3Poses(trajectoryFrames->poses, trajectoryFrames->posesLinks, refinedPosesSE3);

    cout << "Result of the loop closure" << endl;
    genModel(trajectoryFrames->frames, refinedPosesSE3, cameraMatrix)->show(voxelSize);

    vector<Mat> refinedPosesSE3RgbdICP;
    const float pointsPart = 0.05f;

    // fill posesLinks with empty Rt because we want that they will be recomputed
    vector<PosesLink> posesLinks;
    preparePosesLinksWithoutRt(trajectoryFrames->posesLinks, posesLinks);
    refineSE3RgbdICPPoses(trajectoryFrames->frames, refinedPosesSE3, posesLinks, cameraMatrix, pointsPart, refinedPosesSE3RgbdICP);

    cout << "Result of RgbdICP for camera poses" << endl;
    genModel(trajectoryFrames->frames, refinedPosesSE3RgbdICP, cameraMatrix)->show(voxelSize);

    vector<Ptr<RgbdFrame> > objectFrames; // with mask for object points only,
                                          // they will modified while refining the object points
    prepareFramesForModelRefinement(trajectoryFrames, objectFrames);

    vector<Mat> refinedSE3ICPSE3ModelPoses;
    refineSE3RgbdICPModel(objectFrames, refinedPosesSE3RgbdICP, posesLinks, cameraMatrix, refinedSE3ICPSE3ModelPoses);

    model = genModel(objectFrames, refinedSE3ICPSE3ModelPoses, cameraMatrix);
    model->show();
}

Ptr<ObjectModel> ModelReconstructor::genModel(const std::vector<Ptr<RgbdFrame> >& frames, const std::vector<Mat>& poses, const Mat& cameraMatrix)
{
    CV_Assert(frames.size() == poses.size());

    Ptr<ObjectModel> model = new ObjectModel();

    cout << "Convert frames to model" << endl;
    for(size_t i = 0; i < frames.size(); i++)
    {
        CV_Assert(!frames[i]->image.empty());
        CV_Assert(frames[i]->depth.size() == frames[i]->image.size());
        CV_Assert(frames[i]->mask.size() == frames[i]->image.size());
        CV_Assert(frames[i]->normals.empty() || frames[i]->normals.size() == frames[i]->image.size());

        const int maskElemCount = countNonZero(frames[i]->mask);
        model->colors.reserve(model->colors.size() + maskElemCount);
        model->points3d.reserve(model->points3d.size() + maskElemCount);
        if(!frames[i]->normals.empty())
            model->normals.reserve(model->normals.size() + maskElemCount);

        Mat cloud;
        depthTo3d(frames[i]->depth, cameraMatrix, cloud);
        Mat transfPoints3d;
        perspectiveTransform(cloud.reshape(3,1), transfPoints3d, poses[i]);
        transfPoints3d = transfPoints3d.reshape(3, cloud.rows);

        Mat transfNormals;
        if(!frames[i]->normals.empty())
        {
            perspectiveTransform(frames[i]->normals.reshape(3,1), transfNormals, poses[i]);
            transfNormals = transfNormals.reshape(3, frames[i]->normals.rows);
        }

        for(int y = 0, pointIndex = 0; y < frames[i]->mask.rows; y++)
        {
            const uchar* maskRow = frames[i]->mask.ptr<uchar>(y);
            for(int x = 0; x < frames[i]->mask.cols; x++, pointIndex++)
            {
                if(maskRow[x] && isValidDepth(frames[i]->depth.at<float>(y,x)))
                {
                    model->colors.push_back(frames[i]->image.at<Vec3b>(y,x));
                    model->points3d.push_back(transfPoints3d.at<Point3f>(y,x));
                    if(!frames[i]->normals.empty())
                        model->normals.push_back(transfNormals.at<Point3f>(y,x));
                }
            }
        }
        CV_Assert(model->colors.size() == model->points3d.size());
        CV_Assert(model->normals.empty() || model->normals.size() == model->points3d.size());
    }

    return model;
}
