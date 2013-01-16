#include <opencv_candidate/feature2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

static
Mat_<float> affineSkew(const Mat& image, const Mat& mask, float tilt, float phi,
                       Mat& transformedImage, Mat& transformedMask)
{
    transformedImage = image.clone();
    transformedMask = mask.clone();

    Mat_<float> A(2,3);
    A << 1.f, 0.f, 0.f,
         0.f, 1.f, 0.f;

    CV_Assert(tilt > 1.f);

    // rotate image
    if(phi != 0.f)
    {
        const float sinPhi = std::sin(phi / 180.f * CV_PI);
        const float cosPhi = std::cos(phi / 180.f * CV_PI);

        Mat_<float> R(2,2);
        R << cosPhi, -sinPhi,
             sinPhi,  cosPhi;

        Mat_<float> rectPoints(4,2);
        rectPoints <<        0.f,        0.f,
                             0.f, image.rows,
                      image.cols, image.rows,
                      image.cols,        0.f;
        Mat transformedRectPoints = R * rectPoints.t();
        transformedRectPoints = transformedRectPoints.t();
        transformedRectPoints = transformedRectPoints.reshape(2,1);

        Rect r = boundingRect(transformedRectPoints);

        A << cosPhi, -sinPhi, -r.x,
             sinPhi,  cosPhi, -r.y;

        Mat dst;
        warpAffine(transformedImage, dst, A, r.size(), INTER_LINEAR, BORDER_REPLICATE);
        transformedImage = dst;
    }

    // blur image
    const float sigmaX = 0.8f * sqrt(tilt*tilt - 1.f);
    const float sigmaY = 0.01f;
    Mat bluredImage;
    GaussianBlur(transformedImage, bluredImage, Size(0,0), sigmaX, sigmaY);

    // tilt image
    resize(bluredImage, transformedImage, Size(0,0), 1.f/tilt, 1.f, INTER_NEAREST);
    A.row(0) /= tilt;

    // transform mask if it's need
    if(phi != 0.f || !transformedMask.empty())
    {
        if(transformedMask.empty())
            transformedMask = Mat(image.size(), CV_8UC1, Scalar::all(255));

        Mat dst;
        warpAffine(transformedMask, dst, A, transformedImage.size(), INTER_NEAREST);
        transformedMask = dst;
    }

    Mat Ainv;
    invertAffineTransform(A, Ainv);

    return Ainv;
}

namespace feature2d
{

AffineAdaptedFeature2D::AffineAdaptedFeature2D(const Ptr<Feature2D>& feature2d) : feature2d(feature2d)
{}

AffineAdaptedFeature2D::AffineAdaptedFeature2D(const Ptr<FeatureDetector>& featureDetector,
                       const Ptr<DescriptorExtractor>& descriptorExtractor) :
    featureDetector(featureDetector),
    descriptorExtractor(descriptorExtractor)
{}

int AffineAdaptedFeature2D::descriptorSize() const
{
    if(feature2d)
        return feature2d->descriptorSize();

    CV_Assert(descriptorExtractor);
    return descriptorExtractor->descriptorSize();
}

int AffineAdaptedFeature2D::descriptorType() const
{
    if(feature2d)
        return feature2d->descriptorType();

    CV_Assert(descriptorExtractor);
    return descriptorExtractor->descriptorType();
}

void AffineAdaptedFeature2D::detectAndComputeImpl(const Mat& image, const Mat& mask, vector<KeyPoint>& keypoints, Mat& descriptors) const
{
    if(feature2d)
        (*feature2d)(image, mask, keypoints, descriptors);
    else
    {
        CV_Assert(featureDetector);
        CV_Assert(descriptorExtractor);

        featureDetector->detect(image, keypoints, mask);
        descriptorExtractor->compute(image, keypoints, descriptors);
    }
}

void AffineAdaptedFeature2D::operator()(InputArray _image, InputArray _mask,
                                        vector<KeyPoint>& keypoints,
                                        OutputArray _descriptors,
                                        bool useProvidedKeypoints) const
{
    Mat image = _image.getMat();
    Mat mask = _mask.getMat();

    CV_Assert(useProvidedKeypoints == false);

    const float a = sqrt(2);
    const float b = 72.f;

    float tilt = 1.f;
    const int tiltN = 5;

    keypoints.clear();
    Mat descriptors;

    for(int deg = 0; deg <= tiltN; deg++)
    {
        if(tilt == 1.f)
        {
            vector<KeyPoint> kpts;
            Mat descs;
            detectAndComputeImpl(image, mask, kpts, descs);
            keypoints.insert(keypoints.end(), kpts.begin(), kpts.end());
            descriptors.push_back(descs);
        }
        else
        {
            float phi = 0.f;
            const int phiN = static_cast<int>(std::floor(180.f * tilt / b));
            for(int step = 0; step <= phiN; step++)
            {
                Mat transformedImage, transformedMask;
                Mat Ainv = affineSkew(image, mask, tilt, phi, transformedImage, transformedMask);

                vector<KeyPoint> kpts;
                Mat descs;
                detectAndComputeImpl(transformedImage, transformedMask, kpts, descs);

                size_t prevSize = keypoints.size();
                keypoints.resize(prevSize + kpts.size());
                CV_Assert(Ainv.type() == CV_32FC1);
                const float* Ainv_ptr = Ainv.ptr<const float>();
                for(size_t srcIndex = 0, dstIndex = prevSize; srcIndex < kpts.size(); srcIndex++, dstIndex++)
                {
                    KeyPoint kp = kpts[srcIndex];
                    float tx = Ainv_ptr[0] * kp.pt.x + Ainv_ptr[1] * kp.pt.y + Ainv_ptr[2];
                    float ty = Ainv_ptr[3] * kp.pt.x + Ainv_ptr[4] * kp.pt.y + Ainv_ptr[5];
                    kp.pt.x = tx;
                    kp.pt.y = ty;
                    keypoints[dstIndex] = kp;
                }
                descriptors.push_back(descs);

                phi += b / tilt;
            }
        }
        tilt *= a;
    }

    _descriptors.create(descriptors.size(), descriptors.type());
    Mat _descriptorsMat = _descriptors.getMat();
    descriptors.copyTo(_descriptorsMat);
}

void AffineAdaptedFeature2D::computeImpl(const Mat& /*image*/, vector<KeyPoint>& /*keypoints*/, Mat& /*descriptors*/) const
{
    CV_Error(CV_StsNotImplemented, "Not implemented method because it's not efficient to split feature detection and description extraction here\n");
}

void AffineAdaptedFeature2D::detectImpl(const Mat& /*image*/, vector<KeyPoint>& /*keypoints*/, const Mat& /*mask*/) const
{
    CV_Error(CV_StsNotImplemented, "Not implemented method because it's not efficient to split feature detection and description extraction here\n");
}

}
