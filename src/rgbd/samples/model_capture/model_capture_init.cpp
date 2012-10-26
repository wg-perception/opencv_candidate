#include "model_capture.hpp"
#include <opencv2/core/core.hpp>

// TODO: remove this fix when it'll became available in OpenCV

#define CV_INIT_ALGORITHM_FIX(classname, algname, memberinit) \
    static ::cv::Algorithm* create##classname() \
    { \
        return new classname; \
    } \
    \
    static ::cv::AlgorithmInfo& classname##_info() \
    { \
        static ::cv::AlgorithmInfo classname##_info_var(algname, create##classname); \
        return classname##_info_var; \
    } \
    \
    static ::cv::AlgorithmInfo& classname##_info_auto = classname##_info(); \
    \
    ::cv::AlgorithmInfo* classname::info() const \
    { \
        static volatile bool initialized = false; \
        \
        if( !initialized ) \
        { \
            initialized = true; \
            classname obj; \
            memberinit; \
        } \
        return &classname##_info(); \
    }

CV_INIT_ALGORITHM_FIX(TableMasker, "ModelCapture.TableMasker",
    obj.info()->addParam(obj, "planeComputer", obj.planeComputer);
    obj.info()->addParam(obj, "z_filter_min", obj.z_filter_min);
    obj.info()->addParam(obj, "z_filter_max", obj.z_filter_max);
    obj.info()->addParam(obj, "min_table_part", obj.min_table_part);
    obj.info()->addParam(obj, "cameraMatrix", obj.cameraMatrix);)


CV_INIT_ALGORITHM_FIX(OnlineCaptureServer, "ModelCapture.OnlineCaptureServer",
    obj.info()->addParam(obj, "tableMasker", obj.tableMasker);
    obj.info()->addParam(obj, "odometry", obj.odometry);
    obj.info()->addParam(obj, "cameraMatrix", obj.cameraMatrix);
    obj.info()->addParam(obj, "maxCorrespColorDiff", obj.maxCorrespColorDiff);
    obj.info()->addParam(obj, "maxCorrespDepthDiff", obj.maxCorrespDepthDiff);
    obj.info()->addParam(obj, "minInliersRatio", obj.minInliersRatio);
    obj.info()->addParam(obj, "skippedTranslation", obj.skippedTranslation);
    obj.info()->addParam(obj, "minTranslationDiff", obj.minTranslationDiff);
    obj.info()->addParam(obj, "maxTranslationDiff", obj.maxTranslationDiff);
    obj.info()->addParam(obj, "isInitialied", obj.isInitialied, true);
    obj.info()->addParam(obj, "isFinalized", obj.isFinalized, true);)
