#ifndef MARCO_ANNUNZIATA_HPP
#define MARCO_ANNUNZIATA_HPP

#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include "shared.hpp"
#include <iostream>
#include <streambuf>
#include <vector>
#include <fstream>

struct Detection{
    cv::Rect boundingBox = cv::Rect(0,0,0,0);
    float classConfidence = 0;
    int classId = -1;
};

struct Detections{
    std::vector<cv::Rect> boundingBoxes;
    std::vector<float> classConfidences;
    std::vector<int> classIds;
};

class YOLO_model{
    private:
        Ort::Env                         env;
        static void                      logger(void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location, const char* message);
        Ort::Session                     session;
        Ort::SessionOptions              sessionOptions;
        Ort::AllocatorWithDefaultOptions allocator;
        bool                             usingGPU = false;
        const int                        YOLO_TARGET_INPUT_SIZE = 640; // MUST be multiple of 32. See YOLO_model::detectObjects implementation.
        std::vector<Detection>           detections;
        std::string                      modelName = "Yolo";
        std::vector<std::string>         classNames;
        const float CLASS_CONFIDENCE_THRESHOLD = 0.5;
        const float NMS_THRESHOLD = 0.5;
    public:
        YOLO_model();
        bool                     isAvailableGPU();
        std::vector<Detection>   detectionPipeline(cv::Mat &img, bool enable_letterbox_padding=true);
        void                     clearDetections();
        Detections               detect(const cv::Mat &img, bool enable_letterbox_padding=true);
        void                     mergeDetections(Detections& dest, const Detections& source);
        std::vector<std::string> getDataClasses(std::string labelsFilename="../data/model/labels.txt");
        std::vector<Detection>   filterDetectionsNMS(Detections goodDetections);
        cv::Mat                  drawBoundingBoxes(cv::Mat &img, std::vector<Detection> &detections, cv::Scalar color);
        cv::Mat                  drawBoundingBoxes(cv::Mat &img, cv::Scalar color);
        void                     setModelName(std::string modelName);
        std::string              getModelName();
        std::vector<Detection>   getDetections();
        std::vector<std::string> getDetectionsClassNames(Detections detections);
        std::string              getClassName(int classId);
};

#endif
