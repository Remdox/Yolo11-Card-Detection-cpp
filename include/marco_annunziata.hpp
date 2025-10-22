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
#include <vector>
#include <fstream>

struct Detection{
    int classId = -1;
    std::string className = "undefined";
    float classConfidence = 0;
    cv::Rect boundingBox = cv::Rect(0,0,0,0);
};

class YOLO_model{
    private:
        Ort::Env                         env;
        static void                      logger(void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location, const char* message);
        Ort::Session                     session;
        Ort::SessionOptions              sessionOptions;
        Ort::AllocatorWithDefaultOptions allocator;
        const int                        YOLO_TARGET_INPUT_SIZE = 640; // MUST be multiple of 32. See YOLO_model::detectObjects implementation.
        std::vector<Detection>           detections;
        std::string                      modelName = "Yolo";
    public:
        YOLO_model();
        std::vector<Detection>   detectObjects(cv::Mat &img, std::vector<std::string> dataClasses, bool enable_letterbox_padding=true);
        std::vector<std::string> getDataClasses(std::string labelsFilename="../data/model/labels.txt");
        cv::Mat                  drawBoundingBoxes(int inputWidth, int inputHeight, cv::Mat &img, std::vector<Detection> &detections, cv::Scalar color);
        cv::Mat                  drawBoundingBoxes(int inputWidth, int inputHeight, cv::Mat &resultImg, cv::Scalar color);
        void                     setModelName(std::string modelName);
        std::string              getModelName();
        std::vector<Detection>   getDetections();
};

#endif
