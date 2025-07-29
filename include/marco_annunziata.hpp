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
    int classConfidence = 0;
    cv::Rect boundingBox = cv::Rect(0,0,0,0);
};

class YOLO_model{
    private:
        Ort::Env env;
        static void logger(void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location, const char* message);
        Ort::Session session;
        Ort::SessionOptions sessionOptions;
        Ort::AllocatorWithDefaultOptions allocator;
        const int YOLO_TARGET_INPUT_SIZE = 640; // MUST be multiple of 32. See YOLO_model::detectObjects implementation.
        std::vector<Detection> detections;
    public:
        YOLO_model();
        void detectObjects(cv::Mat &img, int inputSize);
};

#endif
