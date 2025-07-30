//includes
#include <opencv2/core/types.hpp>
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <array>
#include <filesystem>

//libs
#include "../include/shared.hpp"
#include "../include/marco_annunziata.hpp"
#include "../include/hermann_serain.hpp"
#include "../include/sveva_turola.hpp"

using namespace std;
using namespace cv;
using namespace Shared;

int main(int argc, char** argv){
    // TODO: batch processing
    if(argc < 2){
        cerr << "Usage: <test image path>\n";
        return -1;
    }

    string labelsFilename = "../data/model/labels.txt";
    YOLO_model model;
    model.setModelName("YOLO11s");

    string image_path = argv[1];
    Mat frame = imread(image_path, cv::IMREAD_COLOR);

    vector<string> dataClasses = model.getDataClasses(labelsFilename);
    model.detectObjects(frame, dataClasses, true);
    model.drawBoundingBoxes(frame.rows, frame.cols, frame);
    // destroyAllWindows();


    return(0);
}
