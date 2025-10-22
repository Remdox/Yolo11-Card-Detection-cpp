#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <filesystem>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

#include "./../include/sveva_turola.hpp"
#include "./../include/marco_annunziata.hpp"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int frameCapture(string data_path, string labels_path) {
    string outputDir = "../output/frames/";     //output directory
    int frameInterval = 10;

    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
    }
    VideoCapture cap;
    if(data_path == "0"){
        cap.open(0);
    } else {
        cap.open(data_path);
    }

    if (!cap.isOpened()) {
        cerr << "Error opening video" << endl;
        return -1;
    }

    YOLO_model model;
    model.setModelName("YOLO11s");
    Mat frame;
    vector<string> dataClasses = model.getDataClasses(labels_path);
    int frameCount = 0;
    int savedCount = 0;

    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
    VideoWriter out;

    cap.read(frame);
    if (frame.empty()){
        return -1;
    }
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    out.open("../output/detections.mp4", codec, frameInterval, Size(frame_width, frame_height), true);
    if(!out.isOpened()){
        cerr << "Could not open the output video file for write\n";
        return -1;
    }

    cout << "Loading video..." << endl;
    while (true) {
        cap.read(frame);
        if (frame.empty()){
            break;
        }

        if (frameCount % frameInterval == 0) {
            // string filename = outputDir + "frame_" + to_string(frameCount) + ".jpg";
            // imwrite(filename, frame);
            savedCount++;
            // cout << "Saved: " << filename << endl;

            auto detections = model.detectObjects(frame, dataClasses, true);
            Mat outputFrame = cardValues(detections, model, frame);
            out.write(outputFrame);
            imshow("Output Video", outputFrame);
            waitKey(1);
        }
        frameCount++;
    }

    cout << "\nExtraction completed! Frames saved: " << savedCount << endl;
    cout << "Video saved in output directory" << endl;

    cap.release();
    out.release();
    destroyAllWindows();
    return 0;
}

Mat cardValues(vector<Detection> detections, YOLO_model &model, Mat &frame){
    vector<Detection> green, blue, red;

    for(auto detection : detections){
        char cardNumber = detection.className[0];

        if(cardNumber >= '2' && cardNumber <= '6'){
            green.push_back(detection);
        }
        else if(cardNumber >= '7' && cardNumber <= '9'){
            blue.push_back(detection);
        }
        else { // tutto il resto, inclusi 10,J,Q,K,A,JOKER
            red.push_back(detection);
        }
    }

    Mat outputFrame = frame.clone();

    if(!green.empty()){
        outputFrame = model.drawBoundingBoxes(frame.rows, frame.cols, outputFrame, green, Scalar(0, 255, 0));
    }
    if(!blue.empty()){
        outputFrame = model.drawBoundingBoxes(frame.rows, frame.cols, outputFrame, blue, Scalar(255, 0, 0));
    }
    if(!red.empty()){
        outputFrame = model.drawBoundingBoxes(frame.rows, frame.cols, outputFrame, red, Scalar(0, 0, 255));
    }

    return outputFrame;
}