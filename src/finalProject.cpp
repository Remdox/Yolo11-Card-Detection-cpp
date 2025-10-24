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
    string choice = "X";
    string data_path;
    string labels_path = "../data/model/labels.txt";;
    string allowedImgType[] = {".png", ".jpg", ".jpeg"};
    string allowedVidType[] = {".mp4"};
    YOLO_model model;
    model.setModelName("YOLO11s");

    while(true){
        cout << "Write C to use camera or F to use a file: ";
        cin >> choice;
        if(choice == "C"){
            cout << "Selected camera (press q to close the camera)\n";
            cout << "Insert number of fps for output video or write default: ";
            string inputFps;
            cin >> inputFps;
            int result = frameCapture("0", labels_path, inputFps);
            break;
        } else if(choice == "F"){
            string data_path;
            cout << "Insert file path: ";
            cin >> data_path;
            cout << "Insert number of fps for output video or write default: ";
            string inputFps;
            cin >> inputFps;
            int frames = frameCapture(data_path, labels_path, inputFps);
            break;
        } else {
            cout << "Usage: write C or F!\n";
        }
    }

    // TODO: batch processing
    /* TODO
    THIS SHOULD BE ADJUSTED TO BECOME A PROPER FUNCTION FOR PARSING THE COMMAND AND/OR FILE GIVEN
    AS INPUT AND ROUTING THE CORRESPONDING FUNCTION TO ANALYZE IMAGE, VIDEO, WEBCAM */
    enum class fileCategories {IMAGE, VIDEO, UNKNOWN};
    fileCategories fileType = fileCategories::UNKNOWN;
    if(labels_path.rfind(".txt") == string::npos){
        cerr << "The label file provided has an invalid file type. Please provide a .txt file.\n";
        return -1;
    }

    /* check if user chooses CAMERA (or default to this if image/video not provided)
    if(CAMERA){
        fileType = fileCategories::CAMERA;
        //runCameraPipeline()
    }
    else{
        [[check file provided]] */

    /* check if file is image */
    if(fileType == fileCategories::UNKNOWN){
        for(auto type : allowedImgType){
            if(data_path.rfind(type) != string::npos){
                fileType = fileCategories::IMAGE;
                //runImagePipeline()
                Mat frame = imread(data_path, cv::IMREAD_COLOR);

                vector<string> dataClasses = model.getDataClasses(labels_path);
                model.detectObjects(frame, dataClasses, true);
                Mat resultImg = model.drawBoundingBoxes(frame.rows, frame.cols, frame, Scalar(255, 0, 0));
                std::string windowTitle = model.getModelName() + " - " + std::to_string(model.getDetections().size()) + " detections";
                namedWindow(windowTitle, WINDOW_NORMAL);
                imshow(windowTitle, resultImg);
                waitKey(0);
                break;
            }
        }
    }

    /* check if file is video */
    if(fileType == fileCategories::UNKNOWN){
        for(auto type: allowedVidType){
            if(data_path.rfind(type) != string::npos){
                fileType = fileCategories::VIDEO;
                //runVideoPipeline()
                //int frames = frameCapture(data_path, labels_path);
                break;
            }
        }
    }
    
    /* invalid file */
    if(fileType == fileCategories::UNKNOWN){
            cerr << "The image or video provided has an invalid file type. Please retry with a different one.\n";
            return -1;
        }

    // destroyAllWindows();
    return 0;
}
