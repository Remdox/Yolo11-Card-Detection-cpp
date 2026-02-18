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
namespace fs = std::filesystem;

int main(int argc, char** argv){
    string choice = "X";
    string data_path;
    string labels_path = "../data/model/labels.txt";;
    enum class fileCategories {IMAGE, VIDEO, UNKNOWN};
    auto fileType = fileCategories::UNKNOWN;
    string allowedImgType[] = {".png", ".jpg", ".jpeg"};
    string allowedVidType[] = {".mp4"};
    YOLO_model model;
    model.setModelName("YOLO11s");
    UserData userData = readInput(argc, argv);

    switch (userData.choice)
    {
        case Choice::Camera:
        {
            frameCapture("0", labels_path);
        }
        break;

        case Choice::File:
        {
            /* check if file is image */
            for(auto type : allowedImgType){
                if(userData.data_path.rfind(type) != string::npos){
                    fileType = fileCategories::IMAGE;
                    Mat frame = imread(userData.data_path, cv::IMREAD_COLOR);

                    vector<string> dataClasses = model.getDataClasses(labels_path);
                    cout << "GPU available? " << (model.isAvailableGPU() ? "YES" : "NO") << endl;
                    model.detectionPipeline(frame);
                    auto detections = model.getDetections();
                    Mat resultImg = cardValues(detections, model, frame);
                    string outputDir = "../output/";
                    if (!fs::exists(outputDir)) {
                        fs::create_directory(outputDir);
                    }
                    fs::path p(userData.data_path);
                    string filename = outputDir + p.stem().string() + "_detections.jpg";
                    imwrite(filename, resultImg);
                    computeImageMetrics(userData.data_path, detections);
                    std::string windowTitle = model.getModelName() + " - " + std::to_string(detections.size()) + " detections";
                    namedWindow(windowTitle, WINDOW_NORMAL);
                    imshow(windowTitle, resultImg);
                    waitKey(0);
                    break;
                }
            }

            /* check if file is video */
            if(fileType == fileCategories::UNKNOWN){
                for(auto type: allowedVidType){
                    if(userData.data_path.rfind(type) != string::npos){
                        fileType = fileCategories::VIDEO;
                        cout << "Using ONNX Runtime GPU? " << (model.isAvailableGPU() ? "YES" : "NO") << endl;
                        frameCapture(userData.data_path, labels_path);
                        break;
                    }
                }
            }

            /* invalid file */
            if(fileType == fileCategories::UNKNOWN){
                    cerr << "The image or video provided has an invalid file type. Please retry with a different one.\n";
                    return -1;
            }


        }
        break;

        case Choice::Invalid:
        case Choice::Help:
        {
            //TODO: Verifica se questa parte va bene, di norma in questo caso non può ricaderci l'utente
            throw std::logic_error("Invalid operation: ...");
        }
    }

    // TODO: batch processing
    if(labels_path.rfind(".txt") == string::npos){
        cerr << "The label file provided has an invalid file type. Please provide a .txt file.\n";
        return -1;
    }

    // destroyAllWindows();
    return 0;
}
