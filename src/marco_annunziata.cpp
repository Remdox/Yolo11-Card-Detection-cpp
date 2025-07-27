
#include "marco_annunziata.hpp"
#include "shared.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp> 
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::dnn;

const float CLASS_CONFIDENCE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.1;

// TODO: setupNet with ONNX Runtime for dynamic input
YOLO_model::YOLO_model(){
    model = readNet("../data/model/YOLO11s_small_last_static.onnx");

    cout << "Running con CPU\n";
    model.setPreferableBackend(DNN_BACKEND_OPENCV);
    model.setPreferableTarget(DNN_TARGET_CPU);
}

// YOLO's grid needs a square image as input
cv::Mat formatYoloInput(const Mat &img){
    int imgSize = max(img.rows, img.cols);
    Mat squareImg = Mat::zeros(imgSize, imgSize, CV_8UC3);
    img.copyTo(squareImg(Rect(0, 0, img.cols, img.rows)));
    // namedWindow("square");
    // imshow("square", squareImg);
    // waitKey(0);
    return squareImg;
}


void YOLO_model::detectObjects(Mat &img, int inputSize){
    // retrieve classes for later
    std::vector<std::string> dataClasses;
    std::ifstream ifs("../data/model/labels.txt");
    std::string line;
    while (getline(ifs, line)) dataClasses.push_back(line);

    // format input correctly and pass it to the OpenCV DNN
    Mat inputImg = formatYoloInput(img);
    Mat blob;

    double normalizationFactor = 1.0 / 255.0;
    blobFromImage(inputImg, blob, normalizationFactor, Size(640, 640), Scalar(), true, false);
    model.setInput(blob);


    // In the OpenCV API, forward returns a vector of 3D matrices [batch_size, dimensions, rows], where each matrix is an output layer
    // In the case of YOLO11, only 1 output layer is returned
    std::vector<cv::Mat> outputs;
    model.forward(outputs, model.getUnconnectedOutLayersNames());

    float scalingFactor[2] = {
        static_cast<float>(inputImg.cols) / inputSize, 
        static_cast<float>(inputImg.rows) / inputSize
    };

    // 3D Matrices are computationally inefficient at accessing elements
    // tensor data is saved as float32, so a pointer is faster than the 3D Mat to iterate through the elements
    float *data = (float *)outputs[0].data;

    // data[0..4...] = (x, y, width, height, confidence, classProb1, classProb2, ...)

    // YOLO11 is pre-trained by default on the COCO dataset, so the number of channels sould be:
    //                      84 = 4 + 80 (= class confidence, one for each class)
    // BUT, we fine-tuned the training of YOLO11 with our custom dataset, so the output layer is restructured to represent 53 classes.
    // This means the number of channels should be 57 = 4 (bounding box parameters) + 53 (= class confidence, one for each class)
    //  we instead used a pre-trained YOLO11s model, so the output layer is still the 84-channels one used for the default COCO dataset!
    // Note: class probability       = probability of having detected one class in that region
    //       class confidence        = confidence of having detected an object of a class in that region, measured as class probability * objectness probability
    //       objectness probability  = probability of having an object in that region, useful for YOLOv5 but not for YOLO11, since there is already class confidence, so it's been removed.

    // Note, Yolov5 has a different representation of the output layer compared to Yolov8/Yolo11:
    // Yolov5 -> channels = 85 = 4 (bb_parameters) + 1 (objectness probability) + 80 (= class probabilities of corresponding 80 classes of the COCO dataset)
    // So, in this case the objectness is separate from class confidence and each class has the class probability instead of the class confidence.
    // To get class confidence of each class you have to multiply each class probability*objectness. So in YOLOv5 the objectness probability was useful!

    // To recap, our custom YOLO11 output tensor (3D Mat) has shape (1, 57, 8400):
    // 1 or 0 (1 for default model)      - is the batch size. If =0, it can change dynamically depending on the input of the net.
    // 57 (84 for default model)         - is the number of channels, where the first 4 are the bounding box parameters (x, y, width, height)
    //                                      and the other ones are class probabilities for COCO'S 80 classes (There is no objectiveness parameter in Yolov8/11)
    // 8400 or 0 (8400 default model)    - is the total number of regions where the predictions are made (regions = grid points = anchor locations).
    //                              If it is =0, it can change dynamically.
    // The first and the third value become 0 if the model has been exported into ONNX format with dynamic=True
    // Dynamic input/output shapes are not supported by OpenCV, but they are supported by ONNX Runtime.
    // See: https://github.com/orgs/ultralytics/discussions/17254
    const int channels = outputs[0].size[1];
    const int rows = 8400;
    cout << "num. of output layer channels: " << channels << endl;

    vector<int> classIds;
    vector<float> detectedClassConfidences;
    vector<Rect> boundingBoxes;

    for(int i = 0; i < rows; i++){
        if(i == 0){
            for(int j = 0; j < channels; j++){
                cout << "data " << j << ": " << data[j] << endl;
            }
        }
        // The Mat constructor doesn't copy the data, it just wraps pointer classes_scores in a matrix header,
        // so this is used instead of the typical Mat constructor as it is more efficient.
        // Also, minMaxLoc accepts as input a Mat and as output a Point, so this is how scores and classId will be defined.
        float * firstClassScore = data + 4;
        Mat classScores(1, channels-4, CV_32FC1, firstClassScore);
        Point classId;
        double maxClassScore;
        minMaxLoc(classScores, 0, &maxClassScore, 0, &classId);

        if(maxClassScore > CLASS_CONFIDENCE_THRESHOLD){
            detectedClassConfidences.push_back(maxClassScore);
            classIds.push_back(classId.x);

            // computing the bounding box of the detected object
            Rect box = Rect(
                int((data[0] - 0.5f*data[2])*scalingFactor[0]), // x position
                int((data[1] - 0.5f*data[3])*scalingFactor[1]), // y position
                int(data[2]*scalingFactor[0]), // box width
                int(data[3]*scalingFactor[1])  // box height
            );
            boundingBoxes.push_back(box);
        }

        data += channels;
    }
    
    // Apply Non-Maxima Suppression
    std::vector<int> resultNMS;
    cv::dnn::NMSBoxes(boundingBoxes, detectedClassConfidences, CLASS_CONFIDENCE_THRESHOLD, NMS_THRESHOLD, resultNMS);
    Mat resultImg = img.clone();

    // Draw the NMS filtered boxes
    for(auto finalBoxIndex : resultNMS){
        Detection finalDetection;
        finalDetection.classId = classIds[finalBoxIndex];
        finalDetection.className = dataClasses[finalDetection.classId];
        finalDetection.classConfidence = detectedClassConfidences[finalBoxIndex];
        finalDetection.boundingBox = boundingBoxes[finalBoxIndex];
        detections.push_back(finalDetection);
        
        rectangle(resultImg, finalDetection.boundingBox, Scalar(0, 255, 0), 2);
        putText(resultImg, dataClasses[finalDetection.classId] + " " + std::to_string(finalDetection.classConfidence), Point(finalDetection.boundingBox.x, finalDetection.boundingBox.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        cout << "Class confidence score:" << finalDetection.classConfidence << endl;
    }
    namedWindow("Prediction with Box", WINDOW_NORMAL);
    imshow("Prediction with Box", resultImg);
    waitKey(0);
    destroyAllWindows();
}