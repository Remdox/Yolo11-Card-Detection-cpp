
#include "marco_annunziata.hpp"
#include "shared.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp> 
#include <onnxruntime_cxx_api.h>
#include <algorithm>

using namespace std;
using namespace cv;

const float CLASS_CONFIDENCE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.1;

YOLO_model::YOLO_model(): env(ORT_LOGGING_LEVEL_WARNING, "YOLOModel"), session(nullptr), sessionOptions(){
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    const char* modelPath = "../data/model/last.onnx";
    session = Ort::Session(env, modelPath, sessionOptions);
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

    // OPTION 1: Use original image size -> slow for big images
    int inputHeight = img.rows;
    int inputWidth = img.cols;
    
    // Ensure dimensions are multiples of 32, as required by YOLO11
    inputHeight = ((inputHeight + 31) / 32) * 32;
    inputWidth = ((inputWidth + 31) / 32) * 32;
    
    Mat inputImg;
    resize(img, inputImg, Size(inputWidth, inputHeight));
    
    // No scaling factors needed since we're using original proportions
    float scaleX = static_cast<float>(img.cols) / inputWidth;
    float scaleY = static_cast<float>(img.rows) / inputHeight;

    /*TODO Adaptive sizing based on image aspect ratio
    float aspectRatio = static_cast<float>(img.cols) / img.rows;
    int inputSize;
    
    if (aspectRatio > 1.0f) {
        // Landscape: fix width, adjust height
        inputSize = 640;
        inputWidth = inputSize;
        inputHeight = static_cast<int>(inputSize / aspectRatio);
    } else {
        // Portrait: fix height, adjust width  
        inputSize = 640;
        inputHeight = inputSize;
        inputWidth = static_cast<int>(inputSize * aspectRatio);
    }
    
    // Round to nearest multiple of 32
    inputWidth = ((inputWidth + 31) / 32) * 32;
    inputHeight = ((inputHeight + 31) / 32) * 32;
    
    resize(img, inputImg, Size(inputWidth, inputHeight));
    float scaleX = static_cast<float>(img.cols) / inputWidth;
    float scaleY = static_cast<float>(img.rows) / inputHeight;*/

    
    cout << "Dynamic input size: " << inputWidth << "x" << inputHeight << endl;

    // Create dynamic input tensor
    std::vector<int64_t> inputShape = {1, 3, inputHeight, inputWidth};
    size_t inputTensorSize = 1 * 3 * inputHeight * inputWidth;
    std::vector<float> inputData(inputTensorSize);

    // Convert Mat to tensor
    for (int y = 0; y < inputHeight; ++y) {
        for (int x = 0; x < inputWidth; ++x) {
            Vec3b pixel = inputImg.at<Vec3b>(y, x);
            inputData[0 * inputHeight * inputWidth + y * inputWidth + x] = pixel[2] / 255.0f; // R
            inputData[1 * inputHeight * inputWidth + y * inputWidth + x] = pixel[1] / 255.0f; // G  
            inputData[2 * inputHeight * inputWidth + y * inputWidth + x] = pixel[0] / 255.0f; // B
        }
    }

    // Create ONNX tensor with dynamic shape
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputData.data(), inputTensorSize, inputShape.data(), inputShape.size()
    );

    // Run inference
    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto outputName = session.GetOutputNameAllocated(0, allocator);
    
    std::vector<const char*> inputNames = {inputName.get()};
    std::vector<const char*> outputNames = {outputName.get()};
    
    auto outputTensors = session.Run(Ort::RunOptions{nullptr},
                                   inputNames.data(), &inputTensor, 1,
                                   outputNames.data(), 1);

    // Get dynamic output tensor info
    auto tensorInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    auto shape = tensorInfo.GetShape();
    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    
    int batchSize = static_cast<int>(shape[0]);
    int numChannels = static_cast<int>(shape[1]); 
    int numDetections = static_cast<int>(shape[2]);  // This will vary based on input size!
    
    cout << "Dynamic output shape: [" << batchSize << ", " << numChannels << ", " << numDetections << "]" << endl;



    /* format input correctly and pass it to the OpenCV DNN
    Mat inputImg = formatYoloInput(img);
    Mat blob;

    double normalizationFactor = 1.0 / 255.0;
    blobFromImage(inputImg, blob, normalizationFactor, Size(640, 640), Scalar(), true, false);
    model.setInput(blob);

    std::vector<int64_t> inputDims = {1, 3, h, w};
    std::vector<float> inputTensorValues(blob.begin<float>(), blob.end<float>()); 


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
    destroyAllWindows();*/
}