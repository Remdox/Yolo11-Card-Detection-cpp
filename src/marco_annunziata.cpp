
#include "marco_annunziata.hpp"
#include "shared.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp> 
#include <onnxruntime_cxx_api.h>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::dnn;

const float CLASS_CONFIDENCE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.1;

YOLO_model::YOLO_model(): env(ORT_LOGGING_LEVEL_VERBOSE, "YOLOModel", YOLO_model::logger, this), session(nullptr), sessionOptions(){
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions.SetLogId("YOLOModel");

    const char* modelPath = "../data/model/last.onnx";
    session = Ort::Session(env, modelPath, sessionOptions);
}

void YOLO_model::logger(void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location,
    const char* message){
    static std::mutex logMutex;
    static std::ofstream logFile("./onnxruntime.log", std::ios::out | std::ios::app);

    const char* severityStr = "";
    switch (severity) {
        case ORT_LOGGING_LEVEL_VERBOSE: severityStr = "VERBOSE"; break;
        case ORT_LOGGING_LEVEL_INFO:    severityStr = "INFO";    break;
        case ORT_LOGGING_LEVEL_WARNING: severityStr = "WARNING"; break;
        case ORT_LOGGING_LEVEL_ERROR:   severityStr = "ERROR";   break;
        case ORT_LOGGING_LEVEL_FATAL:   severityStr = "FATAL";   break;
        default:                        severityStr = "UNKNOWN"; break;
    }

    // concatenating the output into a single string buffer to use later when writing into the log file
    std::ostringstream os;
    os << "[" << (logid ? logid : "no-logid") << "] "
       << severityStr << " | " << (category ? category : "") << " @ "
       << (code_location ? code_location : "") << "\n    "
       << (message ? message : "") << "\n";

    std::lock_guard<std::mutex> lock(logMutex);

    if (severity <= ORT_LOGGING_LEVEL_INFO) {
        // INFO and VERBOSE go to file
        if (logFile) {
            logFile << os.str();
            logFile.flush();
        }
    } else {
        // WARNING, ERROR, FATAL go to stderr
        std::cerr << os.str();
    }
}


void YOLO_model::detectObjects(Mat &img, int inputSize){
    // retrieve classes for later
    vector<string> dataClasses;
    ifstream ifs("../data/model/labels.txt");
    string line;
    while (getline(ifs, line)) dataClasses.push_back(line);

    // Giving the original image size is possible in OnnxRuntime because it supports dynamic input, but the processing of the model is way slower.
    // The input needs to be resized for the YOLO model anyway to have good accuracy, so pre-processing of the image is still needed and dynamic input is not used for YOLO
    int inputHeight = img.rows;
    int inputWidth  = img.cols;

    /* YOLO: PRE-PROCESSING required for the image
        the YOLO model requires the image to be:
            1. Of dimensions multiples of 32, because we know YOLO has a CNN in its architecture:
                    -There are 5 convolutional layers with stride=2 where the convolutions downsample the input by a factor of 2^5=32, so the feature map has size inputSize/32
                    -The stride of convolution replaces max pooling of typical CNNs
                    -If the inputSize is not divisible by 32, the feature map become of size non-integer which leads to feature misalignment
            2. Of overall size square (multiple of 32), typically 640x640 or 416x416
            3. not deformed (deformed = stretched image), since YOLO doesn't recognize deformed objects and, even if you train it for that, the predictions are still made in fixed grid cells
        This means that:
            - If an image is smaller or bigger than 640x640, it has to be resized to the same scale of 640x640 (requirement 2)
            - If the image is not square, you need to resize it while keeping its aspect ratio w:h, so that the result is not deformed (requirement 3)
               => the scaling factor has to be saved so that you can correctly resize the bounding box to the original size of the image
            - If the image is not square, after resizing you also need to add padding to have a resulting image of exactly 640x640 size (requirement 2)
        So the pre-processing of the images is composed of these steps:
            Case A: If the image is square, just resize it if needed and you're done
            Case B: If the image is not square (most of the time):
                1. Resize the image to a multiple of 32 while keeping its aspect ratio, which means: resize the longest side to 640 and adjust the shorter side following proportions w:h
                2. add padding to the shorter side in order to reach the square size.
        Note: to guarantee the best performance, the padding has to be done in the same we YOLO was trained for. 
        YOLO was trained using images with letterbox padding, which means that the image is centered and the padding is around the image.
        This is useful because in the uncentered padding the model would have a small bias of detecting objects which are closer to the corner.
        A symmetric padding (letterbox) is more robust to this positional bias.
    */
    
    Mat resizedImg;
    float resizedWidth = -1;
    float resizedHeight = -1;
    int xPaddedImgCorner  = -1; // TODO move inside else
    int yPaddedImgCorner  = -1;

    float imgAspectRatio    = static_cast<float>(inputWidth) / inputHeight;
    if(inputWidth == inputHeight){ /* Case A: image is square, just resize */
        resizedWidth   = YOLO_TARGET_INPUT_SIZE;
        resizedHeight  = YOLO_TARGET_INPUT_SIZE;
        resize(img, resizedImg, Size(static_cast<int>(resizedWidth), static_cast<int>(resizedHeight)));
    }
    else{ 
        /* Case B: image is not square. Resize following aspect ratio, then add letterbox padding*/
        if(imgAspectRatio > 1.0f){
            resizedWidth  = YOLO_TARGET_INPUT_SIZE;
            resizedHeight = YOLO_TARGET_INPUT_SIZE / imgAspectRatio; // imgAspectRatio > 1 -> have to divide so that the height follows the width increase/decrease
            //cout << "h before adjusting: " << resizedHeight << endl;
            resizedHeight = (static_cast<int>(resizedHeight) / 32) * 32; // Ensure dimensions of the image are multiples of 32, as required by YOLO11 (here width is already multiple of 32)
            
            xPaddedImgCorner = 0;                                    // It represents the corner of the centered image after letterbox padding, will be useful later to center the image
            yPaddedImgCorner = static_cast<int>((YOLO_TARGET_INPUT_SIZE/2) - resizedHeight/2);
        }
        else{
            resizedHeight = YOLO_TARGET_INPUT_SIZE;
            resizedWidth  = YOLO_TARGET_INPUT_SIZE * imgAspectRatio; // imgAspectRatio < 1 -> have to multiply to make the height follow the width value
            //cout << "w before adjusting: " << resizedWidth << endl;
            resizedWidth  = (static_cast<int>(resizedWidth) / 32) * 32;

            yPaddedImgCorner = 0;                                    // It represents the corner of the centered image after letterbox padding, will be useful later to center the image
            xPaddedImgCorner = static_cast<int>((YOLO_TARGET_INPUT_SIZE/2) - resizedWidth/2);
        }

        Mat tempImg;
        resize(img, resizedImg, Size(static_cast<int>(resizedWidth), static_cast<int>(resizedHeight)));
        resize(img, tempImg, Size(static_cast<int>(resizedWidth), static_cast<int>(resizedHeight)));
        //letterbox padding
        resizedImg = Mat::zeros(YOLO_TARGET_INPUT_SIZE, YOLO_TARGET_INPUT_SIZE, CV_8UC3);
        tempImg.copyTo(resizedImg(Rect(xPaddedImgCorner, yPaddedImgCorner, resizedWidth, resizedHeight)));
    }

    float resizeScalingFactor[2] = {
        static_cast<float>(inputWidth)  / resizedWidth,
        static_cast<float>(inputHeight) / resizedHeight 
    };
    
    // testing correct resizing
    cout << "Dynamic input size: W = " << inputWidth << " h = " << inputHeight << endl;
    cout << "Resized input size (after adjusting): W = " << resizedWidth << " h = " << resizedHeight << endl;
    cout << "Padded corner: X = " << xPaddedImgCorner << " Y = " << yPaddedImgCorner << endl;
    cout << "resize scaling factor: X = " << resizeScalingFactor[0] << " Y = " << resizeScalingFactor[1] << endl;

    namedWindow("resized", WINDOW_NORMAL);
    imshow("resized", resizedImg);
    waitKey(0);


    /* TENSORS
    OnnxRuntime needs a tensor as input.
    The ONNX tensor is not the mathematical tensor (although it is inspired by it).
    It's a multi-dimensional array of heterogeneous data types (float, int ...) which is optimized for computations.
    Tensors are useful because they are a convenient data structure for parallelizing GPU computations during DNN inference.*/
    
    // In OnnxRuntime the tensor is defined by its data (an array) and its shape (dimensions of the multi-dimensional array)
    // Create dynamic input tensor
    vector<int64_t> inputShape = {1, 3, static_cast<int>(resizedWidth), static_cast<int>(resizedHeight)}; // 4D tensor
    size_t inputTensorSize = 1 * 3 * resizedHeight * resizedWidth;
    vector<float> inputData(inputTensorSize); // the input image is flattened into a mono-dimensional array

    double normalizationFactor = 1.0 / 255.0;

    // Converting image data from cv::Mat to tensor
    for (int row = 0; row < resizedHeight; row++) {
        for (int col = 0; col < resizedWidth; col++) {
            Vec3b pixel = resizedImg.at<Vec3b>(row, col);                                                      // cv::Mat -> Tensor
            inputData[2 * resizedHeight * resizedWidth + row * resizedWidth + col] = pixel[0] * normalizationFactor; //    B    ->   R
            inputData[1 * resizedHeight * resizedWidth + row * resizedWidth + col] = pixel[1] * normalizationFactor; //    G         G
            inputData[0 * resizedHeight * resizedWidth + row * resizedWidth + col] = pixel[2] * normalizationFactor; //    R    ->   B
        }
    }

    // Allocating the tensor on CPU
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Creating the input tensor (which is a view of the data, NOT a deep copy!)
    Ort::Value inputOnnxTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputData.data(), inputTensorSize, inputShape.data(), inputShape.size());

    // the allocations of these strings need to be freed with some boilerplate code later
    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto outputName = session.GetOutputNameAllocated(0, allocator);
    
    // The API needs the array of inputs to set and the array of outputs to get (or even just a portion of output if needed).
    // So here you are setting the ORDER of the inputs such that the outputs are ordered accordingly
    vector<const char*> inputNames = {inputName.get()};
    vector<const char*> outputNames = {outputName.get()};
    
    // Running the inference
    auto outputTensors = session.Run(
        Ort::RunOptions{nullptr},               // e.g set a verbosity level only for this run
        inputNames.data(), &inputOnnxTensor, 1, // input to set
        outputNames.data(),                     // output to take
        1);

    /* OUTPUT TENSOR
    // YOLO11 is pre-trained by default on the COCO dataset, so the number of channels sould be:
    //                      84 = 4 + 80 (= class confidence, one for each class)
    // BUT, we fine-tuned the training of YOLO11 with our custom dataset, so the output layer is restructured to represent 53 classes.
    // This means the number of channels should be 57 = 4 (bounding box parameters) + 53 (= class confidence, one for each class)
    // Note: class probability       = probability of having detected one class in that region
    //       class confidence        = confidence of having detected an object of a class in that region, measured as class probability * objectness probability
    //       objectness probability  = probability of having an object in that region, useful for YOLOv5 but not for YOLO11, since there is already class confidence, so it's been removed.

    // Note, Yolov5 has a different representation of the output layer compared to Yolov8/Yolo11:
    // Yolov5 -> channels = 85 = 4 (bb_parameters) + 1 (objectness probability) + 80 (= class probabilities of corresponding 80 classes of the COCO dataset)
    // So, in this case the objectness is separate from class confidence and each class has the class probability instead of the class confidence.
    // To get class confidence of each class you have to multiply each class probability*objectness. So in YOLOv5 the objectness probability was useful!

    // To recap, our custom YOLO11 output tensor (3D Mat) has shape (1, 57, 8400*):
    // 1                               - is the batch size. If =0, it can change dynamically depending on the input of the net (not needed since in YOLO batch size = 1).
    // 57 (84 for default model)       - is the number of channels, where the first 4 are the bounding box parameters (x, y, width, height)
    //                                    and the other ones are class probabilities for COCO'S 80 classes (There is no objectiveness parameter in Yolov8/11)
    // 8400 *or lower                  - is the total number of regions where the predictions are made ("regions" = "grid points" = anchor locations).
    //                                    *Since the ONNX model is dynamic, the output tensor can have a lower number of detections depending on how big is the padding area.
    // The first and the third value become 0 when exported into ONNX format with dynamic=True, indicating that in the dynamic model these parameters can change on runtime.
    // Dynamic input/output shapes are not supported by OpenCV, but they are supported by ONNX Runtime.
    // In this case, OnnxRuntime is adopted instead of OpenCV more because of the superior inference efficency rather than the support of dynamic models.
    // See: https://github.com/orgs/ultralytics/discussions/17254
    */

    // Get dynamic output tensor info (data, shape, count)
    auto& outputTensor = outputTensors[0]; // In the case of YOLO11, only 1 output layer is used, so there is only 1 output tensor (outputTensors[0])
    auto tensorInfo = outputTensor.GetTensorTypeAndShapeInfo();
    auto tensorCount = tensorInfo.GetElementCount();
    auto tensorShape = tensorInfo.GetShape();
    float* outputData = outputTensor.GetTensorMutableData<float>(); // the output values are flattened into a 1D array
    // outputData[0..4...] = (x, y, width, height, confidence, classProb1, classProb2, ..., classProb80)
    
    int batchSize = static_cast<int>(tensorShape[0]);
    int numChannels = static_cast<int>(tensorShape[1]); 
    int numDetections = static_cast<int>(tensorShape[2]);  // This will vary based on input size and padding area
    
    cout << "Dynamic output shape:[" << batchSize << ", " << numChannels << ", " << numDetections << "], count:" << tensorCount << endl;

    vector<int> classIds;
    vector<float> detectedClassConfidences;
    vector<Rect> boundingBoxes;

    for(int i = 0; i < numDetections; i++){
        
        float* currentDetection = outputData + (i * numChannels); // 1D array of the 57 elements of one detection
        if(i == 0){
            for(int j = 0; j < numChannels; j++){
                cout << "data " << j << ": " << currentDetection[j] << endl;
            }
        }
        // The Mat constructor doesn't copy the data, it just wraps pointer classes_scores in a matrix header,
        // so this is used instead of the typical Mat constructor as it is more efficient.
        // Also, minMaxLoc accepts as input a Mat and as output a Point, so this is how scores and classId will be defined.
        float * firstClassScore = currentDetection + 4;
        Mat classScores(1, numChannels-4, CV_32FC1, firstClassScore);
        Point classId;
        double maxClassScore;
        minMaxLoc(classScores, 0, &maxClassScore, 0, &classId);

        if(maxClassScore > CLASS_CONFIDENCE_THRESHOLD){
            detectedClassConfidences.push_back(maxClassScore);
            classIds.push_back(classId.x);

            float xCenter   = currentDetection[0];
            float yCenter   = currentDetection[1];
            float boxWidth  = currentDetection[2];
            float boxHeight = currentDetection[3];

            // removing coordinates translation caused by the padding
            xCenter -= xPaddedImgCorner;
            yCenter -= yPaddedImgCorner;

            // // computing the bounding box of the detected object and scaling it to the original image size
            int xPosScaled = int((xCenter - 0.5f*boxWidth) * resizeScalingFactor[0]);
            int yPosScaled = int((yCenter - 0.5f*boxHeight) * resizeScalingFactor[1]);
            int bWScaled = int(boxWidth  * resizeScalingFactor[0]);
            int bHScaled = int(boxHeight * resizeScalingFactor[1]);

            Rect box = Rect(
                xPosScaled,  // x corner position
                yPosScaled,  // y corner position
                bWScaled,    // box width
                bHScaled     // box height
            );

            // Clamping bounding box to image boundaries
            box.x = max(0, box.x);
            box.y = max(0, box.y);
            box.width = min(box.width, img.cols - box.x);
            box.height = min(box.height, img.rows - box.y);


            boundingBoxes.push_back(box);
        }
    }
    
    // Apply Non-Maxima Suppression
    vector<int> resultNMS;
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
        string label = finalDetection.className + " " + to_string(static_cast<int>(finalDetection.classConfidence * 100)) + "%";
        putText(resultImg, label, Point(finalDetection.boundingBox.x, finalDetection.boundingBox.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        cout << "Class: " << finalDetection.className << " Confidence: " << finalDetection.classConfidence << endl;
    }
    namedWindow("Prediction with Box", WINDOW_NORMAL);
    imshow("Prediction with Box", resultImg);
    waitKey(0);
    destroyAllWindows();
}