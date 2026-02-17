#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

#include "./../include/sveva_turola.hpp"
#include "./../include/marco_annunziata.hpp"
#include "./../include/hermann_serain.hpp"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// TODO fare commenti doxygen

int frameCapture(string data_path, string labels_path) {
    string outputDir = "../output/frames/"; //output directory
    Mat frame;
    int savedCount = 0;
    VideoCapture cap;
    VideoWriter out;

    // creation of output directory: at every run, creates a new directory, in order to have only the frames for that run 
    if (fs::exists(outputDir)){
        fs::remove_all(outputDir);
    }
    fs::create_directory(outputDir);

    if(data_path == "0"){
        // if camera was selected
        cap.open(0);

        if (!cap.isOpened()) {
            cerr << "Error opening camera!" << endl;
            return -1;
        }

        cout << "Loading camera video..." << endl;
        savedCount = processStream(cap, out, frame, savedCount, labels_path);
    } else {
        // if video file was selected
        cap.open(data_path);

        if (!cap.isOpened()) {
            cerr << "Error opening video!" << endl;
            return -1;
        }

        cout << "Loading video..." << endl;
        initObjectsForVideoMetrics(data_path);
        savedCount = processStream(cap, out, frame, savedCount, labels_path);
        printFinalVideoMetrics();
    }

    cout << "\nExtraction completed! Frames used: " << savedCount << endl;
    cout << "Video saved in output directory" << endl;

    return 0;
}

int processStream(VideoCapture cap, VideoWriter out, Mat frame, int savedCount, string labels_path){
    // initialization of YOLO model
    YOLO_model model;
    model.setModelName("YOLO11s");
    vector<string> dataClasses = model.getDataClasses(labels_path);

    // setup of video variables
    int frameCount = 0;
    int codec = VideoWriter::fourcc('m', 'p', '4', 'v');
    int fps = cap.get(CAP_PROP_FPS);

    vector<Detection> detections;

    // checks if input is the camera
    bool isCamera = (cap.get(CAP_PROP_FRAME_COUNT) <= 0);

    cap.read(frame);
    if (frame.empty()){
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    // opens output video
    out.open("../output/detections.mp4", codec, fps, Size(frame_width, frame_height), true);
    if(!out.isOpened()){
        cerr << "Could not open the output video file for write\n";
        return -1;
    }

    int cardCount = -1;

    while (true){
        cap.read(frame);

        if (frame.empty()){
            break;
        }

        savedCount++;
        int key = waitKey(1) & 0xFF;

        if (frameCount % (fps/2) == 0){
            detections = model.detectionPipeline(frame);
            computeVideoMetrics(detections, frameCount+1); 
        }

        frameCount++;
        Mat outputFrame = cardValues(detections, model, frame);

        int currentCardCount = detections.size();

        // saving only relevant frames, that are the frames in which the number of detections changes
        if(currentCardCount != cardCount){
            string filename = "../output/frames/frame_" + to_string(savedCount) + ".jpg";
            imwrite(filename, outputFrame);
            cardCount = currentCardCount;
        }

        // adds each frame to the video
        out.write(outputFrame);

        if(isCamera == true){
            // stops the camera
            if (key == 'q') {
                cout << "Closed camera!\n";
                break;
            }
        }
    }

    cap.release();
    out.release();
    destroyAllWindows();

    // calls the method to show the video
    playOutputVideo("../output/detections.mp4", fps);

    return savedCount;
}

void playOutputVideo(string path, double fps){
    // opens the output video detections.mp4
    VideoCapture playback;
    playback.open(path);
    
    if (!playback.isOpened()) {
        cerr << "Error in opening " << path << endl;
        return;
    }

    namedWindow("Final video", WINDOW_NORMAL);
    setWindowProperty("Final video", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

    Mat playFrame;

    cout << "Reproduction of the output video. Press 'q' to close it or 'p' to pause it." << endl;
    while (playback.read(playFrame)) {
        // shows the output video
        imshow("Final video", playFrame);
        
        // computes the delay for the playback of the video
        int delay = 30;

        if(fps > 0){
            delay = 1000 / (int) fps;
        }
        
        int key = waitKey(delay);
        // used to stop the playback of the output video
        if (key == 'q') {
            break;
        } else if (key == 'p'){
            // used to pause the playback of the output video
            waitKey(0);
        }
    }
    
    playback.release();
    destroyWindow("Final video");
}

double getDistance(const Rect& r1, const Rect& r2){
    // computes center x and center y for both rectangles
    double cx1 = r1.x + (r1.width / 2.0);
    double cy1 = r1.y + (r1.height / 2.0);
    double cx2 = r2.x + (r2.width / 2.0);
    double cy2 = r2.y + (r2.height / 2.0);

    // returns euclidean distance
    return sqrt(pow(cx2 - cx1, 2) + pow(cy2 - cy1, 2));
}

Mat cardValues(vector<Detection> detections, YOLO_model &model, Mat &frame){ // TODO da rivedere
    vector<Detection> green, blue, red;

    // groups the detections by value
    map<int, vector<Detection>> classGroups;
    for(const Detection& d : detections){
        classGroups[d.classId].push_back(d);
    }

    // iterates on each group of classGroups
    for(auto& [id, dets] : classGroups){
        // used to mark the symbols already processed
        vector<bool> processed(dets.size(), false);

        for (size_t i = 0; i < dets.size(); i++){
            if (processed[i]){
                continue;
            }

            // takes the first detection not already processed
            Detection fullCard = dets[i];
            processed[i] = true;

            // searches among the other symbols of the same group
            for (size_t j = i + 1; j < dets.size(); j++){
                // if it finds another identical and very close symbol, then they belong to the same card
                if (!processed[j] && getDistance(dets[i].boundingBox, dets[j].boundingBox) < 280.0) { // TODO cambiare valore
                    // draws the bounding box of the entire card based on the coordinates of the detections
                    int minX = min(fullCard.boundingBox.x, dets[j].boundingBox.x);
                    int minY = min(fullCard.boundingBox.y, dets[j].boundingBox.y);
                    int maxX = max(fullCard.boundingBox.x + fullCard.boundingBox.width, dets[j].boundingBox.x + dets[j].boundingBox.width);
                    int maxY = max(fullCard.boundingBox.y + fullCard.boundingBox.height, dets[j].boundingBox.y + dets[j].boundingBox.height);

                    Rect mergedBox(minX, minY, maxX - minX, maxY - minY);

                    // TODO da commentare
                    int minSize = 100; 
    
                    if (mergedBox.width < minSize) {
                        int diff = minSize - mergedBox.width;
                        mergedBox.x -= diff / 2;
                        mergedBox.width = minSize;
                    }
    
                    if (mergedBox.height < minSize) {
                        int diff = minSize - mergedBox.height;
                        mergedBox.y -= diff / 2;
                        mergedBox.height = minSize;
                    }

                    mergedBox &= Rect(0, 0, frame.cols, frame.rows);

                    fullCard.boundingBox = mergedBox;
                    processed[j] = true;
                }
            }

            // Hi-Lo system
            char cardNumber = model.getClassName(id)[0];
            
            if(cardNumber >= '2' && cardNumber <= '6'){
                green.push_back(fullCard);
            } else if(cardNumber >= '7' && cardNumber <= '9'){
                blue.push_back(fullCard);
            } else {
                red.push_back(fullCard);
            }
        }
    }

    Mat outputFrame = frame.clone();
    
    // draws original bounding boxes using white
    if(!detections.empty()){
        Mat detectionOverlay = outputFrame.clone();
        detectionOverlay = model.drawBoundingBoxes(detectionOverlay, detections, Scalar(255, 255, 255));

        addWeighted(detectionOverlay, 0.7, outputFrame, 0.3, 0, outputFrame);
    }

    // draws the bounding box for the complete card
    drawCardGroup(outputFrame, green, Scalar(0, 255, 0), false, model);
    drawCardGroup(outputFrame, blue, Scalar(255, 0, 0), false, model);
    drawCardGroup(outputFrame, red, Scalar(0, 0, 255), false, model);

    // fills the bounding box with the respective color
    Mat overlay = outputFrame.clone();

    drawCardGroup(overlay, green, Scalar(0, 255, 0), true, model);
    drawCardGroup(overlay, blue, Scalar(255, 0, 0), true, model);
    drawCardGroup(overlay, red, Scalar(0, 0, 255), true, model);

    addWeighted(overlay, 0.3, outputFrame, 0.7, 0, outputFrame);

    // computes and shows the running count
    drawGameStatus(outputFrame, green, red);

    return outputFrame;
}

// TODO da commentare
void drawCardGroup(Mat& img, const vector<Detection>& list, Scalar color, bool filled, YOLO_model& model){
    int thickness = max(1, int(max(img.rows, img.cols) / 640));

    int boxThickness;
    if(filled == true){
        boxThickness = FILLED;
    } else {
        boxThickness = 2 * thickness;
    }
    
    for(size_t i = 0; i < list.size(); i++) {
        Rect box = list[i].boundingBox;
        rectangle(img, box, color, boxThickness);

        string label = model.getClassName(list[i].classId);

        double fontScale = 0.5 * thickness;
        int fontThickness = max(1, int(1.5 * thickness));
        int baseLine = 0;
            
        Size textSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, fontScale, fontThickness, &baseLine);

        int textX = box.x + (box.width - textSize.width) / 2;
        int textY = box.y + box.height + textSize.height + (10 * thickness);

        Point textPosition(textX, textY);

        putText(img, label, textPosition, FONT_HERSHEY_SIMPLEX, 0.5 * thickness, color, 1.5 * thickness);
    }
}

void drawGameStatus(Mat& frame, const vector<Detection>& green, const vector<Detection>& red){
    // computes running count
    int runningCount = (int)green.size() - (int)red.size();

    // setup to visualize the running count
    int xPos = 20;
    int yPos = 20;
    int width = 300;
    int height = 70;

    // draws the background for the running count
    Mat overlay;
    frame.copyTo(overlay);
    rectangle(overlay, Rect(xPos, yPos, width, height), Scalar(0, 0, 0), FILLED);
    addWeighted(overlay, 0.5, frame, 0.5, 0, frame);
    
    // draws the border for the background
    rectangle(frame, Rect(xPos, yPos, width, height), Scalar(255, 255, 255), 2);

    // defines the color and the string 
    Scalar countColor = Scalar(255, 255, 255); // using white for running count = 0
    string prefix = "";
    string statusText = "Neutral";

    if (runningCount > 0) {
        // using green for player's advantage
        countColor = Scalar(0, 255, 0);
        prefix = "+";
        statusText = "Player Advantage";
    } else if (runningCount < 0) {
        // using red for house's advantage
        countColor = Scalar(0, 0, 255);
        statusText = "House Advantage";
    }

    string countStr = "Count: " + prefix + to_string(runningCount);
    
    // writes running count
    putText(frame, countStr, Point(xPos + 20, yPos + 45), FONT_HERSHEY_SIMPLEX, 1.2, countColor, 2);

    if (runningCount >= 2) {
        // in this case, the count is high and this indicates an advantage for the players
        int alertY = yPos + height + 10;

        // defines the alert rectangle
        Rect alertBox;
        alertBox.x = xPos;
        alertBox.y = alertY;
        alertBox.width = width;
        alertBox.height = 50;

        Mat overlay;
        frame.copyTo(overlay);

        rectangle(overlay, alertBox, Scalar(0, 0, 255), FILLED);

        addWeighted(overlay, 0.7, frame, 0.3, 0, frame);
        
        Point textPosition;
        textPosition.x = alertBox.x + 10;
        textPosition.y = alertBox.y + 35;

        // warns to keep an eye on the players' moves
        putText(frame, "Monitor players' moves!", textPosition, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1);
    }
}