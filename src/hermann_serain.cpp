#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <numeric>
#include <fstream>
#include <sstream>
#include <array>
#include <stdexcept>
#include <filesystem>
#include <termios.h>
#include <unistd.h>

#include "./../include/hermann_serain.hpp"
#include "./../include/marco_annunziata.hpp"
#include "./../include/shared.hpp"

namespace fs = std::filesystem;
using namespace cv;
using namespace std;
using namespace Shared;

//User parameters consts
const std::string DEFAULT_VIDEO_PATH = "../data/test/videos/default_video.mp4";
const std::string DEFAULT_IMAGE_PATH = "../data/test/images/default_image.jpg";
const std::string FLAG_VIDEO = "--default-video";
const std::string FLAG_IMAGE = "--default-image";

//Tab utilities
struct TerminalRawMode
{
    termios oldt;

    TerminalRawMode()
    {
        tcgetattr(STDIN_FILENO, &oldt);
        termios newt = oldt;

        newt.c_lflag &= ~(ICANON | ECHO); // no buffer, no echo
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    }

    ~TerminalRawMode()
    {
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    }
};

void splitPath(const string& fullPath, string& directory, string& partial);
vector<string> findMatches(const string& directory, const string& partial);
void redrawLine(const string& text);


//Help page methods
void printHelp();
void handleCamera();
string handleFile();


//WARNING: THIS PROGRAM USE THE TERMINAL API AND PROCESS THE TERMINAL IN RAW MODE, 
//IF THE TERMINAL DOESN'T WORK AS BEFORE TYPE BLINDED "reset"
UserData readInput(int argc, char** argv)
{
    //NOTE: Here I'm gonna read the user input from the terminal
    UserData result(Choice::Invalid, "");

    if(argc > 0) {
        std::vector<std::string> args(argv, argv + argc);
        
        for (int i = 1; i < argc; ++i) {
            if (args[i] == FLAG_VIDEO) {
                result.data_path = DEFAULT_VIDEO_PATH;
                result.choice = Choice::File;
            } 
            else if (args[i] == FLAG_IMAGE) {
                result.data_path = DEFAULT_IMAGE_PATH;
                result.choice = Choice::File;
            }
        }
    }

    if(result.choice != Choice::Invalid)
        return result;

    //NOTE: I'm gonna read the user input from now on
    string choiceStr;

    // NOTE: Let's keep asking if the input is invalid
    while (result.choice == Choice::Invalid)
    {
        cout << "--------------------------------------------\n";
        cout << " C = Camera | F = File | H = Help\n";
        cout << "--------------------------------------------\n";
        cout << "Write your choice: ";

        cin >> choiceStr;
        result.choice = parseChoice(choiceStr);

        switch (result.choice)
        {
            case Choice::Camera:
            {
                handleCamera();
            }
            break;

            case Choice::File:
            {
                result.data_path = handleFile();
            }
            break;

            case Choice::Help:
            {
                printHelp();
                result.choice = Choice::Invalid; //NOTE: continuing loop
            }
            break;

            case Choice::Invalid:
            {
                cout << "Usage: write C or F (or H for help)!\n";
            }
        }
    }

    return result;
}


//NOTE: Print the help page
void printHelp()
{
    cout << "\n";
    cout << "============================================\n";
    cout << "            APPLICATION HELP PAGE            \n";
    cout << "============================================\n";
    cout << "\n";
    cout << "Commands:\n";
    cout << "  C            Use camera as input\n";
    cout << "  F            Use file as input\n";
    cout << "  H / -h       Show this help page\n";
    cout << "\n";
    cout << "Details:\n";
    cout << "  - Camera closes by pressing 'q'\n";
    cout << "  - File path must not contain spaces\n";
    cout << "  - TAB completion not implemented yet\n";
    cout << "\n";
    cout << "============================================\n";
    cout << endl;
}

void handleCamera()
{
    cout << "Selected camera (press q to close the camera)\n";
}

string handleFile()
{
    TerminalRawMode rawMode; // abilita modalità raw

    string currentPath;
    cout << "Insert file path: ";
    cout.flush();

    while (true)
    {
        char ch;
        read(STDIN_FILENO, &ch, 1);

        // ENTER
        if (ch == '\n')
        {
            cout << endl;
            return currentPath;
        }

        // BACKSPACE
        if (ch == 127)
        {
            if (!currentPath.empty())
            {
                currentPath.pop_back();
                redrawLine(currentPath);
            }
            continue;
        }

        // TAB
        if (ch == '\t')
        {
            string dir, partial;
            splitPath(currentPath, dir, partial);

            auto matches = findMatches(dir, partial);

            if (matches.size() == 1)
            {
                currentPath.erase(currentPath.size() - partial.size());
                currentPath += matches[0];
            }
            else if (matches.size() > 1)
            {
                cout << "\n";
                for (const auto& m : matches)
                    cout << "  " << m << "\n";
            }

            redrawLine(currentPath);
            continue;
        }

        // Normal char
        if (isprint(ch))
        {
            currentPath.push_back(ch);
            cout << ch;
            cout.flush();
        }
    }
}


//Print utilities implementations
void splitPath(const string& fullPath, string& directory, string& partial)
{
    size_t pos = fullPath.find_last_of('/');

    if (pos == string::npos)
    {
        directory = ".";
        partial = fullPath;
    }
    else
    {
        directory = fullPath.substr(0, pos);
        partial = fullPath.substr(pos + 1);
    }
}

vector<string> findMatches(const string& directory, const string& partial)
{
    vector<string> matches;

    if (!fs::exists(directory))
        return matches;

    for (const auto& entry : fs::directory_iterator(directory))
    {
        string name = entry.path().filename().string();

        if (name.rfind(partial, 0) == 0)
            matches.push_back(name);
    }

    return matches;
}

void redrawLine(const string& text)
{
    cout << "\rInsert file path: " << text;
    cout << "\033[K"; // delete at the row end
    cout.flush();
}

//NOTE: Metrics
bool checkPath(const string& fullPath) {
    try {
        if (fs::exists(fullPath) && fs::is_regular_file(fullPath)) {
            return true;
        }
    } catch (const fs::filesystem_error& e) {
        cerr << "[ERROR] Filesystem access error: " << e.what() << endl;
    }
    return false;
}

bool canComputeMetrics(string filePath, string directoryName) {
    fs::path p(filePath);
    string filename = p.stem().string();
    
    // Construct the path dynamically based on directoryName
    string labelPath = "./../data/test/ground_truths/" + directoryName + "/" + filename + ".txt";
    
    if (checkPath(labelPath)) {
        cout << "[METRICS] Ground truth found in " << directoryName << " for: " << filename << ". Proceeding with calculation." << endl;
        return true;
    }

    cout << "[INFO] No ground truth labels found at: " << labelPath << ". Skipping metrics." << endl;
    return false;
}

bool canComputeImageMetrics(string imgPath) {
    return canComputeMetrics(imgPath, "images");
}

bool canComputeVideoMetrics(string videoPath) {
    return canComputeMetrics(videoPath, "videos");
}

//NOTE: Helper method to compute the IoU
float calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    int intersectionArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = box1.area() + box2.area() - intersectionArea;

    if (unionArea == 0) return 0.0f;
    return static_cast<float>(intersectionArea) / unionArea;
}

//NOTE: Method used to compute the Detections for both images and videos
MetricsCounter updateMetricsCounters(std::vector<Detection> predictedDetections, std::vector<Detection> actualDetections) {
    const float IOU_THRESHOLD = 0.5f;
    MetricsCounter frameResults;
    
    // Copy of ground truth to manage matching and removals
    std::vector<Detection> remainingGT = actualDetections;

    for (const auto& pred : predictedDetections) {
        int bestMatchIdx = -1;
        float maxIoU = 0.0f;

        // Search for the best Ground Truth match (Same Class + Highest IoU)
        for (int i = 0; i < remainingGT.size(); ++i) {
            if (pred.classId == remainingGT[i].classId) {
                float currentIoU = calculateIoU(pred.boundingBox, remainingGT[i].boundingBox);
                if (currentIoU > maxIoU) {
                    maxIoU = currentIoU;
                    bestMatchIdx = i;
                }
            }
        }

        // Validate the match against the threshold
        if (bestMatchIdx != -1 && maxIoU >= IOU_THRESHOLD) {
            frameResults.tp++;
            
            //WARNING: CHECK CORRECTNESS WITH DEBUGING

            /* CARD CONSOLIDATION LOGIC 
               Per your requirement: remove the matched GT and other GTs of the 
               same class within the surrounding "card-sized" region.
            */
            cv::Rect matchedRect = remainingGT[bestMatchIdx].boundingBox;
            int matchedClass = remainingGT[bestMatchIdx].classId;
            
            // Remove the exact match first
            remainingGT.erase(remainingGT.begin() + bestMatchIdx);

            // Remove nearby boxes of the same card (approx. same region)
            remainingGT.erase(
                std::remove_if(remainingGT.begin(), remainingGT.end(),
                    [&](const Detection& d) {
                        return (d.classId == matchedClass && calculateIoU(matchedRect, d.boundingBox) > 0.1f);
                    }),
                remainingGT.end()
            );

        } else {
            frameResults.fp++; //NOTE: No valid ground truths found for this prediction
        }
    }

    // NOTE: Ricordati che marco ha etichettato il video in modo strano !
    // Remaings ground truths are false negative
    frameResults.fn = static_cast<int>(remainingGT.size());

    return frameResults;
}

void printFinalMetrics(const MetricsCounter& total) {
    if (total.tp + total.fp == 0 || total.tp + total.fn == 0) {
        cout << "[METRICS] Warning: Not enough data to compute Precision/Recall." << endl;
        return;
    }

    float precision = static_cast<float>(total.tp) / (total.tp + total.fp);
    float recall = static_cast<float>(total.tp) / (total.tp + total.fn);
    float f1Score = (precision + recall > 0) ? (2.0f * (precision * recall) / (precision + recall)) : 0.0f;

    cout << "\n--- Final Performance Metrics ---" << endl;
    cout << "Total True Positives (TP): " << total.tp << endl;
    cout << "Total False Positives (FP): " << total.fp << endl;
    cout << "Total False Negatives (FN): " << total.fn << endl;
    cout << "---------------------------------" << endl;
    cout << "Precision: " << precision << endl;
    cout << "Recall:    " << recall << endl;
    cout << "F1-Score:  " << f1Score << endl;
    cout << "---------------------------------\n" << endl;
}

std::vector<Detection> parseYoloLabels(string labelPath, int imgW, int imgH) {
    std::vector<Detection> actualDetections;
    ifstream file(labelPath);
    string line;

    try {
        while (getline(file, line)) {
            if (line.empty()) continue;

            stringstream ss(line);
            int classId;
            float x_center, y_center, w, h;

            if (!(ss >> classId >> x_center >> y_center >> w >> h)) {
                throw std::runtime_error("Invalid YOLO format in file: " + labelPath);
            }

            //NOTE: Converting bounding box width and height from percentage to actual values
            int pixelW = static_cast<int>(w * imgW); 
            int pixelH = static_cast<int>(h * imgH);
            //NOTE: Moving the bounding box center to the left top corner.
            int pixelX = static_cast<int>((x_center * imgW) - (pixelW / 2.0f));
            int pixelY = static_cast<int>((y_center * imgH) - (pixelH / 2.0f));

            Detection gt;
            gt.classId = classId;
            gt.boundingBox = cv::Rect(pixelX, pixelY, pixelW, pixelH);
            gt.classConfidence = 1.0f; 

            actualDetections.push_back(gt);
        }
    } catch (const std::exception& e) {
        cerr << "[ERROR] Reading image ground truth: " << e.what() << endl;
        file.close();
        return {}; // Return empty vector on error
    }
    file.close();
    return actualDetections;
}

void computeImageMetrics(string imagePath, std::vector<Detection> predictedDetections) {
    if (!canComputeImageMetrics(imagePath)) {
        return;
    }

    fs::path p(imagePath);
    string filename = p.stem().string();
    string labelPath = "../data/test/ground_truths/images/" + filename + ".txt";
    
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        cerr << "[ERROR] Could not open image to get dimensions: " << imagePath << endl;
        return;
    }

    // Extraction of ground truth using the helper method
    std::vector<Detection> actualDetections = parseYoloLabels(labelPath, img.cols, img.rows);

    if (!actualDetections.empty()) {
        MetricsCounter results = updateMetricsCounters(predictedDetections, actualDetections);
        printFinalMetrics(results);
    }
}


//NOTE: Compute Video Metrics

std::ifstream g_motFile;
bool g_videoMetricsEnabled = false;
MetricsCounter g_cumulativeVideoMetrics;
std::string g_lastLine; 

void initObjectsForVideoMetrics(string videoPath) {
    if (g_motFile.is_open()) g_motFile.close();
    g_videoMetricsEnabled = false;
    g_cumulativeVideoMetrics = MetricsCounter();
    g_lastLine = "";

    if (!canComputeVideoMetrics(videoPath)) return;

    fs::path p(videoPath);
    string labelPath = "./../data/test/ground_truths/videos/" + p.stem().string() + ".txt";

    g_motFile.open(labelPath);
    if (g_motFile.is_open()) {
        g_videoMetricsEnabled = true;
        cout << "[METRICS] File MOT aperto per la lettura sequenziale." << endl;
    }
}

void computeVideoMetrics(std::vector<Detection> predictedDetections, int frameCount) {
    if (!g_videoMetricsEnabled || !g_motFile.is_open()) return;

    std::vector<Detection> actualDetections;
    string line;
    
    // Se abbiamo una riga salvata dal ciclo precedente, iniziamo da quella
    if (!g_lastLine.empty()) {
        line = g_lastLine;
        g_lastLine = "";
    } else if (!getline(g_motFile, line)) {
        return; // Fine del file
    }

    while (true) {
        stringstream ss(line);
        string token;
        vector<string> tokens;
        
        // Parsing veloce della riga (CSV)
        while (getline(ss, token, ',')) tokens.push_back(token);

        if (tokens.size() >= 8) {
            int currentLineFrame = stoi(tokens[0]);

            if (currentLineFrame == frameCount) {
                Detection d;
                d.boundingBox = cv::Rect(stof(tokens[2]), stof(tokens[3]), stof(tokens[4]), stof(tokens[5]));
                d.classId = stoi(tokens[7]);
                d.classConfidence = 1.0f;
                actualDetections.push_back(d);
            } 
            else if (currentLineFrame > frameCount) {
                g_lastLine = line;
                break;
            }
        }

        if (!getline(g_motFile, line)) break;
    }
    MetricsCounter frameResults = updateMetricsCounters(predictedDetections, actualDetections);
    
    g_cumulativeVideoMetrics.tp += frameResults.tp;
    g_cumulativeVideoMetrics.fp += frameResults.fp;
    g_cumulativeVideoMetrics.fn += frameResults.fn;
}


void printFinalVideoMetrics() {
    if (!g_videoMetricsEnabled) {
        cout << "[METRICS] No data accumulated for this video." << endl;
        return;
    }

    cout << "\n--- FINAL CUMULATIVE VIDEO STATISTICS ---" << endl;
    printFinalMetrics(g_cumulativeVideoMetrics);

    if (g_motFile.is_open()) {
        g_motFile.close();
    }

    g_videoMetricsEnabled = false;
    g_lastLine = "";
    g_cumulativeVideoMetrics.tp = 0;
    g_cumulativeVideoMetrics.fp = 0;
    g_cumulativeVideoMetrics.fn = 0;

    cout << "[INFO] Metrics state reset for the next session." << endl;
}