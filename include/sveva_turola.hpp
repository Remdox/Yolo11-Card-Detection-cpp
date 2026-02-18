#ifndef SVEVA_TUROLA_HPP
#define SVEVA_TUROLA_HPP

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "marco_annunziata.hpp"

using namespace std;
using namespace cv;

int frameCapture(string data_path, string labels_path);
int processStream(string data_path, VideoCapture cap, VideoWriter out, Mat frame, int savedCount, string labels_path);
void playOutputVideo(string path, double fps);
double getDistance(const Rect& r1, const Rect& r2);
Mat cardValues(vector<Detection> detections, YOLO_model &model, Mat &frame);
void drawCardGroup(Mat& img, const vector<Detection>& list, Scalar color, bool filled, YOLO_model& model);
void drawGameStatus(Mat& frame, const vector<Detection>& green, const vector<Detection>& red);

#endif
