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
Mat cardValues(vector<Detection> detections, YOLO_model &model, Mat &frame);

#endif
