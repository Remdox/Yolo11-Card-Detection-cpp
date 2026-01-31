#ifndef HERMANN_SERAIN_HPP
#define HERMANN_SERAIN_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <numeric>

#include "shared.hpp"

using namespace cv;
using namespace std;
using namespace Shared;

//NOTE: Contains the user choices
struct UserData {
    Choice choice;
    string data_path;

    UserData(Choice _choice, string _data_path) 
        : choice(_choice), data_path(_data_path) {} 
};

//NOTE: Public methods
UserData readInput(int argc, char** argv);

#endif
