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
    if(argc < 3){
        cerr << "Usage: <test image path> <object_detection_dataset path>\n";
        return -1;
    }

    return(0);
}
