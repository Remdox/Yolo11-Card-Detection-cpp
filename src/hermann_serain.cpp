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

#include "./../include/hermann_serain.hpp"
#include "./../include/shared.hpp"

using namespace cv;
using namespace std;
using namespace Shared;



UserData readInput() {

    string choiceStr;
    UserData result(Choice::Invalid, "");

    //NOTE: Let's keep asking if the input it's invalid
    while (result.choice == Choice::Invalid)
    {
        //TODO: Improve the help page by adding commands
        cout << "Write C to use camera or F to use a file: ";
        cin >> choiceStr;
        result.choice = parseChoice(choiceStr);

        switch (result.choice)
        {
            case Choice::Camera:
            {
                cout << "Selected camera (press q to close the camera)\n";
            }
            break;

            case Choice::File:
            {
                cout << "Insert file path: ";
                //TODO: Implement the tab function right here
                cin >> result.data_path;
            }
            break;

            case Choice::Invalid:
            {
                cout << "Usage: write C or F!\n";
            }
        }


    }

    return result;
}