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
#include "./../include/shared.hpp"

namespace fs = std::filesystem;
using namespace cv;
using namespace std;
using namespace Shared;

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
UserData readInput()
{
    string choiceStr;
    UserData result(Choice::Invalid, "");

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