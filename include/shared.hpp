#ifndef SHARED_HPP
#define SHARED_HPP

#include <string>
#include <opencv2/highgui.hpp>

namespace Shared{

    //NOTE: Possible choices
    enum class Choice {Camera, File, Invalid, Help};

    Choice parseChoice(std::string s);
}

#endif
