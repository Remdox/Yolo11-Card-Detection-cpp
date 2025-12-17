#include <string>
#include <stdexcept>

#include "./../include/shared.hpp"

namespace Shared{

    Choice parseChoice(std::string s) {
        //NOTE: Manual Trim
        while (!s.empty() && std::isspace((unsigned char)s.front())) s.erase(s.begin());
        while (!s.empty() && std::isspace((unsigned char)s.back()))  s.pop_back();

        if (s.size() != 1) return Choice::Invalid;

        char c = (char)std::toupper((unsigned char)s[0]);
        if (c == 'C' || c == 'c') return Choice::Camera;
        if (c == 'F' || c == 'f') return Choice::File;
        return Choice::Invalid;
    }
}
