#ifndef CONTRA_ARGS_HPP
#define CONTRA_ARGS_HPP

#include <map>
#include <string>

namespace contra {

//! \brief Process the argument list for this app.
int processArguments(
    int argc,
    char** argv,
    std::map<std::string, std::string> & key_value_pair); 

}

#endif // CONTRA_ARGS_HPP
