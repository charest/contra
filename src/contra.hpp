#ifndef CONTRA_CONTRA_HPP
#define CONTRA_CONTRA_HPP

#include "codegen.hpp"
#include "parser.hpp"

namespace contra {

// top ::= definition | external | expression | ';'
void mainLoop( Parser &, CodeGen &, bool, bool );

}


#endif //CONTRA_CONTRA_HPP
