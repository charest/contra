#ifndef CONTRA_CONTRA_HPP
#define CONTRA_CONTRA_HPP

#include "codegen.hpp"
#include "parser.hpp"

namespace contra {

// top ::= definition | external | expression | ';'
int MainLoop( Parser &, CodeGen &, bool );

}


#endif //CONTRA_CONTRA_HPP
