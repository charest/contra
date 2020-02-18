#ifndef CONTRA_CONTRA_HPP
#define CONTRA_CONTRA_HPP

#include "codegen.hpp"
#include "parser.hpp"

namespace contra {

// top ::= definition | external | expression | ';'
void MainLoop( Parser &, CodeGen &, bool );

}


#endif //CONTRA_CONTRA_HPP
