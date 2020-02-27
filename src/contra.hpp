#ifndef CONTRA_CONTRA_HPP
#define CONTRA_CONTRA_HPP

#include "codegen.hpp"
#include "parser.hpp"

namespace contra {

struct InputsType;

// top ::= definition | external | expression | ';'
void mainLoop( Parser &, CodeGen &, const InputsType & );

}


#endif //CONTRA_CONTRA_HPP
