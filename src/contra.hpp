#ifndef CONTRA_CONTRA_HPP
#define CONTRA_CONTRA_HPP

#include "ContraJIT.hpp"
#include "codegen.hpp"
#include "parser.hpp"

namespace contra {

// top ::= definition | external | expression | ';'
void MainLoop( Parser &, CodeGen &, ContraJIT & );

// Top-Level parsing and JIT Driver
void InitializeModuleAndPassManager(CodeGen &, ContraJIT &);

}


#endif //CONTRA_CONTRA_HPP
