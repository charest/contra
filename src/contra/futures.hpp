#ifndef CONTRA_FUTURES_HPP
#define CONTRA_FUTURES_HPP

#include "config.hpp"
#include "recursive.hpp"

#include <forward_list>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// AST plotting class
////////////////////////////////////////////////////////////////////////////////
class FutureIdentifier : public RecursiveAstVisiter {
public:

  void runVisitor(FunctionAST&e)
  { e.accept(*this); }

};



}

#endif // CONTRA_FUTURES_HPP
