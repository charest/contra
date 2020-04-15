#ifndef CONTRA_FLOW_HPP
#define CONTRA_FLOW_HPP

#include "config.hpp"
#include "recursive.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// AST plotting class
////////////////////////////////////////////////////////////////////////////////
class Flow : public RecursiveAstVisiter {

  bool IsFuture_ = false;

public:

  // Codegen function
  void runVisitor(FunctionAST&e)
  { e.accept(*this); }

private:
   
  // Codegen function
  void runVisitor(ExprAST&e)
  { e.accept(*this); }

  // Codegen function
  void runVisitor(StmtAST&e)
  {
    IsFuture_ = false;
    e.accept(*this);
  }

};



}

#endif // CONTRA_FLOW_HPP
