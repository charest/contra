#ifndef CONTRA_LEAF_HPP
#define CONTRA_LEAF_HPP

#include "config.hpp"
#include "recursive.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// AST plotting class
////////////////////////////////////////////////////////////////////////////////
class LeafIdentifier : public RecursiveAstVisiter {

  bool CallsTask_ = false;
  
public:

  void runVisitor(FunctionAST&e)
  {
    CallsTask_ = false;
    e.accept(*this);
    e.setLeaf(!CallsTask_);
  }
  
  void postVisit(CallExprAST& e) override;
  void postVisit(AssignStmtAST& e) override;
  void postVisit(ForeachStmtAST& e) override;
  void postVisit(IndexTaskAST& e) override;

};

} // namespace

#endif // CONTRA_LEAF_HPP
