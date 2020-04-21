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

  std::map<VariableDef*, std::set<VariableDef*>> VariableTable_;

  void addFlow(VariableDef* Left, VariableDef* Right)
  { VariableTable_[Left].emplace(Right); }
  
public:

  void runVisitor(FunctionAST&e);
  
  void postVisit(CallExprAST& e) override;
  void visit(AssignStmtAST& e) override;
  void visit(VarDeclAST& e) override;
  void visit(FieldDeclAST& e) override;

};

} // namespace

#endif // CONTRA_FUTURES_HPP
