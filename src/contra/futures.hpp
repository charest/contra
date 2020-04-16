#ifndef CONTRA_FUTURES_HPP
#define CONTRA_FUTURES_HPP

#include "config.hpp"
#include "scope.hpp"
#include "recursive.hpp"

#include <forward_list>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// AST plotting class
////////////////////////////////////////////////////////////////////////////////
class FutureIdentifier : public RecursiveAstVisiter, public Scoper {
public:
  struct VariableEntry {
    unsigned Id;
    VarDeclAST * Node;
  };

private:
  std::forward_list< std::map<std::string, VariableEntry> > VariableTable_;

  void popScope() { VariableTable_.pop_front(); }
  void pushScope() { VariableTable_.push_front({}); }


  void addVariable(const std::string &Name, const VariableEntry & VarE)
  { VariableTable_.front().emplace(Name, VarE); }

  VariableEntry* findVariable(const std::string &Name);

public:

  void runVisitor(FunctionAST&e)
  {
    VariableTable_.clear();
    pushScope();
    e.accept(*this);
    popScope();
  }

  virtual void visit(VarAccessExprAST&e) override;
  
  virtual bool preVisit(VarDeclAST&e) override;
  virtual void postVisit(VarDeclAST&e) override;

  virtual void visit(AssignStmtAST&e) override;

  virtual void postVisit(CallExprAST&e) override;

  virtual bool preVisit(ForStmtAST&e) override;
  virtual void postVisit(ForStmtAST&e) override;

  virtual bool preVisit(ForeachStmtAST&e) override;
  virtual void postVisit(ForeachStmtAST&e) override;

  virtual bool preVisit(IfStmtAST&e) override;
  virtual void postVisit(IfStmtAST&e) override;

};



}

#endif // CONTRA_FUTURES_HPP
