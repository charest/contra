#ifndef CONTRA_FLOW_HPP
#define CONTRA_FLOW_HPP

#include "config.hpp"
#include "visiter.hpp"

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// AST plotting class
////////////////////////////////////////////////////////////////////////////////
class Flow : public AstVisiter {

  bool IsFuture_ = false;

public:

  void visitBlock(const ASTBlock &);
  
  // Codegen function
  template<typename T>
  void runFuncVisitor(T&e)
  { e.accept(*this); }

private:
   
  // Codegen function
  template<typename T>
  void runExprVisitor(T&e)
  { e.accept(*this); }

  // Codegen function
  template<typename T>
  void runStmtVisitor(T&e)
  {
    IsFuture_ = false;
    e.accept(*this);
  }

  void visit(ValueExprAST<int_t>&) override;
  void visit(ValueExprAST<real_t>&) override;
  void visit(ValueExprAST<std::string>&) override;
  void visit(VarAccessExprAST&) override;
  void visit(ArrayAccessExprAST&) override;
  void visit(ArrayExprAST&) override;
  void visit(CastExprAST&) override;
  void visit(UnaryExprAST&) override;
  void visit(BinaryExprAST&) override;
  void visit(CallExprAST&) override;
  void visit(ForStmtAST&) override;
  void visit(ForeachStmtAST&) override;
  void visit(IfStmtAST&) override;
  void visit(AssignStmtAST&) override;
  void visit(VarDeclAST&) override;
  void visit(PrototypeAST&) override;
  void visit(FunctionAST&) override;
  void visit(TaskAST&) override;
  void visit(IndexTaskAST&) override;
  
};



}

#endif // CONTRA_FLOW_HPP
