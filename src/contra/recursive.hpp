#ifndef CONTRA_RECURSIVE_HPP
#define CONTRA_RECURSIVE_HPP

#include "ast.hpp"
#include "visiter.hpp"

namespace contra {

class RecursiveAstVisiter: public AstVisiter
{

  void visitBlock( const ASTBlock & Stmts ){
    for ( const auto & Stmt : Stmts ) Stmt->accept(*this);
  }

public:

  virtual bool preVisit(ValueExprAST&) { return false; }
  virtual void postVisit(ValueExprAST&) {}
  virtual void visit(ValueExprAST&e) {
    if (preVisit(e)) { return; }
    postVisit(e);
  }

  virtual bool preVisit(VarAccessExprAST&) { return false; }
  virtual void postVisit(VarAccessExprAST&) {}
  virtual void visit(VarAccessExprAST&e) {
    if (preVisit(e)) { return; }
    postVisit(e);
  }

  virtual bool preVisit(ArrayAccessExprAST&) { return false; }
  virtual void postVisit(ArrayAccessExprAST&) {}
  virtual void visit(ArrayAccessExprAST&e) {
    if (preVisit(e)) { return; }
    e.getIndexExpr()->accept(*this);
    postVisit(e);
  }
  
  
  virtual bool preVisit(ArrayExprAST&) { return false; }
  virtual void postVisit(ArrayExprAST&) {}
  virtual void visit(ArrayExprAST&e) {
    if (preVisit(e)) { return; }
    visitBlock(e.getValExprs());
    if (e.hasSize()) e.getSizeExpr()->accept(*this);
    postVisit(e);
  }
  
  virtual bool preVisit(RangeExprAST&) { return false; }
  virtual void postVisit(RangeExprAST&) {}
  virtual void visit(RangeExprAST&e) {
    if (preVisit(e)) { return; }
    e.getStartExpr()->accept(*this);
    e.getEndExpr()->accept(*this);
    postVisit(e);
  }
  
  virtual bool preVisit(FieldDeclExprAST&) { return false; }
  virtual void postVisit(FieldDeclExprAST&) {}
  virtual void visit(FieldDeclExprAST&e) {
    if (preVisit(e)) { return; }
    e.getIndexExpr()->accept(*this);
    postVisit(e);
  }
  
  virtual bool preVisit(CastExprAST&) { return false; }
  virtual void postVisit(CastExprAST&) {}
  virtual void visit(CastExprAST&e) {
    if (preVisit(e)) { return; }
    e.getFromExpr()->accept(*this);
    postVisit(e);
  }
  
  virtual bool preVisit(UnaryExprAST&) { return false; }
  virtual void postVisit(UnaryExprAST&) {}
  virtual void visit(UnaryExprAST&e) {
    if (preVisit(e)) { return; }
    e.getOpExpr()->accept(*this);
    postVisit(e);
  }
  
  virtual bool preVisit(BinaryExprAST&) { return false; }
  virtual void postVisit(BinaryExprAST&) {}
  virtual void visit(BinaryExprAST&e) {
    if (preVisit(e)) { return; }
    e.getLeftExpr()->accept(*this);
    e.getRightExpr()->accept(*this);
    postVisit(e);
  }  
  
  virtual bool preVisit(CallExprAST&) { return false; }
  virtual void postVisit(CallExprAST&) {}
  virtual void visit(CallExprAST&e) {
    if (preVisit(e)) { return; }
    visitBlock(e.getArgExprs());
    postVisit(e);
  }
  
  virtual bool preVisit(ExprListAST&) { return false; }
  virtual void postVisit(ExprListAST&) {}
  virtual void visit(ExprListAST&e) {
    if (preVisit(e)) { return; }
    for (const auto & Expr : e.getExprs()) Expr->accept(*this);
    postVisit(e);
  }

  virtual bool preVisit(IfStmtAST&) { return false; }
  virtual void postVisit(IfStmtAST&) {}
  virtual void visit(IfStmtAST&e) {
    if (preVisit(e)) { return; }
    e.getCondExpr()->accept(*this);
    visitBlock(e.getThenExprs());
    visitBlock(e.getElseExprs());
    postVisit(e);
  }
  
  virtual bool preVisit(ForStmtAST&) { return false; }
  virtual void postVisit(ForStmtAST&) {}
  virtual void visit(ForStmtAST&e) {
    if (preVisit(e)) { return; }
    e.getStartExpr()->accept(*this);
    visitBlock(e.getBodyExprs());
    postVisit(e);
  }
  
  virtual bool preVisit(ForeachStmtAST&) { return false; }
  virtual void postVisit(ForeachStmtAST&) {}
  virtual void visit(ForeachStmtAST&e) {
    if (preVisit(e)) { return; }
    e.getStartExpr()->accept(*this);
    visitBlock(e.getBodyExprs());
    postVisit(e);
  }
  
  virtual bool preVisit(AssignStmtAST&) { return false; }
  virtual void postVisit(AssignStmtAST&) {}
  virtual void visit(AssignStmtAST&e) {
    if (preVisit(e)) { return; }
    visitBlock(e.getLeftExprs());
    visitBlock(e.getRightExprs());
    postVisit(e);
  }
  
  virtual bool preVisit(PartitionStmtAST&) { return false; }
  virtual void postVisit(PartitionStmtAST&) {}
  virtual void visit(PartitionStmtAST&e) {
    if (preVisit(e)) { return; }
    e.getPartExpr()->accept(*this);
    postVisit(e);
  }
  
  virtual bool preVisit(PrototypeAST&) { return false; }
  virtual void postVisit(PrototypeAST&) {}
  virtual void visit(PrototypeAST&e) {
    if (preVisit(e)) { return; }
    postVisit(e);
  }
    
  virtual bool preVisit(FunctionAST&) { return false; }
  virtual void postVisit(FunctionAST&) {}
  virtual void visit(FunctionAST&e) {
    if (preVisit(e)) { return; }
    visitBlock(e.getBodyExprs());
    if (e.hasReturn()) e.getReturnExpr()->accept(*this);
    postVisit(e);
  }
    
  virtual bool preVisit(TaskAST&) { return false; }
  virtual void postVisit(TaskAST&) {}
  virtual void visit(TaskAST&e) {
    if (preVisit(e)) { return; }
    visitBlock(e.getBodyExprs());
    if (e.hasReturn()) e.getReturnExpr()->accept(*this);
    postVisit(e);
  }
  
  virtual bool preVisit(IndexTaskAST&) { return false; }
  virtual void postVisit(IndexTaskAST&) {}
  virtual void visit(IndexTaskAST&e) {
    if (preVisit(e)) { return; }
    visitBlock(e.getBodyExprs());
    if (e.hasReturn()) e.getReturnExpr()->accept(*this);
    postVisit(e);
  } 
};

}

#endif // CONTRA_RECURSIVE_HPP
