#ifndef CONTRA_VISITER_HPP
#define CONTRA_VISITER_HPP

#include "config.hpp"

#include <string>

namespace contra {

class ExprListAST;
class ValueExprAST;
class VarAccessExprAST;
class ArrayAccessExprAST;
class ArrayExprAST;
class RangeExprAST;
class CastExprAST;
class UnaryExprAST;
class BinaryExprAST;
class CallExprAST;
class IfStmtAST;
class ForStmtAST;
class ForeachStmtAST;
class AssignStmtAST;
class PartitionStmtAST;
class VarDeclAST;
class FieldDeclAST;
class PrototypeAST;
class FunctionAST;
class TaskAST;
class IndexTaskAST;

class AstVisiter {
public:

  virtual ~AstVisiter() = default;
  
  virtual void visit(ExprListAST&) = 0;
  virtual void visit(ValueExprAST&) = 0;
  virtual void visit(VarAccessExprAST&) = 0;
  virtual void visit(ArrayAccessExprAST&) = 0;
  virtual void visit(ArrayExprAST&) = 0;
  virtual void visit(RangeExprAST&) = 0;
  virtual void visit(CastExprAST&) = 0;
  virtual void visit(UnaryExprAST&) = 0;
  virtual void visit(BinaryExprAST&) = 0;
  virtual void visit(CallExprAST&) = 0;

  virtual void visit(IfStmtAST&) = 0;
  virtual void visit(ForStmtAST&) = 0;
  virtual void visit(ForeachStmtAST&) = 0;
  virtual void visit(AssignStmtAST&) = 0;
  virtual void visit(PartitionStmtAST&) = 0;

  //virtual void visit(VarDeclAST&) = 0;
  //virtual void visit(FieldDeclAST&) = 0;
  virtual void visit(PrototypeAST&) = 0;
  
  virtual void visit(FunctionAST&) = 0;
  virtual void visit(TaskAST&) = 0;
  virtual void visit(IndexTaskAST&) = 0;
};

}

#endif // CONTRA_VISITOR_HPP
