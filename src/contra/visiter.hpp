#ifndef CONTRA_VISITER_HPP
#define CONTRA_VISITER_HPP

#include "config.hpp"

#include <string>

namespace contra {

template<typename T> class ValueExprAST;

class VarAccessExprAST;
class ArrayAccessExprAST;
class ArrayExprAST;
class CastExprAST;
class UnaryExprAST;
class BinaryExprAST;
class CallExprAST;
class IfStmtAST;
class ForStmtAST;
class ForeachStmtAST;
class VarDeclAST;
class ArrayDeclAST;
class PrototypeAST;
class FunctionAST;
class TaskAST;
class IndexTaskAST;

class AstVisiter {
public:

  virtual ~AstVisiter() = default;
  
  virtual void visit(ValueExprAST<int_t>&) = 0;
  virtual void visit(ValueExprAST<real_t>&) = 0;
  virtual void visit(ValueExprAST<std::string>&) = 0;
  virtual void visit(VarAccessExprAST&) = 0;
  virtual void visit(ArrayAccessExprAST&) = 0;
  virtual void visit(ArrayExprAST&) = 0;
  virtual void visit(CastExprAST&) = 0;
  virtual void visit(UnaryExprAST&) = 0;
  virtual void visit(BinaryExprAST&) = 0;
  virtual void visit(CallExprAST&) = 0;

  virtual void visit(IfStmtAST&) = 0;
  virtual void visit(ForStmtAST&) = 0;
  virtual void visit(ForeachStmtAST&) = 0;

  virtual void visit(VarDeclAST&) = 0;
  virtual void visit(ArrayDeclAST&) = 0;
  virtual void visit(PrototypeAST&) = 0;
  
  virtual void visit(FunctionAST&) = 0;
  virtual void visit(TaskAST&) = 0;
  virtual void visit(IndexTaskAST&) = 0;
};

}

#endif // CONTRA_VISITOR_HPP
