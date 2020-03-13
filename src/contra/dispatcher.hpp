#ifndef CONTRA_DISPATCHER_HPP
#define CONTRA_DISPATCHER_HPP

#include "config.hpp"

namespace contra {

template<typename T> class ValueExprAST;

class VariableExprAST;
class ArrayExprAST;
class CastExprAST;
class UnaryExprAST;
class BinaryExprAST;
class CallExprAST;
class IfStmtAST;
class ForStmtAST;
class VarDeclAST;
class ArrayDeclAST;
class PrototypeAST;
class FunctionAST;

class AstDispatcher {
public:

  virtual ~AstDispatcher() = default;
  
  virtual void dispatch(ValueExprAST<int_t>&) = 0;
  virtual void dispatch(ValueExprAST<real_t>&) = 0;
  virtual void dispatch(ValueExprAST<std::string>&) = 0;
  virtual void dispatch(VariableExprAST&) = 0;
  virtual void dispatch(ArrayExprAST&) = 0;
  virtual void dispatch(CastExprAST&) = 0;
  virtual void dispatch(UnaryExprAST&) = 0;
  virtual void dispatch(BinaryExprAST&) = 0;
  virtual void dispatch(CallExprAST&) = 0;

  virtual void dispatch(IfStmtAST&) = 0;
  virtual void dispatch(ForStmtAST&) = 0;

  virtual void dispatch(VarDeclAST&) = 0;
  virtual void dispatch(ArrayDeclAST&) = 0;
  virtual void dispatch(PrototypeAST&) = 0;
  
  virtual void dispatch(FunctionAST&) = 0;
};

}

#endif // CONTRA_DISPATCHER_HPP
