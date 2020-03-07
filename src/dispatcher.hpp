#ifndef CONTRA_DISPATCHER_HPP
#define CONTRA_DISPATCHER_HPP

namespace contra {

class VariableExprAST;
class ArrayExprAST;
class BinaryExprAST;
class CallExprAST;
class IfExprAST;
class ForExprAST;
class UnaryExprAST;
class VarExprAST;
class ArrayVarExprAST;
class PrototypeAST;
class FunctionAST;

template<typename T>
class ValueExprAST;

class AbstractDispatcher {
public:

  virtual ~AbstractDispatcher() = default;
  
  virtual void dispatch(ExprAST&) = 0;
  virtual void dispatch(ValueExprAST<int_t>&) = 0;
  virtual void dispatch(ValueExprAST<real_t>&) = 0;
  virtual void dispatch(ValueExprAST<std::string>&) = 0;
  virtual void dispatch(VariableExprAST&) = 0;
  virtual void dispatch(ArrayExprAST&) = 0;
  virtual void dispatch(BinaryExprAST&) = 0;
  virtual void dispatch(CallExprAST&) = 0;
  virtual void dispatch(IfExprAST&) = 0;
  virtual void dispatch(ForExprAST&) = 0;
  virtual void dispatch(UnaryExprAST&) = 0;
  virtual void dispatch(VarExprAST&) = 0;
  virtual void dispatch(ArrayVarExprAST&) = 0;
  virtual void dispatch(PrototypeAST&) = 0;
  virtual void dispatch(FunctionAST&) = 0;
};

}

#endif // CONTRA_DISPATCHER_HPP
