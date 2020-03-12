#ifndef CONTRA_ANALYSIS_HPP
#define CONTRA_ANALYSIS_HPP

#include "config.hpp"
#include "context.hpp"
#include "dispatcher.hpp"
#include "precedence.hpp"
#include "symbols.hpp"

#include <iostream>
#include <fstream>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
class Analyzer : public AstDispatcher {

  int Scope_ = 0;
  std::map<std::string, std::shared_ptr<Symbol>> SymbolTable_;
  std::map<std::string, std::shared_ptr<VariableDef>> VariableTable_;
  std::map<std::string, std::shared_ptr<FunctionDef>> FunctionTable_;
  
  std::shared_ptr<BinopPrecedence> BinopPrecedence_;

  VariableType I64Type  = Context::I64Symbol;
  VariableType F64Type  = Context::F64Symbol;
  VariableType StrType  = Context::StrSymbol;
  VariableType BoolType = Context::BoolSymbol;
  VariableType VoidType = Context::VoidSymbol;

  VariableType  TypeResult_;

public:

  Analyzer(std::shared_ptr<BinopPrecedence> Prec) : BinopPrecedence_(std::move(Prec))
  {
    SymbolTable_.emplace( Context::I64Symbol->getName(),  Context::I64Symbol);
    SymbolTable_.emplace( Context::F64Symbol->getName(),  Context::F64Symbol);
    SymbolTable_.emplace( Context::StrSymbol->getName(),  Context::StrSymbol);
    SymbolTable_.emplace( Context::BoolSymbol->getName(), Context::BoolSymbol);
    SymbolTable_.emplace( Context::VoidSymbol->getName(), Context::VoidSymbol);
  }

  virtual ~Analyzer() = default;

  std::shared_ptr<FunctionDef> getFunction(const std::string &);
  
  template<
    typename T,
    typename = typename std::enable_if_t<
      std::is_same<T, FunctionAST>::value || std::is_same<T, PrototypeAST>::value >
  >
  void runFuncVisitor(T&e)
  { dispatch(e); }

private:
  
  template<typename T>
  auto runExprVisitor(T&e)
  {
    TypeResult_ = VariableType{};
    dispatch(e);
    return TypeResult_;
  }

  void dispatch(ExprAST&) override;
  void dispatch(ValueExprAST<int_t>&) override;
  void dispatch(ValueExprAST<real_t>&) override;
  void dispatch(ValueExprAST<std::string>&) override;
  void dispatch(VariableExprAST&) override;
  void dispatch(ArrayExprAST&) override;
  void dispatch(UnaryExprAST&) override;
  void dispatch(BinaryExprAST&) override;
  void dispatch(CallExprAST&) override;
  void dispatch(ForExprAST&) override;
  void dispatch(IfExprAST&) override;
  void dispatch(VarExprAST&) override;
  void dispatch(ArrayVarExprAST&) override;
  void dispatch(PrototypeAST&) override;
  void dispatch(FunctionAST&) override;

};



}

#endif // CONTRA_ANALYSIS_HPP
