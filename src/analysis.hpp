#ifndef CONTRA_ANALYSIS_HPP
#define CONTRA_ANALYSIS_HPP

#include "config.hpp"
#include "dispatcher.hpp"
#include "symbols.hpp"

#include <iostream>
#include <fstream>

namespace contra {

class Analyzer : public AstDispatcher {

  int Scope_ = 0;
  std::map<std::string, std::shared_ptr<Symbol>> SymbolTable_;
  std::map<std::string, std::shared_ptr<VariableSymbol>> VariableTable_;
  std::map<std::string, std::shared_ptr<FunctionSymbol>> FunctionTable_;

public:


  virtual ~Analyzer() = default;

  std::shared_ptr<FunctionSymbol> getFunction(const std::string &);

  void dispatch(ExprAST&) override;
  void dispatch(ValueExprAST<int_t>&) override;
  void dispatch(ValueExprAST<real_t>&) override;
  void dispatch(ValueExprAST<std::string>&) override;
  void dispatch(VariableExprAST&) override;
  void dispatch(ArrayExprAST&) override;
  void dispatch(BinaryExprAST&) override;
  void dispatch(CallExprAST&) override;
  void dispatch(ForExprAST&) override;
  void dispatch(IfExprAST&) override;
  void dispatch(UnaryExprAST&) override;
  void dispatch(VarExprAST&) override;
  void dispatch(ArrayVarExprAST&) override;
  void dispatch(PrototypeAST&) override;
  void dispatch(FunctionAST&) override;

};



}

#endif // CONTRA_ANALYSIS_HPP
