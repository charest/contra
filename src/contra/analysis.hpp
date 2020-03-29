#ifndef CONTRA_ANALYSIS_HPP
#define CONTRA_ANALYSIS_HPP

#include "ast.hpp"
#include "config.hpp"
#include "context.hpp"
#include "dispatcher.hpp"
#include "precedence.hpp"
#include "scope.hpp"
#include "symbols.hpp"

#include <iostream>
#include <forward_list>
#include <fstream>
#include <set>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// Semantec analyzer class
////////////////////////////////////////////////////////////////////////////////
class Analyzer : public AstDispatcher, public Scoper {
public:

  using TypeEntry = std::shared_ptr<TypeDef>;
  using FunctionEntry = std::shared_ptr<FunctionDef>;
  using VariableEntry = std::shared_ptr<VariableDef>;

private:

  std::map<std::string, TypeEntry> TypeTable_;
  std::map<std::string, FunctionEntry> FunctionTable_;
  std::set<std::string> TaskTable_; 
  
  std::forward_list< std::map<std::string, VariableEntry> > VariableTable_;
  
  std::shared_ptr<BinopPrecedence> BinopPrecedence_;

  VariableType I64Type_  = VariableType(Context::I64Type);
  VariableType F64Type_  = VariableType(Context::F64Type);
  VariableType StrType_  = VariableType(Context::StrType);
  VariableType BoolType_ = VariableType(Context::BoolType);
  VariableType VoidType_ = VariableType(Context::VoidType);

  VariableType  TypeResult_;
  VariableType  DestinationType_;

  bool HaveTopLevelTask_ = false;
 
  Scoper::value_type createScope() override {
    VariableTable_.push_front({});
    return Scoper::createScope();
  }
  
  void resetScope(Scoper::value_type Scope) override {
    for (int i=Scope; i<getScope(); ++i)
      VariableTable_.pop_front();
    Scoper::resetScope(Scope);
  }

public:

  Analyzer(std::shared_ptr<BinopPrecedence> Prec) : BinopPrecedence_(std::move(Prec))
  {
    TypeTable_.emplace( Context::I64Type->getName(),  Context::I64Type);
    TypeTable_.emplace( Context::F64Type->getName(),  Context::F64Type);
    TypeTable_.emplace( Context::StrType->getName(),  Context::StrType);
    TypeTable_.emplace( Context::BoolType->getName(), Context::BoolType);
    TypeTable_.emplace( Context::VoidType->getName(), Context::VoidType);
    VariableTable_.push_front({}); // global table
  }

  virtual ~Analyzer() = default;
  
  // visitor interface
  template<
    typename T,
    typename = typename std::enable_if_t<
      std::is_same<T, FunctionAST>::value || std::is_same<T, PrototypeAST>::value >
  >
  void runFuncVisitor(T&e)
  { e.accept(*this); }

private:
  
  template<typename T>
  auto runExprVisitor(T&e)
  {
    TypeResult_ = VariableType{};
    e.accept(*this);
    return TypeResult_;
  }

  template<typename T>
  auto runStmtVisitor(T&e)
  {
    auto OrigScope = getScope();
    auto V = runExprVisitor(e);
    resetScope(OrigScope);
    return V;
  }


  void dispatch(ValueExprAST<int_t>&) override;
  void dispatch(ValueExprAST<real_t>&) override;
  void dispatch(ValueExprAST<std::string>&) override;
  void dispatch(VariableExprAST&) override;
  void dispatch(ArrayExprAST&) override;
  void dispatch(CastExprAST&) override;
  void dispatch(UnaryExprAST&) override;
  void dispatch(BinaryExprAST&) override;
  void dispatch(CallExprAST&) override;

  void dispatch(ForStmtAST&) override;
  void dispatch(IfStmtAST&) override;
  void dispatch(VarDeclAST&) override;
  void dispatch(ArrayDeclAST&) override;
  void dispatch(PrototypeAST&) override;

  void dispatch(FunctionAST&) override;
  
  // base type interface
  TypeEntry getBaseType(const std::string & Name, const SourceLocation & Loc);
  TypeEntry getBaseType(Identifier Id);

  // variable interface
  VariableEntry
    getVariable(const std::string & Name, const SourceLocation & Loc);

  VariableEntry getVariable(Identifier Id);

  VariableEntry
    insertVariable(const Identifier & Id, const VariableType & VarType);
  
  struct VariableTableResult {
    decltype(VariableTable_)::value_type::iterator Result;
    bool IsFound = false;
  };

  VariableTableResult findVariable(const std::string & Name);

  // function interface
  FunctionEntry
    insertFunction(const Identifier & Id, const VariableTypeList & ArgTypes,
      const VariableType & RetType);
  
  FunctionEntry getFunction(const std::string &, const SourceLocation &);

  FunctionEntry getFunction(const Identifier & Id);
 
public:
  void removeFunction(const std::string & Name);

private:

  // type checking interface
  void checkIsCastable(const VariableType & FromType, const VariableType & ToType,
      const SourceLocation & Loc);
    
  void checkIsAssignable(const VariableType & LeftType, const VariableType & RightType,
      const SourceLocation & Loc);

  std::unique_ptr<CastExprAST>
    insertCastOp( std::unique_ptr<NodeAST> FromExpr, const VariableType & ToType );

  VariableType promote(const VariableType & LeftType, const VariableType & RightType,
      const SourceLocation & Loc);
  
  // Task interface
  void insertTask(const std::string & Name)
  { TaskTable_.emplace(Name); }

  bool isTask(const std::string & Name) const
  { return TaskTable_.count(Name); }

};



}

#endif // CONTRA_ANALYSIS_HPP
