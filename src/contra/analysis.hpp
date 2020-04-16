#ifndef CONTRA_ANALYSIS_HPP
#define CONTRA_ANALYSIS_HPP

#include "ast.hpp"
#include "config.hpp"
#include "context.hpp"
#include "recursive.hpp"
#include "precedence.hpp"
#include "scope.hpp"
#include "symbols.hpp"

#include <deque>
#include <iostream>
#include <forward_list>
#include <fstream>
#include <set>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// Semantec analyzer class
////////////////////////////////////////////////////////////////////////////////
class Analyzer : public RecursiveAstVisiter, public Scoper {
public:

  using TypeEntry = std::shared_ptr<TypeDef>;
  using FunctionEntry = std::shared_ptr<FunctionDef>;
  using VariableEntry = std::shared_ptr<VariableDef>;

private:

  std::map<std::string, TypeEntry> TypeTable_;
  std::map<std::string, FunctionEntry> FunctionTable_;
  std::set<std::string> TaskTable_; 
  
  std::forward_list< std::map<std::string, VariableEntry> > VariableTable_;
  std::deque< std::set<std::string> > VarAccessTable_;
  
  std::shared_ptr<BinopPrecedence> BinopPrecedence_;

  VariableType I64Type_  = VariableType(Context::I64Type);
  VariableType F64Type_  = VariableType(Context::F64Type);
  VariableType StrType_  = VariableType(Context::StrType);
  VariableType BoolType_ = VariableType(Context::BoolType);
  VariableType VoidType_ = VariableType(Context::VoidType);
  
  bool HaveTopLevelTask_ = false;

  VariableType  TypeResult_;
  VariableType  DestinationType_;
 
  bool IsInsideTask_;

public:

  Analyzer(std::shared_ptr<BinopPrecedence> Prec) : BinopPrecedence_(std::move(Prec))
  {
    TypeTable_.emplace( Context::I64Type->getName(),  Context::I64Type);
    TypeTable_.emplace( Context::F64Type->getName(),  Context::F64Type);
    TypeTable_.emplace( Context::StrType->getName(),  Context::StrType);
    TypeTable_.emplace( Context::BoolType->getName(), Context::BoolType);
    TypeTable_.emplace( Context::VoidType->getName(), Context::VoidType);
    VariableTable_.push_front({}); // global table
    VarAccessTable_.push_front({});
  }

  virtual ~Analyzer() = default;

  // visitor interface
  void runFuncVisitor(FunctionAST&e)
  {
    IsInsideTask_ = false;
    e.accept(*this);
  }

private:
  
  void runProtoVisitor(PrototypeAST&e)
  { e.accept(*this); }

  auto runExprVisitor(NodeAST &e)
  {
    TypeResult_ = VariableType{};
    e.accept(*this);
    return TypeResult_;
  }

  auto runStmtVisitor(NodeAST &e)
  {
    DestinationType_ = VariableType{};
    auto OrigScope = getScope();
    auto TypeResult = runExprVisitor(e);
    resetScope(OrigScope);
    return TypeResult;
  }


  void visit(ValueExprAST&) override;
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
    int Scope = 0;
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

  // Scope interface
  Scoper::value_type createScope() override {
    VariableTable_.push_front({});
    VarAccessTable_.push_front({});
    return Scoper::createScope();
  }
  
  void resetScope(Scoper::value_type Scope) override {
    for (int i=Scope; i<getScope(); ++i) {
      VariableTable_.pop_front();
      VarAccessTable_.pop_front();
    }
    Scoper::resetScope(Scope);
  }


};



}

#endif // CONTRA_ANALYSIS_HPP
