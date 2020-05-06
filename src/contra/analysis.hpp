#ifndef CONTRA_ANALYSIS_HPP
#define CONTRA_ANALYSIS_HPP

#include "ast.hpp"
#include "config.hpp"
#include "context.hpp"
#include "recursive.hpp"
#include "precedence.hpp"
#include "symbols.hpp"

#include <deque>
#include <iostream>
#include <forward_list>
#include <fstream>
#include <set>
#include <memory>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
/// Semantec analyzer class
////////////////////////////////////////////////////////////////////////////////
class Analyzer : public RecursiveAstVisiter {
public:

private:

  std::shared_ptr<BinopPrecedence> BinopPrecedence_;

  VariableType I64Type_  = VariableType(Context::instance().getInt64Type());
  VariableType F64Type_  = VariableType(Context::instance().getFloat64Type());
  VariableType StrType_  = VariableType(Context::instance().getStringType());
  VariableType BoolType_ = VariableType(Context::instance().getBoolType());
  VariableType VoidType_ = VariableType(Context::instance().getVoidType());

  VariableType RangeType_ = VariableType(Context::instance().getInt64Type(),
      VariableType::Attr::Range);
  
  bool HaveTopLevelTask_ = false;
  bool TrackVariableAccess_ = true;

  VariableType  TypeResult_;
  VariableType  DestinationType_;
  FunctionDef*  ParentFunction_;

public:

  Analyzer(std::shared_ptr<BinopPrecedence> Prec);

  virtual ~Analyzer() = default;

  // visitor interface
  void runFuncVisitor(FunctionAST&e)
  {
    ParentFunction_ = nullptr;
    e.accept(*this);
  }

private:
  
  void runProtoVisitor(PrototypeAST&e)
  { e.accept(*this); }

  auto runExprVisitor(NodeAST &e)
  {
    e.setParentFunctionDef(ParentFunction_);
    TypeResult_ = VariableType{};
    e.accept(*this);
    return TypeResult_;
  }

  auto runStmtVisitor(NodeAST &e)
  {
    DestinationType_ = VariableType{};
    auto TypeResult = runExprVisitor(e);
    return TypeResult;
  }

  void visitFor(ForStmtAST&);


  void visit(ValueExprAST&) override;
  void visit(VarAccessExprAST&) override;
  void visit(ArrayAccessExprAST&) override;
  void visit(ArrayExprAST&) override;
  void visit(RangeExprAST&) override;
  void visit(CastExprAST&) override;
  void visit(UnaryExprAST&) override;
  void visit(BinaryExprAST&) override;
  void visit(CallExprAST&) override;

  void visit(ForStmtAST&) override;
  void visit(ForeachStmtAST&) override;
  void visit(IfStmtAST&) override;
  void visit(AssignStmtAST&) override;
  void visit(PartitionStmtAST&) override;

  void visit(VarDeclAST&) override;
  void visit(FieldDeclAST&) override;
  void visit(PrototypeAST&) override;

  void visit(FunctionAST&) override;
  void visit(TaskAST&) override;
  void visit(IndexTaskAST&) override;
  
  // base type interface
  TypeDef* getType(const std::string & Name, const SourceLocation & Loc);
  TypeDef* getType(const Identifier & Id);

  // variable interface
  VariableDef* getVariable(
      const std::string & Name,
      const SourceLocation & Loc);
  VariableDef* getVariable(const Identifier & Id);
  VariableDef* insertVariable(const Identifier & Id, const VariableType & VarType);

  // function interface
  FunctionDef* insertFunction(
      const Identifier & Id,
      const VariableTypeList & ArgTypes,
      const VariableType & RetType);
  
  FunctionDef* getFunction(const std::string &, const SourceLocation &);
  FunctionDef* getFunction(const Identifier & Id);
 
public:
  void removeFunction(const std::string & Name);

private:

  // type checking interface
  void checkIsCastable(
      const VariableType & FromType,
      const VariableType & ToType,
      const SourceLocation & Loc);
    
  void checkIsAssignable(
      const VariableType & LeftType,
      const VariableType & RightType,
      const SourceLocation & Loc);

  std::unique_ptr<CastExprAST> insertCastOp(
      std::unique_ptr<NodeAST> FromExpr,
      const VariableType & ToType );

  VariableType promote(
      const VariableType & LeftType,
      const VariableType & RightType,
      const SourceLocation & Loc);
  

  // Scope interface
  void createScope() const
  { Context::instance().createScope(); }

  void popScope() const
  { Context::instance().popScope(); }

  bool isGlobalScope() const
  { return Context::instance().isGlobalScope(); }

};



}

#endif // CONTRA_ANALYSIS_HPP
