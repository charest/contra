#ifndef CONTRA_ANALYSIS_HPP
#define CONTRA_ANALYSIS_HPP

#include "ast.hpp"
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
  std::map<std::string, std::shared_ptr<TypeDef>> TypeTable_;
  std::map<std::string, std::shared_ptr<VariableDef>> VariableTable_;
  std::map<std::string, std::shared_ptr<FunctionDef>> FunctionTable_;
  
  std::shared_ptr<BinopPrecedence> BinopPrecedence_;

  VariableType I64Type_  = VariableType(Context::I64Type);
  VariableType F64Type_  = VariableType(Context::F64Type);
  VariableType StrType_  = VariableType(Context::StrType);
  VariableType BoolType_ = VariableType(Context::BoolType);
  VariableType VoidType_ = VariableType(Context::VoidType);


  VariableType  TypeResult_;
  VariableType  DestinationType_;

public:

  Analyzer(std::shared_ptr<BinopPrecedence> Prec) : BinopPrecedence_(std::move(Prec))
  {
    TypeTable_.emplace( Context::I64Type->getName(),  Context::I64Type);
    TypeTable_.emplace( Context::F64Type->getName(),  Context::F64Type);
    TypeTable_.emplace( Context::StrType->getName(),  Context::StrType);
    TypeTable_.emplace( Context::BoolType->getName(), Context::BoolType);
    TypeTable_.emplace( Context::VoidType->getName(), Context::VoidType);
  }

  virtual ~Analyzer() = default;

  template<
    typename T,
    typename = typename std::enable_if_t<
      std::is_same<T, FunctionAST>::value || std::is_same<T, PrototypeAST>::value >
  >
  void runFuncVisitor(T&e)
  {
    Scope_ = 0;
    e.accept(*this);
  }

private:
  
  template<typename T>
  auto runExprVisitor(T&e)
  {
    TypeResult_ = VariableType{};
    e.accept(*this);
    return TypeResult_;
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
  
  auto getBaseType(const std::string & Name, const SourceLocation & Loc) {
    auto it = TypeTable_.find(Name);
    if ( it == TypeTable_.end() )
      THROW_NAME_ERROR("Unknown type specifier '" << Name << "'.", Loc);
    return it->second;
  }
  auto getBaseType(Identifier Id) { return getBaseType(Id.getName(), Id.getLoc()); }

  auto getVariable(const std::string & Name, const SourceLocation & Loc) {
    auto it = VariableTable_.find(Name);
    if (it == VariableTable_.end())
      THROW_NAME_ERROR("Variable '" << Name << "' has not been"
          << " previously defined", Loc);
    return it->second;
  }
  auto getVariable(Identifier Id) { return getVariable(Id.getName(), Id.getLoc()); }

  auto insertVariable(const Identifier & Id, const VariableType & VarType)
  {
    const auto & Name = Id.getName();
    auto Loc = Id.getLoc();
    auto S = std::make_shared<VariableDef>( Name, Loc, VarType);
    auto it = VariableTable_.emplace(Name, std::move(S));
    if (!it.second)
      THROW_NAME_ERROR("Variable '" << Name << "' has been"
          << " previously defined", Loc);
    return it.first->second;
  }

  auto insertFunction(const Identifier & Id, const VariableTypeList & ArgTypes,
      const VariableType & RetType)
  { 
    const auto & Name = Id.getName();
    auto Sy = std::make_shared<UserFunction>(Name, Id.getLoc(), ArgTypes, RetType);
    auto fit = FunctionTable_.emplace( Name, std::move(Sy) );
    if (!fit.second)
      THROW_NAME_ERROR("Prototype already exists for '" << Name << "'.",
        Id.getLoc());
    return fit.first->second;
  }
  
  std::shared_ptr<FunctionDef> getFunction(const std::string &, const SourceLocation &);

  auto getFunction(const Identifier & Id)
  { return getFunction(Id.getName(), Id.getLoc()); }
  
  
  void checkIsCastable(const VariableType & FromType, const VariableType & ToType,
      const SourceLocation & Loc)
  {
    auto IsCastable = FromType.isCastableTo(ToType);
    if (!IsCastable)
      THROW_NAME_ERROR("Cannot cast from type '" << FromType << "' to type '"
          << ToType << "'.", Loc);
  }
    
  void checkIsAssignable(const VariableType & LeftType, const VariableType & RightType,
      const SourceLocation & Loc)
  {
    auto IsAssignable = RightType.isAssignableTo(LeftType);
    if (IsAssignable)
      THROW_NAME_ERROR("A variable of type '" << RightType << "' cannot be"
           << " assigned to a variable of type '" << LeftType << "'." , Loc);
  }

  auto insertCastOp( std::unique_ptr<NodeAST> FromExpr, const VariableType & ToType )
  {
    auto Loc = FromExpr->getLoc();
    auto E = std::make_unique<CastExprAST>(Loc, std::move(FromExpr),
          Identifier(ToType.getBaseType()->getName(), Loc));
    return E;
  }

  VariableType promote(const VariableType & LeftType, const VariableType & RightType,
      const SourceLocation & Loc)
  {
    if (LeftType == RightType) return LeftType;

    if (LeftType.isNumber() && RightType.isNumber()) {
      if (LeftType == F64Type_ || RightType == F64Type_)
        return F64Type_;
      else
        return LeftType;
    }
    
    THROW_NAME_ERROR("No promotion rules between the type '" << LeftType
         << " and the type '" << RightType << "'." , Loc);

    return {};
  }

};



}

#endif // CONTRA_ANALYSIS_HPP
