#include "ast.hpp"
#include "analysis.hpp"
#include "token.hpp"

#include "librt/librt.hpp"

#include <string>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Base type interface
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
Analyzer::TypeEntry
  Analyzer::getBaseType(const std::string & Name, const SourceLocation & Loc)
{
  auto it = TypeTable_.find(Name);
  if ( it == TypeTable_.end() )
    THROW_NAME_ERROR("Unknown type specifier '" << Name << "'.", Loc);
  return it->second;
}

//==============================================================================
Analyzer::TypeEntry Analyzer::getBaseType(Identifier Id)
{ return getBaseType(Id.getName(), Id.getLoc()); }

  
////////////////////////////////////////////////////////////////////////////////
// Function routines
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
void Analyzer::removeFunction(const std::string & Name)
{ FunctionTable_.erase(Name); }

//==============================================================================
Analyzer::FunctionEntry Analyzer::getFunction(const std::string & Name,
    const SourceLocation & Loc) {
  
  // If not, check whether we can codegen the declaration from some existing
  // prototype.
  auto FP = FunctionTable_.find(Name);
  if (FP != FunctionTable_.end()) 
    return FP->second;
  
  // see if this is an available intrinsic, try installing it first
  if (auto F = librt::RunTimeLib::tryInstall(Name))
    return F;
  
  THROW_NAME_ERROR("No valid prototype for '" << Name << "'.", Loc);

  // if found it, make sure its not a variable in scope
  return nullptr;
}

//==============================================================================
Analyzer::FunctionEntry Analyzer::getFunction(const Identifier & Id)
{ return getFunction(Id.getName(), Id.getLoc()); }
  
//==============================================================================
Analyzer::FunctionEntry
Analyzer::insertFunction(
    const Identifier & Id,
    const VariableTypeList & ArgTypes,
    const VariableType & RetType)
{ 
  const auto & Name = Id.getName();
  auto Sy = std::make_shared<UserFunction>(Name, Id.getLoc(), RetType, ArgTypes);
  auto fit = FunctionTable_.emplace( Name, std::move(Sy) );
  if (!fit.second)
    THROW_NAME_ERROR("Prototype already exists for '" << Name << "'.",
      Id.getLoc());
  return fit.first->second;
}

////////////////////////////////////////////////////////////////////////////////
// Variable interface
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
Analyzer::VariableEntry
Analyzer::getVariable(const std::string & Name, const SourceLocation & Loc)
{
  for ( const auto & ST : VariableTable_ ) {
    auto it = ST.find(Name);
    if (it != ST.end()) return it->second;
  }
  THROW_NAME_ERROR("Variable '" << Name << "' has not been"
     << " previously defined", Loc);
  return {};
}

//==============================================================================
Analyzer::VariableEntry Analyzer::getVariable(Identifier Id)
{ return getVariable(Id.getName(), Id.getLoc()); }

//==============================================================================
Analyzer::VariableEntry
Analyzer::insertVariable(const Identifier & Id, const VariableType & VarType)
{
  const auto & Name = Id.getName();
  const auto & Loc = Id.getLoc();
  auto S = std::make_shared<VariableDef>(Name, Loc, VarType);
  auto it = VariableTable_.front().emplace(Name, std::move(S));
  if (!it.second)
    THROW_NAME_ERROR("Variable '" << Name << "' has been"
        << " previously defined", Loc);
  return it.first->second;
}

////////////////////////////////////////////////////////////////////////////////
// type checking interface
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
void Analyzer::checkIsCastable(
    const VariableType & FromType,
    const VariableType & ToType,
    const SourceLocation & Loc)
{
  auto IsCastable = FromType.isCastableTo(ToType);
  if (!IsCastable)
    THROW_NAME_ERROR("Cannot cast from type '" << FromType << "' to type '"
        << ToType << "'.", Loc);
}
  
//==============================================================================
void Analyzer::checkIsAssignable(
    const VariableType & LeftType,
    const VariableType & RightType,
    const SourceLocation & Loc)
{
  auto IsAssignable = RightType.isAssignableTo(LeftType);
  if (!IsAssignable)
    THROW_NAME_ERROR("A variable of type '" << RightType << "' cannot be"
         << " assigned to a variable of type '" << LeftType << "'." , Loc);
}

//==============================================================================
std::unique_ptr<CastExprAST>
Analyzer::insertCastOp(
    std::unique_ptr<NodeAST> FromExpr,
    const VariableType & ToType )
{
  auto Loc = FromExpr->getLoc();
  auto E = std::make_unique<CastExprAST>(Loc, std::move(FromExpr), ToType);
  return E;
}

//==============================================================================
VariableType
Analyzer::promote(
    const VariableType & LeftType,
    const VariableType & RightType,
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

////////////////////////////////////////////////////////////////////////////////
// Visitors
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
void Analyzer::dispatch(ValueExprAST<int_t>& e)
{
  TypeResult_ = I64Type_;
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::dispatch(ValueExprAST<real_t>& e)
{
  TypeResult_ = F64Type_;
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::dispatch(ValueExprAST<std::string>& e)
{
  TypeResult_ = StrType_;
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::dispatch(VariableExprAST& e)
{
  const auto & Name = e.getName();
  auto Var = getVariable(Name, e.getLoc());
  auto VarType = Var->getType();

  // array index
  if (e.IndexExpr_) {
    auto Loc = e.IndexExpr_->getLoc();
    
    if (!VarType.isArray())
      THROW_NAME_ERROR( "Cannot index scalar using '[]' operator", Loc);
    
    auto IndexType = runExprVisitor(*e.IndexExpr_);
    if (IndexType != I64Type_)
      THROW_NAME_ERROR( "Array index for variable '" << Name << "' must "
          << "evaluate to an integer.", Loc );

    VarType.setArray(false); // revert to scalar
  }

  // result
  TypeResult_ = VarType;
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::dispatch(ArrayExprAST& e)
{
  if (e.SizeExpr_) {
    auto SizeType = runExprVisitor(*e.SizeExpr_);
    if (SizeType != I64Type_)
      THROW_NAME_ERROR( "Size expression for arrays must be an integer.",
          e.SizeExpr_->getLoc());
  }

  int NumVals = e.ValExprs_.size();
  
  VariableTypeList ValTypes;
  ValTypes.reserve(NumVals);
  
  VariableType CommonType;

  for (int i=0; i<NumVals; ++i) {
    auto & ValExpr = *e.ValExprs_[i];
    auto ValType = runExprVisitor(ValExpr);
    if (i==0) CommonType = ValType;
    else      CommonType = promote(ValType, CommonType, ValExpr.getLoc());
    ValTypes.emplace_back(ValType);
  }

  if (DestinationType_) {
    CommonType = DestinationType_;
    CommonType.setArray(false);
  }

  for (int i=0; i<NumVals; ++i) {
    const auto & ValType = ValTypes[i];
    if (CommonType != ValType) {
      auto Loc = e.ValExprs_[i]->getLoc();
      checkIsCastable(ValType, CommonType, Loc);
      e.ValExprs_[i] = insertCastOp(std::move(e.ValExprs_[i]), CommonType);
    }
  }

  CommonType.setArray();
  TypeResult_ = CommonType;
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::dispatch(CastExprAST& e)
{
  auto FromType = runExprVisitor(*e.FromExpr_);
  auto TypeId = e.TypeId_;
  auto ToType = VariableType(getBaseType(TypeId));
  checkIsCastable(FromType, ToType, e.getLoc());
  TypeResult_ = VariableType(ToType);
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::dispatch(UnaryExprAST& e)
{
  auto OpCode = e.OpCode_;
  auto OpType = runExprVisitor(*e.OpExpr_);
  auto Loc = e.getLoc();

  if (OpType.isArray())
      THROW_NAME_ERROR( "Unary operation '" << OpCode << "' "
          << "not allowed for array expressions.", Loc );

  if (!OpType.isNumber())
      THROW_NAME_ERROR( "Unary operators only allowed for scalar numeric "
          << "expressions. Expression is of type '" << OpType << "'.", Loc );


  switch (OpCode) {
  default:
    THROW_NAME_ERROR( "Unknown unary operator '" << OpCode << "'", Loc);
  case tok_sub:
  case tok_add:
    TypeResult_ = OpType;
  };
  
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::dispatch(BinaryExprAST& e)
{
  auto Loc = e.getLoc();
  auto OpCode = e.OpCode_;

  auto & RightExpr = *e.RightExpr_;
  auto & LeftExpr = *e.LeftExpr_;
  
  auto RightLoc = RightExpr.getLoc();
  auto LeftLoc = LeftExpr.getLoc();
  
  auto RightType = runExprVisitor(RightExpr);
  auto LeftType = runExprVisitor(LeftExpr);

  if (OpCode == tok_asgmt) {
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    auto LHSE = dynamic_cast<VariableExprAST*>(e.LeftExpr_.get());
    if (!LHSE)
      THROW_NAME_ERROR("destination of '=' must be a variable", LeftLoc);

    auto Name = LHSE->getName();
    auto Var = getVariable(Name, LeftLoc);
   
    checkIsAssignable( LeftType, RightType, Loc );

    if (RightType.getBaseType() != LeftType.getBaseType()) {
      checkIsCastable(RightType, LeftType, Loc);
      e.RightExpr_ = insertCastOp(std::move(e.RightExpr_), LeftType);
      RightExpr = *e.RightExpr_;
    }
    
    TypeResult_ = LeftType;
    e.setType(TypeResult_);

    return;
  }
 
  if ( !LeftType.isNumber() || !RightType.isNumber())
      THROW_NAME_ERROR( "Binary operators only allowed for scalar numeric "
          << "expressions.", Loc );
  
  auto CommonType = LeftType;
  if (RightType != LeftType) {
    checkIsCastable(RightType, LeftType, RightLoc);
    checkIsCastable(LeftType, RightType, LeftLoc);
    CommonType = promote(LeftType, RightType, Loc);
    if (RightType != CommonType)
      e.RightExpr_ = insertCastOp(std::move(e.RightExpr_), CommonType );
    else
      e.LeftExpr_ = insertCastOp(std::move(e.LeftExpr_), CommonType );
  }

  switch (OpCode) {
  case tok_add:
  case tok_sub:
  case tok_mul:
  case tok_div:
  case tok_mod:
    TypeResult_ = CommonType;
    e.setType(TypeResult_);
    return;
  case tok_eq:
  case tok_ne:
  case tok_lt:
  case tok_le:
  case tok_gt:
  case tok_ge:
    TypeResult_ = BoolType_;
    e.setType(TypeResult_);
    return;
  } 
  
  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  auto F = getFunction(std::string("binary") + OpCode, Loc);
  TypeResult_ = F->getReturnType();
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::dispatch(CallExprAST& e)
{
  auto FunRes = getFunction(e.Callee_, e.getLoc());

  int NumArgs = e.ArgExprs_.size();
  int NumFixedArgs = FunRes->getNumArgs();

  if (FunRes->isVarArg()) {
    if (NumArgs < NumFixedArgs)
      THROW_NAME_ERROR("Variadic function '" << e.Callee_
          << "', must have at least " << NumFixedArgs << " arguments, but only "
          << NumArgs << " provided.", e.getLoc());
  }
  else {
    if (NumFixedArgs != NumArgs)
      THROW_NAME_ERROR("Incorrect number of arguments specified for '" << e.Callee_
          << "', " << NumArgs << " provided but expected " << NumFixedArgs, e.getLoc());
  }

  for (int i=0; i<NumFixedArgs; ++i) {
    auto ArgType = runExprVisitor(*e.ArgExprs_[i]);
    auto ParamType = FunRes->getArgType(i);
    if (ArgType != ParamType) {
      checkIsCastable(ArgType, ParamType, e.ArgExprs_[i]->getLoc());
      e.ArgExprs_[i] = insertCastOp( std::move(e.ArgExprs_[i]), ParamType);
    }
  }

  for (int i=NumFixedArgs; i<NumArgs; ++i)
    auto ArgType = runExprVisitor(*e.ArgExprs_[i]);

  TypeResult_ = FunRes->getReturnType(); 
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::dispatch(ForStmtAST& e)
{
  auto VarId = e.VarId_;
  
  auto OldScope = Scope_;
  auto InnerScope = createScope();

  auto LoopVar = insertVariable(VarId, I64Type_);

  auto StartType = runStmtVisitor(*e.StartExpr_, InnerScope);
  if (StartType != I64Type_ )
    THROW_NAME_ERROR( "For loop start expression must result in an integer type.",
        e.StartExpr_->getLoc() );

  auto EndType = runStmtVisitor(*e.EndExpr_, InnerScope);
  if (EndType != I64Type_ )
    THROW_NAME_ERROR( "For loop end expression must result in an integer type.",
        e.EndExpr_->getLoc() );

  if (e.StepExpr_) {
    auto StepType = runStmtVisitor(*e.StepExpr_, InnerScope);
    if (StepType != I64Type_ )
      THROW_NAME_ERROR( "For loop step expression must result in an integer type.",
          e.StepExpr_->getLoc() );
  }

  for ( auto & stmt : e.BodyExprs_ ) runStmtVisitor(*stmt, InnerScope);

  resetScope(OldScope);
  TypeResult_ = VoidType_;
}

//==============================================================================
void Analyzer::dispatch(IfStmtAST& e)
{
  auto & CondExpr = *e.CondExpr_;
  auto CondType = runExprVisitor(CondExpr);
  if (CondType != BoolType_ )
    THROW_NAME_ERROR( "If condition must result in boolean type.", CondExpr.getLoc() );

  auto OldScope = Scope_;
  auto InnerScope = createScope();
  for ( auto & stmt : e.ThenExpr_ ) runStmtVisitor(*stmt, InnerScope);
  
  resetScope(OldScope);
  InnerScope = createScope();
  for ( auto & stmt : e.ElseExpr_ ) runStmtVisitor(*stmt, InnerScope);
  
  resetScope(OldScope);
  TypeResult_ = VoidType_;
}

//==============================================================================
void Analyzer::dispatch(VarDeclAST& e)
{
  // check if there is a specified type, if there is, get it
  auto TypeId = e.TypeId_;
  VariableType VarType;
  if (TypeId) {
    VarType = VariableType(getBaseType(TypeId), e.isArray());
    DestinationType_ = VarType;
  }
  
  auto InitType = runExprVisitor(*e.InitExpr_);
  if (!VarType) VarType = InitType;

  if (VarType != InitType) {
    checkIsCastable(InitType, VarType, e.InitExpr_->getLoc());
    e.InitExpr_ = insertCastOp(std::move(e.InitExpr_), VarType);
  }

  int NumVars = e.VarIds_.size();
  for (int i=0; i<NumVars; ++i) {
    auto VarId = e.VarIds_[i];
    insertVariable(VarId, VarType);
  }

  TypeResult_ = VarType;
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::dispatch(ArrayDeclAST& e)
{

  // check if there is a specified type, if there is, get it
  auto TypeId = e.TypeId_;
  VariableType VarType;
  if (TypeId) {
    VarType = VariableType(getBaseType(TypeId), e.isArray());
    DestinationType_ = VarType;
  }
  
  auto InitType = runExprVisitor(*e.InitExpr_);
  if (!VarType) VarType = InitType;

  //----------------------------------------------------------------------------
  // Array already on right hand side
  if (InitType.isArray()) {
  }
  //----------------------------------------------------------------------------
  //  scalar on right hand side
  else {
  
    auto ElementType = VariableType(VarType, false);
    if (ElementType != InitType) {
      checkIsCastable(InitType, ElementType, e.InitExpr_->getLoc());
      e.InitExpr_ = insertCastOp(std::move(e.InitExpr_), ElementType);
    }
 
    if (e.SizeExpr_) {
      auto SizeType = runExprVisitor(*e.SizeExpr_);
      if (SizeType != I64Type_)
        THROW_NAME_ERROR( "Size expression for arrays must be an integer.",
           e.SizeExpr_->getLoc());
    }

  }
  //----------------------------------------------------------------------------

  int NumVars = e.VarIds_.size();
  for (int i=0; i<NumVars; ++i) {
    auto VarId = e.VarIds_[i];
    insertVariable(VarId, VarType);
  }

  TypeResult_ = VarType;
  e.setType(TypeResult_);


}

//==============================================================================
void Analyzer::dispatch(PrototypeAST& e)
{
  int NumArgs = e.ArgIds_.size();

  auto & ArgTypes = e.ArgTypes_;
  ArgTypes.reserve( NumArgs );
  
  for (int i=0; i<NumArgs; ++i) {
    // check type specifier
    const auto & TypeId = e.ArgTypeIds_[i];
    auto ArgType = VariableType( getBaseType(TypeId), e.ArgIsArray_[i] );
    ArgTypes.emplace_back(std::move(ArgType));
  }

  auto & RetType = e.ReturnType_ = VoidType_;
 
  if (e.ReturnTypeId_)
    RetType = VariableType( getBaseType(*e.ReturnTypeId_) );

  insertFunction(e.Id_, ArgTypes, RetType);

}

//==============================================================================
void Analyzer::dispatch(FunctionAST& e)
{
  auto OldScope = Scope_;
  auto InnerScope = createScope();

  auto & ProtoExpr = *e.ProtoExpr_;
  const auto & FnId = ProtoExpr.Id_;
  auto FnName = FnId.getName();
  auto Loc = FnId.getLoc();

  runFuncVisitor(ProtoExpr);
  auto ProtoType = getFunction(FnId);

  const auto & ArgIds = ProtoExpr.ArgIds_;
  const auto & ArgTypes = ProtoType->getArgTypes();
  auto NumArgs = ArgTypes.size();
  
  if (NumArgs != ArgIds.size())
    THROW_NAME_ERROR("Numer of arguments in prototype for function '" << FnName
        << "', does not match definition.  Expected " << ArgIds.size()
        << " but got " << NumArgs, Loc);

  // If this is an operator, install it.
  if (ProtoExpr.isBinaryOp())
    BinopPrecedence_->operator[](ProtoExpr.getOperatorName()) = ProtoExpr.getBinaryPrecedence();

  // Record the function arguments in the NamedValues map.
  for (unsigned i=0; i<NumArgs; ++i)
    insertVariable(ArgIds[i], ArgTypes[i]);
  
  for ( auto & B : e.BodyExprs_ ) runStmtVisitor(*B, InnerScope);
  
  if (e.ReturnExpr_) {
    auto RetType = runStmtVisitor(*e.ReturnExpr_, InnerScope);
    if (ProtoExpr.isAnonExpr())
      ProtoExpr.setReturnType(RetType);
    else if (RetType != ProtoType->getReturnType())
      THROW_NAME_ERROR("Function return type does not match prototype for '"
          << FnName << "'.  The type '" << RetType << "' cannot be "
          << "converted to the type '" << ProtoType->getReturnType() << "'.",
          e.ReturnExpr_->getLoc());
  }
  
  resetScope(OldScope);
  
}

}
