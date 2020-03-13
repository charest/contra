#include "ast.hpp"
#include "analysis.hpp"
#include "token.hpp"

#include "librt/librt.hpp"

namespace contra {

//==============================================================================
// Get the function
//==============================================================================
std::shared_ptr<FunctionDef> Analyzer::getFunction(const std::string & Name,
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
void Analyzer::dispatch(ValueExprAST<int_t>& e)
{
  TypeResult_ = I64Type;
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::dispatch(ValueExprAST<real_t>& e)
{
  TypeResult_ = F64Type;
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::dispatch(ValueExprAST<std::string>& e)
{
  TypeResult_ = StrType;
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
    if (IndexType != I64Type)
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
    if (SizeType != I64Type)
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

  if (DestinationType_) CommonType = DestinationType_;

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

  if (OpType.isNumber())
      THROW_NAME_ERROR( "Unary operators only allowed for scalar numeric "
          << "expressions.", Loc );


  switch (OpCode) {
  case tok_sub:
  case tok_add:
    TypeResult_ = OpType;
  default:
    THROW_NAME_ERROR( "Unknown unary operator '" << OpCode << "'", Loc);
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

  if (OpCode == '=') {
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
    TypeResult_ = CommonType;
    e.setType(TypeResult_);
    return;
  case tok_lt:
    TypeResult_ = BoolType;
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
  int FunArgs = FunRes->getNumArgs();
  if (FunArgs != NumArgs)
    THROW_NAME_ERROR("Incorrect number of arguments specified for '" << e.Callee_
        << "', " << NumArgs << " provided but expected " << FunArgs, e.getLoc());

  for (int i=0; i<NumArgs; ++i) {
    auto ArgType = runExprVisitor(*e.ArgExprs_[i]);
    auto ParamType = FunRes->getArgType(i);
    if (ArgType != ParamType) {
      checkIsCastable(ArgType, ParamType, e.ArgExprs_[i]->getLoc());
      e.ArgExprs_[i] = insertCastOp( std::move(e.ArgExprs_[i]), ParamType);
    }

  }

  TypeResult_ = FunRes->getReturnType(); 
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::dispatch(ForStmtAST& e)
{
  auto VarId = e.VarId_;
  
  auto OldScope = Scope_;
  Scope_++;

  auto LoopVar = insertVariable(VarId, I64Type);

  auto & StartExpr = *e.StartExpr_;
  auto StartType = runExprVisitor(StartExpr);
  if (StartType != I64Type )
    THROW_NAME_ERROR( "For loop start expression must result in an integer type.",
        StartExpr.getLoc() );

  auto & EndExpr = *e.EndExpr_;
  auto EndType = runExprVisitor(EndExpr);
  if (EndType != I64Type )
    THROW_NAME_ERROR( "For loop end expression must result in an integer type.",
        EndExpr.getLoc() );

  if (e.StepExpr_) {
    auto & StepExpr = *e.StepExpr_;
    auto StepType = runExprVisitor(StepExpr);
    if (StepType != I64Type )
      THROW_NAME_ERROR( "For loop step expression must result in an integer type.",
          StepExpr.getLoc() );
  }

  for ( auto & stmt : e.BodyExprs_ ) runExprVisitor(*stmt);

  Scope_ = OldScope;
  TypeResult_ = VoidType;
}

//==============================================================================
void Analyzer::dispatch(IfStmtAST& e)
{
  auto & CondExpr = *e.CondExpr_;
  auto CondType = runExprVisitor(CondExpr);
  if (CondType != BoolType )
    THROW_NAME_ERROR( "If condition must result in boolean type.", CondExpr.getLoc() );

  auto OldScope = Scope_;
  for ( auto & stmt : e.ThenExpr_ ) { Scope_ = OldScope+1; runExprVisitor(*stmt); }
  for ( auto & stmt : e.ElseExpr_ ) { Scope_ = OldScope+1; runExprVisitor(*stmt); }
  Scope_ = OldScope;

  TypeResult_ = VoidType;
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

  { // scope
    auto VarAST = static_cast<VarDeclAST*>(&e);
    dispatch(*VarAST);
  }

  //---------------------------------------------------------------------------
  // Array already on right hand side
  if (TypeResult_.isArray()) {
  }

  //---------------------------------------------------------------------------
  //  on right hand side
  else {
    auto SizeType = runExprVisitor(*e.SizeExpr_);
    if (SizeType != I64Type)
      THROW_NAME_ERROR( "Size expression for arrays must be an integer.",
          e.SizeExpr_->getLoc());
  }

}

//==============================================================================
void Analyzer::dispatch(PrototypeAST& e)
{
  int NumArgs = e.ArgIds_.size();

  VariableTypeList ArgTypes;
  ArgTypes.reserve( NumArgs );
  
  for (int i=0; i<NumArgs; ++i) {
    // check type specifier
    const auto & TypeId = e.ArgTypeIds_[i];
    auto ArgType = VariableType( getBaseType(TypeId), e.ArgIsArray_[i] );
    ArgTypes.emplace_back(std::move(ArgType));
  }

  VariableType RetType = VoidType;
 
  if (e.ReturnTypeId_) { 
    auto RetId = *e.ReturnTypeId_;
    RetType = VariableType( getBaseType(RetId) );
  }

  insertFunction(e.Id_, ArgTypes, RetType);

}

//==============================================================================
void Analyzer::dispatch(FunctionAST& e)
{
  auto OldScope = Scope_;
  Scope_++;

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
  VariableTable_.clear();
  for (unsigned i=0; i<NumArgs; ++i) {
    const auto & Name = ArgIds[i].getName();
    const auto & Loc = ArgIds[i].getLoc();

    auto S = std::make_shared<VariableDef>(Name, Loc, ArgTypes[i]);
    auto it = VariableTable_.emplace( Name, std::move(S) );
    if (!it.second)
      THROW_NAME_ERROR("Duplicate definition for argument " << i+1
          << ", '" << Name << "' of function '" << FnName << "'", Loc);
  }

  for ( auto & B : e.BodyExprs_ ) {
    DestinationType_ = VariableType{};
    runExprVisitor(*B);
  }
  
  if (e.ReturnExpr_) {
    auto RetType = runExprVisitor(*e.ReturnExpr_);
    if (RetType != ProtoType->getReturnType() )
      THROW_NAME_ERROR("Function return type does not match prototype for '"
          << FnName << "'.  The type '" << RetType << "' cannot be "
          << "converted to the type '" << ProtoType->getReturnType() << "'.",
          e.ReturnExpr_->getLoc());
  }
  
  Scope_ = OldScope;
  
}

}
