#include "ast.hpp"
#include "analysis.hpp"
#include "token.hpp"

#include "librt/librt.hpp"

namespace contra {

//==============================================================================
// Get the function
//==============================================================================
std::shared_ptr<FunctionDef> Analyzer::getFunction(const std::string & Name) {
  
  // If not, check whether we can codegen the declaration from some existing
  // prototype.
  auto FP = FunctionTable_.find(Name);
  if (FP != FunctionTable_.end()) 
    return FP->second;
  
  // see if this is an available intrinsic, try installing it first
  if (auto F = librt::RunTimeLib::tryInstall(Name))
    return F;
  
  // if found it, make sure its not a variable in scope
  return nullptr;
}

//==============================================================================
void Analyzer::dispatch(ExprAST& e)
{ e.accept(*this); }

//==============================================================================
void Analyzer::dispatch(ValueExprAST<int_t>& e)
{
  TypeResult_ = I64Type;
}

//==============================================================================
void Analyzer::dispatch(ValueExprAST<real_t>& e)
{
  TypeResult_ = F64Type;
}

//==============================================================================
void Analyzer::dispatch(ValueExprAST<std::string>& e)
{
  TypeResult_ = StrType;
}

//==============================================================================
void Analyzer::dispatch(VariableExprAST& e)
{
  auto Var = getVariable(e.Name_, e.getLoc());
  auto VarType = Var->getType();

  // array index
  if (e.IndexExpr_) {
    auto Loc = e.IndexExpr_->getLoc();
    
    if (!VarType.isArray())
      THROW_NAME_ERROR( "Cannot index scalar using '[]' operator", Loc);
    
    auto IndexType = runExprVisitor(*e.IndexExpr_);
    if (IndexType != I64Type)
      THROW_NAME_ERROR( "Array index for variable '" << e.Name_ << "' must "
          << "evaluate to an integer.", Loc );

    VarType.setArray(false); // revert to scalar
  }

  // result
  TypeResult_ = VarType;
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
}

//==============================================================================
void Analyzer::dispatch(CastExprAST& e)
{
  auto FromType = runExprVisitor(*e.FromExpr_);
  auto TypeId = e.TypeId_;
  auto ToType = getType( TypeId.getName(), TypeId.getLoc() );
  checkIsCastable(FromType, ToType, e.getLoc());
  TypeResult_ = VariableType(ToType);
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
    return;
  case tok_lt:
    TypeResult_ = BoolType;
    return;
  } 
  
  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  auto F = getFunction(std::string("binary") + OpCode);
  if (!F) THROW_NAME_ERROR("binary operator not found!", Loc);
  TypeResult_ = F->getReturnType();
}

//==============================================================================
void Analyzer::dispatch(CallExprAST& e)
{
  auto FunRes = getFunction(e.Callee_);
  if (!FunRes)
    THROW_NAME_ERROR("Prototype for '" << e.Callee_ << "' not found!", e.getLoc());

  auto NumArgs = e.ArgExprs_.size();
  auto FunArgs = FunRes->getNumArgs();
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
}

//==============================================================================
void Analyzer::dispatch(ForExprAST& e)
{
  auto VarId = e.VarId_;
  
  auto OldScope = Scope_;
  Scope_++;

  auto LoopVar = insertVariable(VarId, I64Type);
  
  auto it = VariableTable_.find(VarId.getName());
  if (it != VariableTable_.end())
    THROW_NAME_ERROR("Variable '" << VarId.getName() << "' has been"
         << " previously defined", VarId.getLoc());

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
void Analyzer::dispatch(IfExprAST& e)
{
  auto IfLoc = e.getLoc();

  auto & CondExpr = *e.CondExpr_;
  auto CondType = runExprVisitor(CondExpr);
  if (CondType != Context::BoolType )
    THROW_NAME_ERROR( "If condition must result in boolean type.", CondExpr.getLoc() );

  auto OldScope = Scope_;
  for ( auto & stmt : e.ThenExpr_ ) { Scope_ = OldScope+1; runExprVisitor(*stmt); }
  for ( auto & stmt : e.ElseExpr_ ) { Scope_ = OldScope+1; runExprVisitor(*stmt); }
  Scope_ = OldScope;

  TypeResult_ = VoidType;
}

//==============================================================================
void Analyzer::dispatch(VarDefExprAST& e)
{

  // check if there is a specified type, if there is, get it
  auto TypeId = e.TypeId_;
  VariableType VarType;
  if (TypeId) {
    auto it = TypeTable_.find(TypeId.getName());
    if ( it == TypeTable_.end() )
      THROW_NAME_ERROR("Unknown type specifier '" << TypeId.getName() << "'.",
        TypeId.getLoc());
    VarType = VariableType(it->second, e.isArray());
    DestinationType_ = VarType;
  }
  
  auto InitType = runExprVisitor(*e.InitExpr_);

  
  auto NumVars = e.VarIds_.size();
  for (int i=0; i<NumVars; ++i) {
    auto VarId = e.VarIds_[i];
    auto VarName = VarId.getName();
    auto it = VariableTable_.find(VarName);
    if (it != VariableTable_.end())
      THROW_NAME_ERROR("Variable '" << VarName << "' has been"
           << " previously defined", VarId.getLoc());
    auto S = std::make_shared<VariableDef>( VarName, VarId.getLoc(), VarType);
    VariableTable_.emplace( VarId.getName(), std::move(S) );
  }

  if (InitType.isArray()) {
    std::stringstream ss;
    ss << e.VarIds_[0].getName();
    if (NumVars == 2) ss << ", " << e.VarIds_[1].getName();
    else if (NumVars > 2) ss << ", etc...";
    THROW_NAME_ERROR("Scalar variables '" << ss.str() << "' cannot be "
        << " assigned to an array.", e.VarIds_[0].getLoc());
  }

  if ( VarType != InitType ) {
    // auto OrigInit = std::move(e.Init_);
    // e.Init_ = std::make_unique<CastExpr>( std::move(OrigInit, InitType.Type, VarType );
    //std::cerr << "Casting from " << InitType.Type->getName() << " to " << VarType->getName()
    //  << std::endl;
    THROW_NAME_ERROR( "Cast operations not implemented yet.", e.VarIds_[0].getLoc() );
  }

  TypeResult_ = VarType;
}

//==============================================================================
void Analyzer::dispatch(ArrayDefExprAST& e)
{
  auto InitType = runExprVisitor(*e.InitExpr_);

  auto TypeId = e.TypeId_;
  auto it = TypeTable_.find(TypeId.getName());
  if ( it == TypeTable_.end() )
    THROW_NAME_ERROR("Unknown type specifier '" << TypeId.getName() << "'.",
        TypeId.getLoc());
  auto VarType = VariableType(it->second, e.isArray());
 
  auto NumVars = e.VarIds_.size();
  for (int i=0; i<NumVars; ++i) {
    auto VarId = e.VarIds_[i];
    auto VarName = VarId.getName();
    auto it = VariableTable_.find(VarId.getName());
    if (it != VariableTable_.end())
      THROW_NAME_ERROR("Variable '" << VarId.getName() << "' has been"
           << " previously defined", VarId.getLoc());
    auto S = std::make_shared<VariableDef>( VarId.getName(), VarId.getLoc(), VarType);
    VariableTable_.emplace( VarId.getName(), std::move(S) );
  }

  if ( VarType != InitType ) {
    THROW_NAME_ERROR( "Cast operations not implemented yet.", e.VarIds_[0].getLoc() );
  }

  //---------------------------------------------------------------------------
  // Array already on right hand side
  if (InitType.isArray()) {
  }

  //---------------------------------------------------------------------------
  //  on right hand side
  else {
    auto SizeType = runExprVisitor(*e.SizeExpr_);
    if (SizeType != I64Type)
      THROW_NAME_ERROR( "Size expression for arrays must be an integer.",
          e.SizeExpr_->getLoc());
  }

  TypeResult_ = VarType;
}

//==============================================================================
void Analyzer::dispatch(PrototypeAST& e)
{
  auto FnName = e.getName();
  auto NumArgs = e.ArgIds_.size();

  VariableTypeList ArgTypes;
  ArgTypes.reserve( NumArgs );
  
  for (int i=0; i<NumArgs; ++i) {
    // check type specifier
    const auto & TypeId = e.ArgTypeIds_[i];
    auto sit = TypeTable_.find(TypeId.getName());
    if ( sit == TypeTable_.end() )
      THROW_NAME_ERROR("Unknown type specifier '" << TypeId.getName() << "' in prototype"
          " for function '" << FnName << "'.", TypeId.getLoc());
    VariableType ArgType(sit->second, e.ArgIsArray_[i]);
    ArgTypes.emplace_back(std::move(ArgType));
  }

  VariableType RetType = VoidType;
 
  if (e.ReturnTypeId_) { 
    auto RetId = *e.ReturnTypeId_;
    auto it = TypeTable_.find(RetId.getName());
    if ( it == TypeTable_.end() )
      THROW_NAME_ERROR("Unknown return type specifier '" << RetId.getName() << "' in prototype"
          " for function '" << FnName << "'.", RetId.getLoc());
    RetType = VariableType(it->second);
  }

  auto Sy = std::make_shared<UserFunction>(FnName, e.Loc_, ArgTypes, RetType);
  auto fit = FunctionTable_.emplace( FnName, std::move(Sy) );
  if (!fit.second)
    THROW_NAME_ERROR("Prototype already exists for '" << FnName << "'.",
        e.Loc_);

}

//==============================================================================
void Analyzer::dispatch(FunctionAST& e)
{
  auto OldScope = Scope_;
  Scope_++;

  auto & ProtoExpr = *e.ProtoExpr_;
  auto FnName = ProtoExpr.getName();
  auto Loc = ProtoExpr.getLoc();

  runFuncVisitor(ProtoExpr);
  auto ProtoType = getFunction(FnName);
  if (!ProtoType)
    THROW_NAME_ERROR("No valid prototype for '" << FnName << "'.", Loc);

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
          << FnName << "'.", e.ReturnExpr_->getLoc());
  }
  
  Scope_ = OldScope;
  
}

}
