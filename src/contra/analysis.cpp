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
  auto it = VariableTable_.find(e.Name_);
  if (it == VariableTable_.end())
    THROW_NAME_ERROR("Variable '" << e.Name_ << "' has not been"
          << " previously defined", e.getLoc());
  auto VarType = *it->second;

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
  TypeResult_ = *it->second;
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

  std::unique_ptr<VariableType> ValType;
  int NumVals = e.ValExprs_.size();

  for (int i=0; i<NumVals; ++i) {
    auto & ValExpr = *e.ValExprs_[i];
    auto VTy = runExprVisitor(ValExpr);
    if (!ValType) {
      ValType = std::make_unique<VariableType>(VTy);
    }
    else if (*ValType != VTy) {
      THROW_NAME_ERROR( "Cast operations not implemented yet.", ValExpr.getLoc() );
    }
  }

  ValType->setArray();
  TypeResult_ = *ValType;
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

  if (OpType != I64Type && OpType != F64Type)
      THROW_NAME_ERROR( "Unary operators only allowed for numeric expressions.", Loc );


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
    auto LHSE = std::dynamic_pointer_cast<VariableExprAST>(e.LeftExpr_);
    if (!LHSE)
      THROW_NAME_ERROR("destination of '=' must be a variable", LeftLoc);

    auto Name = LHSE->getName();
    auto it = VariableTable_.find(Name);
    if (it == VariableTable_.end())
      THROW_NAME_ERROR("Variable '" << Name << "' has not been"
           << " previously defined", LeftLoc);
    
    if (!LeftType.isArray() && RightType.isArray())
      THROW_NAME_ERROR("Scalar variable '" << Name << "' cannot be"
           << " assigned to an array.", Loc);

    if (RightType.getSymbol() != LeftType.getSymbol()) 
      THROW_NAME_ERROR( "Cast operations not implemented yet.", Loc );
    
    TypeResult_ = LeftType;

    return;
  }
  
  if (LeftType != I64Type && LeftType != F64Type)
      THROW_NAME_ERROR( "Binary operators only allowed for numeric expressions.",
          LeftLoc );

  if (RightType != I64Type && RightType != F64Type)
      THROW_NAME_ERROR( "Binary operators only allowed for numeric expressions.",
          RightLoc );
  
  if (LeftType.isArray())
      THROW_NAME_ERROR( "Binary operators only allowed for scalar numeric expressions.",
          LeftLoc );
  if (RightType.isArray())
      THROW_NAME_ERROR( "Binary operators only allowed for scalar numeric expressions.",
          RightLoc );
  
  if (RightType != LeftType)
    THROW_NAME_ERROR( "Cast operations not implemented yet.", LeftLoc );

  bool has_double = (RightType == F64Type || LeftType == F64Type);
    
  switch (OpCode) {
  case tok_add:
  case tok_sub:
  case tok_mul:
  case tok_div:
    TypeResult_ = has_double ? F64Type : I64Type;
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
  auto FunArgs = FunRes->getArgTypes().size();
  if (FunArgs != NumArgs)
    THROW_NAME_ERROR("Incorrect number of arguments specified for '" << e.Callee_
        << "', " << NumArgs << " provided but expected " << FunArgs, e.getLoc());

  for (int i=0; i<NumArgs; ++i) {
    auto & ArgExpr = *e.ArgExprs_[i]; 
    auto ArgType = runExprVisitor(ArgExpr);
    auto ParamType = FunRes->getArgTypes()[i];
    if (ArgType != ParamType)
      THROW_NAME_ERROR("Incorrect argument type for parameter " << i+1
          << " to function '" << e.Callee_ << "'. Passed argument of type '" 
          << ArgType.getSymbol()->getName() << "', but expected '"
          << ParamType.getSymbol()->getName() << "'.", ArgExpr.getLoc());

  }

  TypeResult_ = FunRes->getReturnType(); 
}

//==============================================================================
void Analyzer::dispatch(ForExprAST& e)
{
  auto VarId = e.VarId_;
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

  TypeResult_ = VoidType;
}

//==============================================================================
void Analyzer::dispatch(IfExprAST& e)
{
  auto IfLoc = e.getLoc();

  auto & CondExpr = *e.CondExpr_;
  auto CondType = runExprVisitor(CondExpr);
  if (CondType != Context::BoolSymbol )
    THROW_NAME_ERROR( "If condition must result in boolean type.", CondExpr.getLoc() );

  for ( auto & stmt : e.ThenExpr_ ) runExprVisitor(*stmt);
  for ( auto & stmt : e.ElseExpr_ ) runExprVisitor(*stmt);

  TypeResult_ = VoidType;
}

//==============================================================================
void Analyzer::dispatch(VarExprAST& e)
{
  auto TypeId = e.TypeId_;
  auto it = SymbolTable_.find(TypeId.getName());
  if ( it == SymbolTable_.end() )
    THROW_NAME_ERROR("Unknown type specifier '" << TypeId.getName() << "'.",
        TypeId.getLoc());
  auto VarType = VariableType(it->second);
  
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

  auto InitType = runExprVisitor(*e.InitExpr_);

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
void Analyzer::dispatch(ArrayVarExprAST& e)
{
  auto TypeId = e.TypeId_;
  auto it = SymbolTable_.find(TypeId.getName());
  if ( it == SymbolTable_.end() )
    THROW_NAME_ERROR("Unknown type specifier '" << TypeId.getName() << "'.",
        TypeId.getLoc());
  auto VarType = VariableType(it->second, true);
 
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

  auto InitType = runExprVisitor(*e.InitExpr_);
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
    auto sit = SymbolTable_.find(TypeId.getName());
    if ( sit == SymbolTable_.end() )
      THROW_NAME_ERROR("Unknown type specifier '" << TypeId.getName() << "' in prototype"
          " for function '" << FnName << "'.", TypeId.getLoc());
    VariableType ArgType(sit->second, e.ArgIsArray_[i]);
    ArgTypes.emplace_back(std::move(ArgType));
  }

  VariableType RetType = VoidType;
 
  if (e.ReturnTypeId_) { 
    auto RetId = *e.ReturnTypeId_;
    auto it = SymbolTable_.find(RetId.getName());
    if ( it == SymbolTable_.end() )
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

  for ( auto & B : e.BodyExprs_ ) dispatch(*B);
  
  if (e.ReturnExpr_) {
    auto RetType = runExprVisitor(*e.ReturnExpr_);
    if (RetType != ProtoType->getReturnType() )
      THROW_NAME_ERROR("Function return type does not match prototype for '"
          << FnName << "'.", e.ReturnExpr_->getLoc());
  }
  
}

}
