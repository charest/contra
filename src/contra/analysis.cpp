#include "analysis.hpp"
#include "token.hpp"

#include "librt/librt.hpp"

#include <algorithm>
#include <memory>
#include <string>

namespace contra {

////////////////////////////////////////////////////////////////////////////////
// Constructor
////////////////////////////////////////////////////////////////////////////////
Analyzer::Analyzer(std::shared_ptr<BinopPrecedence> Prec) :
  BinopPrecedence_(std::move(Prec))
{
  auto & ctx = Context::instance();

  std::vector< std::tuple<std::string, VariableType, std::vector<VariableType>> >
    fun = {
      {I64Type_.getBaseType()->getName(), I64Type_, {F64Type_}},
      {F64Type_.getBaseType()->getName(), F64Type_, {I64Type_}},
      {"len", I64Type_, {RangeType_}},
      {"part", setPartition(I64Type_), {RangeType_, setArray(I64Type_)}},
      {"part", setPartition(I64Type_), {RangeType_, setPartition(I64Type_), setField(I64Type_)}},
    };

  for (const auto & f : fun) {
    auto Sy = std::make_unique<BuiltInFunction>(
        std::get<0>(f),
        std::get<1>(f),
        std::get<2>(f));
    ctx.insertFunction( std::move(Sy) );
  }
}



////////////////////////////////////////////////////////////////////////////////
// Base type interface
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
TypeDef* Analyzer::getType(const Identifier & Id)
{
  const auto & Name = Id.getName();
  auto res = Context::instance().getType(Name);
  if (!res)
    THROW_NAME_ERROR("Unknown type specifier '" << Name << "'.", Id.getLoc());
  return res.get();
}

  
////////////////////////////////////////////////////////////////////////////////
// Function routines
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
void Analyzer::removeFunction(const std::string & Name)
{ Context::instance().eraseFunction(Name); }

//==============================================================================
FunctionDef* Analyzer::getFunction(
    const std::string & Name,
    const LocationRange & Loc,
    int NumArgs = -1)
{
  
  // If not, check whether we can codegen the declaration from some existing
  // prototype.
  auto FP = Context::instance().getFunction(Name);
  if (FP) {
    if (FP.get()->size()>1) {
      if (NumArgs>-1) {
        for (const auto & F : *FP.get()) {
          if (F.get()->getNumArgs() == static_cast<unsigned>(NumArgs))
            return F.get();
        }
      }
      THROW_NAME_ERROR("Too many functions to choose from.", Loc);
    }
    return FP.get()->front().get();
  }
  
  // see if this is an available intrinsic, try installing it first
  if (auto F = librt::RunTimeLib::tryInstall(Name)) {
    auto res = Context::instance().insertFunction(std::move(F));
    return res.first;
  }
  
  THROW_NAME_ERROR("No valid prototype for '" << Name << "'.", Loc);

  // if found it, make sure its not a variable in scope
  return nullptr;
}

//==============================================================================
FunctionDef* Analyzer::getFunction(const Identifier & Id)
{ return getFunction(Id.getName(), Id.getLoc()); }
  
//==============================================================================
FunctionDef* Analyzer::insertFunction(
    const Identifier & Id,
    const VariableTypeList & ArgTypes,
    const VariableType & RetType)
{ 
  const auto & Name = Id.getName();
  auto Sy = std::make_unique<UserFunction>(Name, Id.getLoc(), RetType, ArgTypes);
  auto res = Context::instance().insertFunction( std::move(Sy) );
  return res.first;
}

////////////////////////////////////////////////////////////////////////////////
// Variable interface
////////////////////////////////////////////////////////////////////////////////


//==============================================================================
VariableDef* Analyzer::getVariable(const Identifier & Id)
{
  const auto & Name = Id.getName();
  auto res = Context::instance().getVariable(Name);
  if (!res)
    THROW_NAME_ERROR("Variable '" << Name << "' has not been"
       << " previously defined", Id.getLoc());
  return res.get();
}

//==============================================================================
VariableDef* Analyzer::insertVariable(
    const Identifier & Id,
    const VariableType & VarType)
{
  const auto & Name = Id.getName();
  const auto & Loc = Id.getLoc();
  auto S = std::make_unique<VariableDef>(Name, Loc, VarType);
  auto res = Context::instance().insertVariable( std::move(S) );
  if (!res.isInserted())
    THROW_NAME_ERROR("Variable '" << Name << "' has been"
        << " previously defined", Loc);
  return res.get();
}

//==============================================================================
std::pair<VariableDef*, bool> Analyzer::getOrInsertVariable(
    const Identifier & Id,
    const VariableType & VarType)
{
  const auto & Name = Id.getName();
  const auto & Loc = Id.getLoc();
  
  {
    auto res = Context::instance().getVariable(Name);
    if (res) return {res.get(), false};
  }
 
  {
    auto S = std::make_unique<VariableDef>(Name, Loc, VarType);
    auto res = Context::instance().insertVariable( std::move(S) );
    return {res.get(), true};
  }
}

////////////////////////////////////////////////////////////////////////////////
// type checking interface
////////////////////////////////////////////////////////////////////////////////

//==============================================================================
void Analyzer::checkIsCastable(
    const VariableType & FromType,
    const VariableType & ToType,
    const LocationRange & Loc)
{
  auto IsCastable = FromType.isCastableTo(ToType);
  if (!IsCastable) {
    THROW_NAME_ERROR("Cannot cast from type '" << FromType << "' to type '"
        << ToType << "'.", Loc);
  }
}
  
//==============================================================================
void Analyzer::checkIsAssignable(
    const VariableType & LeftType,
    const VariableType & RightType,
    const LocationRange & Loc)
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

std::unique_ptr<CastExprAST>
Analyzer::insertCastOp(
    NodeAST* FromExpr,
    const VariableType & ToType )
{
  auto Loc = FromExpr->getLoc();
  auto E = std::make_unique<CastExprAST>(Loc, FromExpr, ToType);
  return E;
}

//==============================================================================
VariableType
Analyzer::promote(
    const VariableType & LeftType,
    const VariableType & RightType,
    const LocationRange & Loc)
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
void Analyzer::visit(ValueExprAST& e)
{
  switch (e.getValueType()) {
  case ValueExprAST::ValueType::Int:
    TypeResult_ = I64Type_;
    break;
  case ValueExprAST::ValueType::Real:
    TypeResult_ = F64Type_;
    break;
  case ValueExprAST::ValueType::String:
    TypeResult_ = StrType_;
    break;
  }
  e.setType(TypeResult_);
}


//==============================================================================
void Analyzer::visit(VarAccessExprAST& e)
{
  const auto & Name = e.getName();

  auto VarDef = getVariable(Identifier{Name, e.getLoc()});
  auto VarType = VarDef->getType();

  // result
  TypeResult_ = VarType;
  e.setType(TypeResult_);
  e.setVariableDef(VarDef);
}

//==============================================================================
void Analyzer::visit(ArrayAccessExprAST& e)
{
  const auto & Name = e.getName();

  auto VarDef = getVariable(Identifier{Name, e.getLoc()});
  auto & VarType = VarDef->getType();

  // array index
  auto Loc = e.getIndexExpr()->getLoc();
  
  auto IndexType = runExprVisitor(*e.getIndexExpr());
  if (IndexType.isRange()) {
    VarType.setField();
  }
  else if (IndexType != I64Type_)
    THROW_NAME_ERROR( "Array index for variable '" << Name << "' must "
        << "evaluate to an integer.", Loc );
  
  if (!VarType.isIndexable())
    THROW_NAME_ERROR( "Cannot index scalar using '[]' operator", Loc);
  
  
  TypeResult_ = VarType.getIndexedType();

  // result
  e.setType(TypeResult_);
  e.setVariableDef(VarDef);
}

//==============================================================================
void Analyzer::visit(ArrayExprAST& e)
{
  if (e.hasSize()) {
    auto SizeType = runExprVisitor(*e.getSizeExpr());
    if (SizeType != I64Type_)
      THROW_NAME_ERROR( "Size expression for arrays must be an integer.",
          e.getSizeExpr()->getLoc());
  }

  int NumVals = e.getNumVals();
  
  VariableTypeList ValTypes;
  ValTypes.reserve(NumVals);
  
  VariableType CommonType;

  for (int i=0; i<NumVals; ++i) {
    auto & ValExpr = *e.getValExpr(i);
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
      auto Loc = e.getValExpr(i)->getLoc();
      checkIsCastable(ValType, CommonType, Loc);
      e.setValExpr(i, insertCastOp(std::move(e.moveValExpr(i)), CommonType) );
    }
  }

  CommonType.setArray();
  TypeResult_ = CommonType;
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::visit(RangeExprAST& e)
{
  // start
  auto StartType = runExprVisitor(*e.getStartExpr());
  auto EndType = runExprVisitor(*e.getEndExpr());

  const auto & Loc = e.getLoc();
  if (DestinationType_ && strip(DestinationType_) != I64Type_)
    THROW_NAME_ERROR( "Only integer types supported for range expressions.", Loc );

  if (StartType != I64Type_) {
    checkIsCastable(StartType, I64Type_, Loc);
    e.setStartExpr( insertCastOp(std::move(e.moveStartExpr()), I64Type_) );
  }

  if (EndType != I64Type_) {
    checkIsCastable(EndType, I64Type_, Loc);
    e.setEndExpr( insertCastOp(std::move(e.moveEndExpr()), I64Type_) );
  }

  TypeResult_ = VariableType(I64Type_, VariableType::Attr::Range);
  e.setType(TypeResult_);
}


//==============================================================================
void Analyzer::visit(CastExprAST& e)
{
  auto FromType = runExprVisitor(*e.getFromExpr());
  auto TypeId = e.getTypeId();
  auto ToType = VariableType(getType(TypeId));
  checkIsCastable(FromType, ToType, e.getLoc());
  TypeResult_ = VariableType(ToType);
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::visit(UnaryExprAST& e)
{
  auto OpCode = e.getOperand();
  auto OpType = runExprVisitor(*e.getOpExpr());
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
void Analyzer::visit(BinaryExprAST& e)
{
  auto Loc = e.getLoc();
  auto OpCode = e.getOperand();

  auto RightLoc = e.getRightExpr()->getLoc();
  auto LeftLoc = e.getLeftExpr()->getLoc();
  
  auto RightType = runExprVisitor(*e.getRightExpr());
  auto LeftType = runExprVisitor(*e.getLeftExpr());

  if ( !LeftType.isNumber() || !RightType.isNumber())
      THROW_NAME_ERROR( "Binary operators only allowed for scalar numeric "
          << "expressions.", Loc );
  
  auto CommonType = LeftType;
  if (RightType != LeftType) {
    checkIsCastable(RightType, LeftType, RightLoc);
    checkIsCastable(LeftType, RightType, LeftLoc);
    CommonType = promote(LeftType, RightType, Loc);
    if (RightType != CommonType)
      e.setRightExpr( insertCastOp(std::move(e.moveRightExpr()), CommonType ) );
    else
      e.setLeftExpr( insertCastOp(std::move(e.moveLeftExpr()), CommonType ) );
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
void Analyzer::visit(CallExprAST& e)
{
  const auto & FunName = e.getName();
  auto FunRes = getFunction(FunName, e.getLoc(), e.getNumArgs());
  e.setFunctionDef(FunRes);
  
  int NumArgs = e.getNumArgs();
  int NumFixedArgs = FunRes->getNumArgs();

  auto IsTask = FunRes->isTask();

  if (IsTask && isGlobalScope()) {
    if (HaveTopLevelTask_)  
      THROW_NAME_ERROR("You are not allowed to have more than one top-level task.",
          e.getLoc());
    if (NumArgs > 0)
      THROW_NAME_ERROR("You are not allowed to pass arguments to the top-level task.",
          e.getLoc());
    HaveTopLevelTask_ = true;
    e.setTopLevelTask();
  }

  if (FunRes->isVarArg()) {
    if (NumArgs < NumFixedArgs)
      THROW_NAME_ERROR("Variadic function '" << FunName
          << "', must have at least " << NumFixedArgs << " arguments, but only "
          << NumArgs << " provided.", e.getLoc());
  }
  else {
    if (NumFixedArgs != NumArgs)
      THROW_NAME_ERROR("Incorrect number of arguments specified for '" << FunName
          << "', " << NumArgs << " provided but expected " << NumFixedArgs, e.getLoc());
  }
  
  std::vector<VariableType> ArgTypes;
  ArgTypes.reserve(NumArgs);

  for (int i=0; i<NumArgs; ++i) {
    auto ArgExpr = e.getArgExpr(i);
    auto ArgType = runExprVisitor(*ArgExpr);

    if (i<NumFixedArgs) {
      auto ParamType = FunRes->getArgType(i);
      if (ArgType != ParamType) {
        checkIsCastable(ArgType, ParamType, ArgExpr->getLoc());
        e.setArgExpr(i, insertCastOp( std::move(e.moveArgExpr(i)), ParamType) );
      }
    }

    ArgTypes.emplace_back(ArgType);
  } // args

  TypeResult_ = FunRes->getReturnType(); 

  e.setArgTypes( ArgTypes );
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::visit(ExprListAST& e)
{
  std::vector<VariableType> Types;
  for (const auto & Expr : e.getExprs()) 
  { Types.emplace_back( runExprVisitor(*Expr) );  }

  VariableType StructType(Types);
  TypeResult_ = StructType;
  e.setType(TypeResult_);
}

//==============================================================================
void Analyzer::visitFor(ForStmtAST&e)
{
  // Range for
  DestinationType_ = RangeType_;
  auto RangeType = runExprVisitor(*e.getStartExpr());
  if (RangeType != RangeType_)
    THROW_NAME_ERROR( "Range-based for loops must iterate over a range expression.",
        e.getStartExpr()->getLoc() );
}

//==============================================================================
void Analyzer::visit(ForStmtAST& e)
{
  visitFor(e);
  
  createScope();
  auto VarId = e.getVarId();
  insertVariable(VarId, I64Type_);
  for ( const auto & stmt : e.getBodyExprs() ) runStmtVisitor(*stmt);
  popScope();
  
  TypeResult_ = VoidType_;
}

//==============================================================================
void Analyzer::visit(ForeachStmtAST& e)
{
  visitFor(e);
      
  createScope();
  
  const auto & VarId = e.getVarId();
  auto VarD = insertVariable(VarId, I64Type_);
  
  auto & BodyExprs = e.getBodyExprs();
  
  unsigned NumParts = 0;
  bool DoParts = true;
  for ( const auto & stmt : BodyExprs ) {
    if (dynamic_cast<const PartitionStmtAST *>(stmt.get())) {
      if (!DoParts)
        THROW_NAME_ERROR("Partition statements must appear first in 'foreach' loops.",
            stmt->getLoc());
      NumParts++;
    }
    else if (DoParts) {
      DoParts = false;
    }
  }
  
  createScope();
  for (unsigned i=0; i<NumParts; ++i) {
    auto Expr = dynamic_cast<PartitionStmtAST*>(BodyExprs[i].get());
    runStmtVisitor(*Expr);
  }
  popScope();

  createScope();
  auto NumExprs = BodyExprs.size();
  for (unsigned i=NumParts; i<NumExprs; ++i) runStmtVisitor(*BodyExprs[i]);
  auto AccessedVars = Context::instance().getVariablesAccessedFromAbove();
  {
    auto it = std::find(AccessedVars.begin(), AccessedVars.end(), VarD);
    if (it != AccessedVars.end()) AccessedVars.erase(it);
  }
  popScope();

  e.setNumPartitions(NumParts);
  
  e.setAccessedVariables(AccessedVars);
  
  popScope();

  TypeResult_ = VoidType_;
}

//==============================================================================
void Analyzer::visit(IfStmtAST& e)
{
  auto CondType = runExprVisitor(*e.getCondExpr());
  if (CondType != BoolType_ )
    THROW_NAME_ERROR( "If condition must result in boolean type.", e.getCondExpr()->getLoc() );

  createScope();
  for ( const auto & stmt : e.getThenExprs() ) runStmtVisitor(*stmt);
  for ( const auto & stmt : e.getElseExprs() ) runStmtVisitor(*stmt);
  popScope();

  TypeResult_ = VoidType_;
}

//==============================================================================
void Analyzer::visit(AssignStmtAST& e)
{
  auto Loc = e.getLoc();
  
  // Assignment requires the LHS to be an identifier.
  // This assume we're building without RTTI because LLVM builds that way by
  // default.  If you build LLVM with RTTI this can be changed to a
  // dynamic_cast for automatic error checking.
  
  auto NumLeft = e.getNumLeftExprs();
  auto NumRight = e.getNumRightExprs();

  if ( (NumLeft != NumRight) && (NumRight != 1) )
      THROW_NAME_ERROR("Number of expressions on left- and right-hand sides "
        << "must match.", Loc);

  //------------------------------------
  // First figure out if it has a prescribed type
    
  auto LHSE = dynamic_cast<VarAccessExprAST*>(e.getLeftExpr(0));
  if (!LHSE)
    THROW_NAME_ERROR("destination of '=' must be a variable", LHSE->getLoc());

  VariableType VarType;
  if (LHSE->hasTypeId())
    VarType = VariableType(getType(LHSE->getTypeId()));


  //------------------------------------
  // Loop over variables
  for (unsigned il=0, ir=0; il<NumLeft; il++) {

    auto LeftExpr = e.getLeftExpr(il);
    auto LeftLoc = LeftExpr->getLoc();
    auto LHSE = dynamic_cast<VarAccessExprAST*>(LeftExpr);
    if (!LHSE)
      THROW_NAME_ERROR("destination of '=' must be a variable", LeftLoc);
    
    auto RightExpr = e.getRightExpr(ir);

    auto VarId = LHSE->getVarId();
    VariableDef* LeftDef = nullptr;
    bool WasInserted = false;
    if (VarType) {
      LeftDef = insertVariable(VarId, VarType);
      WasInserted = true;
    }
    else {
      auto DefPair = getOrInsertVariable(VarId);
      LeftDef = DefPair.first;
      WasInserted = DefPair.second;
    }

    auto LeftType = runExprVisitor(*LHSE);
    DestinationType_ = LeftType;
    auto RightType = runExprVisitor(*RightExpr);
  
    if (RightType.isStruct())
      RightType = RightType.getMember(il);

    if (!LeftType) {
      LeftType.setBaseType( RightType.getBaseType() );
      if (!WasInserted)
        THROW_NAME_ERROR("Redeclaration of '" << VarId.getName()
            << "' is not allowed.", VarId.getLoc());
    }

    if (WasInserted) {
      LeftType.setAttributes( RightType.getAttributes() );
      LeftType.setField( LeftDef->getType().isField() );
      LHSE->setType(LeftType);
      LeftDef->getType() = LeftType;
    }

    checkIsAssignable( LeftType, RightType, Loc );

    if (RightType.getBaseType() != LeftType.getBaseType()) {
      checkIsCastable(RightType, LeftType, Loc);
      if (NumLeft==NumRight)
        e.setRightExpr( ir, insertCastOp(std::move(e.moveRightExpr(ir)), LeftType) );
      else
        e.addCast(il, LeftType);
    }

    if (NumRight>1) ir++;

  } // for
  
  TypeResult_ = VariableType();
}

//==============================================================================
void Analyzer::visit(PartitionStmtAST& e)
{

  auto NumRanges = e.getNumVars();

  for (unsigned i=0; i<NumRanges; ++i) {
    const auto & RangeId = e.getVarId(i);
    auto RangeDef = getVariable(RangeId);
    if (!RangeDef->getType().isRange() && !RangeDef->getType().isField())
      THROW_NAME_ERROR("Identifier '" << RangeId.getName()
          << "' is not a valid range or field.", RangeId.getLoc());
    e.setVarDef(i, RangeDef);
  }

  auto PartExpr = e.getPartExpr();
  auto PartType = runExprVisitor(*PartExpr);
  if ( PartType != I64Type_ && !PartType.isPartition())
      THROW_NAME_ERROR(
          "Only partitions expected in 'use' statement.",
          PartExpr->getLoc());

  TypeResult_ = PartType;
}

//==============================================================================
#if 0
void Analyzer::visit(VarDeclAST& e)
{
  // check if there is a specified type, if there is, get it
  auto TypeId = e.getTypeId();
  VariableType VarType;
  if (TypeId) {
    VarType = VariableType(getType(TypeId));
    VarType.setArray( e.isArray() );
    VarType.setRange( e.isRange() );
    VarType.setRange( e.isPartition() );
    DestinationType_ = VarType;
  }
  
  auto InitType = runExprVisitor(*e.getInitExpr());
  if (!VarType) {
    VarType = InitType;
    e.setArray(InitType.isArray());
    e.setRange(InitType.isRange());
    e.setPartition(InitType.isPartition());
  }

  //----------------------------------------------------------------------------
  // Array Variable
  if (e.isArray()) {

    // If Array already on right hand side, nothing to do
    
    //  scalar on right hand side
    if (!InitType.isArray()) {
    
      auto ElementType = VariableType(VarType);
      ElementType.setArray(false);
      if (ElementType != InitType) {
        checkIsCastable(InitType, ElementType, e.getInitExpr()->getLoc());
        e.setInitExpr( insertCastOp(std::move(e.moveInitExpr()), ElementType) );
      }
 
      if (e.hasSize()) {
        auto SizeType = runExprVisitor(*e.getSizeExpr());
        if (SizeType != I64Type_)
          THROW_NAME_ERROR( "Size expression for arrays must be an integer.",
             e.getSizeExpr()->getLoc());
      }

    } // scalar init

  }
  //----------------------------------------------------------------------------
  // Range Variable
  else if (e.isRange()) {
    if (! dynamic_cast<RangeExprAST*>(e.getInitExpr()) )
      THROW_NAME_ERROR( "Range expressions can only be inialized with ranges.",
          e.getInitExpr()->getLoc());
  }
  //----------------------------------------------------------------------------
  // Partition Variable
  else if (e.isPartition()) {
    if (! dynamic_cast<PartitionStmtAST*>(e.getInitExpr()) )
      THROW_NAME_ERROR( "Partition expressions can only be inialized with partitions.",
          e.getInitExpr()->getLoc());
  }
  //----------------------------------------------------------------------------
  // Scalar variable
  else {

    if (VarType != InitType) {
      checkIsCastable(InitType, VarType, e.getInitExpr()->getLoc());
      e.setInitExpr( insertCastOp(std::move(e.moveInitExpr()), VarType) );
    }

  }
  // End
  //----------------------------------------------------------------------------
  
  auto NumVars = e.getNumVars();
  for (unsigned i=0; i<NumVars; ++i) {
    auto VarId = e.getVarId(i);
    auto VarDef = insertVariable(VarId, VarType);
    e.setVariableDef(i, VarDef);
  }

  TypeResult_ = VarType;
}

//==============================================================================
void Analyzer::visit(FieldDeclAST& e)
{
  // check if there is a specified type, if there is, get it
  auto TypeId = e.getTypeId();
  VariableType VarType;
  if (TypeId) {
    VarType = VariableType(getType(TypeId));
    VarType.setArray( e.isArray() );
    DestinationType_ = VarType;
  }
  
  auto InitType = runExprVisitor(*e.getInitExpr());
  if (InitType.isArray())
    THROW_NAME_ERROR( "Only scalar initializers allowed for fields.",
        e.getInitExpr()->getLoc());
  
  if (!VarType) {
    VarType = InitType;
  }

  //----------------------------------------------------------------------------
  // Array Variable
  if (e.isArray()) {

    auto ElementType = VariableType(VarType);
    ElementType.setArray(false);
    if (ElementType != InitType) {
      checkIsCastable(InitType, ElementType, e.getInitExpr()->getLoc());
      e.setInitExpr( insertCastOp(std::move(e.moveInitExpr()), ElementType) );
    }
 
    if (!e.hasSize()) 
      THROW_NAME_ERROR( "Array size is explicitly required for fields.", e.getLoc() );

    auto SizeType = runExprVisitor(*e.getSizeExpr());
    if (SizeType != I64Type_)
      THROW_NAME_ERROR( "Size expression for arrays must be an integer.",
         e.getSizeExpr()->getLoc());
  }
  //----------------------------------------------------------------------------
  // Scalar variable
  else {

    if (VarType != InitType) {
      checkIsCastable(InitType, VarType, e.getInitExpr()->getLoc());
      e.setInitExpr( insertCastOp(std::move(e.moveInitExpr()), VarType) );
    }

  }
  // End
  //----------------------------------------------------------------------------
 
  // Field specific
  auto PartType = runExprVisitor(*e.getPartExpr());
  if (PartType != RangeType_)
    THROW_NAME_ERROR( "Partition expression for fields must be a range.",
       e.getPartExpr()->getLoc());

  
  VarType.setField();

  auto NumVars = e.getNumVars();
  for (unsigned i=0; i<NumVars; ++i) {
    auto VarId = e.getVarId(i);
    auto VarDef = insertVariable(VarId, VarType);
    e.setVariableDef(i, VarDef);
  }

  TypeResult_ = VarType;
}
#endif

//==============================================================================
void Analyzer::visit(PrototypeAST& e)
{
  int NumArgs = e.getNumArgs();

  std::vector<VariableType> ArgTypes;
  ArgTypes.reserve( NumArgs );
  
  for (int i=0; i<NumArgs; ++i) {
    // check type specifier
    const auto & TypeId = e.getArgTypeId(i);
    auto ArgType = VariableType( getType(TypeId) );
    ArgType.setArray( e.isArgArray(i) );
    ArgTypes.emplace_back(std::move(ArgType));
  }

  e.setArgTypes(ArgTypes);

  VariableType RetType = VoidType_;

  const auto & ReturnTypeIds = e.getReturnTypeIds();
  auto NumRet = ReturnTypeIds.size();
  if (NumRet) {
    std::vector<VariableType> RetTypes = {};
    for ( const auto & Id : e.getReturnTypeIds() )
      RetTypes.emplace_back( VariableType( getType(Id) ) );
    if (NumRet == 1) {
      RetType = RetTypes.front();
    }
    else {
      RetType = VariableType(RetTypes);   
    }
  }
  
  e.setReturnType(RetType);

  insertFunction(e.getId(), ArgTypes, RetType);

}

//==============================================================================
void Analyzer::visit(FunctionAST& e)
{
  bool CreatedScope = false;
  if (!e.isTopLevelExpression()) {
    CreatedScope = true;
    createScope();
  }

  auto & ProtoExpr = *e.getProtoExpr();
  const auto & FnId = ProtoExpr.getId();
  auto FnName = FnId.getName();
  auto Loc = FnId.getLoc();

  runProtoVisitor(ProtoExpr);
  auto FunDef = getFunction(FnId);
  if (!FunDef)  
    THROW_NAME_ERROR("No valid prototype for function '" << FnName << "'", Loc);

  e.setFunctionDef(FunDef);

  auto NumArgIds = ProtoExpr.getNumArgs();
  const auto & ArgTypes = FunDef->getArgTypes();
  auto NumArgs = ArgTypes.size();
  
  if (NumArgs != NumArgIds)
    THROW_NAME_ERROR("Numer of arguments in prototype for function '" << FnName
        << "', does not match definition.  Expected " << NumArgIds
        << " but got " << NumArgs, Loc);
 
  if (e.isTask()) FunDef->setTask();

  // If this is an operator, install it.
  if (ProtoExpr.isBinaryOp())
    BinopPrecedence_->operator[](ProtoExpr.getOperatorName()) = ProtoExpr.getBinaryPrecedence();

  // Record the function arguments in the NamedValues map.
  for (unsigned i=0; i<NumArgs; ++i) {
    insertVariable(ProtoExpr.getArgId(i), ArgTypes[i]);
  }
  
  for ( const auto & B : e.getBodyExprs() ) runStmtVisitor(*B);
  
  if (e.getReturnExpr()) {
    DestinationType_ = ProtoExpr.hasReturn() ?
      FunDef->getReturnType() : VariableType{};
    auto RetType = runExprVisitor(*e.getReturnExpr());
    auto DeclRetType = FunDef->getReturnType();
    
    if (!ProtoExpr.hasReturn() || ProtoExpr.isAnonExpr()) {
      ProtoExpr.setReturnType(RetType);
      FunDef->setReturnType(RetType);
    }
    else if (RetType != DeclRetType) {
      if (!RetType.isCastableTo(DeclRetType))
        THROW_NAME_ERROR("Function return type does not match prototype for '"
            << FnName << "'.  The type '" << RetType << "' cannot be "
            << "converted to the type '" << DeclRetType << "'.",
            e.getReturnExpr()->getLoc());
      e.setReturnExpr(insertCastOp(std::move(e.moveReturnExpr()), DeclRetType) );
    }
  }
  
  if (CreatedScope) popScope();
  
}

//==============================================================================
void Analyzer::visit(TaskAST& e)
{
  visit( static_cast<FunctionAST&>(e) );
}

//==============================================================================
void Analyzer::visit(IndexTaskAST& e)
{}

}
