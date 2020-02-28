#include "ast.hpp"
#include "errors.hpp"
#include "parser.hpp"
#include "token.hpp"
#include "string_utils.hpp"
#include "vartype.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Verifier.h"

using namespace llvm;

namespace contra {

//==============================================================================
// Insert 'size' spaces.
//==============================================================================
raw_ostream &indent(raw_ostream &O, int size) {
  return O << std::string(size, ' ');
}


//==============================================================================
// Static initialization
//==============================================================================
bool BaseAST::IsVerbose = false;

//==============================================================================
// IntegerExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
Value *IntegerExprAST::codegen(CodeGen & TheCG, int Depth)
{
  echo( Formatter() << "CodeGen integer '" << Val << "'", Depth++ );
  TheCG.emitLocation(this);
  return ConstantInt::get(TheCG.TheContext, APInt(64, Val, true));
}

//------------------------------------------------------------------------------
raw_ostream &IntegerExprAST::dump(raw_ostream &out, int ind) {
  return ExprAST::dump(out << Val, ind);
}

//==============================================================================
// RealExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
Value *RealExprAST::codegen(CodeGen & TheCG, int Depth)
{
  echo( Formatter() << "CodeGen real '" << Val << "'", Depth++ );
  TheCG.emitLocation(this);
  return ConstantFP::get(TheCG.TheContext, APFloat(Val));
}

//------------------------------------------------------------------------------
raw_ostream &RealExprAST::dump(raw_ostream &out, int ind) {
  return ExprAST::dump(out << Val, ind);
}

//==============================================================================
// StringExprAST - Expression class for string literals like "hello".
//==============================================================================
Value *StringExprAST::codegen(CodeGen & TheCG, int Depth)
{
  echo( Formatter() << "CodeGen string '" << escape(Val) << "'", Depth++ );
  TheCG.emitLocation(this);
  auto & TheContext = TheCG.TheContext;
  auto & TheModule = *TheCG.TheModule;
  auto ConstantArray = ConstantDataArray::getString(TheContext, Val);
  auto GVStr = new GlobalVariable(TheModule, ConstantArray->getType(), true,
      GlobalValue::InternalLinkage, ConstantArray);
  Constant* zero = Constant::getNullValue(IntegerType::getInt32Ty(TheContext));
  Constant* strVal = ConstantExpr::getGetElementPtr(IntegerType::getInt8Ty(TheContext), GVStr, zero, true);
  return strVal;
}

//------------------------------------------------------------------------------
raw_ostream &StringExprAST::dump(raw_ostream &out, int ind) {
  return ExprAST::dump(out << Val, ind);
}
 
//==============================================================================
// VariableExprAST - Expression class for referencing a variable, like "a".
//==============================================================================
Value *VariableExprAST::codegen(CodeGen & TheCG, int Depth)
{
  echo( Formatter() << "CodeGen variable expression '" << Name << "'", Depth++ );
  // Look this variable up in the function.
  Value *V = TheCG.NamedValues[Name];
  if (!V) 
    THROW_NAME_ERROR(Name, getLine());
  TheCG.emitLocation(this);
  
  // Load the value.
  auto Ty = V->getType();
  if (!Ty->isPointerTy()) THROW_CONTRA_ERROR("why are you NOT a pointer");
  Ty = Ty->getPointerElementType();
    
  auto Load = TheCG.Builder.CreateLoad(Ty, V, "ptr."+Name);

  if ( !Index ) {
    if (TheCG.NamedArrays.count(Name))
      THROW_SYNTAX_ERROR("Array accesses require explicit indices", getLine());
    return Load;
  }
  else {
    Ty = Ty->getPointerElementType();
    auto IndexVal = Index->codegen(TheCG, Depth);
    auto GEP = TheCG.Builder.CreateGEP(Load, IndexVal, Name+"aoffset");
    return TheCG.Builder.CreateLoad(Ty, GEP, Name+"[i]");
  }
}

//------------------------------------------------------------------------------
raw_ostream &VariableExprAST::dump(raw_ostream &out, int ind) {
  return ExprAST::dump(out << Name, ind);
}

//==============================================================================
// BinaryExprAST - Expression class for a binary operator.
//==============================================================================
Value *BinaryExprAST::codegen(CodeGen & TheCG, int Depth) {
  echo( Formatter() << "CodeGen binary expression", Depth++ );
  TheCG.emitLocation(this);
  
  // Special case '=' because we don't want to emit the LHS as an expression.
  if (Op == '=') {
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
    if (!LHSE)
      THROW_SYNTAX_ERROR("destination of '=' must be a variable", LHSE->getLine());
    // Codegen the RHS.
    Value *Val = RHS->codegen(TheCG, Depth);

    // Look up the name.
    const auto & VarName = LHSE->getName();
    Value *Variable = TheCG.NamedValues[VarName];
    if (!Variable)
      THROW_NAME_ERROR(VarName, LHSE->getLine());

    if (TheCG.NamedArrays.count(VarName)) {
      auto Ty = Variable->getType()->getPointerElementType();
      auto Load = TheCG.Builder.CreateLoad(Ty, Variable, "ptr."+VarName);
      auto IndexVal = LHSE->Index->codegen(TheCG, Depth);
      auto GEP = TheCG.Builder.CreateGEP(Load, IndexVal, VarName+"aoffset");
      TheCG.Builder.CreateStore(Val, GEP);
    }
    else {
      TheCG.Builder.CreateStore(Val, Variable);
    }
    return Val;
  }

  Value *L = LHS->codegen(TheCG, Depth);
  Value *R = RHS->codegen(TheCG, Depth);

  auto l_is_double = L->getType()->isDoubleTy();
  auto r_is_double = R->getType()->isDoubleTy();
  bool is_double =  (l_is_double || r_is_double);

  if (is_double) {
    auto TheBlock = TheCG.Builder.GetInsertBlock();
    if (!l_is_double) {
      auto cast = CastInst::Create(Instruction::SIToFP, L, Type::getDoubleTy(TheCG.TheContext), "castl", TheBlock);
      L = cast;
    }
    else if (!r_is_double) {
      auto cast = CastInst::Create(Instruction::SIToFP, R, Type::getDoubleTy(TheCG.TheContext), "castr", TheBlock);
      R = cast;
    }
  }

  if (is_double) {
    switch (Op) {
    case tok_add:
      return TheCG.Builder.CreateFAdd(L, R, "addtmp");
    case tok_sub:
      return TheCG.Builder.CreateFSub(L, R, "subtmp");
    case tok_mul:
      return TheCG.Builder.CreateFMul(L, R, "multmp");
    case tok_div:
      return TheCG.Builder.CreateFDiv(L, R, "divtmp");
    case tok_lt:
      return TheCG.Builder.CreateFCmpULT(L, R, "cmptmp");
    default:
      THROW_SYNTAX_ERROR( "'" << getTokName(Op) << "' not supported yet for reals", getLine() );
    } 
  }
  else {
    switch (Op) {
    case tok_add:
      return TheCG.Builder.CreateAdd(L, R, "addtmp");
    case tok_sub:
      return TheCG.Builder.CreateSub(L, R, "subtmp");
    case tok_mul:
      return TheCG.Builder.CreateMul(L, R, "multmp");
    case tok_div:
      return TheCG.Builder.CreateSDiv(L, R, "divtmp");
    case tok_lt:
      return TheCG.Builder.CreateICmpSLT(L, R, "cmptmp");
    default:
      THROW_SYNTAX_ERROR( "'" << getTokName(Op) << "' not supported yet for ints", getLine() );
    }
  }

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  auto F = TheCG.getFunction(std::string("binary") + Op, getLine(), Depth);
  if (!F) THROW_CONTRA_ERROR("binary operator not found!");

  Value *Ops[] = { L, R };
  return TheCG.Builder.CreateCall(F, Ops, "binop");
}

//------------------------------------------------------------------------------
raw_ostream &BinaryExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "binary" << Op, ind);
  LHS->dump(indent(out, ind) << "LHS:", ind + 1);
  RHS->dump(indent(out, ind) << "RHS:", ind + 1);
  return out;
}

//==============================================================================
// CallExprAST - Expression class for function calls.
//==============================================================================
Value *CallExprAST::codegen(CodeGen & TheCG, int Depth) {
  echo( Formatter() << "CodeGen call expression '" << Callee << "'", Depth++ );
  TheCG.emitLocation(this);

  // Look up the name in the global module table.
  auto CalleeF = TheCG.getFunction(Callee, getLine(), Depth);
  if (!CalleeF)
    THROW_NAME_ERROR(Callee, getLine());

  // If argument mismatch error.
  if (CalleeF->arg_size() != Args.size() && !CalleeF->isVarArg()) {
    THROW_SYNTAX_ERROR(
        "Incorrect number of arguments, expected " << CalleeF->arg_size() 
        << " but got " << Args.size() << Formatter::to_str, getLine() );
  }

  auto FunType = CalleeF->getFunctionType();
  auto NumFixedArgs = FunType->getNumParams();

  std::vector<Value *> ArgsV;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    // what is the arg type
    auto A = Args[i]->codegen(TheCG, Depth);
    if (i < NumFixedArgs) {
      auto TheBlock = TheCG.Builder.GetInsertBlock();
      if (FunType->getParamType(i)->isDoubleTy() && A->getType()->isIntegerTy()) {
        auto cast = CastInst::Create(Instruction::SIToFP, A,
            Type::getDoubleTy(TheCG.TheContext), "cast", TheBlock);
        A = cast;
      }
      else if (FunType->getParamType(i)->isIntegerTy() && A->getType()->isDoubleTy()) {
        auto cast = CastInst::Create(Instruction::FPToSI, A,
            Type::getInt64Ty(TheCG.TheContext), "cast", TheBlock);
        A = cast;
      }
    }
    ArgsV.push_back(A);
  }

  return TheCG.Builder.CreateCall(CalleeF, ArgsV, "calltmp");
}
  
//------------------------------------------------------------------------------
raw_ostream &CallExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "call " << Callee, ind);
  for (const auto &Arg : Args)
    Arg->dump(indent(out, ind + 1), ind + 1);
  return out;
}

//==============================================================================
// IfExprAST - Expression class for if/then/else.
//==============================================================================
Value *IfExprAST::codegen(CodeGen & TheCG, int Depth) {
  echo( Formatter() << "CodeGen if expression", Depth++ );
  TheCG.emitLocation(this);

  if ( Then.empty() && Else.empty() )
    return Constant::getNullValue(Type::getInt64Ty(TheCG.TheContext));
  else if (Then.empty())
    THROW_SYNTAX_ERROR( "Can't have else with no if!", getLine() );


  Value *CondV = Cond->codegen(TheCG, Depth);

  auto TheFunction = TheCG.Builder.GetInsertBlock()->getParent();

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  BasicBlock *ThenBB = BasicBlock::Create(TheCG.TheContext, "then", TheFunction);
  BasicBlock *ElseBB = Else.empty() ? nullptr : BasicBlock::Create(TheCG.TheContext, "else");
  BasicBlock *MergeBB = BasicBlock::Create(TheCG.TheContext, "ifcont");

  if (ElseBB)
    TheCG.Builder.CreateCondBr(CondV, ThenBB, ElseBB);
  else
    TheCG.Builder.CreateCondBr(CondV, ThenBB, MergeBB);

  // Emit then value.
  TheCG.Builder.SetInsertPoint(ThenBB);

  for ( auto & stmt : Then ) {
    stmt->codegen(TheCG, Depth);
  }

  // get first non phi instruction
  auto ThenV = ThenBB->getFirstNonPHI();

  TheCG.Builder.CreateBr(MergeBB);

  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ThenBB = TheCG.Builder.GetInsertBlock();

  if (ElseBB) {

    // Emit else block.
    TheFunction->getBasicBlockList().push_back(ElseBB);
    TheCG.Builder.SetInsertPoint(ElseBB);

    for ( auto & stmt : Else ) {
      stmt->codegen(TheCG, Depth);
    }

    // get first non phi
    auto ElseV = ElseBB->getFirstNonPHI();

    TheCG.Builder.CreateBr(MergeBB);
    // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
    ElseBB = TheCG.Builder.GetInsertBlock();

  } // else

  // Emit merge block.
  TheFunction->getBasicBlockList().push_back(MergeBB);
  TheCG.Builder.SetInsertPoint(MergeBB);
  //ElseV->getType()->print(outs());
  //outs() << "\n";
  //PHINode *PN = TheCG.Builder.CreatePHI(ThenV->getType(), 2, "iftmp");

  //if (ThenV) PN->addIncoming(ThenV, ThenBB);
  //if (ElseV) PN->addIncoming(ElseV, ElseBB);
  //return PN;
  
  // for expr always returns 0.
  return Constant::getNullValue(Type::getInt64Ty(TheCG.TheContext));
}
  
//------------------------------------------------------------------------------
raw_ostream &IfExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "if", ind);
  Cond->dump(indent(out, ind) << "Cond:", ind + 1);
  indent(out, ind+1) << "Then:\n";
  for ( auto & I : Then ) I->dump(out, ind+2);
  indent(out, ind+1) << "Else:\n";
  for ( auto & I : Else ) I->dump(out, ind+2);
  return out;
}

//------------------------------------------------------------------------------
std::unique_ptr<ExprAST> IfExprAST::make( 
  std::list< std::pair< SourceLocation, std::unique_ptr<ExprAST> > > & Conds,
  std::list< std::vector< std::unique_ptr<ExprAST> > > & Blocks )
{
  auto TopPair = std::move(Conds.front());
  Conds.pop_front();

  auto TopIf = std::make_unique<IfExprAST>( TopPair.first, std::move(TopPair.second) );

  TopIf->Then = std::move( Blocks.front() );
  Blocks.pop_front();

  if ( !Blocks.empty() ) {
    if ( Conds.empty() )
      TopIf->Else = std::move(Blocks.back());
    else
      TopIf->Else.emplace_back( IfExprAST::make( Conds, Blocks ) );
  }

  return std::move(TopIf);
}

//==============================================================================
// ForExprAST - Expression class for for/in.
//
// Output for-loop as:
//   ...
//   start = startexpr
//   goto loop
// loop:
//   variable = phi [start, loopheader], [nextvariable, loopend]
//   ...
//   bodyexpr
//   ...
// loopend:
//   step = stepexpr
//   nextvariable = variable + step
//   endcond = endexpr
//   br endcond, loop, endloop
// outloop:
//==============================================================================
Value *ForExprAST::codegen(CodeGen & TheCG, int Depth) {
  echo( Formatter() << "CodeGen for expression", Depth++ );
  auto TheFunction = TheCG.Builder.GetInsertBlock()->getParent();

  // Create an alloca for the variable in the entry block.
  AllocaInst *Alloca = TheCG.createEntryBlockAlloca(TheFunction, VarName,
      VarTypes::Int, getLine());
  
  // Within the loop, the variable is defined equal to the PHI node.  If it
  // shadows an existing variable, we have to restore it, so save it now.
  AllocaInst *OldVal = TheCG.NamedValues[VarName];
  TheCG.NamedValues[VarName] = Alloca;
  TheCG.emitLocation(this);

  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->codegen(TheCG, Depth);
  if (StartVal->getType()->isDoubleTy())
    THROW_IMPLEMENTED_ERROR("Cast required for start value");

  // Store the value into the alloca.
  TheCG.Builder.CreateStore(StartVal, Alloca);

  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *BeforeBB = BasicBlock::Create(TheCG.TheContext, "beforeloop", TheFunction);
  BasicBlock *LoopBB =   BasicBlock::Create(TheCG.TheContext, "loop", TheFunction);
  BasicBlock *IncrBB =   BasicBlock::Create(TheCG.TheContext, "incr", TheFunction);
  BasicBlock *AfterBB =  BasicBlock::Create(TheCG.TheContext, "afterloop", TheFunction);

  TheCG.Builder.CreateBr(BeforeBB);
  TheCG.Builder.SetInsertPoint(BeforeBB);

  // Load value and check coondition
  Value *CurVar = TheCG.Builder.CreateLoad(Type::getInt64Ty(TheCG.TheContext), Alloca);

  // Compute the end condition.
  // Convert condition to a bool by comparing non-equal to 0.0.
  Value *EndCond = End->codegen(TheCG, Depth);
  if (EndCond->getType()->isDoubleTy())
    THROW_IMPLEMENTED_ERROR("Cast required for end condition");
  EndCond = TheCG.Builder.CreateICmpSLE(CurVar, EndCond, "loopcond");

  // Insert the conditional branch into the end of LoopEndBB.
  TheCG.Builder.CreateCondBr(EndCond, LoopBB, AfterBB);

  // Start insertion in LoopBB.
  //TheFunction->getBasicBlockList().push_back(LoopBB);
  TheCG.Builder.SetInsertPoint(LoopBB);
  
  // Emit the body of the loop.  This, like any other expr, can change the
  // current BB.  Note that we ignore the value computed by the body, but don't
  // allow an error.
  for ( auto & stmt : Body ) {
    stmt->codegen(TheCG, Depth);
  }

  // Insert unconditional branch to increment.
  TheCG.Builder.CreateBr(IncrBB);

  // Start insertion in LoopBB.
  //TheFunction->getBasicBlockList().push_back(IncrBB);
  TheCG.Builder.SetInsertPoint(IncrBB);
  

  // Emit the step value.
  Value *StepVal = nullptr;
  if (Step) {
    StepVal = Step->codegen(TheCG, Depth);
    if (StepVal->getType()->isDoubleTy())
      THROW_IMPLEMENTED_ERROR("Cast required for step value");
  } else {
    // If not specified, use 1.0.
    StepVal = ConstantInt::get(TheCG.TheContext, APInt(64, 1, true));
  }


  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  CurVar = TheCG.Builder.CreateLoad(Type::getInt64Ty(TheCG.TheContext), Alloca);
  Value *NextVar = TheCG.Builder.CreateAdd(CurVar, StepVal, "nextvar");
  TheCG.Builder.CreateStore(NextVar, Alloca);

  // Insert the conditional branch into the end of LoopEndBB.
  TheCG.Builder.CreateBr(BeforeBB);

  // Any new code will be inserted in AfterBB.
  //TheFunction->getBasicBlockList().push_back(AfterBB);
  TheCG.Builder.SetInsertPoint(AfterBB);

  // Restore the unshadowed variable.
  if (OldVal)
    TheCG.NamedValues[VarName] = OldVal;
  else
    TheCG.NamedValues.erase(VarName);

  // for expr always returns 0.
  return Constant::getNullValue(Type::getInt64Ty(TheCG.TheContext));
}
  

//------------------------------------------------------------------------------
raw_ostream &ForExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "for", ind);
  Start->dump(indent(out, ind) << "Cond:", ind + 1);
  End->dump(indent(out, ind) << "End:", ind + 1);
  Step->dump(indent(out, ind) << "Step:", ind + 1);
  indent(out, ind+1) << "Body:\n";
  for ( auto & B : Body )
    B->dump(out, ind+2);
  return out;
}

//==============================================================================
// UnaryExprAST - Expression class for a unary operator.
//==============================================================================
Value *UnaryExprAST::codegen(CodeGen & TheCG, int Depth) {
  echo( Formatter() << "CodeGen unary expression", Depth++ );
  auto OperandV = Operand->codegen(TheCG, Depth);
  
  if (OperandV->getType()->isDoubleTy()) {
  
    switch (Opcode) {
    case tok_sub:
      return TheCG.Builder.CreateFNeg(OperandV, "negtmp");
    default:
      THROW_SYNTAX_ERROR( "Uknown unary operator '" << static_cast<char>(Opcode)
          << "'", getLine() );
    }

  }
  else {
    switch (Opcode) {
    case tok_sub:
      return TheCG.Builder.CreateNeg(OperandV, "negtmp");
    default:
      THROW_SYNTAX_ERROR( "Uknown unary operator '" << static_cast<char>(Opcode)
          << "'", getLine() );
    }
  }

  auto F = TheCG.getFunction(std::string("unary") + Opcode, getLine(), Depth);
  if (!F)
    THROW_SYNTAX_ERROR("Unknown unary operator", getLine());

  TheCG.emitLocation(this);
  return TheCG.Builder.CreateCall(F, OperandV, "unop");
}
  
//------------------------------------------------------------------------------
raw_ostream &UnaryExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "unary" << Opcode, ind);
  Operand->dump(out, ind + 1);
  return out;
}

//==============================================================================
// VarExprAST - Expression class for var/in
//==============================================================================
Value *VarExprAST::codegen(CodeGen & TheCG, int Depth) {
  echo( Formatter() << "CodeGen var expression", Depth++ );

  auto TheFunction = TheCG.Builder.GetInsertBlock()->getParent();

  // Emit the initializer before adding the variable to scope, this prevents
  // the initializer from referencing the variable itself, and permits stuff
  // like this:
  //  var a = 1 in
  //    var a = a in ...   # refers to outer 'a'.
  auto InitVal = Init->codegen(TheCG, Depth);

  std::size_t NumVals = 0;
  auto IType = InitVal->getType();

  Value * SizeExpr = nullptr;

  if (IsArray) {
    if (Size) {
      SizeExpr = Size->codegen(TheCG, Depth);
    }
    else if (IType->isSingleValueType()) {
      NumVals = 1;
    }
    else {
      NumVals = IType->getArrayNumElements();
      IType = IType->getArrayElementType(); 
    }
  }
 
  std::vector<AllocaInst*> ArrayAllocas;

  // Register all variables and emit their initializer.
  for (const auto & VarName : VarNames) {
  std::cout << "Var " << VarName << " is a " << getVarTypeName(VarType) << std::endl;
    
    // cast init value if necessary
    auto TheBlock = TheCG.Builder.GetInsertBlock();
    if (VarType == VarTypes::Real && !InitVal->getType()->isDoubleTy()) {
      auto cast = CastInst::Create(Instruction::SIToFP, InitVal,
          Type::getDoubleTy(TheCG.TheContext), "cast", TheBlock);
      InitVal = cast;
    }
    else if (VarType == VarTypes::Int && !InitVal->getType()->isIntegerTy()) {
      auto cast = CastInst::Create(Instruction::FPToSI, InitVal,
          Type::getInt64Ty(TheCG.TheContext), "cast", TheBlock);
      InitVal = cast;
    }

    AllocaInst* Alloca;

    // create array of var
    if (IsArray) {
   
      auto Array = TheCG.createArray(TheFunction, VarName, VarType, NumVals, getLine(),
          SizeExpr);

      Alloca = TheCG.createEntryBlockAlloca(TheFunction, VarName, VarType,
          getLine(), true);
      ArrayAllocas.emplace_back(Alloca);

      TheCG.Builder.CreateStore(Array, Alloca);
    }
    else {

      Alloca = TheCG.createEntryBlockAlloca(TheFunction, VarName, VarType,
          getLine());
      TheCG.Builder.CreateStore(InitVal, Alloca);
    }
  
    // Remember this binding.
    TheCG.NamedValues[VarName] = Alloca;
  }

  if (IsArray) TheCG.initArrays(TheFunction, ArrayAllocas, InitVal, NumVals, SizeExpr);
  
  TheCG.emitLocation(this);

#if 0
  // Codegen the body, now that all vars are in scope.
  Value *BodyVal = Body->codegen(TheCG);
  if (!BodyVal)
    return nullptr;

  // Pop all our variables from scope.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
    TheCG.NamedValues[VarNames[i].first] = OldBindings[i];

  // Return the body computation.
  return BodyVal;
#endif

  return InitVal;
}

//------------------------------------------------------------------------------
raw_ostream &VarExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "var", ind);
  for (const auto &NamedVar : VarNames)
    Init->dump(indent(out, ind) << NamedVar << ':', ind + 1);
  return out;
}

//==============================================================================
// ArrayExprAST - Expression class for arrays.
//==============================================================================
Value *ArrayExprAST::codegen(CodeGen & TheCG, int Depth)
{
  echo( Formatter() << "CodeGen array expression", Depth++ );
  std::cout << "ERHERHEHREHRE" << std::endl;
  abort();

  return nullptr;
}

//------------------------------------------------------------------------------
raw_ostream &ArrayExprAST::dump(raw_ostream &out, int ind) {
  return out;
}

//==============================================================================
/// PrototypeAST - This class represents the "prototype" for a function.
//==============================================================================
Function *PrototypeAST::codegen(CodeGen & TheCG, int Depth) {
  echo( Formatter() << "CodeGen prototype expression '" << Name << "'", Depth++ );

  // Make the function type:  double(double,double) etc.
  
  std::vector<Type *> ArgTypes;
  ArgTypes.reserve(Args.size());

  for ( const auto & A : Args ) {
    switch (A.second) {
    case VarTypes::Int:
      ArgTypes.emplace_back( Type::getInt64Ty(TheCG.TheContext) );
      break;
    case VarTypes::Real:
      ArgTypes.emplace_back( Type::getDoubleTy(TheCG.TheContext) );
      break;
    default:
      THROW_SYNTAX_ERROR( "Unknown argument type of '" << getVarTypeName(A.second)
          << "' in prototype for function '" << Name << "'", Line );
    }
  }
  
  Type * ReturnType;
  switch (Return) {
  case VarTypes::Int:
    ReturnType = Type::getInt64Ty(TheCG.TheContext);
    break;
  case VarTypes::Real:
    ReturnType = Type::getDoubleTy(TheCG.TheContext);
    break;
  case VarTypes::Void:
    ReturnType = Type::getVoidTy(TheCG.TheContext);
    break;
  default:
    THROW_SYNTAX_ERROR( "Unknown return type of '" << getVarTypeName(Return)
        << "' in prototype for function '" << Name << "'", Line );
  }

  FunctionType *FT = FunctionType::get(ReturnType, ArgTypes, false);

  Function *F =
      Function::Create(FT, Function::ExternalLinkage, Name, TheCG.TheModule.get());

  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto &Arg : F->args())
    Arg.setName(Args[Idx++].first);

  return F;
}

//==============================================================================
/// FunctionAST - This class represents a function definition itself.
//==============================================================================
Function *FunctionAST::codegen(CodeGen & TheCG,
    std::map<char, int> & BinopPrecedence, int Depth)
{
  echo( Formatter() << "CodeGen function expression", Depth++ );
  
  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto &P = *Proto;
  TheCG.FunctionProtos[Proto->getName()] = std::move(Proto);
  auto TheFunction = TheCG.getFunction(P.getName(), P.getLine(), Depth);

  // If this is an operator, install it.
  if (P.isBinaryOp())
    BinopPrecedence[P.getOperatorName()] = P.getBinaryPrecedence();

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(TheCG.TheContext, "entry", TheFunction);
  TheCG.Builder.SetInsertPoint(BB);

  // Create a subprogram DIE for this function.
  auto Unit = TheCG.createFile();
  unsigned LineNo = P.getLine();
  unsigned ScopeLine = LineNo;
  DISubprogram *SP = TheCG.createSubprogram( LineNo, ScopeLine, P.getName(),
      TheFunction->arg_size(), Unit);
  if (SP)
    TheFunction->setSubprogram(SP);

  // Push the current scope.
  TheCG.pushLexicalBlock(SP);

  // Unset the location for the prologue emission (leading instructions with no
  // location in a function are considered part of the prologue and the debugger
  // will run past them when breaking on a function)
  TheCG.emitLocation(nullptr);

  // Record the function arguments in the NamedValues map.
  TheCG.NamedValues.clear();
  TheCG.NamedArrays.clear();
  unsigned ArgIdx = 0;
  for (auto &Arg : TheFunction->args()) {

    // get arg type
    auto ArgType = P.Args[ArgIdx].second;
    
    // Create an alloca for this variable.
    AllocaInst *Alloca = TheCG.createEntryBlockAlloca(TheFunction, Arg.getName(),
        ArgType, LineNo);
    
    // Create a debug descriptor for the variable.
    TheCG.createVariable( SP, Arg.getName(), ++ArgIdx, Unit, LineNo, Alloca);

    // Store the initial value into the alloca.
    TheCG.Builder.CreateStore(&Arg, Alloca);

    // Add arguments to variable symbol table.
    TheCG.NamedValues[Arg.getName()] = Alloca;
  }
 

  for ( auto & stmt : Body )
  {
    TheCG.emitLocation(stmt.get());
    auto RetVal = stmt->codegen(TheCG, Depth);
  }

  // garbage collection
  TheCG.destroyArrays();
    
  // Finish off the function.
  if ( Return ) {
    auto RetVal = Return->codegen(TheCG, Depth);
    if (RetVal->getType()->isVoidTy() )
      TheCG.Builder.CreateRetVoid();
    else
      TheCG.Builder.CreateRet(RetVal);
  }
  else {  
    TheCG.Builder.CreateRetVoid();
  }

  // Validate the generated code, checking for consistency.
  verifyFunction(*TheFunction);
    
  return TheFunction;

#if 0

  if (Value *RetVal = Body->codegen(TheCG)) {
    // Pop off the lexical block for the function.
    TheCG.popLexicalBlock();

    // Validate the generated code, checking for consistency.
    verifyFunction(*TheFunction);

    // Run the optimizer on the function.
    TheCG.TheFPM->run(*TheFunction);

    return TheFunction;
  }

  // Error reading body, remove function.
  TheFunction->eraseFromParent();
  
  if (P.isBinaryOp())
    TheParser.BinopPrecedence.erase(Proto->getOperatorName());

  // Pop off the lexical block for the function since we added it
  // unconditionally.
  TheCG.popLexicalBlock();
#endif
}
  
//------------------------------------------------------------------------------
raw_ostream &FunctionAST::dump(raw_ostream &out, int ind) {
  indent(out, ind) << "FunctionAST\n";
  ++ind;
  indent(out, ind) << "Body:\n";
  for ( auto & B : Body )
    B->dump(out, ind+1);
  return out;
}

} // namespace
