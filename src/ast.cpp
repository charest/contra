#include "ast.hpp"
#include "config.hpp"
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
// IntegerExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
template<>
Value *IntegerExprAST::codegen(CodeGen & TheCG)
{
  TheCG.emitLocation(this);
  return ConstantInt::get(TheCG.TheContext, APInt(64, Val_, true));
}

//------------------------------------------------------------------------------
template<>
raw_ostream &IntegerExprAST::dump(raw_ostream &out, int ind) {
  return ExprAST::dump(out << Val_, ind);
}

//==============================================================================
// RealExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
template<>
Value *RealExprAST::codegen(CodeGen & TheCG)
{
  TheCG.emitLocation(this);
  return ConstantFP::get(TheCG.TheContext, APFloat(Val_));
}

//------------------------------------------------------------------------------
template<>
raw_ostream &RealExprAST::dump(raw_ostream &out, int ind) {
  return ExprAST::dump(out << Val_, ind);
}

//==============================================================================
// StringExprAST - Expression class for string literals like "hello".
//==============================================================================
template<>
Value *StringExprAST::codegen(CodeGen & TheCG)
{
  TheCG.emitLocation(this);
  auto & TheContext = TheCG.TheContext;
  auto & TheModule = *TheCG.TheModule;
  auto ConstantArray = ConstantDataArray::getString(TheContext, Val_);
  auto GVStr = new GlobalVariable(TheModule, ConstantArray->getType(), true,
      GlobalValue::InternalLinkage, ConstantArray);
  Constant* zero = Constant::getNullValue(IntegerType::getInt32Ty(TheContext));
  Constant* strVal = ConstantExpr::getGetElementPtr(IntegerType::getInt8Ty(TheContext), GVStr, zero, true);
  return strVal;
}

//------------------------------------------------------------------------------
template<>
raw_ostream &StringExprAST::dump(raw_ostream &out, int ind) {
  return ExprAST::dump(out << Val_, ind);
}
 
//==============================================================================
// VariableExprAST - Expression class for referencing a variable, like "a".
//==============================================================================
Value *VariableExprAST::codegen(CodeGen & TheCG)
{
  // Look this variable up in the function.
  auto it = TheCG.NamedValues.find(Name_);
  if (it == TheCG.NamedValues.end()) 
    THROW_NAME_ERROR(Name_, getLine());
  TheCG.emitLocation(this);

  Value* V = it->second;
  
  // Load the value.
  auto Ty = V->getType();
  if (!Ty->isPointerTy()) THROW_CONTRA_ERROR("why are you NOT a pointer");
  Ty = Ty->getPointerElementType();
    
  auto Load = TheCG.Builder.CreateLoad(Ty, V, "ptr."+Name_);

  if ( !Index_ ) {
    if (TheCG.NamedArrays.count(Name_))
      THROW_SYNTAX_ERROR("Array accesses require explicit indices", getLine());
    return Load;
  }
  else {
    Ty = Ty->getPointerElementType();
    auto IndexVal = Index_->codegen(TheCG);
    auto GEP = TheCG.Builder.CreateGEP(Load, IndexVal, Name_+".offset");
    return TheCG.Builder.CreateLoad(Ty, GEP, Name_+".i");
  }
}

//------------------------------------------------------------------------------
raw_ostream &VariableExprAST::dump(raw_ostream &out, int ind) {
  return ExprAST::dump(out << Name_, ind);
}

//==============================================================================
// BinaryExprAST - Expression class for a binary operator.
//==============================================================================
Value *BinaryExprAST::codegen(CodeGen & TheCG) {
  TheCG.emitLocation(this);
  
  // Special case '=' because we don't want to emit the LHS as an expression.
  if (Op_ == '=') {
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    auto LHSE = std::dynamic_pointer_cast<VariableExprAST>(LHS_);
    if (!LHSE)
      THROW_SYNTAX_ERROR("destination of '=' must be a variable", LHSE->getLine());
    // Codegen the RHS.
    Value *Val = RHS_->codegen(TheCG);

    // Look up the name.
    const auto & VarName = LHSE->getName();
    Value *Variable = TheCG.NamedValues[VarName];
    if (!Variable)
      THROW_NAME_ERROR(VarName, LHSE->getLine());

    if (TheCG.NamedArrays.count(VarName)) {
      if (!LHSE->isArray())
        THROW_SYNTAX_ERROR("Arrays must be indexed using '[i]'", LHSE->getLine());
      auto Ty = Variable->getType()->getPointerElementType();
      auto Load = TheCG.Builder.CreateLoad(Ty, Variable, "ptr."+VarName);
      auto IndexVal = LHSE->getIndex()->codegen(TheCG);
      auto GEP = TheCG.Builder.CreateGEP(Load, IndexVal, VarName+"aoffset");
      TheCG.Builder.CreateStore(Val, GEP);
    }
    else {
      TheCG.Builder.CreateStore(Val, Variable);
    }
    return Val;
  }

  Value *L = LHS_->codegen(TheCG);
  Value *R = RHS_->codegen(TheCG);

  auto l_is_real = L->getType()->isFloatingPointTy();
  auto r_is_real = R->getType()->isFloatingPointTy();
  bool is_real =  (l_is_real || r_is_real);

  if (is_real) {
    auto TheBlock = TheCG.Builder.GetInsertBlock();
    if (!l_is_real) {
      auto cast = CastInst::Create(Instruction::SIToFP, L,
          llvmRealType(TheCG.TheContext), "castl", TheBlock);
      L = cast;
    }
    else if (!r_is_real) {
      auto cast = CastInst::Create(Instruction::SIToFP, R,
          llvmRealType(TheCG.TheContext), "castr", TheBlock);
      R = cast;
    }
  }

  if (is_real) {
    switch (Op_) {
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
      THROW_SYNTAX_ERROR( "'" << getTokName(Op_) << "' not supported yet for reals", getLine() );
    } 
  }
  else {
    switch (Op_) {
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
      THROW_SYNTAX_ERROR( "'" << getTokName(Op_) << "' not supported yet for ints", getLine() );
    }
  }

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  auto F = TheCG.getFunction(std::string("binary") + Op_, getLine());
  if (!F) THROW_CONTRA_ERROR("binary operator not found!");

  Value *Ops[] = { L, R };
  return TheCG.Builder.CreateCall(F, Ops, "binop");
}

//------------------------------------------------------------------------------
raw_ostream &BinaryExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "binary" << Op_, ind);
  LHS_->dump(indent(out, ind) << "LHS:", ind + 1);
  RHS_->dump(indent(out, ind) << "RHS:", ind + 1);
  return out;
}

//==============================================================================
// CallExprAST - Expression class for function calls.
//==============================================================================
Value *CallExprAST::codegen(CodeGen & TheCG) {
  TheCG.emitLocation(this);

  // Look up the name in the global module table.
  auto CalleeF = TheCG.getFunction(Callee_, getLine());
  if (!CalleeF)
    THROW_NAME_ERROR(Callee_, getLine());

  // If argument mismatch error.
  if (CalleeF->arg_size() != Args_.size() && !CalleeF->isVarArg()) {
    THROW_SYNTAX_ERROR(
        "Incorrect number of arguments, expected " << CalleeF->arg_size() 
        << " but got " << Args_.size() << Formatter::to_str, getLine() );
  }

  auto FunType = CalleeF->getFunctionType();
  auto NumFixedArgs = FunType->getNumParams();

  std::vector<Value *> ArgsV;
  for (unsigned i = 0, e = Args_.size(); i != e; ++i) {
    // what is the arg type
    auto A = Args_[i]->codegen(TheCG);
    if (i < NumFixedArgs) {
      auto TheBlock = TheCG.Builder.GetInsertBlock();
      if (FunType->getParamType(i)->isFloatingPointTy() && A->getType()->isIntegerTy()) {
        auto cast = CastInst::Create(Instruction::SIToFP, A,
            llvmRealType(TheCG.TheContext), "cast", TheBlock);
        A = cast;
      }
      else if (FunType->getParamType(i)->isIntegerTy() && A->getType()->isFloatingPointTy()) {
        auto cast = CastInst::Create(Instruction::FPToSI, A,
            llvmIntegerType(TheCG.TheContext), "cast", TheBlock);
        A = cast;
      }
    }
    ArgsV.push_back(A);
  }

  return TheCG.Builder.CreateCall(CalleeF, ArgsV, "calltmp");
}
  
//------------------------------------------------------------------------------
raw_ostream &CallExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "call " << Callee_, ind);
  for (const auto &Arg : Args_)
    Arg->dump(indent(out, ind + 1), ind + 1);
  return out;
}

//==============================================================================
// IfExprAST - Expression class for if/then/else.
//==============================================================================
Value *IfExprAST::codegen(CodeGen & TheCG) {
  TheCG.emitLocation(this);

  if ( Then_.empty() && Else_.empty() )
    return Constant::getNullValue(llvmIntegerType(TheCG.TheContext));
  else if (Then_.empty())
    THROW_SYNTAX_ERROR( "Can't have else with no if!", getLine() );


  Value *CondV = Cond_->codegen(TheCG);

  auto TheFunction = TheCG.Builder.GetInsertBlock()->getParent();

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  BasicBlock *ThenBB = BasicBlock::Create(TheCG.TheContext, "then", TheFunction);
  BasicBlock *ElseBB = Else_.empty() ? nullptr : BasicBlock::Create(TheCG.TheContext, "else");
  BasicBlock *MergeBB = BasicBlock::Create(TheCG.TheContext, "ifcont");

  if (ElseBB)
    TheCG.Builder.CreateCondBr(CondV, ThenBB, ElseBB);
  else
    TheCG.Builder.CreateCondBr(CondV, ThenBB, MergeBB);

  // Emit then value.
  TheCG.Builder.SetInsertPoint(ThenBB);

  for ( auto & stmt : Then_ ) stmt->codegen(TheCG);

  // get first non phi instruction
  auto ThenV = ThenBB->getFirstNonPHI();

  TheCG.Builder.CreateBr(MergeBB);

  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ThenBB = TheCG.Builder.GetInsertBlock();

  if (ElseBB) {

    // Emit else block.
    TheFunction->getBasicBlockList().push_back(ElseBB);
    TheCG.Builder.SetInsertPoint(ElseBB);

    for ( auto & stmt : Else_ ) stmt->codegen(TheCG);

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
  return Constant::getNullValue(llvmIntegerType(TheCG.TheContext));
}
  
//------------------------------------------------------------------------------
raw_ostream &IfExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "if", ind);
  Cond_->dump(indent(out, ind) << "Cond:", ind + 1);
  indent(out, ind+1) << "Then:\n";
  for ( auto & I : Then_ ) I->dump(out, ind+2);
  indent(out, ind+1) << "Else:\n";
  for ( auto & I : Else_ ) I->dump(out, ind+2);
  return out;
}

//------------------------------------------------------------------------------
std::unique_ptr<ExprAST> IfExprAST::makeNested( 
  ExprLocPairList & Conds,
  ExprBlockList & Blocks )
{
  auto TopCond = std::move(Conds.front());
  Conds.pop_front();

  auto TopIf = std::make_unique<IfExprAST>( TopCond.Loc, std::move(TopCond.Expr),
      std::move(Blocks.front()) );
  Blocks.pop_front();

  if ( !Blocks.empty() ) {
    if ( Conds.empty() )
      TopIf->Else_ = std::move(Blocks.back());
    else
      TopIf->Else_.emplace_back( IfExprAST::makeNested( Conds, Blocks ) );
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
Value *ForExprAST::codegen(CodeGen & TheCG) {
  auto TheFunction = TheCG.Builder.GetInsertBlock()->getParent();

  // Create an alloca for the variable in the entry block.
  AllocaInst *Alloca = TheCG.createEntryBlockAlloca(TheFunction, VarName_,
      VarTypes::Int, getLine());
  
  // Within the loop, the variable is defined equal to the PHI node.  If it
  // shadows an existing variable, we have to restore it, so save it now.
  AllocaInst *OldVal = TheCG.NamedValues[VarName_];
  TheCG.NamedValues[VarName_] = Alloca;
  TheCG.emitLocation(this);

  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start_->codegen(TheCG);
  if (StartVal->getType()->isFloatingPointTy())
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
  Value *CurVar = TheCG.Builder.CreateLoad(llvmIntegerType(TheCG.TheContext), Alloca);

  // Compute the end condition.
  // Convert condition to a bool by comparing non-equal to 0.0.
  Value *EndCond = End_->codegen(TheCG);
  if (EndCond->getType()->isFloatingPointTy())
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
  for ( auto & stmt : Body_ ) stmt->codegen(TheCG);

  // Insert unconditional branch to increment.
  TheCG.Builder.CreateBr(IncrBB);

  // Start insertion in LoopBB.
  //TheFunction->getBasicBlockList().push_back(IncrBB);
  TheCG.Builder.SetInsertPoint(IncrBB);
  

  // Emit the step value.
  Value *StepVal = nullptr;
  if (Step_) {
    StepVal = Step_->codegen(TheCG);
    if (StepVal->getType()->isFloatingPointTy())
      THROW_IMPLEMENTED_ERROR("Cast required for step value");
  } else {
    // If not specified, use 1.0.
    StepVal = ConstantInt::get(TheCG.TheContext, APInt(64, 1, true));
  }


  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  CurVar = TheCG.Builder.CreateLoad(llvmIntegerType(TheCG.TheContext), Alloca);
  Value *NextVar = TheCG.Builder.CreateAdd(CurVar, StepVal, "nextvar");
  TheCG.Builder.CreateStore(NextVar, Alloca);

  // Insert the conditional branch into the end of LoopEndBB.
  TheCG.Builder.CreateBr(BeforeBB);

  // Any new code will be inserted in AfterBB.
  //TheFunction->getBasicBlockList().push_back(AfterBB);
  TheCG.Builder.SetInsertPoint(AfterBB);

  // Restore the unshadowed variable.
  if (OldVal)
    TheCG.NamedValues[VarName_] = OldVal;
  else
    TheCG.NamedValues.erase(VarName_);

  // for expr always returns 0.
  return Constant::getNullValue(llvmIntegerType(TheCG.TheContext));
}
  

//------------------------------------------------------------------------------
raw_ostream &ForExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "for", ind);
  Start_->dump(indent(out, ind) << "Cond:", ind + 1);
  End_->dump(indent(out, ind) << "End:", ind + 1);
  Step_->dump(indent(out, ind) << "Step:", ind + 1);
  indent(out, ind+1) << "Body:\n";
  for ( auto & B : Body_ )
    B->dump(out, ind+2);
  return out;
}

//==============================================================================
// UnaryExprAST - Expression class for a unary operator.
//==============================================================================
Value *UnaryExprAST::codegen(CodeGen & TheCG) {
  auto OperandV = Operand_->codegen(TheCG);
  
  if (OperandV->getType()->isFloatingPointTy()) {
  
    switch (Opcode_) {
    case tok_sub:
      return TheCG.Builder.CreateFNeg(OperandV, "negtmp");
    default:
      THROW_SYNTAX_ERROR( "Uknown unary operator '" << static_cast<char>(Opcode_)
          << "'", getLine() );
    }

  }
  else {
    switch (Opcode_) {
    case tok_sub:
      return TheCG.Builder.CreateNeg(OperandV, "negtmp");
    default:
      THROW_SYNTAX_ERROR( "Uknown unary operator '" << static_cast<char>(Opcode_)
          << "'", getLine() );
    }
  }

  auto F = TheCG.getFunction(std::string("unary") + Opcode_, getLine());
  if (!F)
    THROW_SYNTAX_ERROR("Unknown unary operator", getLine());

  TheCG.emitLocation(this);
  return TheCG.Builder.CreateCall(F, OperandV, "unop");
}
  
//------------------------------------------------------------------------------
raw_ostream &UnaryExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "unary" << Opcode_, ind);
  Operand_->dump(out, ind + 1);
  return out;
}

//==============================================================================
// VarExprAST - Expression class for var/in
//==============================================================================
Value *VarExprAST::codegen(CodeGen & TheCG) {

  auto TheFunction = TheCG.Builder.GetInsertBlock()->getParent();
  
  // Emit the initializer before adding the variable to scope, this prevents
  // the initializer from referencing the variable itself, and permits stuff
  // like this:
  //  var a = 1 in
  //    var a = a in ...   # refers to outer 'a'.

  // Emit initializer first
  auto InitVal = Init_->codegen(TheCG);
  auto IType = InitVal->getType();

  // Register all variables and emit their initializer.
  for (const auto & VarName : VarNames_) {
    
    // cast init value if necessary
    auto TheBlock = TheCG.Builder.GetInsertBlock();
    if (VarType_ == VarTypes::Real && !InitVal->getType()->isFloatingPointTy()) {
      auto cast = CastInst::Create(Instruction::SIToFP, InitVal,
          llvmRealType(TheCG.TheContext), "cast", TheBlock);
      InitVal = cast;
    }
    else if (VarType_ == VarTypes::Int && !InitVal->getType()->isIntegerTy()) {
      auto cast = CastInst::Create(Instruction::FPToSI, InitVal,
          llvmIntegerType(TheCG.TheContext), "cast", TheBlock);
      InitVal = cast;
    }

    auto Alloca = TheCG.createEntryBlockAlloca(TheFunction, VarName, VarType_,
          getLine());
    TheCG.Builder.CreateStore(InitVal, Alloca);
  
    // Remember this binding.
    TheCG.NamedValues[VarName] = Alloca;
  }


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
  for (const auto &NamedVar : VarNames_)
    Init_->dump(indent(out, ind) << NamedVar << ':', ind + 1);
  return out;
}

//==============================================================================
// ArrayVarExprAST - Expression class for array vars
//==============================================================================
Value *ArrayVarExprAST::codegen(CodeGen & TheCG) {

  auto TheFunction = TheCG.Builder.GetInsertBlock()->getParent();
  
  // Emit the initializer before adding the variable to scope, this prevents
  // the initializer from referencing the variable itself, and permits stuff
  // like this:
  //  var a = 1 in
  //    var a = a in ...   # refers to outer 'a'.

  Value* ReturnInit = nullptr;

  //---------------------------------------------------------------------------
  // Array already on right hand side
  auto ArrayAST = std::dynamic_pointer_cast<ArrayExprAST>(Init_);
  if (ArrayAST) {

    // transfer to first
    const auto & VarName = VarNames_[0];
    auto InitVal = ArrayAST->special_codegen(VarName, TheCG);
    auto ArrayAlloca = std::get<0>(InitVal);
    auto Array = std::get<1>(InitVal);
    auto SizeExpr = std::get<2>(InitVal);
    ReturnInit = Array;
    auto FirstAlloca = TheCG.createEntryBlockAlloca(TheFunction, VarName,
        VarType_, getLine(), true);
    TheCG.Builder.CreateStore(Array, FirstAlloca);
    TheCG.NamedValues[VarName] = FirstAlloca;
    TheCG.NamedArrays[VarName] = ArrayAlloca;
  
    // more complicated
    if (VarNames_.size() > 1) {
    
      std::vector<AllocaInst*> ArrayAllocas;
      ArrayAllocas.reserve(VarNames_.size());

      // Register all variables and emit their initializer.
      for (int i=1; i<VarNames_.size(); ++i) {
        const auto & VarName = VarNames_[i];
        auto Array = TheCG.createArray(TheFunction, VarName, VarType_, 0, getLine(), SizeExpr);
        auto Alloca = TheCG.createEntryBlockAlloca(TheFunction, VarName, VarType_, getLine(), true);
        TheCG.Builder.CreateStore(Array.second, Alloca);
        TheCG.NamedValues[VarName] = Alloca;
        TheCG.NamedArrays[VarName] = Array.first;
        ArrayAllocas.emplace_back( Alloca ); 
      }
      TheCG.copyArrays(TheFunction, FirstAlloca, ArrayAllocas, SizeExpr );
    }

  }
  
  //---------------------------------------------------------------------------
  // Scalar Initializer
  else {
  
    // Emit initializer first
    auto InitVal = Init_->codegen(TheCG);

    // create a size expr
    std::size_t NumVals = 0;
    auto IType = InitVal->getType();
    Value * SizeExpr = nullptr;

    if (Size_) {
      SizeExpr = Size_->codegen(TheCG);
    }
    else if (IType->isSingleValueType()) {
      NumVals = 1;
    }
    else {
      NumVals = IType->getArrayNumElements();
      IType = IType->getArrayElementType(); 
    }
 
    std::vector<AllocaInst*> ArrayAllocas;

    // Register all variables and emit their initializer.
    for (const auto & VarName : VarNames_) {
      
      // cast init value if necessary
      auto TheBlock = TheCG.Builder.GetInsertBlock();
      if (VarType_ == VarTypes::Real && !InitVal->getType()->isFloatingPointTy()) {
        auto cast = CastInst::Create(Instruction::SIToFP, InitVal,
            llvmRealType(TheCG.TheContext), "cast", TheBlock);
        InitVal = cast;
      }
      else if (VarType_ == VarTypes::Int && !InitVal->getType()->isIntegerTy()) {
        auto cast = CastInst::Create(Instruction::FPToSI, InitVal,
            llvmIntegerType(TheCG.TheContext), "cast", TheBlock);
        InitVal = cast;
      }

      AllocaInst* Alloca;

      // create array of var
      auto Array = TheCG.createArray(TheFunction, VarName, VarType_, NumVals, getLine(),
          SizeExpr);
  
      TheCG.NamedArrays[VarName] = Array.first;

      Alloca = TheCG.createEntryBlockAlloca(TheFunction, VarName, VarType_,
          getLine(), true);
      ArrayAllocas.emplace_back(Alloca);

      TheCG.Builder.CreateStore(Array.second, Alloca);
    
      // Remember this binding.
      TheCG.NamedValues[VarName] = Alloca;
    }

    TheCG.initArrays(TheFunction, ArrayAllocas, InitVal, NumVals, SizeExpr);

    ReturnInit = InitVal;

  } // else
  //---------------------------------------------------------------------------

  TheCG.emitLocation(this);


  return ReturnInit;
}

//------------------------------------------------------------------------------
raw_ostream &ArrayVarExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "var", ind);
  for (const auto &NamedVar : VarNames_)
    Init_->dump(indent(out, ind) << NamedVar << ':', ind + 1);
  return out;
}


//==============================================================================
// ArrayExprAST - Expression class for arrays.
//==============================================================================
std::tuple<AllocaInst*, Value*, Value*>
ArrayExprAST::special_codegen(const std::string & Name, CodeGen & TheCG)
{
  
  auto TheFunction = TheCG.Builder.GetInsertBlock()->getParent();    

  std::size_t NumVals{0};
  Value* SizeExpr = nullptr;
  
  std::vector<Value*> InitVals;
  InitVals.reserve(Vals_.size());
  for ( auto & E : Vals_ ) InitVals.emplace_back( E->codegen(TheCG) );

  if (Size_) {
    SizeExpr = Size_->codegen(TheCG);
    if (Vals_.size() != 1 )
      THROW_SYNTAX_ERROR("Only one value expected in [Val; N] syntax", getLine());
  }
  else {
    NumVals = Vals_.size();
  }

  auto Array = TheCG.createArray(TheFunction, "__tmp", InferredType, NumVals, getLine(), SizeExpr );
  auto Alloca = TheCG.createEntryBlockAlloca(TheFunction, "__tmp", InferredType,
          getLine(), true);
  TheCG.Builder.CreateStore(Array.second, Alloca);

  if (Size_) 
    TheCG.initArrays(TheFunction, {Alloca}, InitVals[0], NumVals, SizeExpr);
  else
  {
    TheCG.initArray(TheFunction, Alloca, InitVals);
  }

  if (!SizeExpr) SizeExpr = ConstantInt::get(TheCG.TheContext, APInt(64, NumVals, true)); 
  return {Alloca, Array.second, SizeExpr};
}

//------------------------------------------------------------------------------
raw_ostream &ArrayExprAST::dump(raw_ostream &out, int ind) {
  return out;
}

//==============================================================================
/// PrototypeAST - This class represents the "prototype" for a function.
//==============================================================================
Function *PrototypeAST::codegen(CodeGen & TheCG) {

  // Make the function type:  double(double,double) etc.
  
  std::vector<Type *> ArgTypes;
  ArgTypes.reserve(Args_.size());

  for ( const auto & A : Args_ ) {
    auto VarType = A.second.getType();
    Type * LLType;
    try {
      LLType = getLLVMType(VarType, TheCG.TheContext);
    }
    catch (const ContraError & e) {
      THROW_SYNTAX_ERROR( "Unknown argument type of '" << getVarTypeName(VarType)
          << "' in prototype for function '" << Name_ << "'", Line_ );
    }
    ArgTypes.emplace_back(LLType);
  }
  
  Type * ReturnType;
  try {
    ReturnType = getLLVMType(Return_, TheCG.TheContext);
  }
  catch (const ContraError & e) {
    THROW_SYNTAX_ERROR( "Unknown return type of '" << getVarTypeName(Return_)
        << "' in prototype for function '" << Name_ << "'", Line_ );
  }

  FunctionType *FT = FunctionType::get(ReturnType, ArgTypes, false);

  Function *F =
      Function::Create(FT, Function::ExternalLinkage, Name_, TheCG.TheModule.get());

  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto &Arg : F->args())
    Arg.setName(Args_[Idx++].first);

  return F;
}

//==============================================================================
/// FunctionAST - This class represents a function definition itself.
//==============================================================================
Function *FunctionAST::codegen(CodeGen & TheCG,
    std::map<char, int> & BinopPrecedence)
{
  
  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto &P = *Proto_;
  TheCG.FunctionProtos[Proto_->getName()] = std::move(Proto_);
  auto TheFunction = TheCG.getFunction(P.getName(), P.getLine());

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
    auto ArgType = P.getArgSymbol(ArgIdx).getType();
    
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
 

  for ( auto & stmt : Body_ )
  {
    TheCG.emitLocation(stmt.get());
    auto RetVal = stmt->codegen(TheCG);
  }

  // garbage collection
  TheCG.destroyArrays();
    
  // Finish off the function.
  if ( Return_ ) {
    auto RetVal = Return_->codegen(TheCG);
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
  for ( auto & B : Body_ )
    B->dump(out, ind+1);
  return out;
}

} // namespace
