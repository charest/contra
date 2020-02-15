#include "ast.hpp"
#include "errors.hpp"
#include "token.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Verifier.h"

using namespace llvm;

namespace contra {

Value *LogErrorV(const char *Str) {
  LogError(Str);
  return nullptr;
}
 
Value *NumberExprAST::codegen(CodeGen & TheCG)
{
  return ConstantFP::get(TheCG.TheContext, APFloat(Val));
}

Value *VariableExprAST::codegen(CodeGen & TheCG)
{
  // Look this variable up in the function.
  Value *V = TheCG.NamedValues[Name];
  if (!V)
    return LogErrorV("Unknown variable name");
  return V;
}

Value *BinaryExprAST::codegen(CodeGen & TheCG) {
  Value *L = LHS->codegen(TheCG);
  Value *R = RHS->codegen(TheCG);
  if (!L || !R)
    return nullptr;

  switch (Op) {
  case op_add:
    return TheCG.Builder.CreateFAdd(L, R, "addtmp");
  case op_sub:
    return TheCG.Builder.CreateFSub(L, R, "subtmp");
  case op_mul:
    return TheCG.Builder.CreateFMul(L, R, "multmp");
  case op_div:
    return TheCG.Builder.CreateFDiv(L, R, "divtmp");
  case op_lt:
    L = TheCG.Builder.CreateFCmpULT(L, R, "cmptmp");
    // Convert bool 0/1 to double 0.0 or 1.0
    return TheCG.Builder.CreateUIToFP(L, Type::getDoubleTy(TheCG.TheContext), "booltmp");
  default:
    return LogErrorV("invalid binary operator");
  }
}

Value *CallExprAST::codegen(CodeGen & TheCG) {
  // Look up the name in the global module table.
  Function *CalleeF = TheCG.getFunction(Callee);
  if (!CalleeF)
    return LogErrorV("Unknown function referenced");

  // If argument mismatch error.
  if (CalleeF->arg_size() != Args.size())
    return LogErrorV("Incorrect # arguments passed");

  std::vector<Value *> ArgsV;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    ArgsV.push_back(Args[i]->codegen(TheCG));
    if (!ArgsV.back())
      return nullptr;
  }

  return TheCG.Builder.CreateCall(CalleeF, ArgsV, "calltmp");
}

Value *IfExprAST::codegen(CodeGen & TheCG) {
  Value *CondV = Cond->codegen(TheCG);
  if (!CondV)
    return nullptr;

  // Convert condition to a bool by comparing non-equal to 0.0.
  CondV = TheCG.Builder.CreateFCmpONE(
      CondV, ConstantFP::get(TheCG.TheContext, APFloat(0.0)), "ifcond");

  Function *TheFunction = TheCG.Builder.GetInsertBlock()->getParent();

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  BasicBlock *ThenBB = BasicBlock::Create(TheCG.TheContext, "then", TheFunction);
  BasicBlock *ElseBB = BasicBlock::Create(TheCG.TheContext, "else");
  BasicBlock *MergeBB = BasicBlock::Create(TheCG.TheContext, "ifcont");

  TheCG.Builder.CreateCondBr(CondV, ThenBB, ElseBB);

  // Emit then value.
  TheCG.Builder.SetInsertPoint(ThenBB);

  Value *ThenV = Then->codegen(TheCG);
  if (!ThenV)
    return nullptr;

  TheCG.Builder.CreateBr(MergeBB);
  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ThenBB = TheCG.Builder.GetInsertBlock();

  // Emit else block.
  TheFunction->getBasicBlockList().push_back(ElseBB);
  TheCG.Builder.SetInsertPoint(ElseBB);

  Value *ElseV = Else->codegen(TheCG);
  if (!ElseV)
    return nullptr;

  TheCG.Builder.CreateBr(MergeBB);
  // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
  ElseBB = TheCG.Builder.GetInsertBlock();

  // Emit merge block.
  TheFunction->getBasicBlockList().push_back(MergeBB);
  TheCG.Builder.SetInsertPoint(MergeBB);
  PHINode *PN = TheCG.Builder.CreatePHI(Type::getDoubleTy(TheCG.TheContext), 2, "iftmp");

  PN->addIncoming(ThenV, ThenBB);
  PN->addIncoming(ElseV, ElseBB);
  return PN;
}

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
Value *ForExprAST::codegen(CodeGen & TheCG) {
  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->codegen(TheCG);
  if (!StartVal)
    return nullptr;

  // Make the new basic block for the loop header, inserting after current
  // block.
  Function *TheFunction = TheCG.Builder.GetInsertBlock()->getParent();
  BasicBlock *PreheaderBB = TheCG.Builder.GetInsertBlock();
  BasicBlock *LoopBB = BasicBlock::Create(TheCG.TheContext, "loop", TheFunction);

  // Insert an explicit fall through from the current block to the LoopBB.
  TheCG.Builder.CreateBr(LoopBB);

  // Start insertion in LoopBB.
  TheCG.Builder.SetInsertPoint(LoopBB);

  // Start the PHI node with an entry for Start.
  PHINode *Variable =
      TheCG.Builder.CreatePHI(Type::getDoubleTy(TheCG.TheContext), 2, VarName);
  Variable->addIncoming(StartVal, PreheaderBB);

  // Within the loop, the variable is defined equal to the PHI node.  If it
  // shadows an existing variable, we have to restore it, so save it now.
  Value *OldVal = TheCG.NamedValues[VarName];
  TheCG.NamedValues[VarName] = Variable;

  // Emit the body of the loop.  This, like any other expr, can change the
  // current BB.  Note that we ignore the value computed by the body, but don't
  // allow an error.
  if (!Body->codegen(TheCG))
    return nullptr;

  // Emit the step value.
  Value *StepVal = nullptr;
  if (Step) {
    StepVal = Step->codegen(TheCG);
    if (!StepVal)
      return nullptr;
  } else {
    // If not specified, use 1.0.
    StepVal = ConstantFP::get(TheCG.TheContext, APFloat(1.0));
  }

  Value *NextVar = TheCG.Builder.CreateFAdd(Variable, StepVal, "nextvar");

  // Compute the end condition.
  Value *EndCond = End->codegen(TheCG);
  if (!EndCond)
    return nullptr;

  // Convert condition to a bool by comparing non-equal to 0.0.
  EndCond = TheCG.Builder.CreateFCmpONE(
      EndCond, ConstantFP::get(TheCG.TheContext, APFloat(0.0)), "loopcond");

  // Create the "after loop" block and insert it.
  BasicBlock *LoopEndBB = TheCG.Builder.GetInsertBlock();
  BasicBlock *AfterBB =
      BasicBlock::Create(TheCG.TheContext, "afterloop", TheFunction);

  // Insert the conditional branch into the end of LoopEndBB.
  TheCG.Builder.CreateCondBr(EndCond, LoopBB, AfterBB);

  // Any new code will be inserted in AfterBB.
  TheCG.Builder.SetInsertPoint(AfterBB);

  // Add a new entry to the PHI node for the backedge.
  Variable->addIncoming(NextVar, LoopEndBB);

  // Restore the unshadowed variable.
  if (OldVal)
    TheCG.NamedValues[VarName] = OldVal;
  else
    TheCG.NamedValues.erase(VarName);

  // for expr always returns 0.0.
  return Constant::getNullValue(Type::getDoubleTy(TheCG.TheContext));
}

Function *PrototypeAST::codegen(CodeGen & TheCG) {
  // Make the function type:  double(double,double) etc.
  std::vector<Type *> Doubles(Args.size(), Type::getDoubleTy(TheCG.TheContext));
  FunctionType *FT =
      FunctionType::get(Type::getDoubleTy(TheCG.TheContext), Doubles, false);

  Function *F =
      Function::Create(FT, Function::ExternalLinkage, Name, TheCG.TheModule.get());

  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto &Arg : F->args())
    Arg.setName(Args[Idx++]);

  return F;
}

Function *FunctionAST::codegen(CodeGen & TheCG) {
  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto &P = *Proto;
  TheCG.FunctionProtos[Proto->getName()] = std::move(Proto);
  Function *TheFunction = TheCG.getFunction(P.getName());
  if (!TheFunction)
    return nullptr;

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(TheCG.TheContext, "entry", TheFunction);
  TheCG.Builder.SetInsertPoint(BB);

  // Record the function arguments in the NamedValues map.
  TheCG.NamedValues.clear();
  for (auto &Arg : TheFunction->args())
    TheCG.NamedValues[std::string(Arg.getName())] = &Arg;

  if (Value *RetVal = Body->codegen(TheCG)) {
    // Finish off the function.
    TheCG.Builder.CreateRet(RetVal);

    // Validate the generated code, checking for consistency.
    verifyFunction(*TheFunction);

    // Run the optimizer on the function.
    TheCG.TheFPM->run(*TheFunction);

    return TheFunction;
  }

  // Error reading body, remove function.
  TheFunction->eraseFromParent();
  return nullptr;
}

} // namespace
