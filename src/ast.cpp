#include "ast.hpp"
#include "errors.hpp"
#include "parser.hpp"
#include "token.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Verifier.h"

using namespace llvm;

namespace contra {

//==============================================================================
// Error Utility
//==============================================================================
Value *LogErrorV(const char *Str) {
  LogError(Str);
  return nullptr;
}

//==============================================================================
// Insert 'size' spaces.
//==============================================================================
raw_ostream &indent(raw_ostream &O, int size) {
  return O << std::string(size, ' ');
}

//==============================================================================
// NumberExprAST - Expression class for numeric literals like "1.0".
//==============================================================================
Value *NumberExprAST::codegen(CodeGen & TheCG)
{
  TheCG.emitLocation(this);
  return ConstantFP::get(TheCG.TheContext, APFloat(Val));
}

raw_ostream &NumberExprAST::dump(raw_ostream &out, int ind) {
  return ExprAST::dump(out << Val, ind);
}

//==============================================================================
// VariableExprAST - Expression class for referencing a variable, like "a".
//==============================================================================
Value *VariableExprAST::codegen(CodeGen & TheCG)
{
  // Look this variable up in the function.
  Value *V = TheCG.NamedValues[Name];
  if (!V)
    return LogErrorV("Unknown variable name");
  TheCG.emitLocation(this);
  // Load the value.
  return TheCG.Builder.CreateLoad(V, Name.c_str());
}

raw_ostream &VariableExprAST::dump(raw_ostream &out, int ind) {
  return ExprAST::dump(out << Name, ind);
}

//==============================================================================
// BinaryExprAST - Expression class for a binary operator.
//==============================================================================
Value *BinaryExprAST::codegen(CodeGen & TheCG) {
  TheCG.emitLocation(this);
  
  // Special case '=' because we don't want to emit the LHS as an expression.
  if (Op == '=') {
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
    if (!LHSE)
      return LogErrorV("destination of '=' must be a variable");
    // Codegen the RHS.
    Value *Val = RHS->codegen(TheCG);
    if (!Val)
      return nullptr;

    // Look up the name.
    Value *Variable = TheCG.NamedValues[LHSE->getName()];
    if (!Variable)
      return LogErrorV("Unknown variable name");

    TheCG.Builder.CreateStore(Val, Variable);
    return Val;
  }

  Value *L = LHS->codegen(TheCG);
  Value *R = RHS->codegen(TheCG);
  if (!L || !R)
    return nullptr;

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
    L = TheCG.Builder.CreateFCmpULT(L, R, "cmptmp");
    // Convert bool 0/1 to double 0.0 or 1.0
    return TheCG.Builder.CreateUIToFP(L, Type::getDoubleTy(TheCG.TheContext), "booltmp");
  }

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = TheCG.getFunction(std::string("binary") + Op);
  assert(F && "binary operator not found!");

  Value *Ops[] = { L, R };
  return TheCG.Builder.CreateCall(F, Ops, "binop");
}

raw_ostream &BinaryExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "binary" << Op, ind);
  LHS->dump(indent(out, ind) << "LHS:", ind + 1);
  RHS->dump(indent(out, ind) << "RHS:", ind + 1);
  return out;
}

//==============================================================================
// CallExprAST - Expression class for function calls.
//==============================================================================
Value *CallExprAST::codegen(CodeGen & TheCG) {
  TheCG.emitLocation(this);

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
  
raw_ostream &CallExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "call " << Callee, ind);
  for (const auto &Arg : Args)
    Arg->dump(indent(out, ind + 1), ind + 1);
  return out;
}

//==============================================================================
// IfExprAST - Expression class for if/then/else.
//==============================================================================
Value *IfExprAST::codegen(CodeGen & TheCG) {
  TheCG.emitLocation(this);

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
  
raw_ostream &IfExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "if", ind);
  Cond->dump(indent(out, ind) << "Cond:", ind + 1);
  Then->dump(indent(out, ind) << "Then:", ind + 1);
  Else->dump(indent(out, ind) << "Else:", ind + 1);
  return out;
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
  Function *TheFunction = TheCG.Builder.GetInsertBlock()->getParent();

  // Create an alloca for the variable in the entry block.
  AllocaInst *Alloca = TheCG.createEntryBlockAlloca(TheFunction, VarName);
  
  TheCG.emitLocation(this);

  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->codegen(TheCG);
  if (!StartVal)
    return nullptr;

  // Store the value into the alloca.
  TheCG.Builder.CreateStore(StartVal, Alloca);

  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *LoopBB = BasicBlock::Create(TheCG.TheContext, "loop", TheFunction);

  // Insert an explicit fall through from the current block to the LoopBB.
  TheCG.Builder.CreateBr(LoopBB);

  // Start insertion in LoopBB.
  TheCG.Builder.SetInsertPoint(LoopBB);

  // Within the loop, the variable is defined equal to the PHI node.  If it
  // shadows an existing variable, we have to restore it, so save it now.
  AllocaInst *OldVal = TheCG.NamedValues[VarName];
  TheCG.NamedValues[VarName] = Alloca;

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

  // Compute the end condition.
  Value *EndCond = End->codegen(TheCG);
  if (!EndCond)
    return nullptr;

  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  Value *CurVar = TheCG.Builder.CreateLoad(Alloca);
  Value *NextVar = TheCG.Builder.CreateFAdd(CurVar, StepVal, "nextvar");
  TheCG.Builder.CreateStore(NextVar, Alloca);

  // Convert condition to a bool by comparing non-equal to 0.0.
  EndCond = TheCG.Builder.CreateFCmpONE(
      EndCond, ConstantFP::get(TheCG.TheContext, APFloat(0.0)), "loopcond");

  // Create the "after loop" block and insert it.
  BasicBlock *AfterBB =
      BasicBlock::Create(TheCG.TheContext, "afterloop", TheFunction);

  // Insert the conditional branch into the end of LoopEndBB.
  TheCG.Builder.CreateCondBr(EndCond, LoopBB, AfterBB);

  // Any new code will be inserted in AfterBB.
  TheCG.Builder.SetInsertPoint(AfterBB);

  // Restore the unshadowed variable.
  if (OldVal)
    TheCG.NamedValues[VarName] = OldVal;
  else
    TheCG.NamedValues.erase(VarName);

  // for expr always returns 0.0.
  return Constant::getNullValue(Type::getDoubleTy(TheCG.TheContext));
}

raw_ostream &ForExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "for", ind);
  Start->dump(indent(out, ind) << "Cond:", ind + 1);
  End->dump(indent(out, ind) << "End:", ind + 1);
  Step->dump(indent(out, ind) << "Step:", ind + 1);
  Body->dump(indent(out, ind) << "Body:", ind + 1);
  return out;
}

//==============================================================================
// UnaryExprAST - Expression class for a unary operator.
//==============================================================================
Value *UnaryExprAST::codegen(CodeGen & TheCG) {
  Value *OperandV = Operand->codegen(TheCG);
  if (!OperandV)
    return nullptr;

  Function *F = TheCG.getFunction(std::string("unary") + Opcode);
  if (!F)
    return LogErrorV("Unknown unary operator");

  TheCG.emitLocation(this);
  return TheCG.Builder.CreateCall(F, OperandV, "unop");
}
  
raw_ostream &UnaryExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "unary" << Opcode, ind);
  Operand->dump(out, ind + 1);
  return out;
}

//==============================================================================
// VarExprAST - Expression class for var/in
//==============================================================================
Value *VarExprAST::codegen(CodeGen & TheCG) {
  std::vector<AllocaInst *> OldBindings;

  Function *TheFunction = TheCG.Builder.GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();

    // Emit the initializer before adding the variable to scope, this prevents
    // the initializer from referencing the variable itself, and permits stuff
    // like this:
    //  var a = 1 in
    //    var a = a in ...   # refers to outer 'a'.
    Value *InitVal;
    if (Init) {
      InitVal = Init->codegen(TheCG);
      if (!InitVal)
        return nullptr;
    } else { // If not specified, use 0.0.
      InitVal = ConstantFP::get(TheCG.TheContext, APFloat(0.0));
    }
  
    AllocaInst *Alloca = TheCG.createEntryBlockAlloca(TheFunction, VarName);
    TheCG.Builder.CreateStore(InitVal, Alloca);
  
    // Remember the old variable binding so that we can restore the binding when
    // we unrecurse.
    OldBindings.push_back(TheCG.NamedValues[VarName]);
  
    // Remember this binding.
    TheCG.NamedValues[VarName] = Alloca;
  }
  
  TheCG.emitLocation(this);

  // Codegen the body, now that all vars are in scope.
  Value *BodyVal = Body->codegen(TheCG);
  if (!BodyVal)
    return nullptr;

  // Pop all our variables from scope.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
    TheCG.NamedValues[VarNames[i].first] = OldBindings[i];

  // Return the body computation.
  return BodyVal;
}

raw_ostream &VarExprAST::dump(raw_ostream &out, int ind) {
  ExprAST::dump(out << "var", ind);
  for (const auto &NamedVar : VarNames)
    NamedVar.second->dump(indent(out, ind) << NamedVar.first << ':', ind + 1);
  Body->dump(indent(out, ind) << "Body:", ind + 1);
  return out;
}

//==============================================================================
/// PrototypeAST - This class represents the "prototype" for a function.
//==============================================================================
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

//==============================================================================
/// FunctionAST - This class represents a function definition itself.
//==============================================================================
Function *FunctionAST::codegen(CodeGen & TheCG, Parser & TheParser) {
  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto &P = *Proto;
  TheCG.FunctionProtos[Proto->getName()] = std::move(Proto);
  Function *TheFunction = TheCG.getFunction(P.getName());
  if (!TheFunction)
    return nullptr;

  // If this is an operator, install it.
  if (P.isBinaryOp())
    TheParser.BinopPrecedence[P.getOperatorName()] = P.getBinaryPrecedence();

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
  unsigned ArgIdx = 0;
  for (auto &Arg : TheFunction->args()) {
    // Create an alloca for this variable.
    AllocaInst *Alloca = TheCG.createEntryBlockAlloca(TheFunction, Arg.getName());
    
    // Create a debug descriptor for the variable.
    TheCG.createVariable( SP, Arg.getName(), ++ArgIdx, Unit, LineNo, Alloca);

    // Store the initial value into the alloca.
    TheCG.Builder.CreateStore(&Arg, Alloca);

    // Add arguments to variable symbol table.
    TheCG.NamedValues[Arg.getName()] = Alloca;
  }
  
  TheCG.emitLocation(Body.get());

  if (Value *RetVal = Body->codegen(TheCG)) {
    // Finish off the function.
    TheCG.Builder.CreateRet(RetVal);
    
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

  return nullptr;
}
  
raw_ostream &FunctionAST::dump(raw_ostream &out, int ind) {
  indent(out, ind) << "FunctionAST\n";
  ++ind;
  indent(out, ind) << "Body:";
  return Body ? Body->dump(out, ind) : out << "null\n";
}

} // namespace
