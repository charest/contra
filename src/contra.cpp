#include "contra.hpp"

#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"

#include <iostream>

using namespace llvm;
//using namespace llvm::orc;

namespace contra {

//==============================================================================
// Top-Level parsing and JIT Driver
//==============================================================================

void InitializeModuleAndPassManager(CodeGen & TheCG, ContraJIT & TheJIT) {

  auto & TheModule = TheCG.TheModule;
  auto & TheContext = TheCG.TheContext;
  auto & TheFPM = TheCG.TheFPM; 

  // Open a new module.
  TheModule = std::make_unique<Module>("my cool jit", TheContext);
  TheModule->setDataLayout(TheJIT.getTargetMachine().createDataLayout());

  // Create a new pass manager attached to it.
  TheFPM = std::make_unique<legacy::FunctionPassManager>(TheModule.get());

  // Do simple "peephole" optimizations and bit-twiddling optzns.
  TheFPM->add(createInstructionCombiningPass());
  // Reassociate expressions.
  TheFPM->add(createReassociatePass());
  // Eliminate Common SubExpressions.
  TheFPM->add(createGVNPass());
  // Simplify the control flow graph (deleting unreachable blocks, etc).
  TheFPM->add(createCFGSimplificationPass());

  TheFPM->doInitialization();
}

void HandleDefinition(Parser & TheParser, CodeGen & TheCG, ContraJIT & TheJIT)
{
  if (auto FnAST = TheParser.ParseDefinition()) {
    if (auto *FnIR = FnAST->codegen(TheCG)) {
      std::cerr << "Read function definition:";
      FnIR->print(errs());
      std::cerr << std::endl;
      TheJIT.addModule(std::move(TheCG.TheModule));
      InitializeModuleAndPassManager(TheCG, TheJIT);
    }
  } else {
    // Skip token for error recovery.
    TheParser.getNextToken();
  }
}

static void HandleExtern(Parser & TheParser, CodeGen & TheCG) {
  if (auto ProtoAST = TheParser.ParseExtern()) {
    if (auto *FnIR = ProtoAST->codegen(TheCG)) {
      std::cerr << "Read extern: ";
      FnIR->print(errs());
      std::cerr << std::endl;
      TheCG.FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
    }
  } else {
    // Skip token for error recovery.
    TheParser.getNextToken();
  }
}

static void HandleTopLevelExpression(Parser & TheParser, CodeGen & TheCG, ContraJIT & TheJIT) {
  // Evaluate a top-level expression into an anonymous function.
  if (auto FnAST = TheParser.ParseTopLevelExpr()) {
    if (FnAST->codegen(TheCG)) {
      // JIT the module containing the anonymous expression, keeping a handle so
      // we can free it later.
      auto H = TheJIT.addModule(std::move(TheCG.TheModule));
      InitializeModuleAndPassManager(TheCG, TheJIT);

      // Search the JIT for the __anon_expr symbol.
      auto ExprSymbol = TheJIT.findSymbol("__anon_expr");
      assert(ExprSymbol && "Function not found");

      // Get the symbol's address and cast it to the right type (takes no
      // arguments, returns a double) so we can call it as a native function.
      double (*FP)() = (double (*)())(intptr_t)cantFail(ExprSymbol.getAddress());
      std::cerr << "Evaluated to " << FP() << std::endl;

      // Delete the anonymous expression module from the JIT.
      TheJIT.removeModule(H);
    }
  } else {
    // Skip token for error recovery.
    TheParser.getNextToken();
  }
}

//==============================================================================
/// top ::= definition | external | expression | ';'
//==============================================================================
void MainLoop( Parser & TheParser, CodeGen & TheCG, ContraJIT & TheJIT ) {
  while (true) {
    std::cerr << "ready> ";
    switch (TheParser.CurTok) {
    case tok_eof:
      return;
    case ';': // ignore top-level semicolons.
      TheParser.getNextToken();
      break;
    case tok_def:
      HandleDefinition(TheParser, TheCG, TheJIT);
      break;
    case tok_extern:
      HandleExtern(TheParser, TheCG);
      break;
    default:
      HandleTopLevelExpression(TheParser, TheCG, TheJIT);
      break;
    }
  }
}


} // namespace
