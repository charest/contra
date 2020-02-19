#include "contra.hpp"

#include <iostream>

using namespace llvm;

namespace contra {

//==============================================================================
// Top-Level definition handler
//==============================================================================
void HandleDefinition(Parser & TheParser, CodeGen & TheCG)
{
  if (auto FnAST = TheParser.ParseDefinition()) {
    auto *FnIR = FnAST->codegen(TheCG, TheParser);
    if (!TheCG.isDebug()) {
      if (FnIR) {
        std::cerr << "Read function definition:";
        FnIR->print(errs());
        std::cerr << "\n";
        TheCG.doJIT();
      }
    }
    else {
      if (!FnIR) std::cerr << "Error reading function definition:";
    }
  } else {
    // Skip token for error recovery.
    TheParser.getNextToken();
  }
}

//==============================================================================
// Top-Level external handler
//==============================================================================
void HandleExtern(Parser & TheParser, CodeGen & TheCG) {
  if (auto ProtoAST = TheParser.ParseExtern()) {
    auto FnIR = ProtoAST->codegen(TheCG);
    if (!TheCG.isDebug()) {
      if (FnIR) {
        std::cerr << "Read extern: ";
        FnIR->print(errs());
        std::cerr << "\n";
        TheCG.FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
      }
    }
    else {
      if (!FnIR) std::cerr << "Error reading extern";
    }
  } else {
    // Skip token for error recovery.
    TheParser.getNextToken();
  }
}

//==============================================================================
// Top-Level expression handler
//==============================================================================
int HandleTopLevelExpression(Parser & TheParser, CodeGen & TheCG, bool is_interactive) {
  // Evaluate a top-level expression into an anonymous function.
  if (auto FnAST = TheParser.ParseTopLevelExpr()) {
    auto FnIR = FnAST->codegen(TheCG, TheParser);
    if (FnIR) {
      if (!TheCG.isDebug()) {
        // JIT the module containing the anonymous expression, keeping a handle so
        // we can free it later.
        auto H = TheCG.doJIT();

        // Search the JIT for the __anon_expr symbol.
        auto ExprSymbol = TheCG.findSymbol("__anon_expr");
        assert(ExprSymbol && "Function not found");

        // Get the symbol's address and cast it to the right type (takes no
        // arguments, returns a double) so we can call it as a native function.
        double (*FP)() = (double (*)())(intptr_t)cantFail(ExprSymbol.getAddress());
        std::cerr << "Evaluated to " << FP() << "\n";

        // Delete the anonymous expression module from the JIT.
        TheCG.removeJIT( H );
      }
    }
    else if (!is_interactive) {
      return -1;
    }
  }
  else {
    // Skip token for error recovery.
    TheParser.getNextToken();
  }
  return 0;
}

//==============================================================================
/// top ::= definition | external | expression | ';'
//==============================================================================
int MainLoop( Parser & TheParser, CodeGen & TheCG, bool is_interactive ) {
  while (true) {
    int res = 0;
    if (is_interactive) std::cerr << "ready> ";
    switch (TheParser.CurTok) {
    case tok_eof:
      return 0;
    case ';': // ignore top-level semicolons.
      TheParser.getNextToken();
      break;
    case tok_def:
      HandleDefinition(TheParser, TheCG);
      break;
    case tok_extern:
      HandleExtern(TheParser, TheCG);
      break;
    default:
      res = HandleTopLevelExpression(TheParser, TheCG, is_interactive);
      break;
    }
    if (res) return res;
  }
}


} // namespace
