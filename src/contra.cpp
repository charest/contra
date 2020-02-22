#include "contra.hpp"
#include "errors.hpp"

#include <iostream>

using namespace llvm;

namespace contra {

//==============================================================================
// Top-Level definition handler
//==============================================================================
void handleFunction(Parser & TheParser, CodeGen & TheCG, bool is_interactive,
    bool is_verbose)
{
  if (is_verbose) std::cerr << "Handling function" << std::endl;

  try {
    auto FnAST = TheParser.parseFunction(1);
    auto FnIR = FnAST->codegen(TheCG, TheParser.BinopPrecedence, 1);
    if (is_verbose) FnIR->print(errs());
    if (!TheCG.isDebug()) {
      TheCG.doJIT();
    }
  }
  catch (const ContraError & e) {
    std::cerr << e.what() << std::endl;
    // Skip token for error recovery.
    if (is_interactive) {
      TheParser.getNextToken();
    }
    // otherwise keep throwing the error
    else {
      throw e;
    }
  }
}

//==============================================================================
// Top-Level definition handler
//==============================================================================
void handleDefinition(Parser & TheParser, CodeGen & TheCG, bool is_interactive,
    bool is_verbose)
{
  if (is_verbose) std::cerr << "Handling definition" << std::endl;

  try {
    auto FnAST = TheParser.parseDefinition(1);
    auto FnIR = FnAST->codegen(TheCG, TheParser.BinopPrecedence, 1);
    if (is_verbose) FnIR->print(errs());
    if (!TheCG.isDebug()) {
      TheCG.doJIT();
    }
  }
  catch (const ContraError & e) {
    std::cerr << e.what() << std::endl;
    // Skip token for error recovery.
    if (is_interactive) {
      TheParser.getNextToken();
    }
    // otherwise keep throwing the error
    else {
      throw e;
    }
  }
}

//==============================================================================
// Top-Level external handler
//==============================================================================
void handleExtern(Parser & TheParser, CodeGen & TheCG, bool is_interactive,
    bool is_verbose)
{
  if (is_verbose) std::cerr << "Handling extern" << std::endl;

  try {
    auto ProtoAST = TheParser.parseExtern(1);
    auto FnIR = ProtoAST->codegen(TheCG);
    if (is_verbose) FnIR->print(errs());
    if (!TheCG.isDebug()) {
      TheCG.FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
    }
  }
  catch (const ContraError & e) {
    std::cerr << e.what() << std::endl;
    // Skip token for error recovery.
    if (is_interactive) {
      TheParser.getNextToken();
    }
    // otherwise keep throwing the error
    else {
      throw e;
    }
  }
}

//==============================================================================
// Top-Level expression handler
//==============================================================================
void handleTopLevelExpression(Parser & TheParser, CodeGen & TheCG,
    bool is_interactive, bool is_verbose)
{
  if (is_verbose) std::cerr << "Handling top level expression" << std::endl;

  // Evaluate a top-level expression into an anonymous function.
  try {
    auto FnAST = TheParser.parseTopLevelExpr(1);
    auto FnIR = FnAST->codegen(TheCG, TheParser.BinopPrecedence);
    if (is_verbose) FnIR->print(errs());
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
      if (is_verbose) std::cerr << "---Begin Result--- " <<  "\n";
      auto ans = FP();
      std::cerr << "Ans = " << ans << "\n";
      if (is_verbose) std::cerr << "---End Result--- " <<  "\n";

      // Delete the anonymous expression module from the JIT.
      TheCG.removeJIT( H );
    }
  }
  catch (const ContraError & e) {
    std::cerr << e.what() << std::endl;
    // Skip token for error recovery.
    if (is_interactive) {
      TheParser.getNextToken();
    }
    // otherwise keep throwing the error
    else {
      throw e;
    }
  }
}

//==============================================================================
/// top ::= definition | external | expression | ';'
//==============================================================================
void mainLoop( Parser & TheParser, CodeGen & TheCG, bool is_interactive,
    bool is_verbose ) {

  BaseAST::IsVerbose = is_verbose;
  
  // Prime the first token.
  if (is_interactive) std::cerr << "contra> " << std::flush;
  TheParser.getNextToken();

  while (true) {

    if (TheParser.CurTok == tok_eof) {
      if (is_interactive) std::cerr << std::endl;
      return;
    }

    switch (TheParser.CurTok) {
    case tok_sep: // ignore top-level semicolons.
      TheParser.getNextToken();
      break;
    case tok_def:
      handleDefinition(TheParser, TheCG, is_interactive, is_verbose);
      if (is_interactive) std::cerr << "contra> " << std::flush;
      break;
    case tok_function:
      handleFunction(TheParser, TheCG, is_interactive, is_verbose);
      if (is_interactive) std::cerr << "contra> " << std::flush;
      break;
    case tok_extern:
      handleExtern(TheParser, TheCG, is_interactive, is_verbose);
      if (is_interactive) std::cerr << "contra> " << std::flush;
      break;
    default:
      handleTopLevelExpression(TheParser, TheCG, is_interactive, is_verbose); 
      if (is_interactive) std::cerr << "contra> " << std::flush;
    }

  }
}


} // namespace
