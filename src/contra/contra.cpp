#include "contra.hpp"
#include "errors.hpp"
#include "inputs.hpp"

#include <iostream>

using namespace llvm;

namespace contra {

//==============================================================================
// Top-Level definition handler
//==============================================================================
void Contra::handleFunction()
{

  if (IsVerbose_) std::cerr << "Handling function" << std::endl;

  try {
    auto FnAST = TheParser_->parseFunction();
    if (dumpDot()) TheViz_->runVisitor(*FnAST);
    TheAnalyser_->runFuncVisitor(*FnAST);
    auto FnIR = TheCG_->runFuncVisitor(*FnAST);
    if (dumpIR()) FnIR->print(*IRFileStream_);
    if (IsOptimized_) TheCG_->optimize(FnIR);
    if (!IsDebug_) TheCG_->doJIT();
  }
  catch (const ContraError & e) {
    reportError(e);
    // Skip token for error recovery.
    if (!IsInteractive_) throw e;
    TheParser_->getNextToken();
  }

}

//==============================================================================
// Top-Level definition handler
//==============================================================================
void Contra::handleDefinition()
{
  if (IsVerbose_) std::cerr << "Handling definition" << std::endl;

  try {
    auto FnAST = TheParser_->parseDefinition();
    if (dumpDot()) TheViz_->runVisitor(*FnAST);
    auto FnIR = TheCG_->runFuncVisitor(*FnAST);
    if (dumpIR()) FnIR->print(*IRFileStream_);
    if (!IsDebug_) TheCG_->doJIT();
  }
  catch (const CodeError & e) {
    reportError(e);
    // Skip token for error recovery.
    if (IsInteractive_) TheParser_->getNextToken();
    // otherwise keep throwing the error
    else throw e;
  }
}

//==============================================================================
// Top-Level external handler
//==============================================================================
void Contra::handleExtern()
{
  if (IsVerbose_) std::cerr << "Handling extern" << std::endl;

  try {
    auto ProtoAST = TheParser_->parseExtern();
    auto FnIR = TheCG_->runFuncVisitor(*ProtoAST);
    if (dumpIR()) FnIR->print(*IRFileStream_);
    if (!IsDebug_) TheCG_->insertFunction(std::move(ProtoAST));
  }
  catch (const ContraError & e) {
    reportError(e);
    // Skip token for error recovery.
    if (IsInteractive_) TheParser_->getNextToken();
    // otherwise keep throwing the error
    else throw e;
  }
}

//==============================================================================
// Top-Level expression handler
//==============================================================================
void Contra::handleTopLevelExpression()
{
  if (IsVerbose_) std::cerr << "Handling top level expression" << std::endl;

  const std::string Name = "__anon_expr";

  // Evaluate a top-level expression into an anonymous function.
  try {
    auto FnAST = TheParser_->parseTopLevelExpr();
    //if (IsVerbose_) FnAST->accept(viz);
    TheAnalyser_->runFuncVisitor(*FnAST);
    auto FnIR = TheCG_->runFuncVisitor(*FnAST);
    if (dumpIR()) FnIR->print(*IRFileStream_);
    // get return type
    auto RetType = FnIR->getReturnType();
    auto is_real = RetType->isFloatingPointTy();
    auto is_int = RetType->isIntegerTy();
    auto is_void = RetType->isVoidTy();
    // execute it 
    if (!IsDebug_) {
      // JIT the module containing the anonymous expression, keeping a handle so
      // we can free it later.
      auto H = TheCG_->doJIT();

      // Search the JIT for the __anon_expr symbol.
      auto ExprSymbol = TheCG_->findSymbol(Name.c_str());
      assert(ExprSymbol && "Function not found");

      // Get the symbol's address and cast it to the right type (takes no
      // arguments, returns a double) so we can call it as a native function.
      if (is_real) {
        real_t (*FP)() = (real_t (*)())(intptr_t)cantFail(ExprSymbol.getAddress());
        if (IsVerbose_) std::cerr << "---Begin Real Result--- " <<  "\n";
        auto ans = FP();
        std::cerr << "Ans = " << ans << "\n";
        if (IsVerbose_) std::cerr << "---End Real Result--- " <<  "\n";
      }
      else if (is_int) {
        int_t (*FP)() = (int_t(*)())(intptr_t)cantFail(ExprSymbol.getAddress());
        if (IsVerbose_) std::cerr << "---Begin Int Result--- " <<  "\n";
        auto ans = FP();
        std::cerr << "Ans = " << ans << "\n";
        if (IsVerbose_) std::cerr << "---End Int Result--- " <<  "\n";
      }
      else if (is_void) {
        void (*FP)() = (void(*)())(intptr_t)cantFail(ExprSymbol.getAddress());
        if (IsVerbose_) std::cerr << "---Begin Void Result--- " <<  "\n";
        FP();
        if (IsVerbose_) std::cerr << "---End Void Result--- " <<  "\n";
      }
      else {
        THROW_CONTRA_ERROR("Unknown type of final result!");
      }
      
      // Delete the anonymous expression module from the JIT.
      TheCG_->removeJIT( H );
      TheAnalyser_->removeFunction(Name);
    }
  }
  catch (const ContraError & e) {
    reportError(e);
    // Skip token for error recovery.
    if (IsInteractive_) TheParser_->getNextToken();
    // otherwise keep throwing the error
    else throw e;
  }
}

//==============================================================================
/// top ::= definition | external | expression | ';'
//==============================================================================
void Contra::mainLoop() {

  // Prime the first token.
  if (IsInteractive_) std::cerr << "contra> " << std::flush;
  TheParser_->getNextToken();

  while (true) {

    if (TheParser_->getCurTok() == tok_eof) {
      if (IsInteractive_) std::cerr << std::endl;
      return;
    }

    switch (TheParser_->getCurTok()) {
    case tok_sep: // ignore top-level semicolons.
      TheParser_->getNextToken();
      break;
    case tok_def:
      handleDefinition();
      if (IsInteractive_) std::cerr << "contra> " << std::flush;
      break;
    case tok_task:
    case tok_function:
      handleFunction();
      if (IsInteractive_) std::cerr << "contra> " << std::flush;
      break;
    case tok_extern:
      handleExtern();
      if (IsInteractive_) std::cerr << "contra> " << std::flush;
      break;
    default:
      handleTopLevelExpression();
      if (IsInteractive_) std::cerr << "contra> " << std::flush;
    }

  }
}


} // namespace
