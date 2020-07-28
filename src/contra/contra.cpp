#include "contra.hpp"
#include "errors.hpp"
#include "futures.hpp"
#include "leafs.hpp"
#include "loops.hpp"

#include "utils/file_utils.hpp"

#include <iostream>

using namespace llvm;

namespace contra {

//==============================================================================
//  Main setup function
//==============================================================================
void Contra::setup(const std::string & FileName)
{
  ThePrecedence_ = std::make_shared<BinopPrecedence>();

  if (FileName.empty())
    TheParser_ = std::make_unique<Parser>(ThePrecedence_);
  else
    TheParser_ = std::make_unique<Parser>(ThePrecedence_, FileName);

  TheCG_ = std::make_unique<CodeGen>(BackendType_, IsDebug_, Arguments_);

  if (IRFileName_ == "-") {
    IRFileStream_ = &llvm::outs();
  }
  else if (!IRFileName_.empty()) {
    std::error_code EC;
    if (!isOverwrite() && utils::file_exists(IRFileName_))
      THROW_CONTRA_ERROR("File '" << IRFileName_
          << "' already exists!  Use -f to overwrite.");
    IRFile_ = std::make_unique<llvm::raw_fd_ostream>(IRFileName_, EC);
    IRFileStream_ = IRFile_.get();
  }

  if (DotFileName_ == "-") {
    TheViz_ = std::make_unique<Vizualizer>(std::cout);
  }
  else if (!DotFileName_.empty()) {
    TheViz_ = std::make_unique<Vizualizer>(DotFileName_, isOverwrite());
  }
  if (TheViz_) TheViz_->start();


  TheAnalyser_ = std::make_unique<Analyzer>(ThePrecedence_);
}

//==============================================================================
// Top-Level definition handler
//==============================================================================
std::vector<std::unique_ptr<FunctionAST>>
  Contra::optimizeFunction(std::unique_ptr<FunctionAST> F)
{
  // lift index tasks
  LoopLifter TheLifter;
  TheLifter.runVisitor(*F);

  std::vector<std::unique_ptr<FunctionAST>> Fs;
  while( auto FnAST = TheLifter.getNextFunctionAST() )
    Fs.emplace_back( std::move(FnAST) );

  Fs.emplace_back( std::move(F) );
  
  // identify futures
  FutureIdentifier TheFut;
  for ( const auto & FnAST : Fs )  TheFut.runVisitor(*FnAST);
  
  // identify leafs
  LeafIdentifier TheLeaf;
  for ( const auto & FnAST : Fs )  TheLeaf.runVisitor(*FnAST);
  
  return Fs;
}

//==============================================================================
// Top-Level definition handler
//==============================================================================
void Contra::handleFunction()
{

  if (IsVerbose_) std::cerr << "Handling function" << std::endl;

  try {
    auto FnAST = TheParser_->parseFunction();
    auto Name = FnAST->getName();
		TheAnalyser_->runFuncVisitor(*FnAST);
    
    auto FnASTs = optimizeFunction(std::move(FnAST));
    
		for (auto & FnAST : FnASTs) {
    	if (dumpDot()) TheViz_->runVisitor(*FnAST);
    	auto FnIR = TheCG_->runFuncVisitor(*FnAST);
    	if (IsOptimized_) TheCG_->optimize(FnIR);
    	if (dumpIR()) FnIR->print(*IRFileStream_);
    	if (!isCompiled()) TheCG_->doJIT();
		}

  }
  catch (const ContraError & e) {
    reportError(e);
    // Skip token for error recovery.
    if (!IsInteractive_) throw e;
    TheParser_->getNextToken();
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
    if (!isCompiled()) {
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
    case tok_task:
    case tok_function:
      handleFunction();
      if (IsInteractive_) std::cerr << "contra> " << std::flush;
      break;
    default:
      handleTopLevelExpression();
      if (IsInteractive_) std::cerr << "contra> " << std::flush;
    }

  }
}


} // namespace
