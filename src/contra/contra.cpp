#include "contra.hpp"
#include "errors.hpp"
#include "futures.hpp"
#include "leafs.hpp"
#include "lexer.hpp"
#include "loops.hpp"

#include "utils/file_utils.hpp"

#include <iostream>

using namespace llvm;

namespace contra {

//==============================================================================
//  Main setup function
//==============================================================================
Contra::Contra(ContraBuilder builder) : Builder_(builder)
{
  Tokens_ = make_contra_tokens();

  ThePrecedence_ = std::make_shared<BinopPrecedence>();
  
  auto FileName = getSourceFileName();
  if (FileName.size()) {
    InputStream_.open(FileName.c_str());
    if (!InputStream_.good()) {
      std::stringstream ss;
      ss << "File '" << FileName << "' does not exists" << std::endl;
      throw std::runtime_error( ss.str() );
    }
    In_ = &InputStream_;
  }

  if (FileName.empty())
    TheParser_ = std::make_unique<Parser>(ThePrecedence_);
  else
    TheParser_ = std::make_unique<Parser>(ThePrecedence_, FileName);

  TheCG_ = std::make_unique<CodeGen>(getBackendType(), isDebug());

  auto IRFileName = getIRFileName();
  if (IRFileName == "-") {
    IRFileStream_ = &llvm::outs();
  }
  else if (!IRFileName.empty()) {
    std::error_code EC;
    if (!isOverwrite() && utils::file_exists(IRFileName))
      THROW_CONTRA_ERROR("File '" << IRFileName
          << "' already exists!  Use -f to overwrite.");
    IRFile_ = std::make_unique<llvm::raw_fd_ostream>(IRFileName, EC);
    IRFileStream_ = IRFile_.get();
  }

  auto DotFileName = getDotFileName();
  if (DotFileName == "-") {
    TheViz_ = std::make_unique<Vizualizer>(std::cout);
  }
  else if (!DotFileName.empty()) {
    TheViz_ = std::make_unique<Vizualizer>(DotFileName, isOverwrite());
  }
  if (TheViz_) TheViz_->start();


  TheAnalyser_ = std::make_unique<Analyzer>(ThePrecedence_);
}

//==============================================================================
//  Main parse function
//==============================================================================
void Contra::getNextToken()
{
  auto res = lex(Tokens_, *In_);
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

  if (isVerbose()) std::cerr << "Handling function" << std::endl;

  try {
    auto FnAST = TheParser_->parseFunction();
    auto Name = FnAST->getName();
		TheAnalyser_->runFuncVisitor(*FnAST);
    
    auto FnASTs = optimizeFunction(std::move(FnAST));
    
		for (auto & FnAST : FnASTs) {
    	if (dumpDot()) TheViz_->runVisitor(*FnAST);
    	auto FnIR = TheCG_->runFuncVisitor(*FnAST);
    	if (isOptimized()) TheCG_->optimize(FnIR);
    	if (dumpIR()) FnIR->print(*IRFileStream_);
    	if (!isCompiled()) TheCG_->doJIT();
		}

  }
  catch (const ContraError & e) {
    reportError(e);
    // Skip token for error recovery.
    if (!isInteractive()) throw e;
    getNextToken();
  }

}

//==============================================================================
// Top-Level expression handler
//==============================================================================
void Contra::handleTopLevelExpression()
{
  if (isVerbose()) std::cerr << "Handling top level expression" << std::endl;

  const std::string Name = "__anon_expr";

  // Evaluate a top-level expression into an anonymous function.
  try {
    auto FnAST = TheParser_->parseTopLevelExpr();
    //if (isVerbose()) FnAST->accept(viz);
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
      auto H = TheCG_->doJIT(true);

      // Search the JIT for the __anon_expr symbol.
      auto ExprSymbol = ExitOnErr(TheCG_->findSymbol(Name.c_str()));

      // Get the symbol's address and cast it to the right type (takes no
      // arguments, returns a double) so we can call it as a native function.
      if (is_real) {
        real_t (*FP)() = ExprSymbol.getAddress().toPtr<real_t (*)()>();
        if (isVerbose()) std::cerr << "---Begin Real Result--- " <<  "\n";
        auto ans = FP();
        std::cerr << "Ans = " << ans << "\n";
        if (isVerbose()) std::cerr << "---End Real Result--- " <<  "\n";
      }
      else if (is_int) {
        int_t (*FP)() = ExprSymbol.getAddress().toPtr<int_t (*)()>();
        if (isVerbose()) std::cerr << "---Begin Int Result--- " <<  "\n";
        auto ans = FP();
        std::cerr << "Ans = " << ans << "\n";
        if (isVerbose()) std::cerr << "---End Int Result--- " <<  "\n";
      }
      else if (is_void) {
        void (*FP)() = ExprSymbol.getAddress().toPtr<void(*)()>();
        if (isVerbose()) std::cerr << "---Begin Void Result--- " <<  "\n";
        FP();
        if (isVerbose()) std::cerr << "---End Void Result--- " <<  "\n";
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
    if (isInteractive()) getNextToken();
    // otherwise keep throwing the error
    else throw e;
  }
}

//==============================================================================
/// top ::= definition | external | expression | ';'
//==============================================================================
void Contra::mainLoop() {

  // Prime the first token.
  if (isInteractive()) std::cerr << "contra> " << std::flush;
  getNextToken();

  while (true) {

    if (TheParser_->getCurTok() == tok_eof) {
      if (isInteractive()) std::cerr << std::endl;
      return;
    }

    switch (TheParser_->getCurTok()) {
    case tok_sep: // ignore top-level semicolons.
      getNextToken();
      break;
    case tok_task:
    case tok_function:
      handleFunction();
      if (isInteractive()) std::cerr << "contra> " << std::flush;
      break;
    default:
      handleTopLevelExpression();
      if (isInteractive()) std::cerr << "contra> " << std::flush;
    }

  }
}


} // namespace
