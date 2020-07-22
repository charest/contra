#ifndef CONTRA_CONTRA_HPP
#define CONTRA_CONTRA_HPP

#include "analysis.hpp"
#include "backends.hpp"
#include "codegen.hpp"
#include "parser.hpp"
#include "vizualizer.hpp"

#include "compiler.hpp"

#include "llvm/Support/raw_ostream.h"

namespace contra {

class FunctionAST;

class Contra : public ErrorVisiter {

  bool IsInteractive_ = false;
  bool IsVerbose_ = false;
  bool IsDebug_ = false;
  bool IsOptimized_ = false;
  bool IsOverwrite_ = false;

  std::string OutputFileName_;
  std::string IRFileName_;
  std::string DotFileName_;

  llvm::raw_ostream* IRFileStream_ = nullptr;
  std::unique_ptr<llvm::raw_ostream> IRFile_;

  std::shared_ptr<BinopPrecedence> ThePrecedence_;
  std::unique_ptr<Parser> TheParser_;
  std::unique_ptr<CodeGen> TheCG_;
  
  std::unique_ptr<Vizualizer> TheViz_;
  std::unique_ptr<Analyzer> TheAnalyser_;

  SupportedBackends BackendType_ = static_cast<SupportedBackends>(0);

  std::string Arguments_;

public:


  ~Contra() {
    // Print out all of the generated code.
    //TheCG.TheModule->print(llvm::errs(), nullptr);
    // Compile if necessary
    if (!OutputFileName_.empty()) compile( TheCG_->getModule(), OutputFileName_ );
    IRFileStream_ = nullptr;
  }


  void setup(const std::string & FileName);
  
  bool isInteractive() const { return IsInteractive_; };
  void setInteractive(bool IsInteractive=true)
  { IsInteractive_ = IsInteractive; }
  
  bool isCompiled() const { return !OutputFileName_.empty(); }
  void setCompile(const std::string & OutputFileName)
  { OutputFileName_ = OutputFileName; }

  bool isVerbose() const { return IsVerbose_; }
  void setVerbose(bool IsVerbose=true) { IsVerbose_=IsVerbose; }
  
  bool isDebug() const { return IsDebug_; }
  void setDebug(bool IsDebug=true) { IsDebug_=IsDebug; }

  bool isOverwrite() const { return IsOverwrite_; }
  void setOverwrite(bool IsOverwrite=true) { IsOverwrite_=IsOverwrite; }

  bool isOptimized() const { return IsDebug_; }
  void setOptimized(bool IsOptimized=true) { IsOptimized_=IsOptimized; }

  bool dumpIR() const { return !IRFileName_.empty(); }
  void setDumpIR(const std::string & IRFileName) { IRFileName_ = IRFileName; }

  bool dumpDot() const { return !DotFileName_.empty(); }
  void setDumpDot(const std::string & DotFileName) { DotFileName_ = DotFileName; }

  void setBackend(const std::string & Backend)
  {
    auto lower = utils::tolower(Backend);
    BackendType_ = SupportedBackends::Size;
#ifdef HAVE_LEGION
    if (lower == "legion") BackendType_ = SupportedBackends::Legion; 
#endif
#ifdef HAVE_KOKKOS
    if (lower == "kokkos") BackendType_ = SupportedBackends::Kokkos;
#endif
#ifdef HAVE_CUDA
    if (lower == "cuda") BackendType_ = SupportedBackends::Cuda;
#endif
#ifdef HAVE_ROCM
    if (lower == "rocm") BackendType_ = SupportedBackends::ROCm;
#endif
    if (lower == "serial") BackendType_ = SupportedBackends::Serial;
    if (BackendType_ == SupportedBackends::Size)
      THROW_CONTRA_ERROR("Unsupported backend requested: '" << Backend << "'.");
  }

  void setArgs(const std::string & Args)
  { Arguments_ = Args; }

  // top ::= definition | external | expression | ';'
  void mainLoop();

private:

  void handleFunction();
  void handleTopLevelExpression();

  std::vector<std::unique_ptr<FunctionAST>>
    optimizeFunction(std::unique_ptr<FunctionAST>);

  template<typename T>
  void reportError(const T&e) const
  { e.accept(*this); }

  void visit(const CodeError & e) const {
    std::cerr << e.what() << std::endl;
    std::cerr << std::endl;
    TheParser_->barf(std::cerr, e.getLoc());
    std::cerr << std::endl;
  }
  
  void visit(const ContraError & e) const {
    std::cerr << e.what() << std::endl;
  }
};

}


#endif //CONTRA_CONTRA_HPP
