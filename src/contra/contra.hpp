#ifndef CONTRA_CONTRA_HPP
#define CONTRA_CONTRA_HPP

#include "analysis.hpp"
#include "codegen.hpp"
#include "llvm.hpp"
#include "parser.hpp"
#include "vizualizer.hpp"

#include "llvm/Support/raw_ostream.h"

namespace contra {

class Contra : public ErrorDispatcher {

  bool IsInteractive_ = false;
  bool IsVerbose_ = false;
  bool IsDebug_ = false;
  bool IsOptimized_ = false;

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

public:

  ~Contra() {
    // Finalize whatever needs to be
    TheCG_->finalize();
    // Print out all of the generated code.
    //TheCG.TheModule->print(llvm::errs(), nullptr);
    // Compile if necessary
    if (!OutputFileName_.empty()) compileLLVM( TheCG_->getModule(), OutputFileName_ );
    IRFileStream_ = nullptr;
  }


  void setup(const std::string & FileName)
  {
    ThePrecedence_ = std::make_shared<BinopPrecedence>();

    if (FileName.empty())
      TheParser_ = std::make_unique<Parser>(ThePrecedence_);
    else
      TheParser_ = std::make_unique<Parser>(ThePrecedence_, FileName);

    TheCG_ = std::make_unique<CodeGen>(IsDebug_);

    if (IRFileName_ == "-") {
      IRFileStream_ = &llvm::outs();
    }
    else if (!IRFileName_.empty()) {
      std::error_code EC;
      IRFile_ = std::make_unique<llvm::raw_fd_ostream>(IRFileName_, EC, llvm::sys::fs::F_None);
      IRFileStream_ = IRFile_.get();
    }

    if (!DotFileName_.empty()) TheViz_ = std::make_unique<Vizualizer>(DotFileName_);

    TheAnalyser_ = std::make_unique<Analyzer>(ThePrecedence_);
  }
  
  bool isInteractive() const { return IsInteractive_; };
  void setInteractive(bool IsInteractive=true)
  { IsInteractive_ = IsInteractive; }
  
  bool doCompile() const { return !OutputFileName_.empty(); }
  void setCompile(const std::string & OutputFileName)
  { OutputFileName_ = OutputFileName; }

  bool isVerbose() const { return IsVerbose_; }
  void setVerbose(bool IsVerbose=true) { IsVerbose_=IsVerbose; }
  
  bool isDebug() const { return IsDebug_; }
  void setDebug(bool IsDebug=true) { IsDebug_=IsDebug; }

  bool isOptimized() const { return IsDebug_; }
  void setOptimized(bool IsOptimized=true) { IsOptimized_=IsOptimized; }

  bool dumpIR() const { return !IRFileName_.empty(); }
  void setDumpIR(const std::string & IRFileName) { IRFileName_ = IRFileName; }

  bool dumpDot() const { return !DotFileName_.empty(); }
  void setDumpDot(const std::string & DotFileName) { DotFileName_ = DotFileName; }

  // top ::= definition | external | expression | ';'
  void mainLoop();

private:

  void handleFunction();
  void handleDefinition();
  void handleExtern();  
  void handleTopLevelExpression();

  template<typename T>
  void reportError(const T&e) const
  { e.accept(*this); }

  void dispatch(const CodeError & e) const {
    std::cerr << e.what() << std::endl;
    std::cerr << std::endl;
    TheParser_->barf(std::cerr, e.getLoc());
    std::cerr << std::endl;
  }
  
  void dispatch(const ContraError & e) const {
    std::cerr << e.what() << std::endl;
  }
};

}


#endif //CONTRA_CONTRA_HPP
