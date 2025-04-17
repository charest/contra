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

class ContraBuilder {
  
  bool IsVerbose_ = false;
  bool IsDebug_ = false;
  bool IsOptimized_ = false;
  bool IsOverwrite_ = false;
  
  std::string OutputFileName_;
  std::string IRFileName_;
  std::string DotFileName_;
  std::string SourceFileName_;

  SupportedBackends BackendType_ = SupportedBackends::Serial;

public:

  bool isInteractive() const { return SourceFileName_.empty(); };
  void setSource(const std::string & SourceFileName)
  { SourceFileName_ = SourceFileName; }
  
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

  const std::string getIRFileName() const
  { return IRFileName_; }
  
  const std::string getDotFileName() const
  { return DotFileName_; }
  
  const std::string getSourceFileName() const
  { return SourceFileName_; }


  void setBackend(const std::string & Backend)
  {
    BackendType_ = getBackend(Backend);
    if (BackendType_ == SupportedBackends::Size)
      THROW_CONTRA_ERROR("Unsupported backend requested: '" << Backend << "'.");
  }
  auto getBackendType() const
  { return BackendType_; }


};

class Contra : public ErrorVisiter {

  const ContraBuilder Builder_;

  llvm::raw_ostream* IRFileStream_ = nullptr;
  std::unique_ptr<llvm::raw_ostream> IRFile_;
  
  std::ifstream InputStream_;
  std::istream *In_ = &std::cin;

  Tokens Tokens_;

  std::shared_ptr<BinopPrecedence> ThePrecedence_;
  std::unique_ptr<Parser> TheParser_;
  std::unique_ptr<CodeGen> TheCG_;
  
  std::unique_ptr<Vizualizer> TheViz_;
  std::unique_ptr<Analyzer> TheAnalyser_;

public:

  Contra(ContraBuilder builder);

  ~Contra() {
    // Print out all of the generated code.
    //TheCG.TheModule->print(llvm::errs(), nullptr);
    // Compile if necessary
    //if (!OutputFileName_.empty()) compile( TheCG_->getModule(), OutputFileName_ );
    IRFileStream_ = nullptr;
  }
  
  bool isInteractive() const { return Builder_.isInteractive(); };
  bool isCompiled() const { return Builder_.isCompiled(); }
  bool isVerbose() const { return Builder_.isVerbose(); }
  bool isDebug() const { return Builder_.isDebug(); }
  bool isOverwrite() const { return Builder_.isOverwrite(); }
  bool isOptimized() const { return Builder_.isDebug(); }
  bool dumpIR() const { return Builder_.dumpIR(); }
  bool dumpDot() const { return Builder_.dumpDot(); }
  const std::string getIRFileName() const { return Builder_.getIRFileName(); }
  const std::string getDotFileName() const { return Builder_.getDotFileName(); }
  const std::string getSourceFileName() const { return Builder_.getSourceFileName(); }
  auto getBackendType() const { return Builder_.getBackendType(); }
  
  // top ::= definition | external | expression | ';'
  void mainLoop();

  void getNextToken();

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
