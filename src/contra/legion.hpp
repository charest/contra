#ifndef CONTRA_LEGION_HPP
#define CONTRA_LEGION_HPP

#include "tasking.hpp"

namespace llvm {
class AllocaInst;
}

namespace contra {

class LegionTasker : public AbstractTasker {
protected:
  llvm::AllocaInst* ContextAlloca_ = nullptr;
  llvm::AllocaInst* RuntimeAlloca_ = nullptr;

public:
 
  LegionTasker(llvm::IRBuilder<> & TheBuilder, llvm::LLVMContext & TheContext) :
    AbstractTasker(TheBuilder, TheContext)
  {}

  virtual PreambleResult taskPreamble(llvm::Module &, const std::string &,
      llvm::Function*) override;
  virtual void taskPostamble(llvm::Module &, llvm::Value*) override;
  
  virtual void preregisterTask(llvm::Module &, const std::string &, const TaskInfo &) override;
  virtual void postregisterTask(llvm::Module &, const std::string &, const TaskInfo &) override;
  
  virtual void setTopLevelTask(llvm::Module &, int) override;
  
  virtual llvm::Value* startRuntime(llvm::Module &, int, char **) override;
  
  virtual void launch(llvm::Module &, const std::string &, const TaskInfo &,
      const std::vector<llvm::Value*> &, const std::vector<llvm::Value*> &) override;
  
  virtual ~LegionTasker() = default;

protected:

  void reset() {
    ContextAlloca_ = nullptr;
    RuntimeAlloca_ = nullptr;
  }

};

} // namepsace

#endif // LIBRT_LEGION_HPP
