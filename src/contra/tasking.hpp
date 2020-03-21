#ifndef CONTRA_TASKING_HPP
#define CONTRA_TASKING_HPP

#include "llvm/IR/IRBuilder.h"

#include <string>

namespace contra {


struct TaskInfo {
  int Id = -1;
  llvm::Function * Func = nullptr;
};

class AbstractTasker {

protected:

  llvm::IRBuilder<> & Builder_;
  llvm::LLVMContext & TheContext_;

public:
  
  AbstractTasker(llvm::IRBuilder<> & TheBuilder, llvm::LLVMContext & TheContext) :
    Builder_(TheBuilder), TheContext_(TheContext)
  {}

  virtual llvm::Function* wrap(llvm::Module &, const std::string &, llvm::Function*) const = 0;
  virtual void preregister(llvm::Module &, const std::string &, const TaskInfo &) const = 0;
  virtual void set_top(llvm::Module &, int) const = 0;
  virtual llvm::Value* start(llvm::Module &, int, char **) const = 0;

  virtual ~AbstractTasker() = default;
};

} // namespace

#endif // CONTRA_TASKING_HPP
