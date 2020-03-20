#ifndef CONTRA_TASKING_HPP
#define CONTRA_TASKING_HPP

#include "llvm/IR/IRBuilder.h"

#include <string>

namespace contra {

class AbstractTasker {

protected:

  llvm::IRBuilder<> & Builder_;
  llvm::LLVMContext & TheContext_;

public:
  
  AbstractTasker(llvm::IRBuilder<> & TheBuilder, llvm::LLVMContext & TheContext) :
    Builder_(TheBuilder), TheContext_(TheContext)
  {}

  virtual llvm::Function* wrap(llvm::Module &, const std::string &, llvm::Function*) const = 0;

  virtual ~AbstractTasker() = default;
};

} // namespace

#endif // CONTRA_TASKING_HPP
