#ifndef CONTRA_TASKARG_HPP
#define CONTRA_TASKARG_HPP

#include "llvm/IR/IRBuilder.h"

#include <string>

namespace contra {

//==============================================================================
// Task argument
//==============================================================================
class TaskArgument {
  
  llvm::Type* Type_ = nullptr;
  llvm::Value* Value_ = nullptr;

public:

  TaskArgument(llvm::Type* Type, llvm::Value* Value) :
    Type_(Type), Value_(Value)
  {}

  auto getValue() const { return Value_; }
  auto getType() const { return Type_; }
};

} // namespace

#endif // CONTRA_TASKINFO_HPP
