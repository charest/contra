#ifndef CONTRA_TASKING_HPP
#define CONTRA_TASKING_HPP

#include "llvm/IR/IRBuilder.h"

#include <iostream>
#include <string>

namespace contra {


class TaskInfo {
  int Id_ = -1;
  std::string Name_;
  llvm::Function * Function_ = nullptr;
  intptr_t Address_ = 0;

public:

  TaskInfo(int Id, const std::string & Name, llvm::Function * Func)
    : Id_(Id), Name_(Name), Function_(Func) {}

  auto getId() const { return Id_; }
  const auto & getName() const { return Name_; }
  auto getFunction() const { return Function_; }

  void setAddress(intptr_t Address) { Address_ = Address; }
  intptr_t getAddress() const { 
    if (!Address_) return reinterpret_cast<intptr_t>(Function_);
    else return Address_;
  }
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
